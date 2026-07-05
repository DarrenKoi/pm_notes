---
tags: [llmops, tracing, observability, cost, latency]
level: intermediate
last_updated: 2026-07-06
---

# 03. 트레이싱 및 관측성 (Observability)

> LLM 앱의 각 실행을 **재구성 가능한 기록(trace)**으로 남긴다. "왜 이 답이 나왔는가"를 사후에 추적하고, 품질·비용·지연을 함께 본다.

## 왜 필요한가? (Why)

- LLM 파이프라인은 여러 단계(검색 → 프롬프트 조립 → LLM 호출 → 후처리)를 거친다. 답이 이상할 때 **어느 단계에서 틀어졌는지** 로그 없이는 알 수 없다.
- 평가 점수가 떨어졌을 때, trace가 있으면 **실패 케이스를 그대로 재현**해 원인(검색 실패 vs 생성 실패)을 가른다. → [08](./08-rag-evaluation.md), [09](./09-agent-tool-evaluation.md)
- 품질만 보면 함정에 빠진다. **토큰·비용·지연**을 같이 기록해야 "품질 +2%, 비용 +200%" 같은 나쁜 거래를 걸러낸다.

## 핵심 개념 (What)

### 1) Trace / Span
- **Span**: 하나의 작업 단위(예: `retrieve`, `llm_call`). 시작·끝 시각, 입력·출력, 메타데이터를 가진다.
- **Trace**: 하나의 요청을 처리한 span들의 트리. 사용자 질문 1건 = trace 1개.
- 이 구조는 분산 트레이싱(OpenTelemetry)의 개념을 LLM에 가져온 것.

### 2) 반드시 남길 필드
| 카테고리 | 필드 |
|----------|------|
| 식별 | `trace_id`, `span`, `timestamp` |
| 버전 | `prompt_version`, `model`, `retriever_version` |
| 입출력 | `input`, `output`(민감정보 마스킹) |
| 비용 | `prompt_tokens`, `completion_tokens`, `est_cost` |
| 성능 | `latency_ms`, `error` |
| 피드백 | `user_feedback`(👍/👎), `eval_score`(사후 채점) |

### 3) 사내 자체 관측성
LangSmith/Langfuse 같은 SaaS는 외부망이라 막힌다. 대안:
- **표준 로깅 + 사내 저장소**(OpenSearch/파일)에 위 스키마로 JSON을 적재.
- 사내에 Langfuse를 **self-host** 하는 방법도 있으나, 우선 표준 로깅으로 스키마를 확정한 뒤 도구를 붙이는 편이 안전.

## 어떻게 사용하는가? (How)

### 데코레이터 기반 경량 트레이서
프레임워크 없이 span을 기록하는 최소 구현. `usage`에서 토큰을 뽑아 비용까지 남긴다.

```python
import time, json, uuid, functools
from contextvars import ContextVar

_trace_id = ContextVar("trace_id", default=None)

def new_trace():
    _trace_id.set(uuid.uuid4().hex[:12])
    return _trace_id.get()

def log_span(record: dict):
    record = {"trace_id": _trace_id.get(), **record}
    # 로컬: 파일에 append / 사내: OpenSearch 인덱스로 bulk 적재
    print(json.dumps(record, ensure_ascii=False))   # 데모용

def span(name):
    def deco(fn):
        @functools.wraps(fn)
        def wrap(*a, **kw):
            t0 = time.time()
            err = None
            try:
                return fn(*a, **kw)
            except Exception as e:
                err = repr(e); raise
            finally:
                log_span({"span": name, "latency_ms": int((time.time()-t0)*1000), "error": err})
        return wrap
    return deco
```

### LLM 호출 span에서 토큰·비용 기록

```python
from openai import OpenAI
client = OpenAI(base_url="http://llm-gateway.internal/v1", api_key="EMPTY")

# 사내 단가는 예시(요금이 아닌 GPU-시간 환산일 수 있음). 상대 비교용으로만.
PRICE = {"Kimi-K2.5": {"in": 0.0, "out": 0.0}}   # 사내: 실제 단가/쿼터로 대체

@span("llm_call")
def traced_chat(model, messages, prompt_version="v4", temperature=0):
    r = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
    u = r.usage
    log_span({
        "span": "llm_call.usage", "model": model, "prompt_version": prompt_version,
        "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
    })
    return r.choices[0].message.content
```

### RAG 파이프라인 전체를 하나의 trace로

```python
def rag_answer(question):
    new_trace()                          # 요청 1건 = trace 1개
    ctx = traced_retrieve(question)      # @span("retrieve")
    msgs = build_prompt(question, ctx)   # 프롬프트 조립
    ans = traced_chat("Kimi-K2.5", msgs) # @span("llm_call")
    return ans
# → 같은 trace_id 아래 retrieve/llm_call span이 묶여, 실패 케이스를 그대로 복원 가능
```

### 관측 대시보드에서 보는 4가지
1. **품질 추이** — 일별 평균 eval_score(사후 채점) / 👎 비율.
2. **비용** — 요청당 평균 토큰, 프롬프트 버전별 비교.
3. **지연** — p50/p95 latency, 단계별(retrieve vs llm) 분해.
4. **실패 로그** — error가 있는 span, 낮은 점수 케이스 상위 N개(개선 후보).

> 로컬 개발 환경에는 OpenSearch가 없으므로, 여기서는 **JSON 표준 로깅까지만** 검증하고 적재/대시보드는 사내에서 붙인다. → [12. 모니터링](./12-monitoring-drift.md)

## 관련 문서
- [02. 프롬프트 버전 관리](./02-prompt-management-versioning.md) — 각 span에 `prompt_version`을 남기는 이유
- [09. Agent·Tool 평가](./09-agent-tool-evaluation.md) — trajectory 평가는 span 트리 위에서 이뤄진다
- [12. 모니터링 & 드리프트](./12-monitoring-drift.md) — trace를 집계해 production 지표로

## 참고 자료 (References)
- OpenTelemetry(트레이싱 개념 원류): https://opentelemetry.io/docs/concepts/
- Langfuse(self-host 가능한 LLM observability): https://langfuse.com/docs
