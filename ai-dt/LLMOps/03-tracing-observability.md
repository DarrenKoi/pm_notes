---
tags: [llmops, tracing, observability, cost, latency]
level: intermediate
last_updated: 2026-07-06
---

# 03. 트레이싱 및 관측성 (Observability)

> LLM 앱의 각 실행을 **재구성 가능한 기록(trace)**으로 남긴다. "왜 이 답이 나왔는가"를 사후에 추적하고, 품질·비용·지연을 함께 본다.

## 왜 필요한가? (Why)

- LLM 파이프라인은 여러 단계(검색 → 프롬프트 조립 → LLM 호출 → 후처리)를 거친다. 답이 이상할 때 **어느 단계에서 틀어졌는지** 로그 없이는 알 수 없다.
- 평가 점수가 떨어졌을 때 trace가 있으면 **실패 케이스를 그대로 재현**해 원인(검색 실패 vs 생성 실패)을 가른다. → [08](./08-rag-evaluation.md), [09](./09-agent-tool-evaluation.md)
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
| 버전 | `release_id`, `prompt_version`, `model`, `retriever_version`, `tool_schema_version` |
| 입출력 | `input`, `output`(민감정보 마스킹) |
| 비용 | `prompt_tokens`, `completion_tokens`, `est_cost` |
| 성능 | `latency_ms`, `error` |
| 피드백 | `user_feedback`(👍/👎), `eval_score`(사후 채점) |

### 3) 사내 관측성 도구 — Arize Phoenix
LangSmith/Langfuse 같은 SaaS는 외부망이라 막힌다. 우리 회사는 **Arize Phoenix**를 self-host하여 LLM 관측성을 확보한다.

- **Arize Phoenix**: 오픈소스 LLM observability 플랫폼. **OpenTelemetry 기반** 트레이싱, 평가, 모니터링을 하나의 UI로 제공한다.
- **self-host 가능**: Docker/pip로 사내 서버에 설치하므로 외부망 차단 환경에서도 사용 가능.
- **OpenTelemetry 네이티브**: OTel GenAI Semantic Conventions을 기본 지원하므로 위 4)에서 맞춰둔 필드와 바로 연결된다.
- **주요 기능**: trace 시각화(워터폴 뷰), span 상세 조회, 평가 결과 연동, 대시보드·드리프트 모니터링.
- 표준 로깅(JSON → OpenSearch)은 **백업·장기 보관 채널**로 병행 유지한다.

### 4) OpenTelemetry GenAI 관점으로 맞춰두기
OpenTelemetry는 GenAI 전용 semantic conventions를 별도 저장소로 관리한다. 지금 당장 OTel collector를 붙이지 않더라도 필드명을 아래처럼 맞춰두면 나중에 표준 트레이싱으로 옮기기 쉽다.

| 내부 필드 | OTel GenAI 대응 | 메모 |
|---|---|---|
| `span="llm_call"` | `gen_ai.operation.name=chat` | chat/completion/embedding/retrieval 등 작업명 |
| `model` | `gen_ai.request.model`, `gen_ai.response.model` | 요청 모델과 실제 응답 모델이 다를 수 있음 |
| `prompt_version` | `gen_ai.prompt.version` | [14](./14-artifact-lineage-governance.md)의 manifest와 연결 |
| `prompt_tokens` | `gen_ai.usage.input_tokens` | 캐시 토큰 포함 여부를 일관되게 정의 |
| `completion_tokens` | `gen_ai.usage.output_tokens` | reasoning token이 있으면 별도 보관 |
| `streaming` | `gen_ai.request.stream` | streaming이면 time-to-first-token도 기록 |

주의: OTel의 `gen_ai.input.messages`, `gen_ai.output.messages`, `gen_ai.system_instructions`는 민감정보를 포함하기 쉬운 opt-in 성격의 필드다. 사내 기본값은 **원문 저장 금지 + 마스킹/해시/외부 보관 포인터**로 둔다.

## 어떻게 사용하는가? (How)

### Arize Phoenix 설치 및 연동

```bash
# Phoenix 설치 (사내 미러/프록시 사용)
pip install arize-phoenix openinference-instrumentation-openai
```

```python
# ── Phoenix 서버 시작 (개발 환경에서 로컬 실행) ──────────────────
import phoenix as px
session = px.launch_app()   # http://localhost:6006 에서 UI 접근

# ── 사내 서버에 띄운 Phoenix에 연결하는 경우 ─────────────────────
import os
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://phoenix.internal:6006"

# ── OpenAI 클라이언트 자동 계측 (한 줄) ──────────────────────────
from openinference.instrumentation.openai import OpenAIInstrumentor
from phoenix.otel import register
tracer_provider = register()                       # Phoenix에 trace 전송
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
# 이제 client.chat.completions.create() 호출이 자동으로 Phoenix에 span으로 기록된다
```

> Phoenix UI(`http://localhost:6006`)에서 trace 워터폴, 토큰 사용량, 지연 분포를 바로 확인할 수 있다.

### 데코레이터 기반 경량 트레이서 (대안)
Phoenix 없이 span을 기록하는 최소 구현. `usage`에서 토큰을 뽑아 비용까지 남긴다. Phoenix가 있으면 위 자동 계측을 우선 사용하고, 커스텀 span이 필요할 때 아래를 보조로 쓴다.

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
def traced_chat(model, messages, prompt_version="v4", release_id="rag-photo-2026-07-06-01", temperature=0):
    r = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
    u = r.usage
    log_span({
        "span": "llm_call.usage", "release_id": release_id,
        "model": model, "prompt_version": prompt_version,
        "gen_ai.operation.name": "chat",
        "gen_ai.request.model": model,
        "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
        "gen_ai.usage.input_tokens": u.prompt_tokens,
        "gen_ai.usage.output_tokens": u.completion_tokens,
    })
    return r.choices[0].message.content
```

### RAG 파이프라인 전체를 하나의 trace로

Phoenix 자동 계측이 활성화되어 있으면 LLM 호출은 자동으로 span이 생긴다. 검색 등 커스텀 단계를 묶으려면 `using_span`을 사용한다.

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

def rag_answer(question):
    with tracer.start_as_current_span("rag_pipeline"):       # 최상위 span
        ctx = traced_retrieve(question)                       # @span("retrieve") 또는 커스텀 span
        msgs = build_prompt(question, ctx)
        ans = traced_chat("Kimi-K2.5", msgs)                  # Phoenix가 LLM span 자동 기록
        return ans
# → Phoenix UI에서 rag_pipeline > retrieve > llm_call 워터폴을 바로 확인
```

### Phoenix에서 보는 4가지
1. **품질 추이** — 일별 평균 eval_score(사후 채점) / 👎 비율. Phoenix Evaluations 탭에서 평가 결과를 trace에 연결해 확인.
2. **비용** — 요청당 평균 토큰, 프롬프트 버전별 비교. Phoenix가 `gen_ai.usage.*` 필드를 자동 집계.
3. **지연** — p50/p95 latency, 단계별(retrieve vs llm) 분해. Phoenix 워터폴 뷰에서 병목 구간을 시각적으로 파악.
4. **실패 로그** — error가 있는 span, 낮은 점수 케이스 상위 N개(개선 후보). Phoenix에서 status=ERROR 필터링.

> 로컬에서는 `px.launch_app()`으로 Phoenix를 띄워 trace를 바로 확인할 수 있다. production에서는 사내 Phoenix 서버(`http://phoenix.internal:6006`)에 집계하고, OpenSearch에 장기 보관용 로그를 병행 적재한다. → [12. 모니터링](./12-monitoring-drift.md)

## 관련 문서
- [02. 프롬프트 버전 관리](./02-prompt-management-versioning.md) — 각 span에 `prompt_version`을 남기는 이유
- [09. Agent·Tool 평가](./09-agent-tool-evaluation.md) — trajectory 평가는 span 트리 위에서 이뤄진다
- [12. 모니터링 & 드리프트](./12-monitoring-drift.md) — trace를 집계해 production 지표로
- [14. 아티팩트 계보와 거버넌스](./14-artifact-lineage-governance.md) — trace와 release manifest 연결

## 참고 자료 (References)
- Arize Phoenix(사내 사용 LLM observability): https://docs.arize.com/phoenix
- OpenTelemetry(트레이싱 개념 원류): https://opentelemetry.io/docs/concepts/
- OpenTelemetry GenAI Semantic Conventions: https://github.com/open-telemetry/semantic-conventions-genai
- OpenInference(Phoenix의 OTel 계측 라이브러리): https://github.com/Arize-ai/openinference
