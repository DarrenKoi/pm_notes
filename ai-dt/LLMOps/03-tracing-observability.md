---
tags: [llmops, tracing, observability, cost, latency]
level: intermediate
last_updated: 2026-07-06
reviewed_on: 2026-10-04
review_status: partial
document_type: learning
category_major: "AI·DT"
category_middle: "LLM 평가·운영"
category_minor: "운영 기초·관측"
note_kind: "학습"
classified_on: "2026-10-05"
---

> [!info] 검토 범위 — 2026-10-04
> 공식·일차 근거와 로컬 검증은 [공통 적용 조건](./verified-conditions.md), 변경·미확인은 [정리 기록](./organization-log.md)에 있다. 실제 사내 접속·모델 품질·운영 승인과 Claude 협의는 미확인이다. 원래17개를 개별 검토했다. 실제 운영·읽기 화면 검증은 미완료다.


# 03. 트레이싱 및 관측성 (Observability)

> LLM 앱의 각 실행을 **재구성 가능한 기록(trace)**으로 남긴다. "왜 이 답이 나왔는가"를 사후에 추적하고, 품질·비용·지연을 함께 본다.

## 왜 필요한가? (Why)

- LLM 파이프라인은 여러 단계(검색 → 프롬프트 조립 → LLM 호출 → 후처리)를 거친다. 답이 이상할 때 **어느 단계에서 틀어졌는지** 로그 없이는 알 수 없다.
- 평가 점수가 떨어졌을 때 trace에 입력·버전·실행 조건이 충분하면 실패 케이스를 조사해 원인(검색 실패 vs 생성 실패)을 가른다. → [08](./08-rag-evaluation.md), [09](./09-agent-tool-evaluation.md)
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
Phoenix는 자체 호스팅 가능한 관측성 후보다. 원래 노트의 “우리 회사가 채택했다”는 주장과 실제 OpenSearch 병행 운영은 미확인이다.

- SDK 계측·collector 전송·서버 저장/UI는 서로 다른 단계다. 패키지 설치만으로 운영 환경이 구성되지는 않는다.
- OpenInference의 OTel span과 OTel GenAI convention은 판본·설정에 따라 속성이 다를 수 있다. 실제 수집 필드와 UI 집계를 확인해야 한다.
- trace 워터폴·평가 연결을 조사할 수 있지만 기본 화면/드리프트 기능의 위치와 실제 수집 성공은 이번에 검증하지 않았다.

### 4) OpenTelemetry GenAI 관점으로 맞춰두기
OpenTelemetry는 GenAI 전용 semantic conventions를 별도 저장소로 관리한다. 지금 당장 OTel collector를 붙이지 않더라도 필드명을 아래처럼 맞춰두면 키 대응을 기록할 수 있다. 2026-10-04 공식 main의 GenAI span 명세는 Development 상태이므로 고정 판본·실제 계측 결과와 따로 대조해야 한다.

| 내부 필드 | OTel GenAI 대응 | 메모 |
|---|---|---|
| `span="llm_call"` | `gen_ai.operation.name=chat` | chat/completion/embedding/retrieval 등 작업명 |
| `model` | `gen_ai.request.model`, `gen_ai.response.model` | 요청 모델과 실제 응답 모델이 다를 수 있음 |
| `prompt_version` | `gen_ai.prompt.version` | prompt name도 기록. [14](./14-artifact-lineage-governance.md)의 manifest와 연결 |
| `prompt_tokens` | `gen_ai.usage.input_tokens` | 캐시 토큰 포함 여부를 일관되게 정의 |
| `completion_tokens` | `gen_ai.usage.output_tokens` | reasoning token이 있으면 별도 보관 |
| `streaming` | `gen_ai.request.stream` | streaming이면 time-to-first-token도 기록 |

주의: OTel의 `gen_ai.input.messages`, `gen_ai.output.messages`, `gen_ai.system_instructions`는 민감정보를 포함하기 쉬운 opt-in 성격의 필드다. 사내 기본값은 **원문 저장 금지 + 마스킹/해시/외부 보관 포인터**로 둔다.

## 어떻게 사용하는가? (How)

### Arize Phoenix 설치 및 연동

```bash
# Phoenix 설치 (사내 미러/프록시 사용)
pip install arize-phoenix-otel==0.17.2 openinference-instrumentation-openai==0.1.63
```

```python
import os
from openinference.instrumentation import TraceConfig
from openinference.instrumentation.openai import OpenAIInstrumentor
from phoenix.otel import register

# 승인된 collector endpoint를 설정한다. Phoenix server는 별도로 설치/구성한다.
tracer_provider = register(
    endpoint=os.environ["PHOENIX_COLLECTOR_ENDPOINT"],
    project_name="llmops-learning", batch=True, auto_instrument=False, verbose=False,
)
# 수집 전에 원문·도구 schema·embedding을 숨긴다. 자체 attribute/예외도 별도 검토한다.
privacy = TraceConfig(
    hide_inputs=True, hide_outputs=True, hide_llm_invocation_parameters=True,
    hide_llm_tools=True, hide_embedding_vectors=True, hide_embeddings_text=True,
)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider, config=privacy)
# 앱 종료 시 tracer_provider.shutdown()으로 flush/종료한다.
```

> 서버 접속·수집 인증·retention을 구성한 뒤 UI에서 수신 여부를 확인한다. 이 전송/UI 경로는 미검증이며, 로컬 검증은 InMemorySpanExporter로 실제 SDK 계측 속성만 확인했다.

### 데코레이터 기반 경량 트레이서 (대안)
다음은 JSON 타이머 로그이며 OTel span/parent-child 트리를 만들지 않는다. OTel 계측의 보조 기록으로 사용할 수 있다. 로그 ID와 실제 OTel trace ID를 혼동하지 않는다.

```python
import time, json, uuid, functools
from contextvars import ContextVar
from contextlib import contextmanager
from datetime import datetime, timezone

_trace_id = ContextVar("log_trace_id", default=None)

@contextmanager
def new_trace():
    token = _trace_id.set(uuid.uuid4().hex)
    try:
        yield _trace_id.get()
    finally:
        _trace_id.reset(token)  # 요청 종료 후 이전 ContextVar 복원

def log_span(record: dict):
    record = {**record, "log_trace_id": _trace_id.get(),
              "timestamp": datetime.now(timezone.utc).isoformat()}
    print(json.dumps(record, ensure_ascii=False, allow_nan=False))

def span(name):
    def deco(fn):
        @functools.wraps(fn)
        def wrap(*args, **kwargs):
            started = time.perf_counter()
            error = None
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                error = type(exc).__name__  # 예외 원문에는 민감한 입력이 있을 수 있음
                raise
            finally:
                log_span({"span": name, "latency_ms": (time.perf_counter()-started)*1000,
                          "error": error})
        return wrap
    return deco
```

### LLM 호출 span에서 토큰·비용 기록

```python
import os
from openai import OpenAI
client = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"])
TARGET_MODEL = os.environ["LLM_MODEL"]
import math

# 승인된 같은 단위의 입력/출력 100만 토큰 단가를 넣는다. 미설정은 무료가 아닌 미확인.
PRICE = {}

@span("llm_call")
def traced_chat(model, messages, prompt_version="v4", release_id="learning-example", temperature=0):
    r = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
    usage = r.usage
    input_tokens = None if usage is None else usage.prompt_tokens
    output_tokens = None if usage is None else usage.completion_tokens
    if any(value is not None and (type(value) is not int or value < 0)
           for value in (input_tokens, output_tokens)):
        raise ValueError("토큰 사용량 계약 오류")
    rates = PRICE.get(model)
    cost = None
    if rates is not None and input_tokens is not None and output_tokens is not None:
        if any(not math.isfinite(v) or v < 0 for v in rates.values()):
            raise ValueError("단가 입력 오류")
        cost = (input_tokens*rates["in"] + output_tokens*rates["out"]) / 1_000_000
    log_span({"span": "llm_call.usage", "release_id": release_id,
              "requested_model": model, "response_model": r.model,
              "prompt_version": prompt_version,
              "prompt_tokens": input_tokens, "completion_tokens": output_tokens,
              "est_cost": cost})
    if not r.choices or not isinstance(r.choices[0].message.content, str):
        raise ValueError("텍스트 응답 없음")
    return r.choices[0].message.content
```

### RAG 파이프라인 전체를 하나의 trace로

Phoenix 자동 계측이 활성화되어 있으면 LLM 호출은 자동으로 span이 생긴다. 검색 등 커스텀 단계는 실제 `start_as_current_span`으로 묶는다. 아래 retrieve_fn/build_prompt_fn은 호출자가 전달하는 함수다.

```python
from opentelemetry import trace
tracer = trace.get_tracer(__name__)

def rag_answer(question, retrieve_fn, build_prompt_fn):
    with new_trace(), tracer.start_as_current_span("rag_pipeline") as pipeline:
        pipeline.set_attribute("app.release_id", "learning-example")
        pipeline.set_attribute("app.prompt_version", "v4")
        with tracer.start_as_current_span("retrieve"):
            contexts = retrieve_fn(question)
        messages = build_prompt_fn(question, contexts)
        return traced_chat(TARGET_MODEL, messages)
# JSON 데코레이터는 OTel span이 아니다. SDK 자동 계측이 LLM child span을 만든다.
```

### Phoenix에서 보는 4가지

다음은 관측 요구사항이다. 해당 server/UI 판본의 필드 매핑과 대시보드 구성은 실제 확인해야 한다.
1. **품질 추이** — 일별 평균 eval_score(사후 채점) / 👎 비율. Phoenix의 평가 기능에서 평가 결과를 trace에 연결해 확인.
2. **비용** — 요청당 평균 토큰, 프롬프트 버전별 비교. 실제 계측의 token 속성과 미수집 비율을 대조한다.
3. **지연** — p50/p95 latency, 단계별(retrieve vs llm) 분해. Phoenix 워터폴 뷰에서 병목 구간을 시각적으로 파악.
4. **실패 로그** — error가 있는 span, 낮은 점수 케이스 상위 N개(개선 후보). 수집된 status=ERROR와 애플리케이션 실패 정의를 대조한다.

> `arize-phoenix` server 패키지의 `px.launch_app()`은 별도 로컬 서버 시작 방식이며 이 client 계측 예제와 다르다. 실제 회사 collector·OpenSearch·대시보드 운영은 미확인이다. → [12. 모니터링](./12-monitoring-drift.md)

## 관련 문서
- [02. 프롬프트 버전 관리](./02-prompt-management-versioning.md) — 각 span에 `prompt_version`을 남기는 이유
- [09. Agent·Tool 평가](./09-agent-tool-evaluation.md) — trajectory 평가는 span 트리 위에서 이뤄진다
- [12. 모니터링 & 드리프트](./12-monitoring-drift.md) — trace를 집계해 production 지표로
- [14. 아티팩트 계보와 거버넌스](./14-artifact-lineage-governance.md) — trace와 release manifest 연결

## 참고 자료 (References)
- Arize Phoenix(자체 호스팅 후보): https://docs.arize.com/phoenix
- OpenTelemetry(트레이싱 개념 원류): https://opentelemetry.io/docs/concepts/
- OpenTelemetry GenAI Semantic Conventions: https://github.com/open-telemetry/semantic-conventions-genai
- OpenInference(Phoenix의 OTel 계측 라이브러리): https://github.com/Arize-ai/openinference
