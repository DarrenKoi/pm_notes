---
tags: [harness-engineering, observability, tracing, opentelemetry, cost]
level: intermediate
last_updated: 2026-09-12
---

# 08. 관측과 비용 (Observability & Cost)

> 비결정적인 다단계 시스템은 로그 몇 줄로 디버깅할 수 없다. 실행 하나를 처음부터
> 끝까지 재생할 수 있는 트레이스와, 성공 1건당 비용을 보여주는 지표가 필요하다.

## 왜 필요한가? (Why)

- "어제 그 작업 왜 실패했어?"에 답하려면 그 실행의 **모든 모델 입력, 출력, 도구
  호출, 결과**가 필요하다. 최종 에러 메시지만으로는 원인이 20턴 전에 있었는지 알
  수 없다
- 에이전트 비용은 선형으로 늘지 않는다. 컨텍스트가 누적되므로 턴이 두 배가 되면
  입력 토큰은 그 이상으로 는다. 루프 폭주 한 건이 하루 예산을 다 쓸 수 있다
- eval([05](./05-verification-and-evals.md))의 새 태스크는 대부분 운영
  트레이스에서 나온다
- OpenAI 팀은 로그, 메트릭, 트레이스를 **에이전트 자신도** 조회할 수 있게
  열어줬다. 관측은 사람만을 위한 것이 아니다

## 핵심 개념 (What)

### 1. 트레이스 구조 (OpenTelemetry GenAI 시맨틱 컨벤션)

```text
invoke_agent  (run 1건)
 ├─ chat              ← LLM 호출 1회
 ├─ execute_tool      ← 도구 실행 1회
 ├─ chat
 ├─ execute_tool
 └─ chat              ← 최종 답
```

주요 속성 이름:

| 속성 | 뜻 |
|---|---|
| `gen_ai.request.model` | 요청한 모델 |
| `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens` | 토큰 사용량 |
| `gen_ai.response.finish_reasons` | 생성 종료 이유 (stop, tool_calls, length 등) |
| `gen_ai.agent.name` | 에이전트 이름 |

표준 이름은 이동을 쉽게 하지만 exporter·스키마 버전까지 자동 호환되지는 않는다.
실제 사용한 컨벤션 버전을 기록하고 필드 변경을 확인한다.
[공식 GenAI 컨벤션](https://opentelemetry.io/docs/specs/semconv/gen-ai/)

### 2. 무엇을 기록하나

| 항목 | 이유 |
|---|---|
| 전체 프롬프트와 응답 | 재현과 디버깅. 개인정보·기밀 마스킹 정책을 함께 정한다 |
| 도구 이름, 인자, 결과(잘린 것), 소요 시간, 에러 | 어느 도구가 병목이고 실패가 잦은지 파악 |
| 토큰, 캐시 적중 토큰, 지연 시간 | 비용과 성능 |
| 종료 이유 (정상, 턴 한도, 예산 초과, 거부) | 실패 분류 |
| **하네스 버전** (프롬프트, 도구 정의, 설정의 해시) | "언제부터 나빠졌나"를 변경 이력과 연결 |
| run_id, 사용자와 요청 ID | 사용자 문의와 트레이스를 연결 |

하네스 버전 기록이 가장 자주 빠진다. 이게 없으면 프롬프트를 고친 날과 성공률이
떨어진 날을 연결할 수 없다.

### 3. 봐야 할 지표

| 지표 | 질문 |
|---|---|
| 태스크 성공률 (검증 통과 기준) | 일을 제대로 하는가? |
| **성공 1건당 비용** | 실패한 실행 비용까지 포함하면 얼마인가? |
| 태스크당 턴 수, 토큰 수 분포 (p50/p95) | 폭주하는 꼬리가 있는가? |
| 도구별 에러율 | 어느 도구 설계를 고쳐야 하는가? |
| 종료 이유 분포 | 턴 한도, 예산 초과가 늘고 있는가? |
| 사람 승인률, 거절률 | 정책이 너무 느슨하거나 빡빡하지 않은가? |
| 캐시 적중률 | 컨텍스트 설계가 캐시를 깨고 있지 않은가? ([03](./03-context-engineering.md)) |

### 4. 예산은 관측이 아니라 강제다

지표를 보는 것만으로는 폭주를 막지 못한다. 실행 단위로 토큰, 비용, 시간 상한을
두고, 넘으면 루프가 스스로 멈추게 한다. 사용자와 팀 단위 일일 한도도 둔다.

### 5. 사내 환경

외부 SaaS로 트레이스를 보낼 수 없다면 자체 호스팅이 가능한 도구를 쓴다(Langfuse,
Arize Phoenix, MLflow 트레이싱 등). 그 전 단계에서는 **JSONL 파일로도
충분하다**. 필드 이름만 OTel 컨벤션을 따라두면 나중에 옮기기 쉽다.

## 어떻게 사용하는가? (How)

### Step 1. JSONL 트레이서

```python
import json
import time
import uuid
from contextlib import contextmanager


class Tracer:
    """JSONL 트레이서.

    필드 이름을 OTel GenAI 컨벤션에 맞춰 나중에 OTel로 옮기기 쉽게 한다.
    """

    def __init__(self, path: str, harness_version: str,
                 run_id: str | None = None):
        self.path, self.harness_version = path, harness_version
        self.run_id = run_id or uuid.uuid4().hex

    @contextmanager
    def span(self, name: str, **attrs):
        rec = {"run_id": self.run_id, "span": name,
               "harness.version": self.harness_version, **attrs}
        start = time.time()
        try:
            yield rec
            rec["status"] = "ok"
        except Exception as e:
            rec["status"], rec["error"] = "error", repr(e)
            raise
        finally:
            rec["duration_ms"] = round((time.time() - start) * 1000)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
```

루프에서 쓰는 방법:

```python
with tracer.span("chat", **{"gen_ai.request.model": model}) as rec:
    resp = client.chat.completions.create(
        model=model, messages=messages, tools=TOOL_SPECS)
    rec["gen_ai.usage.input_tokens"] = resp.usage.prompt_tokens
    rec["gen_ai.usage.output_tokens"] = resp.usage.completion_tokens
    rec["gen_ai.response.finish_reasons"] = [resp.choices[0].finish_reason]

tool_attrs = {"gen_ai.tool.name": call.function.name}
with tracer.span("execute_tool", **tool_attrs) as rec:
    result = execute(call)
    rec["tool.result_chars"] = len(result)
```

### Step 2. 실행 단위 예산

```python
class BudgetExceeded(Exception):
    pass


class Budget:
    def __init__(self, max_tokens: int = 500_000, max_seconds: float = 1800):
        self.max_tokens, self.deadline = max_tokens, time.time() + max_seconds
        self.tokens = 0

    def charge(self, usage) -> None:
        """매 LLM 호출 뒤에 부른다. 한도를 넘으면 예외로 루프를 멈춘다."""
        self.tokens += usage.prompt_tokens + usage.completion_tokens
        if self.tokens > self.max_tokens:
            raise BudgetExceeded(
                f"token budget exceeded: {self.tokens}/{self.max_tokens}")
        if time.time() > self.deadline:
            raise BudgetExceeded("time budget exceeded")
```

### Step 3. 트레이스로 지표 뽑기

```python
import collections


def summarize(path: str) -> dict:
    runs = collections.defaultdict(
        lambda: {"tokens": 0, "turns": 0, "tool_errors": 0})
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            run = runs[r["run_id"]]
            if r["span"] == "chat":
                run["turns"] += 1
                run["tokens"] += (r.get("gen_ai.usage.input_tokens", 0)
                                  + r.get("gen_ai.usage.output_tokens", 0))
            elif r["span"] == "execute_tool" and r["status"] == "error":
                run["tool_errors"] += 1
    tokens = sorted(r["tokens"] for r in runs.values())
    return {"runs": len(runs),
            "tokens_p50": tokens[len(tokens) // 2] if tokens else 0,
            "tokens_max": tokens[-1] if tokens else 0}
```

## 성공을 관측하는 것과 예외를 관측하는 것은 다르다

위 트레이서의 `status=ok`는 Python 예외가 없었다는 뜻이다. [02](./02-agent-loop.md)의
`execute`는 오류도 문자열로 반환하므로, 도구가 실패해도 span은 `ok`일 수 있다.
운영에서는 구조화된 도구 결과의 성공 여부를 별도 속성으로 기록한다.

| 층 | 별도로 기록할 결과 |
|---|---|
| 모델 호출 | 응답 수신·잘림·타임아웃 |
| 도구 | 성공·빈 결과·부분 성공·권한 거부 |
| 검증 | 검증 종류·통과 여부·증거 위치 |
| 업무 | 완료·실패·대기·취소·예산 중단 |

이 표는 이 노트의 권고 스키마다. 학습용 JSONL은 OTel exporter가 아니며,
부모 span ID·분산 전파·동시 쓰기 처리를 구현하지 않는다. 운영에서 도구 서비스와
연결할 때는 trace/span ID를 전파하되, 비밀값을 baggage에 넣지 않는다.
[MCP의 OTel 전파 규약](https://modelcontextprotocol.io/specification/2026-07-28/changelog)

### 호출 전 예산과 호출 후 사용량을 함께 확인한다

위 `Budget.charge`는 응답 이후 한도를 확인한다. 이미 사용한 토큰을 되돌릴 수
없고, 진행 중인 호출을 시간 제한에 맞춰 취소하지도 않는다. 다음은 보강 기준이다.

1. 호출 전에 남은 시간·토큰을 확인하고 가능한 출력 한도를 제한한다.
2. 응답·도구 결과를 기록하고 체크포인트를 저장한 뒤 실제 사용량을 정산한다.
3. 서브에이전트·재시도·검증·압축 비용까지 같은 업무 예산에서 차감한다.
4. 병렬 워커는 예산을 원자적으로 예약한다. 각자 같은 잔액을 보고 실행하지 않는다.
5. 경과 시간 측정에는 단조 시계, 감사 기록에는 날짜·시각을 쓴다.

### 성공 1건당 비용과 사람의 개입을 같이 본다

`전체 시행 비용 / 검증 성공 건수`로 계산한다. 성공이 0건이면 0원으로 표시하지
말고 계산 불가로 보고한다. 비용이 낮아져도 사람의 재작업이 늘었다면 개선이라고
단정하기 어렵다. 승인 대기 시간·검토 시간·재작업 비율을 함께 기록한다.

운영 트레이스는 보존 기간·열람 권한·마스킹을 정한다. 원문 전체가 필요하면
접근 통제된 별도 저장소에 두고 트레이스에는 참조를 남긴다. 기록만으로 동일 모델
응답을 재현할 수는 없지만, 원인과 실행 순서를 추적할 수는 있어야 한다.

## 학습 체크리스트

- [ ] 실행 하나의 전체 트레이스를 파일로 남기고, 턴별로 무슨 일이 있었는지 다시
      읽어봤다
- [ ] 모든 트레이스에 하네스 버전(프롬프트와 도구 정의 해시)을 기록했다
- [ ] 태스크당 토큰 수의 p50과 p95를 계산하고, p95 실행의 트레이스를 열어 원인을
      찾았다
- [ ] 실행 단위 토큰과 시간 예산을 강제했다
- [ ] 사내에서 쓸 수 있는 자체 호스팅 트레이싱 도구를 하나 골라 JSONL을 옮겨봤다

## 참고 자료 (References)

- [OpenTelemetry GenAI semantic
  conventions](https://github.com/open-telemetry/semantic-conventions-genai) —
  agent / chat / tool span 정의
- [Inside the LLM Call: GenAI Observability with
  OpenTelemetry](https://opentelemetry.io/blog/2026/genai-observability/) —
  OpenTelemetry 블로그, 2026
- [Demystifying evals for AI
  agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
  — Anthropic, 운영 모니터링과 트랜스크립트 리뷰
- [Harness engineering](https://openai.com/index/harness-engineering/) — OpenAI,
  에이전트가 직접 조회하는 로그·메트릭·트레이스

## 관련 문서

- [05. 검증과 평가](./05-verification-and-evals.md)
- [07. 상태와 복구](./07-state-and-recovery.md)
- [10. 프로덕션 체크리스트](./10-production-checklist.md)
