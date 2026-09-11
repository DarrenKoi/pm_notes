---
tags: [harness-engineering, agent-loop, tool-calling, python]
level: intermediate
last_updated: 2026-09-12
---

# 02. 에이전트 루프 (Agent Loop)

> 모든 하네스의 뼈대는 `모델 호출 → 도구 실행 → 결과를 컨텍스트에 추가`를
> 반복하는 while 루프다. 프로덕션 품질은 이 루프가 언제, 어떻게 멈추느냐에서
> 갈린다.

## 왜 필요한가? (Why)

프레임워크(LangGraph, Agent SDK 등)를 쓰더라도 내부는 결국 이 루프다. 루프를
직접 짜본 적이 없으면 다음과 같은 문제를 디버깅할 수 없다.

- 에이전트가 같은 도구를 무한히 호출한다
- 잘못된 JSON 인자 하나 때문에 전체 작업이 예외로 죽는다
- 도구 결과 10만 자가 컨텍스트에 그대로 들어가 다음 호출이 실패한다
- 모델이 "완료했습니다"라고 했지만 실제로는 아무것도 확인하지 않았다

HumanLayer의 *12-Factor Agents*가 "Own your control flow"를 원칙으로 두는 이유도
같다. 루프를 이해하고 통제해야 운영할 수 있다.

## 핵심 개념 (What)

### 1. 루프 구조

```text
┌──────────────────────────────────────────────────────────┐
│ messages = [system, user_task]                           │
│ loop (turn < MAX_TURNS, budget 남음):                    │
│   response = model(messages, tools)                      │
│   messages += response                                   │
│   if tool_calls 없음: return response.content   ← 종료   │
│   for call in tool_calls:                                │
│       result = policy_check → execute → truncate         │
│       messages += tool_result(call.id, result)           │
│ return "STOPPED: limit reached"                ← 강제 종료│
└──────────────────────────────────────────────────────────┘
```

### 2. 종료 조건은 여러 개여야 한다

| 종료 조건 | 이유 |
|---|---|
| 모델이 도구 없이 답함 | 정상 종료 |
| 최대 턴 수 도달 | 무한 루프 방지 |
| 토큰·비용·시간 예산 소진 | 비용 폭주 방지 ([08](./08-observability-and-cost.md)) |
| 연속 에러 N회 | 같은 실패 반복 방지 |
| 사람의 승인 대기 | 위험 행동 전 일시정지 ([06](./06-guardrails-and-permissions.md), [07](./07-state-and-recovery.md)) |
| 외부 취소 요청 | 사용자 중단 |

"모델이 끝났다고 말함"은 작업 완료가 아니라 루프 종료일 뿐이다. 실제 완료 여부는
검증 단계가 판정한다([05](./05-verification-and-evals.md)).

### 3. 에러는 예외가 아니라 관측 결과다

도구 실패(파일 없음, 잘못된 인자, 타임아웃)를 예외로 던지면 루프 전체가 죽는다.
실패를 **모델이 읽을 수 있는 문장**으로 바꿔 tool result로 돌려주면 모델이
스스로 고칠 기회가 생긴다. 이때 에러 메시지에는 무엇이 틀렸고 어떻게 다시
호출하면 되는지를 적는다.

### 4. 워크플로우 vs 에이전트

Anthropic은 둘을 구분한다.

- **워크플로우**: 코드가 정한 경로를 따라 LLM을 호출한다. prompt chaining,
  routing, parallelization, orchestrator-workers, evaluator-optimizer가 여기
  속한다.
- **에이전트**: LLM이 다음 행동을 동적으로 결정한다.

실무에서는 대부분 둘을 섞는다. 경로가 예측 가능한 구간은 워크플로우로 고정하고,
탐색이 필요한 구간에만 루프를 쓴다. 자율성이 클수록 비용과 실패 가능성도 함께
커진다.

## 어떻게 사용하는가? (How)

### Step 1. 최소 하네스 (약 90줄)

Chat Completions의 `tools`와 `tool_calls`를 지원하는 엔드포인트용 학습 예제다.
OpenAI 호환이라는 이름만으로 도구 호출 지원이 보장되지는 않는다.
서버의 모델·도구 파서 설정과 응답 형식을 확인한다. `pip install openai pytest`

이 예제는 파일 경로 격리, JSON Schema 검증, 업무 완료 판정을 구현하지 않는다.
`read_file`은 임의 경로를 읽을 수 있고 `run_tests`는 저장소 코드를 실행한다.
신뢰하는 연습 파일을 별도 작업 공간에서 사용하고, 운영 연결 전에는
[06](./06-guardrails-and-permissions.md)의 실행 경계를 적용한다.

```python
"""mini_harness.py — 최소 에이전트 하네스.

실행:
  LLM_BASE_URL=http://<internal-endpoint>/v1 LLM_MODEL=Kimi-K2.5 \
  python mini_harness.py "테스트가 왜 실패하는지 찾아줘"
"""
import json
import os
import subprocess
import sys

MAX_TURNS = 20           # 무한 루프 방지: 상한은 반드시 둔다
MAX_TOOL_OUTPUT = 4_000  # 도구 결과가 컨텍스트를 잡아먹지 않게 자른다 (문자 수)

SYSTEM = """You are a coding assistant working in the current directory.
Use tools to inspect files and run tests.
Run run_tests to verify before you finish.
When done, reply with a short final answer and no tool calls."""


# --- 도구 구현 -------------------------------------------------------------
def read_file(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def run_tests() -> str:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q"],
                       capture_output=True, text=True, timeout=300)
    return f"exit_code={r.returncode}\n{r.stdout[-3000:]}\n{r.stderr[-1000:]}"


TOOLS = {"read_file": read_file, "run_tests": run_tests}

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": ("Read a UTF-8 text file. "
                        "Use a path relative to the working directory."),
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "run_tests",
        "description": ("Run the pytest suite. "
                        "Returns exit_code and the tail of the output."),
        "parameters": {"type": "object", "properties": {}},
    }},
]


# --- 하네스 핵심 -----------------------------------------------------------
def truncate(text: str) -> str:
    if len(text) <= MAX_TOOL_OUTPUT:
        return text
    cut = len(text) - MAX_TOOL_OUTPUT
    return (text[:MAX_TOOL_OUTPUT]
            + f"\n...[truncated {cut} chars; request a narrower range]")


def execute(call) -> str:
    """도구 호출 1건 실행.

    어떤 실패도 루프를 죽이지 않고, 모델이 읽을 문장으로 돌려준다.
    """
    name = call.function.name
    if name not in TOOLS:
        return f"ERROR: unknown tool '{name}'. Available tools: {sorted(TOOLS)}"
    try:
        args = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError as e:
        return (f"ERROR: arguments are not valid JSON ({e}). "
                "Call the tool again with valid JSON.")
    try:
        return truncate(str(TOOLS[name](**args)))
    except Exception as e:  # 도구 에러도 관측 결과(observation)다
        return f"ERROR: {type(e).__name__}: {e}"


def assistant_dict(msg) -> dict:
    d = {"role": "assistant", "content": msg.content}
    if msg.tool_calls:
        d["tool_calls"] = [
            {"id": c.id, "type": "function",
             "function": {"name": c.function.name,
                          "arguments": c.function.arguments}}
            for c in msg.tool_calls
        ]
    return d


def run_agent(client, model: str, task: str) -> tuple[str, list[dict]]:
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": task}]
    for _ in range(MAX_TURNS):
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=TOOL_SPECS)
        msg = resp.choices[0].message
        messages.append(assistant_dict(msg))
        if not msg.tool_calls:  # 종료 조건 1: 모델이 도구 없이 답했다
            return msg.content or "", messages
        for call in msg.tool_calls:
            result = execute(call)
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": result})
    # 종료 조건 2: 상한 도달
    return f"STOPPED: reached MAX_TURNS={MAX_TURNS}", messages


if __name__ == "__main__":
    from openai import OpenAI

    # 429/5xx 재시도는 SDK 내장 기능(max_retries)으로 충분하다
    client = OpenAI(base_url=os.environ["LLM_BASE_URL"],
                    api_key=os.environ.get("LLM_API_KEY", "none"),
                    max_retries=5, timeout=120)
    model = os.environ.get("LLM_MODEL", "Kimi-K2.5")
    answer, _ = run_agent(client, model, sys.argv[1])
    print(answer)
```

### Step 2. 모델 없이 하네스를 테스트한다

하네스 로직(종료 조건, 에러 처리)은 결정적이어야 한다. 그러니 실제 LLM 대신
**대본대로 응답하는 가짜 클라이언트**로 테스트한다. 비용도 들지 않고 결과가 매번
같다.

```python
"""test_mini_harness.py — 실행: python test_mini_harness.py"""
from types import SimpleNamespace as NS

from mini_harness import run_agent


class ScriptedClient:
    """미리 정한 응답을 순서대로 돌려주는 가짜 LLM."""
    def __init__(self, replies):
        replies = iter(replies)
        self.chat = NS(completions=NS(
            create=lambda **kw: NS(choices=[NS(message=next(replies))])))


def call(name, arguments):
    fn = NS(name=name, arguments=arguments)
    return NS(content=None, tool_calls=[NS(id="c1", function=fn)])


def final(text):
    return NS(content=text, tool_calls=None)


# 1) 없는 도구나 깨진 JSON이 와도 루프가 죽지 않고 에러를 모델에게 돌려준다
client = ScriptedClient([
    call("delete_all", "{}"),
    call("read_file", "{bad json"),
    final("done"),
])
answer, messages = run_agent(client, "fake", "task")
results = [m["content"] for m in messages if m["role"] == "tool"]
assert answer == "done"
assert results[0].startswith("ERROR: unknown tool")
assert results[1].startswith("ERROR: arguments are not valid JSON")

# 2) 모델이 끝없이 도구만 부르면 MAX_TURNS에서 멈춘다
loop_forever = (call("read_file", '{"path": "x.txt"}') for _ in range(100))
client = ScriptedClient(loop_forever)
answer, _ = run_agent(client, "fake", "task")
assert answer.startswith("STOPPED"), answer
print("ok")
```

### Step 3. 프로덕션으로 가며 추가할 것

최소 하네스에서 빠진 것들이다. 뒤 문서에서 하나씩 다룬다.

| 빠진 것 | 문서 |
|---|---|
| 도구 실행 전 권한 판정 (allow / ask / deny) | [06](./06-guardrails-and-permissions.md) |
| 컨텍스트 한도 접근 시 compaction | [03](./03-context-engineering.md) |
| 매 턴 체크포인트와 재개 | [07](./07-state-and-recovery.md) |
| 토큰·비용 예산, 트레이스 기록 | [08](./08-observability-and-cost.md) |
| 완료 선언 뒤 검증 단계 | [05](./05-verification-and-evals.md) |
| 스트리밍, 취소, 병렬 도구 실행 | 사용자 경험(UX) 요구가 생기면 추가 |

## 확장할 때 유지할 실행 계약

아래는 학습용 루프에 추가할 설계 기준이다. 도구가 없는 응답은
`answer_ready`일 뿐이고, 검증이 통과해야 `completed`로 판정한다.

| 상황 | 처리 | 기록할 이유 |
|---|---|---|
| 정상 답변 후 검증 실패 | 전체 예산 안에서 교정 | `verification_failed` |
| 연결 끊김·출력 한도로 응답 불완전 | 부분 JSON을 실행하지 않음 | `model_error` |
| 추가 정보·승인 필요 | 요청과 대기 상태 저장 | `waiting_input` / `waiting_approval` |
| 취소 요청 | 새 행동 중단, 진행 중 호출 확인 | `cancel_requested` |
| 턴·시간·토큰 한도 소진 | 부분 산출물과 미완료 항목 반환 | `budget_exceeded` |

각 `tool_call_id`에 결과를 대응시킨다. 독립적인 읽기만 병렬화하고,
쓰기나 앞선 결과에 의존하는 호출은 순서를 지킨다. 정책 거부도 명시적인
결과여야 한다. 취소 응답을 받았다고 외부 쓰기가 취소됐다고 단정하지 않는다.
[MCP Tasks의 취소 의미](https://tasks.extensions.modelcontextprotocol.io/specification/2026-07-28/tasks)

재시도와 검증 재실행도 최초 작업의 예산을 공유한다. 새 루프마다 예산을 초기화하면
최대 턴 제한이 있어도 전체 실행은 끝없이 길어질 수 있다.

## 학습 체크리스트

- [ ] 위 코드를 사내 엔드포인트로 실행하고 `messages`를 JSON으로 덤프해 한 턴씩
      읽어봤다
- [ ] 모델이 잘못된 JSON 인자를 보낸 뒤 에러 메시지를 읽고 스스로 고치는 장면을
      확인했다
- [ ] 워크플로우로 충분한 작업과 에이전트가 필요한 작업을 내 업무에서 하나씩
      구분해봤다
- [ ] 가짜 클라이언트로 종료 조건 테스트를 작성했다

## 참고 자료 (References)

- [Building effective
  agents](https://www.anthropic.com/engineering/building-effective-agents) —
  Anthropic, 워크플로우 패턴과 에이전트의 구분
- [12-Factor Agents](https://github.com/humanlayer/12-factor-agents) —
  HumanLayer, "Own your control flow", "Tools are just structured outputs"
- [Building agents with the Claude Agent
  SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk)
  — Anthropic, "에이전트에게 컴퓨터를 줘라"

## 관련 문서

- [01. 핵심 개념](./01-core-concept.md)
- [03. 컨텍스트 엔지니어링](./03-context-engineering.md)
- [04. 도구 설계](./04-tool-design.md)
