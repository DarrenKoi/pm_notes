---
tags: [harness-engineering, durable-execution, checkpoint, idempotency]
level: intermediate
last_updated: 2026-09-12
---

# 07. 상태와 복구 (State & Recovery)

> 긴 작업은 반드시 중간에 멈춘다. 모델 API 타임아웃, 배포 재시작, 사람 승인
> 대기가 원인이다. 멈춘 지점에서 안전하게 이어가는 것이 프로덕션 하네스의 기본
> 요건이다.

## 왜 필요한가? (Why)

- 40턴짜리 작업이 35턴째 죽었을 때 처음부터 다시 하면 비용이 두 배가 되고, 이미
  실행한 부수효과(메일 발송, 티켓 생성)가 **중복 실행**된다
- 사람 승인이 몇 시간 뒤에 온다면 프로세스를 그동안 붙잡아 둘 수 없다
- 장애를 분석하려면 "그 시점의 상태"를 다시 볼 수 있어야 한다

## 핵심 개념 (What)

### 1. 상태란 무엇인가

```text
에이전트 상태 = 대화 이력(messages)
              + 실행 메타데이터(turn, status, pending action)
              + 외부 산출물(작업 폴더 파일, git 커밋, DB 레코드)
```

12-Factor Agents는 에이전트를 **상태 없는 리듀서(stateless reducer)**로 만들라고
한다. `(현재 상태, 새 이벤트) → 다음 상태` 형태의 순수 함수에 가깝게 만들면,
상태만 저장해 두면 어느 프로세스에서든 이어서 실행할 수 있다.

### 2. 체크포인트 저장 시점

| 방식 | 설명 | 트레이드오프 |
|---|---|---|
| 매 스텝 동기 저장 | 다음 스텝 전에 저장 완료 | 가장 안전, 약간 느림 |
| 비동기 저장 | 다음 스텝과 병렬로 저장 | 빠름, 크래시 시 마지막 스텝 유실 가능 |
| 종료 시 저장 | 성공, 에러, 인터럽트 때만 저장 | 가장 빠름, 중간 크래시에 취약 |

LangGraph의 durability 모드(`sync` / `async` / `exit`)가 바로 이 선택지다.
부수효과가 있는 에이전트라면 동기 저장이 기본이다.

### 3. 체크포인트만으로는 부족하다: 멱등성

체크포인트는 "스텝이 끝난 뒤" 저장된다. 도구가 메일을 보낸 직후, 체크포인트를
저장하기 직전에 프로세스가 죽으면 어떻게 될까? 재개하면 같은 도구 호출이 **다시
실행된다**. 그래서 외부 부수효과가 있는 도구에는 **멱등성 키(idempotency
key)**가 필요하다.

```text
key = run_id : turn : tool_call_id
도구 서버: 같은 key로 이미 처리했으면 → 새로 실행하지 않고 이전 결과 반환
```

### 4. 실패 유형별 대응

| 유형 | 예 | 대응 |
|---|---|---|
| 일시적 인프라 오류 | 429, 503, 연결 끊김, 타임아웃 | 지수 백오프 + jitter로 재시도 |
| 영구 오류 | 인증 실패, 권한 없음, 존재하지 않는 리소스 | 재시도하지 않는다. 모델이나 사람에게 보고 |
| 모델 출력 오류 | 깨진 JSON, 없는 도구 호출 | 교정 메시지를 tool result로 돌려준다 ([02](./02-agent-loop.md)) |
| 논리 실패 | 같은 행동 반복, 진전 없음 | 반복 감지 후 중단하거나 전략 변경을 지시 |
| 프로세스 중단 | 배포, OOM, 노드 장애 | 체크포인트에서 재개 |

모델 API 호출의 일시적 오류는 OpenAI SDK의 내장 재시도(`max_retries`)로
충분하다. 직접 재시도 코드를 짜야 하는 곳은 사내 API 같은 **도구 호출** 쪽이다.

### 5. 사람 개입 = 일시정지 + 재개

승인이 필요한 행동을 만나면 이렇게 처리한다.

1. 상태를 `waiting_approval`로, 대기 중인 도구 호출을 `pending_call`로
   체크포인트에 저장한다
2. 승인 요청을 보내고 프로세스는 종료한다
3. 승인 이벤트가 오면 체크포인트를 읽고, 인자·승인 유효성을 확인한 뒤 도구를
   실행한다. **실제 실행 결과**를 tool result로 기록하고 재개한다. 거절이면
   실행하지 않고 거절 결과를 기록한다. 승인 자체를 실행 성공으로 기록하지 않는다

### 6. 여러 세션에 걸친 작업

컨텍스트 윈도우를 넘는 작업은 대화 이력만으로 이어갈 수 없다. Anthropic의 장기
실행 하네스처럼 **기능 목록, 진행 파일, git 커밋** 같은 외부 산출물을 상태로
삼는다. 각 세션은 깨끗한 상태(테스트 통과, 커밋 완료)로 끝내고 다음 세션용
인수인계를 남긴다 ([03](./03-context-engineering.md)의 progress.md).

## 어떻게 사용하는가? (How)

### Step 1. 원자적 체크포인트

```python
import json
import os
import tempfile


def save_checkpoint(path: str, state: dict) -> None:
    """임시 파일에 쓴 뒤 원자적으로 교체한다.

    쓰는 도중 죽어도 이전 체크포인트가 깨지지 않는다.
    """
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)
    os.replace(tmp, path)


def load_checkpoint(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# state 예시
state = {
    "run_id": "2026-09-11-a1b2",
    "turn": 12,
    "status": "running",        # running | waiting_approval | done | failed
    "pending_call": None,       # 승인 대기 중인 tool_call
    "messages": [],
}
```

루프에서는 매 턴 도구 결과를 `messages`에 추가한 직후 `save_checkpoint`를
호출한다. 시작할 때는 `load_checkpoint`로 이전 상태가 있는지 확인한다.

### Step 2. 도구 호출 재시도 (일시적 오류만)

```python
import random
import time


def retry(fn, transient=(TimeoutError, ConnectionError),
          attempts: int = 5, cap: float = 60):
    """일시적 오류만 지수 백오프로 재시도한다. 영구 오류는 바로 올린다."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except transient:
            if attempt == attempts:
                raise
            time.sleep(min(cap, 2 ** attempt) + random.random())
```

### Step 3. 멱등성 키를 받는 도구

```python
def create_ticket(title: str, body: str,
                  idempotency_key: str, store: dict) -> str:
    """같은 idempotency_key면 새로 만들지 않는다.

    store는 실제로는 DB 테이블이다 (key에 UNIQUE 제약).
    """
    if idempotency_key in store:
        return f"already created: {store[idempotency_key]}"
    ticket_id = f"T-{len(store) + 1}"  # 실제로는 외부 시스템 호출
    store[idempotency_key] = ticket_id
    return f"created: {ticket_id}"


store = {}
key = "run-a1b2:12:call_7"
assert create_ticket("x", "y", key, store) == "created: T-1"
# 재개로 같은 호출이 반복돼도 중복 생성되지 않는다
assert create_ticket("x", "y", key, store) == "already created: T-1"
```

이 예제는 단일 프로세스의 메모리 딕셔너리로 재호출만 보여준다. 실제 구현에서는
DB UNIQUE 제약이 필요하지만, **로컬 DB의 키만으로 외부 티켓 생성까지 exactly-once가
되지는 않는다.** 외부 시스템의 멱등성 지원이나 결과 조회·조정 절차가 필요하다.
애플리케이션 코드의 `if` 검사는 동시 실행 경쟁 조건에도 취약하다.

## 실패 지점을 나눠야 안전하게 재개할 수 있다

다음은 부수효과가 있는 작업의 설계 예다.

```text
planned → 승인 확인 → dispatched → confirmed → 검증 → completed
                         ↓ 응답 유실
                       unknown → 외부 결과 조회 → confirmed 또는 수동 조정
```

| 중단 지점 | 재개 시 행동 |
|---|---|
| 외부 요청 전 | 저장된 동일 작업 ID로 실행 |
| 요청 전달 후 응답 전 | 실패로 단정하지 않고 외부 결과 조회 |
| 성공 응답 후 체크포인트 전 | 멱등성 키로 기존 결과 회수 |
| 취소 요청 후 | 작업 중단 여부·이미 발생한 결과를 따로 확인 |

업무 멱등성 키는 실행 전에 영속화하고 재시도에도 그대로 사용한다. 모델을 다시
호출해 새 `tool_call_id`를 만들면 같은 업무가 새 키로 실행될 수 있다.
MCP에서도 연결 단절 뒤 새 요청 ID로 재발행하는 것과 업무 중복 방지는 별개다.
[2026-07-28 변경 기록](https://modelcontextprotocol.io/specification/2026-07-28/changelog)

타임아웃은 작업 실패의 증거가 아니다. 외부 시스템이 멱등성·조회 기능을 제공하지
않으면 결과가 불명확한 쓰기를 자동 재시도하지 말고 조정 대상으로 남긴다.
위 `retry` 예제도 이 조건을 만족하는 작업에만 적용한다.

### 원자적 교체와 영속성의 차이

`os.replace` 예제는 파일을 쓰다 죽을 때 부분 JSON이 현재 파일로 노출되는 문제를
줄인다. 전원 장애 뒤 데이터 보존, 여러 워커의 동시 쓰기, 외부 작업과의 원자성은
보장하지 않는다. 요구 수준에 따라 파일·디렉터리 동기화, DB 트랜잭션, 단일 쓰기
담당자를 선택한다. 체크포인트 버전과 마이그레이션 정책도 정한다.

### 장기 작업 프로토콜도 업무 판정기를 대체하지 않는다

MCP Tasks의 `completed`에는 `isError: true` 결과가 들어갈 수 있고, 취소는
협력적 요청이다. 프로토콜 상태, 도구 결과 성공 여부, 업무 검증 결과를 분리한다.
[Tasks 안정 버전](https://tasks.extensions.modelcontextprotocol.io/specification/2026-07-28/tasks)

장애 연습에서는 위 각 지점에서 프로세스를 중단한 뒤, 산출물 수·외부 ID·검증
기록을 확인한다. “이어 실행됐다”만으로 복구 검증을 끝내지 않는다.

## 학습 체크리스트

- [ ] 에이전트를 실행 중에 `kill`하고, 재시작했을 때 이어서 진행되는지 확인했다
- [ ] 외부 부수효과가 있는 도구를 모두 찾아 멱등성 키를 적용했다
- [ ] 실패 로그를 일시적 / 영구 / 모델 출력 / 논리 실패로 분류해봤다
- [ ] 승인 대기 → 프로세스 종료 → 승인 후 재개 흐름을 한 번 구현해봤다
- [ ] 체크포인트와 durable execution의 차이(부수효과 exactly-once 보장 여부)를
      설명할 수 있다

## 참고 자료 (References)

- [Durable
  execution](https://docs.langchain.com/oss/python/langgraph/durable-execution)
  — LangGraph 공식 문서, durability 모드
- [Why Checkpoints Aren't Durable
  Execution](https://www.diagrid.io/blog/checkpoints-are-not-durable-execution-why-langgraph-crewai-google-adk-and-others-fall-short-for-production-agent-workflows)
  — Diagrid, 체크포인트의 한계
- [12-Factor Agents](https://github.com/humanlayer/12-factor-agents) —
  HumanLayer, stateless reducer와 pause/resume
- [Effective harnesses for long-running
  agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
  — Anthropic, 세션 간 인수인계

## 관련 문서

- [02. 에이전트 루프](./02-agent-loop.md)
- [06. 가드레일과 권한 — 사람 승인](./06-guardrails-and-permissions.md)
- [08. 관측과 비용](./08-observability-and-cost.md)
