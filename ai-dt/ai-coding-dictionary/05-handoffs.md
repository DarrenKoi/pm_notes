---
tags: [ai-coding, handoff, spec, ticket, compaction, clearing]
level: intermediate
last_updated: 2026-05-05
source: https://github.com/mattpocock/dictionary-of-ai-coding
---

# Section 5 — Handoffs (인계)

> 한 [Session](./02-sessions-context-windows-turns.md#session)이 감당하기엔 작업이 너무 클 때, 어떻게 **다음 세션으로 컨텍스트를 넘길 것인가**.

## 왜 이 섹션이 중요한가? (Why)

- 큰 작업을 한 세션에 욱여넣으면 [dumb zone](./04-failure-modes.md#smart-zone)에 빠진다.
- 그래서 **나누고 → 인계하고 → 새 smart zone에서 시작**하는 패턴이 필수.
- 이 섹션의 용어들이 그 패턴의 어휘다.

## 용어 (What & How)

### Clearing (클리어링)

현재 [Session](./02-sessions-context-windows-turns.md#session)을 끝내고 **새 세션을 시작**하는 것. 다음 메시지는 빈 세션과 빈 [Context window](./02-sessions-context-windows-turns.md#context-window)로 시작한다. 보통 사용자가 트리거.

**💬 실전 대화 예시**
> "실패하는 테스트에서 루프에 갇혔어."
> "그냥 clear 해 — 계획 문서랑 테스트 파일 들고 새 세션 시작해. 기존 [Context](./02-sessions-context-windows-turns.md#context)랑 싸워봤자 의미 없어."

---

### Handoff (핸드오프, 인계)

[Agent](./02-sessions-context-windows-turns.md#agent) [Context](./02-sessions-context-windows-turns.md#context)를 한 [Session](./02-sessions-context-windows-turns.md#session)에서 다른 세션으로 **돌아오지 않게** 옮기는 것. 운반 메커니즘은 다양 — 작성된 [Handoff artifact](#handoff-artifact), 인메모리 요약([Compaction](#compaction)), 등등. [Clearing](#clearing)과 다름(전혀 옮기지 않음). 이유도 다양: 역할 전환(planner → implementer), [AFK](./07-patterns-of-work.md#afk) 런 시작, 병렬 세션 fan-out, [Context window](./02-sessions-context-windows-turns.md#context-window) 공간 확보.

**💬 실전 대화 예시**
> "계획 세션이 무거워지고 있어 — 그냥 계속 갈까?"
> "Handoff 해. 결정사항을 문서에 적고, clear 하고, 새 세션을 그 문서 읽으면서 implementation으로 시작해."

🏢 **실무 적용**: 사내에서 **Spec → Ticket** 단위로 작업을 쪼개고, 각 ticket을 별도 세션에서 처리하는 패턴이 그대로 적용된다.

---

### Handoff artifact (핸드오프 아티팩트)

[Handoff](#handoff)의 **운반 매체로 쓰이는 문서** — 한 [Session](./02-sessions-context-windows-turns.md#session)이 작성하고, 다른 세션이 읽는다. 여러 운반 방법 중 하나(다른 방법: [Compaction](#compaction)).

**💬 실전 대화 예시**
> "기획 [Agent](./02-sessions-context-windows-turns.md#agent)와 구현 agent를 어떻게 나눠?"
> "Planner가 handoff artifact를 쓰게 해 — 파일 경로, 결정, 제약. Implementer 세션은 그 artifact를 가리키며 시작해서, 그걸 브리핑으로 일해."

---

### Spec (스펙)

**여러 [Session](./02-sessions-context-windows-turns.md#session)에 걸친 작업**을 기술하는 [Handoff artifact](#handoff-artifact) — *무엇을 만드는가*, 각 세션이 *어떻게 그 일부를 처리하는가는 아님*. 작업이 진행되며 변형된다. **[Tickets](#ticket)으로 구성**된다.

**💬 실전 대화 예시**
> "이거 한 세션으로 다 할 수 있을까?"
> "안 돼, spec으로 적어 — ticket으로 쪼개서 각각 자기 세션에서 돌려. 한 [Context](./02-sessions-context-windows-turns.md#context)에 다 넣으려고 하면 절반도 가기 전에 [dumb zone](./04-failure-modes.md#smart-zone) 만나."

---

### Ticket (티켓)

**한 [Session](./02-sessions-context-windows-turns.md#session) 분량의 작업**을 스코프하는 [Handoff artifact](#handoff-artifact). 단독으로 서거나, [Spec](#spec)에 자식으로 매달려 있거나. 형제 ticket 사이에 **block / blocked-by 관계**가 있을 수 있어, 작업 순서가 선형 계획이 아니라 **의존성 그래프**에서 떨어진다.

**💬 실전 대화 예시**
> "마이그레이션 spec 어디서부터 시작해?"
> "Ticket 그래프 봐 — 스키마 변경이 backfill을 막고, backfill이 API 전환을 막아. Leaf 하나 골라서 거기에 한 세션 돌려."

---

### Compaction (컴팩션)

**인메모리로 하는 [Handoff](#handoff)**: 이전 [Session](./02-sessions-context-windows-turns.md#session)의 히스토리를 요약해서 새 세션의 시드로 삼는다. **Lossy(손실 있음)** — 디테일을 헤드룸과 맞바꾸는 거래. 사용자가 수동으로 트리거하거나 [자동으로](#autocompact) 트리거됨.

**💬 실전 대화 예시**
> "[Context](./02-sessions-context-windows-turns.md#context)가 무거워지는데 아직 테스트 통과 작업이 남았어."
> "시작 전에 compact해 — 부담 가는 부분(스키마 결정 같은 것)을 요약 프롬프트에 적어두고, 새 세션이 그걸 유지하고 탐색은 버리도록."

---

### Autocompact (자동 컴팩션)

[Harness](./01-the-model.md#harness)가 [Context window](./02-sessions-context-windows-turns.md#context-window)가 거의 찼을 때 **자동으로 트리거**하는 [Compaction](#compaction).

**💬 실전 대화 예시**
> "아까 결정한 스키마 내용을 기억 못 하는 것 같아."
> "[Turn](./02-sessions-context-windows-turns.md#turn) 사이에 autocompact가 발동했네 — 초기 결정이 요약돼버려서 뭔가 잃어버린 거야. 계획 문서 다시 로드하든가, 다음엔 수동으로 compact해서 *무엇을 유지할지* 네가 통제해."

🏢 **실무 적용**: Autocompact는 편하지만 **무엇이 잘려나갔는지 모른다**. 중요 작업에서는 수동 compaction(또는 명시적 handoff artifact)을 권장.

## 이 섹션 요약 (Cheatsheet)

```
[큰 작업]
   │
   ├─ Spec 작성 ── (여러 세션 분량)
   │    │
   │    ├─ Ticket 1 ─ Session A (처음부터 smart zone)
   │    ├─ Ticket 2 ─ Session B
   │    └─ Ticket 3 ─ Session C
   │
   └─ 세션 한 번이 무거워질 때:
        ├─ Clearing       (전혀 안 옮김 — 그냥 시작)
        ├─ Handoff artifact (문서로 옮김 — 통제 가능, 권장)
        └─ Compaction       (요약으로 옮김 — 자동/편함, lossy)
```

| 옵션 | 운반 매체 | 통제 가능성 |
|---|---|---|
| Clearing | (없음) | n/a — 그냥 처음부터 |
| Handoff artifact | 작성된 문서 | 높음 — 무엇을 적을지 명시적 |
| Compaction | 인메모리 요약 | 중간 — 무엇을 유지할지 프롬프트로 가이드 |
| Autocompact | 인메모리 요약 (자동) | 낮음 — 하네스가 알아서 |

## 관련 문서

- 이전: [04 - Failure Modes](./04-failure-modes.md)
- 다음: [06 - Memory and Steering](./06-memory-and-steering.md)
- 인덱스: [README](./README.md)

## 참고 자료 (References)

- 원문: [mattpocock/dictionary-of-ai-coding — Section 5](https://github.com/mattpocock/dictionary-of-ai-coding#section-5--handoffs)
