---
tags: [ai-coding, handoff, spec, ticket, compaction, clearing]
level: intermediate
last_updated: 2026-05-05
source: https://github.com/mattpocock/dictionary-of-ai-coding
---

# Section 5 — Handoffs (인계)

> 한 [Session](./02-sessions-context-windows-turns.md#session)이 감당하기엔 작업이 너무 클 때, 어떻게 **다음 세션으로 컨텍스트를 넘길 것인가**.

## 왜 이 섹션이 중요한가? (Why)

- 큰 작업을 한 세션에 통째로 욱여넣으면 결국 [dumb zone](./04-failure-modes.md#smart-zone)에 빠지고 만다.
- 그래서 **작업을 나누고 → 다음 세션으로 인계하고 → 새 smart zone에서 다시 시작**하는 패턴이 필수다.
- 이 섹션에서 다루는 용어들이 그 패턴을 이야기할 때 쓰는 어휘다.

## 용어 (What & How)

### Clearing (클리어링)

현재 [Session](./02-sessions-context-windows-turns.md#session)을 끝내고 **새 세션을 시작**하는 동작. 다음 메시지는 빈 세션과 빈 [Context window](./02-sessions-context-windows-turns.md#context-window) 위에서 새로 시작된다. 보통 사용자가 직접 트리거한다.

**💬 실전 대화 예시**
> "실패하는 테스트에서 루프에 갇혀 버렸어."
> "그냥 clear 하는 게 나아. 계획 문서랑 테스트 파일만 들고 새 세션을 시작해. 지금 [Context](./02-sessions-context-windows-turns.md#context)랑 계속 싸워봐야 의미가 없어."

---

### Handoff (핸드오프, 인계)

[Agent](./02-sessions-context-windows-turns.md#agent)의 [Context](./02-sessions-context-windows-turns.md#context)를 한 [Session](./02-sessions-context-windows-turns.md#session)에서 다른 세션으로 **되돌아오지 않는 방향으로** 넘기는 것. 운반 방식은 다양해서, 명시적으로 작성된 [Handoff artifact](#handoff-artifact)를 쓸 수도 있고, 인메모리 요약([Compaction](#compaction))을 쓸 수도 있다. 정보를 전혀 옮기지 않는 [Clearing](#clearing)과는 분명히 다르다. 핸드오프를 하는 이유도 다양한데, 역할 전환(planner → implementer), [AFK](./07-patterns-of-work.md#afk) 런 시작, 병렬 세션으로의 fan-out, [Context window](./02-sessions-context-windows-turns.md#context-window) 여유 공간 확보 등이 대표적이다.

**💬 실전 대화 예시**
> "계획 세션이 슬슬 무거워지는데, 그냥 이어서 갈까?"
> "Handoff 하는 게 좋아. 결정사항을 문서에 정리해 두고, clear 한 다음, 새 세션에서 그 문서를 읽으면서 implementation 단계로 넘어가."

🏢 **실무 적용**: 사내에서 **Spec → Ticket** 단위로 작업을 쪼개고, 각 ticket을 별도 세션에서 처리하는 워크플로우가 이 패턴과 그대로 맞물린다.

---

### Handoff artifact (핸드오프 아티팩트)

[Handoff](#handoff)의 **운반 매체 역할을 하는 문서**. 한 [Session](./02-sessions-context-windows-turns.md#session)이 작성하고, 다른 세션이 그것을 읽어들이는 구조다. 여러 운반 방식 중 하나이며, 다른 방식으로는 [Compaction](#compaction)이 있다.

**💬 실전 대화 예시**
> "기획 [Agent](./02-sessions-context-windows-turns.md#agent)랑 구현 agent를 어떻게 나누는 게 좋아?"
> "Planner가 handoff artifact를 쓰게 만들어 줘. 파일 경로, 결정사항, 제약 조건 같은 걸 거기에 정리하는 거지. Implementer 세션은 그 artifact를 시작점으로 잡고, 그걸 브리핑 삼아서 일을 시작하면 돼."

---

### Spec (스펙)

**여러 [Session](./02-sessions-context-windows-turns.md#session)에 걸쳐 진행되는 작업 전체**를 기술하는 [Handoff artifact](#handoff-artifact). spec에는 *무엇을 만들 것인가*가 담기지만, 각 세션이 *그 일부를 어떻게 처리할지*까지는 다루지 않는다. 작업이 진행되면서 spec도 함께 변형된다. 내부적으로는 **여러 [Tickets](#ticket)으로 구성**된다.

**💬 실전 대화 예시**
> "이거 한 세션으로 다 처리할 수 있을까?"
> "그건 무리야. spec으로 적어 두고, 거기서 ticket으로 쪼개서 각각 별도 세션에서 돌려. 한 [Context](./02-sessions-context-windows-turns.md#context) 안에 다 욱여넣으려고 하면, 절반 가기도 전에 [dumb zone](./04-failure-modes.md#smart-zone)에 빠질 거야."

---

### Ticket (티켓)

**한 [Session](./02-sessions-context-windows-turns.md#session) 분량의 작업**으로 스코프된 [Handoff artifact](#handoff-artifact). 단독으로 존재할 수도 있고, [Spec](#spec) 아래에 자식으로 매달려 있을 수도 있다. 형제 ticket들 사이에는 **block / blocked-by 관계**가 걸릴 수 있기 때문에, 작업 순서는 단순한 선형 계획이 아니라 **의존성 그래프**에서 자연스럽게 도출된다.

**💬 실전 대화 예시**
> "마이그레이션 spec, 어디서부터 시작해야 하지?"
> "Ticket 그래프부터 보자. 스키마 변경이 backfill을 막고 있고, backfill은 다시 API 전환을 막고 있네. Leaf에 있는 ticket 하나를 골라서, 거기에 한 세션을 붙여 돌리는 게 좋아."

---

### Compaction (컴팩션)

**인메모리 형태로 수행하는 [Handoff](#handoff)**. 이전 [Session](./02-sessions-context-windows-turns.md#session)의 히스토리를 요약한 뒤, 그 요약을 새 세션의 시드로 삼는 방식이다. 본질적으로 **lossy(손실이 있는)** 작업이다. 디테일을 잃는 대신 컨텍스트 헤드룸을 확보하는 일종의 트레이드오프라고 보면 된다. 사용자가 수동으로 트리거할 수도 있고, [자동으로](#autocompact) 트리거될 수도 있다.

**💬 실전 대화 예시**
> "[Context](./02-sessions-context-windows-turns.md#context)가 슬슬 무거워지는데, 아직 테스트 통과 작업이 남아 있어."
> "이어서 시작하기 전에 한 번 compact 하는 게 좋아. 꼭 유지해야 할 부분(예: 스키마 결정 같은 것)을 요약 프롬프트에 명시해 두고, 새 세션이 그건 유지하면서 탐색 흔적은 버리도록 잡아 줘."

---

### Autocompact (자동 컴팩션)

[Harness](./01-the-model.md#harness)가 [Context window](./02-sessions-context-windows-turns.md#context-window)가 거의 가득 찼다고 판단했을 때 **자동으로 발동시키는** [Compaction](#compaction).

**💬 실전 대화 예시**
> "아까 결정한 스키마 내용을 잊어버린 것 같아."
> "[Turn](./02-sessions-context-windows-turns.md#turn) 사이에 autocompact가 한 번 발동했네. 초기 결정이 요약 안에 묻히면서 일부가 빠진 거야. 계획 문서를 다시 로드하든가, 아니면 다음번에는 수동으로 compact해서 *무엇을 유지할지를* 네가 직접 통제해."

🏢 **실무 적용**: Autocompact는 편리하지만 **정확히 무엇이 잘려나갔는지를 알 수 없다**는 단점이 있다. 중요한 작업에서는 수동 compaction이나 명시적인 handoff artifact를 쓰는 쪽이 안전하다.

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
