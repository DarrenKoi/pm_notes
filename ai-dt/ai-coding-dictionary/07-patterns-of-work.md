---
tags: [ai-coding, afk, vibe-coding, grilling, human-in-the-loop, automated-review]
level: intermediate
last_updated: 2026-05-05
source: https://github.com/mattpocock/dictionary-of-ai-coding
---

# Section 7 — Patterns of Work (작업 패턴)

> "Vibe coding", "AFK", "Grilling" — 사람들이 AI 에이전트와 **실제로 어떻게 일하는가**의 패턴들.

## 왜 이 섹션이 흥미로운가? (Why)

- 같은 도구라도 **누가 어디까지 들여다보느냐**에 따라 결과 품질이 천차만별이다.
- 이 섹션은 "사람-에이전트 협업의 자세"에 이름을 붙인다 — 어떤 자세는 빠르지만 위험하고, 어떤 자세는 느리지만 안전하다.

## 용어 (What & How)

### Human-in-the-loop (HITL, 휴먼 인 더 루프)

[Session](./02-sessions-context-windows-turns.md#session) 동안 한 명 이상의 인간이 [Agent](./02-sessions-context-windows-turns.md#agent)와 **함께 짝을 이루는** 작업 패턴 — 리뷰, 재방향, 협업을 실시간으로. 인간은 **현장에 있고 관여한다** — 개별 동작을 게이팅만 하는 게 아니라.

**💬 실전 대화 예시**
> "이거 [AFK](#afk)로 밤새 돌릴까?"
> "아니, 스키마 마이그레이션이야 — human-in-the-loop 유지해. 매 단계 보고, 잘못된 컬럼에서 backfill하려고 하면 내가 끼어들고 싶어."

---

### AFK (Away From Keyboard, 자리 비움)

사용자가 [Session](./02-sessions-context-windows-turns.md#session)을 시작해 놓고 [Agent](./02-sessions-context-windows-turns.md#agent)가 **무인으로 돌게 두는** 작업 패턴. **AI 코딩의 처리량 배수**(throughput multiplier) — 자고 있거나, 식사 중이거나, 다른 일 하는 동안 **여러 AFK 세션을 병렬로** 돌릴 수 있다. 보통 느슨한 [Permission mode](./03-tools-environment.md#permission-mode) + [Sandbox](./03-tools-environment.md#sandbox) 조합이 안전을 위해 필요하다.

**❌ 피할 표현**: "background agent" — 기계 중심이지("백그라운드에서 돌고 있다") 인간 패턴 중심이 아니다("사용자가 자리를 비웠다"). **AFK의 핵심 사실은: 사용자가 보고 있지 않다는 것**.

**💬 실전 대화 예시**
> "이거 AFK로 돌려 — sandbox 친 에이전트 3개가 리팩토링하고, 나는 아침에 PR 리뷰할 거야."
> "[Bypass permissions](./03-tools-environment.md#agent-mode)?"
> "응, read-only [filesystem](./03-tools-environment.md#filesystem), 외부망 차단."

🏢 **실무 적용**: 사내 폐쇄망에서도 같은 패턴이 가능 — 격리된 컨테이너에서 read-only DB credential, 인터넷 차단, 그러면 밤새 여러 작업을 병렬로 돌릴 수 있다. 단, **automated check / review** 게이트가 잘 걸려 있어야 안전하다.

---

### Automated check (자동화 체크)

[Environment](./03-tools-environment.md#environment)에서 돌아가는 **결정론적(deterministic) 검증** — 테스트, 타입체크, 린트, 빌드, pre-commit 훅. **Pass/fail, 판단 없음**. [Agent](./02-sessions-context-windows-turns.md#agent)가 누군가를 끼우지 않고 자체 교정할 수 있는 신호. **flaky test는 *깨진 체크*이지 *체크가 아닌 것*이 아니다** — automated check는 *설계상* 결정론적이다.

**❌ 피할 표현**: "feedback loop" / "backpressure" — 둘 다 [Review](#automated-review)와 한 데 묶어버린다. "test" — 테스트는 automated check이지만, 모든 automated check가 테스트는 아니다.

**💬 실전 대화 예시**
> "[AFK](#afk) 런에서 에이전트가 자꾸 망가진 코드 보내."
> "[Sandbox](./03-tools-environment.md#sandbox)에 어떤 automated check가 걸려 있어?"
> "그냥 unit test."
> "Typecheck하고 lint 추가해 — PR 떨어지기 전에 거기서부터 자체 교정할 거야."

---

### Automated review (자동화 리뷰)

[Agent](./02-sessions-context-windows-turns.md#agent)가 **다른 에이전트의 결과물을 리뷰**하는 것. 종종 다른 [Model](./01-the-model.md#model)이나 [System prompt](./02-sessions-context-windows-turns.md#system-prompt)를 쓴다. **비결정론적**: 판단을 만든다. 어디서나 돌 수 있음 — PR pre-merge, 커밋 히스토리 post-hoc, 세션 도중 [subagent](./06-memory-and-steering.md#subagent)로. **CI에 LLM-as-judge가 있다면 그건 automated review이지 [automated check](#automated-check)가 아니다** — 어디서 도느냐가 아니라 *assertion이 무엇을 하느냐*가 분류 기준.

**❌ 피할 표현**: "AI review" / "agent review" — 일하는 에이전트와 구분이 안 될 만큼 모호함.

**💬 실전 대화 예시**
> "[AFK](#afk) 런에서 나오는 PR 품질이 너무 안 좋아."
> "Merge 전에 automated review 단계 추가해 — 다른 model, 별도 system prompt, 보안과 contract 변경에 스코프해서."

---

### Human review (휴먼 리뷰)

[Agent](./02-sessions-context-windows-turns.md#agent)가 만든 코드를 **사용자가 직접 읽고 판단**하는 것. **diff나 변경 파일을 읽어야 카운트**된다. 에이전트의 *서술*만 읽는 건 카운트 안 된다 — **narration is not the artifact (서술은 산출물이 아니다)**.

**❌ 피할 표현**: 그냥 "code review"라고만 쓰지 말 것 — human인지 [automated](#automated-review)인지 모호.

**💬 실전 대화 예시**
> "[AFK](#afk) 결과 휴먼 리뷰했어."
> "Diff를 봤어, 아니면 요약만?"
> "Diff. 요약은 *dead code 지웠다*고 했는데, 알고 보니 그 함수가 generated 파일에서 호출되고 있었어."

🏢 **실무 적용**: 사내 PR 리뷰에서도 동일 — *"Claude가 짠 코드인데 잘 됐다고 했어요"* 가 아니라, **변경 diff를 직접** 보는 게 진짜 리뷰. 에이전트의 자기 서술은 검증이 아니다.

---

### Vibe coding (바이브 코딩)

사용자가 [Agent](./02-sessions-context-windows-turns.md#agent)의 코드를 **[Human review](#human-review) 없이 받아들이는** 작업 패턴. Diff를 **불투명**하게 다룬다 — 중요한 건 프로그램이 *동작하느냐*이지 *그 안이 뭐냐*가 아니다. [Automated review](#automated-review)와 [Automated checks](#automated-check)는 여전히 돌 수 있다 — vibe coding은 그것들에 대해선 침묵.

**❌ 피할 표현**: "vibe coding"을 "저품질 AI 코딩"의 동의어로 쓰지 말 것 — 이 용어는 **리뷰 자세**에 이름을 붙이는 거지, 결과 코드 자체를 비하하는 게 아니다.

**💬 실전 대화 예시**
> "Auth flow에서 뭘 바꿨는지 읽어봤어?"
> "Vibe coded — 로그인 되더라, 그것만 확인했어."
> "Push 전에 diff 봐, auth에서 vibe하는 게 secret이 로그에 새는 길이야."

🏢 **실무 적용**: 프로토타입·사이드 프로젝트에서는 vibe coding이 빠르고 효율적. 단, **auth/secrets/DB 마이그레이션** 같은 영역에서는 항상 human review 필수.

---

### Design concept (디자인 컨셉)

**무엇을 만드는가에 대한 공유된 이해** — 사용자와 [Agent](./02-sessions-context-windows-turns.md#agent) 사이에 공통으로 자리잡는 것이지만 어떤 자산과도 별개. Brookes의 용어(*The Design of Design*): 대화, [Handoff artifacts](./05-handoffs.md#handoff-artifact), 코드 모두 이 design concept을 *포착하거나 도달하려는 자산*이지만, 어느 것도 design concept *그 자체*는 아니다. **Design concept의 품질은 그것을 만든 대화의 품질로 체감된다**.

**💬 실전 대화 예시**
> "내가 부탁한 그대로 짜고 있는데도 여전히 틀려."
> "아직 design concept을 공유 못 했어 — 가정으로 빈 곳을 메우고 있는 거야. 취소·환불·부분 이행이 너랑 에이전트 사이에 다 같은 의미가 될 때까지 대화 더 해, [spec](./05-handoffs.md#spec) 쓰게 두지 말고."

---

### Grilling (그릴링)

[Agent](./02-sessions-context-windows-turns.md#agent)와 [Design concept](#design-concept)을 만들어가는 기법: **에이전트가 사용자를 소크라테스식으로 인터뷰** — 한 번에 결정 하나씩, 각 결정마다 추천 답을 제안한다. **완성된 계획으로 가는 돌진을 늦춘다** — concept이 안정될 때까지 [Handoff artifact](./05-handoffs.md#handoff-artifact)를 쓰지 않는다.

**💬 실전 대화 예시**
> "곧장 [spec](./05-handoffs.md#spec)을 쓰러 가더니 cancellation 로직을 망쳐놨어."
> "먼저 grilling 시켜 — partial cancel, 환불, 타이밍을 문서에 commit하기 전에 *너에게* 묻게 해. 코드보다 대화에서 푸는 게 싸."

🏢 **실무 적용**: Spec/RFC 작성 전에 **에이전트가 PM/엔지니어를 인터뷰**하게 하는 패턴. "잠깐, 이 케이스는 어떻게 처리할까요?" 류의 질문을 5~10개 받고 답하면, 그 다음 spec 품질이 압도적으로 올라간다.

## 이 섹션 요약 (Cheatsheet)

```
[리뷰 강도 스펙트럼]

가장 강함 ─────────────────────────────────── 가장 약함
   │                                                │
[Human-in-the-loop]   [Human review]            [Vibe coding]
   매 단계 함께         완성 후 diff 읽음         diff 안 봄
                          │                       │
                          ├─ Automated review (LLM judge)
                          └─ Automated checks (test/lint/typecheck)
```

```
[작업 패턴]
                    AFK     ↔     Human-in-the-loop
        (1명이 N개 병렬, 무인)   (1명이 1개, 함께)

[Design 단계 패턴]
                    Grilling  →  Spec  →  Tickets  →  세션들
              (에이전트가 PM 역할)  (문서)   (분할)   (실행)
```

| 용어 | 한 줄 |
|---|---|
| HITL | 함께 일함 |
| AFK | 자리 비움, 병렬 다중 세션 |
| Automated check | 결정론적 검증 (test/lint) |
| Automated review | LLM-as-judge (비결정론적 판단) |
| Human review | diff를 직접 읽음 |
| Vibe coding | diff 안 읽음, 동작만 봄 |
| Design concept | 공유된 이해 (어떤 자산도 그 자체 아님) |
| Grilling | 에이전트가 사용자를 소크라테스식 인터뷰 |

## 관련 문서

- 이전: [06 - Memory and Steering](./06-memory-and-steering.md)
- 인덱스: [README](./README.md)
- 사내 연결: 코드 리뷰 컨벤션, PR 워크플로우 — 사내 가이드와 매핑

## 참고 자료 (References)

- 원문: [mattpocock/dictionary-of-ai-coding — Section 7](https://github.com/mattpocock/dictionary-of-ai-coding#section-7--patterns-of-work)
- Frederick Brooks, *The Design of Design* (Design concept의 출처)
