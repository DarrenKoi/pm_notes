---
tags: [ai-coding, afk, vibe-coding, grilling, human-in-the-loop, automated-review]
level: intermediate
last_updated: 2026-05-05
source: https://github.com/mattpocock/dictionary-of-ai-coding
---

# Section 7 — Patterns of Work (작업 패턴)

> "Vibe coding", "AFK", "Grilling" — 사람들이 AI 에이전트와 **실제로 어떻게 일하는가**의 패턴들.

## 왜 이 섹션이 흥미로운가? (Why)

- 같은 도구를 쓰더라도 **누가 어디까지 들여다보느냐**에 따라 결과물의 품질이 크게 달라진다.
- 이 섹션은 사람과 에이전트가 협업할 때 취하는 "자세"에 이름을 붙인다. 어떤 자세는 빠른 대신 위험하고, 어떤 자세는 느린 대신 안전하다.

## 용어 (What & How)

### Human-in-the-loop (HITL, 휴먼 인 더 루프)

[Session](./02-sessions-context-windows-turns.md#session) 동안 한 명 이상의 인간이 [Agent](./02-sessions-context-windows-turns.md#agent)와 **실시간으로 짝을 이루어** 함께 작업하는 패턴. 리뷰, 방향 재조정, 협업이 실시간으로 일어난다. 사람이 단순히 개별 동작을 게이팅만 하는 수준이 아니라, **현장에 직접 자리하고 적극적으로 관여한다**는 점이 특징이다.

**💬 실전 대화 예시**
> "이거 [AFK](#afk)로 밤새 돌릴까?"
> "아니, 이건 스키마 마이그레이션이야. human-in-the-loop를 유지하자. 매 단계 직접 보면서, 잘못된 컬럼에 backfill하려고 하면 바로 끼어들고 싶어."

---

### AFK (Away From Keyboard, 자리 비움)

사용자가 [Session](./02-sessions-context-windows-turns.md#session)만 시작해 두고, [Agent](./02-sessions-context-windows-turns.md#agent)가 **무인으로 알아서 돌도록 내버려 두는** 작업 패턴. AI 코딩에서 처리량을 몇 배로 뻥튀기해 주는 핵심 패턴이라고도 할 수 있다(throughput multiplier). 사용자가 자거나, 식사하거나, 다른 일을 하는 동안 **여러 AFK 세션을 병렬로** 돌리는 게 가능하기 때문이다. 안전을 위해 보통은 느슨한 [Permission mode](./03-tools-environment.md#permission-mode)와 [Sandbox](./03-tools-environment.md#sandbox)를 함께 묶어서 사용한다.

**❌ 피할 표현**: "background agent". 이 말은 "백그라운드에서 돌고 있다"는 *기계 관점*에 가깝지, "사용자가 자리를 비웠다"는 *사람 관점*을 담고 있지 않다. **AFK의 핵심은 어디까지나 "사용자가 지금 보고 있지 않다"는 사실**이다.

**💬 실전 대화 예시**
> "이거 AFK로 돌리자. sandbox 띄운 에이전트 3개가 리팩토링하고, 나는 아침에 일어나서 PR 리뷰만 하면 돼."
> "[Bypass permissions](./03-tools-environment.md#agent-mode) 모드로 갈 거야?"
> "응. 대신 [filesystem](./03-tools-environment.md#filesystem)은 read-only로 잡고, 외부망은 차단해 둘 거야."

🏢 **실무 적용**: 사내 폐쇄망에서도 같은 패턴이 충분히 가능하다. 격리된 컨테이너에 read-only DB credential만 주고 인터넷을 차단해 두면, 밤새 여러 작업을 병렬로 돌릴 수 있다. 단, **automated check / review** 게이트가 제대로 걸려 있어야 안전하게 굴릴 수 있다.

---

### Automated check (자동화 체크)

[Environment](./03-tools-environment.md#environment) 안에서 돌아가는 **결정론적(deterministic) 검증**. 테스트, 타입체크, 린트, 빌드, pre-commit 훅 등이 여기에 해당한다. **결과는 pass/fail로만 나오고, 그 안에 어떤 판단도 들어가지 않는다.** [Agent](./02-sessions-context-windows-turns.md#agent)가 사람을 거치지 않고도 스스로 교정에 활용할 수 있는 신호 역할을 한다. **flaky test는 "깨진 체크"이지, "체크가 아닌 것"이 아니다.** automated check는 *설계상* 결정론적인 게 정상이기 때문이다.

**❌ 피할 표현**: "feedback loop"나 "backpressure" 같은 표현. 둘 다 [Review](#automated-review)와 한 묶음으로 뭉뚱그려 버리는 단점이 있다. 또 "test"라는 표현도 주의해야 한다. 테스트는 automated check의 한 종류일 뿐, 모든 automated check가 테스트는 아니다.

**💬 실전 대화 예시**
> "[AFK](#afk) 런에서 에이전트가 자꾸 망가진 코드를 PR로 올려."
> "[Sandbox](./03-tools-environment.md#sandbox) 쪽에는 어떤 automated check가 걸려 있어?"
> "지금은 그냥 unit test 하나뿐이야."
> "Typecheck랑 lint도 같이 추가해. PR로 올라오기 전에, 그 단계에서부터 에이전트가 스스로 교정할 수 있도록."

---

### Automated review (자동화 리뷰)

[Agent](./02-sessions-context-windows-turns.md#agent)가 **다른 에이전트의 결과물을 리뷰**하는 패턴. 리뷰어 쪽에는 보통 다른 [Model](./01-the-model.md#model)이나 다른 [System prompt](./02-sessions-context-windows-turns.md#system-prompt)를 쓰는 경우가 많다. **비결정론적**이고, 결과적으로 일종의 판단을 만들어 낸다. 돌리는 위치는 자유로워서, PR merge 직전에도, 커밋 히스토리에 대한 post-hoc 리뷰로도, 세션 도중 [subagent](./06-memory-and-steering.md#subagent)로도 가능하다. **CI에 LLM-as-judge가 들어 있다면 그건 [automated check](#automated-check)가 아니라 automated review에 해당한다.** 분류 기준은 "어디서 도는가"가 아니라 "*assertion이 무엇을 하는가*"이기 때문이다.

**❌ 피할 표현**: "AI review"나 "agent review" 같은 표현. 일을 수행하는 에이전트와 구분이 가지 않을 만큼 모호하다.

**💬 실전 대화 예시**
> "[AFK](#afk) 런에서 올라오는 PR 품질이 너무 안 좋아."
> "Merge 전에 automated review 단계를 하나 추가하자. 다른 model, 별도 system prompt를 쓰되, 보안 이슈랑 contract 변경 쪽에만 스코프해서."

---

### Human review (휴먼 리뷰)

[Agent](./02-sessions-context-windows-turns.md#agent)가 만든 코드를 **사용자가 직접 눈으로 읽고 판단하는 행위**. 어디까지나 **diff나 실제 변경 파일을 직접 봐야 리뷰로 인정**된다. 에이전트가 작성한 *서술*만 읽는 건 리뷰로 카운트되지 않는다. **서술은 산출물이 아니다(narration is not the artifact).**

**❌ 피할 표현**: 그냥 "code review"라고만 부르는 것은 피한다. human review인지 [automated review](#automated-review)인지 구분이 흐려진다.

**💬 실전 대화 예시**
> "[AFK](#afk) 결과 휴먼 리뷰했어."
> "Diff를 직접 봤어, 아니면 요약만 봤어?"
> "Diff까지 다 봤어. 요약에는 *dead code를 지웠다*고 적혀 있었는데, 알고 보니 그 함수가 generated 파일에서 호출되고 있더라고."

🏢 **실무 적용**: 사내 PR 리뷰에서도 똑같은 원칙이 적용된다. *"Claude가 짠 코드인데 잘 됐다고 했어요"* 가 진짜 리뷰가 아니라, **변경 diff를 직접 들여다보는 행위**가 진짜 리뷰다. 에이전트의 자기 서술은 검증이 될 수 없다.

---

### Vibe coding (바이브 코딩)

사용자가 [Agent](./02-sessions-context-windows-turns.md#agent)의 코드를 **[Human review](#human-review) 없이 그대로 받아들이는** 작업 패턴. Diff는 **불투명한 블랙박스**처럼 다룬다. 중요한 건 프로그램이 *제대로 동작하느냐*뿐이고, *내부에 무엇이 있는지*는 따지지 않는다는 자세다. 이때도 [Automated review](#automated-review)나 [Automated check](#automated-check)는 여전히 따로 돌 수 있다. vibe coding 자체는 그 부분에 대해서는 따로 규정하지 않는다.

**❌ 피할 표현**: "vibe coding"을 "저품질 AI 코딩"의 동의어처럼 사용하지 말 것. 이 용어는 어디까지나 **리뷰 태도**에 붙이는 이름이지, 결과물 자체를 깎아내리는 표현이 아니다.

**💬 실전 대화 예시**
> "Auth flow에서 뭘 바꿨는지 직접 읽어봤어?"
> "Vibe coding이라 안 봤어. 로그인되는 것만 확인했지."
> "Push 전에 diff는 한 번 봐. auth에서 vibe로 넘기는 건 secret이 로그에 새 나가는 지름길이야."

🏢 **실무 적용**: 프로토타입이나 사이드 프로젝트에서는 vibe coding이 빠르고 효율적인 선택일 수 있다. 다만 **auth, secrets, DB 마이그레이션** 같은 민감한 영역에서는 반드시 human review를 거쳐야 한다.

---

### Design concept (디자인 컨셉)

**무엇을 만들 것인지에 대해 사용자와 [Agent](./02-sessions-context-windows-turns.md#agent)가 공유하고 있는 머릿속의 그림**. 양쪽 모두에게 자리 잡고 있는 것이지만, 어떤 구체적 자산과도 동일시할 수는 없다. Frederick Brooks가 *The Design of Design*에서 사용한 용어로, 대화도, [Handoff artifacts](./05-handoffs.md#handoff-artifact)도, 코드도 결국은 이 design concept을 *포착하거나 그것에 가까워지려는 자산*일 뿐, 그 자체는 아니다. 어느 것도 design concept *자체*가 되지는 못한다는 점이 핵심이다. **Design concept의 품질은, 그것을 만들어낸 대화의 품질로 체감된다.**

**💬 실전 대화 예시**
> "내가 부탁한 그대로 짜고 있다는데도 결과는 계속 어긋나."
> "아직 사용자와 에이전트 사이에 design concept이 제대로 공유되지 않아서 그래. 빈 곳을 가정으로 메우고 있는 상태야. 취소, 환불, 부분 이행 같은 개념이 너와 에이전트 양쪽에서 똑같은 의미가 될 때까지 대화를 더 나누자. 그 전에 [spec](./05-handoffs.md#spec) 쓰는 단계로 넘어가게 두지 말고."

---

### Grilling (그릴링)

[Agent](./02-sessions-context-windows-turns.md#agent)와 [Design concept](#design-concept)을 함께 다듬어 가는 기법. **에이전트가 사용자를 소크라테스식으로 인터뷰**하는 방식이다. 한 번에 결정 하나씩만 다루되, 각 결정마다 에이전트가 자기 추천안을 함께 제시한다. 핵심은 **완성된 계획으로 곧장 돌진하는 흐름을 의도적으로 늦추는 것**이다. concept이 어느 정도 안정되기 전에는 [Handoff artifact](./05-handoffs.md#handoff-artifact) 작성으로 넘어가지 않는다.

**💬 실전 대화 예시**
> "곧장 [spec](./05-handoffs.md#spec)부터 쓰더니 cancellation 로직을 완전히 망쳐 놨어."
> "다음엔 먼저 grilling부터 시키자. partial cancel, 환불, 타이밍 같은 걸 문서에 박아 넣기 전에 *너한테* 먼저 묻도록 만들어. 코드 단계에서 푸는 것보다 대화 단계에서 푸는 게 훨씬 싸."

🏢 **실무 적용**: Spec/RFC 작성에 들어가기 전에 **에이전트가 PM이나 엔지니어를 직접 인터뷰**하게 하는 패턴이라고 보면 된다. "잠깐, 이 케이스는 어떻게 처리할까요?" 같은 질문을 5~10개 정도 받고 거기에 답하고 나면, 이어지는 spec 품질이 눈에 띄게 좋아진다.

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
