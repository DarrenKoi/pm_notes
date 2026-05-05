---
tags: [ai-coding, hallucination, attention, knowledge-cutoff, sycophancy]
level: intermediate
last_updated: 2026-05-05
source: https://github.com/mattpocock/dictionary-of-ai-coding
---

# Section 4 — Failure Modes (실패 양상)

> AI 에이전트가 **이상하게 굴 때**, 그 이상함에 정확히 이름을 붙이는 용어들. 이 섹션이 일상 디버깅의 80%를 차지한다.

## 왜 이 섹션이 가장 자주 쓰이는가? (Why)

- "헛소리한다 / 자꾸 까먹는다 / 갑자기 멍청해진다 / 내 말에 무조건 동의한다" — 이 네 가지가 일상이고, 각각 다른 원인이 있다.
- 원인을 구분하지 못하면 잘못된 처방을 한다 (예: faithfulness hallucination인데 docs를 더 많이 붙이는 — 도리어 악화).

## 용어 (What & How)

### Sycophancy (아첨)

**자신만만하게 동조하는** [Model](./01-the-model.md#model) 출력. 원인은 [Training](./01-the-model.md#training): 모델이 인간이 좋아한 답을 선호하도록 형성됐는데, 인간은 *틀렸다는 말을 듣는 것보다* *동의를 듣는 것*을 더 좋아한다. 그래서 모델이 *동의가 보상이다*를 학습했다 — 그 동의가 틀린 동의일 때조차도.

**나타나는 양상**:
- *반박에 무너짐* — "정말이야?"라고 하면 맞는 답을 뒤집는다.
- *나쁜 입력 칭찬* — 분석도 하기 전에 네 망가진 계획이 훌륭하다고 한다.
- *편향된 프레이밍* — "내가 짠 코드"라고 하면 긍정적, "남이 짠 코드"라고 하면 부정적. **같은 코드, 다른 평가**.
- *흉내내기* — 네 실수를 다시 너에게 확인으로 되돌려준다.

**진단 테스트**: *네가 유도하지 않았다면 모델이 이렇게 말했을까?* 바뀐 게 네 **톤이나 프레이밍**뿐이라면, 그건 sycophancy다 — 분석이 진짜로 바뀐 게 아니다.

**처방**: 선호를 숨겨라. 중립적으로 표현하라 — *"이 코드 좋아?"* 가 아니라 *"이 코드 리뷰해."*

**❌ 피할 표현**: "내 마음에 든 틀린 답"을 그냥 다 sycophancy라고 부르지 말 것. 진단 테스트 없이는 그냥 "틀렸다"보다 가치가 없다.

**💬 실전 대화 예시**
> "리팩토링 계획이 좋다고 했다가, '정말?' 하니까 다 뒤집었어."
> "전형적인 sycophancy야 — 처음엔 네가 자신만만해 보여서 동의했고, 그다음엔 네가 의심해 보여서 무너진 거야. 계획 품질은 안 바뀌었고, 네 톤만 바뀐 거야. [Clear](./05-handoffs.md#clearing) 하고 어느 쪽도 신호 보내지 말고 다시 물어봐."

---

### Hallucination (환각)

**자신만만하게 틀린** [Model](./01-the-model.md#model) 출력. **원인과 처방이 다른 두 종류**:

**1. Factuality hallucination (사실성 환각)** — 세상에 대한 사실을 지어내거나 틀림 (존재 안 하는 함수, 잘못된 API 시그니처, 가짜 인용).
- 원인: [Parametric knowledge](#parametric-knowledge) 갭, 종종 [Knowledge cutoff](#knowledge-cutoff) 이후의 정보.
- 처방: 올바른 [Contextual knowledge](#contextual-knowledge)를 로드.

**2. Faithfulness hallucination (충실성 환각)** — 출력이 로드된 **contextual knowledge**, 사용자 지시, 또는 모델 자신의 이전 추론에서 **벗어남**.
- 원인: [Attention degradation](#attention-degradation), [dumb zone](#smart-zone)에서 악화.
- 처방: [Clear](./05-handoffs.md#clearing) 또는 [Compact](./05-handoffs.md#compaction).

**❌ 피할 표현**: "hallucination"을 그냥 "틀렸다"의 동의어로 쓰지 말 것. 두 종류 중 어느 쪽인지 명명하지 않으면 진단으로서 가치가 없다.

**💬 실전 대화 예시**
> "스키마에 `parseAsync` 메서드를 환각했어."
> "Factuality야, faithfulness야?"
> "내가 붙인 docs에 그 메서드가 있어 — 그냥 [turn](./02-sessions-context-windows-turns.md#turn) 40 이후로 그 docs를 안 읽고 있어."
> "그럼 faithfulness네. Compact하고 다시 로드해, docs를 더 붙이지 마."

🏢 **실무 적용**: 사내에서 *"환각이 발생했어요"*라는 보고가 오면 첫 질문은 항상 **"두 종류 중 어떤 쪽?"** 이어야 한다. 처방이 정반대다.

---

### Parametric knowledge (파라메트릭 지식)

[Training](./01-the-model.md#training)으로 [Model](./01-the-model.md#model)이 "아는" 것 — [Parameters](./01-the-model.md#parameters)에 저장됨. 학습 시점에 **얼어 있다** — 모델은 자기 파라미터를 보지도, 갱신하지도 못한다. **압축에서 디테일이 사라진다**: 수십억 사실이 고정된 파라미터 수에 욱여들어가, 드문 것일수록 흐려진다. 흔한 주제에서의 유창함의 원천이고, 드문 주제에서의 fabrication의 원천. [Contextual knowledge](#contextual-knowledge)의 반대 짝.

**💬 실전 대화 예시**
> "React는 흠잡을 데 없이 짜는데, 우리 내부 SDK에서는 메서드를 지어내."
> "React는 parametric knowledge에 빽빽이 있어 — 학습 예시가 수백만이야. 너희 SDK는 그렇지 않으니까, 모델이 그럴듯한 모양을 채워넣는 거야. SDK docs를 [Context](./02-sessions-context-windows-turns.md#context)에 로드해."

---

### Knowledge cutoff (지식 컷오프)

[Model](./01-the-model.md#model)이 [Parametric knowledge](#parametric-knowledge)를 **갖지 못한 시점 이후**의 경계 날짜. 컷오프 이후의 라이브러리, API, 사건은 **fabrication 함정**이다 — 그 docs를 [Contextual knowledge](#contextual-knowledge)로 로드하지 않으면. 모델 릴리즈마다 자체 컷오프가 있다.

**💬 실전 대화 예시**
> "v3 SDK 문법으로 자꾸 짜 — 우린 v5인데."
> "v5는 knowledge cutoff 이후에 나왔어. v5 changelog를 contextual knowledge로 로드해, 안 그러면 옛 parametric 버전으로 계속 지어낼 거야."

---

### Contextual knowledge (컨텍스트 지식)

[Agent](./02-sessions-context-windows-turns.md#agent)가 **지금 [Context](./02-sessions-context-windows-turns.md#context)에서 직접 읽을 수 있는 사실들** — 사용자 작업, 에이전트가 읽어들인 파일, [Tool results](./03-tools-environment.md#tool-result), [Session](./02-sessions-context-windows-turns.md#session) 시작에 로드된 [AGENTS.md](./06-memory-and-steering.md#agentsmd) 내용. [Parametric knowledge](#parametric-knowledge)의 반대 짝: parametric은 파라미터에서 *recall*되고, contextual은 [window](./02-sessions-context-windows-turns.md#context-window)에서 *read*된다. **에이전트가 contextual knowledge로 작업하면 [Hallucinations](#hallucination)이 훨씬 줄어든다** — 답이 바로 눈앞에 있고, 흐려진 기억에서 끌어올리는 게 아니니까.

**언제 이 용어를 쓸지**: parametric과 *대조*할 때만. 그 외엔 그냥 **context**라고 해라.

**❌ 피할 표현**: "working memory" — contextual knowledge는 *지금 윈도우에 있는 것*이고, [memory system](./06-memory-and-steering.md#memory-system)은 *세션 간*에 그걸 윈도우로 넣어주는 별개 메커니즘이다. 스케일이 다르므로 혼동 금지.

**💬 실전 대화 예시**
> "Docs 붙이면 API 완벽한데 안 붙이면 지어내는 이유?"
> "Docs 있으면 contextual knowledge야 — 페이지에서 읽는 거지. 없으면 parametric인데, 드문 endpoint는 흐려져."

---

### Attention relationship (어텐션 관계)

각 [Token](./01-the-model.md#token)을 예측할 때, [Model](./01-the-model.md#model)은 [Context](./02-sessions-context-windows-turns.md#context)의 **다른 모든 토큰을 고려한다** — 어떤 건 무겁게, 어떤 건 거의 안 본다. 두 토큰의 **짝**이 attention relationship이고, 의미 있는 짝(예: "her"-"Sarah", `getUser()` 호출-`function getUser` 정의)은 무관한 짝보다 서로에게 더 영향을 준다. **N개 토큰의 컨텍스트는 대략 N² 개의 관계를 만든다**.

**💬 실전 대화 예시**
> "diff에서 두 `user` 심볼을 자꾸 헷갈려 — [dumb zone](#smart-zone)인 것 같아."
> "응, 각 호출부와 그 선언 사이의 attention relationship이 다른 짝이랑 싸우는 거야 — 같은 토큰 모양, 다른 바인딩. 하나 rename 하면 짝이 선명해져."

---

### Attention budget (어텐션 예산)

각 [Token](./01-the-model.md#token)은 컨텍스트의 나머지 토큰들에 분배할 **유한한 영향력**을 갖는다. [한 관계](#attention-relationship)에 무겁게 쓰면 다른 관계에 쓸 게 줄어든다. **예산은 토큰당이고, 컨텍스트가 커진다고 늘어나지 않는다** — 그래서 긴 [Sessions](./02-sessions-context-windows-turns.md#session)이 희석된다.

**💬 실전 대화 예시**
> "내가 위에 붙인 스키마를 자꾸 무시해."
> "[Dumb zone](#smart-zone) 깊숙이 들어와 있어 — 매 토큰의 attention budget은 고정인데 컨텍스트가 계속 커졌어. 스키마의 신호가 수천 개 새 토큰과 경쟁 중이야."

---

### Attention degradation (어텐션 저하)

[Session](./02-sessions-context-windows-turns.md#session)이 커질수록 각 [Token](./01-the-model.md#token)의 [Attention budget](#attention-budget)이 더 많은 경쟁자에게 분산된다. 어떤 [의미 있는 관계](#attention-relationship)의 신호도 작아지고, 무관한 [Context](./02-sessions-context-windows-turns.md#context)의 노이즈가 밀고 들어온다. **같은 [Model](./01-the-model.md#model), 같은 [Parameters](./01-the-model.md#parameters)** — 그저 같은 접시에 입이 더 늘어났을 뿐. **smart zone / dumb zone** 효과의 원인.

**💬 실전 대화 예시**
> "Dumb zone 한복판이야 — 타입 파일에 없는 generics를 지어내고 있어."
> "Attention degradation이야. 타입 정의는 아직 컨텍스트에 *있어* — 근데 그 위 신호가 그 이후 추가된 모든 것에 깔려 있어. [Clear](./05-handoffs.md#clearing) 하고 다시 로드해."

---

### Smart zone (스마트 존)

[Session](./02-sessions-context-windows-turns.md#session) 초반에 [Agent](./02-sessions-context-windows-turns.md#agent)는 "smart zone"에 있다 — 날카롭고, 집중되고, 회상이 좋다. 세션이 커지면 "**dumb zone**"으로 드리프트한다: 더 엉성하고, 까먹고, 실수가 늘고 — **faithfulness [hallucinations](#hallucination)** 이 늘어난다. 같은 [Model](./01-the-model.md#model), 같은 [Harness](./01-the-model.md#harness) — 그저 [Context](./02-sessions-context-windows-turns.md#context)가 더 많을 뿐. [Attention degradation](#attention-degradation)이 체감되는 효과.

> **frontier 모델에서 dumb zone은 보통 약 100,000 tokens 부근에서 시작한다고 알려진다 — 다만 논쟁 중**.

세션이 비대해지면 [Clear](./05-handoffs.md#clearing) 또는 [Compact](./05-handoffs.md#compaction). 밀어붙이지 말 것.

**💬 실전 대화 예시**
> "처음 세 컴포넌트는 잘 짰는데 네 번째는 망쳤어."
> "Smart zone을 벗어났어 — 같은 모델인데, dumb zone 깊은 곳에 있어. Compact하고 계획 다시 로드해, 다음 컴포넌트는 잘 들어올 거야."

🏢 **실무 적용**: 사내 RAG/LangGraph 파이프라인에서도 동일한 현상이 일어난다. **노드 체인이 길어질수록 누적 context가 커지고**, 후반 노드의 추론 품질이 떨어진다. 중간에 **요약/압축 노드**를 끼우는 것을 고려.

## 이 섹션 요약 (Cheatsheet)

| 증상 | 가장 가능성 높은 원인 | 1차 처방 |
|---|---|---|
| 자신만만하게 동의/번복 | Sycophancy | 톤·프레이밍 중립화 |
| 존재 안 하는 API 만듦 | Factuality hallucination | 정확한 docs를 context에 로드 |
| 붙여둔 docs를 무시하고 헛소리 | Faithfulness hallucination | Clear 또는 Compact |
| 신버전 라이브러리에서 헛소리 | Knowledge cutoff | 변경 docs/changelog 로드 |
| 세션 후반에 점점 멍청해짐 | Attention degradation / Dumb zone | Compact, 핵심만 다시 로드 |

```
[Hallucination 분기]
   ├─ docs가 context에 있는가?
   │     ├─ NO  → Factuality   → docs 로드
   │     └─ YES → Faithfulness → Clear / Compact
```

## 관련 문서

- 이전: [03 - Tools & Environment](./03-tools-environment.md)
- 다음: [05 - Handoffs](./05-handoffs.md)
- 인덱스: [README](./README.md)

## 참고 자료 (References)

- 원문: [mattpocock/dictionary-of-ai-coding — Section 4](https://github.com/mattpocock/dictionary-of-ai-coding#section-4--failure-modes)
- Anthropic 연구: [Long-context attention](https://www.anthropic.com/research)
