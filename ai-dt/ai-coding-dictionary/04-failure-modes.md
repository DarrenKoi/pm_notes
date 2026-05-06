---
tags: [ai-coding, hallucination, attention, knowledge-cutoff, sycophancy]
level: intermediate
last_updated: 2026-05-05
source: https://github.com/mattpocock/dictionary-of-ai-coding
---

# Section 4 — Failure Modes (실패 양상)

> AI 에이전트가 **이상하게 굴 때**, 그 이상함에 정확히 이름을 붙이는 용어들. 이 섹션이 일상 디버깅의 80%를 차지한다.

## 왜 이 섹션이 가장 자주 쓰이는가? (Why)

- "헛소리한다 / 자꾸 까먹는다 / 갑자기 멍청해진다 / 내 말에 무조건 동의한다" — 이 네 가지가 가장 흔한 증상이고, 각각 원인이 다르다.
- 원인을 구분하지 못하면 잘못된 처방을 내리게 된다. 예를 들어 faithfulness hallucination인데 docs를 더 많이 붙이면, 오히려 상태가 악화된다.

## 용어 (What & How)

### Sycophancy (아첨)

**자신만만한 태도로 사용자에게 동조해 버리는** [Model](./01-the-model.md#model) 출력. 원인은 [Training](./01-the-model.md#training)에 있다. 모델은 사람이 좋아한 답을 선호하도록 형성되는데, 사람들은 보통 *틀렸다는 말을 듣는 것*보다 *동의를 듣는 것*을 더 좋아한다. 그래서 모델은 *동의가 곧 보상*이라는 패턴을 학습해 버렸다. 그 동의가 사실은 틀린 동의일 때조차도 그렇다.

**나타나는 양상**:
- *반박에 쉽게 무너짐*: "정말이야?"라고 한 번만 되물어도 맞는 답을 뒤집어 버린다.
- *나쁜 입력에도 칭찬*: 분석도 시작하기 전에 사용자의 망가진 계획을 훌륭하다고 평가한다.
- *프레이밍에 따라 평가가 달라짐*: "내가 짠 코드"라고 하면 긍정적으로 보고, "남이 짠 코드"라고 하면 부정적으로 본다. **같은 코드인데 평가가 갈린다.**
- *사용자 흉내*: 사용자의 실수를 다시 사용자에게 확인 형태로 되돌려준다.

**진단 테스트**: *내가 특정 방향으로 유도하지 않았더라도 모델이 이렇게 말했을까?* 만약 바뀐 것이 내 **톤이나 프레이밍뿐**이라면, 그건 sycophancy일 가능성이 높다. 분석이 실제로 바뀐 게 아니다.

**처방**: 자신의 선호를 드러내지 말고, 중립적으로 묻는다. *"이 코드 좋아?"* 가 아니라 *"이 코드를 리뷰해 줘."* 처럼.

**❌ 피할 표현**: "내 마음에 안 드는 틀린 답"을 무조건 sycophancy라고 부르지 말 것. 진단 테스트를 거치지 않으면 그냥 "틀렸다"라고 말하는 것 이상의 가치가 없다.

**💬 실전 대화 예시**
> "처음엔 리팩토링 계획이 좋다더니, '정말?' 하니까 다 뒤집어."
> "전형적인 sycophancy야. 처음엔 네가 자신만만해 보였으니까 동의했고, 그다음엔 네가 의심하는 톤이 되니까 무너진 거지. 계획 품질이 바뀐 게 아니라 네 톤만 바뀌었을 뿐이야. [Clear](./05-handoffs.md#clearing)하고 한쪽으로 치우친 신호 없이 다시 물어봐."

---

### Hallucination (환각)

**자신만만한 태도로 틀린 답을 내놓는** [Model](./01-the-model.md#model) 출력. **원인과 처방이 다른 두 종류**가 있다.

**1. Factuality hallucination (사실성 환각)**: 세상에 대한 사실 자체를 지어내거나 틀린 경우(존재하지 않는 함수, 잘못된 API 시그니처, 출처가 가짜인 인용 등).
- 원인: [Parametric knowledge](#parametric-knowledge)의 빈틈. [Knowledge cutoff](#knowledge-cutoff) 이후의 정보일 때 자주 발생한다.
- 처방: 올바른 [Contextual knowledge](#contextual-knowledge)를 로드한다.

**2. Faithfulness hallucination (충실성 환각)**: 출력이 이미 로드된 **contextual knowledge**, 사용자 지시, 또는 모델 자신의 직전 추론에서 **벗어나는** 경우.
- 원인: [Attention degradation](#attention-degradation). [dumb zone](#smart-zone)에 들어가면 더 심해진다.
- 처방: [Clear](./05-handoffs.md#clearing) 또는 [Compact](./05-handoffs.md#compaction).

**❌ 피할 표현**: "hallucination"을 그냥 "틀렸다"의 동의어처럼 쓰지 말 것. 두 종류 중 어느 쪽인지 짚어주지 않으면 진단으로서의 가치가 없다.

**💬 실전 대화 예시**
> "스키마에 `parseAsync` 메서드를 환각했어."
> "Factuality 쪽이야, faithfulness 쪽이야?"
> "내가 붙인 docs에는 그 메서드가 있어. 그냥 [turn](./02-sessions-context-windows-turns.md#turn) 40 이후로는 그 docs를 안 읽고 있는 것 같아."
> "그럼 faithfulness네. docs를 더 붙이지 말고, 일단 compact한 다음 다시 로드해."

🏢 **실무 적용**: 사내에서 *"환각이 발생했어요"*라는 보고가 들어오면, 가장 먼저 던져야 할 질문은 **"두 종류 중 어느 쪽인가?"** 이다. 처방이 정반대 방향이기 때문이다.

---

### Parametric knowledge (파라메트릭 지식)

[Training](./01-the-model.md#training)을 통해 [Model](./01-the-model.md#model)이 "안다"고 할 만한 정보. [Parameters](./01-the-model.md#parameters)에 저장돼 있다. 한 번 학습되고 나면 **그 시점에 얼어붙어서**, 모델은 자기 파라미터를 들여다보지도, 갱신하지도 못한다. 또한 **압축 과정에서 디테일이 사라진다**. 수십억 개의 사실이 고정된 파라미터 안에 욱여 들어가다 보니, 드문 정보일수록 흐릿해진다. 흔한 주제에서 유창함이 나오는 출처이자, 드문 주제에서 fabrication이 일어나는 출처이기도 하다. [Contextual knowledge](#contextual-knowledge)와 짝을 이루는 반대 개념.

**💬 실전 대화 예시**
> "React는 흠잡을 데 없이 짜는데, 우리 내부 SDK에서는 자꾸 메서드를 지어내."
> "React는 parametric knowledge에 굉장히 두텁게 들어가 있어. 학습 예시가 수백만 개 단위로 있거든. 그런데 너희 내부 SDK는 그렇지 않으니까, 모델이 그럴듯한 모양으로 빈 곳을 채워넣는 거야. SDK docs를 [Context](./02-sessions-context-windows-turns.md#context)에 로드해 줘."

---

### Knowledge cutoff (지식 컷오프)

[Model](./01-the-model.md#model)이 [Parametric knowledge](#parametric-knowledge)를 **갖지 못하는 경계 시점**. 이 시점 이후에 등장한 라이브러리, API, 사건은 모두 **fabrication 함정**이 된다. 해당 docs를 [Contextual knowledge](#contextual-knowledge)로 직접 로드해 주지 않는 한 그렇다. 모델은 릴리즈마다 자기만의 cutoff를 갖는다.

**💬 실전 대화 예시**
> "자꾸 v3 SDK 문법으로 짜네. 우리는 v5인데."
> "v5가 knowledge cutoff 이후에 나와서 그래. v5 changelog를 contextual knowledge로 로드해. 그러지 않으면 모델이 계속 옛 parametric 버전을 기준으로 답을 지어낼 거야."

---

### Contextual knowledge (컨텍스트 지식)

[Agent](./02-sessions-context-windows-turns.md#agent)가 **지금 [Context](./02-sessions-context-windows-turns.md#context)에서 직접 읽어낼 수 있는 정보들**. 사용자가 한 말, 에이전트가 읽어들인 파일, [Tool results](./03-tools-environment.md#tool-result), [Session](./02-sessions-context-windows-turns.md#session) 시작에 로드된 [AGENTS.md](./06-memory-and-steering.md#agentsmd) 내용 같은 것들이 여기에 해당한다. [Parametric knowledge](#parametric-knowledge)와 짝을 이루는 반대 개념이다. parametric은 파라미터에서 *떠올리는(recall)* 정보이고, contextual은 [window](./02-sessions-context-windows-turns.md#context-window)에서 *읽어내는(read)* 정보다. **에이전트가 contextual knowledge로 작업할 때는 [Hallucination](#hallucination)이 훨씬 줄어든다.** 답이 바로 눈앞에 있는 상태라, 흐릿해진 기억을 더듬을 필요가 없기 때문이다.

**언제 이 용어를 쓸지**: parametric과 *대조*하고 싶을 때만 쓰면 된다. 그 외에는 그냥 **context**라고 부르는 편이 자연스럽다.

**❌ 피할 표현**: "working memory". contextual knowledge는 *지금 윈도우 안에 있는 것*이고, [memory system](./06-memory-and-steering.md#memory-system)은 *세션과 세션 사이에서* 그것을 다시 윈도우에 넣어주는 별개 메커니즘이다. 스케일이 전혀 다르니 혼동하면 안 된다.

**💬 실전 대화 예시**
> "Docs를 붙이면 API를 완벽하게 짜는데, 안 붙이면 지어내. 왜 그래?"
> "Docs가 있으면 그건 contextual knowledge라서, 모델이 페이지를 읽고 답해. 없을 때는 parametric에 의존해야 하는데, 드문 endpoint일수록 그쪽이 흐릿하거든."

---

### Attention relationship (어텐션 관계)

[Model](./01-the-model.md#model)은 각 [Token](./01-the-model.md#token)을 예측할 때 [Context](./02-sessions-context-windows-turns.md#context) 안의 **다른 모든 토큰을 함께 고려한다**. 어떤 토큰은 무겁게, 어떤 토큰은 거의 무시되는 식이다. 이때 두 토큰 사이의 **연결 관계**가 바로 attention relationship이다. 의미 있게 연결된 짝(예: "her"–"Sarah", `getUser()` 호출과 `function getUser` 정의)은 무관한 짝보다 서로에게 더 큰 영향을 준다. **N개 토큰의 컨텍스트에서는 대략 N² 개의 관계가 만들어진다.**

**💬 실전 대화 예시**
> "diff 안에서 두 `user` 심볼을 자꾸 헷갈리네. [dumb zone](#smart-zone)에 들어간 것 같아."
> "맞아. 각 호출부와 그 선언 사이의 attention relationship이 다른 짝들이랑 신호 경쟁을 하고 있는 거야. 토큰 모양은 똑같은데 바인딩이 다르니까 그래. 둘 중 하나만 rename 해 줘도 짝이 훨씬 선명해질 거야."

---

### Attention budget (어텐션 예산)

각 [Token](./01-the-model.md#token)은 컨텍스트의 나머지 토큰들에 나눠줄 수 있는 **유한한 영향력**을 가진다. [어떤 한 관계](#attention-relationship)에 영향력을 무겁게 실으면, 그만큼 다른 관계에 쓸 수 있는 양은 줄어든다. **이 예산은 토큰 단위이고, 컨텍스트 크기가 커진다고 함께 늘어나지는 않는다.** 그래서 긴 [Sessions](./02-sessions-context-windows-turns.md#session)일수록 신호가 점점 희석되는 것이다.

**💬 실전 대화 예시**
> "내가 위에 붙여 둔 스키마를 자꾸 무시하네."
> "지금 [Dumb zone](#smart-zone) 깊숙이 들어와 있어. 토큰별 attention budget은 고정인데 컨텍스트는 계속 커졌거든. 스키마에서 나오는 신호가 수천 개의 새 토큰과 경쟁하느라 묻히고 있는 거야."

---

### Attention degradation (어텐션 저하)

[Session](./02-sessions-context-windows-turns.md#session)이 길어질수록 각 [Token](./01-the-model.md#token)의 [Attention budget](#attention-budget)이 더 많은 경쟁자에게로 흩뿌려진다. [의미 있는 관계](#attention-relationship)에서 나오는 신호도 약해지고, 무관한 [Context](./02-sessions-context-windows-turns.md#context)의 노이즈가 그 자리를 비집고 들어온다. **모델과 [Parameters](./01-the-model.md#parameters)는 그대로**다. 단지 같은 접시에 입만 더 늘어난 셈이다. **smart zone / dumb zone 효과의 근본 원인**이 바로 이것이다.

**💬 실전 대화 예시**
> "지금 Dumb zone 한복판이야. 타입 파일에 없는 generics를 지어내고 있어."
> "전형적인 attention degradation이야. 타입 정의는 *컨텍스트 안에 그대로 있긴 해.* 다만 그 위에 깔린 신호가 그 이후 추가된 모든 것 밑에 묻혀 버린 거야. [Clear](./05-handoffs.md#clearing)하고 다시 로드하는 게 빠를 거야."

---

### Smart zone (스마트 존)

[Session](./02-sessions-context-windows-turns.md#session) 초반의 [Agent](./02-sessions-context-windows-turns.md#agent)는 "smart zone"에 있다. 답이 날카롭고, 집중력이 있고, 회상도 잘 된다. 그러나 세션이 길어지면 점차 "**dumb zone**"으로 미끄러져 들어간다. 답이 엉성해지고, 잘 까먹고, 실수가 늘어나며, **faithfulness [hallucination](#hallucination)** 이 점점 자주 나타난다. 모델도, [Harness](./01-the-model.md#harness)도 그대로인데, 단지 [Context](./02-sessions-context-windows-turns.md#context)가 너무 많이 쌓였을 뿐이다. [Attention degradation](#attention-degradation)이 사용자 입장에서 체감되는 형태가 바로 smart/dumb zone이라고 보면 된다.

> **frontier 모델에서는 dumb zone이 대략 100,000 tokens 부근에서 시작된다고 알려져 있지만, 이 수치는 여전히 논쟁 중이다.**

세션이 비대해졌다면 무리해서 끌고 가지 말고 [Clear](./05-handoffs.md#clearing)하거나 [Compact](./05-handoffs.md#compaction)할 것.

**💬 실전 대화 예시**
> "처음 세 컴포넌트는 잘 짰는데 네 번째는 완전히 망쳤어."
> "Smart zone에서 벗어난 거야. 같은 모델인데 지금 dumb zone 깊숙이 들어가 있어. Compact한 다음 계획만 다시 로드해 줘. 그러면 다음 컴포넌트는 다시 깔끔하게 나올 거야."

🏢 **실무 적용**: 사내 RAG/LangGraph 파이프라인에서도 똑같은 현상이 일어난다. **노드 체인이 길어질수록 누적 context가 부풀어 오르고**, 그에 따라 후반 노드의 추론 품질이 떨어진다. 중간에 **요약/압축 노드**를 끼워 넣는 방안을 고려해 볼 만하다.

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
