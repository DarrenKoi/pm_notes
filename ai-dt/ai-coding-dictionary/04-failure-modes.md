---
tags: [ai-coding, hallucination, attention, knowledge-cutoff, sycophancy]
level: intermediate
last_updated: 2026-05-05
source: https://github.com/mattpocock/dictionary-of-ai-coding
type: learning
reviewed_on: 2026-10-04
review_status: partial
---

# Section 4 — Failure Modes (실패 양상)

> [!info] 검토 범위 · 2026-10-04
> 원래 2026-05-05 학습 노트의 용어·대화 사례를 유지했다. 대화는 설명용 가상 사례이며 사내 시스템의 관측 결과가 아니다. 저자의 용어 선택은 보편 표준과 구분한다. 버전별 실제 조건은 [검증된 적용 조건](./verified-conditions.md), 개별 변경·남은 미확인은 [정리 기록](./organization-log.md)을 함께 읽는다.


> AI 에이전트가 **이상하게 굴 때**, 그 이상함에 정확히 이름을 붙이는 용어들. 이 섹션의 사례 빈도는 측정하지 않았다.

## 왜 이 섹션이 가장 자주 쓰이는가? (Why)

- "헛소리한다 / 자꾸 까먹는다 / 갑자기 멍청해진다 / 내 말에 무조건 동의한다" — 이 네 가지가 가장 흔한 증상이고, 각각 원인이 다르다.
- 원인을 구분하지 못하면 잘못된 처방을 내리게 된다. 예를 들어 faithfulness hallucination인데 docs를 더 많이 붙이면, 오히려 상태가 악화된다.

## 용어 (What & How)

### Sycophancy (아첨)

**자신만만한 태도로 사용자에게 동조해 버리는** [Model](./01-the-model.md) 출력. [Training](./01-the-model.md)의 선호도 최적화는 가능한 기여 요인이다. 단일 원인을 확정한 진단은 아니다. 모델은 사람이 좋아한 답을 선호하도록 형성되는데, 사람들은 보통 *틀렸다는 말을 듣는 것*보다 *동의를 듣는 것*을 더 좋아한다. 그래서 모델은 *동의가 곧 보상*이라는 패턴을 학습해 버렸다. 그 동의가 사실은 틀린 동의일 때조차도 그렇다.

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
> "전형적인 sycophancy야. 처음엔 네가 자신만만해 보였으니까 동의했고, 그다음엔 네가 의심하는 톤이 되니까 무너진 거지. 계획 품질이 바뀐 게 아니라 네 톤만 바뀌었을 뿐이야. [Clear](./05-handoffs.md)하고 한쪽으로 치우친 신호 없이 다시 물어봐."

---

### Hallucination (환각)

**자신만만한 태도로 틀린 답을 내놓는** [Model](./01-the-model.md) 출력. 사실성·충실성은 구분에 유용한 관점이며 상호 배타적인 원인 진단이 아니다. 같은 출력이 둘 다 위반할 수 있다.

**1. Factuality hallucination (사실성 환각)**: 세상에 대한 사실 자체를 지어내거나 틀린 경우(존재하지 않는 함수, 잘못된 API 시그니처, 출처가 가짜인 인용 등).
- 원인: [Parametric knowledge](./04-failure-modes.md)의 빈틈. [Knowledge cutoff](./04-failure-modes.md) 이후의 정보일 때 자주 발생한다.
- 처방: 올바른 [Contextual knowledge](./04-failure-modes.md)를 로드한다.

**2. Faithfulness hallucination (충실성 환각)**: 출력이 이미 로드된 **contextual knowledge**, 사용자 지시, 또는 모델 자신의 직전 추론에서 **벗어나는** 경우.
- 가능 요인: 관련 정보 누락·잘림·상충 지시·검색/도구 오류·모델 능력·긴 입력에서의 활용 실패 등. [Attention degradation](./04-failure-modes.md)만으로 확정하지 않는다.
- 처방 후보: 실제 입력·schema·도구 결과를 확인하고 핵심 근거 재배치·작업 축소·[Clear](./05-handoffs.md)/[Compact](./05-handoffs.md)를 비교한다. 압축도 근거를 잃을 수 있다.

**❌ 피할 표현**: "hallucination"을 그냥 "틀렸다"의 동의어처럼 쓰지 말 것. 두 종류 중 어느 쪽인지 짚어주지 않으면 진단으로서의 가치가 없다.

**💬 실전 대화 예시**
> "스키마에 `parseAsync` 메서드를 환각했어."
> "Factuality 쪽이야, faithfulness 쪽이야?"
> "내가 붙인 docs에는 그 메서드가 있어. 그냥 [turn](./02-sessions-context-windows-turns.md) 40 이후로는 그 docs를 안 읽고 있는 것 같아."
> "실제 요청에도 docs가 남았는지 먼저 확인해. 지시 충돌과 version/schema를 확인한 뒤 축소·압축을 비교해."

🏢 **실무 적용**: 사내에서 *"환각이 발생했어요"*라는 보고가 들어오면, 가장 먼저 던져야 할 질문은 **"두 종류 중 어느 쪽인가?"** 이다. 관점만으로 단일 원인이나 반대 처방을 확정하지 말고 실제 입력·출력 증거를 함께 본다.

---

### Parametric knowledge (파라메트릭 지식)

[Training](./01-the-model.md)을 통해 [Model](./01-the-model.md)이 "안다"고 할 만한 정보. [Parameters](./01-the-model.md)에 저장돼 있다. 한 번 학습되고 나면 **그 시점에 얼어붙어서**, 모델은 자기 파라미터를 들여다보지도, 갱신하지도 못한다. 또한 **압축 과정에서 디테일이 사라진다**. 수십억 개의 사실이 고정된 파라미터 안에 욱여 들어가다 보니, 드문 정보일수록 흐릿해진다. 흔한 주제에서 유창함이 나오는 출처이자, 드문 주제에서 fabrication이 일어나는 출처이기도 하다. [Contextual knowledge](./04-failure-modes.md)와 짝을 이루는 반대 개념.

**💬 실전 대화 예시**
> "React는 흠잡을 데 없이 짜는데, 우리 내부 SDK에서는 자꾸 메서드를 지어내."
> "React가 학습 자료에 얼마나 포함되었는지는 공개되지 않았다면 확인할 수 없어. 그런데 너희 내부 SDK는 그렇지 않으니까, 모델이 그럴듯한 모양으로 빈 곳을 채워넣는 거야. SDK docs를 [Context](./02-sessions-context-windows-turns.md)에 로드해 줘."

---

### Knowledge cutoff (지식 컷오프)

[Model](./01-the-model.md)이 [Parametric knowledge](./04-failure-modes.md)를 갖추었다고 보장하지 않는 기준 시점. 그 전의 사실도 정확한 학습·회상을 보장하지 않는다. 이후 사실은 검색·도구·별도 업데이트로 알 수 있어 cutoff 하나로 답의 진위를 결정하지 않는다. 해당 docs를 [Contextual knowledge](./04-failure-modes.md)로 직접 로드해 주지 않는 한 그렇다. 모델은 릴리즈마다 자기만의 cutoff를 갖는다.

**💬 실전 대화 예시**
> "자꾸 v3 SDK 문법으로 짜네. 우리는 v5인데."
> "v5가 knowledge cutoff 이후에 나와서 그래. v5 changelog를 contextual knowledge로 로드해. 그러지 않으면 모델이 계속 옛 parametric 버전을 기준으로 답을 지어낼 거야."

---

### Contextual knowledge (컨텍스트 지식)

[Agent](./02-sessions-context-windows-turns.md)가 **지금 [Context](./02-sessions-context-windows-turns.md)에서 직접 읽어낼 수 있는 정보들**. 사용자가 한 말, 에이전트가 읽어들인 파일, [Tool results · Tool result (도구 결과) 절](./03-tools-environment.md), [Session](./02-sessions-context-windows-turns.md) 시작에 로드된 [AGENTS.md](./06-memory-and-steering.md#agentsmd) 내용 같은 것들이 여기에 해당한다. [Parametric knowledge](./04-failure-modes.md)와 짝을 이루는 반대 개념이다. parametric은 파라미터에서 *떠올리는(recall)* 정보이고, contextual은 [window](./02-sessions-context-windows-turns.md)에서 *읽어내는(read)* 정보다. 검증된 관련 자료를 주입하는 것은 [Hallucination](./04-failure-modes.md)을 줄이는 방법이지만 자료 정확도·모델의 실제 사용 여부를 확인해야 한다. 감소 폭과 완벽한 답은 보장하지 않는다. 답이 바로 눈앞에 있는 상태라, 흐릿해진 기억을 더듬을 필요가 없기 때문이다.

**언제 이 용어를 쓸지**: parametric과 *대조*하고 싶을 때만 쓰면 된다. 그 외에는 그냥 **context**라고 부르는 편이 자연스럽다.

**❌ 피할 표현**: "working memory". contextual knowledge는 *지금 윈도우 안에 있는 것*이고, [memory system](./06-memory-and-steering.md)은 *세션과 세션 사이에서* 그것을 다시 윈도우에 넣어주는 별개 메커니즘이다. 스케일이 전혀 다르니 혼동하면 안 된다.

**💬 실전 대화 예시**
> "Docs를 붙이면 API를 완벽하게 짜는데, 안 붙이면 지어내. 왜 그래?"
> "Docs가 있으면 그건 contextual knowledge라서, 모델이 페이지를 읽고 답해. 없을 때는 parametric에 의존해야 하는데, 드문 endpoint일수록 그쪽이 흐릿하거든."

---

### Attention relationship (어텐션 관계)

dense self-attention의 한 query는 mask로 허용된 key 위치들에 가중치를 계산한다. 학습된 head·layer에 따라 관계는 달라지며 높은 가중치가 곧 의미적 중요성·정확한 변수 binding이라는 보장은 없다. 전체 길이 N의 dense 처리에는 대략 N² 쌍이 있지만 causal mask·sparse 구조·KV cache를 사용하는 한 토큰 decode의 계산은 구분한다.

**💬 실전 대화 예시**
> "diff 안에서 두 `user` 심볼을 자꾸 헷갈리네. [dumb zone · Smart zone (스마트 존) 절](./04-failure-modes.md)에 들어간 것 같아."
> "그럴 수도 있지만 실제 binding과 입력 누락을 먼저 확인해. 각 호출부와 그 선언 사이의 attention relationship이 다른 짝들이랑 신호 경쟁을 하고 있는 거야. 토큰 모양은 똑같은데 바인딩이 다르니까 그래. 둘 중 하나만 rename 해 줘도 짝이 훨씬 선명해질 거야."

---

### Attention budget (어텐션 예산)

이 사전의 설명용 비유다. 일반 softmax attention에서 한 query/head의 허용 key 가중치 합은 1이다. 모델 전체의 고정된 “주의력 총량”이 측정된 것은 아니다. 길이가 늘어도 특정 관련 key에 큰 가중치를 줄 수 있어 균등 희석이나 품질 저하가 수학적으로 필연이라는 뜻은 아니다.

**💬 실전 대화 예시**
> "내가 위에 붙여 둔 스키마를 자꾸 무시하네."
> "지금 [Dumb zone · Smart zone (스마트 존) 절](./04-failure-modes.md) 깊숙이 들어와 있어. 토큰별 attention budget은 고정인데 컨텍스트는 계속 커졌거든. 스키마에서 나오는 신호가 수천 개의 새 토큰과 경쟁하느라 묻히고 있는 거야."

---

### Attention degradation (어텐션 저하)

긴 입력에서 필요한 정보를 잘 활용하지 못하는 현상을 설명하는 용어다. Lost in the Middle의 위치별 성능 차이는 해당 모델·검색/질문응답 실험의 관측이다. 모든 모델·모든 세션에서 품질이 단조 감소한다는 법칙이나 유일한 원인으로 확대하지 않는다. 실제 입력 잘림, 잘못된 도구 결과, 지시 충돌도 함께 비교한다.

**💬 실전 대화 예시**
> "지금 Dumb zone 한복판이야. 타입 파일에 없는 generics를 지어내고 있어."
> "긴 입력 활용 실패 가능성이 있지만 아직 단정할 수 없어. 타입 정의는 *컨텍스트 안에 그대로 있긴 해.* 다만 그 위에 깔린 신호가 그 이후 추가된 모든 것 밑에 묻혀 버린 거야. [Clear](./05-handoffs.md)하고 다시 로드하는 게 빠를 거야."

---

### Smart zone (스마트 존)

smart/dumb zone은 이 사전의 실무 비유이며 표준 성능 구간이 아니다. 기존 100,000 token 임계값을 보편 법칙으로 사용할 근거는 확인하지 못했다. 모델·과제·자료 위치·평가에 따라 실용 입력 길이를 측정한다. clear/compact는 후보 조치이며 초기 지시·중요 근거를 보존하고 결과를 다시 평가한다.

**💬 실전 대화 예시**
> "처음 세 컴포넌트는 잘 짰는데 네 번째는 완전히 망쳤어."
> "그 비유만으로 원인이 확정되지는 않아. 실제 입력·코드·오류부터 비교해. Compact한 다음 계획만 다시 로드해 줘. 핵심 계약을 다시 읽히고 다음 결과가 실제로 개선되는지 검사해."

🏢 **실무 적용**: 사내 RAG/LangGraph 파이프라인에서도 똑같은 현상이 일어난다. **노드 체인이 길어질수록 누적 context가 부풀어 오르고**, 그에 따라 후반 노드의 추론 품질이 떨어진다. 중간에 **요약/압축 노드**를 끼워 넣는 방안을 고려해 볼 만하다.

## 이 섹션 요약 (Cheatsheet)

| 증상 | 점검 후보 | 비교할 조치 |
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
   │     └─ YES → Faithfulness 여부 확인 → 입력·지시·도구 검증 후 재구성
```

## 관련 문서

- 이전: [03 - Tools & Environment](./03-tools-environment.md)
- 다음: [05 - Handoffs](./05-handoffs.md)
- 인덱스: [README](./README.md)

## 참고 자료 (References)

- 원문: [mattpocock/dictionary-of-ai-coding — Section 4](https://github.com/mattpocock/dictionary-of-ai-coding#section-4--failure-modes)
- Anthropic 연구: [Lost in the Middle: 실험 조건별 긴 입력 활용](https://arxiv.org/abs/2307.03172)
