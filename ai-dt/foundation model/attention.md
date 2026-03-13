---
tags: [attention, self-attention, transformer, seq2seq]
level: beginner-to-intermediate
last_updated: 2026-03-14
status: in-progress
---

# Attention이란 무엇인가?

> Attention은 "현재 계산에 필요한 정보를 입력 전체에서 선택적으로 끌어오는 방법"이다. Transformer에서는 이 메커니즘이 모델의 중심이 된다.

## 왜 필요한가? (Why)

초기 seq2seq RNN은 입력 문장 전체를 **하나의 고정 길이 벡터**에 압축한 뒤 decoder가 그 벡터만 보고 출력을 생성하는 구조였다. 문장이 길어질수록 문제가 생겼다.

- 긴 문장의 앞부분 정보가 희미해진다
- 어떤 단어를 번역하거나 생성할 때 입력의 어느 위치를 봐야 하는지 명시적으로 알기 어렵다
- RNN은 시간축 순서대로 계산하므로 병렬화가 어렵다

Bahdanau et al. (2014)은 이 병목을 완화하기 위해, decoder가 매 시점마다 **입력의 여러 hidden state를 다시 참조**하도록 하는 attention을 도입했다.

## 한 문장으로 직관 잡기

문장을 읽을 때 사람도 항상 모든 단어를 똑같이 보지 않는다.

- 주어를 해석할 때는 주어 근처를 더 본다
- 대명사 `it`, `she`, `they`를 해석할 때는 앞 문맥에서 관련 명사를 더 본다
- 번역할 때 현재 생성 중인 단어와 가장 관련 있는 원문 위치를 더 본다

Attention은 이 "어디를 볼지"를 수치적으로 학습하는 메커니즘이다.

## 핵심 아이디어

현재 위치의 표현을 **Query(Q)** 라고 하고, 참고 가능한 각 토큰의 표현을 **Key(K)** 와 **Value(V)** 로 둔다.

1. Query와 각 Key의 관련도를 계산한다
2. 관련도를 확률처럼 정규화한다
3. 그 가중치로 Value들을 가중합한다
4. 그 결과가 "지금 이 위치가 참고한 문맥 정보"가 된다

즉 attention은 **soft lookup** 이다. 메모리 전체를 뒤지는 대신, 필요한 부분에 더 큰 가중치를 주어 읽는다.

## Q / K / V를 실무 감각으로 이해하기

| 요소 | 질문으로 바꾸면 | 역할 |
|------|----------------|------|
| Query | "지금 나는 무엇이 필요한가?" | 현재 토큰의 요구 |
| Key | "나는 어떤 특징을 가진 정보인가?" | 각 토큰의 검색용 표지 |
| Value | "실제로 전달할 내용은 무엇인가?" | 읽어올 내용 |

예를 들어 `철수는 영희에게 책을 건넸고, 그녀는 바로 읽기 시작했다.` 같은 문장에서 `그녀는`을 해석할 때 현재 위치의 Query는 앞쪽 토큰 중 `영희` 관련 Key와 높은 점수를 만들 가능성이 크다.

## Scaled Dot-Product Attention

Transformer 논문은 attention을 다음처럼 정리한다.

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

이 식을 순서대로 읽으면:

1. `QK^T`
   - Query와 Key의 유사도를 계산한다
2. `/ sqrt(d_k)`
   - 차원이 커질수록 dot product가 너무 커지는 문제를 완화한다
3. `softmax(...)`
   - 각 위치의 점수를 0~1 사이 가중치로 바꾼다
4. `... V`
   - Value들을 가중합해 최종 문맥 벡터를 만든다

### 왜 `softmax`가 필요한가?

- 한 위치만 딱 고르는 hard selection이 아니라
- 여러 위치를 동시에 참고하되
- 더 중요한 위치는 더 크게, 덜 중요한 위치는 더 작게 반영하기 위해서다

## Self-Attention이 특별한 이유

기존 attention은 보통 decoder가 encoder 출력에 접근하는 형태였다. Transformer는 한 단계 더 가서 **문장 내부 토큰끼리 서로를 참조하는 self-attention**을 중심에 놓았다.

예를 들어 문장 내 각 토큰은:

- 자기 자신을 볼 수 있고
- 앞뒤 토큰을 모두 볼 수 있으며
- 멀리 떨어진 토큰과도 직접 연결될 수 있다

이 때문에 장거리 의존성을 다루기 쉬워진다. RNN처럼 정보를 여러 step에 걸쳐 전달할 필요가 없기 때문이다.

## Self-Attention, Cross-Attention, Causal Attention

### 1. Self-Attention

- Query, Key, Value가 모두 같은 시퀀스에서 나온다
- encoder 내부에서 많이 사용된다
- 문장 전체 문맥을 통합하는 데 적합하다

### 2. Cross-Attention

- Query는 decoder에서 오고, Key/Value는 encoder 출력에서 온다
- decoder가 입력 문장을 참고하면서 출력을 생성할 때 사용된다
- 번역, 요약, speech-to-text 같은 입력-출력 변환 태스크에 중요하다

### 3. Causal Self-Attention

- decoder용 self-attention
- 현재 토큰은 **미래 토큰을 보면 안 된다**
- 그래서 attention mask를 사용해 `t` 시점이 `t+1`, `t+2`를 보지 못하게 막는다

이 causal mask 덕분에 decoder는 **다음 토큰 예측(next-token prediction)** 학습을 정합적으로 수행할 수 있다.

## Multi-Head Attention

Transformer는 attention을 한 번만 하지 않고 여러 개의 head로 나눠 동시에 수행한다.

왜 head를 여러 개 둘까?

- 어떤 head는 문법 관계를 본다
- 어떤 head는 coreference를 본다
- 어떤 head는 지역적 문맥을 본다
- 어떤 head는 멀리 떨어진 토큰 관계를 본다

즉 multi-head attention은 "한 가지 방식으로만 보지 말고 여러 관점으로 문맥을 읽자"는 설계다.

## Attention이 LLM으로 이어진 이유

Attention, 특히 self-attention이 LLM의 핵심이 된 이유는 다음과 같다.

- **병렬화**: RNN처럼 step-by-step recurrence가 없어 GPU/TPU에서 대규모 학습이 유리하다
- **장거리 연결**: 먼 토큰도 직접 참조할 수 있어 긴 문맥을 다루기 쉽다
- **범용성**: 번역, 요약, 언어모델링, 질의응답 등 다양한 텍스트 작업에 동일한 블록을 재사용할 수 있다
- **확장성**: 더 많은 데이터, 더 긴 문맥, 더 큰 모델에 맞춰 스케일업하기 쉬웠다

Transformer 논문 제목이 `Attention Is All You Need`인 이유도 여기에 있다. 핵심 연산을 attention 중심으로 재구성했기 때문이다.

## Attention에 대한 자주 하는 오해

### 오해 1. Attention = explanation

attention weight가 무엇을 봤는지 일부 힌트를 주는 것은 맞지만, 그것만으로 모델의 완전한 "이유"를 설명한다고 보기는 어렵다. 실제 출력은 residual connection, MLP block, layer norm, 여러 head의 결합 영향을 함께 받는다.

### 오해 2. Attention만 있으면 순서 정보가 자동으로 생긴다

아니다. self-attention은 기본적으로 집합(set)처럼 입력을 볼 수 있기 때문에, Transformer는 **positional encoding / positional embedding** 같은 별도 순서 정보를 넣어야 한다.

### 오해 3. 모든 토큰을 똑같이 자세히 본다

아니다. attention은 모든 토큰 사이 점수를 계산하지만, 실제로는 특정 head와 특정 layer에서 정보가 선택적으로 강조된다.

## 한계도 있다

- 표준 self-attention은 길이 `n`에 대해 대략 `O(n^2)` 메모리/연산 비용이 든다
- 문맥 창(context window)이 길어질수록 비용이 빠르게 커진다
- 그래서 긴 문맥 처리에는 sparse attention, linear attention, chunking, KV cache 최적화 같은 보완 기법이 발전했다

## 실무에서 이렇게 정리하면 된다

- attention은 **정보 검색 방식**
- self-attention은 **토큰들이 서로를 읽는 방식**
- causal self-attention은 **생성 모델의 읽기 규칙**
- multi-head attention은 **여러 관점으로 동시에 읽는 방식**

LLM은 결국 "엄청 큰 텍스트 말뭉치 위에서 self-attention 기반 구조를 대규모로 학습한 언어 모델"이라고 보면 된다.

## 참고 자료 (Primary Sources)

- Bahdanau et al., *Neural Machine Translation by Jointly Learning to Align and Translate* (2014): <https://arxiv.org/abs/1409.0473>
- Vaswani et al., *Attention Is All You Need* (2017): <https://arxiv.org/abs/1706.03762>
