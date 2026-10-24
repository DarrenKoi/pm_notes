---
tags: [attention, transformer, llm, forward-pass, causal-mask, next-token-prediction]
level: beginner-to-intermediate
last_updated: 2026-10-09
type: learning
category_major: "AI·DT"
category_middle: "AI 기초"
category_minor: "파운데이션 모델"
note_kind: "학습"
---

# Attention은 LLM 안에서 어떻게 쓰이는가?

> LLM은 "지금까지의 토큰 → 다음 토큰 확률"을 계산하는 함수이고, attention은 그 함수 안에서 토큰끼리 정보를 주고받는 유일한 통로다.

**시각화**: [how-attention-powers-llm.html](./how-attention-powers-llm.html) — 브라우저로 열어 단계별로 눌러 볼 수 있다. 이 문서의 숫자와 같은 값을 쓴다.

> [!info] 범위
> 공개 논문에 나온 표준 decoder-only Transformer(GPT 계열 구조)를 기준으로 설명한다. 특정 제품의 내부 구현은 다루지 않는다. 예제의 벡터와 확률은 원리 설명용으로 손으로 정한 값이다.

## 왜 필요한가? (Why)

- [attention.md](./attention.md)는 attention 연산 자체를, [transformer-to-llm.md](./transformer-to-llm.md)는 계보를 다룬다. 둘을 읽어도 "그래서 attention을 어디에 꽂으면 문장이 나오는가"는 따로 이어 붙여야 한다. 이 문서가 그 연결 부분이다.
- 사내 LLM API에 프롬프트를 보낼 때 실제로 일어나는 일을 알면 다음 질문에 답할 수 있다.
  - 프롬프트에 넣은 문서가 어떻게 답변에 반영되는가? (RAG가 동작하는 이유)
  - 컨텍스트가 길어지면 왜 느려지고 비싸지는가?
  - temperature는 무엇을 바꾸는가?

## 핵심 개념 (What)

### 1. LLM이 하는 일은 하나다

```text
입력: 토큰 n개            출력: 다음 토큰 1개의 확률 분포
[철수는, 영희에게, 책을, 건넸고, 그녀는]  →  {바로: 0.5, 책을: 0.2, ...}
```

대화, 요약, 코드 생성은 모두 이 함수를 반복 호출한 결과다. 토큰을 하나 뽑아 입력 뒤에 붙이고 다시 호출한다.

### 2. 한 번의 계산(forward pass)

```text
텍스트
 → 토큰화            글자 조각을 번호로
 → 임베딩 + 위치      번호를 벡터로, 순서 정보 추가
 → ┌ attention       토큰끼리 정보 교환        ┐
   └ MLP             토큰별로 받은 정보 가공    ┘ × N층
 → 출력층            마지막 위치 벡터 → 어휘 전체 점수(logits)
 → softmax · 샘플링   점수 → 확률 → 토큰 하나 선택
 ↺ 선택한 토큰을 입력 뒤에 붙이고 반복
```

### 3. 블록 안의 역할 분담

| 부품 | 하는 일 | 다른 토큰을 보는가 |
|------|---------|--------------------|
| Attention | 다른 토큰에서 필요한 정보를 끌어온다 | 본다 (유일) |
| MLP (feed-forward) | 끌어온 정보를 토큰별로 변환한다 | 안 본다 |
| Residual 연결 | 각 부품의 결과를 원래 벡터에 **더한다** | - |

토큰마다 벡터 한 줄이 처음부터 끝까지 흐르고, 각 층은 그 벡터에 정보를 덧붙인다고 보면 된다. attention이 없으면 각 토큰은 자기 자신만 아는 채로 끝난다. 그래서 문맥을 이해하는 능력은 전부 attention을 통해 들어온다.

### 4. Causal mask가 학습과 생성을 모두 가능하게 한다

GPT 계열은 각 토큰이 **자기 왼쪽만** 보도록 미래 위치를 가린다. 이 제약 하나로 두 가지가 해결된다.

**학습**: 문장 하나가 문제 n개가 된다.

```text
[철수는]                               → 정답: 영희에게
[철수는, 영희에게]                       → 정답: 책을
[철수는, 영희에게, 책을]                  → 정답: 건넸고
[철수는, 영희에게, 책을, 건넸고]           → 정답: 그녀는
[철수는, 영희에게, 책을, 건넸고, 그녀는]    → 정답: 바로
```

모든 위치가 동시에 계산되고 서로의 정답을 미리 볼 수 없으므로, 한 번의 통과로 n문제를 한꺼번에 채점한다. RNN은 토큰을 차례로 처리해야 했는데 Transformer는 이 부분을 행렬 연산 한 번으로 끝낸다. 방대한 텍스트로 학습할 수 있게 된 직접적인 이유다.

**생성**: 학습 때와 같은 조건(왼쪽만 본다)으로 마지막 위치의 예측만 꺼내 쓰면 된다.

학습 신호는 정답 토큰의 확률에 대한 cross-entropy 하나뿐이다. "대명사는 앞의 명사를 봐야 한다" 같은 규칙은 누구도 넣지 않는다. 다음 토큰을 잘 맞히려면 그렇게 보는 편이 유리하므로 Q·K·V를 만드는 행렬이 그 방향으로 학습된다.

### 5. 층을 쌓고 head를 여러 개 두는 이유

- **Multi-head**: 한 층 안에서 여러 attention이 병렬로 돈다. head마다 다른 관계(예: 대명사와 선행사, 동사와 목적어)를 볼 수 있다.
- **층 쌓기**: 아래층이 만든 결과를 위층이 다시 attention으로 조합한다. 위층의 벡터는 "단어"가 아니라 이미 문맥이 섞인 표현이다.

| 모델 | 층 수 | 벡터 차원 | head 수 | 파라미터 |
|------|-------|-----------|---------|----------|
| Transformer base (2017) | 6 (encoder) + 6 (decoder) | 512 | 8 | 약 6,500만 |
| GPT-2 small (2019) | 12 | 768 | 12 | 약 1.2억 |
| GPT-3 (2020) | 96 | 12,288 | 96 | 약 1,750억 |

구조는 거의 같고 크기와 데이터가 달라졌다.

## 어떻게 사용하는가? (How)

### attention 한 번을 직접 계산하기

시각화 페이지 STEP 2·3과 같은 숫자다. 벡터의 네 칸은 `[사람, 여성, 사물, 행위]`로 사람이 읽을 수 있게 정했다. 실제 모델의 차원은 수백~수천 개이고 이런 이름표가 없다.

```python
import numpy as np

tokens = ["철수는", "영희에게", "책을", "건넸고", "그녀는"]

# Key: "나는 이런 정보다"   Query: "나는 이런 정보를 찾는다"
K = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0.5, 0.5, 0, 0]])
Q = np.array([[1, 0, 0, 0], [3, -1, 0, 2], [1, 1, 0, 2], [2, 1, 3, 0], [2, 6, 0, 0]])
V = K  # 설명용: 실제로는 Q, K, V 모두 별도의 학습된 행렬로 만든다


def attention_weights(Q, K, causal=True):
    scores = Q @ K.T / np.sqrt(K.shape[1])  # 모든 토큰 쌍의 점수를 한 번에
    if causal:  # 자기보다 뒤(미래)는 -inf로 가린다
        scores = np.where(np.tril(np.ones_like(scores)) == 1, scores, -np.inf)
    e = np.exp(scores - scores.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


W = attention_weights(Q, K)
out = W @ V  # 각 토큰이 끌어온 문맥 정보 (residual로 원래 벡터에 더해진다)

print(np.round(W, 2))
assert np.allclose(W.sum(axis=1), 1)      # 각 줄은 확률 분포
assert np.allclose(np.triu(W, 1), 0)      # 미래는 전혀 보지 않는다
assert tokens[W[4].argmax()] == "영희에게"  # "그녀는"은 "영희에게"를 가장 많이 본다
```

출력:

```text
[[1.   0.   0.   0.   0.  ]
 [0.62 0.38 0.   0.   0.  ]
 [0.31 0.51 0.19 0.   0.  ]
 [0.21 0.35 0.35 0.08 0.  ]
 [0.04 0.82 0.01 0.01 0.11]]
```

마지막 줄이 `그녀는`의 시선이다. 82%가 `영희에게`로 간다. `out[4]`에는 "여성인 사람"이라는 정보가 실리고, 이 벡터가 층을 더 거친 뒤 출력층으로 가서 다음 토큰 점수가 된다.

### 생성 루프

```python
def generate(model, tokens, max_new=50, temperature=1.0):
    for _ in range(max_new):
        logits = model(tokens)[-1]                 # 마지막 위치의 점수만 쓴다
        probs = softmax(logits / temperature)      # 낮을수록 1등에 몰린다
        nxt = sample(probs)
        if nxt == EOS:
            break
        tokens.append(nxt)                         # 뒤에 붙이고 다시
    return tokens
```

`model`, `softmax`, `sample`, `EOS`는 자리표시자다. 구조를 보여 주는 의사 코드다.

### 실무와 연결하기

- **RAG가 동작하는 이유**: 검색한 문서를 프롬프트에 넣으면 그 문서의 토큰이 Key/Value가 된다. 답변을 생성하는 위치의 Query가 그 토큰들을 attention으로 읽어 온다. 모델 가중치는 바뀌지 않고, 볼 수 있는 재료만 늘어난다.
- **컨텍스트 길이와 비용**: 토큰 n개가 서로를 모두 보므로 attention 계산량은 n²에 비례해 늘어난다. 프롬프트에 문서를 통째로 넣는 것보다 필요한 부분만 넣는 편이 유리한 이유다.
- **KV cache**: 생성 중 앞 토큰들의 Key/Value는 causal mask 때문에 뒤 토큰의 영향을 받지 않아 변하지 않는다. 서빙 엔진은 이것을 저장해 두고 새 토큰 한 개의 계산만 추가한다. 출력 토큰이 입력 토큰보다 느리고 비싼 이유이기도 하다(출력은 한 개씩 순차 생성).
- **temperature**: 모델 계산이 끝난 뒤 점수를 나누는 값일 뿐이다. 모델이 아는 내용을 바꾸지 않고 선택의 무작위성만 바꾼다. 장비 데이터 추출처럼 재현성이 필요한 작업은 낮게 둔다.

## 참고 자료 (References)

- Vaswani et al., *Attention Is All You Need* (2017): <https://arxiv.org/abs/1706.03762>
- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2, 2019): <https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf>
- Brown et al., *Language Models are Few-Shot Learners* (GPT-3, 2020): <https://arxiv.org/abs/2005.14165>

## 관련 문서

- [Attention이란 무엇인가?](./attention.md) - Q/K/V, multi-head의 개념 설명
- [Transformer가 어떻게 LLM으로 이어졌는가?](./transformer-to-llm.md) - RNN부터 chat model까지의 계보
- [Encoder와 Decoder란 무엇인가?](./encoder-and-decoder.md) - 이 문서가 기준으로 삼은 decoder-only와 다른 구조의 차이
- [Foundation LLM은 어떻게 만들어지는가?](./how-foundation-llms-are-built.md) - 데이터·사전학습·후처리 파이프라인
