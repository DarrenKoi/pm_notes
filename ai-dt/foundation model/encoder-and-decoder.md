---
tags: [encoder, decoder, transformer, bert, gpt, t5]
level: beginner-to-intermediate
last_updated: 2026-03-14
status: in-progress
---

# Encoder와 Decoder란 무엇인가?

> Encoder는 입력을 읽어 문맥 표현으로 바꾸는 쪽이고, Decoder는 그 표현 또는 이전 출력들을 바탕으로 다음 출력을 생성하는 쪽이다.

## 가장 짧은 정의

- **Encoder**: 입력 시퀀스를 읽고 contextual representation으로 변환한다
- **Decoder**: 그 표현을 사용해 출력 시퀀스를 한 토큰씩 생성한다

Transformer에서는 이 둘이 각각 attention block의 조합으로 구현된다.

## 왜 둘을 나누는가?

입력과 출력의 역할이 다르기 때문이다.

- 입력을 잘 이해하려면 문장 전체를 양방향으로 보는 것이 유리하다
- 출력을 생성하려면 미래 정답을 보면 안 되므로 autoregressive 제약이 필요하다

그래서 encoder와 decoder는 attention 사용 방식이 다르다.

## 원래 seq2seq에서의 Encoder / Decoder

초기 seq2seq RNN을 기준으로 보면:

```text
입력 문장 --> Encoder --> 문맥 벡터 --> Decoder --> 출력 문장
```

- encoder는 입력 문장을 압축한다
- decoder는 그 압축 정보를 바탕으로 번역/요약/생성을 한다

문제는 긴 입력을 하나의 벡터에 과도하게 압축해야 했다는 점이다. attention이 나오면서 decoder가 encoder의 여러 상태를 다시 참고할 수 있게 되었고, Transformer는 이 흐름을 attention 중심으로 다시 설계했다.

## Transformer Encoder는 무엇을 하나?

Transformer encoder block은 보통 다음 순서로 생각하면 된다.

1. 입력 토큰 임베딩 + 위치 정보
2. self-attention
3. feed-forward network
4. residual connection + layer normalization

encoder의 핵심 성질:

- 입력 전체를 서로 보게 한다
- 각 토큰을 문맥 반영 표현(contextualized embedding)으로 바꾼다
- 보통 양방향 문맥을 허용한다

예를 들어 BERT의 encoder는 어떤 단어를 해석할 때 **왼쪽과 오른쪽 문맥을 모두 사용**한다.

## Transformer Decoder는 무엇을 하나?

Transformer decoder block은 보통 세 부분으로 생각하면 된다.

1. masked self-attention
2. encoder-decoder cross-attention (필요한 경우)
3. feed-forward network

decoder의 핵심 성질:

- 현재 위치는 과거 토큰만 볼 수 있다
- 미래 토큰은 causal mask로 가린다
- 필요하면 encoder 출력도 cross-attention으로 참고한다

즉 decoder는 "지금까지 생성한 것"을 기반으로 다음 토큰을 예측하는 구조다.

## Self-Attention 관점에서 차이 보기

| 구조 | 무엇을 볼 수 있나? | 주된 목적 |
|------|-------------------|-----------|
| encoder self-attention | 입력 전체 | 이해, 표현 학습 |
| decoder masked self-attention | 현재까지의 출력만 | 생성 |
| decoder cross-attention | encoder 출력 | 입력 조건부 생성 |

## 세 가지 대표 구조

### 1. Encoder-only

대표 예: BERT

특징:

- encoder만 사용
- 양방향 문맥 이해에 강함
- 보통 문장 분류, 토큰 분류, retrieval embedding 같은 작업에 적합

학습 목표 예:

- masked language modeling

질문으로 바꾸면:

- "이 입력이 무슨 뜻인가?"
- "이 단어가 어떤 문맥에서 쓰였는가?"

### 2. Decoder-only

대표 예: GPT 계열

특징:

- decoder만 사용
- 과거 토큰만 보며 다음 토큰을 생성
- open-ended text generation, code generation, chat에 매우 강함

학습 목표 예:

- next-token prediction

질문으로 바꾸면:

- "다음에 올 토큰은 무엇인가?"

### 3. Encoder-Decoder

대표 예: T5, 번역기, 요약 모델

특징:

- encoder가 입력을 읽고
- decoder가 encoder 출력을 참고하며 결과를 생성
- 입력과 출력이 분명히 다른 태스크에 잘 맞음

학습 목표 예:

- text-to-text generation
- denoising / span corruption 계열

질문으로 바꾸면:

- "이 입력을 바탕으로 어떤 출력 시퀀스를 생성해야 하는가?"

## 그림으로 보면

### Encoder-only

```text
입력 토큰 --> Encoder stack --> contextual embeddings --> 분류/검색/태스크 head
```

### Decoder-only

```text
입력 프롬프트 --> Decoder stack (causal mask) --> 다음 토큰 --> 다음 토큰 ...
```

### Encoder-Decoder

```text
입력 토큰 --> Encoder stack ----+
                                |
출력 시작 토큰 --> Decoder stack +--> 다음 토큰 생성
```

## BERT, GPT, T5를 여기서 정리하면

| 모델 | 구조 | 무엇에 강한가? | 왜 그런가? |
|------|------|---------------|-------------|
| BERT | encoder-only | 이해, 분류, 검색 | 양방향 문맥 표현이 강함 |
| GPT | decoder-only | 생성, chat, code | autoregressive objective가 생성과 직접 연결됨 |
| T5 | encoder-decoder | 번역, 요약, text-to-text | 입력 이해와 출력 생성이 분리됨 |

## 자주 헷갈리는 포인트

### 1. "encode"는 토크나이징과 같은가?

아니다. 토크나이징은 텍스트를 토큰 ID로 바꾸는 전처리 단계다. encoder는 그 토큰 시퀀스를 받아 **문맥 표현**을 만드는 신경망 블록이다.

### 2. decoder는 항상 encoder가 있어야 하나?

아니다. GPT처럼 decoder-only 모델은 encoder 없이도 동작한다. 이 경우 decoder는 프롬프트 자체를 조건으로 삼아 생성한다.

### 3. encoder가 더 "좋은" 구조인가?

태스크에 따라 다르다.

- 분류 / 검색 / 표현 학습: encoder 계열이 유리한 경우가 많다
- 자유 생성 / 대화 / 코드 생성: decoder 계열이 유리한 경우가 많다
- 번역 / 요약 / 변환: encoder-decoder가 자연스러운 경우가 많다

## 오늘날 LLM 맥락에서 왜 중요한가?

LLM을 이해할 때 "이 모델이 encoder인가, decoder인가?"를 먼저 구분하면 아래가 바로 정리된다.

- 학습 목표가 무엇인지
- 문맥을 어떤 방식으로 보는지
- 어떤 작업에 강한지
- 왜 chat LLM 대부분이 decoder-only인지

즉 encoder / decoder는 단순 용어가 아니라 **모델의 행동 방식**을 결정하는 핵심 분류다.

## 참고 자료 (Primary Sources)

- Vaswani et al., *Attention Is All You Need* (2017): <https://arxiv.org/abs/1706.03762>
- Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (2018): <https://arxiv.org/abs/1810.04805>
- Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (T5, 2019): <https://arxiv.org/abs/1910.10683>
- Brown et al., *Language Models are Few-Shot Learners* (GPT-3, 2020): <https://arxiv.org/abs/2005.14165>
