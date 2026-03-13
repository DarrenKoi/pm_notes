---
tags: [transformer, llm, foundation-model, bert, gpt, t5]
level: beginner-to-advanced
last_updated: 2026-03-14
status: in-progress
---

# Transformer가 어떻게 LLM으로 이어졌는가?

> LLM은 갑자기 등장한 별도 기술이 아니라, attention 기반 Transformer가 대규모 자기지도 사전학습과 결합되면서 발전한 결과다.

## 큰 흐름 먼저 보기

```text
RNN / LSTM seq2seq
-> attention 도입
-> Transformer 등장
-> 대규모 pretraining
-> BERT / GPT / T5 계열 분화
-> 더 큰 데이터 + 더 큰 모델 + 더 큰 compute
-> foundation model
-> instruct / chat model
```

## 1. Transformer 이전: 왜 한계가 있었나?

Transformer 이전의 대표 구조는 RNN/LSTM/GRU 기반 seq2seq였다.

이 구조는 분명 잘 작동했지만, 대규모 언어모델로 커지기에는 몇 가지 약점이 있었다.

- **순차 계산 의존성**: 토큰을 앞에서 뒤로 차례대로 처리해야 한다
- **긴 거리 의존성 문제**: 멀리 떨어진 단어 간 관계를 잡는 데 어려움이 있다
- **병렬화 한계**: GPU/TPU에서 대규모 학습 효율이 떨어진다
- **고정 길이 병목**: 초기 encoder-decoder는 입력 정보를 좁은 bottleneck에 압축하려는 경향이 강했다

Bahdanau attention은 병목을 완화했지만, 여전히 본체는 recurrent 구조였다.

## 2. Transformer의 핵심 전환

Vaswani et al. (2017)은 recurrence와 convolution 없이도 sequence transduction이 가능하다고 보였고, 핵심 연산을 self-attention으로 재구성했다.

Transformer의 핵심 조합은:

- self-attention
- multi-head attention
- position information
- feed-forward network
- residual connection + layer normalization

이 설계의 의미는 단순히 "새 모델이 나왔다"가 아니다. **언어를 GPU 친화적으로 크게 학습할 수 있는 공통 블록**이 생겼다는 점이 중요하다.

## 3. 왜 Transformer가 LLM에 유리했나?

### 병렬 학습이 가능했다

RNN은 토큰별 계산이 직렬적이지만, Transformer는 한 layer 안에서 토큰 간 관계를 행렬 연산으로 처리할 수 있다. 이 점이 대규모 학습 인프라와 매우 잘 맞았다.

### 긴 문맥 연결이 쉬워졌다

멀리 떨어진 단어라도 attention score만 높으면 직접 정보를 주고받을 수 있다. RNN처럼 많은 time step을 거쳐 정보가 전달되지 않아도 된다.

### 같은 블록을 다양한 목적에 재사용할 수 있었다

- 문장 이해
- 문서 분류
- 번역
- 요약
- 질의응답
- 언어 생성

하나의 Transformer 계열 블록을 학습 목적만 바꿔 다양한 태스크에 적용할 수 있었다.

## 4. Transformer 이후 세 가지 큰 계열

Transformer가 등장한 뒤, 어떤 부분을 강조하느냐에 따라 크게 세 계열이 강해졌다.

| 계열 | 대표 모델 | 핵심 특징 | 잘 맞는 작업 |
|------|-----------|-----------|--------------|
| encoder-only | BERT | 양방향 문맥 이해 | 분류, 검색, NLU |
| decoder-only | GPT 계열 | autoregressive 생성 | 텍스트 생성, chat, code |
| encoder-decoder | T5 | 입력 읽기 + 출력 생성 분리 | 번역, 요약, text-to-text |

### BERT: "이해" 중심의 대규모 사전학습

BERT는 deep bidirectional Transformer encoder를 대규모 비라벨 텍스트에 사전학습한 뒤, 다양한 downstream task에 fine-tuning 하는 방식을 밀어 올렸다.

핵심 포인트:

- 입력 양쪽 문맥을 모두 보며 표현을 만든다
- masked language modeling으로 학습한다
- 자연어 이해 계열 작업에서 매우 강했다

즉 Transformer가 "큰 범용 표현 모델"로 작동할 수 있다는 점을 널리 보여준 사례다.

### GPT 계열: 생성과 스케일링 중심

GPT 계열은 decoder-only Transformer를 autoregressive next-token prediction으로 학습한다.

이 방향의 장점:

- 학습 목표가 단순하고 대규모 웹 텍스트에 맞추기 쉽다
- 생성 태스크와 구조가 자연스럽게 맞는다
- prompt만 바꿔 다양한 작업을 시킬 수 있다

GPT-3는 이 계열이 충분히 커지면 **few-shot / zero-shot 능력**이 emergent하게 나타날 수 있음을 크게 보여줬다.

### T5: text-to-text 통합

T5는 거의 모든 NLP 작업을 "텍스트를 입력받아 텍스트를 출력하는 문제"로 통일했다.

이 관점이 중요한 이유:

- 번역도 text-to-text
- 요약도 text-to-text
- 질의응답도 text-to-text
- 분류도 label string을 출력하게 만들면 text-to-text

즉 Transformer를 task-specific head 모음이 아니라 **범용 텍스트 변환 엔진**으로 볼 수 있게 했다.

## 5. "Transformer"가 "LLM"이 되는 순간

Transformer 자체는 아키텍처다. 여기에 다음 요소가 결합될 때 우리가 흔히 말하는 LLM이 된다.

### 1. 대규모 데이터

- 웹 문서
- 책
- 위키
- 코드
- 대화 데이터

같은 broad data를 매우 크게 사용한다.

### 2. 자기지도 사전학습

정답 라벨을 사람이 일일이 붙이지 않아도 되는 objective를 사용한다.

- next-token prediction
- masked language modeling
- span corruption / denoising

### 3. 대규모 파라미터와 compute

모델이 커질수록 더 많은 패턴을 압축할 수 있고, 충분한 토큰과 연산량이 함께 주어질 때 성능이 크게 올라간다.

### 4. 범용 전이 가능성

하나의 base model을 여러 downstream task에 재사용할 수 있어야 한다.

이 지점에서 Stanford의 Foundation Models 보고서가 말하는 "foundation model" 개념이 등장한다. 즉, **broad data에서 학습되고 다양한 작업의 기반이 되는 모델**이라는 뜻이다.

## 6. 왜 decoder-only LLM이 특히 강해졌나?

오늘날 chat LLM 대부분은 decoder-only 계열이다. 이유는 비교적 명확하다.

- next-token prediction이 웹 규모 텍스트 학습과 매우 잘 맞는다
- 생성 인터페이스 자체가 제품 형태(chat, completion, code generation)와 바로 연결된다
- instruction tuning, RLHF/DPO 같은 post-training을 붙이기 쉽다
- 긴 context에서 연속 생성하는 사용 패턴과 궁합이 좋다

즉 "Transformer -> LLM" 경로에서 가장 산업적으로 크게 확장된 가지가 decoder-only였다.

## 7. Foundation model에서 assistant로

중요한 구분:

- **Base model**: broad data 위에서 사전학습된 모델
- **Instruct / Chat model**: base model을 사람 지시를 따르도록 추가 학습한 모델

InstructGPT는 여기에 대표적이다.

일반적인 흐름:

1. base LM pretraining
2. supervised fine-tuning
3. preference data 수집
4. reward model 또는 preference optimization
5. safety / refusal / policy tuning

그래서 사용자가 체감하는 "대화형 LLM"은 보통 Transformer base model 위에 여러 단계의 post-training이 쌓인 결과물이다.

## 8. 한 줄 결론

Transformer가 LLM으로 이어진 이유는 단순히 attention이 좋아서가 아니다. **attention 기반 구조가 대규모 병렬 사전학습, 범용 전이, 그리고 후속 정렬(alignment) 단계와 결합되기에 가장 적합했기 때문**이다.

## 참고 자료 (Primary Sources)

- Vaswani et al., *Attention Is All You Need* (2017): <https://arxiv.org/abs/1706.03762>
- Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (2018): <https://arxiv.org/abs/1810.04805>
- Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (T5, 2019): <https://arxiv.org/abs/1910.10683>
- Brown et al., *Language Models are Few-Shot Learners* (GPT-3, 2020): <https://arxiv.org/abs/2005.14165>
- Bommasani et al., *On the Opportunities and Risks of Foundation Models* (2021): <https://arxiv.org/abs/2108.07258>
- Ouyang et al., *Training language models to follow instructions with human feedback* (InstructGPT, 2022): <https://arxiv.org/abs/2203.02155>
