---
tags: [foundation-model, llm, pretraining, scaling, alignment, rlhf]
level: intermediate
last_updated: 2026-03-14
reviewed_on: 2026-10-04
review_status: partial
type: learning
category_major: "AI·DT"
category_middle: "AI 기초"
category_minor: "파운데이션 모델"
note_kind: "학습"
classified_on: "2026-10-05"
---

# Foundation LLM은 어떻게 만들어지는가?

> Foundation LLM은 단순히 "매우 큰 모델"이 아니라, broad data 위에서 자기지도 사전학습을 수행하고 다양한 작업에 적응할 수 있도록 만든 기반 모델이다.

> [!info] 검토 범위 · 2026-10-04
> 공개 원 논문에 근거한 학습 문서다. 논문의 발표·개정 연도와 오늘의 제품 구현은 구분한다. 최신 제품의 구조·학습 공정·시장 점유율은 미확인이다. [정리 기록](./organization-log.md)에 근거와 검증 한계를 남겼다.

## 먼저 용어부터

Bommasani et al. (2021)은 foundation model을 **광범위한 데이터로 대규모 학습하여 다양한 downstream task에 적응 가능한 모델**이라는 관점으로 정리했다.

중요한 구분:

- **foundation model**: 범용 기반 모델
- **base language model**: 다음 토큰 예측 등 기본 objective로 사전학습된 언어모델
- **instruction / chat model**: base model에 사람 지시를 따르도록 추가 정렬을 한 모델

즉 사용자가 접하는 chat model은 보통 foundation model 제작 과정의 "후반 단계 산출물"이다.

## 이 문서의 범위

이 문서는 BERT, T5, GPT-3, Chinchilla, InstructGPT 같은 1차 자료를 종합한 **일반화된 제작 파이프라인**이다. 실제 기업별 세부 데이터 소스, 필터링 규칙, 학습 인프라는 공개 범위가 다르므로 구체 공정은 조금씩 달라진다.

## 전체 파이프라인

```text
목표 정의
-> 아키텍처 선택
-> 데이터 수집 / 정제
-> 토크나이저 설계
-> 사전학습(pretraining)
-> 평가 / 체크포인트 선택
-> post-training (SFT, preference tuning, RLHF/DPO 등)
-> 안전성 / 배포 / 지속 개선
```

## 1. 목표와 아키텍처를 먼저 정한다

처음부터 "모델을 얼마나 크게 만들까?"보다 먼저 정해야 하는 것은 **모델이 주로 무엇을 하게 할 것인가**다.

### 대표 선택지

- **encoder-only**: 이해, 분류, retrieval, embedding
- **decoder-only**: 생성, 대화, 코드, agent
- **encoder-decoder**: 번역, 요약, text-to-text 변환

GPT-3는 decoder-only 자동회귀 언어모델의 공개 사례다. next-token prediction은 생성 인터페이스와 연결된다. 현재 제품들의 구조 비중은 이 문서에서 검증하지 않았다.

## 2. 데이터 수집과 정제가 가장 큰 공정 중 하나다

Foundation LLM은 broad data를 사용한다. 예시는 다음과 같다.

- 웹 문서
- 책
- 위키 / 백과
- 코드 저장소
- 수학 / 과학 문서
- 대화 / instruction 데이터

하지만 "많이 모은다"로 끝나지 않는다. 실제로는 대규모 필터링이 중요하다.

### 왜 정제가 중요한가?

- 중복이 많으면 학습 효율이 떨어진다
- 스팸, 저품질, 깨진 텍스트가 많다
- 유해 콘텐츠나 정책 위반 데이터가 섞여 있을 수 있다
- 언어 분포가 치우치면 편향된 모델이 된다

그래서 보통 다음을 수행한다.

- deduplication
- 품질 필터링
- 언어 식별
- 포맷 정규화
- 데이터 mixture 비율 조정

데이터와 아키텍처의 기여도를 항상 같은 순위로 놓을 수는 없다. 품질·mixture를 바꿀 때는 다른 조건을 통제한 목표 평가로 효과를 확인한다.

## 3. 토크나이저를 만든다

텍스트를 모델의 vocabulary ID로 변환하는 tokenizer가 필요하다. 단위는 subword, 문자, byte 등 설계에 따라 다르다. 기존 checkpoint를 이어 학습할 때는 그 tokenizer와 special token 규약을 먼저 확인한다. 새 tokenizer를 만들면 임베딩 크기·ID 대응까지 함께 맞춰야 한다.

토크나이저가 중요한 이유:

- 같은 문장을 몇 개 토큰으로 표현하는지 결정한다
- 언어별 효율이 달라진다
- 코드, 숫자, 수식, 한국어 같은 도메인에서 품질 차이가 생긴다
- context window 실효 길이에 직접 영향을 준다

즉 tokenizer는 단순 전처리가 아니라 **모델이 세상을 자르는 방식**이다.

## 4. 사전학습 objective를 정한다

사전학습은 사람이 정답 라벨을 다 붙이지 않고도 수행할 수 있어야 한다. 그래서 자기지도(self-supervised) objective를 쓴다.

### 대표 objective

- **next-token prediction**: 다음 토큰 맞히기
- **masked language modeling**: 가린 토큰 복원하기
- **denoising / span corruption**: 손상된 입력을 복원하기

각 objective는 구조와 잘 맞아야 한다.

- BERT: encoder-only + masked LM
- GPT 계열: decoder-only + next-token prediction
- T5: encoder-decoder + text-to-text / denoising

## 5. 대규모 분산 학습으로 base model을 만든다

이제 모델 파라미터를 실제로 학습한다. 여기서 필요한 것은:

- 매우 큰 데이터셋
- 많은 GPU/TPU
- 안정적인 optimizer / scheduler
- mixed precision
- checkpointing
- distributed training

이 단계의 산출물은 보통 **base model**이다. 아직 사람 지시를 잘 따르는 assistant가 아니라, 텍스트 분포를 잘 예측하는 모델에 가깝다.

## 6. 스케일링은 "크게만" 하면 되는 문제가 아니다

한동안은 "파라미터를 크게 만들수록 좋다"는 인식이 강했지만, Chinchilla는 **모델 크기와 학습 토큰 수 사이 균형**이 중요하다고 보여줬다.

핵심 메시지:

- 파라미터만 늘리고 데이터가 부족하면 비효율적일 수 있다
- 같은 compute budget 안에서도 더 좋은 배분이 있다
- 모델 크기, 학습 토큰 수, 연산량은 함께 설계해야 한다

Chinchilla (2022, v1)는 고정된 **학습 compute budget**과 논문 실험 범위에서 크기·토큰 배분을 분석했다. 특정 비율을 모든 데이터·배포 비용의 최적값으로 일반화하지 않는다. 추론 지연·메모리·사용량까지 고려하는 제품 목표는 별도로 정한다.

## 7. 평가와 체크포인트 선택

사전학습 중에는 계속 평가를 수행한다.

### 평가하는 것

- validation loss / perplexity
- downstream benchmark
- reasoning / code / math 성능
- 안전성 및 유해 응답
- 데이터 오염(data contamination) 여부

모델은 단순히 loss가 낮다고 끝나지 않는다. 실제 제품 목표와 안전 요구를 함께 보며 checkpoint를 선택한다.

## 8. Post-Training: foundation model을 assistant로 바꾸는 단계

많은 사용자가 여기서 LLM의 "성격"을 체감한다.

### 1. Supervised Fine-Tuning (SFT)

사람이 작성한 예시 답변이나 고품질 instruction-response 데이터를 사용해, 모델이 원하는 응답 형식을 따르게 만든다.

효과:

- 지시를 더 잘 따른다
- 답변 형식이 안정된다
- assistant처럼 대화하는 습관을 학습한다

### 2. Preference Learning

두 개 이상의 응답 후보를 비교해 "어느 쪽이 더 좋은가"를 수집한다.

이 데이터는:

- reward model 학습
- RLHF
- DPO 같은 preference optimization

으로 이어질 수 있다.

### 3. RLHF / Preference Optimization

InstructGPT의 대표 흐름은 다음과 같다.

1. 사전학습된 LM 준비
2. 시범 답변으로 SFT 수행
3. 인간 선호 데이터를 모아 reward model 학습
4. RL 기반으로 응답 정책을 더 선호 방향으로 조정

[DPO (2023, arXiv v3)](https://arxiv.org/abs/2305.18290v3)는 선호 쌍을 이용해 정책을 직접 최적화하는 대안이다. InstructGPT의 별도 reward model 학습과 PPO 절차를 그대로 수행하는 방법은 아니다. 선택한 방법·선호 데이터·평가 조건을 구분한다. 확인 2026-10-04.

## 9. 안전성, 정책, 제품화

실제 배포 가능한 모델은 성능만 좋아서는 부족하다.

- 유해 요청 대응
- 정책 위반 거절
- 개인정보 / 보안 이슈 대응
- hallucination 완화
- 툴 사용, retrieval, memory 연결

그래서 production LLM은 보통 "pretraining으로 끝난 모델"이 아니라, 정책과 UX 요구까지 반영된 여러 단계의 조합물이다.

## 10. Base model과 Chat model의 차이

| 항목 | Base model | Chat / Instruct model |
|------|------------|-----------------------|
| 주된 학습 | broad data pretraining | SFT, 선호 최적화, 안전성 조정 등 선택한 후처리 |
| 기본 행동 | 텍스트 계속 쓰기 | 지시 따르기, 대화하기 |
| 출력 경향 | completion 중심 | assistant 답변 중심 |
| 장점 | 범용성, 재학습 재료 | 사용자 경험, 제어 가능성 |

즉 "foundation model이 만들어진다"와 "좋은 챗봇이 만들어진다"는 같은 말이 아니다.

## 11. 실무 감각으로 정리

Foundation LLM 제작의 핵심은 아래 네 축이다.

- **Architecture**: 무엇을 잘하게 만들 것인가
- **Data**: 무엇을 얼마나 깨끗하게 먹일 것인가
- **Scale**: compute budget 안에서 얼마나 균형 있게 키울 것인가
- **Alignment**: 사람 기대에 맞게 어떻게 다듬을 것인가

앞의 세 축은 기반 모델 제작을, alignment는 지시·대화 목적의 추가 조정을 살피는 관점이다. 사용 목적에 맞는 평가 없이 경쟁력을 보장하는 공식으로 쓰지 않는다.

## 한 줄 결론

Foundation model의 정의는 광범위한 데이터의 대규모 학습과 여러 작업으로의 적응 가능성에 있다. 이 문서의 자기지도 언어모델 파이프라인은 대표 사례이며, **post-training alignment는 foundation model의 필수 정의 조건이 아니다**. assistant를 만들 목적이라면 추가 학습과 행동 평가를 설계한다.

## 참고 자료 (Primary Sources)

- Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (2018): <https://arxiv.org/abs/1810.04805>
- Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (T5, 2019): <https://arxiv.org/abs/1910.10683>
- Brown et al., *Language Models are Few-Shot Learners* (GPT-3, 2020): <https://arxiv.org/abs/2005.14165>
- Bommasani et al., *On the Opportunities and Risks of Foundation Models* (2021): <https://arxiv.org/abs/2108.07258>
- Hoffmann et al., *Training Compute-Optimal Large Language Models* (Chinchilla, 2022): <https://arxiv.org/abs/2203.15556>
- Ouyang et al., *Training language models to follow instructions with human feedback* (InstructGPT, 2022): <https://arxiv.org/abs/2203.02155>

- Rafailov et al., *Direct Preference Optimization* (2023; arXiv v3, 2024): <https://arxiv.org/abs/2305.18290v3>
