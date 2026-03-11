# 로컬 sLLM 파인튜닝 워크플로우

> 강한 로컬 API 모델을 teacher / judge로 쓰고, GPU에 올라간 작은 모델을 student로 파인튜닝하는 방식이 현재 가장 실용적이다.

## 전체 그림

```text
로컬 API LLM (teacher / judge)
        |
        +--> seed data 확장
        +--> synthetic data 생성
        +--> eval / ranking / error analysis

로컬 GPU sLLM (student)
        |
        +--> Unsloth QLoRA / SFT
        +--> adapter or merged weights
        +--> GGUF / Ollama / vLLM 배포
```

## Step 1. 태스크를 하나로 좁힌다

처음부터 "우리 도메인 전체를 똑똑하게 만들기"로 시작하면 실패하기 쉽다. 다음처럼 좁은 태스크로 잘라야 한다.

- 지원 티켓을 정해진 JSON으로 분류하기
- 장애 리포트를 5줄 요약으로 정리하기
- 내부 문서 질의응답을 특정 tone으로 답하게 하기
- 사내 툴 호출을 위한 function-call style 출력 맞추기

태스크가 좁을수록 작은 모델이 학습한 패턴을 안정적으로 재현한다.

## Step 2. student 모델을 고른다

첫 시도 기준으로는 instruct 모델 + QLoRA가 가장 안전하다.

### 모델 선택 기준

- VRAM이 작다: 3B ~ 8B instruct 모델부터 시작
- 데이터가 300개 미만이다: instruct 모델 우선
- 데이터가 1,000개 이상이고 패턴이 분명하다: base 모델도 검토

### 현실적인 첫 선택

- 범용 assistant 튜닝: 4B ~ 8B instruct
- JSON 추출/분류: 3B ~ 8B instruct
- 코딩 쪽: 가능하면 code-specialized instruct 계열 우선

## Step 3. teacher 모델의 역할을 분리한다

teacher 모델을 student 대신 쓰는 것이 아니라, 학습 준비와 평가를 맡긴다.

### teacher에 맡길 일

- seed examples 확장
- bad example rewriting
- edge case 생성
- student output 채점
- pairwise ranking
- 오류 유형 태깅

### student에 맡길 일

- 실제 배포 대상 behavior 학습
- 지연시간/메모리/비용 최적화

## Step 4. seed dataset을 만든다

최소한 다음 3종류가 필요하다.

- 정상 사례
- 실패하기 쉬운 경계 사례
- 금지/거절/예외 처리 사례

처음부터 1,000개를 만들 필요는 없다. 사람이 직접 만든 고품질 seed 50 ~ 200개가 보통 더 중요하다.

## Step 5. synthetic data를 늘린다

Unsloth datasets guide는 synthetic data 활용 자체를 적극적으로 다루지만, 그대로 넣기보다 검수 루프를 넣어야 한다.

권장 방식:

1. 사람이 소량의 gold examples 작성
2. teacher 모델로 변형 예시 생성
3. 중복/허위/과잉 길이 제거
4. 사람이 샘플 검수
5. train / eval 분리

## Step 6. chat format을 고정한다

학습 전 반드시 다음을 하나로 통일한다.

- system prompt 사용 여부
- role 이름 (`system`, `user`, `assistant`)
- tool-call 출력 형식
- EOS 처리
- 추론 시 generation prompt 처리

여기서 틀리면 학습이 잘 돼도 배포 시 성능이 무너진다.

## Step 7. 첫 학습은 작게 돌린다

권장 초기값:

- `max_seq_length=2048`
- `r=16`
- `lora_alpha=16` 또는 `32`
- `per_device_train_batch_size=2`
- `gradient_accumulation_steps=8`
- `learning_rate=2e-4`
- `num_train_epochs=1`

목표는 최고 점수가 아니라, 파이프라인이 정상 작동하는지 확인하는 것이다.

## Step 8. base vs fine-tuned를 반드시 비교한다

평가 항목 예시:

- JSON schema validity
- instruction following rate
- refusal correctness
- hallucination rate
- latency
- token length

fine-tuned 모델이 항상 더 좋은 것은 아니다. 데이터가 약하면 오히려 base instruct 모델보다 나빠질 수 있다.

## Step 9. teacher judge로 자동 평가를 붙인다

강한 로컬 API 모델을 judge로 활용하면 반복 속도가 빨라진다. 다만 judge score만 믿지 말고, 사람이 직접 보는 검수 세트를 따로 유지해야 한다.

권장 평가 세트:

- small manual set: 30 ~ 100개
- larger automatic set: 200개 이상

## Step 10. export 경로를 미리 정한다

학습 전에 배포 목표를 먼저 정하는 편이 좋다.

- Ollama / llama.cpp 예정: GGUF export 필요
- vLLM 예정: merged weights 또는 호환 가능한 HF 형식 필요
- adapter만 공유 예정: PEFT adapter 저장

## 추천 운영 방식

### 가장 안전한 시작점

1. 4B 또는 8B instruct student 선택
2. 고품질 seed 100개 작성
3. teacher로 500 ~ 3,000개까지 확장
4. 사람이 샘플 검수
5. Unsloth QLoRA 1 epoch
6. held-out eval + 수동 검수
7. GGUF export 후 로컬 serving 비교

### 피해야 할 시작점

- 처음부터 full fine-tuning
- template 정의 없이 데이터부터 대량 생성
- eval set 없이 train loss만 보고 판단
- API 모델과 동일한 품질을 작은 student에 바로 기대

## 참고 자료

- Unsloth Datasets Guide: <https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/datasets-guide>
- Unsloth Model Selection Guide: <https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/what-model-should-i-use>
- Unsloth Fine-Tuning Guide: <https://unsloth.ai/docs/get-started/fine-tuning-guide>
