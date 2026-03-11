# Unsloth 개요

> Unsloth는 "모델 품질을 자동으로 올려주는 마법"이라기보다, Hugging Face 기반 파인튜닝을 더 빠르고 가볍게 돌리게 해주는 최적화 레이어에 가깝다.

## Unsloth란?

Unsloth는 작은 LLM과 일반 LLM의 LoRA / QLoRA / SFT / 일부 RL 학습 워크플로우를 최적화해서, 같은 계열의 학습을 더 적은 VRAM과 더 빠른 속도로 수행하게 해주는 도구다. 실무에서는 보통 다음 조합으로 이해하면 가장 정확하다.

- 모델 로딩: `transformers`
- 데이터/트레이너: `datasets`, `trl`
- 파라미터 효율 학습: `peft`
- 메모리/속도 최적화: `unsloth`

즉, 기존 Hugging Face 생태계를 버리지 않고 얹어 쓰는 형태다.

## 무엇이 특별한가?

공식 문서와 Hugging Face TRL 통합 문서 기준으로 Unsloth의 핵심 포인트는 다음과 같다.

- Triton 기반 커널과 수작업 backprop 최적화로 학습 속도와 VRAM 사용량을 줄인다
- LoRA / QLoRA를 로컬 GPU에서 더 쉽게 시도할 수 있게 한다
- 긴 context 학습과 추론 시나리오를 비교적 낮은 메모리로 다루기 좋다
- 학습 후 GGUF, Ollama, llama.cpp, vLLM으로 연결하기 쉽다

TRL 문서는 Unsloth를 사용할 때 "`up to 2x faster`" 그리고 "`up to 80% less VRAM`" 같은 효과를 언급한다. 이 수치는 환경마다 달라지므로 절대값으로 보기보다, "동일한 Hugging Face 계열 학습을 로컬 환경에서 훨씬 실용적으로 만든다" 정도로 이해하는 편이 좋다.

## 왜 사람들이 쓰는가?

### 1. 로컬 GPU 실험 속도가 빨라진다

파인튜닝은 한 번에 끝나지 않는다. 데이터셋 정제, chat template 수정, 하이퍼파라미터 조정, export 검증을 반복해야 한다. Unsloth는 이 반복 주기를 짧게 만든다.

### 2. 작은 VRAM에서도 시작할 수 있다

공식 requirements 문서는 QLoRA 기준 대략 다음 수준을 제시한다.

- 3B: 약 3.5GB+
- 7B: 약 5GB+
- 8B: 약 6GB+
- 14B: 약 8.5GB+

실전에서는 데이터 길이, batch size, gradient accumulation, target modules 수에 따라 더 필요할 수 있지만, "작은 GPU로도 시작 가능하다"는 점이 가장 큰 매력이다.

### 3. 기존 Hugging Face 워크플로우와 충돌이 적다

완전히 새로운 프레임워크를 배우는 것이 아니라 `FastLanguageModel`을 앞단에 두고 `SFTTrainer`를 활용하는 식이라, 이미 HF 생태계를 쓰던 사람에게 진입 장벽이 낮다.

### 4. 학습 후 배포 경로가 좋다

adapter 저장, merged 저장, GGUF 변환, Ollama / llama.cpp / vLLM 연결까지 이어지는 흐름이 명확하다.

## Unsloth가 특히 잘 맞는 작업

- 특정 JSON schema를 안정적으로 출력하게 만들기
- 사내 문서 스타일에 맞는 요약/분류/추출
- 툴 호출 형식과 assistant tone 고정
- 작은 도메인에 맞춘 상담/지원 응답 패턴 적응
- 특정 코드베이스나 API 스타일에 맞춘 coding assistant 조정

핵심은 "행동 패턴"을 학습시키는 데 강하다는 점이다.

## Unsloth만으로 부족한 작업

### 최신 지식 주입

파인튜닝은 모델 파라미터를 바꾸므로, 자주 바뀌는 사실 정보를 넣는 용도로는 비효율적이다.

### 출처 기반 QA

정확한 citation, 원문 추적, 문서별 권한 제어가 필요하면 RAG가 중심이어야 한다.

### 모델 weights 없이 API만 있는 경우

Unsloth는 로컬 또는 다운로드 가능한 모델 weights를 기준으로 학습한다. API endpoint 뒤에 있는 모델은 teacher / judge로는 써도, 그 모델 자체를 Unsloth로 직접 파인튜닝할 수는 없다.

## 로컬 환경에서의 현실적 포지션

질문처럼 이미 다음 자원이 있는 경우가 Unsloth에 잘 맞는다.

- 로컬 API LLM: 강한 teacher / evaluator
- GPU에 상주한 sLLM: 실제 student 모델

이 조합이면 다음 패턴이 가장 효율적이다.

1. teacher 모델로 synthetic data 생성
2. 사람이 샘플 검수 및 정제
3. student 모델을 Unsloth로 QLoRA/SFT
4. held-out eval set과 judge model로 비교 평가
5. GGUF 또는 merged weights로 배포

## instruct 모델 vs base 모델

Unsloth의 model selection 가이드는 데이터 양에 따라 다음처럼 안내한다.

- 300개 미만: instruct 모델 쪽이 유리한 경우가 많다
- 300 ~ 1,000개: instruct 또는 base 모두 가능
- 1,000개 이상: base 모델도 적극 고려 가능

첫 실험에서는 보통 instruct 모델이 더 안전하다. 이미 대화 형식과 assistant behavior가 잡혀 있어, 작은 데이터셋에서도 원하는 형식으로 수렴하기 쉽기 때문이다.

## 주의할 점

### Chat template mismatch

공식 문서는 잘못된 chat template 사용을 대표적인 실패 원인으로 강조한다. 학습 때 쓴 template과 추론 때 쓰는 template이 다르면 성능이 무너지기 쉽다.

### 데이터 품질이 하이퍼파라미터보다 중요하다

Synthetic data를 쉽게 많이 만들 수 있어도, 잘못된 예시를 넣으면 모델이 그대로 나쁜 습관을 배운다.

### QLoRA부터 시작하는 것이 안전하다

처음부터 full fine-tuning에 가기보다 QLoRA로 충분한지 먼저 확인하는 편이 비용 대비 효율이 좋다.

## 참고 자료

- Unsloth Fine-Tuning Guide: <https://unsloth.ai/docs/get-started/fine-tuning-guide>
- Unsloth Requirements: <https://docs.unsloth.ai/get-started/fine-tuning-for-beginners/unsloth-requirements>
- Unsloth Model Selection Guide: <https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/what-model-should-i-use>
- Hugging Face TRL Unsloth Integration: <https://huggingface.co/docs/trl/en/unsloth_integration>
