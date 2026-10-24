---
type: learning
tags: [finetuning, sllm, framework, unsloth, peft, trl, llama-factory, axolotl, mlx]
level: intermediate
last_updated: 2026-10-11
review_status: partial
category_major: "AI·DT"
category_middle: "모델 학습"
category_minor: "sLLM 파인튜닝"
note_kind: "학습"
---

# 사내 sLLM 튜닝: 프레임워크 선택과 전체 순서

> 오픈 웨이트 sLLM을 사내용으로 튜닝할 때 프레임워크는 대부분 같은 스택(`transformers` + `peft` + `trl`)을 감싼 것이라 바꿔 끼울 수 있다. 바꿔 끼울 수 없는 것은 평가셋과 학습 데이터이므로 거기에 시간을 쓴다.

> [!info] 검토 범위 · 2026-10-11
> 각 프레임워크의 공개 문서를 기준으로 정리한 학습 자료다. 이 저장소에서 설치·학습·서빙을 실행하지 않았다. 멀티 GPU 지원 범위와 지원 모델 목록은 버전마다 바뀌므로 도입 시점의 공식 문서로 다시 확인한다. 공통 조건은 [적용 조건](./verified-conditions.md)을 따른다.

## 왜 필요한가? (Why)

- 이 묶음의 다른 문서는 Unsloth를 전제로 한다. 실제 도입 때는 "Unsloth여야 하는가, 다른 선택지는 무엇인가"를 먼저 답해야 한다.
- 사내 환경은 외부 LLM API가 막혀 있고 패키지·모델 반입에 절차가 있다. 프레임워크를 고를 때 기능보다 **반입 가능성과 GPU 구성**이 먼저 걸린다.
- 프레임워크 비교에 시간을 쓰다가 데이터 준비가 밀리는 경우가 흔하다. 어디에 시간을 써야 하는지 순서를 고정해 둔다.

## 핵심 개념 (What)

### 1. "튜닝"의 종류부터 정한다

| 방식 | 바꾸는 것 | 필요한 데이터 | 언제 |
|------|-----------|---------------|------|
| LoRA / QLoRA SFT | 작은 어댑터만 학습 | 지시-응답 쌍 수백~수천 건 | **기본 선택** |
| Full fine-tuning | 전체 가중치 | 수천~수만 건, 큰 VRAM | LoRA로 한계가 확인된 뒤 |
| 도메인 추가 사전학습 | 전체 가중치, 원문 텍스트 | 대량의 사내 원문 | 용어·표기 자체를 못 읽을 때만 |
| 증류 (distillation) | 위 SFT와 같음 | 큰 모델(teacher)이 만든 응답 | 큰 모델의 특정 작업을 작은 모델로 옮길 때 |
| 선호 학습 (DPO 등) | 어댑터 | 좋은 답/나쁜 답 쌍 | SFT 이후 품질 다듬기 |

사내 지식을 "알게" 하려는 목적이면 튜닝보다 검색(retrieval)이 먼저다. 튜닝은 출력 형식, 판단 습관, 도메인 표기처럼 **행동**을 바꾸는 데 쓴다.

### 2. 프레임워크 비교

| 프레임워크 | 성격 | 잘 맞는 경우 | 주의 |
|------------|------|--------------|------|
| HF `transformers` + `peft` + `trl` | 기준이 되는 원 스택 | 제어가 필요하다, 멀티 GPU, 다른 도구가 지원하지 않는 모델 | 설정을 직접 다 정해야 한다 |
| Unsloth | 위 스택을 메모리·속도 최적화로 감싼 것 | 단일 GPU에서 빠른 반복, GGUF export | 멀티 GPU·최신 모델 지원 범위는 문서 확인 |
| LLaMA-Factory | Web UI + YAML 설정 | 코드를 쓰지 않는 사람도 실험해야 한다, Qwen 계열 등 다양한 모델 | UI가 가린 설정을 모른 채 돌리기 쉽다 |
| Axolotl | YAML 설정 중심 | 설정 파일로 재현성 관리, 멀티 GPU | 설정 항목이 많다 |
| MLX-LM | Apple Silicon 전용 | 맥북에서 파이프라인 리허설 | 사내 CUDA 서버와 결과·산출물 형식이 다르다 |

다섯 모두 `messages` JSONL을 입력으로 받는다. MLX-LM을 뺀 넷은 출력도 PEFT 어댑터 또는 병합 가중치로 같다. MLX-LM은 MLX 형식 어댑터를 만들므로 CUDA 서버로 옮기려면 병합 후 변환이 필요하다. 데이터 형식만 [데이터셋 가이드](./dataset-and-chat-template-guide.md)대로 고정해 두면 프레임워크는 나중에 바꿔도 된다.

### 3. 선택 기준

```text
GPU 1장, 코드로 실험        -> Unsloth
GPU 여러 장 / 사내 표준화   -> transformers + peft + trl (또는 Axolotl)
비개발자도 실험해야 함      -> LLaMA-Factory
아직 GPU 서버가 없음        -> MLX-LM으로 데이터·평가 파이프라인만 먼저 검증
```

### 4. 프레임워크 밖에서 필요한 것

| 단계 | 도구 | 비고 |
|------|------|------|
| 데이터 생성·채점 | 사내 OpenAI 호환 API의 큰 모델 (teacher / judge) | 별도 프레임워크 없이 스크립트로 충분 |
| 문서 추출 | 스크린샷 + VLM | DRM 문서. 추출 오류율을 먼저 잰다 |
| 평가 | 직접 만든 평가셋 + 채점 스크립트 | 공개 벤치마크는 사내 작업 성능을 말해 주지 않는다 |
| 서빙 | vLLM (OpenAI 호환, LoRA 어댑터 로드 가능) 또는 Ollama / llama.cpp (GGUF) | 학습 전에 정한다 |

## 어떻게 사용하는가? (How)

### 전체 순서

```text
0. 태스크 하나로 좁히기        "장애 리포트를 정해진 JSON으로 요약" 수준까지
1. 평가셋 만들기               실제 업무 사례 50~100건 + 채점 기준
2. 기준선 측정                 튜닝 없이 프롬프트(+검색)로 같은 평가셋 채점
   -> 여기서 충분하면 튜닝하지 않는다
3. 반입 확인                   모델 가중치 라이선스, 패키지 미러, GPU 사양
4. student 모델 선택           instruct 계열, 서빙 런타임이 지원하는 구조
5. seed 데이터 작성            사람이 50~200건 (정상 / 경계 / 거절·모름)
6. teacher로 확장              500~3,000건, train 그룹 안에서만, 표본 검수
7. chat template 고정          모델 토크나이저의 template 사용
8. 작은 학습 1회               QLoRA 1 epoch, 파이프라인 동작 확인이 목적
9. base vs tuned 비교          1번의 평가셋으로, 사람이 직접 읽는 검수 포함
10. 틀린 유형 분석 -> 5번으로  건수보다 틀린 유형을 겨냥한 샘플 추가
11. 병합·양자화·서빙           단계마다 같은 평가셋으로 다시 채점
```

1~2번을 건너뛰면 튜닝이 효과가 있었는지 말할 수 없다. 각 단계의 상세는 [워크플로우](./local-sllm-finetuning-workflow.md)에 있다.

### 준비할 데이터셋

| 데이터셋 | 건수 감각 | 만드는 사람 | 용도 |
|----------|-----------|-------------|------|
| 평가셋 | 50~100건 | 현업 담당자 | 성공 기준. 학습에 절대 넣지 않는다 |
| seed | 50~200건 | 현업 담당자 | 정답의 기준이 되는 gold 사례 |
| 확장 학습셋 | 500~3,000건 | teacher 모델 + 사람 표본 검수 | 실제 학습 입력 |
| 자동 평가셋 | 200건 이상 | teacher 모델 + 검수 | 반복 실험의 빠른 채점 |

건수는 시작 후보이며 품질 보장 기준이 아니다. 네 가지 모두 원천 문서(또는 장비, 사건) 단위로 먼저 나눈 뒤 만든다. 같은 원천에서 나온 변형이 학습과 평가 양쪽에 들어가면 점수가 부풀려진다.

seed에는 세 종류가 모두 있어야 한다.

- 정상 사례
- 실패하기 쉬운 경계 사례
- 근거가 없어 "모른다"고 답하거나 거절해야 하는 사례

### 프레임워크 없이 쓰는 최소 학습 코드

Unsloth 버전은 [학습 및 배포 레시피](./training-and-deployment-recipe.md)에 있다. 아래는 같은 일을 원 스택으로 하는 형태다. 두 코드를 비교하면 Unsloth가 무엇을 대신해 주는지 보인다.

```python
# 실행 전제: CUDA GPU, datasets / peft / trl 설치, 모델 가중치가 사내 경로에 반입되어 있음
# 이 저장소에서 실행 검증하지 않은 예제다
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

# train.jsonl: 한 줄에 {"messages": [{"role": ..., "content": ...}, ...]} 하나
train = load_dataset("json", data_files="train.jsonl", split="train").select_columns(["messages"])

trainer = SFTTrainer(
    model="/models/student-instruct",  # 사내에 반입한 모델 경로
    train_dataset=train,               # messages 컬럼은 모델의 chat template로 자동 변환된다
    args=SFTConfig(
        output_dir="out",
        num_train_epochs=1,
        learning_rate=2e-4,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        logging_steps=10,
    ),
    peft_config=LoraConfig(r=16, lora_alpha=16, target_modules="all-linear", task_type="CAUSAL_LM"),
)
trainer.train()
trainer.save_model("out/adapter")  # 어댑터만 저장된다. base 모델 경로·revision을 함께 기록한다
```

이 코드는 모델을 양자화하지 않고 올린다. VRAM이 부족하면 4-bit 로드(QLoRA)가 필요하고, 그 설정을 대신 해 주는 것이 Unsloth의 주된 편의다.

### 실무와 연결하기

- **폐쇄망**: 프레임워크마다 의존 패키지가 다르다. 반입 승인을 한 번 받을 거라면 원 스택(`transformers`, `peft`, `trl`, `datasets`, `accelerate`, `bitsandbytes`)을 기준으로 받는다. Unsloth·Axolotl·LLaMA-Factory 모두 그 위에 올라간다.
- **teacher 모델**: 사내 API의 큰 텍스트 모델을 데이터 확장과 채점에 쓴다. 같은 모델이 만들고 같은 모델이 채점하면 점수가 후해지므로 사람이 읽는 검수 세트를 따로 둔다.
- **서빙 정합**: 사내 서빙이 OpenAI 호환 API라면 튜닝한 모델도 vLLM으로 같은 인터페이스에 올릴 수 있다. 기존 클라이언트 코드는 모델 이름만 바꾸면 된다.
- **기밀**: 학습 데이터에 들어간 내용은 출력으로 재현될 수 있다. 구체적인 장비 데이터는 학습 데이터에 넣기 전에 모델의 접근 범위와 함께 판단한다.

## 참고 자료 (References)

- Hugging Face TRL, *SFTTrainer*: <https://huggingface.co/docs/trl/sft_trainer>
- Hugging Face PEFT, *LoRA*: <https://huggingface.co/docs/peft/conceptual_guides/lora>
- Unsloth Fine-Tuning Guide: <https://unsloth.ai/docs/get-started/fine-tuning-guide>
- LLaMA-Factory: <https://github.com/hiyouga/LLaMA-Factory>
- Axolotl: <https://github.com/axolotl-ai-cloud/axolotl>
- MLX-LM: <https://github.com/ml-explore/mlx-lm>
- vLLM, *LoRA Adapters*: <https://docs.vllm.ai/en/latest/features/lora.html>
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021): <https://arxiv.org/abs/2106.09685>
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (2023): <https://arxiv.org/abs/2305.14314>

## 관련 문서

- [로컬 sLLM 파인튜닝 워크플로우](./local-sllm-finetuning-workflow.md) - 각 단계의 상세
- [데이터셋과 Chat Template 가이드](./dataset-and-chat-template-guide.md) - 데이터 형식과 template 불일치 방지
- [학습 및 배포 레시피](./training-and-deployment-recipe.md) - Unsloth 학습 코드와 GGUF export
- [적용 조건](./verified-conditions.md) - 버전·메모리·평가 조건
