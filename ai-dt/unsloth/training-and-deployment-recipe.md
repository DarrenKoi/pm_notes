---
type: learning
tags: [unsloth, finetuning]
reviewed_on: 2026-10-04
review_status: partial
---

# 학습 및 배포 레시피

> 처음에는 QLoRA 기반 SFT를 가장 작은 성공 단위로 잡는 편이 좋다.

> [!info] 검토 범위 · 2026-10-04
> 공식 문서와 코드 예제를 대조한 학습 자료다. GPU 학습·모델 다운로드·export·serving은 실행하지 않았다. [적용 조건](./verified-conditions.md)과 [정리 기록](./organization-log.md)에 버전 경계와 미확인을 남겼다.

## 1. 환경 준비

아래는 **Core Python 패키지**의 설치 진입점이다. Studio/Desktop 설치와 구분한다. 기존 환경의 torch 등 의존성이 바뀔 수 있으므로 별도 시험 환경에서 실제 해결된 버전을 기록한다.

```bash
pip install unsloth
```

실제 프로젝트에서는 보통 다음 패키지도 함께 사용한다.

```bash
pip install unsloth transformers datasets trl peft accelerate bitsandbytes
```

공식 설치 문서는 CUDA / PyTorch 버전 조합에 따라 권장 설치 경로가 달라질 수 있으므로, 새 환경을 만들 때는 설치 문서를 다시 확인하는 편이 안전하다.

## 2. 하드웨어 전제

Unsloth requirements 문서 기준으로 QLoRA 최소 VRAM 예시는 다음과 같다.

| 모델 크기 | 대략적인 최소 VRAM |
|------|------|
| 3B | 3.5GB+ |
| 7B | 5GB+ |
| 8B | 6GB+ |
| 14B | 8.5GB+ |

주의:

- 이 값은 시작점일 뿐이다
- 긴 context, 큰 batch, 더 많은 target modules를 쓰면 메모리가 더 든다
- 현재 requirements는 Studio의 Mac 학습/MLX 지원과 Core 부분의 진행 중 안내를 함께 담고 있다. 설치 방식별 적용 범위를 확인해야 하며 이 CUDA/bitsandbytes Core 예제가 Mac에서 실행된다는 근거로 사용하지 않는다

## 3. 첫 학습 설정 예시

첫 시도에서 추천하는 조합:

- student: 4B ~ 8B instruct
- 방식: QLoRA
- task: narrow SFT
- context: `2048`
- epoch: `1`

왜 이렇게 시작하나?

- 문제 원인이 데이터인지 설정인지 분리하기 쉽다
- 학습 시간이 짧아 반복 속도가 빠르다
- base model 대비 개선 여부를 빨리 판단할 수 있다

## 4. 기본 SFT 코드 예시

`train.jsonl`은 데이터셋 문서의 conversations 형식을 준비한 로컬 파일이다. 모델 ID는 원래 예시이며 다운로드/접근·라이선스·지원 여부를 확인해야 한다. Unsloth를 TRL/transformers보다 먼저 import한다. 아래 context·target_modules는 Llama 계열 예시다.

```python
from unsloth import FastLanguageModel
from typing import Any
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import get_chat_template

MODEL_NAME = "unsloth/llama-3.1-8b-unsloth-bnb-4bit"  # example placeholder
MAX_SEQ_LENGTH = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype="auto",
    load_in_4bit=True,
)

tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

dataset = load_dataset("json", data_files="train.jsonl", split="train")

def format_batch(batch: dict[str, Any]) -> dict[str, list[str]]:
    return {
        "text": [
            tokenizer.apply_chat_template(
                x,
                tokenize=False,
                add_generation_prompt=False,
            )
            for x in batch["conversations"]
        ]
    }

dataset = dataset.map(format_batch, batched=True)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="outputs",
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        report_to="none",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_steps=10,
        num_train_epochs=1,
        logging_steps=10,
        optim="adamw_8bit",
        lr_scheduler_type="linear",
        seed=3407,
    ),
)

trainer.train()

model.save_pretrained("outputs/adapter")
tokenizer.save_pretrained("outputs/adapter")
```

위 코드는 미실행 학습 예제다. TRL **v0.23.1** 문서의 `processing_class`/`max_length` API로 수정했다. Unsloth 전체 dependency 조합을 설치 검증한 버전 lock은 아니다. `FastLanguageModel`의 `max_seq_length`와 TRL 설정의 `max_length`는 다른 API 인자다. 공식 main 소스에서 `get_peft_model(max_seq_length=...)`는 미사용 인자로 표시되어 예제에서 제거했다. 길이는 모델 로딩과 TRL 설정에서 정한다. preformatted `text` 전체에 대한 SFT이며 eval_dataset·response-only masking·실제 inference 검증은 포함하지 않는다. 실제 모델 family에 맞춰 template와 model name은 조정해야 한다.

## 5. LoRA 하이퍼파라미터 시작점

Unsloth hyperparameter guide의 핵심 포인트를 실무적으로 요약하면 다음과 같다.

- `r`: 보통 `16` 또는 `32`부터 시작
- `lora_alpha`: 보통 `16` 또는 `32`
- `lora_dropout`: 처음엔 `0`도 자주 사용
- `target_modules`: 주요 linear layer를 넓게 포함하는 편이 일반적

초기 추천:

```python
r = 16
lora_alpha = 16
lora_dropout = 0
```

작은 태스크에서는 이 정도로도 충분한 경우가 많다.

## 6. 주요 target modules

아래 이름은 Llama 계열 예시다. 다른 architecture에서는 실제 `named_modules()`와 해당 모델 가이드로 확인한다. 이름이 없거나 fused projection을 사용하는 모델에 그대로 적용하지 않는다.

```python
[
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
```

공식 가이드도 주요 linear layers를 넓게 포함하는 쪽을 권장한다.

## 7. 학습 중 체크할 것

### loss만 보지 않는다

다음을 같이 본다.

- held-out 샘플 생성 결과
- JSON validity
- 과도한 verbosity 여부
- refusal behavior

### overfitting 신호

- train loss만 빠르게 내려가고 eval output이 오히려 나빠짐
- 답변이 training set phrasing을 과도하게 복제
- 작은 변화에도 schema가 깨짐

## 8. 평가 루프 추천

학습 후 최소한 다음 3단계는 확인한다.

1. base model과 동일 프롬프트 비교
2. 사람 수동 검수 30 ~ 100개
3. teacher judge 자동 채점

judge는 빠른 비교용으로 좋지만, 최종 판단은 사람이 해야 한다.

## 9. GGUF 및 배포

Unsloth는 GGUF 저장 경로를 제공한다. Ollama / llama.cpp로 연결할 계획이면 매우 유용하다.

```python
model.save_pretrained_gguf(
    "outputs/gguf",
    tokenizer,
    quantization_method="q4_k_m",
)
```

주의:

- 학습 시점의 chat template과 EOS 처리 방식을 serving 시점에도 유지해야 한다
- GGUF export 전에 base 대비 결과를 먼저 검증하는 편이 좋다

export는 runtime 호환과 품질 검증의 완료가 아니다. adapter는 정확한 base model/revision과 함께 로드하며 merged weights와 구분한다. GGUF 양자화 뒤에도 동일한 held-out prompt/template로 결과를 비교한다. 관련 build 도구와 llama.cpp 변환 경로는 설치 환경에 따라 달라진다.

## 10. 언제 full fine-tuning을 고려하나?

다음 조건이 동시에 맞을 때만 검토하는 편이 좋다.

- 데이터가 충분히 많다
- GPU 여유가 있다
- adapter만으로 원하는 behavior가 잘 안 나온다

QLoRA를 첫 비교 실험으로 사용할 수 있지만 충분성은 실제 baseline·평가 결과로 판단한다.

## 11. 추천 실전 루프

1. 작은 데이터셋으로 1 epoch 학습
2. 결과 검토
3. 데이터셋 수정
4. 다시 학습
5. 필요할 때만 하이퍼파라미터 조정
6. 결과가 확인되면 GGUF 또는 merged weights export

데이터·template·설정을 한 번에 모두 바꾸지 말고 원인별 실험을 기록한다. 데이터와 하이퍼파라미터 기여도의 우열을 확인한 로컬 근거는 없다.

## 참고 자료

- Unsloth Install Guide: <https://unsloth.ai/docs/get-started/install/pip-install>
- Unsloth Requirements: <https://docs.unsloth.ai/get-started/fine-tuning-for-beginners/unsloth-requirements>
- Unsloth LoRA Hyperparameters Guide: <https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide>
- Unsloth Saving to GGUF: <https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf>
- Hugging Face TRL Unsloth Integration: <https://huggingface.co/docs/trl/en/unsloth_integration>
