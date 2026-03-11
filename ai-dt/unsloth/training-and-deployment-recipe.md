# 학습 및 배포 레시피

> 처음에는 QLoRA 기반 SFT를 가장 작은 성공 단위로 잡는 편이 좋다.

## 1. 환경 준비

Unsloth의 기본 설치 진입점은 매우 단순하다.

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
- Apple Silicon / MLX 지원은 공식 문서 기준 아직 진행 중으로 안내된다

## 3. 가장 안전한 첫 학습 설정

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

```python
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel
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
    max_seq_length=MAX_SEQ_LENGTH,
)

dataset = load_dataset("json", data_files="train.jsonl", split="train")

def format_batch(batch):
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
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="outputs",
        max_seq_length=MAX_SEQ_LENGTH,
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

위 코드는 "처음 성공하는 최소 파이프라인"에 초점을 둔 예시다. 실제 모델 family에 맞춰 template와 model name은 조정해야 한다.

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

대부분의 decoder-only 모델에서는 다음 계열이 기본 출발점이다.

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

## 10. 언제 full fine-tuning을 고려하나?

다음 조건이 동시에 맞을 때만 검토하는 편이 좋다.

- 데이터가 충분히 많다
- GPU 여유가 있다
- adapter만으로 원하는 behavior가 잘 안 나온다

대부분의 첫 프로젝트는 QLoRA로 시작해도 충분하다.

## 11. 추천 실전 루프

1. 작은 데이터셋으로 1 epoch 학습
2. 결과 검토
3. 데이터셋 수정
4. 다시 학습
5. 필요할 때만 하이퍼파라미터 조정
6. 결과가 확인되면 GGUF 또는 merged weights export

좋은 결과는 보통 "더 어려운 하이퍼파라미터"보다 "더 깨끗한 데이터와 일관된 template"에서 나온다.

## 참고 자료

- Unsloth Install Guide: <https://docs.unsloth.ai/get-started/installing-%2B-updating>
- Unsloth Requirements: <https://docs.unsloth.ai/get-started/fine-tuning-for-beginners/unsloth-requirements>
- Unsloth LoRA Hyperparameters Guide: <https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide>
- Unsloth Saving to GGUF: <https://docs.unsloth.ai/basics/running-and-saving-models/saving-to-gguf>
- Hugging Face TRL Unsloth Integration: <https://huggingface.co/docs/trl/en/unsloth_integration>
