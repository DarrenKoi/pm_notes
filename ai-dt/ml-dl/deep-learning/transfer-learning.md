---
tags: [transfer-learning, fine-tuning, pretrained, pytorch]
level: intermediate
last_updated: 2026-02-14
reviewed_on: 2026-10-04
review_status: partial
document_type: learning_note
---

# 전이학습(Transfer Learning)

> 사전학습 가중치를 초기값 또는 고정 피처 추출기로 재사용하는 기법. 적은 데이터에서 유리할 수 있지만 도메인 차이·라벨 품질·분할에 따라 성능이 나빠지는 negative transfer도 가능하다.

> [!info] 읽는 목적과 검증 범위
> CNN 문서는 입력/증강/분류 흐름, 이 문서는 동결 범위·학습률·해동의 선택을 설명한다. 1절 모드 함수를 2·3절에서 재사용하며 5절은 별도 스크립트다. PyTorch2.14/torchvision0.29, 텍스트 조각은 Transformers5.17.0 공식 계약을2026-10-04 확인했다. 실제 CPU 검증에서는 사전학습 파일/CIFAR/BERT를 다운로드하지 않아 전이 효과나 정확도를 입증하지 않는다. CUDA·업무 데이터·Claude 협의·Obsidian 읽기 화면은 미확인이다.

## 왜 필요한가? (Why)

- **데이터 부족**: 사전학습 피처를 재사용해 처음부터 학습하는 비용을 줄일 수 있다. 필요한 표본 수는 모델·태스크·평가 설계에 따라 달라진다.
- **학습 비용**: 동결 범위가 넓으면 gradient 계산/optimizer 상태가 줄지만 전체 forward와 입력 전처리는 여전히 필요하다. GPU 수·시간·성능 향상 배수는 측정해야 한다.
- **피처 재사용 조건**: 앞쪽 층의 저수준 피처도 입력 채널·영상 특성·도메인 차이에 영향을 받는다. 독립 validation으로 피처 추출/부분·전체 미세조정과 scratch 기준을 비교한다.
- **실무 적용 사례**:
  - 제조 결함 검출 (수백 장의 불량 이미지로 분류기 구축)
  - 의료 이미지 분석 (소량의 라벨링된 X-ray로 질병 감지)
  - 사내 문서 분류 (소규모 라벨 데이터로 텍스트 분류)

## 핵심 개념 (What)

### 피처 추출(Feature Extraction) vs 파인튜닝(Fine-tuning)

| 구분 | 피처 추출 | 파인튜닝 |
|------|-----------|----------|
| 사전 학습 가중치 | 전부 동결(freeze) | 일부 또는 전부 학습 |
| 학습 대상 | 새로 추가한 분류 헤드만 | 분류 헤드 + 사전 학습 레이어 일부 |
| 필요 데이터량 | 검증으로 판단; 고정 임계값 없음 | 자유도가 늘어 과적합 관리 필요 |
| 학습 시간 | backward 범위가 작음; 시간은 측정 | backward 범위가 큼; 시간은 측정 |
| 적합한 상황 | 타겟 도메인이 사전 학습 도메인과 유사 | 도메인 차이가 크거나 데이터가 충분 |

### 동결 레이어(Frozen Layers)

```
[Input] → [Conv1] → [Conv2] → ... → [ConvN] → [FC] → [Output]
          ←────── frozen (학습 안 함) ──────→   ←학습→

- 앞쪽 레이어: 범용적 저수준 특징 (엣지, 색상, 텍스처)
- 뒤쪽 레이어: 태스크에 특화된 고수준 특징 (객체 부분, 의미 패턴)
- FC (Fully Connected): 목표 클래스 수/의미가 다르면 새 head로 교체
```

### 학습률(Learning Rate) 전략

- **단일 학습률**: 모든 파라미터에 동일한 LR 적용 (단순한 baseline; 최적인지는 검증)
- **차등 학습률(Discriminative LR)**: 사전 학습 레이어에는 작은 LR, 새 레이어에는 큰 LR
- **단계적 언프리징(Gradual Unfreezing)**: 정한 단계에서 뒤쪽 레이어부터 순차적으로 해동

## 어떻게 사용하는가? (How)

### 1. 피처 추출(Feature Extraction)

사전 학습된 ResNet의 모든 레이어를 동결하고, 마지막 FC 레이어만 교체하여 학습한다.

```python
import torch
import torch.nn as nn
from torchvision import models

# 사전 학습된 ResNet18 로드
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# 모든 파라미터 동결
for param in model.parameters():
    param.requires_grad = False

# 마지막 FC 레이어 교체 (클래스 수에 맞게)
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)
# 새로 추가한 FC는 requires_grad=True가 기본값

def set_resnet_training_mode(model):
    """ResNet의 완전히 동결된 child는 BN 통계도 고정한다."""
    model.train()
    for child in model.children():
        if not any(p.requires_grad for p in child.parameters()):
            child.eval()

# 매 학습 epoch 시작에 다시 호출; validate의 model.eval() 뒤에도 필요
set_resnet_training_mode(model)

# 학습 가능한 파라미터만 옵티마이저에 전달
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

print(f"전체 파라미터: {sum(p.numel() for p in model.parameters()):,}")
print(f"학습 파라미터: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
# 전체 파라미터: 11,181,642
# 학습 파라미터: 5,130  (FC 레이어만)
```

동결은 gradient 갱신을 막는 것이며 `model.train()`의 BatchNorm running statistics 갱신을 자동으로 막지 않는다. 위 함수는 ResNet의 완전히 동결된 child에 eval을 적용한다. layer4처럼 일부 child를 전부 해동하면 그 child의 BN도 학습 모드다. 임의의 섞인 동결 구조에 보편적으로 적용하는 함수는 아니다. weights의 권장 평가 전처리는 `ResNet18_Weights.IMAGENET1K_V1.transforms()`로 얻는다.

### 2. 파인튜닝(Fine-tuning)

ResNet의 layer4 residual stage를 해동하여, 새 분류 헤드와 함께 학습한다.

```python
import torch
import torch.nn as nn
from torchvision import models

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# 모든 파라미터 동결
for param in model.parameters():
    param.requires_grad = False

# 마지막 FC 교체
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

# layer4 (마지막 residual block)만 해동
for param in model.layer4.parameters():
    param.requires_grad = True

# 1절 함수 필요: frozen child는 eval, layer4/FC는 train
set_resnet_training_mode(model)

# 학습 가능한 파라미터 확인
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"학습 파라미터: {trainable:,}")  # layer4 + FC

# 차등 학습률 적용
optimizer = torch.optim.Adam([
    {"params": model.layer4.parameters(), "lr": 1e-4},   # 사전 학습 레이어 → 작은 LR
    {"params": model.fc.parameters(),     "lr": 1e-3},   # 새 레이어 → 큰 LR
])
```

### 3. 단계적 언프리징(Gradual Unfreezing)

에폭이 진행될수록 뒤쪽 레이어부터 순차적으로 해동한다. 점진적으로 적응시키는 선택지이며 원래 가중치/성능 보존을 보장하지 않는다. 아래 조각은 trigger epoch를0부터 빠짐없이 호출하는 흐름이며 임의 epoch 재개를 구현하지 않는다. optimizer를 재생성하므로 Adam moment 상태를 초기화하고0epoch부터 LR도1e-4로 바뀐다. 상태 유지가 필요하면 별도 add_param_group 설계와 scheduler 상태 검증이 필요하다.

```python
import torch
import torch.nn as nn
from torchvision import models

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# 전체 동결 후 FC 교체
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 10)

# 언프리징 대상 레이어 목록 (뒤쪽부터)
unfreeze_schedule = [
    (0, [model.fc]),                          # 에폭 0: FC만
    (3, [model.fc, model.layer4]),            # 에폭 3: + layer4
    (6, [model.fc, model.layer4, model.layer3]),  # 에폭 6: + layer3
]

def apply_unfreeze(model, epoch, schedule):
    """에폭에 따라 레이어를 순차적으로 해동한다."""
    for trigger_epoch, layers in schedule:
        if epoch == trigger_epoch:
            # 먼저 모든 파라미터 동결
            for param in model.parameters():
                param.requires_grad = False
            # 지정된 레이어만 해동
            for layer in layers:
                for param in layer.parameters():
                    param.requires_grad = True

            # 옵티마이저 재구성 (학습 가능한 파라미터만)
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(trainable_params, lr=1e-4)
            count = sum(p.numel() for p in trainable_params)
            print(f"[Epoch {epoch}] 학습 파라미터: {count:,}")
            return optimizer
    return None

# 학습 루프에서 사용
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

num_epochs = 10
for epoch in range(num_epochs):
    new_optimizer = apply_unfreeze(model, epoch, unfreeze_schedule)
    if new_optimizer is not None:
        optimizer = new_optimizer

    # 1절 모드 함수 필요: validate 뒤에도 epoch마다 재적용
    set_resnet_training_mode(model)
    # ... 실제 train_one_epoch/validate 연결은 이 조각에 포함하지 않음
```

### 4. 학습률 차등 적용 (param_groups)

PyTorch `optimizer`의 `param_groups`를 활용하여 레이어별로 다른 학습률을 적용한다.

```python
import torch
import torch.nn as nn
from torchvision import models

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 10)

# 파라미터 그룹 정의
# 그룹 1: 초기 레이어 (conv1, bn1, layer1, layer2) → 매우 작은 LR
# 그룹 2: 후반 레이어 (layer3, layer4) → 작은 LR
# 그룹 3: 새 FC 레이어 → 큰 LR

param_groups = [
    {
        "params": list(model.conv1.parameters())
                + list(model.bn1.parameters())
                + list(model.layer1.parameters())
                + list(model.layer2.parameters()),
        "lr": 1e-5,
        "name": "early_layers",
    },
    {
        "params": list(model.layer3.parameters())
                + list(model.layer4.parameters()),
        "lr": 1e-4,
        "name": "late_layers",
    },
    {
        "params": model.fc.parameters(),
        "lr": 1e-3,
        "name": "classifier",
    },
]

optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-2)

# 각 그룹의 LR 확인
for i, group in enumerate(optimizer.param_groups):
    n_params = sum(p.numel() for p in group["params"])
    print(f"Group {i} ({group.get('name', 'N/A')}): lr={group['lr']}, params={n_params:,}")
```

### 5. 완전한 파인튜닝 예제: ResNet18 + CIFAR-10

독립 교육용 스크립트다. weights/CIFAR 다운로드와 메모리·학습 시간이 필요하다. CIFAR-10은 기법 흐름의 예시이며 업무 데이터의 성능 proxy라고 단정하지 않는다. 아래는 전체 백본 학습이므로 model.train()을 사용한다. 피처 추출로 바꾸면1절 모드/optimizer를 함께 적용해야 한다. Resize224는 이 예제의 ImageNet 가중치 평가 조건이며 ResNet 자체가224입력만 받는다는 뜻은 아니다. 수평 반전/회전은 목표 라벨을 보존하는지 확인한다. 기본worker0; worker>0은 main guard/spawn과 운영 환경을 별도로 검증한다.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

torch.manual_seed(42)

# ── 설정 ──────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_EPOCHS = 10
NUM_CLASSES = 10
LR_PRETRAINED = 1e-4
LR_CLASSIFIER = 1e-3

# ── 데이터 전처리 ────────────────────────────────
# ImageNet 정규화 값 사용 (사전 학습 모델 입력 분포에 맞춤)
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize(224),              # ResNet 입력 크기
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

val_transform = models.ResNet18_Weights.IMAGENET1K_V1.transforms()

# ── 데이터 로드 ──────────────────────────────────
train_source = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
val_source = datasets.CIFAR10(root="./data", train=True, download=True, transform=val_transform)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=val_transform)
indices = torch.randperm(len(train_source), generator=torch.Generator().manual_seed(42))
split = int(0.8 * len(indices))
train_dataset = Subset(train_source, indices[:split].tolist())
val_dataset = Subset(val_source, indices[split:].tolist())
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── 모델 구성 ────────────────────────────────────
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# ── 차등 학습률 + 옵티마이저 ─────────────────────
pretrained_params = []
classifier_params = []
for name, param in model.named_parameters():
    if "fc" in name:
        classifier_params.append(param)
    else:
        pretrained_params.append(param)

optimizer = optim.AdamW([
    {"params": pretrained_params, "lr": LR_PRETRAINED},
    {"params": classifier_params, "lr": LR_CLASSIFIER},
], weight_decay=1e-2)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
criterion = nn.CrossEntropyLoss()

# ── 학습 함수 ────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    if not isinstance(criterion, nn.CrossEntropyLoss) or criterion.reduction != "mean" or criterion.weight is not None:
        raise ValueError("비가중 mean CrossEntropyLoss만 지원")
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        if (labels == criterion.ignore_index).any():
            raise ValueError("ignore 라벨은 이 표본 평균 예제에서 지원하지 않음")
        loss = criterion(outputs, labels)
        if not torch.isfinite(loss):
            raise ValueError("non-finite loss")
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    if total == 0:
        raise ValueError("빈 loader")
    return running_loss / total, correct / total

# ── 검증 함수 ────────────────────────────────────
@torch.no_grad()
def validate(model, loader, criterion, device):
    if not isinstance(criterion, nn.CrossEntropyLoss) or criterion.reduction != "mean" or criterion.weight is not None:
        raise ValueError("비가중 mean CrossEntropyLoss만 지원")
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        if (labels == criterion.ignore_index).any():
            raise ValueError("ignore 라벨은 이 표본 평균 예제에서 지원하지 않음")
        loss = criterion(outputs, labels)
        if not torch.isfinite(loss):
            raise ValueError("non-finite loss")

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    if total == 0:
        raise ValueError("빈 loader")
    return running_loss / total, correct / total

# ── 학습 루프 ────────────────────────────────────
best_val_acc = float("-inf")

for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
    scheduler.step()

    print(
        f"[Epoch {epoch+1:02d}/{NUM_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    )

    # 베스트 모델 저장
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_resnet18_cifar10.pth")
        print(f"  → Best model saved (Val Acc: {val_acc:.4f})")

print(f"\n최종 Best Validation Accuracy: {best_val_acc:.4f}")

# validation으로 선택한 가중치만 official test에 한 번 평가
model.load_state_dict(torch.load("best_resnet18_cifar10.pth", map_location=DEVICE, weights_only=True))
test_loss, test_acc = validate(model, test_loader, criterion, DEVICE)
print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
```

> [!warning] 원래 수치 주장: 미확인
> 원문의 validation93~95%·scratch 대비+5~8%·수렴3~5배, 수십만~수백만 장/수백 GPU-hours/단일GPU 수 시간은 실행 설정·분할·측정 출처가 없어 확인할 수 없다. 현재 예제의 결과나 업무 예상치로 사용하지 않는다.

### 6. 텍스트 모델 전이학습 참고

Hugging Face Transformers5.17.0 공식 API를 기준으로 한 입력 준비 후 학습 조각이다. 설치/모델 파일이 필요하며 standalone 전체 스크립트는 아니다. `tokenized_train`/`tokenized_val`은 서로 분리된 텍스트 Dataset이고 tokenizer로 만든 input_ids/attention_mask와 정수 labels0~4를 포함해야 한다. 위의 CIFAR train_dataset을 재사용하지 않는다. text→tokenizer(truncation=True)→collator(padding)→BERT+새 분류 head→validation loss 선택 순서다. 토큰 길이/라벨 사전·최종 test는 별도로 고정한다. 이 CPU 검증에서는 실제 BERT 다운로드/학습을 하지 않았다.

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding

# 사전 학습 모델 로드 (예: BERT 계열)
model_name = "google-bert/bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=5,  # 분류 클래스 수
)

# TrainingArguments로 학습률, 에폭 등 설정
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,           # 시작값 예시; validation으로 선택
    weight_decay=0.01,
    warmup_steps=0.1,             # v5.17.0: 0~1 미만 float는 전체 step 비율
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    push_to_hub=False,
)

# Trainer로 학습
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train, # 위에서 별도로 준비한 텍스트 Dataset
    eval_dataset=tokenized_val,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
)
trainer.train()
```

> [!note] 원래 사내 메모: 현재 정책/모델 미확인
> **사내 환경 참고**: 회사 내부에서는 외부 LLM API(OpenAI, Anthropic 등)가 차단되어 있으므로, 추론(inference) 시에는 내부 LLM API(Kimi-K2.5 등)를 OpenAI-compatible 클라이언트로 호출한다. 파인튜닝 자체는 로컬 GPU 또는 사내 GPU 서버에서 수행한다는 원래 메모다. 2026-10-04 정책/설비는 확인하지 않았다. BERT 분류 모델의 직접 forward와 생성형 OpenAI-compatible endpoint는 서로 다른 인터페이스이며 이 학습 코드를 해당 endpoint에 그대로 연결할 수 없다.

### 7. 전이학습 전략 선택 가이드

데이터 크기·도메인 유사도·라벨 품질·평가 단위로 선택한다. 아래 도식은 가설을 만드는 휴리스틱이며 성능 보장이나 고정 표본 임계값이 아니다. 증강 강도를 높이는 것 자체가 해법은 아니며 label 보존을 확인한다.

```
                    타겟 데이터셋 크기
                    작음                    큼
                ┌─────────────────┬─────────────────┐
    유사도 높음  │  피처 추출       │  전체 파인튜닝    │
    (ImageNet   │  (FC만 학습)     │  (차등 LR 적용)   │
     과 비슷)   │  과적합 위험 낮음  │  최고 성능 가능    │
                ├─────────────────┼─────────────────┤
    유사도 낮음  │  피처 추출       │  단계적 언프리징   │
    (의료, 위성  │  + 데이터 증강    │  또는 전체 파인튜닝 │
     등 특수)   │  어려움, 더 많은  │  주의: 앞쪽 레이어  │
                │  데이터 확보 필요  │  도 학습 필요      │
                └─────────────────┴─────────────────┘
```

**의사결정 흐름** (원문의1,000/5,000장 기준은 출처 없는 휴리스틱):

1. 타겟 데이터가 적고 도메인이 유사 → **피처 추출**로 시작
2. 피처 추출 성능이 부족 → **layer4만 해동**하여 파인튜닝
3. 독립 validation에서 충분한 근거가 있으면 → **차등 학습률로 전체 파인튜닝**
4. 도메인이 매우 다름 (예: 자연 이미지 → 반도체 웨이퍼) → 부분/전체 미세조정·scratch 기준을 비교하고 라벨을 보존하는 증강을 검증

## 참고 자료 (References)

2026-10-04 공식/일차 자료 대조. API 문서 판본은 PyTorch2.14·torchvision0.29·Transformers5.17.0이며 학습 성능 재현 근거와 구분한다.

- [PyTorch 전이학습 튜토리얼](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html): 초기값/고정 추출기와 best 가중치 복원.
- [PyTorch2.14 Autograd](https://docs.pytorch.org/docs/2.14/notes/autograd.html): requires_grad와 eval/BN 통계의 독립성.
- [PyTorch2.14 Optimizer](https://docs.pytorch.org/docs/2.14/optim.html): param_groups·add_param_group·state_dict. 재생성은 기존 moment를 자동 보존하지 않는다.
- [torchvision0.29 ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html): IMAGENET1K_V1과 평가 transforms.
- [CIFAR 원 저자 자료](https://www.cs.toronto.edu/~kriz/cifar.html): train50,000/test10,000; 예제 validation은 train에서 별도 분할한다.
- [CS231n 전이학습](https://cs231n.github.io/transfer-learning/): 데이터/도메인 차이에 따른 전략; 작은 LR도 검증 대상.
- [Transformers5.17.0 Trainer](https://huggingface.co/docs/transformers/v5.17.0/en/main_classes/trainer): eval_strategy·processing_class·warmup_steps 비율·best 선택 조건. 예전 evaluation_strategy/tokenizer/warmup_ratio 조각과 판본을 섞지 않는다.
- [다국어 BERT 모델 카드](https://huggingface.co/google-bert/bert-base-multilingual-cased): masked LM 사전학습 모델과 downstream fine-tuning 구분.
- [ULMFiT, Howard & Ruder2018](https://arxiv.org/abs/1801.06146): 텍스트 분류의 gradual unfreezing 제안; 이 ResNet schedule의 우월성을 입증하는 자료는 아니다.

## 관련 문서

- [PyTorch 기초](./pytorch-basics.md): tensor/gradient/data/model 계약.
- [CNN 이미지 분류](./cnn-image-classification.md): 입력·증강·단일 이미지 추론.
- [학습 루프](./training-loop-template.md): scheduler·early stopping·저장/재개 계약.
