---
tags: [transfer-learning, fine-tuning, pretrained, pytorch]
level: intermediate
last_updated: 2026-02-14
status: in-progress
---

# 전이학습(Transfer Learning)

> 사전 학습된 모델의 지식을 새로운 태스크에 재활용하여, 적은 데이터로도 높은 성능을 달성하는 기법

## 왜 필요한가? (Why)

- **데이터 부족 문제 해결**: 딥러닝 모델을 처음부터(from scratch) 학습하려면 수십만~수백만 장의 데이터가 필요하다. 실무에서는 수백~수천 장만 확보 가능한 경우가 대부분이다.
- **학습 시간 및 비용 절감**: ImageNet으로 학습된 ResNet은 수백 GPU-hours가 소요된다. 전이학습을 사용하면 단일 GPU에서 수 시간 이내로 유사한 성능을 얻을 수 있다.
- **범용적 특징(feature) 재사용**: 사전 학습 모델의 앞쪽 레이어는 엣지, 텍스처 등 범용적 특징을 학습한다. 이 특징은 도메인이 달라도 유효하다.
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
| 필요 데이터량 | 적음 (수백~수천) | 상대적으로 많음 (수천~수만) |
| 학습 시간 | 빠름 | 상대적으로 느림 |
| 적합한 상황 | 타겟 도메인이 사전 학습 도메인과 유사 | 도메인 차이가 크거나 데이터가 충분 |

### 동결 레이어(Frozen Layers)

```
[Input] → [Conv1] → [Conv2] → ... → [ConvN] → [FC] → [Output]
          ←────── frozen (학습 안 함) ──────→   ←학습→

- 앞쪽 레이어: 범용적 저수준 특징 (엣지, 색상, 텍스처)
- 뒤쪽 레이어: 태스크에 특화된 고수준 특징 (객체 부분, 의미 패턴)
- FC (Fully Connected): 최종 분류 레이어 → 반드시 교체 필요
```

### 학습률(Learning Rate) 전략

- **단일 학습률**: 모든 파라미터에 동일한 LR 적용 (단순하지만 최적은 아님)
- **차등 학습률(Discriminative LR)**: 사전 학습 레이어에는 작은 LR, 새 레이어에는 큰 LR
- **단계적 언프리징(Gradual Unfreezing)**: 에폭마다 뒤쪽 레이어부터 순차적으로 해동

## 어떻게 사용하는가? (How)

### 1. 피처 추출(Feature Extraction)

사전 학습된 ResNet의 모든 레이어를 동결하고, 마지막 FC 레이어만 교체하여 학습한다.

```python
import torch
import torch.nn as nn
from torchvision import models

# 사전 학습된 ResNet18 로드
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 모든 파라미터 동결
for param in model.parameters():
    param.requires_grad = False

# 마지막 FC 레이어 교체 (클래스 수에 맞게)
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)
# 새로 추가한 FC는 requires_grad=True가 기본값

# 학습 가능한 파라미터만 옵티마이저에 전달
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

print(f"전체 파라미터: {sum(p.numel() for p in model.parameters()):,}")
print(f"학습 파라미터: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
# 전체 파라미터: 11,181,642
# 학습 파라미터: 5,130  (FC 레이어만)
```

### 2. 파인튜닝(Fine-tuning)

마지막 N개 레이어를 해동하여, 새 분류 헤드와 함께 학습한다.

```python
import torch
import torch.nn as nn
from torchvision import models

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 모든 파라미터 동결
for param in model.parameters():
    param.requires_grad = False

# 마지막 FC 교체
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

# layer4 (마지막 residual block)만 해동
for param in model.layer4.parameters():
    param.requires_grad = True

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

에폭이 진행될수록 뒤쪽 레이어부터 순차적으로 해동한다. 사전 학습 가중치를 안전하게 보존하면서 점진적으로 적응시킬 수 있다.

```python
import torch
import torch.nn as nn
from torchvision import models

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

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

    # ... 학습 코드 (train_one_epoch, validate 등)
```

### 4. 학습률 차등 적용 (param_groups)

PyTorch `optimizer`의 `param_groups`를 활용하여 레이어별로 다른 학습률을 적용한다.

```python
import torch
import torch.nn as nn
from torchvision import models

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
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

실행 가능한 전체 코드이다. CIFAR-10을 커스텀 데이터셋의 프록시(proxy)로 사용한다.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

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

val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

# ── 데이터 로드 ──────────────────────────────────
train_dataset = datasets.CIFAR10(root="./data", train=True,  download=True, transform=train_transform)
val_dataset   = datasets.CIFAR10(root="./data", train=False, download=True, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ── 모델 구성 ────────────────────────────────────
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
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
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

# ── 검증 함수 ────────────────────────────────────
@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

# ── 학습 루프 ────────────────────────────────────
best_val_acc = 0.0

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
```

**예상 결과**: CIFAR-10에서 ResNet18 파인튜닝 시 약 **93~95%** validation accuracy 달성 가능 (from scratch 대비 +5~8% 향상, 수렴 속도 3~5배 빠름).

### 6. 텍스트 모델 전이학습 참고

HuggingFace `transformers` 라이브러리를 사용한 텍스트 분류 전이학습 패턴이다.

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer

# 사전 학습 모델 로드 (예: BERT 계열)
model_name = "bert-base-multilingual-cased"
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
    learning_rate=2e-5,           # BERT 파인튜닝 시 권장: 2e-5 ~ 5e-5
    weight_decay=0.01,
    warmup_ratio=0.1,             # 전체 스텝의 10%를 warmup으로
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Trainer로 학습
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,   # HuggingFace Dataset 객체
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

> **사내 환경 참고**: 회사 내부에서는 외부 LLM API(OpenAI, Anthropic 등)가 차단되어 있으므로, 추론(inference) 시에는 내부 LLM API(Kimi-K2.5 등)를 OpenAI-compatible 클라이언트로 호출한다. 파인튜닝 자체는 로컬 GPU 또는 사내 GPU 서버에서 수행한다.

### 7. 전이학습 전략 선택 가이드

데이터 크기와 타겟 도메인의 유사도에 따라 전략이 달라진다.

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

**의사결정 흐름**:

1. 타겟 데이터가 **1,000장 미만**이고 도메인이 유사 → **피처 추출**로 시작
2. 피처 추출 성능이 부족 → **layer4만 해동**하여 파인튜닝
3. 데이터가 **5,000장 이상** → **차등 학습률로 전체 파인튜닝**
4. 도메인이 매우 다름 (예: 자연 이미지 → 반도체 웨이퍼) → **단계적 언프리징** + 강한 데이터 증강

## 참고 자료 (References)

- [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) - 공식 튜토리얼
- [CS231n: Transfer Learning](https://cs231n.github.io/transfer-learning/) - Stanford 강의 노트
- [HuggingFace Fine-tuning Guide](https://huggingface.co/docs/transformers/training) - 텍스트 모델 파인튜닝
- [ULMFiT Paper (Howard & Ruder, 2018)](https://arxiv.org/abs/1801.06146) - 단계적 언프리징 원논문
- [torchvision.models](https://pytorch.org/vision/stable/models.html) - 사용 가능한 사전 학습 모델 목록

## 관련 문서

- [딥러닝 기초](./deep-learning-basics.md)
- [CNN 아키텍처](./cnn-architectures.md)
- [데이터 증강 기법](./data-augmentation.md)
- [학습률 스케줄링](./learning-rate-scheduling.md)
