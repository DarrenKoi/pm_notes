---
tags: [cnn, image-classification, torchvision, pretrained]
level: intermediate
last_updated: 2026-02-14
reviewed_on: 2026-10-04
review_status: partial
document_type: learning_note
category_major: "AI·DT"
category_middle: "머신러닝·딥러닝"
category_minor: "딥러닝 학습"
note_kind: "학습"
classified_on: "2026-10-05"
---

# CNN 이미지 분류 (Image Classification)

> CNN(Convolutional Neural Network)을 활용한 이미지 분류의 기초부터 사전학습 모델 전이학습까지 실무 패턴 정리

> [!info] 실행 범위와 검증일
> 2026-10-04 공식 PyTorch2.14/torchvision0.29 문서를 대조했다. 로컬 설치는2.14.1/0.29.1이다. CIFAR 흐름은1→3→6→7절, 2절과4절은 별도 데이터/모델 선택 예제다. CIFAR·가중치 다운로드/원래50epoch·GPU/멀티프로세스·실업무 성능·읽기 화면은 미확인이다. 조각을 파일로 실행할 때 worker>0이면 생성/반복을 main guard 안에 둔다.

## 왜 필요한가? (Why)

- 이미지 분류(Image Classification)는 딥러닝의 **"Hello World"** 에 해당하는 가장 기본적인 태스크다
- CNN의 핵심 구성 요소(합성곱, 풀링, 배치 정규화)를 이해하면 Object Detection, Segmentation 등 상위 태스크로 자연스럽게 확장할 수 있다
- 사전학습 모델은 제한된 라벨로 특징을 재사용하는 선택지다. 입력 도메인·사전학습 데이터·비용에 따라 scratch와 fine-tuning을 비교한다
- 반도체 공정에서의 결함 분류, 웨이퍼 맵 패턴 인식 등에도 입력 형식·분할 단위·라벨 보존 조건을 정의해 적용할 수 있다

## 핵심 개념 (What)

### CNN 아키텍처 구성 요소

| 레이어 | PyTorch 클래스 | 역할 |
|--------|---------------|------|
| 합성곱(Convolution) | `nn.Conv2d` | 이미지에서 지역적 특징(엣지, 텍스처 등)을 추출 |
| 풀링(Pooling) | `nn.MaxPool2d` | 공간 해상도를 줄이면서 주요 특징을 유지 |
| 배치 정규화(Batch Norm) | `nn.BatchNorm2d` | 배치 통계/추적 통계를 사용해 정규화. 안정화·속도 향상은 조건부 |
| 활성화 함수 | `nn.ReLU` | 비선형성 부여 |
| 완전 연결층(FC) | `nn.Linear` | 최종 분류 수행 |

### 일반적인 CNN 블록 흐름

```
Input Image
  → Conv2d → BatchNorm2d → ReLU → MaxPool2d   (특징 추출 블록 반복)
  → Flatten
  → Linear → ReLU → Dropout → Linear          (분류기)
  → Output (class logits)
```

### torchvision transforms 주요 구성

- **전처리**: `Resize`, `ToTensor`, `Normalize` -- 모델의 크기·dtype·채널·학습 정규화 계약에 맞춰 선택
- **데이터 증강**: `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter` -- 학습 데이터에만 적용
- `transforms.Compose`로 체이닝하여 사용

## 어떻게 사용하는가? (How)

### 1. 데이터 준비 -- torchvision 내장 데이터셋 (CIFAR-10)

```python
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# --- 전처리 파이프라인 정의 ---
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616]),
])

test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616]),
])

# --- 데이터셋 & 데이터로더 ---
torch.manual_seed(42)
train_source = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=train_transform
)
val_source = datasets.CIFAR10(root="./data", train=True, download=False, transform=test_transform)
indices = torch.randperm(len(train_source), generator=torch.Generator().manual_seed(42)).tolist()
train_dataset = Subset(train_source, indices[:40000])
val_dataset = Subset(val_source, indices[40000:])
test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=test_transform
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

print(f"학습 데이터: {len(train_dataset)}장, 테스트 데이터: {len(test_dataset)}장")
print(f"검증 데이터: {len(val_dataset)}장, 클래스: {train_source.classes}")
```

공식 train50,000개에서40,000/10,000을 분리하며 val은 별도 dataset 객체의 증강 없는 transform을 쓴다. 같은 transform 객체를 공유한 Subset의 transform을 바꾸지 않는다. 공식 test10,000개는6절의 선택 완료 뒤 한 번 평가한다. 분할은 독립 표본 학습용이며 장비/lot/시간/중복 이미지 그룹에는 별도 분할 단위가 필요하다. 위 Normalize 수치의 산출 표본은 원문에서 확인하지 못했으므로 교육용 상수로 보존했다. 실데이터 통계는 train에서만 산출한다.

### 2. 커스텀 이미지 데이터셋

#### 방법 A: `ImageFolder` -- 폴더 구조만 맞추면 끝

```
data/
├── train/
│   ├── cat/        # 클래스명 = 폴더명
│   │   ├── 001.jpg
│   │   └── 002.jpg
│   └── dog/
│       ├── 001.jpg
│       └── 002.jpg
└── val/
    ├── cat/
    └── dog/
```

```python
from torchvision.datasets import ImageFolder

folder_train_dataset = ImageFolder(root="data/train", transform=train_transform)
folder_val_dataset = ImageFolder(root="data/val", transform=test_transform)
if folder_train_dataset.class_to_idx != folder_val_dataset.class_to_idx:
    raise ValueError("train/val 클래스 이름과 index 계약이 다릅니다")

# class_to_idx 확인
print(folder_train_dataset.class_to_idx)  # {'cat': 0, 'dog': 1}
```

#### 방법 B: 커스텀 Dataset -- CSV/JSON 라벨 등 유연한 구조

```python
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    """CSV 파일로 라벨을 관리하는 이미지 데이터셋.

    CSV 형식:
        filename,label
        img_001.jpg,0
        img_002.jpg,1
    """
    def __init__(self, csv_path: str, img_dir: str, transform=None, num_classes: int = 10):
        self.df = pd.read_csv(csv_path, dtype={"filename": "string"})
        if not {"filename", "label"} <= set(self.df.columns):
            raise ValueError("filename/label 열이 필요합니다")
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 1:
            raise ValueError("num_classes는 양의 정수여야 합니다")
        labels = pd.to_numeric(self.df["label"], errors="raise")
        if labels.isna().any() or not np.isfinite(labels).all() or ((labels % 1) != 0).any() or ((labels < 0) | (labels >= num_classes)).any():
            raise ValueError("label은 클래스 범위 안의 finite 정수여야 합니다")
        if self.df["filename"].isna().any() or self.df["filename"].str.strip().eq("").any():
            raise ValueError("filename이 비어 있습니다")
        self.df["label"] = labels.astype("int64")
        self.img_dir = Path(img_dir).resolve()
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = (self.img_dir / row["filename"]).resolve()
        if not img_path.is_relative_to(self.img_dir):
            raise ValueError("이미지 경로가 img_dir 밖을 가리킵니다")
        with Image.open(img_path) as source:
            image = source.convert("RGB")
        label = int(row["label"])

        if self.transform:
            image = self.transform(image)

        return image, label
```

ImageFolder는 class 폴더를 정렬해 index를 만들므로 val 폴더의 클래스 누락/추가도 검사한다. 실제 이미지가 없는 빈 class 폴더는 기본 설정에서 로딩 실패다. 두 대안은 새 DataLoader와 num_classes/클래스 목록을 함께 구성해야 하며 위 CIFAR loader를 자동으로 교체하지 않는다. CSV 예제는 num_classes=10이 기본이고 다른 작업의 클래스 수를 명시한다. transform=None이면 PIL 이미지를 반환해 기본 tensor collate 학습에 바로 쓸 수 없다.

### 3. 간단한 CNN 구현 (from scratch)

```python
import torch.nn as nn

class SimpleCNN(nn.Module):
    """Conv → BN → ReLU → Pool 블록을 반복하는 기본 CNN.

    CIFAR-10 (32x32x3) 기준 설계. 입력 크기가 다르면 fc1의 입력 차원 조정 필요.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()

        # --- 특징 추출부 ---
        self.features = nn.Sequential(
            # Block 1: 3 → 32 채널
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x32 → 16x16

            # Block 2: 32 → 64 채널
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x16 → 8x8

            # Block 3: 64 → 128 채널
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8x8 → 4x4
        )

        # --- 분류기 ---
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def set_training_mode(model):
    model.train()
    if getattr(model, "_frozen_head", None) is not None:
        model.eval()  # 백본 BN/Dropout 상태도 고정
        getattr(model, model._frozen_head).train()  # 새 head만 학습 모드


# --- 모델 생성 및 확인 ---
model = SimpleCNN(num_classes=10)
print(model)

# 파라미터 수 확인
total_params = sum(p.numel() for p in model.parameters())
print(f"총 파라미터 수: {total_params:,}")
```

### 4. 사전학습 모델 사용 (Transfer Learning)

ImageNet 특징을 재사용할 때 분류기 교체는 출발점이다. 도메인 차이·라벨 수·검증 성능에 따라 백본의 일부/전체를 학습한다. 모델과 해당 weights.transforms()의 평가 전처리를 함께 선택한다. 32×32/CIFAR 정규화와 ImageNet224×224 정규화를 무조건 혼용하지 않는다.

#### ResNet18

```python
from torchvision import models

def create_resnet18(num_classes: int, freeze_backbone: bool = True, weights=models.ResNet18_Weights.IMAGENET1K_V1):
    """사전학습된 ResNet18의 마지막 FC 레이어를 교체하여 반환."""
    model = models.resnet18(weights=weights)

    # 백본 동결 (선택)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # 마지막 FC 레이어 교체
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    model._frozen_head = "fc" if freeze_backbone else None
    return model


resnet_eval_transform = models.ResNet18_Weights.IMAGENET1K_V1.transforms()
model = create_resnet18(num_classes=10, freeze_backbone=True)
```

#### EfficientNet-B0

```python
def create_efficientnet(num_classes: int, freeze_backbone: bool = True, weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1):
    """사전학습된 EfficientNet-B0의 classifier를 교체하여 반환."""
    model = models.efficientnet_b0(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    model._frozen_head = "classifier" if freeze_backbone else None
    return model


efficientnet_eval_transform = models.EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()
model = create_efficientnet(num_classes=10, freeze_backbone=True)
```

weights=None은 구조만 만들고 사전학습을 하지 않는다. weights enum이 미캐시이면 다운로드한다. requires_grad=False는 가중치 gradient만 끄므로 백본 BatchNorm running statistics와 Dropout을 고정하려면 위 set_training_mode를 적용한다. 새 head의 Dropout은 train 모드를 유지한다. 6절은 scratch SimpleCNN이고 전이학습을 자동 실행하지 않는다. 백본을 풀면 _frozen_head=None으로 바꾸고 optimizer가 새 trainable parameter를 포함하는지도 확인한다.

> **Tip**: 데이터가 적으면 `freeze_backbone=True`로 시작하고, 성능이 부족하면 일부 레이어를 풀어서 미세조정한다. 자세한 전략은 [Transfer Learning 가이드](./transfer-learning.md) 참고.

### 5. 데이터 증강 (Data Augmentation) 상세

데이터 증강은 학습 데이터의 다양성을 인위적으로 늘려 라벨을 보존할 때 과적합을 줄이는 데 도움이 될 수 있다. 방향/색/질감이 라벨 의미인 경우 flip·rotation·blur가 라벨을 훼손할 수 있어 검증한다.

```python
# 상황별 증강 레시피

# 기본 후보 (방향 변환이 라벨을 보존할 때)
basic_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 강한 증강 (데이터가 적거나 과적합이 심할 때)
strong_augmentation = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 사전학습 모델용 (ImageNet 정규화 + 224x224)
pretrained_augmentation = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),          # 평가 시
    # transforms.RandomResizedCrop(224), # 학습 시에는 이것으로 교체
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

5절은 독립 증강 후보이며 basic은 원래 크기를 유지한다. SimpleCNN 입력은32×32, weights 기반 평가 transform은224×224이므로 크기를 맞춰 loader를 새로 구성한다. 증강 강도가 높다고 성능이 자동으로 좋아지지 않는다.

### 6. 학습 & 평가

```python
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# --- 설정 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

NUM_EPOCHS = 50


def train_one_epoch(model, loader, criterion, optimizer, device):
    """1 에폭 학습 수행. 평균 loss와 accuracy를 반환."""
    set_training_mode(model)
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        if criterion.reduction != "mean" or criterion.weight is not None or torch.any((labels < 0) | (labels >= outputs.size(1))):
            raise ValueError("이 집계는 비가중 mean/유효한 class-index만 지원합니다")
        loss = criterion(outputs, labels)
        if not torch.isfinite(loss).item():
            raise ValueError("손실이 finite가 아닙니다")
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    if total == 0:
        raise ValueError("DataLoader가 비어 있습니다")
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """평가 수행. 평균 loss와 accuracy를 반환."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        if criterion.reduction != "mean" or criterion.weight is not None or torch.any((labels < 0) | (labels >= outputs.size(1))):
            raise ValueError("이 집계는 비가중 mean/유효한 class-index만 지원합니다")
        loss = criterion(outputs, labels)
        if not torch.isfinite(loss).item():
            raise ValueError("손실이 finite가 아닙니다")

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    if total == 0:
        raise ValueError("DataLoader가 비어 있습니다")
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# --- 학습 루프 ---
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val_acc = float("-inf")

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(
        f"[Epoch {epoch:3d}/{NUM_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    )

    # 최고 성능 모델 저장
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f"  ★ Best model saved (Val Acc: {val_acc:.4f})")

# 선택을 고정한 뒤 best 복원 → 최종 test 한 번 (반복 선택에 재사용하지 않음)
model.load_state_dict(torch.load("best_model.pth", map_location=device, weights_only=True))
test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"Final Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
```

#### Accuracy / Loss 커브 시각화

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss 커브
axes[0].plot(history["train_loss"], label="Train Loss")
axes[0].plot(history["val_loss"], label="Val Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Loss Curve")
axes[0].legend()
axes[0].grid(True)

# Accuracy 커브
axes[1].plot(history["train_acc"], label="Train Acc")
axes[1].plot(history["val_acc"], label="Val Acc")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Accuracy Curve")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()
```

### 7. 추론 코드 -- 저장된 모델로 단일 이미지 예측

```python
from PIL import Image
from torchvision import transforms

# --- 추론 파이프라인 ---
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

inference_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616]),
])


def predict_single_image(model_path: str, image_path: str, device: str = "cpu"):
    """저장된 모델 가중치를 로드하여 단일 이미지를 분류한다.

    Returns:
        dict: {"class": 예측 클래스명, "confidence": 확률, "all_probs": 전체 확률}
    """
    # 모델 로드
    model = SimpleCNN(num_classes=len(CLASSES))
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # 이미지 전처리
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    input_tensor = inference_transform(image).unsqueeze(0).to(device)  # (1, 3, 32, 32)

    # 추론
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze()

    top_prob, top_idx = probs.max(0)
    return {
        "class": CLASSES[top_idx.item()],
        "confidence": top_prob.item(),
        "all_probs": {cls: p.item() for cls, p in zip(CLASSES, probs)},
    }


# 사용 예시
result = predict_single_image("best_model.pth", "test_image.jpg")
print(f"예측: {result['class']} (확률: {result['confidence']:.2%})")
```

#### Top-K 예측과 함께 이미지 시각화

```python
def predict_and_visualize(model_path: str, image_path: str, top_k: int = 5):
    """예측 결과를 이미지와 함께 시각화."""
    result = predict_single_image(model_path, image_path)
    probs = result["all_probs"]

    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= len(CLASSES):
        raise ValueError("top_k는 클래스 수 범위의 양의 정수여야 합니다")
    # Top-K 정렬
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:top_k]
    classes_topk = [c for c, _ in sorted_probs]
    values_topk = [v for _, v in sorted_probs]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # 원본 이미지
    with Image.open(image_path) as source:
        img = source.convert("RGB")
    axes[0].imshow(img)
    axes[0].set_title(f"Predicted: {result['class']} ({result['confidence']:.1%})")
    axes[0].axis("off")

    # 확률 바 차트
    axes[1].barh(classes_topk[::-1], values_topk[::-1])
    axes[1].set_xlabel("Probability")
    axes[1].set_title("Top-K Predictions")
    axes[1].set_xlim(0, 1)

    plt.tight_layout()
    plt.show()
```

7절은 저장한 SimpleCNN·CIFAR10 클래스 순서·32×32 전처리 계약 전용이다. ImageFolder/CSV/ResNet/EfficientNet 가중치를 그대로 읽지 않는다. confidence는 softmax 점수이며 정답일 보정 확률/불확실성 보장이 아니다. 새 분포·장비/이미지 품질·클래스 빈도에서 별도 평가한다. 입력 파일과 자신이 만든 checkpoint를 준비한 뒤 실행한다.

## 참고 자료 (References)

2026-10-04 확인: 공식 PyTorch2.14/torchvision0.29, 설치2.14.1/0.29.1을 구분한다.

- [ResNet18 weights·평가 전처리](https://docs.pytorch.org/vision/0.29/models/generated/torchvision.models.resnet18.html)
- [EfficientNet-B0 weights·평가 전처리](https://docs.pytorch.org/vision/0.29/models/generated/torchvision.models.efficientnet_b0.html)
- [ImageFolder class/empty 정책](https://docs.pytorch.org/vision/0.29/generated/torchvision.datasets.ImageFolder.html)
- [BatchNorm running statistics/train/eval](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm2d.html)
- [CIFAR 원 저자 train/test 계약](https://cave.cs.toronto.edu/kriz/cifar.html)
- [CrossEntropy 집계 조건](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)
- [DataLoader worker 조건](https://docs.pytorch.org/docs/2.14/data.html)
- [torchvision transforms dtype/범위](https://docs.pytorch.org/vision/stable/transforms.html)
- [Pillow Image 파일 수명](https://pillow.readthedocs.io/en/stable/reference/open_files.html)
- [PyTorch 공식 튜토리얼 - Training a Classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [torchvision.models 공식 문서](https://pytorch.org/vision/stable/models.html)
- [torchvision.transforms 공식 문서](https://pytorch.org/vision/stable/transforms.html)
- [2019 초판/2020 v7 문헌: A Survey of the Recent Architectures of Deep Convolutional Neural Networks](https://arxiv.org/abs/1901.06032)

## 관련 문서

- [Transfer Learning 가이드](./transfer-learning.md) -- 사전학습 모델 전이학습 전략 상세
- [Training Loop Template](./training-loop-template.md) -- 재사용 가능한 학습 루프 템플릿
