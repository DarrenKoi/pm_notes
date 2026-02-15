---
tags: [cnn, image-classification, torchvision, pretrained]
level: intermediate
last_updated: 2026-02-14
status: in-progress
---

# CNN 이미지 분류 (Image Classification)

> CNN(Convolutional Neural Network)을 활용한 이미지 분류의 기초부터 사전학습 모델 전이학습까지 실무 패턴 정리

## 왜 필요한가? (Why)

- 이미지 분류(Image Classification)는 딥러닝의 **"Hello World"** 에 해당하는 가장 기본적인 태스크다
- CNN의 핵심 구성 요소(합성곱, 풀링, 배치 정규화)를 이해하면 Object Detection, Segmentation 등 상위 태스크로 자연스럽게 확장할 수 있다
- 실무에서는 대부분 **사전학습 모델(Pretrained Model)** 을 미세조정(Fine-tuning)하여 사용하므로, 직접 CNN을 쌓는 것과 사전학습 모델을 활용하는 것 모두 알아야 한다
- 반도체 공정에서의 결함 분류, 웨이퍼 맵 패턴 인식 등에도 동일한 파이프라인이 적용된다

## 핵심 개념 (What)

### CNN 아키텍처 구성 요소

| 레이어 | PyTorch 클래스 | 역할 |
|--------|---------------|------|
| 합성곱(Convolution) | `nn.Conv2d` | 이미지에서 지역적 특징(엣지, 텍스처 등)을 추출 |
| 풀링(Pooling) | `nn.MaxPool2d` | 공간 해상도를 줄이면서 주요 특징을 유지 |
| 배치 정규화(Batch Norm) | `nn.BatchNorm2d` | 학습 안정화, 수렴 속도 향상 |
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

- **전처리**: `Resize`, `ToTensor`, `Normalize` -- 모든 파이프라인에 필수
- **데이터 증강**: `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter` -- 학습 데이터에만 적용
- `transforms.Compose`로 체이닝하여 사용

## 어떻게 사용하는가? (How)

### 1. 데이터 준비 -- torchvision 내장 데이터셋 (CIFAR-10)

```python
import torch
from torch.utils.data import DataLoader
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
train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=train_transform
)
test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=test_transform
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

print(f"학습 데이터: {len(train_dataset)}장, 테스트 데이터: {len(test_dataset)}장")
print(f"클래스: {train_dataset.classes}")
```

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

train_dataset = ImageFolder(root="data/train", transform=train_transform)
val_dataset = ImageFolder(root="data/val", transform=test_transform)

# class_to_idx 확인
print(train_dataset.class_to_idx)  # {'cat': 0, 'dog': 1}
```

#### 방법 B: 커스텀 Dataset -- CSV/JSON 라벨 등 유연한 구조

```python
import os
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
    def __init__(self, csv_path: str, img_dir: str, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")
        label = int(row["label"])

        if self.transform:
            image = self.transform(image)

        return image, label
```

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


# --- 모델 생성 및 확인 ---
model = SimpleCNN(num_classes=10)
print(model)

# 파라미터 수 확인
total_params = sum(p.numel() for p in model.parameters())
print(f"총 파라미터 수: {total_params:,}")
```

### 4. 사전학습 모델 사용 (Transfer Learning)

실무에서는 처음부터 CNN을 학습시키는 경우가 드물다. ImageNet으로 사전학습된 모델의 특징 추출부를 재활용하고 **분류기(최종 레이어)만 교체** 하는 것이 일반적이다.

#### ResNet18

```python
from torchvision import models

def create_resnet18(num_classes: int, freeze_backbone: bool = True):
    """사전학습된 ResNet18의 마지막 FC 레이어를 교체하여 반환."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

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
    return model


model = create_resnet18(num_classes=10, freeze_backbone=True)
```

#### EfficientNet-B0

```python
def create_efficientnet(num_classes: int, freeze_backbone: bool = True):
    """사전학습된 EfficientNet-B0의 classifier를 교체하여 반환."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


model = create_efficientnet(num_classes=10, freeze_backbone=True)
```

> **Tip**: 데이터가 적으면 `freeze_backbone=True`로 시작하고, 성능이 부족하면 일부 레이어를 풀어서 미세조정한다. 자세한 전략은 [Transfer Learning 가이드](./transfer-learning.md) 참고.

### 5. 데이터 증강 (Data Augmentation) 상세

데이터 증강은 학습 데이터의 다양성을 인위적으로 늘려 과적합(Overfitting)을 방지하는 핵심 기법이다.

```python
# 상황별 증강 레시피

# 기본 증강 (거의 항상 사용)
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
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

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
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# --- 학습 루프 ---
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val_acc = 0.0

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )
    val_loss, val_acc = evaluate(model, test_loader, criterion, device)
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
    image = Image.open(image_path).convert("RGB")
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

    # Top-K 정렬
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:top_k]
    classes_topk = [c for c, _ in sorted_probs]
    values_topk = [v for _, v in sorted_probs]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # 원본 이미지
    img = Image.open(image_path)
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

## 참고 자료 (References)

- [PyTorch 공식 튜토리얼 - Training a Classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [torchvision.models 공식 문서](https://pytorch.org/vision/stable/models.html)
- [torchvision.transforms 공식 문서](https://pytorch.org/vision/stable/transforms.html)
- [A Survey of CNN Architectures (EfficientNet, ResNet 등)](https://arxiv.org/abs/1901.06032)

## 관련 문서

- [Transfer Learning 가이드](./transfer-learning.md) -- 사전학습 모델 전이학습 전략 상세
- [Training Loop Template](./training-loop-template.md) -- 재사용 가능한 학습 루프 템플릿
