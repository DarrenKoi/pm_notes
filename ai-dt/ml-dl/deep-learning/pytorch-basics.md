---
tags: [pytorch, tensor, dataset, dataloader]
level: beginner
last_updated: 2026-02-14
status: in-progress
---

# PyTorch 기초 가이드

> PyTorch의 핵심 빌딩 블록(Tensor, Autograd, Dataset, DataLoader, nn.Module)을 실전 코드와 함께 정리한다.

## 왜 필요한가? (Why)

- PyTorch는 연구와 프로덕션 모두에서 가장 널리 쓰이는 딥러닝 프레임워크다. 2023년 이후 대부분의 주요 논문과 오픈소스 모델(LLaMA, Stable Diffusion 등)이 PyTorch 기반으로 공개된다.
- **Eager execution** 방식으로 디버깅이 직관적이고, Python 개발 경험과 자연스럽게 연결된다.
- `torchscript`, `torch.compile`, `ONNX` 변환 등을 통해 프로덕션 배포까지 하나의 프레임워크로 커버 가능하다.
- 사내 AI/DT 시스템에서 커스텀 모델을 학습하거나 파인튜닝할 때 PyTorch가 기본 도구가 된다.

## 핵심 개념 (What)

| 개념 | 설명 |
|------|------|
| **Tensor** | 다차원 배열. NumPy ndarray와 유사하지만 GPU 연산과 자동 미분을 지원 |
| **Autograd** | 텐서 연산 그래프를 자동 추적하여 역전파(Backpropagation) 시 기울기를 계산 |
| **Dataset** | 데이터 한 건을 반환하는 인터페이스 (`__len__`, `__getitem__`) |
| **DataLoader** | Dataset을 배치 단위로 묶고 셔플, 멀티프로세싱 로딩을 담당 |
| **nn.Module** | 모든 신경망 레이어/모델의 기본 클래스. `forward()` 메서드에 연산 정의 |
| **Optimizer** | 기울기 기반으로 파라미터를 업데이트하는 알고리즘 (Adam, SGD 등) |
| **Loss Function** | 예측값과 정답 사이의 차이를 스칼라 값으로 계산 |

## 어떻게 사용하는가? (How)

### 1. 텐서 기본 연산

```python
import torch
import numpy as np

# ── 생성 ──
a = torch.tensor([1, 2, 3])                     # 리스트에서 생성
b = torch.zeros(3, 4)                            # 3×4 영행렬
c = torch.ones(2, 3, dtype=torch.float32)        # dtype 지정
d = torch.randn(2, 3)                            # 정규분포 난수
e = torch.arange(0, 10, 2)                       # [0, 2, 4, 6, 8]
f = torch.linspace(0, 1, steps=5)                # 균등 분할

# ── 인덱싱 ──
x = torch.randn(4, 5)
print(x[0])          # 첫 번째 행
print(x[:, 1])       # 두 번째 열
print(x[1:3, 2:4])   # 슬라이싱

# ── Reshape ──
x = torch.randn(6)
y = x.view(2, 3)         # 메모리 연속일 때
z = x.reshape(3, 2)      # 항상 안전
w = x.unsqueeze(0)       # (1, 6) — 차원 추가
v = w.squeeze(0)          # (6,)  — 차원 제거

# ── 연산 ──
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
print(a + b)              # element-wise 덧셈
print(a @ b)              # dot product (스칼라)
print(torch.matmul(
    a.unsqueeze(0),       # (1, 3)
    b.unsqueeze(1)        # (3, 1)
))                        # 행렬 곱 → (1, 1)

# ── dtype 변환 ──
x = torch.tensor([1, 2, 3])
x_float = x.float()      # int64 → float32
x_half = x.half()        # → float16

# ── NumPy 상호 변환 ──
np_arr = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(np_arr)     # NumPy → Tensor (메모리 공유)
back = t.numpy()                  # Tensor → NumPy (CPU 텐서만 가능)
```

### 2. GPU 관리 (Device Management)

```python
import torch

# ── 디바이스 설정 ──
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# CUDA 정보 확인
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# ── 텐서를 GPU로 이동 ──
x = torch.randn(3, 3)
x_gpu = x.to(device)          # 지정된 디바이스로 이동
# 또는
x_gpu = x.cuda()               # 직접 CUDA로 (GPU 있을 때만)

# ── 모델을 GPU로 이동 ──
# model = MyModel()
# model = model.to(device)      # 모델의 모든 파라미터를 GPU로

# ── 주의: CPU-GPU 텐서 혼합 연산은 에러 ──
# a_cpu = torch.randn(3)
# b_gpu = torch.randn(3).to("cuda")
# a_cpu + b_gpu  # RuntimeError! → 같은 디바이스에 있어야 함
```

### 3. Autograd (자동 미분)

```python
import torch

# ── 기본 미분 ──
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x + 1    # y = x² + 2x + 1

y.backward()                # dy/dx 계산
print(x.grad)               # tensor(8.) → 2*3 + 2 = 8

# ── 벡터 입력에 대한 기울기 ──
x = torch.randn(3, requires_grad=True)
y = (x * x).sum()           # 스칼라로 만들어야 backward() 호출 가능
y.backward()
print(x.grad)               # 2 * x

# ── 기울기 추적 중단 (추론 시) ──
with torch.no_grad():
    # 이 블록 안의 연산은 기울기를 추적하지 않음
    pred = x * 2
    print(pred.requires_grad)  # False

# ── 기울기 초기화 (학습 루프에서 중요) ──
# optimizer.zero_grad()  # 매 스텝마다 호출해야 기울기 누적 방지
```

### 4. Custom Dataset 클래스

```python
import torch
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    """간단한 합성 데이터셋 예제: y = 2x + 1 + noise"""

    def __init__(self, num_samples: int = 1000):
        self.x = torch.randn(num_samples, 1)
        self.y = 2 * self.x + 1 + 0.1 * torch.randn(num_samples, 1)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


# 사용
dataset = SyntheticDataset(num_samples=500)
print(f"데이터 수: {len(dataset)}")
sample_x, sample_y = dataset[0]
print(f"x={sample_x.item():.3f}, y={sample_y.item():.3f}")
```

### 5. DataLoader

```python
from torch.utils.data import DataLoader


# ── 기본 사용 ──
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,        # 에포크마다 데이터 순서 섞기
    num_workers=2,       # 멀티프로세스 데이터 로딩 (0이면 메인 프로세스)
    drop_last=True,      # 마지막 불완전 배치 버리기
)

for batch_x, batch_y in dataloader:
    print(f"배치 shape: x={batch_x.shape}, y={batch_y.shape}")
    break  # 첫 배치만 확인


# ── Custom collate_fn 예제 (가변 길이 시퀀스 등) ──
def custom_collate(batch):
    """배치 내 샘플들을 원하는 형태로 묶는 함수"""
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)


dataloader_custom = DataLoader(
    dataset,
    batch_size=16,
    collate_fn=custom_collate,
)
```

### 6. nn.Module 기본

```python
import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    """입력 → 은닉층 → 출력의 2-layer 네트워크"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


# ── 모델 생성 및 확인 ──
model = SimpleNet(input_dim=1, hidden_dim=32, output_dim=1)
print(model)

# 파라미터 수 확인
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"전체 파라미터: {total_params:,}")
print(f"학습 가능 파라미터: {trainable_params:,}")

# 추론 예시
dummy_input = torch.randn(4, 1)  # 배치 4개
output = model(dummy_input)       # forward() 자동 호출
print(f"출력 shape: {output.shape}")  # (4, 1)
```

### 7. 손실 함수 & 옵티마이저

```python
import torch
import torch.nn as nn
import torch.optim as optim

# ── 손실 함수 ──

# 회귀 문제 → MSELoss
criterion_reg = nn.MSELoss()
pred = torch.tensor([2.5, 0.0, 2.1])
target = torch.tensor([3.0, -0.5, 2.0])
loss_mse = criterion_reg(pred, target)
print(f"MSE Loss: {loss_mse.item():.4f}")

# 분류 문제 → CrossEntropyLoss (내부적으로 softmax 포함)
criterion_cls = nn.CrossEntropyLoss()
logits = torch.tensor([[2.0, 1.0, 0.1]])   # 클래스 3개에 대한 로짓
label = torch.tensor([0])                    # 정답: 클래스 0
loss_ce = criterion_cls(logits, label)
print(f"CrossEntropy Loss: {loss_ce.item():.4f}")

# ── 옵티마이저 ──
model = nn.Linear(10, 1)

# Adam: 가장 범용적으로 많이 사용
optimizer_adam = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# SGD + momentum: 대규모 모델 학습 시 일반화 성능이 좋을 수 있음
optimizer_sgd = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

# AdamW: weight decay를 올바르게 적용 (Transformer 계열에서 표준)
optimizer_adamw = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
```

### 8. 간단한 학습 예제 (End-to-End)

합성 데이터로 2-layer 네트워크를 학습하는 전체 예제:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ── 1. 데이터셋 정의 ──
class SyntheticDataset(Dataset):
    """y = 2x + 1 + noise"""

    def __init__(self, n: int = 1000):
        self.x = torch.randn(n, 1)
        self.y = 2 * self.x + 1 + 0.1 * torch.randn(n, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ── 2. 모델 정의 ──
class TwoLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


# ── 3. 학습 설정 ──
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = SyntheticDataset(n=1000)
val_dataset = SyntheticDataset(n=200)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

model = TwoLayerNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ── 4. 학습 루프 ──
num_epochs = 20

for epoch in range(num_epochs):
    # --- Train ---
    model.train()
    train_loss = 0.0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()          # 기울기 초기화
        pred = model(batch_x)          # 순전파
        loss = criterion(pred, batch_y)  # 손실 계산
        loss.backward()                # 역전파
        optimizer.step()               # 파라미터 업데이트

        train_loss += loss.item() * batch_x.size(0)

    train_loss /= len(train_dataset)

    # --- Validation ---
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            val_loss += loss.item() * batch_x.size(0)

    val_loss /= len(val_dataset)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# ── 5. 학습 결과 확인 ──
model.eval()
with torch.no_grad():
    test_x = torch.tensor([[0.0], [1.0], [2.0]]).to(device)
    test_pred = model(test_x)
    print("\n예측 결과 (y = 2x + 1):")
    for x_val, y_pred in zip(test_x, test_pred):
        print(f"  x={x_val.item():.1f} → pred={y_pred.item():.3f} "
              f"(정답: {2*x_val.item()+1:.1f})")

# ── 6. 모델 저장 & 로드 ──
# 저장 (state_dict만 저장하는 것이 권장)
torch.save(model.state_dict(), "model_weights.pt")

# 로드
loaded_model = TwoLayerNet().to(device)
loaded_model.load_state_dict(torch.load("model_weights.pt", map_location=device))
loaded_model.eval()
```

**실행 결과 예시:**
```
Epoch [5/20]  Train Loss: 0.0312 | Val Loss: 0.0298
Epoch [10/20] Train Loss: 0.0104 | Val Loss: 0.0101
Epoch [15/20] Train Loss: 0.0101 | Val Loss: 0.0099
Epoch [20/20] Train Loss: 0.0100 | Val Loss: 0.0099

예측 결과 (y = 2x + 1):
  x=0.0 → pred=1.003 (정답: 1.0)
  x=1.0 → pred=2.998 (정답: 3.0)
  x=2.0 → pred=4.995 (정답: 5.0)
```

## 참고 자료 (References)

- [PyTorch 공식 튜토리얼](https://pytorch.org/tutorials/)
- [PyTorch 공식 문서 - Tensor](https://pytorch.org/docs/stable/tensors.html)
- [PyTorch 공식 문서 - Autograd](https://pytorch.org/docs/stable/autograd.html)
- [PyTorch 공식 문서 - Data Loading](https://pytorch.org/docs/stable/data.html)
- [PyTorch 공식 문서 - nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)

## 관련 문서

- [학습 루프 템플릿](./training-loop-template.md)
- [ML/DL 전체 목차](../README.md)
