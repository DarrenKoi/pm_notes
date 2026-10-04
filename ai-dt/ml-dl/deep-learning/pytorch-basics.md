---
tags: [pytorch, tensor, dataset, dataloader]
level: beginner
last_updated: 2026-02-14
reviewed_on: 2026-10-04
review_status: partial
document_type: learning_note
---

# PyTorch 기초 가이드

> PyTorch의 핵심 빌딩 블록(Tensor, Autograd, Dataset, DataLoader, nn.Module)을 실전 코드와 함께 정리한다.

> [!info] 적용 범위와 검증일
> 2026-10-04 공식 PyTorch 2.14 문서와 설치 판본2.14.1/Python3.14.2의 CPU 실행을 대조했다. CUDA/MPS·멀티프로세스·compile/export/ONNX 실행은 미확인이다. 1~7절은 순서대로 실행하는 블록이며 8절은 독립 예제다. 합성 데이터 학습은 업무 성능의 근거가 아니다.

## 왜 필요한가? (Why)

- PyTorch는 Tensor 연산·자동 미분·신경망 모듈을 제공해 커스텀 모델의 학습과 추론을 구성할 수 있다. 시장 점유율이나 논문 대부분이라는 주장은 이 문서에서 검증하지 않는다.
- **Eager execution** 방식으로 디버깅이 직관적이고, Python 개발 경험과 자연스럽게 연결된다.
- `torch.compile`은 실행 최적화, `torch.export`는 제약을 기록한 그래프 추출, ONNX는 다른 런타임을 위한 변환 경로다. 지원 연산·입력 shape·backend를 따로 확인해야 한다. 공식 문서에서 TorchScript는 deprecated로 안내한다.
- AI/DT의 커스텀 모델·파인튜닝에 적용할 수 있으며 사내 표준 여부는 미확인이다.

## 핵심 개념 (What)

| 개념 | 설명 |
|------|------|
| **Tensor** | 다차원 배열. NumPy ndarray와 유사하지만 GPU 연산과 자동 미분을 지원 |
| **Autograd** | 텐서 연산 그래프를 자동 추적하여 역전파(Backpropagation) 시 기울기를 계산 |
| **Dataset** | 이 예제는 map-style 인터페이스 (`__len__`, `__getitem__`). IterableDataset은 `__iter__` 기반 |
| **DataLoader** | Dataset을 배치 단위로 묶고 셔플, 멀티프로세싱 로딩을 담당 |
| **nn.Module** | 모든 신경망 레이어/모델의 기본 클래스. `forward()` 메서드에 연산 정의 |
| **Optimizer** | 기울기 기반으로 파라미터를 업데이트하는 알고리즘 (Adam, SGD 등) |
| **Loss Function** | 예측과 정답의 목적함수를 계산. 기본 평균 reduction은 스칼라, `none`은 원소별 결과 |

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
y = x.view(2, 3)         # 원소 수와 stride가 새 shape와 호환될 때
z = x.reshape(3, 2)      # 호환되면 view, 아니면 copy; 원소 수 조건은 여전히 필요
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
back = t.numpy()                  # 이 CPU/grad 없는 tensor는 메모리를 공유
# 학습 tensor는 보통 t.detach().cpu().numpy(); 공유 배열 수정은 원본도 바꿀 수 있음
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
    print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── 텐서를 GPU로 이동 ──
x = torch.randn(3, 3)
x_gpu = x.to(device)          # 지정된 디바이스로 이동
# 또는
if torch.cuda.is_available():
    x_gpu = x.cuda()           # CUDA 가용성을 확인한 뒤 호출

# ── 모델을 GPU로 이동 ──
# model = MyModel()
# model = model.to(device)      # 모델의 모든 파라미터를 GPU로

# ── 주의: CPU-GPU 텐서 혼합 연산은 에러 ──
# a_cpu = torch.randn(3)
# b_gpu = torch.randn(3).to("cuda")
# a_cpu + b_gpu  # RuntimeError! → 같은 디바이스에 있어야 함
```

이 선택식은 CUDA가 없으면 CPU를 사용하며 MPS를 자동 선택하지 않는다. CUDA 메모리 속성 접근과 `.cuda()`는 CUDA 장치가 실제로 있을 때만 실행한다.

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
y = (x * x).sum()           # 실수 스칼라는 gradient 생략 가능; 벡터 출력에는 gradient 인자를 제공
y.backward()
print(x.grad)               # 2 * x

# ── 기울기 추적 중단 (추론 시) ──
with torch.no_grad():
    # 이 블록 안의 연산은 기울기를 추적하지 않음
    pred = x * 2
    print(pred.requires_grad)  # False

# ── 기울기 초기화 (학습 루프에서 중요) ──
# optimizer.zero_grad()  # 일반 학습은 step마다 초기화; 의도적 accumulation은 여러 batch 뒤 초기화
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
    num_workers=0,       # 블록 실습은 메인 프로세스; >0은 별도 프로세스 설정 필요
    drop_last=True,      # 마지막 불완전 배치 버리기
)

for batch_x, batch_y in dataloader:
    print(f"배치 shape: x={batch_x.shape}, y={batch_y.shape}")
    break  # 첫 배치만 확인


# ── Custom collate_fn 예제 (같은 shape를 묶는 현재 데이터) ──
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

Windows/macOS의 spawn worker를 쓸 때 Dataset·collate 함수를 파일 최상위에 정의하고 DataLoader 생성/반복을 `if __name__ == "__main__":` 안에 둔다. worker 수는 속도를 보장하지 않는다. 위 collate는 `stack`이므로 가변 길이는 padding/mask 또는 별도 batch 구조가 필요하다. `drop_last=True`는 여기서500개 중 마지막20개를 제외한다.

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

# 분류 문제 → CrossEntropyLoss: unnormalized logits 입력, class-index이면 log_softmax + NLL과 동등
criterion_cls = nn.CrossEntropyLoss()
logits = torch.tensor([[2.0, 1.0, 0.1]])   # 클래스 3개에 대한 로짓
label = torch.tensor([0])                    # 정답: 클래스 0
loss_ce = criterion_cls(logits, label)
print(f"CrossEntropy Loss: {loss_ce.item():.4f}")

# ── 옵티마이저 ──
model = nn.Linear(10, 1)

# Adam: 적응적 학습률 후보; 항상 최선이라는 뜻은 아님
optimizer_adam = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# SGD + momentum: 대규모 모델 학습 시 일반화 성능이 좋을 수 있음
optimizer_sgd = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

# AdamW: weight decay를 gradient 업데이트와 분리; Adam의 기본 L2 방식과 구분
optimizer_adamw = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
```

CrossEntropy의 이 예제는 `(N, C)` logits와 `[0,C)` 범위의 int64 class-index를 사용한다. logits에 먼저 softmax를 적용하지 않는다. 회귀는 예측/타겟 shape를 맞춰 의도하지 않은 broadcasting을 피한다. `model.eval()`은 Dropout/BatchNorm 등 모드만 바꾸며 grad 추적을 끄지 않아 `no_grad()`와 함께 사용한다.

### 8. 간단한 학습 예제 (End-to-End)

합성 데이터로 2-layer 네트워크를 학습하는 전체 예제:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# seed는 같은 환경의 난수를 제어하며 판본/CPU/GPU 간 동일 결과를 보장하지 않음
torch.manual_seed(42)

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
loaded_model.load_state_dict(torch.load("model_weights.pt", map_location=device, weights_only=True))
loaded_model.eval()
```

state_dict는 모델 구조를 함께 저장하지 않으므로 같은 TwoLayerNet 정의를 먼저 만든다. 이 예제는 현재 작업 폴더에 `model_weights.pt`를 쓴다. PyTorch2.6부터 pickle_module 미지정 시 weights_only=True가 기본이지만 여기서는 명시한다. weights_only도 신뢰할 수 없는 파일의 모든 위험을 제거하지 않으므로 자신의 체크포인트만 사용한다. seed/장치에 따라 아래 수치가 달라질 수 있다.

**원래 문서의 미검증 출력 예시(2026-02-14; 현재 실행 결과가 아님):**
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

2026-10-04 확인: 아래 계약은 공식2.14 문서 기준이며 실행 판본2.14.1과 구분한다.

- [reshape/view 계약](https://docs.pytorch.org/docs/2.14/generated/torch.reshape.html), [view stride 조건](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.view.html)
- [Tensor.numpy 공유/grad 조건](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.numpy.html)
- [DataLoader·spawn·map/iterable](https://docs.pytorch.org/docs/2.14/data.html)
- [CrossEntropyLoss logits/target](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)
- [torch.load·weights_only](https://docs.pytorch.org/docs/2.14/generated/torch.load.html), [2.6 기본값 변경·제한](https://docs.pytorch.org/docs/2.14/notes/serialization.html)
- [backward 벡터 gradient](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.backward.html)
- [AdamW decoupled decay](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html), [Adam 기본 decay](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html)
- [Module eval/load_state_dict](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html)
- [재현성의 환경 제약](https://docs.pytorch.org/docs/2.14/notes/randomness.html)
- [CUDA 속성 total_memory 바인딩(공식 v2.14.1 구현)](https://github.com/pytorch/pytorch/blob/v2.14.1/torch/csrc/cuda/Module.cpp)
- [compile 실행 최적화](https://docs.pytorch.org/docs/stable/generated/torch.compile.html), [export 그래프·입력 제약](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export/api_reference.html)
- [TorchScript deprecated 안내](https://docs.pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html)
- [PyTorch 공식 튜토리얼](https://pytorch.org/tutorials/)
- [PyTorch 공식 문서 - Tensor](https://pytorch.org/docs/stable/tensors.html)
- [PyTorch 공식 문서 - Autograd](https://pytorch.org/docs/stable/autograd.html)
- [PyTorch 공식 문서 - Data Loading](https://pytorch.org/docs/stable/data.html)
- [PyTorch 공식 문서 - nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)

## 관련 문서

- [학습 루프 템플릿](./training-loop-template.md)
- [ML/DL 전체 목차](../README.md)
