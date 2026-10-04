---
tags: [pytorch, training-loop, early-stopping, checkpointing]
level: intermediate
last_updated: 2026-02-14
reviewed_on: 2026-10-04
review_status: partial
document_type: learning_note
---

# PyTorch 학습 루프 템플릿

> 단일 장치·비가중 다중 클래스 분류의 학습/검증·early stopping·체크포인트를 구성하는 교육용 템플릿.

> [!info] 적용 범위와 검증일
> 2026-10-04 공식 PyTorch2.14 문서와 설치2.14.1/Python3.14.2를 대조했다. 0~6절은 앞 import와 사용자가 준비한 model/train_loader/val_loader/criterion/optimizer에 의존하는 조각이다. 7절은 별도 스크립트다. 기본 criterion은 weight 없는 CrossEntropyLoss(reduction="mean"), 출력(N,C)/target(N,) int64이다. ignore label·가중 손실·segmentation·회귀 지표에는 집계 코드를 바꿔야 한다. CUDA/MPS·CIFAR 다운로드·worker·원래30epoch 규모의 실행은 미확인이다.

## 왜 필요한가? (Why)

- 학습 루프(Training Loop)는 **모든 딥러닝 프로젝트의 뼈대**다. 모델 정의보다 학습 루프의 품질이 실험 생산성을 좌우한다.
- 매번 처음부터 작성하면 Early Stopping, 체크포인팅, 로깅 등 필수 기능을 빠뜨리기 쉽다.
- 학습/검증의 모드·집계와 저장 상태를 명시하면 새 프로젝트에서 반복 구현을 줄일 수 있다.
- seed는 난수 원인을 줄이지만 장치·판본 간 결과를 보장하지 않는다. 데이터 순서·worker·알고리즘·라이브러리 판본도 함께 기록한다.

---

## 핵심 개념 (What)

| 개념 | 설명 |
|------|------|
| **Training Loop** | 배치 단위로 forward → loss → backward → optimizer step을 반복하는 핵심 루프 |
| **Validation Loop** | 매 에포크 종료 후 검증 데이터로 모델 성능을 평가 (gradient 계산 없음) |
| **Early Stopping** | 검증 손실이 일정 에포크(patience) 동안 개선되지 않으면 학습을 조기 종료 |
| **Checkpointing** | 추론용 best 가중치와 재개용 optimizer/scheduler/epoch 상태를 구분해 저장 |
| **Learning Rate Scheduler** | 에포크/스텝에 따라 학습률을 동적으로 조절 |
| **Seed Fixing** | `torch.manual_seed` 등으로 특정 환경의 난수 원인을 제어 |

---

## 어떻게 사용하는가? (How)

### 0. 공통 설정 및 재현성(Reproducibility) 시드 고정

데이터 생성·분할·모델 생성 전에 seed를 설정한다. `PYTHONHASHSEED`는 Python 시작 전에 환경 변수로 지정해야 현재 프로세스의 hash seed에 반영된다.

```python
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """현재 프로세스의 난수를 제어한다. 판본/장치 간 재현 보장은 아니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU
    # PYTHONHASHSEED는 실행 전에 환경에서 설정; 여기서 hash seed를 바꾸지 않음
    # 결정론적 동작 (약간의 성능 저하 가능)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

### 1. 기본 학습 루프 (Simple Train Loop)

가장 간단한 형태. 손실만 추적한다.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """1 에포크 학습을 수행하고 평균 손실을 반환한다."""
    model.train()
    running_loss = 0.0
    total = 0
    if getattr(criterion, "reduction", None) != "mean" or getattr(criterion, "weight", None) is not None:
        raise ValueError("이 집계는 비가중 mean 손실만 지원합니다")

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward
        outputs = model(inputs)
        if outputs.ndim != 2 or targets.ndim != 1 or targets.dtype != torch.int64:
            raise ValueError("분류 출력(N,C)/int64 target(N,)가 필요합니다")
        if torch.any((targets < 0) | (targets >= outputs.size(1))):
            raise ValueError("ignore/unknown label은 지원하지 않습니다")
        loss = criterion(outputs, targets)

        if not torch.isfinite(loss).item():
            raise ValueError("손실이 finite가 아닙니다")
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * targets.size(0)
        total += targets.size(0)

    if total == 0:
        raise ValueError("학습 DataLoader가 비어 있습니다")
    avg_loss = running_loss / total
    return avg_loss


# --- 사용 예시 ---
# model = MyModel().to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#
# for epoch in range(num_epochs):
#     loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
#     print(f"Epoch {epoch+1} | Train Loss: {loss:.4f}")
```

### 2. 검증 루프 추가 (Train + Validation)

매 에포크마다 검증 데이터로 모델을 평가한다. `torch.no_grad()`로 gradient 계산을 비활성화한다.

```python
@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """검증 데이터로 평균 손실과 정확도를 계산한다."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    if getattr(criterion, "reduction", None) != "mean" or getattr(criterion, "weight", None) is not None:
        raise ValueError("이 집계는 비가중 mean 손실만 지원합니다")

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        if outputs.ndim != 2 or targets.ndim != 1 or targets.dtype != torch.int64:
            raise ValueError("분류 출력(N,C)/int64 target(N,)가 필요합니다")
        if torch.any((targets < 0) | (targets >= outputs.size(1))):
            raise ValueError("ignore/unknown label은 이 집계에서 지원하지 않습니다")
        if not torch.isfinite(loss).item():
            raise ValueError("손실이 finite가 아닙니다")
        running_loss += loss.item() * targets.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    if total == 0:
        raise ValueError("검증 DataLoader가 비어 있습니다")
    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# --- Train + Validation 루프 ---
num_epochs = 50
history = {"train_loss": [], "val_loss": [], "val_acc": []}

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.2f}%"
    )
```

### 3. Early Stopping 구현

검증 손실이 `patience` 에포크 동안 `delta`보다 크게 감소하지 않으면 학습을 조기 종료한다.

```python
class EarlyStopping:
    """검증 손실 기반 Early Stopping.

    Args:
        patience: 개선 없이 허용하는 에포크 수
        delta: 개선으로 인정하는 최소 변화량
        path: 최적 모델 저장 경로
        verbose: 로그 출력 여부
    """

    def __init__(
        self,
        patience: int = 7,
        delta: float = 0.0,
        path: str = "best_model.pt",
        verbose: bool = True,
    ):
        if isinstance(patience, bool) or not isinstance(patience, int) or patience < 1:
            raise ValueError("patience는 양의 정수여야 합니다")
        if not np.isfinite(delta) or delta < 0:
            raise ValueError("delta는 finite/nonnegative여야 합니다")
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose

        self.counter = 0
        self.best_score: float | None = None
        self.early_stop = False
        self.val_loss_min = float("inf")

    def __call__(self, val_loss: float, model: nn.Module):
        if not np.isfinite(val_loss):
            raise ValueError("검증 손실이 finite가 아닙니다")
        score = -val_loss  # 손실이 작을수록 좋으므로 부호 반전

        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save_checkpoint(val_loss, model)
            self.counter = 0

    def _save_checkpoint(self, val_loss: float, model: nn.Module):
        if self.verbose:
            print(
                f"  Val loss decreased ({self.val_loss_min:.4f} → {val_loss:.4f}). "
                f"Saving model to {self.path}"
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# --- 사용 예시 ---
early_stopping = EarlyStopping(patience=10, delta=1e-4, path="best_model.pt")

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print(f"Early stopping at epoch {epoch+1}")
        break

# 최적 모델 복원
model.load_state_dict(torch.load("best_model.pt", map_location=device, weights_only=True))
```

### 4. 모델 체크포인팅 (Save & Resume)

model/optimizer/scheduler/epoch/history를 저장하고 다음 epoch부터 재개한다. RNG·sampler·worker·early stopping 상태는 이 간단한 함수에 포함되지 않으므로 끊김 없는 학습과 동일한 결과를 보장하지 않는다. 아래 Trainer는 early stopping 상태도 저장한다.

```python
def save_checkpoint(
    path: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
    history: dict | None = None,
    **kwargs,
):
    """명시한 model/optimizer/scheduler/epoch/history를 저장한다."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history or {},
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if {"epoch", "model_state_dict", "optimizer_state_dict", "history", "scheduler_state_dict"} & kwargs.keys():
        raise ValueError("메타데이터가 예약된 checkpoint 키와 겹칩니다")
    checkpoint.update(kwargs)  # weights_only로 읽을 수 있는 tensor/기본 타입 메타데이터
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path} (epoch {epoch})")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """체크포인트에서 학습 상태를 복원한다."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    print(f"Checkpoint loaded: {path} (epoch {checkpoint['epoch']})")
    return checkpoint


# --- 사용 예시: 학습 재개 ---
# start_epoch = 0
# resume_path = "checkpoint_epoch_20.pt"
#
# if os.path.exists(resume_path):
#     ckpt = load_checkpoint(resume_path, model, optimizer, scheduler, device)
#     start_epoch = ckpt["epoch"] + 1
#     history = ckpt["history"]
#     print(f"Resuming from epoch {start_epoch}")
#
# for epoch in range(start_epoch, num_epochs):
#     ...
#     save_checkpoint(f"checkpoint_epoch_{epoch}.pt", epoch, model, optimizer,
#                     scheduler=scheduler, history=history)
```

### 5. 학습률 스케줄러 (Learning Rate Scheduler)

| 스케줄러 | 설명 | 사용 시점 |
|----------|------|-----------|
| `StepLR` | 지정 스텝마다 학습률을 `gamma` 배로 감소 | 간단한 실험 |
| `ReduceLROnPlateau` | 지표가 개선되지 않을 때 학습률 감소 | 검증 손실 기반 자동 조절 |
| `CosineAnnealingLR` | 코사인 함수 형태로 학습률을 서서히 감소 | 긴 학습, 미세 조정 |

```python
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --- 1) StepLR: 매 10 에포크마다 학습률을 0.1배로 ---
scheduler_step = StepLR(optimizer, step_size=10, gamma=0.1)

# --- 2) ReduceLROnPlateau: val_loss 기반 자동 감소 ---
scheduler_plateau = ReduceLROnPlateau(
    optimizer,
    mode="min",       # 손실이 줄어들어야 개선
    factor=0.5,       # 학습률을 절반으로
    patience=5,       # 허용 bad epoch 수5; 조건상 여섯째 bad epoch에서 감소
)

# --- 3) CosineAnnealingLR ---
scheduler_cosine = CosineAnnealingLR(
    optimizer,
    T_max=50,         # step 50회에서 eta_min에 도달; 자동 warm restart는 아님
    eta_min=1e-6,     # 최소 학습률
)


# --- 학습 루프에서의 사용 ---
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    # StepLR / CosineAnnealingLR: 에포크마다 step
    scheduler_step.step()
    # scheduler_cosine.step()

    # ReduceLROnPlateau: 검증 지표를 전달
    # scheduler_plateau.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch+1} | LR: {current_lr:.2e} | Val Loss: {val_loss:.4f}")
```

### 6. 학습 기록 시각화 (Plot Loss Curves)

```python
import matplotlib.pyplot as plt


def plot_history(history: dict, save_path: str | None = None):
    """학습/검증 손실 및 정확도 곡선을 시각화한다."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Loss ---
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=3)
    axes[0].plot(epochs, history["val_loss"], "r-o", label="Val Loss", markersize=3)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Train / Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Accuracy ---
    if "val_acc" in history:
        axes[1].plot(
            epochs, history["val_acc"], "g-o", label="Val Accuracy", markersize=3
        )
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_title("Validation Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.show()


# plot_history(history, save_path="training_curves.png")
```

### 7. 통합 학습 루프 (Trainer 클래스)

분류 루프·저장·기록을 Trainer로 묶는다. `tqdm` 표시를 포함하지만 AMP/DDP·원자적 저장·장애 복구·보안·데이터 버전 관리는 구현하지 않아 프로덕션 완성을 주장하지 않는다. 모델/데이터를 만들기 전에도 seed를 설정해야 한다. 생성자 안의 seed는 이미 만들어진 가중치를 되돌리지 않는다.

```python
import copy
import os
import random
import time
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm


# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
@dataclass
class TrainerConfig:
    """학습에 필요한 모든 하이퍼파라미터."""

    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42

    # Early Stopping
    patience: int = 10
    delta: float = 1e-4

    # Scheduler (ReduceLROnPlateau)
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 10

    # Device
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"

    def __post_init__(self):
        for name in ("num_epochs", "patience", "save_every_n_epochs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name}은 양의 정수여야 합니다")
        if not np.isfinite(self.delta) or self.delta < 0:
            raise ValueError("delta는 finite/nonnegative여야 합니다")

    def resolve_device(self) -> torch.device:
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("지원 장치는 auto/cpu/cuda/mps입니다")
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device)


# ──────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────
class Trainer:
    """PyTorch 모델 학습을 위한 범용 Trainer.

    기능:
        - Train / Validation 루프
        - Early Stopping
        - 모델 체크포인팅 (best + periodic)
        - ReduceLROnPlateau 스케줄러
        - tqdm 프로그레스 바
        - 학습 기록 시각화
        - 재현성 시드 고정
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        config: TrainerConfig | None = None,
    ):
        self.config = config or TrainerConfig()
        self.device = self.config.resolve_device()

        # 시드 고정
        self._set_seed(self.config.seed)

        # 모델 & 데이터
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(self.device)
        if not isinstance(criterion, nn.CrossEntropyLoss) or criterion.reduction != "mean" or criterion.weight is not None:
            raise ValueError("Trainer는 비가중 mean CrossEntropyLoss 분류 전용입니다")

        # Optimizer & Scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.config.scheduler_factor,
            patience=self.config.scheduler_patience,
        )

        # History
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
        }

        # Early Stopping 상태
        self._best_val_loss = float("inf")
        self._es_counter = 0
        self._best_model_state: dict | None = None
        self._start_epoch = 0

        # 체크포인트 디렉토리
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

    # ── 시드 고정 ──
    @staticmethod
    def _set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # PYTHONHASHSEED는 실행 전에 환경에서 설정; 여기서 hash seed를 바꾸지 않음
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ── Train 1 epoch ──
    def _train_one_epoch(self) -> float:
        self.model.train()
        running_loss = 0.0
        total = 0
        pbar = tqdm(self.train_loader, desc="  Train", leave=False)

        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            if outputs.ndim != 2 or targets.ndim != 1 or targets.dtype != torch.int64:
                raise ValueError("분류 출력(N,C)/int64 target(N,)가 필요합니다")
            if torch.any((targets < 0) | (targets >= outputs.size(1))):
                raise ValueError("ignore/unknown label은 지원하지 않습니다")
            loss = self.criterion(outputs, targets)
            if not torch.isfinite(loss).item():
                raise ValueError("손실이 finite가 아닙니다")
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * targets.size(0)
            total += targets.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        if total == 0:
            raise ValueError("학습 DataLoader가 비어 있습니다")
        return running_loss / total

    # ── Validate ──
    @torch.no_grad()
    def _validate(self) -> tuple[float, float]:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in self.val_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            if outputs.ndim != 2 or targets.ndim != 1 or targets.dtype != torch.int64:
                raise ValueError("분류 출력(N,C)/int64 target(N,)가 필요합니다")
            if torch.any((targets < 0) | (targets >= outputs.size(1))):
                raise ValueError("ignore/unknown label은 지원하지 않습니다")
            loss = self.criterion(outputs, targets)
            if not torch.isfinite(loss).item():
                raise ValueError("손실이 finite가 아닙니다")

            running_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        if total == 0:
            raise ValueError("검증 DataLoader가 비어 있습니다")
        avg_loss = running_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    # ── 체크포인트 저장 ──
    def _save_checkpoint(self, epoch: int, filename: str):
        path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "history": self.history,
                "best_val_loss": self._best_val_loss,
                "es_counter": self._es_counter,
                "best_model_state": self._best_model_state,
            },
            path,
        )
        return path

    # ── 체크포인트 복원 ──
    def resume(self, path: str) -> int:
        """체크포인트에서 학습 상태를 복원하고, 다음 시작 에포크를 반환한다."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        required = {"epoch", "model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "history", "best_val_loss", "es_counter", "best_model_state"}
        if not required <= ckpt.keys():
            raise ValueError("이 판본의 재개 상태 키가 누락되었습니다")
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.history = ckpt["history"]
        self._best_val_loss = ckpt["best_val_loss"]
        # 이전 템플릿 checkpoint의 누락 상태를 0/default로 추측하지 않는다.
        self._es_counter = ckpt["es_counter"]
        self._best_model_state = ckpt["best_model_state"]
        self._start_epoch = ckpt["epoch"] + 1
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {path} (next epoch: {start_epoch})")
        return start_epoch

    # ── Early Stopping 체크 ──
    def _check_early_stopping(self, val_loss: float, epoch: int) -> bool:
        if not np.isfinite(val_loss):
            raise ValueError("검증 손실이 finite가 아닙니다")
        if val_loss < self._best_val_loss - self.config.delta:
            self._best_val_loss = val_loss
            self._es_counter = 0
            self._best_model_state = copy.deepcopy(self.model.state_dict())
            path = self._save_checkpoint(epoch, "best_model.pt")
            print(f"  ** Best model saved (val_loss={val_loss:.4f}) -> {path}")
            return False
        else:
            self._es_counter += 1
            print(
                f"  EarlyStopping: {self._es_counter}/{self.config.patience}"
            )
            return self._es_counter >= self.config.patience

    # ── 메인 학습 루프 ──
    def fit(self, start_epoch: int | None = None):
        """학습을 실행한다."""
        print(f"Device: {self.device}")
        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Config: {self.config}")
        print("-" * 60)

        if start_epoch is None:
            start_epoch = self._start_epoch
        if not 0 <= start_epoch < self.config.num_epochs:
            raise ValueError("시작 epoch는 남은 학습 범위 안에 있어야 합니다")
        total_start = time.time()

        for epoch in range(start_epoch, self.config.num_epochs):
            epoch_start = time.time()

            # Train
            train_loss = self._train_one_epoch()

            # Validate
            val_loss, val_acc = self._validate()

            # Scheduler step
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # 기록
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            elapsed = time.time() - epoch_start
            print(
                f"Epoch [{epoch+1}/{self.config.num_epochs}] "
                f"({elapsed:.1f}s) | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.2f}% | "
                f"LR: {current_lr:.2e}"
            )

            # Early Stopping
            if self._check_early_stopping(val_loss, epoch):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

            # 주기적 체크포인트
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(epoch, f"checkpoint_epoch_{epoch+1}.pt")

        total_elapsed = time.time() - total_start
        print(f"\nTraining complete in {total_elapsed:.1f}s")
        print(f"Best val loss: {self._best_val_loss:.4f}")

        # 최적 모델 복원
        if self._best_model_state is not None:
            self.model.load_state_dict(self._best_model_state)
            self.model.eval()
            print("Best model weights restored for inference.")

    # ── 시각화 ──
    def plot(self, save_path: str | None = None):
        """학습 기록을 시각화한다."""
        epochs = range(1, len(self.history["train_loss"]) + 1)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Loss
        axes[0].plot(epochs, self.history["train_loss"], "b-", label="Train")
        axes[0].plot(epochs, self.history["val_loss"], "r-", label="Val")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy
        axes[1].plot(epochs, self.history["val_acc"], "g-", label="Val Acc")
        axes[1].set_title("Validation Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("%")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Learning Rate
        axes[2].plot(epochs, self.history["lr"], "m-", label="LR")
        axes[2].set_title("Learning Rate")
        axes[2].set_xlabel("Epoch")
        axes[2].set_yscale("log")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()


# ──────────────────────────────────────────────
# 사용 예시
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import torch.nn.functional as F
    from torchvision import datasets, transforms
    from torch.utils.data import random_split

    # 모델·분할 생성 전에 seed 적용 (Trainer 생성자만으로 초기 가중치는 제어 불가)
    Trainer._set_seed(42)

    # 데이터 (CIFAR-10 예시)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    # 공식 test split은 튜닝/early stopping에 쓰지 않는다.
    train_ds, val_ds = random_split(train_ds, [40000, 10000], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    # 간단한 CNN 모델
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(256, 10),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    # Trainer 실행
    config = TrainerConfig(
        num_epochs=30,
        learning_rate=1e-3,
        patience=7,
        checkpoint_dir="checkpoints/cifar10",
    )

    trainer = Trainer(
        model=SimpleCNN(),
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(),
        config=config,
    )
    trainer.fit()
    trainer.plot(save_path="cifar10_training_curves.png")
```

---

각 독립 run은 별도 checkpoint_dir를 사용한다. Trainer는 메모리에 best state를 deepcopy하고 checkpoint에도 포함하므로 가중치 저장 비용이 늘어난다. 다른 run의 best_model.pt를 자동으로 읽지 않는다. resume은 이 판본에서 저장한 es_counter/best_model_state 키를 요구하고 이전 checkpoint를 임의 기본값으로 복원하지 않는다. RNG·DataLoader shuffle/worker 상태는 저장하지 않아 resume이 끊김 없는 학습과 수치적으로 같다는 보장은 없다. fit 종료는 추론용 best 가중치만 복원하므로 이후 학습은 재개용 checkpoint를 resume해야 optimizer/scheduler와 모델 상태가 맞는다. 파일은 신뢰 가능한 자신의 기록만 읽고 메타데이터는 weights_only가 허용하는 타입으로 제한한다.

통합 CIFAR 예제는 공식 train50,000개를40,000/10,000으로 분할하고 공식 test10,000개를 선택 과정에서 사용하지 않는다. test 최종 평가 코드는 별도로 구성해야 한다. download=True는 네트워크·저장 공간을 요구하며 여기서 실제 다운로드/전체 학습을 검증하지 않았다. worker>0은 파일 최상위 함수와 main guard·worker RNG를 확인한 뒤 설정한다. 미지원/빈 입력·non-finite loss는 학습 성공으로 숨기지 않는다.

## 참고 자료 (References)

2026-10-04 확인: 공식2.14 계약, 로컬 설치2.14.1을 구분한다.

- [ReduceLROnPlateau 현재 signature·bad epoch 조건](https://docs.pytorch.org/docs/2.14/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html)
- [optimizer 이후 scheduler.step](https://docs.pytorch.org/docs/2.14/optim.html)
- [체크포인트 상태·best deepcopy](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- [CrossEntropyLoss reduction/weight/ignore 조건](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)
- [torch.load weights_only 제한](https://docs.pytorch.org/docs/2.14/notes/serialization.html)
- [DataLoader worker 설정](https://docs.pytorch.org/docs/2.14/data.html)
- [CIFAR10 원 저자 데이터 크기/분할](https://www.cs.toronto.edu/~kriz/cifar.html)
- [CIFAR10 공식 dataset 계약](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html)
- [PYTHONHASHSEED 시작 환경](https://docs.python.org/3/using/cmdline.html)
- [PyTorch Training Loop 공식 튜토리얼](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [PyTorch Learning Rate Scheduler 문서](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
- [Reproducibility in PyTorch](https://pytorch.org/docs/stable/notes/randomness.html)
- [torch.save / torch.load 공식 문서](https://pytorch.org/docs/stable/generated/torch.save.html)
- [tqdm 프로그레스 바](https://github.com/tqdm/tqdm)

---

## 관련 문서

- [딥러닝 읽기 순서](./README.md)
- 데이터 전처리/파이프라인은 학습 루프 전에 준비해야 한다. 원래 ../../data-processing/ 링크는 대상이 없어 제거했다.
