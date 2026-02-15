---
tags: [pytorch, training-loop, early-stopping, checkpointing]
level: intermediate
last_updated: 2026-02-14
status: in-progress
---

# PyTorch 학습 루프 템플릿

> 재현 가능하고 견고한 딥러닝 학습 루프를 처음부터 끝까지 구성하는 실전 템플릿.

## 왜 필요한가? (Why)

- 학습 루프(Training Loop)는 **모든 딥러닝 프로젝트의 뼈대**다. 모델 정의보다 학습 루프의 품질이 실험 생산성을 좌우한다.
- 매번 처음부터 작성하면 Early Stopping, 체크포인팅, 로깅 등 필수 기능을 빠뜨리기 쉽다.
- **복붙 가능한 표준 템플릿**을 갖추면 새 프로젝트를 시작할 때 수 시간을 절약할 수 있다.
- 재현성(Reproducibility)을 보장하려면 시드 고정, 결정론적 설정 등을 학습 루프에 내장해야 한다.

---

## 핵심 개념 (What)

| 개념 | 설명 |
|------|------|
| **Training Loop** | 배치 단위로 forward → loss → backward → optimizer step을 반복하는 핵심 루프 |
| **Validation Loop** | 매 에포크 종료 후 검증 데이터로 모델 성능을 평가 (gradient 계산 없음) |
| **Early Stopping** | 검증 손실이 일정 에포크(patience) 동안 개선되지 않으면 학습을 조기 종료 |
| **Checkpointing** | 최적 모델 가중치를 파일로 저장하고, 필요 시 복원(resume) |
| **Learning Rate Scheduler** | 에포크/스텝에 따라 학습률을 동적으로 조절 |
| **Seed Fixing** | `torch.manual_seed` 등으로 난수를 고정해 실험 재현성 확보 |

---

## 어떻게 사용하는가? (How)

### 0. 공통 설정 및 재현성(Reproducibility) 시드 고정

모든 실험 전에 반드시 시드를 고정한다.

```python
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """모든 난수 시드를 고정하여 실험 재현성을 보장한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU
    os.environ["PYTHONHASHSEED"] = str(seed)
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

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
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

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / len(dataloader)
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

검증 손실이 `patience` 에포크 동안 `delta` 이상 개선되지 않으면 학습을 조기 종료한다.

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
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose

        self.counter = 0
        self.best_score: float | None = None
        self.early_stop = False
        self.val_loss_min = float("inf")

    def __call__(self, val_loss: float, model: nn.Module):
        score = -val_loss  # 손실이 작을수록 좋으므로 부호 반전

        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
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
model.load_state_dict(torch.load("best_model.pt", weights_only=True))
```

### 4. 모델 체크포인팅 (Save & Resume)

학습 중간 상태를 저장하고, 중단된 지점부터 재개할 수 있다.

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
    """학습 상태 전체를 체크포인트로 저장한다."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history or {},
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    checkpoint.update(kwargs)  # 추가 메타데이터
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
    checkpoint = torch.load(path, map_location=device, weights_only=False)
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
    patience=5,       # 5 에포크 개선 없으면 감소
    verbose=True,
)

# --- 3) CosineAnnealingLR ---
scheduler_cosine = CosineAnnealingLR(
    optimizer,
    T_max=50,         # 한 주기 = 50 에포크
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

### 7. 완전한 프로덕션 학습 루프 (Trainer 클래스)

위의 모든 요소를 하나의 `Trainer` 클래스로 통합한다. `tqdm` 프로그레스 바 포함.

```python
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

    def resolve_device(self) -> torch.device:
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
        self.criterion = criterion

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

        # 체크포인트 디렉토리
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

    # ── 시드 고정 ──
    @staticmethod
    def _set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ── Train 1 epoch ──
    def _train_one_epoch(self) -> float:
        self.model.train()
        running_loss = 0.0
        pbar = tqdm(self.train_loader, desc="  Train", leave=False)

        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return running_loss / len(self.train_loader)

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
            loss = self.criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        avg_loss = running_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total if total > 0 else 0.0
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
            },
            path,
        )
        return path

    # ── 체크포인트 복원 ──
    def resume(self, path: str) -> int:
        """체크포인트에서 학습 상태를 복원하고, 다음 시작 에포크를 반환한다."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.history = ckpt["history"]
        self._best_val_loss = ckpt["best_val_loss"]
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {path} (next epoch: {start_epoch})")
        return start_epoch

    # ── Early Stopping 체크 ──
    def _check_early_stopping(self, val_loss: float, epoch: int) -> bool:
        if val_loss < self._best_val_loss - self.config.delta:
            self._best_val_loss = val_loss
            self._es_counter = 0
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
    def fit(self, start_epoch: int = 0):
        """학습을 실행한다."""
        print(f"Device: {self.device}")
        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Config: {self.config}")
        print("-" * 60)

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
        best_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            print("Best model weights restored.")

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

    # 데이터 (CIFAR-10 예시)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    val_ds = datasets.CIFAR10("./data", train=False, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2)

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

## 참고 자료 (References)

- [PyTorch Training Loop 공식 튜토리얼](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [PyTorch Learning Rate Scheduler 문서](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
- [Reproducibility in PyTorch](https://pytorch.org/docs/stable/notes/randomness.html)
- [torch.save / torch.load 공식 문서](https://pytorch.org/docs/stable/generated/torch.save.html)
- [tqdm 프로그레스 바](https://github.com/tqdm/tqdm)

---

## 관련 문서

- [상위 폴더: Deep Learning](../deep-learning/)
- [데이터 처리 파이프라인](../../data-processing/)
