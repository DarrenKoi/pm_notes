---
tags: [rnn, lstm, gru, time-series, pytorch]
level: intermediate
last_updated: 2026-02-14
reviewed_on: 2026-10-04
review_status: partial
document_type: learning_note
---

# 시퀀스 모델 (Sequence Models): RNN, LSTM, GRU

> 시계열 및 순차 데이터를 처리하는 딥러닝 모델의 핵심 아키텍처와 PyTorch 구현 가이드

> [!info] 적용 범위와 검증일
> 2026-10-04 공식 PyTorch2.14 문서를 대조했다. 설치2.14.1/Python3.14.2의 CPU 합성 입력을 검증한다. 1~5절은 앞 import/정의를 쓰는 조각, 6절은 독립 합성 예제, 7절은 별도 packing 예제다. 실제 센서·불규칙 시간/결측·GPU/worker·장기 예측 성능·Obsidian 읽기 화면은 미확인이다. 생성·분할·모델 초기화 전 seed를 설정하며 판본/장치 간 같은 결과를 보장하지 않는다.

## 왜 필요한가? (Why)

- **제조/반도체 공정에서 시계열 데이터는 핵심**: 장비 센서 로그, 공정 파라미터 변화, 웨이퍼 계측 추이 등 순차적으로 발생하는 데이터가 예측/감지의 입력이 될 수 있다
- **시간적 의존성 포착**: MLP도 시차/위치를 명시한 입력으로 순서를 표현할 수 있다. 순환 모델은 **이전 시점의 정보를 기억**하며 다음 시점을 예측한다
- **다양한 실무 활용**:
  - 장비 이상 감지(Anomaly Detection): 센서 값 패턴이 정상 범위를 벗어나는 시점 탐지
  - 시계열 예측(Forecasting): 공정 파라미터, 수율 트렌드 예측
  - 시퀀스 분류: 공정 레시피 패턴 분류, 불량 유형 판별
- **왜 RNN 계열인가?**: LSTM/GRU는 시간축을 따라 상태를 갱신하며 가변 길이 입력을 처리하는 후보다. 자원·시간 의존성·baseline과 비교해 선택하고 트렌드만으로 모델을 고르지 않는다

---

## 핵심 개념 (What)

### 1. RNN (Recurrent Neural Network)

가장 기본적인 시퀀스 모델. 각 시간 단계(time step)에서 **이전 hidden state**를 입력과 함께 받아 새로운 hidden state를 생성한다.

```
h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b)
```

**문제점**: 시퀀스가 길어지면 **기울기 소실(Vanishing Gradient)** 문제로 장기 의존성 학습이 어려워질 수 있다.

### 2. LSTM (Long Short-Term Memory)

장기 상태 전달을 돕고 기울기 소실을 완화하기 위해 **게이트 메커니즘** 도입:

| 게이트 | 역할 |
|--------|------|
| **Forget Gate** | 이전 셀 상태에서 버릴 정보 결정 |
| **Input Gate** | 새로운 정보 중 저장할 부분 결정 |
| **Output Gate** | 셀 상태에서 출력할 부분 결정 |

- **Cell State(c_t)**: 장기 기억을 전달하는 "컨베이어 벨트" 역할
- **Hidden State(h_t)**: 현재 시점의 출력

### 3. GRU (Gated Recurrent Unit)

LSTM의 간소화 버전. 별도 cell state 없이 두 게이트를 사용한다. 같은 입력/hidden/layer/bias 설정이면 순환 파라미터는 LSTM보다 적지만 실제 속도는 kernel·장치·배치에 따라 측정한다:

| 게이트 | 역할 |
|--------|------|
| **Reset Gate** | 이전 hidden state를 얼마나 무시할지 결정 |
| **Update Gate** | 이전 hidden state와 새 후보를 어떤 비율로 섞을지 결정 |

- Cell State와 Hidden State를 **하나로 통합** (h_t만 존재)

### 4. Hidden State 이해

```
시퀀스: [x_1, x_2, x_3, ..., x_T]

x_1 → [RNN Cell] → h_1
x_2, h_1 → [RNN Cell] → h_2
x_3, h_2 → [RNN Cell] → h_3
...
x_T, h_{T-1} → [RNN Cell] → h_T  ← 최종 hidden state (시퀀스 요약)
```

- `h_T`는 전체 시퀀스의 **압축된 표현**
- 분류 태스크: `h_T`를 FC layer에 통과시켜 클래스 예측
- 예측 태스크: `h_T` 또는 각 시점의 `h_t`를 사용

### 5. 시퀀스 패딩 (Sequence Padding)

배치 내 시퀀스 길이가 다를 때 **가장 긴 시퀀스 기준으로 짧은 시퀀스를 0으로 채우는** 기법:

```
원본:    [[1,2,3], [4,5], [6]]
패딩 후: [[1,2,3], [4,5,0], [6,0,0]]
```

PyTorch에서는 `pack_padded_sequence`로 패딩된 부분을 무시하고 효율적으로 연산한다.

---

## 어떻게 사용하는가? (How)

### 1. 시계열 데이터 준비: Sliding Window Dataset

시계열 데이터를 모델에 넣으려면 **sliding window**로 (입력, 타겟) 쌍을 만든다.

```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class TimeSeriesDataset(Dataset):
    """x:(window,features), y:(horizon,features)를 안정적으로 반환한다."""

    def __init__(self, data: np.ndarray, window_size: int, horizon: int = 1):
        for name, value in (("window_size", window_size), ("horizon", horizon)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name}은 양의 정수여야 합니다")
        arr = np.asarray(data)
        if arr.ndim not in (1, 2) or (arr.ndim == 2 and arr.shape[1] == 0):
            raise ValueError("data는 (time,) 또는 (time,features)여야 합니다")
        if not np.issubdtype(arr.dtype, np.number) or np.iscomplexobj(arr):
            raise ValueError("data는 실수 수치 배열이어야 합니다")
        self.data = torch.tensor(arr, dtype=torch.float32)
        if not torch.isfinite(self.data).all():
            raise ValueError("결측/non-finite 값은 윈도 생성 전에 처리해야 합니다")
        if self.data.ndim == 1:
            self.data = self.data.unsqueeze(-1)
        self.window_size = int(window_size)
        self.horizon = int(horizon)

    def __len__(self):
        return max(0, len(self.data) - self.window_size - self.horizon + 1)

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        x = self.data[idx:idx + self.window_size]
        y = self.data[idx + self.window_size:idx + self.window_size + self.horizon]
        return x, y


# 사용 예시
np.random.seed(42)
torch.manual_seed(42)
raw_data = np.random.randn(1000)  # 1000개 시점의 센서 데이터
dataset = TimeSeriesDataset(raw_data, window_size=30, horizon=1)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for x_batch, y_batch in loader:
    print(f"입력: {x_batch.shape}")   # (32, 30, 1)
    print(f"타겟: {y_batch.shape}")   # (32, 1, 1)
    break
```

---

각 window의 내부 시간 순서는 유지한다. shuffle=True는 독립 window 표본/매 호출 초기 상태를 사용하는 학습 범위 안에서만 허용하며, 전체 시계열을 window로 만든 뒤 무작위 train/test로 나누지 않는다. 짧은 입력은 dataset 길이0이며 학습 전에 표본 수를 확인한다. 모델 output_size를 horizon×features로 설정한 뒤 (batch,horizon,features)로 reshape하거나 명시적으로 타겟 축을 선택해야 한다.

### 2. 기본 RNN: nn.RNN 이해

```python
import torch
import torch.nn as nn


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,    # 입력 특성 수 (예: 센서 1개면 1)
            hidden_size=hidden_size,  # hidden state 차원
            num_layers=num_layers,    # RNN 층 수
            batch_first=True,         # 입력 shape: (batch, seq_len, input_size)
            dropout=0.1 if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0=None):
        # x: (batch, seq_len, input_size)
        # output: (batch, seq_len, hidden_size) — 모든 시점의 hidden state
        # hn: (num_layers, batch, hidden_size) — 마지막 시점의 hidden state
        output, hn = self.rnn(x, h0)

        # 마지막 시점의 hidden state만 사용하여 예측
        last_hidden = output[:, -1, :]  # (batch, hidden_size)
        pred = self.fc(last_hidden)     # (batch, output_size)
        return pred


# 테스트
model = SimpleRNN(input_size=1, hidden_size=64, output_size=1)
x = torch.randn(32, 30, 1)  # batch=32, seq_len=30, features=1
y_pred = model(x)
print(f"예측 shape: {y_pred.shape}")  # (32, 1)
```

**핵심 포인트**:
- 이 예제는 `batch_first=True`를 명시 (기본값은 `False`이므로 주의)
- `output`은 모든 시점의 hidden state, `hn`은 마지막 시점만
- 이 단방향·고정 길이 예제는 `output[:, -1, :]`를 쓴다. padding 뒤의 마지막 위치는 실제 마지막 시점이 아닐 수 있어 packing/length를 적용한다

---

### 3. LSTM 구현: 시계열 예측

```python
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        # lstm_out: (batch, seq_len, hidden_size)
        # (hn, cn): 각각 (num_layers, batch, hidden_size)
        lstm_out, (hn, cn) = self.lstm(x)

        # 마지막 시점 출력 사용
        last_out = lstm_out[:, -1, :]  # (batch, hidden_size)
        last_out = self.dropout(last_out)
        pred = self.fc(last_out)       # (batch, output_size)
        return pred


# 초기화
model = LSTMPredictor(input_size=1, hidden_size=128, output_size=1, num_layers=2)
print(f"총 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
```

**LSTM vs RNN 차이점**:
- `nn.LSTM`은 hidden state 외에 **cell state(cn)**도 반환한다
- 입력/출력 시퀀스 형태는 비슷하지만 LSTM의 초기/최종 상태는 (h,c) 쌍이고 projection·양방향일 때 차원이 달라 직접 치환 전에 확인한다

---

### 4. GRU 구현: LSTM 대비 경량 대안

```python
class GRUPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # gru_out: (batch, seq_len, hidden_size)
        # hn: (num_layers, batch, hidden_size)  ← cell state 없음!
        gru_out, hn = self.gru(x)

        last_out = gru_out[:, -1, :]
        last_out = self.dropout(last_out)
        pred = self.fc(last_out)
        return pred


# GRU는 LSTM 대비 파라미터가 약 25% 적다
model_gru = GRUPredictor(input_size=1, hidden_size=128, output_size=1, num_layers=2)
model_lstm = LSTMPredictor(input_size=1, hidden_size=128, output_size=1, num_layers=2)

gru_params = sum(p.numel() for p in model_gru.parameters())
lstm_params = sum(p.numel() for p in model_lstm.parameters())
print(f"GRU 파라미터:  {gru_params:,}")
print(f"LSTM 파라미터: {lstm_params:,}")
print(f"GRU/LSTM 비율: {gru_params / lstm_params:.2%}")
```

---

### 5. 양방향(Bidirectional) 모델

시퀀스를 **앞→뒤, 뒤→앞 양방향**으로 처리하여 더 풍부한 문맥 정보를 얻는다. 시계열 분류(classification)에 유용하지만, **시점별 온라인 출력에서 아직 관측하지 못한 미래 입력을 쓰면 누수다. 예측 시점까지 관측된 과거 window 전체를 양방향으로 인코딩해 window 밖 미래를 예측하는 것은 가능하다**하다.

```python
class BiLSTMClassifier(nn.Module):
    """양방향 LSTM 기반 시퀀스 분류기"""

    def __init__(self, input_size, hidden_size, num_classes, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,  # 양방향 활성화
            dropout=0.3 if num_layers > 1 else 0,
        )
        # bidirectional=True이면 출력 차원이 hidden_size * 2
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # lstm_out: (batch, seq_len, hidden_size * 2)
        lstm_out, (hn, cn) = self.lstm(x)

        # 양방향의 마지막 hidden state 결합
        # hn shape: (num_layers * 2, batch, hidden_size)
        # 정방향 마지막 layer: hn[-2], 역방향 마지막 layer: hn[-1]
        forward_last = hn[-2]   # (batch, hidden_size)
        backward_last = hn[-1]  # (batch, hidden_size)
        combined = torch.cat([forward_last, backward_last], dim=1)  # (batch, hidden_size*2)

        out = self.fc(combined)  # (batch, num_classes)
        return out


# 사용 예시: 3-클래스 분류 (정상 / 이상 유형 A / 이상 유형 B)
model = BiLSTMClassifier(input_size=5, hidden_size=64, num_classes=3)
x = torch.randn(16, 50, 5)  # batch=16, seq_len=50, features=5
logits = model(x)
print(f"출력 shape: {logits.shape}")  # (16, 3)
```

**주의**: `bidirectional=True` 사용 시:
- `output`의 마지막 차원이 `hidden_size * 2`로 변한다
- `hn`의 첫 번째 차원이 `num_layers * 2`로 변한다
- 전체 입력이 관측된 분류/라벨링에 적용할 수 있다. 온라인 시점별 라벨링에도 미래 입력이 포함되면 안 된다. 양방향 output[:, -1]의 역방향 부분은 전체 역방향 마지막 state와 다르므로 위 hn[-2]/hn[-1]을 결합한다

---

### 6. 시계열 예측 완전 예제: Sine Wave

합성 사인파 데이터를 생성하고 LSTM으로 학습 후 예측 결과를 시각화하는 end-to-end 예제.

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 설치된 한글 font를 선택; font 자체는 이 예제에서 설치하지 않음
families = {font.name for font in font_manager.fontManager.ttflist}
for family in ("NanumGothic", "Malgun Gothic", "AppleGothic"):
    if family in families:
        plt.rcParams["font.family"] = family
        break
else:
    print("한글 font 미설정: 그림의 한글 표시를 확인하고 font를 설치/설정하세요")
plt.rcParams["axes.unicode_minus"] = False

# ========== 1. 합성 데이터 생성 ==========
def generate_sine_data(n_points=2000, noise_std=0.05):
    """노이즈가 섞인 사인파 데이터 생성"""
    t = np.linspace(0, 80 * np.pi, n_points)
    data = np.sin(t) + noise_std * np.random.randn(n_points)
    return data.astype(np.float32)


# ========== 2. Dataset 클래스 ==========
class SineDataset(Dataset):
    """x:(window,features), y:(horizon,features)를 안정적으로 반환한다."""

    def __init__(self, data: np.ndarray, window_size: int = 50, horizon: int = 1):
        for name, value in (("window_size", window_size), ("horizon", horizon)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name}은 양의 정수여야 합니다")
        arr = np.asarray(data)
        if arr.ndim not in (1, 2) or (arr.ndim == 2 and arr.shape[1] == 0):
            raise ValueError("data는 (time,) 또는 (time,features)여야 합니다")
        if not np.issubdtype(arr.dtype, np.number) or np.iscomplexobj(arr):
            raise ValueError("data는 실수 수치 배열이어야 합니다")
        self.data = torch.tensor(arr, dtype=torch.float32)
        if not torch.isfinite(self.data).all():
            raise ValueError("결측/non-finite 값은 윈도 생성 전에 처리해야 합니다")
        if self.data.ndim == 1:
            self.data = self.data.unsqueeze(-1)
        self.window_size = int(window_size)
        self.horizon = int(horizon)

    def __len__(self):
        return max(0, len(self.data) - self.window_size - self.horizon + 1)

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        x = self.data[idx:idx + self.window_size]
        y = self.data[idx + self.window_size:idx + self.window_size + self.horizon]
        return x, y


# ========== 3. 모델 정의 ==========
class SineLSTM(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2, horizon=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        pred = self.fc(out[:, -1, :])
        return pred.unsqueeze(-1)  # (batch,horizon,1)


# ========== 4. 학습 함수 ==========
def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, device="cpu"):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)
            if pred.shape != y_batch.shape:
                raise ValueError("예측/타겟 shape가 다릅니다")
            loss = criterion(pred, y_batch)
            if not torch.isfinite(loss).item():
                raise ValueError("손실이 finite가 아닙니다")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            train_loss_sum += loss.item() * y_batch.numel()
            train_count += y_batch.numel()

        # --- Validate ---
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                pred = model(x_batch)
                if pred.shape != y_batch.shape:
                    raise ValueError("예측/타겟 shape가 다릅니다")
                loss = criterion(pred, y_batch)
                if not torch.isfinite(loss).item():
                    raise ValueError("손실이 finite가 아닙니다")
                val_loss_sum += loss.item() * y_batch.numel()
                val_count += y_batch.numel()

        if train_count == 0 or val_count == 0:
            raise ValueError("train/val DataLoader가 비어 있습니다")
        avg_train = train_loss_sum / train_count
        avg_val = val_loss_sum / val_count
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        scheduler.step(avg_val)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")

    return history


# ========== 5. 실행 ==========
if __name__ == "__main__":
    WINDOW_SIZE = 50
    BATCH_SIZE = 64
    EPOCHS = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    np.random.seed(42)
    torch.manual_seed(42)
    # 시간 순서로 train60%/val20%/test20%; 각 구간 내부의 window만 사용
    data = generate_sine_data(n_points=2000)
    train_end, val_end = int(len(data) * 0.6), int(len(data) * 0.8)
    train_data, val_data, test_data = data[:train_end], data[train_end:val_end], data[val_end:]

    train_ds = SineDataset(train_data, window_size=WINDOW_SIZE)
    val_ds = SineDataset(val_data, window_size=WINDOW_SIZE)
    test_ds = SineDataset(test_data, window_size=WINDOW_SIZE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 모델 학습
    model = SineLSTM(hidden_size=64, num_layers=2)
    history = train_model(model, train_loader, val_loader, epochs=EPOCHS, device=device)

    # ========== 6. 예측 및 시각화 ==========
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            pred = model(x_batch.to(device)).cpu()
            predictions.extend(pred[:, 0, 0].numpy())
            actuals.extend(y_batch[:, 0, 0].numpy())

    plt.figure(figsize=(14, 5))

    # 예측 결과
    plt.subplot(1, 2, 1)
    plt.plot(actuals, label="실제값", alpha=0.7)
    plt.plot(predictions, label="예측값", alpha=0.7)
    plt.title("LSTM 시계열 예측 결과")
    plt.xlabel("시간 (Time Step)")
    plt.ylabel("값")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 학습 곡선
    plt.subplot(1, 2, 2)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("학습 곡선")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("lstm_sine_prediction.png", dpi=150)
    plt.show()
    print("완료! 그래프가 lstm_sine_prediction.png에 저장되었습니다.")
```

---

6절은 horizon1 단변량 one-step 예측이며 각 test window 안의 실제 관측 과거값을 사용한다. 자기 예측을 다시 넣는 recursive 장기 예측이 아니다. train/val/test를 시간축에서 먼저 나누고 각 구간 내부 window만 만들어 경계의 초기50시점은 입력으로 소비한다. 원래2,000점에서는 train1,150/val350/test350 window다. 검증 손실은 LR 선택에만 쓰고 test는 학습 뒤 시각화에 사용한다. 그래프는 설치된 한글 font를 선택하고, 없으면 별도 설정이 필요하다. 모델은 마지막 epoch이며 best checkpoint 선택은 이 예제에 구현하지 않았다. SineDataset의 window 계약은1절과 같지만 독립 실행을 위해 정의를 포함한다. 실제 장비에서는 그룹·시간 간격·결측·수집 지연을 먼저 정의한다.

### 7. 시퀀스 패딩 & 팩킹

배치 내 시퀀스 길이가 다를 때 `pack_padded_sequence`와 `pad_packed_sequence`를 사용하면 **패딩 timestep을 순환 상태 갱신에서 제외한다. 속도 이득은 길이 분포·backend·overhead에 따라 측정한다**.

```python
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


def collate_variable_length(batch):
    """가변 길이 시퀀스를 패딩하고 길이 정보를 함께 반환하는 collate 함수"""
    if not batch:
        raise ValueError("batch가 비어 있습니다")
    sequences, labels = zip(*batch)
    if any(seq.ndim != 2 or len(seq) == 0 for seq in sequences):
        raise ValueError("길이가 양수인 (time,features) tensor가 필요합니다")

    # 실제 길이 기록 (패딩 전)
    lengths = torch.tensor([len(seq) for seq in sequences])

    # 가장 긴 시퀀스 기준으로 패딩
    padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)

    # enforce_sorted=True 선택에 맞춰 정렬; False라면 내부 정렬 가능
    sorted_idx = lengths.argsort(descending=True)
    padded = padded[sorted_idx]
    lengths = lengths[sorted_idx]
    labels = torch.stack([torch.as_tensor(label) for label in labels])[sorted_idx]

    return padded, lengths, labels, sorted_idx.argsort()  # 원래 batch 순서 복원 index


class PackedLSTM(nn.Module):
    """패딩/팩킹을 지원하는 LSTM 모델"""

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x_padded, lengths):
        if lengths.ndim != 1 or lengths.dtype != torch.int64 or len(lengths) != x_padded.size(0):
            raise ValueError("배치 수에 맞는 int64 lengths가 필요합니다")
        if torch.any((lengths <= 0) | (lengths > x_padded.size(1))):
            raise ValueError("lengths는 실제 padded 시간축 범위 안이어야 합니다")
        # 패딩된 시퀀스를 packed 형태로 변환
        packed = pack_padded_sequence(x_padded, lengths.cpu(), batch_first=True, enforce_sorted=True)

        # LSTM에 packed 시퀀스 전달
        packed_out, (hn, cn) = self.lstm(packed)

        # 다시 패딩된 형태로 복원 (필요 시)
        # output_padded, output_lengths = pad_packed_sequence(packed_out, batch_first=True)

        # 마지막 layer의 hidden state 사용
        last_hidden = hn[-1]  # (batch, hidden_size)
        return self.fc(last_hidden)


# 사용 예시
# 가변 길이 시퀀스 3개 (특성 차원 = 4)
sequences = [
    torch.randn(10, 4),  # 길이 10
    torch.randn(7, 4),   # 길이 7
    torch.randn(15, 4),  # 길이 15
]
labels = [torch.tensor(0), torch.tensor(1), torch.tensor(0)]
batch = list(zip(sequences, labels))

padded, lengths, labels, restore_idx = collate_variable_length(batch)
print(f"패딩된 배치 shape: {padded.shape}")  # (3, 15, 4) — 최대 길이 15 기준
print(f"실제 길이: {lengths}")               # tensor([15, 10, 7])

model = PackedLSTM(input_size=4, hidden_size=32, output_size=2)
output = model(padded, lengths)
print(f"출력 shape: {output.shape}")  # (3, 2)
output_original_order = output[restore_idx]  # inference에서 원 입력 순서가 필요할 때
```

**언제 팩킹이 필요한가?**
- 배치 내 시퀀스 길이가 **크게 다를 때** (예: 장비마다 공정 시간이 다른 경우)
- 패딩만 사용하면 0으로 채워진 부분도 연산하므로 **불필요한 계산 발생**
- 팩킹을 쓰면 실제 데이터 길이만큼만 연산하여 padding을 실제 데이터로 해석하는 상태 오염을 피할 수 있다. 속도/정확도 개선 보장은 아니다

---

### 8. 모델 비교: RNN vs LSTM vs GRU

| 항목 | RNN | LSTM | GRU |
|------|-----|------|-----|
| **게이트 수** | 없음 | 3개 (forget, input, output) | 2개 (reset, update) |
| **상태** | hidden state만 | hidden + cell state | hidden state만 |
| **파라미터 수** (hidden=128 기준) | 가장 적음 | 가장 많음 | LSTM의 ~75% |
| **학습 속도** | 세 모델 모두 입력/장치/구현별 실측 필요 | 실측 필요 | 실측 필요 |
| **메모리 사용** | hidden/layer/길이·activation·optimizer에 따라 측정 | 같은 조건에서 비교 | 같은 조건에서 비교 |
| **장기 의존성** | 긴 경로의 gradient가 어려울 수 있음 | 게이트가 전달을 돕지만 데이터/학습 조건부 | 게이트가 전달을 돕지만 조건부 |
| **기울기 소실** | 긴 경로에서 발생 가능 | 완화, 해결 보장 없음 | 완화, 해결 보장 없음 |
| **추천 상황** | 짧은 시퀀스, 빠른 프로토타입 | 긴 시퀀스, 복잡한 패턴 | LSTM과 비슷한 성능이 필요하지만 리소스가 제한될 때 |

**실무 선택 가이드**:
- persistence/시차 기반 단순 baseline과 RNN/LSTM/GRU를 같은 시간 분할·지표·예산에서 비교한다.
- 원래 <100이면 GRU 충분/>200이면 LSTM 권장이라는 임계값은 근거가 없어 선택 보장으로 사용하지 않는다.
- 75% 순환 파라미터 비율은 같은 input/hidden/layer/bias·projection 없는 설정의 GRU3/LSTM4개 행렬 블록(후보 상태 포함)에서 나온다. 공통 FC를 포함한 전체 비율은 달라지며 속도·품질의 비율이 아니다.

---

## 참고 자료 (References)

2026-10-04 확인: PyTorch2.14 공식 계약과 설치2.14.1을 구분한다. 아래 API를 기준으로 shape·state·packing을 확인했고 속도/장기 성능 단정은 하지 않는다.

- [LSTM 양방향 hn/output 및 dropout](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html)
- [GRU gate/parameter 계약](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html)
- [RNN input/state](https://docs.pytorch.org/docs/2.14/generated/torch.nn.RNN.html)
- [TimeSeriesSplit 시간순서·gap 조건](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [MSELoss shape/reduction](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MSELoss.html)
- [Matplotlib font 설정](https://matplotlib.org/stable/users/explain/text/text_props.html)
- [clip_grad_norm_·non-finite 거부](https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.clip_grad_norm_.html)


- [PyTorch RNN 공식 문서](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
- [PyTorch LSTM 공식 문서](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [PyTorch GRU 공식 문서](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html)
- [Understanding LSTM Networks (Colah's Blog)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Sequence Models - Andrew Ng (Coursera)](https://www.coursera.org/learn/nlp-sequence-models)
- [Packing 공식 계약·CPU lengths·enforce_sorted](https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.rnn.pack_padded_sequence.html)

---

## 관련 문서

- [상위 폴더](../README.md)
- 데이터 전처리는 입력 시간/결측 계약을 먼저 정의한다. 원래 ../../data-processing/ 링크는 대상이 없어 제거했다.
- 모델 배포는 학습 예제 외 별도 단계다. 원래 ../../deployment/ 링크는 대상이 없어 제거했다.
