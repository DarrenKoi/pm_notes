---
tags: [rnn, lstm, gru, time-series, pytorch]
level: intermediate
last_updated: 2026-02-14
status: in-progress
---

# 시퀀스 모델 (Sequence Models): RNN, LSTM, GRU

> 시계열 및 순차 데이터를 처리하는 딥러닝 모델의 핵심 아키텍처와 PyTorch 구현 가이드

## 왜 필요한가? (Why)

- **제조/반도체 공정에서 시계열 데이터는 핵심**: 장비 센서 로그, 공정 파라미터 변화, 웨이퍼 계측 추이 등 순차적으로 발생하는 데이터가 대부분이다
- **시간적 의존성 포착**: 일반 MLP(Fully Connected)는 입력 간 순서를 무시하지만, 시퀀스 모델은 **이전 시점의 정보를 기억**하며 다음 시점을 예측한다
- **다양한 실무 활용**:
  - 장비 이상 감지(Anomaly Detection): 센서 값 패턴이 정상 범위를 벗어나는 시점 탐지
  - 시계열 예측(Forecasting): 공정 파라미터, 수율 트렌드 예측
  - 시퀀스 분류: 공정 레시피 패턴 분류, 불량 유형 판별
- **왜 RNN 계열인가?**: Transformer가 대세지만, 제조 현장의 센서 데이터처럼 **긴 연속 시계열 + 제한된 GPU 환경**에서는 LSTM/GRU가 여전히 실용적이다

---

## 핵심 개념 (What)

### 1. RNN (Recurrent Neural Network)

가장 기본적인 시퀀스 모델. 각 시간 단계(time step)에서 **이전 hidden state**를 입력과 함께 받아 새로운 hidden state를 생성한다.

```
h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b)
```

**문제점**: 시퀀스가 길어지면 **기울기 소실(Vanishing Gradient)** 문제로 장기 의존성을 학습하지 못한다.

### 2. LSTM (Long Short-Term Memory)

RNN의 기울기 소실 문제를 해결하기 위해 **게이트 메커니즘** 도입:

| 게이트 | 역할 |
|--------|------|
| **Forget Gate** | 이전 셀 상태에서 버릴 정보 결정 |
| **Input Gate** | 새로운 정보 중 저장할 부분 결정 |
| **Output Gate** | 셀 상태에서 출력할 부분 결정 |

- **Cell State(c_t)**: 장기 기억을 전달하는 "컨베이어 벨트" 역할
- **Hidden State(h_t)**: 현재 시점의 출력

### 3. GRU (Gated Recurrent Unit)

LSTM의 간소화 버전. 게이트를 2개로 줄여 **파라미터가 적고 학습이 빠르다**:

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
    """슬라이딩 윈도우 방식으로 시계열 데이터를 (input, target) 쌍으로 변환"""

    def __init__(self, data: np.ndarray, window_size: int, horizon: int = 1):
        """
        Args:
            data: 1D or 2D 시계열 배열 (samples,) or (samples, features)
            window_size: 입력으로 사용할 과거 시점 수
            horizon: 예측할 미래 시점 수
        """
        self.data = torch.FloatTensor(data)
        self.window_size = window_size
        self.horizon = horizon

        # 1D → 2D 변환
        if self.data.dim() == 1:
            self.data = self.data.unsqueeze(-1)  # (samples, 1)

    def __len__(self):
        return len(self.data) - self.window_size - self.horizon + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.window_size]            # (window_size, features)
        y = self.data[idx + self.window_size : idx + self.window_size + self.horizon]  # (horizon, features)
        return x, y.squeeze()  # y를 (horizon,) 또는 스칼라로


# 사용 예시
raw_data = np.random.randn(1000)  # 1000개 시점의 센서 데이터
dataset = TimeSeriesDataset(raw_data, window_size=30, horizon=1)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for x_batch, y_batch in loader:
    print(f"입력: {x_batch.shape}")   # (32, 30, 1)
    print(f"타겟: {y_batch.shape}")   # (32,)
    break
```

---

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
- `batch_first=True`를 항상 명시 (기본값은 `False`이므로 주의)
- `output`은 모든 시점의 hidden state, `hn`은 마지막 시점만
- 예측 태스크에서는 보통 `output[:, -1, :]` (마지막 시점)을 사용

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
- 나머지 인터페이스는 `nn.RNN`과 동일하므로 교체가 쉽다

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

시퀀스를 **앞→뒤, 뒤→앞 양방향**으로 처리하여 더 풍부한 문맥 정보를 얻는다. 시계열 분류(classification)에 유용하지만, **미래→과거 방향이 포함되므로 실시간 예측(forecasting)에는 부적합**하다.

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
- 예측(forecasting)이 아닌 **분류/시퀀스 라벨링** 태스크에 적합하다

---

### 6. 시계열 예측 완전 예제: Sine Wave

합성 사인파 데이터를 생성하고 LSTM으로 학습 후 예측 결과를 시각화하는 end-to-end 예제.

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt


# ========== 1. 합성 데이터 생성 ==========
def generate_sine_data(n_points=2000, noise_std=0.05):
    """노이즈가 섞인 사인파 데이터 생성"""
    t = np.linspace(0, 80 * np.pi, n_points)
    data = np.sin(t) + noise_std * np.random.randn(n_points)
    return data.astype(np.float32)


# ========== 2. Dataset 클래스 ==========
class SineDataset(Dataset):
    def __init__(self, data, window_size=50, horizon=1):
        self.data = torch.FloatTensor(data).unsqueeze(-1)  # (N, 1)
        self.window_size = window_size
        self.horizon = horizon

    def __len__(self):
        return len(self.data) - self.window_size - self.horizon + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.window_size]
        y = self.data[idx + self.window_size : idx + self.window_size + self.horizon].squeeze()
        return x, y


# ========== 3. 모델 정의 ==========
class SineLSTM(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        pred = self.fc(out[:, -1, :])
        return pred.squeeze(-1)


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
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)
            loss = criterion(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # --- Validate ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                pred = model(x_batch)
                loss = criterion(pred, y_batch)
                val_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
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

    # 데이터 생성 및 분할
    data = generate_sine_data(n_points=2000)
    split = int(len(data) * 0.8)
    train_data, test_data = data[:split], data[split:]

    train_ds = SineDataset(train_data, window_size=WINDOW_SIZE)
    test_ds = SineDataset(test_data, window_size=WINDOW_SIZE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 모델 학습
    model = SineLSTM(hidden_size=64, num_layers=2)
    history = train_model(model, train_loader, test_loader, epochs=EPOCHS, device=device)

    # ========== 6. 예측 및 시각화 ==========
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            pred = model(x_batch.to(device)).cpu()
            predictions.extend(pred.numpy())
            actuals.extend(y_batch.numpy())

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

### 7. 시퀀스 패딩 & 팩킹

배치 내 시퀀스 길이가 다를 때 `pack_padded_sequence`와 `pad_packed_sequence`를 사용하면 **패딩된 부분을 무시하고 연산 효율을 높인다**.

```python
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


def collate_variable_length(batch):
    """가변 길이 시퀀스를 패딩하고 길이 정보를 함께 반환하는 collate 함수"""
    sequences, labels = zip(*batch)

    # 실제 길이 기록 (패딩 전)
    lengths = torch.tensor([len(seq) for seq in sequences])

    # 가장 긴 시퀀스 기준으로 패딩
    padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)

    # 길이 내림차순 정렬 (pack_padded_sequence 요구사항)
    sorted_idx = lengths.argsort(descending=True)
    padded = padded[sorted_idx]
    lengths = lengths[sorted_idx]
    labels = torch.stack(labels)[sorted_idx]

    return padded, lengths, labels


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

padded, lengths, labels = collate_variable_length(batch)
print(f"패딩된 배치 shape: {padded.shape}")  # (3, 15, 4) — 최대 길이 15 기준
print(f"실제 길이: {lengths}")               # tensor([15, 10, 7])

model = PackedLSTM(input_size=4, hidden_size=32, output_size=2)
output = model(padded, lengths)
print(f"출력 shape: {output.shape}")  # (3, 2)
```

**언제 팩킹이 필요한가?**
- 배치 내 시퀀스 길이가 **크게 다를 때** (예: 장비마다 공정 시간이 다른 경우)
- 패딩만 사용하면 0으로 채워진 부분도 연산하므로 **불필요한 계산 발생**
- 팩킹을 쓰면 실제 데이터 길이만큼만 연산하여 **속도와 정확도** 모두 개선

---

### 8. 모델 비교: RNN vs LSTM vs GRU

| 항목 | RNN | LSTM | GRU |
|------|-----|------|-----|
| **게이트 수** | 없음 | 3개 (forget, input, output) | 2개 (reset, update) |
| **상태** | hidden state만 | hidden + cell state | hidden state만 |
| **파라미터 수** (hidden=128 기준) | 가장 적음 | 가장 많음 | LSTM의 ~75% |
| **학습 속도** | 빠름 | 느림 | 중간 |
| **메모리 사용** | 적음 | 많음 | 중간 |
| **장기 의존성** | 매우 약함 | 강함 | 강함 (LSTM과 유사) |
| **기울기 소실** | 심각 | 게이트로 해결 | 게이트로 해결 |
| **추천 상황** | 짧은 시퀀스, 빠른 프로토타입 | 긴 시퀀스, 복잡한 패턴 | LSTM과 비슷한 성능이 필요하지만 리소스가 제한될 때 |

**실무 선택 가이드**:
- 우선 **GRU로 시작** → 성능이 부족하면 LSTM으로 교체
- 시퀀스 길이 < 100이고 단순한 패턴: GRU로 충분
- 시퀀스 길이 > 200이고 복잡한 장기 의존성: LSTM 권장
- GPU 메모리가 제한적인 제조 환경: GRU 우선 고려

---

## 참고 자료 (References)

- [PyTorch RNN 공식 문서](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
- [PyTorch LSTM 공식 문서](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [PyTorch GRU 공식 문서](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html)
- [Understanding LSTM Networks (Colah's Blog)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Sequence Models - Andrew Ng (Coursera)](https://www.coursera.org/learn/nlp-sequence-models)
- [Pack Padded Sequence 설명 (Stack Overflow)](https://stackoverflow.com/questions/51030782)

---

## 관련 문서

- [상위 폴더](../README.md)
- [데이터 전처리](../../data-processing/)
- [모델 배포](../../deployment/)
