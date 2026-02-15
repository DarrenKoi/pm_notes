---
tags: [model-saving, joblib, torch-save, onnx]
level: beginner
last_updated: 2026-02-14
status: in-progress
---

# 모델 저장과 로딩 (Model Saving & Loading)

> 학습된 모델을 직렬화(Serialization)하여 저장하고, 배포/추론 시 다시 불러오는 방법을 정리한다.

---

## 왜 필요한가? (Why)

- **배포(Deployment)**: 학습은 GPU 서버에서 하지만, 추론은 API 서버나 엣지 디바이스에서 수행한다. 학습된 가중치를 파일로 저장해야 다른 환경에서 로딩할 수 있다.
- **재현성(Reproducibility)**: 실험 결과를 재현하려면 모델 가중치 + 하이퍼파라미터 + 전처리 파이프라인을 함께 저장해야 한다.
- **체크포인트(Checkpoint)**: 학습 중 서버 장애가 발생해도 중간 지점부터 재개할 수 있다.
- **모델 공유**: 팀원 간, 또는 학습 서버 → 서빙 서버 간 모델을 전달해야 한다.

---

## 핵심 개념 (What)

### 직렬화 포맷 비교

| 항목 | pickle / joblib | state_dict (PyTorch) | ONNX | safetensors |
|------|----------------|---------------------|------|-------------|
| **대상 프레임워크** | scikit-learn, 일반 Python 객체 | PyTorch | 크로스 플랫폼 | PyTorch, HuggingFace |
| **저장 내용** | 모델 객체 전체 | 가중치(텐서)만 | 계산 그래프 + 가중치 | 가중치(텐서)만 |
| **파일 확장자** | `.pkl`, `.joblib` | `.pt`, `.pth` | `.onnx` | `.safetensors` |
| **보안** | pickle 취약점 있음 | pickle 기반 (취약) | 안전 | 안전 (설계 목표) |
| **추론 속도** | 보통 | 보통 | 최적화 가능 (빠름) | 보통 |
| **크로스 플랫폼** | Python 전용 | PyTorch 전용 | C++, Java, JS 등 지원 | 다중 프레임워크 |
| **용량** | 보통 | 작음 (가중치만) | 보통 | 작음 |

### 핵심 용어

- **Serialization**: Python 객체를 바이트 스트림으로 변환하여 파일에 저장하는 것
- **state_dict**: PyTorch 모델의 학습 가능한 파라미터(가중치, 바이어스)를 담은 딕셔너리
- **ONNX (Open Neural Network Exchange)**: 딥러닝 모델의 표준 교환 포맷. 프레임워크 간 호환성 제공
- **safetensors**: HuggingFace가 만든 안전한 텐서 직렬화 포맷. pickle 보안 문제 없음

---

## 어떻게 사용하는가? (How)

### 1. scikit-learn 모델 저장 (joblib / pickle)

#### joblib 방식 (권장)

```python
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 모델 학습
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 저장
joblib.dump(model, "random_forest_v1.joblib")

# 로딩
loaded_model = joblib.load("random_forest_v1.joblib")
predictions = loaded_model.predict(X_test)
print(f"Accuracy: {loaded_model.score(X_test, y_test):.4f}")
```

#### pickle 방식 (비교용)

```python
import pickle

# 저장
with open("random_forest_v1.pkl", "wb") as f:
    pickle.dump(model, f)

# 로딩
with open("random_forest_v1.pkl", "rb") as f:
    loaded_model = pickle.load(f)
```

> **joblib vs pickle**: joblib은 내부적으로 numpy 배열을 효율적으로 압축하므로, 대용량 모델(특히 앙상블 모델)에서 joblib이 더 빠르고 파일 크기도 작다. sklearn 공식 문서에서도 joblib을 권장한다.

#### 전체 파이프라인 저장

실무에서는 전처리(Scaler, Encoder 등)와 모델을 Pipeline으로 묶어서 통째로 저장하는 것이 안전하다.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 파이프라인 구성
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

# 파이프라인 통째로 저장 → 전처리 + 모델이 함께 저장됨
joblib.dump(pipeline, "pipeline_v1.joblib")

# 로딩 후 바로 raw 데이터로 추론 가능
loaded_pipeline = joblib.load("pipeline_v1.joblib")
predictions = loaded_pipeline.predict(X_test)
```

---

### 2. PyTorch 모델 저장

#### 예제 모델 정의

```python
import torch
import torch.nn as nn

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

model = SimpleClassifier(input_dim=20, hidden_dim=64, output_dim=2)
```

#### 방법 A: state_dict 저장 (권장)

```python
# ── 저장 ──
torch.save(model.state_dict(), "classifier_v1.pt")

# ── 로딩 ──
# 반드시 동일한 모델 클래스를 먼저 정의/임포트해야 한다
loaded_model = SimpleClassifier(input_dim=20, hidden_dim=64, output_dim=2)
loaded_model.load_state_dict(torch.load("classifier_v1.pt", weights_only=True))
loaded_model.eval()  # 추론 모드 전환 (Dropout, BatchNorm 비활성화)

# 추론
with torch.no_grad():
    sample = torch.randn(1, 20)
    output = loaded_model(sample)
    pred = torch.argmax(output, dim=1)
    print(f"Predicted class: {pred.item()}")
```

> **왜 state_dict가 권장인가?**
> - 모델 구조(코드)와 가중치(데이터)를 분리하여 관리할 수 있다
> - Python/PyTorch 버전이 바뀌어도 가중치 로딩이 안정적이다
> - 파일 크기가 더 작다 (클래스 메타데이터 미포함)

#### 방법 B: 전체 모델 저장

```python
# ── 저장 ──
torch.save(model, "classifier_full_v1.pt")

# ── 로딩 ──
# 모델 클래스 정의가 필요 없다 (파일에 포함됨)
loaded_model = torch.load("classifier_full_v1.pt", weights_only=False)
loaded_model.eval()
```

> **주의**: 전체 모델 저장은 pickle을 사용하므로 Python 버전, 모듈 경로가 바뀌면 로딩이 실패할 수 있다. 프로토타이핑에서만 사용하고, 프로덕션에서는 state_dict 방식을 쓰자.

#### 방법 C: 체크포인트 저장 (학습 재개용)

학습 중간 상태를 모두 저장하여 장애 시 이어서 학습할 수 있다.

```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ── 학습 루프 중 체크포인트 저장 ──
def save_checkpoint(model, optimizer, epoch, loss, path):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "model_config": {  # 모델 재생성에 필요한 하이퍼파라미터
            "input_dim": 20,
            "hidden_dim": 64,
            "output_dim": 2,
        }
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: epoch={epoch}, loss={loss:.4f}")

# 예: 매 10 에폭마다 저장
for epoch in range(100):
    # ... 학습 코드 ...
    loss = 0.5  # placeholder
    if (epoch + 1) % 10 == 0:
        save_checkpoint(model, optimizer, epoch, loss, f"checkpoint_epoch{epoch+1}.pt")

# ── 체크포인트에서 학습 재개 ──
def load_checkpoint(path):
    checkpoint = torch.load(path, weights_only=False)
    config = checkpoint["model_config"]

    model = SimpleClassifier(**config)
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    last_loss = checkpoint["loss"]

    print(f"Resumed from epoch={start_epoch}, last_loss={last_loss:.4f}")
    return model, optimizer, start_epoch

model, optimizer, start_epoch = load_checkpoint("checkpoint_epoch50.pt")
```

---

### 3. ONNX 변환 및 추론

ONNX로 변환하면 PyTorch 없이도 `onnxruntime`만으로 추론이 가능하다. 서빙 서버에 PyTorch를 설치하지 않아도 되므로 배포 환경이 가벼워진다.

#### PyTorch → ONNX 변환

```python
import torch

model.eval()

# 더미 입력 (모델의 입력 shape과 동일해야 함)
dummy_input = torch.randn(1, 20)

torch.onnx.export(
    model,
    dummy_input,
    "classifier_v1.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},   # 배치 크기를 동적으로 설정
        "output": {0: "batch_size"},
    },
    opset_version=17,
)
print("ONNX export complete.")
```

#### ONNX Runtime으로 추론

```python
import onnxruntime as ort
import numpy as np

# 세션 생성
session = ort.InferenceSession("classifier_v1.onnx")

# 입력 데이터 준비 (numpy 배열)
input_data = np.random.randn(5, 20).astype(np.float32)  # batch=5

# 추론
outputs = session.run(
    None,  # 모든 출력 노드
    {"input": input_data}
)

logits = outputs[0]
predictions = np.argmax(logits, axis=1)
print(f"Predictions: {predictions}")
```

#### ONNX 모델 검증

```python
import onnx

onnx_model = onnx.load("classifier_v1.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model is valid.")

# 모델 그래프 정보 출력
print(f"IR version: {onnx_model.ir_version}")
print(f"Opset version: {onnx_model.opset_import[0].version}")
for inp in onnx_model.graph.input:
    print(f"Input: {inp.name}, shape={[d.dim_value for d in inp.type.tensor_type.shape.dim]}")
```

---

### 4. 버전 관리 팁

모델 파일은 Git으로 관리하기 어렵다 (바이너리 + 대용량). 아래 패턴을 활용하자.

#### 메타데이터 파일 함께 저장

```python
import json
from datetime import datetime

metadata = {
    "model_name": "simple_classifier",
    "version": "1.0.0",
    "framework": "pytorch",
    "created_at": datetime.now().isoformat(),
    "metrics": {
        "accuracy": 0.95,
        "f1_score": 0.93,
    },
    "training_config": {
        "epochs": 100,
        "learning_rate": 1e-3,
        "batch_size": 32,
    },
    "input_schema": {
        "shape": [None, 20],
        "dtype": "float32",
    },
    "files": {
        "weights": "classifier_v1.pt",
        "onnx": "classifier_v1.onnx",
    }
}

with open("classifier_v1_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
```

#### 디렉토리 구조 패턴

```
models/
├── simple_classifier/
│   ├── v1.0.0/
│   │   ├── model.pt              # state_dict
│   │   ├── model.onnx            # ONNX 변환본
│   │   ├── metadata.json         # 메타데이터
│   │   └── config.json           # 모델 하이퍼파라미터
│   └── v1.1.0/
│       ├── model.pt
│       └── metadata.json
└── pipeline_rf/
    └── v1.0.0/
        ├── pipeline.joblib
        └── metadata.json
```

#### .gitignore 설정

```gitignore
# 모델 바이너리는 Git에서 제외
*.pt
*.pth
*.onnx
*.joblib
*.pkl
*.safetensors

# 메타데이터는 Git에 포함 (버전 추적용)
!**/metadata.json
!**/config.json
```

> **대용량 모델 관리**: DVC(Data Version Control)나 MLflow Artifacts를 사용하면 모델 바이너리도 버전 관리가 가능하다.

---

### 5. 보안 주의사항

#### pickle의 위험성

`pickle`은 역직렬화 시 **임의 코드를 실행**할 수 있다. 신뢰할 수 없는 출처의 `.pkl`, `.pt` 파일을 로딩하면 시스템이 공격받을 수 있다.

```python
# 위험한 예시 - 악의적인 pickle 파일이 코드를 실행할 수 있음
import pickle

class MaliciousPayload:
    def __reduce__(self):
        import os
        return (os.system, ("echo HACKED > /tmp/pwned",))

# 이런 객체가 pickle로 저장되어 있으면, load 시 os.system이 실행됨
```

#### 안전한 로딩 방법

```python
# PyTorch 2.6+ 기본값: weights_only=True (안전)
state_dict = torch.load("model.pt", weights_only=True)

# weights_only=False는 신뢰할 수 있는 파일에만 사용
checkpoint = torch.load("checkpoint.pt", weights_only=False)
```

#### safetensors 사용 (가장 안전)

```python
from safetensors.torch import save_file, load_file

# 저장 (state_dict의 텐서만 저장, 코드 실행 불가)
save_file(model.state_dict(), "classifier_v1.safetensors")

# 로딩
state_dict = load_file("classifier_v1.safetensors")
model.load_state_dict(state_dict)
```

> **실무 권장**: 프로덕션 환경에서는 safetensors 또는 ONNX를 사용하고, pickle 기반 포맷은 신뢰할 수 있는 내부 환경에서만 사용한다.

---

## 참고 자료 (References)

- [PyTorch - Saving and Loading Models](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
- [scikit-learn - Model Persistence](https://scikit-learn.org/stable/model_persistence.html)
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [safetensors GitHub](https://github.com/huggingface/safetensors)
- [DVC (Data Version Control)](https://dvc.org/)

---

## 관련 문서

- [FastAPI 모델 서빙](./fastapi-model-serving.md) - 저장된 모델을 API로 서빙하는 방법
- [상위: ML/DL Deployment](../deployment/)
