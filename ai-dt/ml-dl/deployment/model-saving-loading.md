---
tags: [model-saving, joblib, torch-save, onnx]
level: beginner
last_updated: 2026-02-14
reviewed_on: 2026-10-04
review_status: partial
document_type: learning_note
---

# 모델 저장과 로딩 (Model Saving & Loading)

> 학습 결과를 파일로 옮기는 과정과 추론용 가중치·재개용 checkpoint·교환용 그래프의 차이를 정리한다.

> [!info] 실행 순서와 확인 범위
> 1절의 학습→pickle 비교→Pipeline을 순서대로 실행한다. 2절부터 새 SimpleClassifier 변수로 바뀌며 A/B/C→3절→4·5절이 앞 정의를 사용한다. SimpleClassifier는 구조/직렬화 학습용 랜덤 모델이며 높은 분류 성능을 보여주는 예제가 아니다. 파일은 작업용 디렉토리에 생성한다. PyTorch2.14·scikit-learn1.9.1·joblib1.6.0 공식 계약을2026-10-04 확인했다. 실제 CPU 판본과 ONNX/safetensors 검증 조건은 아래 근거에 남겼다. 장치·판본 변경/운영 배포·Claude 협의·Obsidian 읽기 화면은 미확인이다.

---

## 왜 필요한가? (Why)

- **배포(Deployment)**: 학습은 GPU 서버에서 하지만, 추론은 API 서버나 엣지 디바이스에서 수행한다. 학습된 가중치를 파일로 저장해야 다른 환경에서 로딩할 수 있다.
- **재현성(Reproducibility)**: 실험 결과를 재현하려면 모델 가중치·구조/하이퍼파라미터·전처리·데이터/분할 식별자·의존성 판본과 실행 조건을 기록해야 한다. 이것만으로 RNG/worker/장치까지 동일한 replay를 보장하지 않는다.
- **체크포인트(Checkpoint)**: 필요한 model/optimizer/scheduler/RNG 등의 상태를 실제 저장했을 때 지원 범위 안에서 재개할 수 있다. 아래 최소 checkpoint는 모든 상태를 저장하지 않는다.
- **모델 공유**: 팀원 간, 또는 학습 서버 → 서빙 서버 간 모델을 전달해야 한다.

---

## 핵심 개념 (What)

### 직렬화 포맷 비교

| 항목 | pickle / joblib | state_dict (PyTorch) | ONNX | safetensors |
|------|----------------|---------------------|------|-------------|
| **대상 프레임워크** | scikit-learn, 일반 Python 객체 | PyTorch | 크로스 플랫폼 | PyTorch, HuggingFace |
| **저장 내용** | 모델/전처리 객체 | 파라미터 + persistent buffer 등 | 계산 그래프 + 가중치; 외부 파일 가능 | dense tensor + 선택 metadata |
| **파일 확장자** | `.pkl`, `.joblib` | `.pt`, `.pth` | `.onnx` | `.safetensors` |
| **보안** | 신뢰한 파일만 load | weights_only 제한 로더; 완전 안전 아님 | parser/runtime·자원/외부 참조 검증 필요 | pickle 코드 실행을 피하는 설계; 자원/출처 검증 필요 |
| **추론 속도** | 포맷 자체가 속도 보장 안 함 | 실행 모델/장치에 의존 | EP/연산 지원·최적화별 측정 | 실행 모델/장치에 의존 |
| **크로스 플랫폼** | Python 전용 | PyTorch 전용 | C++, Java, JS 등 지원 | 다중 프레임워크 |
| **용량** | 압축/배열/객체에 의존 | dtype/텐서/공유에 의존 | dtype/graph/외부 파일에 의존 | dtype/텐서에 의존 |

### 핵심 용어

- **Serialization**: Python 객체를 바이트 스트림으로 변환하여 파일에 저장하는 것
- **state_dict**: PyTorch 모델의 파라미터와 persistent buffer(예: BatchNorm running statistics)를 담은 매핑; 모델 구조·전처리·optimizer는 별도
- **ONNX (Open Neural Network Exchange)**: 딥러닝 모델의 표준 교환 포맷. 프레임워크 간 호환성 제공
- **safetensors**: HuggingFace가 만든 안전한 텐서 직렬화 포맷. pickle의 임의 객체 역직렬화 코드 실행 경로를 피하는 설계; 모델 출처·자원·후속 실행까지 안전을 보장하지 않음

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

> **joblib vs pickle**: joblib은 NumPy 배열을 다루는 pickle 기반 대안이며 기본 dump compress=0은 무압축이다. 압축을 선택하면 파일 크기/CPU·로딩 시간/mmap 조건이 달라진다. 어느 쪽이 더 빠르고 작은지는 측정한다. 공식 scikit-learn 문서는 목적에 따라 joblib/pickle/cloudpickle/skops/ONNX를 비교하며 하나를 무조건 권장하지 않는다. 신뢰한 파일과 학습 때의 의존성 판본을 사용한다; 다른 sklearn 판본 로딩은 지원하지 않는다.

#### 전체 파이프라인 저장

실무에서는 전처리(Scaler, Encoder 등)와 모델을 Pipeline으로 묶어서 묶어 저장하면 학습/추론 전처리 불일치를 줄인다. 직렬화 보안과는 별개이며 입력 컬럼·순서·dtype·unknown 처리/커스텀 transformer의 import 경로도 계약으로 보존한다.

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
        self.config = {"input_dim": input_dim, "hidden_dim": hidden_dim, "output_dim": output_dim}
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
loaded_model.load_state_dict(torch.load("classifier_v1.pt", map_location="cpu", weights_only=True))
loaded_model.eval()  # 추론 모드 전환 (Dropout을 끄고 BatchNorm은 저장된 running stats를 사용; grad 비활성화는 별도)

# 추론
with torch.no_grad():
    sample = torch.randn(1, 20)
    output = loaded_model(sample)
    pred = torch.argmax(output, dim=1)
    print(f"Predicted class: {pred.item()}")
```

> **왜 state_dict가 권장인가?**
> - 모델 구조(코드)와 가중치(데이터)를 분리하여 관리할 수 있다
> - 호환되는 구조·키/shape·dtype·의존성 판본을 확인한다. 판본 변경 로딩을 무조건 보장하지 않는다
> - 구조 객체의 pickle 의존성을 줄인다. 파일 크기는 텐서·공유/압축 조건에 따라 측정한다

#### 방법 B: 전체 모델 저장

```python
# ── 저장 ──
torch.save(model, "classifier_full_v1.pt")

# ── 로딩 ──
# 동일 클래스의 import 가능한 모듈 경로/정의가 필요; 소스가 통째로 포함되지 않음
loaded_model = torch.load("classifier_full_v1.pt", map_location="cpu", weights_only=False)
loaded_model.eval()
```

> **주의**: 전체 모델 저장은 pickle을 사용하므로 Python 버전, 모듈 경로가 바뀌면 로딩이 실패할 수 있다. 프로토타이핑에서만 사용하고, 프로덕션에서는 state_dict 방식을 쓰자.

#### 방법 C: 체크포인트 저장 (학습 재개용)

model/Adam optimizer/완료 epoch/loss/config만 저장하는 최소 예제다. 아래100epoch 루프의 loss=0.5는 placeholder이며 실제 학습·재개 결과가 아니다. scheduler·early stopping·RNG·sampler/worker·AMP scaler·원자적 파일 저장은 구현하지 않아 동일 replay/장애 무손실 복구를 보장하지 않는다. 이 조각은 전체 모델 파라미터를 같은 순서의 단일 group으로 전달한 Adam만 재생성한다. 여러 group·일부 파라미터·다른 optimizer/모델 클래스의 재생성은 별도 계약이 필요하다.

```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ── 학습 루프 중 체크포인트 저장 ──
def save_checkpoint(model, optimizer, epoch, loss, path):
    if not isinstance(optimizer, optim.Adam):
        raise ValueError("이 예제는 Adam 재생성만 지원")
    if len(optimizer.param_groups) != 1 or [id(p) for p in optimizer.param_groups[0]["params"]] != [id(p) for p in model.parameters()]:
        raise ValueError("전체 모델 파라미터와 같은 순서의 단일 optimizer group만 지원")
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "model_config": dict(model.config),  # 실제 모델 구조와 일치
        "optimizer_type": "Adam",
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
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["optimizer_type"] != "Adam":
        raise ValueError("지원하지 않는 optimizer")
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

지원 연산/입력 dtype·shape/opset·IR·Execution Provider(EP)가 맞으면 PyTorch 없이 ONNX Runtime에서 실행할 수 있다. 커스텀 연산/지원하지 않는 연산은 별도 변환·runtime 구현이 필요하다. 패키지 용량·속도는 배포 환경에서 측정한다. 아래는 작은 Linear/ReLU/Dropout(eval) 모델의 CPU 예시다. 변환에는 onnx/onnxscript, 비교에는 onnxruntime가 필요하다. PyTorch2.14의 dynamo=True exporter에서는 dynamic_shapes를 사용한다; 원래 dynamic_axes/opset17 조각은 legacy 조건과 혼합하지 않는다. 이 예제는 opset18/단일 파일 저장을 명시한다; 큰 가중치의 external data는 모든 관련 파일을 함께 전달해야 한다.

#### PyTorch → ONNX 변환

```python
import torch

model.eval()

# 더미 입력 (모델의 입력 shape과 동일해야 함)
dummy_input = torch.randn(2, 20)  # 1은 shape 특수화될 수 있어 예시 batch2
batch_dim = torch.export.Dim("batch_size", min=1, max=128)

torch.onnx.export(
    model,
    dummy_input,
    "classifier_v1.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamo=True,
    dynamic_shapes={"x": {0: batch_dim}},  # forward 인자명 x; ONNX input 이름과 구분
    opset_version=18,
    external_data=False,  # 이 작은 모델은 단일 파일; >2GB 가중치는 별도 조건 필요
)
print("ONNX export complete.")
```

#### ONNX Runtime으로 추론

```python
import onnxruntime as ort
import numpy as np

# 세션 생성
session = ort.InferenceSession("classifier_v1.onnx", providers=["CPUExecutionProvider"])

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

# checker 통과와 수치 동등성은 다른 검사
with torch.no_grad():
    torch_logits = model(torch.from_numpy(input_data)).cpu().numpy()
np.testing.assert_allclose(logits, torch_logits, rtol=1e-4, atol=1e-5)
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
    print(f"Input: {inp.name}, shape={[d.dim_param or d.dim_value for d in inp.type.tensor_type.shape.dim]}")
```

---

onnx.checker는 모델 형식 검증이며 정확도·속도·입력 범위/도메인 적합성·파일 안전성을 보장하지 않는다. ONNX Runtime가 symbolic batch의 min/max 계약을 자동 검사한다고 가정하지 말고 서빙 입력에서 범위1~128/feature20/float32를 검사한다.

### 4. 버전 관리 팁

모델 파일은 Git으로 관리하기 어렵다 (바이너리 + 대용량). 아래 패턴을 활용하자.

#### 메타데이터 파일 함께 저장

```python
import json
from datetime import datetime, timezone
import hashlib

with open("classifier_v1.pt", "rb") as weights_file:
    weights_sha256 = hashlib.sha256(weights_file.read()).hexdigest()

metadata = {
    "model_name": "simple_classifier",
    "version": "1.0.0",
    "framework": "pytorch",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "framework_version": str(torch.__version__),
    "model_config": dict(model.config),
    "metrics_status": "unmeasured_example",
    "metrics": {
        "accuracy": None,
        "f1_score": None,
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
    "output_schema": {"shape": [None, 2], "class_ids": [0, 1]},
    "weights_sha256": weights_sha256,
    "files": {
        "weights": "classifier_v1.pt",
        "onnx": "classifier_v1.onnx",
    }
}

with open("classifier_v1_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
```

원래 metadata의 accuracy0.95/F1=0.93은 미측정 예시였으므로 현재는 null로 표시했다. epochs100도 설정 예시이며 위 placeholder를 실제 학습 이력으로 취급하지 않는다. hash는 내용 동일성 검사로 출처 신뢰를 증명하지 않는다. 전처리·라벨 사전·데이터/코드/환경 식별자는 실제 프로젝트에서 추가해야 한다.

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

위 패턴은 모델 확장자를 제외하며 JSON은 원래 제외 대상이 아니다. ! 규칙은 이미 무시된 부모 디렉토리를 자동 복구하지 않는다. metadata/config에 비밀이나 내부 endpoint가 있으면 Git에 넣지 않는다.

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
        return (print, ("역직렬화가 callable을 실행하는 무해한 시연",))

# load가 callable을 실행할 수 있음을 보여준다. 원래 shell/file 생성 payload 대신 print 사용
```

#### 안전한 로딩 방법

```python
# PyTorch2.6+: pickle_module을 지정하지 않을 때 weights_only=True가 기본
# 허용 객체 제한으로 공격 표면 감소; DoS/메모리/후속 실행까지 안전 보장은 아님
state_dict = torch.load("classifier_v1.pt", map_location="cpu", weights_only=True)

# 이 최소 checkpoint는 tensor/basic 타입으로 구성돼 weights_only=True로 로딩
checkpoint = torch.load("checkpoint_epoch50.pt", map_location="cpu", weights_only=True)
```

#### safetensors 사용 (텐서 데이터 분리)

```python
from safetensors.torch import save_file, load_file

# 이 모델은 dense/contiguous·공유 없는 텐서; tied/shared weights는 별도 API 조건 확인
save_file(model.state_dict(), "classifier_v1.safetensors")

# 로딩
state_dict = load_file("classifier_v1.safetensors")
model.load_state_dict(state_dict)
model.eval()
```

> **선택 조건**: Python 객체/전처리가 필요하면 신뢰한 동일 환경의 joblib, 텐서만 옮기면 state_dict 제한 로더 또는 safetensors, runtime 교환이 필요하면 검증한 ONNX를 검토한다. 어느 포맷도 출처·의존성·입력/자원 조건을 자동 보증하지 않는다. 내부 파일이라는 이유만으로 신뢰하지 않는다.

---

## 참고 자료 (References)

2026-10-04 공식 자료 대조. PyTorch2.14·scikit-learn1.9.1·joblib1.6.0 문서 판본과 로컬 Python3.14.2/torch2.14.1·onnx1.23.1/onnxscript0.7.2/onnxruntime1.30.0/safetensors0.8.0 실행 판본을 구분한다. 판본 간 호환과 운영 성능은 미검증이다. safetensors의 main 문서는 설치 판본 번호의 근거가 아니다.

- [PyTorch 저장/복원 튜토리얼](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html): state_dict/전체 pickle/재개 상태 구분.
- [PyTorch2.14 serialization](https://docs.pytorch.org/docs/2.14/notes/serialization.html): weights_only 기본 조건과 남은 위험.
- [PyTorch2.14 Module](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html): persistent buffer/state_dict/eval 계약.
- [scikit-learn1.9.1 persistence](https://scikit-learn.org/stable/model_persistence.html): pickle 계열/판본 호환·ONNX/skops 목적 비교.
- [joblib1.6.0 dump](https://joblib.readthedocs.io/en/stable/generated/joblib.dump.html): compress=0 기본·압축/mmap tradeoff.
- [PyTorch2.14 ONNX](https://docs.pytorch.org/docs/2.14/onnx.html): dynamo/dynamic_shapes·opset/external_data.
- [ONNX Runtime Python](https://onnxruntime.ai/docs/get-started/with-python.html): InferenceSession/EP 실행.
- [ONNX Runtime 검증 책임](https://onnxruntime.ai/docs/): 악의적 모델의 자원 소모와 정확도/성능 검증 책임.
- [safetensors Torch API](https://huggingface.co/docs/safetensors/api/torch): dense/contiguous 텐서 저장/로딩 조건.
- [DVC 시작 안내](https://doc.dvc.org/start): 데이터/아티팩트 버전 관리. 여기서 실제 DVC 사용을 검증하지 않았다.
- [MLflow tracking](https://mlflow.org/docs/latest/ml/tracking/): artifact 저장 역할. 서버/registry 검증은 별도다.

---

## 관련 문서

- [FastAPI 모델 서빙](./fastapi-model-serving.md): 저장된 모델을 API로 서빙; 이 문서와 목적 차이.
- [배포 읽기 순서](./README.md): 저장→서빙→실험 기록.
