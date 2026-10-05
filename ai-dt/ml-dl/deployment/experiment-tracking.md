---
tags: [mlflow, experiment-tracking, logging]
level: intermediate
last_updated: 2026-02-14
reviewed_on: 2026-10-04
review_status: partial
document_type: learning_note
category_major: "AI·DT"
category_middle: "머신러닝·딥러닝"
category_minor: "모델 저장·배포"
note_kind: "학습"
classified_on: "2026-10-05"
---

# 실험 추적(Experiment Tracking)

> 실험의 설정·평가·모델 파일을 연결해 비교 근거를 남긴다. 추적 기록이 재현성·품질·배포 승인을 자동 보장하는 것은 아니다.

> [!info] 읽기와 실행 순서
> CSV1절과 JSONL8절은 독립 가벼운 로거다. CSV는 평면 표/정렬, JSONL은 run별 중첩 데이터·artifact 복사를 다루므로 같은 설명의 완전 중복이 아니다. 둘 다 단일 writer 교육용이며 동시성/원자적 저장/감사 보장을 구현하지 않는다. MLflow2~6절은 같은 작업용 디렉토리에서 순차 실행하고6절은3절의 학습 모델을 사용한다. fraud 이름은 원래 API 예시 이름이며 Iris 실습을 사기 탐지 성능으로 해석하지 않는다. 7절은 별도 end-to-end 예제다. MLflow3.16.1 API와2026-10-04 공식 자료를 확인했다. 로컬 SQL/CPU 확인과 원격 서버·UI·업무 승인은 구분한다.

## 왜 필요한가? (Why)

- **재현 불가능 문제**: 기록 없이 "지난주에 잘 됐던 모델"의 데이터·코드·환경·seed를 확인하기 어렵다. 어떤 하이퍼파라미터 조합이었는지, 어떤 데이터 전처리를 적용했는지 기억에 의존하게 된다.
- **비교 불가능 문제**: 10번의 실험을 돌렸는데, 어떤 실험이 가장 좋았는지 정리가 안 되면 시간 낭비가 된다.
- **팀 협업**: 동료에게 "이 모델 어떻게 학습시켰어?"라고 물었을 때, 코드 diff만으로는 파악이 어렵다. 추적 로그에 실제 필요한 설정·분할·코드/환경 식별자를 남겼을 때 확인에 도움이 된다.
- **모델 거버넌스**: 운영 배포된 모델이 어떤 조건에서 학습되었는지 감사 추적(audit trail)이 필요하다.

---

## 핵심 개념 (What)

| 개념 | 설명 | 예시 |
|------|------|------|
| **Experiment** | 관련 실험들을 묶는 최상위 단위 | `fraud-detection-v2` |
| **Run** | 하나의 학습 실행 단위 | 특정 하이퍼파라미터 조합으로 1회 학습 |
| **Parameters** | 학습에 사용된 설정값 (입력) | `learning_rate=0.01`, `n_estimators=100` |
| **Metrics** | 학습 결과 성능 지표 (출력) | `accuracy=0.95`, `f1_score=0.87` |
| **Artifacts** | 학습 과정에서 생성된 파일 | 모델 파일, 혼동 행렬 이미지, 피처 중요도 CSV |

### MLflow 아키텍처 요약

```
┌─────────────────────────────────────────────┐
│                 MLflow Server                │
│  ┌─────────────┐  ┌──────────────────────┐  │
│  │ Tracking     │  │ Model Registry       │  │
│  │ - Experiments│  │ - Registered Models  │  │
│  │ - Runs       │  │ - Versions           │  │
│  │ - Params     │  │ - Aliases / Tags     │  │
│  │ - Metrics    │  │   (배포 정책은 별도)  │  │
│  │ - Artifacts  │  │                      │  │
│  └─────────────┘  └──────────────────────┘  │
│                                             │
│  ┌─────────────────────────────────────┐    │
│  │ Artifact Store (local / S3 / GCS)   │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

---

## 어떻게 사용하는가? (How)

추적 backend는 run metadata를 저장하고 artifact store는 파일을 저장한다. local SQL 예제에서는 서버 없이 API를 사용할 수 있다. 원격 server/auth/network/artifact 권한은 별도 검증한다. Model Stage는2.9.0부터 deprecated이며 alias/tag가 실제 승인 절차를 대신하지 않는다.

### 1. 간단한 CSV 로깅: DIY 실험 추적

CSV 평면 표를 직접 남기는 단일 writer 방법이다. 전체 파일을 재작성하므로 동시 writer는 서로의 행을 잃을 수 있다. params는 평면 scalar 예시, 기록된0.92/0.95 등은 API 사용을 보여주는 가상 값이다. 비교 방향(loss는 작을수록 좋음)·같은 분할/태스크를 지정하며 미관측 metric을0으로 채우지 않는다.

```python
import pandas as pd
from datetime import datetime, timezone
import numpy as np
from pathlib import Path


class SimpleExperimentTracker:
    """CSV 기반 간단한 실험 추적기"""

    def __init__(self, log_path: str = "experiments.csv"):
        self.log_path = Path(log_path)
        if self.log_path.exists():
            self.df = pd.read_csv(self.log_path)
        else:
            self.df = pd.DataFrame()

    def log_run(
        self,
        experiment_name: str,
        params: dict,
        metrics: dict,
        notes: str = "",
    ) -> None:
        """하나의 실험 실행을 기록한다."""
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "experiment": experiment_name,
            "notes": notes,
        }
        # params와 metrics를 prefix 붙여서 컬럼으로 저장
        for k, v in params.items():
            row[f"param_{k}"] = v
        for k, v in metrics.items():
            row[f"metric_{k}"] = v

        new_row = pd.DataFrame([row])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(self.log_path, index=False, encoding="utf-8")
        print(f"[logged] {experiment_name} | acc={metrics.get('accuracy', 'N/A')}")

    def best_run(self, metric_col: str = "metric_accuracy", greater_is_better: bool = True,
                 experiment_name: str | None = None) -> pd.Series:
        """비교 가능한 run에서 방향을 지정해 선택; missing은0으로 바꾸지 않는다."""
        df = self.df
        if experiment_name is not None and not df.empty:
            df = df.loc[df["experiment"] == experiment_name]
        if df.empty or metric_col not in df:
            raise ValueError("비교할 run/metric 없음")
        values = pd.to_numeric(df[metric_col], errors="raise")
        if values.notna().sum() == 0 or not np.isfinite(values.dropna()).all():
            raise ValueError("관측된 finite metric 필요")
        index = values.idxmax() if greater_is_better else values.idxmin()
        return df.loc[index].copy()

    def summary(self) -> pd.DataFrame:
        return self.df.copy() if self.df.empty else self.df.sort_values("timestamp", ascending=False)


# === 사용 예시 ===
tracker = SimpleExperimentTracker("my_experiments.csv")

tracker.log_run(
    experiment_name="rf-baseline",
    params={"n_estimators": 100, "max_depth": 10, "random_state": 42},
    metrics={"accuracy": 0.92, "f1_score": 0.89},
    notes="Random Forest 베이스라인",
)

tracker.log_run(
    experiment_name="rf-tuned",
    params={"n_estimators": 300, "max_depth": 20, "random_state": 42},
    metrics={"accuracy": 0.95, "f1_score": 0.93},
    notes="하이퍼파라미터 튜닝 후",
)

print(tracker.best_run())
print(tracker.summary())
```

---

### 2. MLflow 기본 설정

```bash
# 설치
pip install mlflow==3.16.1

# 버전 확인
mlflow --version
```

```python
import mlflow
from pathlib import Path

# 기본 backend를 추측하지 않고 로컬 SQL 위치를 명시
tracking_dir = Path("./mlflow-demo").resolve()
tracking_dir.mkdir(exist_ok=True)
mlflow.set_tracking_uri(f"sqlite:///{tracking_dir / 'tracking.db'}")

# 팀 서버는 승인된 URL/인증/파일 권한을 별도로 구성; 실제 연결 미검증
experiment_name = "fraud-detection-v2"  # 원래 API 예시 이름; 아래 학습은 Iris
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    mlflow.create_experiment(experiment_name,
                             artifact_location=(tracking_dir / "artifacts").as_uri())
mlflow.set_experiment(experiment_name)
experiment = mlflow.get_experiment_by_name(experiment_name)
print(f"Experiment ID: {experiment.experiment_id}")
print(f"Artifact Location: {experiment.artifact_location}")
```

---

### 3. MLflow 자동 로깅(Autolog)

`autolog()`를 호출하면 해당 프레임워크의 학습 과정을 **자동으로** 추적한다. 기록 항목은 integration/설치 판본/설정에 따라 달라진다. 모든 데이터·환경·학습 코드를 자동 기록한다고 가정하지 않는다. 공식 sklearn autolog 지원 범위는1.5.2~1.9.0이며 로컬1.9.1은 범위 밖이다. 아래는 unsupported 판본에서 비활성화하도록 명시한다. 별도 수동 로깅은 계속 가능하다.

#### scikit-learn 자동 로깅

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 호환 범위 밖이면 autolog를 끈다; 학습 자체는 계속 실행
mlflow.sklearn.autolog(disable_for_unsupported_versions=True)

# 데이터 준비
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 학습 실행; 자동 기록 여부/키는 실제 run에서 확인
with mlflow.start_run(run_name="rf-autolog-demo") as run:
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # autolog가 자동으로 기록하는 것들:
    # - Parameters: n_estimators, max_depth, criterion, ...
    # - Metrics: training_accuracy, training_f1_score, ...
    # - Artifacts: 설정/지원 범위에 따른 모델 등; 피처 중요도 plot 자동 보장은 없음

# 이후 수동 예제에 monkey-patch 기록이 섞이지 않도록 종료
mlflow.sklearn.autolog(disable=True)
```

#### PyTorch 자동 로깅

```python
import importlib.util
import mlflow.pytorch

# Lightning Trainer.fit integration 예시; 일반 optimizer loop 자동 추적이 아님
if importlib.util.find_spec("pytorch_lightning") is None:
    print("미확인: Lightning 미설치; autolog/실제 학습을 실행하지 않음")
else:
    mlflow.pytorch.autolog()
    # 호환 판본에서 별도의 Trainer.fit 실행이 필요; 이 조각에는 학습이 없음
    mlflow.pytorch.autolog(disable=True)
```

> **선택 조건**: autolog는 지원 integration/상태를 확인하고, 기록 항목을 직접 통제하려면 수동 로깅을 사용한다. PyTorch autolog 호출만으로 아래 수동 PyTorch 학습 루프가 추적된다고 주장하지 않는다. 운영 방식의 선택/Lightning 호환은 미확인이다.

---

### 4. MLflow 수동 로깅

이 조각은 API 호출 예시라 실제 모델을 학습하지 않는다. accuracy/F1/precision/recall·fake_loss와 RF/Adam 등의 설정 값은 서로 하나의 학습을 증명하는 기록이 아니다. synthetic_api_demo 태그를 남기고 업무 성능/승인과 비교하지 않는다. fixed /tmp/config.json을 덮어쓰지 않도록 임시 디렉토리를 사용한다.

```python
import mlflow
import json
from pathlib import Path
from tempfile import TemporaryDirectory

mlflow.set_experiment("manual-logging-demo")

with mlflow.start_run(run_name="manual-run-001") as run:

    mlflow.set_tag("metrics_source", "synthetic_api_demo")
    # === Parameters: 학습 설정 기록 ===
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 15)
    mlflow.log_param("feature_selection", "top_20_by_importance")

    # 여러 파라미터를 한 번에 기록
    mlflow.log_params({
        "learning_rate": 0.01,
        "batch_size": 64,
        "optimizer": "adam",
    })

    # === Metrics: 성능 지표 기록 ===
    mlflow.log_metric("accuracy", 0.94)
    mlflow.log_metric("f1_score", 0.91)
    mlflow.log_metric("precision", 0.93)
    mlflow.log_metric("recall", 0.89)

    # step별 메트릭 기록 (epoch별 loss 추적 등)
    for epoch in range(10):
        fake_loss = 1.0 / (epoch + 1)
        mlflow.log_metric("train_loss", fake_loss, step=epoch)

    # === Artifacts: 파일 기록 ===
    # 단일 파일 저장
    config = {"preprocessing": "standard_scaler", "feature_count": 20}
    with TemporaryDirectory(prefix="mlflow-config-") as tmp:
        config_path = Path(tmp) / "config.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(config_path))

    # 디렉토리 통째로 저장
    # mlflow.log_artifacts("/path/to/output_dir", artifact_path="outputs")

    # === Model: 모델 저장 ===
    # sklearn 모델 저장 (모델 객체가 있다면)
    # mlflow.sklearn.log_model(model, name="model")

    # === Tags: 메타데이터 ===
    mlflow.set_tag("developer", "daeyoung")
    mlflow.set_tag("purpose", "baseline_comparison")

    print(f"Run ID: {run.info.run_id}")
```

---

### 5. MLflow UI: 실험 시각화 및 비교

```bash
# MLflow UI 실행 (기본 포트 5000)
mlflow ui --backend-store-uri sqlite:///./mlflow-demo/tracking.db --host 127.0.0.1

# 포트 지정
mlflow ui --backend-store-uri sqlite:///./mlflow-demo/tracking.db --host 127.0.0.1 --port 8080

# 특정 backend store 지정
mlflow ui --backend-store-uri sqlite:///./mlflow-demo/tracking.db --host 127.0.0.1
```

2절과 같은 작업 디렉토리의 DB를 선택한다. 실제 UI/socket/팀 서버는 이 로컬 API 검증에 포함하지 않았다. UI 배치/버튼 이름은 판본에 따라 달라질 수 있다. `http://localhost:5000`에서 확인할 기능:

- **Experiment 목록**: 좌측 패널에서 experiment 선택
- **Run 비교**: 여러 run을 체크박스로 선택 후 "Compare" 클릭
- **메트릭 차트**: step별 메트릭 변화를 시각화
- **파라미터 vs 메트릭**: Parallel Coordinates Plot으로 최적 조합 탐색

#### 프로그래밍 방식으로 실험 비교

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# 특정 experiment의 상위5개만 조회; 모든 run/모든 페이지가 아님
experiment = client.get_experiment_by_name("fraud-detection-v2")
if experiment is None:
    raise ValueError("2절 experiment 설정 필요")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"],
    max_results=5,
)

print("=== Top 5 Runs by Accuracy ===")
for run in runs:
    params = run.data.params
    metrics = run.data.metrics
    accuracy = metrics.get("accuracy")
    accuracy_text = f"{accuracy:.4f}" if accuracy is not None else "미관측"
    print(
        f"  Run {run.info.run_id[:8]} | "
        f"n_estimators={params.get('n_estimators', 'N/A')} | "
        f"accuracy={accuracy_text}"
    )

# 특정 조건으로 필터링
filtered_runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="metrics.accuracy > 0.9 AND params.model_type = 'RandomForest'",
    order_by=["metrics.f1_score DESC"],
)
```

---

### 6. 모델 레지스트리(Model Registry)

로깅한 모델 URI를 등록해 반환된 version을 사용한다. 아래는3절의 Iris RF 모델을 iris-demo 이름으로 로컬 등록하는 API 예시이며 사기 탐지/배포 승인이 아니다. 추적 backend와 registry 기능/권한을 확인한다. alias는 가리키는 version이 바뀔 수 있으며 이미 메모리에 로딩한 모델이 자동 교체되지는 않는다.

```python
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

# 먼저 실제 학습된3절 model을 기록; artifact가 없는 run을 등록하지 않음
with mlflow.start_run(run_name="iris-registry-demo") as run:
    logged = mlflow.sklearn.log_model(model, name="model",
                                     signature=infer_signature(X_train, model.predict(X_train)),
                                     skops_trusted_types=["sklearn.tree._tree.Tree"])
    # 같은 호출에 registered_model_name="iris-demo"를 주는 직접 등록도 가능

# Tree 타입은 이 process에서 직접 학습한 모델만 신뢰해 허용; 외부 파일에 자동 적용 금지
model_uri = logged.model_uri  # MLflow3 logged model URI; run 경로를 추측하지 않음
result = mlflow.register_model(model_uri, "iris-demo")
print(f"Model Name: {result.name}")
print(f"Version: {result.version}")
client = MlflowClient()
client.update_model_version(name="iris-demo", version=result.version,
                            description="Iris API fixture; 업무 승인/성능 보장 아님")
client.set_model_version_tag(name="iris-demo", version=result.version,
                             key="validation_status", value="demo_unreviewed")
model = mlflow.sklearn.load_model(f"models:/iris-demo/{result.version}")
```

> **Stage 이력**:2.9.0부터 Model Stage가 deprecated다. 아래 alias/tag는 버전 선택 API일 뿐 승인/트래픽 전환을 자동 구현하지 않는다. 예전 models:/.../latest 대신 반환 version이나 목적이 명시된 alias를 사용한다.

```python
# 두 번째 version도 실제 생성: 같은 artifact의 등록 API 시연이며 모델 우열 비교 아님
challenger = mlflow.register_model(model_uri, "iris-demo")
client.set_registered_model_alias("iris-demo", "champion", result.version)
client.set_registered_model_alias("iris-demo", "challenger", challenger.version)
champion_model = mlflow.sklearn.load_model("models:/iris-demo@champion")
print(client.get_model_version_by_alias("iris-demo", "champion").version)
```

---

### 7. 실전 통합 예제: sklearn + MLflow End-to-End

원래 세 모델·100/300/200 estimators·CV5 설정은 보존한다. CV 전 전체 train에 fit한 Scaler를 제거하고 Pipeline 내부에서 fold마다 학습한다. 같은 Pipeline을 로그/로딩해 raw feature 입력 계약을 유지한다. 341/114/114 train/validation/test로 분리하고 validation F1(악성0)을 기준으로 이번 batch의 모델만 선택한다. test는 선택 모델에 한 번만 사용한다. 교육 데이터 결과는 환자 진단/의료 적용 근거가 아니며 코드·데이터 판본/분할/환경을 추가 관리해야 한다.

```python
"""CPU 교육 fixture: validation 선택 → 선택 모델만 test → 등록 API."""
import json
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import infer_signature
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             confusion_matrix, classification_report, make_scorer)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

tracking_dir = Path("./mlflow-e2e-demo").resolve()
tracking_dir.mkdir(exist_ok=True)
mlflow.set_tracking_uri(f"sqlite:///{tracking_dir / 'tracking.db'}")
experiment_name = "breast-cancer-classification"
if mlflow.get_experiment_by_name(experiment_name) is None:
    mlflow.create_experiment(experiment_name,
                             artifact_location=(tracking_dir / "artifacts").as_uri())
mlflow.set_experiment(experiment_name)
batch_id = uuid.uuid4().hex  # 과거 run을 이번 비교에 섞지 않음

data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names
# target0=malignant,1=benign: 아래 F1/precision/recall의 양성은0
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=0.25, random_state=42, stratify=y_dev)

def run_experiment(model, model_name: str, params: dict) -> str:
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params({**params, "model_name": model_name,
                           "scaler": "StandardScaler", "random_state": 42,
                           "positive_label": 0, "train_rows": len(y_train),
                           "val_rows": len(y_val), "test_rows": len(y_test)})
        # Scaler도 각 CV train fold에서 학습. 배포 artifact도 동일 Pipeline
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=5,
            scoring=make_scorer(f1_score, pos_label=0, zero_division=0))
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_val)
        metrics = {"val_accuracy": accuracy_score(y_val, pred),
                   "val_f1_malignant": f1_score(y_val, pred, pos_label=0, zero_division=0),
                   "val_precision_malignant": precision_score(y_val, pred, pos_label=0, zero_division=0),
                   "val_recall_malignant": recall_score(y_val, pred, pos_label=0, zero_division=0),
                   "cv_f1_mean": float(cv_scores.mean()),
                   "cv_f1_std": float(cv_scores.std())}
        mlflow.log_metrics(metrics)
        # 임시 이름 충돌을 피하고 파일 존재 동안 동기 로깅
        with TemporaryDirectory(prefix="mlflow-eval-") as tmp:
            artifacts = {
                "confusion_matrix.json": confusion_matrix(y_val, pred, labels=[0, 1]).tolist(),
                "classification_report.json": classification_report(
                    y_val, pred, labels=[0, 1], target_names=data.target_names,
                    output_dict=True, zero_division=0)}
            for name, value in artifacts.items():
                path = Path(tmp) / name
                path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
                mlflow.log_artifact(str(path))
        logged = mlflow.sklearn.log_model(
            pipeline, name="model",
            signature=infer_signature(X_train, pipeline.predict(X_train)),
            input_example=X_train[:2],
            skops_trusted_types=["sklearn.tree._tree.Tree"])
        mlflow.set_tags({"batch_id": batch_id, "model_uri": logged.model_uri,
                         "developer": "daeyoung", "dataset": "breast_cancer",
                         "validation_status": "educational_fixture"})
        print(model_name, metrics)
        return run.info.run_id

experiments = [
    {
        "model": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "name": "rf-baseline",
        "params": {"n_estimators": 100, "max_depth": 10},
    },
    {
        "model": RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42),
        "name": "rf-tuned",
        "params": {"n_estimators": 300, "max_depth": 20},
    },
    {
        "model": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
        ),
        "name": "gb-baseline",
        "params": {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1},
    },
]

run_ids = [run_experiment(exp["model"], exp["name"], exp["params"]) for exp in experiments]
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name(experiment_name)
best_run = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string=f"tags.batch_id = '{batch_id}' AND attributes.status = 'FINISHED'",
    order_by=["metrics.val_f1_malignant DESC"], max_results=1)[0]
assert best_run.info.run_id in run_ids
best_model_uri = best_run.data.tags["model_uri"]
best_model = mlflow.sklearn.load_model(best_model_uri)
# 이번 선택이 끝난 뒤 test에 한 번만 예측. 이 점수로 재선택하지 않는다.
test_pred = best_model.predict(X_test)
with mlflow.start_run(run_id=best_run.info.run_id):
    mlflow.log_metrics({"test_accuracy": accuracy_score(y_test, test_pred),
                       "test_f1_malignant": f1_score(y_test, test_pred, pos_label=0, zero_division=0)})
print("선택:", best_run.info.run_id, best_run.data.params["model_name"],
      best_run.data.metrics["val_f1_malignant"])
registered = mlflow.register_model(best_model_uri, "breast-cancer-classifier")
client.set_model_version_tag(registered.name, registered.version,
                             "validation_status", "demo_unreviewed")
print("등록 version:", registered.version)
```

---

### 8. 가벼운 대안: JSON/CSV 로깅 클래스

JSONL은 중첩 params/metrics/tags·run별 artifact 경로를 보존한다. 원래0.93/0.96 등은 미측정 API 예시값이다. 같은 run의 동일 basename artifact는 덮어쓰지 않고 거부한다. 단일 writer·수동 종료·정상 JSON schema를 전제하며 동시 append/원자적 복구/재현성/실제 duration을 보장하지 않는다. active run 누락/중복 시작·비유한 metric·빈 비교를 명시적으로 거부한다.

```python
"""
MLflow 없이 사용하는 경량 실험 추적기.
JSON Lines(.jsonl) 형식으로 저장하여 검색과 분석이 용이하다.
"""
import json
import uuid
from datetime import datetime, timezone
from copy import deepcopy
import math
import numpy as np
from pathlib import Path
from typing import Any, Optional

import pandas as pd


class LightExperimentTracker:
    """JSON Lines 기반 경량 실험 추적기"""

    def __init__(self, base_dir: str = "./experiments"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.base_dir / "runs.jsonl"
        self.artifact_dir = self.base_dir / "artifacts"
        self.artifact_dir.mkdir(exist_ok=True)
        self._current_run: dict[str, Any] | None = None

    def _require_run(self) -> dict[str, Any]:
        if self._current_run is None:
            raise RuntimeError("start_run 먼저 필요")
        return self._current_run

    def start_run(
        self,
        experiment: str,
        run_name: Optional[str] = None,
    ) -> str:
        """새 run을 시작하고 run_id를 반환한다."""
        if self._current_run is not None:
            raise RuntimeError("활성 run을 먼저 end_run으로 종료")
        self._current_run = {
            "run_id": uuid.uuid4().hex,
            "experiment": experiment,
            "run_name": run_name or f"run-{datetime.now(timezone.utc).strftime('%H%M%S')}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "params": {},
            "metrics": {},
            "tags": {},
            "artifacts": [],
        }
        return self._current_run["run_id"]

    def log_params(self, params: dict[str, Any]) -> None:
        json.dumps(params, allow_nan=False)
        self._require_run()["params"].update(deepcopy(params))

    def log_metrics(self, metrics: dict[str, float]) -> None:
        run = self._require_run()
        checked = {}
        for key, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("finite 숫자 metric 필요")
            checked[key] = float(value)
        run["metrics"].update(checked)

    def log_tags(self, tags: dict[str, str]) -> None:
        json.dumps(tags, allow_nan=False)
        self._require_run()["tags"].update(deepcopy(tags))

    def log_artifact(self, file_path: str) -> None:
        """파일을 artifact 디렉토리에 복사하고 경로를 기록한다."""
        import shutil

        src = Path(file_path)
        run = self._require_run()
        run_id = run["run_id"]
        dest_dir = self.artifact_dir / run_id
        dest_dir.mkdir(exist_ok=True)
        dest = dest_dir / src.name
        if dest.exists():
            raise FileExistsError(f"같은 run의 artifact 이름 충돌: {src.name}")
        shutil.copy2(src, dest)
        run["artifacts"].append(str(dest))

    def end_run(self) -> dict:
        """현재 run을 JSONL 파일에 저장한다."""
        run = self._require_run()
        run["duration_note"] = "manual"  # 측정된 duration이 아님
        serialized = json.dumps(run, ensure_ascii=False, allow_nan=False)
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(serialized + "\n")
        saved = deepcopy(run)
        self._current_run = None
        return saved

    def load_all_runs(self) -> pd.DataFrame:
        """전체 실험 기록을 DataFrame으로 반환한다."""
        if not self.log_file.exists():
            return pd.DataFrame()

        runs = []
        with self.log_file.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    run = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{self.log_file}:{line_no} JSON 손상") from exc
                flat = {
                    "run_id": run["run_id"],
                    "experiment": run["experiment"],
                    "run_name": run["run_name"],
                    "timestamp": run["timestamp"],
                }
                for k, v in run["params"].items():
                    flat[f"param_{k}"] = v
                for k, v in run["metrics"].items():
                    flat[f"metric_{k}"] = v
                runs.append(flat)
        return pd.DataFrame(runs)

    def best_run(self, metric: str = "metric_accuracy", greater_is_better: bool = True,
                 experiment: str | None = None) -> pd.Series:
        df = self.load_all_runs()
        if experiment is not None and not df.empty:
            df = df.loc[df["experiment"] == experiment]
        if df.empty or metric not in df:
            raise ValueError("비교할 run/metric 없음")
        values = pd.to_numeric(df[metric], errors="raise")
        if values.notna().sum() == 0 or not np.isfinite(values.dropna()).all():
            raise ValueError("관측된 finite metric 필요")
        index = values.idxmax() if greater_is_better else values.idxmin()
        return df.loc[index].copy()


# === 사용 예시 ===
tracker = LightExperimentTracker("./my_experiments")

run_id = tracker.start_run(experiment="quick-test", run_name="rf-v1")
tracker.log_params({"n_estimators": 100, "max_depth": 10})
tracker.log_metrics({"accuracy": 0.93, "f1_score": 0.90})
tracker.log_tags({"developer": "daeyoung"})
tracker.end_run()

run_id = tracker.start_run(experiment="quick-test", run_name="rf-v2")
tracker.log_params({"n_estimators": 200, "max_depth": 15})
tracker.log_metrics({"accuracy": 0.96, "f1_score": 0.94})
tracker.log_tags({"developer": "daeyoung"})
tracker.end_run()

# 전체 기록 조회
print(tracker.load_all_runs())

# 최고 성능 run
print(tracker.best_run("metric_f1_score"))
```

---

> [!warning] skops 신뢰 조건
> MLflow 3.16.1의 log_model 기본 포맷은 skops다. 이 환경에서 직접 학습한 RF/GB 저장도 sklearn.tree._tree.Tree 신뢰 확인을 요구했다. 예제의 skops_trusted_types는 이 직접 생성 모델에만 한정한다. Tree 데이터가 악의적으로 조작되면 predict의 메모리 접근 오류가 발생할 수 있다. 외부 파일의 타입 목록을 일괄 허용하거나 skops 사용만으로 안전하다고 판단하지 않는다. custom class/다른 판본의 신뢰 및 이관은 미확인이다.

## 참고 자료 (References)

확인일2026-10-04. MLflow3.16.1·sklearn1.9.1 로컬 CPU/SQL 환경과 공식 rolling API 문서의 조건을 구분한다. sklearn autolog 공식 범위1.5.2~1.9.0과 로컬 판본의 차이는 자동 기록 미확인/비활성화이며 수동 log/load 실행과 별도다.

- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/): run/metadata/artifact store 역할·MLflow3 모델 URI.
- [MLflow sklearn API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html): autolog 호환 범위·name 인자·signature·serialization_format(로컬 log_model 기본 skops).
- [MLflow PyTorch API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pytorch.html): Lightning Trainer.fit autolog integration.
- [MLflow Registry workflow](https://mlflow.org/docs/latest/ml/model-registry/workflow/): register 반환 version·alias·tag·Stage2.9.0 deprecation.
- [MLflow Registry](https://mlflow.org/docs/latest/ml/model-registry/): 모델 기록/등록·접근/운영 조건.
- [sklearn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html): Pipeline의 fold별 전처리·test로 모델을 선택하지 않는 이유.
- [skops persistence](https://skops.readthedocs.io/en/stable/persistence.html): 알려지지 않은 타입 검토·제한된 trusted 목록과 포맷의 안전성 경계.
- [sklearn F1](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html): pos_label/zero_division 계약.

## 관련 문서

- [배포 읽기 순서](README.md): 저장 → HTTP 서빙 → 추적의 책임 구분.
- [저장/로딩](model-saving-loading.md): 직렬화·구조·환경·신뢰 조건.
- [FastAPI 서빙](fastapi-model-serving.md): 입력/출력·준비 상태·실제 배포 검증 경계.
