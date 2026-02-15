---
tags: [mlflow, experiment-tracking, logging]
level: intermediate
last_updated: 2026-02-14
status: in-progress
---

# 실험 추적(Experiment Tracking)

> ML 실험의 파라미터, 메트릭, 아티팩트를 체계적으로 기록하고 비교하는 방법을 정리한다.

## 왜 필요한가? (Why)

- **재현 불가능 문제**: 실험 추적 없이는 "지난주에 잘 됐던 모델"을 다시 만들 수 없다. 어떤 하이퍼파라미터 조합이었는지, 어떤 데이터 전처리를 적용했는지 기억에 의존하게 된다.
- **비교 불가능 문제**: 10번의 실험을 돌렸는데, 어떤 실험이 가장 좋았는지 정리가 안 되면 시간 낭비가 된다.
- **팀 협업**: 동료에게 "이 모델 어떻게 학습시켰어?"라고 물었을 때, 코드 diff만으로는 파악이 어렵다. 추적 로그가 있으면 즉시 확인 가능하다.
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
│  │ - Params     │  │ - Stages             │  │
│  │ - Metrics    │  │   (Staging/Prod)     │  │
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

### 1. 간단한 CSV 로깅: DIY 실험 추적

MLflow를 도입하기 전, 혹은 가볍게 실험을 관리하고 싶을 때 pandas DataFrame으로 직접 추적하는 방법이다.

```python
import pandas as pd
from datetime import datetime
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
            "timestamp": datetime.now().isoformat(),
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
        self.df.to_csv(self.log_path, index=False)
        print(f"[logged] {experiment_name} | acc={metrics.get('accuracy', 'N/A')}")

    def best_run(self, metric_col: str = "metric_accuracy") -> pd.Series:
        """특정 메트릭 기준 최고 실험을 반환한다."""
        return self.df.loc[self.df[metric_col].idxmax()]

    def summary(self) -> pd.DataFrame:
        """전체 실험 요약 테이블을 반환한다."""
        return self.df.sort_values("timestamp", ascending=False)


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
pip install mlflow

# 버전 확인
mlflow --version
```

```python
import mlflow

# --- Tracking URI 설정 ---
# 로컬 파일 시스템 (기본값, ./mlruns 디렉토리에 저장)
mlflow.set_tracking_uri("file:///tmp/my-mlflow-runs")

# 원격 서버 사용 시
# mlflow.set_tracking_uri("http://mlflow-server.company.com:5000")

# --- Experiment 설정 ---
# 이미 존재하면 해당 experiment를 사용, 없으면 새로 생성
mlflow.set_experiment("fraud-detection-v2")

# experiment 정보 확인
experiment = mlflow.get_experiment_by_name("fraud-detection-v2")
print(f"Experiment ID: {experiment.experiment_id}")
print(f"Artifact Location: {experiment.artifact_location}")
```

---

### 3. MLflow 자동 로깅(Autolog)

`autolog()`를 호출하면 해당 프레임워크의 학습 과정을 **자동으로** 추적한다. 파라미터, 메트릭, 모델 아티팩트까지 모두 기록된다.

#### scikit-learn 자동 로깅

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# autolog 활성화 — 이 한 줄이면 된다
mlflow.sklearn.autolog()

# 데이터 준비
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 학습 실행 — autolog가 알아서 모든 것을 기록
with mlflow.start_run(run_name="rf-autolog-demo"):
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # autolog가 자동으로 기록하는 것들:
    # - Parameters: n_estimators, max_depth, criterion, ...
    # - Metrics: training_accuracy, training_f1_score, ...
    # - Artifacts: 모델 파일, 피처 중요도 plot
```

#### PyTorch 자동 로깅

```python
import mlflow
import mlflow.pytorch

# PyTorch Lightning 사용 시
mlflow.pytorch.autolog()

# 기록되는 항목:
# - Parameters: epochs, lr, optimizer 설정
# - Metrics: epoch별 loss, 각종 메트릭
# - Artifacts: 최종 모델 체크포인트
```

> **참고**: `autolog()`은 내부적으로 monkey-patching을 사용한다. 디버깅이 어려울 수 있으므로, 프로덕션 수준에서는 수동 로깅을 권장한다.

---

### 4. MLflow 수동 로깅

세밀한 제어가 필요할 때 수동 로깅을 사용한다. 어떤 값을 기록할지 명시적으로 결정할 수 있다.

```python
import mlflow
import json
from pathlib import Path

mlflow.set_experiment("manual-logging-demo")

with mlflow.start_run(run_name="manual-run-001") as run:

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
    config_path = Path("/tmp/config.json")
    config_path.write_text(json.dumps(config, indent=2))
    mlflow.log_artifact(str(config_path))

    # 디렉토리 통째로 저장
    # mlflow.log_artifacts("/path/to/output_dir", artifact_path="outputs")

    # === Model: 모델 저장 ===
    # sklearn 모델 저장 (모델 객체가 있다면)
    # mlflow.sklearn.log_model(model, "model")

    # === Tags: 메타데이터 ===
    mlflow.set_tag("developer", "daeyoung")
    mlflow.set_tag("purpose", "baseline_comparison")

    print(f"Run ID: {run.info.run_id}")
```

---

### 5. MLflow UI: 실험 시각화 및 비교

```bash
# MLflow UI 실행 (기본 포트 5000)
mlflow ui

# 포트 지정
mlflow ui --port 8080

# 특정 backend store 지정
mlflow ui --backend-store-uri file:///tmp/my-mlflow-runs
```

브라우저에서 `http://localhost:5000` 접속하면:

- **Experiment 목록**: 좌측 패널에서 experiment 선택
- **Run 비교**: 여러 run을 체크박스로 선택 후 "Compare" 클릭
- **메트릭 차트**: step별 메트릭 변화를 시각화
- **파라미터 vs 메트릭**: Parallel Coordinates Plot으로 최적 조합 탐색

#### 프로그래밍 방식으로 실험 비교

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# 특정 experiment의 모든 run 조회
experiment = client.get_experiment_by_name("fraud-detection-v2")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"],
    max_results=5,
)

print("=== Top 5 Runs by Accuracy ===")
for run in runs:
    params = run.data.params
    metrics = run.data.metrics
    print(
        f"  Run {run.info.run_id[:8]} | "
        f"n_estimators={params.get('n_estimators', 'N/A')} | "
        f"accuracy={metrics.get('accuracy', 'N/A'):.4f}"
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

학습된 모델을 버전 관리하고, Staging/Production 단계를 관리한다.

```python
import mlflow
from mlflow.tracking import MlflowClient

# === 모델 등록 ===
# 방법 1: run 내에서 직접 등록
with mlflow.start_run() as run:
    # ... 학습 코드 ...
    # mlflow.sklearn.log_model(model, "model", registered_model_name="fraud-detector")
    pass

# 방법 2: 이미 로깅된 모델을 나중에 등록
model_uri = f"runs:/{run.info.run_id}/model"
result = mlflow.register_model(model_uri, "fraud-detector")

print(f"Model Name: {result.name}")
print(f"Version: {result.version}")

# === 모델 버전 관리 ===
client = MlflowClient()

# 모델 버전에 설명 추가
client.update_model_version(
    name="fraud-detector",
    version=1,
    description="RF baseline, accuracy=0.94",
)

# 모델 버전에 태그 추가
client.set_model_version_tag(
    name="fraud-detector",
    version=1,
    key="validation_status",
    value="approved",
)

# === 특정 버전 모델 로드 ===
# 버전 지정
model = mlflow.sklearn.load_model("models:/fraud-detector/1")

# 최신 버전
model = mlflow.sklearn.load_model("models:/fraud-detector/latest")
```

> **Stage 기반 관리**: MLflow 2.x부터는 Stage(Staging/Production/Archived) 대신 **Model Aliases**와 **Tags** 사용이 권장된다.

```python
# Alias 기반 관리 (MLflow 2.x 권장 방식)
client.set_registered_model_alias("fraud-detector", "champion", 1)
client.set_registered_model_alias("fraud-detector", "challenger", 2)

# alias로 모델 로드
champion_model = mlflow.sklearn.load_model("models:/fraud-detector@champion")
```

---

### 7. 실전 통합 예제: sklearn + MLflow End-to-End

```python
"""
실전 예제: sklearn 분류 모델 학습 + MLflow 추적 전체 파이프라인
"""
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
import json
from pathlib import Path

# --- 설정 ---
mlflow.set_tracking_uri("file:///tmp/mlflow-e2e-demo")
mlflow.set_experiment("breast-cancer-classification")

# --- 데이터 준비 ---
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 실험 함수 ---
def run_experiment(model, model_name: str, params: dict) -> str:
    """모델 학습 + MLflow 로깅을 수행하고 run_id를 반환한다."""

    with mlflow.start_run(run_name=model_name) as run:
        # 파라미터 로깅
        mlflow.log_params(params)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("scaler", "StandardScaler")
        mlflow.log_param("test_size", 0.2)

        # 학습
        model.fit(X_train_scaled, y_train)

        # 예측
        y_pred = model.predict(X_test_scaled)

        # 메트릭 계산 및 로깅
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
        }
        mlflow.log_metrics(metrics)

        # Cross Validation 점수
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        mlflow.log_metric("cv_mean_accuracy", cv_scores.mean())
        mlflow.log_metric("cv_std_accuracy", cv_scores.std())

        # Artifact: confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_path = Path("/tmp/confusion_matrix.json")
        cm_path.write_text(json.dumps(cm.tolist(), indent=2))
        mlflow.log_artifact(str(cm_path))

        # Artifact: classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_path = Path("/tmp/classification_report.json")
        report_path.write_text(json.dumps(report, indent=2))
        mlflow.log_artifact(str(report_path))

        # 모델 저장
        mlflow.sklearn.log_model(model, "model")

        # 태그
        mlflow.set_tag("developer", "daeyoung")
        mlflow.set_tag("dataset", "breast_cancer")

        print(f"[{model_name}] accuracy={metrics['accuracy']:.4f}, f1={metrics['f1_score']:.4f}")
        return run.info.run_id


# --- 여러 모델 비교 실험 ---
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

run_ids = []
for exp in experiments:
    run_id = run_experiment(exp["model"], exp["name"], exp["params"])
    run_ids.append(run_id)

# --- 최적 모델 선택 ---
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("breast-cancer-classification")
best_run = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.f1_score DESC"],
    max_results=1,
)[0]

print(f"\n=== Best Run ===")
print(f"Run ID: {best_run.info.run_id}")
print(f"Model: {best_run.data.params.get('model_name')}")
print(f"F1 Score: {best_run.data.metrics.get('f1_score'):.4f}")

# --- 최적 모델 레지스트리 등록 ---
best_model_uri = f"runs:/{best_run.info.run_id}/model"
mlflow.register_model(best_model_uri, "breast-cancer-classifier")
```

---

### 8. 가벼운 대안: JSON/CSV 로깅 클래스

MLflow 서버를 운영하기 부담스럽거나, 소규모 개인 실험에서는 아래와 같은 가벼운 로깅 클래스를 사용할 수 있다.

```python
"""
MLflow 없이 사용하는 경량 실험 추적기.
JSON Lines(.jsonl) 형식으로 저장하여 검색과 분석이 용이하다.
"""
import json
import uuid
from datetime import datetime
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

    def start_run(
        self,
        experiment: str,
        run_name: Optional[str] = None,
    ) -> str:
        """새 run을 시작하고 run_id를 반환한다."""
        self._current_run = {
            "run_id": uuid.uuid4().hex[:12],
            "experiment": experiment,
            "run_name": run_name or f"run-{datetime.now().strftime('%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "params": {},
            "metrics": {},
            "tags": {},
            "artifacts": [],
        }
        return self._current_run["run_id"]

    def log_params(self, params: dict[str, Any]) -> None:
        self._current_run["params"].update(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self._current_run["metrics"].update(metrics)

    def log_tags(self, tags: dict[str, str]) -> None:
        self._current_run["tags"].update(tags)

    def log_artifact(self, file_path: str) -> None:
        """파일을 artifact 디렉토리에 복사하고 경로를 기록한다."""
        import shutil

        src = Path(file_path)
        run_id = self._current_run["run_id"]
        dest_dir = self.artifact_dir / run_id
        dest_dir.mkdir(exist_ok=True)
        dest = dest_dir / src.name
        shutil.copy2(src, dest)
        self._current_run["artifacts"].append(str(dest))

    def end_run(self) -> dict:
        """현재 run을 JSONL 파일에 저장한다."""
        self._current_run["duration_note"] = "manual"
        with open(self.log_file, "a") as f:
            f.write(json.dumps(self._current_run, ensure_ascii=False) + "\n")
        saved = self._current_run.copy()
        self._current_run = {}
        return saved

    def load_all_runs(self) -> pd.DataFrame:
        """전체 실험 기록을 DataFrame으로 반환한다."""
        if not self.log_file.exists():
            return pd.DataFrame()

        runs = []
        with open(self.log_file, "r") as f:
            for line in f:
                run = json.loads(line.strip())
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

    def best_run(self, metric: str = "metric_accuracy") -> pd.Series:
        """특정 메트릭 기준 최고 run을 반환한다."""
        df = self.load_all_runs()
        return df.loc[df[metric].idxmax()]


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

## 참고 자료 (References)

- [MLflow 공식 문서](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking API](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [MLflow sklearn Integration](https://mlflow.org/docs/latest/python_api/mlflow.sklearn.html)
- [MLflow PyTorch Integration](https://mlflow.org/docs/latest/python_api/mlflow.pytorch.html)

## 관련 문서

- [ML/DL 배포 가이드](../deployment/) - 모델 배포 관련 문서
- [데이터 처리 파이프라인](../../data-handling/) - 학습 데이터 전처리
