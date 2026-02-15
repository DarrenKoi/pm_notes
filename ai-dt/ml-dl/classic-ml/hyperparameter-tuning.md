---
tags: [hyperparameter, gridsearch, optuna, tuning]
level: intermediate
last_updated: 2026-02-14
status: in-progress
---

# 하이퍼파라미터 튜닝 (Hyperparameter Tuning)

> 모델의 학습 성능을 극대화하기 위해 최적의 하이퍼파라미터 조합을 체계적으로 탐색하는 방법론 정리

---

## 왜 필요한가? (Why)

- **기본 하이퍼파라미터는 거의 최적이 아니다**: scikit-learn, XGBoost 등의 기본값은 범용적으로 설정되어 있어 특정 데이터셋에 대해 최적 성능을 보장하지 않는다
- **수동 튜닝은 비효율적이다**: 파라미터 조합이 기하급수적으로 늘어나 사람이 직접 시도하는 것은 한계가 있다
- **체계적 탐색이 재현성을 보장한다**: 실험 결과를 기록하고, 동일한 조건에서 재현할 수 있어야 실무에서 신뢰할 수 있다
- **과적합 방지**: Cross-validation 기반 탐색은 일반화 성능을 기준으로 파라미터를 선택하므로 과적합 위험을 줄인다

---

## 핵심 개념 (What)

### Grid Search vs Random Search vs Bayesian Optimization

| 방식 | 원리 | 특징 |
|------|------|------|
| **Grid Search** | 지정한 파라미터 조합을 **모두** 시도 | 완전 탐색, 소규모 파라미터 공간에 적합 |
| **Random Search** | 파라미터 분포에서 **무작위 샘플링** | 고차원에서 Grid보다 효율적, n_iter로 예산 조절 |
| **Bayesian Optimization** | 이전 시도 결과를 기반으로 **다음 탐색 지점을 추론** | 가장 효율적, Optuna/Hyperopt 등이 대표적 |

### 핵심 용어

- **탐색 공간(Search Space)**: 탐색할 하이퍼파라미터의 범위와 타입
- **목적 함수(Objective Function)**: 최적화할 평가 지표 (예: accuracy, RMSE)
- **교차 검증(Cross-Validation)**: 데이터를 K-fold로 나눠 일반화 성능을 추정
- **Pruning**: Bayesian 최적화에서 성능이 낮은 trial을 조기 중단하여 시간 절약

---

## 어떻게 사용하는가? (How)

### 1. GridSearchCV - 완전 탐색

모든 파라미터 조합을 시도한다. 파라미터 공간이 작을 때 확실한 최적값을 찾을 수 있다.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

# 데이터 준비
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 탐색할 파라미터 그리드 정의
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

# GridSearchCV 실행
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring="f1",           # 평가 지표
    cv=5,                   # 5-fold cross-validation
    n_jobs=-1,              # 모든 CPU 코어 사용
    verbose=1,
    refit=True,             # 최적 파라미터로 전체 데이터 재학습
)

grid_search.fit(X_train, y_train)

# 결과 확인
print(f"최적 파라미터: {grid_search.best_params_}")
print(f"최적 CV 점수: {grid_search.best_score_:.4f}")
print(f"\n테스트 성능:")
print(classification_report(y_test, grid_search.predict(X_test)))

# 총 시도 횟수: 3 * 4 * 3 * 3 = 108 조합 x 5 fold = 540회 학습
```

---

### 2. RandomizedSearchCV - 무작위 탐색

확률 분포에서 파라미터를 샘플링한다. `n_iter`로 시도 횟수를 제어하여 예산 관리가 가능하다.

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# 확률 분포 기반 파라미터 공간 정의
param_distributions = {
    "n_estimators": randint(50, 500),           # 50~500 사이 정수
    "max_depth": randint(3, 15),                # 3~15 사이 정수
    "learning_rate": uniform(0.01, 0.29),       # 0.01~0.30 사이 실수
    "subsample": uniform(0.6, 0.4),             # 0.6~1.0 사이 실수
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 10),
}

random_search = RandomizedSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=100,             # 100번만 샘플링 (Grid 대비 훨씬 적은 시도)
    scoring="f1",
    cv=5,
    n_jobs=-1,
    verbose=1,
    random_state=42,
)

random_search.fit(X_train, y_train)

print(f"최적 파라미터: {random_search.best_params_}")
print(f"최적 CV 점수: {random_search.best_score_:.4f}")

# 결과를 DataFrame으로 정리
import pandas as pd
results_df = pd.DataFrame(random_search.cv_results_)
results_df = results_df.sort_values("rank_test_score")
print(results_df[["params", "mean_test_score", "std_test_score", "rank_test_score"]].head(10))
```

---

### 3. Optuna 기본 - Bayesian Optimization

이전 탐색 결과를 학습하여 유망한 영역을 집중적으로 탐색한다. Grid/Random 대비 훨씬 적은 시도로 좋은 결과를 얻을 수 있다.

```python
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# 목적 함수 정의
def objective(trial):
    """Optuna가 최소화/최대화할 목적 함수"""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }

    clf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    score = cross_val_score(clf, X_train, y_train, cv=5, scoring="f1").mean()
    return score

# Study 생성 및 최적화 실행
study = optuna.create_study(
    direction="maximize",           # f1 점수를 최대화
    study_name="rf_tuning",
    sampler=optuna.samplers.TPESampler(seed=42),  # Tree-structured Parzen Estimator
)

study.optimize(
    objective,
    n_trials=50,                    # 50번 시도
    show_progress_bar=True,
)

# 결과 확인
print(f"최적 파라미터: {study.best_params}")
print(f"최적 CV 점수: {study.best_value:.4f}")
print(f"총 trial 수: {len(study.trials)}")

# 최적 파라미터로 최종 모델 학습
best_clf = RandomForestClassifier(**study.best_params, random_state=42, n_jobs=-1)
best_clf.fit(X_train, y_train)
```

#### trial.suggest_* 주요 메서드

| 메서드 | 용도 | 예시 |
|--------|------|------|
| `suggest_int(name, low, high)` | 정수 파라미터 | `trial.suggest_int("depth", 3, 15)` |
| `suggest_float(name, low, high)` | 실수 파라미터 | `trial.suggest_float("lr", 1e-4, 1e-1)` |
| `suggest_float(..., log=True)` | 로그 스케일 실수 | `trial.suggest_float("lr", 1e-5, 1e-1, log=True)` |
| `suggest_categorical(name, choices)` | 범주형 파라미터 | `trial.suggest_categorical("loss", ["gini", "entropy"])` |

---

### 4. Optuna 고급 - Pruning & Visualization

#### Pruning (조기 중단)

성능이 낮은 trial을 중간에 중단하여 탐색 시간을 대폭 줄인다.

```python
import optuna
from sklearn.model_selection import StratifiedKFold
import numpy as np
from xgboost import XGBClassifier

def objective_with_pruning(trial):
    params = {
        "n_estimators": 1000,       # 큰 값으로 설정 (early stopping 사용)
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

        clf = XGBClassifier(**params, random_state=42, eval_metric="logloss")
        clf.fit(
            X_fold_train, y_fold_train,
            eval_set=[(X_fold_val, y_fold_val)],
            verbose=False,
        )

        score = clf.score(X_fold_val, y_fold_val)
        scores.append(score)

        # 중간 결과 보고 → Pruner가 판단
        trial.report(np.mean(scores), fold_idx)

        # Pruner가 중단 결정하면 즉시 종료
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(scores)


# MedianPruner: 중간값 이하 성능의 trial을 조기 중단
study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5,     # 최소 5개 trial은 완료 후 pruning 시작
        n_warmup_steps=2,       # 각 trial에서 최소 2 step 후 pruning 판단
    ),
)

study.optimize(objective_with_pruning, n_trials=100, show_progress_bar=True)

# Pruning 통계 확인
pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
print(f"완료된 trial: {len(complete_trials)}")
print(f"Pruning된 trial: {len(pruned_trials)}")
print(f"최적 점수: {study.best_value:.4f}")
```

#### Visualization

Optuna의 내장 시각화로 탐색 과정을 분석한다.

```python
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_contour,
    plot_slice,
)

# 1. 최적화 진행 히스토리
fig1 = plot_optimization_history(study)
fig1.show()

# 2. 파라미터 중요도 (어떤 파라미터가 성능에 가장 큰 영향?)
fig2 = plot_param_importances(study)
fig2.show()

# 3. 파라미터 간 상호작용 (Contour plot)
fig3 = plot_contour(study, params=["learning_rate", "max_depth"])
fig3.show()

# 4. 각 파라미터별 성능 분포 (Slice plot)
fig4 = plot_slice(study, params=["learning_rate", "max_depth", "subsample"])
fig4.show()

# Matplotlib 기반 시각화도 가능 (Plotly 미설치 환경)
from optuna.visualization.matplotlib import plot_optimization_history as plot_history_mpl
fig5 = plot_history_mpl(study)
```

---

### 5. 실전 하이퍼파라미터 범위

실무에서 자주 사용하는 주요 모델별 권장 탐색 범위이다.

#### Random Forest

```python
rf_space = {
    "n_estimators": (100, 1000),          # 보통 200~500이면 충분
    "max_depth": (5, 30),                 # None도 포함 고려
    "min_samples_split": (2, 20),
    "min_samples_leaf": (1, 10),
    "max_features": ["sqrt", "log2", None],
}
```

#### XGBoost

```python
xgb_space = {
    "n_estimators": (100, 2000),          # early stopping 함께 사용
    "max_depth": (3, 10),                 # 너무 깊으면 과적합
    "learning_rate": (0.001, 0.3),        # log scale 추천
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "reg_alpha": (1e-8, 10.0),            # log scale
    "reg_lambda": (1e-8, 10.0),           # log scale
    "min_child_weight": (1, 10),
    "gamma": (0.0, 5.0),
}
```

#### LightGBM

```python
lgbm_space = {
    "n_estimators": (100, 2000),
    "max_depth": (-1, 15),                # -1은 제한 없음
    "learning_rate": (0.001, 0.3),        # log scale
    "num_leaves": (20, 150),              # 2^max_depth보다 작게
    "subsample": (0.6, 1.0),              # = bagging_fraction
    "colsample_bytree": (0.6, 1.0),       # = feature_fraction
    "reg_alpha": (1e-8, 10.0),            # log scale
    "reg_lambda": (1e-8, 10.0),           # log scale
    "min_child_samples": (5, 100),
}
```

#### Optuna에서 활용 예시 (XGBoost)

```python
def xgb_objective(trial):
    params = {
        "n_estimators": 2000,
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }

    clf = XGBClassifier(**params, random_state=42, eval_metric="logloss")
    score = cross_val_score(clf, X_train, y_train, cv=5, scoring="f1").mean()
    return score
```

---

### 6. Pipeline과 함께 튜닝

전처리와 모델을 Pipeline으로 묶고 함께 튜닝하면 데이터 누출(Data Leakage)을 방지할 수 있다.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Pipeline 구성
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA()),
    ("svc", SVC()),
])

# 파라미터 이름 규칙: step이름__파라미터이름
param_grid = {
    "pca__n_components": [5, 10, 15, 20],
    "svc__C": [0.1, 1, 10, 100],
    "svc__kernel": ["rbf", "poly"],
    "svc__gamma": ["scale", "auto", 0.01, 0.001],
}

grid_search = GridSearchCV(
    pipe,
    param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    verbose=1,
)

grid_search.fit(X_train, y_train)

print(f"최적 파라미터: {grid_search.best_params_}")
print(f"최적 CV 점수: {grid_search.best_score_:.4f}")

# Pipeline 파라미터 이름 확인 방법
print(pipe.get_params().keys())
```

#### Pipeline + Optuna 조합

```python
def pipeline_objective(trial):
    n_components = trial.suggest_int("pca__n_components", 5, 25)
    C = trial.suggest_float("svc__C", 0.01, 100, log=True)
    gamma = trial.suggest_float("svc__gamma", 1e-4, 1e-1, log=True)
    kernel = trial.suggest_categorical("svc__kernel", ["rbf", "poly"])

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components)),
        ("svc", SVC(C=C, gamma=gamma, kernel=kernel)),
    ])

    score = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1").mean()
    return score

study = optuna.create_study(direction="maximize")
study.optimize(pipeline_objective, n_trials=50)
```

---

### 7. 비교표: GridSearch vs RandomSearch vs Optuna

| 항목 | GridSearchCV | RandomizedSearchCV | Optuna |
|------|-------------|-------------------|--------|
| **탐색 전략** | 완전 탐색 (Exhaustive) | 무작위 샘플링 | Bayesian (TPE) |
| **탐색 효율** | 낮음 (조합 폭발) | 중간 | 높음 (이전 결과 활용) |
| **파라미터 수 3~4개** | 적합 | 적합 | 적합 |
| **파라미터 수 5개 이상** | 비현실적 | 적합 | 가장 적합 |
| **연속형 파라미터** | 이산화 필요 | 분포 지정 가능 | 분포 지정 가능 |
| **조기 중단 (Pruning)** | 불가 | 불가 | 지원 (MedianPruner 등) |
| **시각화** | 수동 구현 | 수동 구현 | 내장 시각화 |
| **분산 학습** | 불가 | 불가 | 지원 (RDB 기반) |
| **구현 난이도** | 매우 쉬움 | 쉬움 | 보통 |
| **추천 상황** | 소규모 탐색, 빠른 프로토타입 | 중규모 탐색, 시간 제한 | 대규모 탐색, 최적 성능 추구 |

#### 실무 가이드라인

```
파라미터 조합 < 100개   → GridSearchCV (확실한 최적값)
파라미터 조합 100~1000  → RandomizedSearchCV (n_iter=100~200)
파라미터 조합 > 1000    → Optuna (n_trials=100~300, Pruning 활용)
```

---

## 참고 자료 (References)

- [scikit-learn GridSearchCV 공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [scikit-learn RandomizedSearchCV 공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
- [Optuna 공식 문서](https://optuna.readthedocs.io/)
- [Optuna Tutorial - Efficient Optimization](https://optuna.readthedocs.io/en/stable/tutorial/index.html)
- [Random Search for Hyper-Parameter Optimization (Bergstra & Bengio, 2012)](https://www.jmlr.org/papers/v13/bergstra12a.html)

---

## 관련 문서

- [ML/DL 학습 가이드](../README.md)
