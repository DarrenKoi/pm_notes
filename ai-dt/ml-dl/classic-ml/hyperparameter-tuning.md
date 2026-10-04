---
tags: [hyperparameter, gridsearch, optuna, tuning]
level: intermediate
last_updated: 2026-02-14
reviewed_on: 2026-10-04
review_status: partial
document_type: learning
---

# 하이퍼파라미터 튜닝 (Hyperparameter Tuning)

> 모델의 학습 성능을 극대화하기 위해 최적의 하이퍼파라미터 조합을 체계적으로 탐색하는 방법론 정리

---

## 왜 필요한가? (Why)

- **기본 하이퍼파라미터가 모든 입력의 최적을 보장하지 않는다**: scikit-learn, XGBoost 등의 기본값은 범용적으로 설정되어 있어 특정 데이터셋에 대해 최적 성능을 보장하지 않는다
- **수동 튜닝은 비효율적이다**: 파라미터 조합이 기하급수적으로 늘어나 사람이 직접 시도하는 것은 한계가 있다
- **실험 조건과 난수·판본을 기록한다**: 실험 결과를 기록하고, 동일한 조건에서 재현할 수 있어야 실무에서 신뢰할 수 있다
- **과적합 방지**: Cross-validation 기반 탐색은 일반화 성능을 기준으로 파라미터를 선택하므로 과적합 위험을 줄인다

---

## 핵심 개념 (What)

### Grid Search vs Random Search vs Bayesian Optimization

| 방식 | 원리 | 특징 |
|------|------|------|
| **Grid Search** | 지정한 파라미터 조합을 **모두** 시도 | 완전 탐색, 소규모 파라미터 공간에 적합 |
| **Random Search** | 파라미터 분포에서 **무작위 샘플링** | 중요한 일부 차원에 예산을 집중할 수 있음; 성능은 입력/분포 의존, n_iter로 예산 조절 |
| **Bayesian Optimization** | 이전 시도 결과를 기반으로 **다음 탐색 지점을 추론** | 효율은 목적함수/탐색 공간/예산에 의존; Optuna는 여러 sampler 지원 |

### 핵심 용어

- **탐색 공간(Search Space)**: 탐색할 하이퍼파라미터의 범위와 타입
- **목적 함수(Objective Function)**: 최적화할 평가 지표 (예: accuracy, RMSE)
- **교차 검증(Cross-Validation)**: 데이터를 K-fold로 나눠 일반화 성능을 추정
- **Pruning**: Bayesian 최적화에서 성능이 낮은 trial을 조기 중단하여 시간 절약

---

## 어떻게 사용하는가? (How)

### 1. GridSearchCV - 완전 탐색

모든 파라미터 조합을 시도한다. 주어진 이산 grid와 CV 점수에서 가장 좋은 후보를 선택한다. 모든 가능한 값의 전역 최적을 보장하지 않는다.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, f1_score, make_scorer

# breast cancer: 0=malignant, 1=benign. 이 데모는 0의 F1을 목표로 지정.
POS_LABEL = 0
SCORING = make_scorer(f1_score, pos_label=POS_LABEL, zero_division=0)


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
    scoring=SCORING,           # 평가 지표
    cv=5,                   # 5-fold cross-validation
    n_jobs=-1,              # 모든 CPU 코어 사용
    verbose=1,
    refit=True,             # fit에 전달한 train 전체에 재학습; 보류 test 제외
)

grid_search.fit(X_train, y_train)

# 결과 확인
print(f"최적 파라미터: {grid_search.best_params_}")
print(f"최적 CV 점수: {grid_search.best_score_:.4f}")
# 여러 방식의 선택이 끝날 때까지 test 평가를 보류한다.
# 최종 고정한 모델만 별도 test로 평가: 다음 callout의 예시 참조.

# 총 시도 횟수: 3 * 4 * 3 * 3 = 108 조합 x 5 fold = 540회 CV 학습 + refit 1회
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
    "n_estimators": randint(50, 500),           # 50 이상 500 미만 정수
    "max_depth": randint(3, 15),                # 3 이상 15 미만 정수
    "learning_rate": uniform(0.01, 0.29),       # 0.01~0.30 사이 실수
    "subsample": uniform(0.6, 0.4),             # 0.6~1.0 사이 실수
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 10),
}

random_search = RandomizedSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=100,             # 100 후보 x5 fold + refit1; 다른 model/grid와 시간 우열은 실측
    scoring=SCORING,
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

이전 탐색 결과를 학습하여 유망한 영역을 집중적으로 탐색한다. 동일 예산에서 실제 결과로 Grid/Random과 비교해야 한다. TPE는 확률 모형을 쓰며 기본 startup trial은 무작위 탐색이다.

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
    score = cross_val_score(clf, X_train, y_train, cv=5, scoring=SCORING).mean()
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

중간 결과에 따라 trial을 중단한다. 절약량과 잘못 중단한 후보의 영향은 실험별 확인한다.

```python
import optuna
from sklearn.model_selection import StratifiedKFold
import numpy as np
from xgboost import XGBClassifier

def take_rows(values, indices):
    return values.iloc[indices] if hasattr(values, "iloc") else values[indices]


def objective_with_pruning(trial):
    params = {
        "n_estimators": 1000,
        "early_stopping_rounds": 50,  # inner stopping set 필요
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
        X_fold_train, X_fold_val = take_rows(X_train, train_idx), take_rows(X_train, val_idx)
        y_fold_train, y_fold_val = take_rows(y_train, train_idx), take_rows(y_train, val_idx)
        # stopping은 fold train 안의 별도 holdout, 점수는 untouched fold val
        X_fit, X_stop, y_fit, y_stop = train_test_split(
            X_fold_train, y_fold_train, test_size=0.2,
            stratify=y_fold_train, random_state=42,
        )

        clf = XGBClassifier(**params, random_state=42, eval_metric="logloss")
        clf.fit(
            X_fit, y_fit,
            eval_set=[(X_stop, y_stop)],
            verbose=False,
        )

        score = f1_score(y_fold_val, clf.predict(X_fold_val),
                         pos_label=POS_LABEL, zero_division=0)
        scores.append(score)

        # 중간 결과 보고 → Pruner가 판단
        trial.report(np.mean(scores), fold_idx)

        # Pruner가 중단 결정하면 즉시 종료
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(scores)


# MedianPruner: 같은 step의 완료 trial 중간값보다 trial의 최고 중간값이 나쁜 경우 판단
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5,     # 완료 trial5개가 쌓이기 전 pruning 비활성
        n_warmup_steps=2,       # step<2는 비활성; fold_idx=2(세 번째 fold)부터 판단
    ),
)

study.optimize(objective_with_pruning, n_trials=100, show_progress_bar=True)

# Pruning 통계 확인
pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
print(f"완료된 trial: {len(complete_trials)}")
print(f"Pruning된 trial: {len(pruned_trials)}")
if complete_trials:
    print(f"완료 trial 최고 점수: {study.best_value:.4f}")
else:
    print("완료 trial 없음: 최적값 미확인")
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

교육용 탐색 후보의 예시이며 실제 최적 범위나 업무 표준을 검증하지 않았다. 아래 tuple은 탐색 API에 넣을 분포/후보를 결정하기 위한 표기이며 그대로 RandomizedSearchCV 분포가 되지는 않는다.

#### Random Forest

```python
rf_space = {
    "n_estimators": (100, 1000),          # 실제 예산/수렴/validation으로 범위 결정
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
    "num_leaves": (20, 150),              # max_depth>0이면 leaves <=2^max_depth도 고려
    "subsample": (0.6, 1.0),              # = bagging_fraction
    "subsample_freq": [1],               # 0이면 행 bagging 비활성
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
    score = cross_val_score(clf, X_train, y_train, cv=5, scoring=SCORING).mean()
    return score
```

---

### 6. Pipeline과 함께 튜닝

전처리를 Pipeline 안에 두고 전체 Pipeline을 CV에 전달하면 그 전처리는 학습 fold에서만 fit된다. 외부 누수 피처·잘못된 분할·타겟 포함·test 선택은 별도 점검한다.

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
    scoring=SCORING,
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

    score = cross_val_score(pipe, X_train, y_train, cv=5, scoring=SCORING).mean()
    return score

study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(pipeline_objective, n_trials=50)
```

---

### 7. 비교표: GridSearch vs RandomSearch vs Optuna

| 항목 | GridSearchCV | RandomizedSearchCV | Optuna |
|------|-------------|-------------------|--------|
| **탐색 전략** | 완전 탐색 (Exhaustive) | 무작위 샘플링 | TPE 포함 다양한 sampler |
| **탐색 효율** | grid크기/fit비용 의존 | 분포/예산 의존 | sampler/목표/예산 의존 |
| **파라미터 수 3~4개** | 적합 | 적합 | 적합 |
| **파라미터 수 5개 이상** | grid 크기 확인 | 분포/예산 확인 | sampler/예산 확인 |
| **연속형 파라미터** | 이산화 필요 | 분포 지정 가능 | 분포 지정 가능 |
| **조기 중단 (Pruning)** | 불가 | 불가 | 지원 (MedianPruner 등) |
| **시각화** | 수동 구현 | 수동 구현 | 내장 시각화 |
| **병렬/분산 실행** | n_jobs/joblib backend 조건 | n_jobs/joblib backend 조건 | storage·worker·충돌/재현 조건 |
| **구현 난이도** | 매우 쉬움 | 쉬움 | 보통 |
| **추천 상황** | 소규모 탐색, 빠른 프로토타입 | 중규모 탐색, 시간 제한 | 대규모 탐색, 최적 성능 추구 |

#### 실무 가이드라인

```
파라미터 조합 < 100개   → GridSearchCV (명시 grid의 CV 최고 후보)
파라미터 조합 100~1000  → RandomizedSearchCV (n_iter=100~200)
파라미터 조합 > 1000    → Optuna (n_trials=100~300, Pruning 활용)
```

---

## 확인 근거와 적용 조건

확인일 **2026-10-04**. 공식 scikit-learn **1.9.1**, Optuna **5.0.0**, SciPy 문서 **1.18.0** 기준 대조. XGBoost 문서3.4.2와 실제 설치3.4.1을 구분한다. 블록은 1번 데이터/SCORING을 사용하며 study/grid_search 변수는 뒤 예제에서 덮어쓴다. 한 탐색의 성능을 다른 study의 결과와 섞지 않는다. NumPy·pandas·SciPy·scikit-learn·Optuna·Plotly/Matplotlib 및 native XGBoost runtime이 필요하다.

- [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html): 지정 후보의 scoring 비교와 train refit. refit은 보류 test를 포함한 전체 데이터라는 뜻이 아니다.
- [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html): list/분포 샘플링 조건·seed·n_iter를 확인한다. scipy randint의 high는 제외된다.
- [Nested CV 예제](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html): 탐색에 사용한 best_score는 선택 편향이 있을 수 있다. 다른 방법을 같은 test로 반복 고르지 않고 nested CV/최종 holdout을 사용한다.
- [TPE sampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html): startup과 seed를 확인한다. 모든 Optuna 탐색이 TPE인 것은 아니다.
- [MedianPruner](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html): 완료 trial·같은 step·warmup/NaN 계약을 확인한다. 이 코드는 boosting round별 pruning이 아니라 fold 종료마다 report한다.
- [Optuna FAQ](https://optuna.readthedocs.io/en/stable/faq.html): 병렬 탐색/비결정 objective·storage·schema 판본이 재현과 재개에 영향을 준다. sampler seed 하나가 전체 실행을 보장하지 않는다.
- [SciPy randint](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.randint.html): 범위는 low 이상 high 미만이다.

1번의 RF grid108개는 CV540+refit1, 2번100후보는 CV500+refit1, 6번PCA/SVC grid128개는 CV640+refit1이다. fit마다 model/데이터 비용이 달라 횟수만으로 속도 우열을 단정하지 않는다. 전체 예제는 큰 예산이므로 실행 전 CPU·메모리·trial 수와 병렬 단계 하나를 정한다. 축소 실행을 원래 모든 조합의 성능/최적값 증거로 해석하지 않는다.

모든 이진 점수는 이 데이터의 malignant label=0 F1을 목표로 맞췄다. 사용자의 실제 양성 label·비용은 별도로 결정해야 한다. 같은 장비/시계열 반복은 stratified 랜덤 CV 대신 적용 단위에 맞는 분할을 선택한다. best_score의 표준편차는 독립 test 신뢰구간이 아니다. 4번stopping/튜닝은 fold train 내부에서, 성능 측정은 untouched fold val에서 수행한다. 각 fold/내부 holdout에 모든 클래스가 충분히 존재해야 한다.

탐색 방식/모델/threshold 선택을 학습 범위에서 고정한 뒤, 예를 들어 `chosen_model = grid_search.best_estimator_`로 실제 선택한 객체를 저장해 `classification_report(y_test, chosen_model.predict(X_test))`를 한 번 수행한다. 여기서 마지막 grid_search는 SVC 탐색 결과이므로 처음 RF 객체라는 뜻이 아니다. 완료 trial이 없으면 best_value를 읽을 수 없고 시각화/중요도도 충분한 완료 trial·관련 파라미터·추가 의존성을 필요로 한다. Plotly show는 renderer에 따라 브라우저를 열 수 있으므로 headless 검증은 figure 생성/직렬화만 확인한다. 중요도는 탐색 공간 안의 목적함수 중요도이며 인과/입력 피처 중요도와 다르다.

원래 XGBoost 설치는 macOS libomp.dylib 부재로 import 실패했다. native pruning/early stopping 실행은 미확인이다. 알고리즘별 실제 최적 범위·전체 예산 실행·분산 환경·사용자 plot/한글 화면·Claude 협의는 별도 검증 대상이다.

## 참고 자료 (References)

- [scikit-learn GridSearchCV 공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [scikit-learn RandomizedSearchCV 공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
- [Optuna 공식 문서](https://optuna.readthedocs.io/)
- [Optuna Tutorial - Efficient Optimization](https://optuna.readthedocs.io/en/stable/tutorial/index.html)
- [Random Search for Hyper-Parameter Optimization (Bergstra & Bengio, 2012)](https://www.jmlr.org/papers/v13/bergstra12a.html)

---

## 관련 문서

- [ML/DL 학습 가이드](../README.md)
