---
tags: [ml, workflow, cross-validation, metrics]
level: beginner
last_updated: 2026-02-14
status: in-progress
---

# ML 워크플로우 개요

> 데이터 분할부터 평가까지, 실수 없이 ML 모델을 학습하고 검증하는 표준 파이프라인 가이드

## 왜 필요한가? (Why)

ML 모델 개발에서 가장 흔한 실패 원인은 **알고리즘 선택**이 아니라 **워크플로우의 구조적 결함**이다.

- **데이터 누수(Data Leakage)**: 테스트 데이터 정보가 학습에 섞이면, 실험에서는 성능이 좋지만 실제 배포 시 성능이 급락한다
- **과적합(Overfitting)**: 검증 없이 학습 데이터에만 맞추면 새로운 데이터에 일반화되지 않는다
- **잘못된 메트릭**: 불균형 데이터에 accuracy를 쓰면 모델이 다수 클래스만 예측해도 높은 점수가 나온다
- **재현 불가능**: random seed 관리 없이 실험하면 결과를 재현할 수 없다

표준 워크플로우를 따르면 이런 실수를 **구조적으로 방지**할 수 있다.

---

## 핵심 개념 (What)

### ML 파이프라인 단계

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  데이터 수집  │ →  │  데이터 분할  │ →  │  모델 학습   │ →  │   평가/배포   │
│  & 전처리    │    │ Train/Val/  │    │  + 교차검증   │    │  Test Set   │
│             │    │   Test      │    │             │    │   최종 평가   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

| 단계 | 목적 | 핵심 포인트 |
|------|------|------------|
| **데이터 분할** | 학습/검증/평가 데이터 분리 | 테스트 셋은 마지막까지 건드리지 않음 |
| **학습(Training)** | 모델 파라미터 학습 | Train 셋만 사용 |
| **검증(Validation)** | 하이퍼파라미터 튜닝 | Val 셋 또는 교차검증 사용 |
| **평가(Evaluation)** | 최종 일반화 성능 측정 | Test 셋으로 딱 한 번 평가 |

### 문제 유형 선택 플로차트

```
데이터에 정답(Label)이 있는가?
│
├── YES → 지도학습(Supervised Learning)
│   │
│   ├── 정답이 범주형(카테고리)인가?
│   │   └── YES → 분류(Classification)
│   │       ├── 2개 클래스 → 이진 분류 (Binary)
│   │       └── 3개+ 클래스 → 다중 분류 (Multiclass)
│   │
│   └── 정답이 연속형(숫자)인가?
│       └── YES → 회귀(Regression)
│
└── NO → 비지도학습(Unsupervised Learning)
    │
    ├── 데이터를 그룹으로 묶고 싶은가?
    │   └── YES → 클러스터링(Clustering)
    │       ├── 그룹 수를 아는가? → K-Means
    │       └── 모르는가? → DBSCAN, HDBSCAN
    │
    └── 차원을 줄이고 싶은가?
        └── YES → 차원 축소(Dimensionality Reduction)
            ├── PCA (선형)
            └── t-SNE, UMAP (비선형, 시각화용)
```

---

## 어떻게 사용하는가? (How)

### 0. 공통 임포트 및 데이터 준비

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 재현을 위한 고정 시드
RANDOM_STATE = 42
```

---

### 1. Train/Val/Test 분할

#### 기본 분할 (Train 60% / Val 20% / Test 20%)

```python
# 1단계: Train+Val / Test 분리
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y,  # 분류 문제: 클래스 비율 유지
)

# 2단계: Train / Val 분리
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.25,  # 전체의 0.8 * 0.25 = 0.2
    random_state=RANDOM_STATE,
    stratify=y_temp,
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
```

#### stratify 파라미터가 중요한 이유

```python
# 나쁜 예: stratify 없이 분할 → 클래스 비율이 깨질 수 있음
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 좋은 예: stratify로 클래스 비율 유지
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# 비율 확인
print("전체:", np.bincount(y) / len(y))
print("Train:", np.bincount(y_train) / len(y_train))
print("Test:", np.bincount(y_test) / len(y_test))
```

> **회귀 문제**에서는 `stratify`를 쓰지 않는다 (연속값이라 범주가 없으므로).

---

### 2. K-Fold 교차 검증

#### 기본 KFold

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)

# 기본 KFold (분류에는 StratifiedKFold 권장)
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="accuracy")
print(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

#### StratifiedKFold (분류 문제 필수)

```python
# 각 Fold에서 클래스 비율을 원본과 동일하게 유지
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="f1_weighted")
print(f"CV F1 (weighted): {scores.mean():.4f} (+/- {scores.std():.4f})")
```

#### 수동 KFold (세밀한 제어가 필요할 때)

```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

fold_results = []
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

    model.fit(X_fold_train, y_fold_train)
    y_pred = model.predict(X_fold_val)

    fold_f1 = f1_score(y_fold_val, y_pred, average="weighted")
    fold_results.append(fold_f1)
    print(f"Fold {fold_idx + 1}: F1 = {fold_f1:.4f}")

print(f"\nMean F1: {np.mean(fold_results):.4f} (+/- {np.std(fold_results):.4f})")
```

---

### 3. 메트릭 선택 가이드

#### 분류(Classification) 메트릭

| 메트릭 | 언제 사용 | 주의사항 |
|--------|----------|---------|
| **Accuracy** | 클래스가 균형잡힌 경우 | 불균형 데이터에서는 의미 없음 (99:1이면 다수 클래스만 예측해도 99%) |
| **F1-Score** | 불균형 데이터, Precision과 Recall 모두 중요할 때 | `average` 파라미터 주의: `binary`, `macro`, `weighted` |
| **Precision** | 거짓 양성(FP) 비용이 클 때 (스팸 필터: 정상 메일을 스팸으로 분류하면 안 됨) | Recall과 트레이드오프 |
| **Recall** | 거짓 음성(FN) 비용이 클 때 (질병 진단: 환자를 놓치면 안 됨) | Precision과 트레이드오프 |
| **ROC-AUC** | 이진 분류의 전반적 성능, 임계값 독립적 평가 | 불균형 심할 때는 PR-AUC 고려 |

#### 회귀(Regression) 메트릭

| 메트릭 | 언제 사용 | 주의사항 |
|--------|----------|---------|
| **RMSE** | 큰 오차에 더 큰 페널티를 주고 싶을 때 | 이상치에 민감, 단위가 타겟과 동일 |
| **MAE** | 이상치의 영향을 줄이고 싶을 때 | RMSE보다 이상치에 robust |
| **R²** | 모델이 분산을 얼마나 설명하는지 (0~1, 높을수록 좋음) | 변수 추가하면 무조건 증가 → Adjusted R² 사용 |
| **MAPE** | 비율 기반 오차가 필요할 때 | 실제값이 0에 가까우면 폭발함 |

#### 메트릭 코드 예제

```python
# === 분류 메트릭 ===
y_true = [0, 1, 1, 0, 1, 0, 1, 1]
y_pred = [0, 1, 0, 0, 1, 1, 1, 1]

print("Accuracy:", accuracy_score(y_true, y_pred))
print("F1 (binary):", f1_score(y_true, y_pred, average="binary"))
print("F1 (weighted):", f1_score(y_true, y_pred, average="weighted"))
print("\n", classification_report(y_true, y_pred))

# ROC-AUC (확률값 필요)
# y_proba = model.predict_proba(X_test)[:, 1]
# print("ROC-AUC:", roc_auc_score(y_test, y_proba))

# === 회귀 메트릭 ===
y_true_reg = [3.0, 5.0, 2.5, 7.0]
y_pred_reg = [2.8, 5.2, 2.3, 6.8]

rmse = np.sqrt(mean_squared_error(y_true_reg, y_pred_reg))
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mean_absolute_error(y_true_reg, y_pred_reg):.4f}")
print(f"R²: {r2_score(y_true_reg, y_pred_reg):.4f}")
```

---

### 4. 완전한 ML 워크플로우 템플릿

아래는 데이터 로딩부터 최종 평가까지의 **복사-붙여넣기 가능한** 전체 템플릿이다.

```python
"""
완전한 ML 워크플로우 템플릿
- 분류 문제 기준 (회귀는 메트릭/모델만 교체)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

RANDOM_STATE = 42

# ============================================================
# 1단계: 데이터 로딩
# ============================================================
# 실제 프로젝트에서는 pd.read_csv() 등으로 교체
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

print(f"데이터 형태: {X.shape}")
print(f"클래스 분포:\n{y.value_counts(normalize=True)}")

# ============================================================
# 2단계: 데이터 분할 (Train 60% / Val 20% / Test 20%)
# ============================================================
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=RANDOM_STATE, stratify=y_temp
)

print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# ============================================================
# 3단계: 파이프라인 구성 (전처리 + 모델)
# ============================================================
# Pipeline을 사용하면 전처리가 Train에만 fit되어 데이터 누수 방지
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
    )),
])

# ============================================================
# 4단계: 교차 검증으로 모델 성능 추정
# ============================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(
    pipeline, X_train, y_train, cv=skf, scoring="f1_weighted"
)
print(f"\nCV F1 (weighted): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============================================================
# 5단계: 검증 셋으로 확인
# ============================================================
pipeline.fit(X_train, y_train)
y_val_pred = pipeline.predict(X_val)
val_f1 = f1_score(y_val, y_val_pred, average="weighted")
print(f"Validation F1 (weighted): {val_f1:.4f}")

# ============================================================
# 6단계: 최종 평가 (Test 셋 - 단 한 번만!)
# ============================================================
# 만족스러운 경우에만 Test 셋 사용
# 선택: Train+Val 전체로 재학습 후 Test 평가
pipeline_final = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
    )),
])
X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])

pipeline_final.fit(X_trainval, y_trainval)
y_test_pred = pipeline_final.predict(X_test)

print(f"\n{'='*50}")
print("최종 Test 셋 평가 결과")
print(f"{'='*50}")
print(classification_report(y_test, y_test_pred, target_names=data.target_names))
```

---

### 5. 흔한 실수 모음

#### 실수 1: 데이터 누수 (Data Leakage)

```python
# ❌ 나쁜 예: 전체 데이터에 scaler를 fit한 후 분할
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 테스트 데이터 정보가 스케일링에 포함됨!
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# ✅ 좋은 예: 분할 후 Train에만 fit, Test에는 transform만
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Train에만 fit
X_test_scaled = scaler.transform(X_test)         # Test에는 transform만

# ✅ 더 좋은 예: Pipeline 사용 (자동으로 누수 방지)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier()),
])
pipeline.fit(X_train, y_train)  # scaler.fit은 X_train에만 적용됨
```

#### 실수 2: 불균형 데이터에서 Accuracy 사용

```python
# 클래스 분포: 95% = 0, 5% = 1
# 모든 예측을 0으로 해도 accuracy = 95% → 의미 없음

# ❌ 나쁜 예
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")  # 0.95지만 무의미

# ✅ 좋은 예: F1 또는 ROC-AUC 사용
print(f"F1: {f1_score(y_test, y_pred, average='weighted'):.4f}")
print(classification_report(y_test, y_pred))

# 추가 대응: 클래스 가중치 부여
model = RandomForestClassifier(
    class_weight="balanced",  # 소수 클래스에 더 높은 가중치
    random_state=RANDOM_STATE,
)
```

#### 실수 3: 테스트 셋을 반복 사용

```python
# ❌ 나쁜 예: 하이퍼파라미터 튜닝마다 Test 셋 확인
for n_est in [50, 100, 200, 500]:
    model = RandomForestClassifier(n_estimators=n_est)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)  # Test 셋에 간접적으로 과적합!
    print(f"n_estimators={n_est}: {score:.4f}")

# ✅ 좋은 예: 교차 검증 또는 Validation 셋으로 튜닝
for n_est in [50, 100, 200, 500]:
    model = RandomForestClassifier(n_estimators=n_est, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_weighted")
    print(f"n_estimators={n_est}: CV F1 = {scores.mean():.4f}")
# → 최적 하이퍼파라미터 결정 후, 마지막에 Test 셋으로 한 번만 평가
```

#### 실수 4: random_state 미설정

```python
# ❌ 나쁜 예: 실행할 때마다 결과가 달라짐
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()

# ✅ 좋은 예: 재현 가능한 실험
RANDOM_STATE = 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
```

#### 실수 5: 회귀 문제에 분류 메트릭 사용 (또는 그 반대)

```python
# ❌ 분류 메트릭을 회귀에 사용
# accuracy_score(y_true_continuous, y_pred_continuous)  # 에러 또는 무의미

# ✅ 문제 유형에 맞는 메트릭
# 분류 → accuracy, f1_score, roc_auc_score
# 회귀 → RMSE, MAE, R²
```

---

## 참고 자료 (References)

- [scikit-learn User Guide - Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [scikit-learn User Guide - Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [scikit-learn Pipeline](https://scikit-learn.org/stable/modules/compose.html#pipeline)
- [Google ML Crash Course - Training and Test Sets](https://developers.google.com/machine-learning/crash-course/training-and-test-sets)

---

## 관련 문서

- [분류 모델 레시피](./classification-recipes.md) - 분류 알고리즘별 실전 코드
- [회귀 모델 레시피](./regression-recipes.md) - 회귀 알고리즘별 실전 코드
- [모델 평가 심화](./model-evaluation.md) - 혼동행렬, PR 곡선, 학습 곡선 등 심화 평가 기법
