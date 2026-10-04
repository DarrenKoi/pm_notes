---
tags: [ml, workflow, cross-validation, metrics]
level: beginner
last_updated: 2026-02-14
reviewed_on: 2026-10-04
review_status: partial
document_type: learning
---

# ML 워크플로우 개요

> 데이터 분할부터 평가까지, 누수 위험을 줄이고 평가 조건을 명시하는 교육용 파이프라인 가이드

## 왜 필요한가? (Why)

알고리즘 선택과 함께 **워크플로우의 구조적 결함**을 점검해야 한다. 실패 원인의 빈도 순위는 이 문서에서 검증하지 않았다.

- **데이터 누수(Data Leakage)**: 테스트 데이터 정보가 학습에 섞이면, 실험에서는 성능이 좋지만 실제 배포 시 성능이 급락한다
- **과적합(Overfitting)**: 검증 없이 학습 데이터에만 맞추면 새로운 데이터에 일반화되지 않는다
- **잘못된 메트릭**: 불균형 데이터에 accuracy를 쓰면 모델이 다수 클래스만 예측해도 높은 점수가 나온다
- **재현 불가능**: 난수·데이터·판본·환경을 기록하지 않으면 실행 간 차이를 추적하기 어렵다

분할과 fit 범위를 명확히 하면 이런 실수를 줄일 수 있다. Pipeline 밖에서 만든 누수 피처, 반복 대상·시계열의 잘못된 분할, 평가 데이터에 맞춘 선택은 별도로 점검한다.

---

## 핵심 개념 (What)

### ML 파이프라인 단계

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  데이터 수집  │ →  │  데이터 분할  │ →  │  모델 학습   │ →  │   평가/배포   │
│ & 스키마 점검 │    │ Train/Val/  │    │  + 교차검증   │    │  Test Set   │
│             │    │   Test      │    │             │    │   최종 평가   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

| 단계 | 목적 | 핵심 포인트 |
|------|------|------------|
| **데이터 분할** | 학습/검증/평가 데이터 분리 | 테스트 셋은 마지막까지 건드리지 않음 |
| **학습(Training)** | 모델 파라미터 학습 | Train 셋만 사용 |
| **검증(Validation)** | 하이퍼파라미터 튜닝 | Val 셋 또는 교차검증 사용 |
| **평가(Evaluation)** | 최종 일반화 성능 측정 | Test 셋으로 딱 한 번 평가 |

스키마 확인·형식 파싱은 분할 전에도 할 수 있지만, 평균 대치·스케일·피처 선택처럼 데이터에서 학습하는 변환은 분할 뒤 train 또는 CV의 각 학습 fold에서 fit한다. 타겟·실험 목적에 따라 지도/비지도 선택을 하며, 그룹 수의 사전 지식만으로 군집 알고리즘을 결정하지 않는다.

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

> 이 회귀 예제에서는 연속 타겟 자체를 `stratify`에 전달하지 않는다. 의도적으로 구간을 정의한 별도 층화 설계는 목적과 희소 구간을 검토해야 한다. `np.bincount` 비율 데모는 0 이상의 정수 label만 받는다.

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

#### StratifiedKFold (독립 표본 분류의 클래스 비율 유지)

```python
# 각 Fold의 클래스 비율을 가능한 한 비슷하게 유지
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="f1_weighted")
print(f"CV F1 (weighted): {scores.mean():.4f} (+/- {scores.std():.4f})")
```

독립 표본의 단일 타겟 분류 예제다. 같은 장비·사람·배치가 반복되면 group split, 미래 예측이면 시간 순서를 보존한 split을 검토한다. stratification은 이 종속성을 제거하지 않으며 클래스별 표본 수·각 split에 들어가는 클래스도 확인한다. CV 점수의 표준편차는 독립 test의 신뢰구간이 아니다.

#### 수동 KFold (세밀한 제어가 필요할 때)

```python
# NumPy 배열과 비연속 인덱스의 pandas DataFrame/Series 모두 위치로 선택
def take_rows(values, indices):
    return values.iloc[indices] if hasattr(values, "iloc") else values[indices]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

fold_results = []
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_fold_train, X_fold_val = take_rows(X_train, train_idx), take_rows(X_train, val_idx)
    y_fold_train, y_fold_val = take_rows(y_train, train_idx), take_rows(y_train, val_idx)

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
| **Accuracy** | 클래스가 균형잡힌 경우 | 단독 사용하면 소수 클래스 실패를 가림 (99:1 다수 예측의 accuracy는 99%) |
| **F1-Score** | 불균형 데이터, Precision과 Recall 모두 중요할 때 | 양성 클래스·평균 방식을 지정. weighted는 다수 클래스 영향을 크게 받음 |
| **Precision** | 거짓 양성(FP) 비용이 클 때 (스팸 필터: 정상 메일을 스팸으로 분류하면 안 됨) | Recall과 트레이드오프 |
| **Recall** | 거짓 음성(FN) 비용이 클 때 (질병 진단: 환자를 놓치면 안 됨) | Precision과 트레이드오프 |
| **ROC-AUC** | 이진 분류의 전반적 성능, 임계값 독립적 평가 | 불균형 심할 때는 PR-AUC 고려 |

#### 회귀(Regression) 메트릭

| 메트릭 | 언제 사용 | 주의사항 |
|--------|----------|---------|
| **RMSE** | 큰 오차에 더 큰 페널티를 주고 싶을 때 | 이상치에 민감, 단위가 타겟과 동일 |
| **MAE** | 이상치의 영향을 줄이고 싶을 때 | RMSE보다 이상치에 robust |
| **R²** | 평균 예측 대비 잔차 제곱합 감소 (최대 1, 음수 가능) | 상수 타겟·소표본은 별도 해석. 같은 학습 표본의 절편 포함 OLS에서 중첩 피처를 추가하면 학습 R²는 감소하지 않지만 test R²·다른 모델에 일반화되지 않음 |
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

# ROC-AUC (연속 점수 필요: predict_proba 또는 decision_function)
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

아래는 scikit-learn 내장 breast cancer 데이터로 실행하는 교육용 템플릿이다. 실제 입력에는 스키마·타겟 의미·결측·분할 단위와 평가 비용을 다시 정의해야 한다. 이 데이터의 1은 benign(양성 종양), 0은 malignant(악성)이므로 binary F1의 기본 pos_label=1을 질병 검출 지표로 해석하면 안 된다.

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
# Pipeline 내부 전처리는 해당 학습 fold에만 fit됨; 외부 피처/분할 누수는 별도 점검
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

# ✅ Pipeline 내부 전처리를 학습 fold에 한정 (모든 누수의 자동 방지는 아님)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier()),
])
pipeline.fit(X_train, y_train)  # scaler.fit은 X_train에만 적용됨
```

#### 실수 2: 불균형 데이터에서 Accuracy 사용

```python
# 클래스 분포: 95% = 0, 5% = 1
# 모든 예측을 0으로 해도 accuracy = 95% → 소수 클래스 탐지 성능은 드러나지 않음

# ❌ 나쁜 예
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")  # 이 지표만으로 소수 클래스 성능 판단 금지

# ✅ 아래는 1이 관심 사건인 이진 문제 기준. 실제 양성 label부터 정의
print(f"Positive F1: {f1_score(y_test, y_pred, pos_label=1, zero_division=0):.4f}")
print(f"Macro F1: {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
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

# ✅ 동일 입력/판본/환경에서 난수 변동을 줄이는 실험
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

## 검토 근거와 적용 조건

확인일 **2026-10-04**, 공식 문서 판본 **scikit-learn 1.9.1**. 실행에는 NumPy·pandas·scikit-learn이 필요하다. 독립 표본의 타겟 분류 데모이며 실제 운영의 평가 프로토콜을 대신하지 않는다. `X`, `y`, `model` 등은 앞 예제 또는 호출자가 준비한다. 서로 다른 분할/데모에서 생성한 `y_pred`를 다른 `y_test`와 조합하지 않는다. 잘못된 예는 실행 대상으로 권장하지 않는다.

- [교차 검증: 그룹·시간 의존성](https://scikit-learn.org/stable/modules/cross_validation.html): 분할이 적용 대상과 일치해야 한다.
- [누수와 난수 관리](https://scikit-learn.org/stable/common_pitfalls.html): 학습 범위와 random_state의 효과를 구분한다. seed 하나가 데이터/라이브러리/하드웨어 판본의 재현을 보장하지 않는다.
- [R² API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html): 최대 1, 음수 가능. 상수 타겟은 기본 force_finite 처리도 확인한다.
- [F1 API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html): weighted는 클래스 support로 가중하므로 소수 클래스 성능을 따로 본다.
- [ROC-AUC API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html): 이진 문제의 decision score도 입력 가능하다.
- [breast cancer 데이터](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html): 데모의 label 이름·자료 범위를 확인한다.
- [선형 회귀 목적함수](https://scikit-learn.org/stable/modules/linear_model.html): OLS는 잔차 제곱합을 최소화한다. 같은 표본/절편/중첩 입력의 학습 R² 비교는 이 목적함수와 R² 정의로 설명한다.

이 예제의 weighted F1은 다중 클래스에도 사용 가능한 평균 데모이며 불균형 해결책이라는 뜻이 아니다. 단일 지표 외에 클래스별 recall/precision, 관심 사건의 PR 곡선·임계값 비용을 기록한다. RandomForest의 StandardScaler는 Pipeline 학습 범위를 보여주는 교육용 단계이며 이 모델에 필수라는 뜻이 아니다. test를 한 번 사용한다는 원칙은 최종 선택 전에 보류하라는 뜻이다. 배포 후 새로운 시간 구간의 모니터링까지 금지하지 않는다.

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
