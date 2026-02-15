---
tags: [classification, sklearn, xgboost, lightgbm]
level: intermediate
last_updated: 2026-02-14
status: in-progress
---

# 분류(Classification) 실전 레시피

> 가장 자주 쓰이는 4가지 분류 알고리즘의 복사-붙여넣기용 실전 코드 모음

## 왜 필요한가? (Why)

- **분류(Classification)는 ML 업무의 가장 기본이자 가장 빈번한 태스크**이다. 불량 판정, 고객 이탈 예측, 문서 분류 등 대부분의 비즈니스 문제가 "A인가 B인가"로 귀결된다.
- 알고리즘마다 강점이 다르기 때문에, 데이터 특성에 맞는 모델을 빠르게 골라 baseline을 세울 수 있어야 한다.
- 이 문서는 **매번 처음부터 작성하지 않도록** 검증된 코드 템플릿을 모아두는 것이 목적이다.

## 핵심 개념 (What)

| 알고리즘 | 핵심 강점 | 약점 | 언제 쓰는가 |
|----------|----------|------|-------------|
| **로지스틱 회귀** | 해석 가능, 빠름, 확률 출력 | 비선형 관계 학습 어려움 | baseline, 해석이 중요할 때 |
| **랜덤 포레스트** | 과적합에 강함, 튜닝 적음 | 메모리 사용량 큼 | 중소규모 데이터, feature importance 필요 시 |
| **XGBoost** | 높은 성능, 결측치 자동 처리 | 하이퍼파라미터 많음 | 정형 데이터 대회/실무 범용 |
| **LightGBM** | 대규모 데이터에서 빠름, 범주형 직접 지원 | 소규모 데이터에서 과적합 가능 | 대규모 데이터, 범주형 피처 많을 때 |

## 어떻게 사용하는가? (How)

### 0. 공통 데이터 준비

모든 예제에서 공유하는 데이터셋 생성 코드이다.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 재현 가능한 이진 분류 데이터 생성
X, y = make_classification(
    n_samples=2000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=2,
    weights=[0.7, 0.3],   # 불균형 클래스
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Class distribution (train): {dict(zip(*__import__('numpy').unique(y_train, return_counts=True)))}")
```

---

### 1. 로지스틱 회귀 (Logistic Regression)

규제(Regularization)와 클래스 가중치를 적용한 실전 패턴이다.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# 로지스틱 회귀는 스케일링이 필수 → Pipeline으로 묶는다
pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        C=1.0,                    # 규제 강도 (작을수록 강한 규제)
        penalty="l2",             # L1: 피처 선택 효과, L2: 일반적 규제
        class_weight="balanced",  # 불균형 클래스 자동 보정
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )),
])

pipe_lr.fit(X_train, y_train)
y_pred_lr = pipe_lr.predict(X_test)

print("=== Logistic Regression ===")
print(classification_report(y_test, y_pred_lr, digits=3))

# 계수 확인 (해석용)
import numpy as np
coef = pipe_lr.named_steps["clf"].coef_[0]
feature_names = [f"feat_{i}" for i in range(X_train.shape[1])]
top_features = sorted(zip(feature_names, coef), key=lambda x: abs(x[1]), reverse=True)[:5]
print("Top 5 features by |coefficient|:")
for name, c in top_features:
    print(f"  {name}: {c:+.4f}")
```

---

### 2. 랜덤 포레스트 (Random Forest)

Feature importance 시각화까지 포함한 패턴이다.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,           # None이면 끝까지 분할 (과적합 주의)
    min_samples_leaf=5,       # 리프 최소 샘플 수로 과적합 제어
    class_weight="balanced",  # 불균형 보정
    n_jobs=-1,                # 전체 CPU 코어 사용
    random_state=42,
)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("=== Random Forest ===")
print(classification_report(y_test, y_pred_rf, digits=3))

# --- Feature Importance 시각화 ---
feature_names = [f"feat_{i}" for i in range(X_train.shape[1])]
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1][:15]  # 상위 15개

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(
    range(len(indices)),
    importances[indices][::-1],
    color="steelblue",
)
ax.set_yticks(range(len(indices)))
ax.set_yticklabels([feature_names[i] for i in indices][::-1])
ax.set_xlabel("Feature Importance (MDI)")
ax.set_title("Random Forest - Top 15 Feature Importances")
plt.tight_layout()
plt.savefig("rf_feature_importance.png", dpi=150)
plt.show()
```

---

### 3. XGBoost

Early stopping과 eval_set을 활용한 실전 패턴이다.

```python
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import numpy as np

# 클래스 불균형 비율 계산
scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)

xgb = XGBClassifier(
    n_estimators=1000,          # 충분히 크게 잡고 early stopping으로 제어
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,  # 불균형 보정
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)

# Early stopping: 검증 성능이 50 라운드 연속 개선 안 되면 중단
xgb.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50,                  # 50 라운드마다 로그 출력
)

# early stopping 결과 확인
print(f"Best iteration: {xgb.best_iteration}")
print(f"Best score: {xgb.best_score:.4f}")

y_pred_xgb = xgb.predict(X_test)

print("\n=== XGBoost ===")
print(classification_report(y_test, y_pred_xgb, digits=3))

# --- XGBoost 내장 feature importance ---
import matplotlib.pyplot as plt
from xgboost import plot_importance

fig, ax = plt.subplots(figsize=(10, 6))
plot_importance(xgb, ax=ax, max_num_features=15, importance_type="gain")
ax.set_title("XGBoost - Feature Importance (Gain)")
plt.tight_layout()
plt.savefig("xgb_feature_importance.png", dpi=150)
plt.show()
```

---

### 4. LightGBM

범주형 피처(Categorical Feature)를 직접 지원하는 패턴이다.

```python
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
import numpy as np
import pandas as pd

# --- 범주형 피처가 포함된 데이터 준비 ---
X, y = make_classification(
    n_samples=3000, n_features=15, n_informative=8,
    n_classes=2, weights=[0.7, 0.3], random_state=42,
)

# DataFrame으로 변환 후 일부 피처를 범주형으로 만든다
df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(15)])
df["cat_A"] = np.random.choice(["low", "mid", "high"], size=len(df))
df["cat_B"] = np.random.choice(["type_1", "type_2", "type_3", "type_4"], size=len(df))

# 범주형 컬럼을 pandas category dtype으로 변환 (LightGBM이 자동 인식)
for col in ["cat_A", "cat_B"]:
    df[col] = df[col].astype("category")

from sklearn.model_selection import train_test_split
X_train_lg, X_test_lg, y_train_lg, y_test_lg = train_test_split(
    df, y, test_size=0.2, stratify=y, random_state=42
)

# 불균형 보정
scale_pos_weight = np.sum(y_train_lg == 0) / np.sum(y_train_lg == 1)

lgbm = LGBMClassifier(
    n_estimators=1000,
    max_depth=-1,               # -1이면 제한 없음
    learning_rate=0.05,
    num_leaves=31,              # max_depth 대신 이것으로 복잡도 제어
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

lgbm.fit(
    X_train_lg, y_train_lg,
    eval_set=[(X_test_lg, y_test_lg)],
    callbacks=[
        early_stopping(stopping_rounds=50),
        log_evaluation(period=50),
    ],
)

y_pred_lgbm = lgbm.predict(X_test_lg)

print("=== LightGBM ===")
print(classification_report(y_test_lg, y_pred_lgbm, digits=3))

# --- Feature Importance (범주형 포함) ---
import matplotlib.pyplot as plt

importance = lgbm.feature_importances_
feature_names = X_train_lg.columns.tolist()
sorted_idx = np.argsort(importance)[-15:]

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(len(sorted_idx)), importance[sorted_idx], color="darkorange")
ax.set_yticks(range(len(sorted_idx)))
ax.set_yticklabels([feature_names[i] for i in sorted_idx])
ax.set_xlabel("Feature Importance (Split count)")
ax.set_title("LightGBM - Feature Importance (Categorical 포함)")
plt.tight_layout()
plt.savefig("lgbm_feature_importance.png", dpi=150)
plt.show()
```

---

### 5. 모델 비교 템플릿

4가지 모델을 한 번에 학습하고 성능을 비교하는 통합 코드이다.

```python
import numpy as np
import pandas as pd
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

# --- 데이터 준비 ---
X, y = make_classification(
    n_samples=5000, n_features=20, n_informative=12,
    n_classes=2, weights=[0.7, 0.3], random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)

# --- 모델 정의 ---
models = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000, random_state=42
        )),
    ]),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5,
        class_weight="balanced", n_jobs=-1, random_state=42,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss", random_state=42, n_jobs=-1,
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42, n_jobs=-1, verbose=-1,
    ),
}

# --- 학습 및 평가 ---
results = []

for name, model in models.items():
    start = time.time()

    # XGBoost / LightGBM은 early stopping 적용
    if name == "XGBoost":
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=0,
        )
    elif name == "LightGBM":
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[early_stopping(50), log_evaluation(0)],
        )
    else:
        model.fit(X_train, y_train)

    elapsed = time.time() - start

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1 (macro)": f1_score(y_test, y_pred, average="macro"),
        "F1 (minority)": f1_score(y_test, y_pred, pos_label=1),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "Train Time (s)": round(elapsed, 3),
    })

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, digits=3))

# --- 결과 요약 테이블 ---
df_results = pd.DataFrame(results).set_index("Model")
print("\n" + "=" * 60)
print("  모델 비교 요약")
print("=" * 60)
print(df_results.to_string(float_format="{:.4f}".format))
print()

# 최고 모델 자동 선택
best_model = df_results["F1 (macro)"].idxmax()
print(f"F1 (macro) 기준 최고 모델: {best_model}")
```

---

### 6. 알고리즘 선택 가이드

데이터 상황에 따라 어떤 알고리즘을 먼저 시도할지 결정하는 가이드이다.

#### 의사결정 테이블

| 상황 | 추천 알고리즘 | 이유 |
|------|-------------|------|
| **데이터 < 1,000건** | 로지스틱 회귀 / 랜덤 포레스트 | 트리 부스팅은 소규모 데이터에서 과적합 위험 |
| **데이터 1,000 ~ 100,000건** | XGBoost / 랜덤 포레스트 | 가장 범용적인 영역 |
| **데이터 > 100,000건** | LightGBM | 학습 속도가 압도적으로 빠름 |
| **해석력이 중요** | 로지스틱 회귀 | 계수 기반 해석, 규제 감사 대응 가능 |
| **범주형 피처 많음** | LightGBM | 인코딩 없이 범주형 직접 처리 |
| **결측치 많음** | XGBoost / LightGBM | 트리 부스팅은 결측치 자동 처리 |
| **빠른 baseline 필요** | 로지스틱 회귀 | 설정 최소, 학습 가장 빠름 |
| **최대 성능 필요** | XGBoost + Optuna 튜닝 | 경진대회에서 검증된 조합 |
| **피처 선택 필요** | 로지스틱 회귀(L1) / 랜덤 포레스트 | L1 규제 또는 MDI importance |

#### 실전 순서 추천

```
1. 로지스틱 회귀 (baseline, 1분 컷)
   ↓ 성능 부족 시
2. 랜덤 포레스트 (빠르게 비선형 모델 확인)
   ↓ 성능 부족 시
3. XGBoost or LightGBM (하이퍼파라미터 튜닝과 함께)
   ↓ 추가 성능 필요 시
4. Optuna로 하이퍼파라미터 자동 탐색
```

#### 속도 비교 (참고, 환경에 따라 다름)

| 알고리즘 | 100K 데이터 학습 시간 (대략) | 예측 속도 |
|----------|---------------------------|----------|
| 로지스틱 회귀 | < 1초 | 매우 빠름 |
| 랜덤 포레스트 | 5 ~ 15초 | 빠름 |
| XGBoost | 10 ~ 30초 | 빠름 |
| LightGBM | 3 ~ 10초 | 빠름 |

---

## 참고 자료 (References)

- [scikit-learn 공식 문서 - Classification](https://scikit-learn.org/stable/supervised_learning.html)
- [XGBoost 공식 문서](https://xgboost.readthedocs.io/en/stable/)
- [LightGBM 공식 문서](https://lightgbm.readthedocs.io/en/stable/)
- [XGBoost vs LightGBM 비교 (Neptune.ai)](https://neptune.ai/blog/xgboost-vs-lightgbm)
- [sklearn Pipeline 사용법](https://scikit-learn.org/stable/modules/compose.html)

## 관련 문서

- [데이터 전처리 기초](../../data-handling/)
- [하이퍼파라미터 튜닝 (Optuna)](./hyperparameter-tuning.md)
- [회귀 모델 레시피](./regression-recipes.md)
- [모델 평가 지표 가이드](./evaluation-metrics.md)
