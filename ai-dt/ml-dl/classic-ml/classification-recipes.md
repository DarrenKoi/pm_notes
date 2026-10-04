---
tags: [classification, sklearn, xgboost, lightgbm]
level: intermediate
last_updated: 2026-02-14
reviewed_on: 2026-10-04
review_status: partial
document_type: learning
---

# 분류(Classification) 실전 레시피

> 가장 자주 쓰이는 4가지 분류 알고리즘의 입력·판본·평가 조건을 명시한 교육용 코드 모음

## 왜 필요한가? (Why)

- **분류(Classification)는 범주형 타겟을 다루는 기본 태스크**이다. 불량 판정, 고객 이탈 예측, 문서 분류 등 범주형 타겟을 정한 문제에 적용한다.
- 알고리즘마다 강점이 다르기 때문에, 데이터 특성에 맞는 모델을 빠르게 골라 baseline을 세울 수 있어야 한다.
- 이 문서는 **매번 처음부터 작성하지 않도록** 검증 범위를 기록한 코드 템플릿을 모아두는 것이 목적이다.

## 핵심 개념 (What)

| 알고리즘 | 핵심 강점 | 약점 | 언제 쓰는가 |
|----------|----------|------|-------------|
| **로지스틱 회귀** | 해석 가능, 빠름, 확률 출력 | 비선형 관계 학습 어려움 | baseline, 해석이 중요할 때 |
| **랜덤 포레스트** | bagging으로 분산 감소 가능; 검증·튜닝 필요 | 메모리 사용량 큼 | 중소규모 데이터, feature importance 필요 시 |
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

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
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

# 규제·단위 차이와 최적화 수렴을 고려해 scaler를 Pipeline 내부에 둠
pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        C=1.0,                    # 규제 강도 (작을수록 강한 규제)
        l1_ratio=0.0,             # sklearn >=1.8: L2. L1은 solver 호환도 확인
        class_weight="balanced",  # 클래스 빈도 역수로 학습 가중; 확률/임계값 품질 별도 확인
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )),
])

pipe_lr.fit(X_train, y_train)
y_pred_lr = pipe_lr.predict(X_val)

print("=== Logistic Regression ===")
print(classification_report(y_val, y_pred_lr, digits=3))

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
y_pred_rf = rf.predict(X_val)

print("=== Random Forest ===")
print(classification_report(y_val, y_pred_rf, digits=3))

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
if set(np.unique(y_train)) != {0, 1}:
    raise ValueError("가중치 비율 데모는 두 클래스가 모두 있는 0/1 입력 필요")
scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)

xgb = XGBClassifier(
    n_estimators=1000,          # validation으로 round 선택
    early_stopping_rounds=50,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    subsample_freq=1,          # LightGBM: 0이면 행 bagging 비활성
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,  # 불균형 보정
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)

# Early stopping: 검증 성능이 50 라운드 연속 개선 안 되면 중단
xgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50,                  # 50 라운드마다 로그 출력
)

# early stopping 결과 확인
print(f"Best iteration: {xgb.best_iteration}")
print(f"Best score: {xgb.best_score:.4f}")

y_pred_xgb = xgb.predict(X_val)

print("\n=== XGBoost ===")
print(classification_report(y_val, y_pred_xgb, digits=3))

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
rng = np.random.default_rng(42)
df["cat_A"] = rng.choice(["low", "mid", "high"], size=len(df))
df["cat_B"] = rng.choice(["type_1", "type_2", "type_3", "type_4"], size=len(df))

# 범주형 컬럼을 pandas category dtype으로 변환 (LightGBM이 자동 인식)
for col in ["cat_A", "cat_B"]:
    df[col] = df[col].astype("category")

from sklearn.model_selection import train_test_split
X_temp_lg, X_test_lg, y_temp_lg, y_test_lg = train_test_split(
    df, y, test_size=0.2, stratify=y, random_state=42
)
X_train_lg, X_val_lg, y_train_lg, y_val_lg = train_test_split(
    X_temp_lg, y_temp_lg, test_size=0.25, stratify=y_temp_lg, random_state=42
)

# 불균형 보정
if set(np.unique(y_train_lg)) != {0, 1}:
    raise ValueError("0/1의 두 학습 클래스 필요")
scale_pos_weight = np.sum(y_train_lg == 0) / np.sum(y_train_lg == 1)

lgbm = LGBMClassifier(
    n_estimators=1000,
    max_depth=-1,               # -1이면 제한 없음
    learning_rate=0.05,
    num_leaves=31,              # max_depth 대신 이것으로 복잡도 제어
    subsample=0.8,
    subsample_freq=1,          # LightGBM: 0이면 행 bagging 비활성
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

lgbm.fit(
    X_train_lg, y_train_lg,
    eval_set=[(X_val_lg, y_val_lg)],
    callbacks=[
        early_stopping(stopping_rounds=50),
        log_evaluation(period=50),
    ],
)

y_pred_lgbm = lgbm.predict(X_val_lg)

print("=== LightGBM ===")
print(classification_report(y_val_lg, y_pred_lgbm, digits=3))

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

공통 데이터 준비와 별도로 실행 가능한 네 모델의 통합 데모다. validation으로 비교하고 고정한 선택만 보류 test에 평가한다. 앞 LightGBM category 데모는 별도 입력이라 직접 같은 성능 표로 비교하지 않는다.

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
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)

if set(np.unique(y_train)) != {0, 1}:
    raise ValueError("가중치 비율 데모는 두 클래스가 모두 있는 0/1 입력 필요")
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
        early_stopping_rounds=50,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss", random_state=42, n_jobs=-1,
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        subsample_freq=1,
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
            eval_set=[(X_val, y_val)],
            verbose=0,
        )
    elif name == "LightGBM":
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[early_stopping(50), log_evaluation(0)],
        )
    else:
        model.fit(X_train, y_train)

    elapsed = time.time() - start

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_val, y_pred),
        "F1 (macro)": f1_score(y_val, y_pred, average="macro"),
        "F1 (label=1)": f1_score(y_val, y_pred, pos_label=1),
        "ROC-AUC": roc_auc_score(y_val, y_proba),
        "Train Time (s)": round(elapsed, 3),
    })

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(classification_report(y_val, y_pred, digits=3))

# --- 결과 요약 테이블 ---
df_results = pd.DataFrame(results).set_index("Model")
print("\n" + "=" * 60)
print("  Validation 모델 비교 요약")
print("=" * 60)
print(df_results.to_string(float_format="{:.4f}".format))
print()

# 최고 모델 자동 선택
best_model = df_results["F1 (macro)"].idxmax()
print(f"Validation F1 (macro) 기준 선택: {best_model}")
# 모델/threshold 선택을 고정한 뒤 최종 test. 같은 validation 선택을 test에 재튜닝하지 않음.
selected = models[best_model]
final_pred = selected.predict(X_test)
print("=== 보류 Test (선택된 모델만) ===")
print(classification_report(y_test, final_pred, digits=3))
```

---

### 6. 알고리즘 선택 가이드

데이터 상황에 따라 어떤 알고리즘을 먼저 시도할지 결정하는 가이드이다.

#### 의사결정 테이블

| 상황 | 추천 알고리즘 | 이유 |
|------|-------------|------|
| **데이터 < 1,000건** | 로지스틱 회귀 / 랜덤 포레스트 | 트리 부스팅은 소규모 데이터에서 과적합 위험 |
| **데이터 1,000 ~ 100,000건** | XGBoost / 랜덤 포레스트 | 가장 범용적인 영역 |
| **데이터 > 100,000건** | LightGBM | 입력/하드웨어/파라미터에 따른 속도 비교 필요 |
| **해석력이 중요** | 로지스틱 회귀 | 계수 기반 해석, 계수 의미/스케일·규제·도메인 검토 필요 |
| **범주형 피처 많음** | LightGBM | category dtype/코드/입력 스키마 조건 확인 |
| **결측치 많음** | XGBoost / LightGBM | 트리 부스팅은 결측치 자동 처리 |
| **빠른 baseline 필요** | 로지스틱 회귀 | 설정 최소, 규모·solver별 실제 시간 측정 |
| **최대 성능 필요** | XGBoost + Optuna 튜닝 | 탐색 예산·누수 없는 validation으로 실측 비교 |
| **피처 선택 필요** | 로지스틱 회귀(L1) / 랜덤 포레스트 | L1 계수 0 가능; MDI만으로 피처 제거 결정 금지 |

#### 실전 순서 추천

```
1. 로지스틱 회귀 (baseline, 시간은 실측)
   ↓ 성능 부족 시
2. 랜덤 포레스트 (빠르게 비선형 모델 확인)
   ↓ 성능 부족 시
3. XGBoost or LightGBM (하이퍼파라미터 튜닝과 함께)
   ↓ 추가 성능 필요 시
4. Optuna로 하이퍼파라미터 자동 탐색
```

#### 속도 비교 (미측정; 실험 조건을 먼저 기록)

| 알고리즘 | 100K 데이터 학습 시간 (대략) | 예측 속도 |
|----------|---------------------------|----------|
| 로지스틱 회귀 | 미측정 | 미측정 |
| 랜덤 포레스트 | 미측정 | 미측정 |
| XGBoost | 미측정 | 미측정 |
| LightGBM | 미측정 | 미측정 |

---

## 확인 근거와 적용 조건

확인일 **2026-10-04**. 공식 문서 표시 판본은 scikit-learn **1.9.1**, XGBoost **3.4.2**, LightGBM **4.7.0**이다. 이것을 최신 설치 가능 판본으로 단정하지 않는다. 블록은 0번 셋업에 의존하며 4번 category 데모/5번 전체 비교는 별도 합성 입력을 만든다. pandas·NumPy·Matplotlib·scikit-learn·XGBoost·LightGBM과 각 OS native runtime이 필요하다. 로컬 PyPI 설치 판본 XGBoost 3.4.1/LightGBM 4.7.0은 macOS libomp.dylib 부재로 import 실패했다. native 모델의 실제 stopping/category 처리 실행은 미검증이며 공식 API/AST 대조로 구분한다. PNG를 별도 실험 폴더에 저장한다.

- [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html): >=1.8의 l1_ratio 규제 API 기준, penalty는 1.8에서 deprecated/1.10 제거 예정. 이전 설치 판본의 인자를 구분한다. 계수는 스케일 변환 입력의 log-odds에 작용하며 인과 중요도가 아니다.
- [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html): class_weight는 학습 가중치, MDI는 불순도 기반 중요도. 큰 category/연속 입력의 편향과 validation permutation을 확인한다. 결측 native 지원 여부는 설치 판본·모델·제약 조건을 확인한다.
- [XGBoost API](https://xgboost.readthedocs.io/en/stable/python/python_api.html): early_stopping_rounds와 eval_set을 함께 설정한다. best_iteration과 best_score는 stopping 결과이며 모든 과적합 방지 보장은 아니다.
- [LightGBM classifier](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html): subsample_freq 기본 0은 행 bagging을 비활성화한다. 예제의 subsample=0.8을 활성화하도록 1을 명시했다. class_weight/scale_pos_weight는 확률 추정에 영향을 줄 수 있어 probability calibration과 실제 임계값 비용을 점검한다.
- [LightGBM 범주형 입력](https://lightgbm.readthedocs.io/en/stable/Advanced-Topics.html): pandas category 또는 정수 코드 등 입력 계약을 따른다. 문자열/object를 아무 설정 없이 넣는다는 뜻은 아니다. 이 데모의 category는 seed를 고정한 무작위 값이며 유의미한 회사 피처가 아니다.
- [LightGBM parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html): scale_pos_weight/is_unbalance를 동시에 쓰지 않는다. binary label=1을 관심 사건으로 정의하고 비율은 train에서만 계산한다.

class_weight/scale_pos_weight는 불균형을 해결했다는 증명이 아니다. label=1이 언제나 소수라는 뜻도 아니다. predict_proba 열 순서는 classes_를 확인해야 하며 여기서는 0/1 이진 합성 입력이다. unknown label을 0으로 만들지 않는다. 그룹·장비·시간 종속 입력은 목적에 맞는 분할을 사용하고 stratify가 종속성을 제거한다고 생각하지 않는다. 임계값 선택도 validation에서 수행한다. early stopping에 쓴 validation 점수는 독립 최종 test가 아니다.

표의 1,000/100,000건 경계는 설명용 탐색 출발점이며 최적 모델을 결정하는 검증된 기준이 아니다. 실제 원본·하드웨어·판본·피처 수·round·parallelism이 없던 속도 숫자는 미측정으로 바꿨다. 학습 시간·ROC-AUC·확률/업무 비용은 같은 입력과 실험 조건에서 비교해야 한다.

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
- [모델 평가 지표 가이드](./model-evaluation.md)
