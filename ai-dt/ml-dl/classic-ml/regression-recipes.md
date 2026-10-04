---
tags: [regression, sklearn, xgboost, linear-model]
level: intermediate
last_updated: 2026-02-14
reviewed_on: 2026-10-04
review_status: partial
document_type: learning
---

# 회귀(Regression) 실전 레시피

> 연속형 수치를 예측하는 대표 회귀 모델들을 한 곳에서 비교하고, 입력·판본·분할 조건을 확인하며 실행하는 교육용 가이드

---

## 왜 필요한가? (Why)

- **연속값 예측**은 머신러닝에서 가장 기본적이고 빈번한 과제이다. 매출 예측, 장비 수명 예측, 공정 파라미터 최적화 등 거의 모든 산업 도메인에서 등장한다.
- 단순 선형 모델부터 트리 기반 앙상블까지, **문제 특성에 맞는 모델을 빠르게 선택하고 비교**할 수 있어야 실무에서 시간을 절약할 수 있다.
- 모델 하나를 학습시키는 것보다 **여러 모델을 동일 기준으로 비교**하고, **잔차(Residual)를 분석**해서 모델의 약점을 파악하는 과정이 더 중요하다.

---

## 핵심 개념 (What)

### 선형 모델 vs 트리 기반 모델

| 구분 | 선형 모델 (Linear) | 트리 기반 모델 (Tree-based) |
|------|--------------------|-----------------------------|
| 가정 | 피처와 타겟 간 선형 관계 | 비선형 관계 자동 학습 |
| 장점 | 해석력, 학습 속도, 안정성 | 비선형 패턴, 피처 상호작용 자동 포착 |
| 단점 | 입력 피처에 선형; 다항/비선형 기저를 추가하면 그 관계를 표현 가능 | 과적합 위험, 해석 어려움 |
| 대표 | OLS, Ridge, Lasso | GBR, XGBoost, LightGBM |

### 편향-분산 트레이드오프 (Bias-Variance Trade-off)

- **편향(Bias)이 높은 모델**: 데이터 패턴을 충분히 학습하지 못함 (과소적합). 선형 모델이 비선형 데이터를 다룰 때 해당.
- **분산(Variance)이 높은 모델**: 학습 데이터에 과하게 맞춰짐 (과적합). 깊은 트리, 복잡한 앙상블이 해당.
- **정규화(Regularization)**: 모델 복잡도를 제한해 분산을 줄이는 기법. Ridge(L2)와 Lasso(L1)가 대표적.

### 정규화 비교

| 방법 | 패널티 | 효과 |
|------|--------|------|
| Ridge (L2) | `alpha * sum(w²)` | 계수를 작게 축소, 다중공선성(Multicollinearity) 완화 |
| Lasso (L1) | `alpha * sum(abs(w))` | 계수를 0으로 만들어 **피처 선택(Feature Selection)** 효과 |
| ElasticNet | L1 + L2 혼합 | 두 장점을 결합 |

---

## 어떻게 사용하는가? (How)

### 공통 셋업

모든 예제에서 공유하는 데이터 로딩 및 전처리 코드이다.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- 데이터 로딩 ---
housing = fetch_california_housing(as_frame=True)
X, y = housing.data, housing.target  # 타겟: 중간 주택 가격 (단위: $100k)

# --- Train/Validation/Test 분리: 최종 test는 선택에 사용하지 않음 ---
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42
)

# --- 규제 선형 모델의 단위 영향을 줄이는 데모; OLS/트리에 일괄 필수 아님 ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
print(f"Features: {housing.feature_names}")
```

### 평가 헬퍼 함수

```python
def evaluate(model, X_test, y_test, model_name="Model"):
    """모델 평가 결과를 딕셔너리로 반환한다."""
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"[{model_name}] RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")
    return {"model": model_name, "rmse": rmse, "mae": mae, "r2": r2, "y_pred": y_pred}
```

---

### 1. 선형 회귀 (Linear Regression) - Basic OLS

가장 단순한 베이스라인. 정규화 없이 최소자승법(Ordinary Least Squares)으로 계수를 구한다.

```python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

result_lr = evaluate(lr, X_val_scaled, y_val, "LinearRegression")

# 계수 확인
coef_df = pd.DataFrame({
    "feature": housing.feature_names,
    "coefficient": lr.coef_
}).sort_values("coefficient", key=abs, ascending=False)
print(coef_df.to_string(index=False))
```

---

### 2. Ridge 회귀 - L2 정규화

L2 패널티를 추가해 계수를 전체적으로 축소한다. `alpha`가 클수록 정규화가 강해진다.

```python
from sklearn.linear_model import Ridge

# fold마다 scaler를 fit하려면 Pipeline 전체를 GridSearchCV에 전달
alphas = np.logspace(-3, 3, 50)  # 0.001 ~ 1000
ridge_search = GridSearchCV(
    Pipeline([("scaler", StandardScaler()), ("model", Ridge())]),
    {"model__alpha": alphas}, cv=5, scoring="neg_mean_squared_error",
)
ridge_search.fit(X_train, y_train)
# 최종 train 전체에 fit된 scaler/모델을 계수·검증 데모에서 사용
ridge = ridge_search.best_estimator_.named_steps["model"]
print(f"선택된 alpha: {ridge.alpha:.4f}")
result_ridge = evaluate(ridge_search.best_estimator_, X_val, y_val, "Ridge")

# 계수 비교: OLS vs Ridge
coef_compare = pd.DataFrame({
    "feature": housing.feature_names,
    "OLS": lr.coef_,
    "Ridge": ridge.coef_,
}).sort_values("OLS", key=abs, ascending=False)
print(coef_compare.to_string(index=False))
```

---

### 3. Lasso 회귀 - L1 정규화 & 피처 선택

L1 패널티를 추가해 일부 피처의 계수를 0으로 만들 수 있다. 이 선택이 업무상 불필요함이나 인과관계를 판정하지는 않는다. **자동 피처 선택** 효과가 핵심이다.

```python
from sklearn.linear_model import Lasso

lasso_search = GridSearchCV(
    Pipeline([("scaler", StandardScaler()),
              ("model", Lasso(max_iter=10000, random_state=42))]),
    {"model__alpha": np.logspace(-3, 1, 50)},
    cv=5, scoring="neg_mean_squared_error",
)
lasso_search.fit(X_train, y_train)
lasso = lasso_search.best_estimator_.named_steps["model"]
print(f"선택된 alpha: {lasso.alpha:.6f}")
result_lasso = evaluate(lasso_search.best_estimator_, X_val, y_val, "Lasso")

# 피처 선택 결과: 계수가 0인 피처 확인
coef_lasso = pd.DataFrame({
    "feature": housing.feature_names,
    "coefficient": lasso.coef_,
    "selected": lasso.coef_ != 0
})
print(f"\n선택된 피처 수: {(lasso.coef_ != 0).sum()} / {len(lasso.coef_)}")
print(coef_lasso.to_string(index=False))
```

**Lasso 정규화 경로 시각화** (alpha 변화에 따른 계수 변화):

```python
from sklearn.linear_model import lasso_path

alphas_path, coefs_path, _ = lasso_path(X_train_scaled, y_train - y_train.mean(), alphas=None)

fig, ax = plt.subplots(figsize=(10, 6))
for i, feat in enumerate(housing.feature_names):
    ax.plot(np.log10(alphas_path), coefs_path[i], label=feat)

ax.axvline(np.log10(lasso.alpha), color="k", linestyle="--", label=f"Best alpha={lasso.alpha:.4f}")
ax.set_xlabel("log10(alpha)")
ax.set_ylabel("Coefficients")
ax.set_title("Lasso 정규화 경로 (Regularization Path)")
ax.legend(fontsize=8, loc="best")
plt.tight_layout()
plt.savefig("lasso_path.png", dpi=150)
plt.show()
```

---

### 4. Gradient Boosting Regressor (sklearn)

sklearn 내장 GBR. 트리를 순차적으로 쌓아 잔차를 줄여나가는 앙상블 기법이다.

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

# 기본 모델 (스케일링 불필요 - 트리 모델이므로 원본 사용 가능)
gbr = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    random_state=42,
)
gbr.fit(X_train, y_train)  # 트리 모델은 스케일링 불필요

result_gbr = evaluate(gbr, X_val, y_val, "GBR")

# 피처 중요도
importance = pd.DataFrame({
    "feature": housing.feature_names,
    "importance": gbr.feature_importances_,
}).sort_values("importance", ascending=False)
print(importance.to_string(index=False))

# 하이퍼파라미터 튜닝 (간단 그리드)
param_grid = {
    "n_estimators": [200, 500],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1],
}
grid = GridSearchCV(
    GradientBoostingRegressor(subsample=0.8, random_state=42),
    param_grid,
    cv=3,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=1,
)
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
print(f"Best CV RMSE: {-grid.best_score_:.4f}")
```

---

### 5. XGBoost Regressor - Early Stopping 포함

XGBoost는 트리 부스팅을 제공한다. 속도·성능은 입력과 파라미터에 따라 비교해야 하며 early stopping은 validation metric으로 학습 round를 선택하는 절차다. 모든 과적합을 자동 방지하지 않는다.

```python
# pip install xgboost
from xgboost import XGBRegressor

xgb = XGBRegressor(
    n_estimators=1000,       # 충분히 크게 설정 (early stopping이 멈춰줌)
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=50,
    eval_metric="rmse",
    reg_alpha=0.1,           # L1 정규화
    reg_lambda=1.0,          # L2 정규화
    random_state=42,
    n_jobs=-1,
    tree_method="hist",      # 빠른 히스토그램 기반 분할
)

# Early stopping: 검증 성능이 50라운드 연속 개선 안 되면 중단
xgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50,
)

print(f"Best iteration: {xgb.best_iteration}")
result_xgb = evaluate(xgb, X_val, y_val, "XGBoost")

# 학습 곡선 시각화
results = xgb.evals_result()
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(results["validation_0"]["rmse"], label="Validation RMSE")
ax.axvline(xgb.best_iteration, color="r", linestyle="--", label=f"Best iter={xgb.best_iteration}")
ax.set_xlabel("Boosting Round")
ax.set_ylabel("RMSE")
ax.set_title("XGBoost 학습 곡선")
ax.legend()
plt.tight_layout()
plt.savefig("xgb_learning_curve.png", dpi=150)
plt.show()
```

---

### 6. 모델 비교 템플릿

선택된 규제 파라미터와 후보 모델을 같은 validation으로 비교한다. 이 비교 표로 test 점수까지 튜닝하지 않는다. 각 모델은 별도 fit하고 마지막에 하나만 보류 test로 평가한다.

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

# --- 모델 정의 ---
models = {
    "LinearRegression": (LinearRegression(), True),                       # (모델, 스케일링 필요 여부)
    "Ridge": (clone(ridge), True),
    "Lasso": (clone(lasso), True),
    "GBR": (GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.1, max_depth=5,
        subsample=0.8, random_state=42), False),
    "XGBoost": (XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        tree_method="hist", n_jobs=-1), False),
}

# --- 학습 및 평가 ---
results = []
for name, (model, needs_scaling) in models.items():
    X_tr = X_train_scaled if needs_scaling else X_train
    X_te = X_val_scaled if needs_scaling else X_val

    model.fit(X_tr, y_train)
    res = evaluate(model, X_te, y_val, name)
    results.append(res)

# --- 비교 테이블 ---
compare_df = pd.DataFrame(results)[["model", "rmse", "mae", "r2"]]
compare_df = compare_df.sort_values("rmse")
print("\n===== Validation 모델 비교 결과 (RMSE 기준 정렬) =====")
print(compare_df.to_string(index=False))

# --- 비교 차트 ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = ["rmse", "mae", "r2"]
titles = ["RMSE (낮을수록 좋음)", "MAE (낮을수록 좋음)", "R² (높을수록 좋음)"]

for ax, metric, title in zip(axes, metrics, titles):
    bars = ax.barh(compare_df["model"], compare_df[metric])
    ax.set_title(title)
    ax.invert_yaxis()
    for bar, val in zip(bars, compare_df[metric]):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f" {val:.4f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.show()

# 선택을 고정한 뒤 보류 test 평가 (현재 train에 fit된 모델을 그대로 사용)
selected_name = compare_df.iloc[0]["model"]
selected_model, selected_scaled = models[selected_name]
final_result = evaluate(
    selected_model, X_test_scaled if selected_scaled else X_test,
    y_test, f"{selected_name} final test",
)
```

---

### 7. 잔차 분석 (Residual Analysis)

잔차(Residual) = 실제값 - 예측값. 잔차 패턴을 통해 모델이 놓치고 있는 신호를 파악한다.

**점검할 잔차 특성 (좋은 일반화의 충분조건이 아님):**
- 평균이 0에 가까움
- 예측값에 대해 무작위로 분포 (패턴 없음)
- 추론 가정상 정규성이 필요한 경우 Q-Q plot 등으로 점검; 모든 예측 모델의 필수 조건은 아님

```python
def residual_analysis(y_true, y_pred, model_name="Model"):
    """잔차 분석 4종 플롯을 생성한다."""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"잔차 분석: {model_name}", fontsize=14)

    # 1) 잔차 vs 예측값 (가장 중요)
    ax = axes[0, 0]
    ax.scatter(y_pred, residuals, alpha=0.3, s=10)
    ax.axhline(y=0, color="r", linestyle="--")
    ax.set_xlabel("예측값")
    ax.set_ylabel("잔차 (실제 - 예측)")
    ax.set_title("잔차 vs 예측값")

    # 2) 잔차 히스토그램
    ax = axes[0, 1]
    ax.hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(x=0, color="r", linestyle="--")
    ax.set_xlabel("잔차")
    ax.set_ylabel("빈도")
    ax.set_title(f"잔차 분포 (평균={residuals.mean():.4f}, 표준편차={residuals.std():.4f})")

    # 3) Q-Q Plot (정규성 시각 점검)
    ax = axes[1, 0]
    from scipy import stats
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    ax.scatter(osm, osr, alpha=0.3, s=10)
    ax.plot(osm, slope * np.array(osm) + intercept, color="r", linestyle="--")
    ax.set_xlabel("이론적 분위수")
    ax.set_ylabel("관측 분위수")
    ax.set_title(f"Q-Q Plot (정규성 시각 점검, R={r:.4f})")

    # 4) 실제값 vs 예측값
    ax = axes[1, 1]
    ax.scatter(y_true, y_pred, alpha=0.3, s=10)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="완벽한 예측")
    ax.set_xlabel("실제값")
    ax.set_ylabel("예측값")
    ax.set_title("실제값 vs 예측값")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"residual_{model_name.lower().replace(' ', '_')}.png", dpi=150)
    plt.show()

    # 수치 요약
    print(f"\n--- {model_name} 잔차 요약 ---")
    print(f"  평균: {residuals.mean():.6f}")
    print(f"  표준편차: {residuals.std():.4f}")
    print(f"  최솟값: {residuals.min():.4f}")
    print(f"  최댓값: {residuals.max():.4f}")
    print(f"  |잔차| > 2*std 비율: {(np.abs(residuals) > 2 * residuals.std()).mean():.2%}")


# --- 사용 예시: 각 모델에 대해 잔차 분석 수행 ---
for res in results:
    residual_analysis(y_val.to_numpy(), res["y_pred"], res["model"])
```

**잔차 분석 해석 가이드:**

| 패턴 | 의미 | 조치 |
|------|------|------|
| 잔차가 부채꼴 모양 | 이분산성(Heteroscedasticity) | 타겟 변환 (log, sqrt) |
| U자 또는 곡선 패턴 | 비선형 관계를 놓침 | 다항 피처 추가 또는 비선형 모델 사용 |
| 클러스터가 보임 | 그룹별 오차·누락 변수 등 가능성 | 피처 엔지니어링 필요 |
| 특정 구간에서 큰 잔차 | 희소 구간·측정 오류·분포 차이 등 후보 원인 | 데이터 수집 또는 이상치 처리 |

---

## 확인 근거와 적용 조건

확인일 **2026-10-04**. scikit-learn 공식 문서 **1.9.1**, XGBoost **3.4.2**, SciPy 문서 **1.18.0**를 기준으로 대조했다. 특정 판본이 최신이라는 보장은 하지 않는다. 문서에 표시된 XGBoost 3.4.2는 PyPI 설치 시 찾을 수 없었고 실제 로컬 제공 패키지는 3.4.1이었다. API 문서 대조와 설치 판본 실행 검증을 구분한다. 실제 3.4.1 import는 macOS libomp.dylib 부재로 실패했다. XGBoost의 early stopping·best_iteration·비교 후보 실행은 미검증이며 AST/공식 API 대조만 수행했다. macOS의 OpenMP 등 설치 의존성을 별도로 충족해야 한다. NumPy·pandas·Matplotlib·SciPy·scikit-learn과 XGBoost 설치가 필요하고 블록은 공통 셋업부터 순차 실행한다. PNG는 현재 디렉토리에 저장하므로 별도 실험 폴더를 사용한다.

- [California housing 자료](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset): 미국 1990 인구조사 기반 지역별 주택 가격 데이터이며 타겟은 100,000 USD 단위다. 최초 fetch는 인터넷/cache가 필요하다. 장비 수명·회사 가격 예측의 실데이터가 아니다.
- [선형 모델](https://scikit-learn.org/stable/modules/linear_model.html): 스케일에 민감한 규제 계수는 동일 학습 입력 기준으로 비교한다. 단순 계수 크기나 0 여부는 인과 중요도 판정이 아니다.
- [누수 방지](https://scikit-learn.org/stable/common_pitfalls.html): train 전체에서 미리 fit한 scaler 뒤 RidgeCV/LassoCV를 실행하면 내부 CV 검증 fold의 통계가 섞인다. 여기서는 원본 train에 Pipeline + GridSearchCV를 적용했다. 원래 내부 CV API를 소개하던 목적은 유지하지만 그 방식의 전처리 흐름을 고쳤다.
- [LassoCV 판본 조건](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html): alphas=None/n_alphas의 폐기 조건은 설치 판본을 확인한다. 이 문서는 명시 alpha grid를 사용한다.
- [lasso_path](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.lasso_path.html): 절편 없는 경로이므로 train 타겟 평균을 빼 비교한다. 경로의 자동 alpha 범위와 search 후보가 동일하다는 보장은 없다.
- [XGBoost API](https://xgboost.readthedocs.io/en/stable/python/python_api.html): early_stopping_rounds와 eval_set을 함께 사용해야 best_iteration을 해석할 수 있다. stopping에 쓰인 validation 점수는 독립 최종 test가 아니다. 이전 판본의 fit 인자와 현재 생성자 API를 구분한다.
- [SciPy probplot](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.probplot.html): 분포의 시각 비교이며 반환 r이 정규성 검정 p-value라는 뜻은 아니다.

모델 비교·잔차 분석은 validation에서 수행하고 최종 선택만 test에서 평가한다. 최종 모델의 train+val 재학습은 이 예제에 구현하지 않았다. 개별 XGBoost 절은 early stopping을, 비교 표의 XGBoost 후보는 고정 500 round를 보여주므로 같은 학습 결과라는 뜻은 아니다. 반복 장비·배치·공간/시간 종속 입력은 랜덤 split 대신 적용 단위에 맞는 분할을 검토한다. 이 예제의 랜덤 CV가 공간적 일반화를 보장하지 않는다. 부스팅은 손실에 따라 음의 gradient를 학습하며 모든 loss에서 단순 잔차만을 학습하는 것은 아니다. feature_importances_는 불순도 기반으로 편향될 수 있어 validation permutation 및 도메인 점검이 필요하다.

잔차 변환 제안은 원인 후보에 대한 점검 방향이다. log는 양수 입력, sqrt는 음수가 아닌 입력 등 domain과 역변환/평가 단위를 확인하고 자동 이상치 삭제로 해석하지 않는다. 예측 성능·속도·최적 alpha·실제 한글 plot 화면은 실데이터 검증이 필요하다.

## 참고 자료 (References)

- [scikit-learn 선형 모델 공식 문서](https://scikit-learn.org/stable/modules/linear_model.html)
- [scikit-learn GradientBoostingRegressor](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)
- [XGBoost 공식 문서](https://xgboost.readthedocs.io/en/stable/)
- [Bias-Variance Tradeoff - MLU Explain](https://mlu-explain.github.io/bias-variance/)
- [Regularization in ML (L1/L2)](https://towardsdatascience.com/regularization-in-machine-learning-76441ddcf99a)

---

## 관련 문서

- [Classic ML 목차](./README.md)
- [데이터 전처리](../../data-handling/)
- [딥러닝 회귀](../deep-learning/)

---

*Last updated: 2026-02-14*
