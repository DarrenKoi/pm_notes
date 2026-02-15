---
tags: [regression, sklearn, xgboost, linear-model]
level: intermediate
last_updated: 2026-02-14
status: in-progress
---

# 회귀(Regression) 실전 레시피

> 연속형 수치를 예측하는 대표 회귀 모델들을 한 곳에서 비교하고, 복사-붙여넣기로 바로 사용할 수 있는 실전 가이드

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
| 단점 | 비선형 패턴 학습 불가 | 과적합 위험, 해석 어려움 |
| 대표 | OLS, Ridge, Lasso | GBR, XGBoost, LightGBM |

### 편향-분산 트레이드오프 (Bias-Variance Trade-off)

- **편향(Bias)이 높은 모델**: 데이터 패턴을 충분히 학습하지 못함 (과소적합). 선형 모델이 비선형 데이터를 다룰 때 해당.
- **분산(Variance)이 높은 모델**: 학습 데이터에 과하게 맞춰짐 (과적합). 깊은 트리, 복잡한 앙상블이 해당.
- **정규화(Regularization)**: 모델 복잡도를 제한해 분산을 줄이는 기법. Ridge(L2)와 Lasso(L1)가 대표적.

### 정규화 비교

| 방법 | 패널티 | 효과 |
|------|--------|------|
| Ridge (L2) | `alpha * sum(w²)` | 계수를 작게 축소, 다중공선성(Multicollinearity) 완화 |
| Lasso (L1) | `alpha * sum(|w|)` | 계수를 0으로 만들어 **피처 선택(Feature Selection)** 효과 |
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- 데이터 로딩 ---
housing = fetch_california_housing(as_frame=True)
X, y = housing.data, housing.target  # 타겟: 중간 주택 가격 (단위: $100k)

# --- 학습/테스트 분리 ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- 스케일링 (선형 모델에 필수, 트리 모델에도 무해) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
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

result_lr = evaluate(lr, X_test_scaled, y_test, "LinearRegression")

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
from sklearn.linear_model import RidgeCV

# RidgeCV: 내부 교차 검증으로 최적 alpha 자동 선택
alphas = np.logspace(-3, 3, 50)  # 0.001 ~ 1000

ridge = RidgeCV(alphas=alphas, cv=5, scoring="neg_mean_squared_error")
ridge.fit(X_train_scaled, y_train)

print(f"선택된 alpha: {ridge.alpha_:.4f}")
result_ridge = evaluate(ridge, X_test_scaled, y_test, "Ridge")

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

L1 패널티를 추가해 불필요한 피처의 계수를 정확히 0으로 만든다. **자동 피처 선택** 효과가 핵심이다.

```python
from sklearn.linear_model import LassoCV

lasso = LassoCV(alphas=None, cv=5, max_iter=10000, random_state=42)
# alphas=None → 자동으로 alpha 경로를 생성
lasso.fit(X_train_scaled, y_train)

print(f"선택된 alpha: {lasso.alpha_:.6f}")
result_lasso = evaluate(lasso, X_test_scaled, y_test, "Lasso")

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

alphas_path, coefs_path, _ = lasso_path(X_train_scaled, y_train, alphas=None)

fig, ax = plt.subplots(figsize=(10, 6))
for i, feat in enumerate(housing.feature_names):
    ax.plot(np.log10(alphas_path), coefs_path[i], label=feat)

ax.axvline(np.log10(lasso.alpha_), color="k", linestyle="--", label=f"Best alpha={lasso.alpha_:.4f}")
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

result_gbr = evaluate(gbr, X_test, y_test, "GBR")

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

XGBoost는 속도와 성능 모두 뛰어나며, early stopping으로 과적합을 자동 방지한다.

```python
# pip install xgboost
from xgboost import XGBRegressor

xgb = XGBRegressor(
    n_estimators=1000,       # 충분히 크게 설정 (early stopping이 멈춰줌)
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,           # L1 정규화
    reg_lambda=1.0,          # L2 정규화
    random_state=42,
    n_jobs=-1,
    tree_method="hist",      # 빠른 히스토그램 기반 분할
)

# Early stopping: 검증 성능이 50라운드 연속 개선 안 되면 중단
xgb.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50,
)

print(f"Best iteration: {xgb.best_iteration}")
result_xgb = evaluate(xgb, X_test, y_test, "XGBoost")

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

위에서 학습한 모든 모델을 동일 기준으로 한 번에 비교한다.

```python
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

# --- 모델 정의 ---
models = {
    "LinearRegression": (LinearRegression(), True),                       # (모델, 스케일링 필요 여부)
    "Ridge": (RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5), True),
    "Lasso": (LassoCV(cv=5, max_iter=10000, random_state=42), True),
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
    X_te = X_test_scaled if needs_scaling else X_test

    model.fit(X_tr, y_train)
    res = evaluate(model, X_te, y_test, name)
    results.append(res)

# --- 비교 테이블 ---
compare_df = pd.DataFrame(results)[["model", "rmse", "mae", "r2"]]
compare_df = compare_df.sort_values("rmse")
print("\n===== 모델 비교 결과 (RMSE 기준 정렬) =====")
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
```

---

### 7. 잔차 분석 (Residual Analysis)

잔차(Residual) = 실제값 - 예측값. 잔차 패턴을 통해 모델이 놓치고 있는 신호를 파악한다.

**좋은 모델의 잔차 특성:**
- 평균이 0에 가까움
- 예측값에 대해 무작위로 분포 (패턴 없음)
- 정규분포에 가까움

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

    # 3) Q-Q Plot (정규성 검정)
    ax = axes[1, 0]
    from scipy import stats
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    ax.scatter(osm, osr, alpha=0.3, s=10)
    ax.plot(osm, slope * np.array(osm) + intercept, color="r", linestyle="--")
    ax.set_xlabel("이론적 분위수")
    ax.set_ylabel("관측 분위수")
    ax.set_title(f"Q-Q Plot (정규성 검정, R={r:.4f})")

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
    residual_analysis(y_test.values, res["y_pred"], res["model"])
```

**잔차 분석 해석 가이드:**

| 패턴 | 의미 | 조치 |
|------|------|------|
| 잔차가 부채꼴 모양 | 이분산성(Heteroscedasticity) | 타겟 변환 (log, sqrt) |
| U자 또는 곡선 패턴 | 비선형 관계를 놓침 | 다항 피처 추가 또는 비선형 모델 사용 |
| 클러스터가 보임 | 숨겨진 범주형 변수 존재 | 피처 엔지니어링 필요 |
| 특정 구간에서 큰 잔차 | 해당 구간 데이터 부족 또는 이상치 | 데이터 수집 또는 이상치 처리 |

---

## 참고 자료 (References)

- [scikit-learn 선형 모델 공식 문서](https://scikit-learn.org/stable/modules/linear_model.html)
- [scikit-learn GradientBoostingRegressor](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)
- [XGBoost 공식 문서](https://xgboost.readthedocs.io/en/stable/)
- [Bias-Variance Tradeoff - MLU Explain](https://mlu-explain.github.io/bias-variance/)
- [Regularization in ML (L1/L2)](https://towardsdatascience.com/regularization-in-machine-learning-76441ddcf99a)

---

## 관련 문서

- [상위: Classic ML 개요](../classic-ml/)
- [데이터 전처리](../../data-handling/)
- [딥러닝 회귀](../deep-learning/)

---

*Last updated: 2026-02-14*
