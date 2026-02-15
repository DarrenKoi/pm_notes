---
tags: [evaluation, metrics, roc-auc, confusion-matrix]
level: intermediate
last_updated: 2026-02-14
status: in-progress
---

# 모델 평가 메트릭 (Model Evaluation Metrics)

> 올바른 메트릭을 선택해야 올바른 모델을 얻는다 — 메트릭이 곧 모델의 방향이다.

## 왜 필요한가? (Why)

- **잘못된 메트릭 선택은 프로덕션에서 나쁜 모델을 만든다.** 예를 들어 불균형 데이터셋에서 Accuracy만 보면 99%가 나오지만, 실제로는 소수 클래스를 전혀 맞추지 못하는 모델일 수 있다.
- 비즈니스 목표에 따라 최적화해야 할 메트릭이 달라진다:
  - 스팸 필터 → Precision 중시 (정상 메일을 스팸으로 분류하면 안 됨)
  - 질병 진단 → Recall 중시 (환자를 놓치면 안 됨)
  - 수율 예측 → RMSE/MAE로 오차 크기 파악
- 모델 비교, 하이퍼파라미터 튜닝, A/B 테스트 등 모든 ML 워크플로우의 기반이 되는 핵심 스킬이다.

---

## 핵심 개념 (What)

### 분류(Classification) vs 회귀(Regression) 메트릭

| 문제 유형 | 대표 메트릭 | 사용 시점 |
|-----------|------------|----------|
| 이진 분류 | Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC | 클래스 2개 |
| 다중 클래스 분류 | Macro/Micro/Weighted F1, Multi-class ROC | 클래스 3개 이상 |
| 회귀 | MSE, RMSE, MAE, R², MAPE | 연속값 예측 |
| 클러스터링 | Silhouette, Calinski-Harabasz, Davies-Bouldin | 비지도 학습 |

### 언제 어떤 메트릭을 사용할까?

- **데이터가 균형** → Accuracy, F1-macro
- **데이터가 불균형** → F1, PR-AUC, ROC-AUC
- **False Positive 비용이 큼** → Precision 우선
- **False Negative 비용이 큼** → Recall 우선
- **확률 기반 랭킹** → ROC-AUC, Log Loss
- **오차 크기가 중요** → RMSE, MAE
- **비율 기반 해석** → MAPE, R²

---

## 어떻게 사용하는가? (How)

### 공통 임포트 및 데이터 준비

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

# 한글 폰트 설정 (필요 시)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['figure.dpi'] = 100

# 분류용 샘플 데이터
X_clf, y_clf = make_classification(
    n_samples=1000, n_features=20, n_classes=2,
    weights=[0.7, 0.3], random_state=42
)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clf, y_clf, test_size=0.3, random_state=42
)

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_c, y_train_c)
y_pred_c = clf.predict(X_test_c)
y_prob_c = clf.predict_proba(X_test_c)[:, 1]

# 회귀용 샘플 데이터
X_reg, y_reg = make_regression(
    n_samples=500, n_features=10, noise=15, random_state=42
)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train_r, y_train_r)
y_pred_r = reg.predict(X_test_r)
```

---

### 1. 분류 메트릭 (Classification Metrics)

#### 1-1. 혼동 행렬 (Confusion Matrix)

혼동 행렬은 분류 모델의 성능을 한눈에 파악할 수 있는 가장 기본적인 도구다.

```
              예측 Positive    예측 Negative
실제 Positive     TP              FN
실제 Negative     FP              TN
```

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 혼동 행렬 계산
cm = confusion_matrix(y_test_c, y_pred_c)
print("Confusion Matrix:\n", cm)

# 히트맵 시각화 (ConfusionMatrixDisplay 사용)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 방법 1: ConfusionMatrixDisplay (sklearn 공식 API)
ConfusionMatrixDisplay.from_estimator(
    clf, X_test_c, y_test_c,
    display_labels=['Negative', 'Positive'],
    cmap='Blues',
    ax=axes[0]
)
axes[0].set_title('Confusion Matrix (Counts)')

# 방법 2: 정규화된 혼동 행렬
ConfusionMatrixDisplay.from_estimator(
    clf, X_test_c, y_test_c,
    display_labels=['Negative', 'Positive'],
    normalize='true',  # 'true', 'pred', 'all' 중 선택
    cmap='Blues',
    values_format='.2%',
    ax=axes[1]
)
axes[1].set_title('Confusion Matrix (Normalized)')

plt.tight_layout()
plt.savefig('confusion_matrix.png', bbox_inches='tight')
plt.show()
```

#### 1-2. 정밀도(Precision), 재현율(Recall), F1 Score

```python
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    PrecisionRecallDisplay,
    average_precision_score
)

# classification_report: 한 번에 모든 분류 메트릭 확인
print(classification_report(
    y_test_c, y_pred_c,
    target_names=['Negative', 'Positive'],
    digits=4
))

# 출력 예시:
#               precision    recall  f1-score   support
#     Negative     0.8800    0.9500    0.9137       200
#     Positive     0.8667    0.7222    0.7879       100
#     accuracy                         0.8733       300
#    macro avg     0.8733    0.8361    0.8508       300
# weighted avg    0.8756    0.8733    0.8717       300
```

**Precision-Recall Curve 시각화:**

```python
# Precision-Recall Curve
fig, ax = plt.subplots(figsize=(8, 6))

PrecisionRecallDisplay.from_estimator(
    clf, X_test_c, y_test_c, ax=ax, name='Logistic Regression'
)

# AP(Average Precision) Score 표시
ap_score = average_precision_score(y_test_c, y_prob_c)
ax.set_title(f'Precision-Recall Curve (AP = {ap_score:.4f})')
ax.legend(loc='lower left')
plt.grid(True, alpha=0.3)
plt.savefig('precision_recall_curve.png', bbox_inches='tight')
plt.show()
```

**Threshold에 따른 Precision/Recall 변화:**

```python
precision, recall, thresholds = precision_recall_curve(y_test_c, y_prob_c)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(thresholds, precision[:-1], 'b-', label='Precision')
ax.plot(thresholds, recall[:-1], 'r-', label='Recall')
ax.set_xlabel('Threshold')
ax.set_ylabel('Score')
ax.set_title('Precision & Recall vs Threshold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('precision_recall_vs_threshold.png', bbox_inches='tight')
plt.show()
```

#### 1-3. ROC-AUC

ROC(Receiver Operating Characteristic) 곡선은 다양한 임계값(threshold)에서 TPR(True Positive Rate)과 FPR(False Positive Rate)의 관계를 보여준다.

```python
from sklearn.metrics import roc_auc_score, RocCurveDisplay

# ROC-AUC Score
auc_score = roc_auc_score(y_test_c, y_prob_c)
print(f"ROC-AUC Score: {auc_score:.4f}")

# ROC Curve 시각화
fig, ax = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_estimator(
    clf, X_test_c, y_test_c,
    ax=ax, name='Logistic Regression'
)
ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.5)')
ax.set_title(f'ROC Curve (AUC = {auc_score:.4f})')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.savefig('roc_curve.png', bbox_inches='tight')
plt.show()
```

**다중 클래스 ROC Curve (One-vs-Rest):**

```python
from sklearn.datasets import make_classification
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier

# 다중 클래스 데이터 생성
X_multi, y_multi = make_classification(
    n_samples=1000, n_features=20, n_classes=3,
    n_informative=10, random_state=42
)
y_multi_bin = label_binarize(y_multi, classes=[0, 1, 2])
n_classes = y_multi_bin.shape[1]

X_tr, X_te, y_tr, y_te = train_test_split(
    X_multi, y_multi_bin, test_size=0.3, random_state=42
)

# OvR 분류기 학습
ovr_clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=42))
ovr_clf.fit(X_tr, y_tr)
y_score = ovr_clf.predict_proba(X_te)

# 각 클래스별 ROC Curve
fig, ax = plt.subplots(figsize=(8, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
class_names = ['Class 0', 'Class 1', 'Class 2']

for i, (color, name) in enumerate(zip(colors, class_names)):
    fpr, tpr, _ = roc_curve(y_te[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={roc_auc:.3f})')

# Micro-average ROC
fpr_micro, tpr_micro, _ = roc_curve(y_te.ravel(), y_score.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)
ax.plot(fpr_micro, tpr_micro, 'k--', lw=2,
        label=f'Micro-average (AUC={roc_auc_micro:.3f})')

ax.plot([0, 1], [0, 1], 'gray', linestyle=':', alpha=0.5)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Multi-class ROC Curve (One-vs-Rest)')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.savefig('multiclass_roc.png', bbox_inches='tight')
plt.show()
```

#### 1-4. 다중 클래스 평균화 방식 (Macro / Micro / Weighted)

```python
from sklearn.metrics import f1_score, precision_score, recall_score

# 다중 클래스 예측 (비교용)
X_multi_flat, y_multi_flat = make_classification(
    n_samples=1000, n_features=20, n_classes=3,
    n_informative=10, random_state=42
)
X_tr_m, X_te_m, y_tr_m, y_te_m = train_test_split(
    X_multi_flat, y_multi_flat, test_size=0.3, random_state=42
)
mc_clf = LogisticRegression(max_iter=1000, random_state=42)
mc_clf.fit(X_tr_m, y_tr_m)
y_pred_m = mc_clf.predict(X_te_m)

# 평균화 방식 비교
averaging_methods = ['macro', 'micro', 'weighted']

print("=" * 60)
print(f"{'Averaging':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
print("=" * 60)
for avg in averaging_methods:
    p = precision_score(y_te_m, y_pred_m, average=avg)
    r = recall_score(y_te_m, y_pred_m, average=avg)
    f = f1_score(y_te_m, y_pred_m, average=avg)
    print(f"{avg:<12} {p:<12.4f} {r:<12.4f} {f:<12.4f}")
print("=" * 60)

# 설명:
# macro   : 각 클래스 메트릭의 단순 평균 (클래스 불균형에 민감)
# micro   : 전체 TP, FP, FN 합산 후 계산 (큰 클래스에 가중)
# weighted: 각 클래스의 support(샘플 수)로 가중 평균
```

---

### 2. 회귀 메트릭 (Regression Metrics)

#### 2-1. MSE, RMSE, MAE, R², MAPE

```python
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)

# 모든 회귀 메트릭 계산
mse = mean_squared_error(y_test_r, y_pred_r)
rmse = mean_squared_error(y_test_r, y_pred_r, squared=False)
mae = mean_absolute_error(y_test_r, y_pred_r)
r2 = r2_score(y_test_r, y_pred_r)
mape = mean_absolute_percentage_error(y_test_r, y_pred_r)

print("=" * 40)
print("회귀 모델 평가 결과")
print("=" * 40)
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")
print(f"MAPE : {mape:.4f} ({mape*100:.2f}%)")
print("=" * 40)
```

**메트릭 해석 가이드:**

| 메트릭 | 범위 | 해석 |
|--------|------|------|
| MSE | [0, ∞) | 큰 오차에 패널티가 큼 (이상치에 민감) |
| RMSE | [0, ∞) | MSE의 제곱근, 원래 단위와 동일 |
| MAE | [0, ∞) | 직관적인 평균 오차, 이상치에 강건 |
| R² | (-∞, 1] | 1에 가까울수록 좋음, 음수면 평균보다 못함 |
| MAPE | [0, ∞) | 비율 기반, 스케일 독립적, 0 근처 값에 주의 |

#### 2-2. 잔차 플롯 (Residual Plot)

잔차(residual = 실제값 - 예측값) 분석은 모델의 패턴을 진단하는 핵심 도구다.

```python
residuals = y_test_r - y_pred_r

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1) 예측값 vs 잔차
axes[0, 0].scatter(y_pred_r, residuals, alpha=0.5, edgecolors='k', linewidths=0.5)
axes[0, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 0].set_xlabel('Predicted Values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residuals vs Predicted')
axes[0, 0].grid(True, alpha=0.3)

# 2) 실제값 vs 예측값 (45도 라인)
axes[0, 1].scatter(y_test_r, y_pred_r, alpha=0.5, edgecolors='k', linewidths=0.5)
min_val = min(y_test_r.min(), y_pred_r.min())
max_val = max(y_test_r.max(), y_pred_r.max())
axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
axes[0, 1].set_xlabel('Actual Values')
axes[0, 1].set_ylabel('Predicted Values')
axes[0, 1].set_title(f'Actual vs Predicted (R²={r2:.4f})')
axes[0, 1].grid(True, alpha=0.3)

# 3) 잔차 히스토그램 (정규성 확인)
axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_xlabel('Residuals')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Residual Distribution')
axes[1, 0].grid(True, alpha=0.3)

# 4) QQ Plot (정규성 확인)
from scipy import stats
(osm, osr), (slope, intercept, r_val) = stats.probplot(residuals, dist="norm")
axes[1, 1].scatter(osm, osr, alpha=0.5, edgecolors='k', linewidths=0.5)
axes[1, 1].plot(osm, slope * np.array(osm) + intercept, 'r-', lw=2)
axes[1, 1].set_xlabel('Theoretical Quantiles')
axes[1, 1].set_ylabel('Sample Quantiles')
axes[1, 1].set_title('Q-Q Plot (Normality Check)')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Residual Analysis Dashboard', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('residual_analysis.png', bbox_inches='tight')
plt.show()
```

**잔차 플롯 해석:**
- 잔차가 0 주위에 **랜덤하게 분포** → 모델이 잘 적합됨
- 잔차에 **패턴(곡선, 부채꼴)이 보임** → 비선형 관계 누락, 이분산성(heteroscedasticity)
- 잔차가 **정규분포를 따르지 않음** → 변수 변환 또는 다른 모델 고려

---

### 3. 클러스터링 메트릭 (Clustering Metrics)

클러스터링은 정답 레이블이 없으므로 **내부 평가 지표(Internal Validation)**를 사용한다.

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    calinski_harabasz_score,
    davies_bouldin_score
)

# 클러스터링용 데이터
X_clust, _ = make_blobs(n_samples=500, centers=4, n_features=2, random_state=42)

# 다양한 K값에 대해 메트릭 비교
k_range = range(2, 8)
silhouette_scores = []
calinski_scores = []
davies_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_clust)

    silhouette_scores.append(silhouette_score(X_clust, labels))
    calinski_scores.append(calinski_harabasz_score(X_clust, labels))
    davies_scores.append(davies_bouldin_score(X_clust, labels))

# 메트릭 비교 시각화
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(k_range, silhouette_scores, 'bo-', lw=2)
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Silhouette Score')
axes[0].set_title('Silhouette Score (higher = better)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(k_range, calinski_scores, 'ro-', lw=2)
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Calinski-Harabasz Index')
axes[1].set_title('Calinski-Harabasz (higher = better)')
axes[1].grid(True, alpha=0.3)

axes[2].plot(k_range, davies_scores, 'go-', lw=2)
axes[2].set_xlabel('Number of Clusters (K)')
axes[2].set_ylabel('Davies-Bouldin Index')
axes[2].set_title('Davies-Bouldin (lower = better)')
axes[2].grid(True, alpha=0.3)

plt.suptitle('Clustering Metric Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('clustering_metrics.png', bbox_inches='tight')
plt.show()

# 결과 출력
print(f"{'K':<5} {'Silhouette':<14} {'Calinski-H':<14} {'Davies-B':<14}")
print("=" * 47)
for k, s, c, d in zip(k_range, silhouette_scores, calinski_scores, davies_scores):
    print(f"{k:<5} {s:<14.4f} {c:<14.2f} {d:<14.4f}")
```

**Silhouette 분석 상세 시각화:**

```python
from matplotlib import cm

k_best = 4
km = KMeans(n_clusters=k_best, random_state=42, n_init=10)
labels = km.fit_predict(X_clust)
silhouette_vals = silhouette_samples(X_clust, labels)
avg_score = silhouette_score(X_clust, labels)

fig, ax = plt.subplots(figsize=(8, 6))
y_lower = 10

for i in range(k_best):
    cluster_vals = silhouette_vals[labels == i]
    cluster_vals.sort()

    y_upper = y_lower + len(cluster_vals)
    color = cm.nipy_spectral(float(i) / k_best)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_vals,
                      facecolor=color, edgecolor=color, alpha=0.7)
    ax.text(-0.05, y_lower + 0.5 * len(cluster_vals), str(i))
    y_lower = y_upper + 10

ax.axvline(x=avg_score, color='red', linestyle='--',
           label=f'Average: {avg_score:.3f}')
ax.set_xlabel('Silhouette Coefficient')
ax.set_ylabel('Cluster')
ax.set_title(f'Silhouette Analysis (K={k_best})')
ax.legend()
plt.savefig('silhouette_analysis.png', bbox_inches='tight')
plt.show()
```

| 메트릭 | 범위 | 좋은 값 | 설명 |
|--------|------|---------|------|
| Silhouette Score | [-1, 1] | 1에 가까울수록 | 클러스터 내 응집도 vs 클러스터 간 분리도 |
| Calinski-Harabasz | [0, ∞) | 높을수록 | 클러스터 간 분산 / 클러스터 내 분산 비율 |
| Davies-Bouldin | [0, ∞) | 낮을수록 | 클러스터 간 유사도의 평균 |

---

### 4. 메트릭 선택 가이드 (Decision Table)

| 문제 유형 | 비즈니스 목표 | 추천 메트릭 | 이유 |
|-----------|-------------|------------|------|
| 이진 분류 (균형) | 전반적 성능 | **Accuracy, F1** | 클래스 비율이 비슷하면 Accuracy도 유효 |
| 이진 분류 (불균형) | 소수 클래스 탐지 | **F1, PR-AUC** | ROC-AUC는 불균형에서 과대평가 가능 |
| 이진 분류 | FP 최소화 (스팸 필터) | **Precision** | 정상을 스팸으로 분류하면 안 됨 |
| 이진 분류 | FN 최소화 (질병 진단) | **Recall** | 환자를 놓치면 안 됨 |
| 이진 분류 | 확률 랭킹 | **ROC-AUC, Log Loss** | 예측 확률의 품질 평가 |
| 다중 클래스 (균형) | 전반적 성능 | **Macro F1** | 모든 클래스에 동등한 가중치 |
| 다중 클래스 (불균형) | 전체 정확도 | **Weighted F1** | 클래스 빈도 반영 |
| 회귀 | 큰 오차 패널티 | **RMSE** | 이상치에 민감하게 반응 |
| 회귀 | 강건한 오차 측정 | **MAE** | 이상치에 덜 민감 |
| 회귀 | 설명력 | **R²** | 분산 설명 비율 |
| 회귀 | 스케일 독립적 비교 | **MAPE** | 서로 다른 단위의 모델 비교 |
| 클러스터링 | 최적 K 탐색 | **Silhouette + Elbow** | 종합적으로 판단 |

---

### 5. 커스텀 메트릭 (Custom Scorer with `make_scorer`)

sklearn의 `make_scorer`를 사용하면 교차 검증, 그리드 서치 등에서 자신만의 메트릭을 사용할 수 있다.

```python
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.model_selection import cross_val_score, GridSearchCV

# 예시 1: F-beta Score (beta=2로 Recall에 가중)
f2_scorer = make_scorer(fbeta_score, beta=2)

scores = cross_val_score(clf, X_train_c, y_train_c, cv=5, scoring=f2_scorer)
print(f"F2 Score (CV): {scores.mean():.4f} (+/- {scores.std():.4f})")
```

```python
# 예시 2: 완전 커스텀 메트릭 정의
def custom_profit_metric(y_true, y_pred):
    """
    비즈니스 수익 기반 커스텀 메트릭.
    - TP: +100 (정상 탐지 → 수익)
    - FP: -50  (오탐 → 비용)
    - FN: -200 (미탐 → 큰 손실)
    - TN: 0    (정상 무시 → 비용 없음)
    """
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    profit = tp * 100 + fp * (-50) + fn * (-200)
    return profit

# make_scorer로 감싸기 (greater_is_better=True가 기본값)
profit_scorer = make_scorer(custom_profit_metric)

# GridSearchCV에서 커스텀 메트릭 사용
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2']
}

grid_search = GridSearchCV(
    LogisticRegression(max_iter=1000, solver='saga', random_state=42),
    param_grid,
    scoring=profit_scorer,
    cv=5,
    refit=True
)
grid_search.fit(X_train_c, y_train_c)

print(f"Best params: {grid_search.best_params_}")
print(f"Best profit score (CV): {grid_search.best_score_:.2f}")
```

```python
# 예시 3: 여러 메트릭을 동시에 평가
from sklearn.model_selection import cross_validate

scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc',
    'profit': profit_scorer
}

results = cross_validate(
    clf, X_train_c, y_train_c,
    scoring=scoring, cv=5, return_train_score=True
)

print(f"{'Metric':<15} {'Train (mean)':<15} {'Test (mean)':<15}")
print("=" * 45)
for metric in scoring:
    train_key = f'train_{metric}'
    test_key = f'test_{metric}'
    print(f"{metric:<15} {results[train_key].mean():<15.4f} {results[test_key].mean():<15.4f}")
```

---

## 참고 자료 (References)

- [scikit-learn Metrics and Scoring 공식 문서](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [scikit-learn Classification Metrics](https://scikit-learn.org/stable/modules/classes.html#classification-metrics)
- [scikit-learn Regression Metrics](https://scikit-learn.org/stable/modules/classes.html#regression-metrics)
- [scikit-learn Clustering Metrics](https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation)
- [Google ML Crash Course - Classification Metrics](https://developers.google.com/machine-learning/crash-course/classification)

---

## 관련 문서

- [상위 개념: Classic ML Overview](./README.md)
- [교차 검증(Cross Validation)](./cross-validation.md)
- [하이퍼파라미터 튜닝](./hyperparameter-tuning.md)
- [Feature Engineering](./feature-engineering.md)
