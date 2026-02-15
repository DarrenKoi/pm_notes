---
tags: [clustering, kmeans, dbscan, sklearn]
level: intermediate
last_updated: 2026-02-14
status: in-progress
---

# 클러스터링(Clustering) 실전 가이드

> 비지도 학습 기반 클러스터링 알고리즘의 핵심 개념과 실전 사용법 정리

## 왜 필요한가? (Why)

- **레이블 없는 데이터에서 패턴을 발견**할 때 클러스터링은 가장 기본적인 접근법이다
- 고객 세그먼테이션, 이상 탐지, 데이터 탐색(EDA) 등 다양한 실무에서 활용된다
- 비지도 학습(Unsupervised Learning)이므로 별도의 라벨링 비용 없이 데이터 구조를 파악할 수 있다
- 반도체 공정에서도 장비 로그/센서 데이터의 그룹화, 결함 유형 분류 등에 적용 가능하다

## 핵심 개념 (What)

클러스터링 알고리즘은 크게 세 가지 접근 방식으로 나뉜다:

| 접근 방식 | 대표 알고리즘 | 핵심 아이디어 |
|-----------|-------------|-------------|
| **분할 기반 (Partitioning)** | KMeans, KMedoids | 데이터를 K개 그룹으로 나눔. 중심점(centroid) 기반 |
| **밀도 기반 (Density-based)** | DBSCAN, HDBSCAN | 밀집 영역을 클러스터로 인식. 비구형 클러스터 탐지 가능 |
| **계층적 (Hierarchical)** | Agglomerative, Divisive | 트리 구조로 클러스터를 병합/분할. 덴드로그램으로 시각화 |

### 주요 용어

- **관성(Inertia)**: 각 데이터 포인트와 소속 클러스터 중심 간 거리 제곱합. 낮을수록 좋음
- **실루엣 점수(Silhouette Score)**: 클러스터 내 응집도와 클러스터 간 분리도의 균형. -1 ~ 1 범위, 높을수록 좋음
- **eps (epsilon)**: DBSCAN에서 이웃 탐색 반경
- **min_samples**: DBSCAN에서 코어 포인트가 되기 위한 최소 이웃 수

## 어떻게 사용하는가? (How)

### 0. 공통 셋업

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# 샘플 데이터 생성
X, y_true = make_blobs(
    n_samples=500,
    centers=4,
    cluster_std=0.8,
    random_state=42,
)

# 스케일링 (클러스터링 전 필수)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"데이터 shape: {X_scaled.shape}")
```

---

### 1. KMeans: 기본 사용법 + Elbow Method + 실루엣 분석

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --- 기본 사용 ---
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

print(f"클러스터 레이블: {np.unique(labels)}")
print(f"클러스터별 샘플 수: {np.bincount(labels)}")
print(f"관성(Inertia): {kmeans.inertia_:.2f}")
print(f"실루엣 점수: {silhouette_score(X_scaled, labels):.3f}")
```

```python
# --- Elbow Method ---
K_range = range(2, 11)
inertias = []
silhouette_scores = []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, km.labels_))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Elbow Plot
axes[0].plot(K_range, inertias, "bo-", linewidth=2)
axes[0].set_xlabel("클러스터 수 (K)")
axes[0].set_ylabel("관성 (Inertia)")
axes[0].set_title("Elbow Method")
axes[0].grid(True, alpha=0.3)

# Silhouette Score Plot
axes[1].plot(K_range, silhouette_scores, "rs-", linewidth=2)
axes[1].set_xlabel("클러스터 수 (K)")
axes[1].set_ylabel("실루엣 점수")
axes[1].set_title("Silhouette Score by K")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("elbow_silhouette.png", dpi=150, bbox_inches="tight")
plt.show()

# 최적 K: Elbow 꺾이는 지점 + 실루엣 점수 최대인 K
best_k = K_range[np.argmax(silhouette_scores)]
print(f"실루엣 기준 최적 K: {best_k}")
```

```python
# --- 실루엣 다이어그램 (개별 샘플 시각화) ---
from sklearn.metrics import silhouette_samples

km = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = km.fit_predict(X_scaled)

silhouette_vals = silhouette_samples(X_scaled, labels)
avg_score = silhouette_score(X_scaled, labels)

fig, ax = plt.subplots(figsize=(8, 6))
y_lower = 10

for i in range(4):
    cluster_vals = silhouette_vals[labels == i]
    cluster_vals.sort()
    y_upper = y_lower + len(cluster_vals)
    ax.fill_betweenx(
        np.arange(y_lower, y_upper),
        0,
        cluster_vals,
        alpha=0.7,
        label=f"Cluster {i}",
    )
    y_lower = y_upper + 10

ax.axvline(x=avg_score, color="red", linestyle="--", label=f"평균: {avg_score:.3f}")
ax.set_xlabel("실루엣 계수")
ax.set_ylabel("클러스터 / 샘플 인덱스")
ax.set_title("실루엣 다이어그램")
ax.legend()
plt.tight_layout()
plt.savefig("silhouette_diagram.png", dpi=150, bbox_inches="tight")
plt.show()
```

---

### 2. DBSCAN: eps/min_samples 튜닝 + 노이즈 처리

```python
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# --- k-distance 그래프로 eps 추정 ---
# k = min_samples 값으로 설정 (일반적으로 2*차원수 권장)
k = 4
nn = NearestNeighbors(n_neighbors=k)
nn.fit(X_scaled)
distances, _ = nn.kneighbors(X_scaled)

# k번째 이웃까지의 거리를 정렬하여 시각화
k_distances = np.sort(distances[:, -1])

plt.figure(figsize=(10, 5))
plt.plot(k_distances, linewidth=2)
plt.xlabel("데이터 포인트 (정렬됨)")
plt.ylabel(f"{k}-번째 이웃 거리")
plt.title("k-Distance Graph (eps 결정용)")
plt.grid(True, alpha=0.3)
# 그래프에서 급격히 꺾이는 지점의 y값이 적절한 eps
plt.axhline(y=0.5, color="red", linestyle="--", label="eps 후보: 0.5")
plt.legend()
plt.tight_layout()
plt.savefig("k_distance_graph.png", dpi=150, bbox_inches="tight")
plt.show()
```

```python
# --- DBSCAN 실행 ---
dbscan = DBSCAN(eps=0.5, min_samples=5)
db_labels = dbscan.fit_predict(X_scaled)

n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise = (db_labels == -1).sum()

print(f"발견된 클러스터 수: {n_clusters}")
print(f"노이즈 포인트 수: {n_noise} ({n_noise / len(db_labels) * 100:.1f}%)")
print(f"클러스터별 샘플 수: {dict(zip(*np.unique(db_labels, return_counts=True)))}")

# 노이즈가 아닌 포인트만 실루엣 점수 계산
mask = db_labels != -1
if len(set(db_labels[mask])) > 1:
    score = silhouette_score(X_scaled[mask], db_labels[mask])
    print(f"실루엣 점수 (노이즈 제외): {score:.3f}")
```

```python
# --- eps / min_samples 조합 비교 ---
eps_values = [0.3, 0.5, 0.7, 1.0]
min_samples_values = [3, 5, 10]

print(f"{'eps':>5} | {'min_samples':>11} | {'n_clusters':>10} | {'n_noise':>7} | {'silhouette':>10}")
print("-" * 60)

for eps in eps_values:
    for ms in min_samples_values:
        db = DBSCAN(eps=eps, min_samples=ms)
        lbl = db.fit_predict(X_scaled)
        n_c = len(set(lbl)) - (1 if -1 in lbl else 0)
        n_n = (lbl == -1).sum()
        mask = lbl != -1
        sil = silhouette_score(X_scaled[mask], lbl[mask]) if len(set(lbl[mask])) > 1 else -1
        print(f"{eps:5.1f} | {ms:11d} | {n_c:10d} | {n_n:7d} | {sil:10.3f}")
```

---

### 3. 계층적 군집화 (Agglomerative) + 덴드로그램

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# --- 덴드로그램 시각화 ---
# scipy linkage 사용 (ward, complete, average, single)
Z = linkage(X_scaled, method="ward")

plt.figure(figsize=(14, 6))
dendrogram(
    Z,
    truncate_mode="lastp",   # 마지막 p개 병합만 표시
    p=30,
    leaf_rotation=90,
    leaf_font_size=8,
    show_contracted=True,
)
plt.title("덴드로그램 (Ward Linkage)")
plt.xlabel("클러스터 인덱스")
plt.ylabel("거리")
plt.axhline(y=15, color="red", linestyle="--", label="커팅 기준선")
plt.legend()
plt.tight_layout()
plt.savefig("dendrogram.png", dpi=150, bbox_inches="tight")
plt.show()
```

```python
# --- Agglomerative Clustering 실행 ---
agg = AgglomerativeClustering(
    n_clusters=4,
    linkage="ward",     # ward | complete | average | single
)
agg_labels = agg.fit_predict(X_scaled)

print(f"클러스터 수: {len(np.unique(agg_labels))}")
print(f"클러스터별 샘플 수: {np.bincount(agg_labels)}")
print(f"실루엣 점수: {silhouette_score(X_scaled, agg_labels):.3f}")
```

```python
# --- Linkage 방법 비교 ---
linkages = ["ward", "complete", "average", "single"]

for link in linkages:
    agg = AgglomerativeClustering(n_clusters=4, linkage=link)
    lbl = agg.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, lbl)
    print(f"Linkage: {link:10s} | 실루엣 점수: {sil:.3f}")
```

---

### 4. 최적 클러스터 수 찾기: Elbow + Silhouette 통합 비교

```python
from sklearn.cluster import KMeans, AgglomerativeClustering

K_range = range(2, 11)
results = {"kmeans_inertia": [], "kmeans_sil": [], "agg_sil": []}

for k in K_range:
    # KMeans
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    results["kmeans_inertia"].append(km.inertia_)
    results["kmeans_sil"].append(silhouette_score(X_scaled, km.labels_))

    # Agglomerative
    agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
    agg_labels = agg.fit_predict(X_scaled)
    results["agg_sil"].append(silhouette_score(X_scaled, agg_labels))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Elbow
axes[0].plot(K_range, results["kmeans_inertia"], "bo-", linewidth=2)
axes[0].set_title("Elbow Method (KMeans)")
axes[0].set_xlabel("K")
axes[0].set_ylabel("Inertia")
axes[0].grid(True, alpha=0.3)

# Silhouette 비교
axes[1].plot(K_range, results["kmeans_sil"], "bo-", label="KMeans", linewidth=2)
axes[1].plot(K_range, results["agg_sil"], "rs-", label="Agglomerative", linewidth=2)
axes[1].set_title("실루엣 점수 비교")
axes[1].set_xlabel("K")
axes[1].set_ylabel("Silhouette Score")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("optimal_k_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

best_km = list(K_range)[np.argmax(results["kmeans_sil"])]
best_agg = list(K_range)[np.argmax(results["agg_sil"])]
print(f"KMeans 최적 K: {best_km} (실루엣: {max(results['kmeans_sil']):.3f})")
print(f"Agglomerative 최적 K: {best_agg} (실루엣: {max(results['agg_sil']):.3f})")
```

---

### 5. 클러스터링 결과 시각화: PCA / t-SNE 차원 축소

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 클러스터링 수행
km = KMeans(n_clusters=4, random_state=42, n_init=10)
km_labels = km.fit_predict(X_scaled)

# --- PCA 2D 시각화 ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_pca[:, 0], X_pca[:, 1],
    c=km_labels, cmap="viridis", alpha=0.6, edgecolors="k", linewidth=0.3, s=40,
)
plt.colorbar(scatter, label="클러스터")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
plt.title("KMeans 클러스터링 결과 (PCA 2D)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("clustering_pca.png", dpi=150, bbox_inches="tight")
plt.show()
```

```python
# --- t-SNE 2D 시각화 ---
tsne = TSNE(
    n_components=2,
    perplexity=30,       # 데이터 수에 따라 5~50 조정
    random_state=42,
    n_iter=1000,
)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_tsne[:, 0], X_tsne[:, 1],
    c=km_labels, cmap="viridis", alpha=0.6, edgecolors="k", linewidth=0.3, s=40,
)
plt.colorbar(scatter, label="클러스터")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("KMeans 클러스터링 결과 (t-SNE 2D)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("clustering_tsne.png", dpi=150, bbox_inches="tight")
plt.show()
```

```python
# --- 알고리즘별 결과 비교 시각화 (PCA 기준) ---
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

algorithms = {
    "KMeans (K=4)": KMeans(n_clusters=4, random_state=42, n_init=10),
    "DBSCAN (eps=0.5)": DBSCAN(eps=0.5, min_samples=5),
    "Agglomerative (K=4)": AgglomerativeClustering(n_clusters=4, linkage="ward"),
}

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, algo) in zip(axes, algorithms.items()):
    labels = algo.fit_predict(X_scaled)
    scatter = ax.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=labels, cmap="viridis", alpha=0.6, edgecolors="k", linewidth=0.3, s=30,
    )
    ax.set_title(name)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)

plt.suptitle("알고리즘별 클러스터링 결과 비교", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("algorithm_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
```

---

### 6. 알고리즘 비교표: 언제 어떤 알고리즘을 쓸 것인가

| 기준 | KMeans | DBSCAN | Agglomerative |
|------|--------|--------|---------------|
| **클러스터 수 사전 지정** | 필요 (K) | 불필요 | 필요 (K) |
| **클러스터 형태** | 구형(spherical) | 비정형 가능 | 다양 (linkage에 따라) |
| **노이즈/이상치 처리** | 취약 | 강건 (노이즈 라벨 -1) | 취약 |
| **대용량 데이터** | 빠름 O(nK) | 보통 O(n log n) | 느림 O(n^2) ~ O(n^3) |
| **하이퍼파라미터** | K | eps, min_samples | K, linkage |
| **결정론적** | 아니오 (초기값 의존) | 예 | 예 |
| **추천 상황** | 대용량, 구형 클러스터 | 밀도 차이 큰 데이터, 이상치 존재 | 계층 구조 탐색, 소규모 데이터 |

### 실무 선택 가이드

```
데이터 특성 파악
├── 클러스터가 구형이고 크기 비슷 → KMeans
├── 비정형 클러스터 or 노이즈 많음 → DBSCAN / HDBSCAN
├── 계층 구조가 중요 → Agglomerative + 덴드로그램
├── 클러스터 수를 모름
│   ├── 밀도 기반 탐색 → DBSCAN
│   └── Elbow / Silhouette로 K 탐색 → KMeans
└── 데이터가 매우 큼 (>100K)
    ├── KMeans 또는 MiniBatchKMeans
    └── HDBSCAN (DBSCAN보다 확장성 좋음)
```

---

## 참고 자료 (References)

- [scikit-learn Clustering 공식 문서](https://scikit-learn.org/stable/modules/clustering.html)
- [scikit-learn Clustering 비교 예제](https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html)
- [Silhouette Analysis (sklearn)](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)
- [DBSCAN 파라미터 튜닝 가이드](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
- [HDBSCAN 공식 문서](https://hdbscan.readthedocs.io/en/latest/)

## 관련 문서

- [상위 폴더](../README.md)
- [데이터 전처리](../../data-handling/)
