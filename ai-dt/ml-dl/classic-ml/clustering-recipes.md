---
tags: [clustering, kmeans, dbscan, sklearn]
level: intermediate
last_updated: 2026-02-14
reviewed_on: 2026-10-04
review_status: partial
document_type: learning
category_major: "AI·DT"
category_middle: "머신러닝·딥러닝"
category_minor: "클래식 머신러닝"
note_kind: "학습"
classified_on: "2026-10-05"
---

# 클러스터링(Clustering) 실전 가이드

> 비지도 학습 기반 클러스터링 알고리즘의 핵심 개념과 실전 사용법 정리

## 왜 필요한가? (Why)

- **레이블 없는 데이터에서 패턴을 발견**할 때 클러스터링은 가장 기본적인 접근법이다
- 고객 세그먼테이션, 이상 탐지, 데이터 탐색(EDA) 등 다양한 실무에서 활용된다
- 비지도 학습(Unsupervised Learning)이므로 별도의 라벨링 비용 없이 데이터 구조를 파악할 수 있다
- 반도체 공정에서도 장비 로그/센서 데이터의 그룹화, 결함 유형 분류 등에 적용 가능하다

## 핵심 개념 (What)

이 문서에서는 다음 세 가지 접근 방식을 비교한다. 혼합 모델·spectral 등 다른 군집 방식도 있다:

| 접근 방식 | 대표 알고리즘 | 핵심 아이디어 |
|-----------|-------------|-------------|
| **분할 기반 (Partitioning)** | KMeans, KMedoids | KMeans는 평균 중심, KMedoids는 실제 표본 medoid로 대표 |
| **밀도 기반 (Density-based)** | DBSCAN, HDBSCAN | 밀집 영역을 클러스터로 인식. 비구형 클러스터 탐지 가능 |
| **계층적 (Hierarchical)** | Agglomerative, Divisive | 트리 구조로 클러스터를 병합/분할. 덴드로그램으로 시각화 |

### 주요 용어

- **관성(Inertia)**: KMeans 표본과 소속 중심의 거리 제곱합. 같은 입력·단위에서 K 증가로 줄어들 수 있어 단독 최소화로 K를 고르지 않음
- **실루엣 점수(Silhouette Score)**: 클러스터 내 응집도와 클러스터 간 분리도의 균형. -1 ~ 1 범위, 높을수록 좋음
- **eps (epsilon)**: DBSCAN에서 이웃 탐색 반경
- **min_samples**: DBSCAN에서 코어 포인트가 되기 위한 eps 이웃 표본 수 (자기 자신 포함)

## 어떻게 사용하는가? (How)

### 0. 공통 셋업

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def safe_silhouette(values, labels, *, exclude_noise=False):
    """정의 불가능한 군집 수는 NaN; unknown label을 정상으로 바꾸지 않음."""
    values = np.asarray(values)
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(values) != len(labels):
        raise ValueError("표본 수와 1차원 label 길이가 일치해야 함")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("이 데모는 알려진 정수 군집 label만 받음")
    mask = labels != -1 if exclude_noise else np.ones(len(labels), dtype=bool)
    observed = labels[mask]
    n_samples = len(observed)
    n_labels = len(np.unique(observed))
    if not 2 <= n_labels < n_samples:
        return np.nan
    return silhouette_score(values[mask], observed)


# 샘플 데이터 생성
X, y_true = make_blobs(
    n_samples=500,
    centers=4,
    cluster_std=0.8,
    random_state=42,
)

# 각 피처 단위가 거리에서 갖는 의미를 정한 뒤 스케일링 여부를 선택
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
    silhouette_scores.append(safe_silhouette(X_scaled, km.labels_))

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

# 아래 코드는 silhouette 최대 후보만 계산; Elbow 꺾임을 자동 판정하지 않음
best_k = K_range[np.argmax(silhouette_scores)]
print(f"실루엣 기준 후보 K: {best_k}")
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
# 자기 자신을 포함한 min_samples번째 거리 후보 (2*차원수는 보장된 최적값 아님)
k = 5
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
# 꺾임은 eps 탐색 후보; 실제 단위/밀도/노이즈 비율도 점검
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
score = safe_silhouette(X_scaled, db_labels, exclude_noise=True)
print(f"실루엣 점수 (노이즈 제외; 정의 불가=nan): {score:.3f}")
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
        sil = safe_silhouette(X_scaled, lbl, exclude_noise=True)
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
    truncate_mode="lastp",   # 마지막 p개의 비단말 node를 포함하도록 압축 표시
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
print(f"실루엣 점수: {safe_silhouette(X_scaled, agg_labels):.3f}")
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
    results["kmeans_sil"].append(safe_silhouette(X_scaled, km.labels_))

    # Agglomerative
    agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
    agg_labels = agg.fit_predict(X_scaled)
    results["agg_sil"].append(safe_silhouette(X_scaled, agg_labels))

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
print(f"KMeans silhouette 후보 K: {best_km} (실루엣: {max(results['kmeans_sil']):.3f})")
print(f"Agglomerative silhouette 후보 K: {best_agg} (실루엣: {max(results['agg_sil']):.3f})")
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
    perplexity=30,       # 반드시 표본 수보다 작아야 함; 후보 범위는 목적별 탐색
    random_state=42,
    max_iter=1000,       # sklearn 1.5에서 n_iter 이름 변경
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
| **클러스터 수 사전 지정** | 필요 (K) | 불필요 | K 또는 distance_threshold 설정 |
| **클러스터 형태** | 구형(spherical) | 비정형 가능 | 다양 (linkage에 따라) |
| **노이즈/이상치 처리** | 취약 | 노이즈 라벨 -1 제공; 실제 이상/고장 판정은 별도 | 취약 |
| **대용량 데이터** | 반복·차원·K에 따른 비용 | 구현·반경·차원 의존; sklearn 메모리 최악 O(n²) | 구현/linkage·표본 수에 따른 시간/메모리 비용 |
| **하이퍼파라미터** | K | eps, min_samples | K, linkage |
| **결정론적** | 초기값/판본/환경 조건 고정 필요 | 경계점 소속/번호는 입력 순서 영향 가능 | 동률·입력 순서/구현 영향 가능 |
| **추천 상황** | 대용량, 구형 클러스터 | 동일 eps로 밀도를 구분할 수 있는 입력; 밀도 차이가 크면 OPTICS/HDBSCAN 검토 | 계층 구조 탐색, 소규모 데이터 |

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
    └── HDBSCAN/OPTICS도 차원·거리·메모리·실측 비교 후 검토
```

---

## 확인 근거와 적용 조건

확인일 **2026-10-04**. 공식 scikit-learn **1.9.1**, SciPy 문서 **1.18.0** 기준으로 대조했다. NumPy·Matplotlib·SciPy·scikit-learn을 설치하고 0번 셋업부터 순차 실행한다. 출력 PNG는 별도 실험 폴더에 저장한다. 실제 장비 입력·업무 군집·속도와 한글 plot은 별도 검증한다.

- [군집 User Guide](https://scikit-learn.org/stable/modules/clustering.html): KMeans의 형태 가정과 단위/거리, DBSCAN의 밀도 가정과 입력 순서 영향을 확인한다. KMedoids는 scikit-learn 기본 KMeans의 별칭이 아니며 이 문서에 구현하지 않았다.
- [별도 sklearn-extra KMedoids 구현](https://raw.githubusercontent.com/scikit-learn-contrib/scikit-learn-extra/main/sklearn_extra/cluster/_k_medoids.py): 실제 표본 medoid를 사용한다. 이 문서 실행 의존성으로 추가하거나 해당 모델을 실행한 것은 아니다.
- [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html): min_samples는 자신을 포함하며 eps는 같은 cluster의 최대 거리라는 뜻이 아니다. 큰 eps/낮은 min_samples의 최악 메모리는 O(n²)다. -1은 해당 밀도 설정의 노이즈 label이지 고장/오류 사실이 아니다.
- [silhouette_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html): 2 <= label 수 < 표본 수일 때 정의된다. 전부 노이즈·한 군집·모든 표본이 singleton이면 이 데모는 NaN으로 미평가를 보존한다. -1 점수는 실제 유효한 낮은 점수일 수 있어 미평가 sentinel로 쓰지 않는다.
- [AgglomerativeClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html): Ward는 Euclidean 거리의 분산 기준을 사용한다. distance_threshold를 지정하면 n_clusters=None 등의 계약을 따른다. 여기서는 K를 지정했다.
- [dendrogram](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html): lastp의 node 표시 의미를 확인한다. 그림의 거리15 선과 뒤의 K=4 fit은 서로 자동 연결되지 않는다.
- [TSNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html): max_iter는 1.5에서 n_iter 이름 변경. perplexity < 표본 수 조건, 초기화/seed/비용을 확인한다. t-SNE는 일반적인 신규 표본 transform을 제공하지 않으며 2D 군집 간 거리/크기만으로 원래 공간의 구조를 증명하지 않는다.

합성 입력은 이미 2D라 PCA 2D가 차원 감소 데모인 것은 아니다. 군집은 원래 X_scaled에서 계산하고 PCA/t-SNE는 표시 좌표만 만든다. 다른 알고리즘의 label 번호/색은 서로 같은 군집이라는 뜻이 아니다. y_true는 합성 자료의 생성 label로 준비했지만 이 예제는 이를 업무 정답으로 사용하지 않는다.

scaler를 전체 입력에 fit하는 것은 해당 표본 집합의 탐색 데모다. 미래 표본 성능을 평가한다면 train에만 fit하고 적용 시 transform 계약을 정해야 한다. 실루엣 최대나 Elbow만으로 실제 cluster 수를 확정하지 않고 목적·안정성·도메인 의미를 비교한다. DBSCAN에서 노이즈를 제외한 점수는 제외한 비율도 함께 기록해야 하고 다른 표본 집합의 점수만으로 우열을 판단하지 않는다. 현재 고정500행 입력의 K=2..10 루프는 표본 수/중복 조건이 충족되는 데모이며 임의의 소표본에 그대로 적용하지 않는다.

## 참고 자료 (References)

- [scikit-learn Clustering 공식 문서](https://scikit-learn.org/stable/modules/clustering.html)
- [scikit-learn Clustering 비교 예제](https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html)
- [Silhouette Analysis (sklearn)](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)
- [DBSCAN 파라미터 튜닝 가이드](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
- [HDBSCAN 공식 문서](https://hdbscan.readthedocs.io/en/latest/)

## 관련 문서

- [상위 폴더](../README.md)
- [데이터 전처리](../../data-handling/README.md)
