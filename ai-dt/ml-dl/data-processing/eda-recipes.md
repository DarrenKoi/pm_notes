---
tags: [eda, pandas, matplotlib, seaborn]
level: beginner
last_updated: 2026-02-14
status: in-progress
---

# EDA 레시피 모음 (Exploratory Data Analysis Recipes)

> 모델링 전 데이터를 빠르게 파악하기 위한 실전 EDA 코드 레시피 모음

## 왜 필요한가? (Why)

- **모델링 전 필수 단계**: 데이터의 분포, 결측값, 이상치를 모르고 모델을 만들면 성능이 나올 수 없다
- **데이터 품질 확인**: 수집된 데이터가 분석에 적합한지 판단하는 근거를 마련한다
- **피처 엔지니어링 방향 설정**: 어떤 변수가 유의미하고, 어떤 변환이 필요한지 EDA를 통해 파악한다
- **커뮤니케이션**: 데이터의 현황을 시각적으로 정리하여 팀원/의사결정자에게 공유한다

## 핵심 개념 (What)

EDA는 크게 네 가지 관점에서 데이터를 탐색한다:

| 관점 | 확인 내용 | 주요 도구 |
|------|-----------|-----------|
| **분포(Distribution)** | 각 변수의 값이 어떻게 퍼져 있는가 | histplot, boxplot, KDE |
| **상관관계(Correlation)** | 변수 간 관계가 있는가 | corr(), heatmap, pairplot |
| **결측값(Missing Values)** | 빠진 데이터가 얼마나, 어디에 있는가 | isnull(), missingno |
| **이상치(Outliers)** | 비정상적으로 극단적인 값이 있는가 | IQR, z-score |

## 어떻게 사용하는가? (How)

### 0. 공통 설정: 라이브러리 임포트 및 한글 폰트

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# --- 한글 폰트 설정 ---
import platform

if platform.system() == "Darwin":  # macOS
    plt.rcParams["font.family"] = "AppleGothic"
elif platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
else:  # Linux
    plt.rcParams["font.family"] = "NanumGothic"

plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지

# 기본 스타일
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 100
```

### 1. 기본 통계 요약

데이터를 처음 받으면 가장 먼저 실행하는 코드들이다.

```python
# 샘플 데이터 로드 (실무에서는 pd.read_csv("your_data.csv") 등으로 대체)
df = sns.load_dataset("titanic")

# 데이터 크기 확인
print(f"행: {df.shape[0]:,}, 열: {df.shape[1]}")

# 처음 5행 미리보기
df.head()
```

```python
# 전체 컬럼 정보: 타입, 결측 수, 메모리 사용량
df.info()
```

```python
# 수치형 변수 기본 통계 (count, mean, std, min, 25%, 50%, 75%, max)
df.describe()
```

```python
# 범주형 변수 포함 통계
df.describe(include="all")
```

```python
# 각 컬럼의 데이터 타입 확인
df.dtypes
```

```python
# 각 컬럼의 유니크 값 개수
df.nunique()
```

```python
# 특정 범주형 컬럼의 값 분포
df["embarked"].value_counts()
```

```python
# 모든 object 타입 컬럼의 value_counts 한 번에 보기
for col in df.select_dtypes(include="object").columns:
    print(f"\n--- {col} ---")
    print(df[col].value_counts())
```

### 2. 결측값 분석

```python
# 컬럼별 결측값 수 및 비율
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)

missing_df = pd.DataFrame({
    "결측수": missing,
    "결측비율(%)": missing_pct
}).sort_values("결측비율(%)", ascending=False)

# 결측값이 있는 컬럼만 필터
missing_df[missing_df["결측수"] > 0]
```

```python
# 결측값 시각화 - 바 차트
cols_with_missing = missing_df[missing_df["결측수"] > 0].index.tolist()

if cols_with_missing:
    fig, ax = plt.subplots(figsize=(10, 5))
    missing_df.loc[cols_with_missing, "결측비율(%)"].plot.barh(ax=ax, color="salmon")
    ax.set_xlabel("결측 비율 (%)")
    ax.set_title("컬럼별 결측값 비율")
    plt.tight_layout()
    plt.show()
```

```python
# missingno 라이브러리 활용 (pip install missingno)
import missingno as msno

# 결측값 매트릭스: 흰색 = 결측
msno.matrix(df, figsize=(12, 5), fontsize=10)
plt.title("결측값 매트릭스")
plt.show()
```

```python
# 결측값 히트맵: 결측값 간 상관관계 (함께 빠지는 패턴 확인)
msno.heatmap(df, figsize=(8, 6), fontsize=10)
plt.title("결측값 상관관계 히트맵")
plt.show()
```

```python
# 결측 패턴 막대 그래프
msno.bar(df, figsize=(12, 5), fontsize=10)
plt.title("컬럼별 데이터 존재 비율")
plt.show()
```

### 3. 분포 확인

#### 3-1. 히스토그램 (Histogram)

```python
# 단일 수치형 변수 분포
fig, ax = plt.subplots()
sns.histplot(data=df, x="age", bins=30, kde=True, ax=ax)
ax.set_title("나이 분포")
ax.set_xlabel("나이")
ax.set_ylabel("빈도")
plt.tight_layout()
plt.show()
```

```python
# 모든 수치형 변수 분포 한 번에 보기
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
n_cols = 3
n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    sns.histplot(data=df, x=col, bins=30, kde=True, ax=axes[i])
    axes[i].set_title(f"{col} 분포")

# 빈 subplot 제거
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
```

#### 3-2. 박스플롯 (Boxplot)

```python
# 단일 변수 boxplot
fig, ax = plt.subplots()
sns.boxplot(data=df, y="fare", ax=ax)
ax.set_title("운임 박스플롯")
plt.tight_layout()
plt.show()
```

```python
# 카테고리별 boxplot (그룹 비교)
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x="class", y="fare", hue="survived", ax=ax)
ax.set_title("객실 등급별 운임 분포 (생존 여부)")
plt.tight_layout()
plt.show()
```

#### 3-3. KDE 플롯 (Kernel Density Estimation)

```python
# 그룹별 분포 비교에 유용
fig, ax = plt.subplots()
for survived, group in df.groupby("survived"):
    label = "생존" if survived == 1 else "사망"
    sns.kdeplot(data=group, x="age", label=label, fill=True, alpha=0.4, ax=ax)

ax.set_title("생존 여부별 나이 분포 (KDE)")
ax.set_xlabel("나이")
ax.legend()
plt.tight_layout()
plt.show()
```

### 4. 상관관계 분석

```python
# 수치형 변수 간 상관계수 계산
corr_matrix = df.select_dtypes(include=np.number).corr()

# 히트맵 시각화
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # 상삼각 마스킹
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5,
    ax=ax,
)
ax.set_title("수치형 변수 상관관계 히트맵")
plt.tight_layout()
plt.show()
```

```python
# 특정 타겟 변수와의 상관계수 정렬
target = "survived"
corr_with_target = corr_matrix[target].drop(target).sort_values(ascending=False)

print(f"'{target}'와의 상관계수:\n")
print(corr_with_target.to_string())
```

```python
# Pairplot: 변수 쌍별 산점도 + 분포 (변수가 많으면 느리니 주요 변수만 선택)
selected_cols = ["survived", "age", "fare", "pclass"]
sns.pairplot(df[selected_cols].dropna(), hue="survived", diag_kind="kde", height=2.5)
plt.suptitle("주요 변수 Pairplot", y=1.02)
plt.show()
```

### 5. 카테고리 변수 분석

```python
# 단일 카테고리 변수 빈도 시각화
fig, ax = plt.subplots()
sns.countplot(data=df, x="class", order=df["class"].value_counts().index, ax=ax)
ax.set_title("객실 등급별 승객 수")
ax.set_xlabel("객실 등급")
ax.set_ylabel("승객 수")

# 바 위에 숫자 표시
for p in ax.patches:
    ax.annotate(
        f"{int(p.get_height())}",
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center", va="bottom", fontsize=11,
    )

plt.tight_layout()
plt.show()
```

```python
# 카테고리 변수 vs 타겟 (비율 비교)
fig, ax = plt.subplots()
sns.countplot(data=df, x="class", hue="survived", ax=ax)
ax.set_title("객실 등급별 생존 여부")
ax.legend(title="생존", labels=["사망", "생존"])
plt.tight_layout()
plt.show()
```

```python
# 교차표 (Crosstab) - 두 범주형 변수 간 관계
ct = pd.crosstab(df["class"], df["survived"], margins=True, normalize="index")
ct.columns = ["사망비율", "생존비율", "합계"]
print(ct.round(3))
```

```python
# 교차표 히트맵
ct_counts = pd.crosstab(df["class"], df["sex"])
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(ct_counts, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
ax.set_title("객실 등급 × 성별 교차표")
plt.tight_layout()
plt.show()
```

### 6. 이상치 탐지

#### 6-1. IQR 방법

사분위수 범위(Interquartile Range)를 활용한 이상치 탐지. 가장 널리 쓰이는 방법이다.

```python
def detect_outliers_iqr(df: pd.DataFrame, col: str, factor: float = 1.5) -> pd.Series:
    """IQR 기반 이상치 탐지. True = 이상치."""
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    return (df[col] < lower) | (df[col] > upper)


# 사용 예시
outlier_mask = detect_outliers_iqr(df, "fare")
print(f"'fare' 이상치 수: {outlier_mask.sum()} ({outlier_mask.mean() * 100:.1f}%)")
print(f"이상치 범위 밖 값 예시:\n{df.loc[outlier_mask, 'fare'].describe()}")
```

```python
# 이상치 시각화 (boxplot + 실제 포인트)
fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(data=df, x="fare", ax=ax, color="lightblue")
sns.stripplot(
    data=df[outlier_mask], x="fare", ax=ax,
    color="red", alpha=0.5, size=4, label="이상치",
)
ax.set_title("운임 이상치 시각화 (IQR)")
ax.legend()
plt.tight_layout()
plt.show()
```

#### 6-2. Z-Score 방법

평균에서 표준편차 몇 배 떨어져 있는지로 판단. 정규분포에 가까운 데이터에 적합하다.

```python
from scipy import stats


def detect_outliers_zscore(df: pd.DataFrame, col: str, threshold: float = 3.0) -> pd.Series:
    """Z-score 기반 이상치 탐지. True = 이상치."""
    z_scores = np.abs(stats.zscore(df[col].dropna()))
    mask = pd.Series(False, index=df.index)
    mask.loc[df[col].dropna().index] = z_scores > threshold
    return mask


# 사용 예시
outlier_z = detect_outliers_zscore(df, "fare", threshold=3.0)
print(f"'fare' Z-score 이상치 수: {outlier_z.sum()} ({outlier_z.mean() * 100:.1f}%)")
```

```python
# 두 방법 비교
comparison = pd.DataFrame({
    "IQR_이상치": [detect_outliers_iqr(df, col).sum() for col in numeric_cols],
    "ZScore_이상치": [detect_outliers_zscore(df, col).sum() for col in numeric_cols],
}, index=numeric_cols)

print("=== 이상치 탐지 방법 비교 ===")
print(comparison[comparison.sum(axis=1) > 0])
```

### 7. 원스톱 EDA 함수

위의 모든 분석을 하나의 함수로 묶은 재사용 가능한 EDA 유틸리티다.

```python
def run_full_eda(
    df: pd.DataFrame,
    target: str | None = None,
    max_categories: int = 20,
    figsize_base: tuple = (12, 5),
) -> dict:
    """
    데이터프레임에 대한 전체 EDA를 수행한다.

    Parameters
    ----------
    df : pd.DataFrame
        분석할 데이터프레임
    target : str, optional
        타겟 변수 이름 (있으면 타겟 기준 분석 추가)
    max_categories : int
        카테고리 시각화 시 최대 유니크 값 수 (너무 많으면 스킵)
    figsize_base : tuple
        기본 figure 크기

    Returns
    -------
    dict
        분석 결과 요약 딕셔너리
    """
    results = {}
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # ========== 1. 기본 정보 ==========
    print("=" * 60)
    print("1. 기본 정보")
    print("=" * 60)
    print(f"  행: {df.shape[0]:,},  열: {df.shape[1]}")
    print(f"  수치형: {len(numeric_cols)}개,  범주형: {len(categorical_cols)}개")
    print(f"  메모리: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
    print()
    df.info()
    print()

    # ========== 2. 결측값 ==========
    print("=" * 60)
    print("2. 결측값 분석")
    print("=" * 60)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        "결측수": missing, "결측비율(%)": missing_pct
    }).sort_values("결측비율(%)", ascending=False)
    missing_with = missing_df[missing_df["결측수"] > 0]

    if len(missing_with) > 0:
        print(missing_with)
        fig, ax = plt.subplots(figsize=figsize_base)
        missing_with["결측비율(%)"].plot.barh(ax=ax, color="salmon")
        ax.set_xlabel("결측 비율 (%)")
        ax.set_title("컬럼별 결측값 비율")
        plt.tight_layout()
        plt.show()
    else:
        print("  결측값 없음!")

    results["missing"] = missing_with
    print()

    # ========== 3. 수치형 변수 분포 ==========
    if numeric_cols:
        print("=" * 60)
        print("3. 수치형 변수 분포")
        print("=" * 60)
        print(df[numeric_cols].describe().round(2))
        print()

        n_cols_plot = 3
        n_rows_plot = (len(numeric_cols) + n_cols_plot - 1) // n_cols_plot
        fig, axes = plt.subplots(
            n_rows_plot, n_cols_plot,
            figsize=(5 * n_cols_plot, 4 * n_rows_plot),
        )
        axes = np.array(axes).flatten()

        for i, col in enumerate(numeric_cols):
            sns.histplot(data=df, x=col, bins=30, kde=True, ax=axes[i])
            axes[i].set_title(f"{col} 분포")

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle("수치형 변수 분포", y=1.01, fontsize=14)
        plt.tight_layout()
        plt.show()

    # ========== 4. 이상치 요약 ==========
    if numeric_cols:
        print("=" * 60)
        print("4. 이상치 탐지 (IQR)")
        print("=" * 60)
        outlier_summary = {}
        for col in numeric_cols:
            mask = detect_outliers_iqr(df, col)
            outlier_summary[col] = {"이상치수": mask.sum(), "비율(%)": round(mask.mean() * 100, 2)}

        outlier_df = pd.DataFrame(outlier_summary).T
        outlier_df = outlier_df[outlier_df["이상치수"] > 0].sort_values("비율(%)", ascending=False)
        if len(outlier_df) > 0:
            print(outlier_df)
        else:
            print("  IQR 기준 이상치 없음")
        results["outliers"] = outlier_df
        print()

    # ========== 5. 상관관계 ==========
    if len(numeric_cols) >= 2:
        print("=" * 60)
        print("5. 상관관계 히트맵")
        print("=" * 60)
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(max(8, len(numeric_cols)), max(6, len(numeric_cols) * 0.8)))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, square=True, linewidths=0.5, ax=ax,
        )
        ax.set_title("수치형 변수 상관관계")
        plt.tight_layout()
        plt.show()

        if target and target in numeric_cols:
            print(f"\n'{target}'와의 상관계수:")
            print(corr[target].drop(target).sort_values(ascending=False).round(3))

        results["correlation"] = corr
        print()

    # ========== 6. 범주형 변수 ==========
    if categorical_cols:
        print("=" * 60)
        print("6. 범주형 변수 분포")
        print("=" * 60)
        plot_cats = [c for c in categorical_cols if df[c].nunique() <= max_categories]
        if plot_cats:
            n_cols_plot = min(3, len(plot_cats))
            n_rows_plot = (len(plot_cats) + n_cols_plot - 1) // n_cols_plot
            fig, axes = plt.subplots(
                n_rows_plot, n_cols_plot,
                figsize=(5 * n_cols_plot, 4 * n_rows_plot),
            )
            axes = np.array(axes).flatten()

            for i, col in enumerate(plot_cats):
                order = df[col].value_counts().index
                if target and target in df.columns:
                    sns.countplot(data=df, x=col, hue=target, order=order, ax=axes[i])
                else:
                    sns.countplot(data=df, x=col, order=order, ax=axes[i])
                axes[i].set_title(f"{col}")
                axes[i].tick_params(axis="x", rotation=45)

            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.suptitle("범주형 변수 분포", y=1.01, fontsize=14)
            plt.tight_layout()
            plt.show()

        skipped = [c for c in categorical_cols if df[c].nunique() > max_categories]
        if skipped:
            print(f"  유니크 값이 {max_categories}개 초과하여 스킵된 컬럼: {skipped}")
        print()

    print("=" * 60)
    print("EDA 완료!")
    print("=" * 60)

    return results
```

```python
# 사용 예시
df = sns.load_dataset("titanic")
eda_results = run_full_eda(df, target="survived")
```

```python
# 결과 딕셔너리에서 필요한 것만 꺼내 쓰기
eda_results["missing"]       # 결측값 요약
eda_results["outliers"]      # 이상치 요약
eda_results["correlation"]   # 상관계수 행렬
```

## 참고 자료 (References)

- [pandas 공식 문서 - DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)
- [seaborn 공식 튜토리얼](https://seaborn.pydata.org/tutorial.html)
- [matplotlib 공식 문서](https://matplotlib.org/stable/contents.html)
- [missingno GitHub](https://github.com/ResidentMario/missingno)
- [scipy.stats.zscore](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.zscore.html)

## 관련 문서

- [데이터 전처리 기법](./data-preprocessing.md)
- [피처 엔지니어링](./feature-engineering.md)
- [ML/DL 상위 폴더](../README.md)
