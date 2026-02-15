---
tags: [feature-engineering, sklearn, encoding, scaling]
level: intermediate
last_updated: 2026-02-14
status: in-progress
---

# 피처 엔지니어링(Feature Engineering) 실전 가이드

> 원시 데이터를 ML 모델이 학습하기 좋은 형태로 변환하는 핵심 전처리 기법 모음

## 왜 필요한가? (Why)

- **ML 모델은 숫자만 이해한다** - 범주형 문자열, 날짜, 텍스트 등을 그대로 넣을 수 없다
- **스케일 차이가 학습을 방해한다** - 키(170cm)와 몸무게(70kg)의 단위 차이가 거리 기반 알고리즘(KNN, SVM)에 큰 영향을 준다
- **좋은 피처가 좋은 모델을 만든다** - 복잡한 모델보다 잘 설계된 피처가 성능에 더 큰 영향을 줄 때가 많다
- **실무에서는 raw data가 그대로 쓰이는 경우가 거의 없다** - 결측치, 이상치, 비정형 데이터 등 전처리가 필수

---

## 핵심 개념 (What)

피처 엔지니어링은 크게 세 가지 유형으로 나뉜다:

| 유형 | 설명 | 예시 |
|------|------|------|
| **인코딩(Encoding)** | 범주형 데이터를 숫자로 변환 | 성별 → 0/1, 도시 → 원핫 벡터 |
| **스케일링(Scaling)** | 수치형 데이터의 범위/분포 조정 | 표준화, 정규화 |
| **피처 생성(Creation)** | 기존 피처에서 새로운 피처 유도 | 날짜 → 요일, 텍스트 → TF-IDF |

### sklearn 파이프라인 철학

sklearn은 모든 변환기를 `fit()` → `transform()` 패턴으로 통일한다:

```python
# 학습 데이터로 fit (통계량 학습)
scaler.fit(X_train)

# 학습/테스트 데이터 모두 동일하게 transform
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 또는 한 번에
X_train_scaled = scaler.fit_transform(X_train)
```

> **주의**: `fit`은 반드시 학습 데이터에만 적용해야 한다. 테스트 데이터에 fit하면 데이터 누수(data leakage)가 발생한다.

---

## 어떻게 사용하는가? (How)

### 공통 임포트 및 샘플 데이터

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 샘플 데이터 생성
np.random.seed(42)
df = pd.DataFrame({
    "age": np.random.randint(20, 60, 100),
    "salary": np.random.randint(3000, 12000, 100) * 10000,
    "city": np.random.choice(["서울", "부산", "대구", "인천"], 100),
    "education": np.random.choice(["고졸", "학사", "석사", "박사"], 100),
    "join_date": pd.date_range("2020-01-01", periods=100, freq="W"),
    "review": np.random.choice([
        "좋은 제품입니다",
        "배송이 느려요",
        "가격 대비 괜찮음",
        "다시 구매하고 싶어요",
    ], 100),
    "target": np.random.randint(0, 2, 100),
})

print(df.head())
```

---

### 1. 범주형 인코딩(Categorical Encoding)

#### 언제 어떤 인코더를 쓰는가?

| 인코더 | 적용 대상 | 특징 | 주의점 |
|--------|-----------|------|--------|
| `OneHotEncoder` | 명목형(순서 없음) - 도시, 색상 | 카테고리당 별도 컬럼 생성 | 카테고리 많으면 차원 폭발 |
| `OrdinalEncoder` | 순서형 - 학력, 등급 | 순서 정보 보존 | 순서를 직접 지정해야 함 |
| `LabelEncoder` | 타겟 변수(y) 인코딩 전용 | 단일 컬럼에 정수 할당 | 피처(X)에 사용하면 안 됨 |

#### OneHotEncoder - 명목형 변수

```python
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

# 명목형 변수: 순서 없음 (서울, 부산, 대구, 인천)
city_encoded = ohe.fit_transform(df[["city"]])

# 결과 확인
city_df = pd.DataFrame(
    city_encoded,
    columns=ohe.get_feature_names_out(["city"]),
)
print(city_df.head())
# city_대구  city_부산  city_서울  city_인천
#      0.0      1.0      0.0      0.0
#      ...
```

#### OrdinalEncoder - 순서형 변수

```python
from sklearn.preprocessing import OrdinalEncoder

# 순서를 명시적으로 지정
edu_order = [["고졸", "학사", "석사", "박사"]]

oe = OrdinalEncoder(categories=edu_order)
df["education_encoded"] = oe.fit_transform(df[["education"]])

print(df[["education", "education_encoded"]].drop_duplicates().sort_values("education_encoded"))
# education  education_encoded
# 고졸                     0.0
# 학사                     1.0
# 석사                     2.0
# 박사                     3.0
```

#### LabelEncoder - 타겟 변수 전용

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# 타겟 변수 인코딩 (문자열 → 정수)
y_labels = ["negative", "positive", "neutral", "positive", "negative"]
y_encoded = le.fit_transform(y_labels)
print(y_encoded)          # [1 2 0 2 1]
print(le.classes_)        # ['negative', 'neutral', 'positive']
print(le.inverse_transform(y_encoded))  # 원래 값 복원
```

---

### 2. 수치형 스케일링(Numerical Scaling)

#### 스케일러 비교

| 스케일러 | 수식 | 결과 범위 | 이상치 민감도 | 적합한 경우 |
|----------|------|-----------|---------------|-------------|
| `StandardScaler` | (x - mean) / std | 평균=0, 표준편차=1 | 민감 | 정규분포에 가까운 데이터 |
| `MinMaxScaler` | (x - min) / (max - min) | [0, 1] | 매우 민감 | 범위가 명확한 데이터 |
| `RobustScaler` | (x - median) / IQR | 중앙값=0 | **강건** | 이상치가 많은 데이터 |

#### 스케일러 코드 비교

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# 이상치 포함 데이터
data = df[["age", "salary"]].copy()

scalers = {
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler(),
    "RobustScaler": RobustScaler(),
}

for name, scaler in scalers.items():
    scaled = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(scaled, columns=["age", "salary"])
    print(f"\n--- {name} ---")
    print(scaled_df.describe().round(3))
```

#### 실전 선택 기준

```python
# 1. StandardScaler - 가장 범용적, 선형 모델/SVM/PCA와 궁합
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
X_standard = ss.fit_transform(df[["age", "salary"]])

# 2. MinMaxScaler - 신경망, 이미지 데이터에 적합
from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler(feature_range=(0, 1))  # 범위 지정 가능
X_minmax = mms.fit_transform(df[["age", "salary"]])

# 3. RobustScaler - 이상치에 강건, 의료/금융 데이터에 유용
from sklearn.preprocessing import RobustScaler

rs = RobustScaler()
X_robust = rs.fit_transform(df[["age", "salary"]])
```

---

### 3. 구간화(Binning)

연속형 변수를 구간으로 나누어 범주형으로 변환한다. 비선형 관계를 포착하거나 이상치 영향을 줄일 때 유용하다.

#### pd.cut - 동일 간격 구간화

```python
# 나이를 10세 단위로 구간화
df["age_bin"] = pd.cut(
    df["age"],
    bins=[0, 30, 40, 50, 60],
    labels=["20대", "30대", "40대", "50대"],
)

print(df["age_bin"].value_counts().sort_index())
```

#### pd.qcut - 동일 빈도 구간화

```python
# 급여를 4분위로 나누기 (각 구간에 동일 개수)
df["salary_quartile"] = pd.qcut(
    df["salary"],
    q=4,
    labels=["하위", "중하", "중상", "상위"],
)

print(df["salary_quartile"].value_counts())
```

#### KBinsDiscretizer - sklearn 파이프라인 호환

```python
from sklearn.preprocessing import KBinsDiscretizer

kbd = KBinsDiscretizer(
    n_bins=5,
    encode="ordinal",       # "ordinal", "onehot", "onehot-dense"
    strategy="quantile",    # "uniform", "quantile", "kmeans"
)

df["age_kbd"] = kbd.fit_transform(df[["age"]])

# 각 구간의 경계값 확인
print("구간 경계:", kbd.bin_edges_[0].round(1))
```

---

### 4. 날짜/시간 피처(Datetime Features)

날짜 컬럼에서 유의미한 시간적 패턴을 추출한다.

```python
# datetime 타입 확인/변환
df["join_date"] = pd.to_datetime(df["join_date"])

# 기본 시간 요소 추출
df["join_year"] = df["join_date"].dt.year
df["join_month"] = df["join_date"].dt.month
df["join_day"] = df["join_date"].dt.day
df["join_weekday"] = df["join_date"].dt.weekday       # 0=월요일, 6=일요일
df["join_hour"] = df["join_date"].dt.hour              # 시간 데이터가 있을 경우

# 파생 피처
df["is_weekend"] = df["join_weekday"].isin([5, 6]).astype(int)
df["join_quarter"] = df["join_date"].dt.quarter
df["days_since_join"] = (pd.Timestamp.now() - df["join_date"]).dt.days

# 주기적 인코딩 (cyclical encoding) - 월/요일의 순환 특성 반영
df["month_sin"] = np.sin(2 * np.pi * df["join_month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["join_month"] / 12)

print(df[["join_date", "join_year", "join_month", "join_weekday", "is_weekend"]].head())
```

> **팁**: 요일/월 같은 순환 데이터는 sin/cos 인코딩으로 12월과 1월이 가깝다는 관계를 모델에 알려줄 수 있다.

---

### 5. 텍스트 피처(Text Features)

텍스트 데이터를 수치 벡터로 변환한다.

#### CountVectorizer - 단어 빈도 벡터

```python
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(
    max_features=100,      # 상위 100개 단어만
    min_df=2,              # 최소 2개 문서에 등장
    ngram_range=(1, 2),    # 유니그램 + 바이그램
)

text_counts = cv.fit_transform(df["review"])

print(f"행렬 크기: {text_counts.shape}")
print(f"어휘 목록: {cv.get_feature_names_out()}")

# DataFrame 변환
text_df = pd.DataFrame(
    text_counts.toarray(),
    columns=cv.get_feature_names_out(),
)
print(text_df.head())
```

#### TfidfVectorizer - TF-IDF 가중치 벡터

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_features=100,
    min_df=2,
    ngram_range=(1, 2),
    sublinear_tf=True,     # TF에 로그 스케일링 적용
)

text_tfidf = tfidf.fit_transform(df["review"])

tfidf_df = pd.DataFrame(
    text_tfidf.toarray(),
    columns=tfidf.get_feature_names_out(),
)
print(tfidf_df.head())
```

> **CountVectorizer vs TfidfVectorizer**: CountVectorizer는 단순 빈도, TfidfVectorizer는 문서 전체에서의 희귀도(IDF)까지 반영한다. 일반적으로 TfidfVectorizer가 더 좋은 성능을 보인다.

---

### 6. 피처 상호작용(Feature Interaction)

기존 피처를 조합하여 새로운 피처를 만든다. 개별 피처로는 포착 못하는 비선형 관계를 잡을 수 있다.

#### PolynomialFeatures - 자동 다항/교차 피처

```python
from sklearn.preprocessing import PolynomialFeatures

pf = PolynomialFeatures(
    degree=2,
    interaction_only=False,   # True면 교차항만 (x1*x2), 제곱항(x1^2) 제외
    include_bias=False,        # 상수항(1) 제외
)

X_numeric = df[["age", "salary"]].values
X_poly = pf.fit_transform(X_numeric)

poly_df = pd.DataFrame(
    X_poly,
    columns=pf.get_feature_names_out(["age", "salary"]),
)
print(poly_df.head())
# age, salary, age^2, age salary, salary^2
```

#### 수동 상호작용 피처 생성

```python
# 도메인 지식 기반 피처 생성
df["salary_per_age"] = df["salary"] / df["age"]                    # 나이 대비 급여
df["age_salary_product"] = df["age"] * df["salary"]                # 교차항
df["log_salary"] = np.log1p(df["salary"])                          # 로그 변환 (왜곡 분포 완화)
df["salary_squared"] = df["salary"] ** 2                           # 비선형 관계 포착

# 조건 기반 피처
df["high_salary_young"] = ((df["salary"] > df["salary"].median()) &
                            (df["age"] < 35)).astype(int)

print(df[["age", "salary", "salary_per_age", "log_salary"]].head())
```

---

### 7. 커스텀 변환기(Custom Transformer)

sklearn 파이프라인에 통합 가능한 자체 변환기를 만든다.

#### BaseEstimator + TransformerMixin 활용

```python
from sklearn.base import BaseEstimator, TransformerMixin


class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """날짜 컬럼에서 시간 피처를 추출하는 커스텀 변환기."""

    def __init__(self, column: str, features: list[str] | None = None):
        self.column = column
        self.features = features or ["year", "month", "weekday"]

    def fit(self, X, y=None):
        # 학습할 통계량이 없으므로 self 반환
        return self

    def transform(self, X):
        X = X.copy()
        dt = pd.to_datetime(X[self.column])

        result = pd.DataFrame(index=X.index)
        if "year" in self.features:
            result[f"{self.column}_year"] = dt.dt.year
        if "month" in self.features:
            result[f"{self.column}_month"] = dt.dt.month
        if "day" in self.features:
            result[f"{self.column}_day"] = dt.dt.day
        if "weekday" in self.features:
            result[f"{self.column}_weekday"] = dt.dt.weekday
        if "hour" in self.features:
            result[f"{self.column}_hour"] = dt.dt.hour

        return result

    def get_feature_names_out(self, input_features=None):
        return [f"{self.column}_{f}" for f in self.features]


# 사용 예시
extractor = DatetimeFeatureExtractor(
    column="join_date",
    features=["year", "month", "weekday"],
)
date_features = extractor.fit_transform(df)
print(date_features.head())
```

#### 파이프라인에 통합하기

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# 수치형 + 범주형 + 커스텀 변환 파이프라인
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["age", "salary"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["city"]),
        ("edu", OrdinalEncoder(categories=[["고졸", "학사", "석사", "박사"]]), ["education"]),
    ],
    remainder="drop",
)

pipe = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
])

# 학습 및 예측
X = df[["age", "salary", "city", "education"]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipe.fit(X_train, y_train)
score = pipe.score(X_test, y_test)
print(f"Accuracy: {score:.3f}")

# 전처리된 피처 이름 확인
feature_names = pipe.named_steps["preprocess"].get_feature_names_out()
print(f"피처 목록: {feature_names}")
```

---

## 참고 자료 (References)

- [sklearn Preprocessing 공식 문서](https://scikit-learn.org/stable/modules/preprocessing.html)
- [sklearn ColumnTransformer 가이드](https://scikit-learn.org/stable/modules/compose.html#columntransformer-for-heterogeneous-data)
- [Feature Engineering for Machine Learning (Alice Zheng)](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)
- [Kaggle Feature Engineering 마이크로 코스](https://www.kaggle.com/learn/feature-engineering)

---

## 관련 문서

- [데이터 파이프라인 템플릿](./data-pipeline-template.md)
