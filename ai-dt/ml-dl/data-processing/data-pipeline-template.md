---
tags: [pipeline, sklearn, preprocessing, column-transformer]
level: intermediate
last_updated: 2026-02-14
status: in-progress
---

# sklearn 데이터 파이프라인 템플릿

> sklearn Pipeline과 ColumnTransformer를 활용한 재현 가능하고 누수 없는 전처리 파이프라인 구축 가이드

## 왜 필요한가? (Why)

### 1. 재현 가능한 전처리 (Reproducible Preprocessing)
- 전처리 단계를 코드로 명시적으로 정의하면, 누가 실행해도 동일한 결과를 보장한다.
- 수동으로 스케일링, 인코딩, 결측치 처리를 따로따로 하면 순서가 꼬이거나 누락되기 쉽다.

### 2. 데이터 누수 방지 (Avoid Data Leakage)
- `fit_transform`을 train 전체에 적용한 뒤 train/test를 나누면 **테스트 데이터 정보가 학습에 유입**된다.
- Pipeline은 `fit`과 `transform`을 내부적으로 분리하므로, cross-validation이나 train/test split에서 자동으로 누수를 차단한다.

### 3. 프로젝트 간 재사용
- 한 번 잘 만든 파이프라인 템플릿은 데이터셋만 바꿔서 여러 프로젝트에 즉시 적용할 수 있다.
- 모델 교체도 파이프라인 마지막 단계만 바꾸면 된다.

---

## 핵심 개념 (What)

| 개념 | 설명 |
|------|------|
| **Pipeline** | 여러 변환 단계를 순서대로 체이닝하는 컨테이너. 마지막 단계는 estimator(모델)도 가능 |
| **ColumnTransformer** | 컬럼별로 서로 다른 변환을 적용. 수치형/범주형 컬럼에 각각 다른 전처리를 한 번에 처리 |
| **make_pipeline** | 이름을 자동 생성해주는 Pipeline 헬퍼 함수. 빠른 프로토타이핑에 유용 |
| **FunctionTransformer** | 일반 Python 함수를 sklearn 변환기(transformer) 인터페이스로 감싸는 래퍼 |

### Pipeline 내부 동작 원리

```
fit() 호출 시:
  Step1.fit_transform(X) → Step2.fit_transform(X) → ... → LastStep.fit(X, y)

predict() 호출 시:
  Step1.transform(X) → Step2.transform(X) → ... → LastStep.predict(X)
```

- 중간 단계는 반드시 `transform` 메서드를 가져야 한다 (transformer).
- 마지막 단계만 `predict` 또는 `transform` 중 하나를 가지면 된다.

---

## 어떻게 사용하는가? (How)

### 1. 기본 Pipeline: 스케일러 + 모델

가장 단순한 형태의 파이프라인이다.

```python
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# 데이터 준비
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 방법 1: Pipeline (이름 직접 지정)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=200)),
])

# 방법 2: make_pipeline (이름 자동 생성)
pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=200),
)

# 학습 & 평가 — 스케일링이 자동으로 적용된다
pipe.fit(X_train, y_train)
score = pipe.score(X_test, y_test)
print(f"Accuracy: {score:.4f}")
```

---

### 2. ColumnTransformer: 수치형/범주형 분리 변환

실무 데이터는 수치형과 범주형 컬럼이 섞여 있다. ColumnTransformer로 각각 다른 변환을 적용한다.

```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# 샘플 데이터
df = pd.DataFrame({
    "age": [25, 30, None, 45, 35],
    "salary": [50000, 60000, 55000, None, 70000],
    "department": ["eng", "sales", "eng", "hr", None],
    "level": ["junior", "senior", "mid", "senior", "junior"],
    "target": [0, 1, 0, 1, 1],
})

X = df.drop("target", axis=1)
y = df["target"]

# 컬럼 분류
numeric_cols = ["age", "salary"]
categorical_cols = ["department", "level"]

# 수치형 파이프라인: 결측치 → 스케일링
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# 범주형 파이프라인: 결측치 → 원핫인코딩
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

# ColumnTransformer로 합치기
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols),
])

# 전처리 + 모델 파이프라인
full_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
])

full_pipe.fit(X, y)
print(f"Train score: {full_pipe.score(X, y):.4f}")
```

---

### 3. 완전한 전처리 파이프라인 템플릿

프로덕션에서 바로 사용할 수 있는 클래스 템플릿이다. 수치형/범주형 컬럼을 자동 분류하고, 결측치 처리, 스케일링, 인코딩을 일괄 적용한다.

```python
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder


class AutoPreprocessor(BaseEstimator, TransformerMixin):
    """수치형/범주형 컬럼을 자동 감지하여 전처리 파이프라인을 구성한다.

    Parameters
    ----------
    numeric_impute_strategy : str
        수치형 결측치 대치 전략. "mean", "median", "most_frequent" 중 선택.
    categorical_impute_strategy : str
        범주형 결측치 대치 전략.
    categorical_encoding : str
        "onehot" 또는 "ordinal".
    max_onehot_cardinality : int
        원핫 인코딩 시 고유값 수 상한. 초과하면 ordinal로 대체.
    """

    def __init__(
        self,
        numeric_impute_strategy: str = "median",
        categorical_impute_strategy: str = "most_frequent",
        categorical_encoding: str = "onehot",
        max_onehot_cardinality: int = 20,
    ):
        self.numeric_impute_strategy = numeric_impute_strategy
        self.categorical_impute_strategy = categorical_impute_strategy
        self.categorical_encoding = categorical_encoding
        self.max_onehot_cardinality = max_onehot_cardinality

    def _detect_columns(self, X: pd.DataFrame):
        """수치형과 범주형 컬럼을 자동으로 분류한다."""
        self.numeric_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols_ = X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()
        return self

    def _build_pipeline(self, X: pd.DataFrame) -> ColumnTransformer:
        """감지된 컬럼 타입에 따라 ColumnTransformer를 구성한다."""
        transformers = []

        # 수치형 파이프라인
        if self.numeric_cols_:
            num_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy=self.numeric_impute_strategy)),
                ("scaler", StandardScaler()),
            ])
            transformers.append(("num", num_pipe, self.numeric_cols_))

        # 범주형 파이프라인
        if self.categorical_cols_:
            if self.categorical_encoding == "onehot":
                # 카디널리티가 높은 컬럼은 ordinal로 분리
                onehot_cols = [
                    c for c in self.categorical_cols_
                    if X[c].nunique() <= self.max_onehot_cardinality
                ]
                ordinal_cols = [
                    c for c in self.categorical_cols_
                    if X[c].nunique() > self.max_onehot_cardinality
                ]

                if onehot_cols:
                    onehot_pipe = Pipeline([
                        ("imputer", SimpleImputer(strategy=self.categorical_impute_strategy)),
                        ("encoder", OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                        )),
                    ])
                    transformers.append(("cat_onehot", onehot_pipe, onehot_cols))

                if ordinal_cols:
                    ordinal_pipe = Pipeline([
                        ("imputer", SimpleImputer(strategy=self.categorical_impute_strategy)),
                        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ])
                    transformers.append(("cat_ordinal", ordinal_pipe, ordinal_cols))
            else:
                cat_pipe = Pipeline([
                    ("imputer", SimpleImputer(strategy=self.categorical_impute_strategy)),
                    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ])
                transformers.append(("cat", cat_pipe, self.categorical_cols_))

        self.transformer_ = ColumnTransformer(
            transformers=transformers,
            remainder="drop",  # 분류되지 않은 컬럼은 제거
        )
        return self.transformer_

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self._detect_columns(X)
        self._build_pipeline(X)
        self.transformer_.fit(X, y)
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return self.transformer_.transform(X)

    def get_feature_names_out(self):
        return self.transformer_.get_feature_names_out()
```

**사용 예시:**

```python
from sklearn.ensemble import GradientBoostingClassifier

pipe = Pipeline([
    ("preprocessor", AutoPreprocessor(
        numeric_impute_strategy="median",
        categorical_encoding="onehot",
        max_onehot_cardinality=15,
    )),
    ("model", GradientBoostingClassifier(n_estimators=200, random_state=42)),
])

pipe.fit(X_train, y_train)
print(f"Test accuracy: {pipe.score(X_test, y_test):.4f}")
```

---

### 4. 커스텀 변환기 통합

`FunctionTransformer`를 사용하면 일반 함수를 파이프라인 단계로 추가할 수 있다.

```python
from sklearn.preprocessing import FunctionTransformer
import numpy as np

# 로그 변환 (0 이하 값 방지를 위해 +1)
log_transformer = FunctionTransformer(
    func=np.log1p,
    inverse_func=np.expm1,  # 역변환도 정의 가능
    validate=True,
)

# DataFrame 컬럼 선택 함수
def select_columns(X, columns):
    return X[columns]

column_selector = FunctionTransformer(
    func=select_columns,
    kw_args={"columns": ["age", "salary"]},
)

# 파이프라인에 통합
pipe_with_custom = Pipeline([
    ("log_transform", log_transformer),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression()),
])
```

**pandas DataFrame을 유지하는 커스텀 변환기:**

```python
from sklearn.base import BaseEstimator, TransformerMixin

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """날짜 컬럼에서 연/월/요일 피처를 추출한다."""

    def __init__(self, date_column: str):
        self.date_column = date_column

    def fit(self, X, y=None):
        return self  # stateless transformer

    def transform(self, X):
        X = X.copy()
        dt = pd.to_datetime(X[self.date_column])
        X[f"{self.date_column}_year"] = dt.dt.year
        X[f"{self.date_column}_month"] = dt.dt.month
        X[f"{self.date_column}_dayofweek"] = dt.dt.dayofweek
        X = X.drop(columns=[self.date_column])
        return X
```

---

### 5. 파이프라인 저장/로드

학습된 파이프라인을 `joblib`으로 직렬화하여 저장하고, 배포 시 그대로 로드하여 사용한다.

```python
import joblib

# 저장
joblib.dump(pipe, "pipeline_v1.joblib")

# 로드
loaded_pipe = joblib.load("pipeline_v1.joblib")

# 로드한 파이프라인으로 바로 예측
predictions = loaded_pipe.predict(X_test)
print(f"Loaded pipeline accuracy: {loaded_pipe.score(X_test, y_test):.4f}")
```

**주의사항:**
- `joblib` 파일은 Python/sklearn 버전에 의존한다. 배포 환경의 버전을 맞춰야 한다.
- 커스텀 클래스를 사용했다면, 로드 시 해당 클래스가 import 가능한 상태여야 한다.
- 모델 파일에 sklearn 버전 정보를 함께 기록해두는 것이 좋다.

```python
import sklearn

metadata = {
    "sklearn_version": sklearn.__version__,
    "python_version": "3.11",
    "description": "분류 파이프라인 v1 - 수치형/범주형 전처리 + GBM",
}
joblib.dump({"pipeline": pipe, "metadata": metadata}, "pipeline_v1_with_meta.joblib")
```

---

### 6. 실전 팁

#### 데이터 누수 (Data Leakage) 주의

```python
# [잘못된 예] 전체 데이터에 fit_transform 후 split
X_scaled = scaler.fit_transform(X)  # 테스트 정보가 스케일링에 포함됨
X_train, X_test = train_test_split(X_scaled)

# [올바른 예] Pipeline + cross_val_score
from sklearn.model_selection import cross_val_score

pipe = make_pipeline(StandardScaler(), LogisticRegression())
scores = cross_val_score(pipe, X, y, cv=5)  # 내부적으로 fold별 fit/transform 분리
print(f"CV scores: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

#### fit_transform vs transform

| 메서드 | 언제 사용 | 설명 |
|--------|-----------|------|
| `fit(X)` | 학습 데이터로 통계 학습 | mean, std 등 저장 |
| `transform(X)` | 학습된 통계로 변환 | 테스트/운영 데이터에 사용 |
| `fit_transform(X)` | `fit` + `transform` 동시 | **학습 데이터에만** 사용 |

> Pipeline을 사용하면 이 구분을 자동으로 처리해주므로 실수를 방지할 수 있다.

#### 파이프라인 디버깅

```python
# 중간 단계 결과 확인 — Pipeline의 named_steps 속성 활용
pipe.fit(X_train, y_train)

# 전처리 단계만 적용한 결과 확인
preprocessed = pipe.named_steps["preprocessor"].transform(X_test)
print(f"Preprocessed shape: {preprocessed.shape}")
print(f"Feature names: {pipe.named_steps['preprocessor'].get_feature_names_out()}")
```

#### set_output API (sklearn >= 1.2)

```python
# transform 결과를 pandas DataFrame으로 유지
pipe.set_output(transform="pandas")
pipe.fit(X_train, y_train)

result = pipe[:-1].transform(X_test)  # 모델 전까지만 변환
print(type(result))  # <class 'pandas.core.frame.DataFrame'>
```

---

## 전체 실행 예제 (End-to-End)

아래 코드는 복사하여 바로 실행할 수 있는 완전한 예제이다.

```python
"""
sklearn 데이터 파이프라인 — End-to-End 예제
복사하여 바로 실행 가능.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ── 1. 데이터 로드 (Titanic) ──────────────────────────────────
titanic = fetch_openml("titanic", version=1, as_frame=True, parser="auto")
df = titanic.frame

# 사용할 피처 선택
feature_cols = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
target_col = "survived"

X = df[feature_cols].copy()
y = df[target_col].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"결측치 현황:\n{X_train.isnull().sum()}\n")


# ── 2. 컬럼 분류 ─────────────────────────────────────────────
numeric_cols = ["age", "sibsp", "parch", "fare"]
categorical_cols = ["pclass", "sex", "embarked"]


# ── 3. 전처리 파이프라인 정의 ─────────────────────────────────
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols),
])


# ── 4. 전체 파이프라인 (전처리 + 모델) ───────────────────────
pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )),
])


# ── 5. Cross Validation ──────────────────────────────────────
cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")
print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")


# ── 6. 최종 학습 & 평가 ──────────────────────────────────────
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print(f"\nTest Accuracy: {pipe.score(X_test, y_test):.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")


# ── 7. 변환된 피처 확인 ──────────────────────────────────────
feature_names = pipe.named_steps["preprocessor"].get_feature_names_out()
print(f"\n변환 후 피처 ({len(feature_names)}개): {feature_names.tolist()}")


# ── 8. 파이프라인 저장 ────────────────────────────────────────
import joblib
import sklearn

artifact = {
    "pipeline": pipe,
    "metadata": {
        "sklearn_version": sklearn.__version__,
        "features": feature_cols,
        "target": target_col,
        "cv_accuracy": float(cv_scores.mean()),
    },
}
joblib.dump(artifact, "titanic_pipeline_v1.joblib")
print("\nPipeline saved to titanic_pipeline_v1.joblib")


# ── 9. 로드하여 예측 ─────────────────────────────────────────
loaded = joblib.load("titanic_pipeline_v1.joblib")
loaded_pipe = loaded["pipeline"]
print(f"Loaded metadata: {loaded['metadata']}")
print(f"Loaded pipeline test accuracy: {loaded_pipe.score(X_test, y_test):.4f}")
```

---

## 참고 자료 (References)

- [sklearn Pipeline 공식 문서](https://scikit-learn.org/stable/modules/compose.html#pipeline)
- [sklearn ColumnTransformer 공식 문서](https://scikit-learn.org/stable/modules/compose.html#columntransformer-for-heterogeneous-data)
- [sklearn FunctionTransformer API](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html)
- [sklearn set_output API](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_set_output.html)
- [Data Leakage in ML (Kaggle)](https://www.kaggle.com/code/alexisbcook/data-leakage)

---

## 관련 문서

- [상위 폴더](../README.md)
- [데이터 처리 개요](../../data-handling/)
