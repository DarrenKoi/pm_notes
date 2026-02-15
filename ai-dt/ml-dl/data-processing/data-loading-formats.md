---
tags: [pandas, polars, csv, json, parquet, excel, data-loading]
level: beginner
last_updated: 2026-02-14
status: complete
---

# 데이터 로딩 포맷 가이드 (CSV, JSON, Parquet, Excel)

> 실무에서 자주 만나는 데이터 포맷별 로딩 방법과 트레이드오프를 정리한 실전 가이드

## 왜 필요한가? (Why)

- 실제 프로젝트에서는 CSV, JSON, Parquet, Excel 등 **다양한 포맷의 데이터가 혼재**한다
- 포맷마다 인코딩, 타입 추론, 메모리 효율, 읽기 속도가 크게 다르다
- 잘못된 로딩 옵션 하나로 **타입 깨짐, 한글 깨짐, 메모리 폭발**이 발생할 수 있다
- 포맷 특성을 이해하면 저장/로딩 전략을 최적화해서 파이프라인 전체 성능을 개선할 수 있다

---

## 핵심 개념 (What)

### 포맷별 비교

| 항목 | CSV | JSON | Parquet | Excel |
|------|-----|------|---------|-------|
| **타입 보존** | X (모두 문자열) | 부분적 (숫자/문자열) | O (스키마 내장) | 부분적 |
| **압축 효율** | 낮음 | 낮음 | 매우 높음 (컬럼 압축) | 중간 |
| **읽기 속도** | 보통 | 느림 | 매우 빠름 | 느림 |
| **사람이 읽기** | 쉬움 | 쉬움 | 불가 (바이너리) | 쉬움 (Excel 필요) |
| **스트리밍/청크** | O | O (JSON Lines) | O (row group) | X |
| **주 사용처** | 범용 교환 | API 응답, 설정 | 분석/ML 파이프라인 | 비개발자 협업 |

### 핵심 원칙

1. **분석/ML 파이프라인 내부**: Parquet을 기본으로 사용 (타입 보존 + 빠른 속도)
2. **외부 데이터 수신**: CSV/Excel로 받되, 즉시 Parquet으로 변환 저장
3. **대용량 파일**: 청크(chunk) 단위 처리 또는 Polars 활용
4. **API 연동**: JSON Lines(`.jsonl`) 포맷 우선

---

## 어떻게 사용하는가? (How)

### 1. CSV 로딩

CSV는 가장 흔하지만 가장 주의가 필요한 포맷이다.

```python
import pandas as pd

# --- 기본 로딩 ---
df = pd.read_csv("data.csv")

# --- 실무 로딩: 인코딩 + 타입 지정 + 날짜 파싱 ---
df = pd.read_csv(
    "data.csv",
    encoding="utf-8",          # 한글 데이터: "cp949" 또는 "euc-kr" 시도
    dtype={
        "id": str,             # 숫자처럼 보이지만 문자열인 컬럼 (예: 사번, 장비코드)
        "category": "category", # 카디널리티 낮은 컬럼 → 메모리 절약
        "value": float,
    },
    parse_dates=["created_at", "updated_at"],  # 날짜 컬럼 자동 파싱
    na_values=["", "N/A", "-", "null"],        # 결측치로 처리할 값들
    low_memory=False,          # 대용량 파일에서 dtype 혼합 경고 방지
)

# --- 한글 인코딩 자동 감지 패턴 ---
def read_csv_auto_encoding(filepath: str, **kwargs) -> pd.DataFrame:
    """한글 CSV 인코딩을 자동 감지하여 로딩"""
    encodings = ["utf-8", "cp949", "euc-kr", "utf-8-sig"]
    for enc in encodings:
        try:
            return pd.read_csv(filepath, encoding=enc, **kwargs)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"지원하는 인코딩으로 읽을 수 없음: {filepath}")

df = read_csv_auto_encoding("korean_data.csv", dtype={"code": str})

# --- 특정 컬럼만 로딩 (메모리 절약) ---
df = pd.read_csv(
    "large_data.csv",
    usecols=["id", "name", "value", "date"],
    dtype={"id": str, "value": float},
)

# --- 구분자가 다른 경우 ---
df = pd.read_csv("data.tsv", sep="\t")
df = pd.read_csv("data.txt", sep="|")
```

### 2. JSON 로딩

```python
import pandas as pd

# --- 일반 JSON (list of records) ---
# [{"name": "A", "val": 1}, {"name": "B", "val": 2}]
df = pd.read_json("data.json", orient="records")

# --- JSON Lines (.jsonl) — 대용량에 적합 ---
# {"name": "A", "val": 1}
# {"name": "B", "val": 2}
df = pd.read_json("data.jsonl", orient="records", lines=True)

# --- 중첩 JSON 평탄화 ---
import json

with open("nested.json", "r", encoding="utf-8") as f:
    raw = json.load(f)

# 중첩 구조를 평탄하게 펼침
df = pd.json_normalize(
    raw,
    record_path=["items"],           # 펼칠 리스트 경로
    meta=["request_id", "timestamp"], # 상위 레벨에서 가져올 필드
    sep="_",                          # 중첩 키 구분자
)

# --- API 응답을 바로 DataFrame으로 ---
import requests

resp = requests.get("https://api.example.com/data")
df = pd.json_normalize(resp.json()["results"])

# --- JSON Lines 저장 (다른 시스템 연동용) ---
df.to_json("output.jsonl", orient="records", lines=True, force_ascii=False)
```

### 3. Parquet 로딩

분석/ML 파이프라인에서 **기본 포맷으로 권장**한다.

```python
import pandas as pd

# --- 기본 로딩 (pyarrow 엔진) ---
df = pd.read_parquet("data.parquet", engine="pyarrow")

# --- 특정 컬럼만 로딩 (컬럼 기반 포맷의 최대 장점) ---
df = pd.read_parquet(
    "data.parquet",
    columns=["id", "name", "score"],  # 필요한 컬럼만 읽음 → 매우 빠름
)

# --- 필터 조건으로 읽기 (row group 레벨 스킵) ---
df = pd.read_parquet(
    "data.parquet",
    filters=[
        ("date", ">=", "2026-01-01"),
        ("category", "==", "A"),
    ],
)

# --- 디렉토리 파티션 Parquet 읽기 ---
# data/
#   year=2025/
#     month=01/
#       part-0.parquet
#     month=02/
#       part-0.parquet
df = pd.read_parquet("data/", engine="pyarrow")  # 디렉토리 전체 읽기

# --- CSV → Parquet 변환 저장 (실무 필수 패턴) ---
df = pd.read_csv("raw_data.csv", dtype={"id": str})
df.to_parquet(
    "processed_data.parquet",
    engine="pyarrow",
    compression="snappy",  # 기본값, 속도/압축 균형
    index=False,
)

# --- PyArrow로 직접 읽기 (더 세밀한 제어) ---
import pyarrow.parquet as pq

table = pq.read_table("data.parquet")
print(table.schema)  # 스키마 확인
df = table.to_pandas()
```

### 4. Excel 로딩

비개발자(현업)로부터 받는 데이터의 대부분이 Excel이다.

```python
import pandas as pd

# --- 기본 로딩 ---
df = pd.read_excel("data.xlsx", engine="openpyxl")

# --- 실무 로딩: 시트 지정 + 범위 + 타입 ---
df = pd.read_excel(
    "data.xlsx",
    sheet_name="Sheet1",       # 시트 이름 또는 인덱스(0, 1, ...)
    header=1,                  # 실제 헤더가 2번째 행에 있는 경우 (0-indexed)
    skiprows=[0],              # 건너뛸 행
    usecols="A:F",             # 사용할 컬럼 범위 (Excel 스타일)
    dtype={"장비코드": str},
    na_values=["", "-", "N/A"],
)

# --- 모든 시트를 한번에 읽기 ---
all_sheets: dict[str, pd.DataFrame] = pd.read_excel(
    "data.xlsx",
    sheet_name=None,  # None → 모든 시트를 dict로 반환
    engine="openpyxl",
)
for sheet_name, sheet_df in all_sheets.items():
    print(f"{sheet_name}: {sheet_df.shape}")

# --- 여러 Excel 파일 병합 ---
from pathlib import Path

excel_dir = Path("raw_excels/")
dfs = []
for f in excel_dir.glob("*.xlsx"):
    df_part = pd.read_excel(f, dtype={"id": str})
    df_part["source_file"] = f.name  # 출처 추적용
    dfs.append(df_part)

df_all = pd.concat(dfs, ignore_index=True)
```

### 5. 공통 패턴: 타입 캐스팅 & 결측치 처리

```python
import pandas as pd
import numpy as np

# --- 로딩 후 타입 정리 패턴 ---
def clean_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """로딩 후 공통 타입 정리"""
    df = df.copy()

    # 문자열 컬럼 공백 제거
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # 카디널리티 낮은 문자열 → category
    for col in str_cols:
        if df[col].nunique() / len(df) < 0.05:  # 고유값 비율 5% 미만
            df[col] = df[col].astype("category")

    return df

# --- 결측치 처리 패턴 ---
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """결측치 현황 확인 및 기본 처리"""
    # 결측 현황 출력
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        print("=== 결측치 현황 ===")
        print(missing)
        print(f"전체 행 수: {len(df)}")

    # 숫자형 결측: 0 또는 중앙값으로 채우기 (상황에 맞게 선택)
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(0)

    # 문자열 결측: "unknown"으로 채우기
    str_cols = df.select_dtypes(include=["object", "category"]).columns
    df[str_cols] = df[str_cols].fillna("unknown")

    return df

# 사용 예시
df = pd.read_csv("data.csv", dtype={"id": str})
df = clean_dtypes(df)
df = handle_missing(df)
```

### 6. 대용량 파일 청크 처리

```python
import pandas as pd

# --- 청크 단위로 읽으면서 집계 ---
chunks = pd.read_csv(
    "huge_data.csv",
    chunksize=100_000,     # 10만 행씩 읽기
    dtype={"id": str},
    usecols=["id", "category", "value"],
)

results = []
for chunk in chunks:
    # 청크별로 집계 수행
    agg = chunk.groupby("category")["value"].sum()
    results.append(agg)

# 청크 집계 결과 합산
final = pd.concat(results).groupby(level=0).sum()
print(final)

# --- 청크 → Parquet 변환 (대용량 CSV를 Parquet으로) ---
chunks = pd.read_csv("huge_data.csv", chunksize=500_000, dtype={"id": str})
for i, chunk in enumerate(chunks):
    chunk.to_parquet(f"output/part_{i:04d}.parquet", engine="pyarrow", index=False)
```

### 7. Polars로 빠르게 로딩

Polars는 Rust 기반으로 pandas보다 **대용량 데이터에서 2~10배 빠르다**.

```python
import polars as pl

# --- CSV ---
df = pl.read_csv(
    "data.csv",
    encoding="utf8",
    dtypes={"id": pl.Utf8, "value": pl.Float64},
    null_values=["", "N/A", "-"],
    try_parse_dates=True,       # 날짜 자동 파싱
)

# --- JSON Lines ---
df = pl.read_ndjson("data.jsonl")

# --- Parquet (Polars의 가장 빠른 경로) ---
df = pl.read_parquet("data.parquet")

# 특정 컬럼만 + 필터 (IO 최소화)
df = pl.scan_parquet("data.parquet").filter(
    (pl.col("date") >= "2026-01-01") & (pl.col("category") == "A")
).select(["id", "name", "score"]).collect()

# --- Excel ---
df = pl.read_excel("data.xlsx", sheet_name="Sheet1")

# --- Lazy 처리 (대용량 파일에서 메모리 효율적) ---
lazy = pl.scan_csv("huge_data.csv")
result = (
    lazy
    .filter(pl.col("value") > 0)
    .group_by("category")
    .agg(pl.col("value").sum().alias("total"))
    .sort("total", descending=True)
    .collect()  # 여기서 실제 실행
)
print(result)

# --- Polars → Pandas 변환 (필요 시) ---
pandas_df = df.to_pandas()
```

### 8. 포맷 간 변환 유틸리티

```python
import pandas as pd
from pathlib import Path


def convert_to_parquet(
    src: str | Path,
    dst: str | Path | None = None,
    **read_kwargs,
) -> Path:
    """CSV/Excel/JSON → Parquet 변환. dst 생략 시 확장자만 변경."""
    src = Path(src)
    dst = Path(dst) if dst else src.with_suffix(".parquet")

    ext = src.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(src, **read_kwargs)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(src, **read_kwargs)
    elif ext == ".json":
        df = pd.read_json(src, **read_kwargs)
    elif ext == ".jsonl":
        df = pd.read_json(src, orient="records", lines=True, **read_kwargs)
    else:
        raise ValueError(f"지원하지 않는 포맷: {ext}")

    df.to_parquet(dst, engine="pyarrow", compression="snappy", index=False)
    print(f"변환 완료: {src} → {dst} ({len(df):,} rows)")
    return dst


# 사용 예시
convert_to_parquet("raw_data.csv", dtype={"id": str})
convert_to_parquet("report.xlsx", sheet_name="Data")
```

---

## 참고 자료 (References)

- [pandas I/O 공식 문서](https://pandas.pydata.org/docs/user_guide/io.html)
- [pandas.read_csv API](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html)
- [pandas.read_parquet API](https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html)
- [Polars 공식 문서 - IO](https://docs.pola.rs/user-guide/io/)
- [Apache Parquet 포맷 명세](https://parquet.apache.org/documentation/latest/)
- [PyArrow Parquet 문서](https://arrow.apache.org/docs/python/parquet.html)

---

## 관련 문서

- [EDA 레시피](../eda-recipes.md)
- [AI/DT ML-DL README](../../README.md)
