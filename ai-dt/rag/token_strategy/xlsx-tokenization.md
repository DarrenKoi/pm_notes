---
tags: [rag, tokenization, excel, xlsx, tabular-data]
level: intermediate
last_updated: 2026-02-12
status: complete
---

# Excel 문서 토큰화 전략 (XLSX Tokenization Strategy)

> Excel은 정형 데이터(테이블)가 핵심이므로, 행/열 구조를 보존하면서 검색 가능한 형태로 변환하는 것이 관건이다.

## 왜 필요한가? (Why)

엔지니어링 분야에서 Excel의 활용:
- **스펙 시트**: 장비/재료 사양 비교 테이블
- **데이터 로그**: 측정값, 공정 파라미터 기록
- **관리 문서**: 체크리스트, 진행 현황 추적
- **다중 시트**: 하나의 파일에 여러 관련 테이블 포함

**핵심 과제**:
- 테이블 구조(행/열 관계)를 파괴하지 않으면서 텍스트화
- 여러 시트 간의 관계를 보존
- 빈 셀, 병합 셀, 수식 등의 특수 케이스 처리

## 핵심 개념 (What)

### Excel 데이터의 특수성

일반 텍스트 문서와 달리 Excel은:

1. **2차원 구조**: 행과 열의 교차점에 의미가 있음 (예: "A장비의 RPM은 3000")
2. **헤더 의존성**: 셀 값만으로는 의미 파악 불가 → 헤더와 함께 제공 필요
3. **다중 시트**: 시트 이름 자체가 중요한 컨텍스트
4. **데이터 타입 혼합**: 숫자, 텍스트, 날짜, 수식이 혼합

### 청킹 전략 옵션

| 전략 | 단위 | 적합한 경우 |
|------|------|-------------|
| **시트 단위** | 1 시트 = 1 청크 | 시트가 작고 독립적 |
| **행 그룹 단위** | N행 = 1 청크 | 큰 테이블, 행별 독립 데이터 |
| **테이블 영역 단위** | 데이터 영역별 1 청크 | 시트 내 여러 독립 테이블 |
| **행 단위 + 헤더** | 헤더 + 1행 = 1 청크 | 각 행이 독립 엔티티 (장비 스펙 등) |

## 어떻게 사용하는가? (How)

### 방법 1: openpyxl - 기본 구조 보존 추출

```python
from openpyxl import load_workbook

def extract_xlsx_sheets(xlsx_path: str) -> list[dict]:
    """시트별 데이터를 구조화하여 추출"""
    wb = load_workbook(xlsx_path, data_only=True)  # 수식 → 값
    sheets_data = []

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]

        # 데이터가 있는 영역만 추출
        rows = []
        for row in ws.iter_rows(values_only=True):
            # 완전히 빈 행 스킵
            if any(cell is not None for cell in row):
                rows.append([str(cell) if cell is not None else "" for cell in row])

        if rows:
            sheets_data.append({
                "sheet_name": sheet_name,
                "headers": rows[0] if rows else [],
                "data_rows": rows[1:] if len(rows) > 1 else [],
                "row_count": len(rows) - 1,
                "col_count": len(rows[0]) if rows else 0,
            })

    return sheets_data
```

### 방법 2: 행 그룹 청킹 (대형 테이블용)

```python
def chunk_xlsx_by_row_groups(
    xlsx_path: str,
    rows_per_chunk: int = 20,
) -> list[dict]:
    """큰 테이블을 행 그룹 단위로 청킹 (헤더 항상 포함)"""
    sheets_data = extract_xlsx_sheets(xlsx_path)
    chunks = []

    for sheet in sheets_data:
        headers = sheet["headers"]
        data_rows = sheet["data_rows"]

        # 행 그룹으로 분할
        for i in range(0, len(data_rows), rows_per_chunk):
            group = data_rows[i:i + rows_per_chunk]

            # Markdown 테이블로 변환
            md_lines = []
            md_lines.append("| " + " | ".join(headers) + " |")
            md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for row in group:
                # 열 수 맞추기 (부족하면 빈 셀 추가)
                padded_row = row + [""] * (len(headers) - len(row))
                md_lines.append("| " + " | ".join(padded_row[:len(headers)]) + " |")

            chunks.append({
                "text": f"## {sheet['sheet_name']}\n\n" + "\n".join(md_lines),
                "metadata": {
                    "source": xlsx_path,
                    "sheet_name": sheet["sheet_name"],
                    "row_range": f"{i+2}-{i+1+len(group)}",  # 엑셀 행번호 기준
                    "total_rows": sheet["row_count"],
                    "type": "table",
                }
            })

    return chunks
```

### 방법 3: 행 단위 자연어 변환 (검색 최적화)

각 행을 자연어 문장으로 변환하여 임베딩 검색에 최적화.

```python
def rows_to_natural_language(xlsx_path: str) -> list[dict]:
    """각 행을 '헤더: 값' 형태의 자연어 문장으로 변환"""
    sheets_data = extract_xlsx_sheets(xlsx_path)
    chunks = []

    for sheet in sheets_data:
        headers = sheet["headers"]
        data_rows = sheet["data_rows"]

        for row_idx, row in enumerate(data_rows):
            # "헤더1: 값1, 헤더2: 값2, ..." 형태로 변환
            pairs = []
            for col_idx, value in enumerate(row):
                if col_idx < len(headers) and value.strip():
                    pairs.append(f"{headers[col_idx]}: {value}")

            if pairs:
                text = f"[{sheet['sheet_name']}] " + ", ".join(pairs)
                chunks.append({
                    "text": text,
                    "metadata": {
                        "source": xlsx_path,
                        "sheet_name": sheet["sheet_name"],
                        "row_number": row_idx + 2,
                        "type": "table_row",
                    }
                })

    return chunks
```

**예시 변환**:
```
# 원본 Excel
| 장비명 | RPM | 온도(°C) | 상태 |
|--------|-----|----------|------|
| Pump-A | 3000| 25.5     | 정상 |

# 변환 결과
"[Sheet1] 장비명: Pump-A, RPM: 3000, 온도(°C): 25.5, 상태: 정상"
```

이 방식이 **벡터 검색에 가장 효과적**인 이유:
- "RPM이 3000인 장비" 같은 쿼리에 잘 매칭됨
- 헤더-값 관계가 자연어로 명시되어 임베딩 품질 높음

### 방법 4: pandas + LLM 요약 (데이터 분석 시트용)

데이터가 많은 시트는 요약 청크를 별도로 생성.

```python
import pandas as pd

def create_sheet_summary(xlsx_path: str) -> list[dict]:
    """각 시트의 통계 요약 청크 생성"""
    xls = pd.ExcelFile(xlsx_path)
    summaries = []

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

        if df.empty:
            continue

        summary_parts = [f"# {sheet_name} 시트 요약"]
        summary_parts.append(f"- 총 {len(df)}행 x {len(df.columns)}열")
        summary_parts.append(f"- 컬럼: {', '.join(df.columns.astype(str))}")

        # 수치형 컬럼 통계
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) > 0:
            summary_parts.append("\n수치 데이터 요약:")
            for col in numeric_cols:
                summary_parts.append(
                    f"- {col}: 범위 {df[col].min():.2f} ~ {df[col].max():.2f}, "
                    f"평균 {df[col].mean():.2f}"
                )

        # 카테고리형 컬럼 고유값
        cat_cols = df.select_dtypes(include="object").columns
        for col in cat_cols:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) <= 10:
                summary_parts.append(f"- {col} 종류: {', '.join(str(v) for v in unique_vals)}")

        summaries.append({
            "text": "\n".join(summary_parts),
            "metadata": {
                "source": xlsx_path,
                "sheet_name": sheet_name,
                "type": "table_summary",
            }
        })

    return summaries
```

### 방법 5: Unstructured 활용

```python
from unstructured.partition.xlsx import partition_xlsx

elements = partition_xlsx(
    filename="data.xlsx",
    infer_table_structure=True,
)

for el in elements:
    print(f"[{type(el).__name__}] Sheet: {el.metadata.page_name}")
    print(f"  {el.text[:100]}")
    if hasattr(el.metadata, "text_as_html"):
        print(f"  HTML: {el.metadata.text_as_html[:100]}")
```

### 권장 종합 파이프라인

```python
def process_engineering_xlsx(xlsx_path: str) -> list[dict]:
    """엔지니어링 Excel 종합 처리"""
    all_chunks = []

    # 1. 시트 요약 청크 (검색 진입점 역할)
    summaries = create_sheet_summary(xlsx_path)
    all_chunks.extend(summaries)

    # 2. 행 단위 자연어 변환 (세부 검색용)
    row_chunks = rows_to_natural_language(xlsx_path)

    # 3. 행이 너무 많으면 행 그룹 청킹으로 대체
    sheets_data = extract_xlsx_sheets(xlsx_path)
    for sheet in sheets_data:
        if sheet["row_count"] > 100:
            # 대형 테이블: 행 그룹 청킹
            group_chunks = chunk_xlsx_by_row_groups(xlsx_path, rows_per_chunk=30)
            all_chunks.extend(group_chunks)
        else:
            # 소형 테이블: 행 단위 자연어
            sheet_rows = [c for c in row_chunks
                         if c["metadata"]["sheet_name"] == sheet["sheet_name"]]
            all_chunks.extend(sheet_rows)

    return all_chunks
```

## 특수 케이스 처리

### 병합 셀 (Merged Cells)

```python
from openpyxl import load_workbook

def handle_merged_cells(xlsx_path: str, sheet_name: str) -> list[list[str]]:
    """병합 셀을 풀어서 모든 셀에 값을 채움"""
    wb = load_workbook(xlsx_path)
    ws = wb[sheet_name]

    # 병합 영역 정보 저장
    merged_ranges = list(ws.merged_cells.ranges)

    # 병합 해제 후 값 채우기
    for merged_range in merged_ranges:
        min_row, min_col = merged_range.min_row, merged_range.min_col
        value = ws.cell(min_row, min_col).value
        ws.unmerge_cells(str(merged_range))
        for row in range(merged_range.min_row, merged_range.max_row + 1):
            for col in range(merged_range.min_col, merged_range.max_col + 1):
                ws.cell(row, col).value = value

    # 데이터 추출
    rows = []
    for row in ws.iter_rows(values_only=True):
        rows.append([str(cell) if cell is not None else "" for cell in row])

    return rows
```

### 다중 테이블이 있는 시트

하나의 시트에 여러 테이블이 떨어져 있는 경우:

```python
def detect_table_regions(xlsx_path: str, sheet_name: str) -> list[dict]:
    """빈 행/열을 기준으로 독립 테이블 영역 감지"""
    wb = load_workbook(xlsx_path, data_only=True)
    ws = wb[sheet_name]

    # 모든 셀 데이터를 2D 배열로 변환
    data = []
    for row in ws.iter_rows(values_only=True):
        data.append(row)

    # 완전히 빈 행 인덱스 찾기
    empty_rows = set()
    for i, row in enumerate(data):
        if all(cell is None for cell in row):
            empty_rows.add(i)

    # 빈 행으로 구분된 테이블 영역 추출
    regions = []
    start = 0
    for i in range(len(data)):
        if i in empty_rows:
            if start < i:
                region_rows = data[start:i]
                regions.append({
                    "start_row": start + 1,
                    "end_row": i,
                    "rows": [[str(c) if c else "" for c in r] for r in region_rows],
                })
            start = i + 1

    # 마지막 영역
    if start < len(data):
        region_rows = data[start:]
        if any(any(c is not None for c in r) for r in region_rows):
            regions.append({
                "start_row": start + 1,
                "end_row": len(data),
                "rows": [[str(c) if c else "" for c in r] for r in region_rows],
            })

    return regions
```

## 도구 비교

| 도구 | 구조 보존 | 수식 처리 | 병합 셀 | 다중 시트 | 대용량 | 비용 |
|------|-----------|-----------|---------|-----------|--------|------|
| openpyxl | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 무료 |
| pandas | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 무료 |
| Unstructured | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 무료 |

## 참고 자료 (References)

- [openpyxl Documentation](https://openpyxl.readthedocs.io/)
- [pandas read_excel](https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html)
- [Unstructured XLSX Partition](https://docs.unstructured.io/open-source/core-functionality/partitioning#partition-xlsx)

## 관련 문서

- [청킹 방법론 총론](./overview-chunking-methods.md)
- [PDF 토큰화 전략](./pdf-tokenization.md)
- [PowerPoint 토큰화 전략](./pptx-tokenization.md)
- [Word 토큰화 전략](./docx-tokenization.md)
