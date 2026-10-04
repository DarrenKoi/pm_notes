---
tags: [rag, tokenization, excel, xlsx, tabular-data]
level: intermediate
last_updated: 2026-02-12
reviewed_on: 2026-10-04
review_status: partial
document_type: learning_note
aliases: [Excel 추출과 행 청킹]
---

# Excel 문서 토큰화 전략 (XLSX Tokenization Strategy)

> Excel은 정형 데이터(테이블)가 핵심이므로, 행/열 구조를 보존하면서 검색 가능한 형태로 변환하는 것이 관건이다.

> [!warning] 값·출처·검증 조건 — 2026-10-04
> openpyxl3.1.5·pandas3.0.3/Python3.14.2의 가상 XLSX로 확인했다. data_only는 수식을 계산하지 않고 저장된 캐시를 읽는다. 캐시가 없거나 빈 결과이면 None일 수 있고 존재해도 최신성은 미확인이다. 원수식·타입·좌표를 별도 보존한다. 실제 Excel 재계산/업무자료·Unstructured·검색품질은 미확인이다. 기본20행·대형100행/30행은 원문의 비교 설정이고 토큰 한도/최적값이 아니다.

읽기 순서: [청킹 총론](./overview-chunking-methods.md) → 셀 추출/헤더 가정 → 행 또는 그룹 → 통계 요약 → 병합/여러 표 한계. 위부터 Python 블록을 정의하고 caller가 확인한 `data.xlsx`를 각 함수에 전달한다. 저장/덮어쓰기·병합 해제 파일 저장·수식 평가/외부 요청을 실행하지 않는다.

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
from openpyxl.utils import get_column_letter
import html


def cell_text(value: object) -> str:
    return "" if value is None else str(value)


def extract_xlsx_sheets(xlsx_path: str) -> list[dict]:
    formulas = load_workbook(xlsx_path, data_only=False)
    try:
        cached = load_workbook(xlsx_path, data_only=True)
        try:
            sheets = []
            for ws in formulas.worksheets:
                records = []
                for row in ws.iter_rows():
                    if not any(cell.value is not None for cell in row):
                        continue
                    cells, values = [], []
                    for cell in row:
                        is_formula = cell.data_type == "f"
                        cache = cached[ws.title][cell.coordinate].value if is_formula else None
                        value = cache if is_formula and cache is not None else cell.value
                        values.append(cell_text(value))
                        cells.append({"coordinate": cell.coordinate, "value": cell.value,
                                      "data_type": cell.data_type, "number_format": cell.number_format,
                                      "formula": cell.value if is_formula else None,
                                      "cached_value": cache,
                                      "cache_status": ("present_unverified" if cache is not None else "missing_or_blank") if is_formula else "not_formula"})
                    records.append({"row_number": row[0].row, "values": values, "cells": cells})
                if not records:
                    continue
                header = records[0]  # caller가 첫 비어 있지 않은 행이 header인지 확인해야 한다.
                labels = [f"{get_column_letter(i)}:{v}" if v else get_column_letter(i)
                          for i, v in enumerate(header["values"], 1)]
                sheets.append({"sheet_name": ws.title, "headers": header["values"],
                               "column_labels": labels, "header_row": header["row_number"],
                               "header_cells": header["cells"], "row_records": records[1:],
                               "data_rows": [r["values"] for r in records[1:]],
                               "row_count": len(records)-1, "col_count": len(header["values"]),
                               "merged_ranges": [str(r) for r in ws.merged_cells.ranges]})
            return sheets
        finally:
            cached.close()
    finally:
        formulas.close()


def markdown_rows(headers: list[str], rows: list[list[str]]) -> str:
    width = len(headers)
    def escape(v: str) -> str:
        return html.escape(v).replace("\\", "\\\\").replace("|", "\\|").replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")
    def line(row: list[str]) -> str:
        if len(row) > width:
            raise ValueError("추가 열을 조용히 잘라내지 않습니다")
        return "| " + " | ".join(escape(v) for v in row + [""] * (width-len(row))) + " |"
    return "\n".join([line(headers), line(["---"]*width), *[line(r) for r in rows]])
```

첫 비어 있지 않은 행을 header로 가정하는 단일 표 예제다. 제목/병합헤더/여러 표가 있는 시트에서는 caller가 header/영역을 지정하는 계약이 추가로 필요하다. 원래 행번호와 원수식·타입·number_format·병합range를 자료로 남기며 날짜/표시값을 문자열로 변환하는 단계는 원래 셀 타입/Excel 표시의 완전 재현이 아니다. 첫 행의 빈/중복 헤더도 column letter로 구분한다. hidden row/column·시트/링크/이미지/차트·외부 참조/수식 결과 최신성은 별도 정책/검증 대상이다. worksheet 전체를 물질화하므로 대용량 streaming/메모리 보장은 없다.

### 방법 2: 행 그룹 청킹 (대형 테이블용)

```python
def row_group_chunks(sheets: list[dict], source: str, rows_per_chunk: int) -> list[dict]:
    if type(rows_per_chunk) is not int or rows_per_chunk <= 0:
        raise ValueError("양의 행 그룹 크기가 필요합니다")
    result = []
    for sheet in sheets:
        records = sheet["row_records"]
        for i in range(0, len(records), rows_per_chunk):
            group = records[i:i+rows_per_chunk]
            numbers = [r["row_number"] for r in group]
            result.append({"text": f"## {sheet['sheet_name']}\n\n" + markdown_rows(sheet["headers"], [r["values"] for r in group]),
                           "metadata": {"source": source, "sheet_name": sheet["sheet_name"],
                                        "header_row": sheet["header_row"], "row_numbers": numbers,
                                        "row_range": f"{numbers[0]}-{numbers[-1]}",
                                        "total_rows": sheet["row_count"], "type": "table"}})
    return result


def chunk_xlsx_by_row_groups(xlsx_path: str, rows_per_chunk: int = 20) -> list[dict]:
    return row_group_chunks(extract_xlsx_sheets(xlsx_path), xlsx_path, rows_per_chunk)
```

### 방법 3: 행 단위 자연어 변환 (검색 최적화)

각 행의 헤더-값 관계를 표현하는 검색 비교 후보다. column letter를 함께 표시해 빈/중복헤더를 구분하고 실제 row_number를 유지한다. 원문의 Pump-A 예시처럼 장비명/RPM/온도/상태 관계를 전달하지만 품질 우위는 실측 전 미확인이다.

```python
def natural_row_chunks(sheets: list[dict], source: str) -> list[dict]:
    chunks = []
    for sheet in sheets:
        for record in sheet["row_records"]:
            pairs = [f"{label}: {value}" for label, value in zip(sheet["column_labels"], record["values"]) if value != ""]
            if pairs:
                chunks.append({"text": f"[{sheet['sheet_name']}] " + ", ".join(pairs),
                               "metadata": {"source": source, "sheet_name": sheet["sheet_name"],
                                            "header_row": sheet["header_row"], "row_number": record["row_number"],
                                            "type": "table_row"}})
    return chunks


def rows_to_natural_language(xlsx_path: str) -> list[dict]:
    return natural_row_chunks(extract_xlsx_sheets(xlsx_path), xlsx_path)
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

"RPM이 3000인 장비" 같은 질의에 헤더 관계를 제공할 수 있다. 정확 수치 조건/집계는 벡터 유사도만으로 보장하지 않으며 원자료를 검증한 구조 질의와 구분한다. 위 원문 표시예시는 열문자 없는 개념형이고 현재 함수 출력은 A:장비명 등 좌표 보강형이다.

### 방법 4: pandas + LLM 요약 (데이터 분석 시트용)

아래는 pandas **통계 요약만** 생성하며 LLM을 호출하지 않는다. 원문의 LLM 요약 맥락은 향후 검증 후보로 보존한다. 타입 추론/빈값·캐시/반올림·헤더 중복 처리로 원자료와 다를 수 있어 요약은 검색 진입점이지 원셀 증거 대체가 아니다. keep_default_na=False로 NA/N/A 같은 문자열을 임의 결측으로 처리하지 않는다. pandas행수에는 사이 빈행이 포함될 수 있어 비어 있지 않은 원데이터행수와 구분한다. 수식 캐시 기반 요약은 계산 성공/최신성을 입증하지 않는다.

```python
import pandas as pd


def create_sheet_summary(xlsx_path: str) -> list[dict]:
    summaries = []
    headers_by_sheet = {s["sheet_name"]: s["header_row"] for s in extract_xlsx_sheets(xlsx_path)}
    with pd.ExcelFile(xlsx_path, engine="openpyxl") as xls:
        for sheet_name in xls.sheet_names:
            # 수식 캐시만 보고 원래 header 행이 바뀌지 않도록 원수식 기준 위치를 사용한다.
            if sheet_name not in headers_by_sheet:
                continue
            header_row = headers_by_sheet[sheet_name] - 1
            df = pd.read_excel(xls, sheet_name=sheet_name, header=header_row, keep_default_na=False)
            if df.empty:
                continue
            parts = [f"# {sheet_name} 시트 요약", f"- pandas 데이터 {len(df)}행 x {len(df.columns)}열",
                     f"- 컬럼: {', '.join(map(str, df.columns))}"]
            for col in df.select_dtypes(include="number").columns:
                values = df[col].dropna()
                if values.empty:
                    parts.append(f"- {col}: 유효 숫자 없음")
                else:
                    parts.append(f"- {col}: 범위 {values.min():.2f} ~ {values.max():.2f}, 평균 {values.mean():.2f}")
            for col in df.select_dtypes(include=["object", "str", "string"]).columns:
                unique = [v for v in df[col].dropna().unique() if v != ""]
                if len(unique) <= 10:
                    parts.append(f"- {col} 종류: {', '.join(map(str, unique))}")
            summaries.append({"text": "\n".join(parts), "metadata": {
                "source": xlsx_path, "sheet_name": sheet_name, "header_row": header_row+1,
                "type": "table_summary", "formula_cache_verified": False,
            }})
    return summaries
```

### 방법 5: Unstructured 활용

```python
def unstructured_xlsx(xlsx_path: str) -> list:
    # 별도 설치/판본/실행 검증 후 caller가 호출한다. 본 검토에서는 import하지 않았다.
    from unstructured.partition.xlsx import partition_xlsx
    return partition_xlsx(filename=xlsx_path)

# page_name/text_as_html은 값이 없을 수 있다. None을 확인한 뒤 slicing한다.
# partition 결과의 시트/셀 출처와 원HTML을 확인하며 parser만으로 추론 품질을 보장하지 않는다.
```

### 권장 종합 파이프라인

```python
def process_engineering_xlsx(xlsx_path: str) -> list[dict]:
    all_chunks = create_sheet_summary(xlsx_path)
    sheets = extract_xlsx_sheets(xlsx_path)
    for sheet in sheets:
        # 현재 시트만 전달해 다른 시트를 중복 적재하지 않는다.
        if sheet["row_count"] > 100:
            all_chunks.extend(row_group_chunks([sheet], xlsx_path, rows_per_chunk=30))
        else:
            all_chunks.extend(natural_row_chunks([sheet], xlsx_path))
    return all_chunks
```

전체 pipeline의 threshold100/그룹30은 평가 후보로 보존했다. 기존 코드는 대형시트마다 모든시트 그룹을 다시 추가하여 중복을 만들었으므로 현재시트만 조립한다. 요약과 상세청크의 type을 구분한다. 원수식/셀자료는 extract_xlsx_sheets 결과로 별도 보관하고 표시청크만 근거의 전부로 쓰지 않는다. 모델 토큰 예산·큰셀/큰행 분할은 별도 구현/평가 대상이다.

## 특수 케이스 처리

### 병합 셀 (Merged Cells)

```python
def handle_merged_cells(xlsx_path: str, sheet_name: str) -> dict:
    wb = load_workbook(xlsx_path, data_only=False)
    try:
        ws = wb[sheet_name]
        rows = [[cell_text(v) for v in row] for row in ws.iter_rows(values_only=True)]
        ranges = []
        for merged in ws.merged_cells.ranges:
            value = cell_text(ws.cell(merged.min_row, merged.min_col).value)
            ranges.append({"range": str(merged), "anchor": ws.cell(merged.min_row, merged.min_col).coordinate,
                           "anchor_value": value})
            # 메모리의 표시용 배열만 채운다. workbook unmerge/수식 재작성/save는 하지 않는다.
            for r in range(merged.min_row-1, merged.max_row):
                for c in range(merged.min_col-1, merged.max_col):
                    rows[r][c] = value
        return {"rows": rows, "merged_ranges": ranges, "projection": "anchor_value_repeat"}
    finally:
        wb.close()
```

handle_merged_cells는 원문의 list 반환에서 rows·anchor/range·projection을 함께 반환하도록 수정했다. 병합영역에 anchor값을 반복하는 표시투영이며 원래 빈셀에 실제값이 있었다는 뜻이 아니다. 수식anchor 반복은 상대참조 재계산/수식복사가 아니다. 원 workbook/파일의 병합은 유지한다.

### 다중 테이블이 있는 시트

빈행으로 세로로 떨어진 영역을 나누는 heuristic이다. 완전히 빈 열·가로로 나란한 표/서식 테이블·header/병합영역을 자동 인식하지 않는다. 원문의 0/False를 빈문자열로 잃는 변환을 is None으로 수정했다:

```python
def detect_table_regions(xlsx_path: str, sheet_name: str) -> list[dict]:
    # 빈 행으로만 나누는 비교 heuristic이다. 가로로 나란한 표/빈 열은 감지하지 않는다.
    wb = load_workbook(xlsx_path, data_only=False)
    try:
        data = list(wb[sheet_name].iter_rows(values_only=True))
        regions, start = [], None
        for i in range(len(data)+1):
            present = i < len(data) and any(v is not None for v in data[i])
            if present and start is None:
                start = i
            if not present and start is not None:
                regions.append({"start_row": start+1, "end_row": i,
                                "rows": [[cell_text(v) for v in row] for row in data[start:i]]})
                start = None
        return regions
    finally:
        wb.close()
```

## 도구 비교

| 도구 | 본문에서 사용하는 기능 | 조건/미확인 |
|---|---|---|
| openpyxl3.1.5 | 셀·원수식/저장캐시·좌표·병합range | 수식 계산/Excel 화면/모든 첨부 추출 아님 |
| pandas3.0.3 | sheet별 읽기·통계 요약 | dtype/결측/헤더 추론·캐시·반올림 영향 |
| Unstructured | XLSX partition 후보 | 실제 설치판본/추출/metadata 미확인 |

원문의 별점/무료/대용량 우위는 공통 측정 근거가 없어 조건표로 바꾸었다. 라이브러리와 운영 자원 비용·모델 비용은 구분한다. 운영 도구/병합 해석·threshold 선택과 완전 중복 통합은 Claude 연결 불가로 보류했다.

## 참고 자료 (References)

- [openpyxl Documentation](https://openpyxl.readthedocs.io/)
- [pandas read_excel](https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html)
- [Unstructured XLSX Partition](https://docs.unstructured.io/open-source/core-functionality/partitioning#partition-xlsx)

확인일 **2026-10-04**. [load_workbook 조건](https://openpyxl.readthedocs.io/en/stable/api/openpyxl.reader.excel.html)·[수식/캐시](https://openpyxl.readthedocs.io/en/stable/tutorial.html)·[병합셀](https://openpyxl.readthedocs.io/en/stable/editing_worksheets.html)·[ExcelFile](https://pandas.pydata.org/docs/reference/api/pandas.ExcelFile.html)을 대조했다. rolling docs의 표시판본(openpyxl3.1.3/pandas3.0.6)과 실제검증판본(3.1.5/3.0.3)이 다르므로 설치판본으로 재확인했다. Unstructured는 공식 partition 안내만 확인하고 실행하지 않았다. 실제 업무자료/Excel 재계산·검색품질은 미확인이다.

## 관련 문서

- [청킹 방법론 총론](./overview-chunking-methods.md)
- [PDF 토큰화 전략](./pdf-tokenization.md)
- [PowerPoint 토큰화 전략](./pptx-tokenization.md)
- [Word 토큰화 전략](./docx-tokenization.md)
