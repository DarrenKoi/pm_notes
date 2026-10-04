---
tags: [rag, tokenization, word, docx, document-structure]
level: intermediate
last_updated: 2026-02-12
reviewed_on: 2026-10-04
review_status: partial
document_type: learning_note
aliases: [Word 추출과 구조 청킹]
---

# Word 문서 토큰화 전략 (DOCX Tokenization Strategy)

> DOCX에서 본문 순서와 Heading 스타일·표를 추출하고, 출처/섹션을 유지한 검색 단위로 나누는 학습 예제다.

> [!warning] 추출 범위·판본 — 2026-10-04
> 로컬 python-docx1.2.0·Mammoth1.11.0·BeautifulSoup4.15.0·langchain-text-splitters1.1.3에서 생성한 가상 DOCX로 확인했다. 실제 업무 문서/Word 화면/Unstructured partition 실행은 미확인이다. 1500/200은 문자 기준 예시이며 모델 토큰 한도가 아니다. 자동 파일 저장/모델·외부 API 호출은 하지 않는다. 기존 작성일2026-02-12와 검토일을 구분한다.

읽기 순서: [청킹 총론](./overview-chunking-methods.md) → 이 문서의 추출 범위 → Heading 청킹 → Markdown/HTML 대안. 구조 기반 분할의 품질 우위·도구 선택·고유 예제 완전 통합은 실제 평가/Claude 협의 전 미확인이다.

## 왜 필요한가? (Why)

엔지니어링 분야에서 Word의 활용:
- **기술 보고서**: 수십~수백 페이지의 구조화된 장문 문서
- **SOP (Standard Operating Procedure)**: 절차서, 작업 지침서
- **회의록/리뷰 문서**: 항목별 정리된 기록
- **제안서/기획서**: 섹션별 구조가 명확한 문서

**핵심 과제**:
- 문서의 계층적 헤더 구조(Heading 1/2/3)를 활용한 분할
- 테이블, 이미지 캡션 등 비텍스트 요소 보존
- 긴 섹션은 추가 분할, 짧은 섹션은 병합

## 핵심 개념 (What)

### Word 문서 구조

```
DOCX 파일
├── 문서 속성 (제목, 작성자, 생성일)
├── 본문 (Body)
│   ├── 단락 (Paragraph)
│   │   ├── 스타일 (Heading 1, Heading 2, Normal, ...)
│   │   ├── 텍스트 (Run)
│   │   └── 서식 (볼드, 이탤릭, 폰트)
│   ├── 테이블 (Table)
│   │   ├── 행 (Row)
│   │   └── 셀 (Cell) → 내부에 또 단락 포함
│   └── 이미지 (InlineShape)
├── 헤더/푸터 (Header/Footer)
├── 각주/미주 (Footnotes/Endnotes)
└── 목차 (Table of Contents)
```

도식은 DOCX에 있을 수 있는 구성 요소이며 아래 추출기가 모두 처리한다는 뜻은 아니다. python-docx의 본문 iter_inner_content는 Paragraph/Table 순서를 제공한다. 아래는 본문과 표셀의 단락/중첩표를 다루며 머리글/바닥글·각주/미주·주석·이미지/캡션 연결·수정 추적·텍스트박스·목차 갱신/페이지 렌더는 별도 처리/검증 대상이다. 표의 병합 값 반복과 생략셀 위치는 원래 grid 투영의 한계를 기록한다.

### 핵심: Heading 스타일을 이용한 구조 파악

Heading은 스타일로 지정된 논리적 구조 후보다. 글꼴 크기만 큰 단락이나 사용자 정의/상속/outline 스타일을 무조건 내장 Heading으로 판정하지 않는다. 아래는 정확한 `Heading 1`..`Heading 9`만 인식하며 나머지는 미분류로 보존한다. 기본 내장 스타일의 이름과 현지화 UI 표시를 구분한다:
- `Heading 1` → 대분류 (Chapter)
- `Heading 2` → 중분류 (Section)
- `Heading 3` → 소분류 (Subsection)
- `Normal` → 본문 텍스트

## 어떻게 사용하는가? (How)

위부터 Python 블록을 같은 모듈에 정의한다. caller가 확인한 입력에 `extract_docx_structure("report.docx")`로 추출을 검사하고 `chunk_docx_by_headers("report.docx")` 또는 `chunk_via_markdown("report.docx")`를 호출한다. 파일을 저장/덮어쓰지 않는다. 출처·누락·길이를 확인한 뒤 임베딩 단계로 전달한다. 자동 numbering/문단 서식과 run별 bold/링크 대상은 plain text와 같지 않으며 현재 결과에 완전 보존되지 않는다.

### 방법 1: python-docx - 구조 인식 추출

```python
import re
import html
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph


def table_record(table: Table) -> dict:
    rows, omitted, nested = [], [], []
    for r, row in enumerate(table.rows):
        rows.append([""] * row.grid_cols_before + [cell.text for cell in row.cells]
                    + [""] * row.grid_cols_after)
        omitted.append({"before": row.grid_cols_before, "after": row.grid_cols_after})
        for c, cell in enumerate(row.cells):
            for child in cell.tables:
                nested.append({"row": r, "column": c + row.grid_cols_before, "table": table_record(child)})
    return {"type": "table", "style": "Table", "rows": rows,
            "omitted_cells": omitted, "nested_tables": nested,
            "is_heading": False, "heading_level": 0}


def extract_docx_structure(docx_path: str) -> list[dict]:
    doc = Document(docx_path)
    elements = []
    for block_number, element in enumerate(doc.iter_inner_content(), 1):
        if isinstance(element, Paragraph):
            text = element.text
            if not text.strip():
                continue
            name = element.style.name if element.style is not None else ""
            match = re.fullmatch(r"Heading ([1-9])", name)
            record = {"type": "paragraph", "style": name, "text": text,
                      "is_heading": match is not None,
                      "heading_level": int(match.group(1)) if match else 0}
        elif isinstance(element, Table):
            record = table_record(element)
        else:
            continue
        record["block_number"] = block_number
        elements.append(record)
    return elements


def rows_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(map(len, rows), default=0)
    if not width:
        return ""
    def escape(value: str) -> str:
        return html.escape(value).replace("\\", "\\\\").replace("|", "\\|").replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")
    lines = ["| " + " | ".join(escape(v) for v in row + [""] * (width - len(row))) + " |" for row in rows]
    # 첫 행이 실제 header인지 별도 확인한다. 여기서는 표시용 header로 가정한다.
    return "\n".join([lines[0], "| " + " | ".join(["---"] * width) + " |", *lines[1:]])


def table_markdown(record: dict) -> str:
    parts = [rows_markdown(record["rows"])]
    for child in record["nested_tables"]:
        parts.append(f"중첩표(0기준 row={child['row']}, column={child['column']}):\n" + table_markdown(child["table"]))
    return "\n\n".join(p for p in parts if p)
```

추출 결과는 원래 본문 text/행 값을 자료로 유지하고 표시용 Markdown에서 pipe·줄바꿈·HTML 문자를 처리한다. grid_cols_before/after를 기록하며 merged cell 값은 grid 투영에서 반복될 수 있다. 중첩표는 행/열 위치와 함께 별도로 보존하지만 병합 span/문서 레이아웃의 완전 복원은 아니다.

### 방법 2: 헤더 기반 청킹 (핵심 전략)

```python
from copy import deepcopy
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_docx_by_headers(docx_path: str, max_chunk_size: int = 1500,
                          split_level: int = 2) -> list[dict]:
    if type(max_chunk_size) is not int or max_chunk_size <= 0 or type(split_level) is not int or not 1 <= split_level <= 9:
        raise ValueError("양의 문자 크기와 Heading level1..9가 필요합니다")
    chunks, texts, headers, blocks = [], [], {}, []
    def flush() -> None:
        if texts:
            chunks.append({"text": "\n\n".join(texts), "metadata": {
                "source": docx_path, "section_path": " > ".join(headers[k] for k in sorted(headers)),
                "headers": dict(headers), "block_numbers": list(blocks), "type": "section",
            }})
    for element in extract_docx_structure(docx_path):
        if element["is_heading"]:
            level = element["heading_level"]
            if level <= split_level:
                flush()
                texts, blocks = [], []
            headers = {k: v for k, v in headers.items() if k < level}
            headers[level] = element["text"]
            if level > split_level:
                # 상세 Heading은 본문에 보존하되 한 청크의 section metadata는 split 경계까지만 쓴다.
                headers = {k: v for k, v in headers.items() if k <= split_level}
            texts.append(f"{'#' * level} {element['text']}")
        elif element["type"] == "table":
            texts.append(table_markdown(element))
        else:
            texts.append(element["text"])
        blocks.append(element["block_number"])
    flush()
    return [part for chunk in chunks for part in split_large_chunk(chunk, max_chunk_size)]


def split_large_chunk(chunk: dict, max_size: int) -> list[dict]:
    if type(max_size) is not int or max_size <= 0:
        raise ValueError("양의 문자 크기가 필요합니다")
    texts = RecursiveCharacterTextSplitter(
        chunk_size=max_size, chunk_overlap=min(200, max_size - 1),
        separators=["\n\n", "\n", ". ", " ", ""], length_function=len,
    ).split_text(chunk["text"])
    return [{"text": text, "metadata": {
        **deepcopy(chunk["metadata"]), "sub_part": i + 1, "total_parts": len(texts),
    }} for i, text in enumerate(texts)]
```

split_level 이하 Heading에서 section을 시작한다. 더 깊은 Heading은 본문에 남기지만 section metadata는 경계까지의 상위 경로다. 큰 섹션은 문자 재분할하며 source/block_numbers는 section 단위 출처이지 정확한 subchunk별 Word 페이지/좌표가 아니다. 빈 구분자 fallback으로 긴 무공백 입력도 나눈다. 짧은 section 자동 병합·표의 행 단위 분할/제목 반복은 구현하지 않았다. 큰 표를 문자로 자르면 구조가 깨질 수 있으므로 운영에서 표 전용 경로를 설계/평가해야 한다.

### 방법 3: Markdown 변환 후 청킹

Word → Markdown → 구조 기반 분할의 2단계 접근.

```python
def docx_to_markdown(docx_path: str) -> str:
    parts = []
    for element in extract_docx_structure(docx_path):
        if element["is_heading"]:
            parts.append(f"{'#' * element['heading_level']} {element['text']}")
        elif element["type"] == "table":
            parts.append(table_markdown(element))
        else:
            parts.append(element["text"])
    return "\n\n".join(parts)
```

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


def chunk_via_markdown(docx_path: str) -> list:
    md_text = docx_to_markdown(docx_path)  # report.docx는 원문의 caller 입력 예시
    sections = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")],
        strip_headers=False,
    ).split_text(md_text)
    for section in sections:
        section.metadata["source"] = docx_path
    return RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200).split_documents(sections)
```

### 방법 4: Unstructured 활용

```python
def unstructured_docx(docx_path: str) -> list:
    # 별도 설치/판본 검증 후 caller가 호출한다. 여기서는 import/partition을 실행하지 않았다.
    from unstructured.partition.docx import partition_docx
    from unstructured.chunking.title import chunk_by_title
    elements = partition_docx(filename=docx_path)
    return chunk_by_title(elements, max_characters=1500,
                          new_after_n_chars=1000, combine_text_under_n_chars=200)
```

Unstructured의 Title은 partition 결과의 분류이지 내장 Heading과 항상 일치하지 않는다. max_characters는 문자 상한/new_after_n_chars는 soft 크기이고 combine_text_under_n_chars는 작은 section 결합 기준이다. 테이블/원요소와 metadata.orig_elements·문서출처를 확인한다. 원문의 infer_table_structure=True는 DOCX 공식 예제의 조건이 아니라 PDF 맥락과 혼동할 수 있어 제거했다. 이 경로의 설치 판본/실행은 미확인이다.

### 방법 5: mammoth - HTML 변환 경유

`docx_via_html(path)`는 원문의 문자열만 반환하던 예제에서 `text`·원래 `html`·변환 `messages`를 함께 반환하도록 수정했다. text는 중복을 줄인 검색용 투영이고 HTML은 비교용 자료다. 경고와 추출 누락을 확인하며 변환 HTML을 신뢰한 화면에 자동 삽입하지 않는다.

```python
import mammoth
from bs4 import BeautifulSoup


def html_blocks(markup: str) -> str:
    soup = BeautifulSoup(markup, "html.parser")
    parts = []
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "p", "table", "ul", "ol"]):
        if tag.find_parent(["table", "ul", "ol"]):
            continue  # parent block의 자식을 다시 수집하지 않는다.
        if tag.name.startswith("h"):
            parts.append(f"{'#' * int(tag.name[1])} {tag.get_text(' ', strip=True)}")
        elif tag.name == "table":
            rows = [[cell.get_text(' ', strip=True) for cell in row.find_all(['th', 'td'], recursive=False)]
                    for row in tag.find_all('tr') if row.find_parent('table') is tag]
            parts.append(rows_markdown(rows))
        elif tag.name in ("ul", "ol"):
            for i, li in enumerate(tag.find_all("li", recursive=False), 1):
                marker = f"{i}." if tag.name == "ol" else "-"
                parts.append(f"{marker} {li.get_text(' ', strip=True)}")
        else:
            text = tag.get_text(' ', strip=True)
            if text:
                parts.append(text)
    return "\n\n".join(parts)


def docx_via_html(docx_path: str) -> dict:
    with open(docx_path, "rb") as stream:
        result = mammoth.convert_to_html(stream)
    return {"text": html_blocks(result.value), "html": result.value,
            "messages": [{"type": m.type, "message": m.message} for m in result.messages]}

# HTML은 보존 자료다. Mammoth는 sanitize하지 않으므로 자동으로 신뢰 HTML로 렌더하지 않는다.
# 이 text 투영은 중첩 list/rowspan/colspan·스타일·이미지를 완전 재현하지 않는다.
```

## 계층적 메타데이터 보강

섹션 경로를 출처/검색 입력에 포함하는 비교 후보다. 아래는 원문 text를 덮어쓰지 않고 retrieval_text를 생성한다. prefix를 붙인 후 모델 토큰 길이를 다시 확인한다. 실제 품질 향상은 관련성 평가 전 미확인이다:

```python
from copy import deepcopy


def add_hierarchical_context(chunks: list[dict]) -> list[dict]:
    result = deepcopy(chunks)
    for chunk in result:
        headers = chunk["metadata"].get("headers", {})
        context = " > ".join(headers[k] for k in sorted(headers) if headers[k])
        # 원문 text는 유지하고 검색용 파생문자열만 만든다. 반복 호출에도 prefix가 누적되지 않는다.
        chunk["retrieval_text"] = f"[{context}]\n\n{chunk['text']}" if context else chunk["text"]
    return result
```

**예시**:
```
# 원본 청크
"펌프의 최대 RPM은 3000이며, 정상 운전 범위는 1500-2500이다."

# 컨텍스트 보강 후
"[장비 사양서 > 3. 주요 장비 > 3.2 펌프 시스템]
펌프의 최대 RPM은 3000이며, 정상 운전 범위는 1500-2500이다."
```

→ "펌프 RPM" 검색 시 관련 섹션 컨텍스트가 함께 제공됨

## 도구 비교

| 도구 | 이 문서의 경로 | 조건/미확인 |
|---|---|---|
| python-docx1.2.0 | 본문 순서·내장 Heading·표 투영 | 모든 문서파트/렌더·custom 스타일·완전 병합 레이아웃 아님 |
| Unstructured | DOCX partition/Title chunk 후보 | 설치판본/실행 미확인; Title 분류/원요소metadata 확인 |
| Mammoth1.11.0 | HTML·변환경고 보존/텍스트 투영 | HTML sanitize 안함·스타일/표/중첩list 투영 한계 |
| Pandoc | 원문의 변환 도구 후보 | 이번 CLI/판본/실행 미확인 |

원문의 별점/속도/무료/최우선 권고는 공통 측정 근거가 없어 제거했다. 라이브러리 비용과 운영/변환 자원 비용은 구분한다. 입력 특성과 고유내용 보존 검사를 통과한 경로를 비교하며 운영 도구 선택은 Claude 협의 대기다.

## 참고 자료 (References)

- [python-docx Documentation](https://python-docx.readthedocs.io/)
- [mammoth.js/Python](https://github.com/mwilliamson/python-mammoth)
- [Unstructured DOCX Partition](https://docs.unstructured.io/open-source/core-functionality/partitioning#partition-docx)
- [LangChain MarkdownHeaderTextSplitter](https://docs.langchain.com/oss/python/integrations/splitters/markdown_header_metadata_splitter)

출처 확인일 **2026-10-04**. [Document API](https://python-docx.readthedocs.io/en/latest/api/document.html)·[표/생략셀/중첩표](https://python-docx.readthedocs.io/en/latest/user/tables.html)·[스타일 사용](https://python-docx.readthedocs.io/en/latest/user/styles-using.html)·[Unstructured chunking](https://docs.unstructured.io/open-source/core-functionality/chunking)을 대조했다. Mammoth 공식 README의 convert_to_html/messages·sanitization 경계를 확인했다. 로컬 fixture 실행과 업무 문서/Word 화면 검증을 구분한다. 이미지/주석/보안/DRM·실제 검색품질은 미확인이다.

## 관련 문서

- [청킹 방법론 총론](./overview-chunking-methods.md)
- [PDF 토큰화 전략](./pdf-tokenization.md)
- [PowerPoint 토큰화 전략](./pptx-tokenization.md)
- [Excel 토큰화 전략](./xlsx-tokenization.md)
