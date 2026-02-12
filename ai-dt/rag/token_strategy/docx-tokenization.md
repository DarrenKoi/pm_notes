---
tags: [rag, tokenization, word, docx, document-structure]
level: intermediate
last_updated: 2026-02-12
status: complete
---

# Word 문서 토큰화 전략 (DOCX Tokenization Strategy)

> Word 문서는 계층적 헤더 구조를 가진 장문 문서가 많아, 구조 기반 분할이 가장 효과적이다.

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

### 핵심: Heading 스타일을 이용한 구조 파악

Word의 Heading 스타일은 문서의 **논리적 구조**를 나타냄:
- `Heading 1` → 대분류 (Chapter)
- `Heading 2` → 중분류 (Section)
- `Heading 3` → 소분류 (Subsection)
- `Normal` → 본문 텍스트

## 어떻게 사용하는가? (How)

### 방법 1: python-docx - 구조 인식 추출

```python
from docx import Document
from docx.table import Table

def extract_docx_structure(docx_path: str) -> list[dict]:
    """헤더 계층 구조를 보존하면서 요소별 추출"""
    doc = Document(docx_path)
    elements = []

    for element in doc.element.body:
        tag = element.tag.split("}")[-1]  # 네임스페이스 제거

        if tag == "p":  # 단락
            from docx.text.paragraph import Paragraph
            para = Paragraph(element, doc)

            style_name = para.style.name if para.style else "Normal"
            text = para.text.strip()

            if text:
                elements.append({
                    "type": "paragraph",
                    "style": style_name,
                    "text": text,
                    "is_heading": style_name.startswith("Heading"),
                    "heading_level": int(style_name.split()[-1]) if style_name.startswith("Heading") and style_name.split()[-1].isdigit() else 0,
                })

        elif tag == "tbl":  # 테이블
            table = Table(element, doc)
            rows = []
            for row in table.rows:
                row_data = [cell.text.strip() for cell in row.cells]
                rows.append(row_data)

            if rows:
                elements.append({
                    "type": "table",
                    "style": "Table",
                    "rows": rows,
                    "is_heading": False,
                    "heading_level": 0,
                })

    return elements
```

### 방법 2: 헤더 기반 청킹 (핵심 전략)

```python
def chunk_docx_by_headers(
    docx_path: str,
    max_chunk_size: int = 1500,
    split_level: int = 2,
) -> list[dict]:
    """Heading 스타일을 기준으로 섹션별 청킹"""
    elements = extract_docx_structure(docx_path)
    chunks = []
    current_chunk = {
        "texts": [],
        "headers": {},  # {1: "Chapter 1", 2: "Section 1.1", ...}
    }

    def flush_chunk():
        if current_chunk["texts"]:
            text = "\n\n".join(current_chunk["texts"])
            header_path = " > ".join(
                current_chunk["headers"].get(i, "")
                for i in sorted(current_chunk["headers"])
                if current_chunk["headers"].get(i)
            )
            chunks.append({
                "text": text,
                "metadata": {
                    "source": docx_path,
                    "section_path": header_path,
                    "headers": dict(current_chunk["headers"]),
                    "type": "section",
                }
            })

    for element in elements:
        if element["is_heading"] and element["heading_level"] <= split_level:
            # 새 섹션 시작 → 이전 청크 저장
            flush_chunk()
            level = element["heading_level"]

            # 현재 레벨 이하의 헤더 초기화
            current_chunk = {
                "texts": [f"{'#' * level} {element['text']}"],
                "headers": {
                    k: v for k, v in current_chunk["headers"].items()
                    if k < level
                },
            }
            current_chunk["headers"][level] = element["text"]

        elif element["type"] == "table":
            # 테이블을 Markdown 형태로 변환
            rows = element["rows"]
            if len(rows) > 1:
                header = " | ".join(rows[0])
                separator = " | ".join(["---"] * len(rows[0]))
                body = "\n".join(" | ".join(row) for row in rows[1:])
                table_md = f"| {header} |\n| {separator} |\n" + \
                           "\n".join(f"| {' | '.join(row)} |" for row in rows[1:])
                current_chunk["texts"].append(table_md)
            else:
                current_chunk["texts"].append(" | ".join(rows[0]))

        else:
            current_chunk["texts"].append(element["text"])

    # 마지막 청크
    flush_chunk()

    # 큰 청크는 추가 분할
    final_chunks = []
    for chunk in chunks:
        if len(chunk["text"]) > max_chunk_size:
            sub_chunks = split_large_chunk(chunk, max_chunk_size)
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(chunk)

    return final_chunks


def split_large_chunk(chunk: dict, max_size: int) -> list[dict]:
    """큰 청크를 문단 경계에서 추가 분할"""
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_size,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "],
    )

    texts = splitter.split_text(chunk["text"])
    sub_chunks = []
    for i, text in enumerate(texts):
        sub_chunk = {
            "text": text,
            "metadata": {
                **chunk["metadata"],
                "sub_part": i + 1,
                "total_parts": len(texts),
            }
        }
        sub_chunks.append(sub_chunk)

    return sub_chunks
```

### 방법 3: Markdown 변환 후 청킹

Word → Markdown → 구조 기반 분할의 2단계 접근.

```python
def docx_to_markdown(docx_path: str) -> str:
    """DOCX를 Markdown으로 변환"""
    elements = extract_docx_structure(docx_path)
    md_lines = []

    for element in elements:
        if element["is_heading"]:
            level = element["heading_level"]
            md_lines.append(f"{'#' * level} {element['text']}")
            md_lines.append("")

        elif element["type"] == "table":
            rows = element["rows"]
            if rows:
                md_lines.append("| " + " | ".join(rows[0]) + " |")
                md_lines.append("| " + " | ".join(["---"] * len(rows[0])) + " |")
                for row in rows[1:]:
                    md_lines.append("| " + " | ".join(row) + " |")
                md_lines.append("")

        else:
            md_lines.append(element["text"])
            md_lines.append("")

    return "\n".join(md_lines)
```

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

md_text = docx_to_markdown("report.docx")

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = splitter.split_text(md_text)

# 각 청크에 헤더 메타데이터가 자동 포함됨
for chunk in chunks:
    print(f"Headers: {chunk.metadata}")
    print(f"Content: {chunk.page_content[:100]}...")
```

### 방법 4: Unstructured 활용

```python
from unstructured.partition.docx import partition_docx
from unstructured.chunking.title import chunk_by_title

elements = partition_docx(
    filename="report.docx",
    infer_table_structure=True,
)

# 요소 유형 확인
for el in elements:
    print(f"[{type(el).__name__}] {el.text[:80]}")

# 제목(Heading) 기반 청킹
chunks = chunk_by_title(
    elements,
    max_characters=1500,
    new_after_n_chars=1000,
    combine_text_under_n_chars=200,
)
```

### 방법 5: mammoth - HTML 변환 경유

```python
import mammoth

def docx_via_html(docx_path: str) -> str:
    """DOCX → HTML → 구조화 텍스트"""
    with open(docx_path, "rb") as f:
        result = mammoth.convert_to_html(f)
        html = result.value

    # HTML을 파싱하여 구조화
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    md_parts = []
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "p", "table", "ul", "ol"]):
        if tag.name.startswith("h"):
            level = int(tag.name[1])
            md_parts.append(f"{'#' * level} {tag.get_text()}")
        elif tag.name == "table":
            # 테이블 HTML 보존
            md_parts.append(str(tag))
        elif tag.name in ("ul", "ol"):
            for li in tag.find_all("li"):
                md_parts.append(f"- {li.get_text()}")
        else:
            text = tag.get_text().strip()
            if text:
                md_parts.append(text)

    return "\n\n".join(md_parts)
```

## 계층적 메타데이터 보강

Word 문서의 강점인 계층 구조를 메타데이터로 보존하면 검색 품질이 크게 향상됨:

```python
def add_hierarchical_context(chunks: list[dict]) -> list[dict]:
    """각 청크에 상위 섹션 컨텍스트를 prefix로 추가"""
    for chunk in chunks:
        headers = chunk["metadata"].get("headers", {})
        if headers:
            # "Chapter 1 > Section 1.1 > Subsection 1.1.1" 형태
            context = " > ".join(
                headers[k] for k in sorted(headers) if headers[k]
            )
            # 청크 텍스트 앞에 컨텍스트 추가
            chunk["text"] = f"[{context}]\n\n{chunk['text']}"

    return chunks
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

| 도구 | 헤더 인식 | 테이블 | 이미지 | 스타일 | 속도 | 비용 |
|------|-----------|--------|--------|--------|------|------|
| python-docx | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 무료 |
| Unstructured | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | 무료 |
| mammoth | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | 무료 |
| pandoc | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 무료 |

**Word 문서에서는 python-docx가 가장 권장됨**: 헤더 스타일을 정확히 읽어 구조 기반 분할이 가능

## 참고 자료 (References)

- [python-docx Documentation](https://python-docx.readthedocs.io/)
- [mammoth.js/Python](https://github.com/mwilliamson/python-mammoth)
- [Unstructured DOCX Partition](https://docs.unstructured.io/open-source/core-functionality/partitioning#partition-docx)
- [LangChain MarkdownHeaderTextSplitter](https://python.langchain.com/docs/how_to/markdown_header_metadata_splitter/)

## 관련 문서

- [청킹 방법론 총론](./overview-chunking-methods.md)
- [PDF 토큰화 전략](./pdf-tokenization.md)
- [PowerPoint 토큰화 전략](./pptx-tokenization.md)
- [Excel 토큰화 전략](./xlsx-tokenization.md)
