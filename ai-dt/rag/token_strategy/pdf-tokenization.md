---
tags: [rag, tokenization, pdf, ocr, layout-analysis]
level: intermediate
last_updated: 2026-02-12
status: complete
---

# PDF 문서 토큰화 전략 (PDF Tokenization Strategy)

> PDF는 가장 복잡한 문서 형식이다. 텍스트 기반 PDF와 스캔/이미지 PDF를 구분하여 최적 전략을 적용해야 한다.

## 왜 필요한가? (Why)

PDF는 엔지니어링 분야에서 가장 많이 사용되는 문서 형식:
- 기술 보고서, 장비 매뉴얼, 공정 문서
- 다이어그램, 테이블, 수식이 혼합된 복합 레이아웃
- 텍스트 PDF vs 스캔(이미지) PDF로 나뉘어 처리 전략이 달라짐

**문제점**: 단순 텍스트 추출 시 테이블 구조 파괴, 그림 캡션 분리, 헤더/푸터 노이즈 포함

## 핵심 개념 (What)

### PDF의 두 가지 유형

| 유형 | 특징 | 텍스트 추출 | 예시 |
|------|------|-------------|------|
| **디지털 PDF** | 텍스트 레이어 존재 | 직접 추출 가능 | Word/PPT에서 PDF로 변환한 문서 |
| **스캔 PDF** | 이미지만 존재 | OCR 필요 | 스캐너로 스캔한 문서, 오래된 매뉴얼 |

### PDF 처리 파이프라인

```
PDF 입력
  ├── 디지털 PDF → 텍스트 추출 → 레이아웃 분석 → 청킹
  └── 스캔 PDF   → OCR         → 레이아웃 분석 → 청킹
                                       ↓
                              테이블/그림/텍스트 분리
                                       ↓
                              유형별 토큰화 전략 적용
```

## 어떻게 사용하는가? (How)

### 방법 1: PyMuPDF (fitz) - 빠른 텍스트 추출

가장 빠른 PDF 텍스트 추출. 디지털 PDF에 적합.

```python
import fitz  # PyMuPDF

def extract_text_pymupdf(pdf_path: str) -> list[dict]:
    """페이지 단위로 텍스트와 메타데이터 추출"""
    doc = fitz.open(pdf_path)
    pages = []

    for page_num, page in enumerate(doc):
        text = page.get_text("text")  # 순수 텍스트
        blocks = page.get_text("dict")["blocks"]  # 구조화된 블록

        pages.append({
            "page_num": page_num + 1,
            "text": text,
            "blocks": blocks,  # 위치 정보 포함
            "tables": [],      # 별도 테이블 추출 필요
        })

    return pages
```

```python
# 블록 기반 구조 인식
def extract_structured_blocks(pdf_path: str) -> list[dict]:
    """텍스트 블록을 위치/크기 정보와 함께 추출"""
    doc = fitz.open(pdf_path)
    structured = []

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] == 0:  # 텍스트 블록
                for line in block["lines"]:
                    for span in line["spans"]:
                        structured.append({
                            "page": page_num + 1,
                            "text": span["text"],
                            "font_size": span["size"],
                            "font_name": span["font"],
                            "bbox": span["bbox"],  # 위치 좌표
                            "is_bold": "Bold" in span["font"],
                        })

    return structured
```

**활용**: 폰트 크기/볼드로 헤더를 감지하여 구조 기반 청킹에 활용

### 방법 2: Unstructured - 올인원 문서 파싱

다양한 요소(제목, 본문, 테이블, 이미지)를 자동 분류.

```python
from unstructured.partition.pdf import partition_pdf

# hi_res 전략: 레이아웃 분석 모델 사용 (느리지만 정확)
elements = partition_pdf(
    filename="report.pdf",
    strategy="hi_res",           # "fast", "ocr_only", "hi_res"
    infer_table_structure=True,  # 테이블 구조 인식
    languages=["kor", "eng"],    # OCR 언어 설정
)

# 요소 유형별 분류
for element in elements:
    print(f"Type: {type(element).__name__}")
    print(f"Text: {element.text[:100]}")
    print(f"Metadata: {element.metadata}")
    print("---")
```

**요소 유형**:
- `Title`: 제목/헤더
- `NarrativeText`: 본문 텍스트
- `Table`: 테이블 (HTML 형식으로 구조 보존)
- `Image`: 이미지 (캡션 추출 가능)
- `ListItem`: 리스트 항목
- `Footer` / `Header`: 페이지 헤더/푸터 (보통 제거 대상)

```python
from unstructured.chunking.title import chunk_by_title

# 제목 기반 청킹: 같은 섹션의 요소들을 하나의 청크로 묶음
chunks = chunk_by_title(
    elements,
    max_characters=1500,
    new_after_n_chars=1000,
    combine_text_under_n_chars=200,  # 짧은 텍스트는 이전 청크에 병합
)
```

### 방법 3: PyMuPDF4LLM - LLM 최적화 Markdown 변환

PDF를 Markdown으로 변환하여 LLM이 구조를 잘 이해하도록 함.

```python
import pymupdf4llm

# PDF → Markdown 변환
md_text = pymupdf4llm.to_markdown("report.pdf")

# LangChain과 연동
from langchain.text_splitter import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = splitter.split_text(md_text)
```

### 방법 4: Document AI 서비스 (클라우드 기반)

복잡한 레이아웃이나 스캔 PDF에 대한 고정밀 처리.

| 서비스 | 특징 | 비용 |
|--------|------|------|
| **Azure Document Intelligence** | 테이블/폼 인식 우수, 한국어 지원 | 페이지당 과금 |
| **Google Document AI** | OCR 정확도 높음, 커스텀 모델 학습 가능 | 페이지당 과금 |
| **Amazon Textract** | AWS 생태계 연동, 테이블/폼 추출 | 페이지당 과금 |

```python
# Azure Document Intelligence 예시
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

client = DocumentIntelligenceClient(
    endpoint="https://<resource>.cognitiveservices.azure.com/",
    credential=AzureKeyCredential("<key>")
)

with open("report.pdf", "rb") as f:
    poller = client.begin_analyze_document("prebuilt-layout", body=f)
    result = poller.result()

# 테이블 추출
for table in result.tables:
    print(f"Table: {table.row_count} rows x {table.column_count} cols")
    for cell in table.cells:
        print(f"  [{cell.row_index},{cell.column_index}]: {cell.content}")
```

### 방법 5: Vision LLM 기반 추출 (최신 트렌드)

GPT-4o, Claude 등 멀티모달 모델로 PDF 페이지를 이미지로 인식하여 구조화.

```python
import fitz
import base64
from openai import OpenAI

client = OpenAI()

def extract_with_vision(pdf_path: str, page_num: int) -> str:
    """PDF 페이지를 이미지로 렌더링 후 Vision LLM으로 구조화"""
    doc = fitz.open(pdf_path)
    page = doc[page_num]

    # 페이지를 이미지로 렌더링
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x 해상도
    img_bytes = pix.tobytes("png")
    img_b64 = base64.b64encode(img_bytes).decode()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    "이 문서 페이지의 내용을 구조화된 Markdown으로 변환하세요. "
                    "테이블은 Markdown 테이블로, 리스트는 bullet point로, "
                    "그림은 [Figure: 설명] 형태로 변환하세요."
                )},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{img_b64}"
                }}
            ]
        }],
        max_tokens=4096,
    )

    return response.choices[0].message.content
```

| 장점 | 단점 |
|------|------|
| 복잡한 레이아웃도 정확히 인식 | API 비용 매우 높음 |
| 다이어그램/차트도 설명 가능 | 처리 속도 느림 |
| OCR + 레이아웃 분석 동시 해결 | 대량 문서 처리에 비현실적 |

**적합한 경우**: 소량의 고가치 문서, 복잡한 레이아웃, 스캔 품질이 낮은 문서

## 엔지니어링 PDF를 위한 권장 파이프라인

```python
def process_engineering_pdf(pdf_path: str) -> list[dict]:
    """엔지니어링 PDF 종합 처리 파이프라인"""

    # 1단계: PDF 유형 판별
    doc = fitz.open(pdf_path)
    first_page = doc[0]
    text = first_page.get_text("text").strip()

    if len(text) < 50:
        strategy = "ocr_only"   # 스캔 PDF
    else:
        strategy = "hi_res"      # 디지털 PDF

    # 2단계: Unstructured로 요소 추출
    elements = partition_pdf(
        filename=pdf_path,
        strategy=strategy,
        infer_table_structure=True,
        languages=["kor", "eng"],
    )

    # 3단계: 요소 유형별 후처리
    processed_chunks = []
    for element in elements:
        element_type = type(element).__name__

        # 헤더/푸터 제거
        if element_type in ("Header", "Footer"):
            continue

        # 테이블은 별도 처리: HTML 구조 보존 + 설명 텍스트 추가
        if element_type == "Table":
            processed_chunks.append({
                "text": element.metadata.text_as_html,  # HTML 테이블
                "type": "table",
                "metadata": {
                    "page": element.metadata.page_number,
                    "source": pdf_path,
                }
            })
        else:
            processed_chunks.append({
                "text": element.text,
                "type": element_type.lower(),
                "metadata": {
                    "page": element.metadata.page_number,
                    "source": pdf_path,
                }
            })

    # 4단계: 제목 기반 청킹
    final_chunks = chunk_by_title(
        elements,
        max_characters=1500,
        combine_text_under_n_chars=200,
    )

    return final_chunks
```

## 도구 비교 요약

| 도구 | 속도 | 테이블 | OCR | 레이아웃 | 비용 | 추천 |
|------|------|--------|-----|----------|------|------|
| PyMuPDF | ⭐⭐⭐⭐⭐ | ❌ | ❌ | ❌ | 무료 | 간단한 텍스트 PDF |
| PyMuPDF4LLM | ⭐⭐⭐⭐ | ⭐⭐ | ❌ | ⭐⭐ | 무료 | Markdown 변환용 |
| Unstructured | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 무료/유료 | 범용 (권장) |
| Azure Doc Intel | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 유료 | 고정밀 필요 시 |
| Vision LLM | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 높음 | 소량 고가치 문서 |

## 참고 자료 (References)

- [PyMuPDF Documentation](https://pymupdf.readthedocs.io/)
- [Unstructured.io Documentation](https://docs.unstructured.io/)
- [PyMuPDF4LLM](https://github.com/pymupdf/RAG)
- [Azure Document Intelligence](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/)

## 관련 문서

- [청킹 방법론 총론](./overview-chunking-methods.md)
- [PowerPoint 토큰화 전략](./pptx-tokenization.md)
- [Excel 토큰화 전략](./xlsx-tokenization.md)
- [Word 토큰화 전략](./docx-tokenization.md)
