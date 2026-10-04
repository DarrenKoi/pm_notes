---
tags: [rag, tokenization, pdf, ocr, layout-analysis]
level: intermediate
last_updated: 2026-02-12
reviewed_on: 2026-10-04
review_status: partial
document_type: learning_note
aliases: [PDF 추출과 구조 기반 청킹]
---

# PDF 문서 토큰화 전략 (PDF Tokenization Strategy)

> PDF의 텍스트·표·이미지와 페이지 출처를 추출한 뒤 검색 단위로 나눈다. 추출·청킹·모델 토큰화는 서로 다른 단계다.

> [!info] 검토 범위 — 2026-10-04
> PyMuPDF/PyMuPDF4LLM 1.28.2, langchain-text-splitters 1.1.3, OpenAI SDK 3.24.0의 API와 임시 PDF/모의 HTTP를 검증한다. 실제 OCR·Unstructured 파서·사내 VLM·업무 문서·검색 품질은 미확인이다. 작성일 2026-02-12와 검토일을 구분한다. 다섯 방법과 엔지니어링 조립 예제는 비교 후보이며 품질 순위가 아니다.

## 왜 필요한가? (Why)

PDF는 기술 보고서 등에서 내용과 시각 배치를 함께 전달하는 입력 형식이다. 이 저장소에서 실제 사용 비율은 측정하지 않았다:
- 기술 보고서, 장비 매뉴얼, 공정 문서
- 다이어그램, 테이블, 수식이 혼합된 복합 레이아웃
- 같은 파일에도 디지털 텍스트·스캔·OCR 텍스트 레이어가 섞여 페이지별 처리 조건이 달라짐

**문제점**: 단순 텍스트 추출 시 테이블 구조 파괴, 그림 캡션 분리, 헤더/푸터 노이즈 포함

## 핵심 개념 (What)

### PDF의 두 가지 유형

| 유형 | 특징 | 텍스트 추출 | 예시 |
|------|------|-------------|------|
| **디지털 PDF** | 텍스트 레이어 존재 | 직접 추출 가능 | Word/PPT에서 PDF로 변환한 문서 |
| **스캔 PDF** | 텍스트 레이어 없이 이미지 중심 | OCR 후보 | 스캐너로 스캔한 문서, 오래된 매뉴얼 |
| **혼합/OCR PDF** | 페이지별 유형이 다르거나 이미지 위에 인식 텍스트 존재 | 텍스트 품질·누락 영역 확인 후 선택 | 디지털 보고서에 스캔 첨부, OCR 처리 결과 |

빈 텍스트는 스캔의 확정 증거가 아니다. 빈 페이지·벡터로 그린 글자·암호화·추출 실패도 확인한다. 첫 페이지의 50자 기준은 원문에 있던 미검증 휴리스틱이며 전체 파일 분류에 사용하지 않는다. 읽기 순서·표 셀·수식·그림 설명은 원본과 대조한다.

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

텍스트 레이어에서 페이지별 글자·좌표를 읽는 후보다. 속도 우위는 입력과 설정에 따라 측정해야 한다. `fitz` 호환 이름 대신 설치 패키지와 같은 `pymupdf` 이름을 쓴다.

```python
import pymupdf


def extract_text_pymupdf(pdf_path: str) -> list[dict]:
    pages = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            # Table 객체는 페이지 수명에 의존하므로 열린 동안 값으로 복사한다.
            tables = [
                {"bbox": tuple(table.bbox), "rows": table.extract()}
                for table in page.find_tables().tables
            ]
            pages.append({
                "page_num": page.number + 1,  # 인용용 1부터 시작
                "text": page.get_text("text", sort=True),
                "blocks": page.get_text("dict", sort=True)["blocks"],
                "tables": tables,
                "source": pdf_path,
                "ocr_applied": False,
            })
    return pages
```

```python
def extract_structured_blocks(pdf_path: str) -> list[dict]:
    structured = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            for block in page.get_text("dict", sort=True)["blocks"]:
                if block["type"] != 0:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        structured.append({
                            "page": page.number + 1,
                            "source": pdf_path,
                            "text": span["text"],
                            "font_size": span["size"],
                            "font_name": span["font"],
                            "bbox": tuple(span["bbox"]),
                            "is_bold": bool(span["flags"] & pymupdf.TEXT_FONT_BOLD),
                        })
    return structured
```

**활용**: 폰트 크기/볼드는 제목 추정에 쓸 수 있으나 제목의 확정 규칙이 아니다. 폰트명에 `Bold`가 들어가는지 대신 공식 `TEXT_FONT_BOLD` 비트(16)를 확인한다. OCR 결과는 원래 폰트/볼드 정보를 보존하지 못할 수 있다. `sort=True`도 복잡한 다단/회전 문서의 읽기 순서를 보장하지 않는다. 표 결과는 텍스트에도 포함될 수 있어 색인 시 중복을 확인한다. image block의 원본 바이트가 메타데이터에 들어갈 수 있으므로 그대로 JSON/벡터DB 필드에 넣지 않는다.

PyMuPDF에는 `find_tables()`와 `get_textpage_ocr()`가 있다. OCR은 Tesseract와 해당 언어 데이터가 필요하며 만든 TextPage를 `get_text(..., textpage=...)`에 제공한다. 위 코드는 OCR을 호출하지 않는다. 표 탐지/텍스트 추출/렌더링과 OCR 품질 검증은 별개다. [Page API](https://pymupdf.readthedocs.io/en/latest/page.html)·[OCR](https://pymupdf.readthedocs.io/en/latest/recipes-ocr.html)·[폰트 플래그](https://pymupdf.readthedocs.io/en/latest/vars.html), 확인일 2026-10-04.

### 방법 2: Unstructured - 올인원 문서 파싱

제목·본문·표 등 요소를 분류하는 후보다. `fast`는 추출 텍스트, `ocr_only`는 OCR, `hi_res`는 레이아웃 모델, `auto`는 입력/옵션에 따른 선택이다. 모델/OCR 의존성·언어 데이터·실제 fallback을 설치 판본에서 확인한다. 한국어 설정만으로 언어 데이터가 설치되지는 않는다. `hi_res`가 모든 입력에서 더 정확하다는 보장은 없다. 아래 함수는 준비 전 자동 실행하지 않는다. [partition_pdf 공식 안내](https://docs.unstructured.io/open-source/core-functionality/partitioning#partition-pdf), 확인일 2026-10-04; rolling 문서와 실제 설치/실행 차이는 미확인.

```python
def partition_pdf_elements(pdf_path: str, strategy: str = "hi_res") -> list:
    # 별도 의존성/언어 데이터/모델 준비 후 호출. 이 검토에서 파서는 실행하지 않았다.
    from unstructured.partition.pdf import partition_pdf

    if strategy not in {"auto", "fast", "ocr_only", "hi_res"}:
        raise ValueError("지원 전략을 명시하세요")
    return partition_pdf(
        filename=pdf_path,
        strategy=strategy,
        infer_table_structure=(strategy == "hi_res"),
        languages=["kor", "eng"],
    )

# 준비된 환경에서만 실행: elements = partition_pdf_elements("report.pdf")
# for element in elements:
#     print(type(element).__name__, (element.text or "")[:100], element.metadata)
```

**요소 유형**:
- `Title`: 제목/헤더
- `NarrativeText`: 본문 텍스트
- `Table`: 표. 설정/결과에 따라 metadata의 `text_as_html`이 있으며 없을 수도 있음
- `Image`: 탐지된 이미지 요소. 원본/캡션/내용 설명을 모두 추출했다는 뜻은 아님
- `ListItem`: 리스트 항목
- `Footer` / `Header`: 페이지 머리말/꼬리말. 문서 번호·개정·조건이 있으면 보존하고 제거는 명시적으로 선택

```python
def title_chunks(elements: list) -> list:
    from unstructured.chunking.title import chunk_by_title

    return chunk_by_title(
        elements,
        max_characters=1500,          # 문자 hard maximum, 모델 토큰 수 아님
        new_after_n_chars=1000,       # 새 요소를 붙이는 soft boundary
        combine_text_under_n_chars=200,  # 작은 연속 섹션 병합 조건
        multipage_sections=False,    # 이 예제는 페이지 경계를 유지
    )
```

`Title`은 추정 분류다. 작은 섹션 병합은 항상 이전 청크에 붙이는 규칙이 아니며 섹션 경계를 합칠 수 있다. 1500/1000/200은 문자 설정으로 최적값이 아니다. 표는 따로 청킹되며 큰 표는 TableChunk로 나뉠 수 있다. 여러 페이지/요소의 출처는 `metadata.orig_elements`로 대조한다. [공식 청킹 동작](https://docs.unstructured.io/open-source/core-functionality/chunking), 확인일 2026-10-04. 실제 파서·Title 판정·표 HTML 품질은 미확인이다.

### 방법 3: PyMuPDF4LLM - LLM 최적화 Markdown 변환

PDF를 Markdown으로 변환하고 페이지별 출처와 제목을 함께 청킹한다. 변환이 모델 이해도/검색 품질을 보장하지는 않는다. 1.28.2에는 레이아웃 엔진과 OCR 옵션이 있으며 이 예제는 OCR/이미지 파일 쓰기를 끈다. 디지털 페이지 내용만 검증하고 스캔은 빈 결과/이미지 자리표시/누락을 검사해야 한다. 1.28.2의 metadata.page_number는 1-based이며 page_boxes는 레이아웃 위치 정보다. 변환 결과/제목 splitter는 원래 PDF의 글자·여백을 그대로 보존하지 않는다. [공식 API](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/api.html)·[제목 splitter](https://docs.langchain.com/oss/python/integrations/splitters/markdown_header_metadata_splitter), 확인일 2026-10-04.

```python
import pymupdf4llm
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter,
)


def pdf_markdown_chunks(pdf_path: str) -> list:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")],
        strip_headers=False,
    )
    size_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    result = []
    with pymupdf.open(pdf_path) as doc:
        pages = pymupdf4llm.to_markdown(
            doc, page_chunks=True, use_ocr=False,
            write_images=False, embed_images=False, show_progress=False,
        )
    for page in pages:
        sections = header_splitter.split_text(page["text"])
        for section in sections:
            section.metadata.update({
                "source": pdf_path,
                "page": page["metadata"]["page_number"],
                "conversion_metadata": dict(page["metadata"]),
                "ocr_applied": False,
            })
        result.extend(size_splitter.split_documents(sections))
    return result
```

### 방법 4: Document AI 서비스 (클라우드 기반)

> [!warning] 환경 가정과 입력 조건
> 2026-02-12 원문은 외부 API 방화벽 차단·DRM 해제 후 로컬 처리를 전제로 했다. 현재 네트워크/정책/해제 가능 여부는 미확인이다. 승인된 입력과 전송 조건이 확인된 경우에만 후보를 비교한다. DRM 화면 입력의 당시 계획은 [스크린샷 + VLM 파이프라인](./when_drm/screenshot-vlm-pipeline.md)에 보존한다.

| 서비스 | 공식 기능 범위 | 확인할 조건 |
|--------|----------------|-------------|
| [Azure Document Intelligence](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/overview?view=doc-intel-4.0.0) | OCR·레이아웃·표/양식 모델 | API/모델 버전·입력 한도·지원 언어·지역 |
| [Google Document AI](https://docs.cloud.google.com/document-ai/docs/overview) | 문서 처리 processor·추출/분류 | processor 종류/버전·언어·입력 한도·지역 |
| [Amazon Textract](https://docs.aws.amazon.com/textract/latest/dg/what-is.html) | 텍스트·표·양식 데이터 추출 | 호출 API·언어·동기/비동기 입력 조건 |

공식 개요 확인일 2026-10-04. 한국어 지원/정확도와 가격은 서비스 전체에 일반화하지 않는다. 실제 모델/지역의 지원표와 과금 항목을 별도로 확인해야 하며 이 검토에서 호출·가격·사내 연결은 확인하지 않았다.

### 방법 5: Vision LLM 기반 추출 (사내 VLM 활용)

페이지를 PNG로 렌더링하여 이미지 입력을 지원하는 서버에 구조화를 요청한다. 원문 Qwen3-VL/사내 서버는 당시 후보이며 배포/별칭은 미확인이다. 호출자가 확인한 client와 모델 ID를 주입한다. Chat Completions 이미지 data URL 형식의 SDK 요청만 모의 전송으로 확인한다. [공식 이미지 입력 안내](https://platform.openai.com/docs/guides/images-vision), 확인일 2026-10-04; 사내 compatible 서버의 지원은 별도 검증해야 한다.

```python
import base64
from openai import OpenAI


def extract_with_vision(
    pdf_path: str, page_num: int, *, client: OpenAI, model: str,
) -> str:
    """page_num은 0부터 시작하는 렌더링 인덱스. 인용 페이지는 page_num + 1."""
    if not model.strip():
        raise ValueError("서버에서 확인한 모델 ID를 제공하세요")
    with pymupdf.open(pdf_path) as doc:
        if isinstance(page_num, bool) or not isinstance(page_num, int):
            raise ValueError("페이지 인덱스는 정수여야 합니다")
        if not 0 <= page_num < len(doc):
            raise ValueError("페이지 범위 밖입니다")
        pix = doc[page_num].get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
        image = base64.b64encode(pix.tobytes("png")).decode("ascii")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": (
                "이 페이지를 구조화된 Markdown으로 변환하세요. "
                "테이블은 Markdown 표, 리스트는 bullet point, "
                "그림은 [Figure: 설명]으로 표현하세요. "
                "판독할 수 없는 내용은 미확인으로 표시하세요."
            )},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
        ]}],
        max_tokens=4096,
    )
    if not response.choices:
        raise ValueError("응답 choice가 없습니다")
    choice = response.choices[0]
    if choice.finish_reason != "stop":
        raise ValueError(f"완료되지 않은 응답: {choice.finish_reason}")
    text = choice.message.content
    if not isinstance(text, str) or not text.strip():
        raise ValueError("추출 텍스트가 없습니다")
    return text

# client/base_url/인증은 승인된 환경 설정에서 주입하고 사용 후 닫는다.
# 원문 Qwen3-VL-30B 별칭은 실제 서버 ID/배포 확인 전 사용하지 않는다.
```

| 기대하는 활용 | 검증할 한계 |
|---------------|-------------|
| 복잡한 레이아웃을 Markdown으로 표현 | 누락·순서 오류·근거 없는 보완을 원본과 대조 |
| 다이어그램/차트 설명 생성 | 설명은 생성 결과이며 원문 사실/수치 추출의 증거가 아님 |
| OCR과 구조 표현을 한 요청으로 시도 | 모델 이미지 한도·응답 절단·지연/비용·재현성 측정 필요 |

**적합한 경우**: 소량의 고가치 문서, 복잡한 레이아웃, 낮은 스캔 품질을 비교 실험하는 후보. 2배 렌더링/4096 응답 한도는 예제 설정이며 최적값이 아니다. `length`·거부·빈 응답은 성공으로 저장하지 않는다. 반환 문자열에는 출처가 없으므로 호출자가 source와 1-based page를 별도로 저장한다. 응답 내용의 정확성은 위 완료 상태 검사로 입증되지 않는다.

## 엔지니어링 PDF를 위한 권장 파이프라인

처리 전략은 페이지 표본과 실제 목표를 확인한 호출자가 명시한다. 첫 페이지 50자 분류와 후처리 결과를 무시하던 원문 오류를 제거했다. 명시적 필터를 적용한 **같은 요소**를 청킹하고 반환형을 `list[dict]`로 맞춘다. 표 텍스트·HTML·원래 요소 메타데이터를 함께 유지하며 HTML이 없으면 `None`을 보존한다. 헤더 제거는 기본으로 하지 않는다. partitioner/chunker 주입은 모델 없이 조립 흐름을 검증하기 위한 경계다.

```python
from copy import deepcopy
from collections.abc import Callable


def process_engineering_pdf(
    pdf_path: str, *, strategy: str = "hi_res", drop_headers: bool = False,
    partitioner: Callable | None = None, chunker: Callable | None = None,
) -> list[dict]:
    """파싱 → 명시한 필터 → 같은 요소 청킹 → 출처/표 HTML 보존."""
    if strategy not in {"auto", "fast", "ocr_only", "hi_res"}:
        raise ValueError("지원 전략을 명시하세요")
    partitioner = partitioner or partition_pdf_elements
    chunker = chunker or title_chunks
    elements = list(partitioner(pdf_path, strategy=strategy))
    kept = [element for element in elements if not (
        drop_headers and type(element).__name__ in {"Header", "Footer"}
    )]
    result = []
    for chunk in chunker(kept):
        metadata = deepcopy(chunk.metadata.to_dict())
        originals = getattr(chunk.metadata, "orig_elements", None)
        records = []
        for element in (originals if originals is not None else [chunk]):
            records.append({
                "type": type(element).__name__,
                "text": element.text,
                "metadata": deepcopy(element.metadata.to_dict()),
            })
        result.append({
            "text": chunk.text,
            "type": type(chunk).__name__.lower(),
            "table_html": getattr(chunk.metadata, "text_as_html", None),
            "metadata": {**metadata, "source": pdf_path},
            "original_elements": records,
        })
    return result
```

## 도구 비교 요약

| 도구 | 실제 기능/조건 | 이 검토의 증거 |
|------|----------------|----------------|
| PyMuPDF | 텍스트/좌표·표 탐지·렌더링, 별도 OCR 준비 | 생성 PDF에서 텍스트/볼드/표/PNG 확인; OCR 미실행 |
| PyMuPDF4LLM | 페이지 Markdown·레이아웃 엔진·OCR 설정 | 디지털 PDF의 페이지 변환/청킹 확인; OCR 미실행 |
| Unstructured | 전략별 텍스트/OCR/레이아웃·Title 청킹 | 공식 문서/구문·주입 조립 검증; 실제 파서 미실행 |
| Azure Document Intelligence | API/모델별 문서 처리 | 공식 개요만 확인; 호출/한국어 품질/가격 미확인 |
| Vision LLM | 이미지 요청→구조화 텍스트 후보 | 실제 SDK 모의 HTTP; 모델/서버/추출 품질 미확인 |

속도/정확도 별점은 동일 평가 자료가 없어 제거했다. PyMuPDF의 라이선스는 AGPL/상용 선택 조건이 있으므로 일률적으로 무료라고 쓰지 않는다. 다른 도구도 패키지/모델/서비스별 라이선스와 운영 조건을 따로 확인한다. [공식 라이선스 안내](https://pymupdf.readthedocs.io/en/latest/about.html#license-and-copyright), 확인일 2026-10-04; 특정 사내 사용의 적합성 판단은 미확인이다.

## 검증 결과와 남은 조건

세 단계로 원문 절/고유 예제 보존, API/로컬 fixture, 링크/메타데이터/읽기 화면을 검사하고 폴더 정리 기록에 결과를 남긴다. 임시 환경과 새로 만든 PDF만 사용하며 기존 첨부는 처리하지 않는다. 생성 디지털/이미지 페이지·표·볼드·원파일 bytes, Markdown 페이지 출처, SDK PNG 요청/빈 응답/절단 거부, 필터/표 HTML/unknown metadata 보존을 검사한다. 모의 요소는 Unstructured 실제 클래스/API 실행의 증거가 아니다.

Claude 협의는 HERDR_ENV=1에서 현재 pane_not_found로 연결되지 않았다. 도구 우위/임계값·업무 OCR/VLM 운영 선택·다른 문서와의 완전 통합은 보류했다. 공식 기능과 확인 가능한 오류만 수정한다.

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
