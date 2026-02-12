---
tags: [rag, tokenization, powerpoint, pptx, slides]
level: intermediate
last_updated: 2026-02-12
status: complete
---

# PowerPoint 문서 토큰화 전략 (PPTX Tokenization Strategy)

> PowerPoint는 슬라이드 단위의 시각적 문서로, 텍스트가 짧고 분산되어 있어 일반적인 청킹 전략이 잘 작동하지 않는다.

## 왜 필요한가? (Why)

엔지니어링 분야에서 PowerPoint의 특징:
- **기술 발표 자료**: 공정 설명, 장비 소개, 프로젝트 리뷰
- **짧은 bullet point**: 핵심만 요약된 텍스트 → 맥락 부족
- **다이어그램/차트 의존**: 텍스트만으로 내용 파악 어려움
- **표(Table)**: 데이터 비교, 스펙 정리에 자주 사용
- **Speaker Notes**: 발표자 노트에 상세 설명이 있는 경우

**핵심 과제**: 슬라이드의 시각적 레이아웃과 발표자 노트를 결합하여 의미 있는 청크를 만드는 것

## 핵심 개념 (What)

### PowerPoint 구조 이해

```
PPTX 파일
├── 슬라이드 (Slide)
│   ├── 제목 (Title)
│   ├── 본문 텍스트 (Body Text)
│   ├── 테이블 (Table)
│   ├── 차트 (Chart)
│   ├── 이미지 (Image)
│   └── 도형 내 텍스트 (Shape Text)
├── 발표자 노트 (Speaker Notes)
├── 슬라이드 마스터 (Master/Layout)
└── 미디어 파일 (images, videos)
```

### 청킹 단위 선택

| 전략 | 설명 | 적합한 경우 |
|------|------|-------------|
| **슬라이드 단위** | 1 슬라이드 = 1 청크 | 각 슬라이드가 독립적 토픽 |
| **섹션 단위** | 여러 슬라이드를 섹션으로 묶음 | 연속된 슬라이드가 하나의 토픽 |
| **요소 단위** | 테이블, 차트 등을 개별 청크 | 테이블/차트가 독립적으로 검색되어야 할 때 |

## 어떻게 사용하는가? (How)

### 방법 1: python-pptx - 기본 텍스트 추출

```python
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.shapes import MSO_SHAPE_TYPE

def extract_pptx_by_slide(pptx_path: str) -> list[dict]:
    """슬라이드 단위로 모든 텍스트 요소를 추출"""
    prs = Presentation(pptx_path)
    slides_data = []

    for slide_num, slide in enumerate(prs.slides, 1):
        slide_data = {
            "slide_num": slide_num,
            "title": "",
            "body_texts": [],
            "tables": [],
            "notes": "",
            "shapes_text": [],
        }

        # 슬라이드 제목 추출
        if slide.shapes.title:
            slide_data["title"] = slide.shapes.title.text

        for shape in slide.shapes:
            # 텍스트 프레임 (본문, bullet points)
            if shape.has_text_frame:
                text = shape.text_frame.text.strip()
                if text and text != slide_data["title"]:
                    slide_data["body_texts"].append(text)

            # 테이블
            elif shape.has_table:
                table = shape.table
                table_data = []
                for row in table.rows:
                    row_data = [cell.text for cell in row.cells]
                    table_data.append(row_data)
                slide_data["tables"].append(table_data)

            # 그룹 도형 내 텍스트
            elif shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                for sub_shape in shape.shapes:
                    if sub_shape.has_text_frame:
                        text = sub_shape.text_frame.text.strip()
                        if text:
                            slide_data["shapes_text"].append(text)

        # 발표자 노트
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                slide_data["notes"] = notes

        slides_data.append(slide_data)

    return slides_data
```

### 방법 2: 슬라이드 단위 청킹 (권장 기본 전략)

```python
def chunk_pptx_by_slide(pptx_path: str) -> list[dict]:
    """각 슬라이드를 하나의 의미 있는 청크로 조합"""
    slides_data = extract_pptx_by_slide(pptx_path)
    chunks = []

    for slide in slides_data:
        # 슬라이드 텍스트 조합
        parts = []

        if slide["title"]:
            parts.append(f"# {slide['title']}")

        if slide["body_texts"]:
            parts.append("\n".join(f"- {t}" for t in slide["body_texts"]))

        # 테이블을 Markdown 형태로 변환
        for table in slide["tables"]:
            if len(table) > 1:
                header = " | ".join(table[0])
                separator = " | ".join(["---"] * len(table[0]))
                rows = "\n".join(" | ".join(row) for row in table[1:])
                parts.append(f"{header}\n{separator}\n{rows}")

        # 도형 내 텍스트
        if slide["shapes_text"]:
            parts.append("추가 정보: " + " / ".join(slide["shapes_text"]))

        # 발표자 노트 (상세 설명으로 활용)
        if slide["notes"]:
            parts.append(f"\n상세 설명:\n{slide['notes']}")

        chunk_text = "\n\n".join(parts)

        if chunk_text.strip():
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": pptx_path,
                    "slide_num": slide["slide_num"],
                    "title": slide["title"],
                    "has_table": len(slide["tables"]) > 0,
                    "has_notes": bool(slide["notes"]),
                }
            })

    return chunks
```

### 방법 3: Unstructured 활용

```python
from unstructured.partition.pptx import partition_pptx
from unstructured.chunking.title import chunk_by_title

elements = partition_pptx(
    filename="presentation.pptx",
    include_page_breaks=True,  # 슬라이드 경계 표시
)

# 요소 유형 확인
for el in elements:
    print(f"[{type(el).__name__}] (page {el.metadata.page_number}): {el.text[:80]}")

# 제목 기반 청킹 (슬라이드 제목 단위로 묶임)
chunks = chunk_by_title(
    elements,
    max_characters=1500,
    combine_text_under_n_chars=100,  # 짧은 bullet은 합침
)
```

### 방법 4: Vision LLM으로 슬라이드 이미지 분석

다이어그램이나 차트가 핵심인 슬라이드에 효과적.

```python
import fitz  # PDF 변환 후 이미지 추출
import base64
from openai import OpenAI

def pptx_to_images(pptx_path: str) -> list[bytes]:
    """PPTX를 PDF로 변환 후 페이지별 이미지 추출"""
    import subprocess

    # LibreOffice로 PDF 변환
    subprocess.run([
        "libreoffice", "--headless", "--convert-to", "pdf",
        "--outdir", "/tmp", pptx_path
    ], check=True)

    pdf_path = f"/tmp/{pptx_path.rsplit('/', 1)[-1].replace('.pptx', '.pdf')}"
    doc = fitz.open(pdf_path)

    images = []
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        images.append(pix.tobytes("png"))

    return images


def analyze_slide_with_vision(image_bytes: bytes, slide_num: int) -> str:
    """Vision LLM으로 슬라이드 내용 구조화"""
    client = OpenAI()

    img_b64 = base64.b64encode(image_bytes).decode()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    "이 프레젠테이션 슬라이드의 내용을 구조화된 텍스트로 변환하세요.\n"
                    "- 제목, 본문 텍스트, 테이블, 다이어그램 설명을 포함하세요.\n"
                    "- 다이어그램이나 차트가 있으면 그 내용을 텍스트로 설명하세요.\n"
                    "- 데이터가 있으면 정확히 기록하세요."
                )},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{img_b64}"
                }}
            ]
        }],
        max_tokens=2048,
    )

    return response.choices[0].message.content
```

### 방법 5: 하이브리드 전략 (권장)

텍스트 추출 + 필요 시 Vision 보강.

```python
def hybrid_pptx_chunking(pptx_path: str) -> list[dict]:
    """텍스트가 충분한 슬라이드는 텍스트 기반, 부족하면 Vision 보강"""
    text_chunks = chunk_pptx_by_slide(pptx_path)
    images = pptx_to_images(pptx_path)

    final_chunks = []
    for i, chunk in enumerate(text_chunks):
        text_length = len(chunk["text"].strip())

        if text_length < 50 and i < len(images):
            # 텍스트가 너무 적으면 → Vision LLM으로 보강
            vision_text = analyze_slide_with_vision(images[i], i + 1)
            chunk["text"] = vision_text
            chunk["metadata"]["extraction_method"] = "vision"
        else:
            chunk["metadata"]["extraction_method"] = "text"

        final_chunks.append(chunk)

    return final_chunks
```

## 메타데이터 보강 전략

PowerPoint는 텍스트가 짧아 검색 성능이 떨어질 수 있다. **메타데이터 보강**으로 해결:

```python
def enrich_slide_metadata(chunks: list[dict], pptx_path: str) -> list[dict]:
    """슬라이드 청크에 컨텍스트 메타데이터 추가"""
    prs = Presentation(pptx_path)

    # 전체 프레젠테이션 제목 (첫 슬라이드)
    presentation_title = ""
    if prs.slides[0].shapes.title:
        presentation_title = prs.slides[0].shapes.title.text

    for chunk in chunks:
        # 프레젠테이션 제목 추가
        chunk["metadata"]["presentation_title"] = presentation_title

        # 전후 슬라이드 컨텍스트
        slide_num = chunk["metadata"]["slide_num"]
        if slide_num > 1:
            prev = chunks[slide_num - 2]["metadata"].get("title", "")
            chunk["metadata"]["prev_slide_title"] = prev
        if slide_num < len(chunks):
            next_t = chunks[slide_num]["metadata"].get("title", "")
            chunk["metadata"]["next_slide_title"] = next_t

    return chunks
```

## 도구 비교

| 도구 | 텍스트 | 테이블 | 차트 | 이미지 설명 | 노트 | 비용 |
|------|--------|--------|------|-------------|------|------|
| python-pptx | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ | ❌ | ⭐⭐⭐⭐⭐ | 무료 |
| Unstructured | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ❌ | ⭐⭐⭐ | 무료 |
| Vision LLM | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ | 높음 |
| 하이브리드 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 중간 |

## 참고 자료 (References)

- [python-pptx Documentation](https://python-pptx.readthedocs.io/)
- [Unstructured PPTX Partition](https://docs.unstructured.io/open-source/core-functionality/partitioning#partition-pptx)
- [LibreOffice CLI](https://help.libreoffice.org/latest/en-US/text/shared/guide/start_parameters.html)

## 관련 문서

- [청킹 방법론 총론](./overview-chunking-methods.md)
- [PDF 토큰화 전략](./pdf-tokenization.md)
- [Excel 토큰화 전략](./xlsx-tokenization.md)
- [Word 토큰화 전략](./docx-tokenization.md)
