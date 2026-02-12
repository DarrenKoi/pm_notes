---
tags: [rag, drm, hybrid, migration, pipeline]
level: intermediate
last_updated: 2026-02-12
status: complete
---

# Phase 2: DRM 해제 후 하이브리드 전략

> DRM이 일부 해제되었을 때, 기존 VLM 파이프라인과 직접 파싱을 통합하는 하이브리드 아키텍처.

## 왜 필요한가? (Why)

DRM 해제는 점진적으로 진행될 가능성이 높다:

```
Phase 1 (현재):  100% DRM  →  전부 스크린샷 + VLM
Phase 2 (전환기):  일부 DRM 해제  →  하이브리드
Phase 3 (이상적):  대부분 해제  →  직접 파싱 위주 + VLM 보조
```

**하이브리드가 필요한 이유**:
- 같은 RAG 시스템에 DRM 문서와 일반 문서가 혼재
- 파이프라인을 두 벌 운영하면 유지보수 비용 증가
- **통합 인터페이스**로 입력 문서를 자동 라우팅해야 함

## 핵심 개념 (What)

### 통합 파이프라인 아키텍처

```
              문서 입력
                │
                ▼
        ┌───────────────┐
        │  DRM 감지 게이트  │
        └───────┬───────┘
                │
        ┌───────┴───────┐
        │               │
    DRM 활성         DRM 해제 (또는 일반 파일)
        │               │
        ▼               ▼
  ┌──────────┐   ┌──────────────┐
  │ VLM 경로  │   │  직접 파싱 경로  │
  │          │   │              │
  │ 스크린샷   │   │ python-pptx  │
  │    ↓     │   │ openpyxl     │
  │  VLM API │   │ python-docx  │
  │    ↓     │   │ PyMuPDF      │
  │ Markdown │   │ Unstructured │
  └────┬─────┘   └──────┬───────┘
       │                │
       └────────┬───────┘
                │
                ▼
        ┌───────────────┐
        │  통합 청킹 엔진  │  ← 동일한 청킹 전략 적용
        └───────┬───────┘
                │
                ▼
        ┌───────────────┐
        │  벡터 DB 저장   │  ← 출처(VLM/파싱) 메타데이터 포함
        └───────────────┘
```

### 핵심 설계 원칙

1. **출력 표준화**: VLM 경로와 파싱 경로 모두 동일한 Markdown 형식으로 출력
2. **메타데이터 추적**: 어떤 경로로 추출했는지 기록 (향후 품질 비교 가능)
3. **자동 라우팅**: DRM 여부를 자동 감지하여 적절한 경로로 분배
4. **점진적 전환**: VLM 경로를 제거하지 않고, 비율만 조정

## 어떻게 사용하는가? (How)

### DRM 감지 게이트

```python
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

class DRMStatus(Enum):
    DRM_ACTIVE = "drm_active"         # DRM 활성 → VLM 경로
    DRM_RELEASED = "drm_released"     # DRM 해제됨 → 직접 파싱
    NO_DRM = "no_drm"                 # 원래 DRM 없음 → 직접 파싱


@dataclass
class DocumentInput:
    file_path: str
    doc_type: str           # pptx, xlsx, docx, pdf
    drm_status: DRMStatus
    screenshots_dir: str | None = None  # DRM인 경우 스크린샷 경로


def detect_drm_status(file_path: str) -> DRMStatus:
    """DRM 상태를 감지하는 로직"""
    path = Path(file_path)

    # 방법 1: 파일 확장자/위치 기반 (사내 규칙에 따라)
    # 예: DRM 해제된 파일은 특정 폴더에 저장
    drm_released_dirs = ["/data/released/", "/shared/open/"]
    if any(d in str(path) for d in drm_released_dirs):
        return DRMStatus.DRM_RELEASED

    # 방법 2: 파일 직접 열기 시도
    try:
        if path.suffix == ".pptx":
            from pptx import Presentation
            Presentation(str(path))  # 열리면 DRM 없음
            return DRMStatus.NO_DRM
        elif path.suffix == ".xlsx":
            from openpyxl import load_workbook
            load_workbook(str(path))
            return DRMStatus.NO_DRM
        elif path.suffix == ".docx":
            from docx import Document
            Document(str(path))
            return DRMStatus.NO_DRM
        elif path.suffix == ".pdf":
            import fitz
            fitz.open(str(path))
            return DRMStatus.NO_DRM
    except Exception:
        return DRMStatus.DRM_ACTIVE

    return DRMStatus.DRM_ACTIVE
```

### 통합 파이프라인 구현

```python
from abc import ABC, abstractmethod

class DocumentExtractor(ABC):
    """문서 추출기 공통 인터페이스"""

    @abstractmethod
    async def extract(self, doc: DocumentInput) -> list[dict]:
        """문서를 추출하여 표준 형태의 청크 리스트 반환

        반환 형식:
        [
            {
                "text": "추출된 텍스트 (Markdown)",
                "metadata": {
                    "source": "파일 경로",
                    "page": 페이지 번호,
                    "doc_type": "pptx|xlsx|docx|pdf",
                    "extraction_method": "vlm|direct_parse",
                    ...
                }
            },
            ...
        ]
        """
        pass


class VLMExtractor(DocumentExtractor):
    """DRM 문서용: 스크린샷 + VLM 추출"""

    def __init__(self, model: str = "gpt-4o"):
        self.model = model

    async def extract(self, doc: DocumentInput) -> list[dict]:
        if not doc.screenshots_dir:
            raise ValueError("DRM 문서는 screenshots_dir 필요")

        # 스크린샷 VLM 추출 (screenshot-vlm-pipeline.md 참조)
        from pathlib import Path
        image_paths = sorted(Path(doc.screenshots_dir).glob("*.png"))

        results = []
        for img_path in image_paths:
            result = await extract_single_image(
                str(img_path), doc.doc_type, self.model
            )
            results.append(result)

        # 표준 형태로 변환
        chunks = []
        for i, result in enumerate(results):
            chunks.append({
                "text": result["extracted_text"],
                "metadata": {
                    "source": doc.file_path,
                    "page": i + 1,
                    "doc_type": doc.doc_type,
                    "extraction_method": "vlm",
                    "vlm_model": self.model,
                }
            })

        return chunks


class DirectParseExtractor(DocumentExtractor):
    """DRM 해제 문서용: 직접 파싱"""

    async def extract(self, doc: DocumentInput) -> list[dict]:
        if doc.doc_type == "pptx":
            return self._extract_pptx(doc)
        elif doc.doc_type == "xlsx":
            return self._extract_xlsx(doc)
        elif doc.doc_type == "docx":
            return self._extract_docx(doc)
        elif doc.doc_type == "pdf":
            return self._extract_pdf(doc)
        else:
            raise ValueError(f"Unsupported doc_type: {doc.doc_type}")

    def _extract_pptx(self, doc: DocumentInput) -> list[dict]:
        """PPTX 직접 파싱 (pptx-tokenization.md 참조)"""
        from pptx import Presentation
        prs = Presentation(doc.file_path)
        chunks = []

        for slide_num, slide in enumerate(prs.slides, 1):
            parts = []
            if slide.shapes.title:
                parts.append(f"# {slide.shapes.title.text}")

            for shape in slide.shapes:
                if shape.has_text_frame:
                    text = shape.text_frame.text.strip()
                    if text and text != (slide.shapes.title.text if slide.shapes.title else ""):
                        parts.append(text)

                elif shape.has_table:
                    table = shape.table
                    rows = [[cell.text for cell in row.cells] for row in table.rows]
                    if rows:
                        md = "| " + " | ".join(rows[0]) + " |\n"
                        md += "| " + " | ".join(["---"] * len(rows[0])) + " |\n"
                        md += "\n".join("| " + " | ".join(r) + " |" for r in rows[1:])
                        parts.append(md)

            if slide.has_notes_slide:
                notes = slide.notes_slide.notes_text_frame.text.strip()
                if notes:
                    parts.append(f"\n[발표자 노트]\n{notes}")

            if parts:
                chunks.append({
                    "text": "\n\n".join(parts),
                    "metadata": {
                        "source": doc.file_path,
                        "page": slide_num,
                        "doc_type": "pptx",
                        "extraction_method": "direct_parse",
                    }
                })

        return chunks

    def _extract_xlsx(self, doc: DocumentInput) -> list[dict]:
        """XLSX 직접 파싱 (xlsx-tokenization.md 참조)"""
        from openpyxl import load_workbook
        wb = load_workbook(doc.file_path, data_only=True)
        chunks = []

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            rows = []
            for row in ws.iter_rows(values_only=True):
                if any(cell is not None for cell in row):
                    rows.append([str(c) if c is not None else "" for c in row])

            if len(rows) > 1:
                # 자연어 변환 (행 단위)
                headers = rows[0]
                for row_idx, row in enumerate(rows[1:]):
                    pairs = []
                    for col_idx, val in enumerate(row):
                        if col_idx < len(headers) and val.strip():
                            pairs.append(f"{headers[col_idx]}: {val}")
                    if pairs:
                        chunks.append({
                            "text": f"[{sheet_name}] " + ", ".join(pairs),
                            "metadata": {
                                "source": doc.file_path,
                                "page": 0,
                                "doc_type": "xlsx",
                                "sheet_name": sheet_name,
                                "row_number": row_idx + 2,
                                "extraction_method": "direct_parse",
                            }
                        })

        return chunks

    def _extract_docx(self, doc: DocumentInput) -> list[dict]:
        """DOCX 직접 파싱 (docx-tokenization.md 참조)"""
        from docx import Document as DocxDocument
        docx_doc = DocxDocument(doc.file_path)
        chunks = []
        current_texts = []
        current_headers = {}

        for para in docx_doc.paragraphs:
            style = para.style.name if para.style else "Normal"
            text = para.text.strip()
            if not text:
                continue

            if style.startswith("Heading"):
                level_str = style.split()[-1]
                level = int(level_str) if level_str.isdigit() else 1

                if level <= 2 and current_texts:
                    header_path = " > ".join(
                        current_headers[k] for k in sorted(current_headers)
                    )
                    chunks.append({
                        "text": "\n\n".join(current_texts),
                        "metadata": {
                            "source": doc.file_path,
                            "page": 0,
                            "doc_type": "docx",
                            "section_path": header_path,
                            "extraction_method": "direct_parse",
                        }
                    })
                    current_texts = []
                    current_headers = {
                        k: v for k, v in current_headers.items() if k < level
                    }

                current_headers[level] = text
                current_texts.append(f"{'#' * level} {text}")
            else:
                current_texts.append(text)

        if current_texts:
            header_path = " > ".join(
                current_headers[k] for k in sorted(current_headers)
            )
            chunks.append({
                "text": "\n\n".join(current_texts),
                "metadata": {
                    "source": doc.file_path,
                    "page": 0,
                    "doc_type": "docx",
                    "section_path": header_path,
                    "extraction_method": "direct_parse",
                }
            })

        return chunks

    def _extract_pdf(self, doc: DocumentInput) -> list[dict]:
        """PDF 직접 파싱 (pdf-tokenization.md 참조)"""
        import fitz
        pdf_doc = fitz.open(doc.file_path)
        chunks = []

        for page_num, page in enumerate(pdf_doc, 1):
            text = page.get_text("text").strip()
            if text:
                chunks.append({
                    "text": text,
                    "metadata": {
                        "source": doc.file_path,
                        "page": page_num,
                        "doc_type": "pdf",
                        "extraction_method": "direct_parse",
                    }
                })

        return chunks
```

### 통합 라우터

```python
class HybridDocumentRouter:
    """DRM 상태에 따라 적절한 추출기로 라우팅"""

    def __init__(self, vlm_model: str = "gpt-4o"):
        self.vlm_extractor = VLMExtractor(model=vlm_model)
        self.direct_extractor = DirectParseExtractor()

    async def process(self, doc: DocumentInput) -> list[dict]:
        """문서를 처리하여 표준 청크 리스트 반환"""

        if doc.drm_status == DRMStatus.DRM_ACTIVE:
            extractor = self.vlm_extractor
        else:
            extractor = self.direct_extractor

        chunks = await extractor.extract(doc)

        # 공통 후처리
        chunks = self._add_common_metadata(chunks, doc)
        return chunks

    def _add_common_metadata(
        self, chunks: list[dict], doc: DocumentInput
    ) -> list[dict]:
        """모든 청크에 공통 메타데이터 추가"""
        for chunk in chunks:
            chunk["metadata"]["drm_status"] = doc.drm_status.value
            chunk["metadata"]["file_name"] = Path(doc.file_path).name
        return chunks

    async def process_batch(
        self, documents: list[DocumentInput]
    ) -> list[dict]:
        """여러 문서를 일괄 처리"""
        import asyncio
        all_chunks = []

        tasks = [self.process(doc) for doc in documents]
        results = await asyncio.gather(*tasks)

        for result in results:
            all_chunks.extend(result)

        return all_chunks
```

### 사용 예시

```python
import asyncio

async def main():
    router = HybridDocumentRouter(vlm_model="gpt-4o")

    documents = [
        # DRM 활성 → VLM 경로
        DocumentInput(
            file_path="/docs/drm/report.pptx",
            doc_type="pptx",
            drm_status=DRMStatus.DRM_ACTIVE,
            screenshots_dir="/screenshots/report/",
        ),
        # DRM 해제 → 직접 파싱
        DocumentInput(
            file_path="/docs/released/specs.xlsx",
            doc_type="xlsx",
            drm_status=DRMStatus.DRM_RELEASED,
        ),
        # 일반 파일 → 직접 파싱
        DocumentInput(
            file_path="/docs/manual.pdf",
            doc_type="pdf",
            drm_status=DRMStatus.NO_DRM,
        ),
    ]

    all_chunks = await router.process_batch(documents)

    print(f"총 {len(all_chunks)}개 청크 생성")
    for chunk in all_chunks[:3]:
        print(f"  [{chunk['metadata']['extraction_method']}] "
              f"{chunk['metadata']['file_name']}: "
              f"{chunk['text'][:50]}...")

asyncio.run(main())
```

## 품질 비교: VLM vs 직접 파싱

DRM이 해제된 문서에 대해 **두 경로의 품질을 비교**하여 VLM 파이프라인 개선에 활용:

```python
async def compare_extraction_quality(
    doc: DocumentInput,
    screenshots_dir: str,
    router: HybridDocumentRouter,
) -> dict:
    """같은 문서를 VLM과 직접 파싱으로 추출하여 비교"""

    # 1. 직접 파싱 (Ground Truth)
    doc_direct = DocumentInput(
        file_path=doc.file_path,
        doc_type=doc.doc_type,
        drm_status=DRMStatus.NO_DRM,
    )
    direct_chunks = await router.direct_extractor.extract(doc_direct)

    # 2. VLM 추출
    doc_vlm = DocumentInput(
        file_path=doc.file_path,
        doc_type=doc.doc_type,
        drm_status=DRMStatus.DRM_ACTIVE,
        screenshots_dir=screenshots_dir,
    )
    vlm_chunks = await router.vlm_extractor.extract(doc_vlm)

    # 3. 비교 메트릭
    direct_text = "\n".join(c["text"] for c in direct_chunks)
    vlm_text = "\n".join(c["text"] for c in vlm_chunks)

    # 텍스트 유사도 (간단한 문자 수준)
    from difflib import SequenceMatcher
    similarity = SequenceMatcher(None, direct_text, vlm_text).ratio()

    # 청크 수 비교
    return {
        "direct_chunks": len(direct_chunks),
        "vlm_chunks": len(vlm_chunks),
        "direct_text_length": len(direct_text),
        "vlm_text_length": len(vlm_text),
        "text_similarity": round(similarity, 4),
        "doc_type": doc.doc_type,
    }
```

이 비교 데이터를 축적하면:
- VLM 프롬프트 튜닝에 활용 가능
- 문서 유형별 VLM 정확도 파악
- DRM 해제 우선순위 결정 근거 (VLM 정확도가 낮은 유형부터 해제)

## 전환 로드맵

```
Phase 1 (현재)
├── VLM 파이프라인 구축 및 운영
├── 스크린샷 자동화 도구 개발
└── VLM 프롬프트 최적화

Phase 2 (DRM 일부 해제 시)
├── HybridDocumentRouter 도입
├── DRM 감지 게이트 구현
├── 품질 비교 프레임워크 가동
└── 직접 파싱 대상 문서 점진 확대

Phase 3 (DRM 대부분 해제 시)
├── 직접 파싱을 기본 경로로 전환
├── VLM은 복잡한 레이아웃/스캔 문서 전용으로 축소
└── VLM 비용 대폭 절감
```

## 참고 자료 (References)

- [PDF 토큰화 전략](../pdf-tokenization.md)
- [PPTX 토큰화 전략](../pptx-tokenization.md)
- [XLSX 토큰화 전략](../xlsx-tokenization.md)
- [DOCX 토큰화 전략](../docx-tokenization.md)

## 관련 문서

- [스크린샷 + VLM 파이프라인](./screenshot-vlm-pipeline.md)
- [VLM 추출 결과 청킹 전략](./vlm-chunking-strategy.md)
- [청킹 방법론 총론](../overview-chunking-methods.md)
