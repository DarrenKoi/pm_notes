---
tags: [rag, drm, hybrid, migration, pipeline]
level: intermediate
last_updated: 2026-02-12
reviewed_on: 2026-10-04
review_status: partial
document_type: learning_proposal
---

# Phase 2: DRM 해제 후 하이브리드 전략

> 승인된 파일 입력과 화면 이미지 입력을 공통 레코드로 받는 전환 설계 제안이다. DRM 보호 해제를 구현하거나 실제 사내 정책/배포를 조사한 기록은 아니다.

> [!warning] 검토 범위 — 2026-10-04
> 원래2026-02-12의 Phase1/2/3·점진적 해제·비용 감소는 당시 계획/가정이다. 파일 위치·확장자·파서 성공/실패만으로 DRM을 확정하지 않는다. 실제 DRM 제품/권한·사내 서비스·VLM 품질은 미확인. Herdr pane_not_found로 Claude와의 전환/최적 설계 협의는 보류한다. 생성 파일과 SDK mock으로 예제의 출처/라우팅/실패 동작을 검사한다.

[화면 추출](./screenshot-vlm-pipeline.md) → [추출 결과 청킹](./vlm-chunking-strategy.md) → 이 전환안을 읽는다. 공통 출력 형식은 추출 가능 범위가 같다는 뜻이 아니다. 승인된 입력 경로를 명시하고 파일/이미지 상태가 미확인이면 해당 경로를 실행하지 않는다.

## 왜 필요한가? (Why)

아래는 당시 단계별 가정이다. “현재100% DRM”·해제 가능성/일정·이상적 상태를 현재 사실로 확인하지 않았다:

```
Phase 1 (현재):  100% DRM  →  전부 스크린샷 + VLM
Phase 2 (전환기):  일부 DRM 해제  →  하이브리드
Phase 3 (이상적):  대부분 해제  →  직접 파싱 위주 + VLM 보조
```

**하이브리드가 필요한 이유**:
- 같은 RAG 시스템에 DRM 문서와 일반 문서가 혼재
- 파이프라인을 두 벌 운영하면 유지보수 비용 증가
- **통합 인터페이스**에서 확인된 입력 경로/출처를 관리하는 것이 목적. DRM 상태만으로 자동 캡처를 시작하지 않는다.

## 핵심 개념 (What)

### 통합 파이프라인 아키텍처

원래 도식은 설계 맥락으로 보존한다. 아래 DRM 감지의 이진 분기는 제품별 근거가 있어야 하며 코드에서는 `UNKNOWN`과 명시 입력 경로를 분리한다. 공통 Markdown/청킹을 사용해도 DOCX 블록·XLSX 행·PPTX 슬라이드·PDF 페이지의 출처와 미추출 자료는 다르다.

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
3. **명시 라우팅**: DRM 조회가 미구현이면 unknown. 승인된 파일/이미지 입력 경로를 별도로 지정한다.
4. **점진적 전환**: VLM 경로를 제거하지 않고, 비율만 조정

## 어떻게 사용하는가? (How)

### DRM 감지 게이트

기존 `/data/released/`·`/shared/open/` 부분 문자열 검사는 폴더명 오탐을 만들고, 모든 파서 예외를 DRM 활성으로 바꾸면 손상/미지원/권한/파일 없음도 잘못 분류한다. 파싱 성공도 DRM 해제 증거가 아니다. 조회 API가 없으므로 `detect_drm_status`는 unknown이며 열람·파싱·화면 취득·저장/재사용의 실제 허용 범위는 별도로 확인한다.

```python
from dataclasses import dataclass, field
from enum import Enum


class DRMStatus(Enum):
    DRM_ACTIVE = "drm_active"
    DRM_RELEASED = "drm_released"
    NO_DRM = "no_drm"
    UNKNOWN = "unknown"


class ExtractionRoute(Enum):
    DIRECT = "direct_parse"
    VLM = "vlm"


@dataclass
class DocumentInput:
    file_path: str  # 승인된 입력/출처 식별용. 비밀 정보를 로그에 출력하지 않음.
    doc_type: str
    drm_status: DRMStatus = DRMStatus.UNKNOWN
    route: ExtractionRoute | None = None
    file_input_approved: bool | None = None
    image_input_approved: bool | None = None
    screenshots: list[dict] = field(default_factory=list)  # verified ordered manifest


def detect_drm_status(file_path: str) -> DRMStatus:
    """DRM 제품별 상태 조회 API를 구현하지 않았으므로 unknown 반환."""
    return DRMStatus.UNKNOWN  # 확장자/폴더/파서 성공·실패는 DRM 상태의 확정 근거가 아님
```

### 통합 파이프라인 구현

중복 파싱 구현의 오류를 다시 복제하지 않고 같은 주제의 대표 예제를 주입한다. PPTX는 [슬라이드/그룹/표/노트](../pptx-tokenization.md), XLSX는 [원래 행/수식 캐시/병합](../xlsx-tokenization.md), DOCX는 [Heading/본문·표 블록](../docx-tokenization.md), PDF는 [페이지·표/텍스트 레이어](../pdf-tokenization.md)를 기준으로 읽는다. 기존 PPTX 그룹/표현 누락·같은 제목 본문 제거/notes None, XLSX blankrow 제거로 인한 행번호 변경/수식 캐시, DOCX 표 누락/Heading 일반화, PDF 빈 페이지 삭제/닫기 누락을 대표 예제에서 다룬다.

예제를 같은 Python 문맥에서 먼저 정의해야 한다. Markdown 파일 자체를 Python module로 import하는 구현은 아니다. 공통 `page=None`은 unknown이며 `slide_num`, `sheet_name`, `row_number`, `block_numbers`, PDF `page`와 미추출 metadata를 보존한다. 이 기본안은 모든 문서/구조를 추출한 것이 아니므로 원본 대조와 형식별 누락 기록을 확인한다.

```python
from abc import ABC, abstractmethod
from collections.abc import Callable, Awaitable
from copy import deepcopy
import asyncio


class DocumentExtractor(ABC):
    @abstractmethod
    async def extract(self, doc: DocumentInput) -> list[dict]:
        """출처·추출 단위·누락 상태를 보존한 text/metadata records 반환."""
        raise NotImplementedError


class VLMExtractor(DocumentExtractor):
    def __init__(self, extract_batch: Callable[..., Awaitable[list[dict]]], client, model: str):
        if not model:
            raise ValueError("verified served model required")
        self.extract_batch, self.client, self.model = extract_batch, client, model

    async def extract(self, doc: DocumentInput) -> list[dict]:
        if doc.image_input_approved is not True or not doc.screenshots:
            raise ValueError("approved image manifest required")
        # screenshot-vlm-pipeline.md의 extract_document_batch를 명시적으로 주입.
        prompt_type = "general" if doc.doc_type == "pdf" else doc.doc_type
        records = await self.extract_batch(doc.screenshots, self.client, self.model, prompt_type)
        return [{"text": r["extracted_text"], "metadata": {
            "source": doc.file_path, "page": r.get("page_number"),
            "source_image": r["source_image"], "doc_type": doc.doc_type,
            "extraction_method": "vlm", "vlm_model": r["model"],
            "review_status": "unverified", "quality_issues": deepcopy(r.get("quality_issues")),
        }} for r in records]


class DirectParseExtractor(DocumentExtractor):
    def __init__(self, parsers: dict[str, Callable[[str], list[dict]]]):
        self.parsers = dict(parsers)  # 같은 주제의 대표 문서 함수를 호출자가 주입

    async def extract(self, doc: DocumentInput) -> list[dict]:
        if doc.file_input_approved is not True:
            raise ValueError("approved file input required")
        if doc.doc_type not in self.parsers:
            raise ValueError("unsupported or missing parser")
        # 동기 파싱은 thread에서 수행. 파서 실패를 DRM 확정/자동 캡처로 변환하지 않음.
        records = await asyncio.to_thread(self.parsers[doc.doc_type], doc.file_path)
        if not isinstance(records, list):
            raise ValueError("parser must return records list")
        result = []
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("text"), str) or not isinstance(record.get("metadata"), dict):
                raise ValueError("text/metadata parser record required")
            item = deepcopy(record)
            metadata = item["metadata"]
            if metadata.get("source") not in {None, doc.file_path}:
                raise ValueError("parser source mismatch")
            metadata.update(source=doc.file_path, doc_type=doc.doc_type,
                            original_extraction_method=metadata.get("extraction_method"),
                            extraction_method="direct_parse", review_status="unverified")
            metadata.setdefault("page", None)  # DOCX/XLSX의 실제 페이지를 모르면 None
            result.append(item)  # 빈/이미지 전용 기록도 삭제하지 않음
        return result


def pdf_records(pdf_path: str) -> list[dict]:
    # pdf-tokenization.md의 extract_text_pymupdf를 같은 Python 문맥에 먼저 정의.
    return [{"text": r["text"], "metadata": {
        "source": pdf_path, "page": r["page_num"], "ocr_applied": r["ocr_applied"],
        "tables": r["tables"], "extraction_method": "pymupdf_text",
    }} for r in extract_text_pymupdf(pdf_path)]
```

### 통합 라우터

DRM enum이 아니라 `route`와 해당 입력 승인 `is True`로 실행 여부를 결정한다. 파서 실패를 VLM fallback/자동 캡처로 바꾸지 않는다. `None`은 거절/허용 어느 쪽도 확정하지 않은 값이다. 예제 배치는 순차 실행이며 thread 파서의 강제 취소/운영 처리량은 보장하지 않는다.

```python
from copy import deepcopy
from pathlib import Path
import asyncio


class HybridDocumentRouter:
    def __init__(self, vlm_extractor: DocumentExtractor, direct_extractor: DocumentExtractor):
        self.vlm_extractor = vlm_extractor
        self.direct_extractor = direct_extractor

    async def process(self, doc: DocumentInput) -> list[dict]:
        if doc.doc_type not in {"pptx", "xlsx", "docx", "pdf"}:
            raise ValueError("unsupported doc_type")
        if not isinstance(doc.drm_status, DRMStatus):
            raise ValueError("explicit DRM status including UNKNOWN required")
        if doc.route is ExtractionRoute.VLM:
            if doc.image_input_approved is not True:
                raise ValueError("image input approval unknown or denied")
            extractor = self.vlm_extractor
        elif doc.route is ExtractionRoute.DIRECT:
            if doc.file_input_approved is not True:
                raise ValueError("file input approval unknown or denied")
            extractor = self.direct_extractor
        else:
            raise ValueError("explicit input route required")
        return self._add_common_metadata(await extractor.extract(doc), doc)

    def _add_common_metadata(self, chunks: list[dict], doc: DocumentInput) -> list[dict]:
        result = deepcopy(chunks)
        for chunk in result:
            chunk["metadata"].update(drm_status=doc.drm_status.value,
                                     file_name=Path(doc.file_path).name)
        return result

    async def process_batch(self, documents: list[DocumentInput]) -> list[dict]:
        # 순차 배치. 실패를 전체 성공으로 보고하지 않음. 파서 thread의 강제 중단은 보장하지 않음.
        results = []
        for doc in documents:
            results.extend(await self.process(doc))
        return results
```

### 사용 예시

대표 파서 예제와 화면 배치 예제를 먼저 정의하고, 확인된 내부 client/model을 주입한다. 입력 목록은 사용자가 승인한 자료/manifest만 받는다. 아래 코드는 서비스나 실제 뷰어를 import 시 시작하지 않는다.

```python
def build_router(client, served_model: str) -> HybridDocumentRouter:
    """형식별 대표 문서와 화면 추출 예제를 같은 Python 문맥에 정의한 후 호출."""
    parsers = {
        "pptx": chunk_pptx_by_slide,
        "xlsx": rows_to_natural_language,
        "docx": chunk_docx_by_headers,
        "pdf": pdf_records,
    }
    return HybridDocumentRouter(
        VLMExtractor(extract_document_batch, client, served_model),
        DirectParseExtractor(parsers),
    )


async def main(router: HybridDocumentRouter, approved_documents: list[DocumentInput]) -> list[dict]:
    # 실제 승인/manifest는 호출자가 제공. 아래 예제는 문서/스크린샷을 자동 취득하지 않음.
    return await router.process_batch(approved_documents)

# 입력 예: 승인 여부는 기본None. 실제 승인/확인 없이 True로 설정하지 않는다.
# DocumentInput("report.pptx", "pptx", drm_status=DRMStatus.DRM_ACTIVE,
#               route=ExtractionRoute.VLM, image_input_approved=True, screenshots=verified_manifest)
# DocumentInput("specs.xlsx", "xlsx", drm_status=DRMStatus.DRM_RELEASED,
#               route=ExtractionRoute.DIRECT, file_input_approved=True)
# DocumentInput("manual.pdf", "pdf", drm_status=DRMStatus.NO_DRM,
#               route=ExtractionRoute.DIRECT, file_input_approved=True)
# 실행은 await main(router, approved_documents). import 시 live 작업을 시작하지 않는다.
```

## 품질 비교: VLM vs 직접 파싱

양쪽 입력이 승인된 동일 문서/판본/동일 범위를 비교한다. 직접 파싱은 정답(Ground Truth)이 아니다. 본문/노트/숨은 값/표·이미지 포함 범위가 다르므로 `SequenceMatcher`는 문자열 유사도일 뿐 정확도나 검색 품질이 아니다. 원문 대조로 표 셀/숫자/읽기 순서·출처·누락을 평가하고 기준 데이터와 별도로 비교한다. 반환값은 양쪽 records를 보존하며 실제 범위 정렬은 검토 대기다:

```python
from dataclasses import replace
from difflib import SequenceMatcher


async def compare_extraction_quality(doc: DocumentInput, router: HybridDocumentRouter) -> dict:
    """양쪽 입력 승인을 보존한 비교. 직접 파싱은 ground truth가 아님."""
    if doc.file_input_approved is not True or doc.image_input_approved is not True:
        raise ValueError("both input approvals required")
    direct = await router.process(replace(doc, route=ExtractionRoute.DIRECT))
    vlm = await router.process(replace(doc, route=ExtractionRoute.VLM))
    direct_text = "\n".join(c["text"] for c in direct)
    vlm_text = "\n".join(c["text"] for c in vlm)
    # 빈 입력끼리 ratio1을 품질100%로 해석하지 않음. autojunk=False 명시.
    similarity = (SequenceMatcher(None, direct_text, vlm_text, autojunk=False).ratio()
                  if direct_text.strip() and vlm_text.strip() else None)
    return {"direct_chunks": len(direct), "vlm_chunks": len(vlm),
            "direct_text_length": len(direct_text), "vlm_text_length": len(vlm_text),
            "text_similarity": round(similarity, 4) if similarity is not None else None,
            "doc_type": doc.doc_type, "review_status": "unverified",
            "direct_records": direct, "vlm_records": vlm}
```

이 비교 데이터를 축적하면:
- VLM 프롬프트 튜닝에 활용 가능
- 문서 유형별 출력 차이와 누락 후보 파악; 정확도는 원문/검토 기준으로 별도 측정
- 당시 전환 우선순위 논의의 입력 후보. 문자열 유사도만으로 보호/운영 정책을 결정하지 않는다.

## 전환 로드맵

아래는 원래2026-02-12 계획을 그대로 보존한다. 현재 운영/해제 일정·비용 절감·최적 경로의 확인 결과가 아니다.

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

확인일 **2026-10-04**. Python 로컬3.14.2, SDK3.24.0/Pillow12.3.0/python-pptx1.0.2/python-docx1.2.0/openpyxl3.1.5/PyMuPDF1.28.2/text-splitters1.1.3으로 생성 자료를 검사했다. 공식 stable 페이지의 표시 판본과 로컬 설치 판본은 같다고 가정하지 않는다.

- [Python SequenceMatcher](https://docs.python.org/3/library/difflib.html) — ratio/순서/반복 heuristic; 정확도 지표가 아님. autojunk=False 설정을 명시한다.
- [Python asyncio tasks](https://docs.python.org/3/library/asyncio-task.html) — 동기 파싱 to_thread/async orchestration; 실행 완료와 실제 업무 성공은 별도.
- [python-docx Document API](https://python-docx.readthedocs.io/en/latest/api/document.html) — 본문/표 block 접근 범위.
- [openpyxl load_workbook](https://openpyxl.readthedocs.io/en/stable/api/openpyxl.reader.excel.html) — data_only는 저장된 수식 캐시, 재계산이 아님.

형식별 상세 출처/검증 범위는 다음 대표 문서에 남겼다:

- [PDF 토큰화 전략](../pdf-tokenization.md)
- [PPTX 토큰화 전략](../pptx-tokenization.md)
- [XLSX 토큰화 전략](../xlsx-tokenization.md)
- [DOCX 토큰화 전략](../docx-tokenization.md)

## 관련 문서

- [스크린샷 + VLM 파이프라인](./screenshot-vlm-pipeline.md)
- [VLM 추출 결과 청킹 전략](./vlm-chunking-strategy.md)
- [청킹 방법론 총론](../overview-chunking-methods.md)
