---
tags: [rag, tokenization, document-processing, chunking]
level: intermediate
last_updated: 2026-02-12
reviewed_on: 2026-10-04
review_status: partial
document_type: index
aliases: [문서 추출과 청킹 학습 목차]
---

# 문서 토큰화 전략 (Document Tokenization Strategy)

> 입력 형식별로 내용/구조를 추출하고, 검색 단위로 나누고, 모델의 토크나이저로 길이를 확인하는 흐름을 안내한다.

> [!info] 검토 범위 — 2026-10-04
> 원래11개 중 이 목차·청킹 총론·DOCX/XLSX/PDF/PPTX·2026 전략 메모·DRM 목차·화면 추출·추출 결과 청킹·입력 전환안11개를 모두 개별 검토했다. 실제 사내 환경/모델 품질/Claude 협의는 미확인이다. 파일명의 tokenization은 과거 이름으로 보존하며 추출·청킹·토큰화를 같은 단계로 취급하지 않는다. 방법별 최적/품질 우위는 실측 전 가설이다. 기존 작성일과 이번 검토일을 구분한다.

## 왜 필요한가? (Why)

RAG에서는 원문 내용·구조·출처를 추출하고 검색 단위로 나누는 선택이 검색/답변에 영향을 준다. 토큰화는 선택한 모델의 입력 길이를 확인하는 별도 단계다. 임베딩/생성 모델의 한도가 다를 수 있으므로 청크·메타데이터·프롬프트를 합친 실제 입력을 확인한다. 다음은 형식별 비교 후보이며 항상 맞는 처방은 아니다:

- **PowerPoint**: 다이어그램, 테이블, 짧은 bullet point 위주 → 일반 텍스트 분할이 비효율적
- **Excel**: 행/열 구조의 정형 데이터 → 테이블 단위 처리 필요
- **Word**: 긴 보고서, 계층적 헤더 구조 → 구조 기반 분할이 효과적
- **PDF**: 위 모든 형식의 출력물 + 스캔 문서 → 가장 복잡한 처리 필요

문서 형식만으로 최적 전략을 정할 수 없다. 질의/관련 근거가 있는 평가 세트로 추출 누락·구조/출처 보존·검색 recall·답변 근거성·지연/비용을 비교한다. 현재 문서의 업무 환경/회사 비율·DRM 계획은 확인된 운영 사실이 아니다.

## 읽기 순서

1. [청킹 총론](./overview-chunking-methods.md)에서 문자/토큰 단위·구조·의미·late pooling의 차이를 파악한다.
2. 실제 입력에 맞는 PDF/PPTX/XLSX/DOCX 문서에서 추출 가능한 내용·누락되는 자료·출처 단위를 확인한다. DOCX/XLSX/PDF/PPTX는 본문/표/Heading·셀/캐시/행·페이지/표/렌더링·슬라이드/그룹/노트 fixture를 검토했으며 나머지 형식의 상세 예제는 대기다.
3. [2026 전략 메모](./recent-rag-strategy-2026.md)는 작성 당시 제안/환경 가정으로 읽는다. 제목의 recent/2026을 검증일 현재 최신 보장으로 취급하지 않는다.
4. DRM 조건이 있을 때 [DRM 목차](./when_drm/README.md)에서 접근 가능/승인된 입력 조건을 먼저 확인한다. 화면 추출·문서 추출·향후 해제 계획을 구분하며 정책/실제환경은 미확인이다.

## 문서 목록

| 파일 | 내용 |
|------|------|
| [recent-rag-strategy-2026.md](./recent-rag-strategy-2026.md) | 2026-03-14 당시 사내 전략 기록 + 2026-10-04 근거 검토 |
| [overview-chunking-methods.md](./overview-chunking-methods.md) | 청킹 방법론 총론 |
| [pdf-tokenization.md](./pdf-tokenization.md) | PDF 문서 토큰화 전략 |
| [pptx-tokenization.md](./pptx-tokenization.md) | PowerPoint 문서 토큰화 전략 |
| [xlsx-tokenization.md](./xlsx-tokenization.md) | Excel 문서 토큰화 전략 |
| [docx-tokenization.md](./docx-tokenization.md) | Word 문서 토큰화 전략 |

### DRM 환경 전략

| 파일 | 내용 |
|------|------|
| [when_drm/README.md](./when_drm/README.md) | **DRM 문서 처리 전략 (스크린샷 + VLM)** |
| [when_drm/screenshot-vlm-pipeline.md](./when_drm/screenshot-vlm-pipeline.md) | Phase 1: VLM 기반 추출 파이프라인 |
| [when_drm/vlm-chunking-strategy.md](./when_drm/vlm-chunking-strategy.md) | VLM 추출 결과물 청킹 전략 |
| [when_drm/post-drm-hybrid.md](./when_drm/post-drm-hybrid.md) | Phase 2: DRM 해제 후 하이브리드 전략 |

## 핵심 용어

| 용어 | 설명 |
|------|------|
| **Tokenization** | 텍스트를 모델이 처리할 수 있는 토큰 단위로 분리하는 과정 |
| **Chunking** | 검색/저장/입력에 쓸 조각으로 분할하는 과정. 문자·토큰·문서 구조 등 기준을 선택하며 토큰화의 상위 개념이라는 고정 관계는 아님 |
| **Embedding** | 텍스트 청크를 벡터 공간에 매핑하여 의미적 유사도 검색 가능하게 함 |
| **OCR** | Optical Character Recognition. 이미지/스캔 문서에서 텍스트 추출 |
| **Layout Analysis** | 문서의 시각적 레이아웃(헤더, 테이블, 그림)을 인식하는 과정 |

## 방법론과 형식별 문서의 역할

[총론](./overview-chunking-methods.md)은 공통 방법/단위/API 검증의 대표 문서다. 형식별 문서는 파일에서 실제로 무엇을 읽고 어떤 구조/출처를 유지하는지에 집중한다. 유사한 분할 설명과 고유 표/슬라이드/DRM 예제의 완전 통합은 Claude 연결 불가로 보류하며 원문을 제거하지 않는다.

근거: [공식 splitter 개요](https://docs.langchain.com/oss/python/integrations/splitters/index)·[문자 분할](https://docs.langchain.com/oss/python/integrations/splitters/character_text_splitter)·[구조 분할](https://docs.langchain.com/oss/python/integrations/splitters/markdown_header_metadata_splitter). 확인일2026-10-04. 일반 개념/API 안내이며 아래 모든 형식별 파서 실행을 입증하지 않는다.

## 관련 문서

- [RAG 그래프 학습](../langgraph/README.md)
- [Milvus 벡터 DB](../milvus/README.md)
- [OpenSearch](../opensearch/README.md)
