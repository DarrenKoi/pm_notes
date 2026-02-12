---
tags: [rag, tokenization, document-processing, chunking]
level: intermediate
last_updated: 2026-02-12
status: in-progress
---

# 문서 토큰화 전략 (Document Tokenization Strategy)

> 다양한 문서 형식(PDF, PPTX, XLSX, DOCX)에 대한 최적의 토큰화/청킹 전략을 정리한다.

## 왜 필요한가? (Why)

RAG 시스템의 성능은 **문서를 얼마나 잘 분할(chunking)하고 토큰화(tokenization)하느냐**에 크게 좌우된다. 특히 엔지니어링 분야에서는:

- **PowerPoint**: 다이어그램, 테이블, 짧은 bullet point 위주 → 일반 텍스트 분할이 비효율적
- **Excel**: 행/열 구조의 정형 데이터 → 테이블 단위 처리 필요
- **Word**: 긴 보고서, 계층적 헤더 구조 → 구조 기반 분할이 효과적
- **PDF**: 위 모든 형식의 출력물 + 스캔 문서 → 가장 복잡한 처리 필요

문서 유형별로 최적 전략이 다르며, 잘못된 청킹은 검색 정확도를 크게 떨어뜨린다.

## 문서 목록

| 파일 | 내용 | 상태 |
|------|------|------|
| [overview-chunking-methods.md](./overview-chunking-methods.md) | 청킹 방법론 총론 | 🟢 |
| [pdf-tokenization.md](./pdf-tokenization.md) | PDF 문서 토큰화 전략 | 🟢 |
| [pptx-tokenization.md](./pptx-tokenization.md) | PowerPoint 문서 토큰화 전략 | 🟢 |
| [xlsx-tokenization.md](./xlsx-tokenization.md) | Excel 문서 토큰화 전략 | 🟢 |
| [docx-tokenization.md](./docx-tokenization.md) | Word 문서 토큰화 전략 | 🟢 |

### DRM 환경 전략

| 파일 | 내용 | 상태 |
|------|------|------|
| [when_drm/](./when_drm/) | **DRM 문서 처리 전략 (스크린샷 + VLM)** | 🟢 |
| [when_drm/screenshot-vlm-pipeline.md](./when_drm/screenshot-vlm-pipeline.md) | Phase 1: VLM 기반 추출 파이프라인 | 🟢 |
| [when_drm/vlm-chunking-strategy.md](./when_drm/vlm-chunking-strategy.md) | VLM 추출 결과물 청킹 전략 | 🟢 |
| [when_drm/post-drm-hybrid.md](./when_drm/post-drm-hybrid.md) | Phase 2: DRM 해제 후 하이브리드 전략 | 🟢 |

## 핵심 용어

| 용어 | 설명 |
|------|------|
| **Tokenization** | 텍스트를 모델이 처리할 수 있는 토큰 단위로 분리하는 과정 |
| **Chunking** | 문서를 의미 있는 조각(chunk)으로 나누는 과정. 토큰화의 상위 개념 |
| **Embedding** | 텍스트 청크를 벡터 공간에 매핑하여 의미적 유사도 검색 가능하게 함 |
| **OCR** | Optical Character Recognition. 이미지/스캔 문서에서 텍스트 추출 |
| **Layout Analysis** | 문서의 시각적 레이아웃(헤더, 테이블, 그림)을 인식하는 과정 |

## 관련 문서

- [RAG 개요](../langgraph/)
- [Milvus 벡터 DB](../milvus/)
- [OpenSearch](../opensearch/)
