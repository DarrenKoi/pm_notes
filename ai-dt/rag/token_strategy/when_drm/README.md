---
tags: [rag, tokenization, drm, vlm, screenshot, enterprise]
level: intermediate
last_updated: 2026-02-12
status: in-progress
---

# DRM 환경에서의 문서 토큰화 전략

> DRM(Digital Rights Management)이 적용된 문서를 RAG 시스템에 활용하기 위한 현실적인 전략을 정리한다.

## 왜 필요한가? (Why)

### 현실적 제약

사내 문서의 **99%가 DRM 적용** 상태:
- 파일을 직접 파싱(python-pptx, openpyxl 등) 할 수 없음
- DRM 뷰어에서만 열람 가능 → **스크린샷 캡처 후 VLM으로 추출**이 유일한 방법
- 향후 일부 문서에 대해 DRM 해제 가능성 있음

### 두 가지 시나리오

```
현재 (Phase 1): DRM 활성 상태
  → 스크린샷 + VLM 파이프라인

미래 (Phase 2): DRM 일부 해제
  → 해제된 문서: 직접 파싱 (기존 전략)
  → 여전히 DRM: 스크린샷 + VLM 유지
  → 하이브리드 파이프라인 필요
```

## 문서 목록

| 파일 | 내용 | 상태 |
|------|------|------|
| [screenshot-vlm-pipeline.md](./screenshot-vlm-pipeline.md) | Phase 1: 스크린샷 + VLM 기반 추출 파이프라인 | 🟢 |
| [vlm-chunking-strategy.md](./vlm-chunking-strategy.md) | VLM 추출 결과물의 청킹 전략 | 🟢 |
| [post-drm-hybrid.md](./post-drm-hybrid.md) | Phase 2: DRM 해제 후 하이브리드 전략 | 🟢 |

## 전체 아키텍처 요약

```
┌──────────────────────────────────────────────────────────┐
│                   문서 입력 게이트                          │
│                                                          │
│  DRM 파일?  ─── Yes ──→  스크린샷 캡처  → VLM 추출        │
│      │                                      │            │
│      No                                     ▼            │
│      │                              구조화된 텍스트         │
│      ▼                                      │            │
│  직접 파싱 (python-pptx, openpyxl 등)        │            │
│      │                                      │            │
│      └──────────┬───────────────────────────┘            │
│                 ▼                                         │
│          통합 청킹 파이프라인                                │
│                 │                                         │
│                 ▼                                         │
│          벡터 DB 저장 (Milvus/OpenSearch)                  │
└──────────────────────────────────────────────────────────┘
```

## 관련 문서

- [청킹 방법론 총론](../overview-chunking-methods.md)
- [PDF 토큰화 전략](../pdf-tokenization.md)
- [PPTX 토큰화 전략](../pptx-tokenization.md)
- [XLSX 토큰화 전략](../xlsx-tokenization.md)
- [DOCX 토큰화 전략](../docx-tokenization.md)
