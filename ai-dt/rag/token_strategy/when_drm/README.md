---
tags: [rag, tokenization, drm, vlm, screenshot, enterprise]
level: intermediate
last_updated: 2026-02-12
reviewed_on: 2026-10-04
review_status: partial
document_type: index
aliases: [DRM 문서 추출 계획 목차]
---

# DRM 환경에서의 문서 토큰화 전략

> DRM 문서의 승인된 입력 조건을 확인하고, 화면 추출 계획과 파일 추출 계획을 구분해 읽는 목차다.

> [!info] 검토 범위 — 2026-10-04
> 이 목차는 2026-02-12의 두 단계 계획을 보존한다. 아래 DRM99%·뷰어/파서 제약·향후 해제는 당시 가정이며 현재 사내 운영 조사 결과가 아니다. 스크린샷+VLM이 유일한 방법이라는 주장은 미확인이다. 열람 가능·추출/캡처 가능·검색 저장/재사용 가능은 별도 조건이다. 실제 DRM 제품/버전·정책·권한·캡처·VLM/파일 파서는 이번 목차 검토에서 실행하지 않았다. 화면 추출/청킹/입력 전환안3개를 추가 검토했다. 실제 모델·제품/권한·사내 운영·Claude 협의는 미확인이다.

## 목적과 적용 조건

DRM은 파일 이름/확장자만으로 판정하지 않는다. 일반 파서의 실패도 보호 상태의 확정 증거가 아니다. 입력 손상·암호화·지원 형식/판본·접근 권한·추출 실패를 구분하고 확인되지 않은 상태는 unknown으로 유지한다. 문서별 승인된 파일/내보내기 입력이 있으면 직접 파싱 후보, 승인된 화면 입력만 있으면 화면 추출 후보로 비교한다. 어느 경로도 확인되지 않으면 해당 문서의 추출은 보류한다.

공식 사례로 [PowerPoint IRM 안내](https://support.microsoft.com/en-us/powerpoint/restrict-access-to-presentations-with-information-rights-management-in-powerpoint)와 [Microsoft 사용 권한](https://learn.microsoft.com/en-us/purview/rights-management-usage-rights)은 열람·복사/추출·인쇄/내보내기 등 권한을 구분한다. 확인일2026-10-04; Microsoft365/지원 Office 제품의 안내이며 모든 DRM 제품이나 사내 정책을 증명하지 않는다. 현재 사내 캡처/재사용·해제 승인 여부는 미확인이다. 승인된 내보내기/직접 파싱을 이용할 수 있는지 확인하기 전 “화면 추출이 유일”하다고 단정하지 않는다.

화면 입력은 보이는 영역의 이미지다. 숨김 셀·노트·원 수식·차트 원데이터를 추출했다는 뜻이 아니며 이미지와 문서/개정/페이지·슬라이드 대응을 보존한다. VLM 구조화 결과는 생성 설명/미확인 판독을 원문 사실과 구분하고 검색에 사용할 권한/범위를 함께 유지한다.

## 읽기 순서와 문서 역할

1. 이 목차에서 문서별 입력/권한·unknown과 Phase1/2가 당시 계획임을 확인한다.
2. [스크린샷/VLM 파이프라인](./screenshot-vlm-pipeline.md)에서 화면 취득과 구조화 요청/출처의 흐름을 읽는다. 제품/업무 환경 실행 조건은 개별 검토 전 미확인이다.
3. [VLM 결과 청킹](./vlm-chunking-strategy.md)에서 추출 결과의 의미/구조·검색 단위와 모델 길이를 구분한다. 화면 취득 설명의 대표 문서는 앞 문서다.
4. [DRM 해제 후 하이브리드 계획](./post-drm-hybrid.md)에서 승인된 원본/내보내기 자료가 확보된 경우의 대안과 원 화면 자료의 관계를 읽는다. 해제는 확정 일정/현재 기능이 아니다.

공통 청킹은 [총론](../overview-chunking-methods.md), 형식별 원본 추출은 아래 관련 문서에 둔다. 취득·청킹·향후 입력 전환은 용도가 달라 고유 맥락을 보존한다. HERDR_ENV=1에서 현재 pane_not_found로 Claude 의견을 받지 못했으므로 세 문서의 완전 통합/업무 경로 선택은 보류한다.

## 2026-02-12 당시 계획을 읽는 기준

아래 기존 절·두 도식은 작성 당시 가정/제안으로 보존한다. DRM Yes/No만으로 캡처/파싱 경로를 자동 승인하는 현재 구현은 아니다. 보호 상태와 승인된 입력 경로를 별도로 확인하고 unknown을 False로 바꾸지 않는다.

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

| 파일 | 내용 |
|------|------|
| [screenshot-vlm-pipeline.md](./screenshot-vlm-pipeline.md) | Phase 1: 스크린샷 + VLM 기반 추출 파이프라인 |
| [vlm-chunking-strategy.md](./vlm-chunking-strategy.md) | VLM 추출 결과물의 청킹 전략 |
| [post-drm-hybrid.md](./post-drm-hybrid.md) | Phase 2: DRM 해제 후 하이브리드 전략 |

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
