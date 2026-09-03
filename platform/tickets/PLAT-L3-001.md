# PLAT-L3-001 검색 문서와 질의 계약을 확정한다

## 식별

- 티켓 ID: `PLAT-L3-001`
- 소속 계층: `L3` — `platform/02-architecture.md §3`
- 관련 결정: `D1`, `D4` — `platform/02-architecture.md §결정 요약`
- 선행 티켓: `[PLAT-L0-001, PLAT-L2-001]`

## 맥락(Context)

- 문제: 파생 검색 인덱스가 받아들일 문서와 질의·결과 형식이 없어 색인기와 질의 구현을 독립적으로 만들 수 없다.
- 필요한 이유: MongoDB 카탈로그를 진실 원천으로 유지하면서 OpenSearch를 교체 가능한 파생 인덱스로 격리해야 한다.
- 설계 근거:
  - `platform/02-architecture.md §3 소프트웨어 계층`
  - `platform/02-architecture.md §5.1 agents 컬렉션`
  - `platform/01-requirements.md FR-08, FR-09`
- 시작 파일:
  - `src/registry/catalog/models.py`
  - `schemas/2026-09-03/agent.schema.json`

## 범위(In scope)

- [ ] `SearchDocument`, `SearchQuery`, `SearchHit`과 `SearchBackend` Protocol을 정의한다.
- [ ] 필터 필드, 페이지 경계와 허용 정렬값을 고정하는 계약 테스트를 추가한다.
- 변경 허용 경로: `src/registry/search/contracts.py`, `tests/unit/test_search_contract.py`

## 범위 밖(Out of scope)

- OpenSearch 클라이언트와 실제 색인·질의 구현
- HTTP 검색 라우터와 화면

## 수용 기준(Acceptance Criteria)

- [ ] AC-1: 카탈로그 문서에서 이름·버전·종류·런타임·소유 팀·상태·등급·갱신 시각을 포함한 `SearchDocument`를 만들 수 있다.
- [ ] AC-2: 빈 검색어와 허용된 종류·런타임·팀·상태 필터 및 페이지 크기 1~100은 계약 검증을 통과한다.
- [ ] AC-3: 페이지 크기 0 또는 101과 미지원 정렬값은 검증 오류로 거부된다.

## 검증 방법

| 수용 기준 | 실행 명령 | 기대 결과 |
|---|---|---|
| AC-1 | `pytest tests/unit/test_search_contract.py::test_search_document_contains_catalog_identity -q` | 종료 코드 0, 검색 문서 필수 필드 검증 통과 |
| AC-2 | `pytest tests/unit/test_search_contract.py::test_search_query_accepts_supported_filters -q` | 종료 코드 0, 허용 질의 검증 통과 |
| AC-3 | `pytest tests/unit/test_search_contract.py::test_search_query_rejects_invalid_page_and_sort -q` | 종료 코드 0, 잘못된 질의 거부 검증 통과 |

## 건드리면 안 되는 것

- 수정 금지 파일: `src/registry/catalog/**`, `src/registry/api/**`
- 변경 금지 인터페이스: `PLAT-L2-001`의 카탈로그 문서 스키마
- 동시 작업 소유 경로: `PLAT-L2-001: src/registry/catalog/**, schemas/**`

## 참고 규약

- `platform/tickets/CONVENTIONS.md` — 구현 경로와 표준 명령
- `platform/05-ticket-conventions.md §3` — 계약 수용 기준 작성 방식

## 완료 보고

- 변경: `src/registry/search/contracts.py`, `tests/unit/test_search_contract.py`
- 검증: `pytest tests/unit/test_search_contract.py -q` → 전체 통과
- 남은 일: `PLAT-L3-002`, `PLAT-L3-003`
- 범위 밖 발견: 없음
