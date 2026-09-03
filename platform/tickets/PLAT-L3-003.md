# PLAT-L3-003 권한 범위 안에서 검색 결과를 반환한다

## 식별

- 티켓 ID: `PLAT-L3-003`
- 소속 계층: `L3` — `platform/02-architecture.md §3`
- 관련 결정: `D1`, `D7`, `D15` — `platform/02-architecture.md §결정 요약`
- 선행 티켓: `[PLAT-L2-004, PLAT-L3-001]`

## 맥락(Context)

- 문제: 키워드·필터·정렬을 검색 계약에 맞게 실행하고 권한 밖 문서를 제외하는 질의 서비스가 없다.
- 필요한 이유: HTTP 계층이 OpenSearch 문법이나 인가 튜플을 직접 조립하지 않게 해야 한다.
- 설계 근거:
  - `platform/02-architecture.md §3 L3 Search Index`
  - `platform/02-architecture.md §5.2 소유권`
  - `platform/01-requirements.md FR-07~09, FR-25`
- 시작 파일:
  - `src/registry/search/contracts.py`
  - `src/registry/auth/decide.py`

## 범위(In scope)

- [ ] `SearchQuery`와 호출자 권한 범위를 받아 `SearchHit` 페이지를 반환하는 서비스를 구현한다.
- [ ] 가짜 검색 백엔드와 인가 판정기로 필터·정렬·권한 제외를 검증한다.
- 변경 허용 경로: `src/registry/search/query.py`, `tests/unit/test_search_query.py`

## 범위 밖(Out of scope)

- FastAPI 라우터와 OIDC 토큰 파싱
- 검색 랭킹 튜닝과 UI

## 수용 기준(Acceptance Criteria)

- [ ] AC-1: 키워드와 종류·런타임·팀·상태 필터를 검색 백엔드에 그대로 전달하고 계약 형식의 페이지를 반환한다.
- [ ] AC-2: 인가 판정이 거부한 문서는 결과와 전체 건수에서 제외된다.
- [ ] AC-3: `degraded` 문서는 동일 점수의 `active` 문서보다 뒤에 배치되고 경고 상태가 결과에 보존된다.

## 검증 방법

| 수용 기준 | 실행 명령 | 기대 결과 |
|---|---|---|
| AC-1 | `pytest tests/unit/test_search_query.py::test_query_passes_supported_filters -q` | 종료 코드 0, 필터와 페이지 결과 일치 |
| AC-2 | `pytest tests/unit/test_search_query.py::test_query_excludes_unauthorized_hits -q` | 종료 코드 0, 권한 밖 결과 0건 노출 |
| AC-3 | `pytest tests/unit/test_search_query.py::test_degraded_hits_rank_below_active_hits -q` | 종료 코드 0, active 우선 정렬과 경고 보존 |

## 건드리면 안 되는 것

- 수정 금지 파일: `src/registry/auth/**`, `src/registry/api/**`
- 변경 금지 인터페이스: `SearchQuery`, `SearchBackend`, `PLAT-L2-004` 인가 판정 계약
- 동시 작업 소유 경로: `PLAT-L5-001: src/registry/auth/**`

## 참고 규약

- `platform/tickets/CONVENTIONS.md` — 계층 경계와 가짜 구현 규칙
- `src/registry/search/contracts.py` — 질의·결과 계약

## 완료 보고

- 변경: `src/registry/search/query.py`, `tests/unit/test_search_query.py`
- 검증: `pytest tests/unit/test_search_query.py -q` → 전체 통과
- 남은 일: HTTP 검색 라우터는 L4 티켓
- 범위 밖 발견: 없음
