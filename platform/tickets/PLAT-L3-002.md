# PLAT-L3-002 카탈로그 변경을 검색 인덱스에 반영한다

## 식별

- 티켓 ID: `PLAT-L3-002`
- 소속 계층: `L3` — `platform/02-architecture.md §3`
- 관련 결정: `D1`, `D4` — `platform/02-architecture.md §결정 요약`
- 선행 티켓: `[PLAT-L2-002, PLAT-L3-001, PLAT-L7-002]`

## 맥락(Context)

- 문제: 카탈로그 변경 이벤트를 소비해 OpenSearch 파생 인덱스를 갱신하는 생산자가 없다.
- 필요한 이유: 카탈로그 쓰기 경로와 검색 장애를 분리하고 재시도 가능한 비동기 색인을 제공해야 한다.
- 설계 근거:
  - `platform/02-architecture.md §3 L3 Search Index`
  - `platform/02-architecture.md §8 발행 파이프라인`
  - `platform/01-requirements.md FR-08, NFR-01`
- 시작 파일:
  - `src/registry/search/contracts.py`
  - `src/registry/cache/queue.py`

## 범위(In scope)

- [ ] 큐 이벤트를 카탈로그에서 다시 읽어 upsert 또는 delete하는 색인 워커를 구현한다.
- [ ] 가짜 카탈로그·큐·검색 백엔드로 중복 이벤트와 실패 재시도를 검증한다.
- 변경 허용 경로: `src/registry/search/indexer.py`, `tests/unit/test_search_indexer.py`

## 범위 밖(Out of scope)

- 검색 질의와 전체 재색인
- OpenSearch 인덱스 운영 설정과 통합 테스트

## 수용 기준(Acceptance Criteria)

- [ ] AC-1: upsert 이벤트를 받으면 최신 카탈로그 문서를 한 번 읽어 동일 ID·버전의 검색 문서를 저장한다.
- [ ] AC-2: delete 이벤트를 받으면 해당 ID·버전을 검색 백엔드에서 제거한다.
- [ ] AC-3: 검색 백엔드 실패 시 이벤트를 완료 처리하지 않고 재시도 가능 상태로 남기며, 같은 이벤트 재처리는 문서를 중복 생성하지 않는다.

## 검증 방법

| 수용 기준 | 실행 명령 | 기대 결과 |
|---|---|---|
| AC-1 | `pytest tests/unit/test_search_indexer.py::test_upsert_event_indexes_latest_catalog_document -q` | 종료 코드 0, 최신 문서 1건 upsert |
| AC-2 | `pytest tests/unit/test_search_indexer.py::test_delete_event_removes_index_document -q` | 종료 코드 0, 대상 문서 삭제 호출 1회 |
| AC-3 | `pytest tests/unit/test_search_indexer.py::test_failed_event_is_retryable_and_idempotent -q` | 종료 코드 0, 실패 미확인과 중복 방지 검증 통과 |

## 건드리면 안 되는 것

- 수정 금지 파일: `src/registry/catalog/**`, `src/registry/cache/**`
- 변경 금지 인터페이스: `SearchBackend`, `WorkQueue`, 카탈로그 저장소 인터페이스
- 동시 작업 소유 경로: `PLAT-L7-002: src/registry/cache/queue.py`

## 참고 규약

- `platform/tickets/CONVENTIONS.md` — 외부 서비스 없는 단위 테스트
- `src/registry/search/contracts.py` — 검색 계약

## 완료 보고

- 변경: `src/registry/search/indexer.py`, `tests/unit/test_search_indexer.py`
- 검증: `pytest tests/unit/test_search_indexer.py -q` → 전체 통과
- 남은 일: `PLAT-L3-004`
- 범위 밖 발견: 없음
