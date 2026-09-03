# PLAT-L3-004 카탈로그에서 검색 인덱스를 재구축한다

## 식별

- 티켓 ID: `PLAT-L3-004`
- 소속 계층: `L3` — `platform/02-architecture.md §3`
- 관련 결정: `D1`, `D4` — `platform/02-architecture.md §결정 요약`
- 선행 티켓: `[PLAT-L2-002, PLAT-L3-002]`

## 맥락(Context)

- 문제: 검색 인덱스 유실·스키마 교체 시 카탈로그 진실 원천에서 안전하게 재구축할 경로가 없다.
- 필요한 이유: 파생 인덱스가 손상돼도 카탈로그 쓰기를 멈추거나 수동 문서 복사를 하지 않아야 한다.
- 설계 근거:
  - `platform/02-architecture.md §3 L3 Search Index`
  - `platform/02-architecture.md §5 데이터 모델`
  - `platform/01-requirements.md NFR-03, NFR-04`
- 시작 파일:
  - `src/registry/search/indexer.py`
  - `src/registry/catalog/repository.py`

## 범위(In scope)

- [ ] 카탈로그를 고정 크기 배치로 순회해 새 인덱스 별칭 대상에 적재하는 재색인 서비스를 구현한다.
- [ ] 중단 체크포인트 재개와 성공 시에만 별칭 교체하는 동작을 단위 테스트한다.
- 변경 허용 경로: `src/registry/search/reindex.py`, `tests/unit/test_search_reindex.py`

## 범위 밖(Out of scope)

- 운영 CLI·관리자 화면과 정기 스케줄
- OpenSearch 클러스터 용량 산정

## 수용 기준(Acceptance Criteria)

- [ ] AC-1: 카탈로그 문서를 배치 단위로 빠짐없이 새 인덱스에 적재하고 마지막 체크포인트를 기록한다.
- [ ] AC-2: 체크포인트가 있으면 다음 문서부터 재개해 이미 완료한 배치를 중복 적재하지 않는다.
- [ ] AC-3: 한 배치라도 실패하면 기존 별칭을 유지하고, 전 배치 성공 때만 새 인덱스로 별칭을 교체한다.

## 검증 방법

| 수용 기준 | 실행 명령 | 기대 결과 |
|---|---|---|
| AC-1 | `pytest tests/unit/test_search_reindex.py::test_reindex_loads_all_catalog_batches -q` | 종료 코드 0, 전 문서 적재와 체크포인트 기록 |
| AC-2 | `pytest tests/unit/test_search_reindex.py::test_reindex_resumes_after_checkpoint -q` | 종료 코드 0, 완료 배치 중복 0건 |
| AC-3 | `pytest tests/unit/test_search_reindex.py::test_alias_switches_only_after_complete_reindex -q` | 종료 코드 0, 실패 시 기존 별칭 유지 |

## 건드리면 안 되는 것

- 수정 금지 파일: `src/registry/catalog/**`, `src/registry/api/**`
- 변경 금지 인터페이스: 카탈로그 저장소와 `SearchBackend` 계약
- 동시 작업 소유 경로: `PLAT-L3-002: src/registry/search/indexer.py`

## 참고 규약

- `platform/tickets/CONVENTIONS.md` — 외부 서비스 없는 단위 테스트
- `src/registry/search/indexer.py` — 문서 투영 규칙

## 완료 보고

- 변경: `src/registry/search/reindex.py`, `tests/unit/test_search_reindex.py`
- 검증: `pytest tests/unit/test_search_reindex.py -q` → 전체 통과
- 남은 일: 운영 실행 표면은 후속 티켓
- 범위 밖 발견: 없음
