# PLAT-L2-002 카탈로그 저장소 인터페이스와 MongoDB 구현을 만든다

## 식별

- 티켓 ID: `PLAT-L2-002`
- 소속 계층: `L2` — `platform/02-architecture.md §3`
- 관련 결정: `D1` — `§결정 요약`
- 선행 티켓: `PLAT-L2-001`

## 맥락(Context)

- 문제: 검증된 카탈로그 문서를 저장·조회할 곳이 없다.
- 필요한 이유: API·검색·승격이 모두 이 저장소를 통해 카탈로그에 접근한다. MongoDB 클라이언트를 직접 쓰면 단위 테스트가 DB 를 요구하게 된다.
- 설계 근거:
  - `platform/02-architecture.md §5.1`, `§5 데이터 모델 인덱스 표`
  - `platform/02-architecture.md §3 L2` — MongoDB `$jsonSchema` validator, `validationAction: error`
- 시작 파일:
  - `src/registry/catalog/models.py` (PLAT-L2-001)

## 범위(In scope)

- [ ] `src/registry/catalog/repository.py` — `CatalogRepository` Protocol: `insert`, `get`, `list_versions`, `find_updated_since`
- [ ] `InMemoryCatalogRepository` — 테스트용
- [ ] `MongoCatalogRepository` — motor 기반
- [ ] 컬렉션 부트스트랩 — `$jsonSchema` validator 적용, `validationAction: error`, `§5` 의 인덱스 생성
- [ ] 공용 계약 테스트 (두 구현 모두에 적용)
- 변경 허용 경로: `src/registry/catalog/repository.py`, `tests/unit/test_catalog_repository.py`, `tests/integration/test_catalog_mongo.py`

## 범위 밖(Out of scope)

- 상태 전이 규칙 (`PLAT-L2-003`)
- 소유권 판정 (`PLAT-L2-004`)
- HTTP 라우터 (L4)
- 스키마 변경 — `PLAT-L2-001` 소유

## 수용 기준(Acceptance Criteria)

- [ ] AC-1: `(name, version)` 이 중복이면 `insert` 가 거부된다 (unique 인덱스)
- [ ] AC-2: `list_versions(name)` 이 버전 목록을 반환하고, `isLatest: true` 인 문서가 **정확히 1개**임을 보장한다
- [ ] AC-3: `find_updated_since(ts)` 가 해당 시각 이후 갱신분만 반환하고, **삭제 상태(`deleted`) 문서도 포함**한다
- [ ] AC-4: 스키마에 맞지 않는 문서는 저장소 진입 전에 거부된다
- [ ] AC-5: 공용 계약 테스트가 `InMemoryCatalogRepository` 에서 전부 통과한다
- [ ] AC-6: MongoDB 없이 `pytest tests/unit` 이 통과한다

## 검증 방법

| 수용 기준 | 실행 명령 | 기대 결과 |
|---|---|---|
| AC-1~AC-5 | `pytest tests/unit/test_catalog_repository.py -q` | 종료 코드 0 |
| AC-3 | `pytest tests/unit/test_catalog_repository.py::test_updated_since_includes_deleted -q` | 통과 |
| Mongo 구현 | `docker compose up -d mongodb && pytest tests/integration/test_catalog_mongo.py -m integration -q` | 종료 코드 0 |
| AC-6 | `pytest tests/unit -q` | 종료 코드 0 (Mongo 미기동) |

## 건드리면 안 되는 것

- 수정 금지 파일: `schemas/`, `src/registry/catalog/models.py`
- 변경 금지 인터페이스: `AgentDoc` 모델
- 동시 작업 소유 경로: 없음

## 참고 규약

- `src/registry/store/base.py` — Protocol + 가짜 구현 + 공용 계약 테스트 패턴을 동일하게 따른다

## 완료 보고

- 변경: <파일별 한 줄>
- 검증: <실행한 명령> → <실제 결과>
- 남은 일: 없음
- 범위 밖 발견: <없음 | 기록>
