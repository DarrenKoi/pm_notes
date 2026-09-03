# PLAT-L2-005 teams 컬렉션 스키마와 저장소를 만든다

## 식별

- 티켓 ID: `PLAT-L2-005`
- 소속 계층: `L2` — `platform/02-architecture.md §3`
- 관련 결정: `D13`(팀은 empno 집합을 갖는 그룹) — `§결정 요약`
- 선행 티켓: `PLAT-L2-002`

## 맥락(Context)

- 문제: 팀을 담을 곳이 없다. 인가 판정(`PLAT-L2-004`)이 팀 멤버십을 필요로 한다.
- 필요한 이유: 팀명이 매년 바뀌므로 **불변 `team_id` 와 가변 표시명을 분리**해야 한다. 외부 조직 코드는 불변성이 확인되지 않았으므로 **식별자로 쓰지 않는다**.
- 설계 근거:
  - `platform/02-architecture.md §5.3 조직` — 스키마 원문과 케이스 표
  - `platform/02-architecture.md §6 식별자 체계`
- 시작 파일:
  - `src/registry/catalog/repository.py` (PLAT-L2-002)

## 범위(In scope)

- [ ] `src/registry/orgsync/models.py` — `Team` 모델 (`teamId`, `displayName`, `bizId`, `members`, `observedOrgCode`, `observedOrgName`, `history`, `status`, `successorTeamId`)
- [ ] `src/registry/orgsync/repository.py` — `TeamRepository` Protocol + 가짜 + Mongo 구현
- [ ] `team_id` 발급 함수 (레지스트리 자체 발급, 충돌 없음 보장)
- [ ] `§5` 의 인덱스 생성
- [ ] 테스트
- 변경 허용 경로: `src/registry/orgsync/models.py`, `src/registry/orgsync/repository.py`, `tests/unit/test_team_repository.py`

## 범위 밖(Out of scope)

- 사원 마스터 조회 API 연동 (`PLAT-L2-006`)
- 조직 개편 감지 (`PLAT-L2-007`)
- 멤버십을 사람이 편집하는 UI — **D14 로 만들지 않기로 결정됨**

## 수용 기준(Acceptance Criteria)

- [ ] AC-1: `teamId` 가 unique 하고, 발급 함수를 1000회 호출해 중복이 없다
- [ ] AC-2: `displayName` 변경 시 이전 값이 `history` 에 `from`/`to` 와 함께 남는다
- [ ] AC-3: `displayName` 은 unique 가 **아니다** (같은 이름의 팀이 동시에 존재할 수 있다)
- [ ] AC-4: `observedOrgCode` 로는 팀을 조회할 수 **없다** — 조회 API 를 제공하지 않는다 (식별자가 아님을 코드로 강제)
- [ ] AC-5: `status: "merged"` 인데 `successorTeamId` 가 없으면 저장이 거부된다
- [ ] AC-6: `members` 는 `empno` 문자열 집합이며 중복이 저장되지 않는다

## 검증 방법

| 수용 기준 | 실행 명령 | 기대 결과 |
|---|---|---|
| AC-1 | `pytest tests/unit/test_team_repository.py::test_team_id_unique -q` | 통과 |
| AC-2, AC-3 | `pytest tests/unit/test_team_repository.py::test_display_name -q` | 통과 |
| AC-4 | `pytest tests/unit/test_team_repository.py::test_no_lookup_by_org_code -q` | 해당 메서드 부재 검증 |
| AC-5, AC-6 | `pytest tests/unit/test_team_repository.py -q` | 종료 코드 0 |

## 건드리면 안 되는 것

- 수정 금지 파일: `src/registry/auth/`, `src/registry/catalog/`
- 변경 금지 인터페이스: `CatalogRepository`
- 동시 작업 소유 경로: 없음

## 참고 규약

- `src/registry/catalog/repository.py` — Protocol + 가짜 + Mongo 구현 3단 구성을 동일하게 따른다

## 완료 보고

- 변경: <파일별 한 줄>
- 검증: <실행한 명령> → <실제 결과>
- 남은 일: 없음
- 범위 밖 발견: <없음 | 기록>
