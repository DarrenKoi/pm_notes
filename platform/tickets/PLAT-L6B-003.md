# PLAT-L6B-003 연속 장애를 degraded로 자동 전환한다

## 식별

- 티켓 ID: `PLAT-L6B-003`
- 소속 계층: `L6b` — `platform/02-architecture.md §3`
- 관련 결정: `D11` — `platform/02-architecture.md §결정 요약`
- 선행 티켓: `[PLAT-L2-003, PLAT-L3-002, PLAT-L3-003, PLAT-L6B-002, PLAT-L7-002]`

## 맥락(Context)

- 문제: 헬스체크 관측을 6회 연속 실패 정책과 복구 동작으로 연결하는 상태 조정기가 없다.
- 필요한 이유: 되돌릴 수 있는 보호 조치는 자동화하되 심사 결과인 tier와 폐기 상태는 사람이 결정해야 한다.
- 설계 근거:
  - `platform/02-architecture.md §8 헬스체크 실패 시`
  - `platform/01-requirements.md FR-37`
  - `platform/03-operations.md §Hosted agent 헬스체크와 degraded`
- 시작 파일:
  - `src/registry/broker/health.py`
  - `src/registry/catalog/transitions.py`
  - `src/registry/search/indexer.py`

## 범위(In scope)

- [ ] health 관측을 소비해 연속 실패 수와 `active`↔`degraded` 상태를 조정한다.
- [ ] 접근 차단·소유 팀 통보·검색 재색인 포트를 가짜 구현으로 검증한다.
- 변경 허용 경로: `src/registry/broker/degraded.py`, `tests/unit/test_degraded_transition.py`

## 범위 밖(Out of scope)

- tier 강등, Deprecated·Recalled 전환과 인시던트 승인
- 실제 알림 채널과 검색 랭킹 공식 변경

## 수용 기준(Acceptance Criteria)

- [ ] AC-1: 5회 연속 실패까지 `active`를 유지하고 6회째 실패에서 `degraded`로 전환한다.
- [ ] AC-2: 전환 시 신규 접근·신규 연결을 차단하고 소유 팀 통보와 검색 재색인을 각각 한 번 요청하며 tier는 변경하지 않는다.
- [ ] AC-3: 성공 관측을 받으면 실패 수를 0으로 초기화하고 `degraded`를 `active`로 복구해 신규 접근·연결 차단을 해제한다.

## 검증 방법

| 수용 기준 | 실행 명령 | 기대 결과 |
|---|---|---|
| AC-1 | `pytest tests/unit/test_degraded_transition.py::test_sixth_consecutive_failure_sets_degraded -q` | 종료 코드 0, 5회 active·6회 degraded |
| AC-2 | `pytest tests/unit/test_degraded_transition.py::test_degraded_blocks_new_access_notifies_and_preserves_tier -q` | 종료 코드 0, 차단·통보·재색인 1회와 tier 불변 |
| AC-3 | `pytest tests/unit/test_degraded_transition.py::test_success_recovers_active_and_clears_failure_count -q` | 종료 코드 0, active 복구와 차단 해제 |

## 건드리면 안 되는 것

- 수정 금지 파일: `src/registry/catalog/**`, `src/registry/search/**`, `src/registry/auth/**`
- 변경 금지 인터페이스: 카탈로그 상태 저장소, 검색 색인 이벤트와 인가 판정 계약
- 동시 작업 소유 경로: `PLAT-L6B-002: src/registry/broker/health.py`

## 참고 규약

- `platform/tickets/CONVENTIONS.md` — 가짜 포트 단위 테스트
- `src/registry/broker/contracts.py` — 상태·health 관측 계약

## 완료 보고

- 변경: `src/registry/broker/degraded.py`, `tests/unit/test_degraded_transition.py`
- 검증: `pytest tests/unit/test_degraded_transition.py -q` → 전체 통과
- 남은 일: 사람 강등·폐기는 L10 운영 티켓
- 범위 밖 발견: 없음
