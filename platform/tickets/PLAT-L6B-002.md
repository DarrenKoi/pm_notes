# PLAT-L6B-002 endpoint 헬스체크를 5분마다 수행한다

## 식별

- 티켓 ID: `PLAT-L6B-002`
- 소속 계층: `L6b` — `platform/02-architecture.md §3`
- 관련 결정: `D11` — `platform/02-architecture.md §결정 요약`
- 선행 티켓: `[PLAT-L2-002, PLAT-L6B-001, PLAT-L7-002]`

## 맥락(Context)

- 문제: 등록된 endpoint를 정해진 간격으로 확인하고 결과를 후속 상태 판정에 전달하는 스케줄러가 없다.
- 필요한 이유: 제공 팀 서버 장애를 사용자 신고 전에 감지하면서 네트워크 호출을 상태 정책과 분리해야 한다.
- 설계 근거:
  - `platform/02-architecture.md §8 발행 파이프라인`
  - `platform/02-architecture.md §4 registry-scheduler`
  - `platform/03-operations.md §Hosted agent 헬스체크와 degraded`
- 시작 파일:
  - `src/registry/broker/contracts.py`
  - `src/registry/cache/queue.py`

## 범위(In scope)

- [ ] 활성 endpoint를 5분 간격 대상으로 선택하고 주입된 health client로 확인 결과를 큐에 발행한다.
- [ ] 가짜 시계·카탈로그·health client·큐로 예정 시각, timeout과 중복 실행 방지를 검증한다.
- 변경 허용 경로: `src/registry/broker/health.py`, `tests/unit/test_broker_health.py`

## 범위 밖(Out of scope)

- `degraded` 상태 전이와 알림·검색 순위 변경
- 실제 HTTP 클라이언트 설정과 scheduler 프로세스 배포

## 수용 기준(Acceptance Criteria)

- [ ] AC-1: 마지막 확인 후 5분이 지난 활성 endpoint만 한 번 검사하고 결과 시각·성공 여부를 큐에 발행한다.
- [ ] AC-2: timeout과 비정상 응답은 실패 관측으로 발행하되 스케줄러 반복 실행을 중단하지 않는다.
- [ ] AC-3: 아직 5분이 지나지 않았거나 같은 실행에서 이미 claim된 endpoint는 다시 검사하지 않는다.

## 검증 방법

| 수용 기준 | 실행 명령 | 기대 결과 |
|---|---|---|
| AC-1 | `pytest tests/unit/test_broker_health.py::test_due_endpoint_is_checked_every_five_minutes -q` | 종료 코드 0, 대상 1건 검사·발행 |
| AC-2 | `pytest tests/unit/test_broker_health.py::test_timeout_emits_failure_and_continues -q` | 종료 코드 0, 실패 관측 후 다음 대상 처리 |
| AC-3 | `pytest tests/unit/test_broker_health.py::test_not_due_or_claimed_endpoint_is_skipped -q` | 종료 코드 0, 중복 검사 0건 |

## 건드리면 안 되는 것

- 수정 금지 파일: `src/registry/catalog/**`, `src/registry/cache/**`, `src/registry/search/**`
- 변경 금지 인터페이스: endpoint 계약과 `WorkQueue`
- 동시 작업 소유 경로: `PLAT-L6B-003: src/registry/broker/degraded.py`

## 참고 규약

- `platform/tickets/CONVENTIONS.md` — 외부 Redis 없는 단위 테스트
- `src/registry/broker/contracts.py` — endpoint·health 관측 계약

## 완료 보고

- 변경: `src/registry/broker/health.py`, `tests/unit/test_broker_health.py`
- 검증: `pytest tests/unit/test_broker_health.py -q` → 전체 통과
- 남은 일: `PLAT-L6B-003`
- 범위 밖 발견: 없음
