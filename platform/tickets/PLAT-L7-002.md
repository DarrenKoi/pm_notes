# PLAT-L7-002 Redis 작업 큐를 구현한다

## 식별

- 티켓 ID: `PLAT-L7-002`
- 소속 계층: `L7` — `platform/02-architecture.md §3`
- 관련 결정: `D4` — `platform/02-architecture.md §결정 요약`
- 선행 티켓: `[PLAT-L7-001]`

## 맥락(Context)

- 문제: API와 워커 사이의 작업 전달·확인·재시도를 수행할 Redis 어댑터가 없다.
- 필요한 이유: 색인과 후속 비동기 작업이 요청 프로세스 실패와 분리돼야 한다.
- 설계 근거:
  - `platform/02-architecture.md §3 L7 Cache / Queue`
  - `platform/02-architecture.md §4 프로세스 구성`
  - `platform/01-requirements.md NFR-02, NFR-03`
- 시작 파일:
  - `src/registry/cache/contracts.py`
  - `tests/unit/test_cache_contract.py`

## 범위(In scope)

- [ ] 주입받은 비동기 Redis 클라이언트로 `WorkQueue`를 구현한다.
- [ ] 가짜 Redis 클라이언트로 enqueue·claim·ack·nack와 재시도를 검증한다.
- 변경 허용 경로: `src/registry/cache/queue.py`, `tests/unit/test_redis_queue.py`

## 범위 밖(Out of scope)

- 업무별 payload와 워커 구현
- dead-letter 운영 화면과 Redis 배포 설정

## 수용 기준(Acceptance Criteria)

- [ ] AC-1: enqueue한 메시지를 한 소비자가 claim하면 ID·payload·시도 횟수가 보존된다.
- [ ] AC-2: ack한 메시지는 다시 claim되지 않고 nack한 메시지는 증가한 시도 횟수로 재시도된다.
- [ ] AC-3: 최대 시도 횟수에 도달한 nack 메시지는 재시도 큐가 아닌 실패 큐로 이동한다.

## 검증 방법

| 수용 기준 | 실행 명령 | 기대 결과 |
|---|---|---|
| AC-1 | `pytest tests/unit/test_redis_queue.py::test_enqueue_and_claim_preserve_message -q` | 종료 코드 0, 메시지 필드 일치 |
| AC-2 | `pytest tests/unit/test_redis_queue.py::test_ack_removes_and_nack_retries -q` | 종료 코드 0, ack 제거와 nack 재시도 |
| AC-3 | `pytest tests/unit/test_redis_queue.py::test_exhausted_message_moves_to_failed_queue -q` | 종료 코드 0, 재시도 초과 메시지 격리 |

## 건드리면 안 되는 것

- 수정 금지 파일: `src/registry/search/**`, `src/registry/api/**`, `src/registry/broker/**`
- 변경 금지 인터페이스: `WorkQueue`, `QueueMessage`
- 동시 작업 소유 경로: `PLAT-L7-003: src/registry/cache/rate_limit.py`

## 참고 규약

- `platform/tickets/CONVENTIONS.md` — Redis와 가짜 클라이언트 테스트 규칙
- `src/registry/cache/contracts.py` — 큐 계약

## 완료 보고

- 변경: `src/registry/cache/queue.py`, `tests/unit/test_redis_queue.py`
- 검증: `pytest tests/unit/test_redis_queue.py -q` → 전체 통과
- 남은 일: 업무별 큐 소비자는 각 계층 티켓
- 범위 밖 발견: 없음
