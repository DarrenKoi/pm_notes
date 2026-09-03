# PLAT-L7-003 Redis 레이트리밋을 원자적으로 적용한다

## 식별

- 티켓 ID: `PLAT-L7-003`
- 소속 계층: `L7` — `platform/02-architecture.md §3`
- 관련 결정: `D4` — `platform/02-architecture.md §결정 요약`
- 선행 티켓: `[PLAT-L7-001]`

## 맥락(Context)

- 문제: `INCR`와 `EXPIRE`를 별도 호출하면 실패 사이에 만료 없는 제한 키가 남을 수 있다.
- 필요한 이유: 여러 API 프로세스가 같은 제한을 경쟁해도 카운트와 TTL이 하나의 연산으로 적용돼야 한다.
- 설계 근거:
  - `platform/02-architecture.md §3 L7 Cache / Queue`
  - `platform/01-requirements.md NFR-02, NFR-05`
  - `platform/03-operations.md §인시던트 대응`
- 시작 파일:
  - `src/registry/cache/contracts.py`
  - `tests/unit/test_cache_contract.py`

## 범위(In scope)

- [ ] Redis `INCR`+`EXPIRE` Lua 스크립트를 사용하는 `RateLimiter` 구현을 추가한다.
- [ ] 가짜 script executor로 최초 TTL, 한도 경계와 재시도 시각을 검증한다.
- 변경 허용 경로: `src/registry/cache/rate_limit.py`, `tests/unit/test_rate_limit.py`

## 범위 밖(Out of scope)

- 엔드포인트별 제한 수치 결정과 FastAPI 미들웨어
- 분산 Redis 구성과 운영 대시보드

## 수용 기준(Acceptance Criteria)

- [ ] AC-1: 한 요청은 단일 Lua 실행으로 카운트를 증가시키고 최초 증가 때만 윈도우 TTL을 설정한다.
- [ ] AC-2: 카운트가 한도 이하이면 허용하고 한도를 초과하면 거부하며 남은 횟수를 음수로 반환하지 않는다.
- [ ] AC-3: 거부 결과의 `retry_at`은 Redis가 반환한 잔여 TTL을 기준으로 계산된다.

## 검증 방법

| 수용 기준 | 실행 명령 | 기대 결과 |
|---|---|---|
| AC-1 | `pytest tests/unit/test_rate_limit.py::test_increment_and_expiry_are_one_script_call -q` | 종료 코드 0, Lua 호출 1회와 최초 TTL 설정 |
| AC-2 | `pytest tests/unit/test_rate_limit.py::test_limit_boundary_allows_then_rejects -q` | 종료 코드 0, 경계 허용·초과 거부 |
| AC-3 | `pytest tests/unit/test_rate_limit.py::test_retry_at_uses_redis_ttl -q` | 종료 코드 0, 재시도 시각 계산 일치 |

## 건드리면 안 되는 것

- 수정 금지 파일: `src/registry/api/**`, `src/registry/broker/**`, `src/registry/auth/**`
- 변경 금지 인터페이스: `RateLimiter`, `RateLimitResult`
- 동시 작업 소유 경로: `PLAT-L7-002: src/registry/cache/queue.py`

## 참고 규약

- `platform/tickets/CONVENTIONS.md` — 기술 고정과 외부 서비스 없는 테스트
- `src/registry/cache/contracts.py` — 제한 계약

## 완료 보고

- 변경: `src/registry/cache/rate_limit.py`, `tests/unit/test_rate_limit.py`
- 검증: `pytest tests/unit/test_rate_limit.py -q` → 전체 통과
- 남은 일: 제한 정책 연결은 소비 계층 티켓
- 범위 밖 발견: 없음
