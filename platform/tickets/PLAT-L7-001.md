# PLAT-L7-001 큐와 레이트리밋 계약을 확정한다

## 식별

- 티켓 ID: `PLAT-L7-001`
- 소속 계층: `L7` — `platform/02-architecture.md §3`
- 관련 결정: `D4` — `platform/02-architecture.md §결정 요약`
- 선행 티켓: `[PLAT-L0-001]`

## 맥락(Context)

- 문제: 비동기 작업과 요청 제한이 Redis 클라이언트 호출에 직접 결합될 수 있다.
- 필요한 이유: 외부 Redis 없이 생산자·소비자와 제한 정책을 단위 테스트할 수 있는 좁은 계약이 필요하다.
- 설계 근거:
  - `platform/02-architecture.md §3 L7 Cache / Queue`
  - `platform/02-architecture.md §4 프로세스 구성`
  - `platform/01-requirements.md NFR-02, NFR-05`
- 시작 파일:
  - `src/registry/__init__.py`
  - `tests/unit/`

## 범위(In scope)

- [ ] `WorkQueue`와 `RateLimiter` Protocol, 큐 메시지·제한 결과 모델을 정의한다.
- [ ] 확인(ack)·실패(nack)·재시도와 고정 윈도우 제한 의미를 계약 테스트로 고정한다.
- 변경 허용 경로: `src/registry/cache/contracts.py`, `tests/unit/test_cache_contract.py`

## 범위 밖(Out of scope)

- Redis 연결과 Lua 스크립트 구현
- 검색·스캔·텔레메트리 업무 메시지 정의

## 수용 기준(Acceptance Criteria)

- [ ] AC-1: 큐 계약은 메시지 ID·종류·payload·시도 횟수를 보존하고 명시적 ack 또는 nack을 요구한다.
- [ ] AC-2: nack된 메시지는 최대 시도 횟수 안에서만 재시도 가능하다고 표현한다.
- [ ] AC-3: 레이트리밋 계약은 키·한도·윈도우를 받아 허용 여부, 남은 횟수와 재시도 시각을 반환한다.

## 검증 방법

| 수용 기준 | 실행 명령 | 기대 결과 |
|---|---|---|
| AC-1 | `pytest tests/unit/test_cache_contract.py::test_queue_message_requires_explicit_completion -q` | 종료 코드 0, ack·nack 계약 검증 통과 |
| AC-2 | `pytest tests/unit/test_cache_contract.py::test_nack_respects_max_attempts -q` | 종료 코드 0, 재시도 상한 표현 검증 |
| AC-3 | `pytest tests/unit/test_cache_contract.py::test_rate_limit_result_exposes_remaining_and_retry_at -q` | 종료 코드 0, 제한 결과 필드 검증 |

## 건드리면 안 되는 것

- 수정 금지 파일: `src/registry/search/**`, `src/registry/api/**`, `src/registry/broker/**`
- 변경 금지 인터페이스: 없음
- 동시 작업 소유 경로: `PLAT-L0-001: 저장소 루트 설정 파일`

## 참고 규약

- `platform/tickets/CONVENTIONS.md` — Redis 기술 고정과 단위 테스트 규칙
- `platform/05-ticket-conventions.md §4` — 공유 인터페이스 변경 규칙

## 완료 보고

- 변경: `src/registry/cache/contracts.py`, `tests/unit/test_cache_contract.py`
- 검증: `pytest tests/unit/test_cache_contract.py -q` → 전체 통과
- 남은 일: `PLAT-L7-002`, `PLAT-L7-003`
- 범위 밖 발견: 없음
