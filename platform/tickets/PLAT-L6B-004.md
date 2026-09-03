# PLAT-L6B-004 만료되는 Broker 접근 토큰을 발급한다

## 식별

- 티켓 ID: `PLAT-L6B-004`
- 소속 계층: `L6b` — `platform/02-architecture.md §3`
- 관련 결정: `D7`, `D11`, `D15` — `platform/02-architecture.md §결정 요약`
- 선행 티켓: `[PLAT-L2-004, PLAT-L5-001, PLAT-L6B-001, PLAT-L7-003]`

## 맥락(Context)

- 문제: 승인된 관계 튜플과 endpoint 상태를 짧게 유효한 Broker 접근 토큰으로 바꾸고 만료·회수할 서비스가 없다.
- 필요한 이유: 원격 endpoint의 장기 자격증명을 사용자에게 노출하지 않고 신규 접근 차단을 즉시 적용해야 한다.
- 설계 근거:
  - `platform/02-architecture.md §3 L6b Endpoint Broker`
  - `platform/02-architecture.md §5.2 소유권`
  - `platform/01-requirements.md FR-29, FR-35, FR-37`
- 시작 파일:
  - `src/registry/broker/contracts.py`
  - `src/registry/auth/decide.py`

## 범위(In scope)

- [ ] OIDC subject·관계 튜플·endpoint 상태를 확인해 불투명 단기 접근 토큰을 발급하고 만료·회수한다.
- [ ] 가짜 인가 판정·토큰 저장소·시계·레이트리미터로 허용과 거부 경계를 검증한다.
- 변경 허용 경로: `src/registry/broker/tokens.py`, `tests/unit/test_broker_tokens.py`

## 범위 밖(Out of scope)

- RFC 8693 token exchange와 agent→사내 API 토큰 모델
- endpoint 자체 자격증명 보관과 FastAPI 라우터

## 수용 기준(Acceptance Criteria)

- [ ] AC-1: 유효한 OIDC subject와 허용 튜플, `active` endpoint에는 만료 시각이 있는 불투명 토큰을 발급한다.
- [ ] AC-2: 미인가·만료 관계, `degraded` endpoint 또는 레이트리밋 초과 요청에는 토큰을 발급하지 않는다.
- [ ] AC-3: 만료되거나 명시적으로 회수된 토큰은 검증에 실패하고 원본 endpoint 자격증명은 반환값에 포함되지 않는다.

## 검증 방법

| 수용 기준 | 실행 명령 | 기대 결과 |
|---|---|---|
| AC-1 | `pytest tests/unit/test_broker_tokens.py::test_authorized_active_endpoint_issues_expiring_token -q` | 종료 코드 0, 불투명 토큰과 만료 시각 생성 |
| AC-2 | `pytest tests/unit/test_broker_tokens.py::test_unauthorized_degraded_or_limited_request_is_denied -q` | 종료 코드 0, 모든 거부 조건에서 발급 0건 |
| AC-3 | `pytest tests/unit/test_broker_tokens.py::test_expired_or_revoked_token_fails_without_endpoint_secret -q` | 종료 코드 0, 토큰 거부와 비밀정보 미노출 |

## 건드리면 안 되는 것

- 수정 금지 파일: `src/registry/auth/**`, `src/registry/catalog/**`, `src/registry/cache/**`
- 변경 금지 인터페이스: OIDC 검증, 관계 튜플, `RateLimiter`와 Broker token 모델
- 동시 작업 소유 경로: `PLAT-L5-001: src/registry/auth/**, PLAT-L7-003: src/registry/cache/rate_limit.py`

## 참고 규약

- `platform/tickets/CONVENTIONS.md` — 보안 경계와 단위 테스트
- `src/registry/broker/contracts.py` — 접근 grant·token 계약

## 완료 보고

- 변경: `src/registry/broker/tokens.py`, `tests/unit/test_broker_tokens.py`
- 검증: `pytest tests/unit/test_broker_tokens.py -q` → 전체 통과
- 남은 일: `PLAT-L6B-005`
- 범위 밖 발견: 없음
