# PLAT-L5-001 OIDC 액세스 토큰 검증을 구현한다

## 식별

- 티켓 ID: `PLAT-L5-001`
- 소속 계층: `L5` (Auth) — `platform/02-architecture.md §3`
- 관련 결정: `D12`(사용자 정보는 마스터에서 조회, ID 만 저장) — `§결정 요약`
- 선행 티켓: `PLAT-L0-001`

## 맥락(Context)

- 문제: 요청자가 누구인지 확인할 수단이 없다.
- 필요한 이유: 1차 플랫폼은 **사업부마다 별도 인스턴스**다. 타 인스턴스용으로 발급된 토큰이 우리 인스턴스에서 통과하면 인스턴스 경계가 무너진다. `aud` 검증이 그 경계다.
- 설계 근거:
  - `platform/02-architecture.md §7.1 원칙` — `aud` 가 자기 자신이 아니면 거부, 토큰 패스스루 금지
  - `platform/02-architecture.md §7.2 신원` — subject identifier 는 public, 저장하는 것은 `empno` 뿐
  - `platform/research/agent-skill-registry-research.md` — RFC 9068 `at+jwt` 필수 클레임
- 시작 파일:
  - `src/registry/auth/__init__.py`

## 범위(In scope)

- [ ] `src/registry/auth/token.py` — JWT 검증: 서명, `iss`, `aud`, `exp`, `nbf`
- [ ] JWKS 조회를 Protocol 로 추상화 + 가짜 구현 (단위 테스트가 네트워크를 타지 않도록)
- [ ] 검증 결과를 `Principal(empno, scopes, client_id)` 로 반환
- [ ] 테스트
- 변경 허용 경로: `src/registry/auth/token.py`, `tests/unit/test_token_verify.py`

## 범위 밖(Out of scope)

- 인가 판정 (`PLAT-L2-004` 의 `can()`)
- FastAPI 의존성·미들웨어 결합 (`PLAT-L5-002`)
- RFC 8693 token exchange — IdP 지원 여부가 미확인(Q7c). **구현하지 않는다**
- 사용자 이름·소속 조회 — `PLAT-L2-006` 의 `lookup_now`

## 수용 기준(Acceptance Criteria)

- [ ] AC-1: 유효한 토큰이 `Principal` 로 파싱되고 `empno` 가 `sub` 에서 추출된다
- [ ] AC-2: **`aud` 가 이 인스턴스가 아니면 거부**된다 (다른 Biz 인스턴스용 토큰)
- [ ] AC-3: `iss` 가 신뢰 목록에 없으면 거부된다
- [ ] AC-4: 만료된 토큰(`exp` 과거)이 거부된다
- [ ] AC-5: 서명이 맞지 않으면 거부된다
- [ ] AC-6: RFC 9068 필수 클레임(`iss`, `exp`, `aud`, `sub`, `client_id`, `iat`, `jti`) 중 하나라도 없으면 거부된다
- [ ] AC-7: 검증 실패 시 **토큰 원문이 로그에 남지 않는다**

## 검증 방법

| 수용 기준 | 실행 명령 | 기대 결과 |
|---|---|---|
| AC-1 | `pytest tests/unit/test_token_verify.py::test_valid_token -q` | 통과 |
| AC-2 | `pytest tests/unit/test_token_verify.py::test_wrong_audience_rejected -q` | 거부 검증 후 통과 |
| AC-3~AC-6 | `pytest tests/unit/test_token_verify.py -q` | 각 거부 케이스 검증, 종료 코드 0 |
| AC-7 | `pytest tests/unit/test_token_verify.py::test_no_token_in_logs -q` | caplog 에 토큰 문자열 부재 검증 |

## 건드리면 안 되는 것

- 수정 금지 파일: `src/registry/auth/tuples.py`, `src/registry/auth/decide.py`
- 변경 금지 인터페이스: `can()` 시그니처
- 동시 작업 소유 경로: `PLAT-L2-004`: 인가 판정은 그 티켓 소유

## 참고 규약

- `platform/02-architecture.md §7.1` — **토큰 패스스루는 금지**(MCP 스펙상 MUST NOT). 받은 토큰을 상류로 그대로 전달하는 코드를 만들지 않는다

## 완료 보고

- 변경: <파일별 한 줄>
- 검증: <실행한 명령> → <실제 결과>
- 남은 일: 없음
- 범위 밖 발견: <없음 | 기록>
