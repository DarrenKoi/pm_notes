# PLAT-L5-002 인증과 인가를 FastAPI 의존성으로 결합한다

## 식별

- 티켓 ID: `PLAT-L5-002`
- 소속 계층: `L5` — `platform/02-architecture.md §3`
- 관련 결정: `D3`(FastAPI), `D15`(튜플 판정) — `§결정 요약`
- 선행 티켓: `PLAT-L5-001`, `PLAT-L2-004`

## 맥락(Context)

- 문제: 토큰 검증(`PLAT-L5-001`)과 관계 판정(`PLAT-L2-004`)이 각각 있지만, 엔드포인트에서 둘을 쓰는 표준 경로가 없다.
- 필요한 이유: 각 라우터가 제각기 인가를 구현하면 **한 곳만 빠뜨려도 구멍이 된다.** 하나의 의존성으로 고정한다.
- 설계 근거:
  - `platform/02-architecture.md §7 인증·인가`
  - `platform/02-architecture.md §5.2` — 스코프로 1차 필터, 튜플로 2차 판정
- 시작 파일:
  - `src/registry/auth/token.py` (PLAT-L5-001)
  - `src/registry/auth/decide.py` (PLAT-L2-004)

## 범위(In scope)

- [ ] `src/registry/auth/deps.py` — `require_principal()` 과 `require_permission(relation, resource_getter)` FastAPI 의존성
- [ ] 스코프 1차 필터(`registry:read` / `registry:write`) 후 튜플 2차 판정
- [ ] 실패 시 401(인증 실패)과 403(인가 실패)을 정확히 구분
- [ ] 감사 로그 훅 — 판정 결과를 `empno` 와 함께 기록. **표시명은 기록하지 않는다**
- [ ] 테스트 (FastAPI `TestClient` 로 더미 라우터 1개를 세워 검증)
- 변경 허용 경로: `src/registry/auth/deps.py`, `tests/unit/test_auth_deps.py`

## 범위 밖(Out of scope)

- 실제 업무 엔드포인트 (L4 티켓)
- 토큰 검증 로직 변경 (`PLAT-L5-001` 소유)
- 판정 로직 변경 (`PLAT-L2-004` 소유)

## 수용 기준(Acceptance Criteria)

- [ ] AC-1: 토큰이 없으면 401, 토큰은 유효하나 권한이 없으면 403 이 반환된다
- [ ] AC-2: `registry:write` 스코프가 없는 토큰은 쓰기 엔드포인트에서 403 이다 (튜플 판정까지 가지 않는다)
- [ ] AC-3: 스코프는 있으나 해당 리소스의 관계가 없으면 403 이다
- [ ] AC-4: 스코프와 관계가 모두 있으면 200 이다
- [ ] AC-5: 감사 로그에 `empno` 와 판정 결과가 남고, **표시명·이메일은 남지 않는다**
- [ ] AC-6: 인가 판정이 예외를 던지면 **거부(403)로 처리**된다. 예외가 통과로 이어지지 않는다

## 검증 방법

| 수용 기준 | 실행 명령 | 기대 결과 |
|---|---|---|
| AC-1~AC-4 | `pytest tests/unit/test_auth_deps.py -q` | 각 상태코드 검증, 종료 코드 0 |
| AC-5 | `pytest tests/unit/test_auth_deps.py::test_audit_log_fields -q` | empno 존재·표시명 부재 검증 |
| AC-6 | `pytest tests/unit/test_auth_deps.py::test_exception_denies -q` | 403 검증 |

## 건드리면 안 되는 것

- 수정 금지 파일: `src/registry/auth/token.py`, `src/registry/auth/decide.py`, `src/registry/auth/tuples.py`
- 변경 금지 인터페이스: `Principal`, `can()`
- 동시 작업 소유 경로: 없음

## 참고 규약

- **AC-6 이 이 티켓의 핵심이다.** 판정 함수가 실패했을 때 통과시키는 fail-open 은 인가 구멍이다. 반드시 fail-closed

## 완료 보고

- 변경: <파일별 한 줄>
- 검증: <실행한 명령> → <실제 결과>
- 남은 일: 없음
- 범위 밖 발견: <없음 | 기록>
