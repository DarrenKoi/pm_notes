# PLAT-L6B-005 원격 호출을 중개하고 계량한다

## 식별

- 티켓 ID: `PLAT-L6B-005`
- 소속 계층: `L6b` — `platform/02-architecture.md §3`
- 관련 결정: `D10`, `D11`, `D12` — `platform/02-architecture.md §결정 요약`
- 선행 티켓: `[PLAT-L6B-004, PLAT-L8-002]`

## 맥락(Context)

- 문제: 유효한 접근 토큰으로 hosted agent·원격 MCP server를 호출하고 성공·오류·지연을 빠짐없이 계량하는 경로가 없다.
- 필요한 이유: Broker를 우회하지 않는 원격 호출만 실측으로 인정하고 실패 호출도 승격 증적에서 누락하지 않아야 한다.
- 설계 근거:
  - `platform/02-architecture.md §3 L6b Endpoint Broker`
  - `platform/02-architecture.md §10 텔레메트리`
  - `platform/01-requirements.md FR-13, FR-14, NFR-08`
- 시작 파일:
  - `src/registry/broker/tokens.py`
  - `src/registry/telemetry/otel.py`

## 범위(In scope)

- [ ] 유효한 Broker 토큰을 검증하고 주입된 endpoint client로 호출을 중개한다.
- [ ] 성공·오류·timeout마다 하나의 계량 이벤트를 `TelemetrySink`에 발행하고 민감 값을 제외한다.
- 변경 허용 경로: `src/registry/broker/metering.py`, `tests/unit/test_broker_metering.py`

## 범위 밖(Out of scope)

- endpoint 프로토콜별 실제 HTTP transport와 재시도 정책
- 설치형 자산 실행 계측과 사용량 집계 질의

## 수용 기준(Acceptance Criteria)

- [ ] AC-1: 유효한 토큰의 hosted agent와 원격 MCP 요청을 endpoint client에 전달하고 응답 상태를 반환한다.
- [ ] AC-2: 성공·endpoint 오류·timeout 각각에 대해 사용자·자산·버전·지연·결과가 있는 이벤트를 정확히 한 건 발행한다.
- [ ] AC-3: 무효 토큰과 `degraded` endpoint는 네트워크 호출 전에 거부하며 토큰·요청·응답 본문을 텔레메트리에 넣지 않는다.

## 검증 방법

| 수용 기준 | 실행 명령 | 기대 결과 |
|---|---|---|
| AC-1 | `pytest tests/unit/test_broker_metering.py::test_valid_token_proxies_remote_request -q` | 종료 코드 0, endpoint 호출과 응답 전달 |
| AC-2 | `pytest tests/unit/test_broker_metering.py::test_every_outcome_emits_exactly_one_metering_event -q` | 종료 코드 0, 결과별 이벤트 각 1건 |
| AC-3 | `pytest tests/unit/test_broker_metering.py::test_invalid_or_degraded_request_stops_before_network_without_secrets -q` | 종료 코드 0, 네트워크 호출 0건과 민감 값 미노출 |

## 건드리면 안 되는 것

- 수정 금지 파일: `src/registry/telemetry/**`, `src/registry/auth/**`, `src/registry/catalog/**`
- 변경 금지 인터페이스: Broker token, endpoint client와 `TelemetrySink`
- 동시 작업 소유 경로: `PLAT-L8-002: src/registry/telemetry/otel.py`

## 참고 규약

- `platform/tickets/CONVENTIONS.md` — 보안 경계와 가짜 client 테스트
- `src/registry/telemetry/contracts.py` — 호출 이벤트 계약

## 완료 보고

- 변경: `src/registry/broker/metering.py`, `tests/unit/test_broker_metering.py`
- 검증: `pytest tests/unit/test_broker_metering.py -q` → 전체 통과
- 남은 일: `PLAT-L8-003`
- 범위 밖 발견: 없음
