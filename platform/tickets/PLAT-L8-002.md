# PLAT-L8-002 Broker 계량 이벤트를 OTel로 발행한다

## 식별

- 티켓 ID: `PLAT-L8-002`
- 소속 계층: `L8` — `platform/02-architecture.md §3`
- 관련 결정: `D11`, `D12` — `platform/02-architecture.md §결정 요약`
- 선행 티켓: `[PLAT-L6B-001, PLAT-L8-001]`

## 맥락(Context)

- 문제: Broker가 생성할 호출 결과를 표준 OTel 속성과 span/metric으로 발행하는 구현이 없다.
- 필요한 이유: hosted agent와 원격 MCP server의 호출 수·지연·오류·사용자를 동일 형식으로 관측해야 한다.
- 설계 근거:
  - `platform/02-architecture.md §10 텔레메트리`
  - `platform/02-architecture.md §3 L8 Observability`
  - `platform/01-requirements.md FR-14, FR-22, FR-23`
- 시작 파일:
  - `src/registry/telemetry/contracts.py`
  - `src/registry/broker/contracts.py`

## 범위(In scope)

- [ ] 주입받은 OTel tracer·meter 포트로 Broker 호출 이벤트를 span과 카운터·지연 값으로 발행한다.
- [ ] 가짜 tracer·meter로 성공·오류 속성과 민감 입력 미기록을 검증한다.
- 변경 허용 경로: `src/registry/telemetry/otel.py`, `tests/unit/test_otel_publisher.py`

## 범위 밖(Out of scope)

- Collector·Data Prepper·OpenSearch 배포 설정
- Broker 요청 프록시와 설치형 실행 계측

## 수용 기준(Acceptance Criteria)

- [ ] AC-1: 성공 호출은 `gen_ai.agent.id`, 팀·사업부·버전, 지연과 성공 결과를 span·metric에 기록한다.
- [ ] AC-2: 실패 호출은 같은 식별자와 오류 유형을 기록하며 호출 카운터에서 누락되지 않는다.
- [ ] AC-3: 프롬프트·응답 본문·접근 토큰은 속성 또는 오류 메시지에 기록되지 않는다.

## 검증 방법

| 수용 기준 | 실행 명령 | 기대 결과 |
|---|---|---|
| AC-1 | `pytest tests/unit/test_otel_publisher.py::test_success_event_emits_span_and_metrics -q` | 종료 코드 0, 성공 span·metric 속성 일치 |
| AC-2 | `pytest tests/unit/test_otel_publisher.py::test_error_event_is_counted_with_error_type -q` | 종료 코드 0, 오류 호출 계량 1건 |
| AC-3 | `pytest tests/unit/test_otel_publisher.py::test_sensitive_payload_and_token_are_not_emitted -q` | 종료 코드 0, 민감 값 노출 0건 |

## 건드리면 안 되는 것

- 수정 금지 파일: `src/registry/broker/**`, `src/registry/catalog/**`, 배포 설정 파일
- 변경 금지 인터페이스: `TelemetrySink`, Broker 호출 이벤트 계약
- 동시 작업 소유 경로: `PLAT-L6B-005: src/registry/broker/metering.py`

## 참고 규약

- `platform/tickets/CONVENTIONS.md` — 외부 서비스 없는 단위 테스트
- `src/registry/telemetry/contracts.py` — 이벤트·속성 계약

## 완료 보고

- 변경: `src/registry/telemetry/otel.py`, `tests/unit/test_otel_publisher.py`
- 검증: `pytest tests/unit/test_otel_publisher.py -q` → 전체 통과
- 남은 일: `PLAT-L6B-005`, `PLAT-L8-003`
- 범위 밖 발견: 없음
