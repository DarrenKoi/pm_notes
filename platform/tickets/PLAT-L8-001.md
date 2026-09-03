# PLAT-L8-001 텔레메트리 속성과 이벤트 계약을 확정한다

## 식별

- 티켓 ID: `PLAT-L8-001`
- 소속 계층: `L8` — `platform/02-architecture.md §3`
- 관련 결정: `D10`, `D11`, `D12` — `platform/02-architecture.md §결정 요약`
- 선행 티켓: `[PLAT-L0-001, PLAT-L2-001]`

## 맥락(Context)

- 문제: 자산 유형별 계측 범위와 OTel 속성 이름을 생산자마다 다르게 정의할 수 있다.
- 필요한 이유: Broker 호출과 설치 이벤트를 같은 불변 자산·팀·사업부 축으로 집계하면서 관측 불가능한 실행을 실측처럼 기록하지 않아야 한다.
- 설계 근거:
  - `platform/02-architecture.md §10 텔레메트리`
  - `platform/02-architecture.md §5.1 agents 컬렉션`
  - `platform/01-requirements.md FR-14, NFR-08, NFR-12`
- 시작 파일:
  - `src/registry/catalog/models.py`
  - `tests/unit/`

## 범위(In scope)

- [ ] 설치·Broker 호출 이벤트 모델과 `TelemetrySink`, `UsageStore` Protocol을 정의한다.
- [ ] OTel `gen_ai.*`와 `skhynix.team_id`, `skhynix.biz_id`, `skhynix.agent_version` 속성 계약을 검사한다.
- 변경 허용 경로: `src/registry/telemetry/contracts.py`, `tests/unit/test_telemetry_contract.py`

## 범위 밖(Out of scope)

- OTel SDK·Collector·OpenSearch 구현
- 설치형 자산의 실행 이벤트 추정

## 수용 기준(Acceptance Criteria)

- [ ] AC-1: Broker 호출 이벤트는 자산 ID·버전·종류·사용자 ID·시작·종료·결과를 포함한다.
- [ ] AC-2: 모든 이벤트는 `gen_ai.agent.id`와 사내 팀·사업부·버전 속성으로 변환될 수 있다.
- [ ] AC-3: Skill과 설치형 MCP server 계약은 설치 이벤트만 허용하고 Broker 호출 이벤트 종류로 생성되지 않는다.

## 검증 방법

| 수용 기준 | 실행 명령 | 기대 결과 |
|---|---|---|
| AC-1 | `pytest tests/unit/test_telemetry_contract.py::test_broker_event_contains_required_identity_and_result -q` | 종료 코드 0, 필수 호출 필드 검증 |
| AC-2 | `pytest tests/unit/test_telemetry_contract.py::test_event_maps_to_standard_and_internal_attributes -q` | 종료 코드 0, 표준·사내 속성 일치 |
| AC-3 | `pytest tests/unit/test_telemetry_contract.py::test_installable_assets_reject_broker_execution_event -q` | 종료 코드 0, 설치형 실행 계측 거부 |

## 건드리면 안 되는 것

- 수정 금지 파일: `src/registry/catalog/**`, `src/registry/broker/**`, `src/registry/api/**`
- 변경 금지 인터페이스: 카탈로그 자산 종류와 불변 ID
- 동시 작업 소유 경로: `PLAT-L2-001: src/registry/catalog/**`

## 참고 규약

- `platform/tickets/CONVENTIONS.md` — 텔레메트리 경로와 테스트 규칙
- `platform/02-architecture.md §10` — 자산별 계측 가능 범위

## 완료 보고

- 변경: `src/registry/telemetry/contracts.py`, `tests/unit/test_telemetry_contract.py`
- 검증: `pytest tests/unit/test_telemetry_contract.py -q` → 전체 통과
- 남은 일: `PLAT-L8-002`, `PLAT-L8-003`
- 범위 밖 발견: 없음
