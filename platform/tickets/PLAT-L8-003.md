# PLAT-L8-003 자산별 사용량 집계를 조회한다

## 식별

- 티켓 ID: `PLAT-L8-003`
- 소속 계층: `L8` — `platform/02-architecture.md §3`
- 관련 결정: `D10`, `D11`, `D12` — `platform/02-architecture.md §결정 요약`
- 선행 티켓: `[PLAT-L6B-005, PLAT-L8-001, PLAT-L8-002]`

## 맥락(Context)

- 문제: 원시 설치·Broker 호출 이벤트를 승격과 운영 지표에 쓸 집계로 읽는 서비스가 없다.
- 필요한 이유: 계측 가능한 범위와 누락 구간을 숨기지 않고 자산·버전·팀·사업부별 사용량을 재현해야 한다.
- 설계 근거:
  - `platform/02-architecture.md §10 텔레메트리`
  - `platform/01-requirements.md FR-14, 열린 질문 14`
  - `platform/03-operations.md §운영 지표(KPI)`
- 시작 파일:
  - `src/registry/telemetry/contracts.py`
  - `src/registry/telemetry/otel.py`

## 범위(In scope)

- [ ] 기간·자산·버전·팀·사업부 기준의 사용량 집계 질의 서비스를 구현한다.
- [ ] 가짜 `UsageStore`로 자산 유형별 지표와 계측 누락률 계산을 검증한다.
- 변경 허용 경로: `src/registry/telemetry/usage.py`, `tests/unit/test_usage_query.py`

## 범위 밖(Out of scope)

- 전사 통합 KPI 전송과 대시보드 UI
- OpenSearch 저장 어댑터와 평점 집계

## 수용 기준(Acceptance Criteria)

- [ ] AC-1: hosted agent와 원격 MCP server는 호출 수·활성 사용자·성공률·지연을 기간과 불변 ID 기준으로 집계한다.
- [ ] AC-2: Skill과 설치형 MCP server는 설치 수만 집계하고 실행 수·성공률을 0으로 추정하지 않고 `unavailable`로 표시한다.
- [ ] AC-3: 결과에는 출처 인스턴스, 포함 기간, 포함 이벤트 수와 계측 누락률이 항상 포함된다.

## 검증 방법

| 수용 기준 | 실행 명령 | 기대 결과 |
|---|---|---|
| AC-1 | `pytest tests/unit/test_usage_query.py::test_broker_assets_aggregate_calls_users_success_and_latency -q` | 종료 코드 0, 경유형 호출 집계 일치 |
| AC-2 | `pytest tests/unit/test_usage_query.py::test_installable_assets_report_execution_as_unavailable -q` | 종료 코드 0, 설치 수만 집계하고 실행 추정 없음 |
| AC-3 | `pytest tests/unit/test_usage_query.py::test_usage_result_includes_source_period_and_completeness -q` | 종료 코드 0, 출처·기간·누락률 포함 |

## 건드리면 안 되는 것

- 수정 금지 파일: `src/registry/broker/**`, `src/registry/api/**`, `src/registry/federation/**`
- 변경 금지 인터페이스: `UsageStore`, 텔레메트리 이벤트 속성
- 동시 작업 소유 경로: `PLAT-L8-002: src/registry/telemetry/otel.py`

## 참고 규약

- `platform/tickets/CONVENTIONS.md` — 외부 OpenSearch 없는 단위 테스트
- `src/registry/telemetry/contracts.py` — 집계 입력·출력 계약

## 완료 보고

- 변경: `src/registry/telemetry/usage.py`, `tests/unit/test_usage_query.py`
- 검증: `pytest tests/unit/test_usage_query.py -q` → 전체 통과
- 남은 일: L4 조회 API와 전사 집계 결정
- 범위 밖 발견: 없음
