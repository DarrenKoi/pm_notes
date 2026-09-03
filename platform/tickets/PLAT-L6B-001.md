# PLAT-L6B-001 Endpoint Broker 계약을 확정한다

## 식별

- 티켓 ID: `PLAT-L6B-001`
- 소속 계층: `L6b` — `platform/02-architecture.md §3`
- 관련 결정: `D10`, `D11`, `D15` — `platform/02-architecture.md §결정 요약`
- 선행 티켓: `[PLAT-L0-001, PLAT-L2-001, PLAT-L2-004, PLAT-L5-001]`

## 맥락(Context)

- 문제: endpoint 등록, 상태 확인, 접근 부여와 호출 중개가 공유할 계약이 없다.
- 필요한 이유: hosted agent와 원격 MCP server를 같은 Broker 경계에서 다루되 카탈로그·인가·텔레메트리 구현과 분리해야 한다.
- 설계 근거:
  - `platform/02-architecture.md §3 L6b Endpoint Broker`
  - `platform/02-architecture.md §5.1 remotes 배열`
  - `platform/01-requirements.md FR-29, FR-35, NFR-15`
- 시작 파일:
  - `src/registry/catalog/models.py`
  - `src/registry/auth/decide.py`

## 범위(In scope)

- [ ] endpoint 등록·상태, 접근 grant와 Broker 호출 요청·결과 모델 및 포트 Protocol을 정의한다.
- [ ] hosted agent·원격 MCP만 허용하고 최소 SLO·불변 ID를 검사하는 계약 테스트를 추가한다.
- 변경 허용 경로: `src/registry/broker/contracts.py`, `tests/unit/test_broker_contract.py`

## 범위 밖(Out of scope)

- endpoint 저장, 네트워크 프록시와 토큰 발급 구현
- 헬스체크·상태 전이와 텔레메트리 발행

## 수용 기준(Acceptance Criteria)

- [ ] AC-1: endpoint 계약은 자산 ID·버전·종류·URL·health URL·운영 팀·SLO를 요구한다.
- [ ] AC-2: 월 가용성 0.995 미만 또는 설치형 자산의 endpoint 등록은 검증 오류로 거부된다.
- [ ] AC-3: 접근 grant는 불변 subject·resource·relation·만료 시각을 포함하고 표시명·이메일 subject를 거부한다.

## 검증 방법

| 수용 기준 | 실행 명령 | 기대 결과 |
|---|---|---|
| AC-1 | `pytest tests/unit/test_broker_contract.py::test_endpoint_contract_requires_identity_health_and_slo -q` | 종료 코드 0, endpoint 필수 필드 검증 |
| AC-2 | `pytest tests/unit/test_broker_contract.py::test_endpoint_rejects_low_availability_and_installable_kind -q` | 종료 코드 0, 잘못된 endpoint 거부 |
| AC-3 | `pytest tests/unit/test_broker_contract.py::test_access_grant_accepts_only_immutable_subjects -q` | 종료 코드 0, 표시명·이메일 subject 거부 |

## 건드리면 안 되는 것

- 수정 금지 파일: `src/registry/catalog/**`, `src/registry/auth/**`, `src/registry/telemetry/**`
- 변경 금지 인터페이스: 카탈로그 `remotes[]`, 관계 튜플과 OIDC 검증 계약
- 동시 작업 소유 경로: `PLAT-L2-004: src/registry/catalog/ownership.py, PLAT-L5-001: src/registry/auth/**`

## 참고 규약

- `platform/tickets/CONVENTIONS.md` — Pydantic v2와 단위 테스트 규칙
- `platform/02-architecture.md §8` — endpoint 등록 게이트

## 완료 보고

- 변경: `src/registry/broker/contracts.py`, `tests/unit/test_broker_contract.py`
- 검증: `pytest tests/unit/test_broker_contract.py -q` → 전체 통과
- 남은 일: `PLAT-L6B-002`~`PLAT-L6B-005`
- 범위 밖 발견: 없음
