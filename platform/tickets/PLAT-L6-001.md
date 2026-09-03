# PLAT-L6-001 읽기 전용 배포 어댑터 계약을 확정한다

## 식별

- 티켓 ID: `PLAT-L6-001`
- 소속 계층: `L6` — `platform/02-architecture.md §3`
- 관련 결정: `D1`, `D2`, `D9`, `D10` — `platform/02-architecture.md §결정 요약`
- 선행 티켓: `[PLAT-L0-001, PLAT-L1-001, PLAT-L2-001]`

## 맥락(Context)

- 문제: 런타임별 표현을 추가할 공통 입력·출력과 읽기 전용 경계가 없다.
- 필요한 이유: 새 런타임 지원이 카탈로그 스키마 변경이나 쓰기 권한 확대로 이어지지 않아야 한다.
- 설계 근거:
  - `platform/02-architecture.md §2-1 런타임 중립성`
  - `platform/02-architecture.md §3 L6 Distribution`
  - `platform/01-requirements.md FR-10, FR-29, FR-30`
- 시작 파일:
  - `src/registry/catalog/models.py`
  - `src/registry/store/base.py`

## 범위(In scope)

- [ ] 정규화 카탈로그 문서를 런타임별 배포 문서로 투영하는 `DistributionAdapter` Protocol과 결과 모델을 정의한다.
- [ ] 호환 런타임 필터와 설치형·원격형 표현의 계약 테스트를 추가한다.
- 변경 허용 경로: `src/registry/adapters/contracts.py`, `tests/unit/test_adapter_contract.py`

## 범위 밖(Out of scope)

- 특정 런타임의 JSON 생성 구현
- 카탈로그 수정, 아티팩트 업로드와 HTTP 라우터

## 수용 기준(Acceptance Criteria)

- [ ] AC-1: 어댑터 입력은 불변 카탈로그 문서와 요청 런타임이고 출력은 직렬화 가능한 읽기 전용 투영이다.
- [ ] AC-2: `packages[]` 자산은 설치 참조를, `remotes[]` 자산은 원격 연결 참조를 결과 모델로 표현한다.
- [ ] AC-3: 요청 런타임이 `runtimes[]`에 없으면 어댑터 계약은 빈 투영을 반환하며 입력 문서를 변경하지 않는다.

## 검증 방법

| 수용 기준 | 실행 명령 | 기대 결과 |
|---|---|---|
| AC-1 | `pytest tests/unit/test_adapter_contract.py::test_adapter_contract_returns_read_only_projection -q` | 종료 코드 0, 직렬화 가능한 투영 생성 |
| AC-2 | `pytest tests/unit/test_adapter_contract.py::test_projection_distinguishes_packages_and_remotes -q` | 종료 코드 0, 설치·원격 참조 분리 |
| AC-3 | `pytest tests/unit/test_adapter_contract.py::test_unsupported_runtime_is_empty_and_input_unchanged -q` | 종료 코드 0, 빈 결과와 입력 불변 검증 |

## 건드리면 안 되는 것

- 수정 금지 파일: `src/registry/catalog/**`, `src/registry/store/**`, `src/registry/api/**`
- 변경 금지 인터페이스: 카탈로그 문서 스키마와 `ArtifactStore`
- 동시 작업 소유 경로: `PLAT-L1-001: src/registry/store/**, PLAT-L2-001: src/registry/catalog/**`

## 참고 규약

- `platform/tickets/CONVENTIONS.md` — 구현 경로와 기술 고정
- `platform/02-architecture.md §2-1` — 어댑터는 읽기 전용 투영

## 완료 보고

- 변경: `src/registry/adapters/contracts.py`, `tests/unit/test_adapter_contract.py`
- 검증: `pytest tests/unit/test_adapter_contract.py -q` → 전체 통과
- 남은 일: `PLAT-L6-002`, `PLAT-L6-003`
- 범위 밖 발견: 없음
