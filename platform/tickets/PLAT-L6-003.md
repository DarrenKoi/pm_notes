# PLAT-L6-003 generic 카탈로그 응답을 투영한다

## 식별

- 티켓 ID: `PLAT-L6-003`
- 소속 계층: `L6` — `platform/02-architecture.md §3`
- 관련 결정: `D1`, `D9`, `D10` — `platform/02-architecture.md §결정 요약`
- 선행 티켓: `[PLAT-L1-001, PLAT-L2-002, PLAT-L2-004, PLAT-L6-001]`

## 맥락(Context)

- 문제: 특정 marketplace 규격을 쓰지 않는 오픈소스 런타임이 소비할 중립 배포 응답이 없다.
- 필요한 이유: 새 런타임마다 카탈로그와 API 의미를 다시 정의하지 않고 동일 자산을 소비해야 한다.
- 설계 근거:
  - `platform/02-architecture.md §2-1 런타임 중립성`
  - `platform/02-architecture.md §5.1 agents 컬렉션`
  - `platform/01-requirements.md FR-09, FR-29, FR-30`
- 시작 파일:
  - `src/registry/adapters/contracts.py`
  - `src/registry/catalog/repository.py`

## 범위(In scope)

- [ ] 요청 런타임과 권한 범위에 맞는 generic 서버 목록·상세 투영을 구현한다.
- [ ] 설치형 `packages[]`와 원격형 `remotes[]` 직렬화를 단위 테스트한다.
- 변경 허용 경로: `src/registry/adapters/generic.py`, `tests/unit/test_generic_adapter.py`

## 범위 밖(Out of scope)

- `/v0.1/servers` FastAPI 라우터와 페이지 전송
- 런타임별 설치 명령 생성과 카탈로그 쓰기

## 수용 기준(Acceptance Criteria)

- [ ] AC-1: 요청 런타임이 선언되고 호출자에게 허용된 자산만 목록·상세 투영에 포함된다.
- [ ] AC-2: 설치형은 package 식별자·SHA-256을, 원격형은 endpoint·서비스 상태·SLO를 서로 섞지 않고 반환한다.
- [ ] AC-3: 투영 결과의 이름·버전·종류·원본 인스턴스 식별자는 카탈로그 값과 일치하며 입력은 변경되지 않는다.

## 검증 방법

| 수용 기준 | 실행 명령 | 기대 결과 |
|---|---|---|
| AC-1 | `pytest tests/unit/test_generic_adapter.py::test_generic_projection_filters_runtime_and_access -q` | 종료 코드 0, 허용 자산만 반환 |
| AC-2 | `pytest tests/unit/test_generic_adapter.py::test_generic_projection_separates_packages_and_remotes -q` | 종료 코드 0, 유형별 필드 분리 |
| AC-3 | `pytest tests/unit/test_generic_adapter.py::test_generic_projection_preserves_identity_without_mutation -q` | 종료 코드 0, 식별자 보존과 입력 불변 |

## 건드리면 안 되는 것

- 수정 금지 파일: `src/registry/catalog/**`, `src/registry/store/**`, `src/registry/api/**`
- 변경 금지 인터페이스: `DistributionAdapter`, 카탈로그 저장소, `ArtifactStore`
- 동시 작업 소유 경로: `PLAT-L6-002: src/registry/adapters/marketplace.py`

## 참고 규약

- `platform/tickets/CONVENTIONS.md` — 단위 테스트와 기술 고정
- `src/registry/adapters/contracts.py` — 공통 투영 계약

## 완료 보고

- 변경: `src/registry/adapters/generic.py`, `tests/unit/test_generic_adapter.py`
- 검증: `pytest tests/unit/test_generic_adapter.py -q` → 전체 통과
- 남은 일: L4 generic API 라우터
- 범위 밖 발견: 없음
