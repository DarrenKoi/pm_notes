# PLAT-L6-002 Claude Code marketplace 문서를 투영한다

## 식별

- 티켓 ID: `PLAT-L6-002`
- 소속 계층: `L6` — `platform/02-architecture.md §3`
- 관련 결정: `D1`, `D2`, `D9` — `platform/02-architecture.md §결정 요약`
- 선행 티켓: `[PLAT-L1-001, PLAT-L2-002, PLAT-L2-004, PLAT-L6-001]`

## 맥락(Context)

- 문제: Claude Code가 소비할 `marketplace.json` 투영이 없어 정규화 카탈로그를 설치 표면으로 노출할 수 없다.
- 필요한 이유: Claude Code 전용 필드가 카탈로그 진실 원천으로 역류하지 않게 해야 한다.
- 설계 근거:
  - `platform/02-architecture.md §2-1 런타임 중립성`
  - `platform/02-architecture.md §6 식별자 체계`
  - `platform/01-requirements.md FR-09, FR-10, FR-30`
- 시작 파일:
  - `src/registry/adapters/contracts.py`
  - `src/registry/catalog/repository.py`

## 범위(In scope)

- [ ] 권한 있는 `claude-code` 호환 자산을 `marketplace.json` 구조로 투영한다.
- [ ] 설치 URL·버전·무결성 값과 rename map을 가짜 저장소로 검증한다.
- 변경 허용 경로: `src/registry/adapters/marketplace.py`, `tests/unit/test_marketplace_adapter.py`

## 범위 밖(Out of scope)

- FastAPI 응답 라우터와 presigned URL 발급 구현
- 카탈로그 문서 수정과 다른 런타임 지원

## 수용 기준(Acceptance Criteria)

- [ ] AC-1: `claude-code`가 선언된 접근 가능 설치형 자산만 marketplace 플러그인 목록에 포함된다.
- [ ] AC-2: 각 항목은 불변 이름·정확한 버전·다운로드 참조·SHA-256을 보존한다.
- [ ] AC-3: rename map은 이전 이름을 현재 불변 이름으로 투영하며 원본 카탈로그 문서를 변경하지 않는다.

## 검증 방법

| 수용 기준 | 실행 명령 | 기대 결과 |
|---|---|---|
| AC-1 | `pytest tests/unit/test_marketplace_adapter.py::test_marketplace_filters_runtime_and_access -q` | 종료 코드 0, 미지원·비인가 자산 제외 |
| AC-2 | `pytest tests/unit/test_marketplace_adapter.py::test_marketplace_preserves_version_and_digest -q` | 종료 코드 0, 버전·참조·해시 일치 |
| AC-3 | `pytest tests/unit/test_marketplace_adapter.py::test_marketplace_projects_renames_without_catalog_mutation -q` | 종료 코드 0, rename 투영과 입력 불변 검증 |

## 건드리면 안 되는 것

- 수정 금지 파일: `src/registry/catalog/**`, `src/registry/store/**`, `src/registry/api/**`
- 변경 금지 인터페이스: `DistributionAdapter`, 카탈로그 저장소, `ArtifactStore`
- 동시 작업 소유 경로: `PLAT-L6-003: src/registry/adapters/generic.py`

## 참고 규약

- `platform/tickets/CONVENTIONS.md` — 단위 테스트와 경로 규칙
- `src/registry/adapters/contracts.py` — 투영 계약

## 완료 보고

- 변경: `src/registry/adapters/marketplace.py`, `tests/unit/test_marketplace_adapter.py`
- 검증: `pytest tests/unit/test_marketplace_adapter.py -q` → 전체 통과
- 남은 일: L4 응답 라우터
- 범위 밖 발견: 없음
