# PLAT-L1-001 ArtifactStore 인터페이스 계약을 확정한다

## 식별

- 티켓 ID: `PLAT-L1-001`
- 소속 계층: `L1` (Artifact Store) — `platform/02-architecture.md §3`
- 관련 결정: `D2`(아티팩트 직접 호스팅) — `platform/02-architecture.md §결정 요약`
- 선행 티켓: `PLAT-L0-001`

## 맥락(Context)

- 문제: 아티팩트를 어디에 저장할지가 **아직 결정되지 않았다.** MinIO / GitLab generic package / OCI 레지스트리 중 어느 것이 될지는 GitLab 실측(`platform/06-gitlab-check.md`) 결과에 달렸다.
- 필요한 이유: 결정을 기다리며 멈추지 않기 위해, 저장소를 **좁은 인터페이스 하나 뒤에** 둔다. 이후 계층은 이 계약만 보고 개발한다.
- 설계 근거:
  - `platform/06-gitlab-check.md §결정을 미루기 위한 설계 대응`
  - `platform/02-architecture.md §3 소프트웨어 계층 L1`
- 시작 파일:
  - `src/registry/store/__init__.py` (PLAT-L0-001 이 만든 빈 패키지)

## 범위(In scope)

- [ ] `src/registry/store/base.py` — `ArtifactStore` Protocol 정의. 메서드는 **3개만**: `put`, `get_url`, `exists`
- [ ] 저장 키 형식을 함수 하나로 고정 (`build_key(name, version, filename) -> str`)
- [ ] `InMemoryArtifactStore` — 테스트용 가짜 구현
- [ ] 계약을 검증하는 테스트. 어떤 구현이든 통과해야 하는 **공용 계약 테스트 함수**로 작성한다
- 변경 허용 경로: `src/registry/store/`, `tests/unit/test_store_contract.py`

## 범위 밖(Out of scope)

- MinIO 구현 (`PLAT-L1-002`)
- Object Lock·리텐션 설정 (`PLAT-L1-002`)
- 서명·SBOM 저장 (P2 거버넌스 티켓)
- 인터페이스에 메서드를 더 추가하는 것. **3개로 충분한지 의심되면 보고하고 멈춘다**

## 수용 기준(Acceptance Criteria)

- [ ] AC-1: `ArtifactStore` 가 `put(key, data) -> str`, `get_url(key, ttl) -> str`, `exists(key) -> bool` 세 메서드만 갖는다
- [ ] AC-2: `build_key("kr.co.skhynix.dram.T-004821/defect-summarizer", "1.4.2", "bundle.tar.gz")` 가 결정적(deterministic)이고 같은 입력에 항상 같은 값을 반환한다
- [ ] AC-3: 키에 경로 탈출 문자(`..`, 선행 `/`)가 포함되면 `ValueError` 로 거부된다
- [ ] AC-4: 공용 계약 테스트를 `InMemoryArtifactStore` 에 적용해 전부 통과한다 — put 후 exists 가 True, 미존재 키는 False, get_url 이 비어있지 않은 문자열
- [ ] AC-5: `put` 이 반환하는 식별자(digest 또는 버전 식별자)가 같은 내용에 대해 안정적이다

## 검증 방법

| 수용 기준 | 실행 명령 | 기대 결과 |
|---|---|---|
| AC-1 | `pytest tests/unit/test_store_contract.py::test_protocol_surface -q` | 통과. 메서드 3개 외 공개 메서드 없음 |
| AC-2, AC-5 | `pytest tests/unit/test_store_contract.py::test_key_is_deterministic -q` | 통과 |
| AC-3 | `pytest tests/unit/test_store_contract.py::test_key_rejects_traversal -q` | `ValueError` 발생 검증 후 통과 |
| AC-4 | `pytest tests/unit/test_store_contract.py -q` | 종료 코드 0 |

## 건드리면 안 되는 것

- 수정 금지 파일: `src/registry/catalog/`, `src/registry/api/`
- 변경 금지 인터페이스: 없음 (이 티켓이 최초 정의)
- 동시 작업 소유 경로: 없음

## 참고 규약

- `platform/tickets/CONVENTIONS.md §외부 의존이 없는 환경에서의 테스트` — 단위 테스트는 MinIO 없이 통과해야 한다

## 완료 보고

- 변경: <파일별 한 줄>
- 검증: <실행한 명령> → <실제 결과>
- 남은 일: 없음
- 범위 밖 발견: <없음 | 기록>
