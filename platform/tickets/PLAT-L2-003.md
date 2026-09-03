# PLAT-L2-003 발행 문서의 불변성과 상태 전이 규칙을 강제한다

## 식별

- 티켓 ID: `PLAT-L2-003`
- 소속 계층: `L2` — `platform/02-architecture.md §3`
- 관련 결정: `D1` — `§결정 요약`
- 선행 티켓: `PLAT-L2-002`

## 맥락(Context)

- 문제: 발행된 문서가 아무 필드나 바뀔 수 있으면, 상위 인스턴스의 증분 폴링(`§9`)이 성립하지 않는다.
- 필요한 이유: **발행 메타데이터 불변**이 2차 플랫폼 동기화의 전제다. 변할 수 있는 것은 상태 계열 필드뿐이다.
- 설계 근거:
  - `platform/02-architecture.md §5.1 불변 규칙` — "변할 수 있는 것은 `status` 계열과 `_meta` 의 심사·승격 필드뿐"
  - `platform/02-architecture.md §8 발행 파이프라인` — "⑤ 이후 문서는 불변"
  - `platform/02-architecture.md §9 승격`
- 시작 파일:
  - `src/registry/catalog/repository.py` (PLAT-L2-002)

## 범위(In scope)

- [ ] `src/registry/catalog/transitions.py` — 상태 전이 함수와 허용 전이 표
- [ ] 가변 필드 화이트리스트를 **한 곳에 상수로** 정의하고, 그 외 필드 변경 시도를 거부
- [ ] 전이 시 `statusChangedAt` 자동 기록
- [ ] 전이·불변성 테스트
- 변경 허용 경로: `src/registry/catalog/transitions.py`, `tests/unit/test_catalog_transitions.py`

## 범위 밖(Out of scope)

- 누가 전이시킬 수 있는지 (인가는 `PLAT-L2-004`)
- 심사 워크플로 UI·API (L4, P2)
- 헬스체크로 인한 `degraded` 자동 전이 (`PLAT-L6B-*`, 다른 담당). **이 티켓은 전이 함수만 제공한다**

## 수용 기준(Acceptance Criteria)

- [ ] AC-1: `name`, `version`, `packages[].fileSha256` 을 바꾸려는 갱신은 거부된다
- [ ] AC-2: `status` 는 허용 전이표에 있는 전이만 성공한다. `deleted → active` 는 거부된다
- [ ] AC-3: 상태 전이 시 `_meta` 의 `statusChangedAt` 이 갱신되고 `publishedAt` 은 **바뀌지 않는다**
- [ ] AC-4: `degraded → active` 는 허용된다 (헬스체크 회복 시 자동 해제)
- [ ] AC-5: `tier` 변경은 허용 전이표를 통해서만 가능하고, `degraded` 상태 때문에 자동으로 낮아지지 않는다
- [ ] AC-6: `isLatest` 재계산이 같은 `name` 안에서 정확히 1개만 true 로 유지한다

## 검증 방법

| 수용 기준 | 실행 명령 | 기대 결과 |
|---|---|---|
| AC-1 | `pytest tests/unit/test_catalog_transitions.py::test_immutable_fields_rejected -q` | 거부 검증 후 통과 |
| AC-2, AC-4 | `pytest tests/unit/test_catalog_transitions.py::test_status_transition_table -q` | 허용·거부 전이 각각 검증 |
| AC-3 | `pytest tests/unit/test_catalog_transitions.py::test_timestamps -q` | 통과 |
| AC-5 | `pytest tests/unit/test_catalog_transitions.py::test_tier_not_auto_demoted -q` | 통과 |
| AC-6 | `pytest tests/unit/test_catalog_transitions.py::test_is_latest_single -q` | 통과 |

## 건드리면 안 되는 것

- 수정 금지 파일: `schemas/`, `src/registry/catalog/models.py`, `src/registry/catalog/repository.py`
- 변경 금지 인터페이스: `CatalogRepository` Protocol
- 동시 작업 소유 경로: `PLAT-L6B-*`: 헬스체크 스케줄러가 이 전이 함수를 **호출**한다. 여기서 스케줄러를 만들지 않는다

## 참고 규약

- `platform/02-architecture.md §8` 의 자동/수동 구분 표를 그대로 반영한다 — **자동으로 하는 것은 `degraded` 전이까지, 등급 강등은 사람**

## 완료 보고

- 변경: <파일별 한 줄>
- 검증: <실행한 명령> → <실제 결과>
- 남은 일: 없음
- 범위 밖 발견: <없음 | 기록>
