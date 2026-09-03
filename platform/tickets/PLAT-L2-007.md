# PLAT-L2-007 멤버 집합 비교로 조직 개편 유형을 추론한다

## 식별

- 티켓 ID: `PLAT-L2-007`
- 소속 계층: `L2` — `platform/02-architecture.md §3`
- 관련 결정: `D14` — `§결정 요약`
- 선행 티켓: `PLAT-L2-006`

## 맥락(Context)

- 문제: 동기화가 멤버십은 갱신하지만, **무슨 일이 일어났는지**(rename 인지 분할인지)를 구분하지 못한다.
- 필요한 이유: rename·코드 변경은 자동 처리해도 안전하지만, 분할·병합·해체는 자산 소유권의 행선지를 정해야 하므로 **사람 승인**이 필요하다. 이 구분이 없으면 둘 다 자동이 되거나 둘 다 수동이 된다.
- 설계 근거:
  - `platform/02-architecture.md §5.4` — 관측·해석·처리 표
  - `platform/03-operations.md` 조직 개편 대응 절차
- 시작 파일:
  - `src/registry/orgsync/sync.py` (PLAT-L2-006)

## 범위(In scope)

- [ ] `src/registry/orgsync/detect.py` — 이전/현재 스냅샷을 받아 개편 이벤트 목록을 반환
- [ ] 5가지 유형 판정: `renamed`, `code_changed`, `split`, `merged`, `dissolved`
- [ ] 자동 처리 대상(`renamed`, `code_changed`)은 즉시 반영, 나머지는 **승인 대기 큐에 적재만**
- [ ] 유사도 임계값을 상수로 노출 (기본: 멤버 집합 겹침 70% 이상이면 동일 팀)
- [ ] 테스트
- 변경 허용 경로: `src/registry/orgsync/detect.py`, `tests/unit/test_org_detect.py`

## 범위 밖(Out of scope)

- 승인 UI·API (L4, P2)
- 자산 소유권의 실제 이관 실행 — 승인 후 별도 티켓
- 동기화 배치 자체 (`PLAT-L2-006`)

## 수용 기준(Acceptance Criteria)

- [ ] AC-1: 코드 동일 + 이름만 변경 → `renamed`. **자동 반영**되고 승인 큐에 적재되지 않는다
- [ ] AC-2: 코드 변경 + 멤버 겹침 80% → `code_changed`. `teamId` 가 유지되고 자동 반영된다
- [ ] AC-3: 한 팀 멤버가 2개 팀으로 각각 50%씩 분산 → `split`. **자동 반영되지 않고** 승인 큐에 적재된다
- [ ] AC-4: 두 팀 멤버가 한 팀으로 수렴 → `merged`. 승인 큐에 적재되고 `successorTeamId` 제안이 포함된다
- [ ] AC-5: 팀이 사라지고 멤버가 흩어짐 → `dissolved`. 승인 큐에 적재된다
- [ ] AC-6: 변화 없는 팀은 이벤트를 생성하지 않는다
- [ ] AC-7: 임계값을 60% 로 낮추면 AC-3 의 케이스가 `code_changed` 로 바뀐다 (임계값이 실제로 동작함)

## 검증 방법

| 수용 기준 | 실행 명령 | 기대 결과 |
|---|---|---|
| AC-1~AC-6 | `pytest tests/unit/test_org_detect.py -q` | 종료 코드 0, 5개 유형 각각 검증 |
| AC-3 | `pytest tests/unit/test_org_detect.py::test_split_requires_approval -q` | 자동 반영 안 됨 검증 |
| AC-7 | `pytest tests/unit/test_org_detect.py::test_threshold_is_effective -q` | 통과 |

## 건드리면 안 되는 것

- 수정 금지 파일: `src/registry/orgsync/sync.py`, `src/registry/orgsync/repository.py`
- 변경 금지 인터페이스: `Team`, `TeamRepository`
- 동시 작업 소유 경로: 없음

## 참고 규약

- `platform/02-architecture.md §5.4` 의 표가 판정 규칙의 단일 출처다. 유형을 추가하거나 이름을 바꾸지 않는다

## 완료 보고

- 변경: <파일별 한 줄>
- 검증: <실행한 명령> → <실제 결과>
- 남은 일: 없음
- 범위 밖 발견: <없음 | 기록>
