# PLAT-L2-004 소유권 관계 튜플과 인가 판정을 구현한다

## 식별

- 티켓 ID: `PLAT-L2-004`
- 소속 계층: `L2` — `platform/02-architecture.md §3`
- 관련 결정: `D7`(관계 튜플), `D15`(엔진 없이 MongoDB), `D16`(병합 1홉) — `§결정 요약`
- 선행 티켓: `PLAT-L2-002`

## 맥락(Context)

- 문제: "이 사람이 이 에이전트의 maintainer 인가" 를 판정할 수단이 없다.
- 필요한 이유: 발행·수정·폐기·승격 신청이 전부 이 판정에 걸린다. 그리고 팀 이름이 매년 바뀌므로 **이름 기반 판정은 쓸 수 없다**.
- 설계 근거:
  - `platform/02-architecture.md §5.2 소유권` — 튜플 형태, namespace 4개, 파생 규칙
  - `platform/02-architecture.md §5.3` — 팀은 `empno` 집합을 멤버로 갖는 그룹
  - `platform/research/authorization-at-scale-google.md §5`
- 시작 파일:
  - `src/registry/catalog/repository.py` (PLAT-L2-002)

## 범위(In scope)

- [ ] `src/registry/auth/tuples.py` — 튜플 모델과 `ownership` 컬렉션 저장소 (Protocol + 가짜 + Mongo)
- [ ] `src/registry/auth/decide.py` — `can(subject, relation, resource) -> bool`
- [ ] 파생 규칙 구현: `owner ⊂ maintainer ⊂ reviewer`, `maintainer ⊃ team:<소유팀>#member`
- [ ] `expiresAt` 만료 판정
- [ ] `successorTeamId` **1홉** 추적
- [ ] 판정 테스트
- 변경 허용 경로: `src/registry/auth/tuples.py`, `src/registry/auth/decide.py`, `tests/unit/test_authz_decide.py`

## 범위 밖(Out of scope)

- OIDC 토큰 검증 (`PLAT-L5-001`)
- HTTP 미들웨어 결합 (`PLAT-L5-002`)
- 팀 멤버십을 **채우는** 일 (`PLAT-L2-006` Org Sync)
- 전용 권한 엔진 도입 — **D15 로 명시적으로 배제됨. 검토하지 말 것**

## 수용 기준(Acceptance Criteria)

- [ ] AC-1: `agent:X#maintainer @team:T-1#member` 튜플 1건과 `team:T-1#member @user:E100` 이 있을 때, `can("user:E100", "maintainer", "agent:X")` 가 True 다
- [ ] AC-2: `owner` 튜플만 있는 사용자가 `maintainer`·`reviewer` 판정에서도 True 다 (상위가 하위를 포함)
- [ ] AC-3: `expiresAt` 이 과거인 튜플은 판정에서 **무시**된다
- [ ] AC-4: 소유 팀이 `merged` 이고 `successorTeamId` 가 있으면 승계 팀 멤버가 True 다. **2홉은 따라가지 않고** False + 경고 로그
- [ ] AC-5: subject 문자열에 표시명·이메일 형식(`@` 포함, 한글 등)이 오면 `ValueError` 로 거부된다 — **subject 는 `user:<empno>` 또는 `team:<team_id>#member` 만**
- [ ] AC-6: 관계 없는 사용자는 False 다 (기본 거부)
- [ ] AC-7: 판정 1회가 튜플 조회 **2회 이하**로 끝난다 (호출 횟수 검증)

## 검증 방법

| 수용 기준 | 실행 명령 | 기대 결과 |
|---|---|---|
| AC-1, AC-2, AC-6 | `pytest tests/unit/test_authz_decide.py::test_basic_derivation -q` | 통과 |
| AC-3 | `pytest tests/unit/test_authz_decide.py::test_expired_tuple_ignored -q` | 통과 |
| AC-4 | `pytest tests/unit/test_authz_decide.py::test_successor_one_hop_only -q` | 1홉 True, 2홉 False 검증 |
| AC-5 | `pytest tests/unit/test_authz_decide.py::test_subject_format_enforced -q` | `ValueError` 검증 후 통과 |
| AC-7 | `pytest tests/unit/test_authz_decide.py::test_lookup_count -q` | 조회 2회 이하 검증 |

## 건드리면 안 되는 것

- 수정 금지 파일: `src/registry/catalog/models.py`, `src/registry/catalog/transitions.py`
- 변경 금지 인터페이스: `CatalogRepository`
- 동시 작업 소유 경로: `PLAT-L5-002` 가 이 `can()` 을 호출한다. 여기서 미들웨어를 만들지 않는다

## 참고 규약

- `platform/02-architecture.md §5.2` 의 절대 규칙: **튜플의 subject 에 표시명·이메일을 넣지 않는다.** AC-5 가 이것을 코드로 강제한다

## 완료 보고

- 변경: <파일별 한 줄>
- 검증: <실행한 명령> → <실제 결과>
- 남은 일: 없음
- 범위 밖 발견: <없음 | 기록>
