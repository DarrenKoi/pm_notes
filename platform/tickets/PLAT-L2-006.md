# PLAT-L2-006 사원 마스터에서 팀 멤버십을 동기화하는 배치를 만든다

## 식별

- 티켓 ID: `PLAT-L2-006`
- 소속 계층: `L2` — `platform/02-architecture.md §3`
- 관련 결정: `D14`(자동 동기화), `D17`(1일 1회 야간 배치) — `§결정 요약`
- 선행 티켓: `PLAT-L2-005`

## 맥락(Context)

- 문제: `teams.members` 를 채울 방법이 없다.
- 필요한 이유: 사원 마스터가 `empno` → 이름 / 현재 소속팀명 / 팀 코드명 을 조회 API 로 제공한다. 멤버십을 손으로 관리하지 않는다.
- 설계 근거:
  - `platform/02-architecture.md §5.4 조직 동기화` — 주기, 실패 시 동작, 쓰기 직전 실시간 조회
  - `platform/01-requirements.md` 의 조직 동기화 FR/NFR
- 시작 파일:
  - `src/registry/orgsync/repository.py` (PLAT-L2-005)

## 범위(In scope)

- [ ] `src/registry/orgsync/client.py` — 사원 마스터 조회 클라이언트를 **Protocol 로 추상화** + 가짜 구현
- [ ] `src/registry/orgsync/sync.py` — 1일 1회 배치. 전체 사원 순회 → 팀별 멤버 집합 재구성 → `TeamRepository` 반영
- [ ] 실패 시 **마지막 성공 스냅샷 유지**. 부분 실패로 멤버를 지우지 않는다
- [ ] 연속 실패 카운터와 3일 경보 훅
- [ ] 쓰기 동작용 실시간 조회 함수 `lookup_now(empno)`
- [ ] 테스트
- 변경 허용 경로: `src/registry/orgsync/client.py`, `src/registry/orgsync/sync.py`, `tests/unit/test_org_sync.py`

## 범위 밖(Out of scope)

- 개편 유형 추론 (분할·병합·해체) — `PLAT-L2-007`
- 스케줄러 인프라 (cron/워커 등록은 L0 또는 배포 티켓)
- 사원 마스터 API 의 실제 엔드포인트·인증 — **계약이 확정되기 전에는 Protocol 과 가짜 구현으로 진행한다.** 실제 연동은 후속 티켓

## 수용 기준(Acceptance Criteria)

- [ ] AC-1: 가짜 클라이언트가 사원 3명·팀 2개를 반환할 때, 동기화 후 각 팀의 `members` 가 정확히 일치한다
- [ ] AC-2: 사원 1명이 팀을 옮긴 상태로 다시 동기화하면, 이전 팀에서 빠지고 새 팀에 추가된다. **에이전트 문서와 소유권 튜플은 변경되지 않는다**
- [ ] AC-3: 클라이언트가 예외를 던지면 **기존 `members` 가 그대로 유지**되고 동기화가 실패로 기록된다
- [ ] AC-4: 클라이언트가 사원 목록의 **일부만** 반환하면(부분 응답) 동기화를 **중단**하고 멤버를 지우지 않는다
- [ ] AC-5: 연속 실패 3회 도달 시 경보 훅이 정확히 1회 호출된다
- [ ] AC-6: `lookup_now(empno)` 가 캐시를 거치지 않고 클라이언트를 직접 호출한다
- [ ] AC-7: 새 팀 코드가 나타나면 `team_id` 를 신규 발급하고 `observedOrgCode`/`observedOrgName` 을 기록한다

## 검증 방법

| 수용 기준 | 실행 명령 | 기대 결과 |
|---|---|---|
| AC-1, AC-2, AC-7 | `pytest tests/unit/test_org_sync.py::test_happy_path -q` | 통과 |
| AC-3, AC-4 | `pytest tests/unit/test_org_sync.py::test_failure_preserves_snapshot -q` | 멤버 불변 검증 |
| AC-5 | `pytest tests/unit/test_org_sync.py::test_alert_after_three_failures -q` | 호출 1회 검증 |
| AC-6 | `pytest tests/unit/test_org_sync.py::test_lookup_now_bypasses_cache -q` | 통과 |

## 건드리면 안 되는 것

- 수정 금지 파일: `src/registry/orgsync/models.py`, `src/registry/orgsync/repository.py`
- 변경 금지 인터페이스: `TeamRepository`, `Team`
- 동시 작업 소유 경로: `PLAT-L2-007` 이 이 배치의 결과를 소비한다

## 참고 규약

- **AC-4 가 이 티켓의 핵심이다.** 부분 응답으로 멤버를 지우면 조회 API 장애가 곧 권한 상실이 된다 (`§5.4`)

## 완료 보고

- 변경: <파일별 한 줄>
- 검증: <실행한 명령> → <실제 결과>
- 남은 일: 없음
- 범위 밖 발견: <없음 | 기록>
