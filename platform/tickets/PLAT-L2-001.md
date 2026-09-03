# PLAT-L2-001 카탈로그 문서 스키마 계약을 확정한다

## 식별

- 티켓 ID: `PLAT-L2-001`
- 소속 계층: `L2` (Catalog) — `platform/02-architecture.md §3`
- 관련 결정: `D1`(MCP server.json 확장), `D9`(런타임 중립), `D10`(자산 3종) — `§결정 요약`
- 선행 티켓: `PLAT-L0-001`

## 맥락(Context)

- 문제: 카탈로그 문서의 형태가 설계 문서에만 있고 코드로 강제되지 않는다.
- 필요한 이유: **모든 계층이 이 문서를 소비한다.** 스키마가 흔들리면 전부 흔들린다. 계약을 먼저 못 박고 구현은 뒤에 한다.
- 설계 근거:
  - `platform/02-architecture.md §5.1 agents 컬렉션` — 필드 정의 원문
  - `platform/02-architecture.md §2 자산 종류`, `§2-1 런타임 중립성`
  - `platform/02-architecture.md §11 버전 파편화` — 날짜 스탬프 `$schema` 규약
- 시작 파일:
  - `src/registry/catalog/__init__.py`
  - `schemas/` (PLAT-L0-001 이 만든 빈 디렉터리)

## 범위(In scope)

- [ ] `schemas/2026-09-03/agent.schema.json` — JSON Schema. 파일명에 **날짜 스탬프**
- [ ] `src/registry/catalog/models.py` — 위 스키마에 대응하는 Pydantic v2 모델
- [ ] 검증 함수 `validate_agent_doc(doc) -> AgentDoc`
- [ ] 스키마 계약 테스트
- 변경 허용 경로: `schemas/`, `src/registry/catalog/models.py`, `tests/unit/test_agent_schema.py`

## 범위 밖(Out of scope)

- MongoDB 저장·조회 (`PLAT-L2-002`)
- 상태 전이 로직 (`PLAT-L2-003`)
- 소유권 튜플 (`PLAT-L2-004`)
- OpenSearch 색인 스키마 (`PLAT-L3-*`, 다른 담당)

## 수용 기준(Acceptance Criteria)

- [ ] AC-1: `name` 이 `<reverse-dns>/<agent-name>` 패턴(슬래시 정확히 1개)을 만족하지 않으면 검증 실패한다
- [ ] AC-2: `kind` 는 `skill | mcp-server | hosted-agent` 셋만 허용한다
- [ ] AC-3: `kind: "hosted-agent"` 인데 `remotes` 가 비어 있으면 검증 실패한다. `kind: "skill"` 인데 `packages` 가 비어 있으면 검증 실패한다
- [ ] AC-4: `version` 에 범위 표기(`^1.2.3`, `~1.0`, `>=2`)가 오면 검증 실패한다
- [ ] AC-5: `remotes[].sla.availability` 가 `0.995` 미만이면 검증 실패한다
- [ ] AC-6: `_meta` 의 `status` 는 `active | deprecated | deleted | degraded`, `tier` 는 `experimental | verified | official` 만 허용한다
- [ ] AC-7: 유효한 문서 3종(각 `kind` 마다 1개)이 검증을 통과한다

## 검증 방법

| 수용 기준 | 실행 명령 | 기대 결과 |
|---|---|---|
| AC-1~AC-6 | `pytest tests/unit/test_agent_schema.py -q` | 각 거부 케이스가 `ValidationError` 로 실패함을 검증, 종료 코드 0 |
| AC-7 | `pytest tests/unit/test_agent_schema.py::test_valid_documents -q` | 3종 전부 통과 |
| 전체 | `ruff check src tests` | 종료 코드 0 |

## 건드리면 안 되는 것

- 수정 금지 파일: 없음 (신규)
- 변경 금지 인터페이스: 없음 (이 티켓이 최초 정의)
- 동시 작업 소유 경로: `PLAT-L3-001`: OpenSearch 색인 스키마는 별도 계약이다. 이 티켓에서 정의하지 않는다

## 참고 규약

- `platform/02-architecture.md §5.1` 의 JSON 예시를 **필드 이름 그대로** 옮긴다. 이름을 개선하지 않는다

## 완료 보고

- 변경: <파일별 한 줄>
- 검증: <실행한 명령> → <실제 결과>
- 남은 일: 없음
- 범위 밖 발견: <없음 | 기록>
