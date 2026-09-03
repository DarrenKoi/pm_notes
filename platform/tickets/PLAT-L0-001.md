# PLAT-L0-001 구현 저장소 스켈레톤과 CI 를 세운다

## 식별

- 티켓 ID: `PLAT-L0-001`
- 소속 계층: `L0` (기반) — `platform/02-architecture.md §3`
- 관련 결정: `D3`(FastAPI), `D4`(단일 앱 + 워커) — `platform/02-architecture.md §결정 요약`
- 선행 티켓: 없음

## 맥락(Context)

- 문제: 구현 저장소가 존재하지 않는다. 이후 모든 티켓이 전제하는 디렉터리·테스트 러너·린트가 없다.
- 필요한 이유: 뒤따르는 모든 티켓의 검증 명령이 `pytest`, `ruff check src tests` 로 고정돼 있다. 이 티켓이 그 명령을 성립시킨다.
- 설계 근거:
  - `platform/02-architecture.md §3 소프트웨어 계층`, `§4 프로세스 구성`
  - `platform/tickets/CONVENTIONS.md`
- 시작 파일:
  - 없음 (신규 저장소)

## 범위(In scope)

- [ ] `CONVENTIONS.md` 의 디렉터리 구조 생성. 각 패키지에 `__init__.py` 만 둔다
- [ ] `pyproject.toml` — Python 3.11, 의존성은 CONVENTIONS 기술 고정 목록만, ruff·pytest 설정
- [ ] `pytest.ini` 또는 pyproject 에 `integration` 마커 등록과 기본 제외 설정
- [ ] `docker-compose.yml` — MongoDB, Redis, OpenSearch, MinIO
- [ ] CI 워크플로 1개 — `ruff check` + `pytest tests/unit`
- [ ] 스켈레톤이 살아있음을 증명하는 테스트 1개
- 변경 허용 경로: 저장소 전체 (최초 생성)

## 범위 밖(Out of scope)

- 어떤 계층의 실제 로직도 구현하지 않는다. 빈 패키지만 만든다
- FastAPI 라우터·엔드포인트 정의 (L4 티켓)
- DB 연결 코드 (각 계층 티켓)

## 수용 기준(Acceptance Criteria)

- [ ] AC-1: `pytest tests/unit` 이 종료 코드 0 으로 끝나고 최소 1개 테스트가 통과한다
- [ ] AC-2: `ruff check src tests` 가 종료 코드 0 으로 끝난다
- [ ] AC-3: `pytest` 기본 실행이 `tests/integration` 을 **수집하지 않는다** (마커로 제외)
- [ ] AC-4: `CONVENTIONS.md` 에 나열된 모든 패키지 디렉터리가 존재하고 import 가능하다
- [ ] AC-5: `docker compose config` 가 종료 코드 0 으로 끝나고 4개 서비스를 출력한다

## 검증 방법

| 수용 기준 | 실행 명령 | 기대 결과 |
|---|---|---|
| AC-1 | `pytest tests/unit -q` | 종료 코드 0, `1 passed` 이상 |
| AC-2 | `ruff check src tests` | 종료 코드 0 |
| AC-3 | `pytest --collect-only -q \| grep -c integration` | 0 |
| AC-4 | `pytest tests/unit/test_skeleton.py -q` | 모든 패키지 import 성공 |
| AC-5 | `docker compose config --services` | `mongodb`, `redis`, `opensearch`, `minio` 출력 |

## 건드리면 안 되는 것

- 수정 금지 파일: 없음 (신규)
- 변경 금지 인터페이스: 없음
- 동시 작업 소유 경로: 없음

## 참고 규약

- `platform/tickets/CONVENTIONS.md` — 디렉터리 구조와 기술 고정 목록을 **그대로** 따른다. 목록에 없는 의존성을 추가하지 않는다

## 완료 보고

- 변경: <파일별 한 줄>
- 검증: <실행한 명령> → <실제 결과>
- 남은 일: 없음
- 범위 밖 발견: <없음 | 기록>
