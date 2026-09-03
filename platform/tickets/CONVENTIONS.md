# 티켓 공통 규약 (구현 저장소)

> 모든 티켓이 전제하는 저장소 구조·명령·기술 선택. 티켓은 이 파일을 복사하지 않고 참조한다.

작성 양식과 규율은 [../05-ticket-conventions.md](../05-ticket-conventions.md), 설계 근거는 [../02-architecture.md](../02-architecture.md).

## 구현 저장소는 별도다

이 저장소(`pm_notes`)는 설계 문서만 담는다. 코드는 **새 저장소**에 만든다. 티켓의 모든 경로는 그 구현 저장소 기준이다.

## 저장소 구조

```
src/registry/
  catalog/        # L2 — 카탈로그 문서, 스키마, 소유권 튜플
  store/          # L1 — ArtifactStore 인터페이스와 구현
  auth/           # L5 — OIDC 검증, 인가 판정
  orgsync/        # 조직 동기화 (§5.4)
  search/         # L3 — OpenSearch 색인·질의
  api/            # L4 — FastAPI 라우터
  adapters/       # L6 — 런타임별 배포 표면
  broker/         # L6b — Endpoint Broker
  telemetry/      # L8
  federation/     # L9 — 승격
schemas/          # JSON Schema (날짜 스탬프 파일명)
tests/
  unit/
  integration/    # testcontainers 또는 docker-compose 의존
```

## 기술 고정

| 항목 | 값 |
|---|---|
| 언어 | Python 3.11 |
| 웹 | FastAPI + uvicorn |
| 검증 | Pydantic v2 |
| DB | MongoDB (motor, 비동기) |
| 캐시/큐 | Redis |
| 검색 | OpenSearch |
| 객체 저장 | MinIO (S3 API, boto3 또는 minio-py) |
| 테스트 | pytest, pytest-asyncio |
| 린트/포맷 | ruff |

## 표준 명령

| 목적 | 명령 |
|---|---|
| 전체 테스트 | `pytest` |
| 단위 테스트만 | `pytest tests/unit` |
| 특정 테스트 | `pytest tests/unit/test_x.py::test_y -q` |
| 린트 | `ruff check src tests` |
| 로컬 의존 서비스 기동 | `docker compose up -d` |

**모든 티켓의 검증 명령은 위 형식을 쓴다.** 티켓마다 다른 러너를 발명하지 않는다.

## 티켓 ID 체계

`PLAT-<계층>-<번호>` — 계층은 `L1`~`L10`, `L6B`, 그리고 계층에 속하지 않는 기반 작업은 `L0`.

## 공통 금지 사항

모든 티켓에 적용되므로 개별 티켓에 반복하지 않는다.

- 티켓 범위 밖 리팩터링 금지
- 스키마·API 계약 변경은 **계약 티켓으로만**. 구현 티켓에서 계약을 바꾸지 않는다
- 새 서드파티 의존성 추가 금지 (위 기술 고정 목록 밖). 필요하면 티켓에 사유를 적고 사람 승인
- 테스트 없이 완료 보고 금지
- 미결 사항(설계 문서의 열린 질문)을 임의로 구현하지 않는다. 막히면 보고하고 멈춘다

## 외부 의존이 없는 환경에서의 테스트

로컬 개발 환경에 MongoDB·OpenSearch 가 없을 수 있다. 그러므로:

- **단위 테스트는 외부 서비스 없이 통과해야 한다.** 저장소 계층은 인터페이스 뒤에 두고 가짜 구현으로 테스트한다.
- 실제 서비스가 필요한 것은 `tests/integration/` 에 두고 `@pytest.mark.integration` 을 붙인다. 기본 실행에서 제외한다.
- 티켓의 AC 는 **단위 테스트로 판정 가능해야 한다.** 통합 테스트만으로 판정되는 AC 는 `ready` 가 아니다.
