# PLAT-L1-002 MinIO ArtifactStore 구현과 불변성 설정을 만든다

## 식별

- 티켓 ID: `PLAT-L1-002`
- 소속 계층: `L1` — `platform/02-architecture.md §3`
- 관련 결정: `D8`(Object Lock GOVERNANCE 모드) — `platform/02-architecture.md §결정 요약`
- 선행 티켓: `PLAT-L1-001`

## 맥락(Context)

- 문제: `ArtifactStore` 계약만 있고 실제 저장 구현이 없다.
- 필요한 이유: 발행된 버전은 **불변**이어야 한다. 동시에 보안 사고 시 회수는 가능해야 한다 — 그래서 COMPLIANCE 가 아니라 GOVERNANCE 모드다.
- 설계 근거:
  - `platform/02-architecture.md §3 L1`, `§결정 요약 D8`
  - `platform/research/agent-skill-registry-research.md` — MinIO Object Lock 은 versioning 을 요구하며, 버킷 생성 시 활성화하면 versioning 이 자동으로 켜진다
- 시작 파일:
  - `src/registry/store/base.py` (PLAT-L1-001)

## 범위(In scope)

- [ ] `src/registry/store/minio_store.py` — `MinioArtifactStore`
- [ ] 버킷 부트스트랩 함수 — **버킷 생성 시 Object Lock 활성화**, 발행 객체에 GOVERNANCE 리텐션 부여
- [ ] `get_url` 은 만료되는 presigned URL 을 반환한다
- [ ] 계약 테스트(`PLAT-L1-001`)를 이 구현에 적용 — 통합 테스트로
- [ ] MinIO 클라이언트를 주입 가능하게 하여 **단위 테스트는 가짜 클라이언트로** 통과시킨다
- 변경 허용 경로: `src/registry/store/minio_store.py`, `tests/unit/test_minio_store.py`, `tests/integration/test_minio_store_integration.py`

## 범위 밖(Out of scope)

- GitLab / OCI 구현 (GitLab 실측 결과가 나온 뒤 별도 티켓)
- `ArtifactStore` 인터페이스 변경 — 계약은 `PLAT-L1-001` 소유
- 서명·SBOM 저장

## 수용 기준(Acceptance Criteria)

- [ ] AC-1: 버킷 부트스트랩이 Object Lock **활성 상태로** 버킷을 만들고, 이미 있으면 멱등하게 통과한다
- [ ] AC-2: `put` 이 객체에 **GOVERNANCE 모드 리텐션**을 설정한다 (가짜 클라이언트로 호출 인자 검증)
- [ ] AC-3: `get_url` 이 반환한 URL 에 만료 파라미터가 포함되고, `ttl` 인자가 그 값에 반영된다
- [ ] AC-4: 이미 존재하는 키에 `put` 을 다시 호출하면 **거부**된다 (발행 버전 불변)
- [ ] AC-5: 계약 테스트 전체가 `MinioArtifactStore` 에서 통과한다 (통합 테스트, `@pytest.mark.integration`)
- [ ] AC-6: MinIO 없이 `pytest tests/unit` 이 통과한다

## 검증 방법

| 수용 기준 | 실행 명령 | 기대 결과 |
|---|---|---|
| AC-1, AC-2 | `pytest tests/unit/test_minio_store.py -q` | 통과. GOVERNANCE·Object Lock 인자 검증 |
| AC-3 | `pytest tests/unit/test_minio_store.py::test_presigned_url_ttl -q` | 통과 |
| AC-4 | `pytest tests/unit/test_minio_store.py::test_put_rejects_existing_key -q` | 거부 검증 후 통과 |
| AC-5 | `docker compose up -d minio && pytest tests/integration/test_minio_store_integration.py -m integration -q` | 종료 코드 0 |
| AC-6 | `pytest tests/unit -q` | 종료 코드 0 (MinIO 미기동 상태에서) |

## 건드리면 안 되는 것

- 수정 금지 파일: `src/registry/store/base.py` (계약은 PLAT-L1-001 소유)
- 변경 금지 인터페이스: `ArtifactStore` Protocol
- 동시 작업 소유 경로: 없음

## 참고 규약

- `tests/unit/test_store_contract.py` — 공용 계약 테스트 함수를 재사용한다. 복사하지 않는다

## 완료 보고

- 변경: <파일별 한 줄>
- 검증: <실행한 명령> → <실제 결과>
- 남은 일: 없음
- 범위 밖 발견: <없음 | 기록>
