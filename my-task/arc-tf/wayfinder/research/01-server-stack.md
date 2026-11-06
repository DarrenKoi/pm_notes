# 서버 스택(MinIO·OpenSearch·Redis)으로 1단계를 운영할 수 있는가

> 1단계 서버 측 작업 5종을 MinIO·OpenSearch·Redis 3종에 배정하고, 빠진 부분과 용량 산정 방법, 라이선스 검토 항목을 1차 출처 기준으로 정리한 조사 기록입니다. 모든 출처 확인일은 2026-11-06입니다.

## 결론

3종 조합으로 1단계 서버 작업의 대부분을 처리할 수 있습니다. 다만 관계형 데이터(사용자·권한·감사 기록·요청서 상태)를 담을 곳이 없으므로 PostgreSQL 1종을 추가하는 것이 최소 보완입니다. 또한 MinIO 커뮤니티 저장소가 2026-04-25에 보관(archived) 처리되어 더 이상 유지보수되지 않으므로, 객체 저장소는 도입 전에 별도 검토가 필요합니다.

- 처리 가능: 녹화·스크린샷 저장(MinIO), 키워드+벡터 하이브리드 검색(OpenSearch), 관측 이벤트 수신·작업 큐·세션 상태·Vision LLM 호출 제한(Redis)
- 보완 필요: 관계형 데이터 저장소 없음 → PostgreSQL 추가 권장(PostgreSQL License, BSD/MIT 유사 허용형 라이선스)
- 보완 필요: 스크린샷 "이미지 임베딩" 모델 없음 → BGE-M3는 텍스트 전용임. 1단계는 Vision LLM으로 스크린샷을 텍스트화한 뒤 BGE-M3로 임베딩하는 우회가 현실적임
- 조건부: Redis Streams 작업 큐는 AOF `everysec` 기준 최대 약 1초 분량 유실 가능, at-least-once 전달임 → 작업자 멱등 처리 필수
- 라이선스 검토 필요: MinIO(AGPLv3, 유지보수 종료, 후속 AIStor Free는 단일 노드·수정 금지 조건), Redis 8(RSALv2/SSPLv1/AGPLv3 중 선택 필요)
- 라이선스 문제 적음: OpenSearch(Apache 2.0), Valkey(BSD-3-Clause), PostgreSQL(PostgreSQL License)

## 1. 작업별 담당 구성요소

| 서버 작업 | 담당 | 판단 근거 | 비고 |
|---|---|---|---|
| 화면 녹화·스크린샷 저장 | MinIO | S3 호환 객체 저장소임 | 커뮤니티판 유지보수 종료 → 4절 참고 |
| 지식 자산(절차·노하우·요청서 내용) 저장·검색 | OpenSearch | 텍스트 필드 + `knn_vector` 필드 동시 보유, `hybrid` 쿼리 지원함 | 원본 파일은 MinIO, 색인은 OpenSearch로 분리함 |
| 막힌 상황의 유사 사례 검색 | OpenSearch | 한 문서에 텍스트 벡터·이미지 벡터 필드를 함께 둘 수 있음 | 이미지 임베딩 모델 부재가 실제 공백임 |
| PC 관측 이벤트 수신·작업 큐 | Redis Streams | consumer group, `XACK`, `XPENDING`, `XAUTOCLAIM` 제공함 | 영속성은 AOF 설정에 의존함 |
| 채팅 세션 상태 | Redis | 키-값·해시 + TTL 저장에 적합함 | 장기 보존 대화 기록은 OpenSearch나 PostgreSQL로 이관함 |
| Vision LLM 호출 큐·처리량 제한 | Redis | Streams 큐 + `INCR`/`EXPIRE` 속도 제한 패턴(공식 문서 수록)임 | 동시 실행 상한은 작업자 수로 통제함 |
| 사용자·권한·감사 기록 | (없음) | 3종 모두 관계형 무결성·트랜잭션 조회에 부적합함 | PostgreSQL 추가 권장 |

### 공백 1: 관계형 데이터

사용자·역할·장비 그룹 권한, 요청서의 결재 상태, 누가 어떤 지식 자산을 조회·수정했는지 남기는 감사 기록은 외래 키와 트랜잭션이 필요한 관계형 데이터입니다. OpenSearch Security 플러그인에도 감사 로그 기능이 있지만, 이 기능은 클러스터에 대한 REST·transport 접근(로그인 성공·실패, 권한 허용·거부, 설정 변경)을 기록하는 것으로, 애플리케이션 수준의 업무 감사 기록을 대신하지 않습니다. 따라서 PostgreSQL 1종을 추가하는 것이 가장 작은 보완입니다. PostgreSQL은 BSD·MIT와 유사한 허용형 라이선스이므로 라이선스 부담이 거의 없습니다.

### 공백 2: 내구성 있는 작업 큐

Redis Streams는 consumer group과 pending entries list(PEL)로 미확인 메시지를 재전달할 수 있어 작업 큐로 쓸 수 있습니다. 다만 공식 문서는 메시지가 "여러 번 또는 한 번" 처리될 수 있고 Redis 영속성·복제의 한계가 관여한다고 명시합니다. AOF 기본 정책 `appendfsync everysec`는 장애 시 최대 약 1초 분량의 쓰기를 잃을 수 있습니다. 1단계 규모(엔지니어 10~20명)에서는 별도 메시지 브로커를 추가하기보다, 다음 3가지 조건으로 Redis Streams를 쓰는 편이 단순합니다.

- 관측 이벤트 원본(녹화·스크린샷)은 MinIO에 먼저 저장하고, 큐에는 객체 키만 넣음
- 작업자는 객체 키 기준 멱등 처리함(같은 메시지를 두 번 받아도 결과 동일)
- `appendonly yes` + RDB 스냅샷 병행함(공식 문서가 PostgreSQL 수준 안전성을 원하면 둘 다 쓰라고 권고함)

### 공백 3: 이미지 임베딩 모델

BGE-M3는 텍스트 전용 모델(dense 1024차원, 최대 8192 토큰, MIT 라이선스)입니다. OpenSearch는 어떤 모델이 만든 벡터든 `knn_vector` 필드에 저장할 수 있지만, 스크린샷 자체를 벡터로 바꿀 모델은 현재 사내 제공 목록에 없습니다. 1단계 선택지는 다음과 같습니다.

| 방식 | 내용 | 장점 | 한계 |
|---|---|---|---|
| A. 캡션 경유 | Vision LLM이 스크린샷을 설명 텍스트로 변환 → BGE-M3 임베딩 | 기존 사내 모델만 사용함 | Vision LLM 호출량 증가, 화면 배치 같은 시각 정보 손실 |
| B. 이미지 임베딩 모델 추가 | 이미지·텍스트 공용 임베딩 모델을 사내에 별도 서빙 | 화면 간 직접 유사도 비교 가능함 | 모델 선정·서빙·라이선스 검토 필요함 |

1단계는 A로 시작하고, 유사 화면 검색 품질이 부족할 때 B를 검토하는 순서가 무리가 없습니다. OpenSearch의 멀티모달 검색 기능(`text_image_embedding` 수집 프로세서, 2.11부터)은 ml-commons 커넥터로 외부 모델을 호출하는 방식이므로, 임베딩은 애플리케이션에서 직접 계산한 뒤 벡터를 그대로 색인하는 쪽이 사내 OpenAI 호환 엔드포인트 구조와 맞습니다.

## 2. OpenSearch k-NN 벡터 검색 지원 범위

| 항목 | 내용 | 출처 |
|---|---|---|
| 벡터 필드 | `knn_vector`, 차원 1~16,000 범위(Faiss 기본 상한 16,000) | knn_vector 문서 |
| 데이터 타입 | `float`(기본), `half_float`, `byte`, `binary` | knn_vector 문서 |
| 거리 함수 | `l2`, `cosinesimil`, `innerproduct`, `l1`, `linf`, `hamming`, `hammingbit` | knn_vector 문서 |
| 엔진 | Faiss(HNSW·IVF, 대규모 권장), Lucene(HNSW, 소규모·필터링에 강함), NMSLIB(지원 중단 예정·deprecated), JVector(별도 플러그인, `opensearch-knn`과 공존 불가) | Methods and engines 문서 |
| 텍스트 벡터 검색 | 지원함(외부 계산 벡터 색인 또는 ml-commons 연동) | knn_vector 문서 |
| 이미지 벡터 검색 | 벡터 저장·검색 자체는 모델과 무관하게 지원함. 수집 시 자동 임베딩은 2.11부터 `text_image_embedding` 프로세서 | Multimodal search 문서 |
| 하이브리드 검색 | `hybrid` 쿼리 + 검색 파이프라인 정규화 프로세서(2.10부터), RRF 방식 score ranker 프로세서(2.19부터) | Normalization processor, Score ranker 문서 |
| 하이브리드 제약 | 하위 쿼리 최대 5개, 최상위 쿼리로만 사용, `from > 0`이면 `pagination_depth` 필수 | Hybrid query 문서 |
| 디스크 기반 검색 | `mode: on_disk`(2.17부터), 기본 32배 압축 후 원본 벡터로 재채점 | Disk-based vector search 문서 |

버전 요구사항은 하이브리드 검색과 RRF를 모두 쓰려면 2.19 이상, 디스크 모드까지 고려하면 같은 2.19 이상입니다. 신규 구축이라면 최신 3.x 계열을 기준으로 잡는 것이 안전합니다. 엔진은 1024차원 기준 Faiss HNSW가 기본 선택이며, 장비 그룹·권한 같은 필터를 강하게 거는 검색이 많으면 Lucene도 후보가 됩니다.

## 3. 용량 산정 방법

회사 실측치 없이 쓸 수 있는 산식만 정리합니다. 변수 값은 1단계 시범 운영에서 측정해 채웁니다.

### 3-1. 녹화·스크린샷 저장 용량(MinIO)

```
녹화 용량 = 엔지니어 수 × 일 녹화 시간(h) × 3600 × 평균 비트레이트(bit/s) ÷ 8 × 월 근무일 × 보존 개월 수 × 저장 오버헤드
스크린샷 용량 = 엔지니어 수 × 일 캡처 장수 × 평균 파일 크기 × 월 근무일 × 보존 개월 수 × 저장 오버헤드
저장 오버헤드 = 복제·이레이저 코딩 배수(전체 드라이브 수 ÷ 데이터 드라이브 수)
```

- 비트레이트는 화면 해상도·코덱·프레임 수로 결정됨 → 시범 PC 1대로 1시간 녹화 후 파일 크기를 재는 방식이 가장 정확함
- 저장 오버헤드 배수는 일반적인 이레이저 코딩 원리에 따른 추정치이며, 단일 드라이브 구성이면 1배임
- 장비 그룹 3개는 용량보다 버킷·접근 정책 분리 단위로 반영함

계산 예시(가정값): 15명 × 4시간 × 3600 × 1 Mbit/s ÷ 8 = 약 27 GB/일, 월 21일이면 약 567 GB/월입니다. 보존 기간과 오버헤드를 곱해 총량을 구합니다.

### 3-2. OpenSearch k-NN 메모리

공식 문서의 Faiss 메모리 추정식은 다음과 같습니다.

```
HNSW: 1.1 × (4 × 차원 + 8 × M) 바이트/벡터      (M: 원소당 양방향 링크 수, 기본 16)
IVF : 1.1 × ((4 × 차원 × 벡터 수) + (4 × nlist × 차원)) 바이트
```

- 레플리카는 벡터 수를 그만큼 늘림(레플리카 1개 → 총 벡터 2배)
- k-NN 네이티브 메모리 상한 `knn.memory.circuit_breaker.limit` 기본값은 "JVM 힙을 뺀 나머지 메모리의 50%"임

1024차원, M=16에 대입하면 1.1 × (4 × 1024 + 8 × 16) = 약 4,646 바이트/벡터입니다.

| 벡터 수 N | HNSW 메모리(레플리카 0) | 레플리카 1 |
|---|---|---|
| 100,000 | 약 0.46 GB | 약 0.93 GB |
| 1,000,000 | 약 4.6 GB | 약 9.3 GB |
| 10,000,000 | 약 46 GB | 약 93 GB |

노드 메모리 역산식은 다음과 같습니다.

```
필요 노드 메모리 ≥ JVM 힙 + (HNSW 메모리 ÷ circuit_breaker 비율 0.5)
```

벡터 수 N은 "지식 자산 문서 수 × 문서당 청크 수 × 청크당 벡터 필드 수(텍스트·이미지 캡션 등)"로 추정합니다. 10~20명 규모의 1단계에서는 수십만 벡터 이하가 될 가능성이 높아 메모리는 큰 제약이 아닙니다. 커질 경우 `on_disk` 모드(기본 32배 압축)나 `half_float`·`byte` 타입으로 줄일 수 있습니다. 32배 압축 시 벡터 부분은 1024차원 기준 약 128 바이트로 줄어든다는 계산은 문서 비율을 적용한 추정치입니다.

### 3-3. Redis 메모리

Redis는 공식 산식이 없으므로 실측 기반으로 산정합니다.

```
Redis 메모리 ≈ Σ(키 유형별 개수 × MEMORY USAGE 실측 평균) + 스트림 보존 길이 × 메시지 평균 크기 + 여유분
```

- `MEMORY USAGE key [SAMPLES n]`: 키와 값, 관리 오버헤드를 포함한 바이트 수를 반환함(빈 문자열 키도 약 56 바이트)
- 키 유형 예: 채팅 세션 해시(동시 세션 수 × 크기), 속도 제한 카운터(초 단위 키 + TTL), 작업 스트림(`MAXLEN`으로 상한 고정)
- 여유분: RDB 스냅샷·AOF 재작성 시 `fork()`의 copy-on-write로 메모리가 추가로 쓰일 수 있으므로 `maxmemory`를 물리 메모리보다 낮게 잡음(배수는 실측으로 결정함)
- 큐에는 객체 키만 넣고 이미지·영상 본문은 넣지 않으면 1단계 규모에서 Redis 메모리는 수백 MB 이내로 예상되는 추정치임

## 4. 온프레미스 라이선스 검토

| 구성요소 | 현재 라이선스 | 사내 내부 사용 시 의미 | 검토 필요 여부 |
|---|---|---|---|
| MinIO(커뮤니티) | AGPLv3 | 수정 없이 내부 사용만 하면 소스 공개 의무 발생하지 않음. 수정본을 네트워크로 사용자에게 제공하면 그 사용자에게 소스 제공 의무 발생함 | **필요**: 저장소 2026-04-25 보관 처리, 바이너리 배포 중단(소스 빌드만), 유지보수 없음 |
| MinIO AIStor Free | 독점 무상 라이선스 | 단일 노드(standalone)만 허용, 상용 운영 허용, 수정·역공학·재배포 금지, 지원·SLA 없음 | **필요**: 분산·고가용성 불가, 약관 위반 시 즉시 종료 조항 |
| OpenSearch | Apache 2.0 | 내부 사용·수정 자유, 고지 유지 의무 수준임 | 불필요(표준 오픈소스 고지 관리만) |
| Redis 7.2.x 이하 | BSD-3-Clause | 제약 거의 없음 | 불필요(단, 구버전 유지보수 여부 확인) |
| Redis 7.4~7.8 | RSALv2 / SSPLv1 | 내부 사용 가능, 관리형 서비스 제공·상용화 금지(RSALv2) | **필요** |
| Redis 8 이상 | RSALv2 / SSPLv1 / AGPLv3 중 선택 | AGPLv3 선택 시 수정 없이 내부 사용하면 공개 의무 없음 | **필요**: 어떤 라이선스를 선택하는지 사내 기록 필요 |
| Valkey | BSD-3-Clause | Redis가 라이선스를 바꾸기 직전의 오픈소스판에서 분기한 LF Projects 프로젝트임 | 불필요 |
| PostgreSQL(추가안) | PostgreSQL License | BSD·MIT 유사 허용형임 | 불필요 |

### AGPLv3의 내부 사용 해석

AGPLv3의 핵심은 13조(원격 네트워크 상호작용)입니다. 13조는 "프로그램을 수정한 경우" 그 수정본과 네트워크로 상호작용하는 모든 사용자에게 소스를 받을 기회를 제공하라고 요구합니다. 또한 0조는 "사본 전달 없이 네트워크로 사용자와 상호작용하는 것만으로는 전달(convey)이 아니다"라고 정의합니다. 따라서 MinIO나 AGPLv3판 Redis를 수정 없이 사내에서 S3·Redis 프로토콜로만 호출하는 구조는 소스 공개 의무를 만들지 않는 것으로 읽힙니다. 다만 다음 경우는 법무·오픈소스 검토 대상입니다.

- MinIO·Redis 소스를 사내에서 패치해 운영하는 경우(13조 적용 대상이 됨)
- 플랫폼을 협력사·외부 법인 등 다른 법인 사용자에게 제공하는 경우
- 사내 오픈소스 정책이 AGPL 계열 자체를 금지 목록에 둔 경우

### 대안 정리

- Redis 라이선스 판단이 부담되면 Valkey(BSD-3-Clause)를 대안으로 둠. 1단계에서 쓰는 Streams·해시·TTL·`INCR`은 분기 이전부터 있던 기능임
- MinIO는 라이선스보다 유지보수 종료가 더 큰 위험임. 커뮤니티판 소스 빌드 운영, AIStor Free(단일 노드), 다른 S3 호환 저장소 중 선택이 필요하며, 이는 별도 조사 항목으로 분리함

## 남은 질문

- 사내 오픈소스 정책에서 AGPLv3·RSALv2·SSPLv1의 허용 여부 ⟨확인⟩
- 스크린샷 유사도 검색에 캡션 경유 방식(A)의 품질이 충분한지 ⟨시범 측정⟩
- 녹화 보존 기간과 평균 비트레이트 ⟨시범 측정⟩
- 객체 저장소 후보(MinIO 소스 빌드·AIStor Free·기타 S3 호환) 비교 ⟨별도 조사⟩

## 참고 자료 (References)

모든 출처 확인일: 2026-11-06

- OpenSearch, k-NN vector field type: https://docs.opensearch.org/latest/mappings/supported-field-types/knn-vector/
- OpenSearch, Methods and engines(메모리 추정식·엔진 권장·NMSLIB 지원 중단): https://docs.opensearch.org/latest/mappings/supported-field-types/knn-methods-engines/
- OpenSearch, Hybrid query: https://docs.opensearch.org/latest/query-dsl/compound/hybrid/
- OpenSearch, Normalization processor(2.10): https://docs.opensearch.org/latest/search-plugins/search-pipelines/normalization-processor/
- OpenSearch, Score ranker processor(RRF, 2.19): https://docs.opensearch.org/latest/search-plugins/search-pipelines/score-ranker-processor/
- OpenSearch, Multimodal search(2.11): https://docs.opensearch.org/latest/vector-search/ai-search/multimodal-search/
- OpenSearch, Disk-based vector search(2.17): https://docs.opensearch.org/latest/vector-search/optimizing-storage/disk-based-vector-search/
- OpenSearch, Vector search settings(circuit breaker 기본 50%): https://docs.opensearch.org/latest/vector-search/settings/
- OpenSearch, Audit logs: https://docs.opensearch.org/latest/security/audit-logs/index/
- OpenSearch 저장소(Apache 2.0): https://github.com/opensearch-project/OpenSearch
- MinIO 저장소(AGPLv3, 2026-04-25 보관, 소스 배포만): https://github.com/minio/minio
- MinIO LICENSE: https://github.com/minio/minio/blob/master/LICENSE
- MinIO AIStor Free Tier License Agreement: https://www.min.io/legal/aistor-free-agreement
- GNU AGPLv3 원문(0조·13조): https://www.gnu.org/licenses/agpl-3.0.en.html
- Redis Licenses: https://redis.io/legal/licenses/
- Redis 블로그, AGPLv3 추가 발표(2025-05-01): https://redis.io/blog/agplv3/
- Redis Streams: https://redis.io/docs/latest/develop/data-types/streams/
- Redis persistence: https://redis.io/docs/latest/operate/oss_and_stack/management/persistence/
- Redis MEMORY USAGE: https://redis.io/docs/latest/commands/memory-usage/
- Redis INCR(rate limiter 패턴): https://redis.io/docs/latest/commands/incr/
- Valkey 저장소(BSD-3-Clause, LF Projects): https://github.com/valkey-io/valkey
- PostgreSQL License: https://www.postgresql.org/about/licence/
- BAAI/bge-m3 모델 카드(1024차원, 텍스트 전용, MIT): https://huggingface.co/BAAI/bge-m3
