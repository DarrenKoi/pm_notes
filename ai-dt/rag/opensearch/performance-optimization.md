---
tags: [opensearch, performance, tuning, scaling, sharding, heap]
level: advanced
last_updated: 2026-02-07
reviewed_on: 2026-10-04
review_status: partial
document_type: learning_note
aliases: [OpenSearch 성능과 자원 설계]
category_major: "AI·DT"
category_middle: "RAG"
category_minor: "OpenSearch 검색"
note_kind: "학습"
classified_on: "2026-10-05"
---

# OpenSearch 성능 최적화 (Performance & Scaling)

> 데이터 크기뿐 아니라 검색·쓰기 부하, 복구 목표와 벡터 메모리를 함께 측정하는 설계 가이드

> [!info] 적용 조건 — 확인 2026-10-04
> OpenSearch OSS의 rolling 공식 문서와 별도 Amazon OpenSearch Service 가이드를 대조했다. 아래 수치·설정은 평가 후보이며 회사 환경의 채택 기준이나 실측 결과가 아니다. 서버 판본/엔진/플러그인·노드와 컨테이너 한도를 확인한다. HTTP 블록은 REST Console 형식이며 쉘 명령이 아니다. 이번 검증은 JSON/YAML 구문과 산술뿐이고 운영 설정을 실행하지 않았다.

## 왜 필요한가? (Why)

작은 데이터도 복잡한 질의·동시 요청·큰 벡터 차원으로 병목이 생길 수 있고, 100GB만으로 성능 저하를 단정할 수 없다. 대표 부하에서 처리량·p95/p99 지연·오류/429·GC·복구 시간을 측정하고 한 변수를 바꾸어 비교한다.

- **샤드(Shard)가 너무 많으면**: 메모리 오버헤드가 커지고 관리 비용이 증가한다.
- **샤드가 너무 적으면**: 하나의 샤드가 너무 커져서 이동/복구가 느려지고 검색 병렬성이 떨어진다.
- **JVM 힙(Heap) 설정**: 잘못 설정하면 OOM(Out of Memory)이나 잦은 GC(Garbage Collection)로 멈춘다.

---

## 핵심 개념 (What)

### 1. 샤딩 전략 (Sharding Strategy)

#### 샤드 크기 가이드라인
[Amazon OpenSearch Service의 샤드 가이드](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/bp-sharding.html)는 검색 지연 중심10~30 **GiB**, 쓰기 중심30~50 **GiB**를 출발점으로 제시한다. OSS 모든 엔진/워크로드의 보장값은 아니다. GB와 GiB도 구분하고 벡터·필터·샤드 개수·복구 시간을 실측한다.

#### 100GB 데이터 시나리오
원문100GB→primary 전체120~150GB는 이 문서의 **미평가 가정**이다. 원문 저장/압축·분석기·벡터·삭제 문서 등에 따라 달라진다. 대표 적재에서 primary store를 측정하고 replica·성장·병합 임시 공간을 별도로 합산한다.

- **계산 예시**: primary120~150GB를 균등 분배한다고 가정한다.
    - 3개:40~50GB/shard
    - 5개:24~30GB/shard
    - 1개:120~150GB/shard

샤드를 늘리면 요청 fan-out/메모리 비용도 증가한다.3~5개 중 어느 것이 빠른지나1개가 무조건 부적절한지는 이 산술로 결정하지 않는다. 아래는 별도 실습 index 생성 후보이며 기존 index의 shard 수를 동적으로 바꾸는 요청이 아니다.

```http
PUT /my-large-index
{
  "settings": {
    "number_of_shards": 3,
    "number_of_replicas": 1
  }
}
```

### 2. 메모리 관리 (JVM Heap vs OS Cache)

메모리는 JVM heap, 프로세스/native 메모리, OS/page cache와 다른 프로세스로 나누어 예산을 잡는다. k-NN native index 메모리는 heap 밖에 있을 수 있다. 컨테이너에서는 호스트 전체 RAM을 노드의 가용 메모리로 간주하지 않는다.

[공식 시스템 설정](https://docs.opensearch.org/latest/install-and-configure/configuring-opensearch/configuration-system/)의 heap 약50%는 출발점이다. 남은50%가 모두 OS 캐시가 되는 것은 아니다. 원문의30~32GB 상한은 JVM 압축 포인터에 관한 관행이며 모든 JDK/heap 배치의 고정 기술 제한으로 확인되지 않았다. 대상 JDK/GC·실제 compressed oops 상태·native 예산을 확인한다.

**16GB 가용 예시**: heap8GB이면 남은8GB를 native/OS/기타가 함께 쓴다. 기본 native k-NN 한도50%를 적용하면 남은8GB의4GB이며 전체 노드 메모리4GB를 미리 할당한다는 뜻은 아니다. 실제 사용·압박을 관찰해야 한다.

### 3. 세그먼트 병합 (Segment Merging)

문서가 추가/수정/삭제되면 새로운 "세그먼트" 파일이 계속 생성된다. 백그라운드에서 이들을 병합(Merge)하는데, 검색 성능을 위해 인위적으로 병합할 수 있다.

- **Force Merge**: 쓰기가 완료된 index에서 세그먼트 수를 줄이는 작업이다. 단일 세그먼트 검색이 효율적일 수 있지만 속도 향상 폭은 미확인이다. 쓰기가 계속되면 큰 세그먼트/디스크 증가로 악화할 수 있고 `max_num_segments=1`은 공식 설명상 임시 shard 저장 공간을 약2배 필요로 할 수 있다. 연결이 끊겨도 작업이 계속될 수 있다.

---

## 최적화 방법 (How)

### 1. 대량 색인 속도 높이기 (Indexing Optimization)

재적재 가능한 입력을 가진 독립 initial load에서 검토한다. 원래 설정과 복구 계획을 먼저 기록하고 실제 운영 선택은 별도로 승인·평가한다.

#### A. Refresh Interval 비활성화
기본 `1s`는 고정 검색 가시성 보장이 아니다. 명시하지 않은 refresh_interval은 검색 유휴 상태(index.search.idle.after 기본30s)에 따라 background refresh를 멈출 수 있다. `-1`은 자동 refresh를 끄며 검색 가시성을 지연한다. durability/fsync와 다른 설정이다.

```http
PUT /my-index/_settings
{
  "index": {
    "refresh_interval": "-1"
  }
}
```
*완료 후 원래 명시값을 복원하거나 원래 미설정 상태였다면 설정을 제거해 기본 동작으로 돌린다. 무조건1s로 바꾸지 않는다. 필요한 refresh와 replica 복구/상태를 확인한다.*

#### B. Replica 수 0으로 설정
replica0은 복제 비용을 줄이는 평가 후보다. 그동안 replica에 의한 장애 보호가 없어지므로 손실을 허용하고 원문에서 재적재할 수 있을 때 검토한다. 완료 후 원래 replica 수와 allocation 상태를 복구한다. 모든 부하에서 더 빠르다고 보장하지 않는다.

```http
PUT /my-index/_settings
{
  "index": {
    "number_of_replicas": 0
  }
}
```

#### C. Translog 설정
아래 async는 translog의 **fsync/commit durability** 조건을 바꾼다. Lucene flush나 검색 refresh 주기 조정이 아니다. 기본 request와 달리 마지막 동기화 이후 이미 응답한 쓰기도 노드 장애로 잃을 수 있다. sync_interval5s는 현재 기본값이며 이것만으로 플러시를 늦춘다는 설명은 틀렸다. 손실 허용 여부가 확인되지 않은 환경에서는 적용하지 않는다.

```http
PUT /my-index/_settings
{
  "index": {
    "translog.durability": "async",
    "translog.sync_interval": "5s"
  }
}
```

### 2. 검색 성능 최적화 (Search Optimization)

#### A. 캐시 워밍 (Cache Warming)
프로세스 재시작만으로 OS page cache가 항상 비워지지는 않는다. cold/warm 조건을 분리해 실제 대표 쿼리로 지연을 비교한다. OS 캐시·query/request 캐시·native k-NN 캐시는 서로 다르며 vector warmup은 [벡터 문서](./vector-search-knn.md)의 적용 조건을 따른다.

#### B. _source 필드 제외
검색 결과 리스트에서는 `title`, `summary` 등 필요한 필드만 가져오고, 상세 내용은 별도 조회하거나 ID만 가져온다.

```http
GET /my-index/_search
{
  "_source": ["title", "id"],
  "query": {"match_all": {}}
}
```

#### C. Force Merge (Read-only 인덱스)
모든 writer가 종료되고 재쓰기 계획·임시 디스크가 확인된 concrete index에서만 검토한다. 이 요청은 실제 실행하지 않았다. 전체 index/alias wildcard로 확장하지 않는다.

```http
POST /my-index/_forcemerge?max_num_segments=1
```

### 3. 벡터 검색(k-NN) 최적화

원문100GB만으로 벡터 RAM을 정할 수 없다. 벡터 수/차원·엔진·압축·m·replica를 확인한다. native HNSW float 기준 공식 근사 `1.1*(4*dimension+8*m)` bytes/vector는 특정 구조의 추정이며 프로세스 전체 RAM이 아니다. Lucene/디스크 모드 등에 무조건 대입하지 않는다.

- **HNSW 메모리**: page cache/native 경로와 엔진의 loading 방식을 구분한다.
- **Circuit Breaker**: native library index의 heap 밖 메모리 한도이며 JVM parent breaker와 다르다. 기본50%는 heap을 뺀 가용 RAM 기준이다.60%는 원문의 미평가 후보이고 상향을 권고하지 않는다.

**설정 이름 설명** (YAML은 표기 예시이며 적용하지 않음):
```yaml
# native k-NN: heap을 뺀 메모리 기준. 기본50%, 아래60%는 미평가 후보.
knn.memory.circuit_breaker.limit: 60%
```

이 항목은 dynamic cluster setting이다. live 변경은 Cluster Settings API의 적용/복구 경로로 다룬다. 이번에는 어느 서버에도 전송하지 않았다.

```http
PUT /_cluster/settings
{
  "persistent": {"knn.memory.circuit_breaker.limit": "60%"}
}
```

---

## 100GB 운영 체크리스트

1.  [ ] **하드웨어**: 대표 I/O·복구·p95/p99를 측정했는가? SSD는 평가 후보이며100GB만으로 필수 여부를 정하지 않는다.
2.  [ ] **메모리**: 원문16~32GB는 미평가 가정이다. heap/native/cache/컨테이너 한도와 모델/벡터 수로 용량을 산정했는가?
3.  [ ] **샤드 수**: 3~5개 산술 예시를 실제 부하/복구 시간으로 검증했는가?
4.  [ ] **매핑(Mapping)**:
    - 불필요한 필드는 `index: false`로 설정했는가?
    - 문자열은 `keyword`와 `text` 중 용도에 맞게 설정했는가?
    - 벡터 차원(Dimension)은 모델과 일치하는가?
5.  [ ] **벌크 사이즈**: 원문5~10MB는 예시다. 공식 indexing-only 가이드는5~15MiB 출발점에서 측정한다. 문서 수/직렬화 바이트·동시성·429/항목 실패를 함께 확인했는가?

---

## 참고 자료 (References)

확인일 **2026-10-04**. rolling 문서를 실제 배포 판본/부하 증거로 삼지 않는다. 성능·용량·회복성 측정과 JVM 상한은 미확인이다.

- [OpenSearch indexing-only tuning](https://docs.opensearch.org/latest/tuning-your-cluster/performance/) — benchmark의 부하/기기 조건, replica·bulk 후보
- [시스템/heap 설정](https://docs.opensearch.org/latest/install-and-configure/configuring-opensearch/configuration-system/)
- [Index settings](https://docs.opensearch.org/latest/install-and-configure/configuring-opensearch/index-settings/) — refresh·durability/fsync/flush 차이
- [Force Merge](https://docs.opensearch.org/latest/api-reference/index-apis/force-merge/)
- [Vector settings](https://docs.opensearch.org/latest/vector-search/settings/)·[Methods/engines memory estimation](https://docs.opensearch.org/latest/mappings/supported-field-types/knn-methods-engines/)
- [Amazon OpenSearch Service shard sizing](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/bp-sharding.html) — OSS와 managed 조건 구분

## 관련 문서

- [OpenSearch Python 클라이언트](./python-client.md) - 실제 코드로 구현하기
- [OpenSearch 기초](./opensearch-basics.md)
