---
tags: [opensearch, performance, tuning, scaling, sharding, heap]
level: advanced
last_updated: 2026-02-07
status: in-progress
---

# OpenSearch 성능 최적화 (Performance & Scaling)

> 100GB 이상의 데이터를 운영할 때 고려해야 할 샤딩 전략, 메모리 관리, 색인/검색 최적화 가이드

## 왜 필요한가? (Why)

데이터가 작을 때는 기본 설정으로도 충분하지만, 데이터가 10GB, 100GB, 1TB 단위로 커지면 성능 저하가 발생한다.
"100GB 데이터를 얻었다"면, 이를 효율적으로 저장하고 빠르게 검색하기 위해 아키텍처를 고민해야 한다.

- **샤드(Shard)가 너무 많으면**: 메모리 오버헤드가 커지고 관리 비용이 증가한다.
- **샤드가 너무 적으면**: 하나의 샤드가 너무 커져서 이동/복구가 느려지고 검색 병렬성이 떨어진다.
- **JVM 힙(Heap) 설정**: 잘못 설정하면 OOM(Out of Memory)이나 잦은 GC(Garbage Collection)로 멈춘다.

---

## 핵심 개념 (What)

### 1. 샤딩 전략 (Sharding Strategy)

#### 샤드 크기 가이드라인
OpenSearch(Elasticsearch)에서 **하나의 샤드 크기는 10GB ~ 50GB** 사이가 권장된다.

- **로그 데이터 (시계열)**: 약간 커도 됨 (30GB~50GB)
- **검색 데이터 (RAG/일반)**: 검색 성능을 위해 적절히 유지 (10GB~30GB)

#### 100GB 데이터 시나리오
100GB의 원본 데이터가 인덱싱되면, 인덱스 크기는 보통 원본보다 커진다 (분석기, 인덱스 구조 등). 약 120GB~150GB라고 가정하자.

- **추천 설정**: Primary Shard **3개 ~ 5개**
    - 3개 설정 시: 약 40~50GB/shard (적당함)
    - 5개 설정 시: 약 25~30GB/shard (검색 성능 유리)
    - 1개 설정 시: 100GB+ (너무 큼, 비추천)

```json
PUT /my-large-index
{
  "settings": {
    "number_of_shards": 3,
    "number_of_replicas": 1
  }
}
```

### 2. 메모리 관리 (JVM Heap vs OS Cache)

OpenSearch 노드의 메모리는 크게 두 부분으로 나뉜다.

1.  **JVM Heap**: OpenSearch 프로세스가 직접 사용 (인덱싱 버퍼, 캐시 등).
2.  **OS Filesystem Cache (Lucene)**: 운영체제가 관리. 실제 데이터 파일(세그먼트)을 캐싱하여 검색 속도에 결정적 영향을 줌.

**황금률 (Golden Rule)**:
> **물리 메모리의 50%를 JVM Heap에 할당하고, 나머지 50%는 OS 캐시로 남겨둔다.**
> (단, Heap은 최대 30~32GB를 넘기지 않는다.)

**예시**: 16GB RAM 서버라면?
- JVM Heap: 8GB
- OS Cache: 8GB (자동으로 남음)

### 3. 세그먼트 병합 (Segment Merging)

문서가 추가/수정/삭제되면 새로운 "세그먼트" 파일이 계속 생성된다. 백그라운드에서 이들을 병합(Merge)하는데, 검색 성능을 위해 인위적으로 병합할 수 있다.

- **Force Merge**: 더 이상 데이터 변경이 없는 인덱스(예: 지난달 로그)는 세그먼트를 1개로 병합하면 검색 속도가 비약적으로 향상된다.

---

## 최적화 방법 (How)

### 1. 대량 색인 속도 높이기 (Indexing Optimization)

100GB 데이터를 처음 넣을 때(Initial Load) 적용한다.

#### A. Refresh Interval 비활성화
기본값 `1s`는 매초마다 검색 가능하게 만든다. 대량 색인 중에는 불필요한 부하를 준다.

```json
PUT /my-index/_settings
{
  "index": {
    "refresh_interval": "-1"
  }
}
```
*완료 후 반드시 `"1s"` 등으로 복구해야 한다.*

#### B. Replica 수 0으로 설정
색인 중에는 복제본을 만들지 않고, 색인이 끝난 후 복제본 수를 늘리는 것이 훨씬 빠르다.

```json
PUT /my-index/_settings
{
  "index": {
    "number_of_replicas": 0
  }
}
```

#### C. Translog 설정
트랜잭션 로그(Translog)의 플러시 주기를 늦춘다.

```json
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
서버 재시작 직후에는 OS 캐시가 비어있어 느리다. 주요 쿼리를 미리 실행해준다.

#### B. _source 필드 제외
검색 결과 리스트에서는 `title`, `summary` 등 필요한 필드만 가져오고, 상세 내용은 별도 조회하거나 ID만 가져온다.

```json
GET /my-index/_search
{
  "_source": ["title", "id"],
  "query": { ... }
}
```

#### C. Force Merge (Read-only 인덱스)
데이터 업데이트가 끝난 인덱스에 적용.

```bash
POST /my-index/_forcemerge?max_num_segments=1
```

### 3. 벡터 검색(k-NN) 최적화

100GB 데이터에 벡터가 포함된다면 메모리 사용량이 급증한다.

- **HNSW 메모리**: HNSW 그래프는 메모리에 로드되어야 빠르다.
- **Circuit Breaker**: 힙 메모리 부족을 방지하기 위해 k-NN 메모리 한계를 설정한다.

**opensearch.yml 설정 예시**:
```yaml
# k-NN 인덱스가 사용할 수 있는 힙 메모리 비율 (기본 50%)
knn.memory.circuit_breaker.limit: 60%
```

---

## 100GB 운영 체크리스트

1.  [ ] **하드웨어**: SSD 필수 (HDD는 벡터 검색/랜덤 I/O에 매우 느림).
2.  [ ] **메모리**: 100GB 데이터라면 최소 16GB~32GB RAM 권장.
3.  [ ] **샤드 수**: 3~5개로 설정했는가?
4.  [ ] **매핑(Mapping)**:
    - 불필요한 필드는 `index: false`로 설정했는가?
    - 문자열은 `keyword`와 `text` 중 용도에 맞게 설정했는가?
    - 벡터 차원(Dimension)은 모델과 일치하는가?
5.  [ ] **벌크 사이즈**: Python 클라이언트에서 5~10MB 단위로 보내고 있는가?

---

## 참고 자료 (References)

- [OpenSearch Tuning Guide](https://opensearch.org/docs/latest/tuning-your-cluster/)
- [Size your shards](https://opensearch.org/blog/optimize-elasticsearch-shard-size/)

## 관련 문서

- [OpenSearch Python 클라이언트](./python-client.md) - 실제 코드로 구현하기
- [OpenSearch 기초](./opensearch-basics.md)
