---
tags: [milvus, vector-db, embedding, similarity-search]
level: beginner
last_updated: 2026-01-31
status: in-progress
---

# Milvus 기초 (Milvus Basics)

> 오픈소스 벡터 데이터베이스 Milvus의 아키텍처, 핵심 개념, 기본 사용법을 정리한다.

## 왜 필요한가? (Why)

### 전통적 데이터베이스의 한계

전통적인 RDBMS나 NoSQL은 **정확한 값 매칭**(exact match)에 최적화되어 있다. 하지만 AI 애플리케이션에서는 텍스트, 이미지, 오디오 등을 벡터(고차원 수치 배열)로 변환한 뒤 **"의미적으로 가까운" 데이터를 찾는 유사도 검색**(Approximate Nearest Neighbor, ANN)이 필요하다. 이를 효율적으로 처리하려면 전용 벡터 데이터베이스가 필요하다.

### Milvus vs 다른 Vector DB

| 항목 | Milvus | Pinecone | Chroma | Qdrant |
|------|--------|----------|--------|--------|
| 오픈소스 | O | X (Managed) | O | O |
| 분산 처리 | O (Distributed) | O (Managed) | X | 제한적 |
| 스칼라 필터링 | O | O | O | O |
| 하이브리드 검색 | O | O | 제한적 | O |
| 프로덕션 적합성 | 높음 | 높음 | 낮음 (프로토타입용) | 중간 |
| 관리형 서비스 | Zilliz Cloud | Pinecone | X | Qdrant Cloud |

### 주요 활용 사례

- **RAG (Retrieval-Augmented Generation)**: 문서 임베딩 저장 및 질의 시 유사 문서 검색
- **추천 시스템**: 사용자/아이템 벡터 기반 유사도 추천
- **이미지/영상 검색**: 멀티모달 임베딩 기반 시각 검색
- **이상 탐지**: 정상 패턴 벡터와의 거리 기반 이상 탐지

---

## 핵심 개념 (What)

### 아키텍처 (Architecture)

#### Standalone 모드
- 단일 노드에서 실행, 개발/테스트 환경에 적합
- Docker Compose로 간단히 구동 가능

#### Distributed 모드
- Proxy, Query Node, Data Node, Index Node로 분리
- etcd(메타데이터), MinIO(Object Storage), Pulsar(Message Queue) 활용
- 대규모 프로덕션 환경에 적합

### Collection과 Schema

**Collection**은 RDBMS의 테이블에 해당한다. 각 Collection은 **Schema**를 가진다.

```python
from pymilvus import CollectionSchema, FieldSchema, DataType

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
]
schema = CollectionSchema(fields=fields, description="문서 임베딩 저장소")
```

### 주요 필드 타입 (Field Types)

| 타입 | 설명 | 용도 |
|------|------|------|
| `INT64` | 정수 | Primary Key, 메타데이터 |
| `VARCHAR` | 가변 문자열 | 원본 텍스트, 메타데이터 |
| `BOOL` | 불리언 | 필터 조건 |
| `JSON` | JSON 객체 | 유연한 메타데이터 |
| `FLOAT_VECTOR` | 실수형 벡터 | 임베딩 벡터 |
| `SPARSE_FLOAT_VECTOR` | 희소 벡터 | BM25 등 희소 임베딩 |

### Partition

하나의 Collection을 논리적으로 분할하여 검색 범위를 좁힐 수 있다.

```python
collection.create_partition("category_tech")
collection.create_partition("category_science")
```

### 인덱스 타입 (Index Types)

| 인덱스 | 특징 | 사용 시나리오 |
|--------|------|--------------|
| **FLAT** | 전수 검색, 100% 정확도 | 소규모 데이터 (< 10만 건) |
| **IVF_FLAT** | 클러스터링 후 검색, 빠름 | 중규모 데이터, 정확도 중시 |
| **HNSW** | 그래프 기반, 높은 재현율 | 범용, 메모리 여유 있을 때 |
| **SCANN** | Google 개발, 양자화 활용 | 대규모 데이터, 속도 중시 |

### 메트릭 타입 (Metric Types)

| 메트릭 | 설명 | 사용 시나리오 |
|--------|------|--------------|
| **L2** (Euclidean) | 유클리드 거리, 값이 작을수록 유사 | 정규화되지 않은 벡터 |
| **IP** (Inner Product) | 내적, 값이 클수록 유사 | 정규화된 벡터, 추천 시스템 |
| **COSINE** | 코사인 유사도, 방향 기반 | 텍스트 임베딩 (가장 일반적) |

> **팁**: OpenAI, Sentence-Transformers 등 대부분의 텍스트 임베딩 모델은 **COSINE** 메트릭을 사용한다.

### 메타데이터 필터링 (Metadata Filtering)

벡터 유사도 검색과 스칼라 필터를 결합할 수 있다.

```python
# category가 "tech"인 문서 중에서 유사도 검색
results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"nprobe": 10}},
    limit=5,
    expr='category == "tech"',  # 메타데이터 필터
    output_fields=["text", "category"]
)
```

---

## 어떻게 사용하는가? (How)

### 1. Docker로 Milvus 설치 (Standalone)

```bash
# docker-compose.yml 다운로드
wget https://github.com/milvus-io/milvus/releases/download/v2.4.0/milvus-standalone-docker-compose.yml -O docker-compose.yml

# 실행
docker compose up -d

# 상태 확인
docker compose ps
```

기본 포트: `19530` (gRPC), `9091` (HTTP metrics)

### 2. pymilvus 설치 및 연결

```bash
pip install pymilvus
```

```python
from pymilvus import connections, utility

# Milvus 연결
connections.connect("default", host="localhost", port="19530")

# 연결 확인
print(utility.list_collections())
```

### 3. Collection 생성

```python
from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType, utility
)

connections.connect("default", host="localhost", port="19530")

# 기존 Collection 삭제 (개발 시)
if utility.has_collection("documents"):
    utility.drop_collection("documents")

# Schema 정의
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
]
schema = CollectionSchema(fields=fields, description="RAG 문서 저장소")

# Collection 생성
collection = Collection(name="documents", schema=schema)
print(f"Collection 생성 완료: {collection.name}")
```

### 4. 벡터 삽입 (Insert)

```python
import numpy as np

# 예시 데이터 (실제로는 임베딩 모델 사용)
texts = ["Milvus는 벡터 데이터베이스이다.", "RAG는 검색 증강 생성이다."]
sources = ["doc1.pdf", "doc2.pdf"]
embeddings = np.random.rand(2, 1536).tolist()  # 실제로는 OpenAI 등 사용

# 삽입
data = [texts, sources, embeddings]
mr = collection.insert(data)
print(f"삽입 완료: {mr.insert_count}건, IDs: {mr.primary_keys}")

# 삽입 후 flush (디스크에 영속화)
collection.flush()
```

### 5. 인덱스 생성

```python
# HNSW 인덱스 생성 (범용적으로 추천)
index_params = {
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {"M": 16, "efConstruction": 256}
}
collection.create_index("embedding", index_params)
print("인덱스 생성 완료")

# 검색 전 Collection 로드 (메모리에 올림)
collection.load()
```

### 6. 유사도 검색 (Search)

```python
# 검색 쿼리 벡터 (실제로는 임베딩 모델로 생성)
query_embedding = np.random.rand(1, 1536).tolist()

# 검색 실행
search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
results = collection.search(
    data=query_embedding,
    anns_field="embedding",
    param=search_params,
    limit=5,
    output_fields=["text", "source"]
)

# 결과 출력
for hits in results:
    for hit in hits:
        print(f"ID: {hit.id}, Score: {hit.score:.4f}")
        print(f"  Text: {hit.entity.get('text')}")
        print(f"  Source: {hit.entity.get('source')}")
```

### 7. 하이브리드 검색 (Hybrid Search)

Dense 벡터(의미 검색)와 Sparse 벡터(키워드 검색)를 결합한다.

```python
from pymilvus import AnnSearchRequest, WeightedRanker

# Dense 검색 요청
dense_req = AnnSearchRequest(
    data=[dense_query_vector],
    anns_field="dense_embedding",
    param={"metric_type": "COSINE", "params": {"ef": 64}},
    limit=10
)

# Sparse 검색 요청
sparse_req = AnnSearchRequest(
    data=[sparse_query_vector],
    anns_field="sparse_embedding",
    param={"metric_type": "IP", "params": {}},
    limit=10
)

# 가중 결합 (Dense 0.7, Sparse 0.3)
reranker = WeightedRanker(0.7, 0.3)
results = collection.hybrid_search(
    reqs=[dense_req, sparse_req],
    ranker=reranker,
    limit=5,
    output_fields=["text"]
)
```

---

## 참고 자료 (References)

- [Milvus 공식 문서](https://milvus.io/docs)
- [pymilvus GitHub](https://github.com/milvus-io/pymilvus)
- [Milvus Docker 설치 가이드](https://milvus.io/docs/install_standalone-docker.md)
- [Zilliz Cloud (관리형 Milvus)](https://zilliz.com/)

## 관련 문서

- [Milvus RAG 연동](./milvus-rag-integration.md) - LangChain/LangGraph 통합
- [Milvus 시리즈 목차](./README.md)
- [LangGraph RAG](../langgraph/langgraph-rag.md) - Corrective RAG 파이프라인

---

*Last updated: 2026-01-31*
