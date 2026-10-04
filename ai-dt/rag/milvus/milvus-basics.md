---
tags: [milvus, vector-db, embedding, similarity-search]
reviewed_on: 2026-10-04
review_status: partial
document_type: learning_note
level: beginner
last_updated: 2026-01-31
---

# Milvus 기초 (Milvus Basics)

> 오픈소스 벡터 데이터베이스 Milvus의 아키텍처, 핵심 개념, 기본 사용법을 정리한다.

> [!info] 검토 조건 — 2026-10-04
> 역사적 작성일 2026-01-31을 유지한다. 공식 문서는 확인 시 v3.0.x 표시이며 로컬 확인 판본은 pymilvus3.0.2·milvus-lite3.2.1이다. 최신/운영 검증을 뜻하지 않는다. 예제는 함수를 명시적으로 호출해야 연결·저장이 일어난다. Docker 서버·분산 운영·실제 임베딩 품질·인증은 미확인이다. [정리 기록](../organization-log.md)을 함께 읽는다.


## 왜 필요한가? (Why)

### 전통적 데이터베이스의 한계

벡터 검색은 임베딩 공간에서 가까운 항목을 찾아 검색 후보를 만드는 작업이다. RDBMS 확장이나 다른 저장소도 벡터 검색을 제공하므로 전용 DB가 항상 필수인 것은 아니다. 기존 DB의 기능·데이터 크기·지연·갱신 빈도·운영 인력과 요구 권한을 먼저 비교한다. Milvus는 벡터 인덱스·스칼라 필터·서버 검색을 함께 관리할 때 고려할 수 있다. 가까운 벡터는 문장의 사실성이나 질문에 대한 정답을 보장하지 않는다.

### Milvus vs 다른 Vector DB

| 제품 | 비교할 배포 구성 | 선택 전에 확인할 조건 |
|---|---|---|
| Milvus | Lite·Standalone·Distributed·Zilliz Cloud | 판본별 인덱스/분산 구성·접근 제어·백업·부하 |
| Pinecone | 관리형 서비스 | 계약·지역·비용·필터/검색 API. 이번에 상세 검증하지 않음 |
| Chroma | 로컬 영속 클라이언트·서버·Chroma Cloud | 로컬 객체와 다중 클라이언트 서비스의 운영 방식 구분 |
| Qdrant | 자체 호스팅·Qdrant Cloud | 이번에 분산 동작과 운영 기능을 실행 검증하지 않음 |

기존 표의 `Chroma=메모리 전용/관리형 없음`, `Qdrant=프로덕션 중간` 등은 근거 없는 제품 등급이었다. Chroma의 [PersistentClient](https://docs.trychroma.com/reference/python/client)와 [Chroma Cloud](https://docs.trychroma.com/cloud/getting-started)는 공식 구성이다(확인 2026-10-04). 동일한 데이터/쿼리/배포 조건 없이 제품 간 성능과 운영 적합성 등급을 정하지 않는다. 회사 채택 권고는 Claude 협의 연결 실패로 보류한다.

### 주요 활용 사례

- **RAG (Retrieval-Augmented Generation)**: 문서 임베딩 저장 및 질의 시 유사 문서 검색
- **추천 시스템**: 사용자/아이템 벡터 기반 유사도 추천
- **이미지/영상 검색**: 멀티모달 임베딩 기반 시각 검색
- **이상 탐지**: 정상 패턴 벡터와의 거리 기반 이상 탐지

---

## 핵심 개념 (What)

### 아키텍처 (Architecture)

#### Standalone 모드

한 Milvus 인스턴스에 주요 서비스를 묶는 구성이다. 외부 메타데이터·스토리지와 배포 방법은 선택한 판본/구성 파일을 따른다. 단일 인스턴스라는 사실만으로 개발 전용이나 무장애 운영이라고 판단하지 않는다.

#### Distributed 모드

공식 v3.0.x 설명은 Proxy → Coordinator → Streaming/Query/Data Node → 메타데이터·WAL·객체 저장소의 역할을 구분한다. 쓰기는 WAL 기록과 데이터 처리, 검색은 growing/sealed 데이터 검색 결과 병합을 거친다. etcd·객체 저장소·WAL의 실제 조합은 배포 설정에 따른다. 이전 문서의 별도 Index Node/Pulsar 고정 구성은 모든 판본의 구조가 아니다. 확장·복구는 복제/배포/백업 설계와 부하 검증이 필요하다. [공식 아키텍처](https://milvus.io/docs/architecture_overview.md), 확인 2026-10-04.

### Collection과 Schema

**Collection**은 RDBMS의 테이블에 해당한다. 각 Collection은 **Schema**를 가진다. 아래는 `MilvusClient` API 한 계열로 구성하며 뒤 예제에서도 같은 필드 이름을 사용한다. 1536은 원래 임베딩 예제의 차원이며 모델/설정에 맞게 정해야 한다. 로컬 fixture 검증은 3차원 숫자로 수행했다.

```python
from pymilvus import MilvusClient, DataType

def document_schema(client: MilvusClient, dim: int = 1536):
    if type(dim) is not int or dim <= 0:
        raise ValueError("양의 정수 차원이 필요합니다")
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("text", DataType.VARCHAR, max_length=65535)
    schema.add_field("source", DataType.VARCHAR, max_length=512)
    schema.add_field("category", DataType.VARCHAR, max_length=64)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)
    return schema
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

하나의 Collection을 논리적으로 분할하여 검색 범위를 좁힐 수 있다. 아래는 partition을 지원하는 서버에만 호출한다. Partition은 사용자 인증·권한 경계가 아니다. Lite 공식 제한표는 partition 미지원으로 표시한다. 로컬 Lite와 Docker/Distributed의 동일 동작은 가정하지 않는다. [Lite 제한](https://milvus.io/docs/milvus_lite.md), 확인 2026-10-04.

```python
def create_example_partitions(client: MilvusClient, name: str) -> None:
    # partition을 지원하는 서버에서만 호출. 자동 적재/테넌트 격리는 별도다.
    client.create_partition(name, "category_tech")
    client.create_partition(name, "category_science")
```

### 인덱스 타입 (Index Types)

| 인덱스 | 특징 | 사용 시나리오 |
|--------|------|--------------|
| **FLAT** | 지정 벡터/메트릭의 전수 거리 비교 | ANN 결과 비교 기준. 의미 정확도 100%를 뜻하지 않음 |
| **IVF_FLAT** | 후보 클러스터 선택 후 비교 | nlist/nprobe·recall·지연을 데이터로 측정 |
| **HNSW** | 그래프 탐색 기반 ANN | M/efConstruction/ef·메모리·recall의 절충을 측정 |
| **SCANN** | 양자화 등을 이용한 검색 | 판본/백엔드 지원과 부하 조건을 확인 |

### 메트릭 타입 (Metric Types)

| 메트릭 | 설명 | 사용 시나리오 |
|--------|------|--------------|
| **L2** (Euclidean) | Milvus 반환값은 제곱 유클리드 거리, 작을수록 가까움 | 모델이 요구하는 거리/정규화 계약에 맞춤 |
| **IP** (Inner Product) | 내적, 클수록 가까움; 벡터 크기도 영향 | 두 벡터를 정규화하면 cosine과 관계가 같아짐 |
| **COSINE** | 방향 기반 유사도, 클수록 가까움 | 모델의 학습/권장 메트릭을 확인; 영벡터는 거부 |

같은 임베딩 모델·차원·전처리·정규화를 문서와 질의에 사용하고 인덱스/검색 메트릭을 맞춘다. 모델이 바뀌면 같은 차원이어도 기존 벡터와 섞지 않는다. [공식 메트릭 설명](https://milvus.io/docs/metric.md), 확인 2026-10-04.

### 메타데이터 필터링 (Metadata Filtering)

벡터 유사도 검색과 스칼라 필터를 결합할 수 있다.

```python
def search_tech(client: MilvusClient, name: str, query_vector: list[float],
                dim: int, *, hnsw: bool = False):
    validate_vector(query_vector, dim)
    return client.search(
        collection_name=name, data=[query_vector], anns_field="embedding",
        search_params={"metric_type": "COSINE",
                       "params": {"ef": 64} if hnsw else {}},
        limit=5, filter='category == "tech"',
        output_fields=["text", "source", "category"],
    )
```

---

## 어떻게 사용하는가? (How)

### 1. Docker로 Milvus 설치 (Standalone)

```bash
# 기존 프로젝트/데이터가 없는 새 실습 디렉터리에서만 실행한다.
# 2026-10-04 공식 v3.0.x 페이지의 명시 판본. 업그레이드 절차가 아니다.
mkdir milvus-3.0.2-lab
cd milvus-3.0.2-lab
curl --fail --location https://github.com/milvus-io/milvus/releases/download/v3.0.2/milvus-standalone-docker-compose.yaml --output docker-compose.yaml
# 실행 전에 image/volume/ports/인증값을 검토하고 실습 접근 범위를 설정한다.
docker compose config --quiet
docker compose up -d
docker compose ps
```

[공식 Compose 설치](https://milvus.io/docs/install_standalone-docker-compose.md)는 v3.0.2 기준 etcd·MinIO·Standalone 및 embedded Woodpecker 구성을 제시한다(확인 2026-10-04). 서비스 접근 기본 포트는19530,9091은 WebUI/관리 엔드포인트에도 쓰인다. 실제 공개 범위는 compose 설정을 확인한다. 이번 작업에서 다운로드·Docker 기동·인증 설정·기존 데이터 업그레이드를 실행하지 않았다. 이전 v2.4.0 고정 예제의 재사용은 해당 판본 자료와 호환성을 별도로 확인한다.

### 2. pymilvus 설치 및 연결

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install "pymilvus==3.0.2" "milvus-lite==3.2.1"
```

```python
def connect_milvus(uri: str, *, token: str = "") -> MilvusClient:
    if not isinstance(uri, str) or not uri.strip():
        raise ValueError("대상 URI/실습 DB 경로를 명시하세요")
    return MilvusClient(uri=uri, token=token, timeout=10)
# 사용 후 client.close(). 기존 회사 서버/다른 DB 경로를 추측하지 않는다.
```

문서를 읽는 것만으로 접속하지 않는다. `connect_milvus(uri, token=...)`는 사용자가 정한 서버 또는 새 실습 `.db` 경로에 명시 호출한다. 토큰은 환경 변수/로컬 설정으로 전달한다. 아래 함수들은 정의 순서대로 한 모듈에서 사용한다.

### 3. Collection 생성

```python
def create_documents(client: MilvusClient, name: str, dim: int = 1536) -> None:
    if client.has_collection(name):
        raise ValueError("기존 Collection은 보존합니다. 새 실습 이름을 사용하세요")
    client.create_collection(collection_name=name, schema=document_schema(client, dim))
```

같은 이름의 Collection이 있으면 실패하며 삭제/덮어쓰기하지 않는다. VARCHAR 길이는 UTF-8 바이트 수로 확인한다([공식 VARCHAR](https://milvus.io/docs/string.md), 확인 2026-10-04). 기존 schema 재사용은 별도 모델/메타데이터 계약 검사 후 결정한다.

### 4. 벡터 삽입 (Insert)

```python
import math
from numbers import Real

def validate_vector(vector: list[float], dim: int) -> None:
    if not isinstance(vector, list) or len(vector) != dim:
        raise ValueError("벡터 차원이 다릅니다")
    if any(isinstance(x, bool) or not isinstance(x, Real)
           or not math.isfinite(x) for x in vector) or not any(vector):
        raise ValueError("유한한 비영벡터가 필요합니다")

def insert_rows(client: MilvusClient, name: str, rows: list[dict], dim: int):
    if not rows or len({r.get("id") for r in rows}) != len(rows):
        raise ValueError("비어 있거나 중복 ID인 배치입니다")
    widths = {"id": 64, "text": 65535, "source": 512, "category": 64}
    for row in rows:
        if set(row) != {*widths, "embedding"}:
            raise ValueError("스키마 필드가 다릅니다")
        for key, width in widths.items():
            value = row[key]
            if not isinstance(value, str) or not value.strip() or len(value.encode("utf-8")) > width:
                raise ValueError("빈 문자열 또는 UTF-8 바이트 길이 초과")
        validate_vector(row["embedding"], dim)
    return client.upsert(collection_name=name, data=rows)

# 새 3차원 실습 Collection에만 사용한다. 1536차원 모델 벡터가 아니다.
fixture_rows = [
    {"id": "doc1", "text": "Milvus는 벡터 데이터베이스이다.",
     "source": "doc1.pdf", "category": "tech", "embedding": [1.0, 0.0, 0.0]},
    {"id": "doc2", "text": "RAG는 검색 증강 생성이다.",
     "source": "doc2.pdf", "category": "science", "embedding": [0.0, 1.0, 0.0]},
]
```

fixture는 난수 대신 결정적 3차원 벡터로 API 동작만 시험한다. 실제 문서 의미를 표현하지 않는다. `upsert`는 같은 PK의 재삽입 중복을 줄이나 변경되어 사라진 문서 제거/트랜잭션/동시 적재를 구현하지 않는다. `flush`는 지원 백엔드의 저장 처리 요청이며 백업·재해 복구·권한을 보장하지 않는다.

### 5. 인덱스 생성

```python
def index_and_load(client: MilvusClient, name: str, *, hnsw: bool = False) -> None:
    indexes = client.prepare_index_params()
    indexes.add_index(
        field_name="embedding", index_type="HNSW" if hnsw else "FLAT",
        metric_type="COSINE", params={"M": 16, "efConstruction": 256} if hnsw else {},
    )
    client.create_index(collection_name=name, index_params=indexes)
    client.load_collection(collection_name=name)
# HNSW는 지원 서버에서만 선택. 로컬 검증은 FLAT이다.
# 새 프로세스에서 DB를 다시 열면 검색 전에 client.load_collection(name)을 호출한다.
```

### 6. 유사도 검색 (Search)

```python
def search_documents(client: MilvusClient, name: str, query_vector: list[float],
                     dim: int, *, hnsw: bool = False):
    validate_vector(query_vector, dim)
    return client.search(
        collection_name=name, data=[query_vector], anns_field="embedding",
        search_params={"metric_type": "COSINE",
                       "params": {"ef": 64} if hnsw else {}},
        limit=5, output_fields=["text", "source"],
    )
# MilvusClient 반환값: 질의별 list[list[dict]], 내용은 hit["entity"]에 있다.
# 한 질의라도 results[0]에서 hit["id"], hit["distance"], hit["entity"]를 읽는다.
```

실습 호출 순서는 `connect_milvus(새_실습_DB_URI)` → `create_documents(client, 새_이름, 3)` → `insert_rows(client, 새_이름, fixture_rows, 3)` → `index_and_load(client, 새_이름)` → `search_documents(client, 새_이름, [1.0, 0.0, 0.0], 3)` → `client.close()`다. 실제 모델을 사용할 때는 문서/질의 벡터를 같은 모델로 생성하고 schema 차원을 맞춘다. HNSW 설정을 선택했다면 검색도 `hnsw=True`로 맞춘다.

### 7. 하이브리드 검색 (Hybrid Search)

Dense와 Sparse의 검색 후보를 결합한다. 앞 단일 `embedding` Collection에는 dense/sparse 필드가 없어 아래를 적용할 수 없다. 별도 Collection에 `dense_embedding: FLOAT_VECTOR`와 `sparse_embedding: SPARSE_FLOAT_VECTOR`, 일관된 인코더·각 필드 인덱스와 데이터를 먼저 준비한다. Sparse 벡터가 자동으로 BM25인 것은 아니다. 아래는 준비된 Collection에 호출하는 요청 예제다. 가중치0.7/0.3은 검증된 최적값이 아니다. WeightedRanker는 메트릭별 점수 변환 후 결합하므로 raw dense/sparse 점수의 단순 합과 구분한다. [공식 reranking](https://milvus.io/docs/reranking.md), 확인 2026-10-04. 실제 hybrid 서버 검색·검색 품질은 미확인이다.

```python
from pymilvus import AnnSearchRequest, WeightedRanker

def hybrid_requests(dense_query_vector: list[float], sparse_query_vector: dict[int, float]):
    dense_req = AnnSearchRequest(
        data=[dense_query_vector], anns_field="dense_embedding",
        param={"metric_type": "COSINE", "params": {"ef": 64}}, limit=10,
    )
    sparse_req = AnnSearchRequest(
        data=[sparse_query_vector], anns_field="sparse_embedding",
        param={"metric_type": "IP", "params": {}}, limit=10,
    )
    return [dense_req, sparse_req], WeightedRanker(0.7, 0.3)

def search_prepared_hybrid(client: MilvusClient, hybrid_name: str,
                           dense_query_vector: list[float], sparse_query_vector: dict[int, float]):
    reqs, ranker = hybrid_requests(dense_query_vector, sparse_query_vector)
    return client.hybrid_search(collection_name=hybrid_name, reqs=reqs,
                                ranker=ranker, limit=5, output_fields=["text"])
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
