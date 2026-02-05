---
tags: [opensearch, search-engine, elasticsearch, vector-search]
level: beginner
last_updated: 2026-02-05
status: in-progress
---

# OpenSearch 기초 (OpenSearch Basics)

> AWS가 Elasticsearch를 포크하여 만든 오픈소스 검색/분석 엔진. 벡터 검색과 키워드 검색을 모두 지원한다.

## 왜 필요한가? (Why)

### OpenSearch vs Elasticsearch

2021년 Elastic이 라이선스를 변경하면서 AWS가 Elasticsearch 7.10을 포크하여 **OpenSearch**를 만들었다.

| 항목 | OpenSearch | Elasticsearch |
|------|------------|---------------|
| 라이선스 | Apache 2.0 (완전 오픈소스) | Elastic License / SSPL |
| 관리 주체 | AWS + 커뮤니티 | Elastic |
| 클라우드 서비스 | Amazon OpenSearch Service | Elastic Cloud |
| k-NN (벡터 검색) | 기본 내장 | 8.0부터 지원 |
| API 호환성 | ES 7.10 호환 | - |

### 언제 OpenSearch를 선택하는가?

- **AWS 환경**에서 관리형 서비스 사용 시
- **완전한 오픈소스**가 필요할 때
- **벡터 검색 + 키워드 검색** 모두 필요할 때
- 기존 Elasticsearch 7.x 마이그레이션 시

### 주요 활용 사례

- **RAG (Retrieval-Augmented Generation)**: 문서 임베딩 저장 및 유사 문서 검색
- **로그 분석**: 애플리케이션/인프라 로그 수집 및 분석
- **전문 검색(Full-text Search)**: 웹사이트, 문서 검색 기능
- **보안 분석 (SIEM)**: 보안 이벤트 모니터링

---

## 핵심 개념 (What)

### 아키텍처 구성요소

```
┌─────────────────────────────────────────────────────────────┐
│                     OpenSearch Cluster                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Node 1    │  │   Node 2    │  │   Node 3    │         │
│  │  (Master)   │  │   (Data)    │  │   (Data)    │         │
│  │             │  │             │  │             │         │
│  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │         │
│  │ │ Shard 0 │ │  │ │ Shard 1 │ │  │ │ Shard 2 │ │         │
│  │ │(Primary)│ │  │ │(Primary)│ │  │ │(Primary)│ │         │
│  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │         │
│  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │         │
│  │ │ Shard 2 │ │  │ │ Shard 0 │ │  │ │ Shard 1 │ │         │
│  │ │(Replica)│ │  │ │(Replica)│ │  │ │(Replica)│ │         │
│  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

#### 노드 타입 (Node Types)

| 노드 타입 | 역할 | 설정 |
|----------|------|------|
| **Master** | 클러스터 상태 관리, 인덱스 생성/삭제 | `node.master: true` |
| **Data** | 데이터 저장, 검색/인덱싱 실행 | `node.data: true` |
| **Ingest** | 데이터 전처리 파이프라인 | `node.ingest: true` |
| **Coordinating** | 요청 라우팅, 결과 집계 | 모든 역할 false |

### Index, Document, Field

```
Index (인덱스)           → RDBMS의 Database/Table
  └── Document (문서)    → RDBMS의 Row
       └── Field (필드)  → RDBMS의 Column
```

**예시**:
```json
// Index: products
{
  "_index": "products",
  "_id": "1",
  "_source": {
    "name": "노트북",           // Field
    "description": "고성능...", // Field
    "price": 1500000,          // Field
    "embedding": [0.1, 0.2...] // Field (벡터)
  }
}
```

### Shard와 Replica

- **Primary Shard**: 데이터가 분산 저장되는 단위 (생성 후 변경 불가)
- **Replica Shard**: Primary의 복제본 (고가용성, 읽기 성능 향상)

```python
# 인덱스 생성 시 샤드 설정
{
    "settings": {
        "number_of_shards": 3,      # Primary 샤드 수
        "number_of_replicas": 1     # 각 Primary당 Replica 수
    }
}
```

> **가이드라인**: Shard 하나당 10~50GB 권장. 너무 많은 샤드는 오버헤드 발생.

### Mapping (매핑)

Document의 구조와 필드 타입을 정의한다. RDBMS의 스키마와 유사.

```json
{
  "mappings": {
    "properties": {
      "title": { "type": "text" },
      "category": { "type": "keyword" },
      "price": { "type": "integer" },
      "created_at": { "type": "date" },
      "embedding": {
        "type": "knn_vector",
        "dimension": 1536
      }
    }
  }
}
```

#### 주요 필드 타입

| 타입 | 설명 | 용도 |
|------|------|------|
| `text` | 분석기로 토큰화됨 | 전문 검색 대상 |
| `keyword` | 분석 안 됨, 정확 매칭 | 필터, 집계, 정렬 |
| `integer/long/float` | 숫자 | 범위 검색, 집계 |
| `date` | 날짜/시간 | 시계열 데이터 |
| `boolean` | true/false | 필터 조건 |
| `object` | 중첩 JSON | 복합 데이터 |
| `knn_vector` | 벡터 (k-NN 플러그인) | 유사도 검색 |

### Analyzer (분석기)

텍스트를 검색 가능한 토큰으로 변환하는 과정.

```
"OpenSearch는 검색 엔진이다"
        ↓ Analyzer
[opensearch, 검색, 엔진]  (토큰화 + 소문자화)
```

```
┌─────────────────────────────────────────────────────────┐
│                      Analyzer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Char Filter  │→ │  Tokenizer   │→ │ Token Filter │  │
│  │ (문자 변환)   │  │ (토큰 분리)  │  │ (토큰 가공)   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 어떻게 사용하는가? (How)

### 1. Docker로 OpenSearch 설치

**docker-compose.yml**:
```yaml
version: '3'
services:
  opensearch:
    image: opensearchproject/opensearch:2.11.0
    container_name: opensearch
    environment:
      - discovery.type=single-node
      - bootstrap.memory_lock=true
      - "OPENSEARCH_JAVA_OPTS=-Xms512m -Xmx512m"
      - DISABLE_SECURITY_PLUGIN=true  # 개발용
    ulimits:
      memlock:
        soft: -1
        hard: -1
    volumes:
      - opensearch-data:/usr/share/opensearch/data
    ports:
      - 9200:9200  # REST API
      - 9600:9600  # Performance Analyzer

  opensearch-dashboards:
    image: opensearchproject/opensearch-dashboards:2.11.0
    container_name: opensearch-dashboards
    ports:
      - 5601:5601
    environment:
      - OPENSEARCH_HOSTS=["http://opensearch:9200"]
      - DISABLE_SECURITY_DASHBOARDS_PLUGIN=true

volumes:
  opensearch-data:
```

```bash
# 실행
docker compose up -d

# 상태 확인
curl -X GET "http://localhost:9200/_cluster/health?pretty"
```

### 2. Python 클라이언트 설치

```bash
pip install opensearch-py
```

### 3. 클러스터 연결 및 정보 확인

```python
from opensearchpy import OpenSearch

# 클라이언트 생성 (보안 비활성화 상태)
client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_compress=True,
    use_ssl=False,
    verify_certs=False,
)

# 클러스터 정보
info = client.info()
print(f"Cluster: {info['cluster_name']}")
print(f"Version: {info['version']['number']}")

# 클러스터 상태
health = client.cluster.health()
print(f"Status: {health['status']}")  # green/yellow/red
print(f"Nodes: {health['number_of_nodes']}")
```

### 4. 인덱스 생성 및 관리

```python
index_name = "documents"

# 인덱스 생성
index_body = {
    "settings": {
        "index": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "knn": True  # k-NN 활성화
        }
    },
    "mappings": {
        "properties": {
            "title": {"type": "text"},
            "content": {"type": "text"},
            "category": {"type": "keyword"},
            "embedding": {
                "type": "knn_vector",
                "dimension": 1536,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "nmslib"
                }
            }
        }
    }
}

if not client.indices.exists(index=index_name):
    response = client.indices.create(index=index_name, body=index_body)
    print(f"Index created: {response['acknowledged']}")

# 인덱스 목록 확인
indices = client.cat.indices(format="json")
for idx in indices:
    print(f"{idx['index']}: {idx['docs.count']} docs, {idx['store.size']}")

# 인덱스 삭제
# client.indices.delete(index=index_name)
```

### 5. 문서 CRUD

```python
# 단일 문서 삽입
doc = {
    "title": "OpenSearch 소개",
    "content": "OpenSearch는 오픈소스 검색 엔진입니다.",
    "category": "tutorial",
    "embedding": [0.1] * 1536  # 실제로는 임베딩 모델 사용
}

response = client.index(
    index=index_name,
    body=doc,
    id="doc-001",  # 생략 시 자동 생성
    refresh=True   # 즉시 검색 가능하도록
)
print(f"Created: {response['_id']}")

# 문서 조회
doc = client.get(index=index_name, id="doc-001")
print(doc["_source"])

# 문서 수정
client.update(
    index=index_name,
    id="doc-001",
    body={"doc": {"category": "guide"}}
)

# 문서 삭제
# client.delete(index=index_name, id="doc-001")
```

### 6. 벌크 작업 (Bulk Operations)

대량 데이터 처리 시 필수.

```python
from opensearchpy import helpers

# 벌크 삽입할 문서들
documents = [
    {"title": f"문서 {i}", "content": f"내용 {i}", "category": "bulk"}
    for i in range(100)
]

# 벌크 액션 생성
actions = [
    {
        "_index": index_name,
        "_id": f"bulk-{i}",
        "_source": doc
    }
    for i, doc in enumerate(documents)
]

# 벌크 실행
success, errors = helpers.bulk(client, actions, refresh=True)
print(f"Indexed: {success}, Errors: {len(errors)}")
```

### 7. 클러스터 관리 명령어

```python
# 클러스터 통계
stats = client.cluster.stats()
print(f"Total docs: {stats['indices']['docs']['count']}")
print(f"Store size: {stats['indices']['store']['size_in_bytes'] / 1024 / 1024:.2f} MB")

# 노드 정보
nodes = client.nodes.info()
for node_id, node_info in nodes["nodes"].items():
    print(f"Node: {node_info['name']}, Roles: {node_info['roles']}")

# 인덱스 설정 변경
client.indices.put_settings(
    index=index_name,
    body={"index": {"number_of_replicas": 1}}
)

# 인덱스 새로고침 (검색에 반영)
client.indices.refresh(index=index_name)
```

---

## 참고 자료 (References)

- [OpenSearch 공식 문서](https://opensearch.org/docs/latest/)
- [opensearch-py GitHub](https://github.com/opensearch-project/opensearch-py)
- [OpenSearch Docker 설치](https://opensearch.org/docs/latest/install-and-configure/install-opensearch/docker/)
- [Amazon OpenSearch Service](https://aws.amazon.com/opensearch-service/)

## 관련 문서

- [벡터 검색 (k-NN)](./vector-search-knn.md) - 의미 기반 유사도 검색
- [키워드 검색 (BM25)](./keyword-search-bm25.md) - 전문 검색
- [OpenSearch 시리즈 목차](./README.md)

---

*Last updated: 2026-02-05*
