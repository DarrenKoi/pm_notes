---
tags: [opensearch, vector-search, knn, embedding, semantic-search]
level: intermediate
last_updated: 2026-02-05
status: in-progress
---

# OpenSearch 벡터 검색 (k-NN Vector Search)

> OpenSearch의 k-NN 플러그인을 활용한 의미 기반 유사도 검색(Semantic Search) 구현

## 왜 필요한가? (Why)

### 키워드 검색의 한계

전통적인 키워드 검색은 **정확한 단어 매칭**에 의존한다.

```
쿼리: "자동차 수리"
문서: "차량 정비 방법" → 매칭 실패 ❌ (같은 의미지만 다른 단어)
```

### 벡터 검색의 장점

텍스트를 **고차원 벡터(Embedding)**로 변환하면 **의미적 유사성**을 계산할 수 있다.

```
"자동차 수리" → [0.23, 0.87, -0.12, ...]
"차량 정비"   → [0.25, 0.85, -0.10, ...]  → 코사인 유사도 0.98 ✅
```

### k-NN (k-Nearest Neighbors)

주어진 쿼리 벡터와 가장 가까운 k개의 벡터를 찾는 알고리즘.

```
┌────────────────────────────────────────┐
│            벡터 공간                    │
│                                        │
│    ●doc1      ○query                  │
│         ●doc2                         │
│                    ●doc3              │
│      ●doc4                            │
│                                        │
│   k=2 결과: doc1, doc2 (가장 가까운 2개)│
└────────────────────────────────────────┘
```

---

## 핵심 개념 (What)

### OpenSearch k-NN 플러그인

OpenSearch에 기본 내장된 벡터 검색 플러그인. 별도 설치 불필요.

#### 지원하는 엔진 (Engine)

| 엔진 | 특징 | 사용 시나리오 |
|------|------|--------------|
| **nmslib** | HNSW 알고리즘, 빠른 검색 | 기본 권장, 대부분의 경우 |
| **faiss** | Meta 개발, 다양한 인덱스 | 대규모 데이터, GPU 지원 |
| **lucene** | JVM 네이티브, 필터 친화적 | 필터 조건이 복잡할 때 |

#### 유사도 공간 타입 (Space Type)

| Space Type | 설명 | 점수 계산 |
|------------|------|----------|
| `l2` | 유클리드 거리 | `1 / (1 + l2^2)` (0~1, 클수록 유사) |
| `cosinesimil` | 코사인 유사도 | `(1 + cos) / 2` (0~1, 클수록 유사) |
| `innerproduct` | 내적 | 정규화된 벡터에서 사용 |

> **권장**: 텍스트 임베딩은 대부분 **cosinesimil** 사용

### 인덱스 타입

#### 1. Exact k-NN (정확 검색)

모든 벡터를 전수 비교. 100% 정확하지만 느림.

```json
{
  "type": "knn_vector",
  "dimension": 1536
}
```

#### 2. Approximate k-NN (근사 검색)

HNSW 등 알고리즘으로 빠르게 검색. 약간의 정확도 손실.

```json
{
  "type": "knn_vector",
  "dimension": 1536,
  "method": {
    "name": "hnsw",
    "space_type": "cosinesimil",
    "engine": "nmslib",
    "parameters": {
      "ef_construction": 256,
      "m": 16
    }
  }
}
```

### HNSW 파라미터

| 파라미터 | 설명 | 기본값 | 권장 |
|----------|------|--------|------|
| `m` | 노드당 연결 수 | 16 | 16-64 |
| `ef_construction` | 인덱스 빌드 시 탐색 범위 | 100 | 256-512 |
| `ef_search` | 검색 시 탐색 범위 | 100 | 256+ |

- **m** ↑: 정확도 ↑, 메모리 ↑, 빌드 시간 ↑
- **ef_construction** ↑: 정확도 ↑, 빌드 시간 ↑
- **ef_search** ↑: 정확도 ↑, 검색 시간 ↑

---

## 어떻게 사용하는가? (How)

### 1. 벡터 검색용 인덱스 생성

```python
from opensearchpy import OpenSearch

client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    use_ssl=False,
    verify_certs=False
)

index_name = "semantic-search"

# 인덱스 설정 (k-NN 활성화)
index_body = {
    "settings": {
        "index": {
            "knn": True,  # k-NN 활성화 필수!
            "knn.algo_param.ef_search": 256  # 검색 정확도
        }
    },
    "mappings": {
        "properties": {
            "title": {"type": "text"},
            "content": {"type": "text"},
            "embedding": {
                "type": "knn_vector",
                "dimension": 1536,  # OpenAI ada-002: 1536
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "nmslib",
                    "parameters": {
                        "ef_construction": 256,
                        "m": 16
                    }
                }
            }
        }
    }
}

if client.indices.exists(index=index_name):
    client.indices.delete(index=index_name)

client.indices.create(index=index_name, body=index_body)
print(f"Index '{index_name}' created with k-NN enabled")
```

### 2. 임베딩 생성 및 문서 인덱싱

```python
from openai import OpenAI
from opensearchpy import helpers

# OpenAI 클라이언트 (또는 다른 임베딩 모델)
openai_client = OpenAI()

def get_embedding(text: str, model: str = "text-embedding-ada-002") -> list[float]:
    """텍스트를 임베딩 벡터로 변환"""
    response = openai_client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

# 샘플 문서
documents = [
    {"title": "Python 기초", "content": "Python은 배우기 쉬운 프로그래밍 언어입니다."},
    {"title": "머신러닝 입문", "content": "머신러닝은 데이터에서 패턴을 학습하는 기술입니다."},
    {"title": "웹 개발 가이드", "content": "웹 개발은 프론트엔드와 백엔드로 구성됩니다."},
    {"title": "데이터베이스 기초", "content": "데이터베이스는 데이터를 체계적으로 저장합니다."},
    {"title": "딥러닝 개요", "content": "딥러닝은 인공 신경망을 사용한 머신러닝의 한 분야입니다."},
]

# 임베딩 생성 및 벌크 인덱싱
actions = []
for i, doc in enumerate(documents):
    # 제목 + 내용을 합쳐서 임베딩
    text_to_embed = f"{doc['title']} {doc['content']}"
    embedding = get_embedding(text_to_embed)

    actions.append({
        "_index": index_name,
        "_id": f"doc-{i}",
        "_source": {
            "title": doc["title"],
            "content": doc["content"],
            "embedding": embedding
        }
    })

success, errors = helpers.bulk(client, actions, refresh=True)
print(f"Indexed {success} documents")
```

### 3. 벡터 유사도 검색

```python
def vector_search(query: str, k: int = 5) -> list[dict]:
    """쿼리 텍스트로 유사한 문서 검색"""

    # 쿼리를 임베딩으로 변환
    query_embedding = get_embedding(query)

    # k-NN 검색 쿼리
    search_body = {
        "size": k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_embedding,
                    "k": k
                }
            }
        },
        "_source": ["title", "content"]  # 반환할 필드
    }

    response = client.search(index=index_name, body=search_body)

    results = []
    for hit in response["hits"]["hits"]:
        results.append({
            "id": hit["_id"],
            "score": hit["_score"],
            "title": hit["_source"]["title"],
            "content": hit["_source"]["content"]
        })

    return results

# 검색 실행
query = "인공지능 학습 방법"
results = vector_search(query, k=3)

print(f"Query: {query}\n")
for r in results:
    print(f"[{r['score']:.4f}] {r['title']}")
    print(f"  {r['content']}\n")
```

**출력 예시**:
```
Query: 인공지능 학습 방법

[0.8934] 딥러닝 개요
  딥러닝은 인공 신경망을 사용한 머신러닝의 한 분야입니다.

[0.8756] 머신러닝 입문
  머신러닝은 데이터에서 패턴을 학습하는 기술입니다.

[0.7234] Python 기초
  Python은 배우기 쉬운 프로그래밍 언어입니다.
```

### 4. 필터와 함께 벡터 검색

메타데이터 필터를 벡터 검색과 결합.

```python
def filtered_vector_search(
    query: str,
    category: str = None,
    min_score: float = 0.7,
    k: int = 5
) -> list[dict]:
    """필터 조건과 함께 벡터 검색"""

    query_embedding = get_embedding(query)

    # 기본 k-NN 쿼리
    knn_query = {
        "knn": {
            "embedding": {
                "vector": query_embedding,
                "k": k
            }
        }
    }

    # 필터가 있으면 bool 쿼리로 감싸기
    if category:
        search_body = {
            "size": k,
            "query": {
                "bool": {
                    "must": [knn_query],
                    "filter": [
                        {"term": {"category": category}}
                    ]
                }
            },
            "min_score": min_score
        }
    else:
        search_body = {
            "size": k,
            "query": knn_query,
            "min_score": min_score
        }

    response = client.search(index=index_name, body=search_body)
    return response["hits"]["hits"]

# 사용 예시
results = filtered_vector_search(
    query="머신러닝 학습",
    category="ai",
    min_score=0.75,
    k=5
)
```

### 5. Script Score로 커스텀 유사도 계산

더 세밀한 점수 계산이 필요할 때.

```python
search_body = {
    "size": 5,
    "query": {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "knn_score",
                "lang": "knn",
                "params": {
                    "field": "embedding",
                    "query_value": query_embedding,
                    "space_type": "cosinesimil"
                }
            }
        }
    }
}
```

### 6. 임베딩 모델별 차원 설정

```python
# 임베딩 모델별 차원
EMBEDDING_DIMENSIONS = {
    "text-embedding-ada-002": 1536,      # OpenAI
    "text-embedding-3-small": 1536,      # OpenAI
    "text-embedding-3-large": 3072,      # OpenAI
    "all-MiniLM-L6-v2": 384,             # sentence-transformers
    "all-mpnet-base-v2": 768,            # sentence-transformers
    "multilingual-e5-large": 1024,       # intfloat
    "bge-large-zh-v1.5": 1024,           # BAAI
}

# 인덱스 생성 시 모델에 맞는 차원 사용
model_name = "text-embedding-3-small"
dimension = EMBEDDING_DIMENSIONS[model_name]
```

### 7. 성능 최적화 팁

```python
# 1. 인덱스 설정 최적화
index_settings = {
    "settings": {
        "index": {
            "knn": True,
            "knn.algo_param.ef_search": 512,  # 정확도 ↑
            "refresh_interval": "30s",         # 대량 인덱싱 시
            "number_of_replicas": 0            # 인덱싱 중에는 0
        }
    }
}

# 2. 대량 인덱싱 후 refresh
client.indices.refresh(index=index_name)

# 3. Force merge로 세그먼트 최적화 (인덱싱 완료 후)
client.indices.forcemerge(
    index=index_name,
    max_num_segments=1
)

# 4. 워밍업 (첫 검색 전 캐시 로딩)
warmup_body = {
    "size": 1,
    "query": {
        "knn": {
            "embedding": {
                "vector": [0.0] * 1536,
                "k": 1
            }
        }
    }
}
client.search(index=index_name, body=warmup_body)
```

---

## 실전 예제: RAG용 문서 저장소

```python
from dataclasses import dataclass
from typing import Optional
from opensearchpy import OpenSearch, helpers
from openai import OpenAI

@dataclass
class Document:
    id: str
    title: str
    content: str
    source: str
    embedding: Optional[list[float]] = None

class VectorStore:
    """OpenSearch 기반 벡터 저장소"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9200,
        index_name: str = "rag-documents",
        embedding_dim: int = 1536
    ):
        self.client = OpenSearch(
            hosts=[{"host": host, "port": port}],
            use_ssl=False
        )
        self.openai = OpenAI()
        self.index_name = index_name
        self.embedding_dim = embedding_dim

        self._ensure_index()

    def _ensure_index(self):
        """인덱스가 없으면 생성"""
        if not self.client.indices.exists(index=self.index_name):
            self.client.indices.create(
                index=self.index_name,
                body={
                    "settings": {
                        "index": {"knn": True}
                    },
                    "mappings": {
                        "properties": {
                            "title": {"type": "text"},
                            "content": {"type": "text"},
                            "source": {"type": "keyword"},
                            "embedding": {
                                "type": "knn_vector",
                                "dimension": self.embedding_dim,
                                "method": {
                                    "name": "hnsw",
                                    "space_type": "cosinesimil",
                                    "engine": "nmslib"
                                }
                            }
                        }
                    }
                }
            )

    def _get_embedding(self, text: str) -> list[float]:
        response = self.openai.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding

    def add_documents(self, documents: list[Document]):
        """문서 배치 추가"""
        actions = []
        for doc in documents:
            text = f"{doc.title} {doc.content}"
            embedding = self._get_embedding(text)

            actions.append({
                "_index": self.index_name,
                "_id": doc.id,
                "_source": {
                    "title": doc.title,
                    "content": doc.content,
                    "source": doc.source,
                    "embedding": embedding
                }
            })

        success, _ = helpers.bulk(self.client, actions, refresh=True)
        return success

    def search(
        self,
        query: str,
        k: int = 5,
        source_filter: str = None
    ) -> list[dict]:
        """유사 문서 검색"""
        query_embedding = self._get_embedding(query)

        knn_query = {
            "knn": {
                "embedding": {
                    "vector": query_embedding,
                    "k": k
                }
            }
        }

        if source_filter:
            search_body = {
                "size": k,
                "query": {
                    "bool": {
                        "must": [knn_query],
                        "filter": [{"term": {"source": source_filter}}]
                    }
                }
            }
        else:
            search_body = {"size": k, "query": knn_query}

        response = self.client.search(
            index=self.index_name,
            body=search_body
        )

        return [
            {
                "id": hit["_id"],
                "score": hit["_score"],
                **hit["_source"]
            }
            for hit in response["hits"]["hits"]
        ]

# 사용 예시
store = VectorStore()

# 문서 추가
docs = [
    Document(id="1", title="RAG 소개", content="RAG는 검색 증강 생성...", source="manual"),
    Document(id="2", title="벡터 DB", content="벡터 데이터베이스는...", source="blog"),
]
store.add_documents(docs)

# 검색
results = store.search("검색 증강 생성이란?", k=3)
for r in results:
    print(f"[{r['score']:.3f}] {r['title']}")
```

---

## 참고 자료 (References)

- [OpenSearch k-NN 공식 문서](https://opensearch.org/docs/latest/search-plugins/knn/index/)
- [HNSW 알고리즘 설명](https://opensearch.org/docs/latest/search-plugins/knn/knn-index/#hnsw-algorithm)
- [k-NN 성능 튜닝](https://opensearch.org/docs/latest/search-plugins/knn/performance-tuning/)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)

## 관련 문서

- [OpenSearch 기초](./opensearch-basics.md) - 설치, 기본 개념
- [키워드 검색 (BM25)](./keyword-search-bm25.md) - 전문 검색
- [하이브리드 검색](./hybrid-search.md) - 벡터 + 키워드 결합
- [Milvus 기초](../milvus/milvus-basics.md) - 다른 벡터 DB 비교

---

*Last updated: 2026-02-05*
