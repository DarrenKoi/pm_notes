---
tags: [opensearch, hybrid-search, vector-search, keyword-search, rrf]
level: advanced
last_updated: 2026-02-05
status: in-progress
---

# OpenSearch 하이브리드 검색 (Hybrid Search)

> 벡터 검색(Semantic)과 키워드 검색(BM25)을 결합하여 검색 품질을 극대화하는 방법

## 왜 필요한가? (Why)

### 각 검색 방식의 장단점

| 상황 | 벡터 검색 | 키워드 검색 | 하이브리드 |
|------|----------|------------|-----------|
| 의미적 유사성 | ✅ 강함 | ❌ 약함 | ✅ 강함 |
| 정확한 용어 매칭 | ❌ 약함 | ✅ 강함 | ✅ 강함 |
| 고유명사/코드 검색 | ❌ 약함 | ✅ 강함 | ✅ 강함 |
| 동의어/문맥 이해 | ✅ 강함 | ❌ 약함 | ✅ 강함 |
| Zero-shot 검색 | ✅ 가능 | ❌ 불가 | ✅ 가능 |

### 실제 예시

```
쿼리: "SKU-12345 배송 지연"

벡터 검색만:
  1. "물류 배송 문제 해결 가이드"  (의미는 맞지만 SKU 못 찾음)
  2. "택배 지연 사유 안내"

키워드 검색만:
  1. "SKU-12345 재고 현황"  (SKU 매칭, 배송과 무관)
  2. "SKU-12345 주문 정보"

하이브리드 검색:
  1. "SKU-12345 배송 지연 안내" ✅ (정확한 SKU + 배송 지연 의미)
  2. "SKU-12345 물류 처리 현황"
```

### RAG에서의 중요성

RAG(Retrieval-Augmented Generation) 시스템에서 검색 품질이 전체 응답 품질을 결정한다.

```
사용자 질문
    ↓
┌─────────────┐
│ 검색 단계   │ ← 하이브리드 검색으로 품질 ↑
└─────────────┘
    ↓
관련 문서 (Context)
    ↓
┌─────────────┐
│ LLM 생성    │
└─────────────┘
    ↓
최종 응답
```

---

## 핵심 개념 (What)

### 하이브리드 검색 전략

#### 1. Score Combination (점수 결합)

두 검색의 점수를 가중 합산.

```
final_score = α × vector_score + β × bm25_score

예: α=0.7, β=0.3 (벡터 검색 70%, 키워드 검색 30%)
```

#### 2. RRF (Reciprocal Rank Fusion)

각 검색 결과의 **순위(rank)**를 기반으로 결합. 점수 스케일 차이에 강건함.

```
RRF_score(d) = Σ 1 / (k + rank_i(d))

- k: 상수 (보통 60)
- rank_i(d): i번째 검색에서 문서 d의 순위
```

**예시**:
```
벡터 검색 순위: [A(1), B(2), C(3), D(4)]
키워드 검색 순위: [B(1), A(2), E(3), F(4)]

k=60 일 때:
RRF(A) = 1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325
RRF(B) = 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325
RRF(C) = 1/(60+3) + 0 = 0.0159
RRF(E) = 0 + 1/(60+3) = 0.0159

최종 순위: A, B (동점), C, E, D, F
```

#### 3. Re-ranking

초기 검색 후 **Cross-Encoder** 등으로 재순위화.

```
1차 검색 (빠름): 후보 100개 추출
    ↓
2차 Re-ranking (정밀): 후보 중 top-10 재정렬
```

### OpenSearch 하이브리드 검색 방법

| 방법 | 설명 | 장점 | 단점 |
|------|------|------|------|
| **Search Pipeline** | 내장 기능, 자동 결합 | 간단, 성능 좋음 | OpenSearch 2.10+ |
| **Bool Query** | should로 결합 | 유연함 | 점수 정규화 직접 |
| **Multi-query** | 별도 쿼리 후 병합 | 완전 제어 | 구현 복잡 |

---

## 어떻게 사용하는가? (How)

### 1. 하이브리드 검색용 인덱스 설정

```python
from opensearchpy import OpenSearch

client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    use_ssl=False
)

index_name = "hybrid-search"

index_body = {
    "settings": {
        "index": {
            "knn": True,
            "knn.algo_param.ef_search": 256
        },
        "analysis": {
            "analyzer": {
                "korean": {
                    "type": "custom",
                    "tokenizer": "nori_tokenizer",
                    "filter": ["lowercase"]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "title": {
                "type": "text",
                "analyzer": "korean"
            },
            "content": {
                "type": "text",
                "analyzer": "korean"
            },
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

if client.indices.exists(index=index_name):
    client.indices.delete(index=index_name)

client.indices.create(index=index_name, body=index_body)
```

### 2. 방법 1: Search Pipeline (권장, 2.10+)

OpenSearch의 내장 하이브리드 검색 기능.

```python
# Search Pipeline 생성
pipeline_name = "hybrid-pipeline"

pipeline_body = {
    "description": "Hybrid search pipeline with normalization",
    "phase_results_processors": [
        {
            "normalization-processor": {
                "normalization": {
                    "technique": "min_max"  # 또는 "l2"
                },
                "combination": {
                    "technique": "arithmetic_mean",
                    "parameters": {
                        "weights": [0.7, 0.3]  # [vector, keyword] 가중치
                    }
                }
            }
        }
    ]
}

client.transport.perform_request(
    "PUT",
    f"/_search/pipeline/{pipeline_name}",
    body=pipeline_body
)
print(f"Pipeline '{pipeline_name}' created")
```

```python
from openai import OpenAI

openai_client = OpenAI()

def get_embedding(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def hybrid_search_pipeline(query: str, k: int = 10) -> list[dict]:
    """Search Pipeline을 사용한 하이브리드 검색"""

    query_embedding = get_embedding(query)

    search_body = {
        "query": {
            "hybrid": {
                "queries": [
                    # 벡터 검색
                    {
                        "knn": {
                            "embedding": {
                                "vector": query_embedding,
                                "k": k
                            }
                        }
                    },
                    # 키워드 검색
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["title^2", "content"]
                        }
                    }
                ]
            }
        },
        "size": k,
        "_source": ["title", "content"]
    }

    response = client.search(
        index=index_name,
        body=search_body,
        params={"search_pipeline": pipeline_name}
    )

    return [
        {
            "id": hit["_id"],
            "score": hit["_score"],
            "title": hit["_source"]["title"],
            "content": hit["_source"]["content"]
        }
        for hit in response["hits"]["hits"]
    ]

# 사용
results = hybrid_search_pipeline("머신러닝 학습 방법", k=5)
for r in results:
    print(f"[{r['score']:.4f}] {r['title']}")
```

### 3. 방법 2: Bool Query 결합

Search Pipeline 없이 직접 구현.

```python
def hybrid_search_bool(
    query: str,
    k: int = 10,
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3
) -> list[dict]:
    """Bool query로 하이브리드 검색 (점수 직접 결합)"""

    query_embedding = get_embedding(query)

    search_body = {
        "query": {
            "bool": {
                "should": [
                    # 벡터 검색 (boost로 가중치)
                    {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": f"""
                                    {vector_weight} * cosineSimilarity(params.query_vector, 'embedding') + 1.0
                                """,
                                "params": {
                                    "query_vector": query_embedding
                                }
                            }
                        }
                    },
                    # 키워드 검색
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["title^2", "content"],
                            "boost": keyword_weight
                        }
                    }
                ]
            }
        },
        "size": k
    }

    response = client.search(index=index_name, body=search_body)
    return response["hits"]["hits"]
```

### 4. 방법 3: RRF 직접 구현

가장 유연하지만 복잡함.

```python
from collections import defaultdict

def hybrid_search_rrf(
    query: str,
    k: int = 10,
    rrf_k: int = 60
) -> list[dict]:
    """RRF (Reciprocal Rank Fusion) 하이브리드 검색"""

    query_embedding = get_embedding(query)

    # 1. 벡터 검색
    vector_results = client.search(
        index=index_name,
        body={
            "size": k * 2,  # 더 많이 가져와서 병합
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding,
                        "k": k * 2
                    }
                }
            },
            "_source": ["title", "content"]
        }
    )

    # 2. 키워드 검색
    keyword_results = client.search(
        index=index_name,
        body={
            "size": k * 2,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title^2", "content"]
                }
            },
            "_source": ["title", "content"]
        }
    )

    # 3. RRF 점수 계산
    rrf_scores = defaultdict(float)
    doc_sources = {}

    # 벡터 검색 결과 처리
    for rank, hit in enumerate(vector_results["hits"]["hits"], start=1):
        doc_id = hit["_id"]
        rrf_scores[doc_id] += 1 / (rrf_k + rank)
        doc_sources[doc_id] = hit["_source"]

    # 키워드 검색 결과 처리
    for rank, hit in enumerate(keyword_results["hits"]["hits"], start=1):
        doc_id = hit["_id"]
        rrf_scores[doc_id] += 1 / (rrf_k + rank)
        doc_sources[doc_id] = hit["_source"]

    # 4. 점수 기준 정렬
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # 5. 결과 포맷팅
    results = []
    for doc_id, score in sorted_docs[:k]:
        results.append({
            "id": doc_id,
            "rrf_score": score,
            **doc_sources[doc_id]
        })

    return results

# 사용
results = hybrid_search_rrf("인공지능 학습", k=5)
for r in results:
    print(f"[RRF: {r['rrf_score']:.4f}] {r['title']}")
```

### 5. Re-ranking 추가 (선택사항)

Cross-Encoder로 최종 재순위화.

```python
from sentence_transformers import CrossEncoder

# Cross-Encoder 모델 로드 (한 번만)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def hybrid_search_with_rerank(
    query: str,
    initial_k: int = 20,
    final_k: int = 5
) -> list[dict]:
    """하이브리드 검색 + Re-ranking"""

    # 1. 하이브리드 검색으로 후보 추출
    candidates = hybrid_search_rrf(query, k=initial_k)

    # 2. Cross-Encoder로 재점수화
    pairs = [
        (query, f"{c['title']} {c['content']}")
        for c in candidates
    ]
    rerank_scores = reranker.predict(pairs)

    # 3. 재점수 기준 정렬
    for i, candidate in enumerate(candidates):
        candidate["rerank_score"] = float(rerank_scores[i])

    reranked = sorted(
        candidates,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return reranked[:final_k]

# 사용
results = hybrid_search_with_rerank("딥러닝 신경망 구조", initial_k=20, final_k=5)
for r in results:
    print(f"[Rerank: {r['rerank_score']:.4f}] {r['title']}")
```

---

## 실전 예제: RAG용 하이브리드 검색 시스템

```python
from dataclasses import dataclass
from typing import Optional
from collections import defaultdict
from opensearchpy import OpenSearch, helpers
from openai import OpenAI

@dataclass
class SearchResult:
    id: str
    score: float
    title: str
    content: str
    search_type: str  # "vector", "keyword", "hybrid"

class HybridSearchEngine:
    """RAG용 하이브리드 검색 엔진"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9200,
        index_name: str = "rag-hybrid",
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
        if not self.client.indices.exists(index=self.index_name):
            self.client.indices.create(
                index=self.index_name,
                body={
                    "settings": {
                        "index": {"knn": True},
                        "analysis": {
                            "analyzer": {
                                "korean": {
                                    "type": "custom",
                                    "tokenizer": "nori_tokenizer",
                                    "filter": ["lowercase"]
                                }
                            }
                        }
                    },
                    "mappings": {
                        "properties": {
                            "title": {"type": "text", "analyzer": "korean"},
                            "content": {"type": "text", "analyzer": "korean"},
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

    def add_documents(self, documents: list[dict]) -> int:
        """문서 추가 (임베딩 자동 생성)"""
        actions = []
        for doc in documents:
            text = f"{doc['title']} {doc['content']}"
            embedding = self._get_embedding(text)

            actions.append({
                "_index": self.index_name,
                "_source": {
                    "title": doc["title"],
                    "content": doc["content"],
                    "source": doc.get("source", "unknown"),
                    "embedding": embedding
                }
            })

        success, _ = helpers.bulk(self.client, actions, refresh=True)
        return success

    def search(
        self,
        query: str,
        k: int = 5,
        mode: str = "hybrid",  # "vector", "keyword", "hybrid"
        vector_weight: float = 0.7,
        rrf_k: int = 60
    ) -> list[SearchResult]:
        """검색 실행"""

        if mode == "vector":
            return self._vector_search(query, k)
        elif mode == "keyword":
            return self._keyword_search(query, k)
        else:
            return self._hybrid_search_rrf(query, k, rrf_k)

    def _vector_search(self, query: str, k: int) -> list[SearchResult]:
        query_embedding = self._get_embedding(query)

        response = self.client.search(
            index=self.index_name,
            body={
                "size": k,
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": query_embedding,
                            "k": k
                        }
                    }
                }
            }
        )

        return [
            SearchResult(
                id=hit["_id"],
                score=hit["_score"],
                title=hit["_source"]["title"],
                content=hit["_source"]["content"],
                search_type="vector"
            )
            for hit in response["hits"]["hits"]
        ]

    def _keyword_search(self, query: str, k: int) -> list[SearchResult]:
        response = self.client.search(
            index=self.index_name,
            body={
                "size": k,
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["title^2", "content"]
                    }
                }
            }
        )

        return [
            SearchResult(
                id=hit["_id"],
                score=hit["_score"],
                title=hit["_source"]["title"],
                content=hit["_source"]["content"],
                search_type="keyword"
            )
            for hit in response["hits"]["hits"]
        ]

    def _hybrid_search_rrf(
        self,
        query: str,
        k: int,
        rrf_k: int
    ) -> list[SearchResult]:
        """RRF 기반 하이브리드 검색"""

        # 두 검색 병렬 실행
        vector_results = self._vector_search(query, k * 2)
        keyword_results = self._keyword_search(query, k * 2)

        # RRF 점수 계산
        rrf_scores = defaultdict(float)
        doc_data = {}

        for rank, result in enumerate(vector_results, start=1):
            rrf_scores[result.id] += 1 / (rrf_k + rank)
            doc_data[result.id] = result

        for rank, result in enumerate(keyword_results, start=1):
            rrf_scores[result.id] += 1 / (rrf_k + rank)
            if result.id not in doc_data:
                doc_data[result.id] = result

        # 정렬 및 반환
        sorted_ids = sorted(
            rrf_scores.keys(),
            key=lambda x: rrf_scores[x],
            reverse=True
        )[:k]

        return [
            SearchResult(
                id=doc_id,
                score=rrf_scores[doc_id],
                title=doc_data[doc_id].title,
                content=doc_data[doc_id].content,
                search_type="hybrid"
            )
            for doc_id in sorted_ids
        ]


# 사용 예시
if __name__ == "__main__":
    engine = HybridSearchEngine()

    # 문서 추가
    docs = [
        {"title": "RAG 아키텍처", "content": "RAG는 검색과 생성을 결합...", "source": "blog"},
        {"title": "벡터 데이터베이스", "content": "벡터 DB는 임베딩을 저장...", "source": "docs"},
        {"title": "SKU-12345 상품 안내", "content": "해당 상품은 배송 지연...", "source": "product"},
    ]
    engine.add_documents(docs)

    # 검색 비교
    query = "검색 증강 생성"

    print("=== Vector Search ===")
    for r in engine.search(query, mode="vector"):
        print(f"[{r.score:.4f}] {r.title}")

    print("\n=== Keyword Search ===")
    for r in engine.search(query, mode="keyword"):
        print(f"[{r.score:.4f}] {r.title}")

    print("\n=== Hybrid Search (RRF) ===")
    for r in engine.search(query, mode="hybrid"):
        print(f"[{r.score:.4f}] {r.title}")
```

---

## 파라미터 튜닝 가이드

### 가중치 설정 가이드라인

| 사용 케이스 | 벡터 가중치 | 키워드 가중치 | 이유 |
|------------|------------|--------------|------|
| 일반 문서 검색 | 0.7 | 0.3 | 의미 중심 |
| 기술 문서/코드 | 0.5 | 0.5 | 정확한 용어 중요 |
| 제품 검색 (SKU) | 0.3 | 0.7 | 고유명사 매칭 중요 |
| FAQ 검색 | 0.8 | 0.2 | 질문 의미 파악 중요 |

### RRF k 값 선택

| k 값 | 효과 |
|------|------|
| 1 | 상위 결과에 극도로 높은 가중치 |
| 60 | 기본값, 균형 잡힌 결합 |
| 100+ | 순위 차이 영향 감소 |

---

## 참고 자료 (References)

- [OpenSearch Hybrid Search](https://opensearch.org/docs/latest/search-plugins/neural-search/hybrid-search/)
- [Search Pipelines](https://opensearch.org/docs/latest/search-plugins/search-pipelines/index/)
- [RRF 논문](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [MS MARCO Cross-Encoders](https://www.sbert.net/docs/pretrained-models/ce-msmarco.html)

## 관련 문서

- [벡터 검색 (k-NN)](./vector-search-knn.md) - 의미 기반 검색
- [키워드 검색 (BM25)](./keyword-search-bm25.md) - 전문 검색
- [OpenSearch 기초](./opensearch-basics.md) - 설치, 기본 개념
- [LangGraph RAG](../langgraph/langgraph-rag.md) - RAG 파이프라인 연동

---

*Last updated: 2026-02-05*
