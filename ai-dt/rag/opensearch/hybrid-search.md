---
tags: [opensearch, hybrid-search, vector-search, keyword-search, rrf]
level: advanced
last_updated: 2026-02-05
reviewed_on: 2026-10-04
review_status: partial
document_type: learning_note
aliases: [하이브리드 검색과 RRF]
category_major: "AI·DT"
category_middle: "RAG"
category_minor: "OpenSearch 검색"
note_kind: "학습"
classified_on: "2026-10-05"
---

# OpenSearch 하이브리드 검색 (Hybrid Search)

> 벡터 검색(Semantic)과 키워드 검색(BM25)을 결합하여 후보와 순위를 결합하고 검색 품질을 평가하는 방법

> [!info] 목적과 선행 문서 — 확인 2026-10-04
> 벡터/BM25 후보를 정규화 점수 또는 순위로 결합하는 흐름을 배운다. 먼저 [벡터 검색](./vector-search-knn.md)의 metric/embedding 계약과 [BM25](./keyword-search-bm25.md)의 analyzer/score를 읽는다. 공통 검증·적재·VectorStore 정의는 벡터 문서를 대표로 사용하고 이 문서에는 결합 방식만 둔다. 아래 예제는 앞 문서의 정의를 같은 실습 모듈에 준비한 뒤 순서대로 읽는다. 문서 파일을 Python으로 import하지 않는다.
>
> 가정: Nori와 k-NN/neural-search가 제공·활성화된 OpenSearch3.x, opensearch-py3.2.0, 명시적 client/embedding callback. 서버/모델·한국어 순위/normalization 실제 실행은 미확인이다. 기존 index/pipeline을 자동 삭제·덮어쓰거나 모델을 다운로드하지 않는다.

## 왜 필요한가? (Why)

### 각 검색 방식의 장단점

| 상황 | 후보를 결합할 이유·조건 |
|------|--------------------------|
| 의미·표현 변경 | 모델/전처리에 맞는 벡터 후보와 분석 토큰 후보를 비교한다. |
| 용어·SKU/코드 | 정확 일치는 keyword 필드/term 조건이 필요하다. text BM25도 원문 문자열 전체 일치를 보장하지 않는다. |
| 동의어·오타 | 키워드도 synonym/fuzzy 설정이 가능하며 벡터도 항상 해결하지는 않는다. |
| 추가 학습 없이 검색 | BM25/RRF도 검색 자체를 위한 추가 학습 없이 사용할 수 있다. zero-shot 가능 여부로 한 방식을 배제하지 않는다. |
| 하이브리드의 효과 | 두 후보의 상보성이 정답셋에서 입증될 때 채택한다. 항상 품질이 증가하는 것은 아니다. |

### 실제 예시

아래 순위는 설명용 시나리오이며 실측 결과가 아니다. 마지막 코드의 SKU 제목 예시는 별도 keyword SKU mapping/term 질의를 구현하지 않는다.

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

RAG(Retrieval-Augmented Generation) 시스템에서 검색 품질은 근거 후보에 영향을 준다. 전체 응답은 문서 신뢰성·권한/필터·prompt·생성/인용 검증에도 영향을 받는다.

```
사용자 질문
    ↓
┌─────────────┐
│ 검색 단계   │ ← 하이브리드 후보를 정답셋으로 평가
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

두 검색의 **정규화된 점수**를 가중 합산한다. BM25와 k-NN raw score는 척도가 달라 α/β만 넣어 공정한 비율이 되지 않는다. 0.7/0.3은 미평가 예시다. min_max/l2 같은 정규화는 현재 후보 분포와 누락된 후보 처리에 영향을 받으며 동일점수/한건/빈결과를 수동 식만으로 처리하지 않는다.

```
final_score = α × normalized_vector_score + β × normalized_bm25_score

예: α=0.7, β=0.3 (벡터 검색 70%, 키워드 검색 30%)
```

#### 2. RRF (Reciprocal Rank Fusion)

각 검색 결과의 **순위(rank)**를 기반으로 결합. 점수 스케일 차이에 강건함.

```
RRF_score(d) = Σ 1 / (k + rank_i(d))

- k: 순위 완화 상수 (원 논문/예제60); 반환 개수 k와 구분해 코드에서는 rrf_k
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

최종 순위: A, B (동점), C, E (동점), D, F (동점)
동점은 아래 예제에서 (index, id) 순서로 결정한다.
```

RRF는 raw score 크기를 버리지만 후보 창·중복 identity·순위 변화의 영향을 받는다. 목록에 없는 문서는 기여0이다. 원 논문의60은 보편 최적값이라는 뜻이 아니다. [RRF 원 논문](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf), 확인2026-10-04.

#### 3. Re-ranking

초기 검색 후 질의/문서 쌍을 함께 읽는 **Cross-Encoder** 등으로 재순위화한다. 임베딩을 따로 생성하는 bi-encoder와 다르다. 후보에서 빠진 문서는 복구하지 못하며 비용/언어/입력 길이·출력 score 의미를 평가한다. 아래100/top10도 비용을 설명하는 미평가 예시다.

```
1차 검색 (빠름): 후보 100개 추출
    ↓
2차 Re-ranking (정밀): 후보 중 top-10 재정렬
```

### OpenSearch 하이브리드 검색 방법

| 방법 | 결합 위치·조건 |
|------|----------------|
| **Search Pipeline + hybrid** | normalization-processor2.10+, hybrid 질의2.11+. 서버 query/fetch 사이에서 결합한다. native RRF score-ranker는2.19+. 플러그인/활성 상태를 확인한다. |
| **Bool Query** | matching should의 raw score 합산. normalization pipeline을 선택한 것과 같지 않다. 전수 script는 큰 후보 집합에 비용이 든다. |
| **Multi-query + RRF** | 별도 요청 결과를 Python에서 결합. 아래는 순차 요청이며 native processor/분산 후보 수집과 동일 실행 경로가 아니다. |

---

## 어떻게 사용하는가? (How)

### 1. 하이브리드 검색용 인덱스 설정

```python
from math import isfinite, isclose
from collections.abc import Callable
from opensearchpy import OpenSearch, NotFoundError

# 대표 정의는 vector-search-knn.md의 1·2절과 VectorStore/Document이다.
# 같은 Python 실습 모듈에 먼저 복사해 준비한다. Markdown을 import하는 문법이 아니다.
# 필요 정의: Embed, validate_k, get_embedding, checked_hits, vector_index_body,
# new_vector_index, VectorStore, Document. 모델/연결은 caller가 명시한다.

def new_hybrid_index(client: OpenSearch, index_name: str,
                     dimension: int, embedding_contract: str) -> dict:
    body = vector_index_body(dimension, embedding_contract)
    body["settings"]["analysis"] = {"analyzer": {"korean": {
        "type": "custom", "tokenizer": "nori_tokenizer", "filter": ["lowercase"],
    }}}
    for field in ("title", "content"):
        body["mappings"]["properties"][field]["analyzer"] = "korean"
    if client.indices.exists(index=index_name):
        raise ValueError("existing_index_not_modified")
    return client.indices.create(index=index_name, body=body)

def validate_weights(vector_weight: float, keyword_weight: float) -> list[float]:
    values = [vector_weight, keyword_weight]
    if any(type(x) not in (int, float) or not isfinite(x) or not 0 <= x <= 1 for x in values):
        raise ValueError("finite_weights_in_unit_interval_required")
    if not isclose(sum(values), 1., rel_tol=0., abs_tol=1e-9):
        raise ValueError("weights_must_sum_to_one")
    return values

def search_clauses(query: str, embed: Embed, dimension: int,
                   candidate_k: int) -> list[dict]:
    validate_k(candidate_k)
    vector = get_embedding(query, embed, dimension)
    return [
        {"knn": {"embedding": {"vector": vector, "k": candidate_k}}},
        {"multi_match": {"query": query, "fields": ["title^2", "content"]}},
    ]

def format_hits(hits: list[dict], score_kind: str) -> list[dict]:
    return [{"index": hit["_index"], "id": hit["_id"], "score": hit["_score"],
             "score_kind": score_kind, "title": hit["_source"]["title"],
             "content": hit["_source"]["content"]} for hit in hits]
```

### 2. 방법 1: Search Pipeline (normalization 2.10+, hybrid 2.11+)

서버 processor의 지원/권한을 확인한 독립 실습 공간에서만 새 이름을 생성한다. 기존 고정 pipeline의 PUT는 업데이트가 될 수 있다. UUID 이름과 GET 검사는 같은 이름을 피하기 위한 실습 조치이며 GET→PUT가 원자적 create-only인 것은 아니다. 동시 관리자가 없는 전제가 필요하다. 사용한 이름과 정리 대상 자원을 caller가 기록한다. pipeline의 weights 개수/순서는 hybrid.queries와 같아야 하고 합은1이다. 아래는 [vector,keyword] 순서다. 실제 processor의 동점/누락/후보 집계는 서버에서 확인한다.

```python
from uuid import uuid4

def normalization_pipeline_body(vector_weight: float = 0.7,
                                keyword_weight: float = 0.3) -> dict:
    weights = validate_weights(vector_weight, keyword_weight)
    return {"description": "Explicit two-clause normalization demo",
        "phase_results_processors": [{"normalization-processor": {
            "normalization": {"technique": "min_max"},
            "combination": {"technique": "arithmetic_mean",
                            "parameters": {"weights": weights}},
        }}]}

def create_demo_pipeline(client: OpenSearch, vector_weight: float = 0.7,
                         keyword_weight: float = 0.3) -> str:
    body = normalization_pipeline_body(vector_weight, keyword_weight)
    name = "hybrid-demo-" + uuid4().hex  # 기존 고정 이름을 덮어쓰지 않는다.
    try:
        client.search_pipeline.get(id=name)
    except NotFoundError:
        pass
    else:
        raise ValueError("existing_pipeline_not_modified")
    client.search_pipeline.put(id=name, body=body)
    return name

# 독립 실습 공간에서만 caller가 실행한다. 종료 후 생긴 이름/자원을 별도 관리한다.
# GET→PUT는 원자적 create-only가 아니다. 동시 관리자를 차단한 실습 조건이다.
```

```python
def hybrid_search_pipeline(client: OpenSearch, index_name: str, embed: Embed,
                           dimension: int, pipeline_name: str,
                           query: str, k: int = 10) -> list[dict]:
    validate_k(k)
    if not isinstance(pipeline_name, str) or not pipeline_name:
        raise ValueError("prepared_pipeline_required")
    body = {"query": {"hybrid": {"queries": search_clauses(query, embed, dimension, k)}},
            "size": k, "_source": ["title", "content"]}
    response = client.search(index=index_name, body=body,
                             params={"search_pipeline": pipeline_name})
    return format_hits(checked_hits(response), "pipeline_normalized")

# query 예시: "머신러닝 학습 방법". 준비된 pipeline의 가중치 순서는 [vector, keyword].
# 이 함수는 서버 processor를 재현하지 않는다. k/size 후보 범위도 품질 평가 대상이다.
```

### 3. 방법 2: Bool Query 결합

정규화 processor가 없는 raw 점수 결합 실험이다. 원래 Painless 식 `w*cos + 1`은 `w*(cos+1)`과 다르고 문자열에 가중치를 삽입했다. 아래는 k-NN의 knn_score에 boost를 적용해 전체 vector 점수를 가중한다. BM25와 척도를 맞췄다는 뜻은 아니다. vector clause는 embedding이 있는 모든 문서를 전수 점수화하므로 ANN 후보 결합과 비용이 다르다. 이 방식을 운영 기본으로 채택하지 않았다.

```python
def hybrid_search_bool(client: OpenSearch, index_name: str, embed: Embed,
                       dimension: int, query: str, k: int = 10,
                       vector_weight: float = 0.7,
                       keyword_weight: float = 0.3) -> list[dict]:
    """전수 cosine script와 BM25의 raw weighted sum; 정규화된 결합이 아니다."""
    validate_k(k)
    validate_weights(vector_weight, keyword_weight)
    vector = get_embedding(query, embed, dimension)
    clauses = []
    if vector_weight > 0:
        clauses.append({"script_score": {
            "query": {"exists": {"field": "embedding"}},
            "boost": vector_weight,
            "script": {"lang": "knn", "source": "knn_score", "params": {
                "field": "embedding", "query_value": vector, "space_type": "cosinesimil",
            }},
        }})
    if keyword_weight > 0:
        clauses.append({"multi_match": {"query": query,
            "fields": ["title^2", "content"], "boost": keyword_weight}})
    response = client.search(index=index_name, body={"size": k,
        "query": {"bool": {"should": clauses, "minimum_should_match": 1}},
        "_source": ["title", "content"],
    })
    return format_hits(checked_hits(response), "raw_weighted_sum")
```

### 4. 방법 3: RRF 직접 구현

순위 합산 규칙을 학습하는 Python 구현이다. 대표 (index,id) identity로 결합하고 한 목록의 중복을 거부한다. 두 순차 요청 사이에 원문이 바뀌면 조용히 다른 내용을 선택하지 않고 오류로 표시한다. PIT/동시 writer의 전체 일관성을 구현한 것은 아니다. candidate_k는 비용·recall 조건이며 기본2*k도 보편 최적값이 아니다. 이 실습의 공통 validate_k는 1~100을 허용한다. 기본 후보 2*k를 사용하면 출력 k는 50 이하여야 한다. k가 더 크면 k≤candidate_k≤100을 명시한다. rerank의 initial_k도 기본 후보 2*initial_k 때문에 50 이하로 준비한다.

```python
from collections import defaultdict

def fuse_rrf(rankings: list[list[dict]], k: int = 10, rrf_k: int = 60) -> list[dict]:
    validate_k(k)
    if type(rrf_k) is not int or rrf_k < 1:
        raise ValueError("positive_rrf_constant_required")
    scores, sources = defaultdict(float), {}
    for hits in rankings:
        seen = set()
        for rank, hit in enumerate(hits, start=1):
            identity = (hit["_index"], hit["_id"])
            if identity in seen:
                raise ValueError("duplicate_document_in_one_ranking")
            seen.add(identity)
            source = {"title": hit["_source"]["title"],
                      "content": hit["_source"]["content"]}
            if identity in sources and sources[identity] != source:
                raise RuntimeError("source_changed_between_rankings")
            sources[identity] = source
            scores[identity] += 1 / (rrf_k + rank)
    ordered = sorted(scores, key=lambda identity: (-scores[identity], identity))[:k]
    return [{"index": identity[0], "id": identity[1], "rrf_score": scores[identity],
             "score_kind": "rrf", "title": sources[identity]["title"],
             "content": sources[identity]["content"]} for identity in ordered]

def hybrid_search_rrf(client: OpenSearch, index_name: str, embed: Embed,
                      dimension: int, query: str, k: int = 10,
                      rrf_k: int = 60, candidate_k: int | None = None) -> list[dict]:
    validate_k(k)
    if type(rrf_k) is not int or rrf_k < 1:
        raise ValueError("positive_rrf_constant_required")
    if candidate_k is None:
        candidate_k = k * 2
    validate_k(candidate_k)
    if candidate_k < k:
        raise ValueError("candidate_k_must_cover_output_k")
    rankings = []
    # 동기 search 두 번을 순차 실행한다. 병렬/PIT 일관성을 구현한 코드가 아니다.
    for clause in search_clauses(query, embed, dimension, candidate_k):
        response = client.search(index=index_name, body={"size": candidate_k,
            "query": clause, "_source": ["title", "content"]})
        rankings.append(checked_hits(response))
    return fuse_rrf(rankings, k, rrf_k)

# 예: query="인공지능 학습". 더 큰 후보 창은 비용과 recall을 함께 평가한다.
```

### 5. Re-ranking 추가 (선택사항)

Cross-Encoder로 최종 재순위화할 때 caller가 model.predict callback을 전달한다. 원래 모델 ID의 철자를 공식 카드와 맞췄다. 확인한 cross-encoder/ms-marco-MiniLM-L6-v2는 English/MS MARCO 기반이므로 한국어 질의 품질을 보장하지 않는다. 모델 다운로드/추론은 실행하지 않았다. 빈 후보는 호출을 생략하고 점수 개수·finite값을 확인하며 원래 후보 dict를 수정하지 않는다.

```python
Rerank = Callable[[list[tuple[str, str]]], list[float]]

def hybrid_search_with_rerank(client: OpenSearch, index_name: str, embed: Embed,
                             dimension: int, rerank: Rerank, query: str,
                             initial_k: int = 20, final_k: int = 5) -> list[dict]:
    validate_k(initial_k)
    validate_k(final_k)
    if final_k > initial_k:
        raise ValueError("final_k_exceeds_candidate_k")
    candidates = hybrid_search_rrf(client, index_name, embed, dimension,
                                   query, k=initial_k)
    if not candidates:
        return []
    pairs = [(query, f"{c['title']} {c['content']}") for c in candidates]
    scores = list(rerank(pairs))
    if len(scores) != len(candidates):
        raise ValueError("reranker_score_count_mismatch")
    values = [float(score) for score in scores]
    if any(isinstance(score, (bool, str, bytes)) for score in scores) or not all(isfinite(x) for x in values):
        raise ValueError("finite_reranker_scores_required")
    reranked = [{**candidate, "rerank_score": score, "score_kind": "reranked"}
                for candidate, score in zip(candidates, values)]
    return sorted(reranked, key=lambda c: (-c["rerank_score"], c["index"], c["id"]))[:final_k]

# 원래 모델 ID cross-encoder/ms-marco-MiniLM-L-6-v2는 확인한 ID와 철자가 달랐다.
# 확인한 ID: cross-encoder/ms-marco-MiniLM-L6-v2. English/MS MARCO 모델이다.
# caller가 별도 CrossEncoder를 준비하면: rerank=lambda pairs: model.predict(pairs)
# 자동 load/download 없음. 질의 예: "딥러닝 신경망 구조".
```

---

## 실전 예제: RAG용 하이브리드 검색 시스템

앞 벡터 문서의 VectorStore를 대표 적재/모델 계약으로 조합한다. 기존 class의 암묵적 생성/ada 고정·자동 ID 재적재·미확인 source 기본값을 제거했다. 준비된 Nori hybrid index의 source/title/content와 embedding 계약을 caller가 확인한다. search의 mode를 모르면 오류이며 hybrid는 RRF이다. 원래 사용되지 않던 vector_weight 인자를 제거했다. 가중 점수 비교가 필요하면 pipeline/bool의 명시 인자를 사용한다. 세 mode의 score_kind가 달라 숫자를 같은 품질 척도로 비교하지 않는다.

```python
from dataclasses import dataclass

@dataclass
class SearchResult:
    index: str
    id: str
    score: float
    title: str
    content: str
    search_type: str  # "vector", "keyword", "hybrid"
    score_kind: str  # mode별 score는 공통 척도가 아니다.

class HybridSearchEngine:
    """벡터 문서의 대표 VectorStore 적재/계약 검사를 조합한다."""

    def __init__(self, store: VectorStore):
        self.store = store  # 무통신. client 소유권은 caller에게 있다.

    def validate_index(self) -> None:
        self.store.validate_index()
        # validate_index 대표 검사는 dimension/engine/metric/embedding_contract다.
        # Nori 설치와 title/content의 korean mapping은 caller가 별도 확인한다.

    def add_documents(self, documents: list[dict]) -> dict[str, int]:
        # source가 없는 문서를 확인되지 않은 사실로 default하지 않는다.
        docs = [Document(id=doc["id"], title=doc["title"], content=doc["content"],
                         source=doc["source"], embedding=doc.get("embedding"))
                for doc in documents]
        return self.store.add_documents(docs)

    def search(self, query: str, k: int = 5, mode: str = "hybrid",
               rrf_k: int = 60, candidate_k: int | None = None) -> list[SearchResult]:
        if mode not in ("vector", "keyword", "hybrid"):
            raise ValueError("unknown_search_mode")
        self.store._require_mapping() # 같은 실습 모듈의 대표 mapping gate
        validate_k(k)
        if not isinstance(query, str) or not query.strip():
            raise ValueError("nonempty_query_required")
        client, index = self.store.client, self.store.index_name
        embed, dimension = self.store.embed, self.store.embedding_dim
        if mode == "hybrid":
            hits = hybrid_search_rrf(client, index, embed, dimension, query,
                                     k, rrf_k, candidate_k)
            return [SearchResult(index=h["index"], id=h["id"], score=h["rrf_score"],
                title=h["title"], content=h["content"], search_type=mode,
                score_kind="rrf") for h in hits]
        if mode == "vector":
            clause = search_clauses(query, embed, dimension, k)[0]
        else:
            clause = {"multi_match": {"query": query, "fields": ["title^2", "content"]}}
        response = client.search(index=index, body={"size": k, "query": clause,
                                                    "_source": ["title", "content"]})
        return [SearchResult(index=h["_index"], id=h["_id"], score=h["_score"],
            title=h["_source"]["title"], content=h["_source"]["content"],
            search_type=mode, score_kind="vector_raw" if mode == "vector" else "bm25_raw")
            for h in checked_hits(response)]

def hybrid_demo(store: VectorStore) -> tuple[dict, dict[str, list[SearchResult]]]:
    engine = HybridSearchEngine(store)
    engine.validate_index()
    outcome = engine.add_documents([
        {"id": "rag-architecture", "title": "RAG 아키텍처", "content": "RAG는 검색과 생성을 결합...", "source": "blog"},
        {"id": "vector-database", "title": "벡터 데이터베이스", "content": "벡터 DB는 임베딩을 저장...", "source": "docs"},
        {"id": "sku-shipping", "title": "SKU-12345 상품 안내", "content": "해당 상품은 배송 지연...", "source": "product"},
    ])
    if outcome["failed"]:
        raise RuntimeError("demo_indexing_incomplete")
    query = "검색 증강 생성"
    return outcome, {mode: engine.search(query, mode=mode)
                     for mode in ("vector", "keyword", "hybrid")}
```

---

## 파라미터 튜닝 가이드

### 가중치 설정 가이드라인

아래 표는 원래 미평가 후보 설정을 보존한 것이며 현재 데이터의 권장/최적값이 아니다. 정규화 방식·분석기·embedding 계약·후보 창을 고정하고 NDCG/recall과 응답 근거를 함께 평가한다. RRF 예제에는 이 score weights를 적용하지 않는다.

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
| 60 | 원 논문/예제의 상수; 데이터별 최적값은 미확인 |
| 100+ | 순위 차이 영향 감소 |

---

## 참고 자료 (References)

확인일 **2026-10-04**. rolling 문서와 대상 서버의 고정 판본/실행 증거를 구분한다. 로컬SDK검증판본은opensearch-py **3.2.0**이다.

- [Hybrid search](https://docs.opensearch.org/latest/vector-search/ai-search/hybrid-search/index/)·[Normalization processor](https://docs.opensearch.org/latest/search-plugins/search-pipelines/normalization-processor/)·[Hybrid query](https://docs.opensearch.org/latest/query-dsl/compound/hybrid/) — 도입/플러그인·후보/정규화조건.
- [Search pipelines](https://docs.opensearch.org/latest/search-plugins/search-pipelines/index/)·[기존 pipeline 조회](https://docs.opensearch.org/latest/search-plugins/search-pipelines/retrieving-search-pipeline/) — 새 이름/업데이트구분.
- [native RRF](https://docs.opensearch.org/latest/vector-search/ai-search/hybrid-search/rrf/)·[RRF 원 논문](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — 순위기반공식/상수조건.
- [Script score](https://docs.opensearch.org/latest/query-dsl/specialized/script-score/) — boost와script계산범위.
- [Cross-Encoder 공식 안내](https://www.sbert.net/docs/cross_encoder/pretrained_models.html)·[확인한 모델 카드](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) — 모델ID/언어와reranker출력조건.

> [!todo] 남은 확인
> 대상 서버의 pipeline/mapping·권한·normalization/native RRF 분산 실행·부분 실패·동시쓰기·모델 추론/한국어 품질·후보/가중치 최적값은 미확인이다. 모의 HTTP와 순위/콜백 테스트는 실제 검색 품질을 증명하지 않는다. Claude 연결 불가로 운영 방식 선택·추가 문서 통합 판단은 보류했다.

## 관련 문서

- [벡터 검색 (k-NN)](./vector-search-knn.md) - 의미 기반 검색
- [키워드 검색 (BM25)](./keyword-search-bm25.md) - 전문 검색
- [OpenSearch 기초](./opensearch-basics.md) - 설치, 기본 개념
- [LangGraph RAG](../langgraph/langgraph-rag.md) - RAG 파이프라인 연동

---

*Last updated: 2026-02-05*
