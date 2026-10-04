---
tags: [opensearch, vector-search, knn, embedding, semantic-search]
level: intermediate
last_updated: 2026-02-05
reviewed_on: 2026-10-04
review_status: partial
document_type: learning_note
aliases: [OpenSearch 의미 검색]
---

# OpenSearch 벡터 검색 (k-NN Vector Search)

> OpenSearch의 k-NN 플러그인을 활용한 의미 기반 유사도 검색(Semantic Search) 구현

> [!info] 목적·실행 조건 — 확인 2026-10-04
> 같은 임베딩 모델/전처리로 문서와 질의를 벡터화하고, 선택한 거리 공간에서 후보를 찾는 흐름을 배운다. 아래 ANN 실습은 k-NN이 제공되는 OpenSearch **3.x**, Faiss/HNSW/cosinesimil의 새 인덱스를 가정한다. Faiss cosine 지원은 공식 문서상 **2.19 이상**이다. 준비된 client/embedding callback만 인자로 사용하고 caller가 연결을 닫는다. 기존 index 삭제·모델 자동 다운로드/API 요청·force merge를 실행하지 않는다. 실제 서버·임베딩 품질·ANN recall은 미확인이다.

## 왜 필요한가? (Why)

### 키워드 검색의 한계

키워드 전문 검색은 **분석된 토큰**을 찾는다. 동의어·형태소·fuzzy 설정에 따라 다른 표현도 찾을 수 있다. 아래 실패는 동의어 확장이 없는 개념 예시다.

```
쿼리: "자동차 수리"
문서: "차량 정비 방법" → 매칭 실패 ❌ (같은 의미지만 다른 단어)
```

### 벡터 검색의 장점

텍스트를 **고차원 벡터(Embedding)**로 변환하면 **의미적 유사성**을 계산할 수 있다.

```
"자동차 수리" → [0.23, 0.87, -0.12, ...]
"차량 정비"   → [0.25, 0.85, -0.10, ...]  → 의미가 가까울 수 있음 (숫자/벡터는 설명용, 실제 모델 출력 아님)
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

표준 OpenSearch 배포는 k-NN 플러그인을 포함하지만 설치 판본·커스텀 배포·관리형 서비스의 제공/활성 상태를 확인해야 한다. 별도 Lucene/Faiss 라이브러리를 Python에서 설치하는 것과 서버 플러그인은 다르다.

#### 지원하는 엔진 (Engine)

| 엔진 | 특징·조건 |
|------|-------------|
| **nmslib** | 기존 HNSW 엔진. 3.0부터 deprecated이며 신규 기본 권장으로 두지 않는다. 삭제된 기능이라는 뜻은 아니다. |
| **faiss** | HNSW/IVF 및 인코딩 선택. cosine은 2.19 이상; 효율적 필터는 HNSW2.9+/IVF2.10+. Faiss 원본 라이브러리의 GPU 기능을 이 서버의 GPU 검색 지원으로 단정하지 않는다. |
| **lucene** | Lucene 기반 HNSW; 효율적 필터 2.4+. innerproduct는 2.13+. |

실제 engine 선택은 메모리/필터/적재/정답셋 평가가 필요하다. 현재 rolling 자료에는 추가 plugin 기반 JVector도 있지만 원래 3엔진 맥락을 보존하고 도입 판단은 보류했다.

#### 유사도 공간 타입 (Space Type)

| Space Type | 의미 | 점수 해석 조건 |
|------------|------|----------------|
| `l2` | 제곱 유클리드 거리 `d = Σ(x-y)²` | `1/(1+d)`. 원래 `l2²`에서 l2를 다시 제곱한 것으로 혼동하지 않는다. |
| `cosinesimil` | `d=1-cos`; 영벡터 불가 | 3.0/rolling Spaces 표의 `(2-d)/2=(1+cos)/2`. 이전 자료의 NMSLIB/Faiss 변환과 다를 수 있어 대상판본/엔진 구현 확인 전 보편 변환으로 쓰지 않는다. |
| `innerproduct` | 내적; 벡터 크기도 영향 | 부호에 따른 변환이며 0~1로 제한되지 않는다. 단위 정규화하면 cosine과 순위가 같을 수 있지만 raw score는 같다는 뜻이 아니다. |

거리/점수/의미적 정답률은 별개다. k-NN `_score`는 확률이 아니며 BM25 raw score와 직접 합산할 공통 척도도 아니다. cosine/innerproduct 선택은 모델의 정규화·권장 metric을 확인한다. Faiss cosine은 내부 정규화되어 저장 벡터가 입력과 달라질 수 있다. 판본 자료 차이와 실제 서버 점수는 정리 기록에 남겼다.

### 인덱스 타입

#### 1. Exact k-NN (정확 검색)

질의가 선택한 벡터 집합을 전수 비교해 해당 metric의 정확 최근접 이웃을 찾는다. 의미적 정답이 100%라는 뜻은 아니다. 아래 field mapping만으로 exact 실행이 정해지지 않는다. 5절의 knn_score script를 사용하고 ANN이 필요 없는 index에서는 index.knn=true를 생략할 수 있다.

```json
{
  "type": "knn_vector",
  "dimension": 1536
}
```

#### 2. Approximate k-NN (근사 검색)

HNSW 등의 탐색으로 후보 수를 줄인다. 지연/메모리와 exact top-k 대비 recall의 절충이며 손실이 항상 작다고 보장할 수 없다. 아래 Faiss cosine mapping은 2.19+ 조건이다.

```json
{
  "type": "knn_vector",
  "dimension": 1536,
  "method": {
    "name": "hnsw",
    "space_type": "cosinesimil",
    "engine": "faiss",
    "parameters": {
      "ef_construction": 256,
      "m": 16
    }
  }
}
```

### HNSW 파라미터

| 파라미터 | 작동 방식·조건 |
|----------|----------------|
| `m` | 그래프 연결 수. current 문서 기본 16이며 생성 후 변경 불가; 원래 16~64는 미평가 후보 범위다. |
| `ef_construction` | 생성 탐색 폭. current float HNSW 기본 100, 2.11 이하 생성 index의 이전 값 512 등 조건 차이가 있다. 원래 256~512는 미평가 후보다. |
| `ef_search` | 검색 탐색 폭. Faiss mapping/index/query의 우선순위와 판본 조건을 확인한다. Lucene은 index ef_search를 무시하고 k를 활용한다. 원래 256+는 보편 권장이 아니다. |

탐색/연결을 늘리면 recall이 개선될 수 있지만 메모리·빌드/검색 비용도 증가한다. 정답셋/필터·샤드·세그먼트를 고정해 recall@k와 지연을 함께 측정한다. k는 ANN 후보 조건, size는 최종 반환 수이며 실제 반환은 필터/문서 수/부분 실패에 따라 줄 수 있다.

---

## 어떻게 사용하는가? (How)

### 1. 벡터 검색용 인덱스 생성

```python
from math import isfinite
from collections.abc import Callable
from opensearchpy import OpenSearch

Embed = Callable[[str], list[float]]

def validate_dimension(dimension: int) -> None:
    if type(dimension) is not int or not 1 <= dimension <= 16000:
        raise ValueError("invalid_vector_dimension")

def validate_vector(vector: list[float], dimension: int) -> list[float]:
    validate_dimension(dimension)
    if not isinstance(vector, list) or len(vector) != dimension:
        raise ValueError("vector_dimension_mismatch")
    if any(type(x) not in (int, float) or not isfinite(x) for x in vector):
        raise ValueError("finite_numeric_vector_required")
    if not any(x != 0 for x in vector):
        raise ValueError("cosine_zero_vector_rejected")
    return [float(x) for x in vector]

def validate_k(k: int) -> None:
    # 실습 상한100은 서버 API의 일반 최대 k와 다르다.
    if type(k) is not int or not 1 <= k <= 100:
        raise ValueError("demo_k_must_be_1_to_100")

def checked_hits(response: dict) -> list[dict]:
    if response.get("timed_out") is not False:
        raise RuntimeError("search_completion_unconfirmed")
    if response.get("_shards", {}).get("failed") != 0:
        raise RuntimeError("search_shards_unconfirmed")
    return response["hits"]["hits"]

def vector_index_body(dimension: int, embedding_contract: str) -> dict:
    validate_dimension(dimension)
    if not isinstance(embedding_contract, str) or not embedding_contract.strip():
        raise ValueError("embedding_contract_required")
    return {
        "settings": {"index": {"knn": True, "number_of_shards": 1,
                               "number_of_replicas": 0}},
        "mappings": {
            "_meta": {"embedding_contract": embedding_contract},
            "properties": {
                "title": {"type": "text"}, "content": {"type": "text"},
                "category": {"type": "keyword"}, "source": {"type": "keyword"},
                "embedding": {"type": "knn_vector", "dimension": dimension,
                    "method": {"name": "hnsw", "space_type": "cosinesimil",
                               "engine": "faiss",
                               "parameters": {"ef_construction": 256, "m": 16}}},
            },
        },
    }

def new_vector_index(client: OpenSearch, index_name: str,
                     dimension: int, embedding_contract: str) -> dict:
    body = vector_index_body(dimension, embedding_contract)
    if client.indices.exists(index=index_name):
        raise ValueError("existing_index_not_modified")
    return client.indices.create(index=index_name, body=body)

# client는 기초/클라이언트 문서의 명시 factory로 준비하고 caller가 finally에서 close한다.
# 서버 조건: k-NN이 제공되는 OpenSearch3.x의 Faiss/HNSW/cosinesimil.
# 새 index만 생성한다. 256/16은 평가되지 않은 예제 값이다.
```

### 2. 임베딩 생성 및 문서 인덱싱

같은 문서의 앞 절 정의를 순서대로 사용한다. embedding_contract에는 모델 ID/고정 revision·query/document prefix·pooling/정규화·출력 차원을 기록한 식별자를 넣는다. 같은 차원인 다른 모델을 혼용하지 않는다. OpenAI callback을 선택한다면 호출자가 API client/인증·전송할 텍스트를 명시한다. 실제 공급자 요청/과금은 여기서 실행하지 않았다. 안정 create ID 재실행은409이며 실패 수를 확인한다.

```python
from opensearchpy import helpers

def get_embedding(text: str, embed: Embed, dimension: int) -> list[float]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("nonempty_text_required")
    return validate_vector(embed(text), dimension)

def openai_embedder(api_client, model: str, dimensions: int | None = None) -> Embed:
    """호출자가 준비한 OpenAI SDK client를 빌린다. 생성 시 API 요청 없음."""
    if dimensions is not None:
        validate_dimension(dimensions)
        if model not in ("text-embedding-3-small", "text-embedding-3-large"):
            raise ValueError("dimensions_requires_embedding_v3")
    def embed(text: str) -> list[float]:
        args = {"input": text, "model": model, "encoding_format": "float"}
        if dimensions is not None:
            args["dimensions"] = dimensions
        return api_client.embeddings.create(**args).data[0].embedding
    return embed

documents = [
    {"title": "Python 기초", "content": "Python은 배우기 쉬운 프로그래밍 언어입니다."},
    {"title": "머신러닝 입문", "content": "머신러닝은 데이터에서 패턴을 학습하는 기술입니다."},
    {"title": "웹 개발 가이드", "content": "웹 개발은 프론트엔드와 백엔드로 구성됩니다."},
    {"title": "데이터베이스 기초", "content": "데이터베이스는 데이터를 체계적으로 저장합니다."},
    {"title": "딥러닝 개요", "content": "딥러닝은 인공 신경망을 사용한 머신러닝의 한 분야입니다."},
]

def create_sample_documents(client: OpenSearch, index_name: str,
                            embed: Embed, dimension: int) -> dict[str, int]:
    categories = ["programming", "ai", "web", "database", "ai"]
    actions = []
    for i, doc in enumerate(documents):
        vector = get_embedding(f"{doc['title']} {doc['content']}", embed, dimension)
        actions.append({"_op_type": "create", "_index": index_name,
            "_id": f"vector-demo-{i}", "_source": {
                **doc, "category": categories[i], "source": "sample",
                "embedding": vector,
            }})
    success, failed = helpers.bulk(client, actions, stats_only=True,
        raise_on_error=False, refresh="wait_for", max_chunk_bytes=1024 * 1024)
    return {"success": success, "failed": failed}
```

### 3. 벡터 유사도 검색

```python
def vector_search(client: OpenSearch, index_name: str, embed: Embed,
                  dimension: int, query: str, k: int = 5) -> list[dict]:
    validate_k(k)
    query_embedding = get_embedding(query, embed, dimension)
    response = client.search(index=index_name, body={
        "size": k, "query": {"knn": {"embedding": {
            "vector": query_embedding, "k": k,
        }}}, "_source": ["title", "content", "category", "source"],
    })
    return [{"id": hit["_id"], "score": hit["_score"],
             "title": hit["_source"]["title"],
             "content": hit["_source"]["content"]}
            for hit in checked_hits(response)]

# 준비된client/index와같은모델/전처리 embed를전달한다.
# vector_search(client, prepared_index, embed, dimension, "인공지능 학습 방법", k=3)
```

**설명용 출력 예시 — 실제 서버/모델의 검색 점수 아님**:
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

category는 1절에서 keyword로 정의하고 2절의 적재 함수가 문서에 포함한다. 아래는 knn 내부의 효율적 필터이며 Faiss HNSW 2.9+/Lucene 2.4+ 등 조건을 확인한다. 원래 bool 밖 filter는 ANN 이후 필터라 k보다 적게 반환할 수 있다. top-level min_score는 최종 점수 컷이며 knn 내부 radial min_score와 다르다. 검증 없이 0.7/0.75를 기본값으로 강제하지 않는다. 필터는 권한 검증 자체가 아니다.

```python
def filtered_vector_search(client: OpenSearch, index_name: str, embed: Embed,
                           dimension: int, query: str,
                           category: str | None = None,
                           min_score: float | None = None, k: int = 5) -> list[dict]:
    validate_k(k)
    vector = get_embedding(query, embed, dimension)
    clause = {"vector": vector, "k": k}
    if category is not None:
        if not isinstance(category, str) or not category:
            raise ValueError("nonempty_category_required")
        clause["filter"] = {"term": {"category": category}}
    body = {"size": k, "query": {"knn": {"embedding": clause}},
            "_source": ["title", "content", "category", "source"]}
    if min_score is not None:
        if type(min_score) not in (int, float) or not isfinite(min_score) or min_score < 0:
            raise ValueError("finite_nonnegative_min_score_required")
        body["min_score"] = min_score
    return checked_hits(client.search(index=index_name, body=body))

# category='ai'는위샘플에실제로적재된다. min_score 기본은None(평가전임계값없음).
# 기존0.7/0.75는측정되지않은예시였다. 점수변환/엔진별로보편cosine임계값이아니다.
```

### 5. Script Score로 커스텀 유사도 계산

전수 metric 검색을 비교 기준으로 사용할 때 knn_score script를 사용한다. ANN query와 다르게 필터로 제한된 모든 embedding을 계산하므로 큰 후보 집합에는 비용이 크다. 사용자 정의 Painless 점수와는 구분한다.

```python
def exact_score_body(query_embedding: list[float], dimension: int,
                     k: int = 5, source_filter: str | None = None) -> dict:
    validate_k(k)
    vector = validate_vector(query_embedding, dimension)
    # embedding이없는문서는전수점수계산대상에서제외한다.
    conditions = [{"exists": {"field": "embedding"}}]
    if source_filter is not None:
        if not isinstance(source_filter, str) or not source_filter:
            raise ValueError("nonempty_source_required")
        conditions.append({"term": {"source": source_filter}})
    return {"size": k, "_source": ["title", "content", "source", "category"],
        "query": {"script_score": {
            "query": {"bool": {"filter": conditions}},
            "script": {"source": "knn_score", "lang": "knn", "params": {
                "field": "embedding", "query_value": vector,
                "space_type": "cosinesimil",
            }},
        }}}

# client.search(index=prepared_index, body=exact_score_body(vector, dimension))
# knn_score는k-NN플러그인의script이며Painless임의함수가아니다.
```

### 6. 임베딩 모델별 차원 설정

확인일 2026-10-04의 기본 출력 차원이다. 모델 카드와 제공자 문서만 확인했으며 모델을 다운로드/추론하지 않았다. OpenAI v3는 dimensions로 줄일 수 있고, 다른 모델도 wrapper/pooling/truncation 조건이 있다. Hugging Face의 생략형 이름은 아래 출처의 전체 repo ID를 사용한다. E5의 query/passage prefix 등 모델별 입력 정책도 확인한다. encoder 출력길이가 계약과 mapping에 일치해야 한다.

```python
# 아래는확인한기본출력차원;실제encoder출력길이와mapping을다시대조한다.
EMBEDDING_DIMENSIONS = {
    "text-embedding-ada-002": 1536,      # OpenAI
    "text-embedding-3-small": 1536,      # OpenAI
    "text-embedding-3-large": 3072,      # OpenAI
    "all-MiniLM-L6-v2": 384,             # sentence-transformers
    "all-mpnet-base-v2": 768,            # sentence-transformers
    "multilingual-e5-large": 1024,       # intfloat
    "bge-large-zh-v1.5": 1024,           # BAAI
}

# v3 dimensions 옵션이나다른pooling/truncation이있으면이기본표를그대로쓰지않는다.
model_name = "text-embedding-3-small"
dimension = EMBEDDING_DIMENSIONS[model_name]
```

### 7. 성능 최적화 팁

512/30s/replica0은 원래 미평가 설정 제안이며 실제로 적용하지 않는다. refresh는 검색 가시성, replica는 복제/가용성 조건이다. 적재 중 설정을 바꾸면 정상·실패 양쪽에서 원래 설정 복원이 필요하다. force merge는 쓰기를 완료한 인덱스에 운영 판단으로 수행하며 큰 세그먼트/임시 디스크/백그라운드 지속 비용을 고려한다. 모든 적재 뒤 1segment가 항상 최적이라는 규칙은 없다. cosine 영벡터 워밍업은 불가하므로 유효 대표 질의 벡터를 전달한다. 별도 Warmup API와 검색 한 번의 차이를 구분한다.

```python
# 설정제안만보존한다. 이dict를실제index에자동적용하지않는다.
index_settings_proposal = {"settings": {"index": {
    "knn.algo_param.ef_search": 512,
    "refresh_interval": "30s", "number_of_replicas": 0,
}}}

def warmup_search(client: OpenSearch, index_name: str,
                  representative_vector: list[float], dimension: int) -> dict:
    vector = validate_vector(representative_vector, dimension)
    return client.search(index=index_name, body={"size": 1,
        "_source": False,
        "query": {"knn": {"embedding": {"vector": vector, "k": 1}}},
    })

# refresh가필요한시점은caller가결정: client.indices.refresh(index=prepared_index)
# forcemerge는쓰기완료/운영판단뒤별도관리작업이다. 아래는자동실행하지않는다.
# client.indices.forcemerge(index=prepared_index, max_num_segments=1)
# replica0/refresh변경은실패경로에서도원래설정복원이필요하다.
# 대표검색워밍업은별도WarmupAPI의native캐시선적재와동일하지않다.
```

---

## 실전 예제: RAG용 문서 저장소

앞 절의 helper와 준비된 client·새 인덱스 계약을 사용한다. `validate_index`를 명시 호출해 고유 index의 mapping/dimension/metric/engine/embedding_contract를 확인한 뒤 사용한다. 제공 embedding도 같은 계약으로 생성해야 한다. 검사 flag는 권한이나 서버 판본을 확인하지 않으며, index 재생성/계약 변경 후 다시 검사한다. class가 client를 소유하지 않으므로 caller가 close한다. 결과는 명시한 필드만 반환해 원문의 id/score가 생성 필드를 덮어쓰지 않으며 embedding을 응답에서 제외한다.

```python
from dataclasses import dataclass
from opensearchpy import OpenSearch, helpers

@dataclass
class Document:
    id: str
    title: str
    content: str
    source: str
    embedding: list[float] | None = None

class VectorStore:
    """준비된client와새mapping계약을조합한다. 자동생성/삭제/모델호출없음."""

    def __init__(self, client: OpenSearch, index_name: str, embed: Embed,
                 embedding_dim: int, embedding_contract: str):
        vector_index_body(embedding_dim, embedding_contract) # 로컬인자검사만
        self.client, self.index_name, self.embed = client, index_name, embed
        self.embedding_dim, self.embedding_contract = embedding_dim, embedding_contract
        self._mapping_checked = False

    def validate_index(self) -> None:
        """사용전명시호출. alias/여러index를추정하지않고고유index mapping확인."""
        self._mapping_checked = False
        response = self.client.indices.get_mapping(index=self.index_name)
        if set(response) != {self.index_name}:
            raise ValueError("single_concrete_index_required")
        mapping = response[self.index_name]["mappings"]
        fields = mapping.get("properties", {})
        vector = fields.get("embedding", {})
        method = vector.get("method", {})
        if (mapping.get("_meta", {}).get("embedding_contract") != self.embedding_contract
            or vector.get("type") != "knn_vector"
            or vector.get("dimension") != self.embedding_dim
            or method.get("name") != "hnsw" or method.get("engine") != "faiss"
            or method.get("space_type") != "cosinesimil"
            or any(fields.get(f, {}).get("type") != t for f, t in
                   (("title", "text"), ("content", "text"), ("source", "keyword")) )):
            raise ValueError("index_embedding_contract_mismatch")
        self._mapping_checked = True

    def _require_mapping(self) -> None:
        if not self._mapping_checked:
            raise RuntimeError("validate_index_required")

    def add_documents(self, documents: list[Document]) -> dict[str, int]:
        self._require_mapping()
        actions, ids = [], set()
        for doc in documents:
            if not isinstance(doc.id, str) or not doc.id or doc.id in ids:
                raise ValueError("unique_nonempty_document_id_required")
            if len(doc.id.encode("utf-8")) > 512:
                raise ValueError("document_id_too_long")
            if any(not isinstance(x, str) for x in (doc.title, doc.content, doc.source)):
                raise ValueError("document_text_fields_required")
            ids.add(doc.id)
            # 제공 embedding은 같은 계약에서 생성했을 때만 전달한다. 차원만으로 모델 일치를 증명할 수 없다.
            vector = (get_embedding(f"{doc.title} {doc.content}", self.embed, self.embedding_dim)
                      if doc.embedding is None else validate_vector(doc.embedding, self.embedding_dim))
            actions.append({"_op_type": "create", "_index": self.index_name,
                "_id": doc.id, "_source": {"title": doc.title, "content": doc.content,
                                          "source": doc.source, "embedding": vector}})
        if not actions:
            return {"success": 0, "failed": 0}
        success, failed = helpers.bulk(self.client, actions, stats_only=True,
            raise_on_error=False, refresh="wait_for", max_chunk_bytes=1024 * 1024)
        return {"success": success, "failed": failed}

    def search(self, query: str, k: int = 5,
               source_filter: str | None = None) -> list[dict]:
        self._require_mapping()
        validate_k(k)
        vector = get_embedding(query, self.embed, self.embedding_dim)
        clause = {"vector": vector, "k": k}
        if source_filter is not None:
            if not isinstance(source_filter, str) or not source_filter:
                raise ValueError("nonempty_source_required")
            clause["filter"] = {"term": {"source": source_filter}}
        response = self.client.search(index=self.index_name, body={"size": k,
            "query": {"knn": {"embedding": clause}},
            "_source": ["title", "content", "source"],
        })
        return [{"id": hit["_id"], "score": hit["_score"],
                 "title": hit["_source"]["title"],
                 "content": hit["_source"]["content"],
                 "source": hit["_source"]["source"]}
                for hit in checked_hits(response)]

def vector_store_demo(client: OpenSearch, prepared_index: str, embed: Embed,
                      dimension: int, contract: str) -> tuple[dict, list[dict]]:
    store = VectorStore(client, prepared_index, embed, dimension, contract)
    store.validate_index()
    docs = [
        Document(id="1", title="RAG 소개", content="RAG는 검색 증강 생성...", source="manual"),
        Document(id="2", title="벡터 DB", content="벡터 데이터베이스는...", source="blog"),
    ]
    outcome = store.add_documents(docs)
    if outcome["failed"]:
        raise RuntimeError("demo_indexing_incomplete")
    return outcome, store.search("검색 증강 생성이란?", k=3)
```

---

## 참고 자료 (References)

확인일: **2026-10-04**. rolling latest/main과서버고정판본을구분한다. 로컬SDK검증판본은opensearch-py **3.2.0**이다.

- [ANN](https://docs.opensearch.org/latest/vector-search/vector-search-techniques/approximate-knn/)·[Exact script](https://docs.opensearch.org/latest/vector-search/vector-search-techniques/knn-score-script/) — 후보/전수·size/k조건.
- [Methods and engines](https://docs.opensearch.org/latest/mappings/supported-field-types/knn-methods-engines/)·[효율적 필터](https://docs.opensearch.org/latest/vector-search/filter-search-knn/efficient-knn-filtering/)·[필터 방식 비교](https://docs.opensearch.org/latest/vector-search/filter-search-knn/index/) — 엔진/판본·필터시점.
- [3.0 Spaces](https://docs.opensearch.org/3.0/field-types/supported-field-types/knn-spaces/)·[rolling Spaces](https://docs.opensearch.org/latest/mappings/supported-field-types/knn-spaces/)·[SpaceType 일차 소스](https://github.com/opensearch-project/k-NN/blob/main/src/main/java/org/opensearch/knn/index/SpaceType.java) — metric/점수와영벡터조건.
- [k-NN query](https://docs.opensearch.org/latest/query-dsl/specialized/k-nn/index/)·[Force merge](https://docs.opensearch.org/latest/api-reference/index-apis/force-merge/)·[k-NN API/Warmup](https://docs.opensearch.org/latest/vector-search/api/knn/) — 파라미터/운영조건.
- [OpenAI 임베딩](https://developers.openai.com/api/docs/guides/embeddings) — ada1536/v3기본1536·3072 및dimensions조건.
- [MiniLM 모델 카드](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)·[MPNet](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)·[E5](https://huggingface.co/intfloat/multilingual-e5-large)·[BGE zh](https://huggingface.co/BAAI/bge-large-zh-v1.5) — 기본차원/입력정책.

> [!todo] 남은 확인
> 대상 서버의 플러그인/판본·mapping 수용·ANN/exact 실제 점수·필터/recall/지연/동시 쓰기·공급자 추론/전처리 동일성은 미확인이다. 공식 자료의 점수/필터 설명 차이는 고정 판본 구현과 대조가 필요하다. Claude 연결 불가로 품질 정책/새 엔진 도입·문서 통합은 보류했다.

## 관련 문서

- [OpenSearch 기초](./opensearch-basics.md) - 설치, 기본 개념
- [키워드 검색 (BM25)](./keyword-search-bm25.md) - 전문 검색
- [하이브리드 검색](./hybrid-search.md) - 벡터 + 키워드 결합
- [Milvus 기초](../milvus/milvus-basics.md) - 다른 벡터 DB 비교

---

*Last updated: 2026-02-05*
