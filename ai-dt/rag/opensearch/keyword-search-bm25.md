---
tags: [opensearch, keyword-search, bm25, full-text-search, analyzer]
level: intermediate
last_updated: 2026-02-05
status: in-progress
---

# OpenSearch 키워드 검색 (BM25 Full-text Search)

> OpenSearch의 전문 검색(Full-text Search)과 BM25 알고리즘을 활용한 키워드 기반 문서 검색

## 왜 필요한가? (Why)

### 벡터 검색의 한계

벡터 검색(Semantic Search)은 "의미"를 잘 잡지만, 다음 상황에서 약점이 있다:

| 상황 | 벡터 검색 | 키워드 검색 |
|------|----------|------------|
| 고유명사 검색 (SKU, 제품코드) | ❌ 약함 | ✅ 강함 |
| 정확한 용어 매칭 | ❌ 약함 | ✅ 강함 |
| 오타/동의어 처리 | ✅ 강함 | ❌ 약함 |
| 문맥 이해 | ✅ 강함 | ❌ 약함 |

**예시**:
```
쿼리: "SKU-A123-XYZ"

벡터 검색: "제품 코드 관련 문서" 반환 (❌ 원하는 게 아님)
키워드 검색: "SKU-A123-XYZ" 정확히 포함된 문서 반환 (✅)
```

### BM25 (Best Matching 25)

TF-IDF를 개선한 랭킹 알고리즘. OpenSearch의 기본 점수 계산 방식.

```
BM25 Score = IDF × (TF × (k1 + 1)) / (TF + k1 × (1 - b + b × dl/avgdl))

- TF: 문서 내 용어 빈도 (Term Frequency)
- IDF: 역문서 빈도 (Inverse Document Frequency) - 희귀할수록 중요
- dl: 문서 길이
- avgdl: 평균 문서 길이
- k1, b: 튜닝 파라미터
```

**핵심 아이디어**:
- 검색어가 **문서에 자주** 나오면 점수 ↑
- 검색어가 **전체 문서에서 희귀**하면 점수 ↑
- **짧은 문서**에서 나오면 점수 ↑ (문서 길이 정규화)

---

## 핵심 개념 (What)

### 쿼리 타입 비교

| 쿼리 타입 | 용도 | 예시 |
|----------|------|------|
| `match` | 분석기 적용, 기본 전문 검색 | "머신러닝 입문" |
| `match_phrase` | 순서 유지 구문 검색 | "딥 러닝 기초" |
| `term` | 정확 매칭 (분석 안 함) | "status": "active" |
| `terms` | 여러 값 중 하나 매칭 | ["python", "java"] |
| `bool` | 복합 조건 조합 | must + should + filter |
| `multi_match` | 여러 필드 동시 검색 | title + content |
| `query_string` | Lucene 문법 지원 | "title:python AND content:기초" |

### Analyzer (분석기) 구조

```
┌─────────────────────────────────────────────────────────────┐
│                        Analyzer                              │
├─────────────────────────────────────────────────────────────┤
│  Input: "OpenSearch는 빠른 검색 엔진이다!"                    │
│                          │                                   │
│                          ▼                                   │
│  ┌───────────────────────────────────────┐                  │
│  │         Character Filter              │                  │
│  │   (HTML 제거, 특수문자 변환 등)         │                  │
│  └───────────────────────────────────────┘                  │
│                          │                                   │
│                          ▼                                   │
│  ┌───────────────────────────────────────┐                  │
│  │            Tokenizer                   │                  │
│  │   (공백/형태소 기준 토큰 분리)           │                  │
│  │   → ["OpenSearch", "는", "빠른", ...]  │                  │
│  └───────────────────────────────────────┘                  │
│                          │                                   │
│                          ▼                                   │
│  ┌───────────────────────────────────────┐                  │
│  │          Token Filter                  │                  │
│  │   (소문자화, 불용어 제거, 형태소 분석)    │                  │
│  │   → ["opensearch", "빠르", "검색", ...]│                  │
│  └───────────────────────────────────────┘                  │
│                          │                                   │
│                          ▼                                   │
│  Output: ["opensearch", "빠르", "검색", "엔진"]               │
└─────────────────────────────────────────────────────────────┘
```

### 내장 분석기 (Built-in Analyzers)

| 분석기 | 설명 | 용도 |
|--------|------|------|
| `standard` | 유니코드 텍스트 분할 | 기본값, 영어/한국어 혼용 |
| `simple` | 알파벳만, 소문자화 | 간단한 영문 |
| `whitespace` | 공백 기준 분할 | 코드, ID |
| `keyword` | 토큰화 안 함 (전체가 1토큰) | exact match |
| `nori` | 한국어 형태소 분석 | 한국어 전용 |

### Nori 분석기 (한국어)

OpenSearch에 기본 내장된 한국어 형태소 분석기.

```
"삼성전자 주가가 상승했다"
      ↓ nori analyzer
["삼성전자", "주가", "상승"]
```

---

## 어떻게 사용하는가? (How)

### 1. 기본 키워드 검색 인덱스

```python
from opensearchpy import OpenSearch

client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    use_ssl=False
)

index_name = "articles"

# 인덱스 생성
index_body = {
    "settings": {
        "index": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "analysis": {
            "analyzer": {
                "korean": {
                    "type": "custom",
                    "tokenizer": "nori_tokenizer",
                    "filter": ["lowercase", "nori_part_of_speech"]
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
            "category": {
                "type": "keyword"  # 정확 매칭용
            },
            "tags": {
                "type": "keyword"  # 배열도 가능
            },
            "created_at": {
                "type": "date"
            }
        }
    }
}

if client.indices.exists(index=index_name):
    client.indices.delete(index=index_name)

client.indices.create(index=index_name, body=index_body)
```

### 2. 문서 인덱싱

```python
from opensearchpy import helpers

documents = [
    {
        "title": "파이썬으로 배우는 머신러닝",
        "content": "머신러닝은 데이터에서 패턴을 학습하는 인공지능의 한 분야입니다.",
        "category": "ai",
        "tags": ["python", "ml", "beginner"],
        "created_at": "2026-01-15"
    },
    {
        "title": "딥러닝 기초와 신경망",
        "content": "딥러닝은 다층 신경망을 사용하여 복잡한 패턴을 학습합니다.",
        "category": "ai",
        "tags": ["deeplearning", "neural-network"],
        "created_at": "2026-01-20"
    },
    {
        "title": "OpenSearch 검색 엔진 가이드",
        "content": "OpenSearch는 오픈소스 검색 및 분석 엔진입니다.",
        "category": "database",
        "tags": ["search", "opensearch"],
        "created_at": "2026-02-01"
    },
]

actions = [
    {"_index": index_name, "_id": str(i), "_source": doc}
    for i, doc in enumerate(documents)
]

helpers.bulk(client, actions, refresh=True)
print(f"Indexed {len(documents)} documents")
```

### 3. 기본 검색 쿼리

#### match 쿼리 (기본)

```python
def search_match(query: str, field: str = "content", size: int = 10):
    """기본 전문 검색"""
    response = client.search(
        index=index_name,
        body={
            "query": {
                "match": {
                    field: {
                        "query": query,
                        "operator": "or"  # 기본값: 검색어 중 하나만 매칭되어도 OK
                    }
                }
            },
            "size": size,
            "_source": ["title", "content", "category"]
        }
    )
    return response["hits"]["hits"]

# 사용
results = search_match("머신러닝 학습")
for hit in results:
    print(f"[{hit['_score']:.2f}] {hit['_source']['title']}")
```

#### match_phrase 쿼리 (구문 검색)

```python
def search_phrase(query: str, field: str = "content"):
    """순서가 중요한 구문 검색"""
    response = client.search(
        index=index_name,
        body={
            "query": {
                "match_phrase": {
                    field: {
                        "query": query,
                        "slop": 1  # 허용 단어 거리 (기본 0)
                    }
                }
            }
        }
    )
    return response["hits"]["hits"]

# "딥러닝 기초" 구문이 순서대로 나오는 문서
results = search_phrase("딥러닝 기초", "title")
```

#### multi_match 쿼리 (여러 필드)

```python
def search_multi_field(query: str, fields: list[str] = None):
    """여러 필드에서 동시 검색"""
    if fields is None:
        fields = ["title^2", "content"]  # title에 2배 가중치

    response = client.search(
        index=index_name,
        body={
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": fields,
                    "type": "best_fields"  # 가장 잘 매칭되는 필드 기준
                }
            }
        }
    )
    return response["hits"]["hits"]

# title과 content에서 동시 검색
results = search_multi_field("머신러닝")
```

### 4. Bool 쿼리 (복합 조건)

```python
def search_complex(
    keyword: str,
    category: str = None,
    tags: list[str] = None,
    date_from: str = None
):
    """복합 조건 검색"""

    must = [{"match": {"content": keyword}}]
    filter_conditions = []

    if category:
        filter_conditions.append({"term": {"category": category}})

    if tags:
        filter_conditions.append({"terms": {"tags": tags}})

    if date_from:
        filter_conditions.append({
            "range": {"created_at": {"gte": date_from}}
        })

    response = client.search(
        index=index_name,
        body={
            "query": {
                "bool": {
                    "must": must,           # 반드시 매칭 (점수에 영향)
                    "filter": filter_conditions,  # 필터 (점수 영향 X)
                    # "should": [],         # 있으면 점수 ↑
                    # "must_not": []        # 반드시 제외
                }
            }
        }
    )
    return response["hits"]["hits"]

# AI 카테고리에서 "학습" 키워드 검색
results = search_complex(
    keyword="학습",
    category="ai",
    date_from="2026-01-01"
)
```

### 5. 하이라이팅 (검색어 강조)

```python
def search_with_highlight(query: str):
    """검색 결과에 하이라이트 추가"""
    response = client.search(
        index=index_name,
        body={
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title", "content"]
                }
            },
            "highlight": {
                "fields": {
                    "title": {},
                    "content": {
                        "fragment_size": 100,
                        "number_of_fragments": 3
                    }
                },
                "pre_tags": ["<strong>"],
                "post_tags": ["</strong>"]
            }
        }
    )

    for hit in response["hits"]["hits"]:
        print(f"Title: {hit['_source']['title']}")
        if "highlight" in hit:
            for field, fragments in hit["highlight"].items():
                print(f"  {field}: {' ... '.join(fragments)}")

# 사용
search_with_highlight("머신러닝")
```

**출력 예시**:
```
Title: 파이썬으로 배우는 머신러닝
  title: 파이썬으로 배우는 <strong>머신러닝</strong>
  content: <strong>머신러닝</strong>은 데이터에서 패턴을 학습하는...
```

### 6. 한국어 분석기 커스터마이징

```python
# Nori 분석기 상세 설정
index_body = {
    "settings": {
        "analysis": {
            "tokenizer": {
                "nori_mixed": {
                    "type": "nori_tokenizer",
                    "decompound_mode": "mixed",  # 복합어 분리
                    "discard_punctuation": True,
                    "user_dictionary": "userdict_ko.txt"  # 사용자 사전
                }
            },
            "filter": {
                "nori_posfilter": {
                    "type": "nori_part_of_speech",
                    "stoptags": [
                        "E", "IC", "J", "MAG", "MAJ",  # 조사, 접속사 등 제거
                        "MM", "SP", "SSC", "SSO",
                        "SC", "SE", "XPN", "XSA",
                        "XSN", "XSV", "UNA", "NA",
                        "VSV", "VCP", "VCN", "VX"
                    ]
                },
                "nori_readingform": {
                    "type": "nori_readingform"  # 한자 → 한글
                }
            },
            "analyzer": {
                "korean_analyzer": {
                    "type": "custom",
                    "tokenizer": "nori_mixed",
                    "filter": [
                        "nori_readingform",
                        "lowercase",
                        "nori_posfilter"
                    ]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "content": {
                "type": "text",
                "analyzer": "korean_analyzer"
            }
        }
    }
}
```

### 7. 분석기 테스트

```python
def analyze_text(text: str, analyzer: str = "korean"):
    """분석기 결과 확인"""
    response = client.indices.analyze(
        index=index_name,
        body={
            "analyzer": analyzer,
            "text": text
        }
    )

    tokens = [t["token"] for t in response["tokens"]]
    print(f"Input: {text}")
    print(f"Tokens: {tokens}")
    return tokens

# 테스트
analyze_text("OpenSearch는 빠른 검색 엔진입니다")
# Output: ['opensearch', '빠르', '검색', '엔진']
```

### 8. BM25 파라미터 튜닝

```python
# 인덱스 설정에서 BM25 파라미터 조정
index_body = {
    "settings": {
        "index": {
            "similarity": {
                "custom_bm25": {
                    "type": "BM25",
                    "k1": 1.2,  # 기본 1.2, 높으면 TF 영향 ↑
                    "b": 0.75   # 기본 0.75, 낮추면 문서 길이 영향 ↓
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "content": {
                "type": "text",
                "similarity": "custom_bm25"
            }
        }
    }
}
```

---

## 실전 예제: 문서 검색 API

```python
from dataclasses import dataclass
from typing import Optional
from opensearchpy import OpenSearch, helpers

@dataclass
class SearchResult:
    id: str
    score: float
    title: str
    content: str
    highlight: Optional[dict] = None

class KeywordSearchEngine:
    """BM25 기반 키워드 검색 엔진"""

    def __init__(self, host: str = "localhost", port: int = 9200):
        self.client = OpenSearch(
            hosts=[{"host": host, "port": port}],
            use_ssl=False
        )
        self.index_name = "search-engine"
        self._ensure_index()

    def _ensure_index(self):
        if not self.client.indices.exists(index=self.index_name):
            self.client.indices.create(
                index=self.index_name,
                body={
                    "settings": {
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
                            "category": {"type": "keyword"},
                        }
                    }
                }
            )

    def index_documents(self, documents: list[dict]) -> int:
        """문서 벌크 인덱싱"""
        actions = [
            {"_index": self.index_name, "_source": doc}
            for doc in documents
        ]
        success, _ = helpers.bulk(self.client, actions, refresh=True)
        return success

    def search(
        self,
        query: str,
        category: str = None,
        size: int = 10,
        highlight: bool = True
    ) -> list[SearchResult]:
        """키워드 검색"""

        # 기본 multi_match 쿼리
        must_query = {
            "multi_match": {
                "query": query,
                "fields": ["title^2", "content"],
                "type": "best_fields"
            }
        }

        # Bool 쿼리 구성
        if category:
            search_query = {
                "bool": {
                    "must": [must_query],
                    "filter": [{"term": {"category": category}}]
                }
            }
        else:
            search_query = must_query

        # 검색 바디
        body = {
            "query": search_query,
            "size": size,
            "_source": ["title", "content", "category"]
        }

        # 하이라이트 추가
        if highlight:
            body["highlight"] = {
                "fields": {
                    "title": {},
                    "content": {"fragment_size": 150}
                },
                "pre_tags": ["**"],
                "post_tags": ["**"]
            }

        response = self.client.search(index=self.index_name, body=body)

        results = []
        for hit in response["hits"]["hits"]:
            results.append(SearchResult(
                id=hit["_id"],
                score=hit["_score"],
                title=hit["_source"]["title"],
                content=hit["_source"]["content"],
                highlight=hit.get("highlight")
            ))

        return results

    def suggest(self, prefix: str, field: str = "title", size: int = 5) -> list[str]:
        """자동완성 제안"""
        response = self.client.search(
            index=self.index_name,
            body={
                "query": {
                    "match_phrase_prefix": {
                        field: prefix
                    }
                },
                "size": size,
                "_source": [field]
            }
        )
        return [hit["_source"][field] for hit in response["hits"]["hits"]]


# 사용 예시
engine = KeywordSearchEngine()

# 문서 인덱싱
engine.index_documents([
    {"title": "Python 기초", "content": "파이썬 프로그래밍...", "category": "programming"},
    {"title": "머신러닝 입문", "content": "ML 기초 개념...", "category": "ai"},
])

# 검색
results = engine.search("파이썬", category="programming")
for r in results:
    print(f"[{r.score:.2f}] {r.title}")
    if r.highlight:
        print(f"  → {r.highlight}")
```

---

## 참고 자료 (References)

- [OpenSearch Query DSL](https://opensearch.org/docs/latest/query-dsl/)
- [Text Analysis](https://opensearch.org/docs/latest/analyzers/)
- [Nori 한국어 분석기](https://opensearch.org/docs/latest/analyzers/language-analyzers/#korean-nori)
- [BM25 Similarity](https://opensearch.org/docs/latest/search-plugins/search-relevance/)

## 관련 문서

- [OpenSearch 기초](./opensearch-basics.md) - 설치, 기본 개념
- [벡터 검색 (k-NN)](./vector-search-knn.md) - 의미 기반 검색
- [하이브리드 검색](./hybrid-search.md) - 벡터 + 키워드 결합

---

*Last updated: 2026-02-05*
