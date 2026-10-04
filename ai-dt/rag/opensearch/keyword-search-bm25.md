---
tags: [opensearch, keyword-search, bm25, full-text-search, analyzer]
level: intermediate
last_updated: 2026-02-05
reviewed_on: 2026-10-04
review_status: partial
document_type: learning_note
aliases: [BM25 키워드 검색]
---

# OpenSearch 키워드 검색 (BM25 Full-text Search)

> OpenSearch의 전문 검색(Full-text Search)과 BM25 알고리즘을 활용한 키워드 기반 문서 검색

> [!info] 읽기 목적과 검증 범위 — 2026-10-04
> 분석기가 문자열을 토큰으로 바꾸고 역색인이 후보를 찾은 뒤 BM25가 순위를 매기는 흐름을 배운다. `text` 전문 검색과 `keyword` 정확 일치를 구분한다. 공식 rolling 문서와 opensearch-py 3.2.0을 대조했다. 예제는 같은 문서의 앞 절 정의를 순서대로 불러오며, 준비된 client/index를 인자로 넘길 때만 통신한다. 기존 index를 삭제하지 않는다. 실제 OpenSearch/Nori 토큰·검색 순위·서버 mapping 수용은 미확인이다.

## 왜 필요한가? (Why)

### 벡터 검색의 한계

벡터 검색은 임베딩의 의미 유사도로 후보를 찾는다. 전문 검색은 분석된 용어를 찾고 BM25로 정렬한다. 어느 방식이 항상 우수한지는 데이터와 질의 평가가 필요하다.

| 상황 | 적용 조건 |
|------|----------|
| SKU·제품코드 | `keyword` 필드의 `term`으로 정확 일치. `text`의 기본 분석기는 하이픈/대소문자를 바꿀 수 있어 문자열 전체 일치가 아니다. |
| 정확한 용어 | 분석 결과가 맞으면 전문 검색이 직접 후보를 찾는다. 원문 정확 일치와 분석된 토큰 일치는 구분한다. |
| 오타·동의어 | 키워드 검색도 fuzziness/동의어 분석 설정으로 처리할 수 있다. 벡터 검색도 모든 오타를 해결한다고 보장할 수 없다. |
| 문맥·표현 변경 | 임베딩 모델의 후보와 키워드 후보를 평가하고 필요하면 하이브리드를 검토한다. |

**예시**: `SKU-A123-XYZ`를 찾으려면 SKU를 `keyword`로 적재하고 `term` 질의한다. 아래 articles 예제에는 SKU 필드를 만들지 않았으므로 별도 mapping이 필요하다. 벡터가 반환할 결과는 여기서 실측하지 않았다.

### BM25 (Best Matching 25)

TF-IDF를 개선한 랭킹 알고리즘. OpenSearch의 기본 점수 계산 방식.

```
전통/Legacy 단일 용어 기여 = IDF × TF × (k1 + 1) / (TF + k1 × (1 - b + b × dl/avgdl))
OpenSearch 3.0+ 기본 BM25 단일 용어 기여 = IDF × TF / (TF + k1 × (1 - b + b × dl/avgdl))

- TF: 문서 내 용어 빈도 (Term Frequency)
- IDF: 역문서 빈도 (Inverse Document Frequency) - 희귀할수록 중요
- dl: 문서 길이
- avgdl: 평균 문서 길이
- k1, b: 튜닝 파라미터
```

이는 이해용 단일 용어 식이다. 전체 질의는 여러 용어/필드·boost·norm·통계 조건에 영향을 받는다. OpenSearch 3.0 기본 구현 변경으로 같은 `k1=1.2`에서 기존 대비 배율은 약 1/2.2이다. 상수만 다른 단일 BM25 순위는 유지되지만 raw score 임계값이나 다른 점수와의 합산은 다시 평가해야 한다. 점수는 확률이나 0~1 척도가 아니다. 실제 인덱스의 `_explain`으로 확인한다. [공식 버전 변경 설명](https://docs.opensearch.org/latest/search-plugins/keyword-search/) 확인: 2026-10-04.

**핵심 아이디어**:
- 검색어가 **문서에 자주** 나오면 점수 ↑
- 검색어가 **전체 문서에서 희귀**하면 점수 ↑
- 같은 TF 등 다른 조건이 같고 `b>0`이면 **짧은 필드**에 길이 정규화가 유리하게 작용한다. 분석된 토큰 길이이며 원문 글자 수와 다르다.

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

아래 한국어 토큰은 단계 설명용이며 실제 출력이 아니다. Nori 형태소 분리는 tokenizer가 수행하고 이후 POS/reading/lowercase 필터가 토큰을 조정한다. Char filter와 token filter는 설정에 따라 생략 가능하다.

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
│  │   (소문자화, 품사·불용어 필터)    │                  │
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
| `simple` | letter가 아닌 문자에서 분리, 소문자화; 숫자 제외 | 단순 문자 기반 검색 |
| `whitespace` | 공백 기준 분할; 대소문자 유지 | 공백 분리가 원하는 규칙일 때 |
| `keyword` | 토큰화 안 함 (전체가 1토큰) | exact match |
| `nori` | 한국어 형태소 분석; 추가 플러그인 | `analysis-nori` 제공/설치된 배포 |

### Nori 분석기 (한국어)

`analysis-nori` 추가 플러그인의 한국어 형태소 분석기다. 모든 설치에 기본 내장된다고 가정하지 않는다. 자체 서버에서는 대상 버전의 플러그인 목록을 확인하고 미설치라면 해당 서버의 설치 절차를 따른다. 관리형 서비스는 제공 플러그인/사용자 사전 배포 정책이 다르다. 여기서 플러그인을 설치하거나 서버를 재시작하지 않았다.

아래는 개념 출력이며 사전·POS 필터·decompound 설정/버전에 따라 실제 토큰은 달라진다. `_analyze`에서 token뿐 아니라 position/start_offset/end_offset을 비교한다. [추가 플러그인 목록](https://docs.opensearch.org/latest/install-and-configure/additional-plugins/index/) 확인: 2026-10-04.

```
"삼성전자 주가가 상승했다"
      ↓ nori analyzer
["삼성전자", "주가", "상승"]
```

---

## 어떻게 사용하는가? (How)

### 1. 기본 키워드 검색 인덱스

새 인덱스 생성 권한과 Nori 설치가 필요하다. shard1/replica0은 단일 노드 소량 실습 값이며 고가용성 설계가 아니다. 이미 존재하면 실패시키고 기존 데이터를 보존한다. 기존 인덱스를 재사용하려면 mapping/analyzer를 별도로 확인한다.

```python
from opensearchpy import OpenSearch

def new_keyword_index(client: OpenSearch, index_name: str) -> dict:
    """analysis-nori가 설치된 서버의 새 실습 인덱스만 생성한다."""
    if client.indices.exists(index=index_name):
        raise ValueError("existing_index_not_modified")
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
    return client.indices.create(index=index_name, body=index_body)

def checked_hits(response: dict) -> list[dict]:
    # 서버 오류/시간 초과/부분 shard 실패를 '검색 결과 없음'으로 취급하지 않는다.
    if response.get("timed_out") is not False:
        raise RuntimeError("search_completion_unconfirmed")
    if response.get("_shards", {}).get("failed") != 0:
        raise RuntimeError("search_shards_unconfirmed")
    return response["hits"]["hits"]

def validate_query(query: str, size: int = 10) -> None:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("nonempty_query_required")
    if type(size) is not int or not 1 <= size <= 100:
        raise ValueError("demo_size_must_be_1_to_100")

# 연결: 기초 문서의 local_demo_client 또는 Python 클라이언트 문서의
# tls_client로 명시적으로 준비한다. 사용 후 caller가 finally: client.close().
# new_keyword_index(client, "새로운-실습-index")는 caller가 선택해 실행한다.
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

def create_sample_documents(client: OpenSearch, index_name: str) -> dict[str, int]:
    actions = [
        {"_op_type": "create", "_index": index_name,
         "_id": f"bm25-demo-{i}", "_source": doc}
        for i, doc in enumerate(documents)
    ]
    success, failed = helpers.bulk(
        client, actions, refresh="wait_for", stats_only=True,
        raise_on_error=False, chunk_size=100, max_chunk_bytes=1024 * 1024,
    )
    return {"success": success, "failed": failed}

# 동일 ID 재실행은 409 실패이며 기존 문서를 덮어쓰지 않는다.
# 성공 수뿐 아니라 failed를 확인한다. 전송 예외는 caller로 전파한다.
```

### 3. 기본 검색 쿼리

#### match 쿼리 (기본)

```python
def search_match(client: OpenSearch, index_name: str, query: str, field: str = "content", size: int = 10) -> list[dict]:
    """기본 전문 검색"""
    validate_query(query, size)
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
    return checked_hits(response)
```

#### match_phrase 쿼리 (구문 검색)

```python
def search_phrase(client: OpenSearch, index_name: str, query: str, field: str = "content") -> list[dict]:
    """순서가 중요한 구문 검색"""
    validate_query(query)
    response = client.search(
        index=index_name,
        body={
            "query": {
                "match_phrase": {
                    field: {
                        "query": query,
                        "slop": 0  # 분석된 토큰 위치의 정확 구문; 바꾸면 재배열도 가능
                    }
                }
            }
        }
    )
    return checked_hits(response)
```

#### multi_match 쿼리 (여러 필드)

```python
def search_multi_field(client: OpenSearch, index_name: str, query: str, fields: list[str] | None = None) -> list[dict]:
    """여러 필드에서 동시 검색"""
    validate_query(query)
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
    return checked_hits(response)
```

### 4. Bool 쿼리 (복합 조건)

category/tags는 앞 절의 `keyword`, created_at은 `date`다. tags의 `terms`는 목록 중 하나라도 일치하는 조건이다. `filter`는 점수에 기여하지 않지만 접근 권한을 자동 보장하지 않는다. `must`/`filter`가 있는 bool에서 should의 기본 minimum_should_match는 0, 둘 다 없으면 1이다. 필수 대안을 추가할 때 이 값을 명시한다.

```python
def search_complex(client: OpenSearch, index_name: str,
    keyword: str,
    category: str | None = None,
    tags: list[str] | None = None,
    date_from: str | None = None
) -> list[dict]:
    """복합 조건 검색"""

    validate_query(keyword)
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
    return checked_hits(response)
```

### 5. 하이라이팅 (검색어 강조)

```python
def search_with_highlight(client: OpenSearch, index_name: str, query: str) -> list[dict]:
    """검색 결과에 하이라이트 추가"""
    validate_query(query)
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
                "encoder": "html",
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

    return checked_hits(response)
```

highlight 응답은 원문에 서버가 태그를 삽입한 조각이다. 예제의 `encoder=html`은 원문을 HTML escape하고 허용한 강조 태그를 유지한다. 화면에서 원문/임의 응답을 그대로 innerHTML로 넣지 않는다. Markdown 강조도 자동 안전 렌더링을 뜻하지 않는다. 복잡한 bool 질의에서는 강조가 모든 매칭 논리를 반영한다고 보장되지 않는다.

**개념 출력 예시 — 실제 서버 실행 결과 아님**:
```
Title: 파이썬으로 배우는 머신러닝
  title: 파이썬으로 배우는 <strong>머신러닝</strong>
  content: <strong>머신러닝</strong>은 데이터에서 패턴을 학습하는...
```

### 6. 한국어 분석기 커스터마이징

`custom_nori_body`는 앞 절 index_body와 별개의 새 인덱스 제안이다. mixed는 복합어와 분해 토큰을 함께 유지하는 설정이다. POS 제거 목록은 평가되지 않은 원래 실습 후보이며 업무 용어 누락을 `_analyze`로 확인한다. 예제는 없는 `userdict_ko.txt` 의존성 대신 inline 사전으로 바꿨다. 파일 사전 `user_dictionary`를 선택한다면 노드의 config 경로에 같은 파일을 준비해야 하며 inline 규칙과 동시에 지정하지 않는다. 실제 사전 배포·한국어 품질은 미확인이다.

```python
# Nori 분석기 상세 설정
custom_nori_body = {
    "settings": {
        "analysis": {
            "tokenizer": {
                "nori_mixed": {
                    "type": "nori_tokenizer",
                    "decompound_mode": "mixed",  # 복합어 분리
                    "discard_punctuation": True,
                    "user_dictionary_rules": ["삼성전자"]  # inline 실습 사전
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

기본 인덱스는 `korean`, 별도 custom_nori_body로 생성한 인덱스는 `korean_analyzer`를 전달한다. 실패를 임의 토큰/빈 목록으로 바꾸지 않는다. 서버 Analyze 권한도 필요하다. 예: `analyze_text(client, prepared_index, "OpenSearch는 빠른 검색 엔진입니다", "korean")`; 출력 토큰은 실행한 서버에서 확인한다.

```python
def analyze_text(client: OpenSearch, index_name: str, text: str, analyzer: str = "korean") -> list[str]:
    """분석기 결과 확인"""
    validate_query(text)
    response = client.indices.analyze(
        index=index_name,
        body={
            "analyzer": analyzer,
            "text": text
        }
    )

    tokens = [t["token"] for t in response["tokens"]]
    return tokens
```

### 8. BM25 파라미터 튜닝

`bm25_settings_body`도 독립적인 새 인덱스 설정 예시다. 기본 k1=1.2/b=0.75를 명시했으며 성능 최적값으로 실측한 값이 아니다. k1은 빈도 포화, b는 길이 정규화의 강도를 조절한다. 분석기·청킹·질의별 정답을 고정해 순위 평가한 뒤 바꾼다. 현재 서버에 자동 설정 변경하지 않는다.

```python
# 인덱스 설정에서 BM25 파라미터 조정
bm25_settings_body = {
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

앞 절 정의와 client·Nori mapping을 준비한 후 호출하는 동기 클래스다. HTTP API 서버 자체를 띄우는 예제는 아니다. 생성자는 연결/생성/삭제하지 않고 caller의 client를 빌린다. title/content/category mapping을 미리 확인한다. create 벌크는 항목 실패 수를 반환하고, 검색은 timeout/부분 shard 실패를 전파한다. suggest는 문서 제목의 마지막 토큰 prefix 검색이며 전용 completion 사전/중복 없는 전체 자동완성 목록을 보장하지 않는다.

```python
from dataclasses import dataclass
from opensearchpy import OpenSearch, helpers

@dataclass
class SearchResult:
    id: str
    score: float
    title: str
    content: str
    highlight: dict | None = None

class KeywordSearchEngine:
    """앞 절의 함수와 준비된 index/client를 조합한다. 소유권은 caller에 있다."""

    def __init__(self, client: OpenSearch, index_name: str):
        self.client = client
        self.index_name = index_name
        # 자동 연결/인덱스 생성/기존 mapping 추정은 하지 않는다.

    def index_documents(self, documents: list[dict]) -> dict[str, int]:
        """각 항목은 {id: 안정 ID, source: 원문 dict}; 기존 ID는 create 충돌."""
        actions = []
        ids = set()
        for doc in documents:
            doc_id, source = doc["id"], doc["source"]
            if not isinstance(doc_id, str) or not doc_id or doc_id in ids:
                raise ValueError("unique_nonempty_document_id_required")
            if len(doc_id.encode("utf-8")) > 512:
                raise ValueError("document_id_too_long")
            if not isinstance(source, dict) or any(
                not isinstance(source.get(key), str) for key in ("title", "content")
            ):
                raise ValueError("title_and_content_strings_required")
            ids.add(doc_id)
            actions.append({"_op_type": "create", "_index": self.index_name,
                            "_id": doc_id, "_source": source})
        if not actions:
            return {"success": 0, "failed": 0}
        success, failed = helpers.bulk(
            self.client, actions, stats_only=True, raise_on_error=False,
            refresh="wait_for", chunk_size=100, max_chunk_bytes=1024 * 1024,
        )
        return {"success": success, "failed": failed}

    def search(self, query: str, category: str | None = None,
               size: int = 10, highlight: bool = True) -> list[SearchResult]:
        validate_query(query, size)
        if type(highlight) is not bool:
            raise ValueError("highlight_must_be_boolean")
        must_query = {"multi_match": {"query": query,
                      "fields": ["title^2", "content"], "type": "best_fields"}}
        search_query = must_query
        if category is not None:
            if not isinstance(category, str) or not category:
                raise ValueError("nonempty_category_required")
            search_query = {"bool": {"must": [must_query],
                            "filter": [{"term": {"category": category}}]}}
        body = {"query": search_query, "size": size,
                "_source": ["title", "content", "category"]}
        if highlight:
            body["highlight"] = {
                "encoder": "html",
                "fields": {"title": {}, "content": {"fragment_size": 150}},
                "pre_tags": ["<strong>"], "post_tags": ["</strong>"],
            }
        response = self.client.search(index=self.index_name, body=body)
        return [SearchResult(
            id=hit["_id"], score=hit["_score"],
            title=hit["_source"]["title"], content=hit["_source"]["content"],
            highlight=hit.get("highlight"),
        ) for hit in checked_hits(response)]

    def suggest(self, prefix: str, field: str = "title", size: int = 5) -> list[str]:
        """마지막 분석 토큰의 prefix로 문서를 검색; 전용 completion suggester와 다름."""
        validate_query(prefix, size)
        if field not in ("title", "content"):
            raise ValueError("mapped_text_field_required")
        response = self.client.search(index=self.index_name, body={
            "query": {"match_phrase_prefix": {field: prefix}},
            "size": size, "_source": [field],
        })
        return list(dict.fromkeys(hit["_source"][field] for hit in checked_hits(response)))

def keyword_demo(client: OpenSearch, prepared_index: str) -> tuple[dict, list[SearchResult]]:
    engine = KeywordSearchEngine(client, prepared_index)
    outcome = engine.index_documents([
        {"id": "python-basics", "source": {"title": "Python 기초",
         "content": "파이썬 프로그래밍...", "category": "programming"}},
        {"id": "ml-intro", "source": {"title": "머신러닝 입문",
         "content": "ML 기초 개념...", "category": "ai"}},
    ])
    # failed가 있으면 검색 전에 caller가 실패 항목 정책을 결정한다.
    if outcome["failed"]:
        raise RuntimeError("demo_indexing_incomplete")
    return outcome, engine.search("파이썬", category="programming")
```

---

## 참고 자료 (References)

확인일: **2026-10-04**. `latest`/GitHub main은 rolling 자료이며 고정 서버 판본의 수용 증거가 아니다. SDK 실행 검증 판본은 opensearch-py **3.2.0**이다.

- [Query DSL](https://docs.opensearch.org/latest/query-dsl/)·[bool](https://docs.opensearch.org/latest/query-dsl/compound/bool/) — 분석/복합 질의 조건.
- [Keyword search](https://docs.opensearch.org/latest/search-plugins/keyword-search/)·[Similarity 설정](https://docs.opensearch.org/latest/im-plugin/similarity/) — BM25 기본값과 3.0 점수 변경.
- [내장 분석기](https://docs.opensearch.org/latest/analyzers/supported-analyzers/index/)·[추가 플러그인](https://docs.opensearch.org/latest/install-and-configure/additional-plugins/index/)·[설치 절차](https://docs.opensearch.org/latest/install-and-configure/plugins/) — Nori 설치 조건.
- [NoriTokenizerFactory 일차 소스](https://github.com/opensearch-project/OpenSearch/blob/main/plugins/analysis-nori/src/main/java/org/opensearch/index/analysis/NoriTokenizerFactory.java) — mixed/사전 파일과 inline 규칙 배타 조건.
- [Analyze](https://docs.opensearch.org/latest/api-reference/analyze-apis/)·[phrase slop](https://docs.opensearch.org/latest/query-dsl/full-text/match-phrase/)·[phrase prefix](https://docs.opensearch.org/latest/query-dsl/full-text/match-phrase-prefix/) — 실제 토큰/위치와 prefix 조건.
- [Highlight](https://docs.opensearch.org/latest/search-plugins/searching-data/highlight/) — HTML encoder와 강조 범위 한계.

> [!todo] 남은 확인
> 실제 서버 판본·Nori 설치/사전/POS 결과·BM25 `_explain`/순위·partial shard/timeout·mapping 생성 수용·highlight 화면은 미확인이다. 로컬 모의 HTTP는 SDK 직렬화/예외 처리 검증이다. 사용자별 검색 품질·분류 변경/중복 통합은 Claude 협의 후 판단하며 현재 연결 불가를 정리 기록에 남겼다.

## 관련 문서

- [OpenSearch 기초](./opensearch-basics.md) - 설치, 기본 개념
- [벡터 검색 (k-NN)](./vector-search-knn.md) - 의미 기반 검색
- [하이브리드 검색](./hybrid-search.md) - 벡터 + 키워드 결합

---

*Last updated: 2026-02-05*
