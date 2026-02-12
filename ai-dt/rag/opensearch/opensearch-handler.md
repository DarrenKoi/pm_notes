---
tags: [opensearch, elasticsearch, python, handler, client, bulk, search, aggregation]
level: intermediate
last_updated: 2026-02-12
status: complete
---

# opensearch_handler — 범용 OpenSearch/ES 핸들러

> OpenSearch와 Elasticsearch 7.x를 위한 공유 Python 패키지. 클라이언트 생성, 인덱스 관리, 문서 CRUD, 검색, Aggregation을 단일 인터페이스로 제공한다.

## 왜 필요한가? (Why)

- `opensearch-py`는 저수준 클라이언트라서 매번 `hosts` 포맷 구성, SSL 옵션, bulk action 변환 등 **보일러플레이트**가 반복된다
- knowhow-elasticsearch, history-opensearch 등 여러 프로젝트에서 **동일한 패턴**을 복사하고 있었다
- 프로젝트마다 클라이언트 코드를 유지보수하면 **불일치와 버그**가 생긴다

`opensearch_handler`는 이 공통 패턴을 한 곳에 모아, 프로젝트별로는 **도메인 로직만** 작성하면 되게 한다.

---

## 핵심 개념 (What)

### 패키지 구조

```
opensearch_handler/
├── __init__.py         # 전체 public API re-export
├── config.py           # ConnectionConfig 데이터클래스 + load_config()
├── client.py           # create_client() — OpenSearch 인스턴스 생성
├── index.py            # 인덱스 CRUD (create, exists, delete, settings)
├── document.py         # 문서 CRUD + bulk_index
├── search.py           # 검색 (match, term, bool, knn, hybrid, aggregate)
├── example.py          # 전체 기능 사용 예제
└── requirements.txt    # opensearch-py>=2.4.0
```

### API 전체 목록

| 모듈 | 함수 | 설명 |
|------|------|------|
| **config** | `ConnectionConfig` | 접속 정보 데이터클래스 (host, port, auth, SSL, bulk_chunk) |
| | `load_config(**overrides)` | env vars + 키워드로 ConnectionConfig 생성 |
| **client** | `create_client(config, **overrides)` | OpenSearch 클라이언트 인스턴스 반환 |
| **index** | `index_exists(client, name)` | 인덱스 존재 여부 확인 |
| | `create_index(client, name, mappings, settings, shards, replicas, refresh_interval)` | 인덱스 생성 |
| | `delete_index(client, name)` | 인덱스 삭제 |
| | `get_index_settings(client, name)` | 인덱스 설정 조회 |
| | `update_index_settings(client, name, settings)` | 동적 설정 변경 |
| **document** | `index_document(client, index, doc, doc_id)` | 단일 문서 색인 |
| | `get_document(client, index, doc_id)` | 문서 조회 |
| | `update_document(client, index, doc_id, doc)` | 부분 업데이트 |
| | `delete_document(client, index, doc_id)` | 문서 삭제 |
| | `bulk_index(client, index, docs, id_field, chunk_size)` | 벌크 색인 |
| **search** | `match_search(client, index, field, query, size)` | Full-text 검색 |
| | `term_search(client, index, field, value, size)` | Exact match 검색 |
| | `bool_search(client, index, must, should, filter, must_not, size)` | Bool 복합 쿼리 |
| | `knn_search(client, index, field, vector, k, size)` | k-NN 벡터 검색 |
| | `hybrid_search(client, index, query, text_field, vector_field, vector, k, size)` | 텍스트 + 벡터 하이브리드 |
| | `aggregate(client, index, agg_body, query, size)` | Aggregation 쿼리 |

---

## 어떻게 사용하는가? (How)

### 1. 접속 설정 (ConnectionConfig)

```python
from opensearch_handler import ConnectionConfig, load_config, create_client

# 방법 1: 직접 생성
config = ConnectionConfig(
    host="my-cluster", port=9200,
    user="admin", password="admin",
    use_ssl=True,
)
client = create_client(config=config)

# 방법 2: 환경 변수 + 오버라이드
#   OPENSEARCH_HOST, OPENSEARCH_PORT, OPENSEARCH_USER, OPENSEARCH_PASSWORD,
#   OPENSEARCH_USE_SSL, OPENSEARCH_VERIFY_CERTS, OPENSEARCH_CA_CERTS,
#   OPENSEARCH_BULK_CHUNK
config = load_config()          # env vars 자동 적용
config = load_config(port=9201) # env vars + 명시적 오버라이드

# 방법 3: create_client에 직접 전달 (config=None이면 load_config 호출)
client = create_client(host="my-cluster", use_ssl=False)
```

설정 우선순위: `dataclass 기본값` → `환경 변수` → `명시적 키워드`

`ConnectionConfig` 기본값:

| 필드 | 기본값 | 설명 |
|------|--------|------|
| `host` | `"localhost"` | 호스트명 |
| `port` | `9200` | 포트 |
| `user` / `password` | `"admin"` | HTTP Basic Auth |
| `use_ssl` | `True` | HTTPS 사용 여부 |
| `verify_certs` | `False` | 인증서 검증 |
| `bulk_chunk` | `500` | 벌크 요청당 문서 수 |

### 2. 인덱스 관리

```python
from opensearch_handler import create_index, index_exists, delete_index

# 인덱스 생성 (mappings, settings는 선택)
if not index_exists(client, "my-index"):
    create_index(
        client, "my-index",
        mappings={
            "properties": {
                "title": {"type": "text"},
                "category": {"type": "keyword"},
            }
        },
        shards=1,         # 기본값 1
        replicas=0,       # 기본값 0
        refresh_interval="30s",  # 기본값 "30s"
    )
```

`create_index()`의 `shards`, `replicas`, `refresh_interval`은 **기본값 역할**이다. `settings` dict에 이미 해당 키가 있으면 덮어쓰지 않는다.

### 3. 문서 CRUD

```python
from opensearch_handler import index_document, get_document, bulk_index

# 단일 문서
index_document(client, "my-index",
    doc={"title": "Hello", "category": "test"},
    doc_id="1")

doc = get_document(client, "my-index", "1")
print(doc["_source"]["title"])  # "Hello"

# 벌크 색인
docs = [{"title": f"Doc {i}", "category": "bulk"} for i in range(1000)]
success, errors = bulk_index(client, "my-index", docs,
    id_field=None,      # 자동 ID 생성
    chunk_size=500)     # 500개씩 분할 전송
```

`bulk_index()`에 `id_field`를 지정하면 각 문서의 해당 필드 값을 `_id`로 사용한다.

### 4. 검색

```python
from opensearch_handler import match_search, term_search, bool_search, aggregate

# Full-text 검색
results = match_search(client, "my-index", "title", "Hello", size=10)

# Exact match
results = term_search(client, "my-index", "category", "test", size=10)

# Bool 복합 쿼리
results = bool_search(client, "my-index",
    must=[{"match": {"title": "Hello"}}],
    filter=[{"term": {"category": "test"}}],
    size=10)

# Aggregation
results = aggregate(client, "my-index",
    agg_body={"categories": {"terms": {"field": "category", "size": 100}}})
buckets = results["aggregations"]["categories"]["buckets"]
```

### 5. 벡터 / 하이브리드 검색

```python
from opensearch_handler import knn_search, hybrid_search

# k-NN 벡터 검색 (OpenSearch k-NN 플러그인 필요)
results = knn_search(client, "my-index",
    field="embedding", vector=[0.1, 0.2, 0.3], k=5)

# 하이브리드: full-text + k-NN을 bool should로 결합
results = hybrid_search(client, "my-index",
    query="검색어",
    text_field="content",
    vector_field="embedding",
    vector=[0.1, 0.2, 0.3],
    k=5, size=10)
```

### 6. 다른 프로젝트에서 import하기

`opensearch_handler`는 독립 패키지가 아니라 `Codes/python/` 아래의 디렉토리 모듈이다. 다른 프로젝트에서 사용하려면 `sys.path`에 `Codes/python/`을 추가해야 한다.

```python
# _path_setup.py (프로젝트 루트에 배치)
import sys
from pathlib import Path

_LIB_ROOT = str(Path(__file__).resolve().parent.parent)  # → Codes/python/
if _LIB_ROOT not in sys.path:
    sys.path.insert(0, _LIB_ROOT)
```

```python
# 사용하는 모듈에서
import _path_setup  # noqa: F401
import opensearch_handler as osh

client = osh.create_client(host="localhost", use_ssl=False)
```

> 디렉토리 이름이 `opensearch_handler`인 이유: `opensearch`로 하면 `opensearchpy` 패키지의 내부 `opensearch` 모듈과 **네임스페이스 충돌**이 발생한다.

---

## 설계 원칙

1. **함수 기반**: 클래스로 감싸지 않고 `(client, index, ...)` 시그니처의 순수 함수로 구성. 호출자가 클라이언트 인스턴스 생명주기를 관리한다.
2. **opensearch-py 직접 반환**: 검색 결과를 래핑하지 않고 `dict`(opensearch-py 응답)를 그대로 반환. 호출자가 `hits`, `aggregations` 등을 자유롭게 처리할 수 있다.
3. **ES 7.x 호환**: `opensearch-py`가 Elasticsearch 7.10에서 포크되었으므로, ES 7.x 클러스터에도 그대로 사용 가능하다.
4. **도메인 로직 없음**: 인덱스 매핑, 검색 boost, 필터링 로직은 호출하는 프로젝트에서 정의한다.

---

## 참고 자료 (References)

- [opensearch-py 공식 문서](https://opensearch.org/docs/latest/clients/python-low-level/)
- [opensearch-py GitHub](https://github.com/opensearch-project/opensearch-py)
- [OpenSearch Query DSL](https://opensearch.org/docs/latest/query-dsl/)

## 관련 문서

- [knowhow-elasticsearch 구조](./knowhow-elasticsearch-architecture.md) - 이 핸들러를 사용하는 프로젝트 예시
- [Python 클라이언트 활용](./python-client.md) - 대용량 Bulk 처리, Async, 에러 핸들링
- [하이브리드 검색](./hybrid-search.md) - 벡터 + 키워드 결합 전략
- [실습 코드](../../../Codes/python/opensearch_handler/)
