---
tags: [opensearch, python, handler, client, bulk, search, aggregation]
level: intermediate
last_updated: 2026-02-12
reviewed_on: 2026-10-04
review_status: partial
document_type: implementation_note
aliases: [공유 OpenSearch 래퍼의 책임과 적용 조건]
---

# opensearch_handler — 범용 OpenSearch 핸들러

> 로컬 래퍼에 기록된 설정·CRUD·검색 인터페이스의 의도와 적용 조건을 읽는다.

> [!warning] 과거 구현 설명과 현재 확인 범위 — 2026-10-04
> 패키지 구조/API 표/기본값은 원래2026-02-12 기록이며 현재 로컬 구현과 대조하지 못했다. `Codes/python/opensearch_handler`는 다른 독립 주제이므로 이 폴더에서 import하거나 근거로 읽지 않았다. 아래 wrapper 예제는 해당 구현·시그니처·오류 계약을 별도로 확인한 환경에서만 실행할 수 있다. 이번에는 AST 구문과 표준 Python 경로 검사만 확인했다. SDK3.2.0 공식 문서는 로컬 wrapper의 동작 증거가 아니다. 확인된 직접SDK 예제는 [Python 클라이언트](./python-client.md), 벡터 schema/embedding 계약은 [벡터 검색](./vector-search-knn.md)을 읽는다.

## 왜 필요한가? (Why)

- `opensearch-py`는 저수준 클라이언트라서 매번 `hosts` 포맷 구성, SSL 옵션, bulk action 변환 등 **보일러플레이트**가 반복된다
- 원래 기록은 여러 프로젝트에 같은 패턴을 복사하던 맥락에서 공통화를 설명했다. 현재 사용 프로젝트 수/배포 여부는 미확인
- 프로젝트마다 클라이언트 코드를 유지보수하면 **불일치와 버그**가 생긴다

래퍼의 목적은 반복 조립을 줄이는 것이다. 호출자가 schema·인증/권한·모델/벡터 전처리·오류/재시도·client 종료를 이해할 책임까지 없어지는 것은 아니다. 이 문서는 로컬 래퍼의 의도 기록이고 직접SDK 학습 문서는 API와 실패 처리를 설명하므로 같은 설명을 보편 라이브러리 기능으로 통합하지 않는다.

---

## 핵심 개념 (What)

### 패키지 구조

원래 기록의 구조이며 파일 존재/공개 export·`>=2.4.0` 의존성의 현재성은 미확인이다. 검증일 현재 SDK는3.2.0으로 별도 예제에서 검사했지만 이 래퍼의 실행 호환성을 뜻하지 않는다.

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

원래 기록된 목록을 보존한다. `delete_*`는 이름 목록일 뿐 이번에 실행하지 않았다. 인자 기본값·반환형/빈 결과·bulk 실패·exception 처리·resource close는 실제 구현 검토가 필요하다.

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
import os

def connection_config_examples() -> list:
    # 호출 시에만 로컬 wrapper를 import한다. import/시그니처는 현재 미확인이다.
    from opensearch_handler import ConnectionConfig, load_config
    host = os.environ["OPENSEARCH_HOST"]
    user = os.environ["OPENSEARCH_USER"]
    password = os.environ["OPENSEARCH_PASSWORD"]
    ca_certs = os.environ["OPENSEARCH_CA_CERTS"]
    # 방법1: 직접 생성. 코드에 자격 정보를 쓰지 않고 검증할 CA를 명시한다.
    direct = ConnectionConfig(host=host, port=9200, user=user, password=password,
                              use_ssl=True, verify_certs=True, ca_certs=ca_certs)
    # 방법2: 환경 변수. 환경값 파싱/누락·unknown 처리는 구현 확인 대상이다.
    from_env = load_config()
    overridden = load_config(port=9201, use_ssl=True, verify_certs=True, ca_certs=ca_certs)
    return [direct, from_env, overridden]

# 원래 환경변수 목록: OPENSEARCH_HOST/PORT/USER/PASSWORD/USE_SSL/VERIFY_CERTS/
# CA_CERTS/BULK_CHUNK. 실제 지원/우선순위는 확인한 wrapper 계약으로 검사한다.
# 방법3: create_client에 직접 전달한다는 원래 인터페이스 기록.
# client = create_client(host=verified_host, use_ssl=True, verify_certs=True,
#                        ca_certs=verified_ca_path)
# 사용할 한 client만 caller가 명시 생성하고 종료 시 client.close()한다.
```

원래 기록의 설정 우선순위: `dataclass 기본값` → `환경 변수` → `명시적 키워드`. 현재 구현·bool/숫자 파싱·unknown 처리와 대조하지 못했다.

원래 기록된 `ConnectionConfig` 기본값 (**현재성 미확인**):

| 필드 | 기본값 | 설명 |
|------|--------|------|
| `host` | `"localhost"` | 호스트명 |
| `port` | `9200` | 포트 |
| `user` / `password` | `"admin"` | HTTP Basic Auth |
| `use_ssl` | `True` | HTTPS 사용 여부 |
| `verify_certs` | `False` | 인증서 검증 |
| `bulk_chunk` | `500` | 벌크 요청당 문서 수 |

아래 표의 admin/인증서 검증False는 역사적 기본값 기록이지 사용 권장값이 아니다. 공식SDK와 래퍼의 기본값을 섞지 않는다. 실제 환경에서는 명시 인증/CA·hostname 검증과 client 수명을 확인한다.

### 2. 인덱스 관리

```python
from opensearch_handler import create_index, index_exists

# caller가 별도 준비한 client와 fresh demo 이름을 전달한다. 기존 schema를 추측 재사용하지 않는다.
def create_handler_demo_index(client, index_name: str) -> None:
    if index_exists(client, index_name):
        raise ValueError("demo_index_already_exists")
    create_index(client, index_name,
        mappings={"properties": {
            "title": {"type": "text"}, "content": {"type": "text"},
            "category": {"type": "keyword"}, "doc_id": {"type": "keyword"},
            "embedding": {"type": "knn_vector", "dimension": 3,
                "method": {"name": "hnsw", "engine": "faiss", "space_type": "cosinesimil"}},
        }},
        settings={"index.knn": True}, shards=1, replicas=0, refresh_interval="30s")
# OpenSearch3.x+Faiss cosine/k-NN 전제의 3차원 toy fixture다. 실제 모델 검색 품질이 아니다.
# client와 index_name은 이후 예제에서 같은 실습 대상이다. 자동 실행/삭제 없음.
```

기존 문서는 shards1/replicas0/refresh30s가 wrapper 기본값이고 settings의 기존 키를 덮지 않는다고 설명했다. 현재 구현과 대조하지 못했다. 여기서는 값을 명시하며 replica0/refresh 지연은 독립 실습 조건이다. 생성 응답/매핑·기존 자원 오류와 부분 실패를 확인해야 한다.

### 3. 문서 CRUD

```python
from opensearch_handler import index_document, get_document, bulk_index

# 단일 문서
index_document(client, index_name,
    doc={"title": "Hello", "content": "Hello document", "category": "test",
         "doc_id": "1", "embedding": [0.1, 0.2, 0.3]},
    doc_id="1")

doc = get_document(client, index_name, "1")
print(doc["_source"]["title"])  # "Hello"

# 벌크 색인
docs = [{"doc_id": f"bulk-{i}", "title": f"Doc {i}", "content": f"Document {i}",
         "category": "bulk", "embedding": [0.1, 0.2, 0.3]} for i in range(1000)]
# 동일 toy 벡터는 순위 평가용이 아니다. 원래1000개 적재 흐름을 보존한다.
success, errors = bulk_index(client, index_name, docs,
    id_field="doc_id",  # 재실행 시 새 자동ID가 증가하지 않게 의도를 명시
    chunk_size=500)     # 500개씩 분할 전송
```

원래 설명은 id_field로 문서 필드 값을 _id로 사용한다고 기록했다. SDK Index API는 같은 _id에 대해 기존 문서를 교체할 수 있으므로 안정ID만으로 create-only가 되지 않는다. wrapper의 operation/충돌·실패 반환·누락/중복ID 계약은 미확인이다. 처음 적재한 fresh 실습에서만 사용하고 success/errors를 확인한다. 자동ID는 재실행에 새 문서가 늘 수 있다.30s refresh라 즉시 검색 반영을 보장하지 않으며 refresh는 durability와 별개다.

### 4. 검색

term은 keyword의 색인 토큰을 정확 매칭한다. 분석한 text 원문 전체 문자열 매칭이 아니다. bool의 should 기본 조건은 must/filter 유무에 따라 달라질 수 있으므로 실제 요청 body를 확인한다. terms aggregation size100은 모든 category를 반환한다는 보장이 아니다. timeout/failed shard·항목 실패/누락 상태를 검사하는 직접SDK 예제와 함께 읽는다.

```python
from opensearch_handler import match_search, term_search, bool_search, aggregate

# Full-text 검색
results = match_search(client, index_name, "title", "Hello", size=10)

# Exact match
results = term_search(client, index_name, "category", "test", size=10)

# Bool 복합 쿼리
results = bool_search(client, index_name,
    must=[{"match": {"title": "Hello"}}],
    filter=[{"term": {"category": "test"}}],
    size=10)

# Aggregation
results = aggregate(client, index_name,
    agg_body={"categories": {"terms": {"field": "category", "size": 100}}})
buckets = results["aggregations"]["categories"]["buckets"]
```

### 5. 벡터 / 하이브리드 검색

원문 my-index에는 embedding/content mapping이 없었는데 같은 index에서 검색해 예제가 연결되지 않았다. 위3차원 toy schema/자료로 맞추었다. 실제 임베딩은 문서/질의의 모델 revision·차원·전처리 계약이 같아야 한다. bool should 결합은 [하이브리드 문서](./hybrid-search.md)의 native normalization/RRF와 다른 raw score 합산 경로다. 함수 이름만으로 pipeline을 사용한다고 판단하지 않는다.

```python
from opensearch_handler import knn_search, hybrid_search

# k-NN 벡터 검색 (OpenSearch k-NN 플러그인 필요)
results = knn_search(client, index_name,
    field="embedding", vector=[0.1, 0.2, 0.3], k=5)

# 하이브리드: full-text + k-NN을 bool should로 결합
results = hybrid_search(client, index_name,
    query="검색어",
    text_field="content",
    vector_field="embedding",
    vector=[0.1, 0.2, 0.3],
    k=5, size=10)
```

### 6. 다른 프로젝트에서 import하기

원래 기록은 `Codes/python/` 아래 디렉터리 모듈과 프로젝트별 `_path_setup.py`로 import했다고 설명했다. 현재 packaging·실제 모듈 경로는 미확인이다. 상위 디렉터리를 추측해 연결하지 않고 caller가 확인한 절대 module root를 전달한다. sys.path의 앞 경로가 우선하므로 같은 이름을 가리는 문제도 확인한다.

```python
# _path_setup.py: 아래 함수 자체는 표준 Python 경로 검사다.
import sys
from pathlib import Path

def add_verified_module_root(module_root: Path) -> None:
    if not isinstance(module_root, Path) or not module_root.is_absolute():
        raise ValueError("verified_absolute_module_root_required")
    root = module_root.resolve(strict=True)
    if not (root / "opensearch_handler" / "__init__.py").is_file():
        raise ValueError("expected_package_not_found")
    value = str(root)
    if value not in sys.path:
        sys.path.insert(0, value)
# 파일 존재 검사는 코드 신뢰성/공개 API 검증을 뜻하지 않는다.
```

```python
# 앞 helper를 _path_setup.py로 준비한 별도 실습 환경에서만 명시 호출한다.
import os
from pathlib import Path
from _path_setup import add_verified_module_root

add_verified_module_root(Path(os.environ["VERIFIED_OPENSEARCH_MODULE_ROOT"]))
import opensearch_handler as osh
# osh.__file__로 실제 import 위치/배포 revision을 확인한다. 연결을 자동 생성하지 않는다.
```

원문의 `opensearch` 이름이면 내부 모듈과 반드시 충돌한다는 주장은 근거가 확인되지 않았다. Python은 패키지/모듈 경로에 따라 이름을 해석하므로 이름만으로 필연적 충돌을 단정하지 않는다. 실제 import 위치를 확인한다.

---

## 설계 원칙

1. **함수 기반이라는 원래 의도**: `(client, index, ...)` 형태지만 네트워크 I/O와 서버 상태 변경은 부수 효과여서 순수 함수가 아니다. caller가 client 수명을 관리한다.
2. **SDK 응답 직접 반환이라는 기록**: 실제 wrapper 반환형·오류 처리 확인이 필요하다. caller는 hits/aggregations뿐 아니라 timeout/shard 실패와 누락된 상태를 검사한다.
3. **호환성 검증**: 포크 계보만으로 Elasticsearch7.x 전체 호환을 보장할 수 없다. 공식 SDK matrix는 OpenSearch 판본에 대한 조건부 호환을 설명한다.3.x client는1.x~3.x에서 제거된 기능을 쓰지 않는 조건이 있으며 로컬 wrapper/Elasticsearch 호환은 별도 미확인이다.
4. **도메인 분리 의도**: schema/boost/filter를 caller에서 정의한다는 설계 기록이다. 실제 wrapper가 이를 변경/기본 주입하는지는 구현 검토가 필요하다.

---

## 참고 자료 (References)

확인일 **2026-10-04**. 공식SDK3.2.0와 rolling 자료는 이 로컬 wrapper의 현재 구현/서버 실행 증거가 아니다.

- [SDK compatibility matrix](https://github.com/opensearch-project/opensearch-py/blob/main/COMPATIBILITY.md)
- [Term query](https://docs.opensearch.org/latest/query-dsl/term/term/)·[Bool query](https://docs.opensearch.org/latest/query-dsl/compound/bool/)·[Index document](https://docs.opensearch.org/latest/api-reference/document-apis/index-document/)
- [Python module search path](https://docs.python.org/3/tutorial/modules.html#the-module-search-path)

- [opensearch-py 공식 문서](https://docs.opensearch.org/latest/clients/python-low-level/)
- [opensearch-py GitHub](https://github.com/opensearch-project/opensearch-py)
- [OpenSearch Query DSL](https://docs.opensearch.org/latest/query-dsl/)

## 관련 문서

- [Python 클라이언트 활용](./python-client.md) - 대용량 Bulk 처리, Async, 에러 핸들링
- [하이브리드 검색](./hybrid-search.md) - 벡터 + 키워드 결합 전략
- 별도 코드 주제 이름 `Codes/python/opensearch_handler`는 원래 맥락으로만 보존한다. 이 주제 밖으로 탐색하는 링크는 제거했다.
