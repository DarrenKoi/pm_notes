---
tags: [opensearch, elasticsearch, knowhow, architecture, bulk-index, retrieval]
level: intermediate
last_updated: 2026-02-12
status: complete
---

# knowhow-elasticsearch 프로젝트 구조

> 사내 Knowhow 데이터를 LLM으로 enrichment한 뒤 OpenSearch/Elasticsearch에 색인하고 하이브리드 검색하는 파이프라인

## 왜 필요한가? (Why)

- 사내에 축적된 Knowhow 텍스트(용어 정의, 프로세스, 트러블슈팅 등)를 **검색 가능한 구조화된 데이터**로 변환해야 한다
- 단순 키워드 매칭이 아닌 **하이브리드 검색**(exact match + full-text)으로 관련성 높은 결과를 찾아야 한다
- OpenSearch(dev)와 Elasticsearch 7.14(prod) **멀티 클러스터 환경**을 지원해야 한다

---

## 핵심 개념 (What)

### 전체 파이프라인

```
sample_data/*.json          (원본 Knowhow)
        │
        ▼
    extract_v2.py           (LLM enrichment → summary, category, keywords)
        │
        ▼
    processed_data/         (JSONL 중간 저장)
    progress.jsonl
        │
        ▼
    index.py                (OpenSearch bulk 색인)
        │
        ▼
    retrieval.py            (하이브리드 검색 API)
```

### 프로젝트 구조

```
knowhow-elasticsearch/
├── _path_setup.py          # sys.path에 Codes/python/ 추가 (opensearch_handler 임포트용)
├── os_settings.py          # 클러스터 설정, 인덱스 매핑, ConnectionConfig 생성
├── models.py               # Pydantic 모델 (KnowhowItem, EnrichedKnowhow)
├── llm_processor.py        # LLM enrichment 로직 (summary/category/keywords 추출)
├── extract_v2.py           # LLM enrichment 파이프라인 (중단/재개 지원)
├── index.py                # OpenSearch 색인 (bulk index)
├── retrieval.py            # 검색 API (하이브리드, term, aggregation)
└── sample_data/            # 원본 JSON 데이터

의존하는 공유 모듈:
../opensearch_handler/      # 범용 OpenSearch 클라이언트/인덱스/검색/문서 핸들러
```

### 공유 모듈 의존 관계

`opensearch_handler`는 범용 OpenSearch 연산을 제공하는 패키지다. knowhow-elasticsearch는 이 핸들러를 import하여 클라이언트 생성, 인덱스 관리, 벌크 색인, term 검색, aggregation을 수행한다.

```
knowhow-elasticsearch             opensearch_handler
─────────────────────             ──────────────────
os_settings.py ──────────────►    ConnectionConfig
index.py ────────────────────►    create_client, index_exists, create_index, bulk_index
retrieval.py ────────────────►    create_client, term_search, aggregate
```

`_path_setup.py`가 `sys.path`에 `Codes/python/`을 추가하여 `import opensearch_handler`를 가능하게 한다. `os_settings.py`가 `_path_setup`을 import하므로, 다른 모듈은 `os_settings`를 먼저 import하면 자동으로 path가 설정된다.

---

## 어떻게 사용하는가? (How)

### 1. 데이터 모델

```python
# models.py
class KnowhowItem(BaseModel):
    knowhow_no: int
    KNOWHOW_ID: str = ""
    knowhow: str             # 원본 텍스트
    user_id: str
    user_name: str
    user_department: str

class EnrichedKnowhow(KnowhowItem):
    summary: str = ""        # LLM 생성 요약
    category: str = ""       # LLM 분류 카테고리
    keywords: list[str] = [] # LLM 추출 키워드
```

카테고리 종류: `용어/약어`, `프로세스`, `트러블슈팅`, `매뉴얼`, `부품정보`, `설정/구성`, `기타`

### 2. 멀티 클러스터 설정

```python
# os_settings.py
CLUSTERS = {
    "opensearch-dev": {
        "host": "localhost", "port": 9200,
        "user": "admin", "password": "admin",
        "use_ssl": True,
        "index": "knowhow", "bulk_chunk": 500,
        "shards": 1, "replicas": 0,        # 단일 데이터 노드
    },
    "es-prod": {
        "host": "...", "port": 9200,
        "user": "elastic", "password": "...",
        "use_ssl": False,
        "index": "knowhow", "bulk_chunk": 500,
        "shards": 2, "replicas": 1,        # 3 데이터 노드
    },
}

# 환경 변수로 클러스터 선택
# KNOWHOW_CLUSTER=es-prod python index.py
ACTIVE_CLUSTER = os.environ.get("KNOWHOW_CLUSTER", "opensearch-dev")
```

`get_connection_config()`가 활성 클러스터 설정을 `opensearch_handler.ConnectionConfig` 객체로 변환한다.

### 3. LLM Enrichment 파이프라인 (extract_v2.py)

```bash
python extract_v2.py
```

- `sample_data/*.json`에서 원본 데이터 로드
- 각 항목을 사내 LLM(`gpt-oss-20b`)에 보내 summary, category, keywords 추출
- 결과를 `processed_data/progress.jsonl`에 한 줄씩 저장 (중단 후 재개 가능)
- 이미 처리된 `KNOWHOW_ID`는 스킵

### 4. OpenSearch 색인 (index.py)

```bash
python index.py
```

핵심 흐름:

```python
config = get_connection_config()
client = osh.create_client(config=config)

# 인덱스 생성 (없으면)
if not osh.index_exists(client, OS_INDEX):
    osh.create_index(client, OS_INDEX,
        mappings=INDEX_SETTINGS["mappings"],
        settings=INDEX_SETTINGS["settings"])

# keywords가 비어 있는 항목 필터링 후 bulk 색인
docs = [item.model_dump(exclude={"knowhow_no"})
        for item in enriched if item.keywords]
success, errors = osh.bulk_index(client, OS_INDEX, docs,
    chunk_size=config.bulk_chunk)
```

인덱스 매핑에서 `knowhow`와 `summary` 필드에 **nori 한국어 분석기**를 적용하고, `keywords`, `category`, `user_id` 등은 `keyword` 타입으로 exact match를 지원한다.

### 5. 하이브리드 검색 (retrieval.py)

메인 검색 함수 `retrieve()`는 세 가지 신호를 결합한다:

```python
should = [
    {"terms":  {"keywords": normalized, "boost": 3.0}},   # exact match
    {"match":  {"knowhow": {"query": text, "boost": 2.0}}},  # full-text
    {"match":  {"summary": {"query": text, "boost": 1.5}}},  # full-text
]
# minimum_should_match: 1 → 최소 하나의 조건에 매칭되어야 함
```

| Boost | 필드 | 검색 방식 | 이유 |
|-------|------|-----------|------|
| 3.0 | keywords | exact term match | 정확한 키워드 일치가 가장 신뢰도 높음 |
| 2.0 | knowhow | full-text (nori) | 원본 텍스트에서 의미 매칭 |
| 1.5 | summary | full-text (nori) | LLM 요약에서 보조 매칭 |

유틸리티 함수들은 `opensearch_handler`의 범용 함수를 직접 호출:

```python
# 카테고리별 조회
osh.term_search(client, OS_INDEX, "category", category, size=size)

# 전체 키워드 목록 (aggregation)
osh.aggregate(client, OS_INDEX,
    {"unique_keywords": {"terms": {"field": "keywords", "size": 500}}})
```

### 6. 실행 예시

```bash
# Dev 클러스터에서 실행 (기본값)
python extract_v2.py        # LLM enrichment
python index.py             # 색인
python retrieval.py PVD 박막  # 검색 테스트

# Prod 클러스터로 전환
KNOWHOW_CLUSTER=es-prod python index.py
```

---

## 설계 결정 사항

### opensearch_handler 분리 이유

원래 `os_client.py`에 클라이언트 생성, 인덱스 관리, 벌크 색인이 모두 있었다. 이를 범용 `opensearch_handler` 패키지로 분리한 이유:

1. **재사용**: 다른 프로젝트(history-opensearch 등)에서도 동일한 클라이언트/인덱스/검색 패턴을 사용
2. **관심사 분리**: knowhow-elasticsearch에는 도메인 로직(LLM enrichment, 하이브리드 검색 쿼리)만 남김
3. **네임스페이스**: `opensearch/` → `opensearch_handler/`로 이름 변경하여 `opensearchpy` 패키지와 충돌 방지

### retrieve()에서 handler의 bool_search를 쓰지 않는 이유

`opensearch_handler.bool_search()`는 `minimum_should_match` 파라미터를 지원하지 않는다. knowhow 검색은 `minimum_should_match: 1` + `boost` 전략이 핵심이므로, 직접 query body를 구성하여 `client.search()`를 호출한다.

---

## 참고 자료 (References)

- [opensearch-py 공식 문서](https://opensearch.org/docs/latest/clients/python-low-level/)
- [OpenSearch Bool Query](https://opensearch.org/docs/latest/query-dsl/compound/bool/)
- [Nori 한국어 분석기](https://opensearch.org/docs/latest/analyzers/language-analyzers/#korean-nori)

## 관련 문서

- [OpenSearch Python 클라이언트 활용](./python-client.md) - Bulk helpers, Async, 에러 핸들링
- [하이브리드 검색](./hybrid-search.md) - 벡터 + 키워드 결합 전략
- [키워드 검색 (BM25)](./keyword-search-bm25.md) - Full-text 검색, 분석기
- [실습 코드](../../../Codes/python/knowhow-elasticsearch/)
- [opensearch_handler 코드](../../../Codes/python/opensearch_handler/)
