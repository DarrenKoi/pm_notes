---
tags: [opensearch, search-engine, elasticsearch, vector-search]
reviewed_on: 2026-10-04
review_status: partial
document_type: learning_note
level: beginner
last_updated: 2026-02-05
category_major: "AI·DT"
category_middle: "RAG"
category_minor: "OpenSearch 검색"
note_kind: "학습"
classified_on: "2026-10-05"
---

# OpenSearch 기초 (OpenSearch Basics)

> Elasticsearch OSS에서 출발한 오픈소스 검색/분석 엔진이다. 키워드·벡터 검색을 지원하는 배포 구성을 학습한다.

> [!info] 검토 조건 — 2026-10-04
> 학습 예제다. 역사적 작성일은 유지하고 이번 검토일을 별도로 기록했다. 로컬 확인은 Python3.14.2·opensearch-py3.2.0 및 모의 REST 응답에 한정한다. 실제 OpenSearch/Docker·보안·검색 품질·100GB 부하를 검증하지 않았다. 함수는 대상과 입력을 명시해 호출한다. [정리 기록](../organization-log.md)을 함께 읽는다.


## 왜 필요한가? (Why)

### OpenSearch vs Elasticsearch

2021년의 출발과 현재 프로젝트 운영을 구분한다. OpenSearch는 Elasticsearch OSS 7.10.2 계열에서 출발했다. 현재 OpenSearch Software Foundation은 Linux Foundation 아래에서 프로젝트를 지원한다. 특정 회사의 관리형 서비스와 오픈소스 프로젝트의 거버넌스는 다르다. [OpenSearch FAQ](https://opensearch.org/faq/), [Foundation](https://opensearch.org/foundation/), 확인 2026-10-04.

| 항목 | OpenSearch | Elasticsearch |
|---|---|---|
| 라이선스 | 프로젝트 소프트웨어 Apache 2.0 | 배포본 ELv2; 적용되는 무료 소스 부분은 AGPLv3/SSPL/ELv2 선택 조건을 확인 |
| 프로젝트/서비스 | 커뮤니티 프로젝트·재단 지원; Amazon 서비스는 별도 | Elastic 프로젝트/서비스 |
| 벡터 검색 | k-NN 플러그인·배포/engine/판본 확인 | 제공 API·라이선스/판본 조건을 따로 확인. 8.0부터만 가능하다고 단정하지 않음 |
| API 호환성 | OSS7.10.2 출발을 모든 후속 버전 API 호환으로 확대하지 않음 | 클라이언트/인덱스/플러그인별 마이그레이션 검사 필요 |

Elastic은 2024년 일부 소스의 AGPLv3 선택을 추가했다. 기존 표의 EL/SSPL만으로 현재 모든 소스/배포 라이선스를 설명할 수 없다. 해당 파일/배포의 조건을 확인한다. [Elastic 라이선스 FAQ](https://www.elastic.co/pricing/faq/licensing), 확인 2026-10-04. 이 표는 회사 사용의 법적 적합성 판정이 아니다.

### 언제 OpenSearch를 선택하는가?

- **AWS 환경**에서 관리형 서비스 사용 시
- **완전한 오픈소스**가 필요할 때
- **벡터 검색 + 키워드 검색** 모두 필요할 때
- 기존 Elasticsearch OSS 계열 자료를 옮길 때 클라이언트/API·인덱스 버전·플러그인 호환성과 업그레이드 경로를 확인한 경우

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
│  │ (Mgr/Data)  │  │   (Data)    │  │   (Data)    │         │
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
| **Cluster manager** | 클러스터 상태/메타데이터 관리 | `node.roles: [cluster_manager]` |
| **Data** | 데이터 저장, 검색/인덱싱 실행 | `node.roles: [data]` |
| **Ingest** | 데이터 전처리 파이프라인 | `node.roles: [ingest]` |
| **Coordinating** | 요청 라우팅, 결과 집계 | 전용 노드는 `node.roles: []` |

위 그림은 역할을 겸하는 노드의 개념도다. 전용 cluster-manager 노드는 data 역할 없이 문서 shard를 저장하지 않는다. 실제 역할은 조합/판본에 따른다. 이전 `node.master` 등 Boolean 예제를 현재 3.x 설정으로 복사하지 않는다. [노드 설정](https://docs.opensearch.org/latest/install-and-configure/configuring-opensearch/configuration-system/), [클러스터 구성](https://docs.opensearch.org/latest/tuning-your-cluster/index/), 확인 2026-10-04.

### Index, Document, Field

```
Index (인덱스)           → RDBMS의 Database/Table
  └── Document (문서)    → RDBMS의 Row
       └── Field (필드)  → RDBMS의 Column
```

**예시**:
```json
{
  "_index": "products",
  "_id": "1",
  "_source": {
    "name": "노트북",
    "description": "고성능 노트북",
    "price": 1500000,
    "embedding": [0.1, 0.2, 0.3]
  }
}
```

위 JSON은 문서 구조용 3차원 예시다. 1536차원 mapping과 함께 색인하려면 실제 벡터 차원을 맞춰야 한다.

### Shard와 Replica

- **Primary Shard**: 데이터를 나누어 저장하는 단위. 생성 후 단순 `put_settings`로 수를 바꾸지 않는다. 조건을 갖춘 split/shrink 또는 reindex로 새 인덱스를 만드는 경로를 구분한다.
- **Replica Shard**: Primary 복제본. 다른 노드에 배치될 때 장애 대응/읽기 분산에 도움을 줄 수 있다. 단일 노드의 replica1은 같은 primary 옆에 배치되지 못해 yellow일 수 있다. replica0은 단일 노드 실습 설정이며 고가용성을 제공하지 않는다.

```python
shard_example = {"settings": {"number_of_shards": 3, "number_of_replicas": 1}}
# 설정 구조 예시. 실제 노드/데이터 조건과 무관한 권장 샤드 수가 아니다.
```

샤드 크기 10~50GB는 원래 문서의 검증되지 않은 계획값이다. 보편 권장으로 쓰지 않는다. 데이터/segment·벡터 인덱스 메모리·부하·복구 목표로 측정한다. [Split Index](https://docs.opensearch.org/latest/api-reference/index-apis/split/)의 새 인덱스/쓰기 차단 등 조건도 확인한다(2026-10-04).

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
| `object` | JSON 객체 필드 | 객체 배열 내 항목 관계가 필요하면 `nested` 타입/쿼리를 따로 검토 |
| `knn_vector` | 벡터 (k-NN 플러그인) | 유사도 검색 |

### Analyzer (분석기)

텍스트를 검색 가능한 토큰으로 변환하는 과정이다. 아래 한국어 출력은 설명용이며 특정 기본 analyzer의 실제 출력이 아니다. 사용할 analyzer/플러그인과 `_analyze`로 토큰을 확인하고 색인/검색 분석 조건을 맞춘다.

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

원래 2.11.0 예제의 무인증 실습 목적을 유지하되 실행 판본은 직접 지정하고 loopback에만 바인딩한다. 아래는 새 실습 디렉터리 전용이며 사내/공용 서비스 배포 설정이 아니다. 실제 Docker 기동은 이번에 하지 않았다. Security plugin을 쓰는 별도 구성에서는 2.12 이후 demo 초기 admin 비밀번호 조건·TLS/CA·역할 권한을 갖춘다. 기본 비밀번호를 코드에 두지 않는다. OpenSearch와 Dashboards 판본을 맞춘다. [공식 Docker 안내](https://docs.opensearch.org/latest/install-and-configure/install-opensearch/docker/), 확인 2026-10-04.

**docker-compose.yml**:
```yaml
services:
  opensearch:
    image: opensearchproject/opensearch:${OPENSEARCH_VERSION:?실습_판본을_명시하세요}
    environment:
      - discovery.type=single-node
      - bootstrap.memory_lock=true
      - "OPENSEARCH_JAVA_OPTS=-Xms512m -Xmx512m"
      - DISABLE_INSTALL_DEMO_CONFIG=true  # 격리 실습에서만 demo 설치 생략
      - DISABLE_SECURITY_PLUGIN=true  # 격리 로컬 실습 전용
    ulimits:
      memlock:
        soft: -1
        hard: -1
    volumes:
      - opensearch-data:/usr/share/opensearch/data
    ports:
      - "127.0.0.1:9200:9200"  # 격리 로컬 REST 실습
      - "127.0.0.1:9600:9600"  # 판본/플러그인의 관리 endpoint 확인

  opensearch-dashboards:
    image: opensearchproject/opensearch-dashboards:${OPENSEARCH_VERSION:?동일_판본을_명시하세요}
    ports:
      - "127.0.0.1:5601:5601"
    environment:
      - OPENSEARCH_HOSTS=["http://opensearch:9200"]
      - DISABLE_SECURITY_DASHBOARDS_PLUGIN=true

volumes:
  opensearch-data:
```

```bash
# 위 YAML을 기존 데이터가 없는 새 실습 폴더에 저장한다.
# OPENSEARCH_VERSION을 검토한 정확한 이미지 태그로 설정한 후 실행.
docker compose config --quiet
docker compose up -d

# 상태 확인
curl -X GET "http://localhost:9200/_cluster/health?pretty"
```

### 2. Python 클라이언트 설치

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install "opensearch-py==3.2.0"
```

### 3. 클러스터 연결 및 정보 확인

```python
from opensearchpy import OpenSearch

def local_demo_client() -> OpenSearch:
    return OpenSearch(
        hosts=[{"host": "127.0.0.1", "port": 9200}],
        http_compress=True, use_ssl=False, timeout=10,
        max_retries=0, retry_on_timeout=False,
    )

def cluster_summary(client: OpenSearch) -> dict:
    info = client.info()
    health = client.cluster.health()
    return {"version": info["version"]["number"], "status": health["status"],
            "nodes": health["number_of_nodes"]}
# green은 현재 shard 할당 상태이며 백업/보안/검색 품질 판정이 아니다.
```

아래 함수 정의를 순서대로 같은 모듈에서 사용한다. `local_demo_client()`는 위 격리 실습만 대상으로 하며 실제 서비스는 [Python 클라이언트](./python-client.md)의 TLS factory 조건을 따른다. 객체 생성 후 필요한 함수만 호출하고 `finally`에서 `client.close()` 한다.

### 4. 인덱스 생성 및 관리

```python
def new_document_index(client: OpenSearch, index_name: str, dim: int = 1536) -> dict:
    if not isinstance(index_name, str) or not index_name.strip():
        raise ValueError("새 실습 인덱스 이름이 필요합니다")
    if type(dim) is not int or not 1 <= dim <= 16000:
        raise ValueError("양의 차원과 engine/판본 지원 범위를 확인하세요")
    if client.indices.exists(index=index_name):
        raise ValueError("기존 인덱스는 보존합니다")
    body = {
        "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0, "knn": True}},
        "mappings": {"properties": {
            "title": {"type": "text"}, "content": {"type": "text"},
            "category": {"type": "keyword"},
            "embedding": {"type": "knn_vector", "dimension": dim,
                          "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "faiss"}},
        }},
    }
    return client.indices.create(index=index_name, body=body)
# 동시 생성 경쟁의 최종 충돌은 서버 오류로 유지한다. 삭제 후 재생성하지 않는다.
```

기존 인덱스가 있으면 거부하며 자동 삭제/재설정하지 않는다. mapping 1536은 모델 차원 예시다. 모델 ID·차원·전처리를 별도 계약으로 관리한다.3.x 설명에서는 NMSLIB가 deprecated이며 아래는Faiss 설정을 명시한다. SDK의 요청 형식 검증은 실제 서버가 이 mapping을 수용/검색한다는 증거가 아니다. [k-NN 필드 계약](https://docs.opensearch.org/latest/mappings/supported-field-types/knn-vector/)의 차원/engine 조건도 확인한다. [Breaking changes](https://docs.opensearch.org/latest/breaking-changes/), 확인 2026-10-04.

### 5. 문서 CRUD

```python
import math
from numbers import Real

def create_demo_document(client: OpenSearch, index_name: str, vector: list[float],
                         dim: int = 1536) -> dict:
    if not isinstance(vector, list) or len(vector) != dim:
        raise ValueError("임베딩 차원 불일치")
    if any(isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x) for x in vector) or not any(vector):
        raise ValueError("유한한 비영벡터가 필요합니다")
    doc = {"title": "OpenSearch 소개", "content": "OpenSearch는 오픈소스 검색 엔진입니다.",
           "category": "tutorial", "embedding": vector}
    return client.create(index=index_name, id="doc-001", body=doc, refresh="wait_for")

def read_demo_document(client: OpenSearch, index_name: str) -> dict:
    return client.get(index=index_name, id="doc-001")["_source"]

def update_demo_category(client: OpenSearch, index_name: str) -> dict:
    return client.update(index=index_name, id="doc-001", body={"doc": {"category": "guide"}})
# create는 기존 ID에서 충돌한다. index는 같은 ID를 덮어쓸 수 있다.
# 삭제는 별도 승인/데이터 수명 정책의 작업이며 이 예제에서 자동 호출하지 않는다.
```

### 6. 벌크 작업 (Bulk Operations)

단건 요청의 왕복 비용을 줄일 때 bulk를 사용할 수 있다. 아래 100건은 API 학습 fixture다. 전체bulk HTTP가 성공해도 항목별 실패가 있을 수 있다. 기본helpers.bulk는 항목 오류에서 예외를 던지므로 성공 응답으로 단순 처리하지 않는다. 실패 목록에는 원문이 포함될 수 있어 그대로 출력하지 않는다.

```python
from opensearchpy import helpers

def bulk_demo_documents(client: OpenSearch, index_name: str) -> dict:
    actions = (
        {"_op_type": "create", "_index": index_name, "_id": f"bulk-{i}",
         "_source": {"title": f"문서 {i}", "content": f"내용 {i}", "category": "bulk"}}
        for i in range(100)
    )
    success, failed = helpers.bulk(
        client, actions, stats_only=True, raise_on_error=False,
        chunk_size=100, max_chunk_bytes=1024 * 1024,
    )
    return {"success": success, "failed": failed}
# 부분 실패가 있으면 상위 호출자가 실패 처리한다. 재실행 create 충돌을 무시하지 않는다.
# 검색 반영이 필요할 때 명시 refresh; bulk 응답과 refresh/backup은 다른 조건이다.
```

### 7. 클러스터 관리 명령어

```python
def management_snapshot(client: OpenSearch) -> dict:
    # 별도 클러스터 모니터링 권한이 필요할 수 있다. 문서 원문은 출력하지 않는다.
    stats = client.cluster.stats()
    nodes = client.nodes.info()
    return {"documents": stats["indices"]["docs"]["count"],
            "store_bytes": stats["indices"]["store"]["size_in_bytes"],
            "node_roles": [item["roles"] for item in nodes["nodes"].values()]}

def request_demo_replica(client: OpenSearch, index_name: str, replicas: int) -> dict:
    if type(replicas) is not int or replicas < 0:
        raise ValueError("replica 수가 잘못되었습니다")
    return client.indices.put_settings(index=index_name, body={"index": {"number_of_replicas": replicas}})
# replica1을 요청해도 단일 노드에서 할당될 수 없다. 실제 분리 노드/복구 검증 필요.
# 검색 반영은 client.indices.refresh(index=index_name)를 명시 호출한다.
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
