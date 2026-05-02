---
tags: [normalization, cheatsheet, cross-layer, rdb, mongodb, opensearch, redis, rag, ontology]
level: intermediate
last_updated: 2026-05-02
status: in-progress
---

# 정규화 Cross-Layer Cheatsheet

> 같은 사실(fact)이 RDB 원천부터 RAG chunk까지 어떻게 형태를 바꾸면서도 같은 정체성을 유지하는지를 한 도메인 사례로 추적한다.

## 왜 필요한가? (Why)

각 문서(03 ~ 07)는 한 저장소 안에서 정규화가 어떻게 보이는지를 다룬다. 실무에서는 같은 사실이 여러 저장소를 동시에 통과한다. 한 객체가 OpenSearch에는 어떻게 들어가고, MongoDB에는 어떻게 저장되며, Redis 캐시에서는 어떻게 보이고, RAG chunk 메타데이터로는 어떻게 흘러가는지 한눈에 정렬하지 않으면 다음 문제가 생긴다.

- 같은 객체에 layer마다 다른 ID가 붙어 추적 불가
- canonical term이 한 layer에만 적용되어 검색 누락
- 한 layer에서 갱신했는데 다른 layer에 stale 값이 남음
- RAG citation이 원천 객체로 역추적되지 않음

이 cheatsheet는 한 가지 도메인 사례를 끝까지 따라가서 layer별 책임 경계를 명확히 한다.

## 도메인 사례

> "CVD-2000 장비의 Alarm A-17 발생 원인과 조치 방법"이라는 사용자 질문에 답하는 RAG 시스템

핵심 객체:

- **장비**: `equipment:cvd-2000`
- **알람 코드**: `alarm:a-17`
- **공정 용어**: `term:cvd` (Chemical Vapor Deposition)
- **소스 매뉴얼**: `manual_abc_100` v2026-05-02

## Layer별 정규화 매핑

### Layer 0. Ontology / Glossary (개념 모델)

```json
{
  "id": "term:cvd",
  "type": "Process",
  "canonical_label": "Chemical Vapor Deposition",
  "preferred_label_ko": "화학기상증착",
  "aliases": ["CVD", "chemical vapor deposition"],
  "broader": ["term:deposition"],
  "related_equipment": ["equipment:cvd-2000"]
}

{
  "id": "equipment:cvd-2000",
  "type": "Equipment",
  "model": "CVD-2000",
  "process_type": "term:cvd",
  "alarm_codes": ["alarm:a-17", "alarm:a-18"]
}
```

이 layer는 "어떤 ID가 같은 개념을 가리키는가"의 ground truth를 정한다. 아래 모든 layer는 이 ID를 그대로 전파한다.

### Layer 1. RDB (트랜잭션 원천)

```text
equipments(equipment_id PK, model_code, process_type_code, installed_at)
alarm_codes(alarm_code PK, equipment_model_code, severity, default_action_code)
alarm_events(event_id PK, equipment_id FK, alarm_code FK, occurred_at, status)
maintenance_actions(action_id PK, event_id FK, action_code, performed_by, performed_at)
```

정규화 포인트:

- `equipment_id`는 ontology의 `equipment:cvd-2000`과 매핑되도록 안정 키 사용
- `alarm_code`는 장비 모델 분류와 분리 (코드 자체는 모델 dependent하지만 사건은 instance dependent)
- `alarm_events`는 사건 객체로 분리하여 이력 보존

### Layer 2. MongoDB (운영 문서 + 매뉴얼 메타)

`manuals` collection (reference + snapshot 혼합):

```json
{
  "_id": "manual_abc_100",
  "title": "CVD-2000 운영 매뉴얼",
  "equipment_id": "equipment:cvd-2000",
  "version": "2026-05-02",
  "is_latest": true,
  "source_uri": "s3://kb/manual/abc_100.pdf",
  "source_hash": "sha256:...",
  "language": "ko",
  "owner_team": "process-eng",
  "security_level": "internal"
}
```

`terms` collection (ontology-lite):

```json
{
  "_id": "term:cvd",
  "canonical_label": "Chemical Vapor Deposition",
  "aliases": ["CVD", "화학기상증착"],
  "broader": ["term:deposition"]
}
```

정규화 포인트:

- 매뉴얼 자체는 reference 모델 (장비/팀과 독립)
- 텍스트 본문은 별도 chunk collection으로 분리 (아래 Layer 4)
- `equipment_id`로 RDB와 ontology 어느 쪽으로도 join 가능

### Layer 3. OpenSearch (검색 projection)

`alarm_search` index (반정규화된 검색 문서):

```json
{
  "doc_type": "ALARM_EVENT",
  "event_id": "evt_91021",
  "equipment_id": "equipment:cvd-2000",
  "equipment_model": "CVD-2000",
  "process_type": "Chemical Vapor Deposition",
  "process_type_code": "term:cvd",
  "alarm_code": "A-17",
  "alarm_description": "Chamber pressure deviation",
  "severity": "HIGH",
  "occurred_at": "2026-05-02T03:14:00+09:00",
  "canonical_terms": ["term:cvd", "term:chamber_pressure"],
  "_routing_key": "equipment:cvd-2000"
}
```

정규화 포인트:

- 원천 ID(`event_id`, `equipment_id`)는 모두 보존
- `canonical_terms`로 약어/한국어 검색 동시 지원
- `process_type` (text)과 `process_type_code` (keyword) 분리해 fuzzy + exact 모두 가능

### Layer 4. RAG Chunk Index (OpenSearch 또는 MongoDB Vector Search)

```json
{
  "chunk_id": "manual_abc_100:p012:s03:c02",
  "source_doc_id": "manual_abc_100",
  "version": "2026-05-02",
  "is_latest": true,
  "page": 12,
  "section_path": ["알람", "A-17", "조치"],
  "text": "A-17은 챔버 압력 이상...",
  "embedding": [0.012, -0.031],
  "embedding_model": "bge-m3",
  "embedding_version": "2026-05-02",
  "language": "ko",
  "entity_ids": ["equipment:cvd-2000", "alarm:a-17"],
  "canonical_terms": ["term:cvd"],
  "doc_type": "MANUAL",
  "security_level": "internal"
}
```

정규화 포인트:

- `chunk_id`는 재생성해도 안정적 (source_doc + page + section + index)
- `entity_ids`로 RDB 객체와 1:1 추적 가능
- `version` + `is_latest`로 최신성 필터링
- citation 시 `source_doc_id`로 manuals collection 역참조

### Layer 5. Redis (alias 캐시 + 응답 캐시)

용어 정규화 캐시:

```text
HSET term:cvd canonical_label "Chemical Vapor Deposition" category "Process"

SET term_alias:cvd term:cvd
SET term_alias:화학기상증착 term:cvd
SET term_alias:chemical_vapor_deposition term:cvd
```

장비 ID 캐시:

```text
HSET equipment:cvd-2000 \
  model "CVD-2000" \
  process_type term:cvd \
  alarm_count "37"

SET equipment_alias:cvd2000 equipment:cvd-2000
SET equipment_alias:cvd-2000 equipment:cvd-2000
```

질의 응답 캐시:

```text
SET cache:rag_answer:sha256:<query_hash> "{...}" EX 3600
```

정규화 포인트:

- alias key는 입력 표기, value는 canonical ID
- 응답 캐시 key는 정규화된 query hash 기준 (raw query 기준이면 hit율 폭락)
- 원천 변경 시 무효화는 entity ID 단위로 (`DEL equipment:cvd-2000` + 관련 alias)

## 한 사실의 흐름 (End-to-End)

질문: "CVD 알람 A-17 어떻게 처리해?"

```text
1. Query 정규화 (애플리케이션 + Redis)
   raw_query: "CVD 알람 A-17 어떻게 처리해?"
   ↓ Redis: GET term_alias:cvd → term:cvd
   ↓ Redis: GET equipment_alias:cvd-2000 (없음, 추론 필요)
   normalized_query: {
     intent: "procedure",
     canonical_terms: ["term:cvd"],
     entity_ids: ["alarm:a-17"],
     date_range: null
   }

2. Hybrid Retrieval (OpenSearch RAG chunk index)
   BM25: text match "A-17"
   Vector: embedding(query) → top-k chunks
   Filter: canonical_terms ∋ term:cvd, doc_type=MANUAL, is_latest=true
   ↓ Score normalization (min-max)
   ↓ Reranker

3. Citation Resolution (MongoDB)
   chunk.source_doc_id → manuals._id → title, owner_team, version
   chunk.entity_ids → equipments → 현재 장비 상태

4. LLM 답변 생성
   Context: 검색 chunk + 용어 정의(term:cvd) + 매뉴얼 메타
   Citations: [chunk_id, source_doc_id, page, section_path, version]

5. 응답 캐시 (Redis)
   SET cache:rag_answer:sha256:<normalized_query_hash> ... EX 3600
```

모든 layer가 같은 ID 체계(`term:cvd`, `equipment:cvd-2000`, `alarm:a-17`, `manual_abc_100`)를 공유하므로 답변에서 사용된 어떤 사실도 원천까지 역추적 가능하다.

## Layer별 책임 요약표

| Layer | 정규화 단위 | 책임 | "원천"인가? |
|-------|-------------|------|-------------|
| Ontology | canonical ID, alias, taxonomy | 같은 개념을 같은 ID로 부르는 규약 | 의미의 원천 |
| RDB | table, FK, dependency | 트랜잭션 사실, 무결성 | 사건의 원천 |
| MongoDB | collection, embed/reference, schema validation | 문서/매뉴얼/glossary 저장, 운영 메타 | 컨텐츠 원천 |
| OpenSearch | analyzer, normalizer, nested, projection | 검색 좌표계, hybrid score 결합 | 아니오 (재생성) |
| Vector Index | chunk_id, embedding metadata | 의미 검색 단위 | 아니오 (재생성) |
| Redis | key prefix, alias, TTL | 빠른 조회/무효화 | 아니오 (캐시) |

> 원천을 가리지 못하면 "여러 군데서 다 갱신했는데 답이 틀린다"는 상황이 발생한다. 원천은 하나, projection은 여러 개라는 규칙을 layer 별로 명시하는 것이 cross-layer 정규화의 핵심이다.

## 흔한 실패 패턴

1. **Layer마다 다른 ID** — RDB는 `eq_2000`, OpenSearch는 `CVD2000`, RAG는 `cvd-2000`. 검색 결과를 RDB로 join 못한다.
2. **약어를 한 layer에서만 풀음** — Redis alias만 있고 OpenSearch synonym은 없으면 BM25 검색에서 누락된다.
3. **버전 누락** — chunk에 `version`이 없으면 매뉴얼 개정 시 stale chunk가 retrieval에 섞인다.
4. **응답 캐시 key가 raw query 기준** — 띄어쓰기, 대소문자만 달라도 cache miss. 정규화된 query hash를 key로 써야 한다.
5. **OpenSearch를 원천처럼 사용** — 재색인 경로가 없어 인덱스 손상 시 복구 불가.
6. **Vector embedding model을 mix** — 같은 인덱스에 dimension/model이 다른 embedding 혼재. 거리 비교 무의미.

## 체크리스트

- [ ] 모든 layer에서 같은 객체가 같은 canonical ID를 사용하는가?
- [ ] alias / synonym / canonical term이 ontology, MongoDB, Redis, OpenSearch에 일관되게 전파되는가?
- [ ] chunk와 검색 문서에 `source_doc_id`, `version`, `is_latest`가 있는가?
- [ ] 응답 캐시 key가 정규화된 query hash 기준인가?
- [ ] 각 projection layer(OpenSearch, Redis, Vector index)에 재생성 경로가 정의되어 있는가?
- [ ] embedding model/version이 layer 안에서 단일하게 유지되는가?
- [ ] 원천 변경 이벤트가 어떤 projection을 무효화하는지 매핑이 있는가?

## 관련 문서

- [정규화 핵심 개념](./01-normalization-core.md)
- [모델링 프로세스와 체크리스트](./02-modeling-process-checklist.md)
- [OpenSearch에서의 정규화](./03-opensearch-normalization.md)
- [MongoDB에서의 정규화](./04-mongodb-normalization.md)
- [Redis에서의 정규화](./05-redis-normalization.md)
- [LLM과 RAG에서의 정규화](./06-llm-rag-normalization.md)
- [온톨로지 관점의 정규화](./07-ontology-normalization.md)

## 참고 자료

- [정규화(Normalization)란 무엇인가 - 교과서 너머의 이해](https://wikidocs.net/blog/%40jcnahm/12324/)
- [OpenSearch Normalization Processor](https://docs.opensearch.org/2.15/search-plugins/search-pipelines/normalization-processor/)
- [MongoDB Vector Search Overview](https://www.mongodb.com/docs/atlas/atlas-search/vector-search/)
- [Redis Vector Search Concepts](https://redis.io/docs/latest/develop/ai/search-and-query/vectors/)
