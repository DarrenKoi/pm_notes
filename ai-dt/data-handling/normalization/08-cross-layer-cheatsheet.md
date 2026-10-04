---
tags: [normalization, cheatsheet, cross-layer, rdb, mongodb, opensearch, redis, rag, ontology]
level: intermediate
last_updated: 2026-05-02
reviewed_on: 2026-10-04
review_status: reviewed_with_limits
document_type: learning_note
---

# 정규화 Cross-Layer Cheatsheet

> 같은 사실(fact)이 RDB 원천부터 RAG chunk까지 어떻게 형태를 바꾸면서도 같은 정체성을 유지하는지를 한 도메인 사례로 추적한다.

> [!info] 학습 예시와 검토 범위 — 2026-10-04
> 장비/알람/매뉴얼 값은 가상 설계다. 아래 계약은 DB DDL·실제 검색 mapping·모델 실행·회사 장비 조치 지침이 아니다. 제품 조건은03~05, RAG/의미 조건은06~07과 대조한다. 원천 코드·업무 기록을 고쳐 쓰는 작업이 아니다.

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
[
{
  "id": "term:cvd",
  "type": "Process",
  "canonical_label": "Chemical Vapor Deposition",
  "preferred_label_ko": "화학기상증착",
  "aliases": ["CVD", "chemical vapor deposition"],
  "broader": ["term:deposition"],
  "related_equipment": ["equipment:cvd-2000"]
},
{
  "id": "equipment:cvd-2000",
  "type": "Equipment",
  "model": "CVD-2000",
  "process_type": "term:cvd",
  "alarm_codes": ["alarm:a-17", "alarm:a-18"]
}
]
```

이 layer는 승인한 식별/용어 계약을 저장하는 설계다. JSON 자체가 ground truth를 증명하지 않는다. 이 예는 CVD-2000 모델의 장비 한 대만 다룬다. 실제 여러 장비에는 모델 ID와 serial/asset 기반 instance ID를 나누고 알람 A-17도 모델 namespace를 포함해 구분한다. 아래 가상 ID는 명시적 매핑이 있는 경우에만 같은 대상을 뜻한다.

### Layer 1. RDB (트랜잭션 원천)

```text
equipments(equipment_id PK, model_code, process_type_code, installed_at)
alarm_codes(equipment_model_code, alarm_code, severity, default_action_code,
            PK(equipment_model_code, alarm_code))
alarm_events(event_id PK, equipment_id FK, equipment_model_code, alarm_code, occurred_at, status,
             FK(equipment_model_code, alarm_code) -> alarm_codes)
maintenance_actions(action_id PK, event_id FK, action_code, performed_by, performed_at)
```

정규화 포인트:

- `equipment_id`는 ontology의 `equipment:cvd-2000`과 매핑되도록 안정 키 사용
- 알람 코드는 모델별로 재사용될 수 있어 `(equipment_model_code, alarm_code)` 복합키/FK로 구분한다. event의 모델과 equipment의 실제 모델 일치도 별도 제약/업무 검증이 필요하다.
- `alarm_events`는 사건 객체로 분리하여 이력 보존

### Layer 2. MongoDB (운영 문서 + 매뉴얼 메타)

`manuals` collection (reference + 특정 판본 메타 예시):

```json
{
  "_id": "manual_abc_100@2026-05-02",
  "source_doc_id": "manual_abc_100",
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

`glossary_terms` collection (ontology-lite; 04의 terms validator와 다른 구조):

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
- equipment_id의 타입/매핑과 조회 구현이 일치해야 역참조할 수 있다. ID만으로 cross-DB JOIN이 실행되지는 않는다.
- 여러 매뉴얼 판본을 보존하려면 논리 source_doc_id와 판본별 _id/version을 구분한다. is_latest는 승인 manifest로 관리하며 원자적인 게시/reader 계약을 정한다.

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
- 여기서 canonical_terms는 ID 배열이다. label을 저장한03/04 예와 동일 계약이 아니므로 label/ID 필드 또는 adapter를 명시한다. synonym·query 확장/filter를 구현해야 약어/한국어 검색에 사용된다.
- process_type/text와 process_type_code/keyword의 실제 mapping·query를 구성해야 한다. JSON 필드만으로 fuzzy/exact 기능이 설정되지는 않는다.
- _routing_key는 이 예의 임의 필드이며 routing 요청 파라미터가 아니다. 실제 custom routing은 색인/조회/delete 요청에도 같은 routing 값을 지정하는 계약이 필요하다.

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
  "embedding_preview": [0.012, -0.031],
  "embedding_dimension": 1024,
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

- 같은 원문/청킹 revision에서 위치 ID를 재현하고 실제 저장 key에는 source_doc_id/version/chunking_revision도 포함해 버전 충돌을 막는다.
- entity_ids는 여러 참조의 배열이며 원천과 타입/namespace 매핑이 있어야 추적할 수 있다. 무조건1:1 관계가 아니다.
- version/is_latest의 승인·갱신 규칙과 실제 mapping/filter·폐기 정책을 적용해야 최신 조회가 된다.
- citation은 `(source_doc_id, version)`의 정확한 매뉴얼 판본을 역참조한다. 최신 _id 하나로 과거 citation을 덮어쓰지 않는다.
- BAAI 공식 BGE-M3 모델 카드의 dense dimension은1024다. 위 preview2개는 생략 표시용이며 실제 embedding 필드가 아니다. full vector1024/모델 revision·전처리·거리 함수와 index/query 계약을 일치시킨다. 모델은 실행하지 않았다.

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
SET cache:rag_answer:sha256:<context_hash> "{...}" EX 3600
```

정규화 포인트:

- alias key는 입력 표기, value는 canonical ID
- 응답 캐시 key에는 검증한 query 의미/원문·tenant/permission revision·근거 corpus revision·모델/프롬프트 revision을 포함한다. 질문 문자열만으로 재사용하지 않는다.
- 원천/권한/문서 변경 때 객체 cache와 관련 alias·응답 cache를 무효화/버전 구분한다. equipment 한 key를 DEL하는 것만으로 관련 응답까지 삭제되지 않는다. TTL3600은 정합성 보장이 아니다.

## 한 사실의 흐름 (End-to-End)

질문: "CVD 알람 A-17 어떻게 처리해?"

```text
1. Query 정규화 (애플리케이션 + Redis)
   raw_query: "CVD 알람 A-17 어떻게 처리해?"
   ↓ Redis: GET term_alias:cvd → term:cvd
   ↓ Redis: GET equipment_alias:cvd-2000 (질문에 모델이 없어 미확정, 자동 확정 금지)
   normalized_query: {
     intent: "procedure",
     canonical_terms: ["term:cvd"],
     entity_ids: ["alarm:a-17"],
     date_range: null
   }

2. Hybrid Retrieval (OpenSearch RAG chunk index)
   BM25: text match "A-17"
   Vector: embedding(query) → top-k chunks
   Filter: 승인한 model/entity 범위 + canonical_terms ∋ term:cvd, doc_type=MANUAL, 승인 판본 + 인증 주체의 ACL
   ↓ Score normalization (min-max)
   ↓ Reranker

3. Citation Resolution (MongoDB)
   (chunk.source_doc_id, chunk.version) → manuals의 정확한 판본 → title, owner_team, version
   chunk.entity_ids → equipments → 현재 장비 상태

4. LLM 답변 생성
   Context: 검색 chunk + 용어 정의(term:cvd) + 매뉴얼 메타
   Citations: [chunk_id, source_doc_id, page, section_path, version]

5. 응답 캐시 (Redis)
   SET cache:rag_answer:sha256:<context_hash> ... EX 3600 (미확정 모델/권한/freshness이면 재사용 보류)
```

ID 계약은 추적의 출발점이다. 원천 판본·위치·삭제/권한 상태와 실제 resolver가 확인돼야 추적할 수 있다. 이 질문만으로 특정 장비의 조치 지침을 확정하지 않으며 충돌 근거·미확정 객체는 답변에 표시하거나 조치를 보류한다.

## Layer별 책임 요약표

| Layer | 정규화 단위 | 책임 | "원천"인가? |
|-------|-------------|------|-------------|
| Ontology | canonical ID, alias, taxonomy | 같은 개념을 같은 ID로 부르는 규약 | 의미의 원천 |
| RDB | table, FK, dependency | 트랜잭션 사실, 무결성 | 사건의 원천 |
| MongoDB | collection, embed/reference, schema validation | 문서/매뉴얼/glossary 저장, 운영 메타 | 컨텐츠 원천 |
| OpenSearch | analyzer, normalizer, nested, projection | 검색 좌표계, hybrid score 결합 | 아니오 (재생성) |
| Vector Index | chunk_id, embedding metadata | 의미 검색 단위 | 아니오 (재생성) |
| Redis | key prefix, alias, TTL | 빠른 조회/무효화 | 아니오 (캐시) |

> 원천을 가리지 못하면 "여러 군데서 다 갱신했는데 답이 틀린다"는 상황이 발생한다. 사실 종류별 변경 권한과 source of truth를 정한다는 의미이며 사건 DB·매뉴얼 파일·승인 glossary가 하나의 물리 DB여야 한다는 뜻은 아니다. 위 표는 이 예의 역할 배치다.

## 흔한 실패 패턴

1. **ID 매핑 누락** — RDB eq_2000/OpenSearch CVD2000/RAG cvd-2000의 namespace·동일인 매핑이 없으면 추적이 끊긴다. 서로 다른 저장 ID라도 검증한 매핑이 있으면 사용 가능하다.
2. **query/색인 용어 계약 불일치** — alias만 저장하고 검색어 변환/ID filter를 적용하지 않으면 누락 가능. 애플리케이션이 확장 query를 보내는 경우 synonym filter 자체가 항상 필수인 것은 아니다.
3. **버전 누락** — chunk에 `version`이 없으면 매뉴얼 개정 시 stale chunk가 retrieval에 섞인다.
4. **응답 캐시 문맥 누락** — query hash만 공유하면 다른 권한·근거 버전·모델 답변을 재사용할 수 있다. 공백/대소문자 차이가 의미상 동등한지도 확인하고 문맥/원문을 함께 구분한다.
5. **projection 재생성 계약 누락** — 이 예에서 원천/판본/재색인 경로를 잃으면 복구가 어렵다. 모든 OpenSearch 사용이 원천 역할을 금지한다는 뜻은 아니다.
6. **Vector embedding model을 mix** — 같은 인덱스에 dimension/model이 다른 embedding 혼재. 거리 비교 무의미.

## 체크리스트

- [ ] 모든 layer에서 같은 객체가 같은 canonical ID를 사용하는가?
- [ ] alias / synonym / canonical term이 ontology, MongoDB, Redis, OpenSearch에 일관되게 전파되는가?
- [ ] chunk와 검색 문서에 `source_doc_id`, `version`, `is_latest`가 있는가?
- [ ] 응답 캐시 key가 검증한 query·권한·근거·모델/프롬프트 revision을 구분하는가?
- [ ] 각 projection layer(OpenSearch, Redis, Vector index)에 재생성 경로가 정의되어 있는가?
- [ ] embedding model/version이 layer 안에서 단일하게 유지되는가?
- [ ] 원천 변경 이벤트가 어떤 projection을 무효화하는지 매핑이 있는가?

## 응답 캐시 key의 로컬 계약 예

아래는 외부 서비스를 부르지 않는 학습용 Python 함수다. 각 revision과 tenant는 신뢰한 서버가 확인한 값을 전달해야 한다. context_verified도 LLM 후보가 아닌 그 검증 결과이며 False/None이면 key를 만들지 않는다. 미확인/누락은 cache 재사용 보류로 처리한다. hash는 암호화나 ACL 검사가 아니며 읽기 시 현재 권한과 근거 상태를 다시 확인한다.

```python
import hashlib
import json


def make_answer_cache_key(*, raw_query: str, normalized_query: str,
                          tenant_id: str, permission_revision: str,
                          corpus_revision: str, model_revision: str,
                          prompt_revision: str, context_verified: bool) -> str:
    if context_verified is not True:
        raise ValueError("cache 문맥 미확인: 재사용 보류")
    parts = {
        "raw_query": raw_query, "normalized_query": normalized_query,
        "tenant_id": tenant_id, "permission_revision": permission_revision,
        "corpus_revision": corpus_revision, "model_revision": model_revision,
        "prompt_revision": prompt_revision,
    }
    if any(not isinstance(v, str) or not v.strip() for v in parts.values()):
        raise ValueError("미확인 cache 문맥: 재사용 보류")
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return "cache:rag_answer:sha256:" + hashlib.sha256(payload).hexdigest()
```

원문도 key에 반영해 과도한 의미 병합을 피하는 예다. 같은 문맥은 같은 key, tenant/permission/corpus/model/prompt가 바뀌면 다른 key가 된다. 이 함수가 원천 갱신·무효화·실제 TTL·인증을 구현한 것은 아니다.

## 검토 결과

2026-10-04: 원래 도메인/layer/흐름/실패/체크 항목을 보존했다. 두 연속 JSON을 배열로 묶고 알람 복합키·매뉴얼 판본·BGE-M3 preview·routing·field 계약·cache 문맥을 정정했다. 가상 장비 내용은 실제 조치 지침이 아니다. Claude pane_not_found로 완전 통합·업무별 ID/동기화/권한/cache 구현은 보류했다. 실제 모델/DB/검색/캐시는 미실행이다. [정리 기록](../organization-log.md)을 따른다.

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
- [MongoDB Vector Search Overview](https://www.mongodb.com/docs/vector-search/)
- [Redis Vector Search Concepts](https://redis.io/docs/latest/develop/ai/search-and-query/vectors/)

- [BAAI BGE-M3 공식 모델 카드, 2026-10-04 확인](https://huggingface.co/BAAI/bge-m3)
