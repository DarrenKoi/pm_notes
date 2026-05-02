---
tags: [ontology, normalization, taxonomy, glossary, knowledge-graph, data-modeling]
level: advanced
last_updated: 2026-05-02
---

# 온톨로지 관점의 정규화

> 온톨로지가 "도메인에 어떤 것들이 존재하며 어떻게 관계 맺는가"를 정의한다면, 정규화는 그 개념들이 데이터 구조 안에서 올바른 책임 위치를 갖도록 만드는 논리 모델링 규율이다.

## 온톨로지와 정규화의 차이

| 구분 | 온톨로지 | 정규화 |
|------|----------|--------|
| 핵심 질문 | 이 도메인에는 어떤 개념과 관계가 있는가? | 이 속성은 어느 객체의 사실인가? |
| 산출물 | class, property, relationship, constraint, term | table/collection/document structure, key, dependency |
| 관심사 | 의미, 분류, 추론, 공유 어휘 | 중복, 종속성, 무결성, 변경 비용 |
| 위치 | 개념 모델/의미 모델 | 논리 모델/물리 모델로 가는 중간 규율 |
| 예시 | `Customer`, `Order`, `placesOrder` | `customers`, `orders`, `order_items` |

둘은 경쟁하지 않는다. 온톨로지는 무엇을 모델링해야 하는지 알려주고, 정규화는 그것을 어떻게 안정적인 데이터 구조로 배치할지 알려준다.

## 모델링 계층에서의 위치

```text
1. 업무 사건/계약 관점
   어떤 사건이 데이터를 발생시켰는가?

2. 온톨로지/개념 모델
   어떤 객체, 관계, 분류, 제약이 존재하는가?

3. 정규화된 논리 모델
   각 속성은 어느 객체/관계/사건에 종속되는가?

4. 물리 모델
   RDB table, MongoDB collection, OpenSearch index, Redis key

5. 목적별 projection
   검색 문서, 캐시, 분석 mart, RAG chunk index
```

정규화는 2번과 4번 사이에 있다. 온톨로지를 데이터베이스 구조로 옮길 때 의미가 섞이지 않도록 잡아주는 역할이다.

## 정규화는 ontology를 검증한다

정규화를 하다 보면 ontology의 빈틈이 드러난다.

### 예시 1. NULL이 많다면 subtype이 빠졌을 수 있다

```text
payments(payment_id, payment_type, card_number, bank_account, mobile_provider)
```

온톨로지 관점:

```text
Payment
  - CardPayment
  - BankTransfer
  - SimplePayment
```

정규화된 모델:

```text
payments(payment_id, order_id, amount, payment_type)
card_payments(payment_id, card_token)
bank_transfers(payment_id, bank_code, account_hash)
simple_payments(payment_id, provider_code)
```

NULL은 단순 데이터 품질 문제가 아니라 "분류가 모델에 반영되지 않았다"는 신호다.

### 예시 2. N:M 관계는 relation class일 수 있다

```text
Employee -- participatesIn -- Project
```

처음에는 단순 관계처럼 보이지만 역할, 기간, 기여도, 평가가 붙으면 관계 자체가 객체가 된다.

```text
ProjectMembership
  - employee
  - project
  - role
  - validFrom
  - validTo
```

정규화된 모델:

```text
project_memberships(
  membership_id,
  employee_id,
  project_id,
  role_code,
  valid_from,
  valid_to
)
```

정규화는 relation class가 필요한 지점을 드러낸다.

### 예시 3. 이력은 event ontology가 필요하다

```text
customers(customer_id, grade_code, grade_changed_at)
```

이 구조는 표면적으로 3NF 위반이 아닐 수 있다. 하지만 "등급 변경"이 업무적으로 중요한 사건이라면 별도 event 객체가 필요하다.

```text
CustomerGradeChange
  - customer
  - previousGrade
  - newGrade
  - changedAt
  - reason
```

정규화된 모델:

```text
customer_grade_histories(
  history_id,
  customer_id,
  previous_grade_code,
  new_grade_code,
  changed_at,
  reason_code
)
```

정규화만으로는 모든 사건을 발견하지 못한다. 온톨로지와 업무 계약 관점이 먼저 "기록해야 할 사건"을 드러내야 한다.

## Ontology-lite: glossary, taxonomy, canonical ID

모든 프로젝트가 OWL/RDF 수준의 형식 온톨로지를 도입할 필요는 없다. RAG나 검색 시스템에서는 ontology-lite가 실용적이다.

```json
{
  "id": "term:cvd",
  "type": "Process",
  "canonical_label": "Chemical Vapor Deposition",
  "preferred_label_ko": "화학기상증착",
  "aliases": ["CVD", "chemical vapor deposition"],
  "broader": ["term:deposition"],
  "related": ["term:thin_film"],
  "definition": "기체 원료의 화학 반응으로 박막을 형성하는 공정"
}
```

구성 요소:

| 요소 | 역할 |
|------|------|
| canonical ID | 같은 개념을 하나로 식별 |
| preferred label | 공식 표기 |
| aliases | 약어, 동의어, 다국어 표현 |
| broader/narrower | taxonomy 계층 |
| related | 연관 개념 |
| definition | LLM 컨텍스트에 넣을 짧은 정의 |
| source | 정의의 근거 |

이 ontology-lite는 MongoDB에 저장하고, Redis에 alias cache를 두고, OpenSearch/RAG chunk에는 `canonical_terms`로 투영할 수 있다.

## 정규화와 Knowledge Graph

Knowledge Graph에서는 triple 형태로 사실을 표현한다.

```text
customer:1001  placesOrder  order:9001
order:9001     hasItem      product:p1
product:p1     belongsTo    category:laptop
```

이 구조는 매우 정규화된 의미 표현에 가깝다. 하지만 애플리케이션 조회에는 비용이 클 수 있다. 그래서 보통 다음처럼 함께 쓴다.

```text
Knowledge Graph:
  의미 관계, 추론, 연결 탐색

정규화된 DB:
  트랜잭션 원천, 무결성, 업무 처리

검색 인덱스:
  사용자 질의, RAG retrieval

캐시:
  빈번한 조회와 alias lookup
```

정규화는 KG와 RDB 사이의 번역 규칙을 안정화한다. class는 table/collection으로, property는 column/field로, relationship은 foreign key나 edge로 대응될 수 있다.

## LLM/RAG에서 ontology와 정규화의 연결

RAG에서 ontology는 검색 전후에 모두 쓰인다.

```text
질문:
  "CVD 알람 조치 방법"

Ontology/Glossary:
  CVD -> Chemical Vapor Deposition
  type -> Process
  related equipment -> CVD chamber

정규화된 검색:
  canonical_terms: term:cvd
  entity_type: equipment/process
  doc_type: manual, incident

LLM 컨텍스트:
  검색 chunk + 용어 정의 + 관련 객체 정보
```

효과:

- 약어와 동의어로 인한 누락을 줄인다.
- 잘못된 동명이인 객체를 줄인다.
- 검색 결과를 도메인 개념별로 rerank할 수 있다.
- LLM 답변에서 용어 정의와 출처를 일관되게 유지한다.

## 포지션 정리

정규화의 ontology 관점 포지션은 다음과 같다.

1. 정규화는 ontology 자체가 아니다.
2. 정규화는 ontology를 논리 데이터 모델로 구현할 때 의미가 섞이지 않게 하는 규율이다.
3. 정규화 과정은 숨은 객체, 관계 객체, subtype, event를 드러내므로 ontology를 개선하는 피드백 루프가 된다.
4. 온톨로지는 RAG에서 query expansion, entity linking, metadata filtering, context grounding을 가능하게 한다.
5. 정규화된 데이터와 ontology가 함께 있어야 LLM이 "그럴듯한 텍스트"가 아니라 "식별 가능한 사실"을 근거로 답할 수 있다.

## 체크리스트

- [ ] 도메인의 핵심 class와 relationship을 먼저 정의했는가?
- [ ] 같은 개념을 가리키는 용어와 약어가 canonical ID로 연결되는가?
- [ ] 정규화 과정에서 드러난 관계 객체를 ontology에 반영했는가?
- [ ] NULL이 많은 구조를 subtype 누락 신호로 검토했는가?
- [ ] 이력과 사건을 별도 event class로 볼 필요가 있는가?
- [ ] ontology의 ID가 MongoDB, OpenSearch, Redis, RAG chunk에 일관되게 전달되는가?
- [ ] LLM이 생성한 용어/관계 후보를 검증해 ontology에 반영하는 절차가 있는가?

## 참고 자료

- [정규화(Normalization)란 무엇인가 - 교과서 너머의 이해](https://wikidocs.net/blog/%40jcnahm/12324/)
- [보조 용어 사전 DB](../../rag/auxiliary-glossary-db.md)
- [OpenSearch Object Field Types](https://docs.opensearch.org/latest/mappings/supported-field-types/object-fields/)
- [MongoDB Vector Search Overview](https://www.mongodb.com/docs/atlas/atlas-search/vector-search/)
