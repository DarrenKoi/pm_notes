---
tags: [mongodb, normalization, schema-design, embedding, reference, vector-search, rag]
level: intermediate
last_updated: 2026-05-02
status: in-progress
---

# MongoDB에서의 정규화

> MongoDB에서는 정규화와 반정규화가 "테이블 분해"가 아니라 embedding과 reference 사이의 선택으로 나타난다.

## MongoDB 모델링의 출발점

MongoDB는 문서 단위로 데이터를 저장한다. 관련 데이터를 한 문서에 embed하면 한 번의 읽기로 가져올 수 있고, reference로 분리하면 중복을 줄이고 독립 변경을 쉽게 만든다.

```text
Embedding:
  주문 문서 안에 배송지, 주문항목, 결제 요약을 함께 저장

Reference:
  orders.customer_id -> customers._id
  order_items.product_id -> products._id
```

MongoDB 공식 문서도 embedding은 읽기 성능과 단일 원자적 업데이트에 유리하고, reference는 정규화된 모델로서 중복을 줄이고 복잡한 관계나 자주 바뀌는 데이터를 다루는 데 적합하다고 설명한다.

## 1. Embedding이 적합한 경우

Embedding은 "부모 문서의 일부로만 의미가 있는 데이터"에 적합하다.

```json
{
  "_id": "order_1001",
  "customer_id": "customer_10",
  "ordered_at": "2026-05-02T09:00:00+09:00",
  "items": [
    {
      "product_id": "product_1",
      "product_name_snapshot": "Laptop",
      "unit_price_snapshot": 1500000,
      "quantity": 1
    }
  ],
  "shipping_address_snapshot": {
    "zip_code": "06123",
    "address1": "Seoul ...",
    "address2": "..."
  }
}
```

적합한 상황:

- 부모와 항상 함께 읽힌다.
- 자식이 독립적으로 자주 갱신되지 않는다.
- "주문 당시 가격", "주문 당시 배송지"처럼 스냅샷이 필요하다.
- 문서 크기 제한에 걸리지 않는다.

여기서 `product_name_snapshot`은 중복이지만 나쁜 중복이 아니다. 현재 상품명이 아니라 주문 당시 계약 사실이다.

## 2. Reference가 적합한 경우

Reference는 독립 객체, 자주 변경되는 객체, N:M 관계, 큰 계층 구조에 적합하다.

```json
// customers
{
  "_id": "customer_10",
  "name": "Kim",
  "grade_code": "VIP",
  "email_normalized": "kim@example.com"
}
```

```json
// orders
{
  "_id": "order_1001",
  "customer_id": "customer_10",
  "ordered_at": "2026-05-02T09:00:00+09:00"
}
```

적합한 상황:

- 같은 객체가 여러 문서에서 참조된다.
- 값이 자주 바뀌며 중복 갱신 비용이 크다.
- N:M 관계나 큰 계층 구조를 표현해야 한다.
- 권한, 개인정보, 감사 경계를 분리해야 한다.

## 3. Hybrid 패턴: reference + snapshot

실무에서는 reference와 snapshot을 함께 쓰는 경우가 많다.

```json
{
  "_id": "order_1001",
  "customer_id": "customer_10",
  "customer_snapshot": {
    "name": "Kim",
    "grade_code": "VIP"
  },
  "ordered_at": "2026-05-02T09:00:00+09:00"
}
```

해석:

- `customer_id`: 현재 고객 객체와 연결하기 위한 reference
- `customer_snapshot`: 주문 당시 증빙을 위한 역사적 사실

이 구조는 중복이 아니라 두 종류의 사실을 분리한 것이다. 현재 고객 정보와 주문 당시 고객 정보는 같은 값처럼 보여도 의미가 다르다.

## 4. JSON Schema로 계약을 고정한다

MongoDB는 유연한 스키마를 제공하지만, 운영 단계에서는 JSON Schema validation으로 구조를 보호하는 편이 좋다.

```javascript
db.createCollection("terms", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["canonical_term", "aliases", "category"],
      properties: {
        canonical_term: {
          bsonType: "string",
          description: "정규 용어명"
        },
        aliases: {
          bsonType: "array",
          items: { bsonType: "string" }
        },
        category: {
          bsonType: "string"
        },
        source_ids: {
          bsonType: "array",
          items: { bsonType: "string" }
        }
      }
    }
  }
})
```

정규화 관점에서 schema validation은 "어떤 문서가 유효한 사실인가"를 DB 레벨 계약으로 고정하는 장치다.

## 5. 값 정규화 필드를 별도로 둔다

검색, 유일성, 매칭이 필요한 값은 원문과 정규화 값을 함께 둔다.

```json
{
  "_id": "customer_10",
  "email": "Kim@Example.COM",
  "email_normalized": "kim@example.com",
  "phone": "010-1234-5678",
  "phone_e164": "+821012345678"
}
```

인덱스 예시:

```javascript
db.customers.createIndex({ email_normalized: 1 }, { unique: true })
db.customers.createIndex({ phone_e164: 1 })
```

이렇게 하면 사용자 입력 표기는 보존하면서, 식별과 검색은 canonical value로 수행할 수 있다.

## 6. 용어 사전과 ontology 저장소

MongoDB는 RAG용 glossary, taxonomy, ontology-lite 저장소로 쓰기 좋다.

```json
{
  "_id": "term:cvd",
  "canonical_label": "Chemical Vapor Deposition",
  "aliases": ["CVD", "화학기상증착", "chemical vapor deposition"],
  "category": "deposition_process",
  "broader": ["term:deposition"],
  "related": ["term:thin_film"],
  "definitions": [
    {
      "text": "기체 원료의 화학 반응으로 박막을 형성하는 공정",
      "source_doc_id": "manual_2026_001"
    }
  ]
}
```

활용:

- LLM이 추출한 용어를 사람이 검토한 뒤 canonical term으로 승격한다.
- 사용자 쿼리의 alias를 canonical label로 확장한다.
- OpenSearch 색인 시 `canonical_terms` 필드에 추가한다.
- RAG 답변 생성 시 용어 정의를 컨텍스트에 주입한다.

## 7. MongoDB Vector Search와 정규화

MongoDB Vector Search를 RAG에 사용할 때도 정규화된 메타데이터가 중요하다.

```json
{
  "_id": "chunk:manual_100:3:2",
  "source_doc_id": "manual_100",
  "source_uri": "s3://kb/manual_100.pdf",
  "page": 3,
  "section_path": ["설치", "전원"],
  "text": "...",
  "embedding": [0.012, -0.031],
  "entity_ids": ["equipment:abc-100"],
  "canonical_terms": ["Chemical Vapor Deposition"],
  "language": "ko",
  "version": "2026-05-02"
}
```

Vector Search는 의미적으로 가까운 chunk를 찾지만, 실무 검색 품질은 메타데이터 필터에 크게 의존한다.

예:

- 특정 장비 모델만 검색: `entity_ids`
- 최신 버전만 검색: `version`
- 한국어 문서만 검색: `language`
- 특정 문서 유형만 검색: `doc_type`

정규화되지 않은 메타데이터는 벡터 검색의 recall과 precision을 동시에 떨어뜨린다.

## 8. MongoDB 정규화 체크리스트

- [ ] 문서에 embed된 데이터가 부모 없이는 의미가 없는가?
- [ ] 자주 바뀌거나 여러 곳에서 공유되는 데이터는 reference로 분리했는가?
- [ ] 현재값과 과거 스냅샷을 필드명과 의미로 구분했는가?
- [ ] JSON Schema validation으로 핵심 컬렉션의 계약을 고정했는가?
- [ ] 이메일, 전화번호, 코드처럼 식별에 쓰는 값은 normalized field를 두었는가?
- [ ] N:M 관계를 배열 하나에 밀어 넣어 무한 성장시키고 있지 않은가?
- [ ] RAG chunk 문서에 source, page, section, version, entity, canonical term이 있는가?
- [ ] Vector Search 필터링에 필요한 메타데이터가 정규화되어 있는가?

## 참고 자료

- [MongoDB Embedded Data](https://www.mongodb.com/docs/manual/data-modeling/embedding/)
- [MongoDB Reference Data](https://www.mongodb.com/docs/manual/data-modeling/referencing/)
- [MongoDB Schema Validation](https://www.mongodb.com/docs/current/core/schema-validation/)
- [MongoDB Vector Search Overview](https://www.mongodb.com/docs/atlas/atlas-search/vector-search/)
