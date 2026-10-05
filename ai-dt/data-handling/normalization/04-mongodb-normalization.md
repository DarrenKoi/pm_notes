---
tags: [mongodb, normalization, schema-design, embedding, reference, vector-search, rag]
level: intermediate
last_updated: 2026-05-02
reviewed_on: 2026-10-04
review_status: reviewed_with_limits
document_type: learning_note
category_major: "AI·DT"
category_middle: "데이터 엔지니어링"
category_minor: "정규화·모델링"
note_kind: "학습"
classified_on: "2026-10-05"
---

# MongoDB에서의 정규화

> MongoDB에서는 정규화와 반정규화가 "테이블 분해"가 아니라 embedding과 reference 사이의 선택으로 나타난다.

> [!info] 판본과 검증 범위 — 2026-10-04
> validator·unique index·원자성·참조는 MongoDB 8.0 공식 문서를 대조했다. embedding/reference는 확인 당시 manual 표시9.0, Vector Search는 별도 제품 안내다. 최신 설치 판본이나 사내 기능 지원을 단정하지 않는다. 로컬 구문·구조 계약만 검사했고 mongosh·서버·Atlas·인덱스·실제 벡터 검색은 실행하지 않았다.

## MongoDB 모델링의 출발점

MongoDB는 문서 단위로 데이터를 저장한다. 관련 데이터를 한 문서에 embed하면 한 번의 읽기로 가져올 수 있고, reference로 분리하면 중복을 줄이고 독립 변경을 쉽게 만든다.

```text
Embedding:
  주문 문서 안에 배송지, 주문항목, 결제 요약을 함께 저장

Reference:
  orders.customer_id -> customers._id
  order_items.product_id -> products._id
```

공식 문서는 embedding의 단일 읽기·문서 단위 원자적 쓰기와 reference의 독립 조회·변경/복잡한 관계 조건을 설명한다. 실제 성능은 접근 패턴과 크기로 측정한다. embedding/reference 선택은 관계형 3NF/BCNF 충족의 증명이 아니며 정의는 [핵심 개념](./01-normalization-core.md)을 참고한다. 단일 문서의 여러 속성 변경은 원자적이지만 여러 문서 updateMany 전체가 자동 원자적이지는 않다. 다중 문서 transaction은 배포/드라이버 조건과 비용을 따로 검토한다.

## 1. Embedding이 적합한 경우

Embedding은 부모와 함께 조회하는 contains 관계에 적합한 선택지다. 자식이 부모 밖에서 의미를 가지면 무조건 금지된다는 규칙은 아니다.

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
- BSON 문서 최대 크기16MiB와 배열 성장·쓰기 비용을 고려한다. 처음 작다고 무한 성장 배열이 안전한 것은 아니다.
- 한 문서에 있더라도 동시 갱신 시 기대값/version 조건을 update filter에 넣는 등 덮어쓰기 방지 계약을 정한다.

여기서 `product_name_snapshot`은 중복이지만 나쁜 중복이 아니다. 현재 상품명이 아니라 주문 당시 계약 사실이다.

## 2. Reference가 적합한 경우

Reference는 독립 객체, 자주 변경되는 객체, N:M 관계, 큰 계층 구조에 적합하다.

customers 문서:

```json
{
  "_id": "customer_10",
  "name": "Kim",
  "grade_code": "VIP",
  "email_normalized": "kim@example.com"
}
```

orders 문서:

```json
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
- 권한, 개인정보, 감사 경계를 나눌 필요가 있다. 컬렉션 분리만으로 권한/암호화가 적용되지는 않는다.

여기서는 string `_id`와 같은 타입의 `customer_id`를 사용한다. 필드 이름에 `_id`가 들어간다고 ObjectId로 자동 변환되거나 FK 존재·cascade 삭제가 강제되는 것은 아니다. 애플리케이션 조회 또는 `$lookup`과 참조 누락·삭제·버전/권한 처리를 설계한다. 둘 이상의 문서에서 일관된 스냅샷이 필요하면 별도 읽기/transaction 계약을 확인한다.

## 3. Hybrid 패턴: reference + snapshot

이 주문 예제는 reference와 snapshot을 함께 쓰는 선택지다. 실제 사용 빈도나 항상 최선이라는 주장은 아니다.

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

MongoDB의 `$jsonSchema`는 JSON Schema draft4를 기반으로 BSON 확장·생략이 있다. 일반 JSON Schema validator와 완전히 같은 기능이라고 간주하지 않는다. 아래는 새 학습용 `terms` 컬렉션의 mongosh 예제이며 기존 업무 컬렉션에 바로 적용하는 migration이 아니다.

```javascript
db.createCollection("terms", {
  validationLevel: "strict",
  validationAction: "error",
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

이 validator는 필수 필드와 string/array 타입만 검사한다. 빈 문자열·빈 aliases·중복 aliases·추가 필드·용어 내용의 진실성·참조 존재·권한은 검사하지 않는다. strict/error는 insert/update를 거부하는 계약이며 기존 문서를 자동 수정/전수 정리하지 않는다. 기존 컬렉션 변경에는 현황 조사와 collMod/validationLevel의 적용 범위를 별도로 검토한다.

아래 JSON은 terms의 최소 유효 문서다. glossary 예제와 필드 계약을 혼동하지 않는다.

```json
{
  "canonical_term": "Chemical Vapor Deposition",
  "aliases": ["CVD", "화학기상증착"],
  "category": "deposition_process"
}
```

## 5. 값 정규화 필드를 별도로 둔다

검색과 업무 식별의 변환 규칙을 먼저 구분한다. RFC 5321의 local-part 대소문자 보존 조건은 [핵심 개념](./01-normalization-core.md)을 따른다. 아래 email_normalized는 도메인만 소문자로 바꾸며 `Kim`을 보존한다. 별도 고객 예제의 `kim@example.com`은 그 원형이 이미 소문자라고 가정한다. 두 값을 제공자 확인 없이 같은 계정으로 합치지 않는다. 전화번호 변환은 국가/국내 trunk prefix·내선·입력 검증 정책이 필요하고 문자열 구분자 제거만으로 처리하지 않는다. 숫자는 가상 형식 예시다.

```json
{
  "_id": "customer_10",
  "email": "Kim@Example.COM",
  "email_normalized": "Kim@example.com",
  "phone": "010-1234-5678",
  "phone_e164": "+821012345678"
}
```

인덱스 예시(메일 식별 필드가 항상 존재하는 unsharded 학습 컬렉션 전제):

```javascript
db.customers.createIndex({ email_normalized: 1 }, { unique: true })
db.customers.createIndex({ phone_e164: 1 })
```

입력 표기와 계약에 따른 식별값을 분리할 수 있다. unique index가 올바른 이메일/동일인 판정까지 해주지는 않는다. 기존 중복이 있으면 unique 생성은 실패하며, 위 단일 필드 non-sparse unique에서 누락과 null은 같은 null index key로 취급되어 하나만 허용된다. 이메일이 선택 사항이면 업무 계약에 맞는 partial index와 필수/nullable validator를 따로 설계한다. sharded 컬렉션은 shard key prefix 등 unique 제한이 있으므로 이 예제를 그대로 적용하지 않는다. unknown 값을 빈 문자열로 조용히 대체하지 않는다.

## 6. 용어 사전과 ontology 저장소

MongoDB에 RAG용 glossary/taxonomy 관계를 JSON으로 저장하는 예다. 아래는 별도 `glossary_terms` 구조이며 앞의 `terms` validator를 그대로 적용하면 canonical_term 누락으로 거부된다. canonical_label/broader/related/definitions에 맞는 별도 validator와 참조·순환·출처 검토가 필요하다. 문서 저장만으로 OWL 추론/온톨로지 일관성 검사가 실행되지는 않는다.

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

위 2차원 embedding은 구조 예시다. 수동 임베딩 검색에는 별도 Vector Search index와 실제 모델 차원·거리 함수·query vector 계약이 필요하다. 일반 createIndex나 배열 저장만으로 활성화되지 않는다. pre-filter 필드를 Vector Search index에도 정의해야 한다. 확인한 제품 개요는 Atlas·지원되는 self-managed/local 경로를 설명하므로 Atlas만 가능하거나 임의의 Community 설치에서 곧바로 가능한 것으로 단정하지 않는다. 회사 배포 지원은 미확인이다.

메타데이터 필터는 검색 대상 범위를 제한하며 품질 개선 여부는 평가 데이터로 확인한다. ANN/ENN 지원·인덱스 옵션·배포별 기능은 설치 판본에 맞게 확인한다.

예:

- 특정 장비 모델만 검색: `entity_ids`
- 승인한 특정 버전만 검색: `version` — 문자열 필드가 존재한다고 최신 판별이 자동으로 되는 것은 아님
- 한국어 문서만 검색: `language`
- 특정 문서 유형만 검색: `doc_type` — 위 chunk 예제에는 없으므로 먼저 필드 계약/색인에 추가해야 함

필드 누락·값/타입 불일치는 필터에서 관련 문서를 배제하거나 다른 범위를 포함시킬 수 있다. recall/precision이 언제나 함께 하락한다는 보편적 결론은 피한다. 문서 버전·권한 변경·chunk 재생성과 오래된 index 정리를 함께 검증한다. 특정 장비/언어 조건은 ACL의 대체가 아니다.

## 8. MongoDB 정규화 체크리스트

- [ ] 문서에 embed된 데이터가 부모 없이는 의미가 없는가?
- [ ] 자주 바뀌거나 여러 곳에서 공유되는 데이터는 reference로 분리했는가?
- [ ] 현재값과 과거 스냅샷을 필드명과 의미로 구분했는가?
- [ ] JSON Schema validation으로 핵심 컬렉션의 계약을 고정했는가?
- [ ] 이메일, 전화번호, 코드처럼 식별에 쓰는 값은 normalized field를 두었는가?
- [ ] N:M 관계를 배열 하나에 밀어 넣어 무한 성장시키고 있지 않은가?
- [ ] RAG chunk 문서에 source, page, section, version, entity, canonical term이 있는가?
- [ ] Vector Search 필터링에 필요한 메타데이터가 정규화되어 있는가?

## 검토 결과

2026-10-04: 기존 주문·고객·snapshot·validator·glossary·chunk 예제의 맥락과 모든 절·작성일·경로를 보존했다. 이메일 도메인 처리값을 정정하고 JSON fence 밖으로 컬렉션 설명을 옮겼다. strict/error와 유효 fixture를 보강했다. 01은 정의, 03은 검색 projection, 이 문서는 MongoDB 저장 구조의 선택이다. Claude 협의는 HERDR_ENV=1에서 pane_not_found로 불가했고 완전 통합·업무별 unique/partial/보안/벡터 배포 계약은 보류했다. [정리 기록](../organization-log.md)에 실제 검증과 미확인을 남긴다.

## 참고 자료

모두 2026-10-04 확인. 8.0은 예제 대조 판본이며 설치/최신 판본 보장이 아니다. Vector Search stage/type의 이전/신규 추측 URL과 개요의 ANN/ENN 링크는 조회 오류였고 정상 확인한 개요 범위를 넘어 stage 실행을 검증했다고 주장하지 않는다.

- [MongoDB Embedded Data](https://www.mongodb.com/docs/manual/data-modeling/embedding/)
- [MongoDB Reference Data](https://www.mongodb.com/docs/manual/data-modeling/referencing/)
- [MongoDB Schema Validation](https://www.mongodb.com/docs/v8.0/core/schema-validation/)
- [MongoDB Vector Search Overview](https://www.mongodb.com/docs/vector-search/)

- [JSON Schema와 BSON 차이, 8.0](https://www.mongodb.com/docs/v8.0/core/schema-validation/specify-json-schema/)
- [Validation level, 8.0](https://www.mongodb.com/docs/v8.0/core/schema-validation/specify-validation-level/)
- [Unique index, 8.0](https://www.mongodb.com/docs/v8.0/core/index-unique/)
- [원자성과 transaction, 8.0](https://www.mongodb.com/docs/v8.0/core/write-operations-atomicity/)
- [Manual reference, 8.0](https://www.mongodb.com/docs/v8.0/reference/database-references/)
