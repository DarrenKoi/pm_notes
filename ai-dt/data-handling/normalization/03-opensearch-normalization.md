---
tags: [opensearch, normalization, search, normalizer, nested, hybrid-search, rag]
level: advanced
last_updated: 2026-05-02
status: in-progress
---

# OpenSearch에서의 정규화

> OpenSearch에서 정규화는 관계형 모델처럼 테이블을 분해하는 뜻만이 아니다. 값 표준화, 검색 토큰화, nested 관계 보존, 검색 점수 정규화, 검색용 projection 설계까지 포함한다.

## OpenSearch의 역할을 먼저 정한다

OpenSearch는 보통 원천 데이터베이스가 아니라 검색과 조회에 최적화된 projection이다.

```text
원천 모델:
  PostgreSQL / MongoDB / 업무 DB

검색 projection:
  OpenSearch index

흐름:
  원천 변경 이벤트 -> 정규화/풍부화(enrichment) -> OpenSearch 색인
```

따라서 OpenSearch에서는 무조건 정규형을 유지하기보다, 원천의 식별자와 의미 구조를 보존한 채 검색에 맞게 반정규화하는 것이 일반적이다.

## 1. 값 정규화: `keyword` normalizer

OpenSearch의 `normalizer`는 `keyword` 필드 전체를 하나의 토큰으로 유지하면서 소문자화, 공백 제거, ASCII folding 같은 처리를 적용한다. `text` analyzer가 여러 토큰을 만드는 것과 다르다.

적합한 대상:

- 코드: `ABC-123`, `abc-123`
- 이메일: `USER@EXAMPLE.COM`
- 태그: `OpenSearch`, `opensearch`
- 정렬/집계용 keyword

예시:

```json
PUT /products
{
  "settings": {
    "analysis": {
      "normalizer": {
        "normalized_keyword": {
          "type": "custom",
          "filter": ["trim", "lowercase", "asciifolding"]
        }
      }
    }
  },
  "mappings": {
    "properties": {
      "product_code": {
        "type": "keyword",
        "normalizer": "normalized_keyword"
      },
      "product_name": {
        "type": "text",
        "fields": {
          "keyword": {
            "type": "keyword",
            "normalizer": "normalized_keyword"
          }
        }
      }
    }
  }
}
```

주의할 점:

- `normalizer`는 synonym, stemming처럼 토큰 단위 처리를 하는 도구가 아니다.
- `_source` 원문은 그대로 남고, 색인된 keyword 값만 정규화된다.
- 원문 보존이 필요하면 `raw_value`와 `normalized_value`를 모두 설계한다.

## 2. 텍스트 정규화: analyzer와 synonym

자연어 검색에서는 `normalizer`보다 analyzer가 중요하다.

```text
문서 원문:
  "CVD 공정 온도 조건"

검색 관점 정규화:
  CVD -> Chemical Vapor Deposition -> 화학기상증착
```

설계 방법:

- `text` 필드에는 언어별 analyzer를 적용한다.
- 전문 용어는 synonym filter나 별도 glossary lookup으로 확장한다.
- 정확 매칭이 필요한 용어는 `keyword` 하위 필드도 둔다.
- 한글, 영문, 약어가 섞이는 도메인에서는 canonical term 필드를 별도로 둔다.

```json
{
  "term": "CVD",
  "canonical_term": "Chemical Vapor Deposition",
  "aliases": ["CVD", "화학기상증착", "chemical vapor deposition"]
}
```

## 3. 구조 정규화: `object`, `nested`, `join`

OpenSearch는 JSON 문서를 색인하지만, 배열 안의 객체는 기본적으로 flattened 형태로 색인될 수 있다. 이때 같은 배열 원소 안의 관계가 깨질 수 있다.

문제 예시:

```json
{
  "patients": [
    {"name": "John", "age": 56, "smoker": true},
    {"name": "Mary", "age": 85, "smoker": false}
  ]
}
```

단순 `object`로 색인하면 `age >= 75`와 `smoker = true`가 서로 다른 배열 원소에서 매칭되어도 한 문서가 검색될 수 있다. 객체 단위의 관계를 보존해야 하면 `nested`를 사용한다.

```json
PUT /medical-records
{
  "mappings": {
    "properties": {
      "patients": {
        "type": "nested",
        "properties": {
          "name": {"type": "text"},
          "age": {"type": "integer"},
          "smoker": {"type": "boolean"}
        }
      }
    }
  }
}
```

쿼리도 `nested`로 감싼다.

```json
GET /medical-records/_search
{
  "query": {
    "nested": {
      "path": "patients",
      "query": {
        "bool": {
          "must": [
            {"range": {"patients.age": {"gte": 75}}},
            {"term": {"patients.smoker": true}}
          ]
        }
      }
    }
  }
}
```

선택 기준:

| 구조 | 사용 상황 | 주의점 |
|------|-----------|--------|
| `object` | 단순 중첩, 배열 원소 간 관계가 중요하지 않음 | 배열 객체의 조합 오류 가능 |
| `nested` | 배열 원소별 조건 결합이 중요함 | 색인/쿼리 비용 증가 |
| `join` | 부모/자식 문서를 같은 인덱스에서 연결해야 함 | 운영 복잡도와 성능 비용이 큼 |
| 반정규화 | 검색 결과에 필요한 데이터를 한 문서에 모음 | 원천과 동기화 규칙 필요 |

OpenSearch에서는 `join`을 남용하기보다, 원천 DB에서 정규화된 구조를 유지하고 OpenSearch에는 검색 목적의 문서를 만들어 넣는 편이 실무적으로 단순하다.

## 4. 검색용 반정규화: 원천 ID를 보존한다

검색 문서는 사용자에게 보여줄 내용을 한 번에 담는 것이 유리하다.

```json
{
  "doc_type": "order",
  "order_id": "ord_1001",
  "customer": {
    "customer_id": "cus_10",
    "name": "Kim"
  },
  "items": [
    {
      "product_id": "prd_1",
      "product_name": "Laptop",
      "quantity": 1
    }
  ],
  "delivery_status": "SHIPPED",
  "ordered_at": "2026-05-02T09:00:00+09:00"
}
```

핵심은 중복 자체가 아니라 원천 추적성이다.

- `order_id`, `customer_id`, `product_id`를 반드시 유지한다.
- 스냅샷인지 현재값인지 필드명으로 구분한다.
- 재색인 가능한 파이프라인을 둔다.
- partial update보다 전체 문서 재생성이 단순한 경우가 많다.

## 5. RAG용 인덱스 정규화

RAG 인덱스는 "문서 텍스트"만 넣으면 운영이 어렵다. 최소한 다음 필드를 정규화한다.

```json
{
  "chunk_id": "doc_100:p003:c002",
  "source_doc_id": "doc_100",
  "source_uri": "s3://kb/manual/doc_100.pdf",
  "title": "장비 유지보수 매뉴얼",
  "section_path": ["설치", "전원", "점검"],
  "page": 3,
  "text": "...",
  "embedding": [0.012, -0.031],
  "language": "ko",
  "entity_ids": ["equipment:abc-100"],
  "canonical_terms": ["Chemical Vapor Deposition"],
  "version": "2026-05-02"
}
```

정규화 포인트:

- `chunk_id`는 재생성해도 안정적으로 만들 수 있어야 한다.
- `source_doc_id`와 `source_uri`로 출처를 추적한다.
- `section_path`, `page`, `version`을 넣어 답변 citation을 가능하게 한다.
- `entity_ids`, `canonical_terms`로 용어와 객체 기반 필터링을 가능하게 한다.

## 6. 하이브리드 검색의 점수 정규화

BM25와 벡터 검색 점수는 스케일이 다르다. OpenSearch의 `normalization-processor`는 hybrid query의 여러 검색 점수를 정규화하고 결합하는 데 사용된다.

예시:

```json
PUT /_search/pipeline/rag-hybrid-pipeline
{
  "description": "Normalize and combine BM25 and vector scores",
  "phase_results_processors": [
    {
      "normalization-processor": {
        "normalization": {
          "technique": "min_max"
        },
        "combination": {
          "technique": "arithmetic_mean",
          "parameters": {
            "weights": [0.4, 0.6]
          }
        }
      }
    }
  ]
}
```

이때 정규화는 데이터 모델링의 정규화가 아니라 검색 결과 score normalization이다. 하지만 목적은 비슷하다. 서로 다른 좌표계의 값을 비교 가능한 기준으로 맞추는 것이다.

## 7. OpenSearch 정규화 체크리스트

- [ ] exact match, 집계, 정렬용 필드는 `keyword`와 `normalizer`를 검토했는가?
- [ ] 자연어 검색 필드와 exact match 필드를 분리했는가?
- [ ] 전문 용어, 약어, 다국어 표현을 canonical term으로 연결했는가?
- [ ] 배열 객체에서 원소 단위 관계가 중요하면 `nested`를 사용했는가?
- [ ] 검색 문서가 원천 ID를 보존하는가?
- [ ] OpenSearch 인덱스를 원천으로 착각하지 않도록 재색인 경로가 있는가?
- [ ] RAG chunk에 `chunk_id`, `source_doc_id`, `page`, `section_path`, `version`이 있는가?
- [ ] BM25와 vector 결과를 결합할 때 score normalization 전략을 정했는가?

## 참고 자료

- [OpenSearch Normalizer](https://docs.opensearch.org/latest/mappings/mapping-parameters/normalizer/)
- [OpenSearch Normalizers](https://docs.opensearch.org/docs/latest/analyzers/normalizers/)
- [OpenSearch Object Field Types](https://docs.opensearch.org/latest/mappings/supported-field-types/object-fields/)
- [OpenSearch Nested Query](https://docs.opensearch.org/3.4/query-dsl/joining/nested/)
- [OpenSearch Normalization Processor](https://docs.opensearch.org/2.15/search-plugins/search-pipelines/normalization-processor/)
