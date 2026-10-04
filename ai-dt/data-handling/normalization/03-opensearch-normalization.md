---
tags: [opensearch, normalization, search, normalizer, nested, hybrid-search, rag]
level: advanced
last_updated: 2026-05-02
reviewed_on: 2026-10-04
review_status: reviewed_with_limits
document_type: learning_note
---

# OpenSearch에서의 정규화

> OpenSearch에서 정규화는 관계형 모델처럼 테이블을 분해하는 뜻만이 아니다. 값 표준화, 검색 토큰화, nested 관계 보존, 검색 점수 정규화, 검색용 projection 설계까지 포함한다.

> [!info] 확인 판본과 실행 범위 — 2026-10-04
> normalizer·nested·join은 3.4 URL의 공식 문서, 점수 pipeline은 원래 2.15 문서를 대조했다. 확인한 normalizers 3.4·pipeline 2.15 페이지는 유지보수 종료로 표시됐다. 최신 또는 사내 설치 판본이라는 뜻이 아니다. JSON/계약 fixture만 로컬 검사했고 OpenSearch 서버·분석기·색인·검색은 실행하지 않았다.

## OpenSearch의 역할을 먼저 정한다

이 학습 문서는 OpenSearch를 업무 원천에서 재생성하는 검색 projection으로 사용한다. 모든 시스템이 이 배치를 사용한다는 통계적 주장은 아니다.

```text
원천 모델:
  PostgreSQL / MongoDB / 업무 DB

검색 projection:
  OpenSearch index

흐름:
  원천 변경 이벤트 -> 정규화/풍부화(enrichment) -> OpenSearch 색인
```

이 배치에서는 원천의 식별자·사건 의미·권한·버전을 보존하면서 검색 문서를 구성한다. 관계형 정규형, 값 변환과 검색 점수 변환의 구분은 [핵심 개념](./01-normalization-core.md)을 따른다.

## 1. 값 정규화: `keyword` normalizer

OpenSearch의 `normalizer`는 `keyword` 필드 전체를 하나의 토큰으로 유지하면서 소문자화, 앞뒤 공백 제거(`trim`), ASCII folding 같은 처리를 적용한다. `text` analyzer가 여러 토큰을 만드는 것과 다르다.

적합한 대상:

- 코드: `ABC-123`, `abc-123`
- 이메일 검색 표시: 식별값은 원형 보존하며 전체 lowercase를 동일 계정 판정에 사용하지 않음
- 태그: `OpenSearch`, `opensearch`
- 정렬/집계용 keyword — 코드가 대소문자를 구분하는 업무라면 lowercase를 적용하지 않음

RFC 5321 §2.4의 local-part 대소문자 보존 조건은 [핵심 개념](./01-normalization-core.md)과 같다. lowercase/asciifolding은 서로 다른 식별값을 같은 검색 값으로 합칠 수 있으므로 유일성·인증 키에 그대로 재사용하지 않는다. `trim`은 문자열 안쪽 공백을 모두 삭제하는 필터가 아니다.

예시:

아래 요청은 Dashboards Dev Tools의 HTTP 요청 표기다. fence 안 첫 줄은 JSON이 아니므로 HTTP 행과 JSON 본문을 분리해 읽는다. 예제 이름의 새 학습용 인덱스에서만 사용할 것을 전제로 하며 기존 업무 인덱스 변경 절차가 아니다.

```http
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
- 기본 `_source` 보존은 검색 변환과 별개다. 두 개의 별도 원천 필드가 항상 필요한 것은 아니다. 원형으로도 exact match/집계를 해야 하면 normalizer 없는 keyword 하위 필드 등 목적에 맞는 mapping을 추가한다.

확인된 compatible filter 목록에 trim/lowercase/asciifolding이 있다. 새 인덱스에서 다음 요청으로 단일 토큰과 내부 공백 보존을 확인한다. 기대 문자열은 `abc-123 extra`이며 실제 서버 결과는 아직 미확인이다.

```http
GET /products/_analyze
{
  "normalizer": "normalized_keyword",
  "text": "  ABC-123 EXTRA  "
}
```

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
- 전문 용어는 synonym filter나 별도 glossary lookup으로 확장한다. 다중 단어의 관계에는 `synonym_graph`를 검토하되 tokenizer·filter 순서·검색/색인 시점과 플러그인 설치 조건을 확인한다. 문자열 화살표만으로 synonym 규칙이 구현된 것은 아니다.
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

OpenSearch의 기본 `object` 배열 색인은 필드별 값 목록으로 펼쳐진다. 여기서 flattened는 저장 표현 설명이며 `flat_object` 필드 타입을 지정했다는 뜻이 아니다. 이때 같은 배열 원소 안의 관계가 깨질 수 있다.

문제 예시:

```json
{
  "patients": [
    {"name": "John", "age": 56, "smoker": true},
    {"name": "Mary", "age": 85, "smoker": false}
  ]
}
```

단순 `object`로 색인하면 `age >= 75`와 `smoker = true`가 서로 다른 배열 원소에서 매칭되어도 한 문서가 검색될 수 있다. 같은 배열 원소에 두 조건이 모두 성립해야 하면 `nested` mapping과 nested query를 함께 사용한다. 이 가상의 두 사람 예제는 AND 조건에서 0건이 기대값이다. OR 조건 또는 개별 필드 조건과 구분한다.

```http
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

```http
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

이 설계 예에서는 원천과 projection을 구분한다. join을 선택하면 부모·자식이 같은 shard에 있어야 하므로 자식의 색인에 부모 계열과 같은 routing을 지정해야 한다. 관계형 JOIN과 FK 강제를 그대로 제공하는 기능으로 간주하지 않는다. nested/join의 실제 비용과 적절한 모델은 데이터 크기·변경 패턴에 따라 측정하며 업무 설계 선택은 미확인이다.

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
- partial update와 전체 재생성은 실제 크기·쓰기 빈도·동시 갱신에 따라 선택한다. 둘 다 stale event가 최신 문서를 덮어쓰지 않는 버전 계약이 필요하다.
- 삭제·권한 변경·재색인 실패·역순 이벤트와 현재값/스냅샷의 혼합을 검증한다. 원천 ID가 있다는 것만으로 일관성이나 접근 제어가 보장되지는 않는다.

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

- 동일 원문 버전·청킹 규칙에는 재현 가능한 ID를 쓰고, 버전/청킹 변경 시 충돌 없이 구분한다. 오래된 chunk 삭제와 citation 유지 정책을 정한다.
- `source_doc_id`와 `source_uri`로 출처를 추적한다.
- `section_path`, `page`, `version`을 넣어 답변 citation을 가능하게 한다.
- `entity_ids`, `canonical_terms`로 필터링하려면 실제 keyword mapping과 쿼리 계약도 정한다.
- 위 embedding 2차원 값은 구조 예시다. 실제 모델·차원·거리 함수·`knn_vector` mapping을 맞춰야 하며 JSON 배열만 넣는다고 벡터 검색이 활성화되지 않는다. page/section 필드만으로 근거 정확성도 보장되지 않는다. 권한 필터는 retrieval과 citation 양쪽에 적용한다.

## 6. 하이브리드 검색의 점수 정규화

BM25와 벡터 검색 점수는 스케일이 다르다. OpenSearch의 `normalization-processor`는 hybrid query의 여러 검색 점수를 정규화하고 결합하는 데 사용된다.

예시:

```http
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

pipeline 생성만으로 모든 검색에 적용되지는 않는다. 준비된 `rag-chunks` 인덱스의 text와 2차원 knn_vector embedding, 해당 버전 hybrid/k-NN 기능을 전제로 아래처럼 요청에 지정한다. `[0.012, -0.031]`은 구조 확인용 숫자이고 실제 질문의 임베딩은 문서와 같은 모델·전처리로 생성해야 한다. 이 문서는 해당 인덱스·모델을 생성하거나 실행한 것이 아니다.

```http
GET /rag-chunks/_search?search_pipeline=rag-hybrid-pipeline
{
  "query": {
    "hybrid": {
      "queries": [
        {"match": {"text": "CVD 온도 조건"}},
        {"knn": {"embedding": {"vector": [0.012, -0.031], "k": 10}}}
      ]
    }
  }
}
```

2.15 문서에서 weights는 query 순서와 개수가 같고 각각 0~1, 합계1이어야 한다. 여기서는 text 0.4/vector 0.6의 학습 예시이며 최적 품질 보장이 아니다. processor는 subquery가 반환한 후보만 변환하고 새 후보를 추가하지 않는다. 실제 후보 수·권한 필터·retrieval 품질을 별도 평가한다.

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

## 검토 결과

2026-10-04: 모든 원래 절·구조 예제·체크리스트와 작성일을 보존했다. 이메일·trim·HTTP/JSON·join routing·chunk/벡터 조건과 pipeline 적용 단계를 보완했다. 01은 일반 정의, 이 문서는 OpenSearch 적용이다. Claude 협의는 HERDR_ENV=1에서 `pane_not_found`로 불가했고 완전 통합·업무별 모델 선택은 보류했다. [정리 기록](../organization-log.md)에 보존·로컬·읽기 검증 결과와 실제 서버 미확인을 남긴다.

## 참고 자료

아래 공식 자료를 2026-10-04 읽었다. normalizer 2.15 및 trim 단독 3.4 URL은 조회 오류였으며 정상 조회된 3.4 normalizers/normalizer 목록·예제로 대조했다. 서로 다른 판본을 무조건 호환으로 간주하지 않는다.

- [OpenSearch Normalizer](https://docs.opensearch.org/3.4/mappings/mapping-parameters/normalizer/)
- [OpenSearch Normalizers](https://docs.opensearch.org/3.4/analyzers/normalizers/)
- [OpenSearch object 배열과 nested, 2.15](https://docs.opensearch.org/2.15/field-types/supported-field-types/nested/)
- [OpenSearch Nested Query](https://docs.opensearch.org/3.4/query-dsl/joining/nested/)
- [OpenSearch Normalization Processor](https://docs.opensearch.org/2.15/search-plugins/search-pipelines/normalization-processor/)

- [Join routing, 3.4](https://docs.opensearch.org/3.4/mappings/supported-field-types/join/)
- [Synonym graph, 3.4](https://docs.opensearch.org/3.4/analyzers/token-filters/synonym-graph/)
- [Hybrid search, 2.15](https://docs.opensearch.org/2.15/search-plugins/hybrid-search/)
- [k-NN index 조건, 2.15](https://docs.opensearch.org/2.15/search-plugins/knn/knn-index/)
