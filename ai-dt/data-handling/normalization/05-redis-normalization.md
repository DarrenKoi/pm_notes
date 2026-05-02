---
tags: [redis, normalization, cache, key-design, hash, json, search, vector]
level: intermediate
last_updated: 2026-05-02
---

# Redis에서의 정규화

> Redis에서 정규화는 JOIN을 위한 테이블 분해가 아니라, 키 설계, 캐시 무효화 단위, 중복 데이터의 수명, alias 매핑을 명확히 하는 일이다.

## Redis의 역할을 먼저 정한다

Redis는 대개 다음 역할을 맡는다.

- 캐시
- 세션 저장소
- 카운터
- 큐/스트림
- 랭킹
- 임시 인덱스
- Redis Search/Vector Search 기반 빠른 검색 저장소

대부분의 경우 Redis는 원천 DB가 아니다. 따라서 정규화의 핵심은 "원천 사실을 어디에 둘 것인가"보다 "어떤 단위로 캐시하고 어떻게 무효화할 것인가"에 있다.

## 1. 키 이름을 정규화한다

키 이름은 Redis의 스키마다.

```text
entity:{id}
entity:{id}:field
entity:{id}:relation:{relation_name}
index:{field}:{normalized_value}
cache:{use_case}:{hash}
lock:{resource}:{id}
stream:{event_name}
```

예시:

```text
customer:1001
customer:email:kim@example.com
customer:1001:orders
order:9001
order:9001:items
term:cvd
term_alias:화학기상증착
rag:query_cache:sha256:...
```

좋은 키 설계의 조건:

- 같은 객체는 항상 같은 패턴으로 접근한다.
- 원천 ID를 포함한다.
- normalized value와 raw value를 섞지 않는다.
- TTL이 필요한 키와 영구 키를 구분한다.
- prefix만 봐도 소유 도메인과 무효화 범위가 보인다.

## 2. Hash는 단순 객체에 적합하다

Redis Hash는 field-value 쌍의 record로 단순 객체를 저장하기 좋다.

```text
HSET customer:1001 \
  name "Kim" \
  email "Kim@Example.COM" \
  email_normalized "kim@example.com" \
  grade_code "VIP"
```

정규화 포인트:

- 원문과 normalized field를 함께 둔다.
- 자주 독립적으로 바뀌는 큰 하위 객체는 별도 key로 분리한다.
- `HGETALL`은 편하지만 큰 hash에서는 비용을 고려한다.
- Redis 7.4 이후 hash field TTL도 가능하지만, 복잡한 수명 정책은 운영 난도를 높일 수 있다.

## 3. JSON은 복잡한 중첩 객체에 적합하다

Redis JSON은 중첩 구조와 JSONPath 기반 접근이 필요할 때 유용하다.

```text
JSON.SET order:9001 $ '{
  "order_id": "9001",
  "customer_id": "1001",
  "items": [
    {
      "product_id": "p1",
      "product_name_snapshot": "Laptop",
      "quantity": 1
    }
  ],
  "status": "PAID"
}'
```

하지만 JSON에 모든 것을 넣으면 캐시 무효화가 어려워질 수 있다.

기준:

- 한 번에 같이 읽고 같이 만료되면 JSON으로 묶어도 된다.
- 일부 필드만 매우 자주 바뀌면 별도 key나 hash로 분리한다.
- 원천 DB의 모든 관계를 Redis JSON 하나에 복제하지 않는다.

## 4. 보조 인덱스를 명시적으로 만든다

Redis 기본 자료구조만 쓸 때는 보조 인덱스를 직접 관리해야 한다.

이메일로 고객 ID 찾기:

```text
SET customer:email:kim@example.com 1001
```

고객의 주문 목록:

```text
SADD customer:1001:orders order:9001 order:9002
```

최근 주문 정렬:

```text
ZADD customer:1001:orders_by_time 1777680000 order:9001
```

정규화 관점에서 중요한 것은 "어떤 키가 원천이고 어떤 키가 인덱스인가"를 구분하는 것이다.

```text
원천 캐시:
  customer:1001

인덱스:
  customer:email:kim@example.com -> 1001
  customer:1001:orders -> {order:9001, order:9002}
```

고객 삭제나 이메일 변경 시 관련 인덱스를 함께 지우는 규칙이 필요하다.

## 5. 캐시 반정규화와 무효화

화면/API 응답을 통째로 캐시하는 것은 강한 반정규화다.

```text
cache:order_detail:9001 -> {
  order,
  customer,
  items,
  delivery
}
```

이 패턴은 읽기 성능에는 좋지만 무효화가 핵심이다.

무효화 전략:

| 전략 | 설명 | 적합한 상황 |
|------|------|-------------|
| TTL | 시간이 지나면 자동 만료 | 약간의 stale 허용 |
| 이벤트 기반 삭제 | 원천 변경 이벤트로 관련 cache 삭제 | 정합성이 중요함 |
| 버전 키 | `customer:1001:version`을 cache key에 포함 | 부분 변경 추적이 어려움 |
| write-through | 원천 변경 시 캐시도 갱신 | 쓰기 경로가 통제됨 |
| cache-aside | miss 시 원천 조회 후 저장 | 일반적인 API cache |

정규화된 키 구조가 있으면 무효화 범위를 좁힐 수 있다.

## 6. 용어 alias와 canonical term 매핑

RAG나 검색 시스템에서 Redis는 빠른 용어 정규화 캐시로 쓰기 좋다.

```text
HSET term:cvd \
  canonical_label "Chemical Vapor Deposition" \
  category "deposition_process"

SET term_alias:cvd term:cvd
SET term_alias:chemical_vapor_deposition term:cvd
SET term_alias:화학기상증착 term:cvd
```

사용 흐름:

```text
사용자 쿼리: "CVD 온도 조건"
1. alias 후보 추출: CVD
2. GET term_alias:cvd -> term:cvd
3. HGET term:cvd canonical_label
4. 확장 쿼리: "CVD Chemical Vapor Deposition 온도 조건"
```

이 구조는 LLM 호출 전에 빠르게 쿼리를 정규화하거나, OpenSearch 검색 전에 필터/확장어를 추가하는 데 유용하다.

## 7. Redis Search와 Vector Search

Redis Search를 쓰면 Hash나 JSON 문서에 보조 인덱스를 만들고 검색할 수 있다. Vector Search에서는 embedding과 메타데이터를 Hash 또는 JSON에 저장하고 벡터 필드로 검색한다.

RAG용 예시 key:

```text
HSET rag_chunk:manual_100:3:2 \
  source_doc_id "manual_100" \
  page "3" \
  section_path "설치 > 전원" \
  text "..." \
  language "ko" \
  canonical_terms "Chemical Vapor Deposition|Thin Film"
```

정규화 포인트:

- `chunk_id`를 key에 포함한다.
- 검색 필터로 쓸 메타데이터를 일관된 field로 둔다.
- embedding 모델명, 차원, 버전을 별도로 관리한다.
- 원문 문서 재수집 시 기존 chunk를 재생성/삭제할 기준을 둔다.

## 8. Redis 정규화 체크리스트

- [ ] Redis가 원천 DB인지 cache/projection인지 명확한가?
- [ ] key prefix와 ID 규칙이 일관적인가?
- [ ] raw value와 normalized value를 구분하는가?
- [ ] 보조 인덱스 key의 생성/삭제 규칙이 있는가?
- [ ] 응답 캐시의 TTL, 무효화 이벤트, 버전 전략이 정해져 있는가?
- [ ] 큰 JSON 하나에 독립 변경되는 데이터를 과도하게 넣고 있지 않은가?
- [ ] 용어 alias와 canonical term 매핑을 빠르게 조회할 수 있는가?
- [ ] RAG chunk key가 `source_doc_id`, `chunk_id`, `version`을 보존하는가?

## 참고 자료

- [Redis Hashes](https://redis.io/docs/latest/develop/data-types/hashes/)
- [Redis JSON](https://redis.io/docs/latest/develop/data-types/json/)
- [Redis Data Type Comparison](https://redis.io/docs/latest/develop/data-types/compare-data-types/)
- [Redis Vector Search Concepts](https://redis.io/docs/latest/develop/ai/search-and-query/vectors/)
- [Redis Search Indexing](https://redis.io/docs/latest/develop/ai/search-and-query/indexing/)
