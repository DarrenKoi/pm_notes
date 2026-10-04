---
tags: [redis, normalization, cache, key-design, hash, json, search, vector]
level: intermediate
last_updated: 2026-05-02
reviewed_on: 2026-10-04
review_status: reviewed_with_limits
document_type: learning_note
---

# Redis에서의 정규화

> Redis에서 정규화는 JOIN을 위한 테이블 분해가 아니라, 키 설계, 캐시 무효화 단위, 중복 데이터의 수명, alias 매핑을 명확히 하는 일이다.

> [!info] 확인 범위 — 2026-10-04
> Redis 공식 명령/자료형/Cluster 문서와 Open Source8.0 릴리스 노트를 대조했다. latest URL은 확인일의 페이지이며 사내 설치·최신 판본 보장이 아니다. 문서의 shell 구문·인자·JSON·순수 fixture만 검증했고 Redis 서버·TTL 경과·Cluster·Search/Vector는 실행하지 않았다.

## Redis의 역할을 먼저 정한다

Redis는 대개 다음 역할을 맡는다.

- 캐시
- 세션 저장소
- 카운터
- 큐/스트림
- 랭킹
- 임시 인덱스
- Redis Search/Vector Search 기반 빠른 검색 저장소

이 문서는 Redis를 원천 DB에서 재생성하는 캐시/projection으로 사용한다. 모든 Redis가 이 역할이라는 통계적 주장은 아니다. 이 배치에서는 캐시 단위·무효화·미확인 상태를 설계한다. 원천 역할이면 내구성·복구 계약을 별도로 검토한다. 관계형 정규형과 캐시 키/값 표준화의 차이는 [핵심 개념](./01-normalization-core.md)을 따른다.

## 1. 키 이름을 정규화한다

키 이름은 애플리케이션의 접근 규약이다. 아래 `{id}` 등은 설명용 placeholder이며 Redis가 schema/유일성/참조를 자동 강제하지 않는다. 실제 Cluster key에 `{...}`를 넣으면 중괄호 안 문자열이 hash tag가 되어 같은 slot 배치에 영향을 준다. 의미 구분자 `:`는 자동 계층·소유권·ACL이 아니다.

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
customer:email:Kim@example.com
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

키와 예제 값은 가상 학습용이다. 실제 이메일 같은 개인정보를 key에 직접 넣을지는 노출·접근 범위를 고려한 업무 계약으로 정한다. 이메일 local-part는 전체 소문자화하지 않고 원형을 보존하며 아래는 도메인만 소문자로 바꾼다. 서로 다른 Kim/kim을 제공자 확인 없이 합치지 않는다.

아래 실행 fence는 shell에서 redis-cli를 호출하는 표기다. 승인된 새 학습용 인스턴스/키를 전제로 하며 이 정리 작업에서는 연결·실행하지 않았다. 실제 연결 대상·인증·TLS는 운영 계약에 맞게 설정한다.

## 2. Hash는 단순 객체에 적합하다

Redis Hash는 field-value 쌍의 record로 단순 객체를 저장하기 좋다.

```bash
redis-cli HSET customer:1001 \
  name "Kim" \
  email "Kim@Example.COM" \
  email_normalized "Kim@example.com" \
  grade_code "VIP"
```

정규화 포인트:

- 원문과 normalized field를 함께 둔다.
- 자주 독립적으로 바뀌는 큰 하위 객체는 별도 key로 분리한다.
- `HGETALL`은 전체 field/value를 읽는 O(N) 명령이다. 필요한 소수 필드에는 HGET/HMGET을 검토한다.
- HEXPIRE는 Redis7.4.0부터 제공한다. hash key의 EXPIRE와 개별 field 수명은 다르다. field가 사라질 때 모름/만료/없음의 의미와 재수집 규칙을 정한다. 배포별 명령 지원을 확인한다.
- HSET/JSON.SET을 했다는 것만으로 새 key에 TTL이 설정되지 않는다. key TTL을 설계·설정해야 한다. EXPIRE 문서에서 HSET은 기존 key TTL을 유지하지만 SET으로 덮어쓰면 보통 TTL이 제거된다(KEEPTTL 등 옵션은 별도 확인).

## 3. JSON은 복잡한 중첩 객체에 적합하다

Redis JSON은 중첩 구조와 JSONPath 접근이 필요할 때 검토한다. Open Source8.0 릴리스는 JSON과 Search를 포함한다고 설명한다. 이전 Redis 배포는 해당 모듈/Stack 제공 여부를 확인해야 하며 임의의 오래된 기본 Redis에서 JSON.SET/FT 명령이 된다고 가정하지 않는다. 실제 명령·ACL·배포 지원은 미확인이다.

```bash
redis-cli JSON.SET order:9001 $ '{
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

```bash
redis-cli SET customer:email:Kim@example.com 1001
```

고객의 주문 목록:

```bash
redis-cli SADD customer:1001:orders order:9001 order:9002
```

최근 주문 정렬:

```bash
redis-cli ZADD customer:1001:orders_by_time 1777680000 order:9001
```

정규화 관점에서 중요한 것은 "어떤 키가 원천이고 어떤 키가 인덱스인가"를 구분하는 것이다.

```text
원천 캐시:
  customer:1001

인덱스:
  customer:email:Kim@example.com -> 1001
  customer:1001:orders -> {order:9001, order:9002}
```

고객 삭제나 이메일 변경 시 이전 인덱스 삭제·새 인덱스 생성·원천 캐시 갱신을 함께 처리하는 계약이 필요하다. 위 SET은 기존 고객 매핑을 덮어쓸 수 있어 unique index가 아니다. 동시 writer·재시도·만료/eviction 후 남은 보조 key도 처리한다. Redis MULTI/EXEC·WATCH 또는 script를 선택할 때 Cluster multi-key의 같은 slot 조건을 검토한다. Redis 내부 묶음만으로 외부 원천 DB와의 원자성이 보장되지는 않는다. 구현 선택은 미확인이다.

ZADD의 점수1777680000은 UTC2026-05-02T00:00:00(한국시간09:00)의 초 단위 Unix timestamp 예시다. 원천 timestamp/단위를 명시하고 같은 score의 순서가 업무 사건 순서를 보장한다고 간주하지 않는다.

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

키 규약은 무효화 대상 식별에 도움이 된다. TTL만으로 캐시가 현재 원천과 일치하는지 보장하지는 않는다. miss→원천 읽기→write 사이에 변경 이벤트가 끼면 오래된 값이 삭제 후 다시 채워질 수 있다. 버전 비교·역순/중복 event·복구/재생성·reader의 stale 표시를 함께 설계한다. 원천 갱신과 캐시 갱신도 자동 분산 transaction이 아니다.

RAG 응답 캐시 key에는 질문만이 아니라 검색 근거/문서 버전·모델·프롬프트·권한 범위를 포함할 계약이 필요하다. 같은 질문을 다른 사용자에게 재사용한다고 접근 권한이 같아지지 않는다. 미확인 freshness/권한을 일치로 기본 처리하지 않는다.

## 6. 용어 alias와 canonical term 매핑

RAG나 검색 시스템에서 Redis는 빠른 용어 정규화 캐시로 쓰기 좋다.

```bash
redis-cli HSET term:cvd \
  canonical_label "Chemical Vapor Deposition" \
  category "deposition_process"

redis-cli SET term_alias:cvd term:cvd
redis-cli SET term_alias:chemical_vapor_deposition term:cvd
redis-cli SET term_alias:화학기상증착 term:cvd
```

사용 흐름:

```text
사용자 쿼리: "CVD 온도 조건"
1. alias 후보 추출: CVD
2. GET term_alias:cvd -> term:cvd
3. HGET term:cvd canonical_label
4. 확장 쿼리: "CVD Chemical Vapor Deposition 온도 조건"
```

이 구조는 용어 lookup의 예다. 약어가 여러 개념을 가리키면 단일 SET이 기존 매핑을 덮어쓸 수 있다. 도메인·버전별 ID 또는 후보 집합과 검토 상태를 정하고 unresolved를 임의의 한 의미로 확정하지 않는다. 원문 질문을 보존하고 확장 결과·사전 출처를 추적한다. 실제 품질은 평가하며 완전 통합·업무 사전 정책은 보류다.

## 7. Redis Search와 Vector Search

Redis Search를 쓰면 Hash나 JSON 문서에 보조 인덱스를 만들고 검색할 수 있다. Vector Search에서는 embedding과 메타데이터를 Hash 또는 JSON에 저장하고 벡터 필드로 검색한다.

다음은 **메타데이터만** 저장하는 예다. embedding·FT.CREATE/schema·vector query가 없어 그대로는 벡터 검색이 아니다. Hash의 vector field는 지정한 FLOAT32 등 TYPE/DIM에 맞는 binary-safe byte buffer, JSON은 수치 배열을 사용한다. 모델·차원·거리 함수·실제 인덱스/필터 조건을 맞춰야 한다. Hash 문자열의 page가 숫자 필터가 되는지, pipe 문자열 canonical_terms가 TAG 목록이 되는지는 별도 schema/구분자 규칙에 달려 있다.

RAG용 예시 key:

```bash
redis-cli HSET rag_chunk:manual_100:3:2 \
  chunk_id "manual_100:3:2" \
  version "2026-05-02" \
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

## 검토 결과

2026-10-04: 기존 key/Hash/JSON/보조 index/cache/alias/chunk 예제와 모든 절·작성일·경로를 보존했다. email 식별 예시를 정정하고 실행 fence를 shell 호출로 명확히 했다. chunk/version을 보강하고 원천·TTL·동시성·Cluster·8.0 JSON/Search·벡터 bytes 조건을 설명했다. 01은 일반 정의, 03은 검색 projection, 04는 문서 저장, 이 문서는 캐시 접근/무효화다. Claude 협의는 HERDR_ENV=1에서 pane_not_found로 불가했고 완전 통합·업무별 key/alias/권한/원천 동기화 구현은 보류했다. [정리 기록](../organization-log.md)에 검증·미확인을 남긴다.

## 참고 자료

공식 문서를 2026-10-04 확인했다. HEXPIRE의 since7.4.0, Open Source8.0 GA 릴리스 기능과 latest 명령 안내를 구분하며 최신·설치 호환을 주장하지 않는다. 원래 Data Type Comparison/Indexing 링크는 추가 읽기 목록이며 이 장의 수정 근거로 전문을 확인한 것은 아니다.

- [Redis Hashes](https://redis.io/docs/latest/develop/data-types/hashes/)
- [Redis JSON](https://redis.io/docs/latest/develop/data-types/json/)
- [Redis Data Type Comparison](https://redis.io/docs/latest/develop/data-types/compare-data-types/)
- [Redis Vector Search Concepts](https://redis.io/docs/latest/develop/ai/search-and-query/vectors/)
- [Redis Search Indexing](https://redis.io/docs/latest/develop/ai/search-and-query/indexing/)

- [HEXPIRE since7.4.0](https://redis.io/docs/latest/commands/hexpire/)
- [HGETALL 복잡도](https://redis.io/docs/latest/commands/hgetall/)
- [EXPIRE와 overwrite](https://redis.io/docs/latest/commands/expire/)
- [Transactions](https://redis.io/docs/latest/develop/using-commands/transactions/)
- [Cluster hash tags/multi-key](https://redis.io/docs/latest/operate/oss_and_stack/reference/cluster-spec/)
- [Open Source8.0 릴리스](https://redis.io/docs/latest/operate/oss_and_stack/stack-with-enterprise/release-notes/redisce/redisos-8.0-release-notes/)
