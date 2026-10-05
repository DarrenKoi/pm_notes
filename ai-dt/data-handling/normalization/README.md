---
tags: [normalization, data-modeling, opensearch, mongodb, redis, rag, ontology]
level: intermediate
last_updated: 2026-05-02
reviewed_on: 2026-10-04
review_status: reviewed_with_limits
document_type: topic_index
category_major: "AI·DT"
category_middle: "데이터 엔지니어링"
category_minor: "정규화·모델링"
note_kind: "목차"
classified_on: "2026-10-05"
---

# 정규화 학습 노트

> 정규화를 단순히 "중복 제거"가 아니라, 데이터 안에 숨어 있는 객체, 종속 관계, 분류 체계, 검색 좌표계를 드러내는 모델링 활동으로 이해하기 위한 문서 모음

> [!info] 검토 범위 — 2026-10-04
> 이 목차와01~08 전체를 검토했다. 모든 원래 절·작성일·고유 예제 맥락을 보존하고 적용 조건을 보완했다. 실제 제품 서버/모델·회사 데이터·Claude 통합 협의는 미확인이다. 정규형 이론과 값 표준화·검색 점수 변환은 다른 작업이며 아래 여섯 층위는 학습용 분류다. 판본·보류는 [정리 기록](../organization-log.md)에 남긴다.

## 왜 필요한가? (Why)

정규화는 관계형 데이터베이스의 이론으로 자주 소개되지만, 실무에서는 더 넓은 문제를 다룬다.

- 같은 사실이 여러 곳에 흩어져 서로 다르게 변하는 문제
- 화면이나 API 응답 모양을 그대로 저장해서 운영 중 변경 비용이 커지는 문제
- 검색, 캐시, 문서DB, 벡터DB에서 같은 개념이 다른 이름과 형태로 저장되는 문제
- LLM/RAG가 중복 문서, 모호한 용어, 충돌하는 사실을 근거로 답하는 문제
- 온톨로지나 용어 사전 없이 도메인 개념이 코드와 문서에 흩어지는 문제

참고 글인 "정규화(Normalization)란 무엇인가 - 교과서 너머의 이해"는 정규화를 화면 중심 설계에서 벗어나 현실 세계의 객체와 관계를 데이터 구조에 담는 행위로 설명한다. 이 폴더는 그 저자의 모델링 관점을 학습용으로 확장한다. 이 글은 배경 자료이며 관계형 정규형이나 제품 API의 공식 정의를 대신하지 않는다.

## 문서 구성

01은 용어와 판단 기준, 02는 실제 설계 순서다. 먼저 두 문서를 읽고 저장소 적용(03~05), 의미·근거 설계(06~07), 층 간 비교(08)로 진행한다. 유사 설명은 역할을 구분해 보존했고 완전 통합은 Claude 연결 복구 후 검토한다.

| 순서 | 문서 | 내용 |
|------|------|------|
| 1 | [정규화 핵심 개념](./01-normalization-core.md) | 교과서적 정의를 넘어 객체, 종속, 분류, 좌표계 관점으로 이해 |
| 2 | [모델링 프로세스와 체크리스트](./02-modeling-process-checklist.md) | 정규화 절차, 위반 신호, 반정규화 판단 기준 |
| 3 | [OpenSearch에서의 정규화](./03-opensearch-normalization.md) | analyzer/normalizer, nested, join, 검색용 반정규화, 하이브리드 점수 정규화 |
| 4 | [MongoDB에서의 정규화](./04-mongodb-normalization.md) | embedding vs reference, JSON Schema, 스냅샷, Vector Search/RAG 저장 구조 |
| 5 | [Redis에서의 정규화](./05-redis-normalization.md) | 키 설계, Hash/JSON/Set, 캐시 무효화 단위, 용어 alias 매핑 |
| 6 | [LLM과 RAG에서의 정규화](./06-llm-rag-normalization.md) | 문서 수집, 청킹, 메타데이터, 쿼리 확장, 검색 품질과의 관계 |
| 7 | [온톨로지 관점의 정규화](./07-ontology-normalization.md) | ontology, taxonomy, glossary, logical schema 사이에서 정규화의 위치 |
| 8 | [Cross-Layer Cheatsheet](./08-cross-layer-cheatsheet.md) | 한 사실이 RDB → MongoDB → OpenSearch → Redis → RAG chunk를 거치며 어떻게 같은 정체성을 유지하는가 |

## 핵심 요약

정규화는 하나의 기술이 아니라 여러 층위의 활동이다.

| 층위 | 질문 | 예시 |
|------|------|------|
| 값 정규화 | 같은 값을 같은 형태로 표현하는가? | 이메일 도메인의 대소문자 처리, 전화번호 포맷, 시간대·단위 명시 |
| 구조 정규화 | 이 속성은 어느 객체의 사실인가? | 고객의 현재 주소와 주문 당시 배송 주소 스냅샷을 구분 |
| 관계 정규화 | 관계 자체가 객체인가? | 직원-프로젝트 참여, 주문-상품 주문항목 |
| 의미 정규화 | 같은 개념을 같은 용어와 ID로 부르는가? | CVD, Chemical Vapor Deposition, 화학기상증착을 canonical term으로 연결 |
| 검색 정규화 | 검색 가능한 좌표계가 일관적인가? | OpenSearch `normalizer`, synonym, hybrid score normalization |
| RAG 정규화 | LLM에 들어가는 근거가 추적 가능하고 충돌하지 않는가? | `doc_id`, `chunk_id`, `entity_id`, `source`, `version` 관리 |

이메일 전체를 소문자로 덮어쓰지 않는다. RFC 5321 §2.4는 local-part의 대소문자 보존을 요구하며 도메인은 대소문자를 구분하지 않는다. 서비스별 동일인 판정은 해당 제공자의 계약을 별도로 확인한다. [SMTP 원문](https://www.rfc-editor.org/rfc/rfc5321.html)을 2026-10-04 확인했다.

## 참고 자료

아래 제품 링크는 원래 학습 출발점을 보존한 목록이다. 각 장에 확인한 공식 판본/조회 오류와 적용 조건을 기록했다. 이 목록 자체는 설치 호환·실제 검색·최신 판본 보장이 아니다.

- [정규화(Normalization)란 무엇인가 - 교과서 너머의 이해](https://wikidocs.net/blog/%40jcnahm/12324/)
- [OpenSearch Normalizer](https://docs.opensearch.org/latest/mappings/mapping-parameters/normalizer/)
- [OpenSearch Object Field Types](https://docs.opensearch.org/latest/mappings/supported-field-types/object-fields/)
- [OpenSearch Normalization Processor](https://docs.opensearch.org/2.15/search-plugins/search-pipelines/normalization-processor/)
- [MongoDB Embedded Data](https://www.mongodb.com/docs/manual/data-modeling/embedding/)
- [MongoDB Reference Data](https://www.mongodb.com/docs/manual/data-modeling/referencing/)
- [MongoDB Schema Validation](https://www.mongodb.com/docs/current/core/schema-validation/)
- [MongoDB Vector Search Overview](https://www.mongodb.com/docs/atlas/atlas-search/vector-search/)
- [Redis Hashes](https://redis.io/docs/latest/develop/data-types/hashes/)
- [Redis JSON](https://redis.io/docs/latest/develop/data-types/json/)
- [Redis Vector Search Concepts](https://redis.io/docs/latest/develop/ai/search-and-query/vectors/)
