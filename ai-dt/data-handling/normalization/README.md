---
tags: [normalization, data-modeling, opensearch, mongodb, redis, rag, ontology]
level: intermediate
last_updated: 2026-05-02
status: in-progress
---

# 정규화 학습 노트

> 정규화를 단순히 "중복 제거"가 아니라, 데이터 안에 숨어 있는 객체, 종속 관계, 분류 체계, 검색 좌표계를 드러내는 모델링 활동으로 이해하기 위한 문서 모음

## 왜 필요한가? (Why)

정규화는 관계형 데이터베이스의 이론으로 자주 소개되지만, 실무에서는 더 넓은 문제를 다룬다.

- 같은 사실이 여러 곳에 흩어져 서로 다르게 변하는 문제
- 화면이나 API 응답 모양을 그대로 저장해서 운영 중 변경 비용이 커지는 문제
- 검색, 캐시, 문서DB, 벡터DB에서 같은 개념이 다른 이름과 형태로 저장되는 문제
- LLM/RAG가 중복 문서, 모호한 용어, 충돌하는 사실을 근거로 답하는 문제
- 온톨로지나 용어 사전 없이 도메인 개념이 코드와 문서에 흩어지는 문제

참고 글인 "정규화(Normalization)란 무엇인가 - 교과서 너머의 이해"는 정규화를 화면 중심 설계에서 벗어나 현실 세계의 객체와 관계를 데이터 구조에 담는 행위로 설명한다. 이 폴더는 그 관점을 OpenSearch, MongoDB, Redis, LLM/RAG, 온톨로지까지 확장한다.

## 문서 구성

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
| 값 정규화 | 같은 값을 같은 형태로 표현하는가? | 이메일 소문자화, 전화번호 포맷, 날짜/단위 통일 |
| 구조 정규화 | 이 속성은 어느 객체의 사실인가? | 고객 주소를 주문 테이블이 아니라 고객 엔터티에 둠 |
| 관계 정규화 | 관계 자체가 객체인가? | 직원-프로젝트 참여, 주문-상품 주문항목 |
| 의미 정규화 | 같은 개념을 같은 용어와 ID로 부르는가? | CVD, Chemical Vapor Deposition, 화학기상증착을 canonical term으로 연결 |
| 검색 정규화 | 검색 가능한 좌표계가 일관적인가? | OpenSearch `normalizer`, synonym, hybrid score normalization |
| RAG 정규화 | LLM에 들어가는 근거가 추적 가능하고 충돌하지 않는가? | `doc_id`, `chunk_id`, `entity_id`, `source`, `version` 관리 |

## 참고 자료

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
