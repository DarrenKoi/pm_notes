---
tags: [ontology, normalization, taxonomy, glossary, knowledge-graph, data-modeling]
level: advanced
last_updated: 2026-05-02
reviewed_on: 2026-10-04
review_status: reviewed_with_limits
document_type: learning_note
---

# 온톨로지 관점의 정규화

> 온톨로지가 "도메인에 어떤 것들이 존재하며 어떻게 관계 맺는가"를 정의한다면, 정규화는 그 개념들이 데이터 구조 안에서 올바른 책임 위치를 갖도록 만드는 논리 모델링 규율이다.

> [!info] 관점과 표준 — 2026-10-04
> “ontology-lite”와 아래 계층은 이 학습 문서의 설계 용어다. JSON 저장만으로 OWL 추론이나 정합성 검사가 활성화되지 않는다. W3C SKOS2009·OWL2 Primer2판2012·SHACL2017 자료와 정규형의 차이를 대조했다. 실제 RDF/OWL/SHACL 엔진은 미실행이다.

## 온톨로지와 정규화의 차이

| 구분 | 온톨로지 | 정규화 |
|------|----------|--------|
| 핵심 질문 | 이 도메인에는 어떤 개념과 관계가 있는가? | 이 속성은 어느 객체의 사실인가? |
| 산출물 | class, property, relationship, constraint, term | table/collection/document structure, key, dependency |
| 관심사 | 의미, 분류, 추론, 공유 어휘 | 중복, 종속성, 무결성, 변경 비용 |
| 위치 | 개념 모델/의미 모델 | 논리 모델/물리 모델로 가는 중간 규율 |
| 예시 | `Customer`, `Order`, `placesOrder` | `customers`, `orders`, `order_items` |

이 표의 정규화는 모델링 관점이다. 정확한 관계형 정규형은 [01](./01-normalization-core.md)의 키/FD 기준으로 검사한다. ontology class/property를 테이블/컬럼으로 옮기는 것만으로 충족되지 않는다.

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

위 순서는 학습용 설계 흐름이며 W3C가 정한 필수 실행 순서가 아니다. 이 흐름에서 정규화 검토는2번과4번 사이에 둔다. 온톨로지를 데이터베이스 구조로 옮길 때 의미가 섞이지 않도록 잡아주는 역할이다.

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

NULL은 subtype 점검 신호일 수 있지만 모름·미수집·해당 없음도 가능하다. NULL만으로 분류 누락/정규형 위반을 확정하지 않는다. 분리된 subtype의 일치·필수값·권한은 별도 제약으로 적용한다.

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

이 설계는 참여 관계에 ID를 부여하는 선택지다. 반복 참여·기간 중첩·역할 변경의 업무 규칙과 무손실/키 조건을 확인한다.

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

모든 프로젝트가 OWL/RDF 수준의 형식 온톨로지를 도입할 필요는 없다. 이 문서는 glossary(용어 정의), taxonomy(분류 관계), canonical ID를 묶는 경량 계약을 ontology-lite라고 부른다. 모든 RAG가 OWL 또는 이런 계약을 필수로 갖춰야 한다는 뜻은 아니다.

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

JSON의 broader/related는 애플리케이션 필드다. SKOS로 변환한다면 IRI/언어 태그/직접 계층/연관 관계를 명시한다. SKOS broader는 자체로 transitive가 아니고 broaderTransitive를 구분한다. prefLabel은 한 언어당 최대 하나라는 무결성 조건이 있으며 업무의 공식 승인 표기는 별도 정책이다. source가 위 JSON에 없으므로 정의 승인/출처와 revision을 추가해야 한다.

이 계약은 [04 MongoDB](./04-mongodb-normalization.md)·[05 Redis](./05-redis-normalization.md)·[03 OpenSearch](./03-opensearch-normalization.md)에 투영할 수 있다. canonical_terms에 ID를 넣는지 label을 넣는지 명세로 구분한다.

## 정규화와 Knowledge Graph

다음은 RDF 방식의 subject/predicate/object를 설명한 텍스트다. 실제 Turtle이 아니며 prefix/IRI 선언이 필요하다. 모든 Knowledge Graph가 RDF 표현만 사용하는 것은 아니다.

```text
customer:1001  placesOrder  order:9001
order:9001     hasItem      product:p1
product:p1     belongsTo    category:laptop
```

triple 표현은 관계형3NF/BCNF 충족의 증명이 아니다. 조회 비용은 구현/데이터/인덱스로 측정한다. 다음은 역할을 나누는 한 가지 배치 예다.

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

class/property/relationship를 table/field/FK/edge로 대응하는 것은 선택지이며 일대일 기계 변환이 아니다. OWL의 open-world는 누락 사실을 false로 확정하지 않으며 DB의 NOT NULL/FK 제약과 구분한다. 필수 속성 같은 RDF 데이터 검증에는 SHACL shape/minCount 등의 명시적 조건을 검토한다. JSON 저장이나 OWL class 선언만으로 실행되지 않는다.

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

기대 효과(평가로 확인할 가설):

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
5. ID/용어 계약은 근거 추적에 도움이 될 수 있지만 ontology가 모든 RAG의 필요조건이나 정답 보장은 아니다. 실제 답변/충돌/검색 평가가 필요하다.

## 체크리스트

- [ ] 도메인의 핵심 class와 relationship을 먼저 정의했는가?
- [ ] 같은 개념을 가리키는 용어와 약어가 canonical ID로 연결되는가?
- [ ] 정규화 과정에서 드러난 관계 객체를 ontology에 반영했는가?
- [ ] NULL이 많은 구조를 subtype 누락 신호로 검토했는가?
- [ ] 이력과 사건을 별도 event class로 볼 필요가 있는가?
- [ ] ontology의 ID가 MongoDB, OpenSearch, Redis, RAG chunk에 일관되게 전달되는가?
- [ ] LLM이 생성한 용어/관계 후보를 검증해 ontology에 반영하는 절차가 있는가?

## 검토 결과

2026-10-04: 모든 원래 객체/subtype/참여/event/glossary/KG/RAG 예제와 절·작성일을 보존했다. 학습용 계층과 정규형/표준을 구분하고 NULL·추론·검증·효과의 보장 한계를 정정했다. 업무 ontology 선택·완전 통합은 Claude pane_not_found로 보류했으며 실제 reasoner/shape engine/검색은 미실행이다. [정리 기록](../organization-log.md)을 참고한다.

## 참고 자료

아래 W3C 판본을2026-10-04 확인했다. 미래 개정판이나 설치 라이브러리 호환을 보장하지 않는다. 기존 제품 링크는 같은 주제의 적용 문서로 연결해 해당 판본·조건을 따른다.

- [정규화(Normalization)란 무엇인가 - 교과서 너머의 이해](https://wikidocs.net/blog/%40jcnahm/12324/)
- [OpenSearch 적용](./03-opensearch-normalization.md)
- [MongoDB 적용](./04-mongodb-normalization.md)
- [SKOS Reference, 2009](https://www.w3.org/TR/skos-reference/)
- [OWL2 Primer2판, 2012](https://www.w3.org/TR/owl2-primer/)
- [SHACL, 2017](https://www.w3.org/TR/shacl/)
- [RDF1.1 Concepts, 2014; 같은 확인일](https://www.w3.org/TR/rdf11-concepts/)

- [NIST 연구, 기체 전구체/CVD 박막 증착, 2019; 2026-10-04 확인](https://www.nist.gov/publications/apparatus-characterizing-gas-phase-chemical-precursor-delivery-thin-film-deposition) — 용어 배경이며 가상 장비의 실제 조치 근거가 아님.
