---
tags: [normalization, data-modeling, dependency, normal-form]
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

# 정규화 핵심 개념

> 정규화는 중복을 줄이는 절차이기도 하지만, 더 본질적으로는 "어떤 사실이 어느 객체에 속하는가"를 밝히는 모델링 활동이다.

## 왜 필요한가? (Why)

초기 개발에서는 화면, API 응답, 엑셀 양식, 검색 결과 화면을 그대로 테이블이나 문서 구조로 만드는 일이 많다.

예를 들어 `주문조회_통합` 구조에 고객명, 고객주소, 상품명, 배송상태, 결제수단을 모두 넣으면 처음에는 편하다. 고객의 **현재 주소**를 여러 위치에 원천처럼 저장하면 주소 변경 때 불일치가 생긴다. 다만 과거 주문·영수증의 배송 주소는 주문 당시 사실이므로 현재 주소로 덮어쓰면 안 된다.

정규화는 이 문제를 "나중에 동기화 규칙으로 해결"하는 방식이 아니라, 애초에 스키마가 연결 지도를 품도록 만드는 방식이다.

```text
화면 중심 구조:
  주문조회_통합(customer_name, customer_address, product_name, delivery_status, ...)
  배송조회_통합(customer_name, customer_address, delivery_status, ...)

정규화된 구조:
  customers(customer_id, name, address, ...)
  orders(order_id, customer_id, ordered_at, ...)
  order_items(order_id, product_id, quantity, ...)
  deliveries(delivery_id, order_id, status, ...)
```

정규화된 구조에서는 고객의 현재 주소를 `customers`가 관리한다. 주문 당시 주소는 별도 스냅샷 속성으로 보존한다. 화면은 여러 객체를 조합해서 보여주는 창일 뿐, 그 창의 모양이 데이터의 본래 구조가 되면 안 된다.

## 교과서적 정의와 실무적 정의

교과서적 설명은 보통 다음과 같다.

- 데이터 중복을 제거한다.
- 삽입, 갱신, 삭제 이상을 방지한다.
- 함수 종속성을 기준으로 테이블을 분해한다.

이 정의는 맞지만 결과 중심이다. 실무에서는 다음 정의가 더 유용하다.

> 정규화는 데이터 안에 섞여 있는 객체, 관계, 사건, 분류를 찾아 각자의 책임 위치에 배치하는 일이다.

위 문장은 학습용 모델링 관점이다. 관계형 정규형은 아래 함수 종속성과 키 조건으로 판정하며 객체 분리만으로 정규형 충족을 증명하지 않는다.

즉 테이블이 늘어나는 것이 목적이 아니다. 원래 데이터 안에 섞여 있던 개념들이 제 이름과 위치를 갖게 되는 것이다.

## 정규화가 드러내는 세 가지

### 1. 객체를 드러낸다

하나의 행 안에 고객, 주문, 상품, 배송, 결제가 섞여 있으면 어느 값이 어느 객체의 사실인지 불명확하다.

```text
주문_통합(
  order_id,
  customer_name,
  customer_grade,
  product_name,
  product_price,
  delivery_status
)
```

이 구조를 분석하면 다음 객체가 드러난다.

```text
customers(customer_id, name, grade)
products(product_id, name, current_price)
orders(order_id, customer_id, ordered_at)
order_items(order_id, product_id, order_price, quantity)
deliveries(delivery_id, order_id, status)
```

`product_price`도 주의해야 한다. 현재 상품 가격이라면 `products`에 속하지만, 주문 당시 가격이라면 `order_items`에 속한다. 같은 이름의 컬럼도 "어떤 사건의 사실인가"에 따라 위치가 달라진다.

이 주문항목 예시는 한 주문에 같은 상품이 한 번만 등장한다고 가정한다. 분할 배송·가격별 반복 행이 가능하면 `order_item_id` 등 행 식별자와 업무 유일성 규칙을 별도로 정한다.

### 2. 종속 관계를 밝힌다

정규화의 핵심 질문은 다음이다.

> 이 속성은 무엇에 관한 사실인가?

예를 들어 `직원(employee_id, employee_name, department_code, department_name)`에서 `department_name`은 직원의 사실이 아니라 부서의 사실이다.

```text
employee_id -> department_code
department_code -> department_name
```

위 종속성은 직원별 현재 부서가 하나이고 부서 코드가 현재 부서명을 결정한다는 업무 규칙을 가정한다. 함수 종속성 `X → Y`는 모든 허용 상태에서 같은 X가 같은 Y를 결정한다는 뜻이다. 샘플 행 몇 개가 일치하는 것만으로 증명할 수 없고 이력에는 시점·버전 조건이 필요하다. [교수 강의의 정의](https://www.cs.rpi.edu/~sibel/csci4380/fall2026/lecture_notes/lecture5.html)를 2026-10-04 대조했다.

직원이 부서에 소속된 것은 직원의 사실이지만, 부서명이 무엇인지는 부서의 사실이다. 이행 종속을 제거하면 구조는 다음처럼 바뀐다.

```text
employees(employee_id, employee_name, department_code)
departments(department_code, department_name)
```

### 3. 분류 체계를 드러낸다

NULL이 지나치게 많다면 분류가 구조로 표현되지 않았다는 신호일 수 있다.

```text
payments(
  payment_id,
  payment_type,
  card_number,
  bank_account,
  mobile_provider
)
```

카드 결제에서는 `bank_account`가 해당 없고, 계좌이체에서는 `card_number`가 해당 없다. 이 업무 계약에서는 해당 없는 속성으로 해석할 수 있다. 하지만 NULL 자체로는 모름·미수집·해당 없음을 구분할 수 없으며 NULL이 많다고 정규형 위반인 것도 아니다. 유형과 상태를 명시해 구분한다.

```text
payments(payment_id, order_id, payment_type, amount)
card_payments(payment_id, card_token, card_company)
bank_transfers(payment_id, bank_code, account_hash)
simple_payments(payment_id, provider_code, transaction_key)
```

분류별 구조는 검증 범위를 나누는 데 도움이 된다. 유형과 하위 테이블의 일치·필수값·접근 권한은 별도 제약과 정책으로 적용한다. `card_token`·`account_hash`라는 이름만으로 익명성이나 보안이 보장되지는 않는다.

## 정규형을 실무 언어로 이해하기

후보키는 행 전체를 결정하는 **최소** 속성 집합이고, 슈퍼키는 최소성 없이 행 전체를 결정하는 집합이다. prime 속성은 어느 후보키에라도 포함되는 속성이다. 아래는 고전적 관계 모델의 조건이며 SQL의 NULL·중복 허용은 별도로 다룬다.

| 정규형 | 판정 기준 | 기존 사례의 해석 |
|---|---|---|
| 1NF | 각 속성의 도메인을 원자적으로 취급 | 콤마 목록·반복 컬럼은 모델링 점검 신호이며 문장부호만으로 위반을 판정하지 않음 |
| 2NF | 1NF이고 모든 non-prime 속성이 모든 후보키에 완전 함수 종속 | 주문상세의 상품명·수강의 강좌명이 후보키 일부에만 종속되는지 검사 |
| 3NF | 각 FD `X → A`가 자명하거나 X가 슈퍼키이거나 A가 prime | 직원 부서명·주문 고객등급의 이행 종속 신호와 후보키를 함께 검사 |
| BCNF | 모든 비자명 FD의 결정자 X가 슈퍼키 | 모든 결정자가 최소 후보키여야 한다는 뜻은 아님 |

이 학습 과정은 3NF부터 검토한다. 실제 설계는 업무 FD, 무손실 분해, 종속성 보존과 읽기·쓰기 비용을 함께 평가한다. 인조키 하나를 추가해도 다른 후보키의 부분 종속이 자동으로 사라지지 않는다. 원자성은 도메인의 사용 방식과 함께 판단한다. [저자 교재 5판 7장](https://www.db-book.com/Previous-editions/db5/slide-dir/ch7.pdf), [2NF 저자 설명](https://www.microsoftpressstore.com/articles/article.aspx?p=2730116)을 2026-10-04 확인했다.

### 3NF와 BCNF의 차이를 확인하는 작은 예

교재의 J/K/L 예를 A/B/C로 바꾸면 `R(A,B,C), F={AB → C, C → B}`다. 후보키는 AB·AC이고 모든 속성이 prime이므로 3NF다. C는 슈퍼키가 아니므로 BCNF는 아니다. `R1(C,B)`와 `R2(A,C)` 분해는 무손실이지만 `AB → C`는 각 테이블만 검사해서 보장할 수 없다. BCNF 분해가 언제나 종속성까지 보존한다는 결론은 피한다.

## 정규화와 반정규화의 관계

반정규화는 정규화의 반대편에 있는 실패가 아니다. 잘 정규화된 기준 모델을 바탕으로 읽기 성능, 검색 편의성, 캐시 효율, 분석 편의성을 위해 만드는 파생 모델이다.

```text
기준 모델:
  customers
  orders
  order_items
  products

검색용 문서:
  order_search_documents(
    order_id,
    customer_name,
    product_names,
    delivery_status,
    ordered_at
  )
```

중요한 차이는 원본과 파생물의 구분이다.

- 정규화 모델: 사실의 원천, 변경의 기준, 무결성의 중심
- 반정규화 모델: 조회와 검색을 위한 projection, 재생성 가능한 산출물

원천과 projection을 구분하지 못하면 반정규화는 곧 데이터 불일치가 된다. 기준 모델 외에도 파생물의 버전·멱등성·삭제 반영·이벤트 순서·권한·재생성 계약이 필요하다. 불명확한 동기화 상태를 최신 또는 일치로 간주하지 않는다. 이는 설계 점검 항목이며 실제 동기화 구현은 미확인이다.

## 정규화의 확장된 의미

관계형 DB 밖에서도 같은 용어가 쓰이지만 값 표준화·검색 점수 변환은 관계형 정규형과 다른 작업이다. 이메일 local-part는 전체 소문자화하지 않고 원형을 보존한다. [RFC 5321 §2.4](https://www.rfc-editor.org/rfc/rfc5321.html), 2008 판본을 2026-10-04 확인했다.

| 영역 | 정규화 대상 | 예시 |
|------|-------------|------|
| 값 | 표기, 단위, 형식 | 이메일 도메인 표기, 전화번호 형식·국가 조건, 날짜와 시간대·단위 |
| 구조 | 객체와 종속 관계 | 고객, 주문, 주문항목 분리 |
| 문서 | 중첩과 참조 | MongoDB embedding/reference 선택 |
| 검색 | 토큰, keyword, score | OpenSearch normalizer, synonym, hybrid score normalization |
| 캐시 | 키와 무효화 단위 | Redis `user:{id}`, `term_alias:{alias}` |
| LLM/RAG | chunk, metadata, entity | `source_doc_id`, `chunk_id`, `canonical_entity_id` |
| 온톨로지 | 개념, 관계, 용어 | class, property, canonical label, alias |

## 실무 판단 원칙

- 먼저 현실 세계의 객체와 사건을 식별한다.
- 각 속성이 어느 객체의 사실인지 묻는다.
- 자주 바뀌는 값은 한 곳에 둔다.
- 과거 사실을 보존해야 하면 현재값과 이력 객체를 분리한다.
- 검색과 캐시는 원천 모델이 아니라 projection으로 설계한다.
- 정규화된 구조를 만든 뒤, 목적이 명확할 때만 반정규화한다.

## 현재값과 거래 스냅샷의 로컬 확인

SQLite의 새 임시 `:memory:` 연결에서 아래 SQL을 실행한다. 업무 DB에 적용하는 migration이 아니다. FK 지원 빌드에서 transaction 시작 전에 연결별 `PRAGMA foreign_keys=ON`을 설정한다. [SQLite 공식 FK 안내](https://www.sqlite.org/foreignkeys.html)를 2026-10-04 확인했다.

```sql
PRAGMA foreign_keys=ON;
CREATE TABLE customers (
  customer_id INTEGER PRIMARY KEY,
  current_address TEXT NOT NULL
);
CREATE TABLE orders (
  order_id INTEGER PRIMARY KEY,
  customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
  shipping_address_snapshot TEXT NOT NULL
);
INSERT INTO customers VALUES (1, '서울');
INSERT INTO orders VALUES (10, 1, '서울');
UPDATE customers SET current_address='부산' WHERE customer_id=1;
SELECT c.current_address, o.shipping_address_snapshot
FROM orders AS o JOIN customers AS c ON c.customer_id=o.customer_id;
-- 결과: 부산 | 서울
```

현재 주소 변경은 과거 주문 스냅샷을 덮어쓰지 않는다. 이 작은 예의 결과는 실제 시스템의 모든 이력·동시성·권한 요구 충족을 증명하지 않는다.

## 검토 결과와 적용 범위

2026-10-04: 기존 객체·부서·결제·projection 예제를 보존하고 키/FD 조건, NULL과 이메일, 과거 주소 구분을 보완했다. 교재 5판(2006)과 T-SQL Fundamentals 3판(2016)은 이론 근거이며 최신 제품 판본이 아니다. 제품별 구현은 03~08에서 검토한다. Claude 협의는 `pane_not_found`로 불가했고 파일 통합과 업무별 모델 선택은 보류했다. [정리 기록](../organization-log.md)에 실제 로컬 검증을 남긴다.

## 참고 자료

- [정규화(Normalization)란 무엇인가 - 교과서 너머의 이해](https://wikidocs.net/blog/%40jcnahm/12324/)
