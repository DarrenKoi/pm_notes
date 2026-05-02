---
tags: [normalization, data-modeling, dependency, normal-form]
level: intermediate
last_updated: 2026-05-02
---

# 정규화 핵심 개념

> 정규화는 중복을 줄이는 절차이기도 하지만, 더 본질적으로는 "어떤 사실이 어느 객체에 속하는가"를 밝히는 모델링 활동이다.

## 왜 필요한가? (Why)

초기 개발에서는 화면, API 응답, 엑셀 양식, 검색 결과 화면을 그대로 테이블이나 문서 구조로 만드는 일이 많다.

예를 들어 `주문조회_통합` 구조에 고객명, 고객주소, 상품명, 배송상태, 결제수단을 모두 넣으면 처음에는 편하다. 하지만 고객 주소가 바뀌면 주문 조회, 배송 조회, 영수증, 구매 이력 등 여러 위치를 동시에 수정해야 한다. 하나라도 빠지면 같은 고객의 주소가 화면마다 달라진다.

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

정규화된 구조에서는 고객 주소의 주인이 `customers`로 명확해진다. 화면은 여러 객체를 조합해서 보여주는 창일 뿐, 그 창의 모양이 데이터의 본래 구조가 되면 안 된다.

## 교과서적 정의와 실무적 정의

교과서적 설명은 보통 다음과 같다.

- 데이터 중복을 제거한다.
- 삽입, 갱신, 삭제 이상을 방지한다.
- 함수 종속성을 기준으로 테이블을 분해한다.

이 정의는 맞지만 결과 중심이다. 실무에서는 다음 정의가 더 유용하다.

> 정규화는 데이터 안에 섞여 있는 객체, 관계, 사건, 분류를 찾아 각자의 책임 위치에 배치하는 일이다.

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

### 2. 종속 관계를 밝힌다

정규화의 핵심 질문은 다음이다.

> 이 속성은 무엇에 관한 사실인가?

예를 들어 `직원(employee_id, employee_name, department_code, department_name)`에서 `department_name`은 직원의 사실이 아니라 부서의 사실이다.

```text
employee_id -> department_code
department_code -> department_name
```

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

카드 결제에서는 `bank_account`가 해당 없고, 계좌이체에서는 `card_number`가 해당 없다. 이 NULL은 "모름"이 아니라 "그 유형에는 존재하지 않는 속성"이다.

```text
payments(payment_id, order_id, payment_type, amount)
card_payments(payment_id, card_token, card_company)
bank_transfers(payment_id, bank_code, account_hash)
simple_payments(payment_id, provider_code, transaction_key)
```

분류를 구조로 표현하면 유효성 검증, 보안, 검색 필터링, 분석 기준이 명확해진다.

## 정규형을 실무 언어로 이해하기

| 정규형 | 핵심 질문 | 흔한 위반 신호 | 드러나는 것 |
|--------|-----------|----------------|-------------|
| 1NF | 한 칸에 여러 값이 들어가는가? | 콤마 문자열, 반복 컬럼, 배열에 가까운 컬럼 | 관계 객체 |
| 2NF | 복합키 전체가 아니라 일부에만 종속되는 속성이 있는가? | 주문상세에 상품명, 수강에 강좌명 | 독립 객체 |
| 3NF | 비키 속성이 다른 비키 속성을 결정하는가? | 직원 테이블의 부서명, 주문 테이블의 고객등급 | 숨은 주체 |
| BCNF | 모든 결정자가 후보키인가? | 예외적 업무 규칙이 한 테이블 안에 섞임 | 더 엄격한 결정 관계 |

보통 실무 OLTP 모델에서는 3NF까지를 기본 목표로 삼고, 특수한 제약이 있으면 BCNF 이상을 검토한다. 다만 정규형 이름보다 중요한 것은 "속성이 올바른 주체를 찾았는가"이다.

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

원천과 projection을 구분하지 못하면 반정규화는 곧 데이터 불일치가 된다. 반대로 기준 모델이 분명하면 OpenSearch 인덱스, MongoDB read model, Redis cache는 안전하게 반정규화할 수 있다.

## 정규화의 확장된 의미

관계형 DB 밖에서도 정규화는 계속 등장한다.

| 영역 | 정규화 대상 | 예시 |
|------|-------------|------|
| 값 | 표기, 단위, 형식 | 이메일 소문자화, 전화번호 E.164, 날짜 ISO 8601 |
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

## 참고 자료

- [정규화(Normalization)란 무엇인가 - 교과서 너머의 이해](https://wikidocs.net/blog/%40jcnahm/12324/)
