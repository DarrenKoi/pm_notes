---
tags: [normalization, data-modeling, checklist, denormalization]
level: intermediate
last_updated: 2026-05-02
---

# 모델링 프로세스와 체크리스트

> 정규화는 정규형 이름을 외워서 적용하는 작업이 아니라, 데이터를 발생시킨 객체와 사건을 찾아 구조로 고정하는 작업이다.

## 전체 흐름

```text
1. 화면/API/문서에서 데이터 항목 수집
2. 항목별 의미와 발생 사건 확인
3. 객체, 관계, 사건 후보 분리
4. 함수 종속성과 식별자 확인
5. 1NF, 2NF, 3NF 관점으로 구조 검증
6. 이력, 보안, 권한, 감사 요구사항 반영
7. 조회/검색/캐시 목적의 projection 설계
8. 원천 모델과 파생 모델의 동기화 규칙 명시
```

## Step 1. 화면이 아니라 데이터 항목을 수집한다

화면 설계서는 좋은 출발점이지만, 화면은 특정 목적의 조합 결과다. 화면 이름을 테이블 이름으로 바꾸면 대부분 통합 테이블이 된다.

나쁜 질문:

- 이 화면에 어떤 컬럼이 필요한가?
- 이 API 응답을 그대로 저장하면 되는가?
- JOIN을 줄이기 위해 한 문서에 다 넣을 수 있는가?

좋은 질문:

- 이 값은 어떤 객체의 사실인가?
- 이 값이 바뀌면 어떤 업무 사건이 발생한 것인가?
- 이 값은 현재 상태인가, 과거 특정 시점의 스냅샷인가?
- 이 값은 원천인가, 다른 값으로부터 계산 가능한가?

## Step 2. 객체, 관계, 사건을 나눈다

| 유형 | 설명 | 예시 |
|------|------|------|
| 객체 | 독립적으로 식별되는 대상 | 고객, 상품, 부서, 장비 |
| 관계 | 객체 사이의 연결 | 고객의 주문, 직원의 프로젝트 참여 |
| 사건 | 시간과 조건을 가진 발생 사실 | 주문 생성, 결제 승인, 등급 변경 |
| 속성 | 객체/관계/사건에 관한 사실 | 고객명, 주문일시, 결제금액 |
| 코드/분류 | 값의 의미를 제한하는 체계 | 결제유형, 주문상태, 장비유형 |

관계 자체에 속성이 붙으면 관계도 객체처럼 다뤄야 한다.

```text
employees(employee_id, name)
projects(project_id, name)
project_memberships(employee_id, project_id, role_code, start_date, end_date)
```

`role_code`, `start_date`, `end_date`는 직원도 프로젝트도 아닌 참여 관계의 속성이다.

## Step 3. 식별자를 정한다

정규화는 "무엇이 같은 것인가"라는 식별 문제와 함께 간다.

| 식별자 유형 | 장점 | 주의점 |
|-------------|------|--------|
| 자연키 | 업무 의미가 명확함 | 변경 가능성, 개인정보 포함 위험 |
| 인조키 | 안정적이고 참조가 쉬움 | 별도 유일성 제약이 필요함 |
| 복합키 | 관계의 의미가 잘 드러남 | 자식 테이블로 전파되면 키 폭이 커짐 |
| 외부 ID | 시스템 통합에 유용함 | 출처별 namespace 관리 필요 |

복합키와 인조키는 경쟁 관계가 아니다. 관계가 단순하고 반복되지 않으면 복합키가 자연스럽다. 같은 쌍이 여러 번 반복되거나 관계의 자식 엔터티가 생기면 인조키를 검토한다.

```text
단순 참여:
  employee_projects(employee_id, project_id, role_code)

반복 참여와 자식 이력이 있는 경우:
  project_memberships(membership_id, employee_id, project_id, role_code, start_date, end_date)
  membership_reviews(review_id, membership_id, score, reviewed_at)
```

## Step 4. 함수 종속성을 문장으로 표현한다

수식보다 먼저 자연어로 적는다.

```text
customer_id가 정해지면 customer_name이 정해진다.
product_id가 정해지면 product_name이 정해진다.
order_id와 product_id가 정해지면 주문 당시 단가가 정해진다.
department_code가 정해지면 department_name이 정해진다.
```

그 다음 위치를 확인한다.

| 속성 | 종속 대상 | 있어야 할 위치 |
|------|-----------|----------------|
| `customer_name` | `customer_id` | `customers` |
| `product_name` | `product_id` | `products` |
| `order_price` | `order_id + product_id` | `order_items` |
| `department_name` | `department_code` | `departments` |

## Step 5. 정규형별 위반 신호를 찾는다

### 1NF 위반 신호

- 콤마로 구분된 목록: `"A,B,C"`
- 반복 컬럼: `phone1`, `phone2`, `phone3`
- 하나의 컬럼에 여러 의미가 섞임: `"서울/강남/역삼"`
- 컬럼 수가 업무 증가에 따라 계속 늘어남

대응:

```text
customers(customer_id, name)
customer_phones(customer_id, phone_id, phone_number, phone_type)
```

### 2NF 위반 신호

- 복합키 테이블에서 키 일부만으로 결정되는 속성이 있음
- 관계 테이블에 한쪽 객체의 속성이 섞임

대응:

```text
수강(student_id, course_id, grade, course_name)

-> courses(course_id, course_name)
-> enrollments(student_id, course_id, grade)
```

### 3NF 위반 신호

- A가 B를 결정하고 B가 C를 결정하는데 C가 A 테이블에 있음
- 어떤 값을 수정할 때 테이블 이름과 다른 객체의 사건이 발생함

대응:

```text
employees(employee_id, name, department_code, department_name)

-> employees(employee_id, name, department_code)
-> departments(department_code, department_name)
```

## Step 6. 이력과 스냅샷을 구분한다

정규화만으로 이력 요구사항이 자동으로 드러나지는 않는다. 이력은 "값이 바뀐 사건을 보존할 것인가"의 문제다.

```text
현재 상태:
  customers(customer_id, grade_code)

이력 보존:
  customer_grade_histories(
    history_id,
    customer_id,
    grade_code,
    valid_from,
    valid_to,
    reason_code
  )
```

주문 당시 고객명, 상품명, 가격처럼 과거 거래 증빙에 필요한 값은 의도적으로 스냅샷을 둔다.

```text
order_items(
  order_item_id,
  order_id,
  product_id,
  product_name_snapshot,
  unit_price_snapshot,
  quantity
)
```

스냅샷은 중복처럼 보이지만, 현재 상품 정보가 아니라 주문 당시 계약 사실을 저장하는 별도 속성이다.

## Step 7. 보안 경계를 구조로 만든다

정규화는 개인정보와 민감정보를 분리하는 데 도움이 된다.

```text
users(user_id, login_id, status)
user_profiles(user_id, name, email, phone)
user_auth_secrets(user_id, password_hash, mfa_secret)
orders(order_id, user_id, amount)
```

구조가 분리되면 접근 제어, 암호화, 감사 로그, 데이터 마스킹 범위를 좁힐 수 있다. 반대로 통합 테이블에 모든 정보가 들어 있으면 읽기 권한 하나가 과도한 노출로 이어진다.

## Step 8. 반정규화는 목적과 재생성 방법을 함께 적는다

반정규화를 할 때는 다음을 반드시 명시한다.

| 질문 | 예시 답변 |
|------|-----------|
| 왜 중복하는가? | 검색 결과를 한 번에 보여주기 위해 |
| 원천은 어디인가? | `customers`, `orders`, `products` |
| 언제 갱신되는가? | 주문 생성 이벤트, 고객 정보 변경 이벤트 |
| 불일치 허용 시간은? | 최대 5분 |
| 재생성 가능한가? | 원천 DB에서 전체 rebuild 가능 |
| 누가 읽는가? | OpenSearch, Redis cache, MongoDB read model |

## 실전 체크리스트

- [ ] 화면/API 이름을 그대로 테이블/컬렉션 이름으로 쓰고 있지 않은가?
- [ ] 같은 사실이 여러 테이블, 문서, 인덱스에 원천처럼 저장되어 있지 않은가?
- [ ] 한 컬럼에 여러 값이나 여러 의미가 들어가 있지 않은가?
- [ ] 복합키 일부에만 종속되는 속성이 없는가?
- [ ] 어떤 컬럼을 바꿀 때 "다른 객체"가 바뀌는 느낌이 들지 않는가?
- [ ] NULL이 특정 유형에서 반복적으로 발생하지 않는가?
- [ ] 이력으로 남겨야 할 사건을 단순 UPDATE로 덮어쓰고 있지 않은가?
- [ ] 개인정보와 인증정보가 업무 데이터와 과도하게 섞여 있지 않은가?
- [ ] 검색/캐시용 반정규화 데이터의 원천과 재생성 방법이 분명한가?
- [ ] LLM/RAG용 문서에 원천 ID, chunk ID, 버전, 출처가 포함되어 있는가?

## 참고 자료

- [정규화(Normalization)란 무엇인가 - 교과서 너머의 이해](https://wikidocs.net/blog/%40jcnahm/12324/)
