---
tags: [unit-test, testing, pytest, vitest, jest, tdd]
level: beginner
last_updated: 2026-02-19
status: in-progress
---

# Unit Testing (단위 테스트) 기초

> 코드의 가장 작은 단위가 의도한 대로 동작하는지 자동으로 검증하는 기법

---

## 왜 필요한가? (Why)

### 개발 팀이 Unit Test를 필수로 여기는 이유

많은 팀이 처음에는 "코드 짜기도 바쁜데 테스트까지?"라고 생각한다.
하지만 프로젝트 규모가 커지면 테스트 없이 개발하는 것이 오히려 더 느려진다.

**1. 버그를 가장 저렴한 시점에 잡는다**

버그를 발견하는 시점에 따라 수정 비용이 기하급수적으로 증가한다:

```
개발 중 발견  → 비용 1x    (방금 짠 코드, 맥락이 생생함)
테스트 중 발견 → 비용 10x   (원인 추적에 시간 소요)
프로덕션 발견  → 비용 100x  (사용자 피해 + 롤백 + 긴급 패치)
```

**2. 리팩토링을 두려움 없이 할 수 있다**

테스트가 없으면 코드를 수정할 때마다 "혹시 다른 기능이 깨지지 않을까?" 불안하다.
테스트가 있으면 수정 후 `pytest` 한 번으로 전체 검증이 끝난다.

**3. 코드가 스스로 문서가 된다**

좋은 테스트 코드는 "이 함수가 어떻게 동작해야 하는가"를 예제로 보여준다.
6개월 후의 나, 또는 새 팀원이 함수 사용법을 테스트 코드에서 바로 이해할 수 있다.

**4. 빠른 피드백 루프**

코드 수정 → 테스트 실행 (수 초) → 결과 확인
수동으로 앱을 실행하고 UI를 클릭해가며 확인하는 것보다 훨씬 빠르다.

**5. 협업 시 안전망**

다른 사람이 내 코드를 수정할 때, 내가 기대하는 동작이 깨지면 테스트가 알려준다.
"작동하는 척하는 코드"가 아닌 "실제로 작동하는 코드"를 보장한다.

---

## 핵심 개념 (What)

### Unit(단위)이란?

Unit Test에서 "단위"는 보통 **하나의 함수 또는 메서드**다.

```
❌ 너무 큰 단위: "로그인 기능 전체"
✓ 적절한 단위: "비밀번호가 8자 이상인지 검증하는 함수"
✓ 적절한 단위: "이메일 형식이 올바른지 확인하는 함수"
```

### 핵심 용어

| 용어 | 설명 | 예시 |
|------|------|------|
| **Test Case** | 하나의 테스트 시나리오 | "빈 문자열을 입력하면 False를 반환한다" |
| **Assertion** | 기대값과 실제값을 비교하는 검증 | `assert result == True` |
| **Test Runner** | 테스트를 실행하고 결과를 보여주는 도구 | pytest, vitest, jest |
| **Mock** | 외부 의존성을 가짜로 대체하는 것 | DB 연결 대신 가짜 데이터 반환 |
| **Coverage** | 전체 코드 중 테스트가 실행하는 코드의 비율 | "커버리지 80%" |
| **TDD** | 테스트를 먼저 작성하고 코드를 구현하는 방법론 | Test-Driven Development |

### 테스트 피라미드 (Testing Pyramid)

```
        /\
       /  \
      / E2E \        ← 느림, 비용 높음, 전체 흐름 검증
     /--------\
    / Integration\   ← 중간, 여러 모듈 연동 검증
   /--------------\
  /   Unit Tests   \  ← 빠름, 비용 낮음, 많이 작성
 /------------------\
```

**Unit Test를 가장 많이 작성하는 이유:**
- 실행 속도가 빠름 (ms 단위)
- 실패 원인이 명확함 (어떤 함수의 어떤 케이스)
- 유지보수 비용이 낮음

### AAA 패턴 (Arrange - Act - Assert)

모든 Unit Test는 이 3단계로 구성된다:

```python
def test_something():
    # Arrange (준비): 테스트에 필요한 데이터/환경 설정
    input_value = "hello@email.com"

    # Act (실행): 테스트할 함수를 실제로 실행
    result = is_valid_email(input_value)

    # Assert (검증): 결과가 기대값과 일치하는지 확인
    assert result == True
```

---

## 어떻게 사용하는가? (How)

### Python - pytest

#### 설치

```bash
pip install pytest
# 또는 uv 사용 시
uv add pytest --dev
```

#### 기본 예제

테스트할 함수 (`calculator.py`):

```python
def add(a, b):
    return a + b

def divide(a, b):
    if b == 0:
        raise ValueError("0으로 나눌 수 없습니다")
    return a / b

def is_valid_email(email: str) -> bool:
    return "@" in email and "." in email.split("@")[-1]
```

테스트 파일 (`test_calculator.py`):

```python
import pytest
from calculator import add, divide, is_valid_email


# 기본 테스트
def test_add_positive_numbers():
    # Arrange
    a, b = 3, 5
    # Act
    result = add(a, b)
    # Assert
    assert result == 8


def test_add_negative_numbers():
    assert add(-1, -2) == -3


def test_add_zero():
    assert add(0, 5) == 5


# 예외 테스트
def test_divide_by_zero_raises_error():
    with pytest.raises(ValueError, match="0으로 나눌 수 없습니다"):
        divide(10, 0)


def test_divide_normal():
    assert divide(10, 2) == 5.0


# 이메일 검증 - 여러 케이스를 한번에 (parametrize)
@pytest.mark.parametrize("email, expected", [
    ("user@example.com", True),
    ("invalid-email",    False),
    ("no-at-sign.com",   False),
    ("@nodomain",        False),
    ("",                 False),
])
def test_is_valid_email(email, expected):
    assert is_valid_email(email) == expected
```

#### 실행

```bash
# 기본 실행
pytest

# 자세한 출력
pytest -v

# 특정 파일만
pytest test_calculator.py

# 특정 테스트만
pytest test_calculator.py::test_add_positive_numbers

# 커버리지 포함
pip install pytest-cov
pytest --cov=. --cov-report=term-missing
```

#### Mock 사용 예제

외부 API나 DB 호출을 테스트할 때 실제로 호출하지 않고 가짜(Mock)로 대체:

```python
from unittest.mock import patch, MagicMock
from my_service import get_user_name  # DB에서 유저 이름을 가져오는 함수


def test_get_user_name():
    # DB 호출을 Mock으로 대체
    with patch("my_service.database.query") as mock_query:
        mock_query.return_value = {"name": "Daeyoung", "id": 1}

        result = get_user_name(user_id=1)

        assert result == "Daeyoung"
        mock_query.assert_called_once_with(user_id=1)  # 올바른 인자로 호출됐는지도 검증
```

---

### TypeScript - Vitest (Vue/Vite 프로젝트)

Vue/Vite 프로젝트에서는 **Vitest**를 주로 사용한다. (Jest와 API가 거의 동일하지만 Vite 친화적)

#### 설치

```bash
npm install -D vitest
```

`vite.config.ts`에 추가:

```typescript
import { defineConfig } from 'vite'

export default defineConfig({
  test: {
    globals: true,  // describe, it, expect를 import 없이 사용
  },
})
```

#### 기본 예제

테스트할 유틸 함수 (`src/utils/validator.ts`):

```typescript
export function isValidEmail(email: string): boolean {
  return email.includes('@') && email.split('@')[1]?.includes('.')
}

export function formatPrice(price: number, currency = 'KRW'): string {
  return `${price.toLocaleString()} ${currency}`
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}
```

테스트 파일 (`src/utils/validator.test.ts`):

```typescript
import { describe, it, expect } from 'vitest'
import { isValidEmail, formatPrice, clamp } from './validator'


describe('isValidEmail', () => {
  it('유효한 이메일을 true로 반환한다', () => {
    expect(isValidEmail('user@example.com')).toBe(true)
  })

  it('@가 없으면 false를 반환한다', () => {
    expect(isValidEmail('invalid-email')).toBe(false)
  })

  it('빈 문자열은 false를 반환한다', () => {
    expect(isValidEmail('')).toBe(false)
  })
})


describe('formatPrice', () => {
  it('기본 통화(KRW)로 포맷팅한다', () => {
    expect(formatPrice(10000)).toBe('10,000 KRW')
  })

  it('다른 통화로 포맷팅한다', () => {
    expect(formatPrice(100, 'USD')).toBe('100 USD')
  })
})


describe('clamp', () => {
  it('값이 범위 내에 있으면 그대로 반환한다', () => {
    expect(clamp(5, 0, 10)).toBe(5)
  })

  it('최솟값보다 작으면 최솟값을 반환한다', () => {
    expect(clamp(-5, 0, 10)).toBe(0)
  })

  it('최댓값보다 크면 최댓값을 반환한다', () => {
    expect(clamp(15, 0, 10)).toBe(10)
  })
})
```

#### 실행

```bash
# 한 번 실행
npx vitest run

# watch 모드 (파일 변경 시 자동 재실행)
npx vitest

# UI로 보기
npx vitest --ui

# 커버리지
npx vitest run --coverage
```

---

### 좋은 테스트를 작성하는 원칙

#### 1. 한 테스트는 한 가지만 검증한다

```python
# ❌ 나쁜 예: 여러 가지를 한 번에 검증
def test_user_creation():
    user = create_user("daeyoung", "pass123", "user@email.com")
    assert user.name == "daeyoung"
    assert user.is_active == True
    assert user.email_verified == False
    assert len(user.id) == 36  # UUID 길이

# ✓ 좋은 예: 각각 분리
def test_user_creation_sets_correct_name():
    user = create_user("daeyoung", "pass123", "user@email.com")
    assert user.name == "daeyoung"

def test_new_user_is_active_by_default():
    user = create_user("daeyoung", "pass123", "user@email.com")
    assert user.is_active == True
```

#### 2. 테스트 이름은 "상황 → 기대 결과"로 명확하게

```python
# ❌ 모호한 이름
def test_email():
def test_1():
def test_validate():

# ✓ 명확한 이름
def test_is_valid_email_returns_false_for_empty_string():
def test_divide_raises_value_error_when_divisor_is_zero():
def test_get_user_returns_none_when_user_not_found():
```

#### 3. Edge Case(경계값)를 빼놓지 않는다

```python
# 정상 케이스만 테스트하면 실무에서 버그가 터진다
def test_add():
    assert add(2, 3) == 5  # 이것만 테스트하면 부족!

# Edge case도 함께
def test_add_large_numbers():
    assert add(10**9, 10**9) == 2 * 10**9

def test_add_with_zero():
    assert add(0, 0) == 0

def test_add_negative_and_positive():
    assert add(-5, 5) == 0
```

---

## TDD (Test-Driven Development) 맛보기

TDD는 구현 전에 테스트를 먼저 작성하는 방법론이다.

```
1. Red   → 실패하는 테스트 작성 (아직 구현이 없으니 당연히 실패)
2. Green → 테스트를 통과하는 최소한의 코드 작성
3. Refactor → 코드 개선 (테스트가 있으니 안전하게)
```

실제 예시:

```python
# Step 1 (Red): 먼저 테스트 작성
def test_password_must_be_at_least_8_chars():
    assert is_valid_password("short") == False
    assert is_valid_password("longenough") == True

# → 테스트 실행하면 NameError: is_valid_password 함수가 없음

# Step 2 (Green): 함수 구현
def is_valid_password(password: str) -> bool:
    return len(password) >= 8

# → 테스트 통과!

# Step 3 (Refactor): 요구사항 추가 → 테스트 먼저 추가
def test_password_must_contain_number():
    assert is_valid_password("nodigitshere") == False
    assert is_valid_password("hasdigit1") == True

# → 테스트 실패 → 함수 수정
def is_valid_password(password: str) -> bool:
    return len(password) >= 8 and any(c.isdigit() for c in password)
```

---

## 실무 연결 포인트

### Recipe Setup 자동화 프로젝트에서의 활용
- 레시피 파라미터 파싱 함수에 unit test 적용
- 설비 데이터 변환 로직의 edge case 검증
- API 엔드포인트 입력 검증 함수 테스트

### FastAPI 서비스에서의 활용
- 비즈니스 로직 함수 (데이터 변환, 계산) 단위 테스트
- Pydantic 모델 검증 테스트
- 서비스 레이어 함수를 Mock 활용해 테스트

### Vue/TypeScript 프로젝트에서의 활용
- 유틸리티 함수 (포맷팅, 날짜 처리, 검증) 테스트
- Pinia store action 로직 테스트
- Composable 함수 테스트

---

## 참고 자료 (References)

- [pytest 공식 문서](https://docs.pytest.org/)
- [Vitest 공식 문서](https://vitest.dev/)
- [Python unittest.mock 공식 문서](https://docs.python.org/3/library/unittest.mock.html)

## 관련 문서
- [FastAPI 개발](../python/fastapi/)
- [Vue 3 with TypeScript](../typescript/vue/vue3-with-typescript.md)
- [Vite 기초](../typescript/vite-basics.md)
