---
tags: [pytest, unittest, jest, vitest, mocha, testing-framework]
level: beginner
last_updated: 2026-02-19
status: in-progress
---

# 테스트 프레임워크 비교 및 사용법

> Python과 JavaScript/TypeScript의 주요 Unit Test 프레임워크를 정리하고 실제 사용법을 익힌다

---

## 왜 필요한가? (Why)

테스트를 직접 작성하려면 테스트를 **실행**하고 **결과를 보고**하는 도구가 필요하다.
프레임워크마다 작성 방식, 기능, 생태계가 다르므로 상황에 맞는 선택이 중요하다.

---

## Python 테스트 프레임워크

### 한눈에 비교

| 프레임워크 | 특징 | 추천 상황 |
|-----------|------|----------|
| **pytest** | 가장 많이 쓰임, 간결한 문법, 풍부한 플러그인 | 대부분의 Python 프로젝트 (사실상 표준) |
| **unittest** | Python 표준 라이브러리 내장, 별도 설치 불필요 | 의존성 최소화가 필요한 경우, 레거시 코드베이스 |

> 현재 업계 표준: **pytest**. 새 프로젝트라면 pytest를 선택하면 된다.

---

### 1. pytest (권장)

#### 설치

```bash
pip install pytest

# uv 사용 시
uv add pytest --dev
```

#### 특징

- `assert` 문 그대로 사용 (별도 메서드 불필요)
- 실패 시 assert 값을 상세히 출력해줌
- 강력한 fixture 시스템
- parametrize로 여러 케이스 한 번에 테스트
- 방대한 플러그인 생태계 (`pytest-cov`, `pytest-mock`, `pytest-asyncio` 등)

#### 파일/함수 명명 규칙

```
test_*.py 또는 *_test.py        ← pytest가 자동으로 인식하는 파일명
def test_*():                   ← 함수명이 test_로 시작해야 함
class TestSomething:            ← 클래스는 Test로 시작 (선택)
    def test_method(self):
```

#### 기본 사용법

```python
# test_basic.py

def add(a, b):
    return a + b


# ── 기본 테스트 ──
def test_add():
    assert add(2, 3) == 5


# ── 예외 테스트 ──
import pytest

def divide(a, b):
    if b == 0:
        raise ZeroDivisionError("0으로 나눌 수 없음")
    return a / b

def test_divide_by_zero():
    with pytest.raises(ZeroDivisionError):
        divide(10, 0)


# ── 여러 케이스 한 번에: parametrize ──
@pytest.mark.parametrize("a, b, expected", [
    (1, 2, 3),
    (0, 0, 0),
    (-1, 1, 0),
    (100, 200, 300),
])
def test_add_multiple_cases(a, b, expected):
    assert add(a, b) == expected
```

#### Fixture - 테스트 환경 준비/정리

```python
import pytest


@pytest.fixture
def sample_user():
    """테스트용 유저 딕셔너리를 반환"""
    return {"id": 1, "name": "Daeyoung", "email": "dy@example.com"}


@pytest.fixture
def db_connection():
    """DB 연결 준비 → 테스트 실행 → 연결 해제"""
    conn = create_test_db_connection()
    yield conn          # 여기서 테스트 함수가 실행됨
    conn.close()        # 테스트 끝나면 정리 (teardown)


def test_get_user_name(sample_user):
    assert sample_user["name"] == "Daeyoung"


def test_save_user(db_connection, sample_user):
    db_connection.save(sample_user)
    result = db_connection.find(id=1)
    assert result["name"] == "Daeyoung"
```

#### Mock - 외부 의존성 대체

```python
from unittest.mock import patch, MagicMock


# 외부 API 호출을 Mock으로 대체
def test_fetch_weather():
    with patch("requests.get") as mock_get:
        mock_get.return_value.json.return_value = {"temp": 25, "city": "Seoul"}
        mock_get.return_value.status_code = 200

        result = get_weather("Seoul")  # 내부에서 requests.get 호출

        assert result["temp"] == 25
        mock_get.assert_called_once_with("https://api.weather.com/Seoul")


# pytest-mock 플러그인 사용 시 더 깔끔하게
# pip install pytest-mock
def test_send_email(mocker):
    mock_smtp = mocker.patch("smtplib.SMTP")
    send_welcome_email("user@example.com")
    mock_smtp.return_value.send_message.assert_called_once()
```

#### 비동기 코드 테스트 (FastAPI 등)

```bash
pip install pytest-asyncio
```

```python
import pytest
import pytest_asyncio


@pytest.mark.asyncio
async def test_async_function():
    result = await fetch_data_from_api()
    assert result["status"] == "ok"


# FastAPI 엔드포인트 테스트
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_get_users():
    response = client.get("/users")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
```

#### 커버리지 측정

```bash
pip install pytest-cov

# 실행 + 커버리지
pytest --cov=src --cov-report=term-missing

# HTML 리포트 생성
pytest --cov=src --cov-report=html
# → htmlcov/index.html 브라우저로 열면 어떤 줄이 테스트 안 됐는지 한눈에 보임
```

#### 자주 쓰는 실행 옵션

```bash
pytest                          # 현재 디렉토리 전체 테스트
pytest -v                       # verbose: 각 테스트 이름 출력
pytest -s                       # print() 출력 보이게
pytest -x                       # 첫 번째 실패 시 즉시 중단
pytest -k "login"               # "login"이 포함된 테스트만 실행
pytest test_user.py::test_login # 특정 함수만 실행
pytest --lf                     # 마지막에 실패한 테스트만 재실행
```

---

### 2. unittest (표준 라이브러리)

Python에 기본 내장되어 있어 **별도 설치가 필요 없다.**
Java의 JUnit을 참고해 만들어진 xUnit 스타일.

#### 기본 사용법

```python
# test_calculator.py
import unittest


def add(a, b):
    return a + b


class TestCalculator(unittest.TestCase):

    def setUp(self):
        """각 테스트 전 실행 (pytest의 fixture와 유사)"""
        self.default_value = 10

    def tearDown(self):
        """각 테스트 후 실행 (정리 작업)"""
        pass

    def test_add_positive(self):
        self.assertEqual(add(2, 3), 5)

    def test_add_negative(self):
        self.assertEqual(add(-1, -2), -3)

    def test_add_zero(self):
        self.assertEqual(add(0, self.default_value), 10)

    def test_raises_on_invalid_input(self):
        with self.assertRaises(TypeError):
            add("string", 1)


if __name__ == "__main__":
    unittest.main()
```

#### pytest vs unittest 문법 비교

```python
# ── pytest (간결) ──
def test_add():
    assert add(2, 3) == 5

def test_raises():
    with pytest.raises(ValueError):
        risky_function()

# ── unittest (장황하지만 명시적) ──
def test_add(self):
    self.assertEqual(add(2, 3), 5)

def test_raises(self):
    with self.assertRaises(ValueError):
        risky_function()
```

주요 assert 메서드:

| unittest | pytest 동등 표현 | 의미 |
|----------|-----------------|------|
| `assertEqual(a, b)` | `assert a == b` | 같음 |
| `assertNotEqual(a, b)` | `assert a != b` | 다름 |
| `assertTrue(x)` | `assert x` | 참 |
| `assertFalse(x)` | `assert not x` | 거짓 |
| `assertIsNone(x)` | `assert x is None` | None |
| `assertIn(a, b)` | `assert a in b` | 포함 |
| `assertRaises(Exc)` | `with pytest.raises(Exc):` | 예외 발생 |

> **참고:** pytest는 unittest로 작성된 테스트도 실행 가능하다.

---

## JavaScript / TypeScript 테스트 프레임워크

### 한눈에 비교

| 프레임워크 | 특징 | 추천 상황 |
|-----------|------|----------|
| **Vitest** | Vite 기반, 초고속, Jest와 API 호환 | Vue/Vite/Nuxt 프로젝트 (최신 표준) |
| **Jest** | 오랜 업계 표준, 풍부한 생태계, React 기본 포함 | React, CRA, Next.js, 레거시 프로젝트 |
| **Mocha + Chai** | 유연하고 가볍지만 설정 필요 | Node.js 백엔드, 커스터마이징이 필요한 경우 |

---

### 1. Vitest (Vue/Vite 프로젝트 권장)

#### 설치

```bash
npm install -D vitest
```

`vite.config.ts`:
```typescript
import { defineConfig } from 'vite'

export default defineConfig({
  test: {
    globals: true,           // describe, it, expect를 import 없이 사용
    environment: 'jsdom',    // 브라우저 환경 시뮬레이션 (DOM 조작 테스트 시)
  },
})
```

#### 기본 사용법

```typescript
// src/utils/math.ts
export const add = (a: number, b: number) => a + b
export const multiply = (a: number, b: number) => a * b
export const clamp = (val: number, min: number, max: number) =>
  Math.min(Math.max(val, min), max)
```

```typescript
// src/utils/math.test.ts
import { describe, it, expect, test } from 'vitest'
import { add, multiply, clamp } from './math'


// describe: 관련 테스트를 그룹화
describe('add 함수', () => {
  it('두 양수를 더한다', () => {
    expect(add(2, 3)).toBe(5)
  })

  it('음수를 더한다', () => {
    expect(add(-1, -2)).toBe(-3)
  })

  it('0을 더한다', () => {
    expect(add(0, 5)).toBe(5)
  })
})


describe('clamp 함수', () => {
  it('범위 내 값은 그대로 반환', () => {
    expect(clamp(5, 0, 10)).toBe(5)
  })

  it('최솟값보다 작으면 min 반환', () => {
    expect(clamp(-5, 0, 10)).toBe(0)
  })

  it('최댓값보다 크면 max 반환', () => {
    expect(clamp(15, 0, 10)).toBe(10)
  })
})
```

#### expect 매처(Matcher) 정리

```typescript
// 동등 비교
expect(result).toBe(5)                  // 값과 타입이 정확히 같음 (===)
expect(result).toEqual({ a: 1 })        // 객체/배열 깊은 비교

// 진리값
expect(result).toBeTruthy()             // truthy한 값
expect(result).toBeFalsy()              // falsy한 값
expect(result).toBeNull()               // null
expect(result).toBeUndefined()          // undefined

// 숫자
expect(result).toBeGreaterThan(3)       // > 3
expect(result).toBeLessThanOrEqual(10)  // <= 10
expect(result).toBeCloseTo(0.1 + 0.2)  // 부동소수점 비교

// 문자열
expect(str).toContain('hello')          // 부분 문자열 포함
expect(str).toMatch(/pattern/)          // 정규식 매칭

// 배열
expect(arr).toHaveLength(3)             // 길이
expect(arr).toContain('item')           // 포함 여부

// 예외
expect(() => riskyFn()).toThrow()
expect(() => riskyFn()).toThrow('에러 메시지')

// 부정
expect(result).not.toBe(0)
expect(arr).not.toContain('wrong')
```

#### Mock (vi 객체 사용)

```typescript
import { describe, it, expect, vi, beforeEach } from 'vitest'


describe('API 호출 테스트', () => {
  beforeEach(() => {
    vi.clearAllMocks()  // 각 테스트 전에 mock 초기화
  })

  it('API 호출 성공 시 데이터를 반환한다', async () => {
    // fetch를 mock으로 대체
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ id: 1, name: 'Daeyoung' }),
    })

    const user = await fetchUser(1)

    expect(user.name).toBe('Daeyoung')
    expect(fetch).toHaveBeenCalledWith('/api/users/1')
    expect(fetch).toHaveBeenCalledTimes(1)
  })

  it('API 실패 시 에러를 던진다', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 404,
    })

    await expect(fetchUser(999)).rejects.toThrow('User not found')
  })
})
```

#### setup/teardown hooks

```typescript
import { beforeAll, afterAll, beforeEach, afterEach } from 'vitest'

beforeAll(() => {
  // 이 describe 블록의 모든 테스트 시작 전 1번만 실행
  setupTestDatabase()
})

afterAll(() => {
  // 모든 테스트 끝나고 1번만 실행
  cleanupTestDatabase()
})

beforeEach(() => {
  // 각 테스트 전에 실행
  resetToInitialState()
})

afterEach(() => {
  // 각 테스트 후에 실행
  vi.clearAllMocks()
})
```

#### 실행 명령어

```bash
npx vitest          # watch 모드 (파일 변경 감지하여 자동 재실행)
npx vitest run      # 1회 실행 후 종료 (CI 환경)
npx vitest --ui     # 브라우저 UI로 결과 확인
npx vitest run --coverage  # 커버리지 포함
```

---

### 2. Jest (React / 레거시 프로젝트)

#### 설치

```bash
npm install -D jest @types/jest ts-jest

# TypeScript용 설정
npx ts-jest config:init
```

`jest.config.ts`:
```typescript
export default {
  preset: 'ts-jest',
  testEnvironment: 'node',   // 또는 'jsdom'
}
```

#### 기본 사용법

Jest의 API는 Vitest와 거의 동일하다 (Vitest가 Jest API를 호환하도록 설계됨):

```typescript
// math.test.ts
import { add, multiply } from './math'

describe('add', () => {
  test('1 + 2 = 3', () => {
    expect(add(1, 2)).toBe(3)
  })
})
```

#### Mock (jest 객체 사용)

```typescript
// Vitest의 vi 대신 jest 객체 사용
jest.fn()           // vi.fn()
jest.spyOn()        // vi.spyOn()
jest.mock('./module')  // vi.mock('./module')
jest.clearAllMocks() // vi.clearAllMocks()
```

#### Vitest vs Jest 선택 기준

```
Vite 기반 프로젝트 (Vue, Nuxt, Svelte 등) → Vitest
React + Vite                               → Vitest
React + CRA / Next.js                      → Jest
기존 Jest 코드베이스 유지보수               → Jest 유지
```

> **팁:** Vitest는 Jest API를 거의 동일하게 지원하므로,
> 기존 Jest 테스트를 `jest` → `vi`로 바꾸는 것만으로 대부분 마이그레이션 가능.

---

### 3. Mocha + Chai (Node.js 백엔드)

더 유연하지만 설정이 필요하다. Node.js 백엔드에서 종종 사용된다.

#### 설치

```bash
npm install -D mocha chai @types/mocha @types/chai
```

#### 사용법

```typescript
// test/math.test.ts
import { expect } from 'chai'
import { add } from '../src/math'


describe('add 함수', () => {
  it('두 수를 더한다', () => {
    expect(add(2, 3)).to.equal(5)
  })

  it('음수를 처리한다', () => {
    expect(add(-1, 1)).to.equal(0)
  })
})
```

Chai의 자연어 스타일 체이닝:
```typescript
expect(5).to.equal(5)
expect([1,2,3]).to.include(2)
expect('hello').to.have.lengthOf(5)
expect(null).to.be.null
expect(true).to.be.true
expect({a: 1}).to.deep.equal({a: 1})
```

---

## 프레임워크 선택 가이드

```
Python 프로젝트
├── 새 프로젝트              → pytest
├── 의존성 없이 내장만       → unittest
└── FastAPI/Django           → pytest + pytest-asyncio (FastAPI)
                               또는 pytest + django-pytest (Django)

JavaScript/TypeScript 프로젝트
├── Vue + Vite/Nuxt          → Vitest
├── React + Vite             → Vitest
├── React + Next.js          → Jest (또는 Vitest)
├── Node.js 백엔드           → Vitest 또는 Mocha+Chai
└── 기존 Jest 코드베이스     → Jest 유지
```

---

## 참고 자료 (References)

- [pytest 공식 문서](https://docs.pytest.org/)
- [unittest 공식 문서](https://docs.python.org/3/library/unittest.html)
- [Vitest 공식 문서](https://vitest.dev/)
- [Jest 공식 문서](https://jestjs.io/)
- [Mocha 공식 문서](https://mochajs.org/)
- [Chai Assertion 문서](https://www.chaijs.com/api/bdd/)

## 관련 문서
- [Unit Testing 기초](./unit-testing-basics.md)
- [E2E Testing 기초](./e2e-testing-basics.md)
- [Vite 기초](../typescript/vite-basics.md)
