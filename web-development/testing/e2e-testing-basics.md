---
tags: [e2e, testing, playwright, cypress, automation]
level: beginner
last_updated: 2026-02-19
status: in-progress
---

# End-to-End Testing (E2E 테스트) 기초

> 실제 사용자처럼 브라우저를 직접 조작해 전체 흐름이 올바르게 동작하는지 검증하는 테스트

---

## 왜 필요한가? (Why)

### Unit Test와의 차이

Unit Test가 "부품이 제대로 만들어졌는가"를 검증한다면,
E2E Test는 "조립된 완성품이 실제로 잘 작동하는가"를 검증한다.

```
[로그인 기능을 예로]

Unit Test:
  ✓ 이메일 형식 검증 함수가 잘 작동하는가?
  ✓ 비밀번호 해싱 함수가 올바른가?
  ✓ JWT 토큰 생성 함수가 올바른가?

E2E Test:
  ✓ 사용자가 이메일/비밀번호를 입력하고 로그인 버튼을 누르면
    실제로 대시보드로 이동하는가?
```

### 테스트 피라미드에서 E2E의 위치

```
        /\
       /E2E\        ← 느림, 비용 높음 / 하지만 가장 실제에 가까운 검증
      /------\
     /Integr- \
    / ation   \
   /------------\
  /  Unit Tests  \   ← 빠름, 비용 낮음, 가장 많이 작성
 /----------------\
```

**E2E는 비용이 높지만 이것만 잡아낼 수 있는 버그가 있다:**
- 프론트엔드-백엔드 연동이 실제로 맞는가?
- 여러 화면에 걸친 사용자 흐름(flow)이 끊기지 않는가?
- 브라우저 특성 차이로 발생하는 문제가 없는가?

### 실제로 발생하는 상황

```
Unit Test 전부 통과 ✓
Integration Test 통과 ✓
실제 브라우저에서 로그인 버튼 누르면... 아무 반응 없음 ✗

→ 원인: API 응답 형식 변경됐는데 프론트에서 처리 안 함
→ E2E 테스트가 있었다면 바로 잡힐 버그
```

---

## 핵심 개념 (What)

### E2E 테스트가 하는 일

실제 브라우저(Chromium, Firefox 등)를 코드로 자동 조작한다:

```
1. 브라우저 열기
2. URL 접속
3. 화면 요소 찾기 (버튼, 입력창 등)
4. 클릭, 입력 등 사용자 행동 수행
5. 결과 화면이 기대와 일치하는지 검증
6. 브라우저 닫기
```

### 주요 도구 비교

| 도구 | 특징 | 적합한 상황 |
|------|------|------------|
| **Playwright** | Microsoft 개발, 빠름, 다양한 브라우저 지원, Python/TS 모두 지원 | 신규 프로젝트 권장 |
| **Cypress** | JS/TS 전용, 디버깅 UI 뛰어남, 레코딩 기능 | 프론트엔드 팀 친화적 |
| **Selenium** | 오래된 표준, 모든 언어 지원 | 레거시 환경 유지보수 |

> 현재 업계 트렌드: **Playwright** 가 빠르게 표준이 되고 있음

### 핵심 개념 용어

| 용어 | 설명 |
|------|------|
| **Locator** | 화면의 특정 요소를 찾는 방법 (`page.get_by_text("로그인")`) |
| **Assertion** | 기대값 검증 (`expect(page).to_have_url("/dashboard")`) |
| **Fixture** | 테스트 전/후 공통 작업 (로그인 상태 설정 등) |
| **Headless** | 브라우저 창 없이 백그라운드에서 실행하는 모드 (CI 환경) |
| **Screenshot/Video** | 테스트 실패 시 자동 캡처 (디버깅용) |
| **Trace** | 테스트 실행 과정 전체 기록 (Playwright 특기) |

---

## 어떻게 사용하는가? (How)

### Playwright - Python

#### 설치

```bash
pip install playwright
playwright install  # 브라우저 바이너리 설치 (Chromium, Firefox, WebKit)

# pytest 연동
pip install pytest-playwright
```

#### 기본 예제: 구글 검색 자동화

```python
# test_google.py
from playwright.sync_api import Page, expect


def test_google_search(page: Page):
    # 1. 페이지 접속
    page.goto("https://www.google.com")

    # 2. 검색창에 입력
    page.get_by_name("q").fill("playwright python")

    # 3. 엔터 또는 버튼 클릭
    page.get_by_name("q").press("Enter")

    # 4. 결과 페이지 검증
    expect(page).to_have_title(re.compile("playwright python"))
```

```bash
pytest test_google.py --headed  # 브라우저 창 보면서 실행
pytest test_google.py           # headless로 실행 (빠름)
```

#### 실무 예제: 로그인 → 데이터 조회 흐름

```python
# test_login_flow.py
import pytest
from playwright.sync_api import Page, expect


def test_login_and_view_dashboard(page: Page):
    # ── 1. 로그인 페이지 접속 ──
    page.goto("http://localhost:3000/login")
    expect(page).to_have_title("로그인")

    # ── 2. 자격증명 입력 ──
    page.get_by_label("이메일").fill("daeyoung@example.com")
    page.get_by_label("비밀번호").fill("password123")
    page.get_by_role("button", name="로그인").click()

    # ── 3. 대시보드로 이동됐는지 확인 ──
    expect(page).to_have_url("http://localhost:3000/dashboard")
    expect(page.get_by_text("환영합니다, Daeyoung")).to_be_visible()

    # ── 4. 데이터 테이블이 로드됐는지 확인 ──
    table = page.get_by_role("table")
    expect(table).to_be_visible()
    expect(table.get_by_role("row")).to_have_count(10)  # 10개 행


def test_login_fails_with_wrong_password(page: Page):
    page.goto("http://localhost:3000/login")

    page.get_by_label("이메일").fill("daeyoung@example.com")
    page.get_by_label("비밀번호").fill("wrong-password")
    page.get_by_role("button", name="로그인").click()

    # 에러 메시지 표시 확인
    expect(page.get_by_text("이메일 또는 비밀번호가 올바르지 않습니다")).to_be_visible()
    # URL이 login 페이지에 그대로 있어야 함
    expect(page).to_have_url("http://localhost:3000/login")
```

#### Fixture로 로그인 상태 재사용

매 테스트마다 로그인을 반복하면 느려진다. Fixture로 공통화:

```python
# conftest.py
import pytest
from playwright.sync_api import Page


@pytest.fixture
def logged_in_page(page: Page):
    """로그인된 상태의 page를 반환하는 fixture"""
    page.goto("http://localhost:3000/login")
    page.get_by_label("이메일").fill("daeyoung@example.com")
    page.get_by_label("비밀번호").fill("password123")
    page.get_by_role("button", name="로그인").click()
    page.wait_for_url("**/dashboard")
    return page


# 다른 테스트에서 로그인 상태를 그냥 가져다 씀
def test_create_new_item(logged_in_page: Page):
    logged_in_page.get_by_role("button", name="새로 만들기").click()
    logged_in_page.get_by_label("이름").fill("새 아이템")
    logged_in_page.get_by_role("button", name="저장").click()

    expect(logged_in_page.get_by_text("새 아이템")).to_be_visible()
```

#### API 상태 저장으로 더 빠르게 (Storage State)

로그인 세션을 파일로 저장해서 재사용:

```python
# setup.py - 한 번만 실행해서 로그인 상태 저장
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    context = browser.new_context()
    page = context.new_page()

    page.goto("http://localhost:3000/login")
    page.get_by_label("이메일").fill("daeyoung@example.com")
    page.get_by_label("비밀번호").fill("password123")
    page.get_by_role("button", name="로그인").click()
    page.wait_for_url("**/dashboard")

    # 로그인 상태(쿠키, localStorage) 저장
    context.storage_state(path="auth_state.json")
    browser.close()
```

```python
# conftest.py - 저장된 상태로 브라우저 시작
@pytest.fixture
def authenticated_context(browser):
    context = browser.new_context(storage_state="auth_state.json")
    yield context
    context.close()
```

---

### Playwright - TypeScript (Vue/Nuxt 프로젝트)

#### 설치 및 초기화

```bash
npm init playwright@latest
# → playwright.config.ts 자동 생성
```

`playwright.config.ts`:

```typescript
import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './e2e',
  baseURL: 'http://localhost:5173',  // Vite dev server
  use: {
    trace: 'on-first-retry',         // 실패 시 trace 자동 저장
    screenshot: 'only-on-failure',   // 실패 시 스크린샷
  },
  webServer: {
    command: 'npm run dev',          // 테스트 전 자동으로 dev server 실행
    url: 'http://localhost:5173',
    reuseExistingServer: true,
  },
})
```

#### 기본 예제

```typescript
// e2e/login.spec.ts
import { test, expect } from '@playwright/test'

test.describe('로그인 기능', () => {
  test('올바른 자격증명으로 로그인하면 대시보드로 이동한다', async ({ page }) => {
    await page.goto('/login')

    await page.getByLabel('이메일').fill('user@example.com')
    await page.getByLabel('비밀번호').fill('password123')
    await page.getByRole('button', { name: '로그인' }).click()

    await expect(page).toHaveURL('/dashboard')
    await expect(page.getByText('환영합니다')).toBeVisible()
  })

  test('잘못된 비밀번호는 에러 메시지를 보여준다', async ({ page }) => {
    await page.goto('/login')

    await page.getByLabel('이메일').fill('user@example.com')
    await page.getByLabel('비밀번호').fill('wrong')
    await page.getByRole('button', { name: '로그인' }).click()

    await expect(page.getByRole('alert')).toContainText('비밀번호가 올바르지 않습니다')
  })
})
```

#### 실행

```bash
# 모든 E2E 테스트 실행
npx playwright test

# 브라우저 보면서 실행
npx playwright test --headed

# 특정 파일만
npx playwright test e2e/login.spec.ts

# 디버그 모드 (한 줄씩 실행)
npx playwright test --debug

# 실패한 테스트 trace 보기 (브라우저 타임라인 뷰어)
npx playwright show-trace test-results/.../trace.zip

# UI 모드 (가장 편한 디버깅 방법)
npx playwright test --ui
```

---

### 코드 생성 (Codegen) - 자동으로 테스트 코드 작성

직접 브라우저를 조작하면 Playwright가 코드를 자동 생성해준다:

```bash
# Python
playwright codegen http://localhost:3000

# TypeScript
npx playwright codegen http://localhost:3000
```

브라우저가 열리고, 내가 클릭/입력하는 모든 행동이 왼쪽 창에 코드로 실시간 작성된다.
처음 시작할 때 빠르게 테스트 뼈대를 만드는 데 매우 유용.

---

## Unit Test vs Integration Test vs E2E 정리

```
┌─────────────────┬──────────────┬──────────────────┬─────────────────────┐
│                 │  Unit Test   │ Integration Test  │     E2E Test        │
├─────────────────┼──────────────┼──────────────────┼─────────────────────┤
│ 테스트 대상      │ 함수/메서드  │ 모듈 간 연동      │ 전체 사용자 흐름     │
│ 실행 속도        │ ms          │ 초               │ 분 단위             │
│ 환경            │ 코드만       │ DB/API 일부 포함  │ 실제 브라우저 + 서버 │
│ 실패 원인 파악   │ 쉬움        │ 보통             │ 어려울 수 있음       │
│ 작성 비용        │ 낮음        │ 중간             │ 높음                │
│ 현실 반영도      │ 낮음        │ 중간             │ 가장 높음           │
└─────────────────┴──────────────┴──────────────────┴─────────────────────┘
```

**실무 권장 비율 (테스트 피라미드):**
- Unit Test: 70%
- Integration Test: 20%
- E2E Test: 10%

---

## 실무 팁

### 어떤 케이스를 E2E로 테스트해야 하는가?

E2E는 느리고 비용이 높다. 모든 것을 E2E로 하려 하면 유지보수가 힘들어진다.
**핵심 사용자 흐름(Critical User Journey)만** E2E로 검증한다:

```
E2E로 반드시 테스트해야 할 것:
✓ 로그인 / 로그아웃
✓ 핵심 구매/결제 흐름
✓ 회원가입
✓ 가장 자주 쓰는 핵심 기능 2~3가지

E2E 대신 Unit/Integration으로 하면 충분한 것:
✗ 버튼 색상 확인
✗ 모든 폼 필드 검증
✗ 에러 메시지 문구 (단순한 것)
```

### CI/CD 파이프라인에서 E2E

```yaml
# GitHub Actions 예시
- name: Run E2E Tests
  run: npx playwright test
  env:
    CI: true

# Playwright HTML 리포트 아티팩트로 저장
- uses: actions/upload-artifact@v3
  if: always()
  with:
    name: playwright-report
    path: playwright-report/
```

### 실패 디버깅 순서

```
1. 스크린샷 확인 (test-results/ 폴더)
2. Trace 뷰어로 타임라인 확인 (playwright show-trace)
3. --headed 모드로 직접 눈으로 확인
4. --debug 모드로 한 줄씩 실행
```

---

## 참고 자료 (References)

- [Playwright 공식 문서](https://playwright.dev/)
- [Playwright Python 공식 문서](https://playwright.dev/python/)
- [Playwright Codegen 가이드](https://playwright.dev/docs/codegen)
- [Testing Trophy (Kent C. Dodds)](https://kentcdodds.com/blog/the-testing-trophy-and-testing-classifications)

## 관련 문서
- [Unit Testing 기초](./unit-testing-basics.md)
- [Vue 3 with TypeScript](../typescript/vue/vue3-with-typescript.md)
- [Vite 기초](../typescript/vite-basics.md)
