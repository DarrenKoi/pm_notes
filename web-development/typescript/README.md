---
tags: [typescript, web-development, study-roadmap]
level: beginner
last_updated: 2026-01-31
status: in-progress
---

# TypeScript 웹 개발 학습 로드맵

> TypeScript를 활용한 웹 개발 역량을 체계적으로 쌓기 위한 학습 카테고리와 순서

## 왜 필요한가? (Why)

- JavaScript의 동적 타입으로 인한 런타임 에러를 **컴파일 타임에 방지**
- 대규모 프로젝트에서 **코드 자동 완성, 리팩토링, 문서화**가 크게 개선됨
- 프론트엔드(Vue, React)와 백엔드(Node.js, Deno)를 **하나의 언어**로 통합
- 현재 웹 생태계의 사실상 표준 — 대부분의 라이브러리가 TypeScript 지원

---

## 학습 카테고리

### Phase 1: TypeScript 기초

| 주제 | 내용 | 상태 |
|------|------|------|
| [TypeScript 기본 문법](./typescript-basics.md) | 타입 시스템, 인터페이스, 제네릭, 유니온/인터섹션 타입 | 🔴 시작 전 |
| [tsconfig와 프로젝트 설정](./tsconfig-setup.md) | tsconfig.json 옵션, strict 모드, 모듈 시스템 | 🟡 진행중 |
| [타입 심화](./advanced-types.md) | 유틸리티 타입, 조건부 타입, 템플릿 리터럴 타입, type guard | 🔴 시작 전 |

**학습 목표**: TypeScript 코드를 읽고 쓸 수 있으며, 타입 에러를 스스로 해결할 수 있다.

---

### Phase 2: 웹 프론트엔드 기초

| 주제 | 내용 | 상태 |
|------|------|------|
| [HTML/CSS 핵심](./html-css-essentials.md) | 시맨틱 HTML, Flexbox, Grid, 반응형 디자인 | 🔴 시작 전 |
| [DOM과 이벤트](./dom-and-events.md) | DOM 조작, 이벤트 핸들링, 비동기 처리(Promise, async/await) | 🔴 시작 전 |
| [모던 CSS](./modern-css.md) | CSS 변수, Tailwind CSS, CSS-in-JS 개요 | 🔴 시작 전 |

**학습 목표**: 프레임워크 없이 기본적인 웹 페이지를 구성할 수 있다.

---

### Phase 3: 프론트엔드 프레임워크 — Vue.js

> CLAUDE.md에 Vue/Nuxt가 주요 관심 분야로 명시되어 있으므로 Vue를 우선합니다.

| 주제 | 내용 | 상태 |
|------|------|------|
| [Vue 3 기초](../typescript/vue/vue3-basics.md) | Composition API, ref/reactive, 컴포넌트 구조 | 🔴 시작 전 |
| [Vue 상태 관리](../typescript/vue/state-management.md) | Pinia, props/emit, provide/inject | 🔴 시작 전 |
| [Vue Router](../typescript/vue/vue-router.md) | SPA 라우팅, 네비게이션 가드, 동적 라우트 | 🔴 시작 전 |
| [Vue + TypeScript 패턴](../typescript/vue/vue-typescript.md) | defineComponent, type-safe props/emit, composables 타이핑 | 🔴 시작 전 |

**학습 목표**: Vue 3 + TypeScript로 SPA를 만들 수 있다.

---

### Phase 4: 풀스택 프레임워크 — Nuxt.js

| 주제 | 내용 | 상태 |
|------|------|------|
| [Nuxt 3 기초](../typescript/nuxt/nuxt3-basics.md) | 파일 기반 라우팅, auto-import, SSR/SSG 개념 | 🔴 시작 전 |
| [Nuxt 데이터 패칭](../typescript/nuxt/data-fetching.md) | useFetch, useAsyncData, API 라우트 | 🔴 시작 전 |
| [Nuxt 배포](../typescript/nuxt/deployment.md) | Vercel/Netlify/Docker 배포, Nitro 서버 엔진 | 🔴 시작 전 |

**학습 목표**: Nuxt 3로 SSR/SSG 웹 애플리케이션을 구축하고 배포할 수 있다.

---

### Phase 5: 개발 도구 및 생태계

| 주제 | 내용 | 상태 |
|------|------|------|
| [Vite 기초](./vite-basics.md) | Vite 개념, 설정, TypeScript 연동, 환경 변수 | 🟡 진행중 |
| [Bun 시작하기](./bun-basics.md) | Bun 런타임, 패키지 매니저, 테스트 러너, npm 전환 전략 | 🔴 시작 전 |
| [패키지 관리와 빌드](./package-and-build.md) | npm/pnpm, Vite, ESBuild, 번들링 개념 | 🔴 시작 전 |
| [코드 품질 도구](./code-quality.md) | ESLint, Prettier, Husky, lint-staged | 🔴 시작 전 |
| [테스트](./testing.md) | Vitest (유닛), Playwright (E2E), 테스트 전략 | 🔴 시작 전 |

**학습 목표**: 프로덕션 수준의 개발 환경을 구성할 수 있다.

---

### Phase 6: 백엔드 연동 및 풀스택

| 주제 | 내용 | 상태 |
|------|------|------|
| [REST API 연동](./api-integration.md) | fetch/axios, 에러 핸들링, 타입 안전한 API 클라이언트 | 🔴 시작 전 |
| [인증/인가](./authentication.md) | JWT, OAuth, 세션, 쿠키 기반 인증 | 🔴 시작 전 |
| [Flask + Vue 풀스택](./fullstack-flask-vue.md) | Flask 백엔드 + Vue 프론트엔드 통합 패턴 | 🔴 시작 전 |

**학습 목표**: Flask 백엔드와 Vue 프론트엔드를 연결하여 풀스택 애플리케이션을 만들 수 있다.

---

## 추천 학습 순서

```
Phase 1: TypeScript 기초 (필수, 먼저)
    ↓
Phase 2: 웹 프론트엔드 기초 (HTML/CSS 경험 있으면 빠르게)
    ↓
Phase 3: Vue.js (프론트엔드 프레임워크)
    ↓
Phase 4: Nuxt.js (풀스택 프레임워크)
    ↓
Phase 5 & 6: 병행 가능 (도구 + 백엔드 연동)
```

> Phase 1은 나머지 모든 단계의 기반이므로 반드시 먼저 학습한다. Phase 2는 웹 개발 경험이 있다면 빠르게 넘어갈 수 있다.

## 참고 자료

- [TypeScript 공식 핸드북](https://www.typescriptlang.org/docs/handbook/)
- [Vue 3 공식 문서](https://vuejs.org/guide/introduction.html)
- [Nuxt 3 공식 문서](https://nuxt.com/docs)
- [Vite 공식 문서](https://vite.dev/guide/)

## 관련 문서

- [FastAPI 관련 문서](../python/fastapi/) (백엔드 연동 시 활용)
- [uv 패키지 매니저](../python/uv-package-manager.md) (Python 백엔드 환경 관리)
