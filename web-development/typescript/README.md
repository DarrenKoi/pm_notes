---
tags: [typescript, web-development, study-roadmap]
level: beginner
last_updated: 2026-01-31
reviewed_on: 2026-10-04
review_status: partial
document_type: index
---

# TypeScript 웹 개발 학습 로드맵

> TypeScript를 활용한 웹 개발 역량을 체계적으로 쌓기 위한 학습 카테고리와 순서

## 왜 필요한가? (Why)

- 정적 타입으로 표현한 오류를 **타입 검사 시 발견**; 외부 응답·실행 환경 오류까지 방지하지는 않음
- 대규모 프로젝트에서 **코드 자동 완성, 리팩토링, 문서화**가 크게 개선됨
- 프론트엔드(Vue, React)와 백엔드(Node.js, Deno)를 **하나의 언어**로 통합
- 학습 대상 라이브러리의 타입 지원·버전과 실행 환경을 함께 확인

---

## 학습 카테고리

### Phase 1: TypeScript 기초

| 주제 | 내용 |
|------|------|
| TypeScript 기본 문법 **(계획·미작성)** | 타입 시스템, 인터페이스, 제네릭, 유니온/인터섹션 타입 |
| [tsconfig와 프로젝트 설정](./tsconfig-setup.md) | tsconfig.json 옵션, strict 모드, 모듈 시스템 |
| 타입 심화 **(계획·미작성)** | 유틸리티 타입, 조건부 타입, 템플릿 리터럴 타입, type guard |

**학습 목표**: TypeScript 코드를 읽고 쓸 수 있으며, 타입 에러를 스스로 해결할 수 있다.

---

### Phase 2: 웹 프론트엔드 기초

| 주제 | 내용 |
|------|------|
| HTML/CSS 핵심 **(계획·미작성)** | 시맨틱 HTML, Flexbox, Grid, 반응형 디자인 |
| DOM과 이벤트 **(계획·미작성)** | DOM 조작, 이벤트 핸들링, 비동기 처리(Promise, async/await) |
| 모던 CSS **(계획·미작성)** | CSS 변수, Tailwind CSS, CSS-in-JS 개요 |

**학습 목표**: 프레임워크 없이 기본적인 웹 페이지를 구성할 수 있다.

---

### Phase 3: 프론트엔드 프레임워크 — Vue.js

> 이 학습 순서는 Vue를 먼저 다루고 Nuxt로 확장한다. Nuxt 세부 노트는 아직 작성되지 않았다.

| 주제 | 내용 |
|------|------|
| Vue 3 기초 **(계획·미작성)** | Composition API, ref/reactive, 컴포넌트 구조 |
| [Vue 상태 관리](./vue/props-emit-vs-pinia.md) | Pinia, props/emit, provide/inject |
| Vue Router **(계획·미작성)** | SPA 라우팅, 네비게이션 가드, 동적 라우트 |
| [Vue + TypeScript 패턴](./vue/vue3-with-typescript.md) | defineComponent, type-safe props/emit, composables 타이핑 |

**학습 목표**: Vue 3 + TypeScript로 SPA를 만들 수 있다.

---

### Phase 4: 풀스택 프레임워크 — Nuxt.js

| 주제 | 내용 |
|------|------|
| Nuxt 3 기초 **(계획·미작성)** | 파일 기반 라우팅, auto-import, SSR/SSG 개념 |
| Nuxt 데이터 패칭 **(계획·미작성)** | useFetch, useAsyncData, API 라우트 |
| Nuxt 배포 **(계획·미작성)** | Vercel/Netlify/Docker 배포, Nitro 서버 엔진 |

**학습 목표**: Nuxt 3로 SSR/SSG 웹 애플리케이션을 구축하고 배포할 수 있다.

---

### Phase 5: 개발 도구 및 생태계

| 주제 | 내용 |
|------|------|
| [Vite 기초](./vite-basics.md) | Vite 개념, 설정, TypeScript 연동, 환경 변수 |
| [Bun 시작하기](./bun-basics.md) | Bun 런타임, 패키지 매니저, 테스트 러너, npm 전환 전략 |
| 패키지 관리와 빌드 **(계획·미작성)** | npm/pnpm, Vite, ESBuild, 번들링 개념 |
| 코드 품질 도구 **(계획·미작성)** | ESLint, Prettier, Husky, lint-staged |
| [테스트 프레임워크](../testing/testing-frameworks.md) | Vitest (유닛), Playwright (E2E), 테스트 전략 |

**학습 목표**: 프로덕션 수준의 개발 환경을 구성할 수 있다.

---

### Phase 6: 백엔드 연동 및 풀스택

| 주제 | 내용 |
|------|------|
| REST API 연동 **(계획·미작성)** | fetch/axios, 에러 핸들링, 타입 안전한 API 클라이언트 |
| 인증/인가 **(계획·미작성)** | JWT, OAuth, 세션, 쿠키 기반 인증 |
| Flask + Vue 풀스택 **(계획·미작성)** | Flask 백엔드 + Vue 프론트엔드 통합 패턴 |

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

- FastAPI 노트는 계획·미작성이다. 백엔드 연동은 Vue 타입 문서의 Flask 예제부터 읽는다.
- [uv 패키지 매니저](../python/uv-package-manager.md) (Python 백엔드 환경 관리)

## 현재 검토

확인일 **2026-10-04**. 목차는 실제 파일과 학습 계획을 구분한다. Vue·도구·테스트의 예제별 버전과 미확인은 각 문서의 검토 절을 따른다. Nuxt 3 학습 목표는 기존 로드맵의 목표이며 현재 추천 major나 앱 검증 완료를 뜻하지 않는다.
