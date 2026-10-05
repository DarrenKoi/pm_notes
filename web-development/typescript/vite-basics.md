---
tags: [vite, bundler, typescript, frontend, build-tool]
level: beginner
last_updated: 2026-02-01
reviewed_on: 2026-10-04
review_status: partial
document_type: learning_note
category_major: "웹 개발"
category_middle: "프론트엔드·런타임"
category_minor: "TypeScript·빌드 도구"
note_kind: "학습"
classified_on: "2026-10-05"
---

# Vite 기초 가이드

> 차세대 프론트엔드 빌드 도구 — ESM 네이티브 개발 서버와 최적화된 프로덕션 빌드

## 왜 필요한가? (Why)

- **개발 방식의 차이**: 번들 중심 서버와 ESM 모듈 제공 서버의 시작·변경 비용을 비교한다. 실제 속도는 설정·플러그인·모듈 그래프에 따라 다르다.
- **ESM 네이티브**: Vite는 브라우저의 ES Modules를 직접 활용 → 필요한 소스를 변환하여 서빙; 의존성 사전 번들링은 별도
- **HMR 속도**: 변경된 모듈과 영향받는 의존 그래프를 갱신; 일정한 지연을 보증하지 않음
- **Vue/React/Svelte 공식 지원**: 프레임워크 팀이 Vite를 공식 빌드 도구로 채택

### Webpack vs Vite 비교

| 항목 | Webpack | Vite |
|------|---------|------|
| Dev Server 시작 | 번들·캐시·설정에 따라 다름 | 소스 ESM 제공 + 의존성 최적화 |
| HMR 속도 | 모듈 그래프·설정에 따라 다름 | 영향받는 모듈·플러그인에 따라 다름 |
| 설정 복잡도 | 높음 (loader, plugin 체인) | 낮음 (합리적 기본값) |
| 프로덕션 빌드 | Webpack 자체 | Vite 8: Rolldown (7 이하는 Rollup) |
| 생태계 성숙도 | 매우 넓음 | 빠르게 확대 중 |

## 핵심 개념 (What)

### 개발 서버 vs 프로덕션 빌드

```
[개발 모드]
브라우저 → HTTP 요청 → Vite Dev Server → Vite 8 Oxc로 변환 → ESM 직접 반환
                                          (TypeScript, JSX 등을 JS로)

[프로덕션 빌드]
소스 코드 → Vite 8 Rolldown → 최적화된 정적 파일 (tree-shaking, code-splitting, minify)
```

- **Dev Server**: Vite 8은 Oxc로 소스를 변환하여 ESM으로 제공하고 의존성을 별도로 최적화한다.
- **Build**: Vite 8은 Rolldown으로 번들을 생성한다. tree-shaking, code-splitting 자동 적용.
- **HMR (Hot Module Replacement)**: 파일 변경 시 해당 모듈만 교체. 전체 페이지 새로고침 불필요.

### Pre-bundling (사전 번들링)

Vite 8은 Rolldown으로 의존성을 사전 번들링한다. 기존 esbuild 설명은 Vite 7 이하에 해당한다:

- CommonJS → ESM 변환
- 수백 개의 내부 모듈을 하나로 합침 (예: `lodash-es`의 600+ 모듈)
- `.vite` 폴더에 캐시 → lock·설정·소스 등의 변경에 따라 무효화될 수 있음

## 어떻게 사용하는가? (How)

### 프로젝트 생성

```bash
# 대화형 프로젝트 생성
npm create vite@latest

# 템플릿 직접 지정
npm create vite@latest my-app -- --template vue-ts
npm create vite@latest my-app -- --template react-ts
npm create vite@latest my-app -- --template vanilla-ts
```

주요 템플릿:

| 템플릿 | 설명 |
|--------|------|
| `vanilla-ts` | 프레임워크 없이 순수 TypeScript |
| `vue-ts` | Vue 3 + TypeScript |
| `react-ts` | React + TypeScript |
| `svelte-ts` | Svelte + TypeScript |

### 기본 CLI 명령어

```bash
# 개발 서버 시작 (기본 http://localhost:5173)
npx vite

# 프로덕션 빌드
npx vite build

# 빌드 결과물 미리보기 (로컬 정적 서버)
npx vite preview
```

`package.json` scripts:

```jsonc
{
  "scripts": {
    "dev": "vite",
    "build": "vue-tsc --noEmit && vite build",
    "preview": "vite preview"
  }
}
```

> **`vue-tsc --noEmit`**: Vite는 타입 체크를 하지 않으므로 빌드 전에 별도로 실행한다.

### vite.config.ts 주요 설정

```typescript
// vite.config.ts
import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import { fileURLToPath, URL } from 'node:url';

export default defineConfig({
  // 플러그인
  plugins: [vue()],

  // 경로 별칭
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },

  // 개발 서버
  server: {
    port: 3000,
    open: true, // 브라우저 자동 열기
    proxy: {
      // /api 요청을 백엔드로 프록시
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },

  // 프로덕션 빌드
  build: {
    outDir: 'dist',
    sourcemap: true,
    rolldownOptions: {
      output: {
        // 벤더 청크 분리
        // Vite 8은 객체 manualChunks를 지원하지 않는다.
        // 커스텀 청크는 Rolldown codeSplitting 정책을 별도 검증한다.
      },
    },
  },
});
```

### TypeScript 연동

Vite에서 TypeScript는 **변환만** 하고 **타입 체크는 하지 않는다** (속도를 위해).

```jsonc
// tsconfig.json (Vite용)
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "Bundler",
    "jsx": "preserve",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "paths": {
      "@/*": ["./src/*"]
    }
  },
  "include": ["src/**/*.ts", "src/**/*.tsx", "src/**/*.vue"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

```jsonc
// tsconfig.node.json (Vite 설정 파일용)
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "Bundler",
    "allowImportingTsExtensions": true,
    "noEmit": true,
    "strict": true,
    "skipLibCheck": true
  },
  "include": ["vite.config.ts"]
}
```

> **주의**: Vite 프로젝트는 `module: "ESNext"` + `moduleResolution: "Bundler"`를 사용한다. Node.js 백엔드의 `NodeNext`와 다르다. 자세한 비교는 [tsconfig 설정 가이드](./tsconfig-setup.md)를 참고.

### 환경 변수

Vite는 `.env` 파일에서 환경 변수를 로드한다:

```bash
# .env (모든 환경)
VITE_APP_TITLE=My App

# .env.development (개발 환경)
VITE_API_URL=http://localhost:8000

# .env.production (프로덕션)
VITE_API_URL=https://api.example.com
```

```typescript
// 기본 envPrefix는 VITE_; 내장 변수와 prefix 커스텀 설정은 별도
console.log(import.meta.env.VITE_APP_TITLE);
console.log(import.meta.env.VITE_API_URL);

// 내장 변수
console.log(import.meta.env.MODE);      // 'development' | 'production'
console.log(import.meta.env.DEV);       // true | false
console.log(import.meta.env.PROD);      // true | false
console.log(import.meta.env.BASE_URL);  // base 설정값
```

TypeScript에서 타입 지원:

```typescript
// src/env.d.ts
/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_APP_TITLE: string;
  readonly VITE_API_URL: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
```

> **보안**: 기본 envPrefix에서는 사용자 변수를 VITE_로 선택한다. envPrefix/define/클라이언트 코드의 다른 노출 경로도 검토한다. 비밀키는 절대 `VITE_` 접두사를 사용하지 말 것.

### 자주 쓰는 플러그인

```bash
# Vue
npm install -D @vitejs/plugin-vue

# React (SWC 기반; 상대 속도는 실제 프로젝트에서 측정)
npm install -D @vitejs/plugin-react-swc

# 레거시 브라우저 지원
npm install -D @vitejs/plugin-legacy
```

```typescript
// vite.config.ts — 플러그인 사용 예
import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import legacy from '@vitejs/plugin-legacy';

export default defineConfig({
  plugins: [
    vue(),
    legacy({
      targets: ['defaults', 'not IE 11'],
    }),
  ],
});
```

| 플러그인 | 패키지 | 용도 |
|----------|--------|------|
| Vue | `@vitejs/plugin-vue` | `.vue` SFC 지원 |
| React (SWC) | `@vitejs/plugin-react-swc` | JSX/TSX 변환 (SWC 기반) |
| Legacy | `@vitejs/plugin-legacy` | 구형 브라우저 폴리필 |

### Vue 프로젝트 빠른 시작

```bash
npm create vite@latest my-vue-app -- --template vue-ts
cd my-vue-app
npm install
npm run dev
```

생성되는 구조:

```
my-vue-app/
├── public/
│   └── vite.svg
├── src/
│   ├── assets/
│   ├── components/
│   │   └── HelloWorld.vue
│   ├── App.vue
│   ├── main.ts
│   ├── style.css
│   └── vite-env.d.ts
├── index.html              # 진입점 (Vite는 HTML을 엔트리로 사용)
├── package.json
├── tsconfig.json
├── tsconfig.node.json
└── vite.config.ts
```

> **Webpack과의 차이**: Webpack은 JS를 엔트리로 사용하지만, Vite는 `index.html`을 엔트리로 사용한다. HTML에서 `<script type="module" src="/src/main.ts">`로 직접 참조.

## 참고 자료 (References)

- [Vite 공식 가이드](https://vite.dev/guide/)
- [Vite 설정 레퍼런스](https://vite.dev/config/)
- [Vite 플러그인 목록](https://vite.dev/plugins/)
- [Rollup 공식 문서](https://rollupjs.org/)

## 관련 문서

- [tsconfig와 프로젝트 설정](./tsconfig-setup.md)
- 패키지 관리 심화 문서는 계획·미작성이다.
- [Vue 3 + TypeScript](./vue/vue3-with-typescript.md)

## 현재 적용 조건과 근거

확인일 **2026-10-04**, 본문의 도구 설명은 **Vite 8** 기준이다. [시작 가이드](https://vite.dev/guide/)의 Node 조건은 20.19+ 또는 22.12+이며 일부 템플릿은 더 높다. [7→8 변경](https://vite.dev/guide/migration)은 Oxc/Rolldown 및 객체 manualChunks 제거·rollupOptions 이름 변경을 안내한다. 기존 vendor 청크 의도는 주석으로 보존했으며 동일한 산출물 분할을 검증한 것은 아니다. [의존성 최적화](https://vite.dev/guide/dep-pre-bundling)와 [환경 변수](https://vite.dev/guide/env-and-mode)를 대조했다. 새 프로젝트의 파일 트리는 템플릿 버전에 따라 다르고 preview는 운영 서버 대체가 아니다. 생성·설치·실제 build·typecheck는 미실행이다. 타입 선언은 환경 변수 존재나 값의 런타임 유효성을 보증하지 않는다.

### 이전 버전의 고유 청크 예제 보존

다음은 원 문서의 **Vite 7 이하 Rollup** 청크 정책 조각이다. Vite 8 설정과 합치지 않는다. Vue·Router·Pinia를 vendor로 묶으려던 의도를 보존하며 설치 의존성과 실제 chunk 영향을 확인한다.

```typescript
const legacyBuildExample = {
  rollupOptions: {
    output: {
      manualChunks: { vendor: ['vue', 'vue-router', 'pinia'] },
    },
  },
}
```
