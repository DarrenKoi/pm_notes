---
tags: [vite, bundler, typescript, frontend, build-tool]
level: beginner
last_updated: 2026-02-01
status: in-progress
---

# Vite 기초 가이드

> 차세대 프론트엔드 빌드 도구 — ESM 네이티브 개발 서버와 최적화된 프로덕션 빌드

## 왜 필요한가? (Why)

- **Webpack의 한계**: 프로젝트가 커질수록 개발 서버 시작과 HMR이 느려짐 (모든 파일을 번들링 후 서빙)
- **ESM 네이티브**: Vite는 브라우저의 ES Modules를 직접 활용 → 번들링 없이 즉시 서빙
- **HMR 속도**: 파일 수에 관계없이 거의 일정한 HMR 속도 (변경된 모듈만 교체)
- **Vue/React/Svelte 공식 지원**: 프레임워크 팀이 Vite를 공식 빌드 도구로 채택

### Webpack vs Vite 비교

| 항목 | Webpack | Vite |
|------|---------|------|
| Dev Server 시작 | 전체 번들링 후 서빙 (느림) | ESM 직접 서빙 (즉시) |
| HMR 속도 | 프로젝트 크기에 비례 | 거의 일정 |
| 설정 복잡도 | 높음 (loader, plugin 체인) | 낮음 (합리적 기본값) |
| 프로덕션 빌드 | Webpack 자체 | Rollup 기반 |
| 생태계 성숙도 | 매우 넓음 | 빠르게 확대 중 |

## 핵심 개념 (What)

### 개발 서버 vs 프로덕션 빌드

```
[개발 모드]
브라우저 → HTTP 요청 → Vite Dev Server → esbuild로 변환 → ESM 직접 반환
                                          (TypeScript, JSX 등을 JS로)

[프로덕션 빌드]
소스 코드 → Rollup → 최적화된 정적 파일 (tree-shaking, code-splitting, minify)
```

- **Dev Server**: esbuild로 파일을 개별 변환하여 브라우저에 ESM으로 서빙. 번들링하지 않음.
- **Build**: Rollup을 사용하여 최적화된 번들 생성. tree-shaking, code-splitting 자동 적용.
- **HMR (Hot Module Replacement)**: 파일 변경 시 해당 모듈만 교체. 전체 페이지 새로고침 불필요.

### Pre-bundling (사전 번들링)

Vite는 `node_modules`의 의존성을 esbuild로 사전 번들링한다:

- CommonJS → ESM 변환
- 수백 개의 내부 모듈을 하나로 합침 (예: `lodash-es`의 600+ 모듈)
- `.vite` 폴더에 캐시 → 두 번째 시작부터 즉시 로드

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
import { resolve } from 'path';

export default defineConfig({
  // 플러그인
  plugins: [vue()],

  // 경로 별칭
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
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
    rollupOptions: {
      output: {
        // 벤더 청크 분리
        manualChunks: {
          vendor: ['vue', 'vue-router', 'pinia'],
        },
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
    "baseUrl": ".",
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
// 사용법 — 반드시 VITE_ 접두사 필요
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

> **보안**: `VITE_` 접두사가 없는 변수는 클라이언트에 노출되지 않는다. 비밀키는 절대 `VITE_` 접두사를 사용하지 말 것.

### 자주 쓰는 플러그인

```bash
# Vue
npm install -D @vitejs/plugin-vue

# React (SWC 기반 — 더 빠름)
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
- [패키지 관리와 빌드](./package-and-build.md)
- [Vue 3 기초](./vue/vue3-basics.md)
