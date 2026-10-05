---
tags: [typescript, tsconfig, eslint, prettier, project-setup]
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

# TypeScript 프로젝트 설정 가이드

> TypeScript 프로젝트를 처음부터 설정하는 종합 가이드 — tsconfig, ESLint, Prettier, VS Code까지

## 왜 필요한가? (Why)

- 프로젝트마다 설정을 처음부터 하면 시간 낭비 → **재사용 가능한 표준 설정**이 필요
- `strict` 모드 없이 시작하면 나중에 타입 에러 폭발 → **처음부터 strict로 시작**
- ESLint + Prettier 충돌은 흔한 문제 → **올바른 조합 설정**이 중요
- Windows/Mac/Linux 혼합 환경에서 줄바꿈(CRLF/LF) 문제 → **크로스 플랫폼 설정** 필요

## 핵심 개념 (What)

### tsconfig.json 주요 옵션

| 옵션 | 설명 | 추천값 |
|------|------|--------|
| `target` | 컴파일 결과 JS 버전 | `ES2022` |
| `module` | 모듈 시스템 | `NodeNext` |
| `moduleResolution` | 모듈 해석 방식 | `NodeNext` |
| `strict` | 엄격한 타입 검사 전체 활성화 | `true` |
| `esModuleInterop` | CommonJS ↔ ESM 호환 | `true` |
| `skipLibCheck` | `.d.ts` 파일 타입 검사 건너뛰기 | `true` |
| `forceConsistentCasingInFileNames` | 파일명 대소문자 일관성 강제 | `true` |
| `resolveJsonModule` | JSON import 허용 | `true` |
| `declaration` | `.d.ts` 파일 생성 | `true` (라이브러리일 때) |
| `outDir` | 컴파일 결과물 출력 폴더 | `./dist` |
| `rootDir` | 소스 파일 루트 폴더 | `./src` |
| `paths` | 경로 별칭(alias) 설정 | `{"@/*": ["./src/*"]}` |
| `baseUrl` | TypeScript 6.0 deprecated | 새 예제에서 생략; paths는 tsconfig 상대 경로 |

### 모듈 시스템 선택 가이드

| 상황 | module | moduleResolution | 이유 |
|------|--------|------------------|------|
| Node.js 백엔드 | `NodeNext` | `NodeNext` | Node.js의 ESM/CJS 듀얼 지원 |
| Vite/번들러 프론트엔드 | `ESNext` | `Bundler` | 번들러가 모듈 해석 담당 |
| Node 대상 라이브러리 | `NodeNext` | `NodeNext` | 실제 지원 Node·exports·소비자 환경에 맞춰 검증 |

> **핵심**: `NodeNext`는 `package.json`의 `"type": "module"` 여부에 따라 ESM/CJS를 자동 결정한다.

## 어떻게 사용하는가? (How)

### 1단계: 프로젝트 초기화

```bash
mkdir my-project && cd my-project
npm init -y

# TypeScript 및 실행 도구 설치
npm install -D typescript tsx @types/node
```

- **`typescript`**: TypeScript 컴파일러 (`tsc`)
- **`tsx`**: TypeScript 파일을 직접 실행 (개발 시 `ts-node` 대체)
- **`@types/node`**: Node.js API 타입 정의

### 2단계: tsconfig.json

```bash
npx tsc --init
```

생성된 파일을 아래와 같이 수정:

```jsonc
// tsconfig.json
{
  "compilerOptions": {
    // 컴파일 대상
    "target": "ES2022",
    "lib": ["ES2022"],

    // 모듈 시스템
    "module": "NodeNext",
    "moduleResolution": "NodeNext",

    // 엄격한 타입 검사
    "strict": true,
    "forceConsistentCasingInFileNames": true,
    "noUncheckedIndexedAccess": true,

    // 모듈 호환성
    "esModuleInterop": true,
    "resolveJsonModule": true,
    "isolatedModules": true,

    // 출력
    "outDir": "./dist",
    "rootDir": "./src",
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,

    // 경로 별칭
    "paths": {
      "@/*": ["./src/*"]
    },

    // 성능
    "skipLibCheck": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

`package.json`에 모듈 타입 지정:

```jsonc
// package.json (추가)
{
  "type": "module"
}
```

### 3단계: Prettier 설정

```bash
npm install -D prettier
```

```jsonc
// .prettierrc.json
{
  "semi": true,
  "singleQuote": true,
  "tabWidth": 2,
  "trailingComma": "all",
  "printWidth": 100,
  "endOfLine": "lf",
  "arrowParens": "always"
}
```

```text
# .prettierignore
dist
node_modules
*.md
```

### 4단계: ESLint (Flat Config)

```bash
npm install -D eslint @eslint/js typescript-eslint eslint-config-prettier globals
```

```javascript
// eslint.config.mjs
import js from '@eslint/js';
import tseslint from 'typescript-eslint';
import prettierConfig from 'eslint-config-prettier';
import globals from 'globals';

export default tseslint.config(
  // 무시할 경로
  { ignores: ['dist/', 'node_modules/'] },
  { languageOptions: { globals: globals.node } },

  // JavaScript 기본 추천 규칙
  js.configs.recommended,

  // TypeScript 추천 규칙
  ...tseslint.configs.recommended,

  // Prettier와 충돌하는 ESLint 규칙 비활성화 (항상 마지막에)
  prettierConfig,

  // 커스텀 규칙
  {
    rules: {
      '@typescript-eslint/no-unused-vars': [
        'error',
        { argsIgnorePattern: '^_', varsIgnorePattern: '^_' },
      ],
      '@typescript-eslint/consistent-type-imports': 'error',
    },
  },
);
```

> **Flat Config**는 ESLint v9+의 기본 설정 방식이다. `.eslintrc.*` 파일은 더 이상 권장되지 않는다.

### 5단계: 크로스 플랫폼 호환성

```ini
# .editorconfig
root = true

[*]
charset = utf-8
end_of_line = lf
indent_style = space
indent_size = 2
insert_final_newline = true
trim_trailing_whitespace = true
```

```text
# .gitattributes
* text=auto eol=lf
*.png binary
*.jpg binary
*.jpeg binary
*.gif binary
*.webp binary
*.ico binary
# SVG는 텍스트이므로 별도 의도가 없으면 위 text 규칙을 적용
```

### 6단계: VS Code 설정

```jsonc
// .vscode/settings.json
{
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": "explicit"
  },
  "typescript.tsdk": "node_modules/typescript/lib",
  "typescript.enablePromptUseWorkspaceTsdk": true
}
```

```jsonc
// .vscode/extensions.json
{
  "recommendations": [
    "esbenp.prettier-vscode",
    "dbaeumer.vscode-eslint",
    "editorconfig.editorconfig"
  ]
}
```

### 7단계: package.json scripts

```jsonc
// package.json (scripts 부분)
{
  "scripts": {
    "dev": "tsx watch src/index.ts",
    "build": "tsc",
    "start": "node dist/index.js",
    "type-check": "tsc --noEmit",
    "lint": "eslint .",
    "lint:fix": "eslint . --fix",
    "format": "prettier --write .",
    "format:check": "prettier --check ."
  }
}
```

### 완성된 프로젝트 구조

```
my-project/
├── src/
│   └── index.ts
├── dist/                    # (gitignore)
├── node_modules/            # (gitignore)
├── .vscode/
│   ├── settings.json
│   └── extensions.json
├── .editorconfig
├── .gitattributes
├── .gitignore
├── .prettierrc.json
├── .prettierignore
├── eslint.config.mjs
├── package.json
├── package-lock.json
└── tsconfig.json
```

### 시작 파일 예제

```typescript
// src/index.ts
const greeting = (name: string): string => {
  return `Hello, ${name}!`;
};

console.log(greeting('TypeScript'));
```

```bash
# 실행
npx tsx src/index.ts

# 개발 모드 (파일 변경 시 자동 재실행)
npm run dev

# 빌드
npm run build
```

## 참고 자료 (References)

- [TypeScript 공식 — tsconfig 레퍼런스](https://www.typescriptlang.org/tsconfig)
- [typescript-eslint 공식 문서](https://typescript-eslint.io/)
- [ESLint Flat Config 마이그레이션](https://eslint.org/docs/latest/use/configure/migration-guide)
- [Prettier 공식 옵션](https://prettier.io/docs/en/options.html)
- [EditorConfig 공식](https://editorconfig.org/)

## 관련 문서

- TypeScript 기본 문법 노트는 계획·미작성이다.
- [Vite 기초](./vite-basics.md)
- 코드 품질 심화 노트는 계획·미작성이다.

## 버전·적용 조건 검토

확인일 **2026-10-04**. [TypeScript 6.0](https://www.typescriptlang.org/docs/handbook/release-notes/typescript-6-0.html#deprecated---baseurl)은 baseUrl을 deprecated로 안내한다. [paths](https://www.typescriptlang.org/docs/handbook/modules/reference.html#paths)는 출력 import를 다시 쓰지 않는다. Node가 `@/…`를 자동 해석한다고 가정하지 말고 상대 `.js` import 또는 실제 지원하는 package imports 정책을 사용한다. 위 paths는 타입 해석 설정의 예이며 런타임 구현은 생략됐다. NodeNext의 기본 모듈·지원 기능은 컴파일러 버전에 따라 달라진다.

[typescript-eslint 공식 설정](https://typescript-eslint.io/getting-started/)과 [Git attributes](https://git-scm.com/docs/gitattributes)를 확인했다. Git 패턴은 셸의 중괄호 확장이 아니므로 확장자를 각 행에 썼다. ESLint 예제는 Node 프로젝트용 globals이며 브라우저용은 별도 설정이다. 이 폴더에는 tsc·Vue compiler 및 node_modules가 없어 컴파일·lint 실행은 미확인이다. 설치 명령은 버전을 고정하지 않으므로 실제 lock의 TypeScript/ESLint/plugin 조합을 확인한다.
