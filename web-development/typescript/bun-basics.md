---
tags: [bun, typescript, runtime, package-manager, test-runner]
level: beginner
last_updated: 2026-04-04
status: in-progress
---

# Bun 시작 가이드

> 앞으로 `npm` 대신 `bun`을 쓰려는 사람을 위한 실전 입문서

## 왜 Bun을 보나? (Why)

- `npm`은 주로 패키지 매니저지만, Bun은 **런타임 + 패키지 매니저 + 테스트 러너 + 번들러**를 한 번에 제공한다.
- TypeScript 파일을 별도 실행기 없이 바로 실행할 수 있어서 `ts-node`, `tsx`, 일부 빌드 스크립트를 줄이기 쉽다.
- 설치, 시작 속도, 테스트 실행 속도가 빨라서 로컬 개발 루프를 짧게 가져가기 좋다.
- 기존 `package.json`과 npm 생태계를 상당 부분 그대로 활용할 수 있어서, 새 도구를 배우면서도 기존 자산을 버리지 않아도 된다.

## Bun이 특별한 점은 무엇인가? (What)

### 1. 올인원 툴체인

Node.js 프로젝트에서는 보통 아래처럼 도구가 나뉜다.

- 런타임: `node`
- 패키지 매니저: `npm`
- 일회성 CLI 실행: `npx`
- 테스트 러너: `jest`, `vitest`
- TypeScript 실행 보조: `tsx`, `ts-node`

Bun은 이 중 여러 역할을 한 바이너리에서 처리한다.

| 작업 | 기존 조합 | Bun |
|------|-----------|-----|
| 패키지 설치 | `npm install` | `bun install` |
| 스크립트 실행 | `npm run dev` | `bun run dev` |
| 파일 직접 실행 | `node index.js` / `tsx index.ts` | `bun index.ts` |
| 일회성 CLI 실행 | `npx prettier` | `bunx prettier` |
| 테스트 실행 | `jest`, `vitest` | `bun test` |

### 2. TypeScript-first 실행 경험

- `.ts`, `.tsx`를 바로 실행할 수 있다.
- 작은 도구, 서버, 자동화 스크립트는 "빌드 후 실행" 대신 "바로 실행" 흐름으로 단순화하기 좋다.
- 특히 백엔드 유틸리티나 사내 스크립트에서는 생산성 차이가 크게 난다.

### 3. 빠른 설치와 짧은 개발 루프

- `bun install`은 글로벌 캐시를 활용해 설치가 빠른 편이다.
- 런타임 시작도 빠른 편이라 작은 CLI, 테스트, 개발 서버 체감이 좋다.
- `--watch`와 함께 쓰면 변경 후 재실행 루프를 짧게 가져가기 쉽다.

### 4. Node.js 호환성을 활용한 점진적 전환

- 많은 npm 패키지를 그대로 설치하고 사용할 수 있다.
- 따라서 "프로젝트 전체를 한 번에 갈아엎는" 방식보다, **패키지 매니저부터 Bun으로 바꾸고 런타임은 점진적으로 옮기는 전략**이 실용적이다.

### 5. 내장 테스트 러너와 셸 기능

- `bun test`로 별도 테스트 러너 없이 빠르게 테스트를 돌릴 수 있다.
- `bunx`로 설치 없이 CLI를 실행할 수 있다.
- Bun Shell을 쓰면 크로스 플랫폼 스크립트를 JavaScript/TypeScript 안에서 작성하기 쉬워진다.

## 언제 Bun이 특히 잘 맞나? (When)

- TypeScript 기반의 작은 API 서버
- 사내 자동화 스크립트
- CLI 도구
- 빠른 프로토타이핑
- Vite, React, Vue 같은 프론트엔드 프로젝트의 패키지 관리
- 모노레포에서 빠른 설치와 워크스페이스 관리가 중요한 경우

반대로 아래는 먼저 검증하고 들어가는 편이 낫다.

- 특정 Node 전용 네이티브 모듈 의존성이 강한 프로젝트
- Jest 전용 플러그인이나 특수 러너에 깊게 묶인 테스트 환경
- CI/CD, 배포 런타임이 Node에 강하게 고정된 레거시 환경

## 가장 현실적인 도입 방식 (How)

### 1. 가장 먼저 바꿀 것: 패키지 설치

기존 프로젝트에서 가장 부담이 적은 시작점은 패키지 매니저 교체다.

```bash
# 기존
npm install

# Bun으로 전환
bun install
```

이 단계에서는 애플리케이션 런타임을 꼭 Bun으로 바꾸지 않아도 된다.
즉, **"설치는 Bun, 실행은 Node"** 조합도 가능하다.

### 2. 다음 단계: npx 습관 바꾸기

```bash
# 기존
npx tsc --noEmit
npx prettier --write .

# Bun
bunx tsc --noEmit
bunx prettier --write .
```

### 3. 그다음: TypeScript 실행을 단순화하기

```bash
# 기존
npx tsx src/server.ts

# Bun
bun src/server.ts
```

개발 중에는 아래처럼 watch 모드가 유용하다.

```bash
bun --watch src/server.ts
```

### 4. 마지막 단계: 테스트와 서버 런타임 전환

```bash
bun test
bun run dev
```

여기까지 안정적으로 동작하면, 실제 개발 루프 대부분을 Bun 기준으로 옮겼다고 봐도 된다.

## 예제로 배우기: 간단한 Task API

실행 가능한 예제는 아래 폴더에 두었다.

- [`web-development/typescript/bun/example-task-api/README.md`](/C:/Code/pm_notes/web-development/typescript/bun/example-task-api/README.md)

이 예제는 아래 세 가지를 동시에 보여준다.

- Bun 런타임으로 TypeScript 서버 실행
- `bun test`로 API 핸들러 테스트
- `bun install` 기준 프로젝트 구조

### 실행 방법

```bash
cd web-development/typescript/bun/example-task-api
bun install
bun run dev
```

다른 터미널에서:

```bash
curl http://localhost:3000/health
curl http://localhost:3000/tasks
curl -X POST http://localhost:3000/tasks \
  -H "Content-Type: application/json" \
  -d "{\"title\":\"write Bun docs\"}"
```

테스트:

```bash
bun test
```

### 예제 핵심 코드

```ts
import { createApp } from "./app";

const port = Number(Bun.env.PORT ?? 3000);
const app = createApp();

Bun.serve({
  port,
  fetch: app.fetch,
});
```

핵심은 다음이다.

- `Bun.env`로 환경 변수를 읽는다.
- `Bun.serve()`로 HTTP 서버를 바로 띄운다.
- 핸들러 로직을 `createApp()`에 분리해 두면 테스트가 쉬워진다.

## 예제 파일 읽는 순서

1. [`app.ts`](/C:/Code/pm_notes/web-development/typescript/bun/example-task-api/src/app.ts): 라우팅과 JSON 응답 로직
2. [`server.ts`](/C:/Code/pm_notes/web-development/typescript/bun/example-task-api/src/server.ts): Bun 서버 시작점
3. [`app.test.ts`](/C:/Code/pm_notes/web-development/typescript/bun/example-task-api/src/app.test.ts): Bun 테스트 러너 사용 예시

## npm에서 Bun으로 바꿀 때 기억할 것

### package-lock 대신 lockfile을 관리한다

- `bun install`을 쓰면 `bun.lock`이 생성된다.
- 팀 단위로 재현 가능한 설치를 원하면 lockfile을 커밋하는 편이 좋다.

### package.json 스크립트는 계속 중요하다

아래처럼 `package.json` 스크립트는 그대로 유지하는 편이 좋다.

```json
{
  "scripts": {
    "dev": "bun --watch src/server.ts",
    "start": "bun src/server.ts",
    "test": "bun test"
  }
}
```

이렇게 해 두면 팀원이 `bun run dev`, `bun run test`처럼 일관된 진입점을 쓸 수 있다.

### 모든 것을 한 번에 바꾸려고 하지 않는다

실전에서는 아래 순서가 안전하다.

1. `bun install`로 설치 전환
2. `bunx`로 CLI 실행 전환
3. `bun test` 또는 `bun src/file.ts` 전환
4. 마지막에 실제 서버 런타임 전환

## Bun을 효과적으로 쓰는 실전 팁

### 1. "작은 TypeScript 도구"부터 Bun으로 옮겨라

- 배치 스크립트
- 로그 정리 도구
- 파일 변환 유틸
- 간단한 API 서버

이런 코드는 Bun의 장점이 바로 체감된다.

### 2. 프론트엔드에서는 먼저 패키지 매니저로 써라

Vite, Vue, React 프로젝트라면 처음에는 아래만 바꿔도 충분하다.

```bash
bun install
bun run dev
bun run build
```

즉, 런타임까지 바로 바꾸지 않아도 생산성 이점을 얻을 수 있다.

### 3. `bunx`를 적극적으로 써라

```bash
bunx tsc --noEmit
bunx eslint .
bunx prettier --check .
```

CLI 툴을 임시 실행할 때 특히 편하다.

### 4. 크로스 플랫폼 스크립트를 단순화하라

Node 프로젝트는 종종 `rimraf`, `cross-env` 같은 보조 패키지를 많이 쓰는데, Bun Shell을 활용하면 이런 보조 도구 일부를 줄일 수 있다.

### 5. 호환성 검증은 초반에 짧게 끝내라

도입 초기에 아래만 빠르게 확인하면 된다.

- 설치가 정상적으로 끝나는가
- 개발 서버가 뜨는가
- 테스트가 도는가
- 빌드 산출물이 필요한 배포 환경에서 문제없는가

여기서 막히면 억지로 전부 Bun에 맞추기보다, 해당 단계만 Node를 유지하는 편이 더 현실적이다.

## 추천 학습 경로

1. `bun install`, `bunx`부터 익숙해진다.
2. 작은 `.ts` 파일을 `bun file.ts`로 실행해 본다.
3. `Bun.serve()`로 아주 작은 API를 만든다.
4. `bun test`로 테스트를 붙인다.
5. 마지막에 기존 npm 프로젝트 일부를 Bun 기준으로 옮긴다.

## 한 줄 요약

Bun의 핵심은 "npm의 대체재"에 그치지 않고, **TypeScript 개발에 필요한 실행기와 주변 도구를 하나로 압축해 개발 루프를 짧게 만드는 것**이다.
