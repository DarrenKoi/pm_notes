---
tags: [web-development, typescript]
reviewed_on: 2026-10-04
review_status: partial
document_type: learning_note
category_major: "웹 개발"
category_middle: "프론트엔드·런타임"
category_minor: "Bun·API 실습"
note_kind: "목차"
classified_on: "2026-10-05"
---

# Bun + TypeScript Task API 예제

Bun 런타임·HTTP handler 분리·내장 테스트 사용을 익히는 학습 예제다. [Bun 가이드](../../bun-basics.md)의 동작 개념을 먼저 읽는다.

## 실행과 읽기 순서

이 README가 있는 폴더에서 Bun을 설치한 상태로 실행한다.

```bash
bun install --frozen-lockfile
bun run dev
```

기본 URL은 `http://localhost:3000`, PORT 환경 변수로 포트를 지정할 수 있다. app.ts의 라우팅 → server.ts의 서버 연결 → app.test.ts의 handler 검증 순서로 읽는다.

- [app.ts](./src/app.ts): 입력 처리·메모리 task 목록·응답
- [server.ts](./src/server.ts): Bun.serve와 PORT
- [app.test.ts](./src/app.test.ts): 서버 포트 없이 Request/Response 검사

## 테스트와 요청

```bash
bun test
curl http://localhost:3000/health
curl http://localhost:3000/tasks
curl -X POST http://localhost:3000/tasks \
  -H 'Content-Type: application/json' \
  -d '{"title":"learn bun"}'
```

GET /health는 runtime·개수, GET /tasks는 items·count, POST /tasks는 새 task와 201·Location을 반환한다. 빈 title은 400이다. GET /tasks/:id는 구현되지 않아 Location 주소가 조회 endpoint임을 보장하지 않는다.

## 확인 범위와 적용 조건

**2026-10-04**, Bun **1.3.6**의 기존 테스트 4개가 통과했다. 서버·install·tsc 실행 성공을 뜻하지 않는다. 코드·lock은 원본 그대로다. @types/bun latest와 typescript ^5의 실제 설치/lock 호환은 확인해야 한다. 메모리 저장은 프로세스 재시작 시 초기화된다. 인증·영속 저장·요청 크기 제한·전체 JSON 형태 검증을 제공하는 운영 앱이 아니다. JSON null 입력은 title 접근 전에 객체 검사하지 않아 예외가 날 수 있으며 검토 기록에 남겼다. 실행 코드 수정은 이번 문서 정리 범위에서 하지 않는다.
