---
tags: [codex, vibe-coding, workflow, quality]
level: intermediate
last_updated: 2026-04-28
status: complete
---

# 바이브 코딩 제어 워크플로

> 빠르게 맡기되, 단계마다 멈춤점과 검증 기준을 둔다.

## 전체 흐름

1. **Scope**: 작업 루트와 변경 범위를 좁힌다.
2. **Survey**: 수정 전 관련 파일, 테스트, 실행 명령을 먼저 찾게 한다.
3. **Plan**: 큰 변경이면 구현 전 짧은 계획을 받는다.
4. **Patch**: 한 번에 하나의 논리 변경만 적용한다.
5. **Verify**: lint, typecheck, test, smoke test를 실행한다.
6. **Review**: 변경 이유, 위험, 남은 테스트 공백을 점검한다.
7. **Record**: 중요한 결정과 검증 명령을 문서/PR 설명에 남긴다.

## 1. 작업 시작 전

먼저 현재 상태를 확인한다.

```bash
git status --short
```

그다음 Codex를 좁은 루트에서 시작한다.

```bash
codex -p vibe-code -C web-development/python/flask/job-scheduler
```

작업 프롬프트에는 최소 네 가지를 넣는다.

- 목표: 무엇을 만들거나 고칠 것인가
- 범위: 어느 폴더/파일을 중심으로 볼 것인가
- 제약: 바꾸면 안 되는 것, 승인 받아야 하는 것
- 검증: 완료 후 실행해야 하는 명령

## 2. 좋은 시작 프롬프트

```text
web-development/python/flask/job-scheduler 안에서만 작업해줘.
스케줄 실행 실패 원인을 찾아 수정하고, 관련 테스트나 스모크 테스트를 실행해줘.
먼저 관련 파일과 실행 명령을 확인한 뒤, 수정 범위를 짧게 설명하고 구현해줘.
환경 변수나 외부 서비스 연결이 필요하면 실행 전에 알려줘.
```

## 3. 탐색 단계 하네스

수정 없이 구조만 알고 싶을 때:

```bash
codex -p vibe-research -C Codes/python/opensearch_handler \
  "수정하지 말고 검색 흐름, 테스트 위치, 위험한 의존성만 정리해줘."
```

탐색 단계에서 확인할 것:

- 진입점: app, CLI, script, route, handler
- 데이터 흐름: input -> transform -> output
- 테스트 위치와 실행 명령
- 외부 의존성: DB, Redis, OpenSearch, SaaS, network
- 설정 파일: `.env`, `config.py`, `config.yaml`, `pyproject.toml`, `package.json`

## 4. 구현 단계 하네스

구현 요청은 작게 나눈다.

나쁜 예:

```text
이 앱 전체를 더 좋게 만들어줘.
```

좋은 예:

```text
frontend/pages/index.vue의 필터 UI만 수정해줘.
API 계약은 바꾸지 말고, 변경 후 npm run lint와 npm run typecheck를 실행해줘.
```

큰 작업이면 먼저 계획을 받는다.

```text
구현 전에 관련 파일, 예상 변경 범위, 검증 명령을 5줄 이하로 제시해줘.
계획 확인 후 바로 구현해도 된다.
```

## 5. 검증 단계 하네스

Codex가 변경한 뒤에는 검증 명령을 명시적으로 요구한다.

Python 예:

```text
변경 후 해당 모듈에서 pytest를 실행하고, 실패하면 원인을 분석해서 수정해줘.
외부 OpenSearch가 필요하면 mock 테스트로 대체하고 그 이유를 남겨줘.
```

Nuxt/Vue 예:

```text
변경 후 frontend에서 npm run lint, npm run typecheck를 실행해줘.
UI 변경이면 로컬 dev server로 화면도 확인해줘.
```

문서 작업 예:

```text
링크가 깨지지 않도록 상대 경로를 확인하고, 관련 README 인덱스도 업데이트해줘.
```

## 6. 리뷰 단계 하네스

구현이 끝난 뒤에는 리뷰 모드로 한 번 더 본다.

```bash
codex review --uncommitted
```

리뷰에서 볼 것:

- 요구사항을 벗어난 수정이 있는가
- 테스트 없이 바뀐 핵심 로직이 있는가
- 보안/환경 변수/내부 URL 노출이 있는가
- API 계약, DB schema, 파일 경로가 깨졌는가
- 변경이 너무 넓어 분리해야 하는가

## 7. 중단해야 하는 신호

다음 상황에서는 계속 맡기지 말고 범위를 다시 줄인다.

- 모델이 같은 파일을 반복적으로 크게 고친다.
- 테스트 실패 원인을 설명하지 못하고 무작정 우회한다.
- 네트워크, secret, 외부 DB 연결을 계속 요구한다.
- 관련 없는 대규모 리팩터링을 시작한다.
- "아마 될 것"이라고만 하고 검증 명령을 실행하지 않는다.

## 8. 기록 방식

작업이 끝나면 다음 세 가지를 남긴다.

```text
변경 요약:
- ...

검증:
- npm run lint
- npm run typecheck

남은 위험:
- 외부 API 연동은 로컬에서 mock으로만 검증됨
```

이 기록은 PR 설명, 작업 로그, 문서 업데이트에 그대로 재사용할 수 있다.
