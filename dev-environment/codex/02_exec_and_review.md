---
tags: [codex, cli, exec, review, automation]
level: intermediate
last_updated: 2026-03-07
status: complete
---

# 비대화형 실행과 리뷰 (Exec & Review)

> 반복 가능한 작업은 `codex exec`, 변경 검토는 `codex review`로 분리하는 편이 좋다.

## `codex exec`는 언제 쓰는가?

- 같은 유형의 작업을 여러 저장소에서 반복할 때
- 셸 스크립트, Makefile, CI 잡에 붙이고 싶을 때
- 사람과 여러 턴 대화할 필요 없이 한 번에 결과를 받고 싶을 때

## 기본 패턴

### 인자 한 줄로 실행

```bash
codex exec "README에서 깨진 상대 링크를 찾아 고쳐줘."
```

### 표준 입력으로 긴 지시문 전달

```bash
cat prompt.txt | codex exec -
```

프롬프트를 파일로 관리할 수 있어 재현성이 좋다.

### 특정 디렉토리에서 실행

```bash
codex exec -C web-development/python/flask/job-scheduler \
  "lint 오류만 수정하고 변경 파일을 요약해줘."
```

### 세션 파일을 남기지 않기

```bash
codex exec --ephemeral "간단한 문서 정리만 해줘."
```

일회성 작업에 적합하다.

### 마지막 메시지만 파일로 저장

```bash
codex exec \
  --output-last-message /tmp/codex-last.txt \
  "변경 내용을 5줄 이내로 요약해줘."
```

### JSONL 이벤트로 받기

```bash
codex exec --json "현재 저장소의 Python 파일 수를 세고 설명해줘."
```

자동화 파이프라인에서 후처리하기 좋다.

### 구조화된 최종 응답 요구

```bash
codex exec \
  --output-schema ./result.schema.json \
  "변경점과 검증 결과를 스키마에 맞춰 반환해줘."
```

머신 리더블 출력이 필요할 때 유용하다.

## `codex review`는 언제 쓰는가?

- 커밋 전에 위험 요소를 빠르게 점검할 때
- 특정 브랜치 대비 변경 영향만 확인하고 싶을 때
- "요약"이 아니라 "문제점" 위주로 평가받고 싶을 때

## 기본 패턴

### 현재 작업 트리 리뷰

```bash
codex review --uncommitted
```

staged, unstaged, untracked 변경까지 함께 본다.

### 기준 브랜치 대비 리뷰

```bash
codex review --base main
```

기능 브랜치 작업을 한꺼번에 점검할 때 적합하다.

### 특정 커밋 리뷰

```bash
codex review --commit abc1234
```

### 리뷰 관점 추가

```bash
codex review --uncommitted \
  "버그 가능성과 테스트 누락을 우선적으로 봐줘."
```

## 추천 자동화 흐름

### 문서/정적 파일 정리

```bash
codex exec -C . \
  "README 링크와 제목 구조를 점검하고 필요한 부분만 수정해줘."
```

### 구현 후 리뷰

```bash
codex review --uncommitted \
  "동작 회귀, 예외 처리 누락, 테스트 공백 위주로 리뷰해줘."
```

### 결과를 PR 메모로 저장

```bash
codex review --base main --title "Scheduler retry fix"
```

## 실무 팁

- `exec`는 "수행", `review`는 "비판" 역할로 나누면 프롬프트가 단순해진다
- 자동화에서는 `--ephemeral`, `--json`, `--output-last-message` 조합이 유용하다
- 리뷰는 "좋은 점"보다 "문제점 우선"으로 요청하는 편이 신호대잡음비가 높다

## 관련 문서

- [Codex CLI 실전 가이드](./README.md)
- 이전: [대화형 작업 흐름](./01_interactive_workflow.md)
- 다음: [컨텍스트, 샌드박스, 프로필](./03_context_sandbox_profiles.md)
