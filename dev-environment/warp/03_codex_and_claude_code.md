---
tags: [warp, codex, claude-code, ai-agent]
level: intermediate
last_updated: 2026-03-28
status: complete
---

# Warp에서 Codex와 Claude Code 잘 쓰기

> Warp는 에이전트 자체라기보다, 에이전트를 안정적으로 운영하기 좋은 터미널 작업대다.

## 역할을 먼저 분리하자

- Warp: 입력, 실행 결과, pane 구성, 반복 작업 관리
- Codex / Claude Code: 코드 탐색, 수정, 검증을 수행하는 작업형 에이전트

이 구분이 중요하다. Warp의 장점은 "에이전트를 대신하는 것"보다 "에이전트가 일하기 좋은 환경을 만드는 것"에 있다.

## 가장 추천하는 pane 구성

### 2-pane 기본형

- 왼쪽: `codex` 또는 `claude`
- 오른쪽: `git status -sb`, 테스트, 로그, diff 확인

### 3-pane 실전형

- 왼쪽 크게: 에이전트 메인 pane
- 오른쪽 위: 테스트/빌드 pane
- 오른쪽 아래: Git diff, README, 실패 로그 확인

이유는 단순하다. 에이전트와 검증을 한 pane에 섞으면 스크롤이 엉키고, 실패 원인 추적도 느려진다.

## Codex를 Warp에서 쓸 때 좋은 패턴

이 저장소에는 이미 [Codex CLI 실전 가이드](../codex/README.md)가 있으므로, Warp에서는 운영 방식에 집중하면 된다.

### 시작 예시

```bash
codex -C dev-environment
```

```bash
codex -C dev-environment \
  "warp 폴더에 튜토리얼 문서를 작성해줘.
  기존 문서 톤을 맞추고, 마지막에 검증 결과를 짧게 정리해줘."
```

### 자주 같이 쓰는 명령

```bash
codex review --uncommitted
```

```bash
codex --no-alt-screen
```

`--no-alt-screen`은 Codex의 대화 기록을 Warp 스크롤백과 함께 더 자연스럽게 보고 싶을 때 유용하다.

### Codex용 Warp 운영 팁

1. 항상 정확한 작업 루트에 들어간 뒤 실행한다.
2. 긴 프롬프트는 Warp 멀티라인 입력으로 정리해서 넣는다.
3. 검증 명령은 별도 pane에서 직접 확인한다.
4. 실패 출력은 Block 북마크 후, 그 내용을 바탕으로 다음 턴을 준다.

## Claude Code를 Warp에서 쓸 때 좋은 패턴

로컬 환경의 `claude --help` 기준으로 Claude Code도 대화형 세션, 재개, 권한 모드, worktree 같은 운영 옵션이 잘 갖춰져 있다.

### 시작 예시

```bash
claude
```

```bash
claude --add-dir dev-environment
```

```bash
claude -c
```

### Claude Code용 Warp 운영 팁

1. Claude 세션 하나에 너무 많은 프로젝트를 섞지 않는다.
2. `--add-dir`는 정말 필요한 경로만 추가한다.
3. 장기 작업은 한 pane, 일회성 질문은 다른 pane으로 분리한다.
4. 필요하면 `--worktree` 같은 격리 옵션을 고려한다.

## Codex와 Claude Code를 같이 쓸 때 원칙

### 같은 pane에서 번갈아 쓰지 말 것

세션 로그와 문맥이 섞여서 품질이 떨어진다.

### 같은 파일을 동시에 수정하게 하지 말 것

한 에이전트는 구현, 다른 에이전트는 리뷰나 문서 보조처럼 역할을 나누는 편이 낫다.

### 각자 다른 검증 pane를 두지 말고, 공용 검증 pane를 둔다

테스트 결과를 한곳에서 비교해야 혼선이 줄어든다.

## Warp가 여기서 주는 실제 이점

### 1. 프롬프트 입력이 편하다

- 긴 문장을 멀티라인으로 정리 가능
- 옵션이 많은 명령도 수정하기 쉬움
- 이전 실행을 Block에서 재입력 가능

### 2. 실패 로그를 다시 쓰기 좋다

- Block 단위로 실패 출력을 다시 확인
- 필요한 출력만 복사
- 북마크로 중요한 에러 유지

### 3. 반복 구성을 저장할 수 있다

- "Codex pane + test pane + git pane" 구조를 Launch Configuration으로 저장 가능
- 다음날 바로 같은 작업대를 복원 가능

## 추천 프롬프트 패턴

### Codex

```text
먼저 관련 파일을 파악하고, 수정 계획을 짧게 제시한 뒤 작업해줘.
수정 후에는 실행한 검증 명령과 남은 리스크만 짧게 정리해줘.
```

### Claude Code

```text
작업 범위를 먼저 좁히고, 변경 대상 파일과 검증 방법을 먼저 합의한 뒤 진행해줘.
```

둘 다 "목표 + 범위 + 제약 + 검증"이 들어가면 품질이 좋아진다.

## pm_notes 저장소 기준 추천 예시

### 문서 작업

```bash
codex -C dev-environment
claude --add-dir dev-environment
```

### 특정 앱 작업

```bash
codex -C web-development/python/flask/job-scheduler
claude --add-dir web-development/python/flask/job-scheduler
```

## 한 줄 요약

Warp에서 중요한 것은 AI가 아니라 `AI 세션, 검증, 로그 추적을 분리해서 운영하는 구조`다.

## 관련 문서

- [Codex CLI 실전 가이드](../codex/README.md)
- [Launch Configuration 예제](./04_launch_config_examples.md)
