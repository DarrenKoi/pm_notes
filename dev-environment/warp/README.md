---
tags: [warp, terminal, ai-agent, productivity]
level: intermediate
last_updated: 2026-03-28
status: complete
---

# Warp 터미널 실전 가이드

> Warp를 이미 설치한 뒤, 실제 개발 작업과 AI 코딩 CLI(Codex, Claude Code)를 더 잘 쓰기 위한 사용법을 정리한 시리즈

## 왜 Warp를 쓰는가? (Why)

- 일반 터미널보다 입력창이 에디터처럼 동작해서 긴 명령어와 멀티라인 입력을 다루기 쉽다
- 명령어와 출력이 `Block` 단위로 정리되어, 에러 재현과 로그 재확인이 빠르다
- `Command Search`, `Workflows`, `Launch Configurations` 덕분에 반복 작업을 줄이기 좋다
- 탭/패널/세션 복원 기능이 강해서 프로젝트별 작업 공간을 빠르게 다시 열 수 있다
- Warp 자체 AI를 쓰지 않더라도, Codex나 Claude Code 같은 터미널 기반 에이전트를 운용하는 작업대 역할이 좋다

## 핵심 이점 (What)

| 기능 | 왜 유용한가 | Codex/Claude Code와의 연결 |
|------|-------------|-----------------------------|
| Blocks | 명령과 출력을 한 덩어리로 관리 | 실패 로그를 다시 보고, 복사하고, 재입력하기 쉽다 |
| Modern Input Editor | 터미널 입력이 IDE 입력창처럼 동작 | 긴 프롬프트, 멀티라인 명령, 복잡한 옵션 입력이 편하다 |
| Command Search / History | 예전에 쓴 명령을 빠르게 재사용 | 반복적인 `codex`, `claude`, `pytest`, `git` 명령 회수가 줄어든다 |
| Launch Configurations | 프로젝트별 탭/패널 구성을 저장 | AI 작업 pane, 테스트 pane, Git 확인 pane을 한 번에 복구할 수 있다 |
| Session Restoration | 종료 후에도 작업 문맥이 이어짐 | 어제 보던 로그, 실행 결과, 작업 pane 구성을 이어받기 좋다 |

## 추천 학습 순서

1. [설치 후 바로 익힐 기본기](./01_start_after_install.md)
2. [생산성을 높이는 핵심 기능](./02_productivity_features.md)
3. [Warp에서 Codex와 Claude Code 잘 쓰기](./03_codex_and_claude_code.md)
4. [Launch Configuration 예제](./04_launch_config_examples.md)

## 먼저 이렇게 시작

```bash
# 1) Warp에서 저장소를 연 뒤 작업 폴더로 이동
cd /Users/daeyoung/Codes/pm_notes

# 2) 왼쪽 pane은 AI 작업용
codex -C dev-environment

# 3) 오른쪽 pane은 Git/검증용
git status -sb
```

이후 반복해서 쓰는 구성이 안정되면 Launch Configuration으로 저장하는 흐름이 가장 효율적이다.

## 이 가이드의 기준

- Warp 기능 설명은 2026-03-28 기준 공식 문서를 바탕으로 정리
- Codex/Claude Code 명령 예시는 이 로컬 환경의 `codex --help`, `claude --help` 출력과 이 저장소의 기존 가이드를 함께 참고

## 관련 문서

- [설치 후 바로 익힐 기본기](./01_start_after_install.md)
- [생산성을 높이는 핵심 기능](./02_productivity_features.md)
- [Warp에서 Codex와 Claude Code 잘 쓰기](./03_codex_and_claude_code.md)
- [Launch Configuration 예제](./04_launch_config_examples.md)
- [Codex CLI 실전 가이드](../codex/README.md)
- [개발 환경 인덱스](../README.md)

## 참고 자료

- https://docs.warp.dev/
- https://docs.warp.dev/getting-started/keyboard-shortcuts
- https://docs.warp.dev/terminal/entry
- https://docs.warp.dev/terminal/sessions
