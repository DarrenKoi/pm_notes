---
tags: [codex, cli, ai-agent, productivity]
level: intermediate
last_updated: 2026-03-07
status: complete
---

# Codex CLI 실전 가이드

> `codex-cli 0.111.0` 기준으로 정리한 실전 사용 가이드. 설치 설명은 제외하고, 실제로 빠르게 잘 쓰는 방법에 집중한다.

## 왜 필요한가? (Why)

- Codex CLI는 단순한 채팅 도구가 아니라 코드 탐색, 수정, 검증, 리뷰까지 이어지는 작업형 에이전트다
- 같은 질문을 하더라도 작업 디렉토리, 권한 정책, 프롬프트 구조를 어떻게 주느냐에 따라 결과 품질이 크게 달라진다
- 터미널 기반이라 로컬 저장소, Git 흐름, 자동화 스크립트와 연결하기 좋다

## 핵심 기능 (What)

| 기능 | 명령어/옵션 | 언제 쓰는가 |
|------|-------------|-------------|
| 대화형 작업 | `codex` | 코드베이스를 탐색하고 여러 턴으로 수정/검증할 때 |
| 즉시 시작 프롬프트 | `codex "..."` | 한 줄 요구사항으로 바로 세션을 시작할 때 |
| 비대화형 실행 | `codex exec` | 반복 가능한 작업, 스크립트, CI용 |
| 코드 리뷰 | `codex review` | 변경분의 위험, 버그, 테스트 공백을 점검할 때 |
| 세션 이어가기 | `codex resume`, `codex fork` | 이전 맥락을 재사용하거나 다른 방향으로 갈라서 작업할 때 |
| 샌드박스/승인 제어 | `--sandbox`, `--ask-for-approval`, `--full-auto` | 자동 실행 범위와 안전성을 조절할 때 |
| 작업 범위 제어 | `-C`, `--add-dir` | 정확한 프로젝트 루트만 보게 하고 싶을 때 |
| 프로필/설정 오버라이드 | `-p`, `-c` | 팀별 기본 설정과 일회성 옵션을 분리하고 싶을 때 |
| 모델/로컬 모델 선택 | `-m`, `--oss`, `--local-provider` | 원격/로컬 모델을 상황에 맞게 선택할 때 |
| 이미지 입력 | `-i` | UI 스크린샷이나 다이어그램을 같이 주고 싶을 때 |
| 웹 검색 | `--search` | 최신 문서나 외부 사실 검증이 필요한 작업일 때 |
| MCP 연동 | `codex mcp ...` | 외부 툴/서비스를 Codex 도구로 붙일 때 |
| 패치 반영 | `codex apply <TASK_ID>` | 다른 Codex 작업 결과를 로컬 Git 작업 트리에 반영할 때 |
| 에이전트 규칙 | `AGENTS.md` | 저장소별 작업 원칙, 스타일, 검증 규칙을 고정할 때 |
| 재사용 가능한 전문 기능 | `SKILL.md` 기반 skills | 특정 종류의 작업을 자동으로 더 잘 처리하게 할 때 |

## 추천 학습 순서

1. [대화형 작업 흐름](./01_interactive_workflow.md)부터 읽고 일상적인 사용 패턴을 익힌다
2. [비대화형 실행과 리뷰](./02_exec_and_review.md)로 자동화 가능한 지점을 이해한다
3. [컨텍스트, 샌드박스, 프로필](./03_context_sandbox_profiles.md)로 품질과 안전성을 조정한다
4. [세션 재사용, MCP, 고급 기능](./04_sessions_mcp_advanced.md)로 활용 범위를 넓힌다
5. [에이전트와 스킬](./05_agents_and_skills.md)로 Codex 동작 규칙과 확장 방식을 이해한다

## 먼저 이렇게 시작

```bash
# 1) 저장소 루트 대신 정확한 작업 폴더를 지정
codex -C web-development/python/flask/job-scheduler

# 2) 작업 목표와 완료 조건을 처음부터 명확히 전달
codex -C web-development/python/flask/job-scheduler \
  "백엔드 스케줄 실행 실패 원인을 찾고 수정해줘. \
  변경 후 실행 명령과 검증 결과까지 보여줘."

# 3) 변경분 리뷰만 빠르게 확인
codex review --uncommitted
```

## 이 가이드의 기준

- 명령어 구조와 옵션은 이 환경의 `codex --help` 출력 기준으로 정리
- 세부 옵션은 버전에 따라 달라질 수 있으므로, 실제 사용 전 `codex <subcommand> --help` 확인 권장

## 관련 문서

- [대화형 작업 흐름](./01_interactive_workflow.md)
- [비대화형 실행과 리뷰](./02_exec_and_review.md)
- [컨텍스트, 샌드박스, 프로필](./03_context_sandbox_profiles.md)
- [세션 재사용, MCP, 고급 기능](./04_sessions_mcp_advanced.md)
- [에이전트와 스킬](./05_agents_and_skills.md)
- [개발 환경 인덱스](../README.md)
