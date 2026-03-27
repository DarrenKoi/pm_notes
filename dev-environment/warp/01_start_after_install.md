---
tags: [warp, terminal, basics]
level: beginner
last_updated: 2026-03-28
status: complete
---

# 설치 후 바로 익힐 기본기

> Warp는 "예쁜 터미널"보다 "명령 입력기 + 블록 히스토리 + 작업 공간 관리자"에 가깝다.

## 먼저 달라지는 개념 3가지

### 1. 입력창이 셸 한 줄이 아니라 에디터처럼 동작한다

- 긴 명령어를 여러 줄로 나눠 입력하기 쉽다
- 커서 이동, 단어 단위 삭제, 멀티 커서 같은 편집 감각이 일반 터미널보다 낫다
- 프롬프트가 긴 Codex/Claude Code 명령을 넣을 때 체감 차이가 크다

예시:

```bash
codex -C dev-environment \
  "warp 사용법 튜토리얼 문서를 구조화해서 작성해줘.
  설치 설명은 빼고, 실전 사용 예시를 포함해줘."
```

### 2. 실행 결과가 `Block` 단위로 쌓인다

Warp의 핵심은 명령어와 출력이 한 묶음으로 저장된다는 점이다.

- 에러가 난 실행만 다시 찾기 쉽다
- 특정 출력만 복사하기 쉽다
- 이전 명령을 다시 입력창으로 되돌리기 쉽다
- 중요한 실행 결과를 북마크할 수 있다

CLI 에이전트를 쓸 때는 "어떤 명령이 어떤 출력을 냈는지"를 다시 확인하는 시간이 많은데, Blocks가 그 비용을 줄여준다.

### 3. 탭보다 세션 단위 운영이 강하다

- split pane으로 역할을 분리하기 쉽다
- 세션 이동, 세션 복원, Launch Configuration 저장이 자연스럽다
- 프로젝트별로 같은 작업대를 반복해서 열기 좋다

## 설치 직후 30분 체크리스트

### 입력과 탐색

- `CMD-L`: 입력창 포커스
- `CTRL-J`: 입력창 안에서 줄바꿈
- `CMD-P`: Command Palette 열기
- `CTRL-R`: Command Search 열기

### Block 다루기

- `CMD-UP` 또는 `CMD-DOWN`: 최근 Block 선택 시작
- `CMD-B`: 중요한 Block 북마크
- `CMD-K`: Block 정리
- Block 우클릭: 명령 복사, 출력 복사, 재입력

### 작업 공간

- `CMD-D`: 오른쪽 split pane
- Command Palette에서 `sessions:` 검색: 세션 이동
- `CTRL-CMD-L`: Launch Configuration palette

## 추천하는 첫 작업 배치

### 2-pane 기본형

- 왼쪽: 실제 작업 pane
- 오른쪽: `git status`, 테스트, 로그 확인 pane

이 구성이 가장 무난하다. AI 에이전트가 긴 작업을 하는 동안 오른쪽에서 결과만 확인하면 된다.

### 3-pane 실전형

- 왼쪽 크게: Codex 또는 Claude Code
- 오른쪽 위: 테스트/빌드
- 오른쪽 아래: Git diff, 로그 tail, 문서 확인

이 구조는 "에이전트가 일하는 pane"과 "내가 검증하는 pane"을 분리하기 좋아서 실전성이 높다.

## 자주 쓰게 되는 macOS 기준 단축키

| 단축키 | 의미 | 추천 상황 |
|--------|------|-----------|
| `CMD-L` | 입력창 포커스 | pane 이동 후 바로 명령 입력 |
| `CMD-D` | 오른쪽 분할 | 작업/검증 pane 분리 |
| `CMD-P` | Command Palette | 기능 이름이 기억 안 날 때 |
| `CTRL-R` | Command Search | 예전 명령 재사용 |
| `CTRL-SHIFT-R` | Workflow 검색 | 저장한 반복 작업 실행 |
| `CTRL-CMD-L` | Launch Configuration palette | 프로젝트 작업대 다시 열기 |
| `CMD-B` | Block 북마크 | 중요한 실패/성공 결과 표시 |
| `CMD-K` | Block 정리 | 세션을 깨끗하게 다시 시작 |

## 처음부터 습관 들이면 좋은 것

1. 긴 명령은 한 줄로 우겨 넣지 말고 멀티라인으로 입력한다.
2. 실패한 테스트 결과는 해당 Block을 북마크한다.
3. 반복 명령은 매번 다시 치지 말고 Command Search 또는 Workflow로 꺼낸다.
4. 프로젝트마다 pane 구성이 안정되면 Launch Configuration으로 저장한다.

## 한 줄 요약

Warp 입문 핵심은 `에디터 같은 입력`, `Block 단위 결과`, `pane 기반 작업 분리` 세 가지다.

## 참고 자료

- https://docs.warp.dev/getting-started/keyboard-shortcuts
- https://docs.warp.dev/terminal/blocks
- https://docs.warp.dev/terminal/blocks/block-basics
