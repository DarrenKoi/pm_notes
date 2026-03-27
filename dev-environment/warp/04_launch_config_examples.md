---
tags: [warp, launch-configurations, workspace]
level: intermediate
last_updated: 2026-03-28
status: complete
---

# Launch Configuration 예제

> Warp에서 가장 실무 체감이 큰 기능 하나를 꼽으면 Launch Configuration이다.

## 왜 중요한가

Codex나 Claude Code를 잘 쓰려면, 매번 같은 작업 공간을 빠르게 다시 여는 것이 중요하다.

- AI 작업 pane
- 테스트/빌드 pane
- Git 확인 pane
- 프로젝트별 시작 디렉토리

이 조합을 저장해두면, 터미널이 "명령 실행 창"이 아니라 "프로젝트 작업대"가 된다.

## UI로 저장하는 방법

1. 원하는 창/탭/pane 구성을 만든다.
2. `CMD-P`로 Command Palette를 연다.
3. `Save New Launch Configuration`을 찾는다.
4. 이름을 정해서 저장한다.
5. 다음부터는 `CTRL-CMD-L`로 Launch Configuration palette를 열어 다시 실행한다.

## 파일로 관리하는 방법

Warp 공식 문서 기준으로 Launch Configuration YAML 파일은 다음 위치에 저장된다.

```bash
$HOME/.warp/launch_configurations/
```

중요한 제약:

- `cwd`는 절대 경로여야 한다
- `~`를 쓰거나 빈 경로를 넣으면 목록에 안 보일 수 있다

## 예제 1. 문서 작업용 기본 구성

`pm_notes`처럼 문서와 코드가 함께 있는 저장소에서 무난한 구성이다.

```yaml
---
name: PM Notes Docs
windows:
  - tabs:
      - title: dev-environment
        color: blue
        layout:
          split_direction: horizontal
          panes:
            - cwd: /Users/daeyoung/Codes/pm_notes
            - split_direction: vertical
              panes:
                - cwd: /Users/daeyoung/Codes/pm_notes/dev-environment
                  commands:
                    - exec: git status -sb
                - cwd: /Users/daeyoung/Codes/pm_notes/dev-environment
                  commands:
                    - exec: ls
```

### 추천 용도

- 왼쪽: Codex 또는 Claude Code 실행
- 오른쪽 위: Git 상태 확인
- 오른쪽 아래: 문서/파일 확인

## 예제 2. Job Scheduler 작업용 구성

프론트엔드와 백엔드를 같이 봐야 하는 프로젝트에서 유용하다.

```yaml
---
name: Job Scheduler
windows:
  - tabs:
      - title: backend
        color: green
        layout:
          split_direction: vertical
          panes:
            - cwd: /Users/daeyoung/Codes/pm_notes/web-development/python/flask/job-scheduler
            - cwd: /Users/daeyoung/Codes/pm_notes/web-development/python/flask/job-scheduler
              commands:
                - exec: git status -sb
      - title: frontend
        color: yellow
        layout:
          cwd: /Users/daeyoung/Codes/pm_notes/web-development/python/flask/job-scheduler/frontend
```

## AI CLI용으로 쓸 때 권장 방식

### 권장: pane만 열고 에이전트는 수동 시작

이 방식이 기본값으로 가장 낫다.

- 매번 다른 목표를 가진 세션을 열 수 있다
- 원치 않는 자동 실행을 막을 수 있다
- 비용, 로그, 문맥 낭비가 적다

### 비권장: Launch Configuration에서 항상 에이전트를 자동 시작

가능은 하지만 보통은 비효율적이다.

- 프로젝트를 열기만 해도 에이전트가 시작될 수 있다
- 예전 명령이 자동으로 불필요하게 반복될 수 있다
- 같은 저장소에서 여러 용도를 번갈아 쓸 때 오히려 불편하다

## Launch Configuration과 Workflows의 차이

| 항목 | Launch Configuration | Workflow |
|------|----------------------|----------|
| 초점 | 창/탭/pane 구조 저장 | 명령 템플릿 저장 |
| 강점 | 작업대 복원 | 반복 명령 재사용 |
| AI CLI와의 궁합 | 세션 배치 관리에 좋음 | 검증/리뷰 명령 저장에 좋음 |

둘은 경쟁 기능이 아니라 같이 쓰는 편이 낫다.

## 추천 조합

1. Launch Configuration으로 pane 구조를 연다.
2. 왼쪽 pane에서 `codex` 또는 `claude`를 시작한다.
3. 오른쪽 pane에서는 Workflow나 Command Search로 검증 명령을 꺼낸다.

## 한 줄 요약

Launch Configuration은 Warp를 "터미널"에서 "재사용 가능한 개발 작업대"로 바꿔주는 기능이다.

## 참고 자료

- https://docs.warp.dev/terminal/sessions/launch-configurations
- https://docs.warp.dev/getting-started/keyboard-shortcuts
