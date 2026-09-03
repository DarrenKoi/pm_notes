---
tags: [pi, coding-agent, terminal, packages]
level: intermediate
last_updated: 2026-07-22
---

# Pi 코딩 에이전트 실전 가이드

> Pi가 이미 설치되어 있다고 가정하고, 로컬 PC에서 실제 개발 작업을 시작하는 방법과 추천 Pi 패키지를 정리한다. Pi 자체 설치 과정은 다루지 않는다.

## 이 가이드에서 얻을 것

- Pi의 작은 코어와 확장 구조를 이해한다.
- 모델 연결부터 코드 탐색, 수정, 검증, 세션 재개까지 한 흐름으로 익힌다.
- 프로젝트 지침과 읽기 전용 도구 제한을 이용해 작업 범위를 제어한다.
- 서드파티 패키지를 무조건 많이 설치하지 않고 필요한 기능만 안전하게 추가한다.

이 문서는 2026-07-22 기준 [Pi 공식 문서](https://pi.dev/docs/latest)와 [공식 패키지 카탈로그](https://pi.dev/packages)를 확인해 작성했다. Pi와 패키지는 변화가 빠르므로 실제 명령과 최신 버전은 링크된 문서에서 다시 확인한다.

## 1. Pi를 이해하는 가장 짧은 방법

Pi는 로컬 터미널에서 실행되는 **작은 코딩 에이전트 하네스**다. 기본 기능을 크게 만들기보다 필요한 기능을 확장, 스킬, 프롬프트 템플릿, 테마, 패키지로 추가하는 방향을 택한다.

기본 도구는 다음 일곱 개다.

| 도구 | 역할 |
|------|------|
| `read` | 파일 읽기 |
| `bash` | 셸 명령 실행 |
| `edit` | 기존 파일 수정 |
| `write` | 파일 생성 또는 쓰기 |
| `grep` | 파일 내용 검색 |
| `find` | 파일 찾기 |
| `ls` | 디렉터리 목록 확인 |

Pi 코어에는 MCP, 서브에이전트, 계획 모드, 권한 확인 팝업, 백그라운드 셸이 기본으로 포함되지 않는다. 필요한 기능만 패키지로 붙이는 것이 Pi의 사용 방식이다. 자세한 설계 원칙과 도구 목록은 [Using Pi](https://pi.dev/docs/latest/usage)를 참고한다.

### 확장 요소 구분

| 요소 | 무엇인가 | 추천 용도 |
|------|----------|-----------|
| 컨텍스트 파일 | `AGENTS.md`, `CLAUDE.md`에 저장한 지침 | 저장소 규칙, 검증 명령, 금지 작업 |
| 프롬프트 템플릿 | 슬래시 명령처럼 재사용하는 프롬프트 | 반복 리뷰, 릴리스 점검 |
| 스킬 | 필요할 때 불러오는 절차 지침 | 테스트, 조사, 특정 업무 표준화 |
| 확장(Extension) | 도구, 명령, UI, 키 바인딩을 추가하는 TypeScript 코드 | Pi 동작 자체 확장 |
| Pi 패키지 | 확장, 스킬, 프롬프트, 테마를 묶은 배포 단위 | 기능 묶음 공유와 설치 |

스킬과 패키지의 세부 구조는 [Skills](https://pi.dev/docs/latest/skills)와 [Pi Packages](https://pi.dev/docs/latest/packages)에서 확인할 수 있다.

## 2. 설치 후 첫 세션

### 2.1 프로젝트 루트에서 시작한다

```bash
cd /path/to/project
pi
```

Pi가 보는 기본 작업 범위는 시작한 현재 디렉터리다. 저장소 전체 작업이라면 저장소 루트에서, 특정 모듈만 다룬다면 해당 모듈에서 시작한다.

화면에서 먼저 확인할 부분은 다음과 같다.

- 상단: 로드된 컨텍스트 파일, 스킬, 프롬프트, 확장
- 하단: 현재 디렉터리, 세션 이름, 토큰·캐시·비용, 컨텍스트 사용량, 모델

### 2.2 로그인하고 모델을 고른다

대화형 화면에서 다음 명령을 사용한다.

```text
/login
/model
/scoped-models
```

- `/login`: 구독 OAuth 또는 API 키 프로바이더 연결
- `/model`: 현재 사용할 모델 선택
- `/scoped-models`: 모델 순환 목록을 자주 쓰는 모델로 제한

Pi가 지원하는 프로바이더와 인증 방식은 [Providers](https://pi.dev/docs/latest/providers)에서 확인한다. 모델 이름은 자주 바뀔 수 있으므로 문서에 고정된 추천 모델을 복사하기보다 `/model`의 현재 목록에서 고르는 편이 안전하다.

인증 정보는 기본적으로 `~/.pi/agent/auth.json`에 저장된다. 이 파일이나 API 키를 저장소에 복사하거나 커밋하지 않는다.

### 2.3 세션에 바로 이름을 붙인다

```text
/name 로그인-오류-수정
```

세션이 많아지면 이름 없는 기록을 찾기 어렵다. 작업을 시작할 때 기능이나 문제 이름으로 `/name`을 실행하면 `/resume`에서 다시 찾기 쉽다.

## 3. 첫 번째 실전 작업

Pi에는 내장 계획 모드나 “수정 전 승인” 모드가 없다. 필요한 작업 경계를 프롬프트에 직접 명시한다.

### 3.1 좋은 첫 요청 구조

```text
목표: 로그인 API가 간헐적으로 500을 반환하는 원인을 찾아 수정해줘.

범위:
- backend/auth와 관련 테스트만 확인
- 다른 모듈은 수정하지 않기

진행:
1. 현재 코드와 git 상태를 먼저 읽기
2. 원인과 수정 계획을 설명하고 기다리기
3. 내가 승인하면 구현하기

검증:
- pytest tests/auth
- 변경 후 git diff 요약

완료 시:
- 변경 파일
- 실행한 검증과 결과
- 남은 위험을 보고하기
```

핵심은 `목표`, `범위`, `진행 방식`, `검증`, `완료 조건`을 한 번에 주는 것이다. 작업이 작다면 “계획 후 대기” 단계를 빼고 바로 구현하도록 요청해도 된다.

### 3.2 읽기 전용 검토는 도구로 제한한다

말로만 “수정하지 마”라고 요청하는 것보다 쓰기 도구를 아예 제외하는 편이 강하다.

```bash
pi --tools read,grep,find,ls -p \
  "이 저장소를 읽기 전용으로 검토하고 중요한 위험을 파일 근거와 함께 보고해줘."
```

이 모드에는 `bash`, `edit`, `write`가 없으므로 정적 검토에 적합하다. 테스트 실행까지 허용하려면 `bash`가 필요하지만, 그 순간 셸을 통한 쓰기도 가능해진다는 점을 이해해야 한다.

### 3.3 파일을 빠르게 컨텍스트에 넣는다

대화 입력창에서 `@`를 입력하면 프로젝트 파일을 퍼지 검색할 수 있다.

```text
@src/auth/session.ts @tests/auth/session.test.ts
두 파일의 세션 만료 처리 불일치를 찾아줘.
```

시작할 때 파일을 함께 넘길 수도 있다.

```bash
pi @src/auth/session.ts @tests/auth/session.test.ts \
  "세션 만료 로직을 비교하고 문제를 설명해줘."
```

### 3.4 에이전트가 일하는 동안 방향을 조정한다

| 입력 | 동작 |
|------|------|
| `Enter` | 현재 도구 호출이 끝난 뒤 전달할 방향 수정 메시지 큐잉 |
| `Alt+Enter` | 전체 작업이 끝난 뒤 전달할 후속 메시지 큐잉 |
| `Escape` | 현재 실행 중단, 큐의 메시지를 입력창으로 복원 |
| `Alt+Up` | 큐에 넣은 메시지를 다시 편집기로 가져오기 |
| `Ctrl+O` | 도구 출력 펼치기 또는 접기 |

예를 들어 예상보다 범위가 넓어질 때 다음과 같이 바로 조정한다.

```text
설정 파일은 수정하지 말고 원인 분석 범위를 auth 모듈로 제한해줘.
```

### 3.5 셸 결과를 보낼 때 비밀정보를 주의한다

```text
!git status --short
!npm test
!!git status --short
```

- `!command`: 명령을 실행하고 결과를 모델 컨텍스트에 포함
- `!!command`: 명령을 실행하지만 결과를 모델에 보내지 않음

`!env`, 토큰이 포함된 로그, `.env` 내용처럼 비밀정보가 출력될 수 있는 명령은 사용하지 않는다.

## 4. 권장 일상 워크플로

다음 흐름을 기본 루프로 사용하면 작업이 안정적이다.

1. 저장소 루트에서 `git status`를 확인한다.
2. `pi`를 시작하고 `/name`으로 작업 이름을 붙인다.
3. 관련 파일과 프로젝트 지침을 읽게 한다.
4. 큰 작업은 원인·계획을 먼저 설명하게 한다.
5. 범위가 맞으면 구현을 승인한다.
6. 저장소가 정한 lint, type-check, test, smoke test를 실행하게 한다.
7. `git diff`를 직접 검토한다.
8. `/session`으로 세션 정보와 사용량을 확인한다.
9. 다음에 이어갈 작업이면 종료 후 `pi -c`로 재개한다.

완료 요청 예시는 다음과 같다.

```text
변경을 마무리해줘.

- 저장소의 기존 검증 명령을 찾아 실행
- 실패하면 원인과 이번 변경의 관련 여부를 구분
- git diff에서 의도하지 않은 파일이 없는지 확인
- 변경 파일, 검증 결과, 남은 위험만 간결하게 보고
```

## 5. 프로젝트 지침을 `AGENTS.md`에 남긴다

반복해서 설명하는 규칙은 프로젝트의 `AGENTS.md` 또는 `CLAUDE.md`에 기록한다. Pi는 전역 `~/.pi/agent/AGENTS.md`와 현재 디렉터리까지의 관련 컨텍스트 파일을 시작할 때 로드한다.

```markdown
# Repository Guidelines

## Scope

- 작업 전 `git status --short`를 확인한다.
- 사용자 변경을 덮어쓰지 않는다.
- 요청받은 모듈 밖은 수정하지 않는다.

## Validation

- Python 변경: `pytest`
- Frontend 변경: `npm run lint`, `npm run typecheck`
- 완료 전 `git diff --check`

## Safety

- `.env`, 인증서, 개인 키를 읽거나 출력하지 않는다.
- `git reset --hard`, `git clean`, 강제 push를 실행하지 않는다.
- 삭제나 외부 전송은 먼저 승인을 받는다.
```

주의할 점이 있다. **프로젝트 신뢰를 거절해도 `AGENTS.md`와 `CLAUDE.md`는 기본적으로 로드될 수 있다.** 신뢰하지 않는 저장소라면 파일을 먼저 직접 확인하거나 `--no-context-files`를 사용한다. 자세한 동작은 [Security](https://pi.dev/docs/latest/security)에서 확인한다.

## 6. 세션과 컨텍스트 관리

Pi 세션은 기본적으로 `~/.pi/agent/sessions/` 아래에 작업 디렉터리별 JSONL로 자동 저장된다.

| 명령 | 용도 |
|------|------|
| `pi -c` | 가장 최근 세션 이어가기 |
| `pi -r` 또는 `/resume` | 세션 목록에서 선택 |
| `pi --no-session` | 저장하지 않는 임시 세션 |
| `/session` | 현재 세션 파일, ID, 메시지·토큰·비용 확인 |
| `/tree` | 같은 세션의 이전 지점으로 이동해 다른 분기 만들기 |
| `/fork` | 이전 사용자 메시지부터 새 세션 생성 |
| `/clone` | 현재 활성 분기를 새 세션으로 복제 |
| `/compact [초점]` | 오래된 대화를 요약해 컨텍스트 확보 |

사용 기준은 간단하다.

- 같은 문제를 이어서 해결: `pi -c`
- 이전 시도와 다른 방향을 시험: `/tree`
- 기존 기록을 보존하고 별도 작업으로 분리: `/fork` 또는 `/clone`
- 컨텍스트가 길어져 중요한 내용이 묻힘: `/compact 테스트 실패 원인과 현재 수정안 보존`

세부 동작은 [Sessions](https://pi.dev/docs/latest/sessions)와 [Compaction](https://pi.dev/docs/latest/compaction)을 참고한다.

## 7. 자주 쓰는 명령과 단축키

| 입력 | 용도 |
|------|------|
| `/hotkeys` | 현재 적용된 전체 단축키 확인 |
| `Ctrl+L` | 모델 선택기 |
| `Ctrl+P` / `Shift+Ctrl+P` | 허용된 모델을 앞뒤로 순환 |
| `Shift+Tab` | thinking level 순환 |
| `Shift+Enter` 또는 `Ctrl+J` | 줄바꿈 |
| `Ctrl+X` | 마지막 응답 복사 |
| `Ctrl+G` | 외부 편집기 열기 |
| `/settings` | 일반 실행 설정 |
| `/name` | 세션 이름 지정 |
| `/resume` | 저장된 세션 선택 |
| `/compact` | 긴 컨텍스트 요약 |
| `/reload` | 설정과 리소스 다시 로드 |
| `/changelog` | 현재 버전 변경사항 확인 |

단축키는 `~/.pi/agent/keybindings.json`에서 바꿀 수 있다. 터미널에 따라 `Shift+Enter`나 `Alt+Enter` 전달에 추가 설정이 필요할 수 있으므로 [Keybindings](https://pi.dev/docs/latest/keybindings)와 [Terminal Setup](https://pi.dev/docs/latest/terminal-setup)을 함께 확인한다.

## 8. 최소 설정 예시

전역 설정은 `~/.pi/agent/settings.json`, 프로젝트별 설정은 `.pi/settings.json`에 둔다. 프로젝트 설정이 전역 설정을 덮어쓰거나 병합한다.

```json
{
  "defaultProvider": "YOUR_PROVIDER",
  "defaultModel": "YOUR_MODEL",
  "defaultThinkingLevel": "medium",
  "defaultProjectTrust": "ask",
  "enabledModels": [
    "YOUR_DAILY_MODEL_PATTERN",
    "YOUR_HARD_TASK_MODEL_PATTERN"
  ]
}
```

처음부터 설정을 많이 넣지 말고 `/settings`, `/model`, `/scoped-models`로 실제 사용 패턴을 찾은 뒤 고정하는 편이 좋다. 전체 설정 키는 [Settings](https://pi.dev/docs/latest/settings)를 기준으로 확인한다.

## 9. 반드시 알아야 할 보안 경계

### 프로젝트 신뢰는 샌드박스가 아니다

프로젝트 신뢰는 `.pi/settings.json`, 프로젝트 확장, 스킬, 패키지 등을 로드할지 결정하는 입력 게이트다. Pi가 시작된 뒤 도구가 할 수 있는 작업을 제한하지는 않는다.

### Pi는 현재 사용자 권한으로 실행된다

Pi에는 기본 샌드박스와 권한 팝업이 없다. `bash`, 파일 도구, 확장은 Pi를 실행한 로컬 사용자와 같은 권한을 가진다.

따라서 다음 원칙을 지킨다.

- 정적 검토는 `--tools read,grep,find,ls`로 시작한다.
- 비밀정보와 개인 키는 작업 디렉터리에 두지 않거나 접근을 분리한다.
- 신뢰하지 않는 저장소, 무인 실행, 위험한 생성 코드는 컨테이너·VM·별도 샌드박스에서 실행한다.
- 호스트 디렉터리를 컨테이너에 쓰기 가능으로 마운트하면 호스트 파일도 수정될 수 있음을 기억한다.
- 에이전트 결과를 반영하기 전에 `git diff`와 검증 결과를 사람이 확인한다.

공식 권장 격리 방식은 [Security](https://pi.dev/docs/latest/security)와 [Containerization](https://pi.dev/docs/latest/containerization)을 참고한다.

## 10. PC 개발용 Pi 패키지 추천

여기서 “패키지”는 운영체제 패키지가 아니라 **Pi package**를 뜻한다.

### 먼저 결론

모든 사용자에게 반드시 필요한 서드파티 Pi 패키지는 없다. 코어 도구만으로 파일 탐색, 검색, 수정, 셸 실행, 세션 관리가 가능하다. 아래 추천은 부족한 기능이 실제로 확인됐을 때 하나씩 추가하는 후보 목록이다.

패키지 정보는 2026-07-22 스냅샷이다. [공식 패키지 문서](https://pi.dev/docs/latest/packages)는 모든 서드파티 패키지가 전체 시스템 권한으로 코드를 실행할 수 있으므로 설치 전 소스 검토를 요구한다. 패키지 카탈로그 노출이나 다운로드 수는 공식 보증을 의미하지 않는다.

### 우선 검토할 세 가지

| 패키지 | 스냅샷 버전 | 추천 대상 | 판단 |
|--------|---------------|-----------|------|
| [`cc-safety-net`](https://pi.dev/packages/cc-safety-net) | `1.0.6` | 파괴적인 Git·파일시스템 명령을 한 번 더 막고 싶은 사용자 | 소스 검토 후 기본 안전 보조로 고려 |
| [`pi-notify`](https://pi.dev/packages/pi-notify) | `1.4.0` | Pi가 작업하는 동안 다른 창을 사용하는 사용자 | 지원 터미널이라면 편의성이 높음 |
| [`pi-web-access`](https://pi.dev/packages/pi-web-access) | `0.13.0` | 최신 문서, URL, PDF, GitHub 조사가 잦은 사용자 | 네트워크 조사 작업이 있을 때만 추가 |

#### `cc-safety-net`

- 파괴적인 Git·파일시스템 명령을 실행 전에 분석하고 차단하는 보조 방어선이다.
- Pi에 기본 권한 팝업이 없다는 약점을 일부 보완한다.
- OS 수준 샌드박스가 아니며 파일, 자격 증명, 네트워크 접근을 격리하지 않는다.
- 차단 규칙이 모든 위험한 셸 표현을 완벽히 이해한다고 가정하면 안 된다.

#### `pi-notify`

- 에이전트가 작업을 마치고 입력을 기다릴 때 데스크톱 알림을 보낸다.
- Ghostty, iTerm2, WezTerm, Kitty, Windows Terminal 등을 지원한다.
- macOS Terminal.app과 Alacritty는 현재 패키지 문서상 지원하지 않는다.
- tmux에서는 `allow-passthrough` 설정이 필요하다.

#### `pi-web-access`

- 웹 검색, URL 내용 추출, PDF, GitHub 저장소, 영상 분석 기능을 추가한다.
- 검색 질의와 가져온 내용이 외부 서비스로 전송될 수 있다.
- API 키와 브라우저 쿠키 사용 범위를 확인하고, 사내 코드나 비밀정보를 검색 질의에 넣지 않는다.

### 필요가 명확할 때만 설치할 패키지

| 패키지 | 스냅샷 버전 | 설치할 때 | 주요 비용과 위험 |
|--------|---------------|-----------|------------------|
| [`pi-mcp-adapter`](https://pi.dev/packages/pi-mcp-adapter) | `2.11.0` | 이미 MCP 서버를 운영하고 있을 때 | 서버 프로세스, 자격 증명, 네트워크 경계가 추가됨 |
| [`pi-lens`](https://pi.dev/packages/pi-lens) | `3.8.71` | 대형·다중 언어 저장소에서 LSP, lint, type-check 피드백이 필요할 때 | 도구 자동 실행·포매팅 가능성, 큰 의존성 표면 |
| [`pi-subagents`](https://pi.dev/packages/pi-subagents) | `0.35.1` | 병렬 조사, 다각도 리뷰, 역할별 에이전트가 필요한 복잡한 작업 | 모델 호출 비용, 동시 파일 작업, 오케스트레이션 복잡도 증가 |
| [`@gotgenes/pi-permission-system`](https://pi.dev/packages/%40gotgenes/pi-permission-system) | `20.10.0` | 명령·경로별 `allow`/`ask`/`deny` 정책 UI가 꼭 필요할 때 | 보안 핵심 설정의 복잡도와 빠른 버전 변화 |

이 패키지들도 샌드박스를 대신하지 않는다. 특히 권한 시스템 확장은 Pi 프로세스 안에서 동작하는 가드레일이며, 신뢰하지 않는 작업에는 별도 OS 격리가 필요하다.

### 추천 구성 프로필

#### 일반 로컬 코딩

- 패키지 없이 Pi 코어로 시작
- 프로젝트 `AGENTS.md`
- 저장소 자체 lint, type-check, test

#### 안전 보조 + 장시간 작업

- `cc-safety-net`
- 지원 터미널이라면 `pi-notify`
- 정적 검토에는 여전히 `--tools read,grep,find,ls` 사용

#### 문서 조사 중심

- 위 기본 구성
- `pi-web-access`

#### 기존 MCP 사용자

- 위 기본 구성
- `pi-mcp-adapter`
- 실제 필요한 MCP 서버만 연결

#### 대규모 코드베이스 또는 고급 오케스트레이션

- 언어 피드백이 부족할 때 `pi-lens`
- 병렬 리뷰가 실제로 필요할 때 `pi-subagents`
- 처음부터 두 패키지를 동시에 넣지 말고 각각 효과와 비용을 확인

## 11. 패키지를 안전하게 시험하고 고정한다

### 11.1 영구 설치 전에 한 세션만 시험한다

```bash
pi -e npm:cc-safety-net@1.0.6
pi -e npm:pi-notify@1.4.0
pi -e npm:pi-web-access@0.13.0
```

`-e` 또는 `--extension`은 패키지를 임시 디렉터리에 받아 현재 실행에서만 로드한다.

### 11.2 검토가 끝나면 버전을 고정한다

```bash
pi install npm:cc-safety-net@1.0.6
pi install npm:pi-notify@1.4.0
pi install npm:pi-web-access@0.13.0
```

특정 프로젝트에서만 필요하면 프로젝트 로컬 설정에 기록한다.

```bash
pi install npm:pi-lens@3.8.71 -l
```

버전을 고정한 npm 패키지는 일반 패키지 업데이트에서 자동으로 최신 버전으로 올라가지 않는다. 새 버전은 변경사항과 소스를 다시 확인한 뒤 고정 버전을 직접 바꾼다.

### 11.3 설치한 리소스를 관리한다

```bash
pi list
pi config
pi update --extensions
pi remove npm:pi-web-access
```

- `pi list`: 설정에 기록된 패키지 확인
- `pi config`: 패키지 안의 불필요한 확장·스킬·프롬프트·테마 비활성화
- `pi update --extensions`: 고정되지 않은 패키지 업데이트
- `pi remove`: 패키지 제거

### 설치 전 체크리스트

1. Pi 패키지 페이지에서 소유자와 연결된 저장소를 확인한다.
2. 최근 커밋, 릴리스, 라이선스, 이슈 상태를 본다.
3. `package.json`의 Pi manifest, 의존성, lifecycle script를 확인한다.
4. 파일·명령·네트워크·API 키·설정 경로 접근 범위를 파악한다.
5. 폐기 가능한 저장소에서 `pi -e npm:<package>@<version>`으로 시험한다.
6. 한 프로젝트에만 필요하면 `-l`을 사용한다.
7. 검토한 버전을 고정한다.
8. 업데이트 전 다시 검토한다.

## 12. 자주 생기는 문제

| 증상 | 먼저 확인할 것 |
|------|----------------|
| 모델이 보이지 않음 | `/login`, `/model`, 프로바이더 인증 상태, `pi --list-models` |
| 이전 작업을 찾기 어려움 | 시작할 때 `/name`, 이후 `/resume` 또는 `pi -r` |
| 컨텍스트가 너무 길어짐 | `/compact`에 보존할 초점을 함께 전달 |
| `Shift+Enter` 또는 `Alt+Enter`가 안 됨 | `/hotkeys`, 터미널별 [Terminal Setup](https://pi.dev/docs/latest/terminal-setup) |
| 프로젝트 확장이 로드되지 않음 | 프로젝트 trust 상태와 `.pi/settings.json` 확인 |
| 설정을 바꿨지만 반영되지 않음 | `/reload` 실행 |
| 패키지 설치 후 동작이 이상함 | `pi list`, `pi config`, 패키지 버전과 소스 확인 후 제거·재시험 |
| 안전하게 읽기만 하고 싶음 | `pi --tools read,grep,find,ls -p "..."` |

## 13. 한 장 치트시트

```text
# 시작
cd /path/to/project
pi

# 첫 설정
/login
/model
/name <작업명>

# 작업 중
@파일명          관련 파일 첨부
Enter            방향 수정 메시지 큐
Alt+Enter        작업 완료 후 후속 메시지 큐
Escape           현재 실행 중단
Ctrl+O           도구 출력 펼치기/접기

# 세션
/session
/resume
/tree
/fork
/clone
/compact <보존할 초점>

# 재개
pi -c

# 읽기 전용 검토
pi --tools read,grep,find,ls -p "이 저장소를 읽기 전용으로 검토해줘"

# 패키지
pi -e npm:<package>@<version>       한 세션 시험
pi install npm:<package>@<version>  전역 고정 설치
pi install npm:<package>@<version> -l  프로젝트 로컬 설치
pi list
pi config
pi remove npm:<package>
```

## 참고 자료

- [Pi 공식 문서](https://pi.dev/docs/latest)
- [Using Pi](https://pi.dev/docs/latest/usage)
- [Providers](https://pi.dev/docs/latest/providers)
- [Settings](https://pi.dev/docs/latest/settings)
- [Keybindings](https://pi.dev/docs/latest/keybindings)
- [Sessions](https://pi.dev/docs/latest/sessions)
- [Compaction](https://pi.dev/docs/latest/compaction)
- [Security](https://pi.dev/docs/latest/security)
- [Containerization](https://pi.dev/docs/latest/containerization)
- [Pi Packages](https://pi.dev/docs/latest/packages)
- [Package Catalog](https://pi.dev/packages)
- [Pi 공식 저장소](https://github.com/earendil-works/pi)

## 관련 문서

- [Codex CLI 실전 가이드](../codex/README.md)
- [개발 환경 인덱스](../README.md)
