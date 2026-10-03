---
tags: [pi, coding-agent, mcp, codemode, durable]
level: intermediate
last_updated: 2026-10-04
---

# Pi 1.0.1 코딩 에이전트 활용 가이드

> 터미널에서 코딩하는 Pi와 장기 실행 서비스를 만드는 Pi Durable을 구분하고, 설치부터 실전 개발·확장·자동화까지 익힌다.

## 왜 필요한가? (Why)

Pi는 모델 요청, 도구 실행, 대화 컨텍스트, 세션 저장을 연결하는 확장 가능한 에이전트 하네스다. 개발자는 터미널에서 파일을 읽고 수정하고 테스트하게 할 수 있고, 프로그램에서는 CLI·RPC·SDK로 같은 작업 흐름을 제어할 수 있다. 작은 기본 구성에 프로젝트 지침, 재사용 프롬프트, 스킬, 확장을 더하는 방식이다. [실행 구조](https://github.com/earendil-works/pi/blob/v1.0.1/packages/coding-agent/docs/how-pi-works.md)

**추천:** 개인 개발은 Pi CLI로 시작한다. 반복 작업은 프롬프트 템플릿과 스킬로 정리하고, 시스템 연결은 내장 MCP를 이용한다. 프로세스 재시작 뒤에도 진행 중인 작업을 복구해야 하는 서비스를 직접 만들 때 Pi Durable을 검토한다.

### 조사 기준과 검증 범위

- 조사일: **2026-10-04, 한국 시간**.
- 최신 안정 릴리스: **v1.0.1**. v1.0.0은 10월 1일, v1.0.1은 10월 3일 UTC에 공개됐다. [1.0.0](https://github.com/earendil-works/pi/releases/tag/v1.0.0), [1.0.1](https://github.com/earendil-works/pi/releases/tag/v1.0.1)
- npm 패키지: `@earendil-works/pi-coding-agent@1.0.1` 및 `@earendil-works/pi-durable@1.0.1`. Rust 동명 프로젝트와 구분한다. [CLI manifest](https://github.com/earendil-works/pi/blob/v1.0.1/packages/coding-agent/package.json), [Durable manifest](https://github.com/earendil-works/pi/blob/v1.0.1/packages/durable/package.json)
- 이 PC의 `pi --version` 결과도 **1.0.1**이다. 전역 설정이나 인증 파일은 수정하지 않았다.
- 공식 v1.0.1 소스와 문서를 확인했고, 별도 임시 설정 디렉터리에서 CLI 도움말을 확인했다. 유료 모델 호출, 사용자 프로젝트 수정, 실제 MCP 서버 연결은 수행하지 않았다.
- Durable 저장·복구 예제의 실행 결과는 [별도 가이드](durable.md)에 기록한다.

## 핵심 개념 (What)

### 모델과 하네스의 역할

```text
사용자 요청
  → 지침 + 활성 세션 분기 + 도구 정의로 컨텍스트 구성
  → 선택한 provider/model에 요청
  → 모델의 응답·도구 호출
  → 로컬 도구 실행 및 결과 저장
  → 필요하면 다음 모델 요청
  → 작업 종료 및 다음 사용자 입력 대기
```

모델이 추론하고, Pi가 실행 흐름을 관리한다. 모델을 바꿔도 저장소 지침·세션·도구를 이용하는 방식은 유지된다. `Enter`로 보낸 방향 수정은 현재 assistant turn과 그 도구 호출이 끝난 뒤 들어가고, `Alt+Enter` 후속 요청은 현재 작업이 끝난 뒤 들어간다. [Agent loop](https://github.com/earendil-works/pi/blob/v1.0.1/packages/coding-agent/docs/how-pi-works.md)

### 기본 도구와 기본 활성 도구는 다르다

| 도구 | 용도 | 기본 활성 |
|---|---|---|
| `read` | 텍스트·지원 이미지 읽기 | 예 |
| `bash` | 셸 실행 | 예 |
| `edit` | 기존 텍스트 정확히 치환 | 예 |
| `write` | 파일 생성·덮어쓰기 | 예 |
| `grep`, `find`, `ls` | 내용 검색·경로 검색·목록 | 아니오 |
| `powershell` | Windows PowerShell 실행 | 아니오 |
| `codemode` | JavaScript로 도구 호출·결과 가공 | 아니오, MCP 구성에 따라 자동 활성 |
| `tool_search` | 지연 노출 도구 검색 | 아니오, MCP 구성에 따라 자동 활성 |

Pi 1.0.1에는 **내장 MCP 지원**이 있다. 서브에이전트와 계획 모드는 기본 기능으로 제공하지 않는다. 이 문서의 과거 버전에 있던 “MCP는 기본 기능이 아니다”와 `pi-mcp-adapter` 필수 설치 설명은 최신 기준에서 제거했다. [CLI 도구](https://github.com/earendil-works/pi/blob/v1.0.1/packages/coding-agent/docs/cli.md#tools), [설계 방향](https://github.com/earendil-works/pi/blob/v1.0.1/packages/coding-agent/README.md)

### 커스터마이징 선택 기준

| 필요한 것 | 사용할 수단 | 예 |
|---|---|---|
| 저장소 공통 규칙 | `AGENTS.md` | 작업 범위와 테스트 명령 |
| 반복 요청 | Prompt template | `/review`, `/finish` |
| 특정 업무 절차·자료 | Skill | 회귀 테스트, 데이터 분석 |
| 외부 도구·데이터 연결 | MCP | 문서 검색, 이슈 시스템 |
| 이벤트·도구·UI 변경 | TypeScript Extension | 작업 종료 알림, 도구 정책 |
| 여러 리소스 공유 | Pi package | 팀의 프롬프트·스킬 묶음 |
| 앱에서 코딩 에이전트 제어 | RPC 또는 Coding Agent SDK | IDE, 내부 웹 UI |
| 지속적인 task·conversation 런타임 구축 | Pi Durable | 재시작 가능한 장기 작업 서비스 |

단순 규칙에 실행 코드를 만들기보다 위 표의 가장 작은 수단부터 선택하는 것을 권한다. [Quickstart](https://github.com/earendil-works/pi/blob/v1.0.1/packages/coding-agent/docs/quickstart.md#choose-how-to-customize-pi)

## 어떻게 사용하는가? (How)

## 1. 설치·업데이트

이미 `pi --version`이 1.0.1이면 다시 설치할 필요가 없다.

```bash
pi --version
```

새 설치에는 공식 managed installer를 권한다. 의존성을 고정하고 `pi update`로 갱신한다.

```bash
# macOS / Linux
curl -fsSL https://pi.dev/install.sh | sh

# 설치 후 확인
pi --version
```

Windows PowerShell에서는 다음을 사용한다.

```powershell
irm https://pi.dev/install.ps1 | iex
pi --version
```

npm으로 **이번 조사 버전**을 지정하려면 다음과 같다. Node.js **22.19 이상**이 필요하다.

```bash
npm install -g --ignore-scripts @earendil-works/pi-coding-agent@1.0.1
```

npm 최상위 패키지 버전만 지정해도 전이 의존성까지 고정되지는 않는다. v1.0.1에서 npm shrinkwrap이 제거됐으므로, 설치 재현성이 중요하면 managed installer 또는 앱의 lockfile을 사용한다. [설치](https://github.com/earendil-works/pi/blob/v1.0.1/packages/coding-agent/docs/quickstart.md), [릴리스 변경](https://github.com/earendil-works/pi/releases/tag/v1.0.1)

```bash
pi update                 # Pi 본체
pi update --extensions    # 설치한 Pi 패키지
pi update --models        # 모델 카탈로그
pi update --all           # 본체와 패키지
```

Nix 설치는 `nix profile upgrade pi`로 갱신한다. 일반 npm 설치의 `pi update`는 managed 설치로 이전할 것을 권한다. [업데이트 명령](https://github.com/earendil-works/pi/blob/v1.0.1/packages/coding-agent/docs/cli.md#update-pi-or-packages)

## 2. 첫 실행과 모델 선택

```bash
cd /path/to/project
pi
```

Pi 안에서 순서대로 입력한다.

```text
/login
/model
/thinking
/name 첫-코드-탐색
```

- `/login`: 지원 구독 또는 API 키 연결.
- `/model`: 인증이 준비된 모델 선택. `Ctrl+S`로 새 세션 기본값 저장.
- `/thinking`: 모델이 지원하는 추론 수준 선택. 높은 수준은 어려운 분석에 유용하지만 지연과 비용을 비교한다.
- `/scoped-models`: `Ctrl+P`로 순환할 모델 범위를 제한.

모델 ID는 `/model` 또는 `pi --list-models`의 실제 목록에서 선택한다. 구독 인증 가능 여부와 과금 방식은 provider별로 다르므로 API 키와 같은 방식이라고 가정하지 않는다. 인증은 기본 `~/.pi/agent/auth.json`에 저장된다. [모델](https://github.com/earendil-works/pi/blob/v1.0.1/packages/coding-agent/docs/models.md), [Providers](https://github.com/earendil-works/pi/blob/v1.0.1/packages/coding-agent/docs/providers.md)

### 사내 또는 로컬 호환 API 연결

OpenAI Chat Completions 호환 endpoint라면 `~/.pi/agent/models.json`에 다음처럼 추가한다. 아래 provider/model/URL은 교체해야 하는 예시다.

```json
{
  "providers": {
    "my-compatible-server": {
      "baseUrl": "http://127.0.0.1:8000/v1",
      "api": "openai-completions",
      "apiKey": "${LOCAL_MODEL_API_KEY}",
      "models": [
        { "id": "YOUR_SERVED_MODEL_ID" }
      ]
    }
  }
}
```

환경변수에 키를 설정하고 `/model`을 다시 열면 파일을 다시 읽는다. 인증이 없는 로컬 서버는 서버가 허용하는 dummy key를 쓸 수 있다. Responses API 서버라면 API 타입도 서버와 맞춰야 한다. 이름만 “OpenAI compatible”이라고 해서 tool calling, reasoning, streaming이 전부 같은 것은 아니다. [호환 endpoint 설정](https://github.com/earendil-works/pi/blob/v1.0.1/packages/coding-agent/docs/models.md#configure-a-compatible-endpoint)

## 3. 첫 실습: 탐색 → 수정 → 검증

처음에는 작은 프로젝트에서 파일 수정 없이 흐름을 확인한다.

```text
이 프로젝트의 AGENTS.md와 README를 읽어줘.
실행 진입점, 핵심 모듈, 테스트 명령을 파일 근거와 함께 설명해줘.
파일은 수정하지 말고 마지막에 작은 개선 후보 하나만 제안해줘.
```

정적 탐색만 필요하면 도구도 제한한다.

```bash
pi --tools read,grep,find,ls -p \
  "src와 tests를 읽기 전용으로 검토하고 실제 문제를 파일 근거와 함께 보고해줘."
```

이 명령은 모델의 활성 도구 선택을 제한한다. 확장 자체의 실행 권한이나 OS 권한을 격리하지는 않는다. `bash`를 포함하면 셸을 통한 쓰기도 가능하므로, `edit`만 빼는 것은 읽기 전용 설정이 아니다. [도구 선택](https://github.com/earendil-works/pi/blob/v1.0.1/packages/coding-agent/docs/cli.md#tools), [보안 경계](https://github.com/earendil-works/pi/blob/v1.0.1/packages/coding-agent/docs/security.md)

수정 작업은 다음 형태가 실용적이다.

```text
목표: 로그인 요청에서 만료된 세션을 정상적으로 거절하도록 수정.
범위: auth 모듈과 직접 관련 테스트.
완료 조건: 만료 세션 회귀 테스트 통과, 기존 인증 테스트 통과.
제약: 다른 모듈 수정, 커밋, push는 하지 않기.
진행: 먼저 원인과 수정 계획을 설명하고 기다리기.
완료 보고: 변경 파일, 검증 명령과 결과, 남은 불확실성.
```

계획을 확인한 뒤 “그 범위로 구현하고 검증해줘”라고 이어간다. 작은 수정은 처음부터 구현을 요청해도 된다. 프롬프트의 대기 요청은 모델 지침이며 내장 승인 모드가 아니다.

### 컨텍스트를 직접 선택한다

```text
@src/auth/session.ts @tests/auth/session.test.ts
이 두 파일의 만료 판정이 일치하는지 확인해줘.
```

CLI 시작 인자로도 파일을 넣을 수 있다.

```bash
pi @README.md "실행 방법을 설명해줘"
```

`@`는 파일 검색, `Tab`은 경로 완성, 이미지 붙여넣기는 지원 터미널에서 사용할 수 있다. [입력](https://github.com/earendil-works/pi/blob/v1.0.1/packages/coding-agent/docs/usage.md#enter-a-prompt)

### 실행 중 제어

| 입력 | 의미 |
|---|---|
| `Enter` | 현재 assistant turn과 도구 실행 뒤 방향 수정 |
| `Alt+Enter` | 현재 작업을 마친 뒤 후속 작업 |
| `Escape` | 실행 중단, 큐 메시지를 편집기로 복원 |
| `Alt+Up` | 큐 메시지를 편집기로 가져오기 |
| `Ctrl+O` | 도구 출력 펼치기·접기 |
| `Ctrl+T` | thinking 표시·숨김 |
| `Ctrl+G` | 외부 편집기 |
| `/hotkeys` | 현재 실제 단축키 확인 |

```text
!git status --short
!npm test
!!git status --short
```

`!`는 셸 결과를 모델 컨텍스트에 포함하고, `!!`는 모델에 보내지 않는다. 둘 다 실제 셸 명령을 실행한다. [실행 중 제어](https://github.com/earendil-works/pi/blob/v1.0.1/packages/coding-agent/docs/usage.md)

## 4. 저장소 지침과 설정

프로젝트에 반복 적용할 규칙은 `AGENTS.md`에 적는다.

```markdown
# Repository Guidelines

- 시작 전에 git status --short를 확인한다.
- 사용자가 변경한 파일을 덮어쓰지 않는다.
- 요청한 모듈 안에서만 작업한다.
- 변경과 관련된 lint, type-check, test를 실행한다.
- 완료 시 변경 파일, 실행 결과, 미검증 사항을 보고한다.
- 커밋과 push는 사용자가 요청했을 때만 한다.
```

Pi는 agent directory와 현재 디렉터리 및 부모 디렉터리의 컨텍스트 파일을 발견한다. 시작 시 하위 폴더 전체의 모든 지침을 자동 로드하는 것은 아니다. 하위 작업 지침은 해당 위치에서 시작하거나 직접 읽도록 요청한다. [컨텍스트 파일](https://github.com/earendil-works/pi/blob/v1.0.1/packages/coding-agent/docs/configuration.md#context-files)

| 위치 | 용도 |
|---|---|
| `~/.pi/agent/settings.json` | 전역 기본 설정 |
| `.pi/settings.json` | 프로젝트 설정 |
| `~/.pi/agent/models.json` | endpoint와 모델 |
| `~/.pi/agent/auth.json` | 인증 정보 |
| `.pi/prompts/`, `.pi/skills/`, `.pi/extensions/` | 프로젝트 리소스 |
| `~/.pi/agent/mcp.json`, `.pi/mcp.json` | MCP 서버 |

최소 설정은 다음 정도로 시작한다. `defaultProvider`와 `defaultModel`은 `/model`에서 저장하게 하는 편이 쉽다.

```json
{
  "defaultThinkingLevel": "medium",
  "defaultTools": ["+grep", "+find", "+ls"]
}
```

`+name`은 기존 기본 도구에 추가한다. `["read", "grep"]`처럼 일반 이름을 넣으면 기본 선택을 교체한다. `defaultProjectTrust`는 **전역 설정에서만** 지정할 수 있다. 수동 변경 후 `/reload`를 사용하되 `defaultTools`에서 도구를 제거한 것이 reload만으로 즉시 비활성화되는 것은 아니다. 엄격한 도구 변경은 새 세션 프로세스에서 확인한다. [Settings](https://github.com/earendil-works/pi/blob/v1.0.1/packages/coding-agent/docs/settings.md)

프로젝트 trust는 프로젝트 리소스를 로드할지 결정한다. 도구 실행마다 승인을 받거나 작업 디렉터리 밖 접근을 막는 기능은 아니다. 컨텍스트 지침은 trust와 별개로 로드되며 `--no-context-files`로 끌 수 있다. [Project trust](https://github.com/earendil-works/pi/blob/v1.0.1/packages/coding-agent/docs/security.md#understand-project-trust)

## 5. 세션·분기·컨텍스트 관리

```bash
pi -c                         # 현재 폴더의 최근 세션
pi -r                         # 선택기
pi --session-id auth-fix       # 같은 ID 열기 또는 생성
pi --session SESSION_ID       # 기존 세션 열기
pi --no-session               # 기록을 남기지 않는 실행
```

`--session-id`는 `--continue`, `--resume`, `--session`과 함께 사용하지 않는다. [세션 CLI](https://github.com/earendil-works/pi/blob/v1.0.1/packages/coding-agent/docs/cli.md#sessions)

| 명령 | 효과 | 권장 상황 |
|---|---|---|
| `/name` | 세션 이름 | 시작할 때 |
| `/session` | 파일·ID·토큰·비용 | 비용과 기록 확인 |
| `/tree` | 같은 파일에서 이전 지점으로 이동·분기 | 다른 접근 시험 |
| `/fork` | 이전 사용자 메시지에서 새 세션 | 별도 실험 |
| `/clone` | 활성 분기를 새 세션으로 복사 | 현재 상태 복제 |
| `/compact 초점` | 오래된 컨텍스트 요약 | 긴 작업 |
| `/new` | 새 세션 | 독립 과제 |

중요한 구분: **대화 분기는 Git 상태나 작업 파일을 되돌리지 않는다.** 세션 트리는 대화 이력이다. 다른 코드를 안전하게 비교하려면 Git branch/worktree 또는 별도 복사본을 사용한다. 저장된 원본 이력과 모델에 보내는 요약 컨텍스트도 다르다. [세션](https://github.com/earendil-works/pi/blob/v1.0.1/packages/coding-agent/docs/sessions.md), [포맷](https://github.com/earendil-works/pi/blob/v1.0.1/packages/coding-agent/docs/session-format.md)

```text
/compact 완료한 변경, 실패한 테스트, 다음 할 일, 수정 금지 범위를 보존해줘
```

컨텍스트가 길어지면 자동 compaction이 가능하지만 provider 문제가 있으면 요약도 실패할 수 있다. 세션을 재개할 때는 “현재 git 상태와 마지막 검증 결과를 다시 확인하고 이어가줘”라고 요청하는 것을 권한다. [Compaction](https://github.com/earendil-works/pi/blob/v1.0.1/packages/coding-agent/docs/compaction.md)

## 6. 반복 작업을 프롬프트·스킬로 만든다

### 프롬프트 템플릿

프로젝트 `.pi/prompts/review.md`에 저장한다.

```markdown
---
description: 변경사항을 근거 중심으로 검토
argument-hint: "[검토 범위]"
---
${1:-현재 변경사항}을 읽고 실제 결함만 검토해줘.
파일은 수정하지 말고 파일 위치, 실패 조건, 영향, 수정 방향을 보고해줘.
문제가 없으면 발견 없음이라고 말해줘.
```

```text
/reload
/review "auth 모듈의 현재 변경사항"
```

템플릿은 입력 텍스트를 확장하는 기능이다. 도구 제한이나 승인 정책을 강제하지 않는다. [Prompt templates](https://github.com/earendil-works/pi/blob/v1.0.1/packages/coding-agent/docs/prompt-templates.md)

### 스킬

`.pi/skills/verify-change/SKILL.md`를 만든다.

```markdown
---
name: verify-change
description: 코드 변경을 마무리하고 저장소의 검증 결과를 보고할 때 사용한다.
---

# 변경 검증

1. 변경 범위와 기존 사용자 변경을 확인한다.
2. 저장소 문서에서 해당 모듈의 검증 명령을 찾는다.
3. 관련 lint, type-check, test를 실행한다.
4. 실패를 이번 변경과 기존 문제로 구분한다.
5. git diff --check를 실행한다.
6. 변경 파일, 실행 명령, 실제 결과, 미검증 영역을 보고한다.
```

```text
/reload
/skill:verify-change 이번 auth 수정의 검증을 마무리해줘
```

시작 때는 이름·설명·경로만 컨텍스트에 들어가고, 본문은 필요할 때 읽는다. 모델이 자동으로 선택하지 않으면 `/skill:name`으로 명시한다. `~/.agents/skills`와 프로젝트 `.agents/skills`도 지원한다. [Skills](https://github.com/earendil-works/pi/blob/v1.0.1/packages/coding-agent/docs/skills.md)

## 7. 내장 MCP 사용

최신 Pi에서는 MCP adapter를 먼저 설치할 필요가 없다.

공식 문서의 filesystem 예제로 연결 흐름을 확인할 수 있다. 이 명령은 서버 패키지를 내려받고 실행한다.

```bash
pi mcp add filesystem -- npx -y @modelcontextprotocol/server-filesystem .
pi mcp list
pi
```

프로젝트 전용 설정은 `pi mcp add -l ...`로 추가한다. 진행 중인 세션에서는 `/mcp`로 상태·도구·로그인·노출 방식을 확인하고, 밖에서 설정을 바꿨으면 `/reload`한다. [MCP 사용법](https://github.com/earendil-works/pi/blob/v1.0.1/packages/coding-agent/docs/mcp.md)

HTTP 서버 설정 예시는 다음과 같다. URL은 실제 서버로 교체하고, 토큰은 환경변수에서 주입한다.

```json
{
  "mcpServers": {
    "docs": {
      "url": "https://example.com/mcp",
      "headers": { "Authorization": "Bearer ${DOCS_TOKEN}" },
      "exposure": "codemode",
      "description": "문서 검색과 읽기"
    }
  }
}
```

stdio와 streamable HTTP를 지원하며 legacy SSE는 지원하지 않는다. OAuth가 필요하면 `/mcp login docs` 또는 `pi mcp login docs`를 사용한다. [전송·인증](https://github.com/earendil-works/pi/blob/v1.0.1/packages/coding-agent/docs/mcp.md#authenticate-with-oauth)

### 도구 노출 방식

| exposure | 동작 |
|---|---|
| `codemode` | 기본값. JavaScript에서 도구를 찾고 호출 |
| `deferred` | `tool_search`로 발견한 뒤 모델에 직접 선언 |
| `direct` | 모델에 처음부터 도구 정의 제공 |
| `hidden` | 등록하지만 호출 불가 |

많은 도구가 있는 서버는 검색 후 필요한 것만 쓰게 할 수 있다. 소수 도구를 자주 사용하면 `direct`가 단순하다. v1.0.1에서는 프로젝트 `.pi/mcp.json`의 `{ "enabled": false }` 같은 entry로 전역 서버를 해당 프로젝트에서 끄거나 노출 방식을 바꿀 수 있다. [Exposure 및 project overrides](https://github.com/earendil-works/pi/blob/v1.0.1/packages/coding-agent/docs/mcp.md)

### codemode는 여러 도구를 한 스크립트로 조합한다

MCP 없이 사용하려면 다음처럼 활성화한다.

```json
{ "defaultTools": ["+codemode"] }
```

모델이 생성할 수 있는 스크립트 예:

```javascript
const results = await Promise.allSettled([
  tools.bash({ command: "git status --short" }),
  tools.bash({ command: "git diff --stat" })
]);
for (let i = 0; i < results.length; i++) {
  const result = results[i];
  text({ index: i, ...result });
}
```

JavaScript 자체는 QuickJS 안에서 실행되고 Node·파일시스템·네트워크 API가 없다. 외부 작업은 `tools`를 통해 수행한다. **도구 호출의 부작용은 스크립트 실패 시 자동 rollback되지 않는다.** 독립적인 읽기 작업은 병렬화하고, 의존하는 수정 작업은 순서대로 처리한다. [Codemode](https://github.com/earendil-works/pi/blob/v1.0.1/packages/coding-agent/docs/codemode.md)

## 8. 확장과 Pi 패키지

이벤트나 UI를 바꿔야 할 때만 TypeScript 확장을 만든다. Pi는 jiti로 로컬 TypeScript를 로드한다. 확장의 `pi.registerCommand`, `pi.registerTool`, `pi.on`이 각각 명령·도구·이벤트를 등록한다. 단순 반복 지침은 템플릿·스킬로 충분하다. [Extensions](https://github.com/earendil-works/pi/blob/v1.0.1/packages/coding-agent/docs/extensions.md)

```bash
pi -e ./my-extension.ts             # 이번 프로세스에서 시험
pi install ./my-pi-package -l       # 프로젝트 패키지
pi list
pi config
pi update --extensions
pi remove ./my-pi-package -l
```

npm package는 `pi install npm:<이름>@<검토한-버전>`으로 고정한다. `-e npm:...`는 설정에 영구 기록하지 않고 시험한다. Pi package는 스킬·프롬프트·확장·테마를 배포하는 단위이고, 일반 앱 라이브러리를 모두 Pi plugin으로 설치하는 것은 아니다. **Pi Durable은 앱에서 npm 의존성으로 사용하는 SDK**다. [Packages](https://github.com/earendil-works/pi/blob/v1.0.1/packages/coding-agent/docs/packages.md), [Durable](https://github.com/earendil-works/pi/blob/v1.0.1/packages/durable/README.md)

처음부터 서드파티 기능을 많이 설치할 필요는 없다. 내장 MCP·세션·도구로 부족한 부분을 확인한 뒤 별도 계획 모드나 서브에이전트 패키지를 검토한다. 이 가이드는 과거 패키지 버전 목록을 최신 추천으로 재사용하지 않는다.

## 9. CLI·JSON·RPC·SDK 자동화

| 방식 | 적합한 용도 | 완료 판정 |
|---|---|---|
| `pi -p` | 한 번 실행·텍스트 결과 | 프로세스 결과와 작업 자체 검증 |
| `pi --mode json` | 이벤트·도구 호출 수집 | 최종 이벤트·stopReason·도구 결과 |
| `pi --mode rpc` | 외부 앱에서 지속 제어 | 명령 응답 후 `agent_settled` |
| Coding Agent SDK | Node/TypeScript 앱 내 제어 | `session.prompt()` 및 실제 결과 |
| Pi Durable | 영속 task·conversation 서비스 | runtime·task 상태와 결과 |

```bash
git diff -- src/auth | pi --tools read,grep,find,ls -p \
  "이 diff의 실제 결함을 검토하고 파일 근거를 보고해줘."

pi --tools read,grep,find,ls --mode json \
  "README를 읽고 실행 절차를 설명해줘" > events.jsonl
```

주의할 완료 판정:

- print mode는 마지막 assistant의 `error`/`aborted`에 nonzero exit를 반환한다.
- JSON mode는 실패·중단 assistant 응답이 있어도 그 사실만으로 nonzero exit가 되지는 않는다. 이벤트를 검사해야 한다.
- RPC의 `success: true`는 명령 수락·처리를 의미한다. 코드 수정이나 테스트 성공을 뜻하지 않는다.
- `agent_end` 뒤에도 retry·compaction·queued work가 이어질 수 있다. 자동 작업 종료는 `agent_settled`로 확인한다.

따라서 배치 성공은 “프로세스 종료 + 에이전트 오류 확인 + 원하는 산출물/테스트 통과”를 함께 확인한다. [CLI integration](https://github.com/earendil-works/pi/blob/v1.0.1/packages/coding-agent/docs/cli-integration.md), [RPC](https://github.com/earendil-works/pi/blob/v1.0.1/packages/coding-agent/docs/rpc.md)

앱에 넣는 Coding Agent SDK의 최소 구조는 다음과 같다. 실제 실행은 provider 인증이 필요하며 이번 조사에서 모델 호출은 실행하지 않았다.

```bash
npm install --save-exact --ignore-scripts @earendil-works/pi-coding-agent@1.0.1
```

```javascript
// agent.mjs
import { createAgentSession } from "@earendil-works/pi-coding-agent";

const { session } = await createAgentSession();
try {
  await session.prompt("이 프로젝트의 실행 방법을 파일 근거와 함께 설명해줘.");
  console.log(session.getLastAssistantText());
} finally {
  session.dispose();
}
```

```bash
node agent.mjs
```

기본 SDK session은 현재 폴더의 리소스와 설정·인증을 사용한다. CLI와 달리 SDK factory가 MCP/codemode 내장 확장을 자동 추가하지는 않으므로, 필요하면 resource loader에 명시적으로 추가한다. [SDK](https://github.com/earendil-works/pi/blob/v1.0.1/packages/coding-agent/docs/sdk.md#codemode-mcp)

## 10. Pi Durable은 언제 필요한가?

일반 Pi는 사람이 터미널에서 작업하고, 중단 후 세션을 열어 이어가는 흐름에 적합하다. Durable은 영속 conversation·task·document 상태를 앱에서 관리하고 복구하는 별도 런타임이다. v1.0.1 번호를 쓰지만 **공식적으로 experimental이며 API 안정성을 보장하지 않는다.** [공식 발표](https://earendil.com/posts/pi-durable/), [v1.0.1 README](https://github.com/earendil-works/pi/blob/v1.0.1/packages/durable/README.md)

| 원하는 결과 | 선택 |
|---|---|
| PC에서 개발하고 다음 날 같은 대화를 이어가기 | `pi -c` |
| 외부 UI에서 기존 코딩 에이전트를 제어하기 | RPC 또는 Coding Agent SDK |
| 장기 task와 대화를 저장하고 프로세스 재시작 후 복구하기 | Pi Durable |
| 실패한 외부 작업의 중복 실행까지 방지하기 | Durable 설계에 더해 작업별 멱등성·결과 확인 |

설치, 저장소 선택, `Harness` lifecycle, task 복구, 도구 재실행 한계와 실행 예제는 [Pi Durable 상세 가이드](durable.md)를 따른다.

## 11. 추천 학습 순서와 문제 해결

### 추천 학습 순서

1. **첫 세션:** `/login`, `/model`, `/name`, 읽기 전용 코드 탐색.
2. **작은 수정:** 범위·회귀 테스트·완료 조건을 지정해 수정하고 diff 확인.
3. **작업 재개:** `pi -c`, `/tree`, `/compact` 사용.
4. **반복 작업:** `.pi/prompts/review.md`와 검증 스킬 작성.
5. **도구 연결:** 필요한 MCP 서버 하나만 연결하고 `/mcp`에서 확인.
6. **자동화:** print/JSON으로 시작하고 지속 제어가 필요하면 RPC/SDK.
7. **Durable 실습:** 모델 호출 없이 저장·재시작부터 확인한 뒤 LLM task 연결.

### 문제 해결

| 증상 | 확인 |
|---|---|
| 모델이 없음 | `/login`, `/model`, `pi --list-models`, endpoint 인증 |
| 프로젝트 설정이 무시됨 | `/trust`; 비대화형 실행의 `--approve`/`--no-approve` |
| MCP 도구가 안 보임 | `pi mcp list`, `/mcp`, enabled·exposure·연결 오류 |
| 검색 도구가 기본으로 안 보임 | `defaultTools`에 `+grep`, `+find`, `+ls` |
| 설정 제거가 즉시 반영 안 됨 | `/reload`의 도구 제거 제한 확인 후 프로세스 재시작 |
| 모델 컨텍스트가 길어짐 | `/compact`에 보존할 사실·남은 작업 지정 |
| 자동화가 너무 일찍 완료됨 | `agent_end` 대신 `agent_settled`, 마지막 stopReason 확인 |
| 세션 분기 뒤 파일은 예전 상태가 아님 | 대화 분기와 Git 파일 상태를 별도로 관리 |
| durable 재시작 시 데이터가 없음 | MemoryStorage 대신 영속 storage와 같은 경로 사용 |

모델 오류는 코드 검증 실패와 분리한다. 이 가이드의 소스 확인과 로컬 예제는 실제 provider·사내 endpoint·장비에서의 동작 검증을 대신하지 않는다.

## 참고 자료 (References)

최신 문서는 변경될 수 있으므로 위 설명의 근거는 v1.0.1 태그로 고정했다.

- [Pi v1.0.1 릴리스](https://github.com/earendil-works/pi/releases/tag/v1.0.1)
- [공식 문서 최신판](https://pi.dev/docs/latest)
- [v1.0.1 전체 문서](https://github.com/earendil-works/pi/tree/v1.0.1/packages/coding-agent/docs)
- [CLI](https://github.com/earendil-works/pi/blob/v1.0.1/packages/coding-agent/docs/cli.md)
- [MCP](https://github.com/earendil-works/pi/blob/v1.0.1/packages/coding-agent/docs/mcp.md)
- [보안·trust](https://github.com/earendil-works/pi/blob/v1.0.1/packages/coding-agent/docs/security.md)
- [Pi Durable 소스와 README](https://github.com/earendil-works/pi/tree/v1.0.1/packages/durable)

## 관련 문서

- [Pi Durable 상세 가이드](durable.md)
- [Codex CLI 실전 가이드](../codex/README.md)
- [개발 환경 인덱스](../README.md)
