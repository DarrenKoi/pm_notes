---
tags: [context-mode, claude-code, codex, security, data-flow]
level: intermediate
last_updated: 2026-11-18
---

# context-mode 데이터 흐름: plugin과 gateway CLI 비교

> plugin(2.2.1)은 데이터를 내 PC의 `~/.context-mode`에만 저장하고, gateway CLI(0.12.1)는 모든 모델 요청을 `gateway.contextmode.ai`로 보내도록 에이전트 설정을 바꿉니다.

## 왜 필요한가? (Why)

- 이름이 같은 두 제품이 데이터 경로가 전혀 다름. 하나는 로컬 도구, 하나는 호스팅 프록시
- 사내 기밀 문서를 다루는 PC에서는 "프롬프트와 tool output이 어느 서버를 지나는가"가 설치 판단 기준
- gateway CLI는 `npx` 한 번으로 Claude Code와 Codex의 API 주소를 바꾸므로, 제거와 확인 절차를 문서로 남길 필요가 있음

결론부터 적습니다. plugin 코드에는 사용 데이터를 vendor로 보내는 호출이 없습니다 [P1][P2]. gateway CLI를 연결하면 프롬프트, 응답, tool output 전부가 vendor 서버를 지나고 일부는 저장됩니다 [V2]. 기밀 문서를 다루는 PC에서는 plugin만 쓰고 gateway는 연결하지 않는 구성이 맞습니다. 다만 plugin도 아래 "주의할 점"의 조건이 붙습니다.

## 핵심 개념 (What)

### 비교표

| 항목 | plugin `context-mode` 2.2.1 | gateway CLI `@context-mode/cli` 0.12.1 |
|------|------|------|
| 정체 | Claude Code plugin (MCP server, hook, 로컬 SQLite FTS5) [P3] | 에이전트 설정을 gateway로 바꿔 쓰는 설치 도구 [C1] |
| 모델 요청 경로 | 변경 없음. 에이전트가 원래 API로 직접 전송 [V2] | 모든 모델 요청이 `https://gateway.contextmode.ai`를 거쳐 Anthropic 또는 OpenAI로 전달 [C2][V2] |
| vendor가 받는 데이터 | 없음 (telemetry 0건이라고 README에 명시, 코드 검색 결과와 일치) [P1][P2] | 프롬프트, 에이전트 응답, tool output 전체 [V2] |
| 저장 위치 | 내 PC `~/.context-mode/projects/<프로젝트 키>/context.db` [P4] | vendor 계정별 저장소 (Cloudflare) [V2] |
| 저장 내용 | 프롬프트, edit, 명령, tool error 기록과 색인된 tool output, 파일 내용 [P5][V3] | 오래된 turn과 tool output 전문(archive), 메시지와 응답의 블록당 앞 6,000자, tool call과 입력, Memory용 fact [V2] |
| 보존 기간 | 30일간 읽지 않은 색인 삭제, 프로젝트당 500 MB 상한, 14일 방치 프로젝트 삭제 [P6][V3] | archive는 30, 90, 365일 또는 무기한 중 선택 (기본 90일). Search용 기록은 삭제 요청 전까지 보존 [V2][V4] |
| 계정 | 불필요 [P7] | 필요 (브라우저 로그인, 기기 키 발급) [C3] |
| 네트워크 호출 | 사용자나 에이전트가 시작한 4종뿐 (아래 표) [P1] | 설치 도구 자체 호출 6종 + 이후 모든 모델 요청 [C4] |
| 라이선스 | Elastic-2.0 [R2] | MIT (설치 도구만. gateway 서버 코드는 저장소에서 찾지 못함) [R2][R1] |

### plugin의 네트워크 호출 전부

`server.mjs`, `hook.mjs`, `cli.mjs`, `run.mjs`, `search.mjs`에서 `fetch(`, `node:http`, `node:https`, `node:net`, `node:tls`, child process 호출과 URL 문자열을 검색한 결과입니다 [P2].

| 호출 | 목적지 | 보내는 데이터 | 실행 시점 | 출처 |
|------|------|------|------|------|
| 버전 확인 | `registry.npmjs.org/-/package/context-mode/dist-tags` (GET) | 요청 본문 없음 | `ctx_doctor` tool 또는 `context-mode doctor` 실행 시 | [P8] |
| 버전 확인 대체 경로 | `github.com/mksglu/context-mode.git` (`git ls-remote --tags`) | 없음 | npm 조회 실패 시 | [P8] |
| URL 수집 | 에이전트가 지정한 URL (GET, 사설 IP 차단) | URL 자체와 `user-agent: context-mode/<버전>` | `ctx_index`에 `url` 인자를 줄 때만 | [P9] |
| 디버그 리포트 | `github.com/mksglu/context-mode` issue (`gh issue create`) | secret을 가린 리포트 본문 | `context-mode` CLI에서 확인 질문에 yes를 누를 때만 (기본값 no) | [P10] |
| 설치와 업데이트 | npm 또는 git | 없음 | 사용자가 설치, 업데이트할 때 | [P1] |

- hook(`hook.mjs`)에는 HTTP 호출 없음. child process는 `git worktree list` 하나뿐 [P11]
- `gateway.contextmode.ai`, `console.contextmode.ai`, `x-cm-` 헤더, `ANTHROPIC_BASE_URL` 문자열은 plugin 5개 파일 어디에도 없음 [P2]
- `contextmode.ai` 문자열은 `ctx_stats`와 `ctx_doctor` 출력에 넣는 문서 링크 2건뿐이며 요청을 보내지 않음 [P12]

### "Cloudflare V8 edge sandbox", "knowledge base", "memory graph"는 어느 쪽인가

- plugin 2.2.1의 tool은 9개: `ctx_execute`, `ctx_batch`, `ctx_index`, `ctx_search`, `ctx_resume`, `ctx_stats`, `ctx_doctor`, `ctx_upgrade`, `ctx_purge` [P13]
- `context-mode-run-code`, `context-mode-run-local`, `context-mode-search`, `context-mode-index`, `context-mode-remember`, `context-mode-recall`, "edge sandbox", "memory graph", "knowledge base" 문자열은 plugin 파일에서 0건 [P2]
- 따라서 이 tool들은 plugin이 아니라 gateway가 요청에 끼워 넣는 것임. vendor 문서도 Recall, Search, Memory가 Gateway 서버에서 돈다고 적음 [P7][V5]
- 이 tool들의 데이터는 vendor 계정 저장소(Cloudflare)에 저장됨. Memory는 turn마다 앞 4,000자를 Cloudflare Workers AI의 Llama 3.3 70B로 보냄 [V2]
- `context-mode-run-code`가 "Cloudflare V8 edge sandbox"에서 돈다는 문구는 gateway에 연결된 세션의 tool 설명에서 본 것임. vendor 공개 문서에서는 "Public data in a sandbox" 수준까지만 확인함 [V6][O1]
- plugin의 `ctx_execute`는 이름만 sandbox이고 내 PC에서 내 계정 권한으로 실행됨. 파일, 네트워크, 환경 변수에 그대로 접근함 [P1]

### gateway CLI가 하는 일

- `setup`이 쓰는 설정 [C2][C5]
  - Claude Code `~/.claude/settings.json`의 `env`: `ANTHROPIC_BASE_URL`, `ENABLE_TOOL_SEARCH=true`, `ANTHROPIC_CUSTOM_HEADERS`에 `x-cm-device-key: <기기 키>` 한 줄
  - 같은 파일 최상위: `idleCompaction: false`
  - Codex `~/.codex/config.toml`: `model_provider = "context-mode"`, `[model_providers.context-mode]` (`base_url`, `wire_api`, `requires_openai_auth`, `http_headers`에 기기 키)
  - 쓰기 전에 원본을 `<파일>.cm-setup-bak-<시각>`으로 복사 [C6]
- 로그인 때 gateway로 가는 것 [C3][C7]
  - 에이전트의 자격 증명: `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `CLAUDE_CODE_OAUTH_TOKEN`, macOS Keychain의 Claude Code 로그인 토큰, 없으면 `~/.codex/auth.json`의 토큰. `Authorization` 헤더로 전송
  - Claude 계정 uuid (`~/.claude.json`의 `oauthAccount.accountUuid`)
  - 터미널 이름 label (기본값은 PC의 host name)
- step report (telemetry) [C8]
  - 실행마다 `POST /__cli-event` 1회
  - 내용: CLI 버전, 실행 방식, 명령, OS, CPU arch, Node 버전, shell 이름, 단계별 결과와 reason code, 무작위 id 2개 (`run_id`, `~/.context-mode/install-id`)
  - 경로, 파일 내용, 프롬프트, 이름, 이메일은 본문에 없음
  - 기기 키가 있으면 `x-cm-device-key` 헤더로 함께 감. README의 "키를 담지 않는다"는 본문 기준 설명임
  - 끄는 법: `CONTEXT_MODE_TELEMETRY=0` 또는 `DO_NOT_TRACK=1`
- 그 밖의 CLI 호출: `/__cli-status`, `/__cli-login`, `/__cli-logout`, `/__cli-disconnect`, `/__devices-issue`(CI 키), dry-run 확인 요청 1건(`/v1/messages` 또는 `/responses`, 본문은 "ping") [C4][C9]
- 로컬 상태 `~/.context-mode`: `credentials.json`(기기 키), `install-id`, `setup-state.json` [C10][C8][C5]

### vendor가 밝힌 저장, 보존, 삭제

- 요청이 gateway를 지나며, Recall과 Search를 위해 세션 기록을 검색 가능한 형태로 보관함 [V1]
- 저장 전에 알려진 secret 형식과 이메일 주소를 가림. 그 밖의 내용은 원문 그대로 저장됨 [V4][V5]
- archive는 보존 기간이 지나면 매일 밤 삭제. fact는 남음 [V4]
- "Forget this project"가 해당 프로젝트의 archive, 검색 색인, fact를 삭제함 [V4]
- 열람, 정정, 삭제 요청과 전체 privacy policy는 `sales@context-mode.com`으로 메일 요청. 운영사는 MKSF LTD (London) [V1]
- 모델 학습 사용 여부: 확인하지 못함. 공개 페이지에 관련 문장이 없음
- 이용 약관: 확인하지 못함. `/terms` 등 10개 경로가 404 [V7]

## 어떻게 사용하는가? (How)

### 1. plugin만 안전하게 쓰기

- plugin은 gateway와 계정 없이 완결됨. README가 "complete on its own"이라고 적었고, 코드에도 gateway 주소가 없음 [P7][P2]
- 주의할 점
  - `~/.context-mode/projects/*/context.db`에 프롬프트와 tool output, 파일 내용이 남음. 이 폴더가 클라우드 동기화나 백업 대상인지 확인 [P4][P5]
  - DB 암호화 여부: 확인하지 못함. 평문으로 간주하고 다룸
  - `ctx_index`에 `url`을 주면 그 URL로 GET 요청이 나감. 기밀이 URL에 들어가지 않게 함 [P9]
  - `ctx_execute` 코드는 내 권한으로 네트워크에 접근 가능. 에이전트의 permission prompt를 끄지 않음 [P1]
  - `ctx_doctor`는 npm registry에 버전 조회를 1회 보냄. 폐쇄망에서는 실패해도 동작에 지장 없음 [P8]
  - 기밀 세션이 끝나면 `ctx_purge`로 삭제 가능 (session, source, project 중 하나) [V3]
- 저장 위치를 바꾸려면 `CONTEXT_MODE_DATA_DIR` 환경 변수 사용 [P4]

```text
ctx_purge({ project: true })    # 이 프로젝트의 event, chunk, source 전부 삭제
```

plugin이 안전하다는 말은 "plugin이 데이터를 vendor로 보내지 않는다"는 뜻입니다. Claude Code가 모델 API로 보내는 프롬프트는 plugin과 무관하게 그대로 나갑니다. 사내 정책상 그 전송이 허용되는지는 별도로 판단해야 합니다.

### 2. `ANTHROPIC_BASE_URL`이 비어 있는지 확인

값을 화면에 찍지 않는 명령만 씁니다. `ANTHROPIC_CUSTOM_HEADERS` 줄에는 기기 키가 들어 있을 수 있습니다.

```bash
# 1) 지금 shell 환경: 둘 다 unset이어야 함
echo "BASE_URL=${ANTHROPIC_BASE_URL:-unset}"
[ -n "$ANTHROPIC_CUSTOM_HEADERS" ] && echo "CUSTOM_HEADERS=set" || echo "CUSTOM_HEADERS=unset"

# 2) Claude Code 설정: 건수가 0이어야 함 (-c는 줄 내용을 출력하지 않음)
grep -c 'ANTHROPIC_BASE_URL\|x-cm-device-key' ~/.claude/settings.json

# 3) Codex 설정: 건수가 0이어야 함
grep -c 'context-mode' ~/.codex/config.toml

# 4) shell profile에 남은 것이 없는지: 파일 이름만 출력
grep -l 'ANTHROPIC_BASE_URL\|CONTEXT_MODE_GATEWAY' ~/.zshrc ~/.zprofile ~/.zshenv 2>/dev/null
```

- 설정 파일을 고쳐도 이미 떠 있는 세션은 시작할 때 읽은 환경 변수를 계속 씀. 확인 후 Claude Code와 Codex 세션을 모두 종료하고 다시 시작
- 2026-11-18 이 PC에서 본 상태 [O1]
  - `~/.claude/settings.json`의 `env`와 `~/.codex/config.toml`에 gateway 항목 없음
  - 그런데 조사 당시 실행 중이던 Claude Code 세션의 환경 변수에는 `ANTHROPIC_BASE_URL`이 `gateway.contextmode.ai`로, `ANTHROPIC_CUSTOM_HEADERS`에 `x-cm-device-key`가 남아 있었음
  - 즉 그 세션은 재시작 전까지 gateway를 계속 지남

### 3. gateway CLI 완전 제거 순서

| 순서 | 할 일 | 이 PC의 상태 (2026-11-18) |
|------|------|------|
| 1 | 기기 키 폐기: `npx @context-mode/cli logout` (`POST /__cli-logout`로 키를 폐기한 뒤 로컬 키 삭제) [C1][C4][V8] | 로컬 `credentials.json`에 저장된 gateway 0개. 서버 쪽 폐기 여부는 확인하지 못함 |
| 2 | 설정 되돌리기: `npx @context-mode/cli disconnect` (`ANTHROPIC_BASE_URL`, `ENABLE_TOOL_SEARCH`, 기기 키 헤더 줄, `idleCompaction`, Codex provider 제거) [C11] | 설정 파일에서 이미 제거됨 |
| 3 | 백업 파일 삭제: `~/.claude/settings.json.cm-setup-bak-*`, `~/.codex/config.toml.cm-setup-bak-*` [C6] | 4개 남아 있음. 나중 시각의 2개(`...T20-11-59-...`)에 기기 키 헤더 포함 [O1] |
| 4 | `~/.context-mode`의 CLI 파일 삭제: `credentials.json`, `install-id`, `setup-state.json` [C10][C8][C5] | 3개 모두 남아 있음 |
| 5 | npx cache 삭제: `~/.npm/_npx/<해시>/node_modules/@context-mode/cli` | 이미 삭제됨 |
| 6 | shell profile과 환경 변수 확인 (위 2절) | shell profile은 확인하지 않음 |
| 7 | Claude Code, Codex 세션 전부 재시작 | 필요함 |
| 8 | 서버 쪽 데이터와 계정 삭제 요청 | 아래 참고 |

- 1번과 2번은 gateway에 요청을 보냄. `logout`과 `disconnect`는 기기 키와 함께 에이전트의 자격 증명도 `Authorization` 헤더로 보냄 [C4][C11]. 다시 보내고 싶지 않으면 console에서 기기를 폐기하는 방법을 찾아야 하는데, console에 그 기능이 있는지는 확인하지 못함
- 4번에서 `~/.context-mode` 폴더 전체를 지우면 안 됨. `projects/`는 plugin의 DB임 [P4]. plugin도 지울 때만 전체 삭제
- 3번의 백업 파일은 setup 이전 원본이므로, 지우기 전에 현재 설정 파일이 정상인지 먼저 확인
- 8번 서버 쪽 삭제
  - console(`https://console.contextmode.ai`)의 "Forget this project": 프로젝트 archive, 검색 색인, fact 삭제 [V4]
  - 계정과 전체 데이터 삭제: `sales@context-mode.com`으로 요청 [V1]. console에서 직접 계정을 지우는 기능은 확인하지 못함
  - Search용 기록은 삭제 요청 전까지 보존된다고 vendor가 적었으므로 요청을 빼먹지 않음 [V2]
- 권고: 로그인 과정에서 Claude Code 로그인 토큰이 gateway로 전송됐으므로 [C7], Claude Code에서 `/logout` 후 다시 `/login` 해서 토큰을 새로 받음. 이 조치가 기존 토큰을 무효화하는지는 확인하지 못함

```bash
# 남은 흔적 확인 (값은 출력하지 않음)
ls ~/.claude/settings.json.cm-setup-bak-* ~/.codex/config.toml.cm-setup-bak-* 2>/dev/null
grep -l 'x-cm-device-key' ~/.claude/settings.json* ~/.codex/config.toml* 2>/dev/null
ls ~/.context-mode
ls -d ~/.npm/_npx/*/node_modules/@context-mode 2>/dev/null
```

### 확인하지 못한 것

| 항목 | 부족한 것 |
|------|------|
| gateway의 모델 학습 사용 여부 | 공개 페이지에 문장 없음. 전체 privacy policy는 메일 요청으로만 제공 [V1] |
| 이용 약관 | 공개 URL을 찾지 못함 [V7] |
| gateway 서버의 실제 동작 | 서버 코드가 공개 저장소에 없음. vendor 문서 기준으로만 적음 [R1] |
| 계정 삭제의 처리 기한과 범위 | 메일 창구만 확인 [V1] |
| 이 PC 기기 키의 서버 쪽 폐기 여부 | 로컬 파일만 확인. gateway에 조회하지 않음 |
| console의 기기 폐기, 계정 삭제 기능 | console은 로그인이 필요해 열어 보지 않음 |
| `context-mode-run-code`의 실행 위치 상세 | tool 설명 문구만 확인. vendor 문서의 해당 절은 읽지 못함 [V6] |
| plugin DB 암호화 | 코드에서 확인하지 않음 |
| 설치본과 GitHub `v2.2.1` 소스의 일치 | `SHA256SUMS` 대조와 빌드 재현을 하지 않음 |
| plugin 번들의 모든 실행 경로 | 패턴 검색과 호출 지점 확인까지만 함. 전체 흐름 추적은 아님 |

## 참고 자료 (References)

plugin 설치본 경로는 `~/.claude/plugins/cache/context-mode/context-mode/2.2.1/`입니다. gateway CLI는 npx cache가 삭제되어 package 이름과 내부 경로로 적습니다.

### plugin 2.2.1 (설치본 소스)

- [P1] `README.md` 585~601행 "Security and privacy": zero telemetry, 네트워크 호출 4종 표, `ctx_execute` 실행 권한
- [P2] 5개 번들 전체 검색 결과: `fetch(` 실제 호출은 `server.mjs` 49438행 1건, `node:http`와 `node:https`, `node:net`, `node:tls` import는 `server.mjs` 53502~53506행뿐, `gateway.contextmode`, `x-cm-`, `ANTHROPIC_BASE_URL`, `context-mode-run-code`, `edge sandbox`, `memory graph`, `knowledge base`는 0건
- [P3] `package.json` (`"name": "context-mode"`, `"version": "2.2.1"`), `hooks/claude-code.json` (PreToolUse, PostToolUse, SessionStart, UserPromptSubmit)
- [P4] `server.mjs` 28249~28250행 (`PROJECTS_DIR_NAME`, `DB_FILE_NAME = "context.db"`), 28272~28273행 (`DATA_ROOT = "~/.context-mode"`, `CONTEXT_MODE_DATA_DIR`)
- [P5] `server.mjs` 29191~29236행 schema: `events`(kind: prompt, edit, cmd, toolerr, index, cost, decision, `body` 열), `sources`(kind: path, content, url), `chunks`(FTS5, `body` 열)
- [P6] `server.mjs` 28246~28248행 (`RETENTION_IDLE_DAYS = 30`, `RETENTION_SIZE_CAP_BYTES`, `PROJECT_DEAD_IDLE_DAYS = 14`)
- [P7] `README.md` 603~612행 "Context Mode Gateway": "it is complete on its own", "With the Gateway on, your requests pass through its server"
- [P8] `server.mjs` 28536행 (`D1_REGISTRY_URL`), 49436~49503행 (`npmLatest`, `git ls-remote` 대체 경로), 53011행과 53057행 (`runDoctorChecks`에서만 호출)
- [P9] `server.mjs` 53499~53701행 (`src/core/sandbox/fetch.ts`, 사설 IP 차단, `baseHeaders`의 user-agent), 57598행 (`ctx_index`의 `safeFetch(input.url)`가 유일한 호출)
- [P10] `cli.mjs` 11089~11119행 (`gh auth status`, 확인 질문 `initialValue: false`, `gh issue create`)
- [P11] `hook.mjs` 검색: `node:child_process` 1건, 호출은 `git worktree list --porcelain`
- [P12] `server.mjs` 28443행 부근 (`STATS_FOOTER_URL`), 29015행 부근 (`DOCTOR_DOCS_URL`)
- [P13] `server.mjs` 28059행 (`TOOL_ORDER`)

### gateway CLI (`@context-mode/cli@0.12.1`)

- [C1] `package.json` description, `README.md` 36~43행 명령 표
- [C2] `src/gateway.mjs` 10~13행 (`DEFAULT_GATEWAY`, `DEFAULT_CONSOLE`), `src/setup-claude.mjs` 1~21행과 65~75행
- [C3] `src/commands.mjs` 73~100행 (`login`), `README.md` 45~50행 "The device key"
- [C4] `src/commands.mjs` 14~37행 (`headers`, `/__cli-status`, `/__cli-logout`), 84행 (`/__cli-login`), `src/ci.mjs` 388행 (`/__devices-issue`), `src/disconnect.mjs` 157행, `src/telemetry.mjs` 99행, `src/verify.mjs` 46행
- [C5] `src/setup-codex.mjs` 9~12행과 44~61행, `src/setup-claude.mjs` 37~53행 (`idleCompaction`, `~/.context-mode/setup-state.json`)
- [C6] `src/fsio.mjs` 4행과 19행 (`BAK_TAG = ".cm-setup-bak-"`), `README.md` 310~311행
- [C7] `src/credential.mjs` 1~37행 (`findCredential`, Keychain `Claude Code-credentials`), 44~56행 (`findCodexCredential`), 59~64행 (`credHeaders`), 72~85행 (`claudeAccountUuid`, `agentMetadata`), `bin/context-mode-cli.mjs` 122행 (label 기본값 `hostname()`)
- [C8] `src/telemetry.mjs` 1~18행 주석, 25~35행, 56~63행, 77~105행, `README.md` 415~423행
- [C9] `src/verify.mjs` 1~14행 (dry-run 요청, 본문 "ping")
- [C10] `src/device-key.mjs` 1~19행 (`~/.context-mode/credentials.json`, mode 0600, 헤더 이름 `x-cm-device-key`)
- [C11] `src/disconnect.mjs` 1~28행 주석, 149~160행 (`tellGateway`)

### 공식 저장소와 npm registry

- [R1] https://github.com/mksglu/context-mode (최상위 목록에 gateway 서버 폴더 없음, `v2.2.1` tag의 `src/core/sandbox/fetch.ts` 존재 확인)
- [R2] https://registry.npmjs.org/context-mode (latest 2.2.1, Elastic-2.0), https://registry.npmjs.org/@context-mode%2Fcli (latest 0.12.1, MIT)

### vendor 문서 (2026-11-18 조회)

- [V1] https://contextmode.ai/privacy.md
- [V2] https://contextmode.ai/docs/engine/gateway.md "Side by side" 표
- [V3] https://contextmode.ai/docs/engine/privacy.md
- [V4] https://contextmode.ai/docs/recall.md "Privacy and retention"
- [V5] https://contextmode.ai/docs/search.md
- [V6] https://contextmode.ai/code-mode.md
- [V7] `https://contextmode.ai/` 아래 `terms`, `tos`, `terms-of-service`, `legal`, `legal/terms`, `legal/privacy`, `dpa`, `security`, `trust`, `subprocessors` 모두 404. `llms.txt`에도 terms 항목 없음
- [V8] https://contextmode.ai/docs/cli.md

### 이 PC에서 직접 본 것

- [O1] 2026-11-18 읽기 전용 확인: 설정 파일의 key 이름, 백업 파일 4개의 기기 키 헤더 포함 여부, `~/.context-mode` 파일 목록, 실행 중 세션의 환경 변수 host 이름, 세션에 노출된 tool 이름. 값은 출력하지 않음
