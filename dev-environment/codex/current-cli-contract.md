---
reviewed_on: 2026-10-04
review_status: partial
document_type: learning
tags: [codex, cli, verification]
category_major: "개발 환경"
category_middle: "코딩 에이전트 도구"
category_minor: "Codex CLI"
note_kind: "검토 기록"
classified_on: "2026-10-05"
---

# Codex CLI 버전별 실행 계약

## 목적과 작동 방식

Codex는 작업 지침과 파일 문맥을 모델에 보내고 모델이 요청한 도구를 로컬 권한 안에서 실행한다. 대화형 세션은 탐색·수정·검증을 반복하며, `exec`는 한 요청을 비대화형으로 처리한다. `review`는 변경을 검토한다. 결과 설명과 실제 파일·테스트 결과를 함께 확인한다.

이 폴더의 초기 예제는 0.111.0 기준이었다. **2026-10-04 이 PC에서는 `codex-cli 0.160.0`을 확인했다.** 최신 버전이라는 뜻은 아니다. 아래 계약은 로컬 `--help`와 같은 날 공식 문서에서 확인했다.

| 항목 | 0.160.0에서 확인한 조건 |
|---|---|
| `-C` | 작업 디렉터리를 정한다. 읽기 접근을 그 폴더로 격리하는 기능은 아니다 |
| `--add-dir` | 추가 쓰기 경로를 허용한다. 읽기 전용 allowlist가 아니다 |
| `-s` | `read-only`, `workspace-write`, `danger-full-access` 선택. OS·관리자 정책도 확인 |
| `-a` | 대화형 도움말은 `on-request`, `never`만 표시. `untrusted` 승인 정책은 공식적으로 폐기됨 |
| `--full-auto` | 로컬 도움말에 없다. 기존 조합의 의도는 `-s workspace-write -a on-request`로 표현 |
| `-p NAME` | `$CODEX_HOME/NAME.config.toml`을 기본 사용자 설정 위에 적용. 옛 `[profiles.NAME]` 예제와 구분 |
| `exec` | 로컬 도움말에 `-a`가 없다. 승인 설정은 `-c approval_policy=...`로 지정하고 해당 버전에서 확인 |
| `mcp-server` | 현재 로컬 상위 명령 목록에 없다. 과거 서버 실행 예제를 현재 설치에 복사하지 않는다 |

## 사용 방법

`/path/to/module`은 실제 작업 폴더로 교체한다. 다음은 에이전트를 실행하는 예제이며 이번 검토에서는 도움말만 실행했다.

```bash
codex --version
codex --help
codex exec --help
codex review --help
codex -C /path/to/module -s workspace-write -a on-request
codex exec -C /path/to/module -s read-only -c approval_policy=never \
  "README를 읽고 실행 진입점과 검증 방법을 설명해줘. 파일은 수정하지 마."
codex review --uncommitted
```

`never`는 승인 질문을 생략하는 정책이다. 샌드박스 권한을 확대하지 않으며 거부된 작업은 실패한다. `--ephemeral`은 세션 저장을 생략하지만 파일 쓰기·외부 API 기록까지 없애지는 않는다. `review --title`은 화면에 표시할 제목이며 PR 메모 파일을 저장하는 옵션이 아니다.

## 프로필 예제

기본 `CODEX_HOME`이 `~/.codex`일 때 다음을 **`~/.codex/vibe-research.config.toml`**에 저장하는 형태다. 실제 사용자 설정은 이번 작업에서 읽거나 수정하지 않았다.

```toml
sandbox_mode = "read-only"
approval_policy = "on-request"
model_reasoning_effort = "medium"
```

```bash
codex -p vibe-research -C /path/to/module
```

다른 용도는 파일을 별도로 만든다. 모델 ID·추론 수준·커넥터 도구 이름은 계정과 모델에 따라 다르므로 지원 목록을 먼저 확인한다. 설정 문법과 도움말 확인은 실제 모델·MCP 호출의 성공을 보장하지 않는다.

## 다음 읽기와 근거

기본 개념은 [실전 목차](./README.md), 설정별 설명은 [컨텍스트와 권한](./03_context_sandbox_profiles.md), 반복 작업의 운영 기준은 [하네스 목차](./harness/README.md)로 이어진다. 이 문서는 버전 계약을 모으고 각 가이드는 사용 맥락과 예제를 보존한다.

확인일: **2026-10-04**. 공식 페이지는 변경될 수 있다.

- [공식 CLI 명령](https://learn.chatgpt.com/docs/developer-commands?surface=cli)
- [공식 설정 참조](https://learn.chatgpt.com/docs/config-file/config-reference)
- 로컬 `codex --version`, `codex --help`, `codex exec --help`, `codex review --help`.
