---
tags: [codex, harness, settings, sandbox, mcp]
level: intermediate
last_updated: 2026-04-28
status: complete
---

# 하네스 설정 맵

> 하네스 설정은 "모델 답변 스타일"이 아니라 "모델이 작업하는 환경"을 정하는 값이다.

## 1. 작업 범위 설정

가장 먼저 고정해야 하는 것은 작업 루트다.

```bash
codex -C web-development/python/flask/job-scheduler
```

| 설정 | 하네스 역할 | 권장 기준 |
|------|-------------|-----------|
| `-C`, `--cd` | 모델이 작업할 루트 지정 | 저장소 루트보다 모듈 폴더 우선 |
| `--add-dir` | 추가 읽기/쓰기 경로 허용 | 산출물 폴더처럼 필요한 경로만 추가 |
| `project_root_markers` | 프로젝트 루트 탐지 기준 | monorepo에서 루트 오인 방지 |
| `project_doc_max_bytes` | 프로젝트 지침 최대 로딩 크기 | 핵심 규칙이 잘리지 않게 조정 |
| `project_doc_fallback_filenames` | `AGENTS.md` 외 지침 파일명 | 팀 가이드 파일을 Codex 지침으로 연결 |

## 2. 지침과 컨텍스트 설정

지침은 하네스의 "운영 규칙"이다.

| 설정/파일 | 하네스 역할 | 권장 기준 |
|-----------|-------------|-----------|
| `AGENTS.md` | 저장소별 작업 규칙 | 스타일, 테스트, 보안 규칙을 짧고 명확히 작성 |
| nested `AGENTS.md` | 하위 폴더별 override | frontend/backend처럼 규칙이 다를 때 사용 |
| `developer_instructions` | 세션 공통 추가 지침 | 개인/팀 공통 운영 원칙에만 사용 |
| prompt 첫 메시지 | 작업별 계약 | 목표, 범위, 금지사항, 검증 명령을 포함 |
| conversation state | 이전 결정 보존 | 긴 작업은 중간 요약과 완료 조건 재확인 |

## 3. 모델 설정

모델 설정은 속도, 비용, 추론 깊이를 조절한다.

| 설정 | 하네스 역할 | 권장 기준 |
|------|-------------|-----------|
| `model` | 사용할 모델 선택 | 조직 표준 coding model을 기본값으로 고정 |
| `model_provider` | OpenAI/로컬/커스텀 provider 선택 | 민감 코드, 네트워크 정책에 맞게 분리 |
| `model_reasoning_effort` | 추론 깊이 | 일반 구현 `medium`, 어려운 분석 `high` |
| `review_model` | 리뷰용 모델 | 구현 모델과 분리 가능 |
| `service_tier` | 속도/비용 성향 | 긴급 작업과 일반 작업을 프로필로 분리 |

## 4. 권한과 샌드박스 설정

바이브 코딩을 제어하는 핵심이다.

| 설정 | 값 | 의미 |
|------|----|------|
| `sandbox_mode` | `read-only` | 읽기 위주. 분석, 설계, 리뷰에 적합 |
| `sandbox_mode` | `workspace-write` | 작업 폴더 안 수정 허용. 일반 개발 기본값 |
| `sandbox_mode` | `danger-full-access` | 제한 없음. 격리 환경이 아니면 피함 |
| `approval_policy` | `on-request` | 필요할 때 승인 요청. 대화형 기본값 |
| `approval_policy` | `untrusted` | 신뢰되지 않은 명령 승인 요청. 보수적 설정 |
| `approval_policy` | `never` | 승인 없이 진행. 읽기 전용 자동화에만 적합 |

`workspace-write`를 쓸 때는 쓰기 범위를 명시적으로 관리한다.

```toml
sandbox_mode = "workspace-write"
approval_policy = "on-request"

[sandbox_workspace_write]
network_access = false
writable_roots = ["/tmp"]
```

## 5. 네트워크와 환경 변수 설정

네트워크와 환경 변수는 보안 사고로 이어지기 쉬운 면이다.

| 설정 | 하네스 역할 | 권장 기준 |
|------|-------------|-----------|
| `sandbox_workspace_write.network_access` | 샌드박스 내부 네트워크 허용 | 기본 `false`, 문서 검색/설치 때만 승인 |
| `shell_environment_policy.inherit` | 하위 프로세스 환경 상속 | 기본은 최소 상속 |
| `shell_environment_policy.exclude` | 민감 env 제거 | `*_KEY`, `*_TOKEN`, `*_SECRET` 계열 차단 |
| `shell_environment_policy.include_only` | env allowlist | 배포/CI 자동화에서 유용 |
| `shell_environment_policy.set` | 고정 env 주입 | `CI=1` 같은 검증 환경 고정 |

## 6. 도구와 MCP 설정

도구가 많을수록 모델 선택지가 늘지만 비용, 지연, 오작동 가능성도 커진다.

| 설정 | 하네스 역할 | 권장 기준 |
|------|-------------|-----------|
| `mcp_servers.<id>.enabled` | MCP 서버 활성화 | 작업에 필요한 서버만 켬 |
| `mcp_servers.<id>.enabled_tools` | 도구 allowlist | 필요한 tool만 노출 |
| `mcp_servers.<id>.disabled_tools` | 도구 denylist | 위험하거나 불필요한 tool 차단 |
| `mcp_servers.<id>.tool_timeout_sec` | tool 실행 제한 | 긴 외부 API 호출 방지 |
| `mcp_servers.<id>.required` | 초기화 실패 처리 | 필수 도구만 `true` |
| app tool approval | 앱/커넥터 승인 정책 | destructive/open-world tool은 prompt 우선 |

예시:

```toml
[mcp_servers.github]
url = "https://api.githubcopilot.com/mcp/"
enabled = true
enabled_tools = ["repos/list", "issues/list", "pull_requests/get"]
tool_timeout_sec = 30
```

## 7. 검증과 평가 설정

검증 루프가 없으면 하네스가 아니라 단순 자동완성에 가깝다.

| 범주 | 예시 | 권장 기준 |
|------|------|-----------|
| 정적 검증 | `npm run lint`, `npm run typecheck`, `ruff`, `mypy` | 변경한 모듈 기준으로 실행 |
| 단위 테스트 | `pytest`, `npm test` | 비즈니스 로직 변경 시 추가/수정 |
| 스모크 테스트 | 앱 실행, 샘플 스크립트 실행 | runnable example은 최소 1회 실행 |
| 리뷰 | `codex review --uncommitted` | 큰 변경이나 리팩터링 후 실행 |
| 회귀 eval | 고정 프롬프트/샘플 입력 | LLM/RAG/agent 로직 변경 시 유지 |

## 8. 추천 프로필 예시

```toml
# ~/.codex/config.toml

[profiles.vibe-research]
sandbox_mode = "read-only"
approval_policy = "on-request"
model_reasoning_effort = "medium"

[profiles.vibe-code]
sandbox_mode = "workspace-write"
approval_policy = "on-request"
model_reasoning_effort = "medium"

[profiles.vibe-deep-fix]
sandbox_mode = "workspace-write"
approval_policy = "on-request"
model_reasoning_effort = "high"

[profiles.vibe-ci-readonly]
sandbox_mode = "read-only"
approval_policy = "never"
model_reasoning_effort = "low"
```

사용 예:

```bash
codex -p vibe-research -C Codes/python/opensearch_handler \
  "구조를 파악하고 수정 없이 위험한 부분만 정리해줘."

codex -p vibe-code -C Codes/python/opensearch_handler \
  "검색 결과 정렬 버그를 수정하고 관련 테스트를 실행해줘."
```

## 판단 기준

- 모델이 파일을 바꾸면 안 되는 단계인가? 그렇다면 `read-only`.
- 네트워크가 정말 필요한가? 아니라면 꺼둔다.
- 외부 도구가 20개 이상 노출되는가? allowlist로 줄인다.
- 완료 조건을 자동 검증할 수 있는가? 가능하면 명령으로 고정한다.
- 변경을 되돌릴 수 있는가? 큰 작업 전 `git status`와 브랜치를 확인한다.
