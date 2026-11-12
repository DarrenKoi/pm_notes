---
tags: [reverse-engineering, rea, ghidra, cli, reference]
level: intermediate
last_updated: 2026-11-11
document_type: learning
category_major: "리버스 엔지니어링"
category_middle: "도구"
category_minor: "REA 참조"
note_kind: "참조"
---

# REA 명령, 도구 참조

> REA CLI 명령, MCP 도구 이름, 환경 변수 목록. 2026-11-11에 npm 배포본 `rea-agents` 6.8.0(2026-10-25 배포)과 `morluto/rea` main 브랜치 문서로 확인했다. 버전에 따라 다를 수 있으므로 실제 목록은 `rea --help`와 `binary_session`으로 확인한다.

## 왜 필요한가? (Why)

에이전트에게 도구 이름을 짚어 주면 경로를 덜 헤맨다. 직접 CLI로 확인할 때도 명령을 찾는 데 쓴다.

## 핵심 개념 (What)

- CLI 명령은 MCP 도구 이름의 밑줄을 하이픈으로 바꾼 것이다. 예: `analyze_javascript_application` → `rea analyze-javascript-application`.
- 모든 경로는 절대 경로로 쓴다.
- `--json`을 붙이면 Evidence 레코드를 JSON으로 출력한다.
- 실행 중인 서버에서 쓸 수 있는 도구와 그 이유는 MCP `binary_session` 도구에 `{}`를 넘겨서 확인한다. `result.tool_availability`에 나온다.
- **배포본과 main은 다르다.** upstream 문서는 main 브랜치 코드를 설명한다. main은 매일 바뀌지만 npm 배포는 6.8.0에서 멈춰 있다(2026-11-11 확인). 설치된 버전은 `rea --version --json`, 최신 배포 버전은 `npm view rea-agents dist-tags.latest`로 본다. 이 문서에서 "main 전용"이라고 적은 항목은 다음 배포부터 쓸 수 있다.

## 어떻게 사용하는가? (How)

### 설치, 진단, 업데이트

| 명령 | 하는 일 |
|---|---|
| `npm install --global rea-agents` | REA를 전역 설치해 `rea` 명령을 PATH에 둔다. 이 표의 `rea ...` 명령은 전역 설치를 전제로 한다 |
| `npx rea-agents@latest setup` (전역 설치 후에는 `rea setup`) | 에이전트에 MCP 서버와 스킬을 등록한다. 변경 내용을 보여주고 승인을 받는다. 두 방법의 차이는 [회사 환경 설정](./rea-windows-pi-setup.md#rea를-설치하는-두-가지-방법)에 있다 |
| `rea setup --client codex --dry-run --json` | 특정 에이전트의 등록 계획만 보여준다. 아무것도 바꾸지 않는다 |
| `rea doctor --json` | 전체 환경을 진단한다. 설정을 바꾸지 않는다 |
| `rea doctor --provider ghidra --json` | Ghidra 연결만 진단한다 |
| `rea doctor --client claude_code --json` | 한 에이전트의 MCP 등록만 진단한다 |
| `rea doctor --skill --json` | 설치된 스킬만 진단한다 |
| `rea providers --json` | 쓸 수 있는 분석 엔진 목록 |
| `rea capabilities --json` | 기능 목록 |
| `rea update` | 전역 설치한 REA를 업데이트한다 |
| `rea --version --json` | 버전과 런타임 정보 |

`--client` 값: 회사에서는 `pi`와 `hermes`를 쓴다. 그 밖에 `claude_code`, `claude_desktop`, `codex`, `cursor`, `gemini_cli`, `windsurf`, `devin`, `opencode`, `antigravity`, `copilot_cli`, `commandcode`, `qwen_code`, `kimi`, `vscode`, `grok_build`, `omp`, `hermes`, `qoder`, `droid`, `trae`, `workbuddy`, `zcode`, `dsh`가 있다. `droid`, `dsh`, `kimi`, `qoder`, `trae`, `workbuddy`, `zcode`는 `rea-agents` 6.5.0 이상이 필요하다. Grok Bot은 setup이 아니라 수동으로 연결한다.

### 네이티브 분석 (Ghidra)

| 명령 | 하는 일 |
|---|---|
| `rea inspect PATH --provider ghidra --json` | 대상 개요 |
| `rea analyze PATH --provider ghidra --json` | 분석 실행 |
| `rea search PATH "text" --kind procedures --provider ghidra --json` | 함수 이름 검색. 다른 `--kind` 값은 `rea search --help`로 확인한다 |
| `rea function PATH NAME_OR_ADDR --provider ghidra --json` | 함수 하나의 종합 정보(dossier) |
| `rea decompile PATH 0x1000 --provider ghidra --json` | 의사 코드 |
| `rea instructions PATH NAME_OR_ADDR --provider ghidra --json` | 어셈블리 |
| `rea xrefs PATH 0x1000 --provider ghidra --json` | 상호 참조 |
| `rea trace PATH "text" --provider ghidra --json` | 시작점에서 정적 경로 추적 |
| `rea analyze PATH --provider ghidra --snapshot SNAP.json` | 결과를 스냅샷 파일에 저장하고 재사용 |
| `rea trace-native-values PATH PROCEDURE --provider ghidra` | 값의 출처와 흐름을 함수 경계를 넘어 추적 |
| `rea resolve-native-call-targets PATH CALL_SITE --provider ghidra` | 간접 호출의 대상 후보 |
| `rea inspect-native-instruction PATH ADDR --provider ghidra` | 명령어 하나의 상세 |
| `rea inspect-native-data-type PATH --type /MyStruct --provider ghidra` | 복원된 구조체, 타입 |
| `rea inspect-pe-resources PATH --json` | PE 리소스의 종류, ID, 위치. 내용은 디코딩하지 않는다. 엔진 불필요 |
| `rea inspect-native-exports PATH --json` | DLL 익스포트 목록. 엔진 불필요 |
| `rea inspect-pe-imports PATH --json` | 임포트 테이블과 지연 로드(delay-load) 테이블의 DLL, 함수 이름, 임포트 해시. 엔진 불필요 |
| `rea inspect-pe-load-config PATH --json` | TLS 콜백과 로드 설정(보안 쿠키, SafeSEH, CFG 테이블). 엔진 불필요 |
| `rea inspect-binary-hardening PATH --json` | 컴파일, 링크 단계 보호 설정을 속성별로 `enabled`, `disabled`, `unknown`으로 보여준다. 엔진 불필요. main 전용 |

자주 쓰는 MCP 도구: `open_binary`, `binary_overview`, `list_procedures`, `analyze_function`, `read_function_instructions`, `trace_native_values`, `match_native_functions`(두 빌드의 함수 대응), `inspect_pe_resources`, `inspect_native_exports`, `inspect_pe_imports`, `inspect_pe_load_config`, `close_binary`. `annotate_native_function`, `set_address_name`은 Linux, macOS에서만 된다. Windows에서 실제로 쓸 수 있는 도구는 `binary_session`의 `tool_availability`로 확인한다.

6.8.0부터 MCP `analyze_function`은 기본으로 요약만 돌려준다. 전체 기록(dossier)은 세션에 보관되고, 요약의 `normalized_result.parent_evidence_id`가 그 기록을 가리킨다. 필요한 부분은 `inspect_analysis_view`로 페이지 단위로 읽는다. 전체를 한 번에 받아야 할 때만 `detail: "complete"`를 넘긴다. `compare_functions`, `match_native_functions`, `build_call_path`는 요약 기록을 받지 않으므로 `{"kind": "retained-evidence", "evidence_id": "<parent_evidence_id>"}` 형태로 보관된 전체 기록을 넘긴다. CLI `rea function`은 예전처럼 전체를 출력한다.

### 정적 분석 (엔진 불필요)

| 대상 | CLI | MCP 도구 |
|---|---|---|
| 설치 폴더 | `rea inspect-desktop-application PATH --json` | `inspect_desktop_application` |
| JS, Electron 앱 | `rea analyze-javascript-application PATH --json` | `analyze_javascript_application` |
| 아카이브, 패키지(ASAR, ZIP, PyInstaller 등) | `rea inspect-artifact PATH --json` | `inspect_artifact` |
| 아카이브에서 꺼내기 | `rea extract-artifact PATH --json` | `extract_artifact` |
| .NET 어셈블리 | `rea inspect-managed-artifact PATH --json` | `inspect_managed_artifact` |
| .NET 멤버 | `rea inspect-managed-members PATH --type NS.Class --json` | `inspect_managed_members` |
| `.pyc` | `rea inspect-python-pyc PATH --json` | `inspect_python_pyc` |
| Go 바이너리 | `rea inspect-go-binary PATH --json` | `inspect_go_binary` |
| Rust 바이너리 | `rea inspect-rust-binary PATH --json` | `inspect_rust_binary` |
| PE 서명 인증서 | `rea inspect-pe-certificates PATH --json` | `inspect_pe_certificates` |
| SQLite DB(Node.js 24.15 이상 또는 26 이상 필요) | `rea inspect-sqlite-database PATH --json` | `inspect_sqlite_database` |
| 모델 파일(GGUF, safetensors) | `rea inspect-model-artifact PATH --json` | `inspect_model_artifact` |

### 웹 분석

| 명령 | 하는 일 |
|---|---|
| `rea list-browser-targets http://127.0.0.1:9222 --json` | 디버깅 포트에 열린 페이지 목록 |
| `rea inspect-web-page http://127.0.0.1:9222 TARGET_ID --observation-ms 1000 --json` | DOM, 스크립트, 리소스, 네트워크 메타데이터 |
| `rea analyze-web-bundle http://127.0.0.1:9222 TARGET_ID --json` | 번들의 라우트, 엔드포인트 후보, 소스 맵 |
| `rea observe-web-session http://127.0.0.1:9222 TARGET_ID --observation-ms 10000 --json` | 사람이 조작하는 동안의 탐색, 요청 기록 |
| `rea inspect-web-network-capture /abs/session.har har --json` | 저장한 HAR 파일 분석. URL을 다시 요청하지 않는다 |

에이전트가 직접 클릭하고 입력해야 하면 MCP 도구 `capture_browser_scenario`를 쓴다. 캡처한 스크립트를 JS 정적 분석으로 넘기려면 `export_web_scripts`를 쓴다.

### Evidence 다루기

| 도구 또는 명령 | 하는 일 |
|---|---|
| `inspect_analysis_view` | 저장된 Evidence의 요약, 한 모듈, 한 페이지를 본다. 분석을 다시 돌리지 않는다 |
| `get_evidence_bundle` | 세션의 Evidence를 조건으로 찾는다 |
| `export_evidence_bundle` | 세션 전체 Evidence를 파일로 저장한다 |
| `record_unknown` | 미해결 질문을 세션에 기록한다 |
| `rea evidence-import FILE` / `rea evidence-export IN OUT` | Evidence 파일 가져오기, 내보내기 |
| `rea compare LEFT.json RIGHT.json` | 두 Evidence 비교 |

### MCP 안내 프롬프트

| 프롬프트 | 용도 |
|---|---|
| `investigate_feature` | 기능 하나를 진입점부터 함수 근거까지 추적 |
| `compare_application_versions` | 두 배포본 비교 |
| `verify_reconstruction` | 재구현이 명세를 만족하는지 검증 |
| `trace_crash` | 크래시 증상과 정적 경로 연결 |
| `audit_residual_unknowns` | 남은 미확인 항목 점검 |
| `prepare_bounded_process_capture` | 실행 관찰 실험을 실행 전에 설계 |

### 환경 변수

REA를 띄우는 환경(MCP 등록)에 넣어야 한다. 바꾼 뒤에는 MCP 서버를 재시작한다.

| 변수 | 뜻 | 기본값 |
|---|---|---|
| `GHIDRA_INSTALL_DIR` | Ghidra 설치 경로. 없으면 `PATH`에서 `ghidraRun`을 찾는다 | - |
| `JAVA_HOME` | Ghidra가 쓸 JDK. 21 이상, 호스트 아키텍처와 같아야 한다 | `PATH`의 `java` |
| `GHIDRA_HEADLESS_MAXMEM` | Ghidra JVM 힙(`-Xmx`) | `2G` |
| `REA_GHIDRA_STARTUP_TIMEOUT_MS` | Ghidra 시작 대기 시간 | 330초 |
| `REA_GHIDRA_DECOMPILE_TIMEOUT_SECONDS` | 함수 하나의 디컴파일 제한 시간(초). 넘으면 그 디컴파일만 멈추고 세션은 유지된다. `analyze_function`은 어셈블리, 참조, 기본 블록만 돌려준다 | 없음(무제한) |
| `REA_GHIDRA_DISABLED_ANALYZERS` | 건너뛸 Ghidra 자동 분석기 이름. 쉼표로 구분하고 이름이 정확해야 한다. PE 리소스 분석기는 `WindowsResourceReference`. 끈 분석기가 만들던 정보는 결과에서 빠진다 | - |
| `REA_MCP_MAX_RESPONSE_BYTES` | MCP 응답 크기 상한. 10485760(10 MiB) 이상. 클라이언트의 수신 버퍼도 같은 값으로 늘려야 한다 | - |
| `REA_GHIDRA_LANGUAGE_ID`, `REA_GHIDRA_COMPILER_SPEC_ID` | 헤더 자동 감지 대신 쓸 언어와 컴파일러 지정 | - |
| `REA_MCP_INPUT_SCHEMA_PROFILE` | `full`이면 MCP 입력 스키마를 자세히 노출한다 | 간략형 |

main 전용 설정(6.8.0 배포본에는 없다. 2026-11-11 확인):

| 변수 | 뜻 |
|---|---|
| `REA_GHIDRA_PROJECT_PATH`, `REA_GHIDRA_PROGRAM_NAME` | 이미 만들어 둔 Ghidra 프로젝트(`.gpr`)를 읽기 전용 사본으로 재사용한다. 사람이 GUI에서 붙인 이름과 주석이 REA 결과에 나온다. Windows에서는 쓸 수 없다. upstream main 소스가 Windows에서 기존 프로젝트 열기를 막고, 이 변수를 지우라고 안내한다(6.8.0 배포본의 동작은 미확인) |
| `REA_GHIDRA_CHECKPOINT_DIR` | 분석이 끝난 Program을 `.gzf`로 저장하고 다음 실행에서 이름, 주석, 타입을 복원한다. Linux, macOS 전용 |
| `REA_MCP_TOOL_FAMILIES` | 노출할 MCP 도구 묶음(family)을 고른다. `rea setup --client pi --tool-families session,enhanced,application`으로 등록에 써 넣는다. 알려진 함수의 네이티브 분석은 `session,enhanced,application`, .NET은 `managed,session`이 upstream 권장 범위다 |

### 지원 조건

- Node.js 22.x(22.23.4 이상), 24.x(24.22.1 이상), 또는 26.12.1 이상과 npm.
- Ghidra 12.1.x와 64비트 JDK 21 이상. 검증된 조합은 Ghidra 12.1.4 + JDK 21.
- Ghidra는 Linux, macOS에서 읽기 전용 분석과 함수 이름, 주석 편집을 지원한다. Windows는 실험 단계이고 읽기 전용이다. Windows에서 열 수 있는 대상은 네이티브 x86, x86-64, ARMNT(Thumb-2), ARM64 PE다. ARM64EC와 ARM64X는 열 수 없다.
- Node.js 23과 25는 지원하지 않는다. REA 설치 프로그램은 Node.js를 올려 주지 않으므로 업데이트 전에 직접 맞춘다.
- 실행 파일의 바이트는 바꾸지 않는다. 세션 중 편집은 닫으면 사라진다.

## 참고 자료 (References)

- REA 6.8.0 릴리스 노트: <https://github.com/morluto/rea/releases/tag/rea-agents-6.8.0>
- REA PE 임포트: <https://github.com/morluto/rea/blob/main/docs/pe-imports.md>
- REA PE TLS, 로드 설정: <https://github.com/morluto/rea/blob/main/docs/pe-load-config.md>
- REA CLI와 Evidence: <https://github.com/morluto/rea/blob/main/docs/getting-started/cli.md>
- REA MCP 계약: <https://github.com/morluto/rea/blob/main/docs/reference/mcp-contracts.md>
- REA MCP 안내 프롬프트: <https://github.com/morluto/rea/blob/main/docs/getting-started/mcp-prompts.md>
- REA 설치: <https://github.com/morluto/rea/blob/main/docs/getting-started/installation.md>

## 관련 문서

- [REA + Ghidra로 분석하는 방법](./rea-usage-guide.md)
- [REA 실전 노하우](./rea-knowhow.md)
- [회사 Windows + pi 환경 설정](./rea-windows-pi-setup.md)
- [설치형 Windows 프로그램을 Ghidra로 분석하는 방법](./installed-program-analysis.md)
