---
tags: [reverse-engineering, rea, ghidra, mcp, agent]
level: intermediate
last_updated: 2026-10-24
document_type: learning
category_major: "리버스 엔지니어링"
category_middle: "도구"
category_minor: "REA 사용법"
note_kind: "사용법"
---

# REA + Ghidra로 분석하는 방법

> 에이전트에게 "대상 경로 + 좁은 질문 + 근거 요구 + 실행 범위"를 주면 REA가 Ghidra를 대신 돌리고 근거가 붙은 결론을 돌려준다. 사람이 할 일은 질문을 고르고 결론의 등급을 따지는 것이다.

## 왜 필요한가? (Why)

Ghidra를 직접 쓰면 프로젝트를 만들고, 파일을 가져오고, 자동 분석을 기다리고, 함수 목록과 디컴파일 창을 오가며 클릭해야 한다. 함수 수십 개를 따라가는 데 반나절이 걸린다.

REA는 이 손작업을 에이전트에게 넘긴다. 에이전트가 MCP로 REA 도구를 부르면 REA가 Ghidra를 헤드리스(Headless)로 실행해 결과를 돌려준다. 사람은 GUI를 열지 않고 질문만 한다. 또 REA는 모든 결과에 근거(Evidence)와 한계(limitations)를 붙인다. 에이전트가 근거 없이 지어내는 것을 막는 장치다.

REA는 Ghidra 전용 도구가 아니다. JavaScript, Electron, .NET, PyInstaller, 웹 페이지, HAR 캡처는 Ghidra 없이 분석한다. 사내 웹 앱 분석에 Ghidra가 필요한 경우는 거의 없다.

## 핵심 개념 (What)

### 구성 요소의 관계

```text
사람 ──질문──▶ 에이전트(Claude Code, Codex 등)
                  │ MCP 도구 호출
                  ▼
                REA MCP 서버 (로컬)
                  │
      ┌───────────┼──────────────┬───────────────┐
      ▼           ▼              ▼               ▼
   Ghidra      JS/Electron     .NET/PyInstaller  Chrome(CDP)
  (네이티브)   정적 분석         정적 분석         웹 페이지 관찰
```

- **REA 스킬(`reverse-engineer-anything`)**: 에이전트가 REA를 어떻게 쓸지 정한 지침. `npx rea-agents setup`이 MCP 등록과 함께 설치한다.
- **REA MCP 서버**: 실제 분석 도구 묶음. 에이전트는 이 서버의 도구를 부른다.
- **프로바이더(Provider)**: 네이티브 분석 엔진. Ghidra, Hopper, IDA 중 하나를 고른다. 대상 하나에는 프로바이더 하나가 묶인다.
- **세션(Session)**: `open_binary`로 대상을 열면 시작되고 `close_binary`로 닫으면 끝난다. Ghidra 세션은 임시 데이터베이스를 쓰므로, 세션 중에 붙인 함수 이름은 닫으면 사라진다.
- **Evidence**: 도구 결과 한 건의 기록. `evidence_id`로 가리킨다. 관찰, 추론, 미확인, 한계, 커버리지를 함께 담는다.

### REA를 쓰지 않아도 되는 경우

소스 저장소가 온전히 있으면 REA를 쓰지 않는다. 일반 코드 읽기로 충분하다. REA는 "배포된 결과물에서만 확인할 수 있는 사실"이 필요할 때 쓴다. 예를 들면 저장소 코드와 배포된 `.exe`가 같은지 확인할 때다.

## 어떻게 사용하는가? (How)

### 1단계: 준비 상태를 확인한다

설치는 이미 끝났다고 가정한다. 분석 전에 Ghidra 연결만 확인한다.

```bash
rea doctor --provider ghidra --json
```

`healthy`가 `false`면 아래 두 환경 변수부터 확인한다. 둘 다 REA를 띄우는 환경(에이전트의 MCP 등록)에 들어가 있어야 한다. 터미널에서만 `export`하면 이미 떠 있는 MCP 서버에는 반영되지 않는다.

```bash
export GHIDRA_INSTALL_DIR=/absolute/path/to/ghidra_12.1.4_PUBLIC
export JAVA_HOME=/absolute/path/to/jdk-21
```

REA는 Ghidra 12.1.x와 JDK 21 이상을 요구한다. 환경 변수를 바꾼 뒤에는 에이전트를 재시작한다.

> [!note] Windows에서 쓰는 경우
> Windows의 Ghidra 지원은 실험 단계(P0)이고 읽기 전용이다. 함수 이름 붙이기(`annotate_native_function`, `set_address_name`)는 Linux와 macOS에서만 된다. Windows에서는 이름을 분석 기록에 따로 적는다.

에이전트 안에서 REA 도구가 보이지 않으면 `rea doctor --client claude_code --json`으로 등록 상태를 확인한다. Codex라면 `--client codex`를 쓴다.

### 2단계: 에이전트에게 좋은 질문을 준다

REA 결과의 질은 질문의 질을 넘지 못한다. 프롬프트에 네 가지를 넣는다.

1. **대상**: 절대 경로. "내 앱"처럼 모호하게 쓰면 에이전트가 되묻는다.
2. **질문**: 관찰할 수 있는 시작점과 알고 싶은 결과. [작업 절차 2단계](./re-process.md#2-질문을-정한다)의 "좋은 질문" 형식.
3. **근거 요구**: 결론마다 Evidence ID와 등급(관찰, 추론, 미확인)을 달라고 한다.
4. **실행 범위**: 정적 분석만 할지, 실행해도 되는지. 말하지 않으면 에이전트가 실행 도구를 고를 수 있다.

예시 프롬프트:

```text
대상: /Users/me/re-work/recipe-tool/RecipeTool.exe (SHA-256은 분석 기록에 있음)
질문: "Export" 버튼을 누르면 만들어지는 CSV의 열 구성과 각 열의 계산 방법을 알고 싶다.
방법: 정적 분석만 한다. 프로그램을 실행하지 않는다. 프로바이더는 ghidra.
진입점 후보: 문자열 "Export completed", "csv".
결과 형식: 결론마다 Evidence ID와 관찰/추론/미확인 구분을 붙여라.
         확인하지 못한 부분은 어디까지 찾아봤는지 적어라.
```

나쁜 프롬프트는 "이 exe 분석해줘"다. 에이전트는 함수 목록부터 훑기 시작하고, 결과는 넓고 얕다.

### 3단계: 대상에 맞는 경로를 고른다

에이전트는 대상 형식을 보고 첫 도구를 고른다. 사람이 경로를 알고 있으면 질문이 더 정확해지고 낭비가 줄어든다.

| 대상 | 첫 도구 | Ghidra 필요 |
|---|---|---|
| 형식을 모르는 설치 폴더 | `inspect_desktop_application` | 아니요 |
| C, C++ `.exe`, `.dll`, ELF, Mach-O | `open_binary` → `analyze_function` 등 | 예 |
| .NET 어셈블리 | `inspect_managed_artifact` | 아니요 |
| .NET NativeAOT 빌드 | `open_binary` (Ghidra) | 예 |
| PyInstaller `.exe` | `open_binary` → `inspect_artifact` | 아니요 |
| 단독 `.pyc` 파일 | `inspect_python_pyc` | 아니요 |
| Electron `app.asar`, 압축 푼 JS 폴더 | `analyze_javascript_application` | 아니요 |
| 브라우저에 열린 웹 앱 | `list_browser_targets` → `inspect_web_page` | 아니요 |
| 저장해 둔 HAR 파일 | `inspect_web_network_capture` | 아니요 |
| Go, Rust 바이너리의 의존성 | `inspect_go_binary`, `inspect_rust_binary` | 아니요 |
| SQLite DB 파일 | `inspect_sqlite_database` | 아니요 |

명령과 옵션은 [REA 명령·도구 참조](./rea-reference.md)에 있다.

### 4단계: 대상별 작업 흐름을 따른다

#### 사내 웹 앱 (Vue, Nuxt, React 프런트엔드)

웹 앱은 Ghidra 없이 끝난다. 코드는 이미 브라우저에 있고, 대부분 소스 맵도 함께 배포된다.

1. 분석 전용 프로필로 Chrome을 원격 디버깅 포트와 함께 띄운다. 디버깅 포트는 반드시 `127.0.0.1`에만 연다.

	```bash
	"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
		--remote-debugging-port=9222 \
		--user-data-dir="$HOME/re-work/chrome-profile" \
		http://internal-app.example/
	```

2. 로그인하고 분석할 화면을 연다. 로그인은 사람이 한다.
3. 에이전트에게 페이지 구조와 번들을 분석하게 한다.

	```text
	http://127.0.0.1:9222 에 열린 internal-app 페이지를 REA로 분석해라.
	1) inspect_web_page로 스크립트와 리소스 목록을 보고
	2) analyze_web_bundle로 라우트, API 엔드포인트 후보, 소스 맵 유무를 정리해라.
	페이지를 조작하지 마라.
	```

4. 특정 동작의 요청을 보려면 관찰 창을 열고 사람이 버튼을 누른다. `observe_web_session`이 그동안의 탐색과 요청 메타데이터를 기록한다. REA는 연결하기 전에 일어난 네트워크 요청을 기록하지 않는다. 페이지가 처음 로드될 때의 요청이 필요하면 관찰을 시작한 뒤에 테스트 화면을 새로고침한다. 스크립트 목록은 연결 전에 로드된 것도 나온다.
5. 요청 본문까지 필요하면 DevTools **Network** 탭에서 **Save all as HAR**로 저장하고 `inspect_web_network_capture`로 읽는다.

> [!warning] HAR 파일에는 쿠키와 토큰이 들어 있다
> HAR 파일은 분석이 끝나면 지운다. 이 저장소나 공유 폴더에 올리지 않는다.

#### 네이티브 실행 파일 (Ghidra 사용)

1. 정찰부터 시킨다. 함수 하나를 바로 디컴파일하지 않는다.

	```text
	/abs/path/tool.exe 를 ghidra 프로바이더로 열고, binary_overview와 문자열 검색으로
	"Export" 관련 문자열과 그 문자열을 참조하는 함수를 찾아라. 아직 디컴파일하지 마라.
	```

2. 진입점 함수가 나오면 그 함수만 깊게 본다. `analyze_function`은 디컴파일, 호출 관계, 참조를 한 번에 돌려준다.
3. 경로를 따라가면서 함수에 이름을 붙이게 한다(Linux, macOS). 이름은 세션을 닫으면 사라지므로 분석 기록에도 적는다.
4. 디컴파일 결과가 이상하면 어셈블리를 확인하게 한다. `read_function_instructions`는 디컴파일러를 거치지 않는다.
5. 끝나기 전에 남길 Evidence를 `export_evidence_bundle`로 파일에 저장한다. `close_binary`는 세션의 Evidence와 미확인 항목 기록을 지운다. Evidence ID만 적어 두면 닫은 뒤에는 원문을 다시 볼 수 없다.
6. 저장한 뒤 `close_binary`로 세션을 닫는다.

큰 바이너리는 Ghidra 첫 가져오기(import)에 시간이 오래 걸린다. 시간 초과에는 주체가 둘 있으므로, 오류가 어느 쪽에서 났는지 먼저 구분한다.

- **REA의 Ghidra 시작 제한**: 기본 330초. `REA_GHIDRA_STARTUP_TIMEOUT_MS`를 늘리고 MCP 서버를 재시작한다.
- **에이전트(MCP 클라이언트)의 도구 호출 제한**: 클라이언트마다 다르고, REA 제한보다 먼저 끝날 수 있다. 클라이언트 설정에서 도구 호출 제한을 늘린다. Codex 등록은 `rea setup`이 기본으로 450초를 넣어 준다.

시간 초과 뒤에는 같은 호출을 바로 반복하지 않는다. `close_binary`의 정리 결과를 확인한 뒤 대상을 다시 연다. 메모리가 부족하면 `GHIDRA_HEADLESS_MAXMEM`을 올린다(기본 `2G`).

같은 대상을 여러 날에 걸쳐 분석한다면 CLI의 `--snapshot` 옵션으로 결과를 저장한다. 같은 바이트, 같은 질의는 저장된 결과를 다시 쓴다.

```bash
rea analyze /abs/path/tool.exe --provider ghidra --snapshot /abs/path/re-work/tool-snapshot.json --json
```

#### Python 도구 (PyInstaller로 묶은 `.exe`)

사내 Python 도구는 PyInstaller로 배포된 경우가 많다.

1. `inspect_artifact`로 내부 목록을 본다. Python 버전, 진입 스크립트, 포함된 모듈이 나온다.
2. 필요한 파일만 `extract_artifact`로 새 빈 폴더에 꺼낸다.
3. 꺼낸 `.pyc`는 `inspect_python_pyc`로 이름과 구조를 본다. REA는 `.pyc`를 `.py`로 되돌리지 않는다. 소스 복원이 필요하면 해당 Python 버전을 지원하는 별도 디컴파일러를 쓴다.

#### .NET 도구

`inspect_managed_artifact`로 시작한다. 클래스와 메서드 이름이 남아 있어서 네이티브보다 훨씬 쉽다. 특정 메서드만 보려면 CLI에서 `rea inspect-managed-members PATH --type Namespace.ClassName --json`을 쓴다.

### 5단계: 결과를 읽고 다음 질문을 정한다

REA 결과를 받으면 세 가지를 확인한다.

- **한계(limitations)와 미확인(unknowns)**: 무엇을 보지 못했는지. 간접 호출, 잘린 결과, 부분 커버리지가 여기에 적힌다.
- **커버리지(coverage)**: 결과가 전체인지 일부 페이지인지. 페이지가 끝났다고 해서 "없다"는 뜻은 아니다.
- **등급**: 에이전트의 설명 중 어느 문장이 관찰이고 어느 문장이 추론인지.

남은 질문이 있으면 그 질문 하나만 좁혀서 다시 묻는다. 같은 분석을 그대로 다시 돌리지 않는다.

### 효과적으로 쓰는 요령

- **요약부터 받는다.** 큰 대상은 `detail: "summary"`로 요약을 먼저 받고, 필요한 부분만 `inspect_analysis_view`로 펼친다. 에이전트 컨텍스트를 아낀다.
- **미해결 질문을 등록시킨다.** `record_unknown`으로 남은 질문을 세션에 기록하게 하면 다음 단계에서 빠뜨리지 않는다.
- **질문이 여러 개면 나눈다.** 서로 독립인 질문은 따로 묻는다. 한 프롬프트에 다섯 질문을 넣으면 다섯 개가 다 얕아진다.
- **안내 프롬프트를 쓴다.** REA MCP 서버는 `investigate_feature`, `compare_application_versions`, `verify_reconstruction` 같은 안내 프롬프트를 제공한다. 에이전트가 MCP 프롬프트를 지원하면 이것으로 시작해도 된다.
- **버전 비교에 쓴다.** "어제 배포본과 지난달 배포본이 뭐가 다른가"는 REA가 잘하는 질문이다. JS 앱은 `compare_application_versions`, Evidence 파일끼리는 `rea compare`를 쓴다.
- **재구현은 검증까지 시킨다.** 계산 로직을 옮길 때는 원본과 같은 입력을 넣어 비교하는 테스트까지 요구한다.
- **자주 업데이트한다.** REA는 변경이 잦다. 버그를 만나면 `rea update`부터 한다.

### 자주 만나는 문제

| 증상 | 원인 | 해결 |
|---|---|---|
| 에이전트에 REA 도구가 없다 | MCP 등록이 없거나 재시작을 안 했다 | `rea doctor --client <id> --json` 확인 후 에이전트 재시작 |
| Ghidra 프로바이더를 못 찾는다 | `GHIDRA_INSTALL_DIR`이 MCP 등록 환경에 없다 | MCP 등록의 환경 변수에 넣고 재시작 |
| JDK가 거부된다 | Rosetta 아래의 x86 JDK, 또는 JDK 21 미만 | 호스트 아키텍처와 같은 JDK 21 이상을 `JAVA_HOME`으로 지정 |
| 큰 바이너리가 시간 초과된다 | Ghidra 시작이 330초를 넘었거나, 에이전트의 도구 호출 제한이 먼저 끝났다 | 주체를 구분한 뒤 `REA_GHIDRA_STARTUP_TIMEOUT_MS` 또는 클라이언트 제한을 늘린다 |
| 세션을 닫았더니 Evidence가 없다 | `close_binary`가 세션 기록을 지운다 | 닫기 전에 `export_evidence_bundle`로 저장 |
| 결과가 잘려서 온다 | MCP 응답 크기 제한 | `inspect_analysis_view`로 페이지 단위로 받거나 `export_evidence_bundle`로 파일에 저장 |
| 붙인 함수 이름이 사라졌다 | Ghidra 세션은 임시 DB를 쓴다 | 이름을 분석 기록에 적어 둔다 |

## 참고 자료 (References)

- REA README: <https://github.com/morluto/rea>
- REA 설치와 Ghidra 설정: <https://github.com/morluto/rea/blob/main/docs/installation.md>
- REA 네이티브 분석: <https://github.com/morluto/rea/blob/main/docs/native-investigation.md>
- REA Ghidra 프로바이더 동작: <https://github.com/morluto/rea/blob/main/docs/ghidra-provider.md>
- REA 웹 페이지 관찰: <https://github.com/morluto/rea/blob/main/docs/browser-observation.md>
- REA 스킬 원문: <https://github.com/morluto/rea/blob/main/.agents/skills/reverse-engineer-anything/SKILL.md>

## 관련 문서

- [리버스 엔지니어링 작업 절차](./re-process.md)
- [REA 명령·도구 참조](./rea-reference.md)
- [분석 기록 양식](./templates/investigation-log.md)
