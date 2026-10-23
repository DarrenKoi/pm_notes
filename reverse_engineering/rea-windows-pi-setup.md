---
tags: [reverse-engineering, rea, ghidra, windows, pi, mcp]
level: intermediate
last_updated: 2026-10-24
document_type: learning
category_major: "리버스 엔지니어링"
category_middle: "도구"
category_minor: "회사 환경 설정"
note_kind: "사용법"
---

# 회사 Windows + pi 환경에서 REA를 설정하고 쓰는 방법

> 회사 PC는 Windows이고 에이전트는 pi다. REA는 둘 다 지원하지만 Windows의 Ghidra 연동은 실험 단계이고 읽기 전용이다. 이 문서는 그 환경에서 막히는 지점과 해결 방법을 정리한다.

## 왜 필요한가? (Why)

REA 문서는 macOS와 Linux를 기준으로 쓰인 부분이 많다. Windows에서는 같은 명령이 다르게 동작하거나 아예 지원되지 않는 기능이 있다. 또 pi는 MCP 설정 파일을 엄격한 JSON으로 읽어서, 수동으로 고친 설정 파일 하나 때문에 REA 등록이 통째로 실패할 수 있다. 회사에서 시간을 버리지 않도록 제약을 먼저 알고 시작한다.

## 핵심 개념 (What)

### Windows에서 되는 것과 안 되는 것

| 기능 | Windows | 비고 |
|---|---|---|
| 네이티브 PE(`.exe`, `.dll`) Ghidra 분석 | 됨, 읽기 전용 | x86, x86-64, ARM64 PE. 26개 읽기 전용 작업 |
| Ghidra 함수 이름, 주석 붙이기 | 안 됨 | `annotate_native_function`, `set_address_name`은 Linux, macOS만 |
| ELF, Mach-O를 Ghidra로 분석 | 안 됨 | Windows는 PE와 raw 6502 이미지만 받는다 |
| .NET 어셈블리 | 됨 | Ghidra를 쓰지 않는 별도 분석기 |
| PyInstaller, ASAR, JS 정적 분석 | 됨 | 엔진 불필요 |
| 웹 페이지 관찰(Chrome CDP) | 됨 | Chrome을 디버깅 포트로 띄우면 된다 |
| HAR 파일 분석 | 됨 | |
| 프로세스 실행 관찰(process capture) | 안 됨 | Linux, macOS의 PTY가 필요하다 |
| 패킷 캡처, 펌웨어 분석 | 안 됨 | Linux 전용 도구에 의존한다 |

사내 도구 분석에서 실제로 아쉬운 것은 함수 이름 붙이기 하나다. 나머지 Windows 미지원 기능은 사내 웹 앱과 Windows 도구 분석에 거의 쓰이지 않는다.

### 분석 대상은 로컬 NTFS 디스크에 둔다

Windows의 Ghidra 연동은 **고정 로컬 NTFS 볼륨**의 파일만 받는다. 공유 폴더(네트워크 드라이브), USB 드라이브에 있는 파일은 거부된다. 임시 폴더도 로컬 NTFS여야 한다. 분석 전에 대상을 로컬 분석 폴더(예: `C:\re-work\`)로 복사한다. [작업 절차 3단계](./re-process.md#3-대상을-식별한다)의 "원본을 분석용 폴더에 복사한다"를 지키면 자연히 해결된다.

### pi가 REA를 찾는 방식

- `rea setup --client pi`는 사용자 설정 파일 `~/.pi/agent/mcp.json`(Windows에서는 `%USERPROFILE%\.pi\agent\mcp.json`)의 `mcpServers.rea`에 등록한다.
- 스킬은 `~/.agents/skills` 아래에 설치된다.
- `PI_CODING_AGENT_DIR`이 설정되어 있으면 그 폴더를 따른다.
- 프로젝트 폴더의 `.pi/mcp.json`에 같은 이름(`rea`)의 항목이 있으면 사용자 설정을 덮어쓴다. 사용자 등록은 정상인데 특정 프로젝트에서만 REA가 안 보이면 이 파일을 확인한다.
- pi는 `mcp.json`을 엄격한 JSON으로 읽는다. 주석, 끝 쉼표, BOM, 빈 파일은 모두 오류다. 메모장으로 저장하면 BOM이 붙을 수 있으므로 VS Code 같은 편집기로 "UTF-8(BOM 없음)"으로 저장한다.

## 어떻게 사용하는가? (How)

### 1. 사전 요구 사항을 확인한다

REA는 아래 프로그램을 설치하지 않는다. 미리 설치되어 있어야 한다.

- Windows 10 이상 x64
- Node.js 22.x(22.23.4 이상), 24.x(24.22.1 이상), 또는 26.12.1 이상과 npm. SQLite 분석(`inspect_sqlite_database`)까지 쓰려면 24.x나 26 이상을 고른다
- Ghidra 12.1.x(검증된 버전은 12.1.4)
- 64비트 JDK 21 이상
- REA 자체. 회사 PC에는 설치되어 있다(2026-10-24 확인). 새 PC라면 `npm install --global rea-agents`로 설치한다

### 2. 환경 변수를 정하고 진단한다

PowerShell에서 실행한다. 경로는 실제 설치 위치로 바꾼다.

```powershell
$env:GHIDRA_INSTALL_DIR = "C:\tools\ghidra_12.1.4_PUBLIC"
$env:JAVA_HOME = "C:\Program Files\Java\jdk-21"
$env:REA_ANALYSIS_PROVIDER = "ghidra"

rea doctor --provider ghidra --json
rea providers --json
```

`rea doctor`가 `runtime_unavailable`이나 native self-test 실패를 보고하면 Ghidra나 Java를 다시 설치해도 해결되지 않는다. 보고된 파일 시스템, 권한(DACL), 임시 폴더 문제를 먼저 확인한다. 대부분 임시 폴더가 로컬 NTFS가 아니거나 쓰기 권한이 없는 경우다.

### 3. pi에 등록한다

먼저 계획만 확인한다. 아무것도 바꾸지 않는다.

```powershell
rea setup --client pi --dry-run --json
```

출력에서 바뀔 파일 경로와 백업 위치를 확인한 뒤 적용한다.

```powershell
rea setup --client pi --yes --json
rea doctor --client pi --json
```

setup은 셸에 설정된 유효한 `GHIDRA_INSTALL_DIR`, `JAVA_HOME` 값을 등록에 보존한다. doctor 결과가 정상인데 pi에서 Ghidra를 못 찾으면 `mcp.json`의 `rea` 항목에 `env`가 들어갔는지 직접 본다. 들어가 있어야 할 모양은 다음과 같다.

```json
{
	"mcpServers": {
		"rea": {
			"type": "stdio",
			"command": "...setup이 넣은 값 그대로...",
			"args": ["...setup이 넣은 값 그대로..."],
			"env": {
				"REA_ANALYSIS_PROVIDER": "ghidra",
				"GHIDRA_INSTALL_DIR": "C:\\tools\\ghidra_12.1.4_PUBLIC",
				"JAVA_HOME": "C:\\Program Files\\Java\\jdk-21"
			}
		}
	}
}
```

JSON 안에서는 백슬래시를 두 번 쓴다. `command`와 `args`는 setup이 쓴 값을 바꾸지 않는다.

### 4. pi를 다시 불러온다

이미 열려 있는 pi 세션에서는 `/reload`를 실행한다. 새로 열어도 된다. 그다음 pi에게 REA 도구가 보이는지 묻는다.

```text
REA MCP 도구 목록이 보이는지 확인하고, binary_session 도구를 {}로 호출해서
ghidra 프로바이더가 사용 가능한지 알려줘.
```

### 5. 대상을 복사하고 분석한다

```powershell
New-Item -ItemType Directory -Force C:\re-work\recipe-tool | Out-Null
Copy-Item "\\shared\tools\RecipeTool\RecipeTool.exe" C:\re-work\recipe-tool\
Get-FileHash C:\re-work\recipe-tool\RecipeTool.exe -Algorithm SHA256
```

해시를 [분석 기록 양식](./templates/investigation-log.md)에 적고, [REA + Ghidra로 분석하는 방법](./rea-usage-guide.md)의 프롬프트 형식으로 pi에게 묻는다. 경로는 `C:\re-work\recipe-tool\RecipeTool.exe`처럼 절대 경로로 쓴다.

함수 이름을 붙일 수 없으므로, pi에게 "따라간 함수의 주소와 제안 이름을 표로 정리하라"고 요청해서 분석 기록 5절에 붙인다.

### 6. 웹 앱은 Windows Chrome으로 관찰한다

```powershell
& "C:\Program Files\Google\Chrome\Application\chrome.exe" `
	--remote-debugging-port=9222 `
	--user-data-dir="C:\re-work\chrome-profile" `
	http://internal-app.example/
```

분석 전용 프로필(`--user-data-dir`)을 쓴다. 평소 쓰는 프로필로 띄우면 개인 쿠키와 확장 프로그램이 분석 대상에 섞인다. 이후 순서는 [사용법 문서의 웹 앱 절](./rea-usage-guide.md#사내-웹-앱-vue-nuxt-react-프런트엔드)과 같다.

### 7. 큰 바이너리의 시간 초과를 대비한다

Ghidra의 첫 가져오기는 기본 330초까지 기다린다. 에이전트의 도구 호출 제한이 이보다 짧으면 REA보다 에이전트가 먼저 포기한다. 회사에서 pi로 REA를 쓸 때 지금까지 이 문제는 없었다(2026-10-24 기준). 아주 큰 대상에서 시간 초과가 반복되면 pi의 MCP 설정 문서에서 요청 제한 항목을 찾는다. 그동안에는 CLI로 먼저 분석하고 `--snapshot`으로 결과를 저장해 두는 방법이 있다.

```powershell
rea analyze C:\re-work\recipe-tool\RecipeTool.exe --provider ghidra --snapshot C:\re-work\recipe-tool\snapshot.json --json
```

### 사내 LLM과 쓸 때 주의할 점

회사의 pi는 사내 엔드포인트의 GLM-5.3 모델을 쓰고, 컨텍스트 창은 500K 토큰이다(2026-10-24 기준). 두 가지를 알아둔다.

- **도구 정의 크기**: REA는 도구가 170개 넘게 있다. 도구 정의를 매 요청에 모두 보내는 클라이언트라면 간략형 스키마로도 약 68k 토큰(cl100k 기준)을 차지한다는 측정이 있다. 500K 컨텍스트에서 이 크기는 약 14%이고, 실제로 pi에서 REA를 쓸 때 문제가 없었다. 분석이 길어져서 컨텍스트가 빨리 차면 `env`에 `"REA_MCP_INPUT_SCHEMA_PROFILE": "compact"`가 있는지 본다(현재 기본값이 `compact`이고, `full`로 바꿔 두었다면 되돌린다).
- **스키마 거부**: GLM-5.3에서는 보고된 문제가 없다. 참고로 일부 Kimi 게이트웨이가 REA의 이전 버전 입력 스키마를 "infinite recursion"으로 거부한 사례가 있다. 같은 오류가 나면 `rea update`로 REA를 최신으로 올리고 `compact` 프로필을 확인한다.

도구 결과는 사내 모델로만 간다. 외부 LLM API는 사내망에서 차단되어 있으므로, 외부 모델에 연결된 에이전트로 사내 도구를 분석하지 않는다.

## 참고 자료 (References)

- REA Windows Ghidra P0: <https://github.com/morluto/rea/blob/main/docs/windows-ghidra-p0.md>
- REA 클라이언트 설정(Pi 절): <https://github.com/morluto/rea/blob/main/docs/client-configuration.md#pi>
- REA 지원 에이전트 목록: <https://github.com/morluto/rea/blob/main/docs/installation.md#supported-agents>
- pi MCP 문서: <https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/mcp.md>

## 관련 문서

- [REA + Ghidra로 분석하는 방법](./rea-usage-guide.md)
- [REA 명령·도구 참조](./rea-reference.md)
- [리버스 엔지니어링 작업 절차](./re-process.md)
