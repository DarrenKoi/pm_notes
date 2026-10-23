---
tags: [reverse-engineering, windows, pe, ghidra, rea, installed-program]
level: intermediate
last_updated: 2026-10-24
document_type: learning
category_major: "리버스 엔지니어링"
category_middle: "대상별 분석"
category_minor: "설치형 Windows 프로그램"
note_kind: "사용법"
---

# 설치형 Windows 프로그램을 Ghidra로 분석하는 방법

> 설치 폴더 전체를 복사해서 먼저 런타임을 판별하고, 동료가 직접 짠 실행 파일만 골라서 Ghidra로 깊게 본다. 문자열과 임포트에서 출발해서 질문의 결과까지 함수 경로를 따라가고, 실행으로 확인한다.

## 왜 필요한가? (Why)

사내 설치형 프로그램은 실행 파일 하나가 아니다. `.exe` 하나, 직접 만든 DLL 몇 개, 외부 라이브러리 DLL 수십 개, 설정 파일, 로그 폴더, 로컬 DB가 함께 설치된다. 그중 동료가 직접 짠 코드는 일부다. 처음부터 메인 `.exe`를 Ghidra에 넣으면 외부 라이브러리 코드와 런타임 코드 사이에서 길을 잃는다.

또 "설치형 프로그램"이라는 겉모습은 같아도 속은 다르다. C# 프로그램이면 Ghidra가 필요 없고 메서드 이름이 그대로 보인다. Python을 PyInstaller로 묶은 것이면 압축을 풀어야 한다. Ghidra가 꼭 필요한 대상은 C, C++, Delphi 같은 네이티브 프로그램이다. 런타임을 먼저 판별하면 분석 시간이 몇 배 줄어든다.

## 핵심 개념 (What)

### 설치형 프로그램의 구성

| 위치 | 들어 있는 것 | 분석에서 쓰는 곳 |
|---|---|---|
| 설치 폴더(`C:\Program Files\...`) | `.exe`, DLL, 리소스, 기본 설정 | 코드 분석의 본체 |
| `%ProgramData%\...` | 모든 사용자 공용 설정, DB, 라이선스 | 설정 형식, 데이터 구조 |
| `%AppData%\...`, `%LocalAppData%\...` | 사용자별 설정, 캐시, 로그 | 실제 동작의 흔적 |
| 레지스트리 `HKLM\Software\...`, `HKCU\Software\...` | 설치 경로, 옵션 값 | 바이너리 안의 레지스트리 키 문자열과 맞춰 본다 |
| 로그 파일 | 실행 기록 | 로그 문구는 가장 좋은 진입점 문자열이다 |

### 런타임을 알아보는 단서

`inspect_desktop_application`이 런타임 후보를 알려주지만, 사람도 단서를 알아두면 결과를 검증할 수 있다.

| 런타임 | 설치 폴더나 바이너리에서 보이는 단서 | 분석 경로 |
|---|---|---|
| .NET(C#, VB.NET) | `mscoree.dll` 임포트, `*.dll.config`, `*.deps.json`, `System.*.dll` | `inspect_managed_artifact`. Ghidra 불필요 |
| .NET NativeAOT | .NET 단서가 있는데 CLI 메타데이터가 없다. 일반 apphost나 single-file 배포도 같은 모습이므로 이것만으로는 판별되지 않는다. 함께 배포된 관리 어셈블리와 배포 형태를 더 확인한다 | 판별되면 Ghidra |
| PyInstaller(Python) | `python3xx.dll`, `_MEI` 문자열, `base_library.zip` | `inspect_artifact`로 내부 목록 |
| Electron | `resources\app.asar`, `chrome_*.pak` | `analyze_javascript_application`. Ghidra 불필요 |
| Java | `.jar` 파일, 번들된 `jre\` 폴더 | Java 디컴파일러. REA의 Ghidra 경로가 아니다 |
| C++ MFC | `mfc140u.dll`, `vcruntime140.dll` 의존 | Ghidra |
| C++ Qt | `Qt5Core.dll`, `Qt6Core.dll`, `platforms\qwindows.dll` | Ghidra. Qt 함수 이름이 임포트에 남아 단서가 많다 |
| Delphi | `Borland`, `TObject`, `TForm` 문자열, 리소스의 폼 정의 | Ghidra |
| VB6 | `msvbvm60.dll` 의존 | 전용 도구가 필요하다. Ghidra만으로는 어렵다 |

### 임포트로 프로그램이 하는 일을 짐작한다

Windows 네이티브 프로그램은 운영체제 기능을 쓰려면 Windows API를 임포트해야 한다. 임포트 목록은 "이 프로그램이 할 수 있는 일"의 지도다. 임포트가 있다고 실제로 호출된다는 뜻은 아니므로 후보 단서로만 쓴다.

| 하는 일 | 대표 임포트 | 다음에 볼 것 |
|---|---|---|
| 파일 읽기와 쓰기 | `CreateFileW`, `ReadFile`, `WriteFile` | 경로 문자열, 호출 인수 |
| 설정(INI) | `GetPrivateProfileStringW`, `WritePrivateProfileStringW` | 섹션명, 키 이름 문자열 |
| 레지스트리 | `RegOpenKeyExW`, `RegQueryValueExW`, `RegSetValueExW` | 키 경로 문자열 |
| 네트워크 | `WSAStartup`, `connect`, `send`, `recv`(`ws2_32.dll`) | IP, 포트 상수 |
| HTTP | `WinHttpOpen`, `WinHttpSendRequest`, `InternetOpenW` | URL, 헤더 문자열 |
| 시리얼 통신 | `CreateFileW`("COM1" 등), `SetCommState`, `ReadFile` | 통신 속도 설정, 프레임 형식 |
| DB(ODBC) | `SQLDriverConnectW`, `SQLExecDirectW` | SQL 문자열, 연결 문자열 |
| 암호화, 비밀값 | `BCryptEncrypt`, `CryptProtectData` | 무엇을 보호하는지 |
| 다른 프로그램 실행 | `CreateProcessW`, `ShellExecuteW` | 실행 명령 문자열 |
| 대화 상자 UI | `DialogBoxParamW`, `MessageBoxW`, `SetDlgItemTextW` | PE 리소스의 대화 상자, 문자열 표 |
| COM 객체 | `CoCreateInstance` | CLSID 상수 |
| 숨은 임포트 | `LoadLibraryW`, `GetProcAddress` | 런타임에 불러오는 DLL과 함수 이름 문자열 |

`GetProcAddress`가 있으면 주의한다. 임포트 목록에 없는 함수를 실행 중에 찾아 쓴다는 뜻이다. 인수는 함수 이름 문자열일 수도 있고, 서수(ordinal) 번호나 실행 중에 계산한 값일 수도 있다. 이름 문자열이 보이면 숨은 기능이 바로 나오지만, 문자열이 없다고 해서 동적 조회가 없다고 판단하지 않는다.

### C++ 바이너리를 읽을 때 알아둘 것

- **x64 호출 규약**: MSVC x64는 첫 네 인수를 위치별로 레지스터에 넣는다. 정수와 포인터는 `rcx`, `rdx`, `r8`, `r9`, 실수는 같은 위치의 `xmm0`~`xmm3`을 쓴다. 예를 들어 두 번째 인수가 `double`이면 `rdx`가 아니라 `xmm1`에 들어간다. C++ 멤버 함수는 `this`가 `rcx`로 들어온다. 디컴파일 결과의 `param_1`은 대개 `this`다.
- **가상 함수 표(vtable)**: 가상 함수 호출은 `(**(code **)(*param_1 + 0x18))(param_1)` 같은 간접 호출로 보인다. 정적 분석으로 대상을 못 찾을 수 있으므로 `resolve_native_call_targets`로 후보를 확인한다.
- **RTTI 클래스 이름**: MSVC 바이너리에는 `.?AVRecipeManager@@` 같은 클래스 이름 문자열이 남아 있는 경우가 많다. 문자열 검색으로 클래스 목록을 얻을 수 있다.
- **PDB 경로 문자열**: 릴리스 빌드에도 `C:\...\Release\RecipeTool.pdb` 같은 경로가 남아 있을 수 있다. 원래 프로젝트 이름과 폴더 구조의 단서다.
- **정적으로 링크된 라이브러리**: OpenSSL, zlib, SQLite 같은 라이브러리가 바이너리 안에 들어 있으면 함수 수가 크게 늘어난다. `OpenSSL 1.1.1` 같은 버전 문자열로 알아보고 그 코드는 읽지 않는다. `SQLite format 3`은 라이브러리 버전이 아니라 SQLite DB 파일의 헤더 문자열이다. 이 문자열은 SQLite 파일을 다룬다는 단서일 뿐이다.
- **패킹(Packing)**: UPX 같은 패커로 압축된 바이너리는 코드가 실행 중에 풀린다. 섹션 이름(`UPX0`), 엔트로피가 아주 높은 구간, 임포트가 거의 없는 것이 징후다. 사내 도구에서는 드물지만, 보이면 정적 분석이 거의 막힌다.

## 어떻게 사용하는가? (How)

### 1. 설치 폴더를 통째로 복사한다

REA는 MSI, Inno Setup, NSIS 같은 설치 파일 자체를 풀지 않는다. 테스트 PC나 VM에 설치한 뒤 설치된 폴더를 분석한다. 이미 설치된 PC가 있으면 그 폴더를 복사한다.

```powershell
New-Item -ItemType Directory -Force C:\re-work\recipe-tool | Out-Null
Copy-Item -Recurse "C:\Program Files\RecipeTool" C:\re-work\recipe-tool\install
Get-ChildItem -Recurse C:\re-work\recipe-tool\install -Include *.exe,*.dll |
	Get-FileHash -Algorithm SHA256 |
	Export-Csv C:\re-work\recipe-tool\hashes.csv -NoTypeInformation
```

`%ProgramData%`와 `%AppData%`에 프로그램 폴더가 있으면 같이 복사한다. 운영 데이터가 들어 있을 수 있으므로 분석이 끝나면 지운다.

Windows의 Ghidra 연동은 로컬 NTFS 디스크의 파일만 받는다. 공유 폴더에서 바로 분석하지 않는다. 자세한 내용은 [회사 Windows + pi 환경 설정](./rea-windows-pi-setup.md)에 있다.

### 2. 런타임을 판별한다

pi에게 설치 폴더 전체를 먼저 보게 한다.

```text
C:\re-work\recipe-tool\install 를 REA inspect_desktop_application으로 분석해라.
1) 런타임 후보와 근거
2) 실행 파일과 DLL을 "직접 만든 것으로 보이는 것"과 "외부 라이브러리"로 나눈 표
3) 설정 파일, DB 파일, 로그 파일 목록
4) REA가 제안한 다음 도구(next_probes)
를 정리해라. 아직 Ghidra를 열지 마라.
```

직접 만든 것과 외부 것을 나눌 때는 아래를 근거로 쓴다.

- **서명 인증서**: `inspect_pe_certificates`로 서명자 필드를 본다. Microsoft, Qt 같은 외부 업체 이름이 있으면 외부 것일 가능성이 높다. REA는 서명이 유효한지 검증하지 않으므로 이것은 후보 단서다. 확정이 필요하면 Windows의 Authenticode 검증(`Get-AuthenticodeSignature`)을 따로 한다. 사내 도구는 대개 서명이 없다.
- **이름과 버전 정보**: `Qt5Core.dll`, `libcrypto-1_1-x64.dll` 같은 이름은 외부 라이브러리다.
- **익스포트**: DLL은 `inspect_native_exports`로 내보내는 함수를 본다. 사내 DLL은 `CalcOffset`, `LoadRecipe`처럼 업무 용어가 들어간 이름을 내보내는 경우가 많다.

### 3. 런타임에 따라 경로를 고른다

- .NET이면 [사용법 문서의 .NET 절](./rea-usage-guide.md#net-도구)로 간다. 이 문서의 나머지는 필요 없다.
- PyInstaller면 [사용법 문서의 Python 절](./rea-usage-guide.md#python-도구-pyinstaller로-묶은-exe)로 간다.
- Electron이면 `resources\app.asar`를 `analyze_javascript_application`에 넘긴다.
- 네이티브(C, C++, Delphi)면 아래 4단계부터 이어서 한다.

### 4. 정찰한다: 임포트, 문자열, 리소스

Ghidra로 함수를 읽기 전에 겉을 먼저 본다.

```text
C:\re-work\recipe-tool\install\RecipeTool.exe 를 ghidra 프로바이더로 열어라.
1) binary_overview로 아키텍처(x86/x64), 컴파일러 단서, 섹션을 보고
2) 임포트를 기능별(파일, 레지스트리, 네트워크, 시리얼, DB, UI, 동적 로드)로 묶고
3) 문자열 중 로그 문구, 에러 문구, 파일 경로, 레지스트리 키, SQL, URL을 분류하고
4) inspect_pe_resources로 대화 상자, 메뉴, 문자열 표 리소스가 있는지와 그 ID를 봐라.
디컴파일은 아직 하지 마라. 질문 "Export 버튼의 CSV 형식"에 쓸 진입점 후보 3~5개를 제안해라.
```

대화 상자, 메뉴, 문자열 표 리소스에는 화면 문구와 컨트롤 ID가 들어 있다. 사용자가 보는 버튼 이름에서 출발하면 그 버튼을 처리하는 코드로 가는 길이 짧다. `inspect_pe_resources`는 엔진 없이 리소스의 종류, ID, 위치를 알려주지만 내용은 디코딩하지 않는다. 화면 문구를 읽으려면 별도 리소스 도구로 연다. 회사에서는 오픈 소스인 RisohEditor(GPL-3.0)나 PE-bear(GPL-2.0)를 쓴다. 아래 [보조 도구 표](#회사-pc에-보조-도구를-들이는-기준)를 본다. 이렇게 얻은 컨트롤 ID는 상수 검색으로 버튼 처리 코드를 찾는 데 쓴다.

### 5. 진입점에서 경로를 따라간다

진입점 문자열 하나를 골라 질문의 결과까지 따라간다.

```text
문자열 "Export completed"를 참조하는 함수를 찾아 analyze_function으로 분석해라.
그 함수의 호출자(caller)를 따라 올라가서 버튼 처리 함수를 찾고,
호출 대상(callee)을 따라 내려가서 파일을 쓰는 함수(WriteFile 또는 fprintf 계열)까지 경로를 정리해라.
경로의 각 함수마다 주소, 제안 이름, 하는 일 한 문장, Evidence ID를 표로 줘라.
간접 호출 때문에 끊긴 곳은 따로 표시해라.
```

이 단계에서 쓰는 도구는 아래와 같다.

| 하고 싶은 것 | 도구 |
|---|---|
| 함수 하나의 디컴파일, 호출 관계, 참조를 한 번에 | `analyze_function` |
| 디컴파일러를 거치지 않은 어셈블리 | `read_function_instructions` |
| 값이 어디서 와서 어디로 가는지(계산식 추적) | `trace_native_values` |
| 간접 호출의 대상 후보 | `resolve_native_call_targets` |
| 명령어 하나의 상세 | `inspect_native_instruction` |
| 복원된 구조체, 타입 | `inspect_native_data_type` |

계산 로직이 질문이면 `trace_native_values`가 특히 쓸모 있다. 결과값 하나를 정하고, 그 값이 어느 입력과 상수에서 어떤 연산을 거쳐 만들어지는지 함수 경계를 넘어 추적한다. 메모리 별칭(alias)이나 전역 상태를 거치는 값은 미확인으로 남는다. 그 부분은 어셈블리를 직접 보거나 실행해서 확인한다.

Windows에서는 Ghidra에 함수 이름을 붙일 수 없다. pi에게 매번 "지금까지 붙인 제안 이름 표"를 갱신하게 하고, 그 표를 [분석 기록](./templates/investigation-log.md) 5절에 붙인다.

### 6. 설정과 데이터 파일을 함께 읽는다

코드만 보면 동작의 절반만 보인다. 설정 값과 데이터 형식이 동작을 바꾸기 때문이다.

- INI, XML, JSON 설정 파일은 직접 읽는다. 키 이름을 바이너리 문자열과 맞춰서 각 설정을 읽는 함수를 찾는다.
- SQLite DB는 `inspect_sqlite_database`로 테이블과 컬럼을 본다. 프로그램이 쓰고 있는 DB를 그냥 복사하면 내용이 어긋나거나 `-wal` 파일의 내용이 빠질 수 있다. 프로그램을 종료한 뒤 `.db`와 `-wal` 파일을 함께 복사하거나, 프로그램의 백업 기능으로 받은 파일을 쓴다. 이 도구는 REA를 Node.js 24.15 이상 또는 26 이상으로 실행해야 한다. Node.js 22에서는 쓸 수 없다.
- 바이너리 안의 레지스트리 키 문자열을 찾았으면, 테스트 PC의 레지스트리에서 실제 값을 확인한다.

### 7. 실행해서 확인한다

Windows에서는 REA의 프로세스 관찰 기능을 쓸 수 없다. 테스트 PC나 VM에서 보조 도구로 확인한다. 실행 전에 [작업 절차 1단계의 경고](./re-process.md#1-범위와-허가를-확인한다)대로 운영망과 운영 계정을 분리한다. 도구를 고르는 기준은 아래 "회사 PC에 보조 도구를 들이는 기준"에 있다.

| 확인하고 싶은 것 | 회사에서 쓸 도구 | 방법 |
|---|---|---|
| 어떤 파일, 레지스트리를 건드리는가 | System Informer 4.0 이상(MIT), 또는 Fibratus(Apache-2.0) | 프로그램을 실행하기 전에 기록을 시작한다. 프로세스 트리로 자식 프로세스와 함께 동작하는 서비스까지 확인한 뒤 범위를 정한다. 메인 프로세스 하나만 보면 자식 프로세스가 한 일을 놓친다 |
| 어떤 DLL을 불러오는가 | System Informer | 프로세스의 **Modules** 탭. 설치 없이 보려면 PowerShell `(Get-Process RecipeTool).Modules` |
| 어디로 통신하는가 | System Informer의 **Network** 탭, 또는 기본 제공 `netstat -ano`, 리소스 모니터 | 연결과 소유 프로세스를 본다. 패킷 내용까지는 보지 않는다 |
| 특정 함수에서 값이 무엇인가 | x64dbg(GPL-3.0 기반) | Ghidra에서 찾은 주소에 중단점을 건다 |
| 결과 파일이 가설과 같은가 | 기본 제공 `fc.exe /b`, PowerShell `Compare-Object` | 같은 입력으로 실행해서 만들어진 파일을 비교한다 |

디버거에서 주소를 맞출 때는 기준 주소(image base)를 확인한다. Ghidra는 PE 헤더의 기준 주소로 주소를 보여주지만, 실행 중에는 ASLR 때문에 다른 주소에 올라갈 수 있다. "Ghidra 주소 - Ghidra 기준 주소 + 실행 중 기준 주소"로 바꾼다.

### 8. 두 버전을 비교한다

"새 버전에서 무엇이 바뀌었나"는 설치형 프로그램에서 자주 나오는 질문이다.

```text
C:\re-work\v1\RecipeTool.exe 와 C:\re-work\v2\RecipeTool.exe 를 비교해라.
1) 각각 ghidra로 열고 list_procedures를 detail summary로 받아라.
2) 각 세션을 닫기 전에 export_evidence_bundle로 저장해라.
3) 두 번들을 다시 불러와 match_native_functions로 함수 대응을 만들고,
   대응된 함수와 대응되지 않은 함수를 후보 목록으로 정리해라.
4) 질문과 관련된 함수만 골라 양쪽 analyze_function 결과를 비교해서 실제로 바뀌었는지 확인해라.
```

`close_binary`는 세션 기록을 지운다. 첫 번째 대상을 닫기 전에 반드시 Evidence를 저장한다. 함수 매칭 결과는 후보일 뿐이다. 이름이 같다고 코드가 같다는 뜻은 아니고, 대응되지 않은 함수가 추가되거나 삭제되었다는 증거도 아니다. 변경 여부는 고른 함수의 본문을 양쪽에서 비교해서 확정한다.

### 9. 재구현한다면 비교 테스트까지 만든다

계산 로직을 Python 같은 언어로 옮길 때는 원본 프로그램과 같은 입력을 넣어 결과를 비교하는 테스트를 함께 만든다. 정상 범위만 넣지 말고 0, 음수, 경계값, 빈 입력을 넣는다. 반올림, 정수 나눗셈, 부동소수점 정밀도 차이에서 원본과 어긋나는 경우가 가장 많다.

## 회사 PC에 보조 도구를 들이는 기준

회사 PC에는 설치 금지 목록이 있다. 도구를 들이기 전에 두 가지를 차례로 확인한다.

1. **설치 금지 목록에 없는가.** 목록에 있으면 무료든 오픈 소스든 설치하지 않는다.
2. **회사에서 써도 되는 라이선스인가.** 무료 도구라도 "개인 사용만 무료"이고 회사 사용에는 유료 라이선스를 요구하는 경우가 많다. 공식 사이트의 라이선스 조항을 직접 확인한다.

OSI가 승인한 오픈 소스 라이선스(MIT, Apache-2.0, GPL 등)는 회사에서 쓰는 것 자체를 막지 않으므로 2번이 가장 쉽게 통과한다. Ghidra(Apache-2.0)도 이 기준으로 설치했다. 그래서 보조 도구는 오픈 소스부터 고른다. 무료이지만 오픈 소스가 아닌 도구는 2번을 따로 확인해야 한다.

| 용도 | 오픈 소스 도구(라이선스) | 비고 |
|---|---|---|
| 파일, 레지스트리 접근 추적 | System Informer 4.0 이상(MIT), Fibratus(Apache-2.0) | System Informer는 드라이버를 쓰는 기능이 있다. Fibratus는 ETW 기반이다 |
| 프로세스, DLL, 네트워크 연결 확인 | System Informer(MIT) | 구 Process Hacker |
| 디버거 | x64dbg(GPL-3.0 기반, 플러그인 예외 조항 추가) | 표준 GPL-3.0과 문구가 다르므로 신청할 때 LICENSE 파일을 함께 낸다 |
| PE 구조, 리소스 보기 | PE-bear(GPL-2.0), RisohEditor(GPL-3.0) | RisohEditor는 대화 상자와 문자열 표를 연다 |
| 런타임, 패커 판별 | Detect It Easy(MIT) | 컴파일러, 패커, 설치 파일 형식을 알려준다 |
| PE 구조를 스크립트로 읽기 | pefile(MIT, Python) | 사내 Python 환경에서 쓴다 |
| 헥스 편집기 | ImHex(GPL-2.0) | |
| .NET 디컴파일 | ILSpy(MIT), dnSpyEx(GPL-3.0) | REA의 `inspect_managed_artifact`를 보완한다 |
| PyInstaller 풀기, `.pyc` 디컴파일 | pyinstxtractor(GPL-3.0), pycdc(GPL-3.0) | pycdc는 최신 Python 버전에서 일부 구문을 복원하지 못할 수 있다 |
| 저장된 패킷 캡처 읽기 | Wireshark(GPL-2.0 이상) | 실시간 캡처에 쓰는 Npcap은 오픈 소스가 아니다. Npcap 없이 캡처 파일만 읽는다 |

무료이지만 오픈 소스가 아닌 도구도 있다. Sysinternals Process Monitor와 Process Explorer는 Microsoft 이용 약관을 따른다. Resource Hacker와 HxD는 각자의 프리웨어 라이선스를, Npcap은 자체 라이선스를 따른다. 이 도구들은 금지 목록과 회사 사용 조건을 확인한 뒤에만 쓴다. 이 표의 라이선스는 2026-10-24에 각 공식 저장소에서 확인했다. 설치 패키지에 동봉된 플러그인이나 구성 요소는 라이선스가 다를 수 있다.

## 참고 자료 (References)

- REA 네이티브 분석: <https://github.com/morluto/rea/blob/main/docs/native-investigation.md>
- REA 설치 폴더 분석: <https://github.com/morluto/rea/blob/main/docs/desktop-applications.md>
- REA 빌드 간 함수 매칭: <https://github.com/morluto/rea/blob/main/docs/native-function-matching.md>
- REA 네이티브·패키지 분석 지침: <https://github.com/morluto/rea/blob/main/.agents/skills/reverse-engineer-anything/references/native-and-artifacts.md>
- Microsoft x64 호출 규약: <https://learn.microsoft.com/en-us/cpp/build/x64-calling-convention>
- System Informer: <https://github.com/winsiderss/systeminformer>
- Fibratus: <https://github.com/rabbitstack/fibratus>
- x64dbg: <https://github.com/x64dbg/x64dbg>
- RisohEditor: <https://github.com/katahiromz/RisohEditor>
- PE-bear: <https://github.com/hasherezade/pe-bear>

## 관련 문서

- [회사 Windows + pi 환경 설정](./rea-windows-pi-setup.md)
- [REA + Ghidra로 분석하는 방법](./rea-usage-guide.md)
- [리버스 엔지니어링 작업 절차](./re-process.md)
- [리버스 엔지니어링 기초 개념](./re-fundamentals.md)
