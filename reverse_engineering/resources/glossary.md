---
tags: [reverse-engineering, glossary, beginner]
level: beginner
last_updated: 2026-11-16
document_type: learning
category_major: "리버스 엔지니어링"
category_middle: "개념"
category_minor: "용어 사전"
note_kind: "용어 사전"
---

# 리버스 엔지니어링 용어 사전

> 이 폴더의 문서와 발표 자료에 나오는 어려운 말을 전공이 아닌 사람도 읽을 수 있게 풀어 쓴 사전. 모르는 말이 나오면 이 문서에서 찾는다.

## 왜 필요한가? (Why)

이 폴더의 문서는 ELF, ASAR, CDP, HAR처럼 줄임말과 영어 용어를 많이 쓴다. 문서마다 처음 나올 때 풀어 쓰기는 하지만, 뒤에서 다시 나오면 설명 없이 지나간다. 팀원 대부분은 전기 공학이나 컴퓨터 공학 전공이 아니므로, 한곳에 모아 두고 찾아볼 수 있어야 한다.

같은 말이 분야에 따라 다른 뜻으로 쓰이는 경우도 있다. "어셈블리"는 기계어를 적는 언어이기도 하고 .NET의 배포 파일을 부르는 이름이기도 하다. 이런 말은 마지막 절에 따로 모았다.

## 핵심 개념 (What)

기본 개념(소스 코드, 컴파일, 디스어셈블, 디컴파일, 심볼, 임포트, 상호 참조, 진입점)은 [기초 개념 문서](./re-fundamentals.md)의 "꼭 알아야 할 용어"와 "기초 상식" 절에 있다. 여기서는 그 밖의 말을 다룬다.

### 파일과 형식

| 용어 | 쉬운 뜻 | 예, 비유 |
|---|---|---|
| PE (Portable Executable) | Windows 실행 파일(`.exe`, `.dll`)의 형식. 회사 PC의 REA가 Ghidra로 여는 형식이다. Portable(옮겨 갈 수 있는)은 Windows NT가 여러 CPU에서 돌던 시절, CPU가 달라도 같은 형식을 쓴다는 뜻이다. 파일 맨 앞이 `MZ`(4d 5a)로 시작한다 | 서류 양식 이름 |
| ELF (Executable and Linkable Format) | Linux 실행 파일의 형식. 실행 파일(Executable)과 링크 전의 조각(`.o`), 공유 라이브러리(`.so`)를 같은 형식으로 담아서 Linkable이 붙었다. 파일 맨 앞이 `7f 45 4c 46`(`.ELF`)이다. 회사 Windows PC에서는 REA로 Ghidra 분석을 할 수 없고, Ghidra 화면에서 직접 열어야 한다 | Linux용 서류 양식 |
| Mach-O (Mach Object) | macOS, iOS 실행 파일의 형식. macOS 커널의 뿌리인 Mach 커널에서 이름을 따왔다. ELF와 같은 제약이 있다 | macOS용 서류 양식 |
| DLL (Dynamic Link Library) | 여러 프로그램이 나눠 쓰는 기능 묶음 파일. `.exe`가 시작할 때나 실행 중에 불러 쓴다 | 여러 장비가 같이 쓰는 공용 부품 |
| ASAR (Atom Shell Archive) | Electron 앱이 웹 코드(JavaScript, HTML)를 파일 하나로 묶은 것(`app.asar`). Electron의 옛 이름이 Atom Shell이라서 이 이름이 붙었다. 압축은 아니라서 풀면 넣어 둔 파일이 그대로 나온다. 번들링과 난독화는 그대로 남는다 | 서류를 끈으로 묶은 꾸러미 |
| HAR (HTTP Archive) | 브라우저가 주고받은 요청과 응답을 저장한 기록 파일. 개발자 도구의 Network 탭에서 저장한다. 쿠키와 토큰이 들어 있을 수 있으므로 분석이 끝나면 지운다 | 통화 녹음 파일 |
| JSON (JavaScript Object Notation) | 사람도 읽을 수 있는 데이터 글 형식. `{"이름": "값"}` 모양이다. REA의 결과와 많은 설정 파일이 이 형식이다 | 항목과 값을 적은 표 |
| CSV (Comma-Separated Values) | 쉼표로 열을 나눈 표 파일. 엑셀로 열린다 | 쉼표로 칸을 나눈 표 |
| 해시 (Hash) | 파일 내용으로 계산한 짧은 값. 한 글자만 달라도 값이 바뀐다. 흔히 SHA-256(Secure Hash Algorithm)을 쓴다 | 파일의 지문 |

### 웹과 통신

| 용어 | 쉬운 뜻 | 예, 비유 |
|---|---|---|
| HTTP (HyperText Transfer Protocol) | 브라우저와 서버가 주고받는 약속(규약). 요청을 보내면 응답이 온다 | 주문서와 영수증 양식 |
| API (Application Programming Interface) | 프로그램끼리 데이터를 주고받는 창구 | 식당의 주문 창구 |
| 엔드포인트 (Endpoint) | API에서 요청을 받는 주소 하나 | `/api/recipes` 같은 창구 번호 |
| 쿠키, 토큰 (Cookie, Token) | 로그인한 상태를 기억하는 값. 남이 가지면 내 계정처럼 쓸 수 있다. HAR 파일을 지우는 이유다 | 출입증 |
| 개발자 도구 (DevTools) | 브라우저에 들어 있는 분석 화면. F12로 연다. Sources 탭에서 코드를, Network 탭에서 통신을 본다 | 자동차 정비용 진단기 |
| CDP (Chrome DevTools Protocol) | 다른 프로그램이 Chrome을 원격으로 조종하고 관찰하는 연결 규격. REA가 웹 페이지를 볼 때 쓴다. Chrome을 디버깅 포트를 연 상태로 띄워야 한다 | 원격 진단 단자 |
| 번들 (Bundle) | 웹 앱의 코드 파일 여러 개를 하나로 합치고 줄인 파일. 이름이 짧게 바뀌어 읽기 어렵다 | 압축 요약본 |
| 소스 맵 (Source Map) | 번들을 원래 파일과 줄 번호로 되돌리는 지도 파일(`.map`). 원본 내용이 들어 있으면 원본에 가깝게 읽는다. 자세한 내용은 [소스 맵 문서](./source-map-discovery.md) | 요약본의 원문 찾아보기 |
| 패킷 캡처 (Packet Capture) | 네트워크를 지나는 데이터 조각(패킷)을 모두 기록하는 일. 파일은 `.pcap`, 읽는 도구는 Wireshark다. 회사 Windows PC의 REA로는 할 수 없다 | 전화선 전체 감청 기록 |

HAR와 패킷 캡처의 차이: HAR는 브라우저 안의 요청만 담고, 패킷 캡처는 PC를 드나드는 모든 통신을 담는다. 웹 앱 분석에는 대개 HAR로 충분하다.

### 설치와 실행 환경

| 용어 | 쉬운 뜻 | 예, 비유 |
|---|---|---|
| Node.js | JavaScript를 브라우저 밖에서 실행하는 프로그램. REA가 이 위에서 돈다 | 엔진 |
| npm, npx | Node.js에 딸린 설치 도구. npm은 흔히 Node Package Manager의 줄임말로 통한다(공식적으로는 약어가 아니라고 한다). `npm install --global`은 설치해 두고 쓰고, `npx`는 설치하지 않고 한 번 받아서 실행한다. 차이는 [회사 환경 설정 문서](./rea-windows-pi-setup.md)에 있다 | 앱 스토어 |
| JDK (Java Development Kit) | Java 프로그램을 실행하고 만드는 도구 묶음. Ghidra가 Java로 만들어져서 필요하다 | Ghidra를 돌리는 전원 |
| Gradle | Java 프로그램을 소스에서 만들어 내는 빌드 도구. JDK 위에서 돌며 JDK의 컴파일러를 순서대로 부른다. Ghidra를 소스에서 빌드할 때만 필요하고, REA와 배포본 Ghidra에는 필요 없다 | 작업 지시서와 반장 |
| PowerShell | Windows의 명령 입력 창 | 글로 지시하는 조작판 |
| 환경 변수 (Environment Variable) | 프로그램에게 알려 주는 설정값. 예: Ghidra가 설치된 위치(`GHIDRA_INSTALL_DIR`) | 공용 게시판에 붙인 메모 |
| 헤드리스 (Headless) | 화면(창) 없이 프로그램을 돌리는 방식. REA가 Ghidra를 이렇게 돌린다 | 모니터 없이 돌리는 장비 |
| Chrome 프로필 (Profile) | Chrome의 사용자별 저장 공간(로그인, 쿠키, 확장 기능). 분석 전용 프로필을 따로 만들어 평소 계정 정보와 섞이지 않게 한다 | 분석 전용 작업복 |
| 테스트 환경, VM (Virtual Machine) | 운영 장비, 운영 DB와 떨어진 실험용 환경. VM은 PC 안에 만든 또 하나의 가상 PC다. 네트워크가 열려 있으면 VM 안에서도 운영 DB에 닿을 수 있다 | 시험용 장비 |
| 샌드박스 (Sandbox) | 프로그램을 가둬서 바깥에 손대지 못하게 하는 격리 환경. REA는 샌드박스를 제공하지 않는다 | 모래 놀이터 울타리 |
| NTFS (New Technology File System) | Windows 하드디스크의 기본 저장 방식(파일 시스템). Ghidra로 분석할 파일은 NTFS 로컬 디스크에 복사해 둔다 | 서랍 정리 방식 |
| UNC (Universal Naming Convention) 경로 | `\\서버\공유폴더` 모양으로 네트워크 공유 폴더를 가리키는 경로. Ghidra 분석 대상으로 쓸 수 없다 | 다른 건물 서랍의 주소 |
| 레지스트리 (Registry) | Windows가 프로그램 설정을 모아 저장하는 곳. 설치 경로와 옵션 값이 들어 있다 | Windows의 중앙 설정 장부 |
| MSI | Windows 설치 프로그램 파일 형식. 이름은 Windows Installer의 옛 이름인 Microsoft Installer에서 왔다. 안에 설치할 파일들이 들어 있다 | 조립 키트 상자 |

### 분석 결과에 나오는 말

| 용어 | 쉬운 뜻 | 예, 비유 |
|---|---|---|
| 16진수 (`0x`) | 0~9와 a~f로 세는 숫자 표기. 앞에 `0x`를 붙인다. `0x65`는 101, `0x10`은 16이다 | 같은 수의 다른 표기 |
| 주소 (Address) | 프로그램 안의 위치 번호. `FUN_140012340`의 숫자가 함수의 주소다 | 서랍 번호 |
| VA (Virtual Address) | 프로그램을 메모리에 올렸을 때의 주소. Ghidra가 보여주는 주소가 이것이다. 64비트 `.exe`는 보통 `0x140000000`에서 시작한다(이미지 베이스) | 건물 주소 |
| RVA (Relative Virtual Address) | 이미지 베이스에서 떨어진 거리. VA = 이미지 베이스 + RVA. 실행할 때마다 시작 주소가 바뀌어도 RVA는 같으므로 기록에는 `모듈 + RVA`로 적는다 | 건물 안 몇 호 |
| 파일 오프셋 (File Offset) | 파일 맨 앞에서 몇 바이트 떨어졌는지. 16진수 편집기에서 보는 위치다. 섹션 정렬 때문에 RVA와 값이 다를 수 있다 | 설계도의 몇 쪽 |
| 서수 (Ordinal) | DLL 함수에 붙인 번호. 이름 없이 번호로만 임포트, 익스포트하는 경우가 있어서 함수 이름이 안 보일 수 있다 | 이름 대신 사번 |
| 레지스터 (Register) | CPU 안의 아주 작은 저장 칸. 계산은 여기서 한다. `eax`, `ecx` 같은 이름이 붙어 있다 | 작업대 위의 손 닿는 칸 |
| 포인터 (Pointer) | 값 대신 값이 있는 위치(주소)를 담은 변수 | 물건 대신 서랍 번호를 적은 쪽지 |
| 구조체 (Struct) | 관련 값 여러 개를 하나로 묶은 데이터. 예: 레시피 = 이름 + 온도 + 시간 | 칸이 나뉜 서류 양식 |
| 오프셋 (Offset) | 묶음의 시작에서 몇 칸 떨어졌는지. `param_1 + 0x10`은 시작에서 16바이트 뒤의 값이다 | 양식의 몇 번째 칸 |
| 호출 규약 (Calling Convention) | 함수에 입력값을 어느 칸으로 넘기고 결과를 어디로 돌려주는지 정한 약속. Windows x64는 첫 입력을 `rcx`(`ecx`)로 넘기고 결과를 `rax`(`eax`)로 돌려준다 | 결재 서류를 넣고 받는 서류함 위치 |
| 간접 호출 (Indirect Call) | 부를 함수가 실행 중에 정해지는 호출. 실행 전에는 누가 누구를 부르는지 모를 수 있다 | "그때 당번에게 전달" |
| 인라인 (Inline) | 작은 함수를 부르는 대신 그 내용을 부른 곳에 복사해 넣는 최적화. 그 함수가 목록에서 사라진다 | 별첨 대신 본문에 옮겨 적기 |
| 제어 흐름 그래프 (CFG) | 함수 안의 갈림길과 반복을 그린 그림 | 공정 흐름도 |
| 디버그 정보 (PDB, DWARF) | 함수, 변수 이름과 소스 줄 번호를 담은 정보. Windows는 `.pdb`(Program Database) 파일로 따로 두고, Linux는 DWARF 형식을 쓴다. 있으면 분석이 훨씬 쉽다 | 이름표 목록 |
| 스트립 (Strip) | 배포 전에 이름 정보(심볼)를 지우는 것 | 이름표 떼기 |
| 커버리지 (Coverage) | 도구가 전체 중 얼마나 봤는지. "일부"이면 못 본 곳에 답이 있을 수 있다 | 검사한 범위 |

### 헷갈리는 짝

| 짝 | 차이 |
|---|---|
| 어셈블리 (언어) / .NET 어셈블리 | 어셈블리 언어는 기계어를 사람이 읽게 적은 표기다. .NET 어셈블리는 .NET의 `.exe`, `.dll` 배포 단위를 부르는 이름이다 |
| 기계어 / 바이너리 | 기계어는 CPU가 실행하는 명령이다. 바이너리(실행 파일)는 기계어와 문자열, 임포트 목록 등을 함께 담은 그릇이다 |
| 컴파일 (넓은 뜻) / 컴파일 (좁은 뜻) | 넓은 뜻은 소스에서 실행 파일까지 전체(빌드)다. 좁은 뜻은 C를 어셈블리로 바꾸는 단계만이다 |
| 디스어셈블 / 디컴파일 | 디스어셈블은 기계어를 어셈블리로 1대1로 옮긴다(정확하다). 디컴파일은 C 비슷한 코드로 다시 쓴다(추정이 섞인다) |
| HAR / 패킷 캡처 | HAR는 브라우저 안의 요청만, 패킷 캡처는 PC의 모든 통신을 담는다 |
| 테스트 환경 / 샌드박스 | 테스트 환경은 운영과 떨어진 실험 장소이고, 샌드박스는 프로그램이 바깥에 손대지 못하게 막는 울타리다. VM 안이라도 네트워크가 열려 있으면 샌드박스가 아니다 |

## 어떻게 사용하는가? (How)

- 문서를 읽다가 모르는 말이 나오면 브라우저의 찾기(Ctrl+F)로 이 문서에서 찾는다.
- 입문 deck에도 같은 내용이 부록의 "용어 사전" 슬라이드로 들어 있다. 발표 중 질문이 나오면 그 장을 펼친다.
- 새 문서에 어려운 말을 쓰면 이 사전에 한 줄을 추가한다. deck의 용어 사전 슬라이드는 `slides/build_team_intro.py`의 부록에서 함께 고친다.

## 참고 자료 (References)

- Microsoft PE 형식 문서: <https://learn.microsoft.com/windows/win32/debug/pe-format>
- Chrome DevTools Protocol: <https://chromedevtools.github.io/devtools-protocol/>
- Electron ASAR 문서: <https://www.electronjs.org/docs/latest/tutorial/asar-archives>
- HAR 형식(W3C 초안): <https://w3c.github.io/web-performance/specs/HAR/Overview.html>

## 관련 문서

- [리버스 엔지니어링 기초 개념](./re-fundamentals.md)
- [회사 Windows + pi 환경 설정](./rea-windows-pi-setup.md)
- [REA + Ghidra로 분석하는 방법](./rea-usage-guide.md)
- [설치형 Windows 프로그램을 Ghidra로 분석하는 방법](./installed-program-analysis.md)
