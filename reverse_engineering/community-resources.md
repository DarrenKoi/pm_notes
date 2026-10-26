---
tags: [reverse-engineering, rea, ghidra, llm, community, youtube, reddit]
level: intermediate
last_updated: 2026-10-26
document_type: learning
category_major: "리버스 엔지니어링"
category_middle: "자료"
category_minor: "커뮤니티 작업법"
note_kind: "자료 정리"
---

# 커뮤니티 자료에서 가져온 작업법

> YouTube 제작자, Reddit 실무자, REA 공식 가이드에서 사내 도구 분석에 옮겨 쓸 수 있는 작업법 여섯 가지와 그 출처. 자료 대부분은 게임 해킹이나 악성코드 사례라서, 방법만 가져오고 맥락은 출처로 남긴다.

## 왜 필요한가? (Why)

이 폴더의 다른 문서는 "무엇을 어떤 순서로 하는가"를 정리했다. 그런데 실제로 막히는 지점은 대개 더 작다. 문자열을 찾았는데 참조가 없을 때, 상수 값이 이상할 때, 에이전트가 그럴듯한 결론을 냈을 때 어떻게 하는가. 숙련자들이 영상과 글에서 보여준 습관이 이런 지점에 답한다.

LLM과 MCP로 리버스 엔지니어링을 하는 방법은 빠르게 바뀌고 있고, 공식 문서보다 실무자의 경험담이 먼저 나온다. 다만 경험담은 작성자의 주장이다. 아래 정리는 방법을 가져오되, 성능이나 성공 사례를 일반적인 사실로 옮기지 않는다.

## 핵심 개념 (What)

### 자료를 읽는 기준

- **방법과 결과를 나눈다.** "문자열에서 XREF로 들어간다"는 방법이고, "LLM이 30분 만에 끝냈다"는 그 사람의 결과다. 방법만 가져온다.
- **외부 서비스 구성을 그대로 들이지 않는다.** 많은 자료가 클라우드 LLM이나 외부 웹 서비스에 바이너리를 올리는 구성을 쓴다. 회사에서는 사내 LLM에 연결된 에이전트만 쓴다.
- **도구 설치는 따로 확인한다.** 영상 속 도구(Cheat Engine 등)는 회사 설치 금지 목록과 라이선스를 먼저 확인한다. 기준은 [설치형 프로그램 문서의 보조 도구 기준](./installed-program-analysis.md#회사-pc에-보조-도구를-들이는-기준)에 있다.

### 확인 수준 표시

2026-10-26에 codex와 함께 조사했다. 출처마다 확인한 범위를 아래처럼 표시한다.

| 표시 | 뜻 |
|---|---|
| 확인 | 페이지를 직접 열어 제목과 해당 내용을 봤다. 도구 실행이나 사례 재현은 하지 않았다 |
| 부분 확인 | 페이지와 메타데이터는 확인했지만, 영상 전체나 일부 댓글은 보지 못했다 |
| 2차 확인 못 함 | 1차 조사에서 확인했지만, 재확인 때 로그인 요구로 다시 열지 못했다 |

## 어떻게 사용하는가? (How)

### 1. 문자열에서 XREF로 들어가고, 참조가 없으면 타입부터 의심한다

질문과 관련된 문자열(버튼 문구, 에러 메시지, 설정 키)을 찾고, 그 문자열을 참조하는 함수로 들어간다. Ghidra에서는 `Window → Defined Strings`에서 찾고, 부족하면 `Search → For Strings`로 보완한다.

참조(XREF)가 하나도 없을 때 "이 문자열은 안 쓰인다"고 결론 내리지 않는다. 문자열의 타입이 잘못 정의되었거나, 그 주소를 가리키는 포인터가 데이터로 정의되지 않아서 참조가 안 보이는 경우가 있다. 타입과 포인터 정의를 먼저 확인한다.

- 사내 적용: [작업 절차 4단계(정찰)](./re-process.md#4-정찰한다-triage)의 진입점 찾기. REA를 쓰면 에이전트에게 "문자열 검색 → 참조 함수"를 시키고, 참조가 없다는 결과를 받으면 미확인으로 기록한다.
- 출처: r/ghidra "How do you know what to look for?"의 u/pmrowla 댓글 (확인)

### 2. 이름 → 타입 → 다시 읽기를 반복한다

같은 4바이트 값이 signed `int`로 보면 음수이고 unsigned로 보면 큰 양수다. 의사 코드의 상수가 이상해 보이면 값이 바뀌었다고 판단하기 전에 부호, 비트 폭, 포인터 여부를 확인한다. 오프셋 접근 패턴이 보이면 구조체를 만들고 필드에 이름을 붙인 뒤 의사 코드를 다시 읽는다. 한 번에 다 맞히려 하지 않고 이 순환을 돈다.

- 사내 적용: 보정값, 정수 판정, 비트 필드를 다루는 C++ 도구. Windows에서는 REA로 Ghidra에 이름을 붙일 수 없으므로, 제안 이름과 타입을 [분석 기록](./templates/investigation-log.md) 5절 표에 적고 다음 질문에 함께 넘긴다.
- 출처: r/ghidra "Why is this de-compiled code showing a different value in memory sometimes?"의 u/pmrowla 댓글 (확인), LiveOverflow 영상의 구조체 추론 구간 (부분 확인)

### 3. 실행으로 후보를 줄이고 Ghidra로 읽는다

화면에서 값을 바꿔 가며 메모리 검색으로 후보 주소를 줄이고, 그 주소에 값을 쓰는 명령을 찾는다. 실행 중 DLL 주소는 ASLR 때문에 Ghidra 주소와 다르므로 기준 주소를 맞춘 뒤 Ghidra에서 그 함수를 읽는다. 정적 분석에서 출발하지 않고 동적 관찰에서 출발하는 방향이다.

- 사내 적용: "이 설정 값은 어디서 바뀌는가"처럼 화면 조작과 연결된 질문. 회사에서는 x64dbg(오픈 소스)로 같은 흐름을 만들 수 있다. 주소 변환식은 [설치형 프로그램 문서 7단계](./installed-program-analysis.md#7-실행해서-확인한다)에 있다. 실행은 테스트 PC나 격리된 VM에서만 한다.
- 출처: LiveOverflow, "Windows Game Hacking with Ghidra and Cheat Engine"(2019-08-04)과 같은 내용의 글 "Getting Started With Windows Game Hacking" (부분 확인: 영상 설명·챕터와 글 본문). 영상 기준 05:03 검색 축소, 08:06 쓰는 명령 찾기, 10:27 주소 대응, 11:44 구조체 추론.

### 4. 에이전트 조사 상태를 대화 밖에 남긴다

긴 분석은 에이전트의 컨텍스트가 초기화되거나 요약되면서 앞에서 찾은 근거를 잃는다. 발견 사항, 근거(주소, Evidence ID), 다음 단계를 진행 보고서 파일에 계속 갱신하고, 새 세션은 그 파일부터 읽게 한다. 함수 이름, 주석, 구조체 정의에도 근거를 붙여서 사람이 승인하거나 반려할 수 있게 한다.

결론마다 도구가 실제로 돌려준 주소를 인용하게 하는 것도 같은 목적이다. 인용할 주소가 없는 결론은 사람이 바로 걸러낼 수 있다. 다만 주소를 인용한다고 해석이 맞다는 보장은 없다.

- 사내 적용: [분석 기록 양식](./templates/investigation-log.md)이 그 진행 보고서다. pi에게 "분석 기록 5절 표를 갱신하라"를 매 단계 요청하고, 세션을 새로 열면 그 파일을 먼저 읽힌다. 파일 형식도 확장자 대신 실제 바이트로 판별하게 한다(`file` 명령, `inspect_desktop_application`).
- 출처: r/ghidra "Beyond MCP: A Workspace for Reverse Engineering" 본문과 u/Creative-Brush-6712 댓글 (확인. 글에서 소개하는 외부 서비스는 검토하지 않았다), r/ClaudeAI "My reverse engineering skill for Claude Code" 본문 (확인. 연결된 스킬은 실행하지 않았다)

### 5. 보고서 대신 실행할 수 있는 검증물을 받는다

에이전트에게 "설명해줘"만 요구하지 않고 "같은 결과를 내는 스크립트를 만들어줘"를 함께 요구한다. 스크립트는 실행 결과로 검증하기 쉽다. 설명도 그대로 믿지 않고, 분석자가 따라 할 수 있는 검증 절차를 함께 쓰게 해서 해시·경로·오프셋 같은 근거를 원본과 대조한다. 단, 스크립트가 실제로 입력을 처리하지 않고 결과를 그대로 출력하는 식으로 "속이지" 않는지 코드를 읽고, 실제 입력 파일에 돌려 본다. 저자는 여러 번 검토한 뒤에도 중요한 오류가 남았다고 보고한다.

- 사내 적용: 계산 로직 재구현이 질문이면 Python 함수와 비교 테스트까지 요구한다. [작업 절차 7단계](./re-process.md#7-기록하고-보고한다)의 "테스트가 없는 재구현은 추론일 뿐"과 같은 원칙이다.
- 출처: Karsten Hahn(G DATA, MalwareAnalysisForHedgehogs), "LLMs in malware analysis: Doing things right is difficult"(2026-03) (확인), 같은 저자의 영상 "Build your own AI based Dynamic Reversing Lab, x64dbg automate"(2026-04-19) (부분 확인: 설명·챕터)

### 6. 검증하는 도구도 검증한다

REA 프로젝트 블로그의 게임 재구현 사례에서, 원본과 재구현 결과를 비교하는 도구가 명령어 바이트만 비교하고 그 명령이 가리키는 부동소수점 상수 값은 확인하지 않았다. 그래서 원본은 `128.0`, 재구현은 `0.0`인 함수가 "정확히 일치"로 통과했다. 이후 감사에서 같은 종류의 오류 12개가 더 나왔다. 비교 도구에 일부러 틀린 값을 넣어 거부되는지 테스트하고, 약한 검사로 통과했던 결과를 다시 봤다.

- 사내 적용: 재구현 비교 테스트를 만들면, 일부러 틀린 구현(상수 하나를 바꾼 것)이 실패하는지 먼저 확인한다. 통과만 하는 테스트는 검증이 아니다.
- 출처: REA 블로그, "Reverse Engineering's Industrial Revolution"(2026-10) (확인)

### REA 공식 실습으로 시작하기

REA 가이드 사이트에 대상별 실습이 있다. 사내 대상과 가까운 것부터 고른다. 세 가이드 모두 질문 하나(버튼 하나, 계산식 하나)로 범위를 좁힌 뒤 시작한다.

| 실습 | 사내 대상 | 배울 것 |
|---|---|---|
| [Native Binary Analysis](https://rea.tools/guides/native/) | C++ 도구의 계산식 | 의사 코드가 계산을 숨길 때(`__ftol()` 호출만 보일 때) 명령어와 참조 주소의 실제 상수 바이트를 읽고, 복원한 코드를 원본과 입력별로 비교한다 |
| [JavaScript and Electron Analysis](https://rea.tools/guides/javascript/) | Electron 도구 | Export 버튼 → preload의 `ipcRenderer.invoke` → main 프로세스 처리기 → 파일 쓰기로 좁혀 간다. 정적 연결은 추론으로 남긴다 |
| [Browser and Network Analysis](https://rea.tools/guides/browser/) | 사내 웹 앱 | 분석 전용 Chrome 프로필로 관찰을 시작한 뒤 버튼을 누르고, 요청의 initiator 스크립트에서 코드로 들어간다 |

### 찾았지만 내용을 확인하지 못한 자료

아래 영상은 페이지와 게시일만 확인했다. 설명란에 구체적인 절차가 없었고 영상 전체를 보지 못했으므로, 위 작업법의 근거로 쓰지 않았다.

- Low Level, "I Tested The Viral Reverse Engineering Repo"(2026-10-16): <https://www.youtube.com/watch?v=qFEuYUfauNY>
- OALabs, "Automated AI Reverse Engineering with MCP for IDA and Ghidra (Live VIBE RE)"(2025-04-02): <https://www.youtube.com/watch?v=iFxNuk3kxhk>

LLM과 Ghidra를 쓰는 아래 두 Reddit 글은 1차 조사에서 확인했지만, 재확인 때 로그인 요구로 다시 열지 못했다(2차 확인 못 함). 둘 다 회사 환경과 가까운 이야기라서 남긴다.

- r/LocalLLaMA "Local LLM harness for reverse engineering software?": 의사 코드를 함수 하나씩 넘기면서 PDB 심볼과 앞에서 붙인 이름을 다음 입력에도 유지하는 방법, pi + GhidraMCP로 Windows 도구에서 연동 명령만 뽑아낸 경험담이 댓글에 있다.
- r/ReverseEngineering "Supercharging Ghidra: Using Local LLMs with GhidraMCP": LLM에게 이름과 주석을 제안하게 하는 활용법과 함께, 비트 연산·CRC·구조체 추정의 그럴듯한 오류가 공유 Ghidra DB에 들어가면 이후 분석까지 오염된다는 반대 경험이 있다. 제안과 확정을 나눠 두는 근거로 읽는다.

조사에서 .NET이나 PyInstaller만 다루는 상세 영상, 독립적인 버전 비교(diffing) 시연은 찾지 못했다.

## 참고 자료 (References)

- REA 가이드: <https://rea.tools/guides/>
- REA 블로그, Reverse Engineering's Industrial Revolution: <https://rea.tools/blog/touhou-reconstruction/>
- LaurieWired/GhidraMCP(REA와 다른 Ghidra MCP 서버. 비교용): <https://github.com/LaurieWired/GhidraMCP>
- LiveOverflow, Windows Game Hacking with Ghidra and Cheat Engine: <https://www.youtube.com/watch?v=Pst-4NwY2is>
- LiveOverflow, Getting Started With Windows Game Hacking: <https://liveoverflow.com/getting-started-with-windows-game-hacking-pwn-adventure-3/>
- LiveOverflow, Reverse Engineering PopUnder Trick for Chrome(2017, 이벤트 추적에서 난독화된 호출로 들어가는 순서): <https://www.youtube.com/watch?v=PPzRcZLNCPY>
- G DATA, LLMs in malware analysis: <https://blog.gdatasoftware.com/2026/03/38381-llm-malware-analysis>
- MalwareAnalysisForHedgehogs, Build your own AI based Dynamic Reversing Lab: <https://www.youtube.com/watch?v=QrWzRgPsyTE>
- r/ghidra, How do you know what to look for?: <https://www.reddit.com/r/ghidra/comments/1rotjis/how_do_you_know_what_to_look_for/>
- r/ghidra, Why is this de-compiled code showing a different value: <https://www.reddit.com/r/ghidra/comments/11jj03e/why_is_this_decompiled_code_showing_a_different/>
- r/ghidra, Beyond MCP: A Workspace for Reverse Engineering: <https://www.reddit.com/r/ghidra/comments/1wm6vcw/beyond_mcp_a_workspace_for_reverse_engineering/>
- r/ClaudeAI, My reverse engineering skill for Claude Code: <https://www.reddit.com/r/ClaudeAI/comments/1x4dxgj/my_reverse_engineering_skill_for_claude_code/>
- r/LocalLLaMA, Local LLM harness for reverse engineering software?: <https://www.reddit.com/r/LocalLLaMA/comments/1vzg84e/local_llm_harness_for_reverse_engineering_software/>
- r/ReverseEngineering, Supercharging Ghidra: <https://www.reddit.com/r/ReverseEngineering/comments/1kbfb5a/supercharging_ghidra_using_local_llms_with/>

## 관련 문서

- [리버스 엔지니어링 작업 절차](./re-process.md)
- [REA + Ghidra로 분석하는 방법](./rea-usage-guide.md)
- [설치형 Windows 프로그램을 Ghidra로 분석하는 방법](./installed-program-analysis.md)
- [분석 기록 양식](./templates/investigation-log.md)
