---
tags: [reverse-engineering, rea, ghidra, mcp]
level: intermediate
last_updated: 2026-10-24
document_type: learning
category_major: "리버스 엔지니어링"
category_middle: "주제 안내"
category_minor: "전체 목차"
note_kind: "목차"
---

# 리버스 엔지니어링(Reverse Engineering) 노트

> 동료 엔지니어가 만든 프로그램과 웹 앱을 소스 없이 이해하기 위한 지식, 절차, 그리고 REA + Ghidra 사용법.

## 왜 이 폴더가 있는가

사내 도구는 만든 사람이 떠나거나 소스가 흩어지면 실행 파일만 남는다. 그 도구를 고치거나, 다른 시스템과 연동하거나, 같은 기능을 새로 만들려면 "이 프로그램이 실제로 무엇을 하는가"를 바이너리와 번들에서 다시 읽어내야 한다. 이 폴더는 그 작업을 반복 가능한 절차로 정리한다.

도구는 [REA(Reverse Engineer Anything)](https://github.com/morluto/rea)를 쓴다. REA는 코딩 에이전트에 MCP 서버로 붙어서 Ghidra 같은 분석 엔진을 대신 호출하고, 결론마다 근거(Evidence)를 함께 돌려준다.

## 읽는 순서

| 순서 | 문서 | 성격 | 언제 읽나 |
|---|---|---|---|
| 1 | [리버스 엔지니어링 기초 개념](./re-fundamentals.md) | 설명 | 처음 시작할 때. 정적·동적 분석, 디컴파일, 대상별 특징 |
| 2 | [리버스 엔지니어링 작업 절차](./re-process.md) | 절차 | 새 분석을 시작할 때마다. 범위 확인부터 보고까지 7단계 |
| 3 | [회사 Windows + pi 환경 설정](./rea-windows-pi-setup.md) | 사용법 | 회사 PC에 REA를 처음 연결할 때. Windows 제약과 pi 등록 |
| 4 | [REA + Ghidra로 분석하는 방법](./rea-usage-guide.md) | 사용법 | 실제로 에이전트에 일을 시킬 때. 프롬프트 패턴과 대상별 경로 |
| 5 | [설치형 Windows 프로그램을 Ghidra로 분석하는 방법](./installed-program-analysis.md) | 사용법 | 설치형 프로그램을 분석할 때. 런타임 판별부터 버전 비교까지 |
| 6 | [REA 명령·도구 참조](./rea-reference.md) | 참조 | 명령어, MCP 도구 이름, 환경 변수를 찾을 때 |
| - | [분석 기록 양식](./templates/investigation-log.md) | 양식 | 분석 하나를 시작할 때 복사해서 쓴다 |

회사 환경은 Windows PC + pi 에이전트(사내 LLM)다. 주 분석 대상은 설치형 프로그램이므로, 실무에서는 3번과 5번 문서를 가장 자주 본다.

## 먼저 알아둘 세 가지

1. **허가 범위 안에서만 분석한다.** 동료가 만든 도구라도 소유 부서와 목적을 먼저 확인한다. 절차는 [작업 절차 1단계](./re-process.md#1-범위와-허가를-확인한다)에 있다.
2. **관찰, 추론, 미확인을 구분한다.** 디컴파일 결과는 원본 소스가 아니다. REA 결과도 이 셋을 나눠서 돌려주므로, 기록할 때 그 구분을 지운다면 REA를 쓰는 이유가 사라진다.
3. **분석 결과는 회사 밖으로 내보내지 않는다.** REA는 분석을 로컬에서 하지만, 도구 결과는 에이전트의 모델로 넘어간다. 사내 LLM 엔드포인트에 연결된 에이전트에서만 쓴다. 이 저장소에도 장비명, 내부 시스템 상세, 디컴파일 코드 원문은 넣지 않는다.

## 참고 자료

- REA 저장소: <https://github.com/morluto/rea>
- REA 가이드 사이트: <https://rea.tools/guides/>
- Ghidra: <https://github.com/NationalSecurityAgency/ghidra>
