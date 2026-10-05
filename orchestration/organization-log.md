---
tags: [orchestration, organization-log]
reviewed_on: 2026-10-04
review_status: partial
document_type: maintenance
category_major: "에이전트 오케스트레이션"
category_middle: "문서 관리"
category_minor: "정리·검증 기록"
note_kind: "관리 기록"
classified_on: "2026-10-05"
---

# orchestration 정리 기록 — 2026-10-04

## 범위와 결정

공개 학습/템플릿 Markdown 6개, 숨김 에이전트 기록 Markdown 2개와 실행/설정 파일 5개를 검토했다. Markdown만 수정하고
[검토 보충](./review-notes.md)과 이 기록을 추가했다. 삭제·이동·폴더 간 링크는 없다.
원샷/반복 스케줄/정책/설정 프롬프트는 용도가 달라 유지했다. 설정 JSON 중복은
복사 가능한 프롬프트라는 고유 목적이 있어 원문을 유지하고 값 일치를 검사했다.
일괄 분할·대표 문서 통합 판단은 Claude 협의 부재로 보류한다.

공식/설치 패키지 자료와 코드에서 확인한 오류·과도한 일반화만 한정 수정했다.
2026-09-22 사내 성공 기록과 pi 0.86.1 당시 확인 기록은 역사적 사실의 기록으로
남겼고 이번 검증으로 승격하지 않았다. 실행 템플릿은 이번 작업의 명령이 아니다.
night-run.ps1의 구현 변경·설정 적용·스케줄 등록·commit/push는 수행하지 않았다.

## 근거

확인일 2026-10-04. 로컬 pi-subagents 0.75.0의 package.json과 agents/models/
configuration 문서, Pi 공식 보안 문서, Microsoft schtasks/Start-Process 설명,
Git 공식 git-add 설명을 대조했다. 링크와 적용 조건은
[검토 보충](./review-notes.md)에 모았다. 사내 모델 별칭·성능·한도와 전체
패키지 릴리스 비교는 미확인이다. npm 웹 조회 403은 설치 패키지의 일차 자료로 보완했다.

## Claude 협의 결과

HERDR_ENV=1에서 현재 pane 조회를 재시도했으나 pane_not_found였다. 전용 Claude
pane을 연결/제어하지 않았고 다른 주제 pane도 건드리지 않았다. 의견을 받지 못했다.
야간 래퍼 교체, 엔진 설정·역할 재설계·큰 문서 통합은 보류한다. 변경하지 않은
코드의 위험을 기록하고 독립적인 문서/로컬 검증을 진행했다.

## 문서별 결과

| 문서 | 검토 결과 |
|---|---|
| [README.md](./README.md) | 읽기 순서 추가. 비용·모델 품질·429 성공 보장 일반화 수정. 과거 사내 검증 기록 유지; 현재 엔드포인트 미확인. |
| [office-setup.md](./office-setup.md) | PROMPT-2/3 외부 fence를 4개로 바꿔 내부 JSON과 함께 복사 가능하게 함. 두 JSON은 원본 스니펫과 값 일치. 실행 지시 미수행. |
| [night-setup.md](./night-setup.md) | 래퍼 적용 보류를 등록 직전에 명시. dirty 변경 주체 확인과 자기 변경만 복구하도록 템플릿 수정. Windows 미검증. |
| [oneshot.md](./oneshot.md) | 단일 무인 실행과 반복 실행 용도 충돌 수정. intent-to-add를 배정 경로로 제한. 전체 정책/예제 보존. |
| [agents-md.snippet.md](./agents-md.snippet.md) | git ls-files와 ignore 규칙 구분. intent-to-add의 경로를 제한. 상속은 실제 resolved 역할의 설정 조건. |
| [decisions.example.md](./decisions.example.md) | 결정 정책 템플릿 보존. 프롬프트 고정은 OS 경계 아님, 확신도는 휴리스틱임을 보충 문서에 명시. |

신규 검토 보충은 버전과 정적 한계의 대표 문서, 이 기록은 유지보수 문서다.
각 원래 문서에 같은 검토일과 partial 상태를 넣고 보충으로 연결했다.

## 실행/설정 파일 검토 — 수정 없음

| 파일 | 결과 |
|---|---|
| [merge-settings.py](./merge-settings.py) | 임시 파일 병합 검증 통과. cadence 삭제/원자적 쓰기/동시 편집은 미구현. |
| [smoke.sh](./smoke.sh) | bash 문법과 추출 L0 fixture 정상/실패 검출 통과. 실제 L1~L3 증거 강도는 보충에 구분. |
| [night-run.ps1](./night-run.ps1) | 잠금 전 job 시작, dead owner 잠금 정리, 명령 문자열 기반 프로세스 선택, 종료 코드 한계 확인. 적용 보류. |
| [settings.snippet.json](./settings.snippet.json) | JSON 파싱·office 템플릿 값 일치. 실제 사내 게이트웨이 미실행. |
| [subagent-config.snippet.json](./subagent-config.snippet.json) | JSON 파싱·office 템플릿 값 일치. Windows/실 런타임 상한 미실행. |

## 검증 1 — 목록과 고유 내용

Markdown 6 → 8. 원래 문서·경로·고유 프롬프트와 JSON 예제 유지, 실행/설정 5개
시작 시 SHA256 동일. 대규모 재작성 대신 부분 수정·검토 보충으로 손실을 피했다.

## 검증 2 — 기술·실습

임시 파일로 merge-settings.py preview 무변경, apply 깊은 병합/배열 교체/원래 키
보존/백업/같은 적용 무변경/깨진 JSON 중단을 확인했다. 스니펫에서 생략한 cadence가
유지되는 것도 확인했다. smoke.sh의 Python L0 부분만 추출해 가상 provider 2개·
모델 5개에서 FAIL 0, 잘못된 defaultModel provider/id를 넣어 FAIL 검출했다.
이것은 실제 pi 모델 레지스트리 검증이 아니다. bash -n, Python AST, JSON 2개
파싱, 문서 내 JSON 2개와 스니펫 값 비교가 통과했다. 네트워크 모델 호출·사용자
설정 읽기/쓰기·Windows 실행·pwsh 문법 검증은 하지 않았다.

## 검증 3 — 참조·메타데이터·Obsidian

최종 확인 결과를 아래에 추가한다.

## 보류

Claude 협의, 실제 사내 모델·속도 제한, Windows PowerShell/Task Scheduler,
야간 래퍼 구현 수정과 운영 격리 시험이 남아 있다. 구조·로컬 검증을 마치더라도
partial 상태를 유지하며 전체 저장소 완료로 기록하지 않는다.

숨김 `.remember/now.md`는 시작 시부터 존재한 0바이트 에이전트 상태 파일이다.
`.remember/today-2026-09-26.md`는 당시 세션 압축 상태의 역사적 기록이다.
두 파일은 학습 문서로 재작성하지 않고 원문을 보존했으며 메타데이터 검사만 제외했다.
숨김 상태·로그·캐시 전체도 시작 시 해시를 보존한다. 공개 Markdown 6 → 8과 별도로
숨김 기록 2개를 검토·보존했으므로 전체 Markdown 검토 수는 10개다.

## 최종 확인

공개 문서 8개 링크·앵커 새 오류 0, 기존 오류 0, diff --check 통과.
Obsidian 1.13.7의 확인된 pm_notes vault에서 8개 properties와 reviewed_on을 확인했다.
실제 읽기 화면에서 orchestration/README → orchestration/review-notes를 클릭하고
정확한 폴더·한국어 본문·버전·날짜·태그·alias를 확인했다. 표는 AX 구조에서 확인했다.
8개 전 화면과 Windows 터미널 렌더링을 모두 확인한 것은 아니다.
숨김 에이전트 기록·캐시와 실행/설정 5개 모두 시작 시 해시 동일을 재확인했다.
