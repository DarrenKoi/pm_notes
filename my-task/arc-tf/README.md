---
tags: [my-task, arc-tf, auto-recipe-creation, computer-use, index]
category_major: "업무 기획·산출물"
category_middle: "ARC TF (Auto Recipe Creation)"
category_minor: "전체 목차"
note_kind: "목차"
classified_on: "2026-10-29"
last_updated: 2026-11-06
---

# ARC TF — 레시피 자동 생성 태스크포스

> 장비 레시피(Recipe) 셋업을 컴퓨터 사용(Computer-use) Agent로 자동화하려고, MI 현업 엔지니어와 공통 플랫폼 팀을 묶어 운영하는 TF의 작업공간. 본인이 리드한다.

## 한 문단 요약

레시피 셋업은 숙련 엔지니어가 원격 제어 화면에서 손으로 하는 반복 작업이고, 장비가 늘면 사람도 그만큼 늘어야 한다. 1차 PoC(Align Fail 대응 자동화)에서 사내 VLM으로 API 없는 장비 화면을 보고 조작할 수 있다는 것, VLM과 CV를 나눠 쓰면 오프라인 평가에서 정밀 좌표도 잡힌다는 것을 확인했다. 남은 문제는 두 가지다. 하나는 현장 신뢰성과 실행 엔진 제품화이고, 다른 하나는 장비군마다 다른 레시피 절차다. 뒤의 것은 현업 지식 없이는 풀 수 없다. 그래서 **공통 플랫폼 팀**이 엔진을 만들고, **장비군별 셀**(MI 엔지니어)이 절차와 판정 기준을 맡는 TF를 꾸린다. 2026-11에 기획안을 쓰고, 12월에 임원 승인을 받고, 2027-01에 가동한다.

## 문서

| 순서 | 문서 | 독자 | 내용 | 상태 |
|---|---|---|---|---|
| 0 | [문제 정의](./00-problem-statement.md) | 임원·TF 전원 | 누가 무엇을 하려다 왜 막히는가, 한 문장 문제 정의와 판단 지표 | 초안, 셀 엔지니어 확인 전 |
| 1 | [TF 구성 기획안](./01-tf-proposal.md) | 임원 | 왜 해야 하는가, PoC 성과, 범위, TF 구성, 타임라인, 요청 사항 | 초안, 2026-12 보고용 |
| 2 | [참여 팀과 역할](./02-roles-and-teams.md) | TF 전원 | 플랫폼 팀·셀 구성, 셀 명단, 책임 분담표(RACI) | 초안 |
| 3 | [운영 방식](./03-operating-rhythm.md) | TF 전원 | 회의, 월간 보고 형식, 의사결정, 작업 관리, 안전 원칙 | 초안 |
| 4 | [온보딩 학습 과정](./04-onboarding.md) | 셀 엔지니어 | 2027-01~02 8주 과정, 수료 기준 | 초안 |
| 5 | [PoC 자산과 남은 과제](./05-poc-assets.md) | 플랫폼 팀·임원 | Align Fail PoC에서 가져오는 것, 새로 풀어야 하는 것 | 초안 |
| 6 | [플랫폼 1차 PRD](./06-platform-prd.md) | 플랫폼 팀·셀 엔지니어 | 무엇을 만드나: 사용자 스토리와 수용 기준, 성공·가드레일 지표, 범위 밖, 열린 질문 | 초안, 검토 전 |
| — | [결정 기록](./decision-log.md) | 리드 | 날짜별 결정과 이유 | 2026-10-29 |
| — | [위험 목록](./risk-register.md) | 리드 | 위험, 대응, 담당 | 2026-10-29 |

## 최종 산출물 (docx, pptx)

임원에게 내는 기획안의 최종본은 Word와 PowerPoint 파일이다. `.md`가 원본이고, 두 파일은 손으로 고치지 않고 다시 만든다. 필요 패키지: `python-docx`, `python-pptx`. 한글 폰트는 맑은 고딕.

| 파일 | 내용 | 만드는 명령 |
|---|---|---|
| [deliverables/arc-tf-proposal.docx](https://github.com/DarrenKoi/pm_notes/blob/main/my-task/arc-tf/deliverables/arc-tf-proposal.docx) | 기획안 본문(01) + 별첨 A 문제 정의(00) | `python my-task/arc-tf/deliverables/build_docx.py` |
| [deliverables/arc-tf-proposal.pptx](https://github.com/DarrenKoi/pm_notes/blob/main/my-task/arc-tf/deliverables/arc-tf-proposal.pptx) | 2026-12 임원 보고 deck 11장, 슬라이드마다 발표자 노트 | `python my-task/arc-tf/deliverables/build_pptx.py` |

- docx는 01·00 `.md`를 그대로 변환한다. `.md`를 고치면 명령만 다시 실행한다.
- pptx는 01·00의 **요약**이라 내용이 `build_pptx.py` 안에 있다. 01·00에서 숫자, 범위, 일정, 요청 사항을 바꾸면 그 스크립트의 슬라이드 내용도 같이 고친다.
- 플랫폼 PRD(06)는 팀 작업 문서라 최종 산출물에 넣지 않는다.

## 일정

| 시기 | 단계 |
|---|---|
| 2026-11 | 기획안 작성 |
| 2026-12 | 임원 보고, TF 구성 승인 |
| 2027-01~02 | 온보딩 학습, 셀별 기준선 측정 |
| 2027-03~06 | 1차 구축 (1차 셀 3개: CD-SEM, VeritySEM, Thickness) |
| 2027-07~09 | 파일럿 |
| 2027-10~12 | 2차 셀 확장 |

## 채워야 할 ⟨확인⟩ 목록

### 채운 것 (2026-11-02)

| 항목 | 결정 |
|---|---|
| 1차 장비군 | CD-SEM(리드 직접 담당), VeritySEM, Thickness |
| 겸직 시간 | 주 5시간 (근무일마다 1시간) |
| 공통 플랫폼 팀 | 개발 2명 (본인 포함) |
| 보고 대상 임원 | AI TF 담당 임원, 기반기술센터장 |
| 비생산 검증 | 장비 사용 인가를 받아 하루 1시간 실장비 테스트 |
| 화면 조작 보안 | 설치형 프로그램을 배포해 엔지니어 본인 PC에서 실행. 장비 PC에는 설치하지 않는다 |
| 접속 경로 | CD-SEM은 로컬 RCS, VeritySEM·Thickness는 웹 VDI → RCS/Radmin Viewer. 화면 캡처는 두 경로 모두 됨, 모니터는 대부분 FHD |
| 현재 셋업 시간 | 레시피당 평균 30~60분 (현업 경험치, 셋업 전체 기준) |

### 아직 남은 것

- ⟨확인: VeritySEM·Thickness 셀 참여 엔지니어⟩
- ⟨확인: 설치형 프로그램 배포·녹화 보안 승인 절차⟩
- ⟨확인: 셀별 자동화 대상 구간 실측 — 온보딩 8주차⟩
- ⟨확인: 입력 주입 테스트 — 본인 PC, 로컬 RCS·VDI 경로 각각, 저장 없음⟩
- ⟨확인: VDI 워터마크 여부⟩
- ⟨확인: 작업 관리 도구⟩

사내 수치, 실명, 장비 상세는 `*_사내용.md`(git 제외)에만 둔다.

## 휴대폰, 태블릿에서 읽기

[ARC TF 문서 모음 (claude.ai Artifact, 비공개)](https://claude.ai/artifact/PUq6fvLRzbF7sP6DciTovo) — `reading.html`은 `python scripts/build_folder_html.py my-task/arc-tf`로 다시 만든다. 손으로 고치지 않는다.
