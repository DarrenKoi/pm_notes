---
tags: [my-task, arc-tf, auto-recipe-creation, computer-use, index]
category_major: "업무 기획·산출물"
category_middle: "ARC TF (Auto Recipe Creation)"
category_minor: "전체 목차"
note_kind: "목차"
classified_on: "2026-10-29"
last_updated: 2026-11-09
---

# ARC TF: 레시피 자동 생성 태스크포스

> 장비 레시피(Recipe) 셋업을 컴퓨터 사용(Computer-use) Agent로 자동화하려고, MI 현업 엔지니어와 공통 플랫폼 팀을 묶어 운영하는 TF의 작업공간. 본인이 리드한다.

## 한 문단 요약

2026년 현재 CD(Align Fail 자동 대응 Agent), Thickness(의뢰부터 Recipe 생성까지 ARC Pipeline), Inspection(ARC용 DB 구축, DRB 기반 ARC 생성)이 각자 ARC를 추진 중이고, 생성형 AI 모델에게 Recipe 작업 방식을 학습시키는 방법의 구체화와, Rule 기반 자동화, 이미지 처리(CV), ML/DL을 생성형 AI 모델과 함께 쓰는 구조가 공통 과제로 남았다. PoC(Align Fail 자동 대응)에서 Agent가 Recipe를 다루는 장비군 공통 방식 두 가지(VLM, PyAutoGUI, Win32를 조합한 Computer Use, 장비 밖에서 Recipe 파일을 편집하는 생성)를 확보했다. 동시에 현업 MI 엔지니어의 노하우를 AI가 쓰는 구조로 개발자가 구현하는 두 역할의 유기적인 협업이 필요하고, 조작, 모델 최적화와 운영, 녹화 DB 추출, 기술 부채 해소는 공통 플랫폼으로 풀고, ARC를 위한 효율적인 작업 방법(자동화 방법, 사내 DB 테이블 등)은 주간 미팅에서 공유해야 한다는 것도 배웠다. LLM 모델만으로는 정밀 좌표, 장비 절차, 반복 가능한 판정, 화면 조작을 해결하지 못하므로 그 바깥을 채우는 **공통 플랫폼**이 필요하다. 그래서 **공통 플랫폼 팀**이 엔진을 만들고 **장비군 담당**(MI 엔지니어)이 절차와 판정 기준을 맡는 TF를 꾸린다. 2027 상반기에는 1차 장비군 CD-SEM, Thickness, Inspection 위주로 장비군별 목표와 공통 플랫폼 베타까지 가고, 같은 기간 다른 MI 장비군은 장비 특징, 작업 절차, 필요한 기능을 파악한다. 6월 베타 배포 확인 뒤 하반기에 본격 참여한다. 엔지니어는 합류 시점마다 교육을 받으며 참여한다.

## 문서

| 순서 | 문서 | 독자 | 내용 | 상태 |
|---|---|---|---|---|
| 0 | [문제 정의](./00-problem-statement.md) | 임원, TF 전원 | 누가 무엇을 하려다 왜 막히는가, 한 문장 문제 정의와 판단 지표 | 초안, 담당 엔지니어 확인 전 |
| 1 | [TF 구성 기획안](./01-tf-proposal.md) | 임원 | 업무 전략(PoC에서 확인한 것), 선발대에서 확장으로, 교육하며 운영, 타임라인, 요청 사항 | 초안, 2026-12 보고용 (2026-11-09 흐름 변경) |
| 2 | [참여 팀과 역할](./02-roles-and-teams.md) | TF 전원 | 플랫폼 팀, 장비군 담당 구성, 장비군 담당 명단, 책임 분담표(RACI) | 초안 |
| 3 | [운영 방식](./03-operating-rhythm.md) | TF 전원 | 회의, 월간 보고 형식, 의사결정, 작업 관리, 안전 원칙 | 초안 |
| 4 | [주간 미팅과 노하우 공유](./04-onboarding.md) | 담당 엔지니어 | 주간 미팅 운영, 노하우 공유 주제, 기준선 측정 양식 | 초안 |
| 5 | [PoC 자산과 남은 과제](./05-poc-assets.md) | 플랫폼 팀, 임원 | PoC에서 가져오는 공통 자산, 시사점, 새로 풀어야 하는 것 | 초안 |
| 6 | [플랫폼 1차 PRD](./06-platform-prd.md) | 플랫폼 팀, 담당 엔지니어 | 무엇을 만드나: 사용자 스토리와 수용 기준, 성공, 가드레일 지표, 범위 밖, 열린 질문 | 초안, 일정 기획안과 맞춤 (2026-11-08) |
| 7 | [공통 플랫폼 기술 설계](./07-platform-architecture.md) | 플랫폼 팀, 담당 엔지니어 | 어떻게 만드나: Offline 지식화와 Online 실행, 판단과 규칙 확인과 실행 분리, 안전 관문, 평가, 단계별 구현 순서. Codex 설계 검토 결과 | 초안, 제안 설계 |
| 8 | [한 장 요약](./08-exec-summary.md) | 최고 임원 | 문제, 해결 방향, 단계별 타겟, 요청 사항 | 초안, 2026-12 최종 승인용 |
| - | [Codex 설계 원문](./codex/README.md) | 플랫폼 팀 | MP4 기반 Video-to-Workflow Agent 발표자료 22장의 본문과 이미지 (원문 보존, 수정하지 않음) | 보관 |
| - | [결정 기록](./decision-log.md) | 리드 | 날짜별 결정과 이유 | 2026-10-29 |
| - | [위험 목록](./risk-register.md) | 리드 | 위험, 대응, 담당 | 2026-10-29 |

## 최종 산출물 (docx, pptx)

독자별로 3종을 만든다. 팀원은 전체, 담당 임원(AI TF 담당 임원, 1차 승인)은 기획안, 최고 임원(기반기술센터장, 2차 최종 승인)은 한 장 요약이다. `.md`가 원본이고, 두 파일은 손으로 고치지 않고 다시 만든다. 필요 패키지: `python-docx`, `python-pptx`. 한글 폰트는 맑은 고딕.

| 독자 | docx | pptx |
|---|---|---|
| 팀원 (전체) | [arc-tf-team-full.docx](https://github.com/DarrenKoi/pm_notes/blob/main/my-task/arc-tf/deliverables/arc-tf-team-full.docx): 00, 01, 06, 07, 02~05, 위험 목록, 결정 기록 | [arc-tf-team-full.pptx](https://github.com/DarrenKoi/pm_notes/blob/main/my-task/arc-tf/deliverables/arc-tf-team-full.pptx): 16장 (담당 임원 deck + 기술 설계 5장 + LLM 한계 1장) |
| 담당 임원 | [arc-tf-proposal.docx](https://github.com/DarrenKoi/pm_notes/blob/main/my-task/arc-tf/deliverables/arc-tf-proposal.docx): 기획안(01) + 별첨 A 문제 정의(00) | [arc-tf-proposal.pptx](https://github.com/DarrenKoi/pm_notes/blob/main/my-task/arc-tf/deliverables/arc-tf-proposal.pptx): 10장 (추진 현황, 조직도, 업무 전략, 개발 방향, 효과, 일정, 위험, 결정 요청) |
| 최고 임원 | [arc-tf-exec-summary.docx](https://github.com/DarrenKoi/pm_notes/blob/main/my-task/arc-tf/deliverables/arc-tf-exec-summary.docx): 한 장 요약(08), 1쪽 | [arc-tf-exec-summary.pptx](https://github.com/DarrenKoi/pm_notes/blob/main/my-task/arc-tf/deliverables/arc-tf-exec-summary.pptx): 5장 |

```bash
python my-task/arc-tf/deliverables/build_docx.py          # 3종 모두, 하나만: team | boss | exec
python my-task/arc-tf/deliverables/build_pptx.py          # 3종 모두, 하나만: team | boss | exec
```

- docx는 `.md`를 그대로 변환한다. `.md`를 고치면 명령만 다시 실행한다.
- pptx는 01, 07, 08의 **요약**이라 내용이 `build_pptx.py` 안에 있다(슬라이드 하나 = 함수 하나, 맨 아래 `DECKS`가 deck별 순서). `.md`에서 숫자, 범위, 일정, 요청 사항을 바꾸면 그 스크립트의 슬라이드 내용도 같이 고친다.

## 일정

| 시기 | 단계 |
|---|---|
| 2026-11 | 기획안 작성 |
| 2026-12 | 임원 보고, TF 구성 승인 |
| 2027-01~02 | 선발대(CD-SEM, Thickness, Inspection) 주간 미팅 시작, 기준선 측정, 공통 플랫폼 설계 시작 |
| 2027-03~06 | 1차 장비군별 목표 (CD-SEM, Thickness, Inspection), 공통 플랫폼 설계 완성과 베타, 6월 목표 달성 확인 |
| 2027-07~09 | 2차 장비군(다른 MI 장비군) 합류(주간 미팅 참여), 선발대 절차화 |
| 2027-10~12 | 선발대 지식 자산, 돌파구 추천, 2차 화면 영역 지도와 절차 정의, 12월 목표 달성 확인 |

## 채워야 할 ⟨확인⟩ 목록

### 채운 것 (2026-11-02)

| 항목 | 결정 |
|---|---|
| 1차 장비군 (선발대) | CD-SEM, Thickness, Inspection (2026-11-09 변경, 2차: 다른 MI 장비군) |
| 참여 시간 | 정해 두지 않고 탄력적으로 운영 |
| 공통 플랫폼 팀 | 개발 약 5명 규모: 임원, 팀장급 지정 4명 (백엔드 2, 프론트엔드 2) + 1차 MI팀 현업 엔지니어, 본인(TF 리드)은 총괄 (2026-11-10 변경) |
| 보고 대상 임원 | AI TF 담당 임원(1차 승인) → 기반기술센터장(2차 최종 승인) |
| 비생산 검증 | 담당 엔지니어가 그때그때 판단해 실장비 테스트 |
| 화면 조작 보안 | 설치형 프로그램을 배포해 엔지니어 본인 PC에서 실행. 장비 PC에는 설치하지 않는다 |
| 접속 경로 | CD-SEM은 로컬 RCS, VeritySEM, Thickness는 웹 VDI → RCS/Radmin Viewer. 화면 캡처는 두 경로 모두 됨, 모니터는 대부분 FHD |
| 현재 셋업 시간 | 레시피당 평균 30~60분 (현업 경험치, 셋업 전체 기준) |

### 아직 남은 것

- ⟨확인: 선발대 CD-SEM, Thickness, Inspection 담당 참여 엔지니어 (2026-12), 2차 장비군 담당 (2026-12), 지정 개발 4명 (2026-12)⟩
- ⟨확인: 2차 MI 장비군 목록과 장비군별 접속 경로⟩
- ⟨확인: 장비 화면 녹화분 DB 직접 접근 가능 여부 (R&D MI H/W 담당자). 불가하면 RCS에서 GUI 자동화로 다운로드, 다운로드 자동화 작업 별도 필요⟩
- ⟨확인: 장비군별 자동화 대상 구간 실측, 기준선 측정(2027-02)⟩
- SecureVDI 경로 입력 가능 확인 (2026-11). ⟨확인: 로컬 RCS 경로 입력 주입 테스트, 저장 없음⟩
- ⟨확인: VDI 워터마크 여부⟩
- ⟨확인: 작업 관리 도구⟩

사내 수치, 실명, 장비 상세는 `*_사내용.md`(git 제외)에만 둔다.

## 휴대폰, 태블릿에서 읽기

[ARC TF 문서 모음 (claude.ai Artifact, 비공개)](https://claude.ai/artifact/PUq6fvLRzbF7sP6DciTovo), `reading.html`은 `python scripts/build_folder_html.py my-task/arc-tf`로 다시 만든다. 손으로 고치지 않는다.
