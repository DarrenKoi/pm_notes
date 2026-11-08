---
tags: [my-task, arc-tf, auto-recipe-creation, computer-use, index]
category_major: "업무 기획·산출물"
category_middle: "ARC TF (Auto Recipe Creation)"
category_minor: "전체 목차"
note_kind: "목차"
classified_on: "2026-10-29"
last_updated: 2026-11-08
---

# ARC TF: 레시피 자동 생성 태스크포스

> 장비 레시피(Recipe) 셋업을 컴퓨터 사용(Computer-use) Agent로 자동화하려고, MI 현업 엔지니어와 공통 플랫폼 팀을 묶어 운영하는 TF의 작업공간. 본인이 리드한다.

## 한 문단 요약

레시피 셋업은 숙련 엔지니어가 원격 제어 화면에서 손으로 하는 반복 작업이고, 장비가 늘면 사람도 그만큼 늘어야 한다. 1차 PoC(Align Fail 대응 자동화)에서 사내 VLM으로 API 없는 장비 화면을 보고 조작할 수 있다는 것, VLM과 CV를 나눠 쓰면 오프라인 평가에서 정밀 좌표도 잡힌다는 것을 확인했다. 남은 문제는 두 가지다. 하나는 현장 신뢰성과 실행 엔진 제품화이고, 다른 하나는 장비군마다 다른 레시피 절차다. 뒤의 것은 현업 지식 없이는 풀 수 없다. 그래서 **공통 플랫폼 팀**이 엔진을 만들고, **장비군 담당**(MI 엔지니어)이 절차와 판정 기준을 맡는 TF를 꾸린다. 2026-11에 기획안을 쓰고, 12월에 임원 승인을 받고, 2027-01에 가동한다.

## 문서

| 순서 | 문서 | 독자 | 내용 | 상태 |
|---|---|---|---|---|
| 0 | [문제 정의](./00-problem-statement.md) | 임원, TF 전원 | 누가 무엇을 하려다 왜 막히는가, 한 문장 문제 정의와 판단 지표 | 초안, 담당 엔지니어 확인 전 |
| 1 | [TF 구성 기획안](./01-tf-proposal.md) | 임원 | 왜 해야 하는가, PoC 성과, 범위, TF 구성, 타임라인, 요청 사항 | 초안, 2026-12 보고용 |
| 2 | [참여 팀과 역할](./02-roles-and-teams.md) | TF 전원 | 플랫폼 팀, 장비군 담당 구성, 장비군 담당 명단, 책임 분담표(RACI) | 초안 |
| 3 | [운영 방식](./03-operating-rhythm.md) | TF 전원 | 회의, 월간 보고 형식, 의사결정, 작업 관리, 안전 원칙 | 초안 |
| 4 | [온보딩 학습 과정](./04-onboarding.md) | 담당 엔지니어 | 2027-01~02 8주 과정, 수료 기준 | 초안 |
| 5 | [PoC 자산과 남은 과제](./05-poc-assets.md) | 플랫폼 팀, 임원 | Align Fail PoC에서 가져오는 것, 새로 풀어야 하는 것 | 초안 |
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
| 팀원 (전체) | [arc-tf-team-full.docx](https://github.com/DarrenKoi/pm_notes/blob/main/my-task/arc-tf/deliverables/arc-tf-team-full.docx): 00, 01, 06, 07, 02~05, 위험 목록, 결정 기록 | [arc-tf-team-full.pptx](https://github.com/DarrenKoi/pm_notes/blob/main/my-task/arc-tf/deliverables/arc-tf-team-full.pptx): 21장 (담당 임원 deck + 기술 설계 7장) |
| 담당 임원 | [arc-tf-proposal.docx](https://github.com/DarrenKoi/pm_notes/blob/main/my-task/arc-tf/deliverables/arc-tf-proposal.docx): 기획안(01) + 별첨 A 문제 정의(00) | [arc-tf-proposal.pptx](https://github.com/DarrenKoi/pm_notes/blob/main/my-task/arc-tf/deliverables/arc-tf-proposal.pptx): 14장 (공통 플랫폼 구조 1장 포함) |
| 최고 임원 | [arc-tf-exec-summary.docx](https://github.com/DarrenKoi/pm_notes/blob/main/my-task/arc-tf/deliverables/arc-tf-exec-summary.docx): 한 장 요약(08), 1쪽 | [arc-tf-exec-summary.pptx](https://github.com/DarrenKoi/pm_notes/blob/main/my-task/arc-tf/deliverables/arc-tf-exec-summary.pptx): 4장 |

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
| 2027-01~02 | 온보딩 학습, 장비군별 기준선 측정 |
| 2027-03~06 | 1차 구축 (1차 장비군 3개: CD-SEM, VeritySEM, Thickness) |
| 2027-07~09 | 파일럿 |
| 2027-10~12 | 2차 지정군(Inspection, 선택) 적용 준비 |

## 채워야 할 ⟨확인⟩ 목록

### 채운 것 (2026-11-02)

| 항목 | 결정 |
|---|---|
| 1차 장비군 | CD-SEM(리드 직접 담당), VeritySEM, Thickness |
| 겸직 시간 | 주 5시간 (근무일마다 1시간) |
| 공통 플랫폼 팀 | 개발 2명 (본인 포함) |
| 보고 대상 임원 | AI TF 담당 임원(1차 승인) → 기반기술센터장(2차 최종 승인) |
| 비생산 검증 | 장비 사용 인가를 받아 하루 1시간 실장비 테스트 |
| 화면 조작 보안 | 설치형 프로그램을 배포해 엔지니어 본인 PC에서 실행. 장비 PC에는 설치하지 않는다 |
| 접속 경로 | CD-SEM은 로컬 RCS, VeritySEM, Thickness는 웹 VDI → RCS/Radmin Viewer. 화면 캡처는 두 경로 모두 됨, 모니터는 대부분 FHD |
| 현재 셋업 시간 | 레시피당 평균 30~60분 (현업 경험치, 셋업 전체 기준) |

### 아직 남은 것

- ⟨확인: VeritySEM, Thickness 담당 참여 엔지니어⟩
- ⟨확인: 장비 화면 녹화분 DB 직접 접근 가능 여부 (R&D MI H/W 담당자). 불가하면 RCS에서 GUI 자동화로 다운로드, 다운로드 자동화 작업 별도 필요⟩
- ⟨확인: 장비군별 자동화 대상 구간 실측, 온보딩 8주차⟩
- SecureVDI 경로 입력 가능 확인 (2026-11). ⟨확인: 로컬 RCS 경로 입력 주입 테스트, 저장 없음⟩
- ⟨확인: VDI 워터마크 여부⟩
- ⟨확인: 작업 관리 도구⟩

사내 수치, 실명, 장비 상세는 `*_사내용.md`(git 제외)에만 둔다.

## 휴대폰, 태블릿에서 읽기

[ARC TF 문서 모음 (claude.ai Artifact, 비공개)](https://claude.ai/artifact/PUq6fvLRzbF7sP6DciTovo), `reading.html`은 `python scripts/build_folder_html.py my-task/arc-tf`로 다시 만든다. 손으로 고치지 않는다.
