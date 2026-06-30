# AIX POC — AX 서비스 기획 방법론 + 부문 프로젝트

> SK Hynix "New AI Design Camp" 방법론을 **표준**으로 고정하고, 그 위에 부문 AIX 과제를 **프로젝트 폴더 단위**로 쌓는 저장소.

## 📂 구성 (3축)

| 폴더 | 축 | 설명 |
|------|----|------|
| [`lectures/`](./lectures/README.md) | **표준** | New AI Design Camp **V2.7 완본 강의 덱**(`design-camp-deck-v2.7.md`) + 편집 Template(`design-camp-template-v2.7.md`) + [`captures/`](./lectures/captures/)(초기 캡처 10장 전사). **팀을 가이드하는 기준점** — 모든 틀·적용 문서의 1차 근거. |
| [`_가이드/`](./_가이드/) | **가이드** | 재사용 방법론 틀: [`01-기획문서_AX서비스기획.md`](./_가이드/01-기획문서_AX서비스기획.md)(Why/What/How · 4 Step · 12-Step Track A · 적정성 스코어카드 · Validation/Execution) · [`02-기술문서_AI과제정의구현.md`](./_가이드/02-기술문서_AI과제정의구현.md)(과제 정의서 · To-Be Swimlane · KSF/제약 · Modeling Rule · 일정 트랙) · [`00-템플릿_AI과제발굴/`](./_가이드/00-템플릿_AI과제발굴/00-README.md)(도메인 중립 빈 양식 + 퍼실리테이션 키트). |
| [`프로젝트_smart_align_agent/`](./프로젝트_smart_align_agent/README.md) | **프로젝트** | **Align Agent** — CD-SEM Auto Recipe Creation 중 1차 PoC(**Align Fail 대응 자동화**). Discovery(03)·기술 To-Be(04)·PoC 실험설계(05)·실행기획 11단계 세트(07)·발표물. |
| `tools/` | — | 마크다운 → PPTX 변환기(`md2pptx.py`, `md2pptx_doc.py`). |

## 🧩 프로젝트 폴더 컨벤션

앞으로 신규 AIX 과제는 **`프로젝트_<이름>/` 폴더**로 추가한다.

1. `lectures/`(표준)와 `_가이드/01·02`(틀)를 먼저 읽고,
2. 틀의 섹션 구조(Why→What→How, 산출물 체크리스트)를 **거울처럼 미러링**해 실제 값으로 채운다.
3. 폴더 안에 `README.md`(프로젝트 인덱스)를 두어 문서 목록·읽는 순서를 안내한다.

> 표준(`lectures`)·틀(`_가이드`)은 모든 프로젝트가 공유하고, 과제별 내용은 각 `프로젝트_*/` 안에만 둔다 — 경계를 섞지 않는다.

## 🧭 핵심 흐름 한눈에

```
Baseline 3원칙 (가치·데이터·인간 중심)
   │
4 Step:  Discovery → Design → Validation → Execution
   │
12-Step Design Camp
   ├─ Track A. Idea 도출   (1~7): 업무정의→Scope→PainPoint→근본원인→Ideation→적정성
   └─ Track B. 실행 구체화 (8~12): 과제정의→As-Is→To-Be→KSF→일정
   │
산출물: 과제 정의서 · To-Be Swimlane(AI/Human) · Pain→근본원인→해결아이디어
```

## 📑 원문 인덱스 (lectures/captures/)

| # | 파일 | 내용 |
|---|------|------|
| 01 | [kakao-3step-pain-point](./lectures/captures/01-kakao-3step-pain-point.md) | SOP→Pain Point→과제 선정 3-STEP |
| 02 | [ax-consulting-methodology](./lectures/captures/02-ax-consulting-methodology.md) | Baseline 3원칙 + 4 Step |
| 03 | [design-camp-12steps](./lectures/captures/03-design-camp-12steps.md) | Design Camp 12단계 |
| 04 | [org-goal-task-linkage](./lectures/captures/04-org-goal-task-linkage.md) | 조직목표↔과제 연계 예시 |
| 05 | [process-level-definition](./lectures/captures/05-process-level-definition.md) | L1~L5 프로세스 레벨 정의(영업) |
| 06 | [sales-ai-opportunity](./lectures/captures/06-sales-ai-opportunity.md) | 영업 AI 적용 기회(Agent#1~4) |
| 07 | [ai-agent-scope-template](./lectures/captures/07-ai-agent-scope-template.md) | AI Agent 대상영역 분석 Template |
| 08 | [final-output-example](./lectures/captures/08-final-output-example.md) | Pain Point→근본원인→해결아이디어 |
| 09 | [ai-task-definition-template](./lectures/captures/09-ai-task-definition-template.md) | AI 과제 정의서 양식 |
| 10 | [check-and-review](./lectures/captures/10-check-and-review.md) | To-Be 용어 정의 & 점검 포인트 |

> 완본 강의 덱은 [`lectures/`](./lectures/README.md) 참조. `captures/`는 초기 캡처본이며 `lectures/`가 상위·완본 출처다.

---
*출처: SK Hynix New AI Design Camp 내부 교육자료. 회사 기밀에 해당하는 구체 수치·시스템 상세는 제외하고 방법론 골격만 정리함.*
