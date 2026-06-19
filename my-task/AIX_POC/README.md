# AIX POC — AX 서비스 기획 방법론

> SK Hynix "New AI Design Camp" 교육자료(2026-06-11) 캡처 10장을 전사·정리하고, 우리 부문 AIX POC 추진을 위한 기획/기술 문서로 재구성한 저장소.

## 📂 구성

| 문서 | 설명 |
|------|------|
| [01-기획문서_AX서비스기획.md](./01-기획문서_AX서비스기획.md) | **[틀]** Why/What/How — 3대 원칙, 4 Step, 12-Step, 과제 선정·연계 절차 |
| [02-기술문서_AI과제정의구현.md](./02-기술문서_AI과제정의구현.md) | **[틀]** AI 과제 정의서 · To-Be Swimlane · KSF/제약 · Check & Review |
| [03-적용_CDSEM기획.md](./03-적용_CDSEM기획.md) | **[적용]** CD-SEM Recipe Creation 자동화 — Track A Discovery: End-to-End SOP·망라 Pain·정량 우선순위 선정 (VLM GUI-제어) |
| [04-적용_CDSEM기술.md](./04-적용_CDSEM기술.md) | **[적용]** CD-SEM To-Be 설계·구현 — Track B (과제 정의서·Swimlane·KSF·제약) |
| [05-적용_CDSEM_PoC실험설계.md](./05-적용_CDSEM_PoC실험설계.md) | **[적용]** KSF#1 좌표 재정합 정확도 PoC 실험 설계 (04 Phase 0 검증) |
| [06-적용_CDSEM_DesignCamp장표.md](./06-적용_CDSEM_DesignCamp장표.md) | **[발표]** 03~05를 캠프 12-Step 템플릿 장표로 채운 발표 슬라이드 세트(장표 0~14: SOP→Pain→정량선정→Track B + 발표 스크립트) |
| `06-적용_CDSEM_DesignCamp장표.pptx` | **[발표]** 위 장표를 변환한 편집 가능한 PPTX(네이티브 표·발표자 노트). 생성: `python tools/md2pptx.py <md> <pptx>` |
| [`source/`](./source/) | 캡처 10장 원문 전사 (faithful transcription) |
| [`screenshots/`](./screenshots/) | 원본 이미지 (`.jpg`는 git 미추적) |

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

## 📑 원문 인덱스 (source/)

| # | 파일 | 내용 |
|---|------|------|
| 01 | [kakao-3step-pain-point](./source/01-kakao-3step-pain-point.md) | SOP→Pain Point→과제 선정 3-STEP |
| 02 | [ax-consulting-methodology](./source/02-ax-consulting-methodology.md) | Baseline 3원칙 + 4 Step |
| 03 | [design-camp-12steps](./source/03-design-camp-12steps.md) | Design Camp 12단계 |
| 04 | [org-goal-task-linkage](./source/04-org-goal-task-linkage.md) | 조직목표↔과제 연계 예시 |
| 05 | [process-level-definition](./source/05-process-level-definition.md) | L1~L5 프로세스 레벨 정의(영업) |
| 06 | [sales-ai-opportunity](./source/06-sales-ai-opportunity.md) | 영업 AI 적용 기회(Agent#1~4) |
| 07 | [ai-agent-scope-template](./source/07-ai-agent-scope-template.md) | AI Agent 대상영역 분석 Template |
| 08 | [final-output-example](./source/08-final-output-example.md) | Pain Point→근본원인→해결아이디어 |
| 09 | [ai-task-definition-template](./source/09-ai-task-definition-template.md) | AI 과제 정의서 양식 |
| 10 | [check-and-review](./source/10-check-and-review.md) | To-Be 용어 정의 & 점검 포인트 |

---
*출처: SK Hynix New AI Design Camp 내부 교육자료. 회사 기밀에 해당하는 구체 수치·시스템 상세는 제외하고 방법론 골격만 정리함.*
