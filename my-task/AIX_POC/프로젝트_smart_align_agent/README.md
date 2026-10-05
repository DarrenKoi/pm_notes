---
tags: [aix, cd-sem, smart-align-agent, auto-recipe-creation, vlm, itc]
level: intermediate
last_updated: 2026-06-30
type: project-index
reviewed_on: 2026-10-04
review_status: partial
document_type: project_index
category_major: "업무 기획·산출물"
category_middle: "AIX 적용 사례"
category_minor: "기획·Discovery"
note_kind: "목차"
classified_on: "2026-10-05"
---

# Align Agent — CD-SEM Align Fail 대응 자동화

> [!note] 설계 당시 기록과 현재 확인의 경계
> 2026-06월 기획/설계 문맥을 보존한 문서다. VLM의 좌표 정밀도·실시간 동작·24시간 무중단·Skew Zero는 목표/가설이며 검증된 성과가 아니다. 회사 API 차단·모델 별칭·DRM 비율·GUI/좌표 계약은 현재 환경에서 미확인이다. 체크 완료와 `통과`는 문서 작성/설계 점검이며 실제 PoC·장비 검증/승인을 뜻하지 않는다. 원래 본문은 남기고 아래 검토 절에 현재 적용 조건을 추가했다.


> **Align Agent** = CD-SEM **Auto Recipe Creation**(셋업 ②~⑥) 중 **1차 PoC = Align Fail 대응 자동화**. VLM이 SEM 화면을 판독하고 GUI를 직접 제어해, Recipe Setup 후 **공정 variation으로 실측 이미지가 달라져 발생하는 좌표 shift · align 실패 · 오인식을 실시간 재정합**한다. ①(의뢰 검토)·⑦(최종 승인)은 사람이 유지 — **완전자동화가 아닌 Human-in-the-loop**.

이 폴더는 [`../_가이드/01·02`](../_가이드/README.md) 방법론 틀을 ITC AIX 실제 과제로 채운 **첫 번째 프로젝트 사례**다. 표준 근거는 [`../lectures/`](../lectures/README.md).

## 📑 문서 목록

| 문서 | 성격 | 내용 |
|------|------|------|
| [`03-적용_CDSEM기획.md`](./03-적용_CDSEM기획.md) | 기획 (Track A) | **Discovery** — End-to-End SOP · 망라 Pain · 정량 우선순위 선정. 1차 PoC를 ⑥(+③④) 최대 병목으로 좁힘 — align fail 대응은 ⑥ 오측/인식 실패 수정과 같은 개념이고 align image 재등록이 비교적 단순해 진입점으로 선택. |
| [`04-적용_CDSEM기술.md`](./04-적용_CDSEM기술.md) | 기술 (Track B) | **To-Be 설계** — AI 과제 정의서 · To-Be Swimlane(🤖/🧑) · KSF·제약 4측면. |
| [`05-적용_CDSEM_PoC실험설계.md`](./05-적용_CDSEM_PoC실험설계.md) | 기술 백업 | **KSF#1 검증** — VLM 좌표 재정합 정확도 PoC 실험 *설계*(04 Phase 0). 실행 아님. |
| [`07-적용_CDSEM_실행기획/`](./07-적용_CDSEM_실행기획/00-README.md) | 상세 | **실행기획 11단계 세트** — 전략일관성→…→KSF→개발일정→Review. Validation·Execution 포함, 내년 실행 가능 깊이. |
| [`07-적용_CDSEM_실행기획_발표요약.md`](./07-적용_CDSEM_실행기획_발표요약.md) / `.pptx` | 발표·요약 | 11단계를 **단계당 1장**(14장)으로 압축한 발표 세트. |
| `07-적용_CDSEM_실행기획_세트.pptx` | 발표·상세 | 11단계를 단계별 슬라이드(STEP 디바이더 + Why/What/How)로 변환한 76장 PPTX. |

## 🧭 읽는 순서

```
03 기획(Discovery)  →  04 기술(To-Be·KSF)  →  05 PoC 실험설계(KSF#1)  →  07 실행기획 세트(11단계 상세)
요약층(03·04)이 내러티브, 07 폴더가 단계별 산출물. 발표는 07 발표요약/세트 PPTX.
```

## 🔗 참조

- 방법론 틀: [`../_가이드/01-기획문서_AX서비스기획.md`](../_가이드/01-기획문서_AX서비스기획.md) · [`../_가이드/02-기술문서_AI과제정의구현.md`](../_가이드/02-기술문서_AI과제정의구현.md)
- 표준 강의 덱: [`../lectures/`](../lectures/README.md)
- 재사용 빈 양식: [`../_가이드/00-템플릿_AI과제발굴/`](../_가이드/00-템플릿_AI과제발굴/00-README.md)

---
*테스트 실행은 범위 밖 — 설계·계획까지만. 미확정 수치는 플레이스홀더(`<담당 임원>`, τ/s/r/N)로 둔다. 회사 기밀(구체 장비 수치)은 제외하고 방법론 적용 골격만 정리함.*

## 검토 결과와 후속 읽기 — 2026-10-04

03은 선정 이유,04는 역할/흐름,05는 핵심 가정만 격리하는 시험 설계여서 서로 대체하지 않는다. 07세트는 단계별 상세/발표 파생본이며 개별 검토는 진행 중이다. 2026-10-04 현재 이 저장소의 실행 결과·현재 장비 SW/모델·회사 승인을 확인하지 못했다.

[PPTX 발표 요약](./07-적용_CDSEM_실행기획_발표요약.pptx) · [상세 발표 세트](./07-적용_CDSEM_실행기획_세트.pptx)는 기존 첨부로 보호한다. 아래 추가 검토 절은 기존PPTX에 반영하지 않았고 재생성/재배포하지 않았다. [가이드 목차](../_가이드/README.md)와 [정리 기록](../../organization-log.md)을 함께 읽는다.

첨부 검증2026-10-04: PPTX ZIP의`ppt/slides/slideN.xml`개수는 발표 요약14개·상세 세트79개다. 당시 표의 상세76장 설명과 현재 첨부가 다르므로 현재 파일 수량은79장으로 읽는다. 원본PPTX는 수정하지 않았고 실제Office렌더링/내용 일치는 미검증이다.
