---
tags: [aix, design-camp, captures, lecture]
level: reference
last_updated: 2026-06-30
type: source-reference
reviewed_on: 2026-10-04
review_status: reviewed_with_limits
document_type: source_index
category_major: "업무 기획·산출물"
category_middle: "AIX 방법론"
category_minor: "강의·원본 양식"
note_kind: "목차"
classified_on: "2026-10-05"
---

# lectures/ — New AI Design Camp 강의자료 (V2.7)

> 내부 교육 V2.7 텍스트 추출본과 초기 캡처 전사다. 교육 방법론의 출처를 추적하는 자료이며, 현재 기술 규격·회사 승인·실행 결과를 증명하는 자료는 아니다.

## 파일 (원본 → 추출 md)

기존 기록에는 원본 PDF·PPTX를 추출 후 삭제했다고 적혀 있다. 2026-10-04 이 주제 폴더의 실제 파일 목록에는 아래 이름의 원본이 없으므로 원본 대비 누락·읽기 순서·시각 정보를 재검증할 수 없다. 추출본은 텍스트 학습·출처 추적용이며 원본 바이너리의 완전한 대체물로 단정하지 않는다.

| 추출 md | 기록된 원본 이름 (현재 미발견) | 내용 |
|---------|------|------|
| [`design-camp-deck-v2.7.md`](./design-camp-deck-v2.7.md) | `SKHY New AI Design Camp_V2.7 (배포)_A.pdf` (136p, Day1·Day2) | **강의 본 덱 텍스트 추출**. Agenda, Baseline 3원칙, 4 Step 방법론 상세, 12-Step Focusing Point, 7개 도메인 Process Level 예시, M1~M9 수행방법·Check&Review·산출물 예시, Quiz·Facilitation 가이드, Wrap up(Agentic AI 3관점·Long/Short List 포트폴리오) |
| [`design-camp-template-v2.7.md`](./design-camp-template-v2.7.md) | `SKHY The New Design Camp_Template V2.7.pptx` (25장) | **편집용 빈 양식의 텍스트·표 구조**. M1~M9 모듈별 작성 Template 칸 구조(연계표·스코어카드·정의서·As-Is/To-Be·KSF·일정) |

> [!warning] 원문과 추출 기록의 경계
> `pypdf` 132/136쪽 추출·이미지 전용4쪽 생략, `python-pptx` 텍스트/표 추출·노트 없음은 기존 추출 기록이다. 원본 부재로 생략된 이미지의 정보량·노트·기밀 제거의 완전성은 미확인이다. 연락처·배포 지침은 교육 당시 기록이며 현재 이용 안내로 재배포하지 않는다. 원문 추출2개와 캡처10개는 불변 출처로 그대로 보존하고 정정/의문은 이 목차와 정리 기록에 남긴다.

## 읽기 순서와 적용 조건

1. [강의 덱](./design-camp-deck-v2.7.md)에서 4단계 기획과 M1~M9 산출물의 목적을 읽는다.
2. [빈 템플릿 추출](./design-camp-template-v2.7.md)에서 작성 칸을 확인한다. Markdown을 편집 원본 PPTX로 간주하지 않는다.
3. [초기 캡처 목차](./captures/README.md)에서 요약 도식의 출처를 대조한다.
4. [가이드 목차](../_가이드/README.md)로 이동해 재사용 방법론과 실제 입력 양식을 구분한다.

본문의 `전문`·`완본`·`삭제 가능`은 당시 추출 작성자의 표현이다. 원본 삭제를 권고하지 않는다. 영업 L1~L5와 덱의 L4 중심 분해는 문맥별 표기이므로 동일 명칭을 기계적으로 강제하지 않는다. M1~M9 모듈과 12-Step 번호도 일대일 대응이 아니다. 퀴즈 답 해석·빨간 스티커 우선순위·3-Whys 결과는 작성자의 해석/가설과 근거를 구분한다.

## `captures/`(10캡처)와의 관계

- `captures/01~10` = 초기 캡처 10장의 **불변 전사**. 그대로 유지한다.
- `lectures/` = 동일 방법론의 **범위가 더 넓은 배포 덱의 텍스트 추출본**. 캡처본에 없던 다음 내용을 담고 있어 틀 문서 보강의 근거가 된다.

## 캡처본 대비 신규·심화 내용과 기존 가이드 참조

| 영역 | 신규/심화 내용 | 반영 위치 |
|------|----------------|-----------|
| **Step3. Validation** | 비즈니스 임팩트(수익성·확장성·시급성) 정량화, CapEx/OpEx, **ROI 공식·NPV·투자회수기간**, 도입전략 3대 질문 | [01 §Validation](../_가이드/01-기획문서_AX서비스기획.md) |
| **Step4. Execution** | **PoC→MVP→전사확산 3-Phase**, 이해관계자(영향력×관심도) 분석, **변화관리 저항요인 3종 대응**, 전문조직 육성 | [01 §Execution](../_가이드/01-기획문서_AX서비스기획.md) |
| **M1 후보 선정** | 자동화 적합성 6기준(디지털화·병목·반복성·가치·오류허용·규칙기반) | [01 §후보 선정](../_가이드/01-기획문서_AX서비스기획.md) |
| **M3 Pain Point** | 정량화 6관점, 판단 오류 3유형, Criticality 4기준 | [01 §Pain Point](../_가이드/01-기획문서_AX서비스기획.md) |
| **M4 근본원인** | **근본원인 6유형**(프로세스·시스템·데이터·사람·정책·외부), 해결 아이디어 6유형 | [01 §근본 원인](../_가이드/01-기획문서_AX서비스기획.md) |
| **M5 적정성** | **7항목 스코어카드(3점 척도/21점)** + 항목별 채점 루브릭 | [01 §적정성](../_가이드/01-기획문서_AX서비스기획.md) |
| **Process 체계** | Decomposition vs Map vs SOP, **Agentic AI 3관점 매핑**(LLM 행동지침·Tool·Guardrail) | [02 §Process 체계](../_가이드/02-기술문서_AI과제정의구현.md) |
| **Modeling Rule** | Task = Input/Output/**Constraint**/Mechanism (IDEF0형), BPMN Notation | [02 §Modeling Rule](../_가이드/02-기술문서_AI과제정의구현.md) |
| **적용 AI 기술** | 생성형(LLM)·분석형(ML/DL)·데이터변환 **기술 분류 체계** | [02 §적용 AI 기술](../_가이드/02-기술문서_AI과제정의구현.md) |
| **M9 일정** | LLM/ML-DL **트랙별 개발 Task**, 4구분(프로세스·데이터·AI모델·Agentic Prototype) | [02 §개발 일정](../_가이드/02-기술문서_AI과제정의구현.md) |
| **포트폴리오** | Long List→Short List→Phased Plan 우선순위 기준(가치·실행가능성·시급도·리스크) | [01 §Execution](../_가이드/01-기획문서_AX서비스기획.md) |

---
출처와 추출 범위는 각 파일의 `source`와 서두에 남아 있다. 확인일2026-10-04. 원본 재확인·Claude 협의·Obsidian 읽기 화면 확인은 미완료다. 개별 원문의 보존/검토 결과는 [정리 기록](../../organization-log.md)에 남겼다.
