---
tags: [aix, design-camp, source, lecture]
level: reference
last_updated: 2026-06-30
type: source-reference
---

# lectures/ — New AI Design Camp 원본 강의자료 (V2.7)

> SK Hynix "New AI Design Camp" 2026년 배포본 전체 강의 덱. `source/`(캡처 10장 전사)보다 **상위·완본** 출처이며, 방법론 틀 문서(`01`·`02`)의 근거로 사용한다.

## 파일

| 파일 | 성격 |
|------|------|
| `SKHY New AI Design Camp_V2.7 (배포)_A.pdf` | **강의 본 덱 (136p, Day1·Day2)**. 4 Step 방법론 상세, 12-Step Focusing Point, M1~M9 모듈별 수행방법·Check&Review·예시, Quiz·Facilitation 가이드 포함 |
| `SKHY The New Design Camp_Template V2.7.pptx` | **편집 가능 빈 양식 25장**. M1~M9 모듈별 작성 Template (연계표·스코어카드·정의서·As-Is/To-Be·KSF·일정) |

## `source/`(10캡처)와의 관계

- `source/01~10` = 초기 캡처 10장의 **불변 전사**. 그대로 유지한다.
- `lectures/` = 동일 방법론의 **완본 배포 덱**. 캡처본에 없던 다음 내용을 담고 있어 틀 문서 보강의 근거가 된다.

## 캡처본 대비 신규·심화 내용 (틀 문서에 반영함)

| 영역 | 신규/심화 내용 | 반영 위치 |
|------|----------------|-----------|
| **Step3. Validation** | 비즈니스 임팩트(수익성·확장성·시급성) 정량화, CapEx/OpEx, **ROI 공식·NPV·투자회수기간**, 도입전략 3대 질문 | [01 §Validation](../01-기획문서_AX서비스기획.md) |
| **Step4. Execution** | **PoC→MVP→전사확산 3-Phase**, 이해관계자(영향력×관심도) 분석, **변화관리 저항요인 3종 대응**, 전문조직 육성 | [01 §Execution](../01-기획문서_AX서비스기획.md) |
| **M1 후보 선정** | 자동화 적합성 6기준(디지털화·병목·반복성·가치·오류허용·규칙기반) | [01 §후보 선정](../01-기획문서_AX서비스기획.md) |
| **M3 Pain Point** | 정량화 6관점, 판단 오류 3유형, Criticality 4기준 | [01 §Pain Point](../01-기획문서_AX서비스기획.md) |
| **M4 근본원인** | **근본원인 6유형**(프로세스·시스템·데이터·사람·정책·외부), 해결 아이디어 6유형 | [01 §근본 원인](../01-기획문서_AX서비스기획.md) |
| **M5 적정성** | **7항목 스코어카드(3점 척도/21점)** + 항목별 채점 루브릭 | [01 §적정성](../01-기획문서_AX서비스기획.md) |
| **Process 체계** | Decomposition vs Map vs SOP, **Agentic AI 3관점 매핑**(LLM 행동지침·Tool·Guardrail) | [02 §Process 체계](../02-기술문서_AI과제정의구현.md) |
| **Modeling Rule** | Task = Input/Output/**Constraint**/Mechanism (IDEF0형), BPMN Notation | [02 §Modeling Rule](../02-기술문서_AI과제정의구현.md) |
| **적용 AI 기술** | 생성형(LLM)·분석형(ML/DL)·데이터변환 **기술 분류 체계** | [02 §적용 AI 기술](../02-기술문서_AI과제정의구현.md) |
| **M9 일정** | LLM/ML-DL **트랙별 개발 Task**, 4구분(프로세스·데이터·AI모델·Agentic Prototype) | [02 §개발 일정](../02-기술문서_AI과제정의구현.md) |
| **포트폴리오** | Long List→Short List→Phased Plan 우선순위 기준(가치·실행가능성·시급도·리스크) | [01 §Execution](../01-기획문서_AX서비스기획.md) |

---
*출처: SK Hynix New AI Design Camp V2.7 배포본(2026). 회사 기밀(구체 장비 수치·내부 시스템 상세)은 방법론 골격만 추출하고 제외함. PDF 텍스트 추출은 `pip install pypdf` 후 `PdfReader.extract_text()` 사용.*
