---
tags: [aix, cd-sem, recipe-creation, vlm, computer-use, design-camp, itc]
level: intermediate
last_updated: 2026-06-19
type: 적용사례-기획
---

# AIX 적용 사례 — CD-SEM Recipe Creation 자동화 (기획/Discovery)

> [기획 방법론 틀(01)](./01-기획문서_AX서비스기획.md)을 ITC AIX 실제 과제에 적용한 **첫 번째 사례**. New AI Design Camp Track A(Discovery, 1~7단계)를 따라 **CD-SEM Recipe Creation 자동화** 과제를 발굴·구조화한다. 실행/To-Be 설계(Track B)는 후속 문서 `04-적용_CDSEM기술.md`에서 다룬다.

## 과제 한 줄 정의

**VLM 기반 GUI 제어(computer-use)로 CD-SEM Recipe Creation을 자동화한다.** VLM이 SEM 화면을 직접 보며 GUI를 클릭·입력해 Recipe를 생성하고, **실측 시작 시 발생하는 좌표 shift·align 실패·패턴 오인식을 실시간으로 재정합·수정**한다. 엔지니어는 모니터링하다 **예외만 개입(Human-in-the-loop)** 한다. 완전자동화는 목표가 아니다.

## 왜 이 과제인가? (Why)

상위 방법론의 3대 원칙([원문 02](./source/02-ax-consulting-methodology.md))에 비춰 이 과제가 적합한 이유:

| 원칙 | 이 과제에서의 충족 |
|------|--------------------|
| **가치 중심** | 신규 공정 셋업 **리드타임 단축**이라는 조직 목표에 직결. 최대 병목(오인식 수정 반복)을 정조준. |
| **데이터 중심** | SEM 화면 이미지·reference recipe 등 **이미 존재하는 가용 데이터**로 동작. 추측 설계가 아님. |
| **인간 중심** | 엔지니어를 대체하지 않고 **확장** — 반복 셋업은 AI, 예외 판단·최종 검증은 사람. |

특히 이 과제는 사람의 작업시간이라는 물리적 제약을 풀어 **24h 무중단 셋업**을 가능케 한다는 점에서, 단순 효율화를 넘어 처리량의 상한 자체를 끌어올린다.

## 핵심 구조 (What)

### 1. 조직 목표 ↔ 과제 연계 (Top-Down)

과제는 개인 흥미가 아니라 조직 목표에서 연역한다([원문 04](./source/04-org-goal-task-linkage.md) 양식 적용).

```
조직 목표            →  업무 목표                    →  핵심 업무            →  AI 대상 후보 과제
신규 공정 셋업          MI Recipe Setup 처리량 확대       CD-SEM Recipe           VLM GUI-제어 기반
리드타임 단축           + 숙련도 편차 제거               Creation                Recipe 자동 생성
(핵심)                  + 24h 무중단 셋업                                        + 실측 좌표/align 자동 보정
```

### 2. 업무 분해 — L1~L5 (AI 배치 지점)

업무를 5단계로 분해해 AI 개입 최소 단위(L5 Task)까지 내려간다([원문 05](./source/05-process-level-definition.md) 체계 적용).

```
L1 Mega Process : 계측(MI) 운영
  L2 Process Group : CD-SEM Recipe Management
    L3 Process : Recipe Creation
      L4 Activity : As-Is 프로세스 ①~⑦ (아래)
        L5 Task (AI 배치 지점) : ③ 패턴 등록 · ④ 좌표·박스 설정 · ⑥ 오인식 수정
```

## 발굴 절차 (How)

### Phase 1. As-Is 프로세스 정의 → 병목 식별

CD-SEM Recipe Creation의 실제 As-Is 흐름(엔지니어 확인 완료):

```
① 측정 대상 레이어/패턴 확인 (도면·스펙 검토)
② 장비 GUI에서 신규 Recipe 생성 → 웨이퍼/샘플 로드
③ Alignment / Addressing 포인트 등록 (패턴 인식 템플릿 지정)   ← 병목
④ 측정 포인트(EP) 좌표·측정박스·측정 알고리즘 설정              ← 병목
⑤ 이미징 조건(배율·전압·FOV 등) 설정
⑥ 시험 측정(Try) → 오인식/실패 지점 수정 (반복)                ← 최대 병목
⑦ Recipe 저장·등록 → 검증
```

- **병목 = ③④⑥**, 그중 **⑥ 오인식 수정 반복이 최대 고통**.
- 입력물: 공정/소자 엔지니어가 **MIDAS** 시스템에 의뢰서 작성(현재 PPT 형태: 도면·측정위치·파라미터·측정방식). MIDAS 전산화는 **별도 프로젝트에서 진행 중**이며 본 과제의 외부 변수다.
- ⑤ 이미징 조건은 reference recipe를 참고(동일)하거나, 신규 공정은 엔지니어 협의로 결정한다.

### Phase 2. Pain Point → 근본 원인(3-Why) → 해결 아이디어

Discovery의 핵심 산출물([원문 08](./source/08-final-output-example.md) 양식 적용). Pain Point에서 멈추지 않고 근본 원인을 파고들어 해결 아이디어로 연결한다.

| Pain Point | 근본 원인 (3-Why) | 해결 아이디어 |
|------------|-------------------|----------------|
| 오인식 수정 반복(⑥)이 오래 걸림 | 템플릿을 미리 만들어도 **실측 시작 시 전체 좌표 shift / align 실패** 발생 | **VLM이 SEM 이미지를 보고 실시간 좌표 재정합·재align, 오인식 자동 수정** (핵심) |
| 숙련도별 해석 편차 | 표준 판단기준 부재 + 의뢰자가 의뢰서를 명확히 작성하지 않는 경우 | VLM이 일관된 기준으로 수행 + 의뢰서 부족정보 플래깅 (MIDAS 전산화와 연계) |
| 야간/주말 셋업 공백 | 사람 작업시간이라는 물리적 제약 | VLM 24h 무중단 수행, 사람은 예외만 개입 |

> 집중 근본 원인 = **(a) 패턴 인식 실패, (b) 측정박스/좌표 오류.** 해결 아이디어 전부가 이 둘로 수렴한다.

### Phase 3. 적정성 검토 (Impact Validation)

과제 선정 3-STEP([원문 01](./source/01-kakao-3step-pain-point.md))의 교집합 기준 — **아프고 · 만들 수 있고 · 남는 게 있는가**:

- **고통(아프다)**: ⑥ 오인식 수정은 빈도·공수·숙련 의존도 모두 높은 최대 병목 → ✅
- **구현 난이도(만들 수 있다)**: 사내 VLM(Qwen3-VL-30B)으로 SEM 화면 판독 + GUI 제어 가능 범위. 단 정밀 좌표 재정합 정확도가 핵심 리스크 → ⚠️ (Track B에서 검증)
- **효용(남는다)**: 리드타임 단축 + 24h 무중단으로 처리량 상한 돌파 → ✅

### Phase 4. To-Be 방향성 (Human-AI 경계 — 개념 수준)

상세 Swimlane은 Track B(`04`)에서 그리되, 발굴 단계에서 합의된 역할 경계는 다음과 같다.

- 🤖 **AI**: ②~⑥ GUI 직접 조작 + SEM 이미지 판독 기반 좌표/align 재정합 + 오인식 수정.
- 🧑 **Human**: ① 의뢰서 검토(필요시), 예외 핸드오프 처리, ⑦ 최종 검증·승인.

## 기술/환경 제약 (기획 반영, 상세는 Track B)

- **외부 LLM API 전면 차단.** 사내 OpenAI-호환 내부 엔드포인트만 사용 — **Qwen3-VL-8B / Qwen3-VL-30B(비전)**, Kimi-K2.5(텍스트), BGE-M3(임베딩).
- DB(OpenSearch/ES)는 로컬 미가용 — 검증은 import/문법 수준.
- 99% 사내 문서가 DRM → 스크린샷+VLM 추출이 기본 파이프라인.
- **MIDAS 전산화는 외부 변수** — 본 과제는 의뢰서 정보 부족을 플래깅하는 선에서 연계.

## Discovery 산출물 체크리스트

- [x] 조직 목표 ↔ AI 대상 후보 과제 연계표 (What §1)
- [x] L1~L5 프로세스 레벨 정의 (What §2)
- [x] As-Is 프로세스 + 병목 식별 (Phase 1)
- [x] Pain Point → 근본 원인 → 해결 아이디어표 (Phase 2)
- [x] 적정성 검토 — 고통/난이도/효용 교집합 (Phase 3)
- [x] AI 과제 정의서 (1매) → [Track B `04`](./04-적용_CDSEM기술.md) Step 8
- [x] To-Be Swimlane(🤖 AI / 🧑 Human) → [Track B `04`](./04-적용_CDSEM기술.md) Step 9~10
- [x] KSF·제약 4측면(기술/조직/비용/정책·보안) → [Track B `04`](./04-적용_CDSEM기술.md) Step 11

## 참고 자료 (References)
- 방법론 틀: [01-기획문서_AX서비스기획.md](./01-기획문서_AX서비스기획.md)
- 후속 기술 설계: [04-적용_CDSEM기술.md](./04-적용_CDSEM기술.md) (Track B)
- 원문 전사: [`source/`](./source/) (특히 01·04·05·08)
- 출처: SK Hynix New AI Design Camp 방법론 + ITC AIX 브레인스토밍 합의(2026-06-19). 회사 기밀(구체 장비 수치·시스템 상세)은 제외하고 적용 골격만 정리함.
