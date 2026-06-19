# Handoff — ITC AIX 적용 기획 문서 작성 (CD-SEM Recipe Creation 자동화)

> 다음 세션 목표: 아래 합의된 설계를 바탕으로 **`my-task/AIX_POC/03-적용_CDSEM기획.md`** (Track A Discovery 산출물)를 작성한다.
> 작성 후 다음 단계로 `04-적용_CDSEM기술.md`(Track B)가 이어질 예정이나, **이번에는 기획 문서만** 작성한다.

## 1. 무엇을 만드는가

`AIX_POC/`의 방법론 문서(`01-기획문서_AX서비스기획.md`, `02-기술문서_AI과제정의구현.md`)는 **New AI Design Camp 프레임워크(틀)** 이다.
이번 작업은 그 틀을 **ITC AIX 실제 과제로 채운 "적용 사례" 문서**를 만드는 것.

- **결정된 산출물 구조 (옵션 A)**: 방법론(01·02)은 틀로 보존하고, 우리 과제는 채워진 사례로 분리한다.
  - `03-적용_CDSEM기획.md` ← **이번 세션에서 작성** (Track A: Discovery)
  - `04-적용_CDSEM기술.md` ← 다음 (Track B: To-Be 설계·구현)
- 문서 규칙: CLAUDE.md의 한국어 + Why/What/How 구조, 소문자-하이픈 파일명, frontmatter(tags/level/last_updated/type) 따를 것. 단 파일명은 기존 01·02와 일관성 위해 `03-적용_CDSEM기획.md` 형식 유지.
- **회사 기밀 주의**: 구체 장비 수치·내부 시스템 상세는 제외, 방법론 적용 골격 위주로.

## 2. 과제 한 줄 정의

**VLM 기반 GUI 제어(computer-use)로 CD-SEM Recipe Creation을 자동화** — VLM이 SEM 화면을 보며 GUI를 직접 클릭/입력해 Recipe를 생성하고, 실측 좌표 재정합·오인식 수정을 수행한다. 엔지니어는 모니터링하다 **예외만 개입(Human-in-the-loop)**. 완전자동화는 목표 아님.

## 3. 채워 넣을 실제 내용 (브레인스토밍 확정 사항)

### 조직목표 ↔ 과제 연계 (Top-Down)
- **조직 목표**: 신규 공정 셋업 **리드타임 단축** (핵심)
- **업무 목표**: MI Recipe Setup 처리량 확대 + 숙련도 편차 제거 + **24h 무중단 셋업**(사람 작업시간 제약 돌파)
- **핵심 업무**: CD-SEM Recipe Creation
- **AI 대상 후보 과제**: VLM GUI-제어 기반 Recipe 자동 생성 + 실측 좌표/align 자동 보정

### As-Is 프로세스 (엔지니어 확인 완료)
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
- 입력물: 공정/소자 엔지니어가 **MIDAS** 시스템에 의뢰서 작성(현재 PPT 형태: 도면·측정위치·파라미터·측정방식). → MIDAS 전산화는 **별도 프로젝트에서 진행 중**(이 과제의 외부 변수).
- ⑤ 이미징 조건: reference recipe 참고(동일) 또는 신규 공정은 엔지니어 협의로 결정.

### Pain Point → 근본원인(3-Why) → 해결 아이디어
| Pain Point | 근본 원인 | 해결 아이디어 |
|---|---|---|
| 오인식 수정 반복(⑥)이 오래 걸림 | 미리 템플릿을 만들어도 **실측 시작 시 전체 좌표 shift / align 실패** 발생 | VLM이 SEM 이미지를 보고 실시간 좌표 재정합·재align, 오인식 자동 수정 (핵심) |
| 숙련도별 해석 편차 | 표준 판단기준 부재 + 의뢰자가 의뢰서를 명확히 안 쓰는 경우 | VLM이 일관된 기준으로 수행 + 의뢰서 부족정보 플래깅 (MIDAS 전산화와 연계) |
| 야간/주말 셋업 공백 | 사람 작업시간 제약 | VLM 24h 무중단 수행, 사람은 예외만 |
- **집중 근본원인 = (a) 패턴 인식 실패, (b) 측정박스/좌표 오류** (사용자가 (a)(b) 명시).

### L1~L5 분해 (AI 배치 지점)
- L1 Mega: 계측(MI) 운영 / L2: CD-SEM Recipe Management / L3: Recipe Creation / L4 Activity: ①~⑦ / **L5 Task(AI 배치) = ③ 패턴 등록, ④ 좌표·박스, ⑥ 오인식 수정**.

### To-Be 핵심 (Human-AI 경계)
- 🤖 AI: ②~⑥ GUI 직접 조작 + SEM 이미지 판독 기반 좌표/align 재정합 + 오인식 수정.
- 🧑 Human: ① 의뢰서 검토(필요시), 예외 핸드오프 처리, ⑦ 최종 검증·승인.

## 4. 기술/환경 제약 (메모리 참조 — 04 기술문서에서 본격화하나 기획에도 반영)
- **외부 LLM API 전면 차단**. 사내 OpenAI-호환 내부 엔드포인트만: Kimi-K2.5(텍스트), **Qwen3-VL-8B / Qwen3-VL-30B(비전)**, BGE-M3(임베딩).
- DB(OpenSearch/ES)는 로컬 미가용 — 검증은 import/문법만.
- 99% 사내 문서 DRM → 스크린샷+VLM 추출이 기본.

## 5. 다음 세션 진행 방법
1. `superpowers:brainstorming`은 이미 완료(설계 합의됨). 바로 문서 작성 가능.
2. 기존 `01-기획문서_AX서비스기획.md` 구조(Why→What→How, 산출물 체크리스트)를 **거울처럼** 따르되, 모든 칸을 위 CD-SEM 실제 값으로 채운다.
3. 작성 후 `README.md`의 구성 표에 `03-적용_CDSEM기획.md` 추가.
4. 기술 문서(`04`)는 **이번엔 작성하지 않음** — 사용자 확인 후 진행.

## 6. Suggested skills
- (선택) `superpowers:writing-plans` — 문서를 단계별로 쓰기 전 계획이 필요하면. 단, 본 건은 설계가 이미 확정되어 곧바로 작성해도 됨.
- `superpowers:verification-before-completion` — 작성 완료 주장 전 링크/구조 점검.

## 7. 참조 (중복 금지 — 경로만)
- 방법론 틀: `my-task/AIX_POC/01-기획문서_AX서비스기획.md`, `02-기술문서_AI과제정의구현.md`
- 원문 전사: `my-task/AIX_POC/source/01~10`
- 프로젝트 컨텍스트(메모리): ITC AIX, SKEWNONO CD-SEM 트랙, 사내 LLM 제약.
