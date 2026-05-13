# Grilling Session — 2026-05-14

> ITC AI/DT 로드맵 수립을 위한 `/grill-with-docs` 세션 기록.
> 재개 시 이 파일과 [CONTEXT.md](./CONTEXT.md), [bgk.txt](./bgk.txt)를 같이 읽으면 컨텍스트 복원됨.

## 진행 상황 요약

bgk.txt 기반으로 grilling 시작. 1차 라운드에서 **로드맵의 진짜 동기**, 2차 라운드에서 **스코프 경계**를 다루는 중.

## 해결된 것 (CONTEXT.md에 반영 완료)

- **1차 동기 = A (CEO 보고용)**. 다른 Biz는 로드맵 기반으로 보고하는데 ITC 센터장은 아직 그 수준이 안 됨.
- **2차 동기 = E (상향 평준화)**. 팀별 AIX/DX 격차가 큼 → 하위 팀 끌어올리기.
- **AX Part의 핵심 thesis**:
  > "ITC는 다른 Biz의 의뢰/요청에 AI로 빠르고 정확하게 대응. 반복 요청은 자동화."
- **팀별 우선순위 AI Agent 로드맵은 이미 존재** (개별 로드맵 형태). ITC 로드맵은 그 위에 얹는 통합 레이어가 되어야 함.
- 용어 정리: DT(사업부) vs DX(활동), AIX vs AI/DT 구분, "의뢰/요청"을 자동화 대상의 unit of work로 명명.

## 진행 중인 질문 — Q2: 스코프 경계

bgk.txt 상 ITC의 일은 셋:
1. Biz 목표 지원 ← AX Part thesis가 정확히 겨냥
2. 공정 기술 개발 (proactive R&D)
3. 장비 운영/관리 (operational)

팀 성격이 다름:
- DMI, AT → (1) 요청 처리 중심, thesis와 딱 맞음
- ME, 환경제어 → (3) 장비 운영, **다른 패러다임** (예지보전·이상탐지)
- MASK, 소재 → (2) R&D 사이클

세 가지 옵션을 제시했음:
- **α**: 단일 thesis로 좁힘 ("Biz 요청 자동 대응")
- **β** (Claude 추천): 멀티 트랙 — Track 1 요청 대응 / Track 2 장비·공정 AI / Track 3 R&D 가속
- **γ**: capability(역량) 중심 가로축 구조

**Daeyoung의 답변 대기 중**:
1. α / β / γ 중 어디로 갈지
2. β를 선택한다면, ME·환경제어 팀에 실제로 추진 중인 AI 과제가 있는지 (Track 2가 빈 박스가 되지 않으려면 필요)

## 큐에 쌓인 다음 질문들 (Q2 해결 후 다룰 것)

- **Q3. "의뢰/요청"의 unit 정의**: ticket인지, 협업 요청인지, 정형화된 의뢰서인지. 자동화율 측정 기준이 됨.
- **Q4. 팀별 개별 로드맵과 ITC 로드맵의 관계**: roll-up인가, abstract layer인가, parallel narrative인가.
- **Q5. AIX TF(전사)와 AX Part(ITC)의 책임 경계**: 둘 다 AIX를 다루는데 ITC 로드맵의 소유권/조정 구조.
- **Q6. 3~4년 시간축의 granularity**: 연 단위 마일스톤? 반기? 단계(Crawl-Walk-Run)?
- **Q7. CEO 보고 산출물의 형태**: 1-page narrative + appendix? 슬라이드? 다른 Biz가 쓰는 포맷이 있는가?
- **Q8. 성공 지표**: 노동력 절감 시간? 자동 대응 비율? TAT 단축? 어떤 게 CEO에게 의미 있는가.

## 재개 방법

다음 세션에서 이 파일 + CONTEXT.md를 읽힌 뒤 "Q2부터 이어가자"라고 하면 됨. 또는 `/grill-with-docs` 다시 호출하고 이 폴더를 가리켜도 됨.
