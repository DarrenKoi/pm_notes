---
tags: [my-task, arc-tf, executive-summary, auto-recipe-creation]
category_major: "업무 기획·산출물"
category_middle: "ARC TF (Auto Recipe Creation)"
category_minor: "최고 임원 요약"
note_kind: "요약 보고"
classified_on: "2026-11-08"
last_updated: 2026-11-17
status: 초안, 2026-12 최종 승인 보고용
---

# Auto Recipe Creation TF: 한 장 요약

> 2026년 장비군별 ARC 추진에서 얻은 Lesson Learned를 바탕으로 공통 플랫폼을 만들어, 숙련 엔지니어의 레시피 셋업 노하우를 회사 자산으로 남기고 2029년 1차 장비군 의뢰 Recipe의 ARC 적용 비율 90% 이상을 목표로 합니다.

## 2026 ARC 추진 현황

| 장비군 | 추진 현황 |
|---|---|
| CD | **Align Fail 자동 대응 Agent** 개발 진행 (Recipe Tuning 연계), 현재 Offline 테스트 중. 소형 VLM 기반 장비 컨트롤(Computer-use) 기능 구현 |
| Thickness | **MI 장비군 중 자동화 완성도 가장 높음**: 의뢰부터 Recipe 생성까지 ARC Pipeline 진행. DL 기반 Align Fail 대응 Offline 테스트 중 |
| Inspection | **DRB(Design Recipe Builder)** Tuning 방식으로 ARC 진행 중 (NAND Tech 1개 대상). DRB Tuning에 Rule 기반 Computer-use 사용. Recipe Setup AI 활용 DB 구축 완료 (RAG, 마이다스) |

**Lesson Learned**

1. 생성형 AI 모델만으로 ARC 구현 어려움 (정밀 좌표 클릭, 셋업 절차, 결과 판단, 화면 조작 등의 문제)
2. Rule 기반 자동화, 컴퓨터 비전(CV), ML/DL을 생성형 AI 모델과 함께 쓰는 하이브리드 방식 필수
3. 엔지니어의 노하우인 Recipe 작업 방식을 지식 자산화 (DB)해야 생성형 AI 모델이 활용 가능

## 핵심 개발 방향

Computer Use 기반 RAG Workflow 순환 플랫폼입니다.

| 단계 | 내용 |
|---|---|
| ① 녹화 추출, 라벨링 | 녹화 비디오에서 작업 내용 추출, 현업 엔지니어가 라벨링 |
| ② RAG 저장 | Recipe 셋업 지식 DB에 절차, 노하우, 결정 지점 저장 |
| ③ Workflow 생성, 실행 | RAG에서 셋업 Workflow 생성, Agent가 Computer Use로 실행 |
| ④ 준수 확인, 수정 | Workflow 준수 확인, 틀리면 엔지니어가 개입, 수정. 수정 내용은 ②에 다시 저장해 다음 Workflow에 반영 |

- **공통 플랫폼 팀 역할**: 녹화 내용 추출, RAG DB, Workflow 생성, Computer Use 실행 엔진 개발, 최적화 소형 모델 운영, 기술 부채 해소
- **현업 엔지니어 역할**: 추출 내용 라벨링, Workflow 검토와 승인, 실행 중 개입과 수정, 실전 사용 후 개발자에게 피드백. 외부 Recipe 파일 편집은 장비군별 도구로 진행 (공통 플랫폼 범위 밖)
- **플랫폼의 역할**: 2027 지식 자산화(작업 녹화, Workflow 구축) → 2028 보조(반복 입력 대행) → 2029 자율(Agent가 셋업 진행)

## TF 구성

기반기술센터 AI/DT TF 담당 아래 ARC TF Champion이 총괄하며, 공통 플랫폼 팀과 R&D DMI 장비군 담당 엔지니어가 협업합니다.

| 구분 | 구성 | 2027 상반기 |
|---|---|---|
| 공통 플랫폼 팀 | 개발 약 5명 규모: 백엔드 2, 프론트엔드 2, 1차 MI팀 현업 엔지니어 | 플랫폼 설계, 베타 배포 |
| R&D DMI 장비군 담당: CD-SEM (1차) | 1~2명 | Align Fail 자동 대응 Agent 현장 배치 |
| R&D DMI 장비군 담당: Thickness (1차) | 1~2명 | Film 10 layer 이하 의뢰건 90% 이상 자동화 |
| R&D DMI 장비군 담당: Inspection (1차) | 1~2명 | Optical Inspection 장비군 전체로 ARC 확장 |
| R&D DMI 장비군 담당: 다른 MI 장비군 (2차) | 장비군마다 1~2명 | 인터뷰, Workflow 조사 후 7월 본격 참여 |

- 운영: 매주 미팅(개발 노하우 공유, 진행사항 체크), 월간 AI/DT TF장 보고, 분기 부문장 보고

## 개발 로드맵

| 시기 | 목표 |
|---|---|
| 2027 상반기 | 1차 3개 장비군(CD-SEM, Thickness, Inspection) 목표 달성, 공통 플랫폼 베타, 기준선 측정 |
| 2027 하반기 | 2차 장비군(다른 MI 장비군) 합류, 단위 작업 Agent 수행 성공 비율 ⟨X⟩% |
| 2028 | 1차 장비군 Agent 개입 범위 기준선 대비 ⟨N⟩배, 셋업 시간 단축 (반복 입력 대행) |
| 2029 | 1차 장비군 의뢰 Recipe ARC 적용 비율 90% 이상 (AI가 셋업 진행, 특이 사항과 위험 때만 엔지니어 개입) |

- 수치 확정: 기준선 측정 후 **2027-07**에 2028~2029 목표 수치 확정 보고. 2차 장비군 목표는 합류 후 확정
- 측정: ARC 적용 비율 = ARC 적용 Recipe 수 / 의뢰 Recipe 수 (장비군별 대상 Recipe 기준, 재작업 기준 충족 건만 집계)

## 요청 사항

- **ARC TF 구성(조직도) 승인** (2027-01~12, 1차와 2차 명단 함께 지정)
- 필요 자원: HCP 환경 및 자원, 녹화 스토리지 (2027-01 전)

## 관련 문서

- [TF 구성 기획안](./01-tf-proposal.md)
- [문제 정의](./00-problem-statement.md)
