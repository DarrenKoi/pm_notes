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

> 2026년 장비군별 ARC 추진에서 얻은 Lesson Learned를 바탕으로 공통 플랫폼을 만들어, 숙련 엔지니어의 레시피 셋업 노하우를 회사 자산으로 남기고 장비가 늘어도 셋업 인력을 같은 비율로 늘리지 않는 구조를 만들고자 합니다.

## 2026 ARC 추진 현황

| 장비군 | 추진 현황 |
|---|---|
| CD | **Align Fail 자동 대응 Agent** 개발 진행 (Recipe Tuning 포함), 현재 Offline 테스트 중. 소형 VLM 기반 장비 컨트롤(Computer-use) 기능 구현 |
| Thickness | 의뢰부터 Recipe 생성까지 **ARC Pipeline** 진행. DL 기반 Align Fail 대응 Offline 테스트 중. MI 장비군 중 최고 수준 자동화 구현 |
| Inspection | **DRB(Design Recipe Builder)** Tuning 방식으로 ARC 진행 중 (NAND Tech 1개 대상). DRB Tuning에 Computer-use 사용. Recipe Setup AI 활용 DB 구축 완료 (RAG, 마이다스) |

**Lesson Learned**

1. 생성형 AI 모델만으로 ARC 구현 어려움 (정밀 좌표 클릭, 셋업 절차, 결과 판단, 화면 조작 등의 문제)
2. Rule 기반 자동화, 컴퓨터 비전(CV), ML/DL을 생성형 AI 모델과 함께 쓰는 하이브리드 방식 필수
3. 생성형 AI 모델의 Recipe 작업 방식 학습 방법 구체화 필요

## 업무 전략

- **Agent의 Recipe 작업 방식**: 화면 조작(Computer Use: VLM, PyAutoGUI, Win32 조합)과 외부 Recipe 파일 편집. 기반기술센터의 최적화 소형 모델을 자동화 작업에 사용. Recipe 저장과 적용은 엔지니어 승인
- **공통 플랫폼 대상 공통 과제**: Agent의 마우스, 키보드 조작, 최적화 소형 모델 운영, 녹화 DB 추출, 기술 부채 해소. 장비군별 개별 진행 시 같은 작업 반복
- **설치 방식**: 별도 운영 PC 대신 엔지니어 PC에 설치. 엔지니어가 원할 때 실행하고 즉시 확인, 중단 (Human on Command)

## 핵심 개발 방향

Computer Use 기반 RAG Workflow 순환 플랫폼입니다.

| 단계 | 내용 |
|---|---|
| ① 녹화 추출, 라벨링 | 녹화 비디오에서 작업 내용 추출, 현업 엔지니어가 라벨링 |
| ② RAG 저장 | Recipe 셋업 지식 DB (절차, 노하우, 결정 지점) |
| ③ Workflow 생성, 실행 | RAG에서 셋업 Workflow 생성, Agent가 Computer Use로 실행 |
| ④ 준수 확인, 수정 | Workflow 준수 확인, 틀리면 엔지니어가 개입, 수정. 수정 내용은 ②에 다시 저장해 다음 Workflow에 반영 |

- **공통 플랫폼 팀 역할**: 녹화 내용 추출, RAG DB, Workflow 생성, Computer Use 실행 엔진, 준수 확인(생성형 AI + Rule, CV, ML/DL), 즉시 중단
- **현업 엔지니어 역할**: 추출 내용 라벨링, Workflow 검토와 승인, 실행 중 개입과 수정, 실전 사용 후 개발자에게 피드백
- **플랫폼의 역할**: 2028 보조(반복 입력 대행)에서 2029 자율(Agent 셋업 진행)로 점진적 발전

## TF 구성

기반기술센터 AI/DT TF 담당 아래 ARC TF Champion이 총괄하며, 공통 플랫폼 팀과 R&D DMI 장비군 담당 엔지니어가 협업합니다.

| 구분 | 구성 | 2027 상반기 |
|---|---|---|
| 공통 플랫폼 팀 | 개발 약 5명 규모: 백엔드 2, 프론트엔드 2, 1차 MI팀 현업 엔지니어 | 플랫폼 설계, 베타 배포 |
| R&D DMI 장비군 담당: CD-SEM (1차) | 1~2명 | Align Fail 자동 대응 및 Recipe Tuning(이미지 변경) 구현 |
| R&D DMI 장비군 담당: Thickness (1차) | 1~2명 | Film 10 Layer 이하 의뢰 Recipe 90% 이상 생성 자동화 |
| R&D DMI 장비군 담당: Inspection (1차) | 1~2명 | Optical Inspection 장비군 전체 디바이스향 DRB 기반 ARC 확장 |
| R&D DMI 장비군 담당: 다른 MI 장비군 (2차) | 장비군마다 1~2명 | 인터뷰, Workflow 조사 후 7월 본격 참여 |

- 운영: 매주 미팅(개발 노하우 공유, 진행사항 체크), 월간 AI/DT TF장 보고, 분기 부문장 보고

## 단계별 타겟

| 시기 | 타겟 | 회사 효과 |
|---|---|---|
| 2027 상반기 | 1차 장비군(CD-SEM, Thickness, Inspection) 목표 달성, 공통 플랫폼 베타 | 전 장비군 공통 플랫폼 |
| 2027 하반기 | 2차 장비군(다른 MI 장비군) 합류, 셋업 노하우 지식 자산화 | 장비군 확장 구조, 노하우의 조직 자산화 |
| 2028 | 반복 입력 대행 | Recipe별 셋업의 Agent 진행 비율 증가 |
| 2029 | 1차 장비군에서 AI가 셋업 진행, 특이 사항과 위험 때만 엔지니어 개입 | 셋업 인력을 장비 수에 비례해 늘리지 않음 |

- 측정: Recipe별 셋업 단계 중 Agent 진행 비율(%). 기준선은 합류 첫 달에 실측, 측정 전 목표 숫자 없음

## 요청 사항

- **ARC TF 구성(조직도) 승인** (2027-01~12, 1차와 2차 명단 함께 지정)
- 필요 자원: HCP 환경 및 자원, 녹화 스토리지 (2027-01 전)

## 관련 문서

- [TF 구성 기획안](./01-tf-proposal.md)
- [문제 정의](./00-problem-statement.md)
