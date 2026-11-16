---
tags: [my-task, arc-tf, executive-summary, auto-recipe-creation]
category_major: "업무 기획·산출물"
category_middle: "ARC TF (Auto Recipe Creation)"
category_minor: "최고 임원 요약"
note_kind: "요약 보고"
classified_on: "2026-11-08"
last_updated: 2026-11-16
status: 초안, 2026-12 최종 승인 보고용
---

# Auto Recipe Creation TF: 한 장 요약

> Align Fail 자동 대응 PoC의 공통 자산과 시사점을 바탕으로 공통 플랫폼을 만들어, 숙련 엔지니어의 레시피 셋업 노하우를 회사 자산으로 남기고 장비가 늘어도 셋업 인력을 같은 비율로 늘리지 않는 구조를 만들고자 합니다.

## 2026 ARC 추진 현황

- **CD**: Align Fail 자동 대응 Agent 개발 진행(Recipe Tuning 포함), 현재 Offline 테스트 중. 소형 VLM 기반 장비 컨트롤(Computer-use) 기능 구현
- **Thickness**: 의뢰부터 Recipe 생성까지 ARC Pipeline 진행. DL 기반 Align Fail 대응 Offline 테스트 중. MI 장비군 중 가장 높은 자동화 구현
- **Inspection**: DRB(Design Recipe Builder)를 Tuning하는 방식으로 ARC 진행 중(NAND Tech 1개 대상). DRB Tuning에 Computer-use 사용. Recipe Setup AI 활용을 위한 DB 구축 완료(RAG, 마이다스)

## 출발점: PoC의 성과와 시사점

- PoC(Align Fail 자동 대응)로 Agent가 Recipe를 다루는 장비군 공통 방식 확보: Computer Use(VLM, PyAutoGUI, Win32), 장비 밖 Recipe 파일 편집
- LLM 모델만으로는 정밀 좌표, 장비 절차, 결과 검증, 화면 조작을 해결하지 못함. 조작, 모델 운영, 녹화 DB 추출은 장비군 공통 과제라 **공통 플랫폼**으로 대응
- 현업 MI 엔지니어의 노하우를 AI가 쓰는 구조로 개발자가 구현하는 **두 역할의 협업** 필요. 플랫폼은 전용 PC 대신 **엔지니어 본인 PC에 설치**

## 문제

- MI 장비 레시피 셋업은 숙련 엔지니어가 원격 화면에서 손으로 반복하는 작업임. 레시피당 30~60분 소요
- 판단 기준이 사람에게만 있어 담당자가 바뀌면 노하우가 사라짐
- 장비, 제품이 늘면 셋업 인력도 같은 비율로 늘어야 함. Align Fail 알람이 나면 엔지니어가 하던 일을 멈추고 대응해야 장비가 다시 돎

## 해결 방향

- 엔지니어 작업 녹화에서 절차와 판단 근거를 추출하고 담당 엔지니어가 승인해, 다른 엔지니어가 찾아 쓰는 **지식 자산**으로 축적함
- AI는 현재 화면에 맞는 승인 절차를 찾아 **한 단계씩** 수행하고, 결과가 불명확하면 멈춤. 위험 조작과 저장, 적용은 엔지니어가 승인함
- 초기 개발은 **CD-SEM, Thickness, Inspection 위주**(Align Fail PoC, 기존 PyAutoGUI 자동화와 ARC Pipeline, 의뢰서 DB와 Recipe RAG를 재사용)로 공통 플랫폼을 베타까지 만듦. 같은 기간 다른 MI 장비군은 인터뷰, Recipe Workflow 조사로 준비해 설계에 반영하고, 베타 뒤 본격 참여
- 개발과 AI에 익숙하지 않은 엔지니어는 합류 시점마다 교육하며 TF를 운영함

## 단계별 타겟

| 시기 | 타겟 | 회사에 남는 것 |
|---|---|---|
| 2027 상반기 | 선발대(CD-SEM, Thickness, Inspection) 장비군별 목표, 공통 플랫폼 설계 완성과 베타 | 확장 가능한 플랫폼 |
| 2027 하반기 | 2차 장비군(다른 MI 장비군) 합류, 선발대 셋업 노하우 지식 자산화 | 노하우의 조직 자산화 |
| 2028 | 엔지니어 앞에서 반복 입력 대행 | Recipe별 셋업의 Agent 진행 비율 증가 |
| 2029 | 1차 장비군에서 AI가 셋업을 진행하고 특이 사항, 위험 때만 엔지니어 개입 | 셋업 인력을 장비 수에 비례해 늘리지 않음 |

- 효과는 Recipe별 셋업 단계 가운데 Agent가 진행한 비율(%)로 확인함. 기준선은 합류 첫 달에 실측, 측정 전에는 목표를 숫자로 정하지 않음
- 2차 본격 참여는 2027-06 베타 배포 확인을 거쳐야 함. 단계마다 검증 증거를 확인한 뒤에만 다음 단계로 확대함

## 요청 사항

- **ARC TF 구성(조직도) 승인** (2027-01~12, 조직과 명단은 전체로, 참여는 선발대부터 단계적으로)
- 구성 내용: 공통 플랫폼 팀 개발 인력 4명(백엔드 2, 프론트엔드 2), 1차 CD-SEM, Thickness, Inspection 담당 장비군마다 1~2명, 2차 다른 MI 장비군 담당 장비군마다 1~2명(7월부터 본격 참여)
- 함께 필요한 자원: HCP 환경 및 자원, 녹화 스토리지 (2027-01 전)

## 관련 문서

- [TF 구성 기획안](./01-tf-proposal.md)
- [문제 정의](./00-problem-statement.md)
