---
tags: [index, taxonomy]
document_type: taxonomy_index
category_major: "Agent·Skill 플랫폼"
category_middle: "주제 안내"
category_minor: "전체 목차"
note_kind: "목차"
classified_on: "2026-10-05"
---

# Agent·Skill 플랫폼 — 대·중·소분류 목차

**대분류: Agent·Skill 플랫폼**. 아래 중분류·소분류에서 문서를 선택한다. [기존 읽기 순서](./README.md)도 함께 사용할 수 있다.

> [!tip] 분류를 읽는 방법
> 대분류는 넓은 분야, 중분류는 기술·업무 영역, 소분류는 구체적인 학습·작업 주제다.
> 문서 유형은 학습·실습·기록·양식 등 용도를 나타내며 주제와 별도로 구분한다.
> 파일 경로는 유지했다. 분류 속성은 기술 검증일·업무 승인·실행 완료를 뜻하지 않는다.

## 분류 요약

| 중분류 | 소분류 수 | 문서 수 |
|---|---|---|
| [플랫폼 설계](#%ED%94%8C%EB%9E%AB%ED%8F%BC%20%EC%84%A4%EA%B3%84) | 3 | 3 |
| [기술 조사](#%EA%B8%B0%EC%88%A0%20%EC%A1%B0%EC%82%AC) | 4 | 4 |
| [구현 준비](#%EA%B5%AC%ED%98%84%20%EC%A4%80%EB%B9%84) | 10 | 33 |
| [플랫폼 운영](#%ED%94%8C%EB%9E%AB%ED%8F%BC%20%EC%9A%B4%EC%98%81) | 3 | 3 |
| [주제 안내](#%EC%A3%BC%EC%A0%9C%20%EC%95%88%EB%82%B4) | 1 | 1 |
| [문서 관리](#%EB%AC%B8%EC%84%9C%20%EA%B4%80%EB%A6%AC) | 1 | 1 |

## 플랫폼 설계

### 요구사항

| 문서 | 유형 |
|---|---|
| [사내 AI Agent·Skill 공유 관리 플랫폼 요구사항 정의서](./01-requirements.md) | 설계·계획 |

### 아키텍처

| 문서 | 유형 |
|---|---|
| [아키텍처 설계서 — 사내 AI Agent·Skill 공유 플랫폼 (1차/사업부)](./02-architecture.md) | 설계·계획 |

### 인력·구축 계획

| 문서 | 유형 |
|---|---|
| [구축 계획 — 인력·기간 산정](./04-plan.md) | 설계·계획 |

## 기술 조사

### 조사 안내

| 문서 | 유형 |
|---|---|
| [플랫폼 사전 조사 읽기 안내](./research/README.md) | 목차 |

### 자산 레지스트리

| 문서 | 유형 |
|---|---|
| [사내 Agent/Skill 레지스트리 - 1차 자료 조사](./research/agent-skill-registry-research.md) | 참고 자료 |

### 대규모 인가

| 문서 | 유형 |
|---|---|
| [대규모 인가(Authorization) 1차 자료 조사 — Google 은 어떻게 수만 명에게 권한을 주는가](./research/authorization-at-scale-google.md) | 참고 자료 |

### GitLab 배포

| 문서 | 유형 |
|---|---|
| [GitLab을 배포 백엔드로 쓸 수 있는가 — 1차 출처 조사](./research/gitlab-as-distribution-backend.md) | 참고 자료 |

## 구현 준비

### 티켓 규약·목차

| 문서 | 유형 |
|---|---|
| [코딩 에이전트용 티켓 규약](./05-ticket-conventions.md) | 운영 지침 |
| [티켓 공통 규약 (구현 저장소)](./tickets/CONVENTIONS.md) | 운영 지침 |
| [구현 티켓](./tickets/README.md) | 목차 |

### 기반·CI

| 문서 | 유형 |
|---|---|
| [PLAT-L0-001 구현 저장소 스켈레톤과 CI 를 세운다](./tickets/PLAT-L0-001.md) | 구현 티켓 |

### 아티팩트 저장

| 문서 | 유형 |
|---|---|
| [PLAT-L1-001 ArtifactStore 인터페이스 계약을 확정한다](./tickets/PLAT-L1-001.md) | 구현 티켓 |
| [PLAT-L1-002 MinIO ArtifactStore 구현과 불변성 설정을 만든다](./tickets/PLAT-L1-002.md) | 구현 티켓 |

### 카탈로그·조직·인가

| 문서 | 유형 |
|---|---|
| [PLAT-L2-001 카탈로그 문서 스키마 계약을 확정한다](./tickets/PLAT-L2-001.md) | 구현 티켓 |
| [PLAT-L2-002 카탈로그 저장소 인터페이스와 MongoDB 구현을 만든다](./tickets/PLAT-L2-002.md) | 구현 티켓 |
| [PLAT-L2-003 발행 문서의 불변성과 상태 전이 규칙을 강제한다](./tickets/PLAT-L2-003.md) | 구현 티켓 |
| [PLAT-L2-004 소유권 관계 튜플과 인가 판정을 구현한다](./tickets/PLAT-L2-004.md) | 구현 티켓 |
| [PLAT-L2-005 teams 컬렉션 스키마와 저장소를 만든다](./tickets/PLAT-L2-005.md) | 구현 티켓 |
| [PLAT-L2-006 사원 마스터에서 팀 멤버십을 동기화하는 배치를 만든다](./tickets/PLAT-L2-006.md) | 구현 티켓 |
| [PLAT-L2-007 멤버 집합 비교로 조직 개편 유형을 추론한다](./tickets/PLAT-L2-007.md) | 구현 티켓 |

### 검색

| 문서 | 유형 |
|---|---|
| [PLAT-L3-001 검색 문서와 질의 계약을 확정한다](./tickets/PLAT-L3-001.md) | 구현 티켓 |
| [PLAT-L3-002 카탈로그 변경을 검색 인덱스에 반영한다](./tickets/PLAT-L3-002.md) | 구현 티켓 |
| [PLAT-L3-003 권한 범위 안에서 검색 결과를 반환한다](./tickets/PLAT-L3-003.md) | 구현 티켓 |
| [PLAT-L3-004 카탈로그에서 검색 인덱스를 재구축한다](./tickets/PLAT-L3-004.md) | 구현 티켓 |

### 인증

| 문서 | 유형 |
|---|---|
| [PLAT-L5-001 OIDC 액세스 토큰 검증을 구현한다](./tickets/PLAT-L5-001.md) | 구현 티켓 |
| [PLAT-L5-002 인증과 인가를 FastAPI 의존성으로 결합한다](./tickets/PLAT-L5-002.md) | 구현 티켓 |

### 배포 어댑터

| 문서 | 유형 |
|---|---|
| [PLAT-L6-001 읽기 전용 배포 어댑터 계약을 확정한다](./tickets/PLAT-L6-001.md) | 구현 티켓 |
| [PLAT-L6-002 Claude Code marketplace 문서를 투영한다](./tickets/PLAT-L6-002.md) | 구현 티켓 |
| [PLAT-L6-003 generic 카탈로그 응답을 투영한다](./tickets/PLAT-L6-003.md) | 구현 티켓 |

### Endpoint Broker

| 문서 | 유형 |
|---|---|
| [PLAT-L6B-001 Endpoint Broker 계약을 확정한다](./tickets/PLAT-L6B-001.md) | 구현 티켓 |
| [PLAT-L6B-002 endpoint 헬스체크를 5분마다 수행한다](./tickets/PLAT-L6B-002.md) | 구현 티켓 |
| [PLAT-L6B-003 연속 장애를 degraded로 자동 전환한다](./tickets/PLAT-L6B-003.md) | 구현 티켓 |
| [PLAT-L6B-004 만료되는 Broker 접근 토큰을 발급한다](./tickets/PLAT-L6B-004.md) | 구현 티켓 |
| [PLAT-L6B-005 원격 호출을 중개하고 계량한다](./tickets/PLAT-L6B-005.md) | 구현 티켓 |

### 큐·레이트리밋

| 문서 | 유형 |
|---|---|
| [PLAT-L7-001 큐와 레이트리밋 계약을 확정한다](./tickets/PLAT-L7-001.md) | 구현 티켓 |
| [PLAT-L7-002 Redis 작업 큐를 구현한다](./tickets/PLAT-L7-002.md) | 구현 티켓 |
| [PLAT-L7-003 Redis 레이트리밋을 원자적으로 적용한다](./tickets/PLAT-L7-003.md) | 구현 티켓 |

### 텔레메트리

| 문서 | 유형 |
|---|---|
| [PLAT-L8-001 텔레메트리 속성과 이벤트 계약을 확정한다](./tickets/PLAT-L8-001.md) | 구현 티켓 |
| [PLAT-L8-002 Broker 계량 이벤트를 OTel로 발행한다](./tickets/PLAT-L8-002.md) | 구현 티켓 |
| [PLAT-L8-003 자산별 사용량 집계를 조회한다](./tickets/PLAT-L8-003.md) | 구현 티켓 |

## 플랫폼 운영

### 운영·거버넌스

| 문서 | 유형 |
|---|---|
| [사내 AI Agent·Skill 플랫폼 운영·거버넌스](./03-operations.md) | 운영 지침 |

### 도입 전 실측

| 문서 | 유형 |
|---|---|
| [사내 GitLab 실측 체크리스트](./06-gitlab-check.md) | 검토 기록 |

### 적용 조건

| 문서 | 유형 |
|---|---|
| [플랫폼 현재 검토와 적용 조건](./review-notes.md) | 검토 기록 |

## 주제 안내

### 전체 목차

| 문서 | 유형 |
|---|---|
| [사내 AI Agent·Skill 공유 관리 플랫폼](./README.md) | 목차 |

## 문서 관리

### 정리·검증 기록

| 문서 | 유형 |
|---|---|
| [플랫폼 문서 정리 기록](./organization-log.md) | 관리 기록 |

## Obsidian에서 찾기

일반 문서의 YAML 속성 `category_major`, `category_middle`, `category_minor`, `note_kind`로 분류를 확인한다. 모두 단일 텍스트 값이다. 기존 tags·aliases·작성일·검토 상태는 유지한다.

검색 패널에서 다음처럼 속성을 조합한다. 대·중·소 값은 위 표의 실제 값을 사용한다.

```text
path:"platform/" [category_major:"Agent·Skill 플랫폼"]
path:"platform/" [note_kind:"학습"]
```

목차·검토·관리 기록은 학습 문서와 유형을 구분했다. 불변 원문과 작업 지침은 속성을 추가하지 않고 이 목차에서 분류한다. 따라서 속성 검색만으로 원본 보존 문서 전체를 찾을 수는 없다.

분류 대상은 기존 Markdown 45개이며, 그중 원본 보존 0개다. 이 목차 자체는 대상 수에 포함하지 않는다.

분류일: 2026-10-05. 링크·문법과 실제 읽기 화면의 확인 범위는 [분류 검증 기록](./classification-review.md)에 남긴다.
