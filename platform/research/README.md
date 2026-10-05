---
tags: [platform, research]
aliases: [플랫폼 사전 조사 목차]
reviewed_on: 2026-10-04
review_status: partial
document_type: index
category_major: "Agent·Skill 플랫폼"
category_middle: "기술 조사"
category_minor: "조사 안내"
note_kind: "목차"
classified_on: "2026-10-05"
---

# 플랫폼 사전 조사 읽기 안내

이 폴더의 조사 3개는 2026-09-03 설계 과정의 기록이다. 제품의 현재 동작과 사내 실측을 보증하지 않는다. 새로 확인한 조건은 [현재 검토](../review-notes.md), 검토 범위·미확인은 [정리 기록](../organization-log.md)에 모았다.

| 순서 | 문서 | 목적과 읽을 조건 |
|---|---|---|
| 1 | [Agent·Skill 레지스트리](./agent-skill-registry-research.md) | 번들·카탈로그·설치·인증·관측의 경계를 배운다. MCP·Claude 규격은 버전별 조건을 함께 읽는다. |
| 2 | [Google 규모 인가](./authorization-at-scale-google.md) | 개별 권한과 그룹 멤버십을 분리하는 이유, 관계 튜플·일관성의 작동 방식을 배운다. 규모 비교는 설계 추정이며 엔진 성능 검증이 아니다. |
| 3 | [GitLab 배포 백엔드](./gitlab-as-distribution-backend.md) | 저장·심사 기능을 기존 서비스로 대체할 후보를 비교한다. 사내 버전·티어·권한 확인은 [실측 체크리스트](../06-gitlab-check.md)를 따른다. |

세 문서는 조사 질문이 달라 통합하지 않았다. 신원·서명·불변성처럼 겹치는 **현재 적용 조건**은 상위 검토 문서 하나로 안내한다. 당시 선택·인용·고유 예제는 보존한다. 아래 기록의 “미확인”은 기능이 없다는 증명이 아니다.

## 과거 결론을 적용하기 전에

- Object Lock은 객체 **버전**을 보존한다. 같은 키의 모든 새 쓰기·delete marker·접근까지 자동 차단하는 장치로 읽지 않는다.
- 단일 리전·primary 읽기만으로 인가 캐시와 콘텐츠의 인과 순서가 보장되지는 않는다.
- Google IAM 전파 시간의 “or longer”는 최대 지연이나 SLA가 아니다.
- MCP 2025-06-18 조사와 2026-07-28 인가 규격을 구분한다. OTel GenAI 속성도 Development 상태·계측 대상을 확인한다.
- 공식 문서의 예시를 사내 실제 설치·서명·배포 성공으로 취급하지 않는다. 이번 정리에서는 서비스 호출·플러그인 설치를 실행하지 않았다.
