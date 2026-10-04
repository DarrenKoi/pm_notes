---
tags: [itc, aix, roadmap, documentation-review]
reviewed_on: 2026-10-04
review_status: partial
document_type: index
---

# ITC AI/DT 로드맵 기록

2026년 5월 작성한 CEO 보고용 로드맵 설계 자료다. 공통 AI 능력을 현장 업무와 결합하는 목적, 추진 트랙·KPI·PoC·역할 경계를 검토한 **업무 기록**이며 기술 입문 교재가 아니다. 현재 완료한 사업이나 최신 조직도를 의미하지 않는다.

## 읽기 순서

1. [검토 안내](./review-notes.md): 원문을 다시 사용할 때의 기술 조건과 내부 불일치.
2. [CONTEXT](./CONTEXT.md): ITC·DT/DX·5 stream·암묵지 순환고리의 당시 정의.
3. [통합 outline v0.3](./itc-aix-roadmap-outline.md): 보고서 전체의 골격. 챕터 1~7을 순서대로 안내한다.
4. 아래 상세 초안에서 관심 있는 챕터를 읽는다.
5. [정리 기록](./organization-log.md): 개별 검토 결과·출처·검증·보류.

## 상세 초안

| 순서 | 문서 | 역할 |
|---|---|---|
| 1 | [배경과 As-Is](./ch1-context-asis-draft.md) | 현황 진단과 anchor 슬롯. 수치가 채워진 실측 보고서가 아니다. |
| 2 | [비전과 positioning](./ch2-vision-draft.md) | 보고 메시지·예상 반박. 조직 유일성 주장은 미확인이다. |
| 3 | [To-Be와 KPI](./ch3-tobe-draft.md) | 단계별 업무 변화와 목표. 지표 정의가 필요한 부분은 검토 안내 참조. |
| 4 | [추진 트랙](./ch4-tracks-draft.md) | 5 능력×3 트랙 매트릭스와 횡전개 가설. |
| 5 | [ADR-0001](./docs/adr/0001-aix-tf-ax-part-boundary.md) | 당시 조직 경계 결정. Ch.5 별도 슬라이드 초안은 미작성. |
| 6 | [간트 초안](./ch6-gantt-draft.md) | 5년 horizon과 활동·운영·마일스톤의 표현. |
| 7 | [Quick Win 카드](./quick-win-cards.md) | 5개 제안 과제의 PoC/본 적용 조건·데이터 의존성. |

outline은 전체 설계, 챕터 초안은 슬라이드 본문·speaker notes, 카드와 ADR은 실행·책임 조건이다. 같은 설명이 나와도 용도가 달라 원문을 합치지 않았다. 공통 검토 주의사항만 대표 안내에 모았다.

## 결정 과정과 원본 자료

- [질문 세션](./grilling-session-2026-05-14.md): Round 1~2 당시의 선택과 미해결 큐.
- [브레인스토밍 준비](./team-brainstorm-prep.md): v0.2를 검토할 회의 질문. 회의를 실행한 결과가 아니다.
- [인계 기록](./handoff-2026-05-15.md): v0.3·드래프트 시점의 남은 업무. 과거 에이전트 지시를 이번 정리 작업의 추가 실행 승인으로 취급하지 않는다.
- [입력 메모](./bgk.txt): 원본 텍스트 보존.
- [HTML 읽기 자료](./html/index.html): 기존 정적 산출물 보존. 이번 Markdown 주석이 자동 반영된 것으로 보장하지 않는다.

Cover/Executive Summary와 Ch.5 별도 초안, anchor 실값과 현재 승인·성과 자료는 이 폴더에서 확인되지 않았다. 슬롯과 미해결 질문을 임의로 채우지 않았다.
