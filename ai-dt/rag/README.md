---
tags: [rag, documentation-review]
reviewed_on: 2026-10-04
review_status: partial
document_type: index
category_major: "AI·DT"
category_middle: "RAG"
category_minor: "RAG 학습 안내"
note_kind: "목차"
classified_on: "2026-10-05"
---

# RAG 학습 목차

RAG는 외부 근거를 검색해 모델 입력에 제공하는 방식이다. 문서 추출·청킹·검색·생성을 나누어 설계하고, 실제 근거 적합성과 답변 품질을 측정한다. 검색 결과가 있다고 정답이 보장되지는 않는다.

> [!info] 검토 진행 중 — 2026-10-04
> 원래38개를 발견했다. 현재 대화 메모리·조립 예제·LangGraph 시리즈·고급 RAG 시리즈·Milvus 시리즈·OpenSearch 기초/클라이언트·BM25/벡터/하이브리드 검색·성능/Settings 설계·공유 핸들러·RAG 연동·OpenSearch 대화 메모리의 청킹 목차/총론을 포함한 원문38개를 모두 개별 검토했다. 실제 사내 환경/모델 품질·Claude 협의와 전체 읽기 화면 감사는 미완료다. 아래는 기존 주제 구조를 안내하는 목차다. 출처·실행 조건·미확인은 [정리 기록](./organization-log.md)을 기준으로 읽는다. 공식 확인 없이 최신/완성으로 간주하지 않는다.

## 읽기 순서

| 순서 | 문서 묶음 | 목적과 선택 조건 |
|---|---|---|
| 1 | [LangChain·LangGraph 입문](./langchain-langgraph/README.md) | 구성 요소와 제어 흐름을 구분하고 RAG/tool calling 조립 방식 파악 |
| 2 | [LangGraph](./langgraph/README.md) | 상태·분기·재개 패턴을 개별 예제로 학습 |
| 3 | [문서 처리와 청킹](./token_strategy/README.md) | 입력 형식별 추출/분할 조건 비교. 토큰화와 청킹은 별도 작업 |
| 4 | [Milvus](./milvus/README.md), [OpenSearch](./opensearch/README.md) | 벡터 저장/검색 또는 키워드·벡터 검색을 목적에 따라 선택. 두 엔진 사용이 모두 필수는 아님 |
| 5 | [고급 RAG](./advanced-rag/README.md) | 재검색·판별·다중 단계의 효과와 비용/실패 조건 검토 |
| 6 | [대화 메모리](./llm-conversation-memory.md) | thread 상태·요약·사용자별 장기 검색과 검수 후보를 구분 |

## 유사 문서의 역할

LangGraph 시리즈는 그래프의 상태/제어 기능 학습, LangChain·LangGraph 시리즈는 구성 요소 조립, 고급 RAG는 검색 실패 대응과 오케스트레이션 예제를 다룬다. 같은 API 설명이 일부 겹치지만 각 문서의 예제 맥락을 유지한다. 완전한 중복 통합·분할은 Claude 협의 연결 실패로 보류했다.

대화 메모리 문서는 저장/요약의 개념과 callback 계약을, [OpenSearch 대화 메모리](./opensearch/conversation-memory-opensearch.md)는 해당 엔진에 적용하는 예제를 다룬다. 후자는 mapping/필터/API/조립 경계를 검토했으나 실제 서버·모델·메모리 관리자 구현은 미확인이다.

[DRM 시나리오](./token_strategy/when_drm/README.md)는 추출 가능한 입력 조건을 비교하는 학습 자료다. 원문의 회사 비율·유일한 추출 방법·향후 해제 계획은 이번에 확인되지 않았으며 기술 검토가 남아 있다. 업무 이력/승인 정책으로 읽지 않는다. 여기의 실습은 실제 서비스 운영 이력을 증명하지 않는다.
