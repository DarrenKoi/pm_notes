---
tags: [langgraph, rag, ai-workflow]
level: beginner → advanced
last_updated: 2026-01-31
reviewed_on: 2026-10-04
review_status: partial
document_type: index
---

# LangGraph 학습 시리즈

> [!info] 검토 — 2026-10-04
> 공식 근거·설치 판본·실행 검증은 [RAG 정리 기록](../organization-log.md)에 있다. 실제 모델 품질·사내 권한·운영 서비스는 미확인이다. 입력과 객체를 명시적으로 전달하는 학습 함수이며 자동 모델 호출은 하지 않는다.


> LangGraph를 활용한 AI 워크플로우 및 RAG 시스템 구축 가이드


## 학습 로드맵

```
1. 기초 개념 이해
   ↓
2. RAG 파이프라인 구축
   ↓
3. 고급 패턴 적용
```

## 문서 목차

| # | 문서 | 설명 | 난이도 |
|---|------|------|--------|
| 1 | [LangGraph 기초](./langgraph-basics.md) | State, Node, Edge 등 핵심 개념과 기본 그래프 구성 | ⭐ |
| 2 | [LangGraph RAG](./langgraph-rag.md) | 근거 판별·유한 재검색/재생성의 학습 루프 (논문 전체 구현 아님) | ⭐⭐ |
| 3 | [LangGraph 고급](./langgraph-advanced.md) | Human-in-the-loop, Subgraph, Persistence, Streaming | ⭐⭐⭐ |

## 사전 지식

- Python 기본 문법
- LangChain 기본 개념 (LLM, Chain, Prompt)
- RAG 기본 개념 (Retriever, Vector Store)

## 관련 문서

- [Advanced RAG 완전 가이드](../advanced-rag/README.md) — Agentic RAG + 멀티에이전트 통합
- AI/DT 전체 주제 목차는 이 시리즈의 필수 실행 의존성이 아니다.


## 실행 조건과 예제 차이

1의 분류 그래프는 상태/reducer·정확한 분류 출력 계약, 2의 RAG는 원래 질문과 검색어를 분리한 유한 루프, 3은 승인 대기/거절/재개·공유 key subgraph·SQLite 수명·stream mode를 다룬다. 최소 설치 예시는 같은 RAG 주제의 [조립 입문](../langchain-langgraph/langchain-langgraph-basics.md)을 따른다. SQLite 예제는 별도 `langgraph-checkpoint-sqlite==3.1.1`이 필요하다. Mermaid PNG 외부 렌더러 대신 Mermaid 텍스트를 반환한다.

이 시리즈와 조립 입문/고급 RAG는 용도가 달라 고유 예제를 유지했다. 완전한 중복 통합·재분류는 Claude 연결 실패로 보류했다. checkpointer·프롬프트·모델 판정은 인증된 승인이나 답변 사실성을 보장하지 않는다.
