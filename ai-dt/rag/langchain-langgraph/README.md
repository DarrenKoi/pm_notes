---
tags: [langchain, langgraph, rag, tool-calling, agent]
level: beginner → advanced
last_updated: 2026-04-08
reviewed_on: 2026-10-04
review_status: partial
document_type: index
category_major: "AI·DT"
category_middle: "RAG"
category_minor: "LangChain·Tool Calling"
note_kind: "목차"
classified_on: "2026-10-05"
---

# LangChain + LangGraph 실전 가이드

> [!info] 검토 — 2026-10-04
> 공식 근거·설치 판본·검증 결과는 [RAG 정리 기록](../organization-log.md)에 있다. 실제 모델·서버·회사 권한은 미확인이다. 함수/객체를 명시적으로 조립하는 학습 예제이며 자동 API 접속은 하지 않는다.


> LangChain으로 구성 요소를 만들고, LangGraph로 제어 흐름을 설계하는 방식으로 RAG/Tool Calling 에이전트를 만드는 학습 시리즈


## 이 시리즈로 배우는 것

- LangChain과 LangGraph의 역할 차이
- 단일 체인(Chain)에서 그래프(Graph)로 확장하는 방법
- RAG 파이프라인(수집/임베딩/검색/생성) 구축 방법
- Tool Calling(함수 호출)과 에이전트 루프 구성
- 운영 관점 확장(메모리, 평가, Guardrail, 관측성)

## 추천 학습 순서

1. [기초 사용법](./langchain-langgraph-basics.md)
2. [RAG + Tool Calling 실전](./rag-tool-calling-playbook.md)

## 사전 준비

- Python3.14.2 임시 환경에서 아래 판본의 로컬 fixture 검증. Python3.11+ 전체 호환성은 이번에 확인하지 않음
- 패키지 예시
  - `langchain`
  - `langgraph`
  - `langchain-openai` (또는 사용하는 모델 provider 패키지)
  - `langchain-community`, `langchain-text-splitters`
  - `faiss-cpu` 또는 `chromadb`
- 환경 변수
  - `LLM_MODEL`, `EMBEDDING_MODEL`, `OPENAI_API_KEY`(OpenAI 사용 시). 사내 호환 gateway는 승인된 base_url/api_key로 객체를 별도 구성하고 기능별 지원을 확인

## 함께 보면 좋은 기존 문서

- [LangGraph 시리즈 목차](../langgraph/README.md)
- [Milvus RAG 연동](../milvus/milvus-rag-integration.md)
- MCP 연동은 다른 ai-dt 주제의 기존 참고였으므로 이 목차에서는 기술 이름만 남긴다. RAG 예제 실행의 필수 의존성은 아니다.
