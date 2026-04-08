# LangChain + LangGraph 실전 가이드

> LangChain으로 구성 요소를 만들고, LangGraph로 제어 흐름을 설계하는 방식으로 RAG/Tool Calling 에이전트를 만드는 학습 시리즈

---
tags: [langchain, langgraph, rag, tool-calling, agent]
level: beginner → advanced
last_updated: 2026-04-08
status: in-progress
---

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

- Python 3.11+
- 패키지 예시
  - `langchain`
  - `langgraph`
  - `langchain-openai` (또는 사용하는 모델 provider 패키지)
  - `langchain-community`
  - `faiss-cpu` 또는 `chromadb`
- 환경 변수
  - `OPENAI_API_KEY` (OpenAI 사용 시)

## 함께 보면 좋은 기존 문서

- [LangGraph 시리즈 목차](../langgraph/README.md)
- [Milvus RAG 연동](../milvus/milvus-rag-integration.md)
- [MCP + LangGraph 연동](../../mcp/mcp-langgraph-integration.md)
