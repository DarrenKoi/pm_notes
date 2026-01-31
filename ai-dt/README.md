# AI/DT 학습 노트

> AI/DT 시스템 개발 관련 학습 내용을 정리한다.

## 목차

### RAG (Retrieval-Augmented Generation)

#### LangGraph
- [LangGraph 시리즈 목차](./rag/langgraph/README.md)
  - [LangGraph 기초](./rag/langgraph/langgraph-basics.md) - State, Node, Edge, 기본 그래프 구성
  - [LangGraph RAG](./rag/langgraph/langgraph-rag.md) - Corrective RAG 파이프라인 구현
  - [LangGraph 고급](./rag/langgraph/langgraph-advanced.md) - Human-in-the-loop, Subgraph, Persistence, Streaming

#### Milvus
- [Milvus 시리즈 목차](./rag/milvus/README.md)
  - [Milvus 기초](./rag/milvus/milvus-basics.md) - 아키텍처, Collection, Index, 유사도 검색
  - [Milvus RAG 연동](./rag/milvus/milvus-rag-integration.md) - LangChain/LangGraph와 Milvus 통합

#### Hybrid Search
- [보조 용어 사전 DB (Elasticsearch)](./rag/auxiliary-glossary-db.md) - ES BM25로 전문 용어 관리, 쿼리 확장, LangGraph 통합

### MCP (Model Context Protocol)
- [MCP 시리즈 목차](./mcp/README.md)
  - [MCP 기초](./mcp/mcp-basics.md) - Server/Client 아키텍처, Tools/Resources/Prompts, FastMCP
  - [MCP + LangGraph 연동](./mcp/mcp-langgraph-integration.md) - langchain-mcp-adapters, ReAct Agent

### 데이터 처리 (Data Handling)
- _준비 중_
