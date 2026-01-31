# MCP (Model Context Protocol) 학습 노트

> LLM이 외부 도구와 데이터에 표준화된 방식으로 접근하기 위한 프로토콜

## 목차

### 기초
- [MCP 기초](./mcp-basics.md) - Server/Client 아키텍처, Tools/Resources/Prompts, FastMCP 서버 구현

### 통합
- [MCP + LangGraph 연동](./mcp-langgraph-integration.md) - langchain-mcp-adapters를 활용한 LangGraph Agent 연동

### 예정
- _MCP 서버 실전 패턴_ - 파일 시스템, DB 연동, API 래핑 등 실무 서버 구현
- _MCP Transport 심화_ - SSE, Streamable HTTP, 인증/보안

## 관련 문서
- [LangGraph 기초](../rag/langgraph/langgraph-basics.md)
- [LangGraph RAG](../rag/langgraph/langgraph-rag.md)
