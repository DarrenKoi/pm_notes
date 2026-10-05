---
type: index
tags: [mcp, learning-index]
reviewed_on: 2026-10-04
review_status: partial
category_major: "AI·DT"
category_middle: "에이전트 개발"
category_minor: "MCP 도구 연동"
note_kind: "목차"
classified_on: "2026-10-05"
---

# MCP (Model Context Protocol) 학습 노트

> LLM이 외부 도구와 데이터에 표준화된 방식으로 접근하기 위한 프로토콜

> [!info] 검토 범위 · 2026-10-04
> 프로토콜 사양과 SDK 버전은 별개다. [버전·실행 조건](./version-and-execution-notes.md)과 [정리 기록](./organization-log.md)을 먼저 확인한다. 원래 작성일은 보존했으며 실제 API·원격 서버 실행은 미검증이다.

## 읽기 순서

1. [버전·실행 조건](./version-and-execution-notes.md): 현재 사양과 기존 예제의 경계를 확인한다.
2. MCP 기초: host/client/server와 도구 실행 흐름을 익힌다.
3. LangGraph 연동: 도구를 agent 실행에 연결하고 수명을 관리한다.
4. [LLM 하네스](./harness-engineering-llm.md): MCP를 포함한 실행·평가·관측 설계를 살핀다. 프로토콜 입문과 운영 방법론은 용도가 다르다.

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
