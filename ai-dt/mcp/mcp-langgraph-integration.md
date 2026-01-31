---
tags: [mcp, langgraph, langchain, agent, tool-use]
level: intermediate
last_updated: 2026-01-31
status: complete
---

# MCP + LangGraph 연동

> langchain-mcp-adapters를 사용하여 MCP 서버의 도구를 LangGraph Agent에서 활용하는 방법

## 왜 필요한가? (Why)

### LangGraph Agent의 도구 확장
LangGraph Agent는 도구(Tool)를 호출하며 작업을 수행한다. 기본적으로는 `@tool` 데코레이터로 Python 함수를 도구로 정의하지만, 이 방식은:

- 도구가 Agent 코드에 직접 종속됨
- 다른 프로젝트와 도구 공유가 어려움
- 도구 추가/변경 시 Agent 코드를 수정해야 함

### MCP 연동의 이점
- **분리**: 도구 로직(MCP 서버)과 Agent 로직(LangGraph)을 독립적으로 개발/배포
- **재사용**: 하나의 MCP 서버를 여러 Agent가 공유
- **동적 확장**: Agent 코드 변경 없이 MCP 서버만 추가하면 도구 확장
- **생태계 활용**: 커뮤니티 MCP 서버를 Agent에 바로 연결

## 핵심 개념 (What)

### langchain-mcp-adapters

MCP 도구를 LangChain/LangGraph 호환 `BaseTool`로 변환해주는 어댑터 라이브러리.

```
MCP Server → MCP Client → langchain-mcp-adapters → LangChain BaseTool → LangGraph Agent
```

### 주요 컴포넌트

| 컴포넌트 | 역할 |
|----------|------|
| `MultiServerMCPClient` | 여러 MCP 서버에 동시 연결하는 클라이언트 |
| `load_mcp_tools()` | MCP 서버의 도구를 LangChain Tool로 변환 |

### Tool 변환 과정

MCP 도구의 스키마가 자동으로 LangChain Tool로 매핑된다:

```
MCP Tool                          LangChain Tool
──────────                        ──────────────
name         →                    name
description  →                    description
inputSchema  →                    args_schema (Pydantic)
call()       →                    _run() / _arun()
```

## 어떻게 사용하는가? (How)

### 설치

```bash
pip install langchain-mcp-adapters langgraph langchain-openai
# 또는
uv add langchain-mcp-adapters langgraph langchain-openai
```

### 예제 1: 단일 MCP 서버 + ReAct Agent

#### Step 1: MCP 서버 준비

```python
# math_server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """두 수를 더한다."""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """두 수를 곱한다."""
    return a * b

if __name__ == "__main__":
    mcp.run()
```

#### Step 2: LangGraph Agent에서 MCP 도구 사용

```python
# agent.py
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

async def main():
    model = ChatOpenAI(model="gpt-4o")

    # MCP 클라이언트를 context manager로 사용
    async with MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["math_server.py"],
                "transport": "stdio",
            }
        }
    ) as client:
        # MCP 도구를 LangChain Tool로 변환
        tools = client.get_tools()

        # ReAct Agent 생성
        agent = create_react_agent(model, tools)

        # 실행
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "3과 5를 더하고, 그 결과에 2를 곱해줘"}]}
        )

        for msg in result["messages"]:
            print(f"[{msg.type}] {msg.content}")

asyncio.run(main())
```

**실행 결과** (예시):
```
[human] 3과 5를 더하고, 그 결과에 2를 곱해줘
[ai] (tool_calls: add(a=3, b=5))
[tool] 8
[ai] (tool_calls: multiply(a=8, b=2))
[tool] 16
[ai] 3과 5를 더하면 8이고, 8에 2를 곱하면 16입니다.
```

### 예제 2: 여러 MCP 서버 동시 연결

```python
# weather_server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather")

@mcp.tool()
def get_weather(city: str) -> str:
    """도시의 현재 날씨를 조회한다."""
    weather_data = {
        "서울": "맑음, 3°C",
        "부산": "흐림, 7°C",
        "제주": "비, 10°C",
    }
    return weather_data.get(city, f"{city}: 데이터 없음")

if __name__ == "__main__":
    mcp.run()
```

```python
# multi_agent.py
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

async def main():
    model = ChatOpenAI(model="gpt-4o")

    # 여러 MCP 서버를 동시에 연결
    async with MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["math_server.py"],
                "transport": "stdio",
            },
            "weather": {
                "command": "python",
                "args": ["weather_server.py"],
                "transport": "stdio",
            },
        }
    ) as client:
        # 모든 서버의 도구가 합쳐져서 반환됨
        tools = client.get_tools()
        print(f"사용 가능한 도구: {[t.name for t in tools]}")
        # → ['add', 'multiply', 'get_weather']

        agent = create_react_agent(model, tools)

        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "서울 날씨 알려주고, 기온에서 영하 5도를 더해줘"}]}
        )

        for msg in result["messages"]:
            print(f"[{msg.type}] {msg.content}")

asyncio.run(main())
```

### 예제 3: SSE Transport (원격 MCP 서버)

이미 실행 중인 원격 MCP 서버에 연결하는 경우:

```python
async with MultiServerMCPClient(
    {
        "remote-tools": {
            "url": "http://localhost:8000/mcp",
            "transport": "streamable_http",
        }
    }
) as client:
    tools = client.get_tools()
    agent = create_react_agent(model, tools)
    # ...
```

### 구조 요약

```
┌─────────────────────────────────────────┐
│            LangGraph Agent              │
│  ┌─────────────────────────────────┐    │
│  │    create_react_agent(model,    │    │
│  │           tools=[...])          │    │
│  └────────────┬────────────────────┘    │
│               │                         │
│  ┌────────────▼────────────────────┐    │
│  │    MultiServerMCPClient         │    │
│  │    ┌──────────┐ ┌──────────┐   │    │
│  │    │ Client A │ │ Client B │   │    │
│  │    └────┬─────┘ └────┬─────┘   │    │
│  └─────────┼────────────┼─────────┘    │
└────────────┼────────────┼──────────────┘
             │            │
      ┌──────▼──────┐ ┌──▼──────────┐
      │ MCP Server  │ │ MCP Server  │
      │  (math)     │ │ (weather)   │
      └─────────────┘ └─────────────┘
```

## 참고 자료 (References)

- [langchain-mcp-adapters (GitHub)](https://github.com/langchain-ai/langchain-mcp-adapters)
- [LangChain MCP 문서](https://python.langchain.com/docs/integrations/tools/mcp/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph/)

## 관련 문서
- [MCP 기초](./mcp-basics.md)
- [MCP 시리즈 목차](./README.md)
- [LangGraph 기초](../rag/langgraph/langgraph-basics.md)
- [LangGraph RAG](../rag/langgraph/langgraph-rag.md)
