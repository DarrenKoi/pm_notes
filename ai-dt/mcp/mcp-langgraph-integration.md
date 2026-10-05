---
tags: [mcp, langgraph, langchain, agent, tool-use]
level: intermediate
last_updated: 2026-01-31
reviewed_on: 2026-10-04
review_status: partial
type: learning
category_major: "AI·DT"
category_middle: "에이전트 개발"
category_minor: "MCP 도구 연동"
note_kind: "학습"
classified_on: "2026-10-05"
---

# MCP + LangGraph 연동

> langchain-mcp-adapters를 사용하여 MCP 서버의 도구를 LangGraph Agent에서 활용하는 방법

> [!info] 검토 범위 · 2026-10-04
> 프로토콜 사양과 SDK 버전은 별개다. [버전·실행 조건](./version-and-execution-notes.md)과 [정리 기록](./organization-log.md)을 먼저 확인한다. 원래 작성일은 보존했으며 실제 API·원격 서버 실행은 미검증이다.

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
inputSchema  →                    args_schema (JSON schema)
tools/call   →                    async tool invocation
```

## 어떻게 사용하는가? (How)

### 설치

아래는 유지보수 중단된 **독립 adapter의 기존 구조를 읽기 위한 예제**다. 현재 신규 통합은 별도 버전 조건 문서의 `langchain[mcp]`/`MCPAdapter` 안내를 확인한다. `main`에서 확인한 adapter 소스 버전은 `0.3.2`이며 mcp 의존 범위는 `>=1.24,<2`다. 패키지 배포와 전체 dependency 조합은 로컬 검증하지 않았다.

```bash
pip install "langchain-mcp-adapters==0.3.2" "mcp==1.26.0" langchain langgraph langchain-openai
# 또는
uv add "langchain-mcp-adapters==0.3.2" "mcp==1.26.0" langchain langgraph langchain-openai
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
import sys
from pathlib import Path
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

async def main() -> None:
    model = ChatOpenAI(model="gpt-4o")

    # 독립 adapter API: client 생성은 동기, 도구 조회는 await
    client = MultiServerMCPClient(
        {
            "math": {
                "command": sys.executable,
                "args": [str(Path(__file__).with_name("math_server.py"))],
                "transport": "stdio",
            }
        }
    )
    # MCP 도구를 LangChain Tool로 변환
    tools = await client.get_tools()

    # ReAct Agent 생성
    agent = create_agent(model, tools)

    # 실행
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "3과 5를 더하고, 그 결과에 2를 곱해줘"}]}
    )

    for msg in result["messages"]:
        print(f"[{msg.type}] {msg.content}")

asyncio.run(main())
```

**실행 결과** (가상 예시 · 로컬/API 실행 기록 아님):
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
    """학습용 고정 날씨 문자열을 반환한다."""
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
import sys
from pathlib import Path
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

async def main() -> None:
    model = ChatOpenAI(model="gpt-4o")

    # 여러 MCP 서버를 동시에 연결
    client = MultiServerMCPClient(
        {
            "math": {
                "command": sys.executable,
                "args": [str(Path(__file__).with_name("math_server.py"))],
                "transport": "stdio",
            },
            "weather": {
                "command": sys.executable,
                "args": [str(Path(__file__).with_name("weather_server.py"))],
                "transport": "stdio",
            },
        }
    )
    # 모든 서버의 도구가 합쳐져서 반환됨
    tools = await client.get_tools()
    print(f"사용 가능한 도구: {[t.name for t in tools]}")
    # → ['add', 'multiply', 'get_weather']

    agent = create_agent(model, tools)

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "서울의 예시 날씨를 보여주고, 도구에 저장된 기온 3도에 -5를 더해줘"}]}
    )

    for msg in result["messages"]:
        print(f"[{msg.type}] {msg.content}")

asyncio.run(main())
```

### 예제 3: Streamable HTTP Transport (원격 MCP 서버)

이미 실행 중인 원격 MCP 서버에 연결하는 경우:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from typing import Any

async def load_remote_agent() -> Any:
    client = MultiServerMCPClient(
        {"remote-tools": {"url": "http://localhost:8000/mcp/", "transport": "http"}}
    )
    tools = await client.get_tools()
    return create_agent(ChatOpenAI(model="gpt-4o"), tools)
```

### 실행 조건과 상태

`get_tools()`로 가져온 목록은 실행 시점의 snapshot이다. 설정만 바꾸어 이미 만든 agent에 도구가 즉시 추가되는 것은 아니다. 서버 프로세스·명령 경로·인증·모델 tool calling 지원을 확인하고 목록을 다시 조회·bind해야 한다. 독립 adapter의 기본 호출은 도구 호출마다 새 세션을 열므로 서버 메모리 상태가 이어질 것으로 가정하지 않는다. 상태가 필요하면 해당 버전의 `client.session(...)` 수명 안에서 도구를 로드하고 실행한다. 새 `MCPAdapter`에서는 도구 사용을 adapter context 안에서 끝낸다.

`get_weather`는 현재 날씨가 아니라 고정 문자열을 반환한다. 응답 문자열에서 기온을 읽어 숫자 도구를 호출하는 것은 모델의 해석이며 structured 수치 계약이 아니다. `gpt-4o`는 원래 예제 ID이며 계정 접근·현재 API 사용 가능성은 미검증이다.

### 구조 요약

```
┌─────────────────────────────────────────┐
│            LangGraph Agent              │
│  ┌─────────────────────────────────┐    │
│  │    create_agent(model,          │    │
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
