---
type: learning
tags: [mcp, sdk-version, langchain, integration]
aliases: [MCP 버전과 실행 조건]
reviewed_on: 2026-10-04
review_status: partial
category_major: "AI·DT"
category_middle: "에이전트 개발"
category_minor: "MCP 도구 연동"
note_kind: "검토 기록"
classified_on: "2026-10-05"
---

# MCP 버전과 실행 조건

## 목적과 읽는 방법

MCP는 통신 계약이고 SDK는 그 계약을 구현하는 라이브러리다. 사양 날짜, SDK 버전, host가 지원하는 기능을 따로 기록해야 한다. [기초](./mcp-basics.md)는 v1 FastMCP의 입문 예제, [LangGraph 연동](./mcp-langgraph-integration.md)은 독립 adapter의 기존 예제, [하네스](./harness-engineering-llm.md)는 앱 운영 설계다. 이름이 비슷해도 서로 대체하는 문서는 아니다.

## 확인한 버전 경계 · 2026-10-04

| 대상 | 확인 근거 | 이 묶음에 적용하는 조건 |
|---|---|---|
| MCP 사양 `2025-11-25` | [transport 사양](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports) | stdio·Streamable HTTP를 설명할 때의 이전 구현 맥락 |
| MCP 사양 `2026-07-28` | [변경 기록](https://modelcontextprotocol.io/specification/2026-07-28/changelog) | session/초기화 handshake 제거, 요청별 버전·capabilities, `server/discover` 등 변경. 이전 세션 기반 API와 동일한 wire 동작으로 간주하지 않음 |
| 공식 Python SDK v1 `1.26.0` | [고정 tag README](https://github.com/modelcontextprotocol/python-sdk/blob/v1.26.0/README.md) | 기존 `mcp.server.fastmcp.FastMCP` 예제를 읽기 위한 기준. 실제 설치/프로토콜 통합 실행은 미검증 |
| 공식 Python SDK v2 안내 | [공식 migration](https://py.sdk.modelcontextprotocol.io/migration/) | `MCPServer`로 import·API가 변경됨. 공식 SDK v1의 FastMCP와 별도 `fastmcp` 패키지를 이름만 보고 혼용하지 않음 |
| 독립 `langchain-mcp-adapters` main `0.3.2` | [pyproject](https://github.com/langchain-ai/langchain-mcp-adapters/blob/main/pyproject.toml), [유지보수 중단 안내](https://github.com/langchain-ai/langchain-mcp-adapters) | mcp 의존 조건 `>=1.24,<2`; 해당 예제는 v1 SDK와 함께 읽음. npm/PyPI의 최신 배포라는 주장 아님 |
| 새 `langchain[mcp]` 통합 | [공식 migration](https://docs.langchain.com/oss/python/migrate/langchain-mcp-adapters) | `MCPAdapter`와 `list_tools()`로 전환. 현재 로컬 LangChain `1.2.15`에는 이 namespace가 없음을 확인했으므로 그대로 실행 가능하다고 주장하지 않음 |

`2026-07-28` 변경에서 legacy HTTP+SSE는 Deprecated로 명시된다. Streamable HTTP 응답이 SSE 형식을 사용할 수 있다는 것과 예전 HTTP+SSE transport는 다른 개념이다. 새 사양을 지원하려면 양쪽의 호환 범위를 확인해야 하며, 패키지 이름·host 이름만으로 호환을 확정하지 않는다.

## SDK v2를 선택할 때

공식 migration은 v1의 `from mcp.server.fastmcp import FastMCP` 대신 다음 import를 설명한다.

```python
from mcp.server import MCPServer

mcp = MCPServer("demo")

@mcp.tool()
def add(a: int, b: int) -> int:
    """두 수를 더한다."""
    return a + b
```

이것은 **서버 정의 부분**이다. transport 실행·ASGI lifespan·클라이언트 연결까지 완성한 예제가 아니다. v1의 constructor/run/app 인자를 이름만 바꾸어 이식하지 말고 migration을 확인한다. 현재 로컬에는 mcp가 없어 v2 import/실행은 미검증이다.

## LangChain 새 통합을 선택할 때

기존 독립 adapter는 `MultiServerMCPClient(...)`를 만들고 `await client.get_tools()`로 목록을 가져온다. 확인한 새 안내는 `MCPAdapter` context와 `await adapter.list_tools()`를 사용한다. 아래는 공식 구조에 맞춘 **비실행 설명 예제**이며 namespace를 지원하는 LangChain 버전과 provider 인증이 먼저 필요하다.

```python
import os
import sys
from pathlib import Path
from typing import Any
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter

async def run_math_agent(server_path: Path) -> dict[str, Any]:
    config = {"mcpServers": {"math": {
        "command": sys.executable,
        "args": [str(server_path.resolve())],
    }}}
    async with MCPAdapter(config) as adapter:
        tools = await adapter.list_tools()
        agent = create_agent(os.environ["LANGCHAIN_MODEL"], tools)
        return await agent.ainvoke({"messages": "3과 5를 더하고 2를 곱해줘"})
```

adapter 수명 안에서 도구를 사용한다. `LANGCHAIN_MODEL`에는 설치 provider가 지원하고 계정에서 사용할 수 있는 ID를 넣는다. tool name prefix·인증·prompt/resource 지원도 이전 adapter와 다를 수 있으므로 migration의 차이 표를 확인한다. 로컬에서 이 새 통합을 설치·API 호출하지 않았다.

## 예제를 실행하기 전에

- 각 code fence의 주석 파일명에 맞춰 같은 시험 모듈 안에 파일을 준비한다. 문서 fence는 자동으로 실제 파일이 되는 것이 아니다.
- v1 예제와 v2 예제의 별도 환경을 사용하고 dependency 버전을 기록한다. `uv add`는 프로젝트 의존 파일을 바꾸므로 시험 환경에서 수행한다.
- stdio의 command는 실행할 환경의 Python이고 서버 args는 실제 파일 경로다. HTTP 예제는 이미 떠 있는 MCP 서버와 정확한 endpoint·인증을 전제한다.
- `mcp dev`는 개발 Inspector를 실행한다. 출력 URL을 확인하며 고정 port를 가정하지 않는다. 실제 도구 실행과 서버 로그를 확인한다.
- 고정 날씨 문자열과 가상 agent 응답은 현재 외부 데이터나 실제 모델 품질의 증거가 아니다. schema 통과와 도구 실행 성공도 구분한다.

## 검증과 남은 미확인

[정리 기록](./organization-log.md)에 문서별 변경과 로컬 검증을 남겼다. 공식 사양·소스 대조와 구문/가짜 클라이언트 fixture는 실제 SDK 통신·HTTP·인증·모델 호출의 대체 증거가 아니다. 버전 고정 환경의 실제 stdio/HTTP 왕복, host 연결, 현재 제품 접근 권한은 미완료다.
