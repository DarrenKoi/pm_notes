---
tags: [mcp, llm, tool-use, fastmcp]
level: beginner
last_updated: 2026-01-31
status: complete
---

# MCP 기초 (Model Context Protocol)

> LLM 애플리케이션이 외부 도구, 데이터, 프롬프트에 표준화된 방식으로 접근하는 오픈 프로토콜

## 왜 필요한가? (Why)

### LLM의 한계
LLM은 학습 데이터 내의 지식만 활용할 수 있다. 실시간 데이터 조회, 외부 API 호출, 파일 시스템 접근 등은 자체적으로 불가능하다.

### 기존 방식의 문제
- **Tool/Function Calling**: 각 LLM 프로바이더(OpenAI, Anthropic 등)마다 도구 정의 방식이 다름
- **커스텀 통합**: 도구를 추가할 때마다 애플리케이션 코드를 직접 수정해야 함
- **재사용 불가**: A 프로젝트에서 만든 도구를 B 프로젝트에서 쓰려면 다시 구현

### MCP가 해결하는 것
- **표준화**: 하나의 프로토콜로 도구/데이터/프롬프트를 정의하면, 어떤 MCP 호환 클라이언트든 사용 가능
- **플러그 앤 플레이**: MCP 서버만 띄우면 Claude Desktop, Cursor, VS Code 등에서 바로 연결
- **생태계**: 커뮤니티가 만든 MCP 서버를 가져다 쓸 수 있음 (GitHub, Slack, DB 등)

> USB-C가 다양한 기기를 하나의 규격으로 연결하듯, MCP는 LLM과 외부 세계를 하나의 프로토콜로 연결한다.

## 핵심 개념 (What)

### 아키텍처: Host / Client / Server

```
┌─────────────────────────────────┐
│  Host (Claude Desktop, IDE 등)    │
│                                   │
│  ┌───────────┐  ┌───────────┐   │
│  │  Client A  │  │  Client B  │   │      MCP Client는 Host 내부에
│  └─────┬─────┘  └─────┬─────┘   │      1:1로 Server와 연결
│        │               │         │
└────────┼───────────────┼─────────┘
         │               │
   ┌─────▼─────┐  ┌─────▼─────┐
   │  Server A  │  │  Server B  │         각 Server는 독립적으로
   │ (파일시스템) │  │  (DB 조회)  │         도구/데이터를 제공
   └───────────┘  └───────────┘
```

- **Host**: LLM 애플리케이션 (Claude Desktop, IDE, 커스텀 앱)
- **Client**: Host 내부에서 Server와 1:1 연결을 관리하는 프로토콜 클라이언트
- **Server**: 실제 기능을 제공하는 경량 프로그램

### 3대 기능 (Primitives)

| 기능 | 설명 | 제어 주체 | 예시 |
|------|------|-----------|------|
| **Tools** | LLM이 호출하는 함수 | 모델 (Model-controlled) | 웹 검색, DB 쿼리, 파일 쓰기 |
| **Resources** | LLM이 읽는 데이터 | 애플리케이션 (App-controlled) | 파일 내용, DB 레코드, API 응답 |
| **Prompts** | 재사용 가능한 프롬프트 템플릿 | 사용자 (User-controlled) | 코드 리뷰 프롬프트, 요약 템플릿 |

- **Tools**: 가장 핵심. LLM이 "이 도구를 호출하겠다"고 결정하면 서버가 실행
- **Resources**: REST API의 GET 엔드포인트와 유사. 데이터를 읽기 전용으로 제공
- **Prompts**: 슬래시 커맨드처럼 사용자가 선택하는 미리 정의된 상호작용 패턴

### Transport 방식

| Transport | 사용 환경 | 특징 |
|-----------|-----------|------|
| **stdio** | 로컬 프로세스 | 서버를 자식 프로세스로 실행, stdin/stdout 통신 |
| **Streamable HTTP** | 원격/네트워크 | HTTP 기반, 서버 → 클라이언트 스트리밍 지원 |
| **SSE** (레거시) | 원격/네트워크 | Server-Sent Events, Streamable HTTP로 대체 추세 |

로컬 개발 시에는 **stdio**가 가장 간단하다. 원격 배포 시 Streamable HTTP를 사용한다.

### Flask/FastAPI와의 관계

MCP는 HTTP 프레임워크가 아니라 **자체 프로토콜**이다.

- MCP 서버는 `mcp` Python SDK(FastMCP)로 만든다 — Flask/FastAPI로 만드는 것이 아님
- 단, HTTP transport를 선택하면 내부적으로 웹서버가 동작하므로, 기존 FastAPI 앱에 MCP 엔드포인트를 마운트할 수 있다
- stdio transport는 웹서버 없이 순수 프로세스 통신

```python
# FastAPI에 MCP 서버 마운트 (Streamable HTTP transport)
from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP

app = FastAPI()
mcp = FastMCP("my-server")

# MCP 도구 정의
@mcp.tool()
def hello(name: str) -> str:
    return f"Hello, {name}!"

# FastAPI에 마운트
app.mount("/mcp", mcp.streamable_http_app())
```

## 어떻게 사용하는가? (How)

### 1. 설치

```bash
# uv 사용 (권장)
uv add "mcp[cli]"

# pip 사용
pip install "mcp[cli]"
```

### 2. FastMCP로 서버 구현

```python
# server.py
from mcp.server.fastmcp import FastMCP

# 서버 인스턴스 생성
mcp = FastMCP("demo-server")

# Tool: LLM이 호출할 수 있는 함수
@mcp.tool()
def add(a: int, b: int) -> int:
    """두 수를 더한다."""
    return a + b

@mcp.tool()
def get_weather(city: str) -> str:
    """도시의 현재 날씨를 조회한다."""
    # 실제로는 외부 API 호출
    return f"{city}의 현재 날씨: 맑음, 15°C"

# Resource: 읽기 전용 데이터 제공
@mcp.resource("config://app-settings")
def get_config() -> str:
    """애플리케이션 설정을 반환한다."""
    return "debug=true, version=1.0"

# Dynamic Resource: URI 패턴으로 다양한 데이터 제공
@mcp.resource("users://{user_id}/profile")
def get_user_profile(user_id: str) -> str:
    """사용자 프로필을 반환한다."""
    return f"User {user_id}: name=홍길동, role=admin"

# Prompt: 재사용 가능한 프롬프트 템플릿
@mcp.prompt()
def review_code(code: str) -> str:
    """코드 리뷰 프롬프트를 생성한다."""
    return f"다음 코드를 리뷰해주세요. 버그, 성능, 가독성 관점에서 분석:\n\n{code}"

if __name__ == "__main__":
    mcp.run()  # stdio transport로 실행 (기본값)
```

### 3. MCP Inspector로 테스트

```bash
# Inspector 실행 (서버를 자동으로 띄워줌)
mcp dev server.py

# 브라우저에서 http://localhost:5173 접속
# → Tools, Resources, Prompts 탭에서 각 기능 테스트 가능
```

### 4. Claude Desktop에 연결

`claude_desktop_config.json`에 서버를 등록한다:

```json
{
  "mcpServers": {
    "demo-server": {
      "command": "uv",
      "args": ["run", "server.py"],
      "cwd": "/path/to/project"
    }
  }
}
```

등록 후 Claude Desktop을 재시작하면 도구가 자동으로 인식된다.

### 전체 흐름 요약

```
1. @mcp.tool()로 함수 정의
2. mcp dev server.py로 Inspector 테스트
3. Claude Desktop 또는 커스텀 클라이언트에 연결
4. LLM이 대화 중 필요할 때 자동으로 도구 호출
```

## 참고 자료 (References)

- [MCP 공식 문서](https://modelcontextprotocol.io/)
- [MCP Python SDK (GitHub)](https://github.com/modelcontextprotocol/python-sdk)
- [FastMCP 가이드](https://github.com/modelcontextprotocol/python-sdk/blob/main/README.md)
- [Anthropic MCP 교육 과정](https://github.com/anthropics/courses/tree/master/mcp_course)
- [MCP 서버 목록 (awesome-mcp-servers)](https://github.com/punkpeye/awesome-mcp-servers)

## 관련 문서
- [MCP + LangGraph 연동](./mcp-langgraph-integration.md)
- [MCP 시리즈 목차](./README.md)
- [LangGraph 기초](../rag/langgraph/langgraph-basics.md)
