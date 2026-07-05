---
tags: [langgraph, conditional-edges, branching, toolnode, workflow]
level: intermediate
last_updated: 2026-07-06
---

# 07. Condition, Branching, Tool 결합 Workflow 실습

> 조건부 엣지로 분기·루프를 만들고, `ToolNode`/`tools_condition`으로 도구 호출 루프를 그래프에 통합한다.

## 왜 필요한가? (Why)

- 실무 흐름은 데이터에 따라 갈린다: "문서가 충분한가? → Yes 생성 / No 재검색", "도구 호출이 필요한가? → Yes 도구 / No 종료".
- 이런 **동적 라우팅과 루프**가 LangGraph를 LCEL Chain과 구분 짓는 핵심 기능이다.
- 도구 호출 Agent도 결국 "모델 ↔ 도구" 사이를 오가는 **루프**이므로, 조건부 엣지로 그래프에 자연스럽게 표현된다.

## 핵심 개념 (What)

### `add_conditional_edges`
노드 다음에 **라우터 함수**를 두고, 그 반환값(문자열)으로 다음 노드를 고른다.

```python
def route(state) -> str:
    return "generate" if state["enough_docs"] else "research"

builder.add_conditional_edges(
    "grade",                       # 이 노드 실행 후
    route,                         # 라우터 함수 (state → 키)
    {"generate": "generate", "research": "research"},   # 키 → 목적지 노드
)
```

### 루프
조건부 엣지가 **이전 노드로 되돌아가면** 루프가 된다(예: 재검색). 무한 루프 방지를 위해 상태에 카운터를 두거나 `recursion_limit`을 설정한다.

### 도구 통합 3종 세트 (prebuilt)
- **`ToolNode(tools)`**: 마지막 AI 메시지의 `tool_calls`를 실행해 `ToolMessage`로 돌려주는 노드.
- **`tools_condition`**: "AI가 도구를 부르려 하면 `tools`로, 아니면 `END`로" 라우팅하는 기성 라우터.

## 어떻게 사용하는가? (How)

### 1) 조건 분기 (감정 라우팅 예)
```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="Kimi-K2.5", base_url="http://llm-gateway.internal/v1",
                 api_key="EMPTY", temperature=0)

class State(TypedDict):
    text: str
    sentiment: str
    reply: str

def classify(state):
    s = llm.invoke(f"다음이 긍정/부정 중 뭐야? 단어 하나로: {state['text']}").content
    return {"sentiment": "negative" if "부정" in s else "positive"}

def handle_positive(state): return {"reply": "감사합니다! 😊"}
def handle_negative(state): return {"reply": "불편을 드려 죄송합니다. 상담 연결해 드릴게요."}

def route(state) -> str:
    return state["sentiment"]

b = StateGraph(State)
b.add_node("classify", classify)
b.add_node("handle_positive", handle_positive)
b.add_node("handle_negative", handle_negative)
b.add_edge(START, "classify")
b.add_conditional_edges("classify", route,
                        {"positive": "handle_positive", "negative": "handle_negative"})
b.add_edge("handle_positive", END)
b.add_edge("handle_negative", END)
graph = b.compile()

print(graph.invoke({"text": "배송이 너무 느려요"})["reply"])
```

### 2) 도구 호출 루프 (`ToolNode` + `tools_condition`)
```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool

@tool
def get_status(eqp_id: str) -> str:
    """설비 상태 조회."""
    return f"{eqp_id}: DOWN, 알람=온도과열"

tools = [get_status]
llm_with_tools = llm.bind_tools(tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def agent(state):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

b = StateGraph(State)
b.add_node("agent", agent)
b.add_node("tools", ToolNode(tools))
b.add_edge(START, "agent")
# agent가 도구를 부르면 "tools"로, 아니면 END로
b.add_conditional_edges("agent", tools_condition)
b.add_edge("tools", "agent")     # 도구 결과를 다시 agent에게 → 루프
graph = b.compile()

out = graph.invoke({"messages": [("user", "EQP-102 상태 알려줘")]})
print(out["messages"][-1].content)
```

이 그래프의 흐름: `agent → (도구 필요?) → tools → agent → (더 필요?) → ... → END`. 이것이 곧 ReAct Agent의 뼈대다.

### 3) 루프 안전장치
```python
graph.invoke(inputs, {"recursion_limit": 10})   # 무한 루프 방지
```
또는 상태에 `iterations: int`를 두고 라우터에서 상한 도달 시 `END`로 보낸다.

### 4) 병렬 분기(fan-out/fan-in)
한 노드에서 여러 노드로 엣지를 뻗으면 **병렬 실행**되고, 공통 다음 노드에서 합류한다. 이때 병렬로 쓰는 상태 키는 reducer(`add` 등)로 병합해야 충돌이 없다.

## 관련 문서
- [05. LangGraph 개요 & 상태 머신](./05-langgraph-overview-state-machine.md)
- [08. 시나리오 기반 Agent 구축](./08-langgraph-agent-scenario.md) — 이 패턴을 prebuilt로 축약
- [11. RAG 질의응답 흐름](./11-rag-qa-flow.md) — CRAG 루프에 조건부 엣지 활용

## 참고 자료 (References)
- Conditional edges: https://langchain-ai.github.io/langgraph/concepts/low_level/#conditional-edges
- `ToolNode`, `tools_condition` (langgraph-prebuilt)
