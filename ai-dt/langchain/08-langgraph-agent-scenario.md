---
tags: [langgraph, react-agent, create-react-agent, human-in-the-loop, scenario]
level: advanced
last_updated: 2026-07-06
---

# 08. 사용자 시나리오 기반 LangGraph Agent 구축

> 실제 업무 시나리오를 하나 잡아, LangGraph prebuilt `create_react_agent`와 커스텀 그래프를 함께 써서 메모리·승인(HITL)까지 갖춘 Agent를 만든다.

## 왜 필요한가? (Why)

- 07번까지의 조각(도구 루프, 조건 분기, 메모리)을 **하나의 실사용 에이전트**로 합치는 연습이 필요하다.
- 실무 Agent는 (1) 여러 도구를 쓰고, (2) 대화를 기억하고, (3) 위험한 동작 전엔 사람 승인을 받아야 한다.
- 사내 시나리오: **"설비 이상 대응 어시스턴트"** — 상태 조회 → 원인 분석 → (조치 전) 사람 승인 → 조치 요청.

## 핵심 개념 (What)

### `create_react_agent` (가장 빠른 길)
07번의 "agent ↔ tools 루프"를 LangGraph 그래프로 만든다. 일반 Agent는 [03번](./03-chain-agent-tool.md)의 LangChain `create_agent`가 더 단순하지만, checkpointer, HITL, 조건 분기, 상태 조회/수정이 필요한 시나리오는 LangGraph로 설계한다.

### Human-in-the-loop (HITL)
`interrupt`로 그래프를 특정 지점에서 **일시정지**하고 사람 입력을 기다린다. 승인/거부/수정 후 `Command(resume=...)`로 재개한다. 삭제·변경 같은 부작용 도구 앞에 둔다.

## 어떻게 사용하는가? (How)

### 시나리오 A: prebuilt ReAct Agent + 메모리
```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="Kimi-K2.5", base_url="http://llm-gateway.internal/v1",
                 api_key="EMPTY", temperature=0)

@tool
def get_status(eqp_id: str) -> str:
    """설비 상태/알람 조회."""
    return f"{eqp_id}: DOWN, 알람=온도과열"

@tool
def get_manual(alarm: str) -> str:
    """알람명으로 대응 매뉴얼 조회."""
    return f"{alarm}: 1) 쿨링 확인 2) 팬 점검 3) 재기동"

agent = create_react_agent(
    llm,
    tools=[get_status, get_manual],
    prompt="너는 반도체 설비 이상 대응 어시스턴트다. 도구로 사실을 확인한 뒤 조치를 제안해.",
    checkpointer=InMemorySaver(),
)

cfg = {"configurable": {"thread_id": "shift-A"}}
out = agent.invoke(
    {"messages": [("user", "EQP-102가 이상해. 확인하고 조치 알려줘.")]}, cfg)
print(out["messages"][-1].content)

# 이어지는 대화(메모리 유지)
out2 = agent.invoke({"messages": [("user", "그 조치 순서 다시 정리해줘")]}, cfg)
print(out2["messages"][-1].content)
```

### 시나리오 B: 승인 게이트(HITL)가 있는 커스텀 그래프
"조치 요청(부작용)" 전에 사람이 승인해야 하는 흐름.
```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

class State(TypedDict):
    messages: Annotated[list, add_messages]
    proposed_action: str

def analyze(state):
    ai = llm.invoke(state["messages"] + [("user", "권장 조치를 한 줄로 제안해.")])
    return {"proposed_action": ai.content}

def approval(state):
    decision = interrupt({"proposed_action": state["proposed_action"]})  # ⏸️ 사람 대기
    if decision == "approve":
        return {"messages": [("assistant", f"조치 실행 요청: {state['proposed_action']}")]}
    return {"messages": [("assistant", "조치가 거부되었습니다.")]}

b = StateGraph(State)
b.add_node("analyze", analyze)
b.add_node("approval", approval)
b.add_edge(START, "analyze")
b.add_edge("analyze", "approval")
b.add_edge("approval", END)
graph = b.compile(checkpointer=InMemorySaver())

cfg = {"configurable": {"thread_id": "case-1"}}
# 1) 실행하면 approval에서 멈춤
graph.invoke({"messages": [("user", "EQP-102 온도과열 대응해줘")]}, cfg)
# 2) 사람이 승인 → 재개
result = graph.invoke(Command(resume="approve"), cfg)
print(result["messages"][-1].content)
```

### 시나리오 설계 팁
1. **도구를 먼저 정의**하고, 각 도구의 docstring을 시나리오 언어로 쓴다.
2. **읽기 도구는 자유롭게, 쓰기 도구는 HITL 뒤에** 둔다.
3. 프롬프트에 "반드시 도구로 사실을 확인한 뒤 답하라"를 명시해 환각을 줄인다.
4. `thread_id`를 근무조/케이스 단위로 설계해 대화 맥락을 분리한다.
5. 관측성: `graph.stream(..., stream_mode="updates")`로 각 스텝을 로깅한다.

## 관련 문서
- [03. Chain · Agent · Tool](./03-chain-agent-tool.md) · [04. 외부 API 연동 Agent](./04-external-api-agent.md)
- [07. Condition/Branching/Tool workflow](./07-condition-branching-tool-workflow.md) — 이 에이전트의 내부 구조
- [11. RAG 질의응답 흐름](./11-rag-qa-flow.md) — Agentic RAG로 확장

## 참고 자료 (References)
- LangGraph overview: https://docs.langchain.com/oss/python/langgraph/overview
- Agents / `create_agent`: https://docs.langchain.com/oss/python/langchain/agents
