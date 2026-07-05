---
tags: [langgraph, memory, checkpointer, conversation, state]
level: intermediate
last_updated: 2026-07-06
---

# 06. 상태 기반 멀티스텝 대화 흐름 설계

> 여러 턴에 걸친 대화에서 맥락을 유지하려면 상태에 메시지를 누적하고, checkpointer로 대화를 저장/재개해야 한다. 그 방법을 다룬다.

## 왜 필요한가? (Why)

- 챗봇은 이전 발화를 기억해야 한다("그거 더 자세히" ← "그거"가 뭔지 알아야 함).
- 매 요청마다 전체 히스토리를 앱이 직접 관리하면 번거롭다. LangGraph의 **checkpointer**는 `thread_id`별로 상태를 자동 저장해, 다음 호출 때 이어서 진행한다.
- 사내 멀티스텝 시나리오(예: "설비 조회 → 이상 확인 → 조치 추천"을 대화로 진행)에서 상태 유지가 필수다.

## 핵심 개념 (What)

### `add_messages` reducer
대화는 메시지가 계속 쌓이는 구조다. 상태의 메시지 필드에 `add_messages` reducer를 붙이면, 노드가 반환한 메시지가 **덮어쓰기가 아니라 append**된다(같은 id는 갱신).

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
```

`MessagesState`라는 내장 단축형도 있다:
```python
from langgraph.graph import MessagesState   # messages 필드가 미리 정의됨
```

### Checkpointer (메모리)
- **`InMemorySaver`**: 프로세스 메모리에 저장(개발/테스트용).
- **`SqliteSaver` / `PostgresSaver`**: 영속 저장(운영). 재시작해도 대화 유지.
- 실행 시 `config={"configurable": {"thread_id": "..."}}`로 대화 세션을 구분한다.

## 어떻게 사용하는가? (How)

### 메모리 있는 챗봇 그래프
```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI

# 사내:
llm = ChatOpenAI(model="Kimi-K2.5", base_url="http://llm-gateway.internal/v1",
                 api_key="EMPTY", temperature=0)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}   # 히스토리 전체를 모델에 전달

b = StateGraph(State)
b.add_node("chatbot", chatbot)
b.add_edge(START, "chatbot")
b.add_edge("chatbot", END)

memory = InMemorySaver()
graph = b.compile(checkpointer=memory)     # ← 메모리 연결

# 대화 세션 1
cfg = {"configurable": {"thread_id": "user-42"}}
graph.invoke({"messages": [("user", "내 이름은 대영이야.")]}, cfg)
out = graph.invoke({"messages": [("user", "내 이름 뭐라고 했지?")]}, cfg)
print(out["messages"][-1].content)   # "대영" 이라고 기억함 (같은 thread_id 덕분)
```

### 영속 저장 (운영)
```python
# pip install langgraph-checkpoint-sqlite
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("checkpoints.sqlite") as memory:
    graph = b.compile(checkpointer=memory)
    graph.invoke({"messages": [("user", "안녕")]},
                 {"configurable": {"thread_id": "user-42"}})
# 프로세스를 껐다 켜도 user-42 대화는 그대로 이어짐
```

### 긴 대화의 토큰 관리 (최신 기법)
히스토리가 길어지면 토큰 한도를 넘는다. 노드에서 **메시지를 잘라내거나(trim) 요약**한다.
```python
from langchain_core.messages import trim_messages

def chatbot(state: State) -> dict:
    trimmed = trim_messages(
        state["messages"], max_tokens=4000, strategy="last",
        token_counter=llm, include_system=True,
    )
    return {"messages": [llm.invoke(trimmed)]}
```
> 요약 메모리 패턴: 오래된 메시지를 LLM으로 요약해 하나의 SystemMessage로 압축하고, 최근 메시지만 원문 유지.

### 상태 조회/수정 (디버깅·감사)
```python
snapshot = graph.get_state(cfg)      # 현재 상태 스냅샷
print(snapshot.values["messages"])
# graph.update_state(cfg, {"messages": [...]})  # 상태 수동 수정 가능
```

## 관련 문서
- [05. LangGraph 개요 & 상태 머신](./05-langgraph-overview-state-machine.md)
- [07. Condition/Branching/Tool workflow](./07-condition-branching-tool-workflow.md)
- [08. 시나리오 기반 Agent 구축](./08-langgraph-agent-scenario.md)

## 참고 자료 (References)
- Persistence / checkpointer: https://docs.langchain.com/oss/python/langgraph/overview
- `add_messages`, `MessagesState`, `trim_messages`
