---
tags: [langgraph, state-machine, stategraph, basics]
level: intermediate
last_updated: 2026-07-06
---

# 05. LangGraph 개요 및 상태 머신 이해

> LangGraph는 LLM 앱을 "상태(State)를 공유하는 노드(Node)들이 엣지(Edge)로 연결된 그래프"로 모델링한다. 그 핵심인 상태 머신 개념을 이해한다.

## 왜 필요한가? (Why)

- LCEL Chain은 **한 방향 파이프라인**이라 반복(loop), 조건 분기, 중간 상태 유지, 사람 개입이 어렵다.
- 실무 워크플로우는 "검색 → 품질평가 → (부족하면) 재검색 → 생성"처럼 **루프와 분기**가 있다. 이건 그래프가 자연스럽다.
- LangGraph는 각 단계에서 **상태를 저장(checkpoint)**할 수 있어, 대화 메모리·중단/재개·사람 승인 같은 실무 기능을 그래프 위에서 구현한다.

## 핵심 개념 (What)

### 3대 구성요소
- **State**: 그래프 전체가 공유하는 데이터(dict). 보통 `TypedDict`로 정의.
- **Node**: `state를 받아 부분 업데이트(dict)를 반환하는 함수`. 실제 작업 단위.
- **Edge**: 노드 간 연결. 고정 엣지 또는 조건부 엣지(다음 노드를 런타임에 결정).

### 상태 업데이트와 Reducer
노드는 **전체 state를 덮어쓰지 않고, 바뀐 키만** 반환한다. 기본 동작은 "해당 키를 교체". 하지만 **리스트에 누적**하고 싶으면(예: 메시지 히스토리) `Annotated`로 **reducer**를 지정한다.

```python
from typing import Annotated
from typing_extensions import TypedDict
from operator import add

class State(TypedDict):
    question: str
    documents: Annotated[list, add]   # add reducer → 노드가 반환한 리스트를 "누적"
```

> 메시지 누적에는 LangGraph가 제공하는 `add_messages` reducer를 쓴다(다음 문서).

### 특수 노드 START / END
`START`는 진입점, `END`는 종료점. 최소한 `START → ... → END` 경로가 있어야 한다.

## 어떻게 사용하는가? (How)

### 최소 그래프: 단일 노드
```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

# 사내:
llm = ChatOpenAI(model="Kimi-K2.5", base_url="http://llm-gateway.internal/v1",
                 api_key="EMPTY", temperature=0)

class State(TypedDict):
    question: str
    answer: str

def answer_node(state: State) -> dict:
    resp = llm.invoke(state["question"])
    return {"answer": resp.content}     # 바뀐 키만 반환

builder = StateGraph(State)
builder.add_node("answer", answer_node)
builder.add_edge(START, "answer")
builder.add_edge("answer", END)

graph = builder.compile()
print(graph.invoke({"question": "LangGraph가 뭐야?"})["answer"])
```

### 다중 노드 순차 실행
```python
class State(TypedDict):
    text: str
    summary: str
    keywords: str

def summarize(state):  # 노드 1
    return {"summary": llm.invoke(f"요약:\n{state['text']}").content}

def extract(state):    # 노드 2 (summary를 입력으로)
    return {"keywords": llm.invoke(f"키워드 3개:\n{state['summary']}").content}

b = StateGraph(State)
b.add_node("summarize", summarize)
b.add_node("extract", extract)
b.add_edge(START, "summarize")
b.add_edge("summarize", "extract")
b.add_edge("extract", END)
graph = b.compile()

out = graph.invoke({"text": "긴 공정 리포트 ..."})
print(out["summary"], "\n---\n", out["keywords"])
```

### 그래프 시각화 (디버깅)
```python
# Mermaid 다이어그램 문자열 얻기
print(graph.get_graph().draw_mermaid())
# Jupyter: display(Image(graph.get_graph().draw_mermaid_png()))
```

### 스트리밍으로 각 노드 진행 관찰
```python
for step in graph.stream({"text": "..."}):
    print(step)     # {노드이름: 그 노드가 반환한 부분 상태}
```

## 정리
| 개념 | 한 줄 |
|------|-------|
| State | 그래프가 공유하는 데이터(dict) |
| Node | state → 부분 업데이트 함수 |
| Edge | 노드 연결 (고정/조건부) |
| Reducer | 상태 키를 병합하는 방법(교체 vs 누적) |
| compile() | 그래프를 실행 가능한 Runnable로 변환 |

## 관련 문서
- [06. 멀티스텝 대화 흐름](./06-multistep-conversation-flow.md) — 상태 누적 + 메모리
- [07. Condition/Branching/Tool workflow](./07-condition-branching-tool-workflow.md) — 조건부 엣지

## 참고 자료 (References)
- LangGraph 개요: https://docs.langchain.com/oss/python/langgraph/overview
- StateGraph, reducer, `add_messages`
