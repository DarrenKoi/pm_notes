# LangGraph 고급 패턴

> Human-in-the-loop, Subgraph, Persistence, Streaming 등 LangGraph의 고급 기능을 다룬다.

---
tags: [langgraph, human-in-the-loop, persistence, streaming, subgraph]
level: advanced
last_updated: 2026-01-31
status: in-progress
---

## 왜 필요한가? (Why)

기본 그래프만으로는 프로덕션 환경의 요구사항을 충족하기 어렵다:

- 중요한 결정에서 **사람의 승인**이 필요한 경우
- 복잡한 워크플로우를 **재사용 가능한 단위**로 분리해야 하는 경우
- 긴 워크플로우의 **중간 상태를 저장/복원**해야 하는 경우
- 사용자에게 **실시간 진행 상황**을 보여줘야 하는 경우

## 핵심 개념 (What)

### 1. Human-in-the-loop (사람 개입)

특정 노드 실행 전에 사람의 승인을 받거나, 사람이 직접 State를 수정할 수 있는 패턴.

**사용 사례**:
- 민감한 API 호출 전 승인
- AI 생성 결과에 대한 사람의 검토
- 자동화 중 예외 상황 처리

### 2. Subgraph (하위 그래프)

그래프 안에 또 다른 그래프를 노드로 포함. 복잡한 워크플로우를 모듈화한다.

**사용 사례**:
- RAG 파이프라인을 하나의 서브그래프로 캡슐화
- 팀별로 독립적인 그래프를 개발 후 조합

### 3. Persistence (영속성) - Checkpointer

그래프 실행의 각 단계를 저장하여, 중단 후 이어서 실행하거나 이전 상태로 되돌릴 수 있다.

**사용 사례**:
- Human-in-the-loop에서 승인 대기 중 상태 유지
- 오류 발생 시 마지막 성공 지점부터 재실행
- 대화 히스토리 관리

### 4. Streaming (스트리밍)

그래프 실행 중 각 노드의 결과를 실시간으로 전달한다.

**스트리밍 모드**:
- `stream()`: 노드 단위 스트리밍
- `astream_events()`: 이벤트 단위 세밀한 스트리밍 (LLM 토큰 포함)

## 어떻게 사용하는가? (How)

### 1. Human-in-the-loop 구현

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class State(TypedDict):
    query: str
    plan: str
    result: str

def create_plan(state: State) -> dict:
    """실행 계획을 생성한다."""
    response = llm.invoke(f"다음 요청에 대한 실행 계획을 만들어줘: {state['query']}")
    return {"plan": response.content}

def execute_plan(state: State) -> dict:
    """승인된 계획을 실행한다."""
    response = llm.invoke(f"다음 계획을 실행한 결과를 작성해줘:\n{state['plan']}")
    return {"result": response.content}

# 그래프 구성
workflow = StateGraph(State)
workflow.add_node("create_plan", create_plan)
workflow.add_node("execute_plan", execute_plan)

workflow.add_edge(START, "create_plan")
workflow.add_edge("create_plan", "execute_plan")
workflow.add_edge("execute_plan", END)

# Checkpointer 연결 + interrupt_before로 승인 지점 설정
checkpointer = MemorySaver()
app = workflow.compile(
    checkpointer=checkpointer,
    interrupt_before=["execute_plan"],  # execute_plan 전에 중단
)

# --- 실행 ---

# 1단계: 계획 생성까지 실행 (execute_plan 전에 중단됨)
config = {"configurable": {"thread_id": "task-001"}}
result = app.invoke({"query": "프로젝트 README 작성"}, config)

print(f"생성된 계획:\n{result['plan']}")
print("\n→ 이 계획을 승인하시겠습니까?")

# 2단계: 사용자 승인 후 이어서 실행
# (State를 수정할 수도 있음)
app.update_state(config, {"plan": result["plan"]})  # 필요시 수정
final = app.invoke(None, config)  # None으로 이어서 실행

print(f"\n실행 결과:\n{final['result']}")
```

### 2. Subgraph 구현

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# --- 서브그래프: 문서 처리 ---

class DocState(TypedDict):
    text: str
    summary: str

def summarize(state: DocState) -> dict:
    return {"summary": f"[요약] {state['text'][:50]}..."}

doc_workflow = StateGraph(DocState)
doc_workflow.add_node("summarize", summarize)
doc_workflow.add_edge(START, "summarize")
doc_workflow.add_edge("summarize", END)
doc_subgraph = doc_workflow.compile()

# --- 메인 그래프 ---

class MainState(TypedDict):
    query: str
    text: str       # 서브그래프와 공유하는 키
    summary: str    # 서브그래프의 출력을 받는 키
    answer: str

def search(state: MainState) -> dict:
    return {"text": f"{state['query']}에 대한 검색 결과 문서 내용..."}

def answer(state: MainState) -> dict:
    return {"answer": f"요약 기반 답변: {state['summary']}"}

main_workflow = StateGraph(MainState)
main_workflow.add_node("search", search)
main_workflow.add_node("process_doc", doc_subgraph)  # 서브그래프를 노드로 추가
main_workflow.add_node("answer", answer)

main_workflow.add_edge(START, "search")
main_workflow.add_edge("search", "process_doc")
main_workflow.add_edge("process_doc", "answer")
main_workflow.add_edge("answer", END)

app = main_workflow.compile()

result = app.invoke({"query": "LangGraph 사용법"})
print(result["answer"])
```

### 3. Persistence (Checkpointer) 활용

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

# 메모리 기반 (개발/테스트용)
memory_checkpointer = MemorySaver()

# SQLite 기반 (프로덕션용)
sqlite_checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

# 그래프에 적용
app = workflow.compile(checkpointer=sqlite_checkpointer)

# thread_id로 대화/세션 구분
config = {"configurable": {"thread_id": "user-123-session-1"}}

# 실행 - 각 단계의 상태가 자동 저장됨
result = app.invoke({"question": "LangGraph란?"}, config)

# 상태 히스토리 조회
for state in app.get_state_history(config):
    print(f"Step: {state.metadata.get('step', '?')}")
    print(f"State: {state.values}")
    print("---")

# 특정 체크포인트로 되돌리기
states = list(app.get_state_history(config))
if len(states) > 1:
    previous = states[1]  # 이전 상태
    app.update_state(config, previous.values)
```

### 4. Streaming 구현

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

class State(TypedDict):
    question: str
    answer: str

def generate(state: State) -> dict:
    response = llm.invoke(state["question"])
    return {"answer": response.content}

workflow = StateGraph(State)
workflow.add_node("generate", generate)
workflow.add_edge(START, "generate")
workflow.add_edge("generate", END)
app = workflow.compile()

# 방법 1: 노드 단위 스트리밍
print("=== 노드 단위 스트리밍 ===")
for event in app.stream({"question": "Python의 장점 3가지"}):
    for node_name, output in event.items():
        print(f"[{node_name}] {output}")

# 방법 2: 이벤트 단위 스트리밍 (LLM 토큰 포함)
import asyncio

async def stream_tokens():
    print("\n=== 토큰 단위 스트리밍 ===")
    async for event in app.astream_events(
        {"question": "Python의 장점 3가지"},
        version="v2",
    ):
        if event["event"] == "on_chat_model_stream":
            token = event["data"]["chunk"].content
            if token:
                print(token, end="", flush=True)
    print()

asyncio.run(stream_tokens())
```

## 패턴 조합 예시

실무에서는 위 패턴을 조합하여 사용한다:

```
Persistence + Human-in-the-loop:
  → 승인 대기 중 상태를 DB에 저장, 나중에 이어서 실행

Subgraph + Streaming:
  → 서브그래프 내부의 LLM 호출도 토큰 단위로 스트리밍

Persistence + Streaming:
  → 각 단계를 저장하면서 실시간 진행 상황 표시
```

## 핵심 정리

| 패턴 | 핵심 API | 용도 |
|------|----------|------|
| Human-in-the-loop | `interrupt_before`, `update_state` | 사람 승인/수정 |
| Subgraph | 컴파일된 그래프를 `add_node`에 전달 | 워크플로우 모듈화 |
| Persistence | `MemorySaver`, `SqliteSaver` | 상태 저장/복원 |
| Streaming | `stream()`, `astream_events()` | 실시간 출력 |

## 참고 자료 (References)

- [LangGraph Human-in-the-loop](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/)
- [LangGraph Persistence](https://langchain-ai.github.io/langgraph/how-tos/persistence/)
- [LangGraph Streaming](https://langchain-ai.github.io/langgraph/how-tos/streaming-tokens/)
- [LangGraph Subgraphs](https://langchain-ai.github.io/langgraph/how-tos/subgraph/)

## 관련 문서

- [이전: LangGraph RAG](./langgraph-rag.md)
- [LangGraph 시리즈 목차](./README.md)
