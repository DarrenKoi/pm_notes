---
tags: [langgraph, human-in-the-loop, persistence, streaming, subgraph]
level: advanced
last_updated: 2026-01-31
reviewed_on: 2026-10-04
review_status: partial
document_type: learning
---

# LangGraph 고급 패턴

> [!info] 검토 — 2026-10-04
> 공식 근거·설치 판본·실행 검증은 [RAG 정리 기록](../organization-log.md)에 있다. 실제 모델 품질·사내 권한·운영 서비스는 미확인이다. 입력과 객체를 명시적으로 전달하는 학습 함수이며 자동 모델 호출은 하지 않는다.


> Human-in-the-loop, Subgraph, Persistence, Streaming 등 LangGraph의 고급 기능을 다룬다.


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
- `stream(..., stream_mode="updates")`: 노드의 변경 key; `values`는 전체 state, `messages`는 메시지 chunk·metadata
- `astream_events(version="v2")`: 이 예제에서 검증한 callback 이벤트 형식. rolling 문서의 stream_events v3와 형식/판본을 혼용하지 않음

## 어떻게 사용하는가? (How)

### 1. Human-in-the-loop 구현

```python
from typing import TypedDict, Required
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

class State(TypedDict, total=False):
    query: Required[str]
    plan: str
    result: str
    status: str

def build_approval_app(llm, executor, checkpointer=None):
    def create_plan(state):
        if not isinstance(state.get("query"), str) or not state["query"].strip():
            raise ValueError("요청 필요")
        plan = llm.invoke(f"실행 계획을 작성하세요: {state['query']}").content
        if not isinstance(plan, str) or not plan.strip():
            raise ValueError("텍스트 계획 필요")
        return {"plan": plan, "status": "awaiting_review"}
    def execute_plan(state):
        # interrupt 이후 재개 시 이 노드는 처음부터 재실행된다. 그 앞에 부작용을 두지 않는다.
        approved = interrupt({"plan": state["plan"], "request": state["query"]})
        if type(approved) is not bool:
            raise ValueError("승인 bool 필요; 문자열/누락은 승인 아님")
        if approved is False:
            return {"status": "rejected", "result": "실행하지 않음"}
        result = executor(state["plan"])
        if not isinstance(result, str):
            raise ValueError("executor 텍스트 결과 필요")
        return {"result": result, "status": "executed_callback"}
    workflow = StateGraph(State)
    workflow.add_node("create_plan", create_plan)
    workflow.add_node("execute_plan", execute_plan)
    workflow.add_edge(START, "create_plan")
    workflow.add_edge("create_plan", "execute_plan")
    workflow.add_edge("execute_plan", END)
    return workflow.compile(checkpointer=InMemorySaver() if checkpointer is None else checkpointer)

def start_review(app, query: str, config):
    return app.invoke({"query": query}, config)  # 결과 __interrupt__의 plan을 검토 UI에 표시

def resume_review(app, approved: bool, config):
    if type(approved) is not bool:
        raise ValueError("명시적 bool 승인 필요")
    return app.invoke(Command(resume=approved), config)  # 같은 thread, 실제 검토 응답만 전달

# config = {"configurable": {"thread_id": backend가_소유권_검증한_id}}
# app = build_approval_app(llm, 실제권한을_검사하는_executor)
# pending = start_review(app, "프로젝트 README 작성", config)
# 사람이 payload plan을 검토한 다음 별도 요청에서 resume_review(app, 응답, config)
# app.update_state(config, {"plan": 수정된_plan})도 가능하지만 변경된 plan을 다시 검토해야 한다.
```

위 예제는 외부에서 받은 승인 응답을 검사하는 흐름이며 인증된 승인 시스템 자체가 아니다. thread 소유권·검토 대상 plan 버전·권한·멱등 실행/감사 증빙은 backend가 구현해야 한다. LLM이 실행 결과 문장을 쓰는 것은 실제 작업 수행이 아니므로 executor callback과 분리했다. InMemorySaver는 프로세스 종료 시 사라진다. 기존 interrupt_before는 정적 breakpoint 디버깅에 쓸 수 있지만 멈춘 뒤 자동 invoke(None)만으로 사람 승인이 생기지는 않는다.

### 2. Subgraph 구현

```python
from typing import TypedDict, Required
from langgraph.graph import StateGraph, START, END

class DocState(TypedDict, total=False):
    text: Required[str]
    summary: str

class MainState(TypedDict, total=False):
    query: Required[str]
    text: str
    summary: str
    answer: str

def build_subgraph_app():
    def summarize(state):
        return {"summary": f"[앞50문자 축약] {state['text'][:50]}..."}
    doc_workflow = StateGraph(DocState)
    doc_workflow.add_node("summarize", summarize)
    doc_workflow.add_edge(START, "summarize")
    doc_workflow.add_edge("summarize", END)
    doc_subgraph = doc_workflow.compile()
    def search(state):
        return {"text": f"[가상 검색] {state['query']}에 대한 문서 내용..."}
    def answer(state):
        return {"answer": f"축약 기반 가상 답변: {state['summary']}"}
    main_workflow = StateGraph(MainState)
    main_workflow.add_node("search", search)
    main_workflow.add_node("process_doc", doc_subgraph)
    main_workflow.add_node("answer", answer)
    main_workflow.add_edge(START, "search")
    main_workflow.add_edge("search", "process_doc")
    main_workflow.add_edge("process_doc", "answer")
    main_workflow.add_edge("answer", END)
    return main_workflow.compile()

# result = build_subgraph_app().invoke({"query": "LangGraph 사용법"})
# 서로 다른 state key라면 wrapper에서 입출력을 변환해야 한다.
```

서브그래프 예제의 검색/요약/답변은 가상 문자열 처리다. 실제 retriever/요약 모델·품질 검증이 없다. 공유 text/summary key를 통해 값을 전달하는 구조만 학습한다.

### 3. Persistence (Checkpointer) 활용

```python
from typing import TypedDict, Required
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

class PersistState(TypedDict, total=False):
    question: Required[str]
    count: int
    answer: str

def persistent_workflow():
    # 앞절 query/plan 스키마와 독립적인 question 예제 builder다.
    def answer(state):
        return {"answer": f"[가상 답변] {state['question']}", "count": state.get("count", 0) + 1}
    builder = StateGraph(PersistState)
    builder.add_node("answer", answer)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    return builder

def run_persistence(db_path: str):
    config = {"configurable": {"thread_id": "local-demo"}}
    # from_conn_string은 context manager. 실제 saver/graph를 with 수명 안에서 사용한다.
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        app = persistent_workflow().compile(checkpointer=checkpointer)
        result = app.invoke({"question": "LangGraph란?"}, config)
        states = list(app.get_state_history(config))
        return result, [{"checkpoint_id": s.config["configurable"].get("checkpoint_id"),
                         "step": (s.metadata or {}).get("step"), "next": s.next} for s in states]

# result, history = run_persistence("실습용_별도경로/checkpoints.db")
# 재시작 후 같은 db_path/thread_id로 compile하면 state를 읽을 수 있다.
# 과거 snapshot.config(checkpoint_id 포함)로 invoke(None, snapshot.config)는 그 지점에서 replay/분기.
# update_state(config, previous.values)는 새 checkpoint를 만드는 갱신이며 부작용 롤백이 아니다.
```

SQLite는 로컬 실습용 지속 저장 예시다. 운영 채택은 동시 쓰기·접근 제어·백업/복구·저장 민감정보/암호화·규모 조건을 별도로 검증한다. state/time travel은 외부 시스템 변경을 취소하지 않는다.

### 4. Streaming 구현

```python
from typing import TypedDict, Required
from langgraph.graph import StateGraph, START, END

class StreamState(TypedDict, total=False):
    question: Required[str]
    answer: str

def build_stream_app(llm):
    def generate(state):
        return {"answer": llm.invoke(state["question"]).content}
    workflow = StateGraph(StreamState)
    workflow.add_node("generate", generate)
    workflow.add_edge(START, "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()

def node_updates(app, question: str):
    return list(app.stream({"question": question}, stream_mode="updates"))

async def stream_tokens(app, question: str) -> str:
    pieces = []
    async for event in app.astream_events({"question": question}, version="v2"):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"].content
            if isinstance(chunk, str):
                pieces.append(chunk)
            elif chunk:  # provider의 multimodal/content block은 별도 parser 필요
                raise ValueError("텍스트 외 streaming chunk 계약 확인 필요")
    if not pieces:
        raise ValueError("텍스트 stream 이벤트 미수집")
    return "".join(pieces)

# streaming을 지원하는 승인된 chat model을 전달한다. 모델별 chunk 단위는 tokenizer token과 다를 수 있다.
# script: asyncio.run(stream_tokens(app, "Python의 장점 3가지"))
# notebook의 이미 실행 중인 event loop: await stream_tokens(app, "Python의 장점 3가지")
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
| Human-in-the-loop | `interrupt`, `Command(resume=...)` | 검토 대기/명시 응답; 인증은 별도 |
| Subgraph | 컴파일된 그래프를 `add_node`에 전달 | 워크플로우 모듈화 |
| Persistence | `InMemorySaver`, `SqliteSaver` | 프로세스 상태/로컬 지속 저장·checkpoint 분기 |
| Streaming | `stream()`, `astream_events()` | 실시간 출력 |

## 참고 자료 (References)

- [LangGraph Human-in-the-loop](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [LangGraph Streaming](https://docs.langchain.com/oss/python/langgraph/streaming)
- [LangGraph Subgraphs](https://docs.langchain.com/oss/python/langgraph/use-subgraphs)

## 관련 문서

- [이전: LangGraph RAG](./langgraph-rag.md)
- [LangGraph 시리즈 목차](./README.md)
