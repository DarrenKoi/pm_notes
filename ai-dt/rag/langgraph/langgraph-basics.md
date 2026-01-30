# LangGraph 기초

> LangGraph는 LLM 애플리케이션을 상태 기반 그래프(Stateful Graph)로 구성하는 프레임워크다.

---
tags: [langgraph, state-graph, workflow]
level: beginner
last_updated: 2026-01-31
status: in-progress
---

## 왜 필요한가? (Why)

### LangChain만으로는 부족한 경우

LangChain의 LCEL(LangChain Expression Language)은 **선형 체인(linear chain)** 에 적합하다. 하지만 실무에서는 더 복잡한 흐름이 필요하다:

- **조건 분기**: 사용자 질문 유형에 따라 다른 처리 경로
- **반복(Loop)**: 결과가 불충분하면 다시 검색/생성
- **병렬 처리**: 여러 작업을 동시에 수행 후 결과 합산
- **상태 관리**: 각 단계의 결과를 누적하며 다음 단계에 전달

LangGraph는 이런 **비선형 워크플로우**를 그래프 구조로 깔끔하게 표현한다.

### LangChain vs LangGraph 비교

| 특성 | LangChain (LCEL) | LangGraph |
|------|------------------|-----------|
| 흐름 구조 | 선형 체인 | 그래프 (분기/반복 가능) |
| 상태 관리 | 제한적 | 명시적 State 객체 |
| 조건 분기 | RunnableBranch | Conditional Edge |
| 반복/루프 | 어려움 | 자연스러운 사이클 지원 |
| 적합한 경우 | 단순 파이프라인 | 복잡한 에이전트/RAG |

## 핵심 개념 (What)

### 1. State (상태)

그래프 전체에서 공유되는 데이터 구조. `TypedDict`로 정의한다.

```python
from typing import TypedDict, Annotated
from operator import add

class GraphState(TypedDict):
    question: str                          # 사용자 질문
    documents: list[str]                   # 검색된 문서
    generation: str                        # 생성된 답변
    steps: Annotated[list[str], add]       # 실행 이력 (누적)
```

`Annotated[list[str], add]`는 **리듀서(reducer)** 로, 노드가 반환한 값을 기존 값에 덮어쓰지 않고 **누적(append)** 한다.

### 2. Node (노드)

그래프의 각 처리 단계. 일반 Python 함수로 정의한다. State를 입력받고, 업데이트할 State를 반환한다.

```python
def retrieve(state: GraphState) -> dict:
    """문서 검색 노드"""
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "steps": ["retrieve"]}
```

### 3. Edge (엣지)

노드 간의 연결. 실행 순서를 정의한다.

- **일반 Edge**: A → B (항상 B로 이동)
- **Conditional Edge**: A → B 또는 C (조건에 따라 분기)

### 4. StateGraph

위 요소들을 조합해 그래프를 구성하는 클래스.

```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(GraphState)
```

### 5. 특수 노드

- `START`: 그래프의 시작점
- `END`: 그래프의 종료점

## 어떻게 사용하는가? (How)

### 기본 예제: 질문 분류 → 응답 생성 그래프

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

# 1. State 정의
class State(TypedDict):
    question: str
    category: str
    answer: str

# 2. LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 3. Node 함수 정의
def classify(state: State) -> dict:
    """질문을 분류한다."""
    question = state["question"]
    response = llm.invoke(
        f"다음 질문을 'technical' 또는 'general'로 분류해줘. "
        f"분류 결과만 한 단어로 답해줘.\n\n질문: {question}"
    )
    return {"category": response.content.strip().lower()}

def answer_technical(state: State) -> dict:
    """기술 질문에 대한 상세 답변을 생성한다."""
    response = llm.invoke(
        f"기술 전문가로서 다음 질문에 상세히 답변해줘.\n\n질문: {state['question']}"
    )
    return {"answer": response.content}

def answer_general(state: State) -> dict:
    """일반 질문에 대한 간단한 답변을 생성한다."""
    response = llm.invoke(
        f"다음 질문에 친절하게 답변해줘.\n\n질문: {state['question']}"
    )
    return {"answer": response.content}

# 4. 라우팅 함수 (Conditional Edge에 사용)
def route_question(state: State) -> Literal["answer_technical", "answer_general"]:
    if state["category"] == "technical":
        return "answer_technical"
    return "answer_general"

# 5. 그래프 구성
graph = StateGraph(State)

# 노드 추가
graph.add_node("classify", classify)
graph.add_node("answer_technical", answer_technical)
graph.add_node("answer_general", answer_general)

# 엣지 연결
graph.add_edge(START, "classify")
graph.add_conditional_edges("classify", route_question)
graph.add_edge("answer_technical", END)
graph.add_edge("answer_general", END)

# 6. 컴파일 및 실행
app = graph.compile()

# 실행
result = app.invoke({"question": "FastAPI에서 의존성 주입은 어떻게 작동하나요?"})
print(f"분류: {result['category']}")
print(f"답변: {result['answer']}")
```

### 그래프 시각화

```python
# Jupyter Notebook에서 그래프 구조 확인
from IPython.display import Image, display

display(Image(app.get_graph().draw_mermaid_png()))
```

### 실행 흐름 요약

```
START
  ↓
classify (질문 분류)
  ↓ (conditional)
  ├─ "technical" → answer_technical → END
  └─ "general"  → answer_general  → END
```

## 핵심 정리

| 개념 | 역할 | Python 타입 |
|------|------|-------------|
| State | 그래프 공유 데이터 | `TypedDict` |
| Node | 처리 단계 | 함수 (`state → dict`) |
| Edge | 노드 간 연결 | `add_edge()` |
| Conditional Edge | 조건부 분기 | `add_conditional_edges()` |
| StateGraph | 그래프 빌더 | `StateGraph(State)` |

## 참고 자료 (References)

- [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph/)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)
- [LangGraph Conceptual Guide](https://langchain-ai.github.io/langgraph/concepts/)

## 관련 문서

- [LangGraph 시리즈 목차](./README.md)
- [다음: LangGraph RAG](./langgraph-rag.md)
