---
tags: [langgraph, state-graph, workflow]
level: beginner
last_updated: 2026-01-31
reviewed_on: 2026-10-04
review_status: partial
document_type: learning
---

# LangGraph 기초

> [!info] 검토 — 2026-10-04
> 공식 근거·설치 판본·실행 검증은 [RAG 정리 기록](../organization-log.md)에 있다. 실제 모델 품질·사내 권한·운영 서비스는 미확인이다. 입력과 객체를 명시적으로 전달하는 학습 함수이며 자동 모델 호출은 하지 않는다.


> LangGraph는 LLM 애플리케이션을 상태 기반 그래프(Stateful Graph)로 구성하는 프레임워크다.


## 왜 필요한가? (Why)

### LangChain만으로는 부족한 경우

LangChain의 LCEL(LangChain Expression Language)은 구성 요소 조립을 지원하며 분기/병렬도 표현할 수 있다. 하지만 실무에서는 더 복잡한 흐름이 필요하다:

- **조건 분기**: 사용자 질문 유형에 따라 다른 처리 경로
- **반복(Loop)**: 결과가 불충분하면 다시 검색/생성
- **병렬 처리**: 여러 작업을 동시에 수행 후 결과 합산
- **상태 관리**: 각 단계의 결과를 누적하며 다음 단계에 전달

LangGraph는 이런 **비선형 워크플로우**를 그래프 구조로 깔끔하게 표현한다.

### LangChain vs LangGraph 비교

| 특성 | LangChain (LCEL) | LangGraph |
|------|------------------|-----------|
| 흐름 구조 | chain·branch·parallel 조합 | 상태 기반 그래프/사이클 |
| 상태 관리 | 제한적 | 명시적 State 객체 |
| 조건 분기 | RunnableBranch | Conditional Edge |
| 반복/루프 | 호출자가 루프/수명을 설계 | 그래프 내 사이클·체크포인트 지원 |
| 적합한 경우 | 단순 파이프라인 | 복잡한 에이전트/RAG |

## 핵심 개념 (What)

### 1. State (상태)

그래프 노드가 읽고 갱신하는 데이터 구조. 이 예제는 TypedDict를 사용하며 런타임 검증은 별도다. 필수 입력 question과 노드가 채울 필드를 구분한다.

```python
from typing import TypedDict, Annotated, Required
from operator import add

class GraphState(TypedDict, total=False):
    question: Required[str]                # 사용자 질문
    documents: list[str]                   # 검색된 문서
    generation: str                        # 생성된 답변
    steps: Annotated[list[str], add]       # 실행 이력 (누적)
```

`Annotated[list[str], add]`는 새 리스트를 기존 리스트와 연결한다. 노드는 새 단계만 반환해야 하며 전체 이력을 다시 반환하면 중복된다. 일반 key는 기본적으로 덮어쓴다. 병렬 갱신·messages 삭제 등은 key별 reducer 계약을 설계한다.

### 2. Node (노드)

그래프의 각 처리 단계. 일반 Python 함수로 정의한다. State를 입력받고, 업데이트할 State를 반환한다.

```python
def retrieve(state: GraphState) -> dict:
    """문서 검색 노드"""
    question = state["question"]
    documents = retriever.invoke(question)  # 실제 retriever의 반환은 Document 목록
    return {"documents": [d.page_content for d in documents], "steps": ["retrieve"]}
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
from typing import TypedDict, Required, Literal
from langgraph.graph import StateGraph, START, END

def response_text(response) -> str:
    if not isinstance(response.content, str) or not response.content.strip():
        raise ValueError("비어 있지 않은 텍스트 응답 필요")
    return response.content.strip()

class State(TypedDict, total=False):
    question: Required[str]
    category: Literal["technical", "general"]
    answer: str

def build_classifier(llm):
    def classify(state):
        question = state.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("질문 필요")
        category = response_text(llm.invoke(f"질문을 technical 또는 general 한 단어로 분류: {question}")).lower()
        if category not in {"technical", "general"}:
            raise ValueError("분류 미확인; general로 자동 변환하지 않음")
        return {"category": category}
    def answer_technical(state):
        return {"answer": response_text(llm.invoke(f"기술 전문가로서 상세히 답하세요: {state['question']}"))}
    def answer_general(state):
        return {"answer": response_text(llm.invoke(f"친절하게 간단히 답하세요: {state['question']}"))}
    graph = StateGraph(State)
    graph.add_node("classify", classify)
    graph.add_node("answer_technical", answer_technical)
    graph.add_node("answer_general", answer_general)
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", lambda s: s["category"],
                                {"technical": "answer_technical", "general": "answer_general"})
    graph.add_edge("answer_technical", END)
    graph.add_edge("answer_general", END)
    return graph.compile()

# app = build_classifier(승인된_llm_객체)
# result = app.invoke({"question": "FastAPI에서 의존성 주입은 어떻게 작동하나요?"})
```

### 그래프 시각화

```python
def graph_mermaid(app) -> str:
    # Mermaid 소스 텍스트만 생성하며 외부 이미지 API에 전송하지 않는다.
    return app.get_graph().draw_mermaid()

# print(graph_mermaid(app))  # 별도 renderer 사용은 실행 환경에서 확인
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

- [LangGraph 공식 문서](https://docs.langchain.com/oss/python/langgraph/overview)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)
- [LangGraph Conceptual Guide](https://docs.langchain.com/oss/python/langgraph/graph-api)

## 관련 문서

- [LangGraph 시리즈 목차](./README.md)
- [다음: LangGraph RAG](./langgraph-rag.md)
