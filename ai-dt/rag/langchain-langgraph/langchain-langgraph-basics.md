# LangChain + LangGraph 기초 사용법

## 1) LangChain vs LangGraph

### LangChain이 잘하는 것

- PromptTemplate, OutputParser, Retriever, Tool 같은 **구성 요소(component)** 조합
- LLM 호출을 파이프라인 형태로 빠르게 작성
- 표준화된 인터페이스로 provider 교체가 쉬움

### LangGraph가 잘하는 것

- 노드/엣지 기반 **상태 중심(stateful) 제어 흐름**
- 분기, 반복, 재시도, human-in-the-loop 같은 복잡한 워크플로우
- 대화 상태/중간 결과를 그래프 state에 보관

### 실무에서의 조합

- LangChain: "무엇을 실행할지"(프롬프트, 검색, 툴)
- LangGraph: "어떤 순서/조건으로 실행할지"(라우팅, 루프, 종료 조건)

---

## 2) 최소 설치

```bash
pip install -U langchain langgraph langchain-openai langchain-community faiss-cpu
```

---

## 3) LangChain 최소 예제 (Chain)

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 친절한 기술 튜터다."),
    ("human", "{question}")
])

chain = prompt | llm
result = chain.invoke({"question": "RAG가 뭔지 3줄로 설명해줘"})
print(result.content)
```

핵심은 `prompt | llm`처럼 LCEL로 체인을 선언형으로 연결하는 점이다.

---

## 4) LangGraph 최소 예제 (Graph)

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

class MyState(TypedDict):
    question: str
    answer: str

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def answer_node(state: MyState) -> MyState:
    response = llm.invoke(f"질문에 간단히 답해줘: {state['question']}")
    return {**state, "answer": response.content}

builder = StateGraph(MyState)
builder.add_node("answer", answer_node)
builder.add_edge(START, "answer")
builder.add_edge("answer", END)

app = builder.compile()
print(app.invoke({"question": "LangGraph의 장점은?"})["answer"])
```

---

## 5) 어떤 기능까지 확장 가능한가?

1. **멀티 스텝 에이전트**
   - 계획(Plan) → 도구 실행(Act) → 검증(Check) 루프
2. **조건부 라우팅**
   - 질문 유형(정의형/분석형/코드형)별 다른 노드로 분기
3. **고급 RAG**
   - Query Rewrite, Multi-Query, Re-ranking, Self-RAG/CRAG
4. **Human-in-the-loop**
   - 특정 신뢰도 이하일 때 승인 요청 노드로 이동
5. **장기 메모리/세션 관리**
   - 사용자별 컨텍스트 저장소 연결
6. **외부 시스템 통합**
   - 검색 API, 사내 DB, 티켓 시스템, MCP 서버
7. **평가/관측성**
   - 실행 추적(trace), 노드별 latency/token/cost 모니터링

---

## 6) 설계 체크리스트

- 상태(state)에 무엇을 저장할지 먼저 정의했는가?
- 종료 조건(END)과 최대 반복 횟수를 정의했는가?
- 실패 시 fallback 경로(재시도/간단 답변)를 마련했는가?
- Tool 결과 검증 및 권한 범위를 제한했는가?
- RAG의 chunk/index/retriever 하이퍼파라미터를 측정 기반으로 조정하는가?

이 체크리스트를 먼저 고정하면, 이후 RAG/Tool Calling 확장이 훨씬 쉬워진다.
