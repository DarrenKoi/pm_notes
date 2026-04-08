# LangChain + LangGraph 실전: RAG 연결과 Tool Calling

## 1) 전체 아키텍처

```text
[User Query]
   ↓
[Query Router] ──(일반 질의)──> [Direct Answer]
   │
   ├─(지식 필요)─────────────> [RAG Retrieve] -> [RAG Generate]
   │
   └─(외부 작업 필요)─────────> [Tool Agent Loop]
                                  ├─ call tool
                                  ├─ observe result
                                  └─ finish or repeat
```

핵심 아이디어는 "한 가지 체인"으로 모든 문제를 풀지 않고,
질의 성격에 따라 그래프 라우팅을 통해 경로를 바꾸는 것이다.

---

## 2) RAG 연결: 단계별 구현

## 2-1. 문서 적재 및 청킹

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("docs/product_manual.txt", encoding="utf-8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120
)
chunks = splitter.split_documents(docs)
```

- `chunk_size`는 검색 정밀도/문맥 보존의 균형
- `chunk_overlap`은 문맥 단절 방지

## 2-2. 임베딩 + 벡터 저장소

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

emb = OpenAIEmbeddings(model="text-embedding-3-small")
vs = FAISS.from_documents(chunks, emb)
retriever = vs.as_retriever(search_kwargs={"k": 4})
```

## 2-3. 생성 체인

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "주어진 컨텍스트만 근거로 답하고, 모르면 모른다고 말해."),
    ("human", "질문: {question}\n\n컨텍스트:\n{context}")
])
```

## 2-4. LangGraph 노드로 결합

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    question: str
    route: Literal["direct", "rag", "tool"]
    context: str
    answer: str


def route_node(state: AgentState):
    q = state["question"].lower()
    if "최신" in q or "문서" in q:
        route = "rag"
    elif "예약" in q or "조회" in q:
        route = "tool"
    else:
        route = "direct"
    return {**state, "route": route}


def retrieve_node(state: AgentState):
    docs = retriever.invoke(state["question"])
    context = "\n\n".join(d.page_content for d in docs)
    return {**state, "context": context}


def rag_answer_node(state: AgentState):
    msg = rag_prompt.invoke({"question": state["question"], "context": state["context"]})
    out = llm.invoke(msg)
    return {**state, "answer": out.content}


def direct_answer_node(state: AgentState):
    out = llm.invoke(state["question"])
    return {**state, "answer": out.content}


def route_fn(state: AgentState):
    return state["route"]

builder = StateGraph(AgentState)
builder.add_node("route", route_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("rag_answer", rag_answer_node)
builder.add_node("direct_answer", direct_answer_node)

builder.add_edge(START, "route")
builder.add_conditional_edges("route", route_fn, {
    "rag": "retrieve",
    "direct": "direct_answer",
})
builder.add_edge("retrieve", "rag_answer")
builder.add_edge("rag_answer", END)
builder.add_edge("direct_answer", END)

app = builder.compile()
```

---

## 3) Tool Calling 연결

아래 예시는 "일정 조회" 툴을 LLM이 필요할 때 호출하도록 구성한 패턴이다.

```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

@tool
def get_calendar_events(date: str) -> str:
    """해당 날짜의 일정을 조회한다."""
    # 실제 서비스에서는 DB/API 조회
    return f"{date}: 14:00 아키텍처 리뷰, 16:30 고객 미팅"

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools([get_calendar_events])
```

### Tool 루프 노드 설계 포인트

1. LLM 응답에서 tool call이 있으면 tool 실행 노드로 이동
2. tool 결과를 메시지 히스토리에 추가
3. 다시 LLM 노드로 돌아가 최종 답변 생성
4. tool call이 없으면 종료

이 구조는 ReAct 패턴(Thought/Action/Observation)을 그래프 상태로 명시화한 형태다.

---

## 4) 실무 확장 포인트

## 4-1. RAG 고도화

- Hybrid Search(BM25 + Vector)
- Reranker(교차 인코더) 도입
- Citation 강제 포맷(답변마다 출처 첨부)
- Query Transformation(재작성/확장)
- Retrieval 실패 감지 후 fallback(웹 검색 or clarifying question)

## 4-2. Tool 안정성

- 툴별 timeout/retry/circuit breaker
- 입력 스키마 검증(pydantic)
- 권한 기반 툴 허용 목록(예: 관리자 전용 툴)
- 부작용 툴(삭제/전송)은 human approval 필수

## 4-3. 운영/관측

- 노드별 latency/token/cost 로그
- 세션별 상태 스냅샷 저장
- 오답 사례셋을 기반으로 회귀 평가 자동화

---

## 5) 자주 하는 실수

1. **모든 질의를 RAG로 처리**
   - 상식 질문까지 검색하면 비용/지연 증가
2. **chunk 과대/과소 설정**
   - 너무 작으면 문맥 손실, 너무 크면 검색 정확도 저하
3. **Tool 권한 미제한**
   - 운영 환경에서 위험한 액션이 자동 실행될 수 있음
4. **종료 조건 없는 agent loop**
   - 무한 반복으로 비용 폭증 가능

---

## 6) 추천 학습 실습 과제

1. 문서 20개로 FAQ RAG 챗봇 만들기
2. "주문 조회" API를 tool로 연결하기
3. 라우터 정확도 측정(Direct/RAG/Tool 분류 정확도)
4. 실패 케이스 10개를 모아 그래프 fallback 경로 개선하기

이 과제를 끝내면 단순 챗봇 수준을 넘어,
"업무 자동화 가능한 에이전트"를 설계/운영할 수 있게 된다.
