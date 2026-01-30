# LangGraph 기반 RAG

> LangGraph를 활용해 검색-판단-재검색-생성의 순환 구조를 가진 Corrective RAG 파이프라인을 구축한다.

---
tags: [langgraph, rag, corrective-rag, retrieval]
level: intermediate
last_updated: 2026-01-31
status: in-progress
---

## 왜 필요한가? (Why)

### 단순 RAG의 한계

기본 RAG 파이프라인(검색 → 생성)은 다음 문제를 가진다:

1. **검색 품질 불확실**: 관련 없는 문서가 검색될 수 있음
2. **단일 시도**: 검색 결과가 나쁘면 그대로 잘못된 답변 생성
3. **자기 검증 없음**: 생성된 답변이 질문에 부합하는지 확인하지 않음

### Graph 기반 RAG의 장점

LangGraph를 사용하면:

- **문서 관련성 평가(Grading)**: 검색된 문서가 질문에 관련 있는지 판단
- **자동 재검색**: 관련 문서가 없으면 쿼리를 재작성하여 다시 검색
- **답변 검증**: 생성된 답변이 문서에 근거하는지(hallucination 체크) 확인
- **폴백(Fallback)**: 모든 검색이 실패하면 웹 검색으로 전환

## 핵심 개념 (What)

### Corrective RAG 패턴

```
질문 입력
    ↓
  검색 (Retrieve)
    ↓
  문서 평가 (Grade Documents)
    ↓ (conditional)
    ├─ 관련 문서 있음 → 답변 생성 (Generate)
    │                        ↓
    │                   답변 검증 (Check Hallucination)
    │                        ↓ (conditional)
    │                        ├─ 근거 있음 → END
    │                        └─ 근거 없음 → 답변 재생성
    │
    └─ 관련 문서 없음 → 쿼리 재작성 (Rewrite) → 검색 (다시)
```

### 주요 노드 역할

| 노드 | 역할 |
|------|------|
| Retrieve | 벡터 스토어에서 관련 문서 검색 |
| Grade Documents | 각 문서의 관련성을 LLM으로 평가 |
| Generate | 관련 문서 기반으로 답변 생성 |
| Rewrite Query | 검색 결과가 부족할 때 질문 재작성 |
| Check Hallucination | 답변이 문서에 근거하는지 검증 |

## 어떻게 사용하는가? (How)

### 전체 구현: Corrective RAG

```python
from typing import TypedDict, Literal
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

# --- 1. State 정의 ---

class RAGState(TypedDict):
    question: str
    documents: list[Document]
    generation: str
    retry_count: int

# --- 2. 컴포넌트 초기화 ---

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings()

# 예시 문서로 벡터 스토어 구성
sample_docs = [
    Document(page_content="FastAPI는 Python 기반의 고성능 웹 프레임워크로, 자동 API 문서 생성을 지원한다."),
    Document(page_content="FastAPI의 의존성 주입 시스템은 Depends() 함수를 통해 구현된다."),
    Document(page_content="LangGraph는 LLM 워크플로우를 상태 기반 그래프로 구성하는 프레임워크다."),
]
vectorstore = FAISS.from_documents(sample_docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- 3. 노드 함수 정의 ---

def retrieve(state: RAGState) -> dict:
    """벡터 스토어에서 문서를 검색한다."""
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents}


def grade_documents(state: RAGState) -> dict:
    """검색된 문서의 관련성을 평가한다."""
    question = state["question"]
    documents = state["documents"]

    grading_prompt = ChatPromptTemplate.from_messages([
        ("system", "문서가 질문에 관련 있으면 'yes', 없으면 'no'만 답해줘."),
        ("human", "질문: {question}\n\n문서: {document}"),
    ])
    chain = grading_prompt | llm | StrOutputParser()

    relevant_docs = []
    for doc in documents:
        result = chain.invoke({"question": question, "document": doc.page_content})
        if "yes" in result.lower():
            relevant_docs.append(doc)

    return {"documents": relevant_docs}


def generate(state: RAGState) -> dict:
    """관련 문서를 기반으로 답변을 생성한다."""
    question = state["question"]
    documents = state["documents"]
    context = "\n\n".join(doc.page_content for doc in documents)

    gen_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "다음 컨텍스트를 기반으로 질문에 답변해줘. "
         "컨텍스트에 없는 내용은 답변하지 마.\n\n"
         "컨텍스트:\n{context}"),
        ("human", "{question}"),
    ])
    chain = gen_prompt | llm | StrOutputParser()
    generation = chain.invoke({"context": context, "question": question})

    return {"generation": generation}


def rewrite_query(state: RAGState) -> dict:
    """더 나은 검색을 위해 질문을 재작성한다."""
    question = state["question"]

    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "벡터 스토어 검색에 최적화된 형태로 질문을 재작성해줘. "
         "재작성된 질문만 출력해."),
        ("human", "{question}"),
    ])
    chain = rewrite_prompt | llm | StrOutputParser()
    new_question = chain.invoke({"question": question})

    retry_count = state.get("retry_count", 0)
    return {"question": new_question, "retry_count": retry_count + 1}


def check_hallucination(state: RAGState) -> dict:
    """답변이 문서에 근거하는지 확인한다."""
    # 검증 로직은 라우팅 함수에서 처리
    return state

# --- 4. 라우팅 함수 ---

def route_after_grading(state: RAGState) -> Literal["generate", "rewrite_query"]:
    """관련 문서가 있으면 생성, 없으면 재검색."""
    if state["documents"]:
        return "generate"
    # 재시도 횟수 제한
    if state.get("retry_count", 0) >= 2:
        return "generate"  # 문서 없이라도 생성 시도
    return "rewrite_query"


def route_after_hallucination_check(
    state: RAGState,
) -> Literal["generate", "__end__"]:
    """답변이 문서에 근거하는지 확인하고 라우팅한다."""
    documents = state["documents"]
    generation = state["generation"]

    if not documents:
        return "__end__"

    context = "\n".join(doc.page_content for doc in documents)
    result = llm.invoke(
        f"다음 답변이 컨텍스트에 근거하면 'yes', 아니면 'no'만 답해줘.\n\n"
        f"컨텍스트: {context}\n\n답변: {generation}"
    )
    if "yes" in result.content.lower():
        return "__end__"
    return "generate"

# --- 5. 그래프 구성 ---

workflow = StateGraph(RAGState)

# 노드 추가
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("check_hallucination", check_hallucination)

# 엣지 연결
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges("grade_documents", route_after_grading)
workflow.add_edge("rewrite_query", "retrieve")
workflow.add_edge("generate", "check_hallucination")
workflow.add_conditional_edges("check_hallucination", route_after_hallucination_check)

# 컴파일
app = workflow.compile()

# --- 6. 실행 ---

result = app.invoke({"question": "FastAPI의 의존성 주입이 뭔가요?", "retry_count": 0})
print(f"답변: {result['generation']}")
```

### 실행 흐름 시각화

```
START → retrieve → grade_documents
                        ↓ (conditional)
                        ├─ 문서 있음 → generate → check_hallucination
                        │                              ↓ (conditional)
                        │                              ├─ 근거 있음 → END
                        │                              └─ 근거 없음 → generate (재생성)
                        │
                        └─ 문서 없음 → rewrite_query → retrieve (재검색)
```

### 디버깅: 단계별 실행 확인

```python
# 각 노드의 실행 과정을 스트리밍으로 확인
for event in app.stream({"question": "LangGraph란?", "retry_count": 0}):
    for node_name, output in event.items():
        print(f"--- {node_name} ---")
        if "documents" in output:
            print(f"  문서 수: {len(output['documents'])}")
        if "generation" in output:
            print(f"  답변: {output['generation'][:100]}...")
        if "question" in output:
            print(f"  질문: {output['question']}")
```

## 실무 적용 포인트

- **문서 평가 기준 커스터마이징**: 도메인에 맞게 grading prompt를 조정
- **재시도 횟수 제한**: 무한 루프 방지를 위해 `retry_count` 관리 필수
- **벡터 스토어 교체**: FAISS 대신 Milvus, Chroma 등으로 교체 가능
- **웹 검색 폴백**: 재검색 실패 시 Tavily 등 웹 검색 API로 전환 가능

## 참고 자료 (References)

- [Corrective RAG 논문](https://arxiv.org/abs/2401.15884)
- [LangGraph RAG Tutorial](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag/)
- [LangGraph Adaptive RAG](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag_local/)

## 관련 문서

- [이전: LangGraph 기초](./langgraph-basics.md)
- [다음: LangGraph 고급](./langgraph-advanced.md)
- [LangGraph 시리즈 목차](./README.md)
