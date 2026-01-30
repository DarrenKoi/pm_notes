---
tags: [milvus, rag, langchain, langgraph, vector-store]
level: intermediate
last_updated: 2026-01-31
status: in-progress
---

# Milvus RAG 연동 (Milvus RAG Integration)

> LangChain과 LangGraph를 활용하여 Milvus 기반 RAG 파이프라인을 구축하는 방법을 정리한다.

## 왜 필요한가? (Why)

### In-Memory Store의 한계

프로토타입 단계에서는 Chroma나 FAISS 같은 인메모리 벡터 스토어로 충분하지만, 프로덕션 환경에서는 다음과 같은 문제가 발생한다:

- **데이터 영속성 부재**: 프로세스 종료 시 데이터 소실
- **확장성 제한**: 단일 노드 메모리에 종속
- **동시 접근 불가**: 여러 서비스에서 동시에 접근할 수 없음
- **운영 기능 부재**: 모니터링, 백업, 인덱스 튜닝 등 불가

### Milvus의 프로덕션 장점

- **영속 저장**: 디스크 기반 스토리지로 데이터 안전
- **수평 확장**: 분산 아키텍처로 수십억 벡터 처리 가능
- **멀티테넌시**: Partition 기반으로 여러 서비스/사용자 분리
- **하이브리드 검색**: Dense + Sparse 벡터 결합 검색
- **LangChain 공식 지원**: `langchain-milvus` 패키지로 간편 통합

---

## 핵심 개념 (What)

### LangChain Milvus 래퍼 구조

```
Document Loader → Text Splitter → Embedding Model → Milvus VectorStore
                                                           ↓
                                          Query → Retriever → LLM → Answer
```

### 주요 컴포넌트

| 컴포넌트 | LangChain 클래스 | 역할 |
|----------|-----------------|------|
| VectorStore | `Milvus` | 벡터 저장/검색 인터페이스 |
| Retriever | `MilvusRetriever` | 검색 파라미터를 캡슐화한 검색기 |
| Embedding | `OpenAIEmbeddings` 등 | 텍스트 → 벡터 변환 |
| Document Loader | `PyPDFLoader` 등 | 원본 문서 로딩 |
| Text Splitter | `RecursiveCharacterTextSplitter` | 문서를 적절한 청크로 분할 |

### Retriever vs VectorStore 직접 사용

```python
# VectorStore 직접 사용 - 단순 유사도 검색
docs = vectorstore.similarity_search("query", k=5)

# Retriever 사용 - LangChain 체인/그래프에 통합 가능
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
docs = retriever.invoke("query")
```

> **Retriever**를 사용하면 LangChain의 체인(Chain)이나 LangGraph의 노드에 바로 연결할 수 있다.

---

## 어떻게 사용하는가? (How)

### 1. 패키지 설치

```bash
pip install langchain-milvus langchain-openai langchain-community
pip install pypdf  # PDF 로딩 시
```

### 2. Milvus VectorStore 초기화

```python
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings(model="text-embedding-3-small")

# 새 Collection으로 초기화
vectorstore = Milvus(
    embedding_function=embedding,
    collection_name="rag_documents",
    connection_args={"host": "localhost", "port": "19530"},
    index_params={
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 256},
    },
    search_params={
        "metric_type": "COSINE",
        "params": {"ef": 64},
    },
    auto_id=True,
)
```

### 3. 문서 로딩 및 분할

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# PDF 로딩
loader = PyPDFLoader("example.pdf")
documents = loader.load()

# 텍스트 분할
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_documents(documents)
print(f"총 {len(chunks)}개 청크 생성")
```

### 4. 임베딩 및 저장

```python
# 문서를 Milvus에 저장 (임베딩 자동 생성)
vectorstore.add_documents(chunks)
print("문서 저장 완료")

# 또는 from_documents로 한 번에 생성
vectorstore = Milvus.from_documents(
    documents=chunks,
    embedding=embedding,
    collection_name="rag_documents",
    connection_args={"host": "localhost", "port": "19530"},
)
```

### 5. Retriever 생성 및 검색

```python
# 기본 Retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# MMR (Maximal Marginal Relevance) - 다양성 확보
retriever_mmr = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7}
)

# 메타데이터 필터 포함
retriever_filtered = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,
        "expr": 'source == "manual.pdf"'  # Milvus 필터 표현식
    }
)

# 검색 실행
docs = retriever.invoke("Milvus의 인덱스 타입에 대해 알려줘")
for doc in docs:
    print(f"[{doc.metadata.get('source', 'N/A')}] {doc.page_content[:100]}...")
```

### 6. 기본 RAG 체인

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_template("""
다음 컨텍스트를 기반으로 질문에 답변하세요.
컨텍스트에 답이 없으면 "정보가 부족합니다"라고 답하세요.

컨텍스트:
{context}

질문: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("Milvus에서 HNSW 인덱스는 어떤 특징이 있나요?")
print(answer)
```

### 7. LangGraph 연동 (Corrective RAG 패턴)

LangGraph의 Corrective RAG 패턴에 Milvus를 적용하는 예시이다. 검색 결과의 관련성을 평가하고, 관련성이 낮으면 웹 검색으로 폴백한다.

```python
from typing import TypedDict, List
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

# State 정의
class RAGState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    search_type: str  # "vectordb" | "websearch"

# 노드 함수들
def retrieve(state: RAGState) -> RAGState:
    """Milvus에서 관련 문서를 검색한다."""
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def grade_documents(state: RAGState) -> RAGState:
    """검색된 문서의 관련성을 평가한다."""
    question = state["question"]
    documents = state["documents"]

    grading_prompt = ChatPromptTemplate.from_template("""
    문서가 질문과 관련이 있으면 "yes", 없으면 "no"로 답하세요.

    문서: {document}
    질문: {question}
    답변 (yes/no):
    """)

    grading_chain = grading_prompt | llm | StrOutputParser()

    filtered_docs = []
    for doc in documents:
        result = grading_chain.invoke({
            "document": doc.page_content,
            "question": question
        })
        if "yes" in result.lower():
            filtered_docs.append(doc)

    search_type = "vectordb" if filtered_docs else "websearch"
    return {
        "documents": filtered_docs,
        "question": question,
        "search_type": search_type
    }

def generate(state: RAGState) -> RAGState:
    """필터링된 문서를 기반으로 답변을 생성한다."""
    question = state["question"]
    documents = state["documents"]
    context = "\n\n".join(doc.page_content for doc in documents)

    gen_prompt = ChatPromptTemplate.from_template("""
    컨텍스트를 기반으로 질문에 답변하세요.

    컨텍스트: {context}
    질문: {question}
    """)

    gen_chain = gen_prompt | llm | StrOutputParser()
    generation = gen_chain.invoke({"context": context, "question": question})
    return {"generation": generation}

def web_search(state: RAGState) -> RAGState:
    """벡터 DB 검색 실패 시 웹 검색으로 폴백한다."""
    from langchain_community.tools import TavilySearchResults

    search = TavilySearchResults(max_results=3)
    results = search.invoke(state["question"])

    web_docs = [
        Document(page_content=r["content"], metadata={"source": r["url"]})
        for r in results
    ]
    return {"documents": web_docs, "question": state["question"]}

# 라우팅 함수
def route_after_grading(state: RAGState) -> str:
    return "generate" if state["search_type"] == "vectordb" else "web_search"

# 그래프 구성
workflow = StateGraph(RAGState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("web_search", web_search)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    route_after_grading,
    {"generate": "generate", "web_search": "web_search"}
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

# 컴파일 및 실행
app = workflow.compile()

result = app.invoke({"question": "Milvus에서 하이브리드 검색은 어떻게 하나요?"})
print(result["generation"])
```

**그래프 흐름:**

```
START → retrieve → grade_documents →┬→ generate → END
                                     └→ web_search → generate → END
```

---

## 참고 자료 (References)

- [langchain-milvus 공식 문서](https://python.langchain.com/docs/integrations/vectorstores/milvus/)
- [Milvus 공식 문서](https://milvus.io/docs)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph/)

## 관련 문서

- [Milvus 기초](./milvus-basics.md) - 아키텍처, 인덱스, 기본 사용법
- [Milvus 시리즈 목차](./README.md)
- [LangGraph 기초](../langgraph/langgraph-basics.md) - State, Node, Edge 개념
- [LangGraph RAG](../langgraph/langgraph-rag.md) - Corrective RAG 파이프라인 상세

---

*Last updated: 2026-01-31*
