---
tags: [opensearch, rag, langchain, langgraph, llm, vector-store]
level: advanced
last_updated: 2026-02-07
status: in-progress
---

# OpenSearch RAG 파이프라인 연동 (RAG Integration)

> LangChain 및 LangGraph와 OpenSearch를 연동하여 실제 RAG(Retrieval-Augmented Generation) 시스템을 구축하는 방법

## 왜 필요한가? (Why)

OpenSearch는 단순한 검색 엔진을 넘어, RAG 시스템의 **지식 저장소(Knowledge Base)** 역할을 수행한다. LLM(Large Language Model)이 최신 정보나 사내 데이터(100GB+)를 참조하여 답변하려면, OpenSearch와 애플리케이션 프레임워크(LangChain 등)의 매끄러운 통합이 필수적이다.

---

## 핵심 개념 (What)

### 1. Vector Store (벡터 저장소)
LangChain 등에서 OpenSearch를 추상화하는 개념. `add_documents`, `similarity_search` 같은 표준 인터페이스를 제공한다.

### 2. Retriever (검색기)
단순한 검색을 넘어, LLM 파이프라인의 한 단계로 동작하는 인터페이스.
- **ParentDocumentRetriever**: 작은 청크로 검색하고, 큰 원본 문서를 반환.
- **SelfQueryRetriever**: 자연어 쿼리를 OpenSearch 필터(메타데이터) 쿼리로 자동 변환.
- **EnsembleRetriever**: BM25(키워드) + k-NN(벡터) 결과를 결합 (Hybrid Search).

---

## 어떻게 사용하는가? (How)

### 1. LangChain 연동 (기본)

`langchain-opensearch` 패키지를 사용한다.

```bash
pip install langchain-opensearch langchain-openai
```

```python
from langchain_openai import OpenAIEmbeddings
from langchain_opensearch import OpenSearchVectorSearch

# 1. 임베딩 모델 준비
embeddings = OpenAIEmbeddings()

# 2. OpenSearch VectorStore 초기화
docsearch = OpenSearchVectorSearch(
    index_name="rag-index",
    embedding_function=embeddings,
    opensearch_url="http://localhost:9200",
    http_auth=("admin", "admin"),
    use_ssl=False,
    verify_certs=False,
    engine="nmslib",
    space_type="cosinesimil"
)

# 3. 문서 추가 (이미 인덱스가 있다면 생략 가능)
# docsearch.add_documents(documents)

# 4. 유사도 검색
query = "100GB 데이터 처리 방법"
docs = docsearch.similarity_search(query, k=3)

print(docs[0].page_content)
```

### 2. LangChain Retriever 활용

검색기를 체인(Chain)의 일부로 사용한다.

```python
# 기본 검색기
retriever = docsearch.as_retriever(
    search_type="similarity", # 또는 "mmr", "similarity_score_threshold"
    search_kwargs={"k": 5}
)

# 하이브리드 검색기 (ScriptScore 등으로 구현되거나 Ensemble 사용)
# OpenSearchVectorSearch는 기본적으로 vector search를 수행하므로,
# 하이브리드를 위해서는 search_type="hybrid" (일부 버전 지원) 또는 커스텀 구현 필요
```

### 3. LangGraph 통합 (Agentic RAG)

LangGraph의 상태(State) 기반 워크플로우에 OpenSearch 검색 도구(Tool)를 통합한다.

```python
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool

# 1. 검색 도구 정의
@tool
def retrieve_documents(query: str):
    """OpenSearch에서 관련 문서를 검색합니다."""
    docs = docsearch.similarity_search(query, k=3)
    return "

".join([d.page_content for d in docs])

# 2. 상태 정의
class AgentState(TypedDict):
    messages: List[str]
    context: str

# 3. 노드 함수 정의
def search_node(state: AgentState):
    last_message = state["messages"][-1]
    # 도구 실행
    context = retrieve_documents.invoke(last_message)
    return {"context": context}

def generate_node(state: AgentState):
    context = state["context"]
    query = state["messages"][-1]
    # LLM 호출 (Context 포함)
    response = f"검색된 내용({context[:50]}...)을 바탕으로 답변: ..." 
    return {"messages": [response]}

# 4. 그래프 구성
workflow = StateGraph(AgentState)
workflow.add_node("search", search_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("search")
workflow.add_edge("search", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()
```

### 4. 고급 패턴: Hybrid Search Retriever

LangChain의 `EnsembleRetriever`를 사용하여 OpenSearch의 벡터 검색과 키워드 검색을 결합한다.

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# 1. BM25 검색기 (로컬 메모리 기반 예시, 실제론 OpenSearch의 BM25 기능 사용 권장)
# 주의: 대용량(100GB)에서는 로컬 BM25Retriever 대신 OpenSearch의 query DSL을 사용하는 CustomRetriever를 만들어야 함.
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 5

# 2. 벡터 검색기
vector_retriever = docsearch.as_retriever(search_kwargs={"k": 5})

# 3. 앙상블 (Hybrid)
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7] # 키워드 30%, 벡터 70%
)

docs = ensemble_retriever.invoke("OpenSearch 튜닝")
```

> **100GB 환경 주의사항**: 
> LangChain의 `BM25Retriever`는 문서를 메모리에 로드하므로 100GB 데이터에는 적합하지 않다.
> OpenSearch 자체의 하이브리드 검색 기능(Search Pipeline 등)을 활용하거나, 
> Python Client로 직접 구현한 `CustomRetriever`를 만들어 LangChain에 주입해야 한다.

---

## 실전 팁

1.  **메타데이터 필터링**: `search_kwargs={"filter": {"term": {"category": "news"}}}`와 같이 필터를 적극 활용하여 검색 범위를 좁힌다.
2.  **MMR (Maximal Marginal Relevance)**: `search_type="mmr"`을 사용하여 검색 결과의 다양성을 확보한다 (중복 내용 배제).
3.  **Custom Retriever**: 100GB 규모에서는 LangChain의 고수준 추상화보다, `opensearch-py`를 최적화하여 사용하는 커스텀 클래스가 성능상 유리할 수 있다.

---

## 참고 자료 (References)

- [LangChain OpenSearch Integration](https://python.langchain.com/docs/integrations/vectorstores/opensearch/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

## 관련 문서

- [OpenSearch 하이브리드 검색](./hybrid-search.md)
- [Python 클라이언트 활용](./python-client.md)
