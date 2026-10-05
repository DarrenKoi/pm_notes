---
tags: [opensearch, rag, langchain, langgraph, llm, vector-store]
level: advanced
last_updated: 2026-02-07
reviewed_on: 2026-10-04
review_status: partial
document_type: learning_note
aliases: [OpenSearch 검색 후보와 LangGraph 생성 흐름]
category_major: "AI·DT"
category_middle: "RAG"
category_minor: "OpenSearch 검색"
note_kind: "학습"
classified_on: "2026-10-05"
---

# OpenSearch RAG 파이프라인 연동 (RAG Integration)

> OpenSearch 검색 결과를 Document·Retriever·StateGraph의 입력으로 연결하고 근거와 생성 결과를 분리하는 방법

> [!info] 확인 조건 — 2026-10-04
> 로컬 검증 판본: Python3.14.2, langchain-community0.4.2/core1.6.6/classic1.0.8, LangGraph1.2.12, opensearch-py3.2.0, rank-bm250.2.2. community는 공식 sunset/archived 상태다. 이 문서는 해당 판본의 기존 인터페이스를 학습하며 새 운영 도입을 권고하지 않는다. 별도 패키지 선택/마이그레이션은 미확인·Claude 협의 대기다. OpenSearch3.x+Faiss cosine/k-NN 제공·schema/embedding 계약이 맞는 독립 실습을 가정한다. 실제 서버/모델/100GB 부하를 실행하지 않았다.

## 왜 필요한가? (Why)

OpenSearch는 권한·schema·데이터 갱신 계약을 갖춘 검색 근거 저장소 후보이다. RAG에는 검색 가능한 근거와 생성 입력 연결이 필요하지만 OpenSearch/LangChain이 모두 필수인 것은 아니다. 원문의100GB+는 용량 가정이며 실제 보안/품질/성능 검증 결과가 아니다. 검색 결과의 출처·문서 id를 남기고 근거 부재 시 답변을 보류한다.

---

## 핵심 개념 (What)

### 1. Vector Store (벡터 저장소)
LangChain 등에서 OpenSearch를 추상화하는 개념. `add_documents`, `similarity_search` 같은 표준 인터페이스를 제공한다.

### 2. Retriever (검색기)
단순한 검색을 넘어, LLM 파이프라인의 한 단계로 동작하는 인터페이스.
- **ParentDocumentRetriever**: 작은 청크 검색 후 id로 docstore의 부모를 복원한다. 별도 부모 저장소/매핑·수명 계약이 필요하며 아래 예제는 구현하지 않는다.
- **SelfQueryRetriever**: 모델이 만든 구조화 조건을 translator가 검색 backend의 필터로 바꾼다. 허용 field/type·사용자 권한 조건은 별도 강제해야 하며 아래는 구현하지 않는다.
- **EnsembleRetriever**: 아래는 BM25/벡터 후보의 **가중 RRF 순위**를 결합한다.0.3/0.7은 raw BM25/벡터 점수 비율이 아니다. 동일 corpus의 stable metadata id로 identity를 맞춘다.

---

## 어떻게 사용하는가? (How)

### 1. LangChain 연동 (기본)

원래 OpenAIEmbeddings 모델 준비 선택지는 caller가 제공하는 Embeddings로 일반화해 보존한다. 실제 OpenAI 공급자/모델/추가 패키지 판본은 이번에 검증하지 않았다. 외부 호출을 자동 수행하지 않는다.

공식 provider 문서의 import는 `langchain_community.vectorstores.OpenSearchVectorSearch`다. 원래 langchain-opensearch는 별도 PyPI0.0.2 패키지로 기존 core<0.4 의존성을 가지며 같은 export/현재 LangChain 호환을 단정할 수 없다. 아래는 검증한 legacy 판본만 재현하는 설치 예시다. 실행 환경/잠금 의존성을 별도 관리한다.

```bash
python -m pip install "langchain-community==0.4.2" "langchain-core==1.6.6" \
  "langchain-classic==1.0.8" "langgraph==1.2.12" "opensearch-py==3.2.0" "rank-bm25==0.2.2"
```

```python
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import OpenSearchVectorSearch


def build_docsearch(embeddings: Embeddings, endpoint: str, index_name: str,
                    user: str, password: str, ca_certs: str) -> OpenSearchVectorSearch:
    # caller가 검증한 HTTPS/자격/CA를 전달한다. 모델/서버 호출을 여기서 하지 않는다.
    if not endpoint.startswith("https://") or not all([index_name, user, password, ca_certs]):
        raise ValueError("explicit_tls_connection_values_required")
    return OpenSearchVectorSearch(opensearch_url=endpoint, index_name=index_name,
        embedding_function=embeddings, http_auth=(user, password), use_ssl=True,
        verify_certs=True, ssl_assert_hostname=True, ca_certs=ca_certs,
        engine="faiss", space_type="cosinesimil")


def demo_documents() -> list[Document]:
    # 고유 학습 fixture. 실제 원문/모델 품질/운영 데이터가 아니다.
    return [Document(page_content="100GB 데이터는 샤드와 복구 시간을 측정한다.",
                     metadata={"doc_id": "capacity", "category": "news", "source": "demo"}),
            Document(page_content="OpenSearch 튜닝은 heap과 native 메모리를 구분한다.",
                     metadata={"doc_id": "memory", "category": "news", "source": "demo"}),
            Document(page_content="RAG 답변은 검색 근거와 출처를 확인한다.",
                     metadata={"doc_id": "evidence", "category": "guide", "source": "demo"})]


def ingest_fresh_demo(docsearch: OpenSearchVectorSearch, documents: list[Document]) -> list[str]:
    # 기존 index를 자동 재사용/삭제하지 않는다. 전처리/모델 계약은 caller가 검사한다.
    if not documents:
        raise ValueError("empty_documents")
    ids = [d.metadata["doc_id"] for d in documents]
    if any(not isinstance(i, str) or not i.strip() for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("unique_nonempty_document_ids_required")
    if docsearch.client.indices.exists(index=docsearch.index_name):
        raise ValueError("fresh_demo_index_required")
    return docsearch.add_documents(documents, ids=ids, engine="faiss", space_type="cosinesimil")

# 명시 사용: documents=demo_documents(); docsearch=build_docsearch(verified_embeddings,...)
# 최초 독립 실습에서만 ingest_fresh_demo(docsearch,documents); query="100GB 데이터 처리 방법"
# GET/생성/적재는 transaction이 아니므로 동시 writer/관리자가 없는 독립 실습을 전제로 한다. 기존 index 존재만으로 model/field 계약이 맞지 않는다. 아래는 사전 검사/refresh가 된 상태를 가정한다.
# docs=docsearch.similarity_search(query,k=3); print(docs[0].page_content if docs else "근거 없음")
# 종료 시: docsearch.client.close(); async client는 사용하지 않으면 session 미생성.
# async를 사용했다면 await docsearch.async_client.close()도 caller가 수행한다.
```

### 2. LangChain Retriever 활용

Retriever는 질의→Document 목록 interface다. source/metadata는 답변 인용/권한 검사와 별도로 전달한다. 이 래퍼는 timeout/failed shard를 검사하지 않고 hits를 변환하는 경로가 있어 부분 응답을 완전 근거로 간주하지 않는다. 실패/부분 상태가 중요하면 [직접SDK](./python-client.md)의 응답 검사 계약을 적용한 retriever를 제공한다. 이 문서에서 그 서버 검증은 미완료다.

```python
def build_retriever(docsearch: OpenSearchVectorSearch):
    return docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Retriever search_type: similarity/mmr/similarity_score_threshold.
# wrapper 내부 검색 방식 approximate_search/script_scoring/painless_scripting/hybrid_search와 다르다.
# search_type="hybrid"를 as_retriever에 넣는 것은 검증 판본에서 허용되지 않는다.
# native hybrid_search는 준비된 search_pipeline/query_text와 필드 조건을 별도 확인한다.
```

### 3. LangGraph 통합 (Agentic RAG)

아래는 search→generate의 고정2단계다. 자율 도구 선택/재검색을 구현하지 않았으므로 제목의 Agentic RAG는 확장 맥락이다. 원래 broken join 문자열을 수정하고 messages overwrite 대신 query/context/answer를 분리했다. generate callback은 질의와 전체 context를 받으며 f-string 데모를 실제 LLM 호출로 부르지 않는다. 문서 목록을 남겨 source/id를 잃지 않는다. [그래프 학습 예제](../langgraph/langgraph-rag.md)와 달리 이 문서는 OpenSearch 경계 조립에 집중한다.

```python
from typing import Callable, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool

class AgentState(TypedDict):
    query: str
    documents: list[Document]
    context: str
    answer: str
    status: str


def build_rag_graph(retriever, generate: Callable[[str, str], str]):
    @tool
    def retrieve_documents(query: str) -> list[Document]:
        """명시 제공한 검색기로 근거 문서를 검색한다."""
        return retriever.invoke(query)

    def search_node(state: AgentState) -> dict:
        query = state["query"]
        if not isinstance(query, str) or not query.strip():
            raise ValueError("nonempty_query_required")
        docs = retrieve_documents.invoke({"query": query})
        context = "\n\n".join(d.page_content for d in docs)
        return {"documents": docs, "context": context}

    def generate_node(state: AgentState) -> dict:
        if not state["context"].strip():
            return {"answer": "검색 근거가 없어 답변을 보류합니다.", "status": "no_evidence"}
        answer = generate(state["query"], state["context"])
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("nonempty_generation_required")
        return {"answer": answer, "status": "generated"}

    workflow = StateGraph(AgentState)
    workflow.add_node("search", search_node)
    workflow.add_node("generate", generate_node)
    workflow.add_edge(START, "search")
    workflow.add_edge("search", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()

# caller가 실제 LLM adapter 또는 명시 fixture 함수를 generate로 전달한다.
# result=build_rag_graph(retriever,generate).invoke({"query":"100GB 데이터 처리 방법"})
# query/출처 metadata를 유지하며 answer는 별도 channel이다. checkpointer/재검색은 구현하지 않았다.
```

### 4. 고급 패턴: Hybrid Search Retriever

LangChain의 `EnsembleRetriever`를 사용하여 OpenSearch의 벡터 검색과 키워드 검색을 결합한다.

```python
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever


def build_ensemble(docsearch: OpenSearchVectorSearch, documents: list[Document]):
    if not documents:
        raise ValueError("empty_documents")
    ids = [d.metadata["doc_id"] for d in documents]
    if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("unique_document_ids_required")
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 5
    vector_retriever = build_retriever(docsearch)
    return EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever],
                             weights=[0.3, 0.7], c=60, id_key="doc_id")

# 동일 corpus/metadata id로 적재한 뒤:
# ensemble_retriever=build_ensemble(docsearch,documents)
# docs=ensemble_retriever.invoke("OpenSearch 튜닝")
# 각 branch의 같은 id는 동일 원문/권한으로 조회되어야 한다. 필터를 하나에만 적용하지 않는다.
```

> [!note] 작은 corpus 예제의 범위
> BM25Retriever는 토큰/색인을 프로세스 메모리에 준비한다. 기본 공백 분리는 Nori 분석기와 같은 한국어 분석이 아니다. 원문의100GB는 이 fixture로 검증하지 않았으며 전체 corpus를 무조건 메모리에 올리지 않는다. 서버 BM25/native hybrid 또는 직접SDK adapter는 schema·권한·응답 실패와 부하를 검증한 뒤 선택한다. 로컬 BM25와 서버 벡터에 같은 필터/corpus revision을 적용하지 않으면 다른 문서가 섞일 수 있다.

---

## 실전 팁

1. **메타데이터 필터**: wrapper 기본 metadata 필드는 `metadata`다. 검증 판본 Faiss 경로에서는 `search_kwargs={"k":5, "efficient_filter":{"term":{"metadata.category":"news"}}}`처럼 실제 매핑 경로를 확인한다. filter가 보안 경계를 자동 구현하는 것은 아니며 모든 branch에 동일 소유권 조건이 필요하다. boolean_filter/pre_filter와 실행 경로/판본을 섞지 않는다.
2. **MMR**: 후보 fetch_k와 lambda_mult로 유사성/다양성을 평가한다. 완전 중복 제거·품질 상승 보장은 아니다. 래퍼가 후보 벡터를 가져오는 필드와 response 원문을 확인한다.
3. **Custom Retriever**: 추상화 자체가100GB 성능을 결정하지 않는다. 동일 요청 body/후보/샤드·회수와 client concurrency를 고정해 측정한다. 임베딩 모델은 caller가 선택하고 한국어/전처리/차원/비용을 별도 확인한다.

---

## 참고 자료 (References)

확인일 **2026-10-04**. 고정 검증 판본과 rolling reference를 구분한다. 실서버 schema·ANN/MMR/filter·부분 응답/모델·100GB 성능은 미확인이다.

- [공식 OpenSearch provider](https://docs.langchain.com/oss/python/integrations/providers/opensearch)
- [community OpenSearch 일차 코드](https://raw.githubusercontent.com/langchain-ai/langchain-community/master/libs/community/langchain_community/vectorstores/opensearch_vector_search.py)
- [community sunset](https://github.com/langchain-ai/langchain-community/issues/674)
- [별도 langchain-opensearch 배포 metadata](https://pypi.org/pypi/langchain-opensearch/json)
- [EnsembleRetriever](https://reference.langchain.com/python/langchain-classic/retrievers/ensemble/EnsembleRetriever)
- [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)

## 관련 문서

- [OpenSearch 하이브리드 검색](./hybrid-search.md)
- [Python 클라이언트 활용](./python-client.md)
