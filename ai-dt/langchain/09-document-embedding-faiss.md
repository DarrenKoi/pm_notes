---
tags: [rag, embedding, faiss, vectorstore, bge-m3]
level: intermediate
last_updated: 2026-07-06
---

# 09. 문서 임베딩 및 FAISS 벡터 스토어 구축

> 텍스트를 벡터로 바꾸는 임베딩의 원리와, 그 벡터를 빠르게 검색하는 FAISS 인덱스(Flat/IVF/HNSW)를 이해하고 구축한다.

## 왜 필요한가? (Why)

- LLM은 방대한 사내 문서를 다 기억하지 못한다. 질문과 **의미적으로 유사한** 문서 조각을 찾아 붙여야(RAG) 정확한 답이 나온다.
- 키워드 검색은 "동의어/의역"을 못 잡는다. **임베딩**은 의미를 벡터 공간의 위치로 표현해, 표현이 달라도 의미가 가까우면 찾는다.
- FAISS는 로컬에서 가볍게 쓰는 벡터 검색 라이브러리다. DB(OpenSearch/Milvus) 없이도 PoC를 돌릴 수 있어 사내 초기 실험에 적합하다.

## 핵심 개념 (What)

### 임베딩(Embedding)
텍스트 → 고정 길이 실수 벡터(예: BGE-M3는 1024차원). 의미가 비슷한 문장은 벡터도 가깝다(코사인 유사도 ↑). 사내에서는 **BGE-M3**를 OpenAI 호환 API로 서빙한다.

### 청킹(Chunking)
문서는 통째로 임베딩하지 않고 **적당한 크기 조각**으로 나눈다. 너무 크면 노이즈가 섞이고, 너무 작으면 맥락이 끊긴다. (튜닝은 [10번](./10-retriever-tuning.md))

### FAISS 인덱스 종류
| 인덱스 | 특징 | 언제 |
|--------|------|------|
| **Flat** (IndexFlatL2/IP) | 전수 비교, 100% 정확, 느림 | 수천~수만 건 PoC |
| **IVF** (IVFFlat) | 클러스터로 나눠 일부만 탐색, 빠름 | 수십만~ |
| **HNSW** | 그래프 기반 근사 최근접, 매우 빠름·고정밀 | 대규모 저지연 |

> 유사도 메트릭: 정규화된 임베딩에는 **내적(IP)=코사인 유사도**. BGE-M3는 코사인 기준이 자연스럽다.

## 어떻게 사용하는가? (How)

### 1) 문서 로드 → 청킹 → 임베딩 → FAISS
```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 공개: embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# 사내:
embeddings = OpenAIEmbeddings(model="BGE-M3",
                              base_url="http://llm-gateway.internal/v1",
                              api_key="EMPTY")

raw = [Document(page_content="포토 공정은 웨이퍼에 회로 패턴을 전사한다 ...",
                metadata={"source": "photo.md"})]

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(raw)

vs = FAISS.from_documents(chunks, embeddings)   # 임베딩 계산 + 인덱스 구축
```

### 2) 유사도 검색
```python
docs = vs.similarity_search("웨이퍼에 패턴 만드는 공정은?", k=3)
for d in docs:
    print(d.metadata["source"], d.page_content[:60])

# 점수까지: (문서, 거리) — 거리가 작을수록 유사
for d, score in vs.similarity_search_with_score("...", k=3):
    print(score, d.page_content[:40])
```

### 3) 저장 / 로드 (재임베딩 방지)
```python
vs.save_local("faiss_index")                          # 디스크에 저장
vs2 = FAISS.load_local("faiss_index", embeddings,
                       allow_dangerous_deserialization=True)  # 로드
```

### 4) 인덱스 타입 직접 지정 (대규모, 최신)
기본 `from_documents`는 Flat이다. 대규모에서는 HNSW로 만든다.
```python
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore

dim = 1024                                   # BGE-M3 차원
index = faiss.IndexHNSWFlat(dim, 32)         # 32 = 그래프 이웃 수(M)
index.metric_type = faiss.METRIC_INNER_PRODUCT

vs = FAISS(embedding_function=embeddings, index=index,
           docstore=InMemoryDocstore(), index_to_docstore_id={})
vs.add_documents(chunks)
```

### 5) 증분 추가 / 삭제
```python
vs.add_documents([Document(page_content="새 문서 ...", metadata={"source":"x"})])
# vs.delete(ids=[...])   # id 기반 삭제
```

## 사내 적용 메모
- **BGE-M3**는 dense(임베딩)뿐 아니라 sparse·multi-vector도 지원하지만, LangChain `OpenAIEmbeddings` 경로로는 dense만 쓴다. sparse까지 쓰려면 하이브리드 검색([10번](./10-retriever-tuning.md))에서 BM25와 결합.
- DRM 문서가 많아 **텍스트 추출 자체가 병목**이다 → 스크린샷+VLM(Qwen3-VL)으로 텍스트화한 뒤 임베딩([12번](./12-rag-document-sources.md)).

## 관련 문서
- [10. Retriever 구성 & 튜닝](./10-retriever-tuning.md) — 검색 품질 올리기
- [11. RAG 질의응답 흐름](./11-rag-qa-flow.md) — 이 벡터스토어를 LLM과 연결
- [12. PDF·웹·내부 지식 적용](./12-rag-document-sources.md) — 다양한 소스 로딩

## 참고 자료 (References)
- FAISS wiki: https://github.com/facebookresearch/faiss/wiki
- LangChain FAISS: https://docs.langchain.com/oss/python/integrations/vectorstores/faiss
- BGE-M3: https://huggingface.co/BAAI/bge-m3
