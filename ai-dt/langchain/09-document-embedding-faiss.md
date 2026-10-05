---
tags: [rag, embedding, faiss, vectorstore, bge-m3]
level: intermediate
last_updated: 2026-07-06
reviewed_on: 2026-10-04
review_status: partial
document_type: learning_note
category_major: "AI·DT"
category_middle: "에이전트 개발"
category_minor: "RAG 구축 실습"
note_kind: "학습"
classified_on: "2026-10-05"
---

# 09. 문서 임베딩 및 FAISS 벡터 스토어 구축

> 텍스트를 벡터로 바꾸는 임베딩의 원리와, 그 벡터를 빠르게 검색하는 FAISS 인덱스(Flat/IVF/HNSW)를 이해하고 구축한다.


> [!info] 적용 조건과 실행 순서
> 2026-10-04 개별 검토. [공통 적용 조건](./verified-conditions.md)의 판본·설정·검증 경계를 먼저 확인한다. 같은 문서의 코드 조각은 위에서 아래로 이어 실행하며 개념 조각은 별도로 표시한다. 이전 문서의 vs/chunks/embeddings 등은 관련 절의 선행 예제가 필요하다. 공개·사내 API/실제 데이터·운영 실행은 미확인이며 예제 출력은 보장이 아니다.

## 왜 필요한가? (Why)

- LLM은 방대한 사내 문서를 다 기억하지 못한다. 질문과 **의미적으로 유사한** 문서 조각을 찾아 붙여야(RAG) 관련 근거를 제공할 수 있다. 검색/생성의 정확도는 별도 측정한다.
- 키워드 검색은 "동의어/의역"을 놓칠 수 있다. **임베딩**은 의미를 벡터 공간의 위치로 표현해, 표현이 달라도 의미가 가까우면 찾는다.
- FAISS는 로컬에서 가볍게 쓰는 벡터 검색 라이브러리다. DB(OpenSearch/Milvus) 없이도 PoC를 돌릴 수 있어 사내 초기 실험에 적합하다.

## 핵심 개념 (What)

### 임베딩(Embedding)
텍스트 → 고정 길이 실수 벡터(예: BGE-M3는 1024차원). 의미가 비슷한 문장은 벡터도 가깝다(코사인 유사도 ↑). 공식 BGE-M3 dense는1024차원이다. 사내 서빙/alias/정규화/차원/토큰 제한은 미확인이다.

### 청킹(Chunking)
문서는 통째로 임베딩하지 않고 **적당한 크기 조각**으로 나눈다. 너무 크면 노이즈가 섞이고, 너무 작으면 맥락이 끊긴다. (튜닝은 [10번](./10-retriever-tuning.md))

### FAISS 인덱스 종류
| 인덱스 | 특징 | 언제 |
|--------|------|------|
| **Flat** (IndexFlatL2/IP) | 주어진 벡터·거리의 exact 최근접; 의미/답 정확도100% 아님 | 수천~수만 건 PoC |
| **IVF** (IVFFlat) | 클러스터로 나눠 일부만 탐색, 빠름 | 수십만~ |
| **HNSW** | 그래프 기반 근사 최근접, 매우 빠름·고정밀 | 대규모 저지연 |

> 유사도 메트릭: 정규화된 임베딩에는 **내적(IP)=코사인 유사도**. BGE-M3는 코사인 기준이 자연스럽다.

## 어떻게 사용하는가? (How)

### 1) 문서 로드 → 청킹 → 임베딩 → FAISS
```python
import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 공개: embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# 사내:
embeddings = OpenAIEmbeddings(model=os.environ["EMBEDDING_MODEL"],
                              check_embedding_ctx_length=False, encoding_format="float",
                              base_url=os.environ["LLM_BASE_URL"],
                              api_key=os.environ["LLM_API_KEY"])

raw = [Document(page_content="포토 공정은 웨이퍼에 회로 패턴을 전사한다 ...",
                metadata={"source": "photo.md"})]

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(raw)

vs = FAISS.from_documents(chunks, embeddings, normalize_L2=True)   # 임베딩 계산 + 인덱스 구축
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

save_local은 index와 docstore pickle을 저장한다. allow_dangerous_deserialization=True는 안전 옵션이 아니라 pickle 로딩 허용이다. 이 예제가 직접 만든 파일에만 사용한다. 외부/변조 파일은 로딩하지 않고 같은 임베딩 모델·차원·정규화 설정을 유지한다.
```python
vs.save_local("faiss_index")                          # 디스크에 저장
vs2 = FAISS.load_local("faiss_index", embeddings,
                       allow_dangerous_deserialization=True, normalize_L2=True)  # 직접 생성·신뢰한 파일만
```

### 4) 인덱스 타입 직접 지정 (별도 HNSW 예제)
기본 from_documents의 이 예제는 L2 Flat이다. HNSW는 후보이며 품질·메모리·지연을 실제 데이터로 측정한다. 별도 vs_hnsw를 만들고 예제 삭제는 Flat에만 적용한다.
```python
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore

dim = len(embeddings.embed_query("차원 확인"))                                   # BGE-M3 차원
index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_L2)  # M=32; unit L2와 cosine 순위 동일

vs_hnsw = FAISS(embedding_function=embeddings, index=index, normalize_L2=True,
           docstore=InMemoryDocstore(), index_to_docstore_id={})
vs_hnsw.add_documents(chunks)
```

### 5) 증분 추가 / 삭제
```python
added_ids = vs.add_documents([Document(page_content="새 문서 ...", metadata={"source":"x"})])
vs.delete(ids=added_ids)  # 이 Flat index에서만; HNSW remove_ids는 지원하지 않음
```

## 사내 적용 메모
- **BGE-M3**는 dense(임베딩)뿐 아니라 sparse·multi-vector도 지원하지만, LangChain `OpenAIEmbeddings` 경로로는 dense만 쓴다. BGE learned sparse와 BM25는 별개다. 키워드 검색을 결합하려면 하이브리드 검색([10번](./10-retriever-tuning.md))에서 BM25와 결합.
- DRM 문서가 많아 **텍스트 추출 자체가 병목**이다 → 스크린샷+VLM(Qwen3-VL)으로 텍스트화한 뒤 임베딩([12번](./12-rag-document-sources.md)).

## 관련 문서
- [10. Retriever 구성 & 튜닝](./10-retriever-tuning.md) — 검색 품질 올리기
- [11. RAG 질의응답 흐름](./11-rag-qa-flow.md) — 이 벡터스토어를 LLM과 연결
- [12. PDF·웹·내부 지식 적용](./12-rag-document-sources.md) — 다양한 소스 로딩

## 참고 자료 (References)
- FAISS wiki: https://github.com/facebookresearch/faiss/wiki
- LangChain FAISS: https://github.com/langchain-ai/langchain-community/blob/main/libs/community/langchain_community/vectorstores/faiss.py
- BGE-M3: https://huggingface.co/BAAI/bge-m3
