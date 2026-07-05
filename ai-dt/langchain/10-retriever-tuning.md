---
tags: [rag, retriever, mmr, hybrid-search, reranking, chunking]
level: advanced
last_updated: 2026-07-06
---

# 10. Retriever 구성 및 성능 튜닝

> "검색이 나쁘면 RAG도 나쁘다." Retriever의 파라미터·검색 전략(MMR/하이브리드/리랭킹)과 청킹 전략으로 검색 품질을 끌어올린다.

## 왜 필요한가? (Why)

- RAG 품질의 병목은 대부분 **생성이 아니라 검색(retrieval)**이다. 엉뚱한 문서를 넣으면 아무리 좋은 LLM도 틀린다.
- 단순 임베딩 검색은 (1) 동의어엔 강하지만 **정확한 용어/코드/모델명**엔 약하고, (2) 유사 문서가 **중복**되면 다양성이 떨어진다.
- 하이브리드 검색·리랭킹·청킹 튜닝으로 이 약점을 메운다. 사내 전문 용어(장비명, 파트넘버)가 많을수록 하이브리드가 중요하다.

## 핵심 개념 (What)

### Retriever는 Runnable
`vectorstore.as_retriever(...)`는 "질문 → 문서 리스트" Runnable이다. LCEL 체인에 그대로 꽂힌다.

### 검색 파라미터
| 파라미터 | 의미 |
|----------|------|
| `k` | 가져올 문서 수 (보통 3~8) |
| `search_type="mmr"` | 다양성 고려(중복 억제) |
| `fetch_k` | MMR이 후보로 먼저 뽑는 수 |
| `score_threshold` | 임계값 이하 유사도 문서 제외 |
| `filter` | 메타데이터 필터(예: source, date) |

### 3대 튜닝 레버
1. **청킹** — 크기/중첩/분할 기준. RAG 품질에 가장 큰 영향.
2. **하이브리드 검색** — 벡터(의미) + BM25(키워드)를 결합.
3. **리랭킹(rerank)** — 많이 뽑은 뒤(cross-encoder/LLM) 재정렬해 상위만 사용.

## 어떻게 사용하는가? (How)

### 1) 기본 retriever와 MMR
```python
retriever = vs.as_retriever(search_kwargs={"k": 4})

# MMR: 유사하면서도 서로 다른 문서를 고름(중복 억제)
mmr = vs.as_retriever(search_type="mmr",
                      search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5})

# 임계값 필터 + 메타데이터 필터
filtered = vs.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.3, "k": 5,
                   "filter": {"source": "photo.md"}})
```

### 2) 하이브리드 검색 (BM25 + 벡터)
정확한 용어 매칭(BM25)과 의미 매칭(벡터)을 앙상블한다.
```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

bm25 = BM25Retriever.from_documents(chunks); bm25.k = 4
vector = vs.as_retriever(search_kwargs={"k": 4})

hybrid = EnsembleRetriever(retrievers=[bm25, vector], weights=[0.4, 0.6])
docs = hybrid.invoke("EQP-102 온도과열 알람 대응")
```
> 사내 팁: 장비명·파트넘버·에러코드처럼 **정확 매칭이 중요한 토큰**이 많으면 BM25 가중치를 높인다. (운영 DB가 OpenSearch면 BM25+kNN을 한 쿼리로 처리 가능하지만, 로컬 PoC는 위 조합이 간편)

### 3) 리랭킹 (많이 뽑고 정밀 재정렬)
```python
# LLM을 리랭커로 쓰는 간단 버전 (외부 rerank 모델이 막혀 있어도 동작)
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="Kimi-K2.5", base_url="http://llm-gateway.internal/v1",
                 api_key="EMPTY", temperature=0)

compressor = LLMChainFilter.from_llm(llm)     # 관련 없는 문서를 LLM이 걸러냄
rerank_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vs.as_retriever(search_kwargs={"k": 12}),  # 많이 뽑고
)
docs = rerank_retriever.invoke("포토 공정 결함 원인")           # 정제
```
> 전용 리랭커가 사내에 서빙되면(cross-encoder) 그걸 쓰는 게 더 정확·저렴하다.

### 4) 청킹 전략 튜닝
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 시작점: 500/50. 문서 특성에 맞춰 실험
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],   # 큰 경계부터 시도
)
```
**시맨틱 청킹(최신)**: 문장 임베딩 유사도가 급변하는 지점에서 자르기.
```python
# pip install langchain-experimental
from langchain_experimental.text_splitter import SemanticChunker
sem = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
chunks = sem.create_documents([long_text])
```

### 5) 고급 retriever 패턴 (최신)
- **ParentDocumentRetriever**: 작은 조각으로 검색하되, LLM엔 그 부모(더 넓은 맥락)를 전달 → "정밀 검색 + 충분한 맥락".
- **MultiQueryRetriever**: 질문을 LLM이 여러 표현으로 바꿔 각각 검색 후 합집합 → recall↑.
- **SelfQueryRetriever**: 자연어 질문에서 메타데이터 필터를 LLM이 추출(예: "2024년 photo 문서만").
- **Contextual Retrieval**: 각 청크 앞에 "이 청크가 속한 문서/절 요약"을 붙여 임베딩 → 맥락 손실 완화.

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
mq = MultiQueryRetriever.from_llm(retriever=vs.as_retriever(), llm=llm)
docs = mq.invoke("온도 알람 대응법")
```

## 튜닝 체크리스트
- [ ] 청크 크기/중첩을 2~3개 값으로 A/B 해봤는가
- [ ] 전문 용어가 많으면 하이브리드(BM25+벡터)를 적용했는가
- [ ] `k`를 늘리면 recall↑·noise↑ — 리랭킹으로 상위만 남기는가
- [ ] 메타데이터 필터로 검색 범위를 좁힐 수 있는가
- [ ] 검색 결과를 눈으로(오프라인) 평가했는가

## 관련 문서
- [09. 문서 임베딩 & FAISS](./09-document-embedding-faiss.md)
- [11. RAG 질의응답 흐름](./11-rag-qa-flow.md) — 튜닝한 retriever를 체인에 연결
- [12. PDF·웹·내부 지식 적용](./12-rag-document-sources.md)

## 참고 자료 (References)
- Retrievers: https://docs.langchain.com/oss/python/concepts/retrievers
- EnsembleRetriever, ContextualCompressionRetriever, ParentDocumentRetriever, SelfQueryRetriever
- Anthropic, "Contextual Retrieval" (개념 참고)
