# OpenSearch 학습 노트

> OpenSearch를 활용한 벡터 검색(Vector Search)과 키워드 검색(Keyword Search) 학습 문서

## 목차

### 기초
- [OpenSearch 기초](./opensearch-basics.md) - 아키텍처, 핵심 개념, 설치 및 클러스터 관리

### 검색
- [벡터 검색 (k-NN)](./vector-search-knn.md) - k-NN 플러그인, 임베딩 인덱싱, 유사도 검색
- [키워드 검색 (BM25)](./keyword-search-bm25.md) - Full-text 검색, 분석기, 한국어 처리
- [하이브리드 검색](./hybrid-search.md) - 벡터 + 키워드 결합, Score Normalization, RRF

### 실무 적용 (Production & 100GB+)
- [opensearch_handler 핸들러](./opensearch-handler.md) - 범용 Python 패키지 (클라이언트, 인덱스, 문서 CRUD, 검색, Aggregation) | [실습 코드](../../../Codes/python/opensearch_handler/)
- [Python 클라이언트 활용](./python-client.md) - 대용량 Bulk 처리, Async 클라이언트, 에러 핸들링
- [성능 최적화 (Scaling)](./performance-optimization.md) - 100GB+ 데이터 샤딩 전략, 튜닝, 메모리 관리
- [RAG 파이프라인 연동](./rag-integration.md) - LangChain/LangGraph 통합

### 응용
- [대화 메모리 구현](./conversation-memory-opensearch.md) - 3계층 메모리(단기/중기/장기), 벡터+키워드 검색, 로컬 LLM 연동 | [실습 코드](../../../Codes/python/history-opensearch/)

### 프로젝트 아키텍처
- [knowhow-elasticsearch 구조](./knowhow-elasticsearch-architecture.md) - LLM enrichment → 색인 → 하이브리드 검색 파이프라인, opensearch_handler 연동 | [실습 코드](../../../Codes/python/knowhow-elasticsearch/)

---

## 학습 순서

1. **OpenSearch 기초** → OpenSearch가 무엇인지, 왜 필요한지 이해
2. **벡터 검색** → 의미 기반 검색(Semantic Search) 구현
3. **키워드 검색** → 정확한 용어 매칭, 한국어 처리
4. **하이브리드 검색** → 두 검색의 장점을 결합
5. **Python 클라이언트** → 실제 코드로 대용량 데이터 제어
6. **성능 최적화** → 100GB 이상의 데이터를 운영하기 위한 전략 수립

---

*Last updated: 2026-02-12*