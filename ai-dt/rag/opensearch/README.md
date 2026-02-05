# OpenSearch 학습 노트

> OpenSearch를 활용한 벡터 검색(Vector Search)과 키워드 검색(Keyword Search) 학습 문서

## 목차

### 기초
- [OpenSearch 기초](./opensearch-basics.md) - 아키텍처, 핵심 개념, 설치 및 클러스터 관리

### 검색
- [벡터 검색 (k-NN)](./vector-search-knn.md) - k-NN 플러그인, 임베딩 인덱싱, 유사도 검색
- [키워드 검색 (BM25)](./keyword-search-bm25.md) - Full-text 검색, 분석기, 한국어 처리
- [하이브리드 검색](./hybrid-search.md) - 벡터 + 키워드 결합, Score Normalization, RRF

### 실무 적용
- [Python 클라이언트 활용](./python-client.md) - opensearch-py, 벌크 작업, 비동기 처리
- [RAG 파이프라인 연동](./rag-integration.md) - LangChain/LangGraph 통합 (예정)

---

## 학습 순서

1. **OpenSearch 기초** → OpenSearch가 무엇인지, 왜 필요한지 이해
2. **벡터 검색** → 의미 기반 검색(Semantic Search) 구현
3. **키워드 검색** → 정확한 용어 매칭, 한국어 처리
4. **하이브리드 검색** → 두 검색의 장점을 결합
5. **Python 클라이언트** → 실제 코드로 OpenSearch 제어

---

*Last updated: 2026-02-05*
