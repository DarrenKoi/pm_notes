---
tags: [opensearch, documentation-review]
aliases: [OpenSearch 학습 목차]
reviewed_on: 2026-10-04
review_status: partial
document_type: index
category_major: "AI·DT"
category_middle: "RAG"
category_minor: "OpenSearch 검색"
note_kind: "목차"
classified_on: "2026-10-05"
---

# OpenSearch 학습 노트

> OpenSearch의 키워드·벡터 검색과 클라이언트/운영 조건을 학습한다.

> [!info] 검토 조건 — 2026-10-04
> 학습 예제다. 역사적 작성일은 유지하고 이번 검토일을 별도로 기록했다. 로컬 확인은 Python3.14.2·opensearch-py3.2.0 및 모의 REST 응답에 한정한다. 실제 OpenSearch/Docker·보안·검색 품질·100GB 부하를 검증하지 않았다. 함수는 대상과 입력을 명시해 호출한다. [정리 기록](../organization-log.md)을 함께 읽는다.


## 목차

### 기초
- [OpenSearch 기초](./opensearch-basics.md) - 아키텍처, 핵심 개념, 설치 및 클러스터 관리

### 검색
- [벡터 검색 (k-NN)](./vector-search-knn.md) - k-NN 플러그인, 임베딩 인덱싱, 유사도 검색
- [키워드 검색 (BM25)](./keyword-search-bm25.md) - Full-text 검색, 분석기, 한국어 처리
- [하이브리드 검색](./hybrid-search.md) - 벡터 + 키워드 결합, Score Normalization, RRF

### 운영 조건과 응용 예제
- [opensearch_handler 핸들러](./opensearch-handler.md) - 범용 Python 패키지 (클라이언트, 인덱스, 문서 CRUD, 검색, Aggregation) — 저장소의 독립 코드 주제 `Codes/python/opensearch_handler`와 용도가 다르며 여기에서는 코드 폴더를 이동/통합하지 않는다.
- [Python 클라이언트 활용](./python-client.md) - 대용량 Bulk 처리, Async 클라이언트, 에러 핸들링
- [성능 최적화 (Scaling)](./performance-optimization.md) - 100GB+ 데이터 샤딩 전략, 튜닝, 메모리 관리
- [Settings 실무 가이드](./settings/README.md) - 매핑(토큰화/비토큰화), 템플릿, alias, rollover, ISM 삭제 정책
- [RAG 파이프라인 연동](./rag-integration.md) - LangChain/LangGraph 통합

### 응용
- [대화 메모리 구현](./conversation-memory-opensearch.md) - 3계층 메모리(단기/중기/장기), 벡터+키워드 검색, 로컬 LLM 연동 — `Codes/python/history-opensearch`는 별도 독립 코드 주제다.

---

> [!note] 개별 검토 진행 중
> 원래 11문서 중 이 목차·기초·Python 클라이언트·BM25·벡터·하이브리드·성능·Settings·핸들러·RAG 연동·대화 메모리의 11개 전체를 개별 검토했다. 실제 서버·모델/메모리 구현과 운영 조건은 미확인이다. 100GB/운영 적합성을 문서 제목만으로 보장하지 않는다.

기초는 Index/Document·schema/샤드/실습 연결, Python 클라이언트는 bulk/scan/async/API 실패 처리를 다룬다. 범용 handler와 메모리 문서는 별도 적용 맥락이라 목차에서 역할을 구분한다. 벡터/키워드/hybrid/운영 문서의 중복 재구성과 선택은 Claude 협의 연결 실패로 보류한다.

## 학습 순서

1. **OpenSearch 기초** → OpenSearch가 무엇인지, 왜 필요한지 이해
2. **벡터 검색** → 의미 기반 검색(Semantic Search) 구현
3. **키워드 검색** → 정확한 용어 매칭, 한국어 처리
4. **하이브리드 검색** → 두 검색의 장점을 결합
5. **Python 클라이언트** → 실제 코드로 대용량 데이터 제어
6. **성능 최적화** → 원래 100GB 규모 가정의 샤드/메모리 계획을 실제 부하로 검증
7. **범용 핸들러** → SDK 함수와 래퍼의 책임/예제 차이를 확인
8. **Settings 실무 가이드** → mapping·template·alias·수명 정책의 적용 조건 확인
9. **RAG 연동** → 검색 후보/LLM 입력과 답변 근거의 계약 학습
10. **대화 메모리** → 상태/검색 저장과 사용자 소유권·삭제 조건을 별도로 검토

---

*Last updated: 2026-02-12*
