# AI/DT 학습 노트

> AI/DT 시스템 개발 관련 학습 내용을 정리한다.

## 목차

### RAG (Retrieval-Augmented Generation)

#### LangGraph
- [LangGraph 시리즈 목차](./rag/langgraph/README.md)
  - [LangGraph 기초](./rag/langgraph/langgraph-basics.md) - State, Node, Edge, 기본 그래프 구성
  - [LangGraph RAG](./rag/langgraph/langgraph-rag.md) - Corrective RAG 파이프라인 구현
  - [LangGraph 고급](./rag/langgraph/langgraph-advanced.md) - Human-in-the-loop, Subgraph, Persistence, Streaming

#### Milvus
- [Milvus 시리즈 목차](./rag/milvus/README.md)
  - [Milvus 기초](./rag/milvus/milvus-basics.md) - 아키텍처, Collection, Index, 유사도 검색
  - [Milvus RAG 연동](./rag/milvus/milvus-rag-integration.md) - LangChain/LangGraph와 Milvus 통합

#### OpenSearch
- [OpenSearch 시리즈 목차](./rag/opensearch/README.md)
  - [OpenSearch 기초](./rag/opensearch/opensearch-basics.md) - 아키텍처, 핵심 개념, 설치 및 클러스터 관리
  - [벡터 검색 (k-NN)](./rag/opensearch/vector-search-knn.md) - k-NN 플러그인, 임베딩 인덱싱, 유사도 검색
  - [키워드 검색 (BM25)](./rag/opensearch/keyword-search-bm25.md) - Full-text 검색, 분석기, 한국어 처리
  - [하이브리드 검색](./rag/opensearch/hybrid-search.md) - 벡터 + 키워드 결합, Score Normalization, RRF

#### Hybrid Search
- [보조 용어 사전 DB (Elasticsearch)](./rag/auxiliary-glossary-db.md) - ES BM25로 전문 용어 관리, 쿼리 확장, LangGraph 통합

### MCP (Model Context Protocol)
- [MCP 시리즈 목차](./mcp/README.md)
  - [MCP 기초](./mcp/mcp-basics.md) - Server/Client 아키텍처, Tools/Resources/Prompts, FastMCP
  - [MCP + LangGraph 연동](./mcp/mcp-langgraph-integration.md) - langchain-mcp-adapters, ReAct Agent

### LLM Fine-Tuning
- [Unsloth 기반 sLLM 파인튜닝 가이드](./unsloth/README.md)
  - [Unsloth 개요](./unsloth/unsloth-overview.md) - 무엇이 특별한지, 왜 쓰는지, 언제 맞는지
  - [로컬 sLLM 파인튜닝 워크플로우](./unsloth/local-sllm-finetuning-workflow.md) - 로컬 API teacher + GPU student 운영 방식
  - [데이터셋과 Chat Template 가이드](./unsloth/dataset-and-chat-template-guide.md) - synthetic data, role format, template mismatch 방지
  - [학습 및 배포 레시피](./unsloth/training-and-deployment-recipe.md) - 설치, SFT 코드, 하이퍼파라미터, GGUF export

### 데이터 처리 (Data Handling)
- _준비 중_
