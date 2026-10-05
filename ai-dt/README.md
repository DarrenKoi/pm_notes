---
tags: [ai-dt, study-index]
reviewed_on: 2026-10-04
review_status: reviewed_with_limits
document_type: index
category_major: "AI·DT"
category_middle: "주제 안내"
category_minor: "전체 목차"
note_kind: "목차"
classified_on: "2026-10-05"
---

# AI/DT 학습 노트

## 대·중·소분류로 찾기

[주제별 분류 목차](./taxonomy-index.md)에서 **AI·DT → 중분류 → 소분류**로 탐색한다. 각 문서의 `category_major`·`category_middle`·`category_minor`는 주제, `note_kind`는 용도다. 기존 경로와 아래 읽기 순서는 유지한다.

> AI/DT 시스템 개발 관련 학습 내용을 정리한다.

## 목차

### RAG (Retrieval-Augmented Generation)

#### LangGraph
- [LangGraph 시리즈 목차](./rag/langgraph/README.md)
  - [LangGraph 기초](./rag/langgraph/langgraph-basics.md) - State, Node, Edge, 기본 그래프 구성
  - [LangGraph RAG](./rag/langgraph/langgraph-rag.md) - Corrective RAG 파이프라인 구현
  - [LangGraph 고급](./rag/langgraph/langgraph-advanced.md) - Human-in-the-loop, Subgraph, Persistence, Streaming


#### LangChain + LangGraph
- [LangChain + LangGraph 실전 가이드](./rag/langchain-langgraph/README.md)
  - [기초 사용법](./rag/langchain-langgraph/langchain-langgraph-basics.md) - LangChain vs LangGraph 역할과 기본 코드 패턴
  - [RAG + Tool Calling 실전](./rag/langchain-langgraph/rag-tool-calling-playbook.md) - 라우팅, 검색, 도구 호출, 운영 확장 포인트

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

### LangChain / LangGraph / RAG 커리큘럼 (study_list)
- [LangChain 학습 노트 목차](./langchain/README.md) - `study_list.txt` 4모듈 커리큘럼, 환경변수 기반 접속 조건·실제 API와 fixture 구분
  - **M1 LangChain 기초**: [OpenAI API & 구조](./langchain/01-openai-api-and-langchain-structure.md) · [LCEL](./langchain/02-lcel.md) · [Chain·Agent·Tool](./langchain/03-chain-agent-tool.md) · [외부 API Agent](./langchain/04-external-api-agent.md)
  - **M2 LangGraph**: [상태 머신](./langchain/05-langgraph-overview-state-machine.md) · [멀티스텝 대화](./langchain/06-multistep-conversation-flow.md) · [Condition/Branching/Tool](./langchain/07-condition-branching-tool-workflow.md) · [시나리오 Agent](./langchain/08-langgraph-agent-scenario.md)
  - **M3 RAG**: [임베딩·FAISS](./langchain/09-document-embedding-faiss.md) · [Retriever 튜닝](./langchain/10-retriever-tuning.md) · [RAG 질의응답](./langchain/11-rag-qa-flow.md) · [PDF/웹/내부지식](./langchain/12-rag-document-sources.md)
  - **M4 Mini Project**: [주제선정→구현→테스트→발표](./langchain/13-mini-project.md)

### LLMOps & 평가(Evaluation) 커리큘럼 (study_list)
- [LLMOps & 평가 학습 노트 목차](./LLMOps/README.md) - `study_list.txt` 6영역 커리큘럼, 원문17개 개별 검토·접속 조건/운영 미확인 구분
  - **M1 LLMOps 기초**: [개요·라이프사이클](./LLMOps/01-llmops-overview-lifecycle.md) · [프롬프트 버전관리](./LLMOps/02-prompt-management-versioning.md) · [트레이싱·관측성](./LLMOps/03-tracing-observability.md)
  - **M2 평가 기초**: [평가 개요](./LLMOps/04-llm-evaluation-overview.md) · [평가 데이터셋](./LLMOps/05-eval-dataset-construction.md) · [자동 지표](./LLMOps/06-automatic-metrics.md)
  - **M3 심화 평가**: [LLM-as-a-Judge](./LLMOps/07-llm-as-a-judge.md) · [RAG 평가](./LLMOps/08-rag-evaluation.md) · [Agent·Tool 평가](./LLMOps/09-agent-tool-evaluation.md)
  - **M4 안전·운영**: [안전·환각·가드레일](./LLMOps/10-safety-hallucination-guardrails.md) · [온라인 평가·배포](./LLMOps/11-online-eval-deployment.md) · [모니터링·드리프트](./LLMOps/12-monitoring-drift.md)
  - **M5 Mini Project**: [사내 RAG/Agent 평가 파이프라인](./LLMOps/13-mini-project.md)

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

### Foundation Model / LLM 기초
- [Foundation Model / LLM 기초](./foundation%20model/README.md)
  - [Attention이란 무엇인가?](./foundation%20model/attention.md) - attention의 직관, Q/K/V, self-attention, causal mask
  - [Transformer가 어떻게 LLM으로 이어졌는가?](./foundation%20model/transformer-to-llm.md) - RNN에서 Transformer, BERT/GPT/T5, foundation model로 이어지는 흐름
  - [Encoder와 Decoder란 무엇인가?](./foundation%20model/encoder-and-decoder.md) - encoder-only, decoder-only, encoder-decoder 구조 비교
  - [Foundation LLM은 어떻게 만들어지는가?](./foundation%20model/how-foundation-llms-are-built.md) - 데이터, 사전학습, scaling, alignment, chat model 제작 과정

### 데이터 처리 (Data Handling)
- [Data Handling 시리즈 목차](./data-handling/README.md)
  - [Airflow + MinIO 파이프라인 튜토리얼](./data-handling/airflow-minio-tutorial.md) - 폐쇄망 Airflow에서 Python 코드를 순차 실행하여 MinIO 데이터를 처리하는 ETL 파이프라인
  - [데이터 정규화 시리즈](./data-handling/normalization/README.md) - RDB/OpenSearch/MongoDB/Redis/RAG/온톨로지 관점에서 정규화를 정리한 8편 시리즈 (cross-layer cheatsheet 포함)

### AI Coding Dictionary (용어 사전)
- [AI Coding Dictionary 한국어 학습 노트](./ai-coding-dictionary/README.md) - Matt Pocock의 [dictionary-of-ai-coding](https://github.com/mattpocock/dictionary-of-ai-coding) 한국어 정리 (60+ 용어, 7개 섹션)
  - [01. The Model](./ai-coding-dictionary/01-the-model.md) - Model, Parameters, Token, Inference, Prefix cache 등 모델·비용 구조
  - [02. Sessions, Context Windows & Turns](./ai-coding-dictionary/02-sessions-context-windows-turns.md) - Session/Turn/Request 계층, Stateless vs Stateful
  - [03. Tools & Environment](./ai-coding-dictionary/03-tools-environment.md) - Tool, MCP, Sandbox, Permission/Agent mode
  - [04. Failure Modes](./ai-coding-dictionary/04-failure-modes.md) - Hallucination(2종), Sycophancy, Attention degradation, Smart/Dumb zone
  - [05. Handoffs](./ai-coding-dictionary/05-handoffs.md) - Clearing, Spec, Ticket, Compaction
  - [06. Memory and Steering](./ai-coding-dictionary/06-memory-and-steering.md) - Memory system, AGENTS.md, Skill, Subagent
  - [07. Patterns of Work](./ai-coding-dictionary/07-patterns-of-work.md) - AFK, Vibe coding, Grilling, Human/Automated review

### AI 용어 및 기술 소개
- [AI 용어 및 기술 소개](./ai-terms-and-technologies/README.md) - 전사 구성원을 위한 LLM·RAG·Agent·AI 리터러시·보안·AX 입문 자료
  - [통합본](./ai-terms-and-technologies/all-in-one.md) - 9개 주제를 한 편으로 읽는 문서
  - [01. AI 모델의 종류](./ai-terms-and-technologies/01-ai-model-families.md) - LLM, SLM, VLM, VLA, 지식 증류
  - [02. 벡터, 임베딩, RAG와 온톨로지](./ai-terms-and-technologies/02-vector-embedding-and-rag.md) - 지식 자산화, 의미 검색과 검색 증강 생성
  - [03. 제로샷과 퓨샷](./ai-terms-and-technologies/03-zero-shot-and-few-shot.md) - 프롬프트 예시의 수와 활용 기준
  - [04. 바이브 코딩에서 하네스까지](./ai-terms-and-technologies/04-vibe-coding-harness-agent-orchestration.md) - Harness, AI Agent, Orchestration
  - [05. AI 리터러시, 편향과 슬롭](./ai-terms-and-technologies/05-ai-literacy-bias-and-slop.md) - 책임 있는 AI 활용과 검증
  - [06. 탈옥과 AI 보안](./ai-terms-and-technologies/06-jailbreak-and-ai-security.md) - 탈옥·프롬프트 인젝션과 최근 사례
  - [07. AX와 AI 데이터 센터](./ai-terms-and-technologies/07-ax-and-ai-data-centers.md) - AI 전환과 이를 뒷받침하는 인프라
  - [08. 토큰, 컨텍스트 윈도우, 환각](./ai-terms-and-technologies/08-token-context-window-and-hallucination.md) - 토큰·컨텍스트 한도·환각·비결정성
  - [09. 프롬프트, RAG, 파인튜닝 중 무엇을 선택할까](./ai-terms-and-technologies/09-prompt-rag-finetuning-and-evaluation.md) - 방법 선택 기준과 골든셋 평가

### Agent Documentation
- [OpenWiki 사용법](./openwiki/README.md) - 코드베이스 문서를 생성/갱신하는 에이전트용 CLI 설치, 실행, CI 운영 가이드

### 머신러닝·딥러닝 학습과 실습

- [ML/DL 학습 목차](./ml-dl/README.md) — 데이터 처리 → 클래식 ML → 딥러닝 → 저장·서빙·실험 추적 순서로 읽는다. CPU 로컬 검증과 GPU·원격 서비스 미확인을 구분한다.

### 업무 설계와 출제 기록

- [로드맵 자료 목차](./roadmap/README.md) — 당시 보고서 설계, 발표 노트, ADR, 회의·인계 기록의 역할과 읽기 순서.
- [LLM 경진대회 출제 기록 목차](./llm_question/README.md) — 초안과 개정판의 시간·평가 조건을 구분해 읽는다. 과거 계획은 현재 운영 승인이나 제품 보장이 아니다.

## 문서 역할과 검토 상태

이 목차는 각 주제 안에서 학습을 시작할 위치를 제공한다. LangChain 커리큘럼과 RAG 안의 실전 노트는 학습 순서·시나리오가 다르며 현재 서로 이동·통합하지 않는다. ai-dt의 주제 하위 폴더도 독립적이라는 규칙을 적용한다. roadmap과 llm_question은 업무 설계·출제 기록이며 당시 결정/시간 제약을 현재 제품 사실로 고쳐 쓰지 않는다.

**원문 검토 178/178개, 목차 최종 대조 2026-10-05**: 현재 Markdown은 정리 안내·기록을 포함해 213개다. LLMOps 대소문자 참조를 실제 경로에 맞췄고 13개 하위 주제의 읽기 입구를 제공한다. 기술 근거와 실행 검증의 기준일은 각 문서에 적힌 2026-10-04이며, 다음 날의 목차·참조 대조를 기술 정보 재검증으로 간주하지 않는다. [정리 기록](./organization-log.md)의 범위·미확인을 기준으로 읽는다. 실제 모델·서버·회사 환경, 일부 전체 읽기 화면과 Claude 협의가 필요한 추가 통합은 미완료다.
