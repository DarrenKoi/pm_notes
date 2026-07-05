# LangChain / LangGraph / RAG 학습 노트

> `study_list.txt` 커리큘럼(LangChain → LangGraph → RAG → Mini Project)을 따라, 기초 개념부터 최신 기법까지 단계별로 정리한 실습 노트입니다.

---

## 🎯 이 노트의 방향

- **언어**: 한국어 (기술 용어는 영어 병기)
- **코드**: 실행 가능한 완전한 예제. 모든 LLM 호출은 **공개 OpenAI API**와 **사내 OpenAI 호환 엔드포인트**(Kimi-K2.5 / Qwen3-VL / BGE-M3)를 **병기**합니다.
- **깊이**: 커리큘럼의 각 라인을 별도 문서로 다루고, 2024년 이후 핵심 기법(LCEL Runnable, tool calling, `with_structured_output`, `create_agent`, LangGraph `create_react_agent`, checkpointer, 하이브리드 검색·리랭킹·CRAG/Self-RAG·Agentic RAG 등)을 포함합니다.

> ⚠️ **사내 환경 주의**: 외부 LLM API(OpenAI/Anthropic/Google)는 방화벽으로 차단됩니다. 실제 사내 실행 시에는 항상 "사내 엔드포인트" 코드 블록을 사용하세요. 공개 API 블록은 커리큘럼의 표준 패턴을 이해하기 위한 참고용입니다.

## API 버전 메모

- 일반 Agent는 LangChain의 `create_agent`를 우선 학습합니다. 현재 LangChain 문서는 이를 모델·도구·프롬프트를 조합하는 고수준 agent harness로 설명합니다.
- LangGraph는 장시간 실행, 상태 저장, HITL, 조건 분기처럼 흐름 제어가 중요한 Agent 워크플로우에 사용합니다. 이 문서 묶음에서는 05~08번에서 LangGraph를 별도로 다룹니다.

---

## 📚 목차 (학습 순서)

### 1. LangChain 이해 및 활용
| 문서 | 내용 |
|------|------|
| [커리큘럼 커버리지 점검](./curriculum-coverage.md) | `study_list.txt` 항목별 생성 문서 매핑과 보강 포인트 |
| [01. OpenAI API & LangChain 구조](./01-openai-api-and-langchain-structure.md) | LLM 호출의 원리, LangChain 패키지 구조, Message/Model/Prompt 추상화 |
| [02. LCEL 실습](./02-lcel.md) | Runnable 프로토콜, `\|` 파이프, `invoke/stream/batch`, 병렬·조건 구성 |
| [03. Chain · Agent · Tool](./03-chain-agent-tool.md) | Tool 정의, `bind_tools`, `create_agent`, `with_structured_output`, Agent 루프 |
| [04. 외부 API 연동 Agent](./04-external-api-agent.md) | 실 API를 호출하는 Tool 설계, 인증·에러·재시도·타임아웃, Agent 연결 |

### 2. LangGraph 워크플로우 설계
| 문서 | 내용 |
|------|------|
| [05. LangGraph 개요 & 상태 머신](./05-langgraph-overview-state-machine.md) | State/Node/Edge, StateGraph, reducer, 컴파일 |
| [06. 멀티스텝 대화 흐름](./06-multistep-conversation-flow.md) | 상태 누적, checkpointer 기반 메모리, thread_id |
| [07. Condition · Branching · Tool workflow](./07-condition-branching-tool-workflow.md) | `add_conditional_edges`, `ToolNode`, `tools_condition`, 루프 |
| [08. 시나리오 기반 Agent 구축](./08-langgraph-agent-scenario.md) | LangGraph `create_react_agent`, 커스텀 그래프, HITL |

### 3. RAG 기반 응답 Agent 개발
| 문서 | 내용 |
|------|------|
| [09. 문서 임베딩 & FAISS](./09-document-embedding-faiss.md) | 임베딩 원리, BGE-M3, FAISS 인덱스(Flat/IVF/HNSW), 저장·로드 |
| [10. Retriever 구성 & 튜닝](./10-retriever-tuning.md) | search_type, MMR, 하이브리드 검색, 리랭킹, 청킹 전략 |
| [11. RAG 질의응답 흐름](./11-rag-qa-flow.md) | LCEL RAG chain, 인용, CRAG/Self-RAG, Agentic RAG |
| [12. PDF·웹·내부 지식 적용](./12-rag-document-sources.md) | 로더, DRM 문서 VLM 파이프라인, 메타데이터 필터 |

### 4. 실전 Mini Project
| 문서 | 내용 |
|------|------|
| [13. Mini Project 가이드](./13-mini-project.md) | 주제 선정 → 요구사항 → 구현 → 테스트 → 발표/피드백 |

---

## 🧰 공통 개발 환경

```bash
# 핵심 패키지 (버전은 예시, 실제로는 사내 미러/프록시 사용)
pip install langchain langchain-openai langchain-community langgraph
pip install faiss-cpu pypdf beautifulsoup4 rank-bm25
```

### LLM / 임베딩 클라이언트 (모든 문서 공통 보일러플레이트)

```python
# ── 공개 OpenAI API (커리큘럼 표준, 사내에서는 차단됨) ──────────────
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)          # OPENAI_API_KEY 환경변수 사용
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ── 사내 OpenAI 호환 엔드포인트 (실제 사용) ────────────────────────
llm = ChatOpenAI(
    model="Kimi-K2.5",                    # 사내 텍스트 LLM
    base_url="http://llm-gateway.internal/v1",
    api_key="EMPTY",                      # 게이트웨이가 인증 대행 시 더미값
    temperature=0,
)
embeddings = OpenAIEmbeddings(
    model="BGE-M3",                       # 사내 임베딩 모델
    base_url="http://llm-gateway.internal/v1",
    api_key="EMPTY",
)
```

> `base_url`과 `model`만 바꾸면 나머지 LangChain 코드는 **완전히 동일**합니다. 이것이 "OpenAI 호환"의 핵심 이점입니다.

---

## 📖 참고 자료 (References)
- LangChain (Python): https://docs.langchain.com/oss/python/
- LangGraph: https://docs.langchain.com/oss/python/langgraph/overview
- FAISS: https://github.com/facebookresearch/faiss/wiki
- BGE-M3: https://huggingface.co/BAAI/bge-m3
