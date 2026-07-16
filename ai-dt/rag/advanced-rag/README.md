# Advanced RAG 완전 가이드

> Naive RAG에서 Agentic RAG, 멀티에이전트 통합까지 — 실무 구현 중심 학습 시리즈

---
tags: [rag, advanced-rag, agentic-rag, langgraph, multi-agent]
level: intermediate → advanced
last_updated: 2026-07-16
---

## 왜 필요한가? (Why)

기본 RAG(검색 → 생성)는 **관련 없는 문서가 컨텍스트에 포함**되거나, **검색 실패 시 복구 수단이 없어** 답변 품질이 떨어진다. Advanced RAG는 이 한계를 **조건 분기, 문서 판별, 쿼리 재작성** 등으로 극복한다. 최종적으로 Agentic RAG는 LLM이 스스로 검색 전략을 판단·수정하는 **자율 에이전트 루프**를 구성한다.

## 학습 로드맵

```
1. Advanced RAG 파이프라인 (문서 판별 + 조건 분기)
   ↓
2. Agentic RAG 구현 (StateGraph + 자율 재검색 루프)
   ↓
3. 확장 기법 (HyDE, MemorySaver, Naive vs Agentic 비교)
   ↓
4. 멀티에이전트 통합 (Supervisor + SubAgent 오케스트레이션)
```

## 문서 목차

| # | 문서 | 설명 | 난이도 |
|---|------|------|--------|
| 1 | [Advanced RAG 파이프라인](./advanced-rag-pipeline.md) | 문서 로딩 → 분할 → 임베딩 → 벡터스토어 → Retriever 구성 | ⭐⭐ |
| 2 | [Agentic RAG 구현](./agentic-rag-implementation.md) | StateGraph로 retrieve → grade → generate/rewrite 조건 분기 구현 | ⭐⭐⭐ |
| 3 | [RAG 확장 기법](./rag-extensions.md) | HyDE, MemorySaver, Naive vs Agentic 비교 실험 | ⭐⭐⭐ |
| 4 | [멀티에이전트 RAG 통합](./multi-agent-rag-integration.md) | Supervisor 패턴 + SKILL.md 기반 SubAgent 오케스트레이션 | ⭐⭐⭐⭐ |

## Naive RAG vs Advanced RAG vs Agentic RAG 비교

| 항목 | Naive RAG | Advanced RAG | Agentic RAG |
|------|-----------|-------------|-------------|
| **검색** | 단순 유사도 검색 | 하이브리드 검색 (키워드+벡터) | 자율 검색 전략 결정 |
| **문서 판별** | 없음 | LLM 기반 관련성 판별 | 판별 + 조건 분기 |
| **검색 실패 대응** | 없음 | 쿼리 재작성 | 자율 재검색 루프 |
| **생성** | 검색 결과 전부 사용 | 필터링된 문서만 사용 | 문서 품질 보장 후 생성 |
| **아키텍처** | 선형 파이프라인 | 조건 분기 파이프라인 | StateGraph 기반 에이전트 |
| **확장성** | 제한적 | 노드 추가로 확장 | 멀티에이전트 통합 가능 |

## 기술 스택

| 구성요소 | 기술 | 역할 |
|---------|------|------|
| LLM | OpenAI GPT-4.1 | 생성·판별·재작성 |
| 임베딩 | text-embedding-3-small | 문서 벡터화 |
| 벡터스토어 | ChromaDB | 로컬 파일 기반 벡터 DB |
| 문서 처리 | LangChain (DirectoryLoader, TextSplitter) | 로딩·분할 |
| 그래프 엔진 | LangGraph (StateGraph) | 조건 분기 워크플로우 |
| 구조화 출력 | Pydantic + with_structured_output | 판별 스키마 강제 |
| 멀티에이전트 | langgraph-supervisor / create_agent | 오케스트레이션 |

## 사전 지식

- Python 기본 문법
- LangChain 기본 개념 (LLM, Chain, Prompt, Tool)
- 벡터 검색 기본 개념 (Embedding, Cosine Similarity)
- [LangGraph 기초](../langgraph/langgraph-basics.md) 권장

## 관련 문서

- [LangGraph 기초](../langgraph/langgraph-basics.md)
- [LangGraph RAG (Corrective RAG)](../langgraph/langgraph-rag.md)
- [LangGraph 고급 패턴](../langgraph/langgraph-advanced.md)
- [LangChain-LangGraph 실전 플레이북](../langchain-langgraph/rag-tool-calling-playbook.md)
- [토큰 전략 (문서 분할)](../token_strategy/README.md)
