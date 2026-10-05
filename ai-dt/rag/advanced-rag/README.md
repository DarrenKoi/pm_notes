---
tags: [rag, advanced-rag, agentic-rag, langgraph, multi-agent]
level: intermediate → advanced
last_updated: 2026-07-16
reviewed_on: 2026-10-04
review_status: partial
document_type: index
category_major: "AI·DT"
category_middle: "RAG"
category_minor: "고급 RAG·멀티에이전트"
note_kind: "목차"
classified_on: "2026-10-05"
---

# Advanced RAG 완전 가이드

> [!info] 검토 — 2026-10-04
> 공식 근거·설치 판본·로컬 검증은 [RAG 정리 기록](../organization-log.md)에 있다. 제목의 완전/실무는 원래 학습 시리즈 표현이며 서비스 구현 완료가 아니다. 이 시리즈 원문5개를 개별 검토했으며 문서별 근거/검증 한계는 정리 기록을 따른다. 실제 사내 데이터·모델 품질·회사 권한은 미확인이다.


> Naive RAG에서 Agentic RAG, 멀티에이전트 통합까지 — 실무 구현 중심 학습 시리즈


## 왜 필요한가? (Why)

기본 RAG(검색 → 생성)는 **관련 없는 문서가 컨텍스트에 포함**되거나, **검색 실패 시 복구 수단이 없어** 답변 품질이 떨어진다. Advanced RAG는 조건 분기·문서 판별·쿼리 재작성으로 개선을 시도한다. 추가 모델 오류/비용이 생길 수 있어 같은 평가셋으로 비교해야 한다. 최종적으로 Agentic RAG는 LLM이 스스로 검색 전략을 판단·수정하는 **자율 에이전트 루프**를 구성한다.

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
| **검색** | 예: 단일 유사도 검색 | hybrid·reranking 등 선택 개선 | 검색 선택/재시도 정책을 모델·코드로 제어 |
| **문서 판별** | 없음 | LLM 기반 관련성 판별 | 판별 + 조건 분기 |
| **검색 실패 대응** | 없음 | 쿼리 재작성 | 자율 재검색 루프 |
| **생성** | 기본 검색 근거로 생성 | 예: 판별한 근거로 생성 | 근거 판정 후 생성; 품질 보장 아님 |
| **아키텍처** | 예: 선형 흐름 | 다양한 검색/조건 개선 | 상태/도구 루프. StateGraph만 가능한 것은 아님 |
| **확장성** | 필요에 따라 확장 | 검색/판정 요소 확장 | 단일 agent 또는 복수 agent 선택 |

## 기술 스택

| 구성요소 | 기술 | 역할 |
|---------|------|------|
| LLM | 승인된 모델 id/객체 | 생성·판별·재작성; 원래 GPT-4.1은 예시 |
| 임베딩 | 확인한 모델/revision·차원 | 문서와 질의에 같은 embedding 계약; 원래 모델명은 예시 |
| 벡터스토어 | ChromaDB | 로컬 파일 기반 벡터 DB |
| 문서 처리 | LangChain (DirectoryLoader, TextSplitter) | 로딩·분할 |
| 그래프 엔진 | LangGraph (StateGraph) | 조건 분기 워크플로우 |
| 구조화 출력 | Pydantic + with_structured_output | schema/parser 검증. 사실성/모델 지원은 별도 |
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


## 문서별 역할과 실습 조건

이 시리즈의1은 입력 추출·문자 청킹·Chroma 새 저장/재사용, 2는 구조화 관련성 판정, 3은 HyDE/대화/비교, 4는 전문 도구와 supervisor 조립을 다룬다. 이 분류는 학습 구분이며 모든 Advanced RAG가 hybrid, 모든 Agentic RAG가 StateGraph/복수 agent라는 표준 정의는 아니다.

숫자500/50/k3·회사 정책/워크숍 성능·데이터셋과 sample_data 경로는 검증된 운영 설정이 아니다. 설치 판본은 파이프라인 안내와 정리 기록을 따른다. 승인된 데이터 디렉터리만 읽고, 기존 DB를 자동 삭제하지 않는다. 표·코드 fence의 의미 보존/검색 품질과 회사 접속·모델 권한은 별도 평가한다. 다른 시리즈와 완전 중복 통합/구조 재설계는 Claude 연결 실패로 보류했다.
