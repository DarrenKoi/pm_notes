# RAG 확장 기법

> HyDE(가상 답변 기반 검색), MemorySaver(대화 맥락 유지), Naive vs Agentic 비교를 실험하고 성능 차이를 이해한다

---
tags: [rag, hyde, memory, memorysaver, naive-rag, agentic-rag, comparison]
level: advanced
last_updated: 2026-07-16
---

## 왜 필요한가? (Why)

기본 Agentic RAG(retrieve → grade → generate/rewrite)만으로도 Naive RAG보다 우수하지만, 실무에서는 추가적인 한계에 부딪힌다:

| 한계 | 원인 | 해결 기법 |
|------|------|----------|
| 쿼리-문서 임베딩 불일치 | 질문과 문서의 표현 방식 차이 | **HyDE** (가상 답변 기반 검색) |
| 대화 맥락 소실 | 매 질문마다 독립 실행 | **MemorySaver** (체크포인터) |
| 성능 평가 기준 부재 | 개선 효과 정량화 어려움 | **Naive vs Agentic 비교** |

## HyDE (Hypothetical Document Embedding)

### 핵심 개념

**문제:** 사용자의 질문("Particle 불량 조치는?")과 문서 내용("Particle 수 > 기준치 × 1.5배 시 엔지니어 확인")은 **표현 방식이 다르다.** 질문은 짧고 추상적이지만, 문서는 길고 구체적이다. 이 **임베딩 공간의 밀도 차이**가 검색 실패를 유발한다.

**해결:** LLM에게 질문에 대한 **가상 답변(Hypothetical Document)** 을 먼저 생성시키고, 그 가상 답변을 검색 쿼리로 사용한다. 가상 답변은 실제 문서와 표현 방식이 유사하므로 임베딩 공간에서 더 가까이 위치한다.

```
[기존]  질문 → (임베딩) → 벡터 검색 → 문서
[HyDE]  질문 → LLM → 가상 답변 → (임베딩) → 벡터 검색 → 문서
```

### 구현

#### hyde_node 함수

```python
def hyde_node(state: RAGState) -> dict:
    """질문에 대한 가상 답변을 생성하여 검색 쿼리로 사용한다 (HyDE)."""
    question = state["question"]
    prompt = (
        f"다음 질문에 대해 PM 문서에서 찾을 법한 내용으로 가상 답변을 작성하세요.\n"
        f"(실제 답이 아닌 검색용 가상 답변입니다)\n\n"
        f"질문: {question}\n\n가상 답변:"
    )
    result = llm.invoke(prompt)
    hyde_query = result.content.strip()
    print(f"  [HyDE] 가상 답변 생성 완료 ({len(hyde_query)} chars)")
    return {"question": hyde_query}
```

**도메인별 프롬프트 조정:**

```python
# 반도체 공정 도메인
prompt = (
    f"다음 반도체 공정 질문에 대해 공정 매뉴얼에서 찾을 법한 내용으로 "
    f"가상 답변을 작성하세요.\n"
    f"(실제 답이 아닌 검색용 가상 답변입니다)\n\n"
    f"질문: {question}\n\n가상 답변:"
)
```

"PM 문서에서 찾을 법한 내용" / "공정 매뉴얼에서 찾을 법한 내용"이라는 지시가 핵심이다. 이 지시 없이 일반 답변을 생성하면 문서 스타일과 불일치하여 HyDE의 효과가 감소한다.

#### HyDE 포함 그래프 조립

```python
from langgraph.graph import StateGraph, START, END

workflow_hyde = StateGraph(RAGState)

# 노드 등록 — hyde 노드 추가
workflow_hyde.add_node("hyde",            hyde_node)
workflow_hyde.add_node("retrieve",        retrieve_node)
workflow_hyde.add_node("grade_documents", grade_documents_node)
workflow_hyde.add_node("generate",        generate_node)
workflow_hyde.add_node("rewrite_query",   rewrite_node)

# 엣지 — START → hyde → retrieve로 변경
workflow_hyde.add_edge(START,              "hyde")
workflow_hyde.add_edge("hyde",            "retrieve")
workflow_hyde.add_edge("retrieve",        "grade_documents")
workflow_hyde.add_conditional_edges(
    "grade_documents", route_question,
    {"generate": "generate", "rewrite": "rewrite_query"},
)
workflow_hyde.add_edge("rewrite_query",   "retrieve")
workflow_hyde.add_edge("generate",        END)

rag_agent_hyde = workflow_hyde.compile()
```

**그래프 흐름 변화:**

```
[기존]  START → retrieve → grade → generate/rewrite
[HyDE]  START → hyde → retrieve → grade → generate/rewrite
```

기존 그래프에서 `START → retrieve` 엣지를 `START → hyde → retrieve`로 변경한 것뿐이다. 나머지 노드와 엣지는 모두 동일하다.

### 비교 실험

```python
test_q = "Particle 불량 발생 시 조치 방법은?"

# 기본 Agentic RAG
r_base = rag_agent.invoke({
    "messages": [], "question": test_q, "documents": [], "generation": ""
})

# HyDE Agentic RAG
r_hyde = rag_agent_hyde.invoke({
    "messages": [], "question": test_q, "documents": [], "generation": ""
})

print("=== 기본 ===")
print(f"검색 문서: {[os.path.basename(d.metadata.get('source', '')) for d in r_base['documents']]}")

print("\n=== HyDE ===")
print(f"검색 문서: {[os.path.basename(d.metadata.get('source', '')) for d in r_hyde['documents']]}")
```

### 실험 결과 분석

워크숍 실험에서 관찰된 차이:

| 항목 | 기본 Agentic RAG | HyDE Agentic RAG |
|------|-----------------|-------------------|
| 검색 문서 (Particle 질문) | 증착공정_트러블슈팅.md ×2, Particle_불량_조치_가이드.md | 증착공정_트러블슈팅.md, Particle_불량_조치_가이드.md ×2 |
| 가상 답변 길이 | - | ~656 chars |
| LLM 호출 횟수 | 검색 3 + grade 3 + generate 1 = 7 | **HyDE 1** + 검색 3 + grade 3 + generate 1 = 8 |

**핵심 관찰:**
- HyDE는 Particle_불량_조치_가이드.md를 **2개 청크** 검색하여 더 많은 근거를 확보했다
- 대신 LLM 호출이 1회 추가되어 **비용과 지연 시간이 증가**한다
- 이미 직접 매칭되는 질문에서는 효과가 미미하다 — **간접적·추상적 질문**에서 효과가 크다

### HyDE 적용 판단 기준

| 상황 | HyDE 적용 | 이유 |
|------|----------|------|
| 사용자 질문이 짧고 추상적 | 적용 권장 | 임베딩 불일치 해소 |
| 질문이 문서 용어와 직접 매칭 | 불필요 | 추가 비용만 발생 |
| 검색 실패(rewrite)가 빈번 | 적용 권장 | rewrite 전에 품질 향상 |
| 비용/지연 시간이 중요 | 비적용 | LLM 1회 추가 호출 |

### HyDE의 한계

- **환각(Hallucination) 위험:** 가상 답변에 잘못된 정보가 포함되면 오히려 관련 없는 문서를 검색할 수 있다
- **원래 질문 소실:** `question` 필드가 가상 답변으로 교체되므로, generate 단계에서 원래 질문을 참조할 수 없다

원래 질문을 보존하려면 `RAGState`를 확장한다:

```python
class RAGState(TypedDict):
    messages:          Annotated[list, add_messages]
    question:          str
    original_question: str   # 원래 질문 보존
    documents:         List[Document]
    generation:        str

def hyde_node(state: RAGState) -> dict:
    question = state["question"]
    # ... 가상 답변 생성 ...
    return {
        "question": hyde_query,
        "original_question": question,  # 원래 질문 저장
    }

def generate_node(state: RAGState) -> dict:
    question = state.get("original_question", state["question"])  # 원래 질문 사용
    # ... 답변 생성 ...
```

---

## MemorySaver (대화 맥락 유지)

### 핵심 개념

기본 Agentic RAG는 **매 질문이 독립적**이다. "Particle 불량 조치를 설명해줘" → "방금 설명한 내용에서 가장 중요한 단계는?" 같은 **후속 질문**에 대응할 수 없다.

`MemorySaver`는 LangGraph의 체크포인터(Checkpointer)로, `thread_id`별로 **그래프 상태 스냅샷**을 저장한다. 같은 `thread_id`로 호출하면 이전 상태(messages, documents, generation)가 복원되어 대화 맥락이 유지된다.

```
[스레드 1] Q1: "Particle 불량 조치?" → A1 (messages에 저장)
                                          ↓ (상태 스냅샷 저장)
[스레드 1] Q2: "가장 중요한 단계는?" → A2 (이전 messages 참조 가능)
```

### 구현

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
rag_agent_memory = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "session-1"}}
```

**핵심:** `workflow.compile()` 호출 시 `checkpointer=memory`를 전달한다. 기존 `workflow` 객체를 **그대로 재사용**하므로 그래프 구조 변경이 불필요하다.

#### 멀티턴 테스트

```python
# 1번째 질문
q1 = "Particle 불량 발생 시 조치 방법은?"
r1 = rag_agent_memory.invoke(
    {"messages": [], "question": q1, "documents": [], "generation": ""},
    config=config,
)
print(f"[Q1] {q1}")
print(f"[A1] {r1.get('generation', '')[:300]}\n")

# 후속 질문 — 대화 맥락 활용
q2 = "방금 설명한 조치에서 웨이퍼 검사 단계는 어떻게 진행하나요?"
r2 = rag_agent_memory.invoke(
    {"messages": [], "question": q2, "documents": [], "generation": ""},
    config=config,
)
print(f"[Q2] {q2}")
print(f"[A2] {r2.get('generation', '')[:300]}")
```

`config`의 `thread_id`가 동일하면 이전 대화가 이어진다. 다른 `thread_id`를 사용하면 새로운 세션이 시작된다.

### MemorySaver 동작 원리

```
invoke(state, config={"configurable": {"thread_id": "session-1"}})
  ↓
1. thread_id="session-1"의 스냅샷이 있는지 확인
2. 있으면 → 기존 state를 복원하고 새 입력을 merge
3. 없으면 → 초기 state로 시작
4. 그래프 실행 완료 후 → state 스냅샷 저장
```

| 파라미터 | 설명 |
|---------|------|
| `thread_id` | 세션 식별자 — 같은 값이면 대화 이어짐 |
| `MemorySaver` | 인메모리 저장 — 프로세스 종료 시 소멸 |

프로덕션에서 영속적 대화 이력이 필요하면 `SqliteSaver` 또는 `PostgresSaver`를 사용한다:

```python
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string("sqlite:///rag_memory.db")
rag_agent_memory = workflow.compile(checkpointer=memory)
```

### MemorySaver와 add_messages 리듀서의 관계

`RAGState`의 `messages` 필드에 `add_messages` 리듀서가 정의되어 있어야 한다:

```python
class RAGState(TypedDict):
    messages: Annotated[list, add_messages]  # 이 리듀서가 핵심
```

`add_messages` 리듀서가 없으면 새 호출 시 `messages`가 빈 리스트로 덮어씌워져, 이전 대화 내용이 복원되더라도 의미가 없다. 리듀서가 있으면 새 메시지가 기존 메시지 리스트에 **추가(append)** 된다.

---

## Naive RAG vs Agentic RAG 정량 비교

### 비교 프레임워크

동일한 질문에 대해 두 방식을 실행하고 정량 지표를 비교한다:

```python
def naive_rag(question: str) -> dict:
    """Naive RAG: 검색 후 바로 생성 (grade/rewrite 없음)."""
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    prompt = (
        f"다음 문서를 참고하여 질문에 답변하세요.\n\n"
        f"문서:\n{context}\n\n"
        f"질문: {question}"
    )
    result = llm.invoke(prompt)
    return {"answer": result.content, "docs": docs}
```

```python
test_questions = [
    "프로젝트 리스크 관리 절차를 설명해주세요",
    "스프린트 회고 미팅에서 다뤄야 할 내용은?",
    "프로젝트 품질 검수 기준은 무엇인가요?",
]

print(f"{'질문':30s} {'방법':12s} {'문서수':6s} {'답변길이':8s}")
print("-" * 70)

for q in test_questions:
    # Naive RAG
    nr = naive_rag(q)
    print(f"{q[:28]:30s} {'Naive RAG':12s} {len(nr['docs']):6d} {len(nr['answer']):8d}")

    # Agentic RAG
    ar = rag_agent.invoke({
        "messages": [], "question": q, "documents": [], "generation": ""
    })
    print(f"{'':30s} {'Agentic RAG':12s} "
          f"{len(ar.get('documents', [])):6d} {len(ar.get('generation', '')):8d}")
    print()
```

### PM 도메인 실험 결과

| 질문 | 방법 | 사용 문서 수 | 답변 길이 |
|------|------|------------|----------|
| 리스크 관리 절차 | Naive RAG | 3 | 1,074자 |
| | **Agentic RAG** | **3** | **1,494자** |
| 스프린트 회고 미팅 | Naive RAG | 3 | 111자 |
| | **Agentic RAG** | **1** | **392자** |
| 품질 검수 기준 | Naive RAG | 3 | 524자 |
| | **Agentic RAG** | **2** | **465자** |

### 반도체 공정 도메인 실험 결과

| 질문 | 방법 | 사용 문서 수 | 답변 길이 |
|------|------|------------|----------|
| Particle 불량 조치 | Naive RAG | 3 | 998자 |
| | **Agentic RAG** | **3** | **1,578자** |
| CMP Pad 교체 주기 | Naive RAG | 3 | 476자 |
| | **Agentic RAG** | **1** | **647자** |
| Etch RF Power 이상 | Naive RAG | 3 | 784자 |
| | **Agentic RAG** | **1** | **912자** |

### 결과 분석

#### 1. 문서 필터링 효과

Agentic RAG의 grade 단계가 **관련 없는 문서를 제거**한다:
- CMP 질문: 3개 검색 → **1개만 관련** → 관련 없는 2개 제거
- 스프린트 회고 질문: 3개 검색 → **1개만 관련** → 관련 없는 2개 제거

Naive RAG는 3개 문서를 모두 컨텍스트에 넣으므로, 관련 없는 문서가 **노이즈**로 작용하여 답변 품질이 저하된다.

#### 2. 답변 상세도

Agentic RAG는 관련 문서만 사용하면서도 답변이 더 길고 상세하다. 이는 LLM이 **노이즈 없이 관련 정보에 집중**할 수 있기 때문이다.

- Naive: "스프린트 회고 미팅에서는 프로세스 개선점을 도출합니다." (111자)
- Agentic: 프로세스 개선점 도출 + 팀 작업 방식 평가 + 개선 방안 논의 (392자)

#### 3. rewrite 효과

"스프린트 회고 미팅"이라는 짧은 질문은 grade 단계에서 모두 `no`를 받아 rewrite가 트리거되었다:

```
원래: "스프린트 회고 미팅"
재작성: "애자일 스프린트 회고 미팅의 목적과 효과적인 진행 방법"
→ 재검색 → 1개 관련 문서 확보 → 정확한 답변 생성
```

Naive RAG에서는 이런 복구 메커니즘이 없어 관련 없는 3개 문서로 짧고 부정확한 답변을 생성했다.

#### 4. 비용 트레이드오프

| 항목 | Naive RAG | Agentic RAG |
|------|-----------|-------------|
| LLM 호출 | 1회 (생성만) | 4~8회 (grade + 생성 + rewrite) |
| 비용 | 낮음 | 3~5배 높음 |
| 지연 시간 | 빠름 | 2~4배 느림 |
| 답변 품질 | 불안정 | 안정적으로 높음 |

비용이 중요한 저빈도 질의에는 Naive RAG가, 품질이 중요한 고빈도 운영 질의에는 Agentic RAG가 적합하다.

---

## 확장 기법 조합 가이드

| 기법 | 단독 사용 | 조합 추천 | 주의사항 |
|------|----------|----------|---------|
| HyDE | 간접 질문 많은 도메인 | Agentic + HyDE | LLM 1회 추가 비용 |
| MemorySaver | 멀티턴 대화 | Agentic + Memory | thread_id 관리 필요 |
| HyDE + Memory | 대화형 문서 검색 | Agentic + HyDE + Memory | 복잡도 증가 |

### 전체 확장 그래프 (HyDE + Memory)

```python
workflow_full = StateGraph(RAGState)
workflow_full.add_node("hyde",            hyde_node)
workflow_full.add_node("retrieve",        retrieve_node)
workflow_full.add_node("grade_documents", grade_documents_node)
workflow_full.add_node("generate",        generate_node)
workflow_full.add_node("rewrite_query",   rewrite_node)

workflow_full.add_edge(START,              "hyde")
workflow_full.add_edge("hyde",            "retrieve")
workflow_full.add_edge("retrieve",        "grade_documents")
workflow_full.add_conditional_edges(
    "grade_documents", route_question,
    {"generate": "generate", "rewrite": "rewrite_query"},
)
workflow_full.add_edge("rewrite_query",   "retrieve")
workflow_full.add_edge("generate",        END)

memory = MemorySaver()
rag_agent_full = workflow_full.compile(checkpointer=memory)

# 사용
config = {"configurable": {"thread_id": "full-session-1"}}
result = rag_agent_full.invoke(
    {"messages": [], "question": "Particle 불량 조치는?", "documents": [], "generation": ""},
    config=config,
)
```

## 관련 문서

- [Agentic RAG 구현](./agentic-rag-implementation.md) — 기본 Agentic RAG 구현 (이 문서의 사전 단계)
- [멀티에이전트 RAG 통합](./multi-agent-rag-integration.md) — Supervisor 패턴 통합
- [LangGraph 고급 패턴](../langgraph/langgraph-advanced.md) — Persistence, Streaming 등

## 참고 자료 (References)

- [HyDE 논문 — Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496)
- [LangGraph Checkpointer 공식 문서](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- [LangGraph MemorySaver API](https://langchain-ai.github.io/langgraph/reference/checkpoints/)
