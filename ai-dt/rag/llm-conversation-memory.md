---
tags: [memory, conversation, rag, summarization, user-profiling]
level: intermediate
last_updated: 2026-02-03
status: in-progress
---

# LLM 대화 메모리 시스템 (Conversation Memory)

> 장기간 대화에서 사용자 정보를 축적하고 활용하여 개인화된 서비스를 제공하는 메커니즘

## 왜 필요한가? (Why)

- LLM의 컨텍스트 윈도우는 유한하므로, 긴 대화나 다중 세션에서 이전 맥락이 유실됨
- 사용자의 선호도, 목표, 이전 결정 등을 기억하면 서비스 품질이 크게 향상됨
- 반복적인 설명 없이 연속적인 대화 경험을 제공할 수 있음

---

## 핵심 개념 (What)

### 메모리 3계층 구조

| 계층 | 범위 | 방법 | 예시 |
|------|------|------|------|
| **단기 메모리(Short-term)** | 현재 세션 | 컨텍스트 윈도우에 메시지 직접 포함 | 최근 대화 내용 |
| **중기 메모리(Mid-term)** | 최근 세션들 | 대화 요약(Summarization) | 지난 3일간 대화 요약 |
| **장기 메모리(Long-term)** | 전체 기간 | 사용자 팩트 추출 → DB 저장 | "Python 선호", "RAG 시스템 개발 중" |

### 대화 요약 메커니즘 (Conversation Summarization)

#### 1. 재귀적 요약 (Recursive Summarization)

가장 일반적인 방식으로, 이전 요약에 새 메시지를 합쳐 점진적으로 요약을 갱신한다:

```
[메시지 1-20] → LLM 요약 A
[요약 A + 메시지 21-40] → LLM 요약 B
[요약 B + 메시지 41-60] → LLM 요약 C
```

#### 2. 슬라이딩 윈도우 + 요약

최근 N개 메시지는 원문 유지, 그 이전은 요약으로 압축:

```
[요약된 과거] + [최근 10개 메시지 원문] + [현재 질문]
```

#### 3. 계층적 요약 (Hierarchical Summarization)

세션별 요약 → 세션 간 요약으로 정보 손실을 방지:

```
세션 1 요약 ─┐
세션 2 요약 ─┼─→ 주간 요약 ─┐
세션 3 요약 ─┘              ├─→ 월간 요약
세션 4 요약 ─┐              │
세션 5 요약 ─┼─→ 주간 요약 ─┘
세션 6 요약 ─┘
```

#### 요약 프롬프트 패턴

```python
SUMMARIZE_PROMPT = """
기존 요약과 새 대화를 바탕으로 업데이트된 요약을 생성하세요:
1. 사용자 선호와 결정사항 보존
2. 미해결 질문/작업 유지
3. 잡담 및 중복 교환 제거
4. 구체적 팩트(이름, 숫자, 날짜) 유지

기존 요약: {previous_summary}
새 메시지: {recent_messages}
"""
```

### 사용자 정보 추출 (Meaningful Information Extraction)

#### 구조화된 팩트 추출 (Structured Extraction)

대화 후 LLM 호출로 구조화된 팩트를 추출:

```python
EXTRACT_PROMPT = """
이 대화에서 사용자 팩트를 추출하세요:
- preferences: (예: "Python을 Java보다 선호")
- personal_info: (예: "X 회사 근무")
- goals: (예: "RAG 시스템 구축 중")
- pain_points: (예: "비동기 프로그래밍에 어려움")
- decisions: (예: "PostgreSQL 선택")

규칙:
- 명시적으로 언급되거나 강하게 암시된 것만 추출
- 일시적/세션 한정 정보는 제외
- 새 팩트가 기존 팩트와 충돌하면 새 것이 우선

대화: {messages}
"""
```

#### 메모리 중요도 점수 (Memory Importance Scoring)

Stanford/Google의 "Generative Agents" 논문(2023)에서 영감을 받은 방식:

```
importance = recency × relevance × significance
```

- **Recency(최신성)**: 시간에 따른 지수적 감쇠
- **Relevance(관련성)**: 메모리 임베딩과 현재 쿼리 간 코사인 유사도
- **Significance(중요도)**: 추출 시 LLM이 평가한 중요도 (1-10)

임계값 이상의 메모리만 프롬프트에 주입한다.

---

## 어떻게 사용하는가? (How)

### 전체 아키텍처

```
사용자 메시지
    │
    ├─→ 관련 장기 메모리 검색 (벡터 검색)
    ├─→ 최근 세션 중기 요약 로드
    ├─→ 단기 메시지 (최근 N개) 포함
    │
    ▼
┌─────────────────────────────┐
│  시스템 프롬프트              │
│  + 검색된 메모리             │
│  + 세션 요약                 │
│  + 최근 메시지               │
│  + 현재 사용자 메시지         │
└─────────────────────────────┘
    │
    ▼
  LLM 응답
    │
    ├─→ (비동기) 새 사용자 팩트 추출 → 저장
    └─→ (비동기) 필요 시 세션 요약 갱신
```

### LangGraph 기반 구현 예시

```python
from langgraph.graph import StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage
import json

# 1. State 정의
class ChatState(MessagesState):
    summary: str           # 중기 요약
    user_facts: list[str]  # 장기 메모리 팩트

# 2. 요약 노드
def should_summarize(state: ChatState) -> bool:
    """메시지가 10개 이상이면 요약 트리거"""
    return len(state["messages"]) > 10

def summarize_conversation(state: ChatState):
    """대화를 요약하고 오래된 메시지를 제거"""
    messages = state["messages"]
    existing_summary = state.get("summary", "")

    summary_prompt = f"""
    기존 요약: {existing_summary}
    최근 대화: {messages[:-5]}  # 최근 5개 제외

    위 내용을 종합하여 핵심 정보를 보존한 요약을 작성하세요.
    """

    new_summary = llm.invoke([HumanMessage(content=summary_prompt)])

    return {
        "summary": new_summary.content,
        "messages": messages[-5:],  # 최근 5개만 유지
    }

# 3. 팩트 추출 노드
def extract_user_facts(state: ChatState):
    """대화에서 사용자 팩트를 추출"""
    messages = state["messages"]
    existing_facts = state.get("user_facts", [])

    extract_prompt = f"""
    기존 팩트: {json.dumps(existing_facts, ensure_ascii=False)}
    최근 대화: {messages[-3:]}

    새로 알게 된 사용자 팩트를 추출하세요.
    기존 팩트와 충돌하면 새 것으로 교체하세요.
    JSON 리스트로 반환하세요.
    """

    result = llm.invoke([HumanMessage(content=extract_prompt)])
    new_facts = json.loads(result.content)

    return {"user_facts": new_facts}

# 4. 챗 노드 (메모리 주입)
def chat_with_memory(state: ChatState):
    summary = state.get("summary", "")
    facts = state.get("user_facts", [])

    system_content = "당신은 도움이 되는 AI 어시스턴트입니다.\n"
    if facts:
        system_content += f"\n사용자에 대해 알고 있는 정보:\n"
        for fact in facts:
            system_content += f"- {fact}\n"
    if summary:
        system_content += f"\n이전 대화 요약:\n{summary}\n"

    messages = [SystemMessage(content=system_content)] + state["messages"]
    response = llm.invoke(messages)

    return {"messages": [response]}

# 5. 그래프 구성
graph = StateGraph(ChatState)
graph.add_node("chat", chat_with_memory)
graph.add_node("summarize", summarize_conversation)
graph.add_node("extract_facts", extract_user_facts)

graph.set_entry_point("chat")
graph.add_conditional_edges("chat", should_summarize, {
    True: "summarize",
    False: "extract_facts",
})
graph.add_edge("summarize", "extract_facts")
graph.add_edge("extract_facts", "__end__")

# 체크포인터로 세션 간 상태 유지
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# 사용
config = {"configurable": {"thread_id": "user-123"}}
response = app.invoke(
    {"messages": [HumanMessage(content="안녕, 나는 FastAPI로 RAG 시스템 만들고 있어")]},
    config=config,
)
```

### 벡터 DB를 활용한 장기 메모리 저장

```python
from pymilvus import MilvusClient
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
client = MilvusClient(uri="http://localhost:19530")

# 팩트 저장
def store_memory(user_id: str, fact: str, importance: float):
    vector = embeddings.embed_query(fact)
    client.insert(
        collection_name="user_memories",
        data={
            "user_id": user_id,
            "fact": fact,
            "vector": vector,
            "importance": importance,
            "timestamp": datetime.now().isoformat(),
        }
    )

# 관련 메모리 검색
def retrieve_memories(user_id: str, query: str, top_k: int = 5):
    query_vector = embeddings.embed_query(query)
    results = client.search(
        collection_name="user_memories",
        data=[query_vector],
        filter=f'user_id == "{user_id}"',
        limit=top_k,
        output_fields=["fact", "importance", "timestamp"],
    )
    return results
```

### 실무 프레임워크 비교

| 프레임워크 | 특징 | 적합한 경우 |
|-----------|------|------------|
| **Mem0** | 메모리 추출/저장/검색 자동화, 오픈소스 | 빠른 프로토타이핑 |
| **LangGraph + Checkpointer** | 그래프 기반 상태 관리, 커스터마이징 용이 | 복잡한 워크플로우 |
| **Zep** | 전용 메모리 서버, 요약+팩트 추출 내장 | 멀티유저 프로덕션 |

---

## 설계 시 핵심 결정 사항

| 결정 | 트레이드오프 |
|------|-------------|
| N개 메시지마다 요약 vs. 세션마다 요약 | 세밀함 vs. 비용 |
| 벡터 DB vs. 구조화 DB | 유연한 검색 vs. 정확한 쿼리 |
| 동기 팩트 추출 vs. 비동기 | 지연시간 vs. 즉시 반영 |
| 사용자 편집 가능 메모리 vs. 자동만 | 신뢰/통제 vs. 단순함 |

## 주의사항 (Common Pitfalls)

- **과잉 추출(Over-extraction)**: 사소한 팩트까지 저장하면 검색 품질 저하
- **모순 처리(Contradiction)**: 선호가 바뀌면 이전 팩트를 무효화해야 함
- **프라이버시**: 사용자가 저장된 메모리를 조회/삭제할 수 있어야 함
- **요약 드리프트(Summary Drift)**: 반복 요약 시 디테일 유실 → 계층적 요약으로 완화

---

## 참고 자료 (References)

- [Generative Agents (Stanford/Google, 2023)](https://arxiv.org/abs/2304.03442) - 메모리 중요도 점수 기반 에이전트
- [Mem0 GitHub](https://github.com/mem0ai/mem0) - LLM용 메모리 레이어 오픈소스
- [Zep](https://github.com/getzep/zep) - LLM 메모리 서버
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/) - 상태 관리 및 체크포인팅

## 관련 문서

- [LangGraph 기본 개념](./langgraph/)
- [Milvus 벡터 DB](./milvus/)
