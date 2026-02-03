---
tags: [memory, conversation, rag, summarization, user-profiling, mongodb, milvus, elasticsearch]
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

### 키 관계: user_id vs session_id

```
user_id (1) ──→ (N) session_id
```

- `user_id`: 사용자 식별자. **장기 메모리의 기본 키**. 모든 세션에 걸쳐 팩트를 축적/검색하는 데 사용
- `session_id`: 대화 세션 식별자. **중기 요약 및 체크포인터의 키**. 세션별 메시지 히스토리 관리에 사용

```
user_id: "user-123" (Daeyoung)
  ├── session_id: "sess-001"  (1월 30일 - FastAPI 질문)
  ├── session_id: "sess-002"  (1월 31일 - RAG 구현)
  └── session_id: "sess-003"  (2월 03일 - 메모리 시스템)
```

### 스토리지 아키텍처 (MongoDB + Milvus + ES 7.14)

사용 가능한 DB: MongoDB, Elasticsearch 7.14, Milvus (dense vector only, sparse 미설정)

#### DB별 역할 분담

| 데이터 | 저장소 | 이유 |
|--------|--------|------|
| 대화 메시지 (raw) | **MongoDB** | 문서 지향, 메시지 배열에 자연스러운 구조 |
| 세션 요약 (mid-term) | **MongoDB** | user_id + session_id로 정확한 키 기반 조회 |
| 사용자 팩트 (long-term) | **Milvus** | 시맨틱 유사도 검색으로 관련 팩트 검색 필요 |
| 대화 전문 검색 (optional) | **ES 7.14** | 전체 세션에서 키워드 기반 대화 검색 시 활용 |

> **ES 7.14 참고**: 네이티브 벡터 검색은 8.x부터 지원되므로, 7.14에서는 full-text 검색 용도로만 활용.
> Milvus는 sparse 미설정 상태이므로 dense vector 유사도 검색만 사용.

#### 스토리지 아키텍처 다이어그램

```
사용자 메시지 (+ user_id, session_id from State)
    │
    ├─→ Milvus: 관련 장기 메모리 검색 (dense vector similarity)
    ├─→ MongoDB: session_id로 중기 요약 로드
    ├─→ LangGraph State: 단기 메시지 (최근 N개)
    │
    ▼
┌─────────────────────────────┐
│  시스템 프롬프트              │
│  + Milvus 검색 메모리        │
│  + MongoDB 세션 요약         │
│  + 최근 메시지               │
│  + 현재 사용자 메시지         │
└─────────────────────────────┘
    │
    ▼
  LLM 응답
    │
    ├─→ (비동기) 팩트 추출 → Milvus 저장 (user_id 키)
    ├─→ (비동기) 세션 요약 갱신 → MongoDB 저장 (session_id 키)
    └─→ (비동기) 대화 기록 → MongoDB 저장
```

### 1단계: MongoDB 스토리지 레이어 (대화 + 요약)

```python
from pymongo import MongoClient
from datetime import datetime

mongo = MongoClient("mongodb://localhost:27017")
db = mongo["chat_memory"]

# ─── 대화 기록 저장/조회 ───

def save_conversation(user_id: str, session_id: str, messages: list, summary: str = ""):
    """세션 대화 기록을 MongoDB에 저장 (upsert)"""
    db.conversations.update_one(
        {"user_id": user_id, "session_id": session_id},
        {
            "$set": {
                "messages": [
                    {"role": m.type, "content": m.content}
                    for m in messages
                ],
                "summary": summary,
                "updated_at": datetime.now(),
            },
            "$setOnInsert": {
                "created_at": datetime.now(),
            },
        },
        upsert=True,
    )

def load_session_summary(user_id: str, session_id: str) -> str:
    """현재 세션의 중기 요약을 MongoDB에서 로드"""
    doc = db.conversations.find_one(
        {"user_id": user_id, "session_id": session_id},
        {"summary": 1},
    )
    return doc["summary"] if doc and doc.get("summary") else ""

def load_recent_session_summaries(user_id: str, limit: int = 5) -> list[dict]:
    """최근 N개 세션의 요약을 로드 (크로스 세션 메모리)"""
    docs = db.conversations.find(
        {"user_id": user_id, "summary": {"$ne": ""}},
        {"session_id": 1, "summary": 1, "updated_at": 1},
    ).sort("updated_at", -1).limit(limit)
    return list(docs)
```

#### MongoDB 스키마

```javascript
// Collection: conversations
{
    "user_id": "user-123",
    "session_id": "session-abc-123",
    "messages": [
        {"role": "human", "content": "...", "timestamp": "..."},
        {"role": "ai", "content": "...", "timestamp": "..."}
    ],
    "summary": "사용자가 FastAPI로 RAG 시스템을 구축 중...",
    "created_at": ISODate("2026-02-03T10:00:00Z"),
    "updated_at": ISODate("2026-02-03T11:30:00Z")
}

// 인덱스
db.conversations.createIndex({ "user_id": 1, "session_id": 1 }, { unique: true })
db.conversations.createIndex({ "user_id": 1, "updated_at": -1 })
```

### 2단계: Milvus 스토리지 레이어 (장기 팩트 메모리, dense only)

```python
from pymilvus import MilvusClient
from langchain_openai import OpenAIEmbeddings
from datetime import datetime

embeddings = OpenAIEmbeddings()
milvus = MilvusClient(uri="http://localhost:19530")

# Milvus 컬렉션은 dense vector만 사용 (sparse 미설정)
def store_memory(user_id: str, fact: str, importance: float):
    """사용자 팩트를 Milvus에 dense vector로 저장"""
    vector = embeddings.embed_query(fact)
    milvus.insert(
        collection_name="user_memories",
        data={
            "user_id": user_id,
            "fact": fact,
            "vector": vector,             # dense vector only
            "importance": importance,
            "timestamp": datetime.now().isoformat(),
        },
    )

def retrieve_memories(user_id: str, query: str, top_k: int = 5) -> list[dict]:
    """user_id 기반으로 Milvus에서 관련 팩트를 dense similarity로 검색"""
    query_vector = embeddings.embed_query(query)
    results = milvus.search(
        collection_name="user_memories",
        data=[query_vector],
        filter=f'user_id == "{user_id}"',
        limit=top_k,
        output_fields=["fact", "importance", "timestamp"],
    )
    return [hit.entity for hit in results[0]] if results else []
```

### 3단계: ES 7.14 (선택사항 - 대화 전문 검색)

```python
from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")

def index_conversation_to_es(user_id: str, session_id: str, messages: list):
    """대화 내용을 ES에 인덱싱 (키워드 기반 전문 검색용)"""
    full_text = "\n".join([f"{m.type}: {m.content}" for m in messages])
    es.index(
        index="chat-conversations",
        body={
            "user_id": user_id,
            "session_id": session_id,
            "content": full_text,
            "timestamp": datetime.now().isoformat(),
        },
    )

def search_past_conversations(user_id: str, keyword: str, size: int = 5):
    """과거 대화에서 키워드 검색 (ES 7.14 full-text)"""
    result = es.search(
        index="chat-conversations",
        body={
            "query": {
                "bool": {
                    "must": [{"match": {"content": keyword}}],
                    "filter": [{"term": {"user_id": user_id}}],
                }
            },
            "size": size,
        },
    )
    return [hit["_source"] for hit in result["hits"]["hits"]]
```

### 4단계: LangGraph 통합 (MongoDB + Milvus 연동)

```python
from langgraph.graph import StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage
import json

# 1. State 정의
# user_id와 session_id는 백엔드에서 State를 통해 직접 접근 가능
class ChatState(MessagesState):
    user_id: str           # 백엔드에서 주입되는 사용자 식별자 (장기 메모리 키)
    session_id: str        # 현재 세션 식별자 (중기 요약/체크포인터 키)
    summary: str           # 중기 요약 (MongoDB에 영속화)
    user_facts: list[str]  # 현재 세션에서 추출된 팩트

# 2. 요약 노드
def should_summarize(state: ChatState) -> bool:
    """메시지가 10개 이상이면 요약 트리거"""
    return len(state["messages"]) > 10

def summarize_conversation(state: ChatState):
    """대화를 요약하고 오래된 메시지를 제거, MongoDB에 저장"""
    user_id = state["user_id"]
    session_id = state["session_id"]
    messages = state["messages"]
    existing_summary = state.get("summary", "")

    summary_prompt = f"""
    기존 요약: {existing_summary}
    최근 대화: {messages[:-5]}

    위 내용을 종합하여 핵심 정보를 보존한 요약을 작성하세요.
    """

    new_summary = llm.invoke([HumanMessage(content=summary_prompt)])

    # MongoDB에 요약 영속화
    save_conversation(
        user_id=user_id,
        session_id=session_id,
        messages=messages[-5:],
        summary=new_summary.content,
    )

    return {
        "summary": new_summary.content,
        "messages": messages[-5:],
    }

# 3. 팩트 추출 노드
def extract_user_facts(state: ChatState):
    """대화에서 사용자 팩트를 추출 → Milvus에 저장"""
    user_id = state["user_id"]
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

    # Milvus에 새 팩트 저장 (user_id 키, dense vector)
    for fact in new_facts:
        if fact not in existing_facts:
            store_memory(user_id=user_id, fact=fact, importance=7.0)

    return {"user_facts": new_facts}

# 4. 챗 노드 (MongoDB 요약 + Milvus 메모리 주입)
def chat_with_memory(state: ChatState):
    user_id = state["user_id"]
    session_id = state["session_id"]

    # Milvus: 현재 쿼리와 관련된 장기 팩트 검색 (dense similarity)
    current_query = state["messages"][-1].content
    long_term_memories = retrieve_memories(
        user_id=user_id, query=current_query, top_k=5
    )

    # MongoDB: 현재 세션 요약 + 최근 다른 세션 요약 로드
    current_summary = state.get("summary", "") or load_session_summary(
        user_id=user_id, session_id=session_id
    )
    recent_sessions = load_recent_session_summaries(user_id=user_id, limit=3)

    # 시스템 프롬프트 조립
    system_content = "당신은 도움이 되는 AI 어시스턴트입니다.\n"

    if long_term_memories:
        system_content += "\n[장기 메모리 - Milvus에서 검색된 사용자 팩트]\n"
        for mem in long_term_memories:
            system_content += f"- {mem['fact']}\n"

    if recent_sessions:
        system_content += "\n[이전 세션 요약 - MongoDB]\n"
        for sess in recent_sessions:
            if sess["session_id"] != session_id:
                system_content += f"- {sess['summary']}\n"

    if current_summary:
        system_content += f"\n[현재 세션 요약]\n{current_summary}\n"

    facts = state.get("user_facts", [])
    if facts:
        system_content += "\n[현재 세션에서 파악한 정보]\n"
        for fact in facts:
            system_content += f"- {fact}\n"

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

# 체크포인터로 세션 내 상태 유지
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# 사용 - user_id와 session_id는 백엔드에서 State로 직접 주입
config = {"configurable": {"thread_id": "session-abc-123"}}
response = app.invoke(
    {
        "user_id": "user-123",
        "session_id": "session-abc-123",
        "messages": [HumanMessage(content="안녕, 나는 FastAPI로 RAG 시스템 만들고 있어")],
    },
    config=config,
)
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
| MongoDB + Milvus 분리 vs. 단일 DB | 각 DB의 강점 활용 vs. 운영 복잡도 |
| 동기 팩트 추출 vs. 비동기 | 지연시간 vs. 즉시 반영 |
| 사용자 편집 가능 메모리 vs. 자동만 | 신뢰/통제 vs. 단순함 |
| ES 전문 검색 추가 vs. 미사용 | 과거 대화 키워드 검색 가능 vs. 추가 인프라 |

## 주의사항 (Common Pitfalls)

- **과잉 추출(Over-extraction)**: 사소한 팩트까지 저장하면 Milvus 검색 품질 저하
- **모순 처리(Contradiction)**: 선호가 바뀌면 Milvus의 이전 팩트를 무효화해야 함
- **프라이버시**: 사용자가 MongoDB/Milvus에 저장된 메모리를 조회/삭제할 수 있어야 함
- **요약 드리프트(Summary Drift)**: 반복 요약 시 디테일 유실 → 계층적 요약으로 완화
- **Milvus sparse 미지원**: 현재 dense vector만 사용하므로, 키워드 기반 팩트 검색이 필요하면 ES 7.14 활용

---

## 참고 자료 (References)

- [Generative Agents (Stanford/Google, 2023)](https://arxiv.org/abs/2304.03442) - 메모리 중요도 점수 기반 에이전트
- [Mem0 GitHub](https://github.com/mem0ai/mem0) - LLM용 메모리 레이어 오픈소스
- [Zep](https://github.com/getzep/zep) - LLM 메모리 서버
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/) - 상태 관리 및 체크포인팅

## 관련 문서

- [LangGraph 기본 개념](./langgraph/)
- [Milvus 벡터 DB](./milvus/)
