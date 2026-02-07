---
tags: [opensearch, memory, conversation, rag, embedding, bge-m3]
level: intermediate
last_updated: 2026-02-08
status: in-progress
---

# OpenSearch 대화 메모리 구현 (Conversation Memory with OpenSearch)

> OpenSearch의 k-NN 벡터 검색과 키워드 검색을 활용하여 LLM 대화 메모리 3계층(단기/중기/장기)을 구현하는 방법

## 왜 필요한가? (Why)

- LLM 대화 메모리의 이론적 구조는 [LLM 대화 메모리 시스템](../llm-conversation-memory.md)에 정리됨
- 실제 구현 시 **저장소 선택**이 핵심 — OpenSearch는 벡터 검색(k-NN) + 키워드 검색(BM25)을 단일 엔진에서 지원
- Milvus 같은 전용 벡터 DB 대비, 기존 OpenSearch 인프라를 재활용할 수 있어 운영 부담 감소
- 로컬 LLM(Qwen3, Kimi2)과 로컬 임베딩(BGE-M3) 환경에서 외부 API 의존 없이 구축 가능

---

## 핵심 개념 (What)

### 3계층 인덱스 설계

OpenSearch에서 메모리 계층별로 별도 인덱스를 생성한다:

| 인덱스명 | 계층 | 목적 | 주요 검색 방식 |
|----------|------|------|---------------|
| `chat-messages` | 단기 | 원본 메시지 저장 | 필터(user_id+session_id) + 정렬(timestamp) |
| `chat-sessions` | 중기 | 세션별 요약 | 필터(user_id) + 정렬(end_time) |
| `user-long-memory` | 장기 | 사용자 팩트 | 벡터 검색(k-NN) + 필터(user_id) |

### chat-messages 인덱스 (단기 메모리)

```json
{
  "settings": { "index.knn": true },
  "mappings": {
    "properties": {
      "user_id":    { "type": "keyword" },
      "session_id": { "type": "keyword" },
      "role":       { "type": "keyword" },
      "content":    { "type": "text", "analyzer": "standard" },
      "embedding":  {
        "type": "knn_vector",
        "dimension": 1024,
        "method": {
          "name": "hnsw",
          "space_type": "cosinesimil",
          "engine": "nmslib"
        }
      },
      "timestamp": { "type": "date" }
    }
  }
}
```

- **용도**: 현재 세션의 최근 N개 메시지를 가져와 컨텍스트 윈도우에 직접 포함
- **조회 패턴**: `user_id` + `session_id` 필터 → `timestamp` 정렬 → 최근 20개
- **TTL 관리**: 세션 종료 후 요약이 생성되면 오래된 메시지는 `delete_by_query`로 삭제 가능

### chat-sessions 인덱스 (중기 메모리)

```json
{
  "settings": { "index.knn": true },
  "mappings": {
    "properties": {
      "user_id":       { "type": "keyword" },
      "session_id":    { "type": "keyword" },
      "summary":       { "type": "text" },
      "topics":        { "type": "keyword" },
      "message_count": { "type": "integer" },
      "embedding":     { "type": "knn_vector", "dimension": 1024, ... },
      "start_time":    { "type": "date" },
      "end_time":      { "type": "date" }
    }
  }
}
```

- **용도**: 최근 N개 세션의 요약을 시스템 프롬프트에 주입
- **조회 패턴**: `user_id` 필터 → `end_time` 역순 정렬 → 최근 3개
- **벡터 활용**: 현재 쿼리와 관련된 과거 세션을 벡터 유사도로 검색 가능

### user-long-memory 인덱스 (장기 메모리)

```json
{
  "settings": { "index.knn": true },
  "mappings": {
    "properties": {
      "user_id":       { "type": "keyword" },
      "fact":          { "type": "text" },
      "category":      { "type": "keyword" },
      "importance":    { "type": "float" },
      "embedding":     { "type": "knn_vector", "dimension": 1024, ... },
      "created_at":    { "type": "date" },
      "last_accessed": { "type": "date" }
    }
  }
}
```

- **용도**: 사용자 선호/목표/기술/패턴을 벡터 검색으로 조회
- **조회 패턴**: `user_id` 필터 + k-NN 벡터 검색 → 현재 쿼리와 관련된 팩트 top-K
- **카테고리**: preference, goal, skill, pattern

### BGE-M3 임베딩 연동

BGE-M3(1024차원)는 OpenAI-compatible API로 로컬에서 서빙:

```python
import httpx

resp = httpx.post(
    "http://localhost:8000/v1/embeddings",
    json={"model": "bge-m3", "input": ["텍스트"]},
)
vector = resp.json()["data"][0]["embedding"]  # 1024-dim float list
```

- **다국어 지원**: 한국어/영어 혼용 대화에서도 안정적인 임베딩 품질
- **배치 처리**: 여러 텍스트를 한 번에 임베딩하여 API 호출 횟수 최소화

### 로컬 LLM 요약/추출 패턴

Qwen3 또는 Kimi2를 `/v1/chat/completions` 엔드포인트로 호출:

```python
import httpx

resp = httpx.post(
    "http://localhost:8001/v1/chat/completions",
    json={
        "model": "qwen3",
        "messages": [
            {"role": "system", "content": "요약 전문가입니다..."},
            {"role": "user", "content": "기존 요약: ...\n새 대화: ..."},
        ],
        "temperature": 0.3,
    },
)
summary = resp.json()["choices"][0]["message"]["content"]
```

---

## 어떻게 사용하는가? (How)

### 전체 데이터 흐름

```
[사용자 메시지]
    │
    ▼
┌──────────────────────────────────────────┐
│  1. embed_text(content) → BGE-M3        │
│  2. index_message → chat-messages 저장   │
└──────────────────────────────────────────┘
    │
    │ (세션 종료 시)
    ▼
┌──────────────────────────────────────────┐
│  3. summarize_messages → Qwen3 요약      │
│  4. extract_topics → 토픽 추출           │
│  5. index_session → chat-sessions 저장   │
│  6. extract_facts → Qwen3 팩트 추출      │
│  7. index_fact → user-long-memory 저장   │
└──────────────────────────────────────────┘
    │
    │ (새 쿼리 시)
    ▼
┌──────────────────────────────────────────┐
│  8. get_recent_messages → 단기 메시지     │
│  9. get_recent_sessions → 중기 요약      │
│ 10. search_facts_by_vector → 장기 팩트   │
│ 11. format_system_prompt → 프롬프트 조립  │
└──────────────────────────────────────────┘
```

### OpenSearch k-NN 벡터 검색 쿼리

장기 메모리에서 관련 팩트를 검색하는 쿼리:

```python
body = {
    "size": 5,
    "query": {
        "bool": {
            "filter": [{"term": {"user_id": "user-001"}}],
            "must": [
                {
                    "knn": {
                        "embedding": {
                            "vector": query_vector,  # 현재 쿼리 임베딩
                            "k": 5,
                        }
                    }
                }
            ],
        }
    },
}
```

- `filter`로 `user_id`를 먼저 좁히고, `must`의 `knn`으로 벡터 유사도 검색
- 이 패턴은 [벡터 검색 (k-NN)](./vector-search-knn.md)에서 자세히 다룸

### 하이브리드 검색 활용 (선택적)

장기 메모리 검색에서 벡터 + 키워드를 결합하면 정확도 향상:

```python
body = {
    "size": 5,
    "query": {
        "bool": {
            "filter": [{"term": {"user_id": "user-001"}}],
            "should": [
                {"knn": {"embedding": {"vector": query_vector, "k": 5}}},
                {"match": {"fact": "RAG 시스템"}},
            ],
        }
    },
}
```

- 자세한 하이브리드 검색 방법은 [하이브리드 검색](./hybrid-search.md) 참고

### 컨텍스트 조립 예시

```python
from memory_manager import MemoryManager

mm = MemoryManager()

# 쿼리 시 3계층 메모리를 조합하여 시스템 프롬프트 구성
profile = mm.build_context("user-001", "session-002", "벡터 검색 방법은?")
system_prompt = mm.format_system_prompt(profile)

# 결과:
# 당신은 도움이 되는 AI 어시스턴트입니다.
#
# [장기 메모리 - 사용자 정보]
# - [goal] RAG 시스템을 FastAPI로 구축 중
# - [skill] Python 주력, TypeScript 보조
# - [preference] OpenSearch를 벡터 DB로 사용
#
# [이전 세션 요약]
# - FastAPI + OpenSearch로 RAG 시스템을 개발하며, BGE-M3 임베딩과 ...
```

---

## 설계 결정 사항

| 결정 | 선택 | 이유 |
|------|------|------|
| 인덱스 분리 vs. 단일 인덱스 | 3개 분리 | 계층별 조회 패턴이 다르고, k-NN 인덱스 크기 최적화 |
| 임베딩 모델 | BGE-M3 (1024d) | 다국어 지원, 로컬 서빙 가능, 충분한 품질 |
| 요약 LLM | Qwen3 | 로컬 서빙, 한국어 품질 양호, 비용 무료 |
| 벡터 엔진 | nmslib (HNSW) | OpenSearch 기본 지원, 검색 속도 우수 |
| space_type | cosinesimil | 임베딩 유사도 측정에 코사인 유사도가 표준 |

---

## 참고 자료 (References)

- [OpenSearch k-NN Plugin](https://opensearch.org/docs/latest/search-plugins/knn/index/)
- [BGE-M3 (BAAI)](https://huggingface.co/BAAI/bge-m3) - 다국어 임베딩 모델
- [Qwen3](https://github.com/QwenLM/Qwen) - 로컬 LLM
- [opensearch-py](https://github.com/opensearch-project/opensearch-py) - Python 클라이언트

## 관련 문서

- [LLM 대화 메모리 시스템 (이론)](../llm-conversation-memory.md)
- [OpenSearch 벡터 검색 (k-NN)](./vector-search-knn.md)
- [OpenSearch 하이브리드 검색](./hybrid-search.md)
- [OpenSearch Python 클라이언트](./python-client.md)
- [실습 코드: history-opensearch](../../../Codes/python/history-opensearch/)
