# history-opensearch

> OpenSearch 기반 LLM 대화 메모리 시스템 — 단기/중기/장기 3계층 구조

## 개요

사용자의 대화 이력을 OpenSearch에 저장하고, 로컬 LLM(Qwen3)과 BGE-M3 임베딩으로 대화를 요약·분석하여 사용자 활동을 특성화하는 시스템.

## 아키텍처

```
사용자 메시지
    │
    ├─→ 임베딩 (BGE-M3) → chat-messages 인덱스 (단기)
    │
    ├─→ 세션 종료 시:
    │     ├─→ 요약 (Qwen3) → chat-sessions 인덱스 (중기)
    │     └─→ 팩트 추출 (Qwen3) → user-long-memory 인덱스 (장기)
    │
    └─→ 새 쿼리 시:
          ├─→ 단기: 현재 세션 최근 메시지
          ├─→ 중기: 최근 세션 요약 로드
          └─→ 장기: 벡터 검색으로 관련 팩트 검색
              → 시스템 프롬프트에 주입
```

## OpenSearch 인덱스

| 인덱스 | 계층 | 주요 필드 |
|--------|------|-----------|
| `chat-messages` | 단기 | user_id, session_id, role, content, embedding, timestamp |
| `chat-sessions` | 중기 | user_id, session_id, summary, topics, embedding, start/end_time |
| `user-long-memory` | 장기 | user_id, fact, category, importance, embedding, created_at |

## 파일 구조

```
├── config.py           # OpenSearch/LLM/임베딩 설정
├── models.py           # Pydantic 모델 (Message, Session, UserFact, UserProfile)
├── os_client.py        # OpenSearch 클라이언트, 인덱스 생성/CRUD
├── embedding.py        # BGE-M3 임베딩 (OpenAI-compatible API)
├── summarizer.py       # 대화 요약 (재귀적/계층적)
├── fact_extractor.py   # 사용자 팩트 추출 (장기 메모리)
├── memory_manager.py   # 3계층 메모리 오케스트레이션
├── main.py             # 데모 시나리오 실행
└── requirements.txt    # 의존성
```

## 설치

```bash
pip install -r requirements.txt
```

## 사전 준비

1. **OpenSearch** 실행 (기본: `https://localhost:9200`)
2. **임베딩 서버** 실행 — BGE-M3, OpenAI-compatible API (`http://localhost:8000/v1`)
3. **LLM 서버** 실행 — Qwen3 또는 Kimi2, OpenAI-compatible API (`http://localhost:8001/v1`)

`config.py`에서 엔드포인트와 모델명을 환경에 맞게 수정.

## 사용법

```bash
# 데모 시나리오 실행
python main.py
```

### 코드에서 직접 사용

```python
from memory_manager import MemoryManager

mm = MemoryManager()

# 1) 메시지 저장
mm.add_message("user-001", "session-001", "user", "FastAPI로 RAG 만들고 있어요")
mm.add_message("user-001", "session-001", "assistant", "좋은 프로젝트네요!")

# 2) 세션 종료 → 요약 + 팩트 추출
session = mm.finalize_session("user-001", "session-001")

# 3) 새 세션에서 컨텍스트 조립
profile = mm.build_context("user-001", "session-002", "벡터 검색 방법은?")
system_prompt = mm.format_system_prompt(profile)
```

## 관련 문서

- [LLM 대화 메모리 시스템 (이론)](../../../ai-dt/rag/llm-conversation-memory.md)
- [OpenSearch 대화 메모리 구현 (학습 노트)](../../../ai-dt/rag/opensearch/conversation-memory-opensearch.md)
- [OpenSearch 벡터 검색](../../../ai-dt/rag/opensearch/vector-search-knn.md)
- [OpenSearch 하이브리드 검색](../../../ai-dt/rag/opensearch/hybrid-search.md)
