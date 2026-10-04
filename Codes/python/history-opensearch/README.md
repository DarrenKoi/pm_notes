---
title: OpenSearch 기반 대화 메모리 예제
tags: [python, opensearch, memory]
document_type: learning
reviewed_on: 2026-10-04
verification_status: source-verified-live-unverified
---

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

`config.py`의 설정 구조를 확인하고 실행 환경에 맞는 엔드포인트·모델명을 준비한다. 실제 인증값을 문서나 저장소에 추가하지 않는다.

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

## 실제 구현의 적용 조건

`MemoryManager()` 생성은 `ensure_indices()`를 호출하므로 OpenSearch 인덱스 생성 권한이 필요하다. `finalize_session()`은 최근 메시지를 최대 200건 조회해 요약·팩트를 저장하며 모든 대화의 완전한 기록을 요약한다고 보장하지 않는다. 세션 종료 감지는 호출자가 수행해야 한다.

`build_context()`는 최근 메시지·세션 요약·관련 팩트를 반환하지만 `format_system_prompt()`는 장기 팩트와 세션 요약만 넣는다. 최근 메시지를 LLM 대화 입력에 포함하는 일은 호출자가 별도로 처리해야 한다. 생성 모델이 추출한 팩트는 사용자 사실로 검증된 데이터가 아니며 원문과 대조해야 한다.

## 실행과 검증 경계

저장소 루트에서 실행한다면 먼저 `cd Codes/python/history-opensearch`로 이동한다. Python 타입 문법과 의존성을 고려해 Python 3.10 이상 환경에서 사용하며 설치된 라이브러리 버전은 별도 기록한다. 모델 ID·임베딩 차원·실제 서버 지원은 환경별로 확인한다. 예시 모델명이 있다는 사실만으로 서버가 그 모델을 제공한다는 뜻은 아니다.

이번 검토는 2026-10-04 소스의 함수 호출과 데이터 흐름을 확인했다. OpenSearch·임베딩·LLM 서버는 실행하지 않았으며 대화 저장·요약·검색의 통합 동작은 미확인이다. `config.py`의 예시 인증·TLS 설정을 공유 서버 기본값으로 사용하지 않는다.

## 참고 자료와 다음 문서

- [OpenSearch Python 클라이언트 공식 안내](https://docs.opensearch.org/latest/clients/python-low-level/) — 확인 2026-10-04. 서버와 클라이언트의 실제 버전 조합은 미확인.
- [이 모듈의 오케스트레이터](memory_manager.py), [인덱스·쿼리 구현](os_client.py).
- [Python 예제 목차](../../README.md)에서 다른 예제의 목적을 비교한다.
