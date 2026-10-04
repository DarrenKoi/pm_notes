---
tags: [milvus, rag, documentation-review]
reviewed_on: 2026-10-04
review_status: partial
document_type: index
---

# Milvus Vector DB 시리즈

> Milvus 벡터 데이터베이스의 기본 개념부터 RAG 시스템 연동까지 단계별로 학습한다.

> [!info] 검토 조건 — 2026-10-04
> 역사적 작성일 2026-01-31을 유지한다. 공식 문서는 확인 시 v3.0.x 표시이며 로컬 확인 판본은 pymilvus3.0.2·milvus-lite3.2.1이다. 최신/운영 검증을 뜻하지 않는다. 예제는 함수를 명시적으로 호출해야 연결·저장이 일어난다. Docker 서버·분산 운영·실제 임베딩 품질·인증은 미확인이다. [정리 기록](../organization-log.md)을 함께 읽는다.


## 학습 로드맵

```
1. Milvus 기초 개념 및 설치
   ↓
2. Collection/Index 설계 및 벡터 검색
   ↓
3. LangChain + Milvus RAG 파이프라인
   ↓
4. LangGraph 기반 고급 RAG 연동
```

## 문서 목차

| 순서 | 문서 | 설명 |
|------|------|------|
| 1 | [Milvus 기초](./milvus-basics.md) | 아키텍처, Collection, Index, 유사도 검색 |
| 2 | [Milvus RAG 연동](./milvus-rag-integration.md) | LangChain/LangGraph와 Milvus 통합 |

기초 문서는 DB schema/거리/인덱스와 직접 SDK 검색을, 연동 문서는 LangChain wrapper/PDF 적재/retriever/유한 폴백 그래프를 설명한다. wrapper schema는 기초 예제와 다르므로 같은 Collection을 공유하지 않는다. 새 이름·임시 DB부터 API 동작을 확인하고 실제 모델 품질/서버 운영은 따로 검증한다. 문서는 학습 자료이며 회사 배포 기록이 아니다.

## 사전 지식

- Python 기본 문법
- 임베딩(Embedding) 개념 이해
- Docker 서버 실습을 선택한 경우 Docker 기본 사용법; Lite 실습에서는 Docker 불필요
- (선택) [LangGraph 시리즈](../langgraph/README.md) - RAG 연동 시 필요

## 관련 문서

- [LangGraph 시리즈](../langgraph/README.md) - LangGraph 기반 RAG 파이프라인
- [LangGraph RAG](../langgraph/langgraph-rag.md) - Corrective RAG 구현 (Milvus 연동 대상)

---

*Last updated: 2026-01-31*
