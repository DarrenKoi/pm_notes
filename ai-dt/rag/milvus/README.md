# Milvus Vector DB 시리즈

> Milvus 벡터 데이터베이스의 기본 개념부터 RAG 시스템 연동까지 단계별로 학습한다.

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

## 사전 지식

- Python 기본 문법
- 임베딩(Embedding) 개념 이해
- Docker 기본 사용법
- (선택) [LangGraph 시리즈](../langgraph/README.md) - RAG 연동 시 필요

## 관련 문서

- [LangGraph 시리즈](../langgraph/README.md) - LangGraph 기반 RAG 파이프라인
- [LangGraph RAG](../langgraph/langgraph-rag.md) - Corrective RAG 구현 (Milvus 연동 대상)

---

*Last updated: 2026-01-31*
