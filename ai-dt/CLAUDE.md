---
tags: [ai-dt, repository-guidance]
reviewed_on: 2026-10-04
review_status: partial
document_type: agent_guidance
---

# AI/DT 학습 노트 - CLAUDE.md

> 문서 작성 규칙, 폴더 독립성 원칙 등 공통 가이드는 [상위 CLAUDE.md](../CLAUDE.md)를 따른다.

## 📌 이 디렉토리 정보

- **목적**: AI/DT 관련 기술 학습 문서 저장소
- **검증 명령은 모듈별 확인**: 학습 문서 중심이며 일부 하위 모듈에 실행 자료가 있다. 공통 빌드/테스트 명령이 있다는 전제는 두지 않는다. 실제 파일·각 모듈 README를 확인하고 문서 예제의 로컬 검증과 서비스/설비 실행을 구분한다.

## 📁 주제 구조

주제 폴더 목록만 유지한다 (하위 구조는 각 폴더 README.md 참조).

| 폴더 | 주제 |
|------|------|
| `rag/` | RAG — `langgraph/` `langchain-langgraph/` `milvus/` `opensearch/` `advanced-rag/` `token_strategy/` |
| `langchain/` | LangChain 커리큘럼 |
| `LLMOps/` | LLMOps & 평가 커리큘럼 |
| `mcp/` | Model Context Protocol |
| `data-handling/` | 데이터 처리 — `normalization/` (정규화·온톨로지) |
| `ml-dl/` | 머신러닝·딥러닝 기초 |
| `foundation model/` | 파운데이션 모델 |
| `unsloth/` | 파인튜닝 (Unsloth) |
| `ai-coding-dictionary/` | AI 코딩 용어 사전 |
| `ai-terms-and-technologies/` | AI 용어·기술 해설 |
| `openwiki/` | OpenWiki 가이드 |
| `roadmap/` | ITC AI/DT 로드맵 (`CONTEXT.md` 보유) |
| `llm_question/` | LLM 리포트 경진대회 출제 (`CONTEXT_CODEX.md` 보유) |

> `ai-dt/` 안의 주제 폴더끼리도 서로 독립적이다 — 루트 CLAUDE.md의 폴더 독립성 원칙을 그대로 적용한다.
