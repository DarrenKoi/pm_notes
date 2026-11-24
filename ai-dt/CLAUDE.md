---
tags: [ai-dt, repository-guidance]
reviewed_on: 2026-10-04
review_status: partial
document_type: agent_guidance
---

# AI/DT 학습 노트 - CLAUDE.md

> 이 파일이 `ai-dt/`의 유일한 에이전트 지침이다. 루트 지침 파일은 없고, 공통 규칙은 아래 "저장소 공통 규칙"에 들어 있다.

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

> `ai-dt/` 안의 주제 폴더끼리도 서로 독립적이다. 아래 폴더 독립성 원칙을 주제 폴더 단위로 그대로 적용한다.

## 저장소 공통 규칙

- **폴더 독립성**: pm_notes의 최상위 폴더는 별개 저장소처럼 다룬다. 탐색, 근거, 용어 출처, 링크, 수정, 커밋을 작업 중인 폴더 안으로 한정한다. 작업 폴더가 불분명하면 여러 폴더를 훑지 말고 먼저 묻는다
- **언어**: 한국어가 기본, 기술 용어는 영어 병기 (예: "임베딩(Embedding)"). em dash(—)와 가운데 점(·)을 쓰지 않는다 (나열은 쉼표나 "와/과", 부연은 괄호나 문장 분리, 제목 구분은 콜론)
- **문서 구조**: `# 주제명`, 한 줄 요약, 그다음 Why(해결하는 문제, 실무 맥락) → What(핵심 개념, 용어) → How(실행 가능한 코드, 단계별 설명) → References. 문서 하단에 같은 폴더 안의 관련 문서 링크
- **frontmatter**: `tags`, `level`, `last_updated`
- **파일명**: 소문자와 하이픈, 검색 가능한 이름 (예: `embedding-basics.md`)
- **코드 예제**: 실행 가능한 완전한 코드, 복잡하면 단계별로 분리. Python은 4칸 들여쓰기, `snake_case`, 바꾼 코드에 type hint
- **질문 기반 문서화**: 기술 질문이 기존 주제 폴더와 명확히 맞으면 답변한 뒤 그 폴더에 문서를 만들거나 갱신한다 (예: LangGraph state 질문 → `rag/langgraph/state-management.md`). 맞는 폴더가 불분명하면 답변만 한다
- **품질**: "왜 필요한가"가 분명하고 실무 맥락(Recipe Setup 자동화, SKEWNONO 등)과 이어져야 한다. 공식 문서 단순 번역, 맥락 없는 코드 조각, 추상적인 설명은 피한다
- **기밀과 출처**: 구체 장비 데이터, 내부 시스템 상세, 비밀값, 사내 endpoint는 넣지 않는다. 외부 자료는 출처를 적는다
- **모바일 읽기용 HTML**: 폴더 문서를 새로 쓰거나 크게 고치면 저장소 루트에서 `python scripts/build_folder_html.py <폴더>`로 `reading.html`을 만들어 claude.ai Artifact(비공개)로 게시하고 폴더 README의 "휴대폰, 태블릿에서 읽기" 절에 링크를 둔다. 이미 링크가 있으면 그 URL로 갱신한다. `reading.html`은 손으로 고치지 않는다
- **커밋**: 한 커밋에 한 가지 변경, 폴더 단위로 나눈다. 제목은 `docs(<폴더>): 한 일` 형식
- **Issue와 도메인 문서**: GitHub Issues `DarrenKoi/pm_notes` (`gh` CLI), 절차는 루트 `docs/agents/`. 도메인 용어는 하위 폴더의 `CONTEXT.md` (예: `roadmap/CONTEXT.md`)
