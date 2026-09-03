# pm_notes - CLAUDE.md

> 이 파일은 Claude Code가 이 repository의 컨텍스트를 이해하고 일관된 방식으로 문서를 생성/관리하기 위한 가이드입니다.

## 📌 프로젝트 개요

**목적**: 개인 학습 지식을 체계적으로 정리하고 축적하는 저장소
**소유자**: Daeyoung (SK Hynix AI/DT TF)
**주요 관심 분야**: Web Development, AI/DT 시스템 개발

---

## 📁 폴더 구조

**최상위 폴더 = 하나의 독립 주제.** 하위 트리는 각 폴더의 README.md가 관리하므로 여기서는 폴더 목록만 유지한다.

| 폴더 | 주제 |
|------|------|
| `ai-dt/` | AI/DT 학습 노트 (RAG, MCP, OpenSearch, LLMOps, ITC 로드맵) |
| `web-development/` | 웹 개발 노트 + 예제 앱 (FastAPI / Flask / Vue / Nuxt) |
| `Codes/python/` | 실행 가능한 파이썬 예제 (opensearch_handler, drm-pptx-extraction 등) |
| `dev-environment/` | 개발 환경·툴링 노트 (codex, pi, terminal, vlm) |
| `my-task/` | 진행 중인 업무 산출물 (AIX_POC, 2026_report) — 자체 CLAUDE.md 있음 |
| `RAG/` | RAG 관련 단발 분석 문서 |
| `docs/` | 에이전트 운영 문서 (issue-tracker, triage-labels, domain) |
| `_workspace/` | 세션 임시 작업물 |

### 폴더 독립성 원칙 (Standalone Folders) — 이 저장소의 1순위 규칙

**최상위 폴더는 서로 다른 주제의 별개 저장소처럼 다룬다. 한 대화는 한 폴더 안에서 끝난다.**

작업할 폴더가 정해지면:

- **탐색 범위를 그 폴더로 한정한다** — `grep`/`find`/`Read` 경로를 해당 폴더로 제한한다. 저장소 전역 검색으로 다른 폴더 내용을 끌어오지 않는다.
- **다른 폴더의 문서를 근거·참조·용어 출처로 삼지 않는다** — 같은 단어라도 폴더마다 뜻이 다를 수 있다. 모르면 폴더 밖을 뒤지지 말고 사용자에게 묻는다.
- **폴더 간 크로스 링크를 만들지 않는다** — 관련 문서 링크는 같은 폴더 내부로 한정.
- **변경을 폴더 밖으로 번지게 하지 않는다** — 한 폴더 작업이 다른 폴더의 문서나 README를 수정하게 두지 않는다. 커밋도 폴더 단위로 나눈다.
- 각 폴더는 자체 README.md로 완결된 목차를 유지한다.

**예외 — 전역 파일은 어느 폴더에서든 읽는다:** 루트 `CLAUDE.md` · `AGENTS.md` · `README.md` · `CONTEXT-MAP.md`, 그리고 `docs/agents/`. 그 외에는 폴더 밖을 보지 않는다.

**작업 폴더가 불분명하면 먼저 물어본다** — 여러 폴더를 추측으로 훑지 않는다.

---

## ✍️ 문서 작성 규칙

### 언어
- 기본 언어: **한국어**
- 기술 용어: 영어 병기 (예: "임베딩(Embedding)")
- 코드 주석: 영어 또는 한국어 (일관성 유지)

### 문서 구조
모든 문서는 다음 구조를 따릅니다:

```markdown
# [주제명]

> 한 줄 요약

## 왜 필요한가? (Why)
- 이 기술/개념이 해결하는 문제
- 실무에서의 활용 맥락

## 핵심 개념 (What)
- 주요 개념 설명
- 관련 용어 정리

## 어떻게 사용하는가? (How)
- 실제 코드 예제
- 단계별 설명

## 참고 자료 (References)
- 공식 문서 링크
- 관련 내부 문서 링크
```

### 코드 예제 규칙
- 실행 가능한 완전한 코드 제공
- 복잡한 코드는 단계별로 분리
- 실무에서 바로 사용 가능한 패턴 우선

### 파일명 규칙
- 소문자, 하이픈(-) 사용
- 명확하고 검색 가능한 이름
- 예: `embedding-basics.md`, `fastapi-dependency-injection.md`

---

## 🔗 문서 간 연결

> ⚠️ 문서 링크는 **같은 최상위 폴더 내부**로 한정합니다. 폴더끼리는 독립적(standalone)이므로 폴더 간 크로스 링크는 만들지 않습니다. (위 "폴더 독립성 원칙" 참고)

### 관련 문서 링크
문서 하단에 관련 문서를 링크합니다 (같은 폴더 내 문서만):

```markdown
## 관련 문서
- [상위 개념](../parent-topic.md)
- [관련 주제](./related-topic.md)
- [실습 예제](./examples/example-name.md)
```

### 태그 시스템
문서 상단에 메타데이터 추가:

```markdown
---
tags: [rag, embedding, milvus]
level: intermediate
last_updated: 2025-01-31
---
```

---

## 🧠 질문 기반 자동 문서화

사용자가 특정 기술 질문을 했을 때, 해당 질문이 기존 하위 폴더/주제(예: FastAPI, LangGraph, Vue, MCP 등)와 명확히 매칭되면:

1. **질문에 답변**한다
2. **해당 하위 폴더에 문서를 자동 생성 또는 업데이트**한다 (special topic note)
3. 기존 문서 구조(`Why → What → How → References`)와 파일명 규칙(`소문자-하이픈.md`)을 따른다

### 예시
- 질문: "FastAPI에서 Dependency Injection 어떻게 쓰는 거야?"
  → 답변 후 `web-development/python/fastapi/dependency-injection.md` 생성/업데이트
- 질문: "LangGraph에서 state 관리는?"
  → 답변 후 `ai-dt/rag/langgraph/state-management.md` 생성/업데이트

> 매칭되는 폴더가 불분명하거나 질문이 일반적인 경우에는 문서를 생성하지 않고 답변만 제공한다.

---

## 🎯 품질 기준

### 좋은 문서의 조건
- [ ] "왜 필요한가"가 명확히 설명됨
- [ ] 실행 가능한 코드 예제 포함
- [ ] 내 실무 맥락과 연결됨
- [ ] 관련 문서 링크 연결됨
- [ ] 나중에 봐도 이해 가능한 수준의 설명

### 피해야 할 것
- 공식 문서 단순 번역/복사
- 맥락 없는 코드 조각
- 너무 추상적인 설명
- 업데이트되지 않는 오래된 정보

---

## 학습 ↔ 실무 연결
문서 작성 시 실무 프로젝트가 있으면 연관성을 고려:
- "이 기술이 Recipe Setup 자동화에 어떻게 적용될 수 있는가?"
- "SKEWNONO에서 이 패턴을 사용할 수 있는가?"

---

## ⚠️ 주의사항

1. **회사 기밀 정보 제외**: 구체적인 장비 데이터, 내부 시스템 상세 정보는 포함하지 않음
2. **저작권 준수**: 외부 자료 인용 시 출처 명시
3. **정기 백업**: GitHub에 주기적으로 push
4. **버전 관리**: 의미 있는 커밋 메시지 작성

---

## Agent skills

### Issue tracker

Issues live in GitHub Issues on `DarrenKoi/pm_notes` (via the `gh` CLI). See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical triage labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Multi-context layout — `CONTEXT-MAP.md` at the repo root points to per-subproject `CONTEXT.md` files (e.g. `ai-dt/roadmap/CONTEXT.md`). See `docs/agents/domain.md`.

---

*Last updated: 2026-09-03*

