# Knowledge Base - CLAUDE.md

> 이 파일은 Claude Code가 이 repository의 컨텍스트를 이해하고 일관된 방식으로 문서를 생성/관리하기 위한 가이드입니다.

## 📌 프로젝트 개요

**목적**: 개인 학습 지식을 체계적으로 정리하고 축적하는 저장소
**소유자**: Daeyoung (SK Hynix AI/DT TF)
**주요 관심 분야**: Web Development, AI/DT 시스템 개발

---

## 📁 폴더 구조

```
knowledge-base/
├── CLAUDE.md                           # 이 파일 (Claude Code 컨텍스트)
├── README.md                           # 전체 목차 및 학습 진행 현황
│
├── web-development/
│   ├── README.md
│   ├── python/
│   │   ├── fastapi/                   # FastAPI 백엔드 개발
│   │   └── flask/                     # Flask 웹 애플리케이션
│   └── typescript/
│       ├── vue/                       # Vue.js 프론트엔드
│       └── nuxt/                      # Nuxt.js SSR/SSG
│
├── ai-dt/
│   ├── README.md
│   ├── rag/
│   │   └── langgraph/                 # LangGraph 기반 RAG 시스템
│   ├── mcp/                           # Model Context Protocol
│   └── data-handling/                 # 데이터 처리 및 파이프라인
│
└── _templates/                         # 문서 템플릿
    ├── concept.md                     # 개념 설명용
    ├── tutorial.md                    # 실습/튜토리얼용
    ├── cheatsheet.md                  # 빠른 참조용
    └── troubleshooting.md             # 문제 해결 기록용
```

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

### 관련 문서 링크
문서 하단에 관련 문서를 링크합니다:

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
status: in-progress | complete | needs-review
---
```

---

## 📋 자주 사용하는 Claude Code 명령어 패턴

### 새로운 주제 학습 시작
```
"[분야]/[주제]/ 폴더에 [토픽]에 대해 정리해줘.
기본 개념부터 시작해서 실제 구현 코드까지 단계별로 작성하고,
내가 [관련 프로젝트]할 때 어떻게 연결되는지도 포함해줘"
```

### 기존 문서 확장
```
"[파일 경로] 읽고, [새로운 섹션] 추가해줘.
실무에서 자주 쓰는 패턴 위주로"
```

### 복습용 요약 생성
```
"[폴더 경로] 전체 읽고 핵심 개념 cheatsheet 만들어줘"
```

### 트러블슈팅 기록
```
"[문제 상황] 해결했어. troubleshooting 문서에 기록해줘.
원인, 해결 과정, 최종 해결책 포함해서"
```

### README 업데이트
```
"오늘 [주제] 관련 문서 작성했어.
해당 폴더 README.md 목차 업데이트하고,
루트 README.md에도 진행상황 반영해줘"
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

## 🔄 워크플로우

```
1. 궁금한 주제 발생
   ↓
2. Claude Code: "이 주제 기본 문서 만들어줘"
   ↓
3. 읽고 이해하면서 추가 질문
   ↓
4. Claude Code: "이 부분 더 자세히 / 예제 추가"
   ↓
5. 실습 후 insights 추가
   ↓
6. Git commit으로 버전 관리
   ↓
7. 주기적으로 복습 및 업데이트
```

---

## 📊 학습 진행 현황 관리

루트 README.md에서 전체 진행 상황을 추적합니다:

```markdown
## 학습 진행 현황

| 분야 | 주제 | 상태 | 마지막 업데이트 |
|------|------|------|----------------|
| AI/DT | RAG/LangGraph | 🟡 진행중 | 2025-01-31 |
| Web | FastAPI | 🟢 기본 완료 | 2025-01-30 |
| Web | TypeScript/Vue | 🔴 시작 전 | - |
```

상태:
- 🔴 시작 전
- 🟡 진행 중
- 🟢 기본 완료
- ⭐ 심화 완료

---

### 학습 ↔ 실무 연결
문서 작성 시 실무 프로젝가 있으면 연관성을 고려:
- "이 기술이 Recipe Setup 자동화에 어떻게 적용될 수 있는가?"
- "SKEWNONO에서 이 패턴을 사용할 수 있는가?"

---

## ⚠️ 주의사항

1. **회사 기밀 정보 제외**: 구체적인 장비 데이터, 내부 시스템 상세 정보는 포함하지 않음
2. **저작권 준수**: 외부 자료 인용 시 출처 명시
3. **정기 백업**: GitHub에 주기적으로 push
4. **버전 관리**: 의미 있는 커밋 메시지 작성

---

*Last updated: 2025-01-31*

