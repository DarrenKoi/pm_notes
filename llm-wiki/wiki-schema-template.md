---
tags: [llm-wiki, template, governance]
level: intermediate
last_updated: 2026-04-29
status: complete
---

# WIKI_SCHEMA 템플릿

> 팀 프로젝트 repo 의 `docs/llm-wiki/WIKI_SCHEMA.md` 부트스트랩용. 그대로 복사해 시작하고, 팀 합의로만 수정한다.

## 왜 필요한가? (Why)

LLM에게 "위키에 정리해줘"만 시키면 매번 다른 스타일·다른 인용 규칙·다른 폴더 위치에 글을 만든다. SCHEMA 파일은 LLM이 ingest/query/lint를 할 때마다 읽는 운영 규칙이다. 이게 있어야 위키가 일관성을 유지한다.

## 무엇인가? (What)

- 위치: `docs/llm-wiki/WIKI_SCHEMA.md` (위키 root에 둠)
- 변경 빈도: 분기 1회 이하 (자주 바뀌면 일관성이 깨짐)
- 변경 권한: 팀 합의 + PR 리뷰

## 어떻게 사용하는가? (How)

아래 블록을 통째로 `docs/llm-wiki/WIKI_SCHEMA.md` 에 복사한다. `<프로젝트명>` 등은 실제 값으로 치환.

---

````markdown
# WIKI_SCHEMA — <프로젝트명> LLM Wiki 운영 규칙

> 이 파일은 LLM이 ingest/query/lint 를 수행할 때 매번 읽는 운영 규칙이다.
> 변경은 팀 합의 + PR 리뷰로만.

Last updated: YYYY-MM-DD
Owners: @<owner1>, @<owner2>

---

## 1. 불변 규칙

- `raw/` 하위 파일은 **절대 수정 금지**. 새 정보가 생기면 새 파일 추가.
- 모든 비자명 주장은 raw 경로(`raw/...`) 또는 코드 경로(`src/...:line`) 인용.
- 비밀·자격증명·고객 PII·내부 장비 데이터 ingest 금지.
- 미팅 raw transcript ingest 금지. 사람이 요약한 후 `raw/meetings/` 에 저장.

## 2. 페이지 작성 규칙

- **언어**: 한국어, 기술용어는 영어 병기 (예: "임베딩(Embedding)").
- **구조**: 모든 합성 페이지는 `Why → What → How → References` 4섹션을 포함.
- **파일명**: 소문자, 하이픈(`-`) 구분, 검색 가능한 이름. 예: `embedding-basics.md`.
- **메타데이터**: 모든 wiki/ 페이지 상단에 frontmatter:
  ```yaml
  ---
  tags: [<tag1>, <tag2>]
  level: beginner | intermediate | advanced
  last_updated: YYYY-MM-DD
  status: in-progress | complete | needs-review
  owner: @<username>
  sources: [<raw/path>, <src/path>]
  ---
  ```
- **단일 주제 원칙**: 한 페이지에 한 주제. catch-all 문서 지양.
- **불확실 표기**: 검증 안 된 주장은 `> Unverified:` 인용 블록.
- **충돌 표기**: 페이지 간 모순은 `> Conflict:` 인용 블록 + 양쪽 source 인용.

## 3. 폴더별 용도

### raw/ (불변 입력)
- `journals/` — 개발 저널 (`YYYYMMDD-<topic>.md`)
- `meetings/` — 미팅 요약 (raw transcript 금지)
- `decisions/` — ADR 초안
- `specs/` — 요구사항·설계 문서
- `incidents/` — 장애·이슈 회고
- `learning-logs/` — "이거 처음 봤다" 거친 메모

### wiki/ (LLM 합성, 인간 리뷰)
- `index.md` — 전체 목차. 매 ingest 후 갱신.
- `log.md` — ingest/lint 이력. 매 작업 후 1행 추가.
- `overview.md` — 신규 합류자용 한 페이지 요약.
- `components/` — 코드 컴포넌트 (모듈·서비스).
- `runbooks/` — 사용법·운영 절차.
- `concepts/` — 학습 개념 정리.
- `decisions/` — ADR 합성 페이지.
- `sources/` — 원본 자료 인덱스.

## 4. Ingest 규칙

LLM이 한 번의 ingest에서 수행해야 할 것:

1. 지정된 raw 파일을 읽고 핵심 정보 추출.
2. 영향받는 wiki 페이지 식별 (기존 갱신 + 신규 생성).
3. 각 주장에 raw 경로 또는 코드 경로 인용 추가.
4. 페이지 간 cross-link 생성 (절대경로가 아닌 상대경로).
5. `wiki/index.md` 갱신.
6. `wiki/log.md` 에 1행 추가.
7. 발견된 open question·contradiction 별도 보고.
8. **raw/ 는 절대 건드리지 않음**.

## 5. Query 규칙

- 위키 페이지에서 먼저 답을 찾는다.
- 위키에 정보가 없거나 stale 의심이면 raw/ 에서 보강.
- 답변 가치가 있다고 판단되면 wiki/ 에 새 페이지로 저장 제안.

## 6. Lint 규칙 (주 1회 권장)

다음 항목을 점검해 `wiki/log.md` 에 리포트:

- **Stale**: `last_updated` 가 90일 이상 지났고 source 가 변경된 페이지.
- **Contradiction**: 두 페이지가 같은 사실에 대해 다르게 기술.
- **Missing citation**: 비자명 주장에 raw·코드 인용이 없음.
- **Broken link**: 깨진 cross-link.
- **Orphan**: 어디에서도 link 되지 않은 페이지.
- **Canonical gap**: raw 에 자주 등장하지만 canonical wiki 페이지가 없는 개념.

## 7. log.md 형식

```markdown
## [YYYY-MM-DD] <action> | <target>

- <변경 1>
- <변경 2>
- <발견된 open question / conflict>
```

action: `ingest` | `query-saved` | `lint` | `manual-edit`

예시:
```markdown
## [2026-04-29] ingest | raw/learning-logs/20260428-langgraph-state.md

- Added wiki/concepts/langgraph-state-management.md
- Updated wiki/components/rag-pipeline.md (state 흐름 섹션)
- Open question: checkpoint 저장소 선택은 raw/decisions/ 에 ADR 필요
```

## 8. PR 워크플로우

- 위키 변경은 PR 로만. main 직접 push 금지.
- PR 제목: `[wiki] <action>: <summary>` 예: `[wiki] ingest: langgraph state notes`.
- 리뷰어 1인 이상 승인 필요.
- LLM 자동 ingest PR도 동일 규칙 적용.
````

---

## 참고 자료 (References)

- [팀 위키 구조 가이드](./team-wiki-structure.md)
- [LLM Wiki 패턴 일반론](./README.md)
- [프로젝트 컨벤션](../CLAUDE.md)
