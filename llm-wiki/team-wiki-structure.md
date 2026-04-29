---
tags: [llm-wiki, knowledge-management, team-collaboration]
level: intermediate
last_updated: 2026-04-29
status: in-progress
---

# 팀 프로젝트용 LLM Wiki 구조 가이드

> 코드 구성·사용법·연구노트·학습 노트를 한 곳에 축적하는 팀 위키를 어떤 폴더 구조로 가져갈지에 대한 설계서

## 왜 필요한가? (Why)

팀 프로젝트가 진행되면 정보가 세 갈래로 흩어진다:

1. **코드 구성·사용법** — 팀원들이 "이 모듈 어떻게 쓰지?"에 답하기 위한 정보. README, API docs, 운영 매뉴얼.
2. **연구노트·저널** — "왜 이렇게 결정했지?"에 답하기 위한 개발 과정 기록. ADR, 시도와 실패, 미팅 결정.
3. **학습 노트** — "이 라이브러리 처음 보는데?"라는 개인 막힘을 팀 자산으로 만드는 메모.

이 셋을 흩어진 Notion, 개인 메모, Slack 스크롤백에서 찾는 비용은 매우 비싸다. RAG로 raw chunk만 검색하면 매번 같은 합성을 반복한다. **LLM Wiki 패턴(Karpathy)** 은 raw 원본을 immutable하게 두고, LLM이 합성한 위키 페이지를 Git에 영구 저장해 시간이 지나도 자라는 지식 베이스를 만든다.

→ 이 문서는 **팀 프로젝트 repo 안 `docs/llm-wiki/`** 에 위 패턴을 적용할 때의 구체 폴더 구조를 정의한다.

관련 배경: [`./README.md`](./README.md) (LLM Wiki 패턴 일반론)

## 핵심 개념 (What)

### 3-Layer 구조

| Layer | 역할 | 수정 권한 |
| --- | --- | --- |
| **raw/** | 원본 자료 (저널, 미팅 요약, ADR 초안, 학습 메모) | 추가만 가능, 수정 금지 |
| **wiki/** | LLM이 합성한 마크다운 페이지 (cross-link, 인용 포함) | LLM 생성 + 사람 PR 리뷰 |
| **WIKI_SCHEMA.md** | LLM 운영 규칙 (인용·언어·금지사항) | 팀 합의로만 변경 |

### 사용자 3대 요구사항 → 폴더 매핑

| 요구사항 | 입력 (raw) | 출력 (wiki) |
| --- | --- | --- |
| 코드 구성/사용법 (팀 공유) | `raw/specs/`, 코드 자체 | `wiki/components/`, `wiki/runbooks/`, `wiki/overview.md` |
| 연구노트·저널 (개발 과정) | `raw/journals/`, `raw/decisions/` | `wiki/decisions/` |
| 학습 노트 (모르는 것 정리) | `raw/learning-logs/` | `wiki/concepts/` |

## 어떻게 사용하는가? (How)

### 1. 권장 폴더 구조

```text
docs/llm-wiki/
├── WIKI_SCHEMA.md               # LLM 운영 규칙 (팀 합의)
├── README.md                    # 위키 사용법, ingest/query/lint 워크플로우
│
├── raw/                         # 불변 원본 자료
│   ├── journals/                # 개발 저널: YYYYMMDD-<topic>.md
│   ├── meetings/                # 미팅 요약 (raw transcript 금지)
│   ├── decisions/               # ADR 초안/원본
│   ├── specs/                   # 요구사항·설계 문서
│   ├── incidents/               # 장애·이슈 회고
│   └── learning-logs/           # "이거 처음 봤다" 거친 메모
│
└── wiki/                        # LLM 생성·인간 리뷰 합성 페이지
    ├── index.md                 # 전체 목차 (매 ingest 후 갱신)
    ├── log.md                   # ingest/lint 이력
    ├── overview.md              # 프로젝트 한눈에 보기 (신규 합류자용)
    │
    ├── components/              # 코드 구성 (모듈·서비스·패키지별)
    │   └── <component>.md       # 책임, 핵심 진입점, 의존성, 사용 예
    │
    ├── runbooks/                # 사용법·운영 절차
    │   ├── setup.md
    │   ├── deploy.md
    │   └── <task>.md
    │
    ├── concepts/                # 학습한 개념 (Why→What→How→Refs)
    │   └── <concept>.md
    │
    ├── decisions/               # ADR 합성 (raw/decisions/ 정리본)
    │   └── <YYYYMMDD>-<title>.md
    │
    └── sources/                 # 원본 자료 인덱스/요약
```

### 2. 부트스트랩 명령

팀 프로젝트 repo의 root에서:

```bash
mkdir -p docs/llm-wiki/raw/{journals,meetings,decisions,specs,incidents,learning-logs}
mkdir -p docs/llm-wiki/wiki/{components,runbooks,concepts,decisions,sources}
touch docs/llm-wiki/WIKI_SCHEMA.md
touch docs/llm-wiki/README.md
touch docs/llm-wiki/wiki/{index.md,log.md,overview.md}
```

`WIKI_SCHEMA.md`는 [`./wiki-schema-template.md`](./wiki-schema-template.md) 의 내용을 그대로 복사해 시작한다.

### 3. 운영 워크플로우

#### 일상 (개발 중) — 거친 raw만 쌓는다

- 모르는 라이브러리·개념 마주침 → `raw/learning-logs/YYYYMMDD-<주제>.md` 에 3~5줄 메모.
- 의미 있는 시도·결정 → `raw/journals/YYYYMMDD.md` 에 저널.
- 큰 의사결정 → `raw/decisions/YYYYMMDD-<title>.md` 에 ADR 초안.

> 핵심: 정제하지 않는다. **거친 raw가 자주 쌓이는 게 정제된 wiki가 가끔 쌓이는 것보다 가치 있다.**

#### 주간 (ingest) — LLM이 합성

LLM에 다음과 같이 명령:

```text
docs/llm-wiki/WIKI_SCHEMA.md를 운영 규칙으로 사용.
docs/llm-wiki/raw/learning-logs/ 의 신규 파일을 ingest.
wiki/concepts/ 페이지를 생성·업데이트하고
wiki/index.md, wiki/log.md 를 갱신.
raw/ 는 수정 금지. 인용 경로 포함.
```

LLM이 만든 변경사항을 PR로 올리고, 팀이 리뷰한 후 머지한다.

#### 월간 (lint) — 위생 점검

```text
docs/llm-wiki/wiki/ 전체에 lint 패스.
- stale 주장
- 페이지 간 contradiction
- broken cross-link
- orphan 페이지
- 누락된 canonical 페이지 후보
를 찾아 wiki/log.md 에 리포트.
```

### 4. 페이지 작성 예시

#### `wiki/components/<component>.md` 템플릿

```markdown
---
tags: [component, backend]
last_updated: 2026-04-29
owner: @username
sources: [raw/specs/api-v2.md, src/api/v2/router.py]
---

# <Component Name>

> 한 줄 요약

## 왜 존재하는가? (Why)
- 이 컴포넌트가 해결하는 문제

## 무엇인가? (What)
- 책임 범위
- 핵심 진입점: `src/.../entrypoint.py:42`
- 의존성: <other component>, <library>

## 어떻게 쓰는가? (How)
- 호출 예시 코드
- 자주 쓰는 패턴

## 관련 (References)
- 원본 스펙: [raw/specs/api-v2.md](../../raw/specs/api-v2.md)
- 의사결정: [decisions/20260315-api-versioning.md](../decisions/20260315-api-versioning.md)
```

#### `wiki/concepts/<concept>.md` 템플릿

CLAUDE.md의 표준 구조(Why → What → How → References)를 그대로 적용. `embedding-basics.md`, `vector-database-comparison.md` 같이 학습 노트가 자라난다.

### 5. 검증 체크리스트

위키가 "쓸만한가"는 다음으로 판단:

- [ ] 신규 합류자가 `wiki/overview.md` 한 페이지로 프로젝트를 이해할 수 있는가?
- [ ] `wiki/runbooks/setup.md` 만으로 로컬 환경을 띄울 수 있는가?
- [ ] 주요 의사결정 5건 이상이 `wiki/decisions/` 에 ADR로 정리되었는가?
- [ ] 모든 비자명 주장에 raw 또는 코드 경로 citation이 있는가?
- [ ] `raw/` 에 비밀·자격증명·고객 PII가 들어가지 않았는가?
- [ ] PR 리뷰가 LLM의 잘못된 합성을 잡아낸 사례가 1건 이상 있는가? (없으면 리뷰가 형식적일 가능성)
- [ ] 월간 lint 결과가 stale/contradiction 0건인가? (너무 적으면 lint가 부실)

### 6. 안 하기로 한 것 (스코프 외)

- **별도 web UI** (MkDocs, Docusaurus): 처음에는 GitHub 마크다운 렌더링으로 충분. 검색 한계를 느낄 때 도입.
- **별도 인프라** (`lucasastorian/llmwiki`, Onyx, Open WebUI): 팀이 50명 이상이거나 multi-source 인덱싱이 필요할 때 재검토.
- **raw 미팅 transcript ingest**: 민감 정보 위험. 항상 사람이 요약한 후 `raw/meetings/` 에 저장.

## 참고 자료 (References)

### 내부 문서
- [LLM Wiki 패턴 일반론](./README.md) — Karpathy 패턴, 도구 옵션 비교
- [WIKI_SCHEMA 템플릿](./wiki-schema-template.md) — 부트스트랩용 스키마
- [프로젝트 컨벤션](../CLAUDE.md) — 문서 구조·언어·보안 규칙

### 외부 링크
- Andrej Karpathy, LLM Wiki gist: <https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f>
- Open-source 구현: <https://github.com/lucasastorian/llmwiki>
