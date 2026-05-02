---
tags: [llm-wiki, starter-template]
level: intermediate
last_updated: 2026-05-01
status: complete
---

# LLM Wiki Starter — 사용법

> 이 `llm-wiki/` 폴더 자체를 다른 프로젝트 repo 의 `docs/llm-wiki/` (또는 `doc/llm-wiki/`) 로 복사해 바로 사용한다.

## 왜 필요한가? (Why)

새 프로젝트마다 폴더 구조·운영 규칙·페이지 템플릿을 처음부터 만들면 일관성이 깨진다. 이 폴더는 [README.md](./README.md) 의 설계를 그대로 적용한 부트스트랩 키트다. **복사 → 토큰 치환 → 첫 ingest** 까지 10분 안에 가능하도록 설계됨.

## 무엇이 들어있는가? (What)

```text
llm-wiki/
├── README.md                  # 설계 컨텍스트 — LLM Wiki 패턴 일반론
├── USAGE.md                   # 이 파일
├── WIKI_SCHEMA.md             # ★ 운영 규칙 (LLM 이 매번 읽는 파일)
├── PROMPTS.md                 # ★ 개인용 프롬프트 모음 (모델 무관)
├── prompts_example/           # ★ 채워진 실제 프롬프트 예시 (시나리오별)
│
├── raw/                       # 불변 원본 (LLM 수정 금지)
│   ├── journals/              # 개발 저널
│   ├── decisions/             # ADR 초안
│   ├── specs/                 # 요구사항·설계 문서
│   ├── incidents/             # 장애·이슈 회고
│   └── references/            # 외부 학습 자료 (블로그·강의·논문·책)
│       ├── articles/
│       ├── courses/
│       ├── papers/
│       └── books/
│
├── wiki/                      # LLM 합성·개인 검토
│   ├── index.md               # 전체 목차 (매 ingest 후 갱신)
│   ├── log.md                 # ingest/lint 이력
│   ├── overview.md            # 프로젝트 한눈에 보기
│   ├── components/            # 코드 컴포넌트
│   ├── runbooks/              # 사용법·운영 절차
│   ├── concepts/              # 학습 개념
│   ├── decisions/             # ADR 합성 페이지
│   └── sources/               # 원본 자료 인덱스
│
└── _templates/                # 페이지 템플릿
    ├── component.md
    ├── concept.md
    ├── decision.md
    ├── runbook.md
    └── journal.md
```

## 어떻게 사용하는가? (How)

### 1. 대상 프로젝트로 복사

```bash
# 대상 프로젝트의 root 에서
cp -r /path/to/pm_notes/llm-wiki ./docs/llm-wiki
```

> **선택**: 설계 컨텍스트(`README.md`, `USAGE.md`) 를 대상 프로젝트에서 빼고 싶다면 복사 후 삭제하거나, `_design/` 같은 하위 폴더로 옮긴다. **`WIKI_SCHEMA.md` 는 반드시 root 에 유지** — LLM 이 이 경로를 매번 참조한다.

### 2. 부트스트랩 (프로젝트 셋업 담당자만, 1 회)

[`PROMPTS.md`](./PROMPTS.md) §1 (Bootstrap) 프롬프트를 LLM 에 지시하면 다음을 일괄 처리한다:

- `WIKI_SCHEMA.md`, `wiki/{index,log,overview}.md` 의 `<프로젝트명>` 치환
- 동일 파일들의 `YYYY-MM-DD` 를 오늘 날짜로 치환

### 3. wiki/overview.md 초안 작성

나중에 다시 봐도 한 페이지로 프로젝트를 파악할 수 있도록 손으로 작성한다.
처음에는 거칠어도 된다 — ingest 가 누적되며 LLM 이 보강한다.

### 4. 첫 raw 자료 추가

가장 자신 있는 자료부터 1~2 건 추가:

- README, 아키텍처 다이어그램 → `raw/specs/`
- 최근 의사결정 또는 미팅에서 확정된 선택 → `raw/decisions/YYYYMMDD-<title>.md`
- README, 블로그, 강의, 논문, 책을 보며 얻은 학습 메모
  → `raw/references/<kind>/YYYYMMDD-<topic>.md`
- 작업 중 발견한 흐름·막힌 지점 → `raw/journals/YYYYMMDD-<topic>.md`

### 5. 첫 ingest 명령

[`PROMPTS.md`](./PROMPTS.md) §2 (Ingest) 프롬프트를 그대로 복사해 사용한다.
Claude Code, Codex, ChatGPT 웹, Gemini 등에서 같은 구조로 쓸 수 있다.

### 6. 변경 확인

LLM 변경사항은 `git diff` 로 확인한다. 필요하면 의미 단위로 커밋한다.

## 운영 사이클

| 빈도 | 작업 | 도구 |
| --- | --- | --- |
| 일상 | `raw/` 에 거친 메모 추가 | 손으로 작성 |
| 주간 | `raw/` 신규 항목 ingest | LLM + 직접 검토 |
| 월간 | `wiki/` 전체 lint | LLM + 직접 검토 |

## 검증 체크리스트

- [ ] 나중에 다시 봐도 `wiki/overview.md` 한 페이지로 프로젝트를 이해할 수 있는가?
- [ ] `wiki/runbooks/setup.md` 만으로 로컬 환경을 띄울 수 있는가?
- [ ] 주요 의사결정 5 건 이상이 `wiki/decisions/` 에 ADR 로 정리되었는가?
- [ ] 모든 비자명 주장에 raw 또는 코드 경로 citation 이 있는가?
- [ ] `raw/` 에 비밀·자격증명·고객 PII 가 없는가?
- [ ] `git diff` 검토에서 LLM 의 잘못된 합성을 잡아낸 사례가 있는가?

## 참고 자료 (References)

- [README.md](./README.md) — LLM Wiki 패턴 일반론, 도구 옵션 비교
- Karpathy LLM Wiki gist: <https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f>
