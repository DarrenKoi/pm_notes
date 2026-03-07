---
tags: [codex, cli, agents, skills, instructions]
level: intermediate
last_updated: 2026-03-07
status: complete
---

# 에이전트와 스킬 (Agents & Skills)

> Codex를 잘 쓰려면 "프롬프트만 잘 쓰는 법"보다 "Codex가 어떤 규칙으로 움직이는지"를 이해해야 한다.

## 먼저 구분

| 개념 | 역할 | 보통 어디에 있는가 |
|------|------|--------------------|
| 에이전트(Agent) | 현재 작업을 수행하는 Codex 자체 | CLI 세션 안 |
| `AGENTS.md` | 저장소별 작업 규칙 문서 | 프로젝트 루트 또는 상위 디렉토리 |
| Skill | 특정 작업을 더 잘 수행하게 하는 재사용 가능한 지식/워크플로우 묶음 | 보통 `~/.codex/skills/<skill-name>/` |
| `SKILL.md` | 각 skill의 핵심 정의 파일 | skill 폴더 내부 |

핵심 차이:

- `AGENTS.md`는 "이 저장소에서는 이렇게 일해라"를 정의한다
- Skill은 "이 종류의 작업이면 이 전문 지식과 절차를 써라"를 정의한다

## Agent는 무엇인가?

Codex CLI에서 작업하는 주체가 에이전트다. 단순히 답만 만드는 모델이 아니라:

- 저장소를 읽고
- 필요한 파일을 찾고
- 셸 명령을 실행하고
- 파일을 수정하고
- 결과를 검증하는

작업형 실행 단위라고 보면 된다.

즉, `codex`는 채팅 인터페이스가 아니라 "규칙과 도구를 가진 에이전트"다.

## `AGENTS.md`는 왜 중요한가?

`AGENTS.md`는 특정 저장소에서 Codex가 따라야 하는 로컬 작업 규칙이다.

이 저장소에도 실제로 상위 경로에 [../AGENTS.md](/Users/daeyoung/Codes/pm_notes/AGENTS.md)가 있고, 예를 들어 이런 내용을 담고 있다:

- 프로젝트 구조 설명
- 모듈별 실행 명령
- Python/Vue/Markdown 스타일 규칙
- 테스트 기대치
- 커밋/PR 작성 원칙
- 보안/설정 주의사항

이런 파일이 있으면 Codex는 매번 같은 규칙을 다시 설명받지 않아도 된다.

## `AGENTS.md`에 넣기 좋은 내용

### 1. 저장소 구조

```md
- `backend/`: Flask API
- `frontend/`: Nuxt app
- `docs/`: 운영 문서
```

### 2. 실행/검증 명령

```md
- Backend: `uv run src/app.py`
- Frontend: `npm run dev`
- Validation: `npm run lint`, `npm run typecheck`
```

### 3. 스타일 규칙

```md
- Python: 4 spaces, type hints for changed code
- Vue: follow ESLint + Nuxt defaults
- Markdown: concise sections and relative links
```

### 4. 안전장치

```md
- Do not commit secrets
- Review config changes carefully
- Never change generated files unless requested
```

## `AGENTS.md`를 잘 쓰는 원칙

1. 저장소 공통 규칙만 넣는다
2. 자주 반복해서 설명하는 내용을 넣는다
3. 짧고 구체적으로 쓴다
4. 오래된 명령은 즉시 갱신한다

좋은 후보:

- 항상 필요한 실행 명령
- 팀 코딩 규칙
- 테스트 기본 정책
- 수정 금지 영역

좋지 않은 후보:

- 일회성 작업 메모
- 너무 긴 배경 설명
- 이미 README에만 있어도 충분한 장황한 튜토리얼

## Skill은 무엇인가?

Skill은 Codex에 특정 분야의 작업 절차를 추가하는 모듈이다.

이 환경의 예:

- `doc`: `.docx` 읽기/편집/렌더링 작업용
- `journal`: 세션 내용을 한국어 작업 일지로 저장하는 용도

Skill은 보통 다음을 포함한다:

```text
skill-name/
├── SKILL.md
├── scripts/
├── references/
└── assets/
```

## `SKILL.md`는 어떤 역할인가?

`SKILL.md`는 skill의 진입점이다. 보통 아래를 정의한다:

- skill 이름
- 언제 이 skill을 써야 하는지
- 기본 워크플로우
- 필요한 도구/의존성
- 품질 기준
- 필요하면 추가 참고 파일 위치

예를 들어 `doc` skill은:

- `.docx` 작업일 때 사용
- `python-docx`를 우선 사용
- 가능하면 렌더링으로 시각 검증
- 중간 산출물 위치와 최종 검증 규칙

을 명시하고 있다.

## Skill은 언제 발동하는가?

이 환경 기준으로는 다음 경우 skill을 사용한다:

1. 사용자가 skill 이름을 직접 언급할 때
2. 요청이 특정 skill 설명과 명확히 맞을 때

예시:

- "`journal`로 오늘 작업 로그 남겨줘"
- "이 세션 내용을 한국어 일지로 저장해줘" → `journal` skill
- "docx 문서 레이아웃 유지하면서 수정해줘" → `doc` skill

즉, skill은 "항상 켜진 규칙"이 아니라 "특정 작업에서 불러오는 전문 모드"에 가깝다.

## `AGENTS.md`와 Skill의 차이

| 항목 | `AGENTS.md` | Skill |
|------|-------------|-------|
| 범위 | 특정 저장소/디렉토리 | 특정 작업 유형 |
| 목적 | 저장소 규칙 강제 | 전문 워크플로우 재사용 |
| 적용 시점 | 그 저장소에서 작업할 때 지속 적용 | 트리거될 때만 적용 |
| 내용 | 구조, 실행 명령, 코딩 규칙, 안전 정책 | 절차, 스크립트, 참고 자료, 전용 자산 |

간단히 말하면:

- `AGENTS.md`는 "이 프로젝트의 헌법"
- Skill은 "특수 임무용 플레이북"

## 실제 운영 방식 추천

### 저장소에는 `AGENTS.md`

이런 내용을 둔다:

- 어떻게 빌드하는지
- 무엇을 검증해야 하는지
- 어떤 스타일을 따라야 하는지
- 무엇을 건드리면 안 되는지

### 반복 작업은 Skill로 분리

이런 경우 skill 후보가 된다:

- 내부 API 문서 포맷 생성
- 특정 보고서 템플릿 작성
- 사내 배포 절차 자동화
- 특정 로그/장애 분석 패턴

## 언제 무엇을 만들면 좋은가?

`AGENTS.md`가 맞는 경우:

- 같은 저장소에서 누구나 지켜야 하는 규칙
- 거의 모든 작업에 공통으로 적용되는 지침

Skill이 맞는 경우:

- 특정 작업군에서만 필요한 절차
- 스크립트, 참조 문서, 템플릿을 묶어 재사용하고 싶은 경우
- 여러 저장소에서 공통으로 재사용할 수 있는 작업 지식

## 실전 팁

1. 먼저 `AGENTS.md`로 저장소 규칙을 고정한다
2. 반복되는 전문 작업이 생기면 skill로 분리한다
3. Skill은 짧은 `SKILL.md` + 필요한 참조 파일로 구성한다
4. 둘 다 길게 쓰기보다 "Codex가 실행에 필요한 정보"만 남긴다

## 관련 문서

- [Codex CLI 실전 가이드](./README.md)
- 이전: [세션 재사용, MCP, 고급 기능](./04_sessions_mcp_advanced.md)
