# 06. Skills, Commands, Rules 상세 가이드 및 팀 공유

이 문서는 Roo Code의 커스터마이징 3대 요소(Skills, Commands, Rules)의 작성법, 파일 포맷, 우선순위, 팀 공유 방법을 실무 기준으로 정리한다.

사내 환경 전제:
- 외부 API/마켓플레이스 접근 불가
- 모든 커스터마이징은 Git 저장소 기반으로 팀에 배포
- 로컬 LLM(Kimi-K2.5, GLM-4.7) 사용

---

## 1) 전체 구조 한눈에 보기

```
project-root/
├── .roomodes                          # 프로젝트 커스텀 모드 (YAML/JSON)
├── .roorules                          # 폴백 단일 파일 규칙
├── .roorules-{modeSlug}              # 폴백 모드별 규칙
├── AGENTS.md                          # 크로스 에이전트 규칙 (Cline/Aider 호환)
├── .agents/
│   └── skills/
│       └── {skill-name}/SKILL.md     # 크로스 에이전트 스킬
├── .roo/
│   ├── rules/                         # 프로젝트 규칙 (*.md, *.txt)
│   │   ├── 01-coding-style.md
│   │   └── 02-testing.md
│   ├── rules-{modeSlug}/            # 모드별 프로젝트 규칙
│   ├── skills/                        # 프로젝트 스킬
│   │   └── {skill-name}/
│   │       ├── SKILL.md
│   │       └── template.py           # 번들 리소스 파일
│   ├── skills-{modeSlug}/           # 모드별 프로젝트 스킬
│   ├── commands/                      # 프로젝트 슬래시 명령
│   │   ├── review.md
│   │   └── deploy-check.md
│   └── mcp.json                       # 프로젝트 MCP 설정
```

전역 설정(개인용):
```
~/.roo/
├── rules/                             # 전역 규칙
├── rules-{modeSlug}/                 # 전역 모드별 규칙
├── skills/                            # 전역 스킬
├── skills-{modeSlug}/               # 전역 모드별 스킬
├── commands/                          # 전역 슬래시 명령
```

핵심 원칙:
- **프로젝트(`.roo/`)는 Git으로 팀 공유**, 전역(`~/.roo/`)은 개인 설정
- 같은 이름이 충돌하면 프로젝트 > 전역 우선

---

## 2) Skills (스킬) 상세

### 개념

스킬은 **특정 작업에 특화된 지식 패키지**다. Rules(항상 로드)와 달리 사용자 요청이 스킬의 `description`과 매칭될 때만 활성화된다. 이를 "progressive disclosure"라고 하며, 불필요한 프롬프트 비대화를 방지한다.

Rules와의 핵심 차이:
| 구분 | Skills | Rules | Commands |
|------|--------|-------|----------|
| 활성화 | 요청 매칭 시 자동 | 항상 로드 | 수동 `/` 트리거 |
| 파일 번들 | 가능 (스크립트, 템플릿 등) | 불가 | 불가 |
| 모드 타겟팅 | 가능 | 가능 | 불가 (mode 전환만 가능) |
| 용도 | 작업 워크플로우 | 코딩 표준/규칙 | 반복 프롬프트 |

### 파일 포맷

`SKILL.md` 파일에 YAML frontmatter를 포함한다:

```markdown
---
name: api-doc-generator
description: 사내 REST API 문서를 OpenAPI 3.0 형식으로 자동 생성한다
---
# API 문서 생성 스킬

## 실행 조건
- FastAPI 또는 Flask 프로젝트에서 사용
- `@api-doc` 또는 "API 문서 만들어줘" 요청 시 활성화

## 단계
1. 엔드포인트 라우터 파일 전체 스캔
2. 각 엔드포인트의 method, path, request/response 스키마 추출
3. OpenAPI 3.0 YAML 생성
4. `docs/api/` 폴더에 저장

## 코드 템플릿
아래 bundled 파일을 참조:
- `openapi-template.yaml`: 기본 OpenAPI 뼈대

## 주의사항
- Pydantic v2 모델 기준으로 스키마 추출
- 인증 엔드포인트는 security scheme으로 별도 분리
```

frontmatter 규칙:
- `name`: 디렉토리명과 일치해야 함 (소문자, 하이픈, 1~64자)
- `description`: 1~1024자, Roo가 이 텍스트로 매칭 판단

### 파일 위치 및 우선순위 (높은 순)

1. `.roo/skills-{modeSlug}/{name}/SKILL.md` — 프로젝트 모드별
2. `.roo/skills/{name}/SKILL.md` — 프로젝트 일반
3. `.agents/skills-{modeSlug}/{name}/SKILL.md` — 크로스 에이전트 모드별
4. `.agents/skills/{name}/SKILL.md` — 크로스 에이전트 일반
5. `~/.roo/skills-{modeSlug}/{name}/SKILL.md` — 전역 모드별
6. `~/.roo/skills/{name}/SKILL.md` — 전역 일반
7. `~/.agents/skills-{modeSlug}/{name}/SKILL.md` — 전역 크로스 에이전트 모드별
8. `~/.agents/skills/{name}/SKILL.md` — 전역 크로스 에이전트 일반

### 번들 파일

스킬 디렉토리 안에 보조 파일을 함께 넣을 수 있다:

```
.roo/skills/api-doc-generator/
├── SKILL.md
├── openapi-template.yaml      # 템플릿 파일
└── generate-schema.py         # 헬퍼 스크립트
```

SKILL.md 안에서 이 파일들을 참조하면 Roo가 함께 읽는다.

### 사내 실전 스킬 예제

**스킬 1: DRM 문서 추출**
```
.roo/skills/drm-extract/
├── SKILL.md
└── extract_template.py
```

```markdown
---
name: drm-extract
description: DRM 보호된 PPTX/XLSX를 스크린샷+VLM으로 텍스트 추출한다
---
# DRM 문서 추출 스킬

## 전제
- PowerPoint/Excel COM 자동화 가능 환경 (Windows)
- VLM API: Qwen3-VL-8B-Instruct (사내 엔드포인트)

## 워크플로우
1. COM으로 파일 열기 → 슬라이드/시트별 스크린샷
2. 스크린샷을 VLM에 전송하여 텍스트 추출
3. 추출 결과를 Markdown으로 정리

## 코드 패턴
bundled `extract_template.py` 참조
```

**스킬 2: 사내 코드 리뷰 표준**
```markdown
---
name: code-review-standard
description: 사내 코딩 표준에 따라 Python 코드를 리뷰한다
---
# 코드 리뷰 스킬

## 체크리스트
1. 타입 힌트 완전성
2. docstring 존재 여부 (public 함수)
3. 에러 처리 패턴 (bare except 금지)
4. 로깅 레벨 적정성
5. 보안 취약점 (SQL injection, path traversal 등)

## 출력 포맷
| 파일:라인 | 심각도 | 지적 내용 | 수정 제안 |
```

---

## 3) Slash Commands (슬래시 명령) 상세

### 개념

슬래시 명령은 **재사용 가능한 프롬프트 템플릿**이다. 채팅 입력창에 `/`를 치면 등록된 명령 목록이 표시되며, 선택하면 해당 프롬프트가 즉시 실행된다.

### 파일 포맷

마크다운 파일에 선택적 YAML frontmatter를 붙인다. **파일명이 곧 명령어 이름**이다.

`review.md` → `/review`로 사용:

```markdown
---
description: 현재 변경사항을 사내 코딩 표준으로 리뷰
argument-hint: <파일경로 또는 빈칸>
mode: code
---
다음 기준으로 코드 리뷰를 수행해줘:

1. 사내 Python 코딩 표준 준수 여부
2. 타입 안전성
3. 에러 핸들링 적정성
4. 보안 취약점
5. 성능 이슈

각 지적은 `파일:라인 | 심각도 | 내용 | 수정안` 테이블로 정리.
변경사항이 없으면 git diff 기준으로 분석.
```

frontmatter 필드:
- `description`: 명령 메뉴에 표시되는 설명
- `argument-hint`: 입력창에 회색 힌트로 표시 (사용자가 추가 인자를 입력할 수 있음)
- `mode`: 명령 실행 전 이 모드로 자동 전환 (선택사항)

### 파일명 규칙

- 파일명에서 명령어 이름이 자동 생성됨
- 자동 정규화: "My Cool Command!" → `my-cool-command` → `/my-cool-command`
- 소문자 + 하이픈 사용 권장

### 파일 위치 및 우선순위

| 우선순위 | 위치 | 범위 |
|----------|------|------|
| 1 (최고) | `.roo/commands/` | 프로젝트 |
| 2 | `~/.roo/commands/` | 전역 |
| 3 (최저) | 내장 명령 | 시스템 |

프로젝트 명령이 전역 명령보다 우선. 내장 명령(`/init`, `/code`, `/ask` 등)은 오버라이드 불가.

### 프로그래밍 실행

Roo는 `run_slash_command` 도구를 통해 슬래시 명령을 프로그래밍적으로 실행할 수 있다. 이를 활용하면 명령 체이닝이 가능하다:

```
/architect로 설계 → /review로 리뷰 → /deploy-check로 배포 점검
```

### 사내 실전 명령 예제

**`release-check.md`**:
```markdown
---
description: 릴리즈 전 최종 점검 체크리스트 실행
mode: architect
---
릴리즈 전 최종 점검을 수행해줘:

1. 미완료 TODO/FIXME 검색
2. 테스트 커버리지 확인 (pytest --cov)
3. 의존성 버전 고정 확인 (requirements.txt)
4. CHANGELOG.md 업데이트 여부
5. 보안 취약점 스캔 결과

각 항목을 PASS/FAIL/WARN으로 표시하고 종합 판정.
```

**`onboard.md`**:
```markdown
---
description: 이 프로젝트에 대한 온보딩 브리핑 생성
mode: ask
---
이 프로젝트를 처음 보는 개발자를 위해 다음을 정리해줘:

1. 프로젝트 목적과 아키텍처 개요
2. 핵심 엔트리포인트 3개와 역할
3. 로컬 개발환경 셋업 절차
4. 주요 의존성과 사내 시스템 연동 포인트
5. 자주 사용하는 명령어 (빌드, 테스트, 배포)
```

**`debug-error.md`**:
```markdown
---
description: 에러 로그 기반 디버깅 시작
argument-hint: <에러 메시지 또는 로그 붙여넣기>
mode: debug
---
아래 에러를 분석해줘:

1. 에러 유형 분류 (런타임/설정/네트워크/권한)
2. 원인 가설 3개 (가능성 높은 순)
3. 각 가설 검증 방법
4. 수정 코드 제안
5. 재발 방지 조치
```

---

## 4) Custom Instructions / Rules (규칙) 상세

### 개념

Rules는 **항상 시스템 프롬프트에 로드되는 지시사항**이다. 코딩 스타일, 문서화 표준, 테스트 요구사항 등을 정의한다.

### 파일 포맷

일반 Markdown(`.md`) 또는 텍스트(`.txt`) 파일. frontmatter 불필요.

```markdown
# Python 코딩 표준

- 모든 public 함수에 타입 힌트 필수
- bare except 금지, 구체적 예외 타입 명시
- f-string 사용 (format() 지양)
- import 순서: stdlib → third-party → local (isort 기준)
- 로깅은 structlog 사용, print() 금지
```

### 파일 위치 및 로드 순서

규칙은 아래 순서로 결합되어 시스템 프롬프트에 포함된다:

1. Prompts Tab의 전체 모드 지시
2. Prompts Tab의 모드별 지시
3. `~/.roo/rules-{modeSlug}/` + `.roo/rules-{modeSlug}/` (모드별)
4. `.roorules-{modeSlug}` (모드별 폴백)
5. `~/.roo/rules/` + `.roo/rules/` (일반)
6. `.roorules` (일반 폴백)

파일 정렬:
- 디렉토리 내 파일은 **알파벳순(대소문자 무시)** 정렬
- `01-style.md`, `02-testing.md` 처럼 번호 prefix로 순서 제어 가능

### AGENTS.md (크로스 에이전트)

프로젝트 루트의 `AGENTS.md`는 자동 로드된다. Cline, Aider 등 다른 AI 코딩 도구에서도 인식하는 표준이므로, 팀에서 여러 도구를 혼용할 때 유용하다.

비활성화: `"roo-cline.useAgentRules": false`

### 사내 실전 규칙 예제

**`.roo/rules/01-general.md`**:
```markdown
# 사내 개발 규칙

- 모든 코드와 주석은 한국어 또는 영어로 작성 (혼용 가능)
- 외부 API 호출 금지 (사내 엔드포인트만 사용)
- 민감 정보(IP, 토큰, 사번 등)는 코드에 하드코딩 금지
- 환경변수 또는 사내 시크릿 매니저 사용
```

**`.roo/rules-code/01-python.md`** (Code 모드 전용):
```markdown
# Code 모드 Python 규칙

- Python 3.10+ 문법 사용
- async 코드는 asyncio 기반 (trio 금지)
- DB 쿼리는 반드시 파라미터 바인딩 사용
- 새 함수 작성 시 단위 테스트도 함께 작성
```

---

## 5) Custom Modes (커스텀 모드)와 연동

모드는 Skills/Commands/Rules를 묶는 상위 개념이다. 모드별로 도구 권한을 제한하고, 모드별 규칙/스킬을 분리할 수 있다.

### 파일 포맷 (YAML 권장)

`.roomodes`:
```yaml
customModes:
  - slug: reviewer
    name: "Code Reviewer"
    description: "코드 리뷰 전용 모드"
    roleDefinition: >
      당신은 시니어 코드 리뷰어입니다. 코드 품질, 보안, 성능을
      사내 표준 기준으로 검토합니다.
    whenToUse: >
      코드 리뷰, PR 검토, 품질 점검 시 사용
    customInstructions: >
      수정 제안은 반드시 diff 형식으로 제시.
      심각도를 Critical/Major/Minor로 분류.
    groups:
      - read
      - mcp

  - slug: doc-writer
    name: "Documentation Writer"
    description: "문서 작성 전용 모드"
    roleDefinition: >
      당신은 테크니컬 라이터입니다. 명확하고 구조적인
      기술 문서를 작성합니다.
    groups:
      - read
      - [edit, {fileRegex: '\.(md|mdx|txt|rst)$', description: 'Markdown/Text만 편집'}]

  - slug: safe-coder
    name: "Safe Coder"
    description: "제한된 경로만 편집 가능한 안전 코딩 모드"
    roleDefinition: >
      당신은 개발자입니다. 지정된 디렉토리 내에서만 코드를 수정합니다.
    groups:
      - read
      - [edit, {fileRegex: '^src/.*', description: 'src/ 하위만 편집'}]
      - [command, {allowedCommands: ['pytest', 'python', 'pip']}]
```

### 모드별 규칙/스킬 연동

`reviewer` 모드를 만들었다면:
- `.roo/rules-reviewer/` — 이 모드에서만 적용되는 규칙
- `.roo/skills-reviewer/` — 이 모드에서만 활성화되는 스킬

이 구조를 활용하면 모드 전환만으로 Roo의 행동 방식이 완전히 바뀐다.

---

## 6) 팀 공유 전략

### 방법 1: Git 저장소 커밋 (권장)

가장 간단하고 확실한 방법. `.roo/` 디렉토리를 프로젝트에 커밋한다:

```bash
git add .roo/ .roomodes AGENTS.md
git commit -m "Add Roo Code team configuration"
```

공유되는 것:
| 항목 | 파일/경로 | 효과 |
|------|-----------|------|
| Rules | `.roo/rules/` | 팀 코딩 표준 자동 적용 |
| Skills | `.roo/skills/` | 작업 스킬 공유 |
| Commands | `.roo/commands/` | 슬래시 명령 공유 |
| Modes | `.roomodes` | 커스텀 모드 공유 |
| MCP | `.roo/mcp.json` | MCP 서버 설정 공유 |
| Cross-agent | `AGENTS.md` | Cline/Aider 사용자도 동일 규칙 |

### 방법 2: 팀 템플릿 저장소

사내 Git에 Roo 설정 전용 템플릿 저장소를 만들고, 새 프로젝트 생성 시 복사한다:

```
roo-team-config/        # 사내 템플릿 저장소
├── .roo/
│   ├── rules/
│   │   ├── 01-company-standard.md
│   │   └── 02-security.md
│   ├── skills/
│   │   ├── api-doc-generator/
│   │   └── drm-extract/
│   ├── commands/
│   │   ├── review.md
│   │   ├── onboard.md
│   │   └── release-check.md
│   └── mcp.json
├── .roomodes
└── README.md           # 사용법 안내
```

새 프로젝트에 적용:
```bash
# 템플릿에서 설정 복사
cp -r /path/to/roo-team-config/.roo ./
cp /path/to/roo-team-config/.roomodes ./
```

### 방법 3: 전역 설정 자동 임포트

개인 머신 간 설정 동기화 또는 팀 공통 전역 설정 배포:

VS Code `settings.json`:
```json
{
  "roo-cline.autoImportSettingsPath": "//shared-drive/roo-config/settings.json"
}
```

시작 시 지정 경로의 설정을 자동 머지한다. 사내 공유 드라이브/NAS에 올려두면 팀 전체에 적용 가능.

주의: 이 방법은 API 프로필/키를 포함할 수 있으므로 보안 검토 필요.

### 방법 4: 모드 Import/Export (UI)

커스텀 모드를 YAML 파일로 내보내고, 동료가 UI에서 가져올 수 있다:

1. 모드 설정 화면 → Export → YAML 파일 저장
2. 동료에게 파일 전달
3. 동료가 Import → 프로젝트 또는 전역 수준 선택

소규모 공유나 1회성 전달에 적합.

---

## 7) 사내 환경 맞춤 권장 구성

로컬 LLM 환경에서의 추천 `.roo/` 초기 세트:

```
.roo/
├── rules/
│   ├── 01-language.md          # "한국어 기본, 기술용어 영어 병기"
│   ├── 02-no-external-api.md   # "외부 API 호출 금지, 사내 엔드포인트만"
│   └── 03-security.md          # "민감정보 하드코딩 금지"
├── skills/
│   ├── drm-extract/            # DRM 문서 → VLM 추출
│   └── internal-api-doc/       # 사내 API 문서 생성
├── commands/
│   ├── review.md               # 코드 리뷰
│   ├── onboard.md              # 프로젝트 온보딩
│   └── check-deps.md           # 의존성 점검
└── mcp.json                    # 사내 MCP 서버 (문서검색, Jira 등)
```

`.roomodes`에 최소 3개 커스텀 모드:
- `reviewer`: 읽기 + MCP만 (편집 금지)
- `doc-writer`: 읽기 + `.md` 편집만
- `safe-coder`: `src/` 하위만 편집 + 제한된 명령만

---

## 8) 자주 묻는 질문

**Q: Skills와 Commands의 차이가 뭔가?**
- Commands는 사용자가 `/review`처럼 직접 트리거. 단순 프롬프트 주입.
- Skills는 사용자 요청과 description이 매칭될 때 자동 활성화. 번들 파일 포함 가능.

**Q: Rules가 너무 많으면 토큰 낭비 아닌가?**
- Rules는 항상 로드되므로 맞다. 핵심 규칙만 Rules에 넣고, 상세 워크플로우는 Skills로 분리하는 것이 바람직하다.

**Q: `.roo/`와 `.agents/` 둘 다 있으면?**
- `.roo/`가 우선. `.agents/`는 크로스 에이전트 호환용 폴백이다.

**Q: 로컬 LLM에서 Skills 매칭이 잘 안 되면?**
- description을 더 구체적으로 작성한다. 로컬 모델은 추론 능력이 제한적일 수 있으므로 키워드를 명확히 넣는다.
- 매칭이 안 될 때는 Commands로 대체하여 수동 트리거한다.

**Q: 동료가 Cline을 쓰는데 같은 규칙을 공유할 수 있나?**
- `AGENTS.md`와 `.agents/skills/`를 사용하면 Cline/Aider에서도 인식한다.

---

## 참고 자료

- [Custom Instructions (Rules)](https://docs.roocode.com/features/custom-instructions)
- [Skills](https://docs.roocode.com/features/skills)
- [Custom Modes](https://docs.roocode.com/features/custom-modes)
- [Slash Commands](https://docs.roocode.com/features/slash-commands)
- [MCP in Roo Code](https://docs.roocode.com/features/mcp/using-mcp-in-roo)
- [Settings Management](https://docs.roocode.com/features/settings-management)

## 관련 문서
- [00_company_local_llm_setup.md](./00_company_local_llm_setup.md) — 사내 API 연결 기본
- [02_tutorial_topics_core.md](./02_tutorial_topics_core.md) — 핵심 주제 요약
- [03_mcp_marketplace_and_browser.md](./03_mcp_marketplace_and_browser.md) — MCP 상세
