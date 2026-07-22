---
tags: [ai-coding, documentation, agents, openwiki]
level: beginner
last_updated: 2026-07-07
---

# OpenWiki 사용법

> OpenWiki는 코드베이스를 읽어 에이전트가 참고할 수 있는 저장소 문서를 생성하고 유지하는 CLI 도구다.

## 한 줄 요약

OpenWiki는 저장소 안에 `openwiki/` 문서 폴더를 만들고, 이후 코드 변경에 맞춰 그 문서를 갱신하는 도구다. 일반 위키 서버라기보다, Codex/Claude Code 같은 코딩 에이전트가 저장소 맥락을 빠르게 찾도록 돕는 "에이전트용 코드베이스 위키 생성기"에 가깝다.

## 언제 쓰면 좋은가?

- 새 저장소를 빠르게 파악하기 위한 구조 문서를 만들고 싶을 때
- 여러 에이전트가 같은 코드베이스를 다룰 때 공통 참조 문서를 두고 싶을 때
- `README.md`보다 더 세부적인 모듈/아키텍처/운영 문서를 자동으로 유지하고 싶을 때
- CI에서 정기적으로 문서 갱신 PR을 만들고 싶을 때

반대로 작은 스크립트 몇 개뿐인 저장소, 보안상 LLM에 코드 내용을 보내기 어려운 저장소, 이미 사람이 잘 관리하는 상세 설계 문서가 있는 저장소에는 우선순위가 낮다.

## 설치 전 확인

- Node.js 20 이상이 필요하다. 공식 GitHub Actions 예시는 Node.js 22를 사용한다.
- LLM API 키가 필요하다. 기본 제공 provider는 OpenRouter, Fireworks, Baseten, OpenAI, OpenAI-compatible, Anthropic이다.
- OpenWiki 실행 결과는 저장소 안의 `openwiki/` 폴더를 수정할 수 있다.
- 첫 설정 값과 API 키는 로컬 `~/.openwiki/.env`에 저장된다.
- `AGENTS.md` 또는 `CLAUDE.md`가 있으면 OpenWiki 문서를 참고하라는 프롬프트를 추가할 수 있고, 파일이 없으면 생성할 수 있다.

## 설치

```bash
npm install -g openwiki
```

설치 후 도움말을 확인한다.

```bash
openwiki --help
```

## 첫 실행

저장소 루트에서 초기화를 실행한다.

```bash
cd /path/to/your/repo
openwiki --init
```

첫 interactive 실행에서는 provider, API key, model ID를 설정한다. 설정은 `~/.openwiki/.env`에 저장된다.

비대화형 환경에서는 환경 변수로 설정한다.

```bash
OPENWIKI_PROVIDER=openai \
OPENAI_API_KEY=... \
OPENWIKI_MODEL_ID=gpt-5.5 \
openwiki --init --print
```

## 기본 명령

```bash
# 대화형 CLI 시작
openwiki

# 초기 요청과 함께 실행
openwiki "Please generate documentation for this repository"

# 한 번만 실행하고 결과 출력 후 종료
openwiki -p "Summarize what you can do"

# 초기 문서 생성
openwiki --init

# 기존 openwiki/ 문서 갱신
openwiki --update

# 특정 모델로 실행
openwiki --modelId openai/gpt-5.5

# 기존 문서를 갱신하면서 우선순위 지시
openwiki --update --modelId openai/gpt-5.5 "Please document the API routes first"
```

대화형 CLI 안에서는 `/provider`, `/model`, `/init`, `/update`, `/clear`, `/help`, `/exit` 명령을 사용할 수 있다.

## Provider 설정 예시

### OpenRouter

OpenWiki의 기본 provider는 OpenRouter다.

```bash
OPENWIKI_PROVIDER=openrouter
OPENROUTER_API_KEY=...
OPENWIKI_MODEL_ID=z-ai/glm-5.2
```

### OpenAI

```bash
OPENWIKI_PROVIDER=openai
OPENAI_API_KEY=...
OPENWIKI_MODEL_ID=gpt-5.5
```

### Anthropic 호환 게이트웨이

Anthropic provider를 프록시나 사내 게이트웨이로 보내려면 `ANTHROPIC_BASE_URL`을 함께 둔다.

```bash
OPENWIKI_PROVIDER=anthropic
ANTHROPIC_API_KEY=...
ANTHROPIC_BASE_URL=https://your-gateway.example.com/anthropic
OPENWIKI_MODEL_ID=claude-sonnet-5
```

### OpenAI-compatible 엔드포인트

LiteLLM 같은 OpenAI-compatible gateway를 쓸 때는 base URL이 필수다.

```bash
OPENWIKI_PROVIDER=openai-compatible
OPENAI_COMPATIBLE_API_KEY=...
OPENAI_COMPATIBLE_BASE_URL=https://your-gateway.example.com/v1
OPENWIKI_MODEL_ID=your-gateway-model-name
```

## CI로 자동 갱신하기

GitHub Actions에서는 다음 흐름을 사용한다.

1. repository checkout
2. Node.js 22 설정
3. `npm install --global openwiki`
4. `openwiki --update --print`
5. 변경된 `openwiki/` 폴더를 PR로 생성

최소 예시는 다음과 같다.

```yaml
name: OpenWiki Update

on:
  workflow_dispatch:
  schedule:
    - cron: "0 8 * * *"

permissions:
  contents: write
  pull-requests: write

jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          persist-credentials: true

      - uses: actions/setup-node@v4
        with:
          node-version: "22"

      - name: Install OpenWiki
        run: npm install --global openwiki

      - name: Run OpenWiki
        run: openwiki --update --print
        env:
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
          OPENWIKI_MODEL_ID: z-ai/glm-5.2
          LANGSMITH_API_KEY: ${{ secrets.LANGSMITH_API_KEY }}
          LANGCHAIN_PROJECT: openwiki
          LANGCHAIN_TRACING_V2: "true"

      - name: Create OpenWiki update pull request
        uses: peter-evans/create-pull-request@v7
        with:
          add-paths: openwiki
          branch: openwiki/update
          commit-message: "docs: update OpenWiki"
          title: "docs: update OpenWiki"
          body: Automated OpenWiki documentation update.
```

사내망이나 보안 저장소에서는 CI에 API 키를 넣기 전에 코드 반출 정책을 먼저 확인해야 한다.

## pm_notes에서 시험해볼 때

이 저장소에서 바로 시험한다면 먼저 별도 브랜치를 만들고 실행하는 것이 안전하다.

```bash
cd /Users/daeyoung/Codes/pm_notes
git switch -c docs/openwiki-trial
npm install -g openwiki
openwiki --init
git status --short
```

실행 후에는 최소한 다음 파일을 확인한다.

- `openwiki/`: 생성된 저장소 위키 본문
- `AGENTS.md` 또는 `CLAUDE.md`: OpenWiki 참고 지시가 추가됐는지
- `~/.openwiki/.env`: 로컬에 저장된 provider/API key/model 설정

생성 결과가 마음에 들지 않으면 커밋하지 않고 변경분을 폐기하면 된다. 단, 이미 있던 `AGENTS.md`/`CLAUDE.md`가 수정될 수 있으므로 diff를 확인한 뒤 정리한다.

## 운영 팁

- 처음에는 `--init`으로 큰 구조를 만들고, 이후에는 `--update`만 반복한다.
- API 문서, 배치 작업, 프론트엔드 라우트처럼 우선 문서화할 영역이 있으면 초기 요청에 명시한다.
- 자동 갱신 PR은 사람이 리뷰한다. 에이전트가 만든 문서는 오래된 가정이나 과도한 일반화를 포함할 수 있다.
- API 키는 저장소에 커밋하지 않는다. 로컬은 `~/.openwiki/.env`, CI는 secret store를 사용한다.
- OpenWiki가 생성한 문서를 다른 에이전트 지시 파일과 함께 쓸 때는 중복되거나 충돌하는 운영 규칙이 없는지 확인한다.

## 참고 자료

- OpenWiki GitHub: <https://github.com/langchain-ai/openwiki>
- OpenWiki README: <https://github.com/langchain-ai/openwiki/blob/main/README.md>
- GitHub Actions 예시: <https://github.com/langchain-ai/openwiki/blob/main/examples/openwiki-update.yml>
- GitLab CI 예시: <https://github.com/langchain-ai/openwiki/blob/main/examples/openwiki-update.gitlab-ci.yml>
- package.json: <https://github.com/langchain-ai/openwiki/blob/main/package.json>
