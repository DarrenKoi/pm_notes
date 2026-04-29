---
tags: [llm-wiki, obsidian, local-llm, ollama, plugin]
level: beginner
last_updated: 2026-04-29
status: complete
---

# Obsidian + Local LLM API 연동 가이드

> 사내 환경처럼 Local LLM API만 사용 가능한 상황에서, Obsidian을
> "LLM Wiki" 프론트엔드로 활용하기 위한 플러그인 비교와 연결 방법.

## 왜 필요한가? (Why)

- 회사 정책상 OpenAI/Anthropic 같은 외부 클라우드 LLM은 호출 불가, 사내 호스팅
  Local LLM API(OpenAI 호환 엔드포인트)만 사용 가능한 환경.
- LLM Wiki 패턴(`llm-wiki/README.md`)은 Markdown 기반이어서 Obsidian과 매우
  궁합이 좋음. Vault 자체를 `wiki/` 디렉토리로 두면 그래프 뷰, 백링크,
  태그 검색을 그대로 활용할 수 있음.
- Claude Code/Codex로 ingest·lint를 돌리는 동안, **사용자가 노트를 읽고
  질문하는 인터페이스**는 Obsidian 안에서 끝내는 것이 가장 마찰이 적음.
- 단, Obsidian의 LLM 플러그인 대부분이 OpenAI SaaS를 기본 가정으로
  설계되어 있어, "OpenAI 호환 Local 엔드포인트"를 받아주는 플러그인을
  제대로 골라야 함.

## 핵심 개념 (What)

### "Local LLM API"의 사실상 표준 = OpenAI 호환

대부분의 사내 LLM 게이트웨이는 OpenAI Chat Completions 스펙을 흉내 낸다.
즉, 다음 형태의 엔드포인트를 노출한다.

```
POST {BASE_URL}/v1/chat/completions
Authorization: Bearer <token>
Content-Type: application/json
```

이 점이 중요한 이유: Obsidian 플러그인이 **"Custom OpenAI-compatible
endpoint"**, **"Base URL"**, **"OpenAI Format"** 같은 옵션을 가지고 있다면
사내 LLM에 거의 그대로 붙는다. 반대로 프로바이더 드롭다운에 `OpenAI`,
`Anthropic`, `Google`만 있고 Base URL 입력이 없으면 못 쓴다고 보면 된다.

### 후보 플러그인 한눈에 비교

| 플러그인 | OpenAI 호환 Custom Endpoint | Ollama 직결 | 노트 컨텍스트(RAG/임베딩) | 적합한 용도 |
| --- | --- | --- | --- | --- |
| **Copilot** (logancyang/obsidian-copilot) | O (Model 추가 시 Base URL 입력) | O (네이티브) | O (Vault QA, Relevant Notes) | "Vault 안에서 채팅하며 위키를 같이 읽고 싶다" |
| **Smart Connections** (brianpetro) | O ("Custom Local (OpenAI format)") | O (`http://localhost:11434`) | O (임베딩 기반 관련 노트 자동 표시) | 노트 간 연결/추천, 임베딩 검색 |
| **Local LLM Helper** (manimohans) | O (모든 OpenAI 호환 서버) | O (Ollama, LM Studio, vLLM) | 일부 (semantic search) | 텍스트 변환(요약/번역/액션아이템 추출)을 명령으로 |
| **Local GPT** (pfrankov/obsidian-local-gpt) | O (OpenAI-like) | O | X (선택 텍스트 단위 작업) | 단축키 기반 인라인 텍스트 작업 |
| **Text Generator** (nhaouari) | O (`Custom`, OpenAI Format) | △ (Custom 엔드포인트로) | X | 템플릿 기반 글 생성, 프롬프트 노트 |
| **LLM Wiki Local** (kytmanov) | O (OpenAI 호환) | O (Ollama 기본) | Karpathy LLM Wiki 패턴 자체 구현 | 본 저장소의 LLM Wiki 패턴을 Obsidian 안에서 자동화 |

> 우리 맥락(Local LLM API만 사용, LLM Wiki 운영)에서 **1순위는 Copilot 또는
> Smart Connections**, 위키 자동 성장까지 노린다면 **LLM Wiki Local**을
> 추가로 검토.

### Obsidian이 막히는 단골 이슈: CORS

Obsidian Desktop은 Electron 앱이라 내부적으로 브라우저 fetch를 쓴다. 사내
LLM 게이트웨이나 Ollama가 CORS 헤더(`Access-Control-Allow-Origin`)를 안
주면 플러그인에서 호출이 막힌다. 해결 방법은 두 가지다.

1. 서버 쪽에서 Obsidian Origin 허용
   - Ollama: `OLLAMA_ORIGINS=app://obsidian.md*` 환경 변수
   - 사내 게이트웨이: 운영팀에 `app://obsidian.md*` Origin 허용 요청
2. 플러그인의 "CORS 우회/프록시" 토글 사용
   - Copilot: 모델 행의 CORS 토글을 켜면 내장 프록시로 우회

## 어떻게 사용하는가? (How)

### 0. 사전 준비 — 사내 Local LLM API 정보 확인

다음 4가지를 운영팀에서 확보한다.

- Base URL (예: `https://llm.intra.company.com`)
- Chat 엔드포인트 경로 (대부분 `/v1/chat/completions`)
- Embedding 엔드포인트 경로 (RAG/Smart Connections에 필요, 보통
  `/v1/embeddings`. 없으면 임베딩만 로컬 Ollama로 따로 띄운다)
- 인증 방식 (대개 `Authorization: Bearer <token>`)

테스트는 Obsidian 밖에서 먼저:

```bash
curl -sS "$BASE_URL/v1/chat/completions" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "company-internal-llm",
    "messages": [{"role": "user", "content": "ping"}]
  }'
```

응답이 OpenAI 포맷(`choices[0].message.content`)으로 오면 OK.

### 1. Copilot (1순위 추천)

가장 폴리시가 잘 된 옵션. Vault 전체를 컨텍스트로 채팅 가능.

#### 설치

1. Obsidian → Settings → Community plugins → Browse → `Copilot` 검색
2. Install → Enable

#### 사내 Local LLM API 등록

1. Settings → Copilot → **Model** 탭 → **Add Custom Model**
2. 다음과 같이 입력:

| 필드 | 값 |
| --- | --- |
| Model Name | `company-internal-llm` (서버에서 받는 model 이름) |
| Display Name | `Company LLM` |
| Provider | `OpenAI Format` (또는 `Custom`) |
| Base URL | `https://llm.intra.company.com/v1` |
| API Key | 사내 발급 토큰. 토큰이 없으면 임의 문자열 (`local`) |
| CORS | 호출 실패 시 토글 ON (내장 프록시 경유) |

3. **Default Chat Model**과 **Default QA Model**을 위 모델로 지정.
4. Embedding 모델: 사내 임베딩이 있으면 같은 방식으로 등록, 없으면
   로컬 Ollama로 `nomic-embed-text` 정도를 띄워 별도로 연결.

#### Ollama로 로컬에서 띄우는 경우

```bash
# macOS: Obsidian Origin 허용
launchctl setenv OLLAMA_ORIGINS "app://obsidian.md*"

# Linux/직접 실행
OLLAMA_ORIGINS="app://obsidian.md*" ollama serve

# 모델 받기
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

Copilot 설정의 Base URL은 `http://127.0.0.1:11434/v1`, API Key는 `local`.

#### 사용

- 좌측 사이드바 Copilot 아이콘 → **Chat** / **Vault QA** / **Copilot Plus**
- Vault QA 모드에서 임베딩 인덱싱이 1회 필요. 인덱싱 후 우리
  `llm-wiki/` 노트들에 대해 직접 질의 가능.

### 2. Smart Connections (관련 노트 자동 추천 + 채팅)

위키의 **백링크를 임베딩 기반으로 자동 보완**해 주는 플러그인. Karpathy
패턴에서 Lint/Cross-link 단계와 잘 맞는다.

설정:

1. Settings → Smart Connections → **Model Platform**: `Custom Local (OpenAI format)`
2. 다음 입력

| 필드 | 값 |
| --- | --- |
| Protocol | `https` (사내) 또는 `http` (Ollama) |
| Hostname | `llm.intra.company.com` 또는 `localhost` |
| Port | `443` 또는 `11434` (Ollama) / `1234` (LM Studio) |
| Path | `/v1/chat/completions` |
| Model Name | 사내 모델 이름 |
| API Key | 사내 토큰 또는 `local` |

3. **Notes Embedding Model**: 사내 임베딩 모델 또는 `Nomic-embed-text`
   (Ollama에 `ollama pull nomic-embed-text`).
4. 활성화 후 Vault 인덱싱이 진행됨. 노트를 열면 우측 패널에 **Smart
   Connections**가 의미적으로 가까운 노트를 자동 표시.

### 3. Local LLM Helper / Local GPT (가벼운 텍스트 작업용)

선택 텍스트에 대해 "요약", "Action Items", "톤 변경" 같은 명령을 핫키로
호출하고 싶을 때.

- Local LLM Helper: Settings에서 **Server URL**(예: `http://localhost:1234/v1`),
  Model, API Key만 채우면 됨. LM Studio/Ollama/vLLM/사내 게이트웨이 모두
  OpenAI 호환이면 동작.
- Local GPT: 동일하게 OpenAI-like Base URL을 받음.

### 4. Text Generator (템플릿 기반 글 생성)

LLM Provider → **Custom**(OpenAI Format) → Endpoint:
`https://llm.intra.company.com/v1/chat/completions` 입력. 프론트매터로
모델/temperature 등을 노트별로 다르게 줄 수 있어 "템플릿화된 ingest 노트"
같은 워크플로우에 적합.

### 5. LLM Wiki Local (선택, Karpathy 패턴 자동화)

`kytmanov/obsidian-llm-wiki-local`. 이 저장소의 `llm-wiki/README.md`에서
설명한 Karpathy LLM Wiki 패턴을 Obsidian 플러그인으로 구현한 것.
Markdown 노트를 드롭하면 LLM이 개념을 추출하고 자동으로 위키 페이지로
연결해 준다. 기본은 Ollama, OpenAI 호환 엔드포인트도 지원하므로 사내
LLM에 그대로 붙일 수 있다. 단, 회사 데이터로 돌리기 전에 README/코드를
한 번 검수할 것.

### 추천 조합

우리 LLM Wiki 운영 맥락에서는 다음 조합을 권장.

```
Obsidian Vault = llm-wiki/ 폴더 (또는 docs/llm-wiki/)
├─ Copilot                → Vault QA / 채팅 (사내 LLM)
├─ Smart Connections      → 임베딩 기반 관련 노트 추천 (사내 임베딩 or 로컬 Ollama)
└─ (옵션) LLM Wiki Local  → ingest 자동화 보조
```

ingest/lint 같은 "쓰기" 작업은 가급적 **Claude Code/Codex로 PR 단위**로
진행하고(`WIKI_SCHEMA.md` 규칙 준수), Obsidian은 **읽기·질의·연결 탐색**
중심으로 쓰는 것이 거버넌스 관점에서 안전하다.

## 보안 체크리스트

회사 데이터를 다루기 전에 반드시 확인.

- [ ] Base URL이 진짜 사내 도메인인지 확인 (오타 한 자로 외부로 샐 수 있음)
- [ ] 플러그인이 OpenAI 클라우드로 fallback 호출하지 않는지 네트워크 모니터로 확인
- [ ] API 키는 사용자 머신에만 저장됨을 인지 (Obsidian Sync 사용 시 설정 동기화 범위 점검)
- [ ] `raw/` 아래 비공개 자료는 별도 Vault 또는 별도 폴더로 격리
- [ ] 임베딩 인덱스 파일도 민감 정보로 간주, Git ignore
- [ ] CORS 우회 프록시 옵션 사용 시 트래픽 경로가 어디로 나가는지 플러그인 코드 확인

## 참고 자료 (References)

- [Copilot for Obsidian — local_copilot.md](https://github.com/logancyang/obsidian-copilot/blob/master/local_copilot.md)
- [Copilot Settings 공식 문서](https://www.obsidiancopilot.com/en/docs/settings)
- [Smart Connections (brianpetro)](https://github.com/brianpetro/obsidian-smart-connections)
- [Smart Connections — local embeddings 토론](https://github.com/brianpetro/obsidian-smart-connections/discussions/561)
- [Local LLM Helper (manimohans)](https://github.com/manimohans/obsidian-local-llm-helper)
- [Local GPT (pfrankov)](https://github.com/pfrankov/obsidian-local-gpt)
- [Text Generator Plugin (nhaouari)](https://github.com/nhaouari/obsidian-textgenerator-plugin)
- [LLM Wiki Local (kytmanov) — Karpathy 패턴 Obsidian 구현](https://github.com/kytmanov/obsidian-llm-wiki-local)
- [Obsidian + Smart Connection + Ollama 가이드](https://medium.com/@hunterzhang86/obsidian-smart-connection-ollama-make-local-llm-your-intelligent-note-taking-assistant-d21397f2cd66)
- [Ollama CORS / OLLAMA_ORIGINS 설정](https://github.com/ollama/ollama/blob/main/docs/faq.md)

## 관련 문서

- [LLM Wiki 패턴 개요](./README.md)
- [WIKI_SCHEMA 템플릿](./wiki-schema-template.md)
- [팀 위키 구조 가이드](./team-wiki-structure.md)
