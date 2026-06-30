---
tags: [hermes, agent, harness, local-llm, openai-compatible, claude-code, nous-research]
level: intermediate
last_updated: 2026-07-01
---

# Hermes Agent — 설명, 사용법, 사내 LLM/Claude Code 연동

> Nous Research가 만든 오픈소스(MIT) 자율 에이전트 런타임. "세션이 끝나면 잊는" 일반 프레임워크와 달리, **메모리·학습된 스킬을 디스크에 남겨 쓸수록 똑똑해지는** 상주형 에이전트다. OpenAI 호환 엔드포인트만 있으면 어떤 모델로도 돌릴 수 있어 **사내 LLM(local API)와 구조적으로 연결 가능**하다.

> ⚠️ 이 폴더(`harness/`)는 원래 Codex 하네스 가이드다. Hermes는 repo 내부 코드가 아니라 **외부 도구**이며, 이 문서는 사내 환경(외부 LLM API 차단·DRM 문서·로컬 OpenAI 호환 엔드포인트) 기준으로 평가/사용법을 정리한 리서치 노트다. 모든 사실은 2026-07-01 기준 [공식 저장소·문서](https://github.com/NousResearch/hermes-agent)에서 확인했다.

---

## 왜 보는가? (Why)

- 우리 [하네스 가이드](./README.md)의 관점(컨텍스트·권한·도구·검증·관측성)에서 보면, Hermes는 그 자체가 **하나의 완결된 하네스 + 런타임**이다. Codex/Claude Code가 "IDE 옆 대화형 개발자"라면, Hermes는 "서버에 상주하며 메신저로 호출되는 자율 작업자"에 가깝다.
- 핵심 차별점은 **상태(state)의 영속화**: 작업 결과를 평가해 재사용 가능한 추론 패턴을 **named skill**로 추출하고, 다음 작업에서 검색해 다시 쓴다. 즉 "프롬프트 엔지니어링"이 아니라 "경험 축적"으로 품질이 오른다.
- 우리에게 중요한 이유: **모델을 바꿔 끼울 수 있다.** Anthropic/OpenAI 같은 외부 API가 막힌 사내 환경에서도, OpenAI 호환 사내 엔드포인트(Kimi-K2.5 / Qwen3-VL / BGE-M3)를 그대로 연결할 수 있는지가 도입 가능성의 전부다 → 결론은 "가능, 단 조건부".

---

## 핵심 개념 (What)

### 정체성
| 항목 | 내용 |
|------|------|
| 만든 곳 | Nous Research |
| 라이선스 | MIT (오픈소스, self-host, 무료) |
| 출시 | 2026년 2월 |
| 플랫폼 | Linux / macOS / WSL2 / Termux / Windows(native) |
| 형태 | 대화형 CLI + 메신저 게이트웨이(상주 프로세스) |
| 데이터 | 전부 로컬(`~/.hermes/`). 텔레메트리·추적 없음 |

### 자기 개선 루프(closed learning loop)
Hermes의 정체성. 대략 다섯 단계로 돈다:
1. **실행(execute)** — 도구를 써서 작업 수행
2. **평가(evaluate)** — 결과가 성공/실패였는지 판단
3. **스킬 추출(extract)** — 복잡한 작업이 끝나면 재사용 가능한 절차를 **named skill**로 자동 생성
4. **정련(refine)** — 그 스킬을 계속 쓰면서 다듬음
5. **검색·재사용(retrieve)** — 새 작업에 맞는 스킬을 불러와 다시 적용

여기에 **메모리 시스템**이 붙는다: 에이전트가 직접 큐레이션하는 메모리 엔트리, 사용자 프로필, 세션 전문 full-text 검색(FTS5) + LLM 요약 회상, "Honcho dialectic" 사용자 모델링.

### 도구·스킬·게이트웨이
- **40+ 내장 도구**: 웹 검색, 페이지 추출, 풀 브라우저 자동화(navigate/click/type/screenshot), 비전 분석, 이미지 생성, TTS, 샌드박스 코드 실행 등.
- **스킬 표준**: `agentskills.io` 오픈 표준 호환 — 이게 Claude Code 조합의 핵심 고리다(아래 참조).
- **메신저 게이트웨이**: Telegram·Discord·Slack·WhatsApp·Signal·CLI·Email을 단일 프로세스로 묶어, 메신저로 에이전트를 호출.

---

## 어떻게 쓰는가 (How) — 기본

### 설치
```bash
# Linux / macOS / WSL2 / Termux
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash

# Windows (PowerShell, native)
iex (irm https://hermes-agent.nousresearch.com/install.ps1)
```
설치 후:
```bash
source ~/.bashrc
hermes            # 대화형 CLI 시작
hermes model      # 모델/프로바이더 선택(인터랙티브)
```

> ⚠️ **사내 1차 관문**: 설치가 외부 도메인에서 `curl | bash`로 받아오는 구조다. 회사 방화벽/프록시가 `hermes-agent.nousresearch.com`를 막으면 **설치 자체가 안 된다**. 도입 전에 (1) 해당 도메인 화이트리스트 여부, (2) 사내 미러/오프라인 설치 가능 여부를 먼저 확인할 것.

---

## 어떻게 쓰는가 (How) — 사내 LLM(local API) 연동 ★

이게 질문의 핵심. Hermes는 **"OpenAI 호환 `/v1/chat/completions`만 말하면 어떤 엔드포인트든 `provider: custom`으로 연결"**한다. 우리 사내 LLM(Kimi-K2.5 / Qwen3-VL)이 정확히 이 OpenAI 호환 형식이므로 **구조적으로 연결된다.**

### 설정 파일 (`~/.hermes/config.yaml`)
```yaml
model:
  default: kimi-k2.5            # 사내에서 부르는 모델명 그대로
  provider: custom
  base_url: http://<사내-LLM-host>:<port>/v1   # OpenAI 호환 엔드포인트
  api_key: ${CORP_API_KEY}      # 키 없으면 비워두거나 더미값
  context_length: 65536         # 자동 감지 실패 시 명시(최소 64K 필요)
```

또는 인터랙티브로:
```bash
hermes model
# → "Custom endpoint (self-hosted / VLLM / etc.)" 선택
# → base URL / API key(선택) / model name 입력
```

### 여러 사내 엔드포인트 등록(텍스트/비전 분리)
사내엔 텍스트(Kimi-K2.5)와 비전(Qwen3-VL-8B/30B)이 따로 있으니 named provider로 등록해 두고 세션 중 전환:
```yaml
custom_providers:
  - name: kimi
    base_url: http://<text-llm>:<port>/v1
    key_env: CORP_API_KEY
  - name: qwenvl
    base_url: http://<vision-llm>:<port>/v1
    key_env: CORP_API_KEY
```
```text
/model custom:kimi:kimi-k2.5      # 텍스트/에이전트 추론
/model custom:qwenvl:qwen3-vl-30b # 비전(DRM 스크린샷 분석 등)
```

### 반드시 만족해야 할 2가지 조건 (도입 가능/불가의 분기점)
1. **컨텍스트 ≥ 64K 토큰.** Hermes는 멀티스텝 tool-calling 동안 working memory를 유지하느라 최소 64,000 토큰을 요구한다. 사내 모델 서빙이 이보다 작게 잘려 있으면 **에이전트가 제대로 작동하지 않는다.** (참고: Ollama는 기본 4K → `OLLAMA_CONTEXT_LENGTH=64000 ollama serve`로 서버 측에서 올려야 함)
2. **Tool calling(function calling) 지원.** Hermes는 도구 호출로 동작한다. 사내 엔드포인트가 OpenAI 호환 `tools`/`tool_calls`를 지원해야 한다. Kimi-K2 계열은 에이전트형 도구 호출에 강하고 Qwen3 계열도 function calling을 지원하므로 가능성이 높지만, **사내 서빙(vLLM 등)이 tool-call 파서를 켜고 떴는지**가 관건이다.
   - vLLM 직접 서빙이라면: `--enable-auto-tool-choice --tool-call-parser hermes`(또는 모델군에 맞는 파서)
   - 이미 사내 플랫폼이 게이트웨이로 감싸 제공한다면, 그 게이트웨이가 tool_calls를 패스스루하는지 확인.

### 임베딩(BGE-M3)
Hermes의 메모리 회상은 기본적으로 FTS5(전문 검색) + LLM 요약 기반이다. 별도 벡터 임베딩이 필수는 아니지만, 사내 BGE-M3가 OpenAI 호환 임베딩 API로 서빙된다면 향후 메모리/RAG 보강에 끌어 쓸 여지가 있다(공식 필수 항목은 아님).

> **요약 판단**: 사내 LLM이 "OpenAI 호환 + 64K 컨텍스트 + tool calling" 세 가지를 만족하면 **Hermes는 사내 모델만으로 완전 로컬 구동 가능**하다. 외부 API 한 줄도 안 쓴다. 막히는 건 모델이 아니라 (a) 설치 도메인 접근, (b) 사내 서빙의 tool-call/컨텍스트 설정 두 군데다.

---

## 어떻게 쓰는가 (How) — Claude Code와의 조합 ★

"조합 가능한가?" → **가능하다. 단 '한 프로세스 안에서 합친다'가 아니라 '역할을 나눠 같은 자산을 공유한다'에 가깝다.** 세 가지 결의 조합이 있다:

### 1) 스킬 자산 공유 (가장 실질적)
- Hermes는 스킬을 `agentskills.io` **오픈 표준**으로 만들고 읽는다. 이 표준은 우리가 쓰는 Claude Code/Superpowers 스킬 생태계와 같은 계보다(실제로 Hermes 저장소엔 `opencode` 등 외부 에이전트용 SKILL.md가 들어 있다).
- 즉 **"Claude Code에서 다듬은 절차/스킬을 Hermes가 상주 실행하게 넘기고, Hermes가 경험으로 자동 생성한 스킬을 다시 사람이 Claude Code로 리뷰·정련"**하는 양방향 흐름이 현실적인 조합이다.

### 2) 역할 분담 (권장 운영 모델)
| | Claude Code | Hermes Agent |
|---|---|---|
| 위치 | 내 노트북 / IDE | 서버에 상주($5 VPS~사내 서버) |
| 트리거 | 사람이 대화형으로 | 메신저(Telegram/Slack 등)·스케줄 |
| 강점 | repo 깊은 이해, 정밀 편집, 리뷰 게이트 | 24/7 상주, 메모리 축적, 멀티채널 호출 |
| 모델 | (사내선 제약) | **사내 OpenAI 호환 LLM 그대로** |

→ **개발/리뷰는 Claude Code, 상시 자동화·알림·반복 작업은 Hermes**로 나누면 충돌 없이 보완된다.

### 3) 같은 사내 LLM 백엔드 공유
둘 다 결국 "OpenAI 호환 사내 엔드포인트"를 백엔드로 쓸 수 있으므로, **모델 인프라를 공유**하고 프런트(대화형 vs 상주형)만 다르게 가져갈 수 있다.

> **주의**: Hermes를 *Claude 모델*로 돌리려면 `provider: anthropic`(또는 Bedrock)을 쓰는데, 이는 **외부 Anthropic API 호출**이라 우리 사내 방화벽 정책상 막힌다. 따라서 "Hermes를 Claude로 구동"은 사내선 불가, **"Hermes를 사내 LLM으로 구동 + Claude Code는 별도 도구로 병행"**이 맞는 그림이다.

---

## 사내 도입 체크리스트 (실측 전 확인)

```text
[ ] 설치 도메인(hermes-agent.nousresearch.com) 방화벽 화이트리스트 — 또는 오프라인 설치 경로
[ ] 사내 LLM이 OpenAI 호환 /v1/chat/completions 노출하는가
[ ] tool calling(tool_calls) 패스스루/지원되는가  ← 안 되면 사실상 작동 불가
[ ] 컨텍스트 ≥ 64K 보장되는가
[ ] 사내 모델명/엔드포인트/키를 config.yaml(custom)에 매핑
[ ] 비전 작업용 Qwen3-VL을 named provider로 분리 등록
[ ] 데이터 로컬 보관(~/.hermes) — 사내 보안정책상 서버 위치/접근통제 검토
[ ] 메신저 게이트웨이는 필요한 채널만, 사내 정책 허용 범위에서만 활성화
```

---

## 한 줄 결론

> **사내 LLM 활용**: OpenAI 호환 + 64K 컨텍스트 + tool calling 충족 시 **완전 로컬 구동 가능**(외부 API 0). 막히는 건 모델이 아니라 설치 경로와 사내 서빙 설정.
> **Claude Code 조합**: 한 몸으로 합치는 게 아니라, **스킬 표준(agentskills.io) 공유 + 역할 분담(대화형 개발=Claude Code / 상주 자동화=Hermes) + 사내 LLM 백엔드 공유**의 형태로 보완 가능.

---

## 관련 문서
- [하네스 가이드 개요](./README.md)
- [하네스 설정 맵](./01_harness_settings.md)
- [에이전트와 스킬](../05_agents_and_skills.md)
- [Harness Engineering for LLM](../../../ai-dt/mcp/harness-engineering-llm.md)

## 참고 자료 (References)
- [NousResearch/hermes-agent (GitHub)](https://github.com/NousResearch/hermes-agent)
- [Hermes Agent 공식 사이트](https://hermes-agent.nousresearch.com/)
- [AI Providers 설정 문서](https://github.com/NousResearch/hermes-agent/blob/main/website/docs/integrations/providers.md)
- [agentskills.io 스킬 표준](https://agentskills.io)
