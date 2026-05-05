---
tags: [ai-coding, model, token, inference, billing]
level: beginner
last_updated: 2026-05-05
source: https://github.com/mattpocock/dictionary-of-ai-coding
---

# Section 1 — The Model (모델)

> "Claude Opus 4.7"이나 "GPT-5" 같은 **모델 그 자체**와, 그 모델이 어떻게 학습되고 어떻게 비용이 매겨지는지에 관한 용어들.

## 왜 이 섹션부터인가? (Why)

- AI 코딩 도구를 쓰면서 가장 헷갈리는 게 **"모델"과 "에이전트"의 구분**이다.
- "Claude는 똑똑하다"고 말할 때, 그건 모델 얘기인지 Claude Code(하네스+에이전트) 얘기인지가 다르다.
- 비용은 거의 다 **토큰 단위**로 나오는데, 토큰이 뭔지/어떻게 캐시되는지 모르면 청구서를 해석할 수 없다.

## 용어 (What & How)

### Model (모델)

**파라미터 덩어리**. [Stateless](./02-sessions-context-windows-turns.md#stateless) — 다음 토큰 예측(next-token prediction) 외엔 아무것도 하지 않는다. "Claude Opus 4.7", "GPT-5"가 모델이다. 모델 혼자서는 어떤 에이전트적 동작도 할 수 없다 — [Harness](#harness)로 감싸야 한다.

**💬 실전 대화 예시**
> "계획 단계는 Sonnet에서 Opus로 모델을 바꿀까?"
> "해보자 — 근데 이 작업에서 무거운 일은 하네스가 하고 있어. 시스템 프롬프트와 도구가 잘못돼 있으면 모델만 바꿔서는 안 풀려."

---

### Parameters (파라미터)

[Model](#model) 안의 숫자들 — 보통 수십억 개. 학습(training) 동안 조정된다. 모델이 "아는" 모든 것이 여기에 산다. 학습이 이 값을 *세팅*하고, [Inference](#inference)는 이 값을 *그대로 쓴다*. 별칭: **weights(가중치)**.

**💬 실전 대화 예시**
> "우리 코드베이스에 fine-tune 할 수 있을까?"
> "그러면 파라미터를 업데이트하는 거야 — 그 시점부터 다른 모델이 되는 거지. 한 프로젝트만 보면 재학습보다 코드베이스를 [Context](./02-sessions-context-windows-turns.md#context)에 로드하는 게 거의 항상 더 싸다."

---

### Training (학습)

[Model](#model)의 [Parameters](#parameters)를 세팅하는 과정. 방대한 텍스트에 노출시켜서 [Next-token prediction](#next-token-prediction)이 더 잘 되도록 파라미터를 조정한다. 일회성, 매우 비쌈, 모델 제공자만 한다. Pre-training(대규모)과 post-training(instruction-following, safety 같은 후속 정제)을 모두 포함한다.

**💬 실전 대화 예시**
> "우리 내부 API를 알게 하려면?"
> "학습은 안 돼 — 그건 모델 제공자가 몇 달짜리로 하는 일이야. API 문서를 [Context](./02-sessions-context-windows-turns.md#context)에 로드해. 그게 네가 실제로 쥘 수 있는 레버야."

---

### Inference (추론)

학습된 [Model](#model)을 **실행**해서 출력을 만드는 일. 모든 [Model provider request](#model-provider-request)에서 일어난다. 파라미터는 고정이고, 모델은 주어진 [Context](./02-sessions-context-windows-turns.md#context) 위에서 [Next-token prediction](#next-token-prediction)을 할 뿐이다. 학습보다 싸지만 **토큰당 과금**되며, 모델 사용 비용의 대부분을 차지한다.

**💬 실전 대화 예시**
> "왜 비용이 정액제가 아니라 사용량에 비례해?"
> "추론에 돈을 내는 거야 — 모든 model provider request가 제공자 하드웨어에서 모델을 돌려. 학습은 이미 끝났지만, 추론 비용은 요청마다 쌓이고, 한 [Turn](./02-sessions-context-windows-turns.md#turn)이 [Tool](./03-tools-environment.md#tool) 호출로 여러 요청으로 불어날 수 있어."

---

### Token (토큰)

[Model](#model)이 읽고 쓰는 **원자 단위**. 대략 단어 크기지만 정확히는 아니다 — 흔한 단어는 1토큰, 드물거나 긴 단어는 여러 개로 쪼개진다. [Context window](./02-sessions-context-windows-turns.md#context-window) 크기, 비용, 지연(latency) 모두 토큰으로 센다.

**❌ 피할 표현**: "단어(word)" — 토큰 경계는 단어 경계와 일치하지 않고, **tokens-per-second / tokens-per-dollar** 가 실제로 중요한 단위다.

**💬 실전 대화 예시**
> "이 프롬프트가 얼마나 클까?"
> "토크나이저 돌려봐 — 스키마는 짧지만 JSON 키가 이상해서 생각보다 많이 쪼개질 거야."

---

### Next-token prediction (다음 토큰 예측)

[Model](#model)이 **실제로 하는 일**. 주어진 [Context](./02-sessions-context-windows-turns.md#context)에서 다음 토큰 *하나*를 샘플링하고, 그걸 붙이고, 다시 돌린다. 한 문장이든, [Tool call](./03-tools-environment.md#tool-call)이든, 천 줄짜리 파일이든 — 전부 한 토큰씩 만들어진다. 모델에는 **이것 외의 동작 모드가 없다**.

**💬 실전 대화 예시**
> "[Agent](./02-sessions-context-windows-turns.md#agent)는 도구를 호출할지를 어떻게 '결정'하지?"
> "결정하는 게 아니야 — 끝까지 next-token prediction이야. Tool call은 모델이 출력 스트림에 뱉은 구조화된 문자열이고, [Harness](#harness)가 그걸 파싱해서 실행해."

---

### Non-determinism (비결정성)

같은 입력이 다른 출력을 만들 수 있다. 같은 [Model](#model)을 같은 [Context](./02-sessions-context-windows-turns.md#context)로 두 번 돌리면 다른 답이 나올 수 있다. **네 코드의 어떤 부분도 바뀌지 않아도** 그렇다.

이건 모델이 텍스트를 만드는 방식, 그리고 [Model providers](#model-provider)가 요청을 처리하는 방식의 속성이다. 끄는 스위치는 없다.

같은 작업에서도 결과가 분산된다는 걸 받아들여야 한다. 어떤 날은 날카롭고, 어떤 날은 정신이 나간 것 같다. **사람은 패턴 매칭 기계라서, 안 좋은 결과가 몇 번 연속되면 "이번 주에 모델이 나빠졌다"고 느낀다 — 보통은 그냥 분포일 뿐이다**.

**💬 실전 대화 예시**
> "오늘 Claude 진짜 별로야. 더 나쁜 버전 배포된 거 아냐?"
> "아닐 가능성이 커 — 모델 출력은 비결정적이야. 같은 작업도 좋은 날과 나쁜 날이 있어. 원인 찾기 전에 내일 다시 해봐."

---

### Model provider (모델 제공자)

[Model](#model)을 [Inference](#inference)용으로 서빙하는 주체. 보통 원격 서비스(Anthropic, OpenAI, Google)지만, 로컬일 수도 있다 — 본인 머신에서 도는 Ollama, LM Studio, llama.cpp. **하네스가 모델을 직접 돌리는 게 아니라, 제공자에게 요청하는 것이다**.

**💬 실전 대화 예시**
> "에어갭(폐쇄망) 클라이언트를 위해 오프라인으로 돌릴 수 있어?"
> "Model provider를 로컬로 바꿔 — 그쪽 머신에 Ollama나 llama.cpp 깔고. 하네스는 신경 안 써, 그냥 다른 엔드포인트로 때리는 거야."

🏢 **실무 적용**: 사내 폐쇄망 환경(SK Hynix AI/DT)에서 외부 API를 못 쓸 때, *모델 자체*가 아니라 *Provider 레이어*만 바꾸면 된다는 게 핵심. 동일한 하네스를 그대로 쓸 수 있다.

---

### Harness (하네스)

[Model](#model) 주위에 둘러싸여서 모델을 [Agent](./02-sessions-context-windows-turns.md#agent)로 만들어주는 모든 것: [Tools](./03-tools-environment.md#tool), [System prompt](./02-sessions-context-windows-turns.md#system-prompt), [Context window](./02-sessions-context-windows-turns.md#context-window) 관리, 권한, 훅. **Claude.ai와 Claude Code는 같은 모델을 쓰지만 하네스가 달라서 다르게 행동한다**.

**💬 실전 대화 예시**
> "같은 모델인데 왜 Claude Code는 파일을 편집하고 Claude.ai는 답변만 해?"
> "하네스가 다른 거야 — Claude Code는 [Filesystem](./03-tools-environment.md#filesystem) 도구가 있고, 시스템 프롬프트도 다르고, 권한 레이어도 있어. 모델은 변수가 아니야."

---

### Model provider request (모델 제공자 요청)

[Harness](#harness)에서 [Model provider](#model-provider)로 가는 **한 번의 왕복**. 하네스가 현재 [Context](./02-sessions-context-windows-turns.md#context)를 보내고, 제공자가 응답 하나를 돌려준다([Tool call](./03-tools-environment.md#tool-call) 또는 최종 답변). **사용자 메시지 하나가 도구를 여러 번 부르면 여러 model provider request로 불어난다** — 매 [Tool result](./03-tools-environment.md#tool-result)마다 또 한 번의 요청이 발생.

**💬 실전 대화 예시**
> "질문 하나에 4만 토큰을 태웠다고?"
> "Tool call들 봐 — grep 12번, read 8번, edit 4번. 매 tool result마다 model provider request가 또 생기고, 매번 [Session](./02-sessions-context-windows-turns.md#session) 전체 prefix가 다시 보내져."

---

### Input tokens (입력 토큰)

[Harness](#harness)가 매 [Model provider request](#model-provider-request)마다 보내는 [Tokens](#token). [Output tokens](#output-tokens)보다 단가가 낮다.

**💬 실전 대화 예시**
> "비용은 높은데 [Agent](./02-sessions-context-windows-turns.md#agent)가 거의 아무것도 안 쓰는데?"
> "Input token 때문이야 — 매 [Turn](./02-sessions-context-windows-turns.md#turn)마다 [Session](./02-sessions-context-windows-turns.md#session) 전체가 재전송돼. [Prefix cache](#prefix-cache) 없으면 매 요청마다 히스토리 비용을 다시 내는 거야."

---

### Output tokens (출력 토큰)

[Model](#model)이 만들어내는 [Tokens](#token). [Input tokens](#input-tokens)보다 **단가가 높다** — 만드는 데 더 많은 컴퓨트가 들기 때문.

**💬 실전 대화 예시**
> "리팩토링 [Session](./02-sessions-context-windows-turns.md#session) 입력은 작은데 크레딧이 빨리 사라져."
> "[Agent](./02-sessions-context-windows-turns.md#agent)가 패치를 만드는 게 아니라 파일 전체를 다시 쓰고 있어. Output token이 input의 5배쯤이야 — edit으로 뱉게 만들면 비용이 떨어져."

---

### Prefix cache (프리픽스 캐시)

[Provider](#model-provider) 측 저장소. 연속된 [Model provider requests](#model-provider-request)가 **공통 prefix를 다시 처리하지 않게** 해준다. 어떤 요청의 시작이 최근 요청의 시작과 일치하면(같은 system prompt, 같은 히스토리), 제공자가 이전 작업을 재활용하고 그 토큰들을 [Cache tokens](#cache-tokens)로 훨씬 저렴하게 청구한다.

**Prefix를 바꾸는 모든 것이 캐시를 깬다** — 파일 순서 변경, 세션 도중 system prompt 재작성, 위쪽에 timestamp 주입 등. 그 지점부터 나머지는 풀 input rate로 청구된다.

**💬 실전 대화 예시**
> "왜 세션 중간부터 비용이 튀었지?"
> "[Harness](#harness)가 매 [Turn](./02-sessions-context-windows-turns.md#turn)마다 system prompt에 현재 시각을 주입하기 시작했어. Prefix cache는 처음 바뀐 토큰에서 깨지니까, 그 뒤로 모든 요청이 풀 가격이야."

🏢 **실무 적용**: 사내 LangGraph 파이프라인에서 노드 간 컨텍스트를 재구성할 때, **prefix를 안정적으로 유지**하면(시간/UUID 같은 휘발 값을 위에 두지 않으면) 비용이 크게 떨어진다.

---

### Cache tokens (캐시 토큰)

[Provider](#model-provider)가 이전 [Model provider request](#model-provider-request)에서 캐시해둔 [Input tokens](#input-tokens). 연속된 요청들이 prefix를 공유하면 [Prefix cache](#prefix-cache)로 작업을 재활용하고, 캐시된 부분은 훨씬 싸게 청구된다. **긴 [Session](./02-sessions-context-windows-turns.md#session)을 감당 가능하게 만들어주는 레버** — 없으면 매 [Turn](./02-sessions-context-windows-turns.md#turn)마다 전체 히스토리 비용을 또 낸다.

**💬 실전 대화 예시**
> "긴 세션 비용이 살벌해 — 리팩토링 한 번에 8달러야."
> "Cache token을 봐. [Harness](#harness)가 [Turn](./02-sessions-context-windows-turns.md#turn) 사이에 [System prompt](./02-sessions-context-windows-turns.md#system-prompt)나 파일 순서를 바꾸고 있으면 prefix가 깨지고, 매 요청마다 풀 input rate로 다시 내는 거야."

## 이 섹션 요약 (Cheatsheet)

| 헷갈리는 쌍 | 차이 |
|---|---|
| Model vs Agent | Model은 파라미터 덩어리 (stateless). Agent = Model + Harness |
| Training vs Inference | Training은 파라미터를 *세팅*. Inference는 *사용*. 비용은 거의 다 inference |
| Input tokens vs Output tokens | Input은 보내는 것 (싸다). Output은 받는 것 (~5배 비싸다) |
| Token vs Word | 단어 기준이 아니라 토큰 기준 — JSON, 한국어, 드문 용어는 더 많이 쪼개짐 |
| Prefix cache vs Cache tokens | 메커니즘 vs 그 결과로 청구되는 토큰 종류 |

## 관련 문서

- 다음 섹션: [02 - Sessions, Context Windows & Turns](./02-sessions-context-windows-turns.md)
- 인덱스: [README](./README.md)
- 사내 연결: [Foundation Model 기초](../foundation%20model/README.md), [Unsloth 파인튜닝](../unsloth/README.md)

## 참고 자료 (References)

- 원문: [mattpocock/dictionary-of-ai-coding — Section 1: The Model](https://github.com/mattpocock/dictionary-of-ai-coding#section-1--the-model)
- Anthropic Pricing 문서: https://docs.anthropic.com/en/docs/about-claude/pricing
