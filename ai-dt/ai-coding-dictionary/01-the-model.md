---
tags: [ai-coding, model, token, inference, billing]
level: beginner
last_updated: 2026-05-05
source: https://github.com/mattpocock/dictionary-of-ai-coding
---

# Section 1 — The Model (모델)

> "Claude Opus 4.7"이나 "GPT-5" 같은 **모델 그 자체**와, 그 모델이 어떻게 학습되고 어떻게 비용이 매겨지는지에 관한 용어들.

## 왜 이 섹션부터인가? (Why)

- AI 코딩 도구를 쓸 때 가장 자주 헷갈리는 부분이 **"모델"과 "에이전트"의 구분**이다.
- "Claude는 똑똑하다"는 말도, 모델 자체를 가리키는지 Claude Code(하네스 + 에이전트) 전체를 가리키는지에 따라 의미가 달라진다.
- 비용은 거의 전부 **토큰 단위**로 청구되기 때문에, 토큰이 무엇이고 어떻게 캐시되는지를 모르면 청구서를 해석할 수가 없다.

## 용어 (What & How)

### Model (모델)

**파라미터 덩어리**. [Stateless](./02-sessions-context-windows-turns.md#stateless)이고, 하는 일은 다음 토큰 예측(next-token prediction)이 전부다. "Claude Opus 4.7", "GPT-5" 같은 게 여기서 말하는 모델. 모델 혼자서는 에이전트처럼 행동할 수 없고, [Harness](#harness)로 감싸줘야 비로소 동작한다.

**💬 실전 대화 예시**
> "계획 단계만 Sonnet에서 Opus로 바꿔볼까?"
> "한번 해보자. 다만 이 작업에선 사실 무거운 일은 하네스 쪽에서 하고 있어. 시스템 프롬프트와 도구 구성이 잘못돼 있으면, 모델만 바꿔서는 해결이 안 돼."

---

### Parameters (파라미터)

[Model](#model) 안에 들어 있는 숫자들로, 보통 수십억 개에 이른다. 학습(training) 과정에서 조정되며, 모델이 "안다"고 부를 만한 모든 것이 이 값들 안에 들어 있다. 학습은 이 값을 *세팅하는* 과정이고, [Inference](#inference)는 이 값을 *그대로 사용하는* 과정이다. 다른 이름으로 **weights(가중치)** 라고도 부른다.

**💬 실전 대화 예시**
> "우리 코드베이스에 맞춰서 fine-tune 할 수 있을까?"
> "그건 파라미터 자체를 업데이트하는 거라, 그 시점부터는 사실상 다른 모델이 되는 셈이야. 프로젝트 하나 단위에서는, 재학습하는 것보다 코드베이스를 [Context](./02-sessions-context-windows-turns.md#context)에 로드하는 쪽이 거의 항상 더 싸게 먹혀."

---

### Training (학습)

[Model](#model)의 [Parameters](#parameters)를 세팅하는 과정. 방대한 텍스트를 보여주면서 [Next-token prediction](#next-token-prediction)이 더 잘 맞도록 파라미터를 조정한다. 일회성 작업이고, 매우 비싸며, 모델 제공자만 수행한다. 큰 규모의 pre-training과, instruction-following·safety 같은 후속 정제를 다듬는 post-training이 모두 여기에 포함된다.

**💬 실전 대화 예시**
> "내부 API를 모델한테 알게 하려면 어떻게 해야 해?"
> "학습으로 해결할 수 있는 일이 아니야. 그건 모델 제공자가 몇 달에 걸쳐서 하는 작업이거든. 대신 API 문서를 [Context](./02-sessions-context-windows-turns.md#context)에 로드해 줘. 우리가 실제로 통제할 수 있는 레버는 그쪽이야."

---

### Inference (추론)

학습된 [Model](#model)을 **실행해서** 출력을 만들어내는 일. 모든 [Model provider request](#model-provider-request)가 일어날 때마다 추론이 실행된다. 파라미터는 이미 고정돼 있고, 모델은 주어진 [Context](./02-sessions-context-windows-turns.md#context)를 기반으로 [Next-token prediction](#next-token-prediction)을 할 뿐이다. 학습보다는 훨씬 저렴하지만 **토큰 단위로 과금**되기 때문에, 모델 사용 비용의 대부분이 추론에서 발생한다.

**💬 실전 대화 예시**
> "왜 비용이 정액제가 아니라 사용량에 비례하는 식이야?"
> "우리가 돈을 내는 대상이 사실상 추론(inference)이라서 그래. 모든 model provider request가 제공자 측 하드웨어에서 모델을 한 번 돌리는 거니까. 학습은 이미 끝났지만, 추론 비용은 요청 단위로 계속 쌓이고, 한 [Turn](./02-sessions-context-windows-turns.md#turn) 안에서도 [Tool](./03-tools-environment.md#tool) 호출 때문에 요청이 여러 번으로 늘어날 수 있어."

---

### Token (토큰)

[Model](#model)이 읽고 쓰는 **최소 단위**. 대체로 단어 정도의 크기지만 정확히 일치하지는 않는다. 흔한 단어는 1토큰으로 처리되고, 드물거나 긴 단어는 여러 개로 쪼개진다. [Context window](./02-sessions-context-windows-turns.md#context-window) 크기도, 비용도, 지연(latency)도 모두 토큰을 기준으로 측정한다.

**❌ 피할 표현**: "단어(word)". 토큰 경계는 단어 경계와 일치하지 않고, 실제로 의미 있는 단위는 **tokens-per-second / tokens-per-dollar** 다.

**💬 실전 대화 예시**
> "이 프롬프트가 얼마나 큰 편이야?"
> "토크나이저로 한번 돌려봐. 스키마 자체는 짧지만 JSON 키가 좀 특이해서, 생각보다 많이 쪼개질 거야."

---

### Next-token prediction (다음 토큰 예측)

[Model](#model)이 **실제로 하고 있는 일**. 주어진 [Context](./02-sessions-context-windows-turns.md#context)에서 다음 토큰을 *하나만* 샘플링한 뒤, 그 토큰을 컨텍스트 끝에 붙이고 다시 같은 과정을 반복한다. 한 줄짜리 문장이든, [Tool call](./03-tools-environment.md#tool-call)이든, 천 줄짜리 파일이든 결국 모두 한 토큰씩 쌓아 만든 결과다. 모델에는 **이 동작 외에는 다른 모드가 없다**.

**💬 실전 대화 예시**
> "[Agent](./02-sessions-context-windows-turns.md#agent)는 도구를 호출할지 말지를 어떻게 '결정'하는 거야?"
> "사실 결정한다기보다는, 끝까지 그냥 next-token prediction을 하는 거야. Tool call도 결국 모델이 출력 스트림에 뱉은 구조화된 문자열일 뿐이고, [Harness](#harness)가 그걸 파싱해서 실제로 실행하는 거지."

---

### Non-determinism (비결정성)

같은 입력에도 출력이 달라질 수 있다. 같은 [Model](#model)에 같은 [Context](./02-sessions-context-windows-turns.md#context)를 두 번 넣어도 답이 다르게 나올 수 있다. **내 쪽 코드가 전혀 바뀌지 않은 상황에서도** 마찬가지다.

이건 모델이 텍스트를 생성하는 방식과 [Model providers](#model-provider)가 요청을 처리하는 방식 자체에서 비롯된 성질이라서, 끌 수 있는 스위치 같은 건 없다.

같은 작업이라도 결과는 어느 정도 분산된다는 점을 전제로 깔고 가야 한다. 어떤 날은 답이 날카롭고, 어떤 날은 영 정신이 없어 보인다. **사람은 패턴을 잘 잡아내는 만큼, 결과가 몇 번 연달아 나쁘게 나오면 "이번 주에 모델이 나빠졌다"고 느끼기 쉽다. 하지만 대부분은 그냥 분포 안에서 생기는 변동일 뿐이다.**

**💬 실전 대화 예시**
> "오늘 Claude 진짜 별로야. 더 나쁜 버전 배포된 거 아냐?"
> "그럴 가능성은 낮아. 모델 출력은 원래 비결정적이라서, 같은 작업도 잘 풀리는 날이 있고 영 안 풀리는 날이 있어. 원인을 파헤치기 전에, 일단 내일 다시 한번 돌려봐."

---

### Model provider (모델 제공자)

[Model](#model)을 [Inference](#inference)용으로 서빙해 주는 주체. 보통은 Anthropic, OpenAI, Google 같은 원격 서비스가 이 역할을 하지만, Ollama·LM Studio·llama.cpp처럼 본인 머신에서 도는 로컬 환경도 모델 제공자가 될 수 있다. **하네스가 모델을 직접 실행하는 게 아니라, 모델 제공자에게 요청을 보내는 구조라는 점이 핵심**이다.

**💬 실전 대화 예시**
> "에어갭(폐쇄망) 클라이언트를 위해 오프라인으로 돌릴 수 있어?"
> "Model provider만 로컬로 바꾸면 돼. 그쪽 머신에 Ollama나 llama.cpp 깔아서 띄워두면, 하네스는 어디로 요청하는지 신경 안 써. 그냥 다른 엔드포인트로 보낼 뿐이야."

🏢 **실무 적용**: 사내 폐쇄망 환경(SK Hynix AI/DT)에서 외부 API를 사용할 수 없을 때 핵심은, *모델 자체*가 아니라 *Provider 레이어*만 교체하면 된다는 점이다. 같은 하네스를 그대로 쓸 수 있다.

---

### Harness (하네스)

[Model](#model)을 둘러싸서 그 모델을 [Agent](./02-sessions-context-windows-turns.md#agent)답게 동작하게 만들어 주는 모든 구성 요소를 합쳐서 부르는 말. 여기에는 [Tools](./03-tools-environment.md#tool), [System prompt](./02-sessions-context-windows-turns.md#system-prompt), [Context window](./02-sessions-context-windows-turns.md#context-window) 관리, 권한 처리, 훅 등이 포함된다. **Claude.ai와 Claude Code는 같은 모델을 쓰면서도, 하네스가 다르기 때문에 동작이 다르게 나타난다**.

**💬 실전 대화 예시**
> "같은 모델인데, 왜 Claude Code는 파일을 편집하고 Claude.ai는 답변만 해?"
> "차이는 모델이 아니라 하네스에 있어. Claude Code에는 [Filesystem](./03-tools-environment.md#filesystem) 도구가 붙어 있고, 시스템 프롬프트도 다르고, 권한 레이어도 있거든. 여기서 모델은 변수가 아니야."

---

### Model provider request (모델 제공자 요청)

[Harness](#harness)와 [Model provider](#model-provider) 사이를 오가는 **한 번의 왕복 요청**. 하네스가 현재 [Context](./02-sessions-context-windows-turns.md#context)를 보내면, 제공자가 응답 하나를 돌려준다(이때 응답은 [Tool call](./03-tools-environment.md#tool-call)일 수도 있고 최종 답변일 수도 있다). **사용자 메시지 하나에 도구 호출이 여러 번 끼면, 그만큼 model provider request가 늘어난다.** [Tool result](./03-tools-environment.md#tool-result)가 돌아올 때마다 매번 새 요청이 한 번 더 발생하기 때문이다.

**💬 실전 대화 예시**
> "질문 하나에 4만 토큰을 태웠다고?"
> "Tool call 내역 보면 grep 12번, read 8번, edit 4번이 일어났어. tool result가 돌아올 때마다 model provider request가 또 한 번씩 생기고, 그때마다 [Session](./02-sessions-context-windows-turns.md#session) 전체 prefix가 다시 전송돼."

---

### Input tokens (입력 토큰)

[Harness](#harness)가 매 [Model provider request](#model-provider-request)마다 보내는 [Tokens](#token). [Output tokens](#output-tokens)보다 단가가 낮다.

**💬 실전 대화 예시**
> "비용은 높은데 [Agent](./02-sessions-context-windows-turns.md#agent)가 거의 아무것도 안 쓰는 것 같은데?"
> "Input token이 원인이야. 매 [Turn](./02-sessions-context-windows-turns.md#turn)마다 [Session](./02-sessions-context-windows-turns.md#session) 전체가 다시 전송되는데, [Prefix cache](#prefix-cache)가 없으면 매 요청마다 히스토리 비용을 또 내는 셈이야."

---

### Output tokens (출력 토큰)

[Model](#model)이 생성해 내는 [Tokens](#token). [Input tokens](#input-tokens)보다 **단가가 높은데**, 토큰을 만들어내는 데 더 많은 컴퓨트가 필요하기 때문이다.

**💬 실전 대화 예시**
> "리팩토링 [Session](./02-sessions-context-windows-turns.md#session)인데 입력은 작은데도 크레딧이 빠르게 빠져나가."
> "[Agent](./02-sessions-context-windows-turns.md#agent)가 패치만 만드는 게 아니라 파일 전체를 다시 쓰고 있어서 그래. 지금 output token이 input의 5배쯤 되는데, 패치(edit) 형식으로 출력하게 하면 비용이 확 떨어져."

---

### Prefix cache (프리픽스 캐시)

[Provider](#model-provider) 쪽에 마련된 캐시 저장소. 연속된 [Model provider requests](#model-provider-request)가 **공통 prefix를 매번 다시 처리하지 않도록** 해준다. 어떤 요청의 앞부분이 최근 요청의 앞부분과 일치하면(같은 system prompt, 같은 히스토리), 제공자가 이전 작업 결과를 재활용하고, 그 토큰들은 [Cache tokens](#cache-tokens)로 훨씬 저렴하게 청구된다.

**Prefix를 바꾸는 모든 행동이 캐시를 깨뜨린다.** 파일 순서를 바꾸거나, 세션 도중 system prompt를 다시 쓰거나, 위쪽에 timestamp를 주입하는 식이 대표적인 예다. 캐시가 깨진 그 지점부터 나머지 토큰은 모두 풀(full) input rate로 청구된다.

**💬 실전 대화 예시**
> "왜 세션 중간부터 비용이 튀었지?"
> "[Harness](#harness)가 어느 시점부터 매 [Turn](./02-sessions-context-windows-turns.md#turn)마다 system prompt에 현재 시각을 주입하고 있어. Prefix cache는 처음 바뀐 토큰에서 바로 깨지기 때문에, 그 이후 요청은 전부 풀 가격으로 들어가."

🏢 **실무 적용**: 사내 LangGraph 파이프라인에서 노드 간 컨텍스트를 재구성할 때, 시간이나 UUID 같은 휘발성 값을 prefix 위쪽에 두지 않고 **prefix를 안정적으로 유지**하면 비용이 크게 떨어진다.

---

### Cache tokens (캐시 토큰)

[Provider](#model-provider)가 이전 [Model provider request](#model-provider-request)에서 캐시해 둔 [Input tokens](#input-tokens). 연속된 요청들이 같은 prefix를 공유하면 [Prefix cache](#prefix-cache)가 이전 작업을 재활용하고, 캐시된 부분은 훨씬 저렴한 단가로 청구된다. **긴 [Session](./02-sessions-context-windows-turns.md#session)을 비용적으로 감당 가능하게 만들어 주는 핵심 장치**다. 캐시가 없으면 매 [Turn](./02-sessions-context-windows-turns.md#turn)마다 전체 히스토리 비용을 다시 내야 한다.

**💬 실전 대화 예시**
> "긴 세션 비용이 살벌해. 리팩토링 한 번에 8달러야."
> "Cache token 사용량을 한번 봐봐. [Harness](#harness)가 [Turn](./02-sessions-context-windows-turns.md#turn) 사이에 [System prompt](./02-sessions-context-windows-turns.md#system-prompt)나 파일 순서를 건드리고 있으면 prefix가 깨져서, 결국 매 요청마다 풀 input rate를 다시 무는 셈이 돼."

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
