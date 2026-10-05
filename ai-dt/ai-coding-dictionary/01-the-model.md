---
tags: [ai-coding, model, token, inference, billing]
level: beginner
last_updated: 2026-05-05
source: https://github.com/mattpocock/dictionary-of-ai-coding
type: learning
reviewed_on: 2026-10-04
review_status: partial
category_major: "AI·DT"
category_middle: "AI 기초"
category_minor: "AI 코딩 용어"
note_kind: "학습"
classified_on: "2026-10-05"
---

# Section 1 — The Model (모델)

> [!info] 검토 범위 · 2026-10-04
> 원래 2026-05-05 학습 노트의 용어·대화 사례를 유지했다. 대화는 설명용 가상 사례이며 사내 시스템의 관측 결과가 아니다. 저자의 용어 선택은 보편 표준과 구분한다. 버전별 실제 조건은 [검증된 적용 조건](./verified-conditions.md), 개별 변경·남은 미확인은 [정리 기록](./organization-log.md)을 함께 읽는다.


> "Claude Opus 4.7"이나 "GPT-5" 같은 **모델 그 자체**와, 그 모델이 어떻게 학습되고 어떻게 비용이 매겨지는지에 관한 용어들.

## 왜 이 섹션부터인가? (Why)

- AI 코딩 도구를 쓸 때 가장 자주 헷갈리는 부분이 **"모델"과 "에이전트"의 구분**이다.
- "Claude는 똑똑하다"는 말도, 모델 자체를 가리키는지 Claude Code(하네스 + 에이전트) 전체를 가리키는지에 따라 의미가 달라진다.
- 많은 모델 API는 **토큰 단위**로 청구하지만 구독·로컬 비용은 다르므로, 토큰이 무엇이고 어떻게 캐시되는지를 모르면 청구서를 해석할 수가 없다.

## 용어 (What & How)

### Model (모델)

생성형 언어 모델은 학습된 파라미터와 계산 구조로 입력에 대한 출력을 만든다. 여기서는 autoregressive 텍스트 생성 모델을 중심으로 설명한다. 모든 AI 모델이 다음 토큰 생성만 수행하는 것은 아니다. 파일 편집·명령 실행은 [Harness](./01-the-model.md)가 연결한 도구의 역할이다. 모델명은 원래 노트의 예시이며 현재 제공·권한·가격을 보장하지 않는다.

**💬 실전 대화 예시**
> "계획 단계만 Sonnet에서 Opus로 바꿔볼까?"
> "한번 해보자. 다만 이 작업에선 사실 무거운 일은 하네스 쪽에서 하고 있어. 시스템 프롬프트와 도구 구성이 잘못돼 있으면, 모델만 바꿔서는 해결이 안 돼."

---

### Parameters (파라미터)

[Model](./01-the-model.md) 안에 들어 있는 숫자들로, 보통 수십억 개에 이른다. 학습(training) 과정에서 조정되며, 모델이 "안다"고 부를 만한 모든 것이 이 값들 안에 들어 있다. 학습은 이 값을 *세팅하는* 과정이고, [Inference](./01-the-model.md)는 이 값을 *그대로 사용하는* 과정이다. 다른 이름으로 **weights(가중치)** 라고도 부른다.

**💬 실전 대화 예시**
> "우리 코드베이스에 맞춰서 fine-tune 할 수 있을까?"
> "그건 파라미터 자체를 업데이트하는 거라, 그 시점부터는 사실상 다른 모델이 되는 셈이야. 프로젝트 하나 단위에서는, 재학습하는 것보다 코드베이스를 [Context](./02-sessions-context-windows-turns.md)에 로드하는 쪽이 총비용과 품질을 비교할 출발점이야."

---

### Training (학습)

데이터와 목적 함수로 파라미터를 조정하는 과정이다. pre-training, supervised fine-tuning, 선호도 기반 post-training은 범위와 목표가 다르다. 제공자만 수행하는 일회성 작업은 아니다. 공개 모델에 대한 사용자 fine-tuning과 일부 파라미터만 학습하는 PEFT도 가능하다. 내부 API의 자주 바뀌는 사실은 우선 문서 검색·컨텍스트 주입과 비교하고, 학습은 데이터 권한·자원·평가 조건을 갖춰 결정한다.

**💬 실전 대화 예시**
> "내부 API를 모델한테 알게 하려면 어떻게 해야 해?"
> "학습도 가능하지만 API의 현재 사실을 전달하려면 먼저 API 문서를 [Context](./02-sessions-context-windows-turns.md)에 로드해 줘. 우리가 실제로 통제할 수 있는 레버는 그쪽이야."

---

### Inference (추론)

학습된 [Model](./01-the-model.md)을 **실행해서** 출력을 만들어내는 일. 모든 [Model provider request](./01-the-model.md)가 일어날 때마다 추론이 실행된다. 파라미터는 이미 고정돼 있고, 모델은 주어진 [Context](./02-sessions-context-windows-turns.md)를 기반으로 [Next-token prediction](./01-the-model.md)을 할 뿐이다. 요청당 추론 비용과 학습 전체 비용은 비교 범위가 다르다. 원격 API의 토큰 과금, 구독, 로컬 하드웨어·운영 비용을 구분한다.

**💬 실전 대화 예시**
> "왜 비용이 정액제가 아니라 사용량에 비례하는 식이야?"
> "우리가 돈을 내는 대상이 사실상 추론(inference)이라서 그래. 모든 model provider request가 제공자 측 하드웨어에서 모델을 한 번 돌리는 거니까. 학습은 이미 끝났지만, 추론 비용은 요청 단위로 계속 쌓이고, 한 [Turn](./02-sessions-context-windows-turns.md) 안에서도 [Tool](./03-tools-environment.md) 호출 때문에 요청이 여러 번으로 늘어날 수 있어."

---

### Token (토큰)

[Model](./01-the-model.md)이 읽고 쓰는 **최소 단위**. 대체로 단어 정도의 크기지만 정확히 일치하지는 않는다. 흔한 단어는 1토큰으로 처리되고, 드물거나 긴 단어는 여러 개로 쪼개진다. [Context window](./02-sessions-context-windows-turns.md) 크기도, 비용도, 지연(latency)도 모두 토큰을 기준으로 측정한다.

**❌ 피할 표현**: "단어(word)". 토큰 경계는 단어 경계와 일치하지 않고, 실제로 의미 있는 단위는 **tokens-per-second / tokens-per-dollar** 다.

**💬 실전 대화 예시**
> "이 프롬프트가 얼마나 큰 편이야?"
> "토크나이저로 한번 돌려봐. 스키마 자체는 짧지만 JSON 키가 좀 특이해서, 생각보다 많이 쪼개질 거야."

---

### Next-token prediction (다음 토큰 예측)

autoregressive 언어 모델은 기존 토큰을 조건으로 다음 토큰 분포를 계산하고 decoding 정책으로 선택한다. sampling뿐 아니라 greedy/beam 방식도 있다. 병렬·speculative decoding 최적화가 있을 수 있어 물리적으로 항상 한 토큰씩만 계산한다고 해석하지 않는다. 구조화된 도구 요청을 생성해도 실제 도구를 실행하는 쪽은 하네스다.

**💬 실전 대화 예시**
> "[Agent](./02-sessions-context-windows-turns.md)는 도구를 호출할지 말지를 어떻게 '결정'하는 거야?"
> "사실 결정한다기보다는, 끝까지 그냥 next-token prediction을 하는 거야. Tool call도 결국 모델이 출력 스트림에 뱉은 구조화된 문자열일 뿐이고, [Harness](./01-the-model.md)가 그걸 파싱해서 실제로 실행하는 거지."

---

### Non-determinism (비결정성)

같은 요청에 다른 출력이 나올 수 있다. sampling 설정과 실행 환경을 통제하면 변동을 줄일 수 있지만 원격 API에서 완전한 재현을 보장한다고 단정하지 않는다. 품질 하락이 관측되면 동일 평가 입력, 모델 식별자, 프롬프트·도구·서빙 설정과 반복 결과를 비교한다. 임의의 정규분포나 “대부분 변동일 뿐”이라는 진단은 여기서 확인되지 않았다.

**💬 실전 대화 예시**
> "오늘 Claude 진짜 별로야. 더 나쁜 버전 배포된 거 아냐?"
> "변동과 실제 변경을 구분해야 해. 동일 평가 입력과 모델·설정·로그를 기록해서 반복 비교해 보자."

---

### Model provider (모델 제공자)

[Model](./01-the-model.md)을 [Inference](./01-the-model.md)용으로 서빙해 주는 주체. 보통은 Anthropic, OpenAI, Google 같은 원격 서비스가 이 역할을 하지만, Ollama·LM Studio·llama.cpp처럼 본인 머신에서 도는 로컬 환경도 모델 제공자가 될 수 있다. **하네스가 모델을 직접 실행하는 게 아니라, 모델 제공자에게 요청을 보내는 구조라는 점이 핵심**이다.

**💬 실전 대화 예시**
> "에어갭(폐쇄망) 클라이언트를 위해 오프라인으로 돌릴 수 있어?"
> "하네스가 지원하는 API·도구 schema·template·인증·모델 능력을 먼저 맞춰야 해. 그쪽 머신에 Ollama나 llama.cpp 깔아서 띄워두면, 주소만 교체해도 호환되는지는 실제 요청과 응답으로 확인해야 해."

🏢 **실무 적용**: 사내 폐쇄망 환경(SK Hynix AI/DT)에서 외부 API를 사용할 수 없을 때 provider를 분리하면 교체 범위를 줄일 수 있다. 같은 하네스의 재사용은 API·모델·인증·도구 지원 호환성을 확인한 경우에 한한다. 사내 환경의 실제 지원은 미확인이다.

---

### Harness (하네스)

[Model](./01-the-model.md)을 둘러싸서 그 모델을 [Agent](./02-sessions-context-windows-turns.md)답게 동작하게 만들어 주는 모든 구성 요소를 합쳐서 부르는 말. 여기에는 [Tools · Tool (도구) 절](./03-tools-environment.md), [System prompt](./02-sessions-context-windows-turns.md), [Context window](./02-sessions-context-windows-turns.md) 관리, 권한 처리, 훅 등이 포함된다. 같은 모델을 선택했더라도 하네스·도구·권한·입력이 다르면 동작이 달라질 수 있다. 제품별 실제 사용 모델이 같다는 뜻은 아니다.

**💬 실전 대화 예시**
> "같은 모델인데, 왜 Claude Code는 파일을 편집하고 도구가 없는 채팅 환경은 답변만 해?"
> "차이는 모델이 아니라 하네스에 있어. Claude Code에는 [Filesystem](./03-tools-environment.md) 도구가 붙어 있고, 시스템 프롬프트도 다르고, 권한 레이어도 있거든. 여기서 모델은 변수가 아니야."

---

### Model provider request (모델 제공자 요청)

[Harness](./01-the-model.md)와 [Model provider](./01-the-model.md) 사이를 오가는 **한 번의 왕복 요청**. 하네스가 현재 [Context](./02-sessions-context-windows-turns.md)를 보내면, 제공자가 응답 하나를 돌려준다(이때 응답은 [Tool call](./03-tools-environment.md)일 수도 있고 최종 답변일 수도 있다). **사용자 메시지 하나에 도구 호출이 여러 번 끼면, 그만큼 model provider request가 늘어난다.** 여러 결과를 모아서 한 요청으로 후속 추론할 수도 있다. 도구 호출 수와 모델 요청 수는 일대일이 아니다.

**💬 실전 대화 예시**
> "질문 하나에 4만 토큰을 태웠다고?"
> "Tool call 내역 보면 grep 12번, read 8번, edit 4번이 일어났어. 도구 결과를 어떻게 모아 모델에 보내는지 요청 로그를 확인해야 해. 전체 history 재전송·요약·서버 저장 state 사용 여부도 하네스에 따라 달라."

---

### Input tokens (입력 토큰)

[Harness](./01-the-model.md)가 매 [Model provider request](./01-the-model.md)마다 보내는 [Tokens · Token (토큰) 절](./01-the-model.md). API별 input/output 단가는 가격표로 확인한다. 정액 구독·로컬 추론의 원가 구조는 다르다.

**💬 실전 대화 예시**
> "비용은 높은데 [Agent](./02-sessions-context-windows-turns.md)가 거의 아무것도 안 쓰는 것 같은데?"
> "Input token이 원인이야. 매 [Turn](./02-sessions-context-windows-turns.md)마다 history가 모델 입력에 반영되는데, [Prefix cache](./01-the-model.md)가 없으면 매 요청마다 히스토리 비용을 또 내는 셈이야."

---

### Output tokens (출력 토큰)

[Model](./01-the-model.md)이 생성해 내는 [Tokens · Token (토큰) 절](./01-the-model.md). [Input tokens](./01-the-model.md)보다 높은 단가인 API가 많지만 배수와 과금 종류는 모델·제공자 정책이다. 컴퓨트 비용만으로 가격을 설명하지 않는다.

**💬 실전 대화 예시**
> "리팩토링 [Session](./02-sessions-context-windows-turns.md)인데 입력은 작은데도 크레딧이 빠르게 빠져나가."
> "[Agent](./02-sessions-context-windows-turns.md)가 패치만 만드는 게 아니라 파일 전체를 다시 쓰고 있어서 그래. 지금 output token이 input의 5배쯤 되는데, 패치(edit) 형식으로 출력하게 하면 비용이 확 떨어져."

---

### Prefix cache (프리픽스 캐시)

[Provider](./01-the-model.md) 쪽에 마련된 캐시 저장소. 연속된 [Model provider requests · Model provider request (모델 제공자 요청) 절](./01-the-model.md)가 **공통 prefix를 매번 다시 처리하지 않도록** 해준다. 어떤 요청의 앞부분이 최근 요청의 앞부분과 일치하면(같은 system prompt, 같은 히스토리), 제공자가 이전 작업 결과를 재활용하고, 그 토큰들은 [Cache tokens](./01-the-model.md)로 훨씬 저렴하게 청구된다.

정확히 일치하는 prefix, 최소 길이·TTL·cache breakpoint 등 제공자 조건이 충족되어야 한다. 변경 지점 전의 이미 캐시된 prefix가 재사용될 가능성은 남는다. 파일 순서를 바꾸거나, 세션 도중 system prompt를 다시 쓰거나, 위쪽에 timestamp를 주입하는 식이 대표적인 예다. 재사용·새 입력·cache write 과금은 API별로 다르다. Anthropic의 cache write는 일반 입력보다 비쌀 수 있다.

**💬 실전 대화 예시**
> "왜 세션 중간부터 비용이 튀었지?"
> "[Harness](./01-the-model.md)가 어느 시점부터 매 [Turn](./02-sessions-context-windows-turns.md)마다 system prompt에 현재 시각을 주입하고 있어. Prefix cache는 처음 바뀐 토큰에서 바로 깨지기 때문에, 그 뒤의 usage에서 cache read/write와 일반 입력을 나눠 확인하자."

🏢 **실무 적용**: 사내 LangGraph 파이프라인에서 노드 간 컨텍스트를 재구성할 때, 시간이나 UUID 같은 휘발성 값을 prefix 위쪽에 두지 않고 **prefix를 안정적으로 유지**하면 비용이 크게 떨어진다.

---

### Cache tokens (캐시 토큰)

[Provider](./01-the-model.md)가 이전 [Model provider request](./01-the-model.md)에서 캐시해 둔 [Input tokens](./01-the-model.md). 연속된 요청들이 같은 prefix를 공유하면 [Prefix cache](./01-the-model.md)가 이전 작업을 재활용하고, cache read 단가는 줄어들 수 있으나 최초 write·TTL·모델별 가격을 함께 계산한다. **긴 [Session](./02-sessions-context-windows-turns.md)을 비용적으로 감당 가능하게 만들어 주는 핵심 장치**다. 캐시가 없으면 매 [Turn](./02-sessions-context-windows-turns.md)마다 전체 히스토리 비용을 다시 내야 한다.

**💬 실전 대화 예시**
> "긴 세션 비용이 살벌해. 리팩토링 한 번에 8달러야."
> "Cache token 사용량을 한번 봐봐. [Harness](./01-the-model.md)가 [Turn](./02-sessions-context-windows-turns.md) 사이에 [System prompt](./02-sessions-context-windows-turns.md)나 파일 순서를 건드리고 있으면 prefix가 깨져서, 결국 매 요청마다 풀 input rate를 다시 무는 셈이 돼."

## 이 섹션 요약 (Cheatsheet)

| 헷갈리는 쌍 | 차이 |
|---|---|
| Model vs Agent | Model은 파라미터 덩어리 (stateless). Agent = Model + Harness |
| Training vs Inference | Training은 파라미터를 *세팅*. Inference는 *사용*. 비용 범위는 API·구독·로컬 운용별로 구분 |
| Input tokens vs Output tokens | Input은 모델 입력. Output은 생성 결과. 단가·배수는 가격표 기준 |
| Token vs Word | 단어 기준이 아니라 토큰 기준 — JSON, 한국어, 드문 용어는 더 많이 쪼개짐 |
| Prefix cache vs Cache tokens | 메커니즘 vs 그 결과로 청구되는 토큰 종류 |

## 관련 문서

- 다음 섹션: [02 - Sessions, Context Windows & Turns](./02-sessions-context-windows-turns.md)
- 인덱스: [README](./README.md)
- 사내 연결: [Foundation Model 기초](../foundation%20model/README.md), [Unsloth 파인튜닝](../unsloth/README.md)

## 참고 자료 (References)

- 원문: [mattpocock/dictionary-of-ai-coding — Section 1: The Model](https://github.com/mattpocock/dictionary-of-ai-coding#section-1--the-model)
- Anthropic Pricing 문서: https://docs.anthropic.com/en/docs/about-claude/pricing
