---
tags: [ai-coding, session, context-window, turn, agent, system-prompt]
level: beginner
last_updated: 2026-05-05
source: https://github.com/mattpocock/dictionary-of-ai-coding
---

# Section 2 — Sessions, Context Windows & Turns (세션, 컨텍스트 윈도우, 턴)

> 에이전트가 어떻게 **상태(state)** 를 다루고, 사용자와 어떤 단위로 주고받는가.

## 왜 이 섹션이 중요한가? (Why)

- AI 코딩 도구의 **계층 구조**를 이해하지 않으면 비용도, 멍청해지는 현상도 설명이 안 된다.
- 핵심 계층: `Session > Turn > Model provider request`
- 그리고 핵심 구분: `Stateless (모델)` vs `Stateful (세션 누적)`

## 용어 (What & How)

### Stateless (스테이트리스)

정보를 다음 단계로 **이어서 가져가지 않는다**는 뜻. [Model](./01-the-model.md#model)은 [Model provider requests](./01-the-model.md#model-provider-request) 사이에서 stateless이기 때문에, 매 요청마다 [Context window](#context-window) 전체를 다시 보내야 한다. 모델이 무언가를 인지할 수 있는 통로가 그것뿐이기 때문이다. [Agent](#agent) 역시 기본적으로는 [Sessions](#session) 간에 stateless라서, 새 세션은 항상 빈 상태로 시작한다. [Stateful](#stateful)의 반대 개념.

**💬 실전 대화 예시**
> "왜 [clear](./05-handoffs.md#clearing) 할 때마다 컨벤션을 다시 까먹어?"
> "모델 자체가 stateless라서 그래. 새 세션은 늘 빈 상태로 시작하잖아. 컨벤션을 유지하고 싶으면 [AGENTS.md](./06-memory-and-steering.md#agentsmd) 같은 메모리 파일에 적어둬. 그러면 하네스가 세션 시작에 자동으로 로드해 줘."

---

### Context (컨텍스트)

**[Agent](#agent)가 지금 다루는 작업과 관련해서 알고 있는 모든 정보**를 가리키는 추상 개념. 모델이 직접 보는 raw input(그건 [Context window](#context-window))과도 다르고, 누적된 히스토리(그건 [Session](#session))와도 다르다. 좀 더 정확히 말하면, *agent가 이 작업에 대해 아는 것 그 자체*다. "context에 로드한다"는 말은 이 집합에 정보를 추가한다는 의미이고, "context engineering"은 이 집합을 **어떻게 큐레이션할지** 설계하는 분야다.

**💬 실전 대화 예시**
> "타입 정의에 없는 필드를 자꾸 만들어내네."
> "타입 파일이 지금 context에 안 들어가 있어서 그래. 호출부만 보고 짐작하는 중이야. 정의 파일을 먼저 읽히게 하자."

🏢 **실무 적용**: "Context engineering"은 사실상 사내에서 가장 부족한 스킬 영역이다. RAG/LangGraph 파이프라인을 설계할 때 *어떤 문서를, 어떤 순서로, 어느 분량까지* 컨텍스트에 넣을지를 결정하는 일이 곧 context engineering이다.

---

### Context window (컨텍스트 윈도우)

매 [Model provider request](./01-the-model.md#model-provider-request)마다 [Model](./01-the-model.md#model)이 한 번에 들여다보는 **모든 입력**. 크기가 유한하고, 모델마다 크기가 다르며, **모델이 무언가를 인지할 수 있는 유일한 표면**이다.

**❌ 피할 표현**: "memory". context window는 어디까지나 한 시점의 작업 상태(working state)이지, [Session](#session) 사이에 유지되는 게 아니다. [Memory](./06-memory-and-steering.md#memory-system)는 그 위에 별도로 얹힌 다른 개념이다.

**💬 실전 대화 예시**
> "그냥 모노레포 전체를 프롬프트에 붙여버릴 수 있어?"
> "Context window가 200k [tokens](./01-the-model.md#token)인데, 그건 레포의 1/5 정도밖에 안 돼. 작업에서 실제로 건드리는 파일만 골라서 넣고, 나머지는 [Tool call](./03-tools-environment.md#tool-call)로 필요할 때 가져오게 하자."

---

### Stateful (스테이트풀)

정보를 **다음 단계로 이어서 가져간다**는 뜻. [Session](#session)은 [Turns](#turn) 사이에서 stateful이라서, 세션이 진행될수록 [Context](#context)가 계속 누적된다. 긴 세션이 [dumb zone](./04-failure-modes.md#smart-zone)으로 흘러가는 것도 이 때문이다. [Agent](#agent)는 [Memory system](./06-memory-and-steering.md#memory-system)을 더하면 **세션과 세션 사이에서도** stateful이 될 수 있다. 정보를 [Environment](./03-tools-environment.md#environment)에 영속화해 두고, 다음 세션이 시작될 때 다시 로드하는 방식이다. **다만 [Model](./01-the-model.md#model) 자체가 stateful이 되는 일은 결코 없다.** 연속성처럼 보이는 효과도 결국은 [Harness](./01-the-model.md#harness)가 컨텍스트를 다시 먹여주기 때문에 생기는 것일 뿐이다. [Stateless](#stateless)의 반대 개념.

**💬 실전 대화 예시**
> "어제 내 선호를 기억하던데, 모델이 학습한 거야?"
> "아니, 에이전트가 stateful해진 거야. 하네스가 메모리 파일에 적어뒀다가 세션 시작에 다시 읽어준 거지. 모델 자체는 어제 본 게 아무것도 없어."

---

### Agent (에이전트)

[Tools](./03-tools-environment.md#tool), [System prompt](#system-prompt), [Context window](#context-window) 같은 요소로 [harnessed](./01-the-model.md#harness)된 [Model](./01-the-model.md#model). 사용자와 [Turns](#turn)을 주고받으며 동작한다. **Claude Code도 에이전트, Cursor도 에이전트, Claude.ai도 에이전트다.** 우리가 실제로 대화하는 대상은 바로 이 에이전트, 즉 특정 목적에 맞게 세팅된 채 움직이는 모델이다.

**❌ 피할 표현**: "the AI", "the bot". 너무 모호한 표현이라, 파라미터(모델)를 가리키는 건지 하네스로 감싼 전체를 가리키는 건지 흐려진다.

**💬 실전 대화 예시**
> "마이그레이션할 때 어떤 에이전트 써?"
> "로컬에선 Claude Code, UI 쪽은 Cursor 써. 같은 모델인데 하네스가 다른 케이스야."

---

### System prompt (시스템 프롬프트)

[Harness](./01-the-model.md#harness)가 매 [Model provider request](./01-the-model.md#model-provider-request) 앞에 항상 붙이는 지시문이자, [Agent](#agent)에게 늘 주어지는 **상시 브리핑**. 자기가 누구인지, 어떻게 행동해야 하는지, 어떤 [Tools](./03-tools-environment.md#tool)을 호출할 수 있는지, 어떤 컨벤션을 따라야 하는지가 여기에 담긴다. 보통 [Session](#session) 동안에는 거의 그대로 유지된다.

**💬 실전 대화 예시**
> "하네스 두 개에 같은 [Model](./01-the-model.md#model), 같은 프롬프트를 줬는데 동작이 완전 달라."
> "System prompt가 서로 다른 거야. 하나는 짧은 코드 편집용으로 튜닝돼 있고, 다른 하나는 설명용이거든. 그래서 네 메시지가 도착하기도 전에 이미 분기점이 갈려 있는 거야."

🏢 **실무 적용**: 사내에서 만드는 모든 LangGraph 노드와 Agent는 **자기만의 system prompt**를 갖고 있어야 한다. 어떤 노드 동작이 이상하다면, 가장 먼저 의심할 곳이 system prompt다.

---

### Session (세션)

[Agent](#agent)와 주고받는 **명확한 시작과 끝이 있는 한 번의 상호작용**. 빈 상태로 시작해서 메시지, [Tool results](./03-tools-environment.md#tool-result), 읽어들인 파일들이 차례로 쌓이다가, [clear](./05-handoffs.md#clearing) / 종료 / [compaction](./05-handoffs.md#compaction) 중 하나로 끝난다.

**[Context window](#context-window)를 *채우는* 주체가 바로 세션이다.** context window가 상자라면, session은 그 상자를 시간이 갈수록 천천히 채워가는 내용물이다. 하나의 context window로 감당이 안 되는 일은 여러 세션으로 쪼개서 다뤄야 한다.

**💬 실전 대화 예시**
> "한 세션이 무너지기 전까지 얼마나 끌고 갈 수 있어?"
> "작업 성격에 따라 달라. 집중된 리팩토링은 꽤 오래 가고, 발산적인 리서치 작업은 금방 무너져. 세션이 비대해지면 무리해서 끌고 가지 말고 [hand off](./05-handoffs.md#handoff)하거나 compact 해."

---

### Turn (턴)

**사용자 메시지 하나와 그에 대해 [Agent](#agent)가 수행하는 모든 동작**을, 다시 사용자에게 통제권을 넘길(yield) 때까지 묶은 단위. 그 안에서 도구를 호출하면 [Model provider requests](./01-the-model.md#model-provider-request)가 여러 번 발생할 수 있다. 명확화 질문이 들어오면 턴이 닫히고, 사용자가 답을 하면 그 답이 다음 턴을 연다.

**계층 구조: [Session](#session) > Turn > Model provider request**

**💬 실전 대화 예시**
> "한 턴이 2분이나 걸렸다고?"
> "그 안에서 [tool calls](./03-tools-environment.md#tool-call)을 14번 했거든. 각각이 전부 별도 model provider request라서, 사용자에게 통제권을 다시 넘기기 전까지 지연이 차곡차곡 쌓인 거야."

## 이 섹션 요약 (Cheatsheet)

```
Session ─┬─ Turn 1 ─┬─ Model provider request 1 (모델 사고)
         │          ├─ Model provider request 2 (tool call → tool result → 다시 모델)
         │          └─ Model provider request 3 (최종 답변 → user에게 yield)
         ├─ Turn 2 ─...
         └─ Turn N
```

| 개념 | 단위 | 비유 |
|---|---|---|
| Session | 한 번의 작업 (보통 분~시간) | 박스 |
| Turn | 사용자 메시지 1 + 응답 1 | 박스에 한 번 짐 넣기 |
| Model provider request | API 호출 1번 | 짐 넣기 안의 한 동작 |
| Context (개념) | 에이전트가 *아는 것* | - |
| Context window (자원) | 모델이 *보는 것의 한계* | 박스 크기 |

## 관련 문서

- 이전: [01 - The Model](./01-the-model.md)
- 다음: [03 - Tools & Environment](./03-tools-environment.md)
- 인덱스: [README](./README.md)

## 참고 자료 (References)

- 원문: [mattpocock/dictionary-of-ai-coding — Section 2](https://github.com/mattpocock/dictionary-of-ai-coding#section-2--sessions-context-windows--turns)
