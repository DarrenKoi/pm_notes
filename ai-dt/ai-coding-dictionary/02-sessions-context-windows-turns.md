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

정보를 다음으로 **이어 보내지 않는다**. [Model](./01-the-model.md#model)은 [Model provider requests](./01-the-model.md#model-provider-request) 사이에서 stateless다 — 매 요청이 [Context window](#context-window) 전체를 다시 보내야 한다. 그게 모델이 볼 수 있는 유일한 통로니까. [Agent](#agent)도 기본은 [Sessions](#session) 간에 stateless — 새 세션은 비어 있는 채로 시작한다. [Stateful](#stateful)의 반대.

**💬 실전 대화 예시**
> "왜 [clear](./05-handoffs.md#clearing) 할 때마다 컨벤션을 까먹어?"
> "모델은 stateless야 — 새 세션이 비어서 시작하잖아. 유지하고 싶으면 [AGENTS.md](./06-memory-and-steering.md#agentsmd) 같은 메모리 파일에 적어둬, 하네스가 세션 시작에 로드하게."

---

### Context (컨텍스트)

**[Agent](#agent)가 지금 작업과 관련해서 알고 있는 정보**. 추상 명사다 — 모델이 보는 raw input(그건 [Context window](#context-window))도 아니고, 누적된 히스토리(그건 [Session](#session))도 아니다. *agent가 이 작업에 관해 아는 것*이다. "context에 로드한다"는 이 집합에 추가한다는 뜻이고, "context engineering"은 이 집합을 **큐레이션하는 분야**다.

**💬 실전 대화 예시**
> "타입에 없는 필드를 자꾸 만들어내."
> "타입 파일이 context에 없어 — 호출부만 보고 추측 중이야. 정의 파일을 먼저 읽혀."

🏢 **실무 적용**: "Context engineering"은 사실상 사내에서 가장 부족한 스킬. RAG/LangGraph 파이프라인 만들 때 *어떤 문서를, 어떤 순서로, 어떤 분량까지* 컨텍스트에 넣을지를 설계하는 일이 곧 context engineering.

---

### Context window (컨텍스트 윈도우)

매 [Model provider request](./01-the-model.md#model-provider-request)마다 [Model](./01-the-model.md#model)이 보는 **모든 것**. 유한하고, 모델별로 크기가 다르고, **모델이 무엇을 인지할 수 있는 유일한 표면**이다.

**❌ 피할 표현**: "memory" — context window는 작업 상태(working state)이고 [Session](#session) 간에 유지되지 않는다. [Memory](./06-memory-and-steering.md#memory-system)는 그 위에 얹힌 별개 개념이다.

**💬 실전 대화 예시**
> "그냥 모노레포 전체를 프롬프트에 붙일 수 있어?"
> "Context window가 200k [tokens](./01-the-model.md#token)야 — 레포의 1/5쯤이야. 작업이 건드리는 파일만 골라, 나머지는 [Tool call](./03-tools-environment.md#tool-call) 뒤에 두고."

---

### Stateful (스테이트풀)

정보를 **다음으로 이어 보낸다**. [Session](#session)은 [Turns](#turn) 간에 stateful — 세션이 진행되면서 [Context](#context)가 누적되고, 그래서 긴 세션은 [dumb zone](./04-failure-modes.md#smart-zone)으로 흘러간다. [Agent](#agent)는 [Memory system](./06-memory-and-steering.md#memory-system)을 추가하면 **세션 간**에도 stateful이 될 수 있다 — 정보를 [Environment](./03-tools-environment.md#environment)에 영속화하고 다음 세션 시작에 다시 로드한다. **[Model](./01-the-model.md#model) 자체는 절대 stateful이 아니다** — 연속성처럼 보이는 건 [Harness](./01-the-model.md#harness)가 컨텍스트를 다시 먹이는 것이다. [Stateless](#stateless)의 반대.

**💬 실전 대화 예시**
> "어제 내 선호를 기억했어 — 모델이 학습한 거야?"
> "아니, 에이전트가 stateful해진 거야. 하네스가 메모리 파일에 적어두고 세션 시작에 다시 로드한 거지. 모델 자체는 어제 본 게 아무것도 없어."

---

### Agent (에이전트)

[Tools](./03-tools-environment.md#tool), [System prompt](#system-prompt), [Context window](#context-window)로 [harnessed](./01-the-model.md#harness)된 [Model](./01-the-model.md#model). 사용자와 [Turns](#turn)을 주고받는다. **Claude Code는 에이전트다. Cursor는 에이전트다. Claude.ai는 에이전트다**. 네가 실제로 대화하는 대상이 바로 에이전트 — 목적에 맞게 설정된, 움직이는 모델.

**❌ 피할 표현**: "the AI", "the bot" (너무 모호함 — 파라미터를 말하는지 하네스로 감싼 걸 말하는지 흐려짐)

**💬 실전 대화 예시**
> "마이그레이션은 어떤 에이전트 써?"
> "로컬은 Claude Code, UI는 Cursor — 같은 모델 쓰는데 하네스가 달라."

---

### System prompt (시스템 프롬프트)

[Harness](./01-the-model.md#harness)가 매 [Model provider request](./01-the-model.md#model-provider-request) 앞에 붙이는 지시문 — [Agent](#agent)의 **상시 브리핑**: 누구인지, 어떻게 행동해야 하는지, 어떤 [Tools](./03-tools-environment.md#tool)을 부를 수 있는지, 어떤 컨벤션을 따라야 하는지. 보통 [Session](#session) 동안 안정적으로 유지된다.

**💬 실전 대화 예시**
> "두 하네스, 같은 [Model](./01-the-model.md#model), 같은 프롬프트인데 동작이 완전 달라."
> "System prompt가 다른 거야. 하나는 짧은 코드 편집용으로 튜닝됐고, 다른 하나는 설명용이야 — 분기점이 거기 있어, 네 메시지가 도착하기도 전에."

🏢 **실무 적용**: 사내에서 만드는 모든 LangGraph 노드/Agent는 **자기만의 system prompt**를 가져야 한다. 노드가 동작이 이상하면 항상 system prompt부터 의심.

---

### Session (세션)

[Agent](#agent)와의 **한 번의 경계 지어진 상호작용**. 비어서 시작하고, 메시지·[Tool results](./03-tools-environment.md#tool-result)·읽힌 파일이 누적되고, [cleared](./05-handoffs.md#clearing)·종료·[compaction](./05-handoffs.md#compaction)으로 끝난다.

**[Context window](#context-window)를 *채우는* 게 세션이다**: context window가 상자라면 session은 그 상자를 천천히 채우는 내용물이다. 하나의 context window로 감당 안 되는 일은 여러 세션으로 나눠야 한다.

**💬 실전 대화 예시**
> "한 세션이 무너지기 전까지 얼마나 갈 수 있어?"
> "작업에 따라 달라 — 집중된 리팩토링은 오래 가고, 발산적 리서치는 빨리 무너져. 세션이 비대해지면 [hand off](./05-handoffs.md#handoff)하거나 compact해, 밀어붙이지 말고."

---

### Turn (턴)

**사용자 메시지 하나 + 그에 대한 [Agent](#agent)의 모든 동작**, 사용자에게 다시 양보(yield)할 때까지. 도구를 부르면 안에 [Model provider requests](./01-the-model.md#model-provider-request)가 여러 개 들어간다. 명확화 질문은 턴을 닫고, 사용자의 답이 다음 턴을 연다.

**계층: [Session](#session) > Turn > Model provider request**

**💬 실전 대화 예시**
> "한 턴이 2분 걸렸다고?"
> "그 안에서 [tool calls](./03-tools-environment.md#tool-call)을 14번 했어 — 하나하나가 별도 model provider request야. 결국 너에게 양보하기 전까지 지연이 쌓인 거지."

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
