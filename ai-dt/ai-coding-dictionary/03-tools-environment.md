---
tags: [ai-coding, tool, mcp, environment, sandbox, permission]
level: intermediate
last_updated: 2026-05-05
source: https://github.com/mattpocock/dictionary-of-ai-coding
---

# Section 3 — Tools & Environment (도구와 환경)

> [Agent](./02-sessions-context-windows-turns.md#agent)가 **할 수 있는 일(Tools)** 과 그 일이 **벌어지는 무대(Environment)**.

## 왜 이 섹션이 중요한가? (Why)

- "Claude Code가 파일을 만진다"는 말은 정확히 **무엇**이 무엇을 만지는 것인가?
  - 모델은 텍스트만 뱉는다. 그 텍스트를 보고 **하네스**가 **도구**를 실행한다. 그 도구가 닿는 곳이 **환경**이다.
- MCP, sandbox, permission mode 같은 단어가 다 이 섹션 안에 있다.

## 용어 (What & How)

### Environment (환경)

[Agent](./02-sessions-context-windows-turns.md#agent)가 **작용하는 세계**. [Harness](./01-the-model.md#harness) 바깥에서, agent가 [Tool results](#tool-result)를 통해 인지하고 [Tool calls](#tool-call)로 변경하는 모든 것.

> 하네스는 *agent를 실행*하고, 환경은 *agent가 일하는 작업장*이다.

[`AGENTS.md`](./06-memory-and-steering.md#agentsmd) 같은 파일은 environment에 있고, 하네스는 그걸 [Context window](./02-sessions-context-windows-turns.md#context-window)로 *로드*하는 역할이다. [Filesystem](#filesystem)이 가장 흔한 환경이지만, DB·원격 API·브라우저 세션 모두 환경이 될 수 있다.

**❌ 피할 표현**: 런타임이나 하네스 자체를 "environment"라고 부르지 말 것 — 하네스는 *래퍼*, 환경은 *작업장*.

**💬 실전 대화 예시**
> "에이전트가 staging DB 스키마를 못 봐."
> "Environment에 연결해 — staging에 read-only로 스코프된 `psql` [tool](#tool)을 줘. 하네스는 멀쩡해, 작용할 대상이 없을 뿐이야."

---

### Filesystem (파일시스템)

[Agent](./02-sessions-context-windows-turns.md#agent)가 읽고, 쓰고, 실행하는 **파일과 디렉토리의 트리** — 코딩 에이전트의 **기본 [Environment](#environment)**. [AGENTS.md](./06-memory-and-steering.md#agentsmd), [Skills](./06-memory-and-steering.md#skill), 소스 코드, 빌드 스크립트, [Tool](#tool) 설정이 다 파일시스템에 산다. 하네스가 "프로젝트에서 시작"한다는 건 에이전트를 그 파일시스템에 가리킨다는 뜻이다.

**💬 실전 대화 예시**
> "왜 내 AGENTS.md를 안 읽어?"
> "다른 파일시스템에서 돌고 있어 — [Sandbox](#sandbox)가 프로젝트 루트가 아니라 부모 디렉토리를 마운트했어. 하네스 다시 가리켜."

---

### Tool (도구)

[Harness](./01-the-model.md#harness)가 [Agent](./02-sessions-context-windows-turns.md#agent)에게 노출하는 **함수** — Read, Write, Bash, Search 등. 도구는 에이전트가 [Environment](#environment)를 인지하고 변경하는 **유일한 통로**다: 도구 없이는 환경을 보지도 바꾸지도 못한다. 매 도구 호출은 추가 [Model provider request](./01-the-model.md#model-provider-request) 비용이 든다 — 결과가 모델로 돌아가야 다음 행동을 정할 수 있으니까.

**💬 실전 대화 예시**
> "에이전트가 staging을 직접 쿼리할 수 있어?"
> "하네스에 `psql` tool 추가해, staging에 read-only로 스코프해서. Tool이 없으면 [Filesystem](#filesystem) 바깥은 다 깜깜이야."

🏢 **실무 적용**: 사내에서 LangGraph로 에이전트 짜면서 외부 시스템 붙일 때, 매번 "tool 하나 = model provider request 하나 더"라는 비용을 잊지 말 것. **batch tool**이나 **multi-tool**로 묶을 수 있는 건 묶기.

---

### Tool call (도구 호출)

[Model](./01-the-model.md#model)의 출력 중에서 **도구 이름과 인자를 명시한 부분** — 그냥 구조화된 텍스트일 뿐이다. **그 자체로는 아무 일도 하지 않는다**; [Harness](./01-the-model.md#harness)가 이걸 읽고 실행해야 일이 된다. 하나의 [Model provider request](./01-the-model.md#model-provider-request)에서 모델이 만들어낸다.

**💬 실전 대화 예시**
> "테스트 돌렸다는데 파일 타임스탬프가 안 바뀌었어."
> "트랜스크립트 봐 — 실제로 tool call을 뱉었어, 아니면 돌렸다고 *서술만* 했어? 모델이 호출을 만들어도 하네스가 실행 안 했으면 아무 일도 안 일어난 거야."

---

### Tool result (도구 결과)

[Harness](./01-the-model.md#harness)가 [Tool call](#tool-call)을 실행한 뒤 돌려보내는 것 — 파일 내용, 명령 출력, 에러 등. **[Agent](./02-sessions-context-windows-turns.md#agent)가 [Environment](#environment)를 보는 유일한 창**. *다음* [Model provider request](./01-the-model.md#model-provider-request)에서 모델로 전달되어, 모델이 그걸 가지고 무엇을 할지 결정한다. **Tool call과 tool result는 같은 교환의 양 끝**이고, 둘 다 한 [Turn](./02-sessions-context-windows-turns.md#turn) 안에 있다.

**💬 실전 대화 예시**
> "파일이 비었다는 듯이 추론하고 있어."
> "Tool result가 권한 거부로 돌아왔어, 내용이 아니라. 모델은 에러 문자열만 봤어 — 파일을 들여다본 게 아니야."

---

### MCP (Model Context Protocol)

외부 도구 서버를 [Harness](./01-the-model.md#harness)에 꽂는 **프로토콜** — [Agent](./02-sessions-context-windows-turns.md#agent)가 하네스 기본 도구를 넘어서 더 많은 [Tools](#tool)을 얻는 방식. 에이전트는 절대 "MCP를 호출"하지 않는다 — 에이전트는 **도구**를 호출하고, 그 도구가 우연히 MCP 서버에서 온 것일 뿐이다. 리소스(읽기 전용 데이터)와 프롬프트(재사용 템플릿)도 노출하지만, 주된 용도는 도구 제공.

**💬 실전 대화 예시**
> "에이전트가 Linear에서 티켓 읽어야 해."
> "하네스에 Linear MCP 서버 설정해 — Linear API를 도구로 노출해줘. 커스텀 wrapper 안 짜도 돼."

🏢 **실무 적용**: 자세한 내용은 [MCP 기초](../mcp/mcp-basics.md), [MCP + LangGraph](../mcp/mcp-langgraph-integration.md) 참고.

---

### Permission request (권한 요청)

[Harness](./01-the-model.md#harness)가 사전 승인되지 않은 [Tool call](#tool-call)을 실행하기 전에 사용자에게 보여주는 것. 모델이 tool call을 만들면, 즉시 실행하는 게 아니라 하네스가 멈추고 묻는다. 승인 → 실행. 거부 → 거부 사실을 [Tool result](#tool-result)로 모델에 보고. **하네스가 위험·민감 동작에 [Human-in-the-loop](./07-patterns-of-work.md#human-in-the-loop)을 끼워넣는 메커니즘**.

**💬 실전 대화 예시**
> "권한 요청에서 10분이나 막혀 있었어 — 회의 중이었거든."
> "그게 human-in-the-loop의 비용이야. 안전한 [tools](#tool)은 미리 승인해서, 진짜 위험한 호출에서만 요청이 뜨도록 해."

---

### Permission mode (권한 모드)

[Agent mode](#agent-mode)에서 **권한 게이팅 부분**만 떼어낸 슬라이스 — 어떤 [Tool calls](#tool-call)이 [Permission request](#permission-request)을 띄우고, 어떤 게 자동 실행되는지. 하네스가 행동 지시문을 함께 묶어 팔기 시작하기 *전*의 원래 모드 시스템 목적이다.

**💬 실전 대화 예시**
> "grep 할 때마다 멈춰서 [AFK](./07-patterns-of-work.md#afk) 런이 다 망가졌어."
> "Read-only 도구는 permission mode를 느슨하게, write/shell은 prompt 유지. 리서치 세션의 권한 요청은 대부분 노이즈야."

---

### Agent mode (에이전트 모드)

런타임에 [Agent](./02-sessions-context-windows-turns.md#agent)가 어떻게 동작할지 정하는 **프리셋** — [Permission mode](#permission-mode)와 [System prompt](./02-sessions-context-windows-turns.md#system-prompt)에 주입되는 행동 지시문을 묶어둔 것.

**예시**:
- *기본 모드* — 위험 호출에서 prompt
- *plan mode* — edit을 막고 리서치 쪽으로 유도
- *accept-edits* — edit 자동 승인
- *bypass permissions* (구어로 **YOLO mode**) — 모든 걸 자동 승인

세션 도중에도 전환 가능.

**벤더 차이**: Claude Code는 "permission modes", Codex는 "approval modes" — 둘 다 행동 묶음 이전 세대 명칭.

**💬 실전 대화 예시**
> "계획만 짜라고 했는데 자꾸 파일을 편집해."
> "Plan mode로 바꿔 — write를 막고 리서치만 하게."
> "그럼 나중에 [AFK](./07-patterns-of-work.md#afk) 런은?"
> "Bypass mode인데, [Sandbox](#sandbox) 안에서만."

---

### Sandbox (샌드박스)

[Agent](./02-sessions-context-windows-turns.md#agent)가 그 안에서 도는 **격리된 [Environment](#environment)** — 컨테이너, VM, 일회용 [Filesystem](#filesystem), 권한 제한 셸. 에이전트 동작의 **반경(blast radius)** 을 제한한다: 파괴적 명령을 돌리거나 악성 콘텐츠를 받아도 피해가 격리된다. **[AFK](./07-patterns-of-work.md#afk)를 실용적으로 만드는 안전 기반**.

**💬 실전 대화 예시**
> "밤새 [bypass-permissions](#agent-mode)로 돌리고 싶은데 아직 무서워."
> "Sandbox에 넣어 — 새 컨테이너, credential 마운트 안 함, 외부망 차단. 최악의 경우라도 자기 파일시스템만 날리고, 컨테이너 버리면 끝이야."

🏢 **실무 적용**: 사내에서 자동화 에이전트를 운영할 때 **반드시 read-only DB credential, 격리 네트워크, 폐기 가능 컨테이너** 조합으로 시작. 실패해도 잃을 게 없는 환경에서만 bypass mode를 쓴다.

## 이 섹션 요약 (Cheatsheet)

```
[Agent]
   │
   ├─ 출력: Tool call (그냥 텍스트)
   │       └─→ [Harness] 가 받아서 → Permission check → 실행
   │
   └─ 입력: Tool result
           └─← [Harness] 가 [Environment] (Filesystem/DB/API) 에서 가져옴
```

| 헷갈리는 쌍 | 차이 |
|---|---|
| Tool vs Skill | Tool은 *호출*하는 함수. [Skill](./06-memory-and-steering.md#skill)은 *읽는* 지시문 |
| Tool call vs Tool result | 호출(에이전트→하네스) vs 결과(하네스→에이전트), 둘 다 한 Turn 안 |
| Environment vs Harness | Env는 작업장, Harness는 래퍼 (헷갈리지 말 것) |
| Permission mode vs Agent mode | 권한만 vs 권한+행동 지시문 묶음 |

## 관련 문서

- 이전: [02 - Sessions, Context Windows & Turns](./02-sessions-context-windows-turns.md)
- 다음: [04 - Failure Modes](./04-failure-modes.md)
- 인덱스: [README](./README.md)
- 사내 연결: [MCP 기초](../mcp/mcp-basics.md)

## 참고 자료 (References)

- 원문: [mattpocock/dictionary-of-ai-coding — Section 3](https://github.com/mattpocock/dictionary-of-ai-coding#section-3--tools--environment)
- MCP 공식: https://modelcontextprotocol.io
