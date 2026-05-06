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

[Agent](./02-sessions-context-windows-turns.md#agent)가 **실제로 작용하는 무대**. [Harness](./01-the-model.md#harness) 바깥에 있으며, agent가 [Tool results](#tool-result)를 통해 인지하고 [Tool calls](#tool-call)로 변경하는 모든 대상이 환경에 해당한다.

> 하네스는 *agent를 실행시키는 쪽*, 환경은 *agent가 일하는 작업장 쪽*이다.

[`AGENTS.md`](./06-memory-and-steering.md#agentsmd) 같은 파일은 environment에 존재하고, 하네스는 그것을 [Context window](./02-sessions-context-windows-turns.md#context-window)로 *불러오는* 역할을 한다. 가장 흔한 환경은 [Filesystem](#filesystem)이지만, DB나 원격 API, 브라우저 세션도 모두 환경이 될 수 있다.

**❌ 피할 표현**: 런타임이나 하네스 자체를 "environment"라고 부르지 말 것. 하네스는 *래퍼*고, 환경은 *작업장*이다.

**💬 실전 대화 예시**
> "에이전트가 staging DB 스키마를 못 봐."
> "Environment 쪽에 연결해 줘야 해. staging에 read-only로 스코프된 `psql` [tool](#tool)을 붙여 줘. 하네스 자체는 멀쩡한데, 작용할 대상이 없을 뿐이야."

---

### Filesystem (파일시스템)

[Agent](./02-sessions-context-windows-turns.md#agent)가 읽고, 쓰고, 실행하는 **파일과 디렉토리의 트리**. 코딩 에이전트에서는 이게 **기본 [Environment](#environment)**가 된다. [AGENTS.md](./06-memory-and-steering.md#agentsmd), [Skills](./06-memory-and-steering.md#skill), 소스 코드, 빌드 스크립트, [Tool](#tool) 설정 등이 모두 이 파일시스템 안에 들어 있다. 하네스가 "프로젝트에서 시작한다"는 말은, 결국 에이전트가 그 파일시스템을 바라보도록 세팅한다는 뜻이다.

**💬 실전 대화 예시**
> "왜 내 AGENTS.md를 안 읽어?"
> "지금 다른 파일시스템에서 돌고 있어. [Sandbox](#sandbox)가 프로젝트 루트가 아니라 부모 디렉토리를 마운트해 둬서 그래. 하네스가 가리키는 위치를 다시 잡아 줘."

---

### Tool (도구)

[Harness](./01-the-model.md#harness)가 [Agent](./02-sessions-context-windows-turns.md#agent)에게 노출하는 **함수들**. Read, Write, Bash, Search 같은 것들이 여기에 해당한다. 도구는 에이전트가 [Environment](#environment)를 인지하고 변경할 수 있는 **유일한 통로**다. 도구가 없으면 환경을 들여다볼 수도, 바꿀 수도 없다. 또한 도구 호출은 매번 추가 [Model provider request](./01-the-model.md#model-provider-request) 비용을 발생시킨다. 도구 결과가 모델로 다시 돌아와야 그다음 행동을 결정할 수 있기 때문이다.

**💬 실전 대화 예시**
> "에이전트가 staging을 직접 쿼리할 수 있게 만들 수 있어?"
> "하네스에 `psql` tool을 붙이면 돼. staging에 read-only로 스코프해서 주면 안전해. Tool이 없으면 [Filesystem](#filesystem) 바깥 세계는 에이전트한텐 깜깜이야."

🏢 **실무 적용**: 사내에서 LangGraph로 에이전트를 짜면서 외부 시스템을 붙일 때, "tool 하나 추가 = model provider request 한 번 추가"라는 비용 구조를 항상 염두에 둬야 한다. **batch tool**이나 **multi-tool**로 묶을 수 있는 호출은 가능한 한 묶어서 처리하는 게 좋다.

---

### Tool call (도구 호출)

[Model](./01-the-model.md#model)의 출력 중에서 **도구 이름과 인자를 명시한 부분**. 본질적으로는 그냥 구조화된 텍스트일 뿐이다. **그 자체만으로는 아무 일도 일어나지 않고**, [Harness](./01-the-model.md#harness)가 그것을 읽어서 실제로 실행해야 비로소 동작이 발생한다. 하나의 [Model provider request](./01-the-model.md#model-provider-request) 안에서 모델이 만들어낸다.

**💬 실전 대화 예시**
> "테스트 돌렸다는데 파일 타임스탬프가 안 바뀌었어."
> "트랜스크립트 한번 확인해 봐. 실제로 tool call을 뱉었어, 아니면 그냥 *돌렸다는 서술*만 한 거야? 모델이 호출을 만들었어도 하네스가 실행하지 않았다면 아무 일도 안 일어난 거야."

---

### Tool result (도구 결과)

[Harness](./01-the-model.md#harness)가 [Tool call](#tool-call)을 실행한 뒤 다시 돌려주는 결과물. 파일 내용, 명령어 출력, 에러 메시지 등이 여기에 들어간다. **[Agent](./02-sessions-context-windows-turns.md#agent)가 [Environment](#environment)를 들여다볼 수 있는 유일한 창구**이기도 하다. 이 결과는 *다음* [Model provider request](./01-the-model.md#model-provider-request)에서 모델로 전달되고, 모델은 그걸 토대로 다음 행동을 결정한다. **Tool call과 tool result는 같은 한 차례 교환의 양 끝이고**, 둘 다 한 [Turn](./02-sessions-context-windows-turns.md#turn) 안에서 일어난다.

**💬 실전 대화 예시**
> "파일이 비어 있는 것처럼 추론하고 있어."
> "Tool result가 파일 내용이 아니라 권한 거부 에러로 돌아왔어. 모델은 에러 문자열만 본 셈이지, 파일을 실제로 들여다본 게 아니야."

---

### MCP (Model Context Protocol)

외부 도구 서버를 [Harness](./01-the-model.md#harness)에 꽂아 쓸 수 있게 해주는 **프로토콜**. [Agent](./02-sessions-context-windows-turns.md#agent)가 하네스의 기본 도구 외에도 더 많은 [Tools](#tool)을 확보하는 방식이라고 보면 된다. 에이전트는 절대 "MCP를 호출"하지 않는다는 점에 주의해야 한다. 에이전트가 호출하는 건 어디까지나 **도구**이고, 그 도구의 출처가 MCP 서버일 뿐이다. MCP는 리소스(읽기 전용 데이터)와 프롬프트(재사용 가능한 템플릿)도 노출할 수 있지만, 주된 용도는 도구 제공이다.

**💬 실전 대화 예시**
> "에이전트가 Linear에서 티켓을 읽어야 해."
> "하네스에 Linear MCP 서버를 붙이면 돼. Linear API를 도구 형태로 노출해 주거든. 따로 커스텀 wrapper를 짤 필요가 없어."

🏢 **실무 적용**: 자세한 내용은 [MCP 기초](../mcp/mcp-basics.md), [MCP + LangGraph](../mcp/mcp-langgraph-integration.md) 참고.

---

### Permission request (권한 요청)

[Harness](./01-the-model.md#harness)가 사전 승인되지 않은 [Tool call](#tool-call)을 실행하기 직전에 사용자에게 띄우는 확인 절차. 모델이 tool call을 만들었다고 곧바로 실행되는 게 아니라, 하네스가 일단 멈추고 사용자에게 묻는다. 승인하면 실행되고, 거부하면 거부됐다는 사실이 [Tool result](#tool-result)로 모델에 보고된다. **하네스가 위험하거나 민감한 동작에 [Human-in-the-loop](./07-patterns-of-work.md#human-in-the-loop)을 끼워넣는 핵심 메커니즘**이다.

**💬 실전 대화 예시**
> "권한 요청 화면에서 10분이나 막혀 있었어. 회의 중이었거든."
> "그게 human-in-the-loop를 쓰는 데 따르는 비용이야. 안전한 [tools](#tool)은 미리 승인해 두고, 진짜 위험한 호출에서만 요청이 뜨게 만들어 두면 돼."

---

### Permission mode (권한 모드)

[Agent mode](#agent-mode)에서 **권한 게이팅 부분**만 따로 떼어낸 영역. 어떤 [Tool calls](#tool-call)이 [Permission request](#permission-request)를 띄우고, 어떤 호출이 자동으로 실행되는지를 결정한다. 하네스가 행동 지시문까지 같이 묶어 팔기 *이전의*, 모드 시스템 본래의 목적이라고 볼 수 있다.

**💬 실전 대화 예시**
> "grep 한 번 할 때마다 멈춰 서서 [AFK](./07-patterns-of-work.md#afk) 런이 다 망가졌어."
> "Read-only 도구들은 permission mode를 느슨하게 풀어 두고, write나 shell처럼 위험한 쪽만 prompt를 유지하는 게 좋아. 리서치 세션에서 뜨는 권한 요청은 대부분 노이즈에 가까워."

---

### Agent mode (에이전트 모드)

런타임에서 [Agent](./02-sessions-context-windows-turns.md#agent)가 어떻게 동작할지 결정하는 **프리셋**. [Permission mode](#permission-mode)와 [System prompt](./02-sessions-context-windows-turns.md#system-prompt)에 주입되는 행동 지시문을 한 묶음으로 만들어 둔 것이라고 보면 된다.

**예시**:
- *기본 모드*: 위험한 호출에서는 prompt를 띄움
- *plan mode*: edit을 막고 리서치 쪽으로 유도
- *accept-edits*: edit을 자동으로 승인
- *bypass permissions* (구어로 **YOLO mode**): 모든 호출을 자동 승인

세션 도중에도 모드를 자유롭게 전환할 수 있다.

**벤더 차이**: Claude Code에서는 "permission modes", Codex에서는 "approval modes"라고 부른다. 둘 다 행동 지시문 묶음이 도입되기 이전 세대의 명칭이다.

**💬 실전 대화 예시**
> "계획만 짜 달라고 했는데 자꾸 파일을 편집해."
> "Plan mode로 바꿔. write를 막고 리서치만 하게 해 줘."
> "그럼 나중에 [AFK](./07-patterns-of-work.md#afk) 런 돌릴 때는?"
> "그땐 bypass mode로 가되, 반드시 [Sandbox](#sandbox) 안에서만 돌려야 해."

---

### Sandbox (샌드박스)

[Agent](./02-sessions-context-windows-turns.md#agent)가 그 안에서 동작하는 **격리된 [Environment](#environment)**. 컨테이너, VM, 일회용 [Filesystem](#filesystem), 권한이 제한된 셸 같은 형태가 대표적이다. 샌드박스의 핵심 역할은 에이전트 동작의 **영향 반경(blast radius)을 제한**하는 것이다. 파괴적인 명령을 실행하거나 악성 콘텐츠를 받아도 그 피해가 샌드박스 안에 갇히기 때문이다. **[AFK](./07-patterns-of-work.md#afk)를 실용적으로 만들어 주는 안전 기반**이라고 봐도 된다.

**💬 실전 대화 예시**
> "밤새 [bypass-permissions](#agent-mode)로 돌리고는 싶은데 아직 좀 무서워."
> "Sandbox 안에 넣어 두면 안전해. 새 컨테이너에 credential은 마운트하지 않고, 외부망도 차단해 둬. 최악의 경우라도 자기 파일시스템만 날아가고, 컨테이너만 버리면 끝이야."

🏢 **실무 적용**: 사내에서 자동화 에이전트를 운영할 때는 **read-only DB credential + 격리 네트워크 + 폐기 가능한 컨테이너** 조합으로 시작하는 것이 안전하다. bypass mode는 실패해도 잃을 게 없는 환경에서만 사용해야 한다.

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
