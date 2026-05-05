---
tags: [ai-coding, memory, agents-md, skill, subagent, progressive-disclosure]
level: intermediate
last_updated: 2026-05-05
source: https://github.com/mattpocock/dictionary-of-ai-coding
---

# Section 6 — Memory and Steering (메모리와 조종)

> [Agent](./02-sessions-context-windows-turns.md#agent)가 **여러 세션에 걸쳐 무엇을 기억할지**, 그리고 **그 기억을 어떻게 효율적으로 로드할지**.

## 왜 이 섹션이 중요한가? (Why)

- "어제 가르친 걸 오늘 또 가르쳐야 한다"는 좌절은 [Memory system](#memory-system)이 없어서다.
- 그렇다고 모든 걸 [AGENTS.md](#agentsmd)에 넣으면 **매 [Turn](./02-sessions-context-windows-turns.md#turn)마다 토큰 비용이 든다**.
- 해법은 **[Progressive disclosure](#progressive-disclosure)**: 필요할 때만 로드.

## 용어 (What & How)

### Memory system (메모리 시스템)

[Agent](./02-sessions-context-windows-turns.md#agent)를 [Sessions](./02-sessions-context-windows-turns.md#session) 간에 [Stateful](./02-sessions-context-windows-turns.md#stateful)로 만들려는 시스템. 세션 동안 정보를 [Environment](./03-tools-environment.md#environment)에 영속화하고, 미래 세션 시작 시 [Context window](./02-sessions-context-windows-turns.md#context-window)에 다시 로드한다 — 사용자가 세션을 [Clear](./05-handoffs.md#clearing)한 뒤에도 연속성을 갖도록.

**💬 실전 대화 예시**
> "MySQL 아니라 Postgres라고 매번 다시 말해야 해."
> "Memory system 붙여 — 첫 [Turn](./02-sessions-context-windows-turns.md#turn)에 학습한 걸 [Filesystem](./03-tools-environment.md#filesystem)에 적고, 세션 시작에 다시 로드해. [Model](./01-the-model.md#model) 자체는 [stateless](./02-sessions-context-windows-turns.md#stateless)니까, 메모리 레이어가 연속성을 *흉내*내는 거야."

🏢 **실무 적용**: Claude Code의 `CLAUDE.md`(이 레포 최상위에 있는 그 파일!)도 일종의 memory system. 사용자 선호와 컨벤션을 매 세션 시작에 자동 로드한다.

---

### AGENTS.md

[Environment](./03-tools-environment.md#environment)에 있는 파일로, [Harness](./01-the-model.md#harness)가 [Session](./02-sessions-context-windows-turns.md#session) 시작에 [Context window](./02-sessions-context-windows-turns.md#context-window)로 **로드한다** — 프로젝트가 [Agent](./02-sessions-context-windows-turns.md#agent)에게 주는 **상시 브리핑**. **하네스 간 공통 컨벤션**.

**❌ 피할 표현**: AGENTS.md를 [Progressive disclosure](#progressive-disclosure)되어야 할 콘텐츠로 쓰지 말 것 — 여기 들어가는 모든 건 매 [Turn](./02-sessions-context-windows-turns.md#turn)마다 [Token](./01-the-model.md#token) 비용을 낸다.

**💬 실전 대화 예시**
> "왜 매 세션이 4k 토큰을 이미 태운 채 시작해?"
> "AGENTS.md 봐 — 누군가 스타일 가이드 전체를 거기에 붙였어, [Skill](#skill) 뒤에 두지 않고."

🏢 **실무 적용**: 사내 프로젝트의 `CLAUDE.md`는 **얇게 유지**해야 한다. 자주 안 쓰는 컨벤션은 skill로 분리.

---

### Progressive disclosure (점진적 노출)

[Agent](./02-sessions-context-windows-turns.md#agent)가 **지금 필요한 [Context](./02-sessions-context-windows-turns.md#context)만** 로드하고, 나머지는 *포인터*로 두는 패턴. UI 디자인에서 빌려온 개념.

**💬 실전 대화 예시**
> "스타일 가이드 전체를 [AGENTS.md](#agentsmd)에 부어버릴까?"
> "안 돼 — progressive disclosure 해. 스타일 가이드를 [Skill](#skill)로 참조하고, 에이전트가 컴포넌트 짤 때만 로드하게. AGENTS.md에 두면 매 [Turn](./02-sessions-context-windows-turns.md#turn)마다 [Token](./01-the-model.md#token) 비용 내는 거야."

---

### Skill (스킬)

**가르칠 수 있는 능력 단위로 묶은 것** — 어떤 한 작업을 잘 하기 위한 지시문과 리소스 묶음. [Environment](./03-tools-environment.md#environment)에 보관되고, 관련 있을 때만 [Context window](./02-sessions-context-windows-turns.md#context-window)로 로드된다. **[Harness](./01-the-model.md#harness)에서 [Progressive disclosure](#progressive-disclosure)의 단위**.

**❌ 피할 표현**: "[Tool](./03-tools-environment.md#tool)" — tool은 [Agent](./02-sessions-context-windows-turns.md#agent)가 *호출*하는 것이고, skill은 *읽는* 지시문이다.

**💬 실전 대화 예시**
> "Deploy runbook 어디에 둘까?"
> "Skill로 — 에이전트가 deploy 관련 작업할 때만 로드해. [AGENTS.md](#agentsmd)에 두면 주간 1회 쓸 일에 매 turn token 비용 내는 거야."

---

### Subagent (서브에이전트)

다른 [Agent](./02-sessions-context-windows-turns.md#agent)가 [Tool call](./03-tools-environment.md#tool-call)로 spawn한 에이전트. **자기만의 [Session](./02-sessions-context-windows-turns.md#session), 자기만의 [Context window](./02-sessions-context-windows-turns.md#context-window)** 로 돌고, **단일 [Tool result](./03-tools-environment.md#tool-result)** 를 부모에게 보고한다. [Handoff](./05-handoffs.md#handoff)와 다름 — 부모는 *반환을 명시적으로 기대*하고, handoff는 반환 경로가 *없다*. **subagent는 더 깊은 subagent를 spawn할 수 없다** — 트리는 1단 깊이다. **계층을 쌓기 위해서가 아니라, [Context](./02-sessions-context-windows-turns.md#context)를 격리하기 위해 존재**.

**💬 실전 대화 예시**
> "Grep 결과가 내 컨텍스트를 폭발시켜."
> "Subagent를 spawn해서 검색 시켜 — 자기 context window에 노이즈를 다 태우고, 네가 실제로 필요한 파일 경로 두 개만 보고할 거야."

🏢 **실무 적용**: LangGraph의 **subgraph**, 그리고 OpenAI의 **assistants**도 본질적으로 같은 패턴 — context isolation. 자세한 건 [LangGraph 고급](../rag/langgraph/langgraph-advanced.md) 참고.

## 이 섹션 요약 (Cheatsheet)

```
[Token 비용 관점에서 어디에 무엇을 둘 것인가]

매 turn 비용 ───────────────────────────── 비용 없음 (lazy load)
   │                                                    │
[AGENTS.md]                                          [Skill]
   • 얇게 유지                                         • 큰 가이드/runbook
   • 항상 알아야 하는 컨벤션                           • 가끔 필요한 도메인 지식
   • 프로젝트 정체성                                   • 특정 task 모드
```

| 단어 | 한 줄 정의 |
|---|---|
| Memory system | 세션 간 stateful 흉내. 영속화 + 재로드 |
| AGENTS.md | 매 세션 시작에 자동 로드되는 브리핑 |
| Progressive disclosure | "필요할 때만 로드" 원칙 |
| Skill | progressive disclosure의 단위 — 읽는 지시문 |
| Subagent | context 격리용 1단 깊이 자식 에이전트 |

## 관련 문서

- 이전: [05 - Handoffs](./05-handoffs.md)
- 다음: [07 - Patterns of Work](./07-patterns-of-work.md)
- 인덱스: [README](./README.md)
- 사내 연결: [LangGraph 고급 (Subgraph)](../rag/langgraph/langgraph-advanced.md)

## 참고 자료 (References)

- 원문: [mattpocock/dictionary-of-ai-coding — Section 6](https://github.com/mattpocock/dictionary-of-ai-coding#section-6--memory-and-steering)
- AGENTS.md 컨벤션: https://agentsmd.net (또는 Claude Code의 CLAUDE.md)
