---
tags: [ai-coding, glossary, agent, claude-code, llm]
level: beginner-intermediate
last_updated: 2026-05-05
source: https://github.com/mattpocock/dictionary-of-ai-coding
---

# AI Coding Dictionary (한국어 학습 노트)

> Matt Pocock의 [Dictionary of AI Coding](https://github.com/mattpocock/dictionary-of-ai-coding)을 한국어로 풀어 정리한 학습 노트. Claude Code, Cursor, Codex 같은 AI 코딩 도구를 쓰면서 마주치는 용어를 "왜 필요한가 → 무엇인가 → 어떻게 쓰는가" 순서로 정리했다.

## 왜 필요한가? (Why)

- AI 코딩 도구를 사용하다 보면 토큰, 컨텍스트, 하네스, 에이전트 모드 같은 **낯선 어휘**가 한꺼번에 쏟아진다.
- 같은 모델인데 도구마다 동작이 다르거나 같은 프롬프트인데 비용이 폭증하는 현상은, **용어를 정확히 알지 못하면 디버깅 자체가 불가능**하다.
- 원문 저자의 표현을 빌리면: *"모호함의 상당 부분은 의도적으로 만들어진 것이다. AI 코딩 업계에는, 이 어휘를 어렵게 유지함으로써 이득을 보는 VC 자금이 흐른다."*
- 이 사전을 한 번이라도 훑어 두면 **요금, 어텐션, 컨텍스트 저하 같은 현상에 이름을 붙일 수 있게 된다.** 그리고 이름을 붙일 수 있다는 것이 디버깅의 출발점이다.

## 핵심 개념 (What) — 7개 섹션

원문은 약 60개의 용어를 7개 섹션으로 나누고 있다. 이 노트에서도 같은 구조를 따라, 섹션마다 별도 문서로 정리했다.

| # | 섹션 | 다루는 것 | 문서 |
|---|------|-----------|------|
| 1 | The Model | 모델 자체. 파라미터, 학습, 추론, 토큰, 비용 구조 | [01-the-model.md](./01-the-model.md) |
| 2 | Sessions, Context Windows & Turns | 에이전트가 상태를 유지하고 사용자와 주고받는 단위 | [02-sessions-context-windows-turns.md](./02-sessions-context-windows-turns.md) |
| 3 | Tools & Environment | 에이전트의 능력과 그 능력이 작동하는 환경 | [03-tools-environment.md](./03-tools-environment.md) |
| 4 | Failure Modes | 환각, 어텐션 저하, 지식 한계 등 실패 양상 | [04-failure-modes.md](./04-failure-modes.md) |
| 5 | Handoffs | 세션 간 작업 인계 메커니즘 | [05-handoffs.md](./05-handoffs.md) |
| 6 | Memory and Steering | 영속성과 행동 제어 | [06-memory-and-steering.md](./06-memory-and-steering.md) |
| 7 | Patterns of Work | Vibe coding, AFK, Grilling 등 실무 작업 패턴 | [07-patterns-of-work.md](./07-patterns-of-work.md) |

## 어떻게 사용하는가? (How)

### 학습 순서 추천

1. **Section 1 (The Model)** 부터 시작하는 것이 좋다. 모델/하네스/추론/토큰의 구분이 잡혀 있지 않으면 나머지 섹션들이 모두 흔들린다.
2. **Section 2 (Sessions)** 에서 `Session > Turn > Model provider request` 계층을 이해해 두면, 그제야 비용 구조를 머릿속에서 계산할 수 있게 된다.
3. **Section 3 (Tools & Environment)** 을 보면, Claude Code나 Cursor 같은 하네스가 실제로 어떤 일을 하는지가 분명히 드러난다.
4. **Section 4 (Failure Modes)** 는 일상에서 마주치는 "왜 이상하지?"의 90% 이상이 모여 있는 섹션이다.
5. **5~7번 섹션은 실무 운영 단계**에 해당한다. 긴 작업을 어떻게 나눌지(Handoff), 무엇을 기억시킬지(Memory), 어떤 패턴으로 협업할지(AFK, Grilling) 같은 주제들을 다룬다.

### 빠른 참조 (Cheatsheet 용도)

- **비용이 갑자기 튀었다** → [Prefix cache](./01-the-model.md#prefix-cache), [Cache tokens](./01-the-model.md#cache-tokens), [Output tokens](./01-the-model.md#output-tokens)
- **답변이 점점 멍청해진다** → [Smart zone](./04-failure-modes.md#smart-zone), [Attention degradation](./04-failure-modes.md#attention-degradation), [Compaction](./05-handoffs.md#compaction)
- **존재하지 않는 API를 만든다** → [Hallucination](./04-failure-modes.md#hallucination), [Knowledge cutoff](./04-failure-modes.md#knowledge-cutoff), [Parametric vs Contextual knowledge](./04-failure-modes.md#contextual-knowledge)
- **권한 prompt가 너무 자주 뜬다** → [Permission mode](./03-tools-environment.md#permission-mode), [Agent mode](./03-tools-environment.md#agent-mode)
- **여러 세션으로 나눠 작업하고 싶다** → [Handoff](./05-handoffs.md#handoff), [Spec](./05-handoffs.md#spec), [Ticket](./05-handoffs.md#ticket)
- **에이전트가 매번 같은 컨벤션을 까먹는다** → [Memory system](./06-memory-and-steering.md#memory-system), [AGENTS.md](./06-memory-and-steering.md#agentsmd)

### 표기 규칙 (이 노트 한정)

- 원문 용어는 **영문 그대로** 표기하되, 처음 등장할 때 한글 의역을 함께 적는다. 예: `Harness(하네스)`.
- 원문의 *Avoid* 박스(피해야 할 표현)는 **❌ 피할 표현**으로, *Usage* 박스는 **💬 실전 대화 예시**로 옮겨 적는다.
- 사내 실무와 연결되는 포인트는 **🏢 실무 적용** 박스로 따로 표시한다. Recipe Setup 자동화, SKEWNONO 등 사내 시스템에 어떻게 매핑되는지를 정리하는 자리다.

## 참고 자료 (References)

- [원문 — mattpocock/dictionary-of-ai-coding](https://github.com/mattpocock/dictionary-of-ai-coding)
- [aihero.dev — AI Coding Dictionary 웹페이지](https://www.aihero.dev/ai-coding-dictionary)
- 관련 사내 노트
  - [MCP 기초](../mcp/mcp-basics.md) — Section 3의 MCP 항목과 직접 연결
  - [LangGraph 고급](../rag/langgraph/langgraph-advanced.md) — Subagent / Human-in-the-loop 패턴
  - [Foundation Model 기초](../foundation%20model/README.md) — Parametric knowledge / Knowledge cutoff의 배경
