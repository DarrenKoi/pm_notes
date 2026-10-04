---
tags: [ai-coding, glossary, agent, claude-code, llm]
level: beginner-intermediate
last_updated: 2026-05-05
source: https://github.com/mattpocock/dictionary-of-ai-coding
type: index
reviewed_on: 2026-10-04
review_status: partial
---

# AI Coding Dictionary (한국어 학습 노트)

> [!info] 검토 범위 · 2026-10-04
> 원래 2026-05-05 학습 노트의 용어·대화 사례를 유지했다. 대화는 설명용 가상 사례이며 사내 시스템의 관측 결과가 아니다. 저자의 용어 선택은 보편 표준과 구분한다. 버전별 실제 조건은 [검증된 적용 조건](./verified-conditions.md), 개별 변경·남은 미확인은 [정리 기록](./organization-log.md)을 함께 읽는다.


> Matt Pocock의 [Dictionary of AI Coding](https://github.com/mattpocock/dictionary-of-ai-coding)을 한국어로 풀어 정리한 학습 노트. Claude Code, Cursor, Codex 같은 AI 코딩 도구를 쓰면서 마주치는 용어를 "왜 필요한가 → 무엇인가 → 어떻게 쓰는가" 순서로 정리했다.

## 왜 필요한가? (Why)

- AI 코딩 도구를 사용하다 보면 토큰, 컨텍스트, 하네스, 에이전트 모드 같은 **낯선 어휘**가 한꺼번에 쏟아진다.
- 같은 모델인데 도구마다 동작이 다르거나 같은 프롬프트인데 비용이 폭증하는 현상은, **용어를 정확히 알지 못하면 디버깅 자체가 불가능**하다.
- 아래 업계 평가는 원문 저자의 의견이며 검증된 산업 사실과 구분한다: *"모호함의 상당 부분은 의도적으로 만들어진 것이다. AI 코딩 업계에는, 이 어휘를 어렵게 유지함으로써 이득을 보는 VC 자금이 흐른다."*
- 이 사전을 한 번이라도 훑어 두면 **요금, 어텐션, 컨텍스트 저하 같은 현상에 이름을 붙일 수 있게 된다.** 그리고 이름을 붙일 수 있다는 것이 디버깅의 출발점이다.

## 핵심 개념 (What) — 7개 섹션

원래 노트는 원문의 일부 용어를 7개 섹션으로 정리했다. 현재 원문 main에는 AI/Effort/Primary source/Context pointer 등 추가 항목이 있어 완전 번역본이라고 주장하지 않는다. 이 노트에서도 같은 구조를 따라, 섹션마다 별도 문서로 정리했다.

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
4. **Section 4 (Failure Modes)** 는 오류 증상과 점검 후보를 다룬다. 빈도 비율은 측정하지 않았다.
5. **5~7번 섹션은 실무 운영 단계**에 해당한다. 긴 작업을 어떻게 나눌지(Handoff), 무엇을 기억시킬지(Memory), 어떤 패턴으로 협업할지(AFK, Grilling) 같은 주제들을 다룬다.

### 빠른 참조 (Cheatsheet 용도)

- **비용이 갑자기 튀었다** → [Prefix cache](./01-the-model.md), [Cache tokens](./01-the-model.md), [Output tokens](./01-the-model.md)
- **답변이 점점 멍청해진다** → [Smart zone](./04-failure-modes.md), [Attention degradation](./04-failure-modes.md), [Compaction](./05-handoffs.md)
- **존재하지 않는 API를 만든다** → [Hallucination](./04-failure-modes.md), [Knowledge cutoff](./04-failure-modes.md), [Parametric vs Contextual knowledge · Contextual knowledge (컨텍스트 지식) 절](./04-failure-modes.md)
- **권한 prompt가 너무 자주 뜬다** → [Permission mode](./03-tools-environment.md), [Agent mode](./03-tools-environment.md)
- **여러 세션으로 나눠 작업하고 싶다** → [Handoff](./05-handoffs.md), [Spec](./05-handoffs.md), [Ticket](./05-handoffs.md)
- **에이전트가 매번 같은 컨벤션을 까먹는다** → [Memory system](./06-memory-and-steering.md), [AGENTS.md](./06-memory-and-steering.md#agentsmd)

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


## 참조를 읽는 방법 — 2026-10-05

개념 링크는같은주제의노트를연다. 표시된용어/절이름으로해당본문을찾는다. Obsidian과일반Markdown의제목앵커해석차이가있어새로변환했던GitHub slug참조는노트경로로정돈했다. 대상개념과고유예제는변경하지않았다. renderer별절자동이동을보장하지않는다.
