---
tags: [ai-coding, memory, agents-md, skill, subagent, progressive-disclosure]
level: intermediate
last_updated: 2026-05-05
source: https://github.com/mattpocock/dictionary-of-ai-coding
type: learning
reviewed_on: 2026-10-04
review_status: partial
---

# Section 6 — Memory and Steering (메모리와 조종)

> [!info] 검토 범위 · 2026-10-04
> 원래 2026-05-05 학습 노트의 용어·대화 사례를 유지했다. 대화는 설명용 가상 사례이며 사내 시스템의 관측 결과가 아니다. 저자의 용어 선택은 보편 표준과 구분한다. 버전별 실제 조건은 [검증된 적용 조건](./verified-conditions.md), 개별 변경·남은 미확인은 [정리 기록](./organization-log.md)을 함께 읽는다.


> [Agent](./02-sessions-context-windows-turns.md)가 **여러 세션에 걸쳐 무엇을 기억할지**, 그리고 **그 기억을 어떻게 효율적으로 로드할지**.

## 왜 이 섹션이 중요한가? (Why)

- "어제 가르친 내용을 오늘 또 가르쳐야 한다"는 식의 좌절은 보통 [Memory system](./06-memory-and-steering.md)이 없어서 생긴다.
- 그렇다고 모든 정보를 [AGENTS.md](#agentsmd)에 몰아넣으면, 이번에는 **매 [Turn](./02-sessions-context-windows-turns.md)마다 토큰 비용을 다시 내는 문제**가 생긴다.
- 그래서 해법은 **[Progressive disclosure](./06-memory-and-steering.md)**다. 필요할 때만 로드하는 식으로 풀어야 한다.

## 용어 (What & How)

### Memory system (메모리 시스템)

[Agent](./02-sessions-context-windows-turns.md)가 [Sessions · Session (세션) 절](./02-sessions-context-windows-turns.md) 사이에서도 [Stateful](./02-sessions-context-windows-turns.md)로 동작하도록 만들기 위한 시스템. 세션 동안 알게 된 정보를 [Environment](./03-tools-environment.md)에 영속화해 두었다가, 다음 세션이 시작될 때 [Context window](./02-sessions-context-windows-turns.md)로 다시 불러온다. 사용자가 세션을 [Clear](./05-handoffs.md)한 뒤에도 연속성이 유지되도록 만드는 게 목적이다.

**💬 실전 대화 예시**
> "MySQL이 아니라 Postgres라고 매번 다시 알려줘야 해."
> "Memory system을 붙이자. 첫 [Turn](./02-sessions-context-windows-turns.md)에 학습한 내용을 [Filesystem](./03-tools-environment.md)에 적어 두고, 세션 시작 때 다시 로드하면 돼. [Model](./01-the-model.md) 자체는 [stateless](./02-sessions-context-windows-turns.md)니까, 결국 메모리 레이어가 연속성을 *흉내내는* 셈이지."

🏢 **실무 적용**: Claude Code의 `CLAUDE.md`(이 레포 최상위에 있는 바로 그 파일!)도 일종의 memory system이라고 볼 수 있다. 사용자 선호와 컨벤션을 매 세션 시작 시점에 자동으로 로드해 준다.

---

### AGENTS.md

[Environment](./03-tools-environment.md)에 놓이는 파일로, [Harness](./01-the-model.md)가 [Session](./02-sessions-context-windows-turns.md) 시작 시점에 [Context window](./02-sessions-context-windows-turns.md)로 지원·버전·파일 탐색 범위·설정이 맞으면 로드한다. 이름만 같다고 모든 하네스가 자동으로 읽지는 않는다. 프로젝트가 [Agent](./02-sessions-context-windows-turns.md)에게 매번 전해 주는 **상시 브리핑** 역할을 한다. **여러 하네스에 공통으로 적용되는 컨벤션**의 표준 위치이기도 하다.

**❌ 피할 표현**: AGENTS.md를 원래 [Progressive disclosure](./06-memory-and-steering.md) 대상이어야 할 콘텐츠를 담아두는 용도로 쓰지 말 것. 여기에 들어간 로드된 정보는 입력 공간과 처리 비용에 영향을 준다. 실제 과금은 요청 재구성·cache read/write·구독 정책에 따라 다르다.

**💬 실전 대화 예시**
> "왜 매 세션이 4k 토큰을 이미 태운 채로 시작해?"
> "AGENTS.md를 한번 봐. 누군가 스타일 가이드 전체를 [Skill](./06-memory-and-steering.md) 뒤로 빼지 않고 그대로 거기에 붙여 놨어."

🏢 **실무 적용**: 사내 프로젝트의 `CLAUDE.md`도 가능한 한 **얇게 유지**해야 한다. 자주 사용하지 않는 컨벤션은 skill로 분리해 두는 편이 좋다.

---

### Progressive disclosure (점진적 노출)

[Agent](./02-sessions-context-windows-turns.md)가 **지금 당장 필요한 [Context](./02-sessions-context-windows-turns.md)만** 로드하고, 나머지는 *포인터*만 남겨 두는 패턴. UI 디자인 분야에서 빌려온 개념이다.

**💬 실전 대화 예시**
> "스타일 가이드 전체를 [AGENTS.md](#agentsmd)에 그냥 부어 버릴까?"
> "그건 안 돼. progressive disclosure로 가야 해. 스타일 가이드는 [Skill](./06-memory-and-steering.md)로 참조해 두고, 에이전트가 실제로 컴포넌트를 짤 때만 로드되도록 하는 거야. AGENTS.md에 박아 두면, 사용하지 않는 turn에서도 매번 [Token](./01-the-model.md) 비용이 계속 나가."

---

### Skill (스킬)

**하나의 작업을 잘 수행할 수 있도록 묶어 둔 능력 단위**. 특정 작업을 잘 처리하기 위한 지시문과 리소스가 함께 묶여 있다. 평소에는 [Environment](./03-tools-environment.md)에 보관돼 있다가, 관련 작업이 발생할 때만 [Context window](./02-sessions-context-windows-turns.md)로 로드된다. 한마디로, **[Harness](./01-the-model.md)에서 [Progressive disclosure](./06-memory-and-steering.md)를 적용하는 단위**라고 보면 된다.

**❌ 피할 표현**: "[Tool](./03-tools-environment.md)"과 혼동하지 말 것. tool은 [Agent](./02-sessions-context-windows-turns.md)가 *호출하는* 함수이고, skill은 *읽어들이는* 지시문이다.

**💬 실전 대화 예시**
> "Deploy runbook은 어디에 두는 게 좋을까?"
> "Skill로 만들어 두자. 에이전트가 deploy 관련 작업을 할 때만 로드되게 하는 거야. [AGENTS.md](#agentsmd)에 박아 두면 일주일에 한 번 쓰는 내용에 대해 매 turn token 비용을 내야 하니까."

---

### Subagent (서브에이전트)

다른 [Agent](./02-sessions-context-windows-turns.md)가 [Tool call](./03-tools-environment.md)을 통해 새로 띄운(spawn) 에이전트. **자기만의 [Session](./02-sessions-context-windows-turns.md)과 [Context window](./02-sessions-context-windows-turns.md)** 안에서 동작하며, 작업이 끝나면 **하나의 [Tool result](./03-tools-environment.md)** 를 부모 에이전트에게 보고한다. [Handoff](./05-handoffs.md)와는 분명히 다른데, subagent는 위임 결과를 부모로 전달할 수 있고 비동기 실행도 가능하다. handoff와의 경계는 이 사전의 작업 패턴 분류다. 중첩 spawn 가능 여부와 깊이는 하네스·버전·설정별이다. Claude Code 공식 문서는 v2.1.219 이후 기본 깊이 3과 변경 설정을 설명한다. 1단 제한을 모든 에이전트의 정의로 쓰지 않는다. 계층 구조를 깊게 쌓기 위해서가 아니라, **[Context](./02-sessions-context-windows-turns.md)를 격리하기 위해 존재하는 메커니즘**이라는 점이 핵심이다.

**💬 실전 대화 예시**
> "Grep 결과가 내 컨텍스트를 폭발시키고 있어."
> "Subagent를 띄워서 검색을 맡기자. 노이즈는 그쪽 context window에서 다 태우고, 너에게는 실제로 필요한 파일 경로 두 개만 보고하게 해."

🏢 **실무 적용**: LangGraph의 subgraph는 그래프 구성/실행 단위이며 공유 state 설계에 따라 context 격리 여부가 다르다. 제품 API 이름을 곧 서브에이전트와 동치로 취급하지 않는다. 자세한 내용은 [LangGraph 고급](../rag/langgraph/langgraph-advanced.md) 참고.

## 이 섹션 요약 (Cheatsheet)

```
[Token 비용 관점에서 어디에 무엇을 둘 것인가]

매 turn 비용 ───────────────────────────── 로드 전 본문 비용 감소 (metadata·로드 후 비용은 존재)
   │                                                    │
[AGENTS.md]                                          [Skill]
   • 얇게 유지                                         • 큰 가이드/runbook
   • 항상 알아야 하는 컨벤션                           • 가끔 필요한 도메인 지식
   • 프로젝트 정체성                                   • 특정 task 모드
```

| 단어 | 한 줄 정의 |
|---|---|
| Memory system | 세션 간 stateful 흉내. 영속화 + 재로드 |
| AGENTS.md | 지원·탐색·설정 조건에 따라 로드되는 지침 |
| Progressive disclosure | "필요할 때만 로드" 원칙 |
| Skill | progressive disclosure의 단위 — 읽는 지시문 |
| Subagent | 별도 context로 위임하는 에이전트; 중첩 깊이는 구현별 |

## 관련 문서

- 이전: [05 - Handoffs](./05-handoffs.md)
- 다음: [07 - Patterns of Work](./07-patterns-of-work.md)
- 인덱스: [README](./README.md)
- 사내 연결: [LangGraph 고급 (Subgraph)](../rag/langgraph/langgraph-advanced.md)

## 참고 자료 (References)

- 원문: [mattpocock/dictionary-of-ai-coding — Section 6](https://github.com/mattpocock/dictionary-of-ai-coding#section-6--memory-and-steering)
- AGENTS.md 컨벤션: https://agents.md/ (또는 Claude Code의 CLAUDE.md)
