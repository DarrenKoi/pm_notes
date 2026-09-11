---
tags: [harness-engineering, ai-agent, concept]
level: intermediate
last_updated: 2026-09-11
---

# 01. 하네스 엔지니어링 핵심 개념

> 모델은 지능을 제공하고, 하네스는 그 지능을 쓸모 있는 일로 바꾼다.

## 왜 필요한가? (Why)

LLM 자체는 텍스트를 받아 텍스트를 내놓는 함수다. 파일을 읽고, 테스트를 돌리고,
결과를 확인하고, 실수를 고치는 일은 모두 모델 바깥 코드가 한다. 이 바깥 코드가
하네스(Harness)다.

프로덕션에서 이 차이가 드러나는 사례가 있다.

- LangChain은 모델과 가중치를 그대로 두고 하네스만 바꿔 Terminal Bench 2.0에서
  Top 30 밖에서 Top 5로 올라갔다.
- OpenAI의 한 팀은 5개월 동안 사람이 직접 짠 코드 없이 약 100만 줄짜리 제품을
  에이전트(Codex)로 만들었다. 이들이 꼽은 가장 큰 병목은 모델 능력이 아니라
  에이전트가 환경을 읽어낼 수 있는가(legibility)였다.

결론은 같다. 모델 교체는 선택지가 적고 비싸다. 반면 하네스는 우리가 직접 통제할
수 있는 영역이고, 품질 차이는 대부분 여기서 난다.

> 사내 환경 메모: 사내 오픈소스 모델(OpenAI 호환 API)은 프런티어 API 모델보다
> tool calling 형식 오류나 긴 컨텍스트 성능 저하가 더 잦을 수 있다. 그만큼
> 하네스가 떠안아야 할 몫이 커진다. JSON 인자 검증, 좁고 명확한 도구, 짧은
> 컨텍스트 유지가 더 중요해진다.

## 핵심 개념 (What)

### 1. 정의

> "Agent = Model + Harness. A harness is every piece of code, configuration, and
> execution logic that isn't the model itself." — LangChain

이름은 말의 마구(馬具)에서 왔다. 고삐, 안장, 재갈은 힘은 세지만 예측하기 어려운
동물을 원하는 방향으로 이끈다.

### 2. 용어의 등장

개념 자체는 오래됐지만, 이 이름은 2026년 초에 굳어졌다.

| 시점 | 출처 | 기여 |
|---|---|---|
| 2026-02-05 | Mitchell Hashimoto, *My AI Adoption Journey* | "에이전트가 실수하면, 그 실수를 다시는 못 하게 만드는 장치를 설계한다" |
| 2026-02-11 | OpenAI, *Harness engineering* | 에이전트 전용 코드베이스를 운영하며 얻은 구체적 실천법 |
| 2026-03-10 | LangChain, *The Anatomy of an Agent Harness* | 하네스 구성요소 분류 |
| 2026-04-02 | Birgitta Böckeler (martinfowler.com) | Guides / Sensors 멘탈 모델 |

### 3. Prompt → Context → Harness

세 개념은 서로를 대체하지 않는다. 뒤의 것이 앞의 것을 포함하며 범위가 넓어진다.

| 층위 | 묻는 질문 |
|---|---|
| Prompt engineering | 모델에게 무엇을 어떻게 말할까? |
| Context engineering | 매 스텝마다 모델이 무엇을 보게 할까? |
| Harness engineering | 시스템이 무엇을 막고, 측정하고, 교정할까? |

### 4. 하네스 구성요소 지도

LangChain의 분류를 이 폴더 문서에 매핑했다.

| 구성요소 | 역할 | 문서 |
|---|---|---|
| 에이전트 루프, 계획 도구 | 모델 호출 → 도구 실행 → 결과 반영 반복 | [02](./02-agent-loop.md) |
| 시스템 프롬프트, 메모리, 컨텍스트 관리 | 매 턴 들어갈 토큰 선별·압축 | [03](./03-context-engineering.md) |
| 도구, 스킬, MCP, 코드 실행 | 모델이 세상에 행동하는 수단 | [04](./04-tool-design.md) |
| 검증 루프 | 테스트·린터·판정기로 자기 교정 | [05](./05-verification-and-evals.md) |
| 샌드박스, 권한 | 피해 범위 제한 | [06](./06-guardrails-and-permissions.md) |
| 파일시스템, 상태 저장 | 세션 간 지속, 중단 후 재개 | [07](./07-state-and-recovery.md) |
| (운영 계층) 트레이싱, 비용 | 무슨 일이 있었는지 재생 | [08](./08-observability-and-cost.md) |

### 5. Guides와 Sensors (Böckeler)

하네스 요소는 두 축으로 나눌 수 있다.

| | Computational (결정적, 빠름) | Inferential (LLM 기반, 느리고 비쌈) |
|---|---|---|
| **Guide** (feedforward: 행동 전에 방향 제시) | 코드 템플릿, 타입 정의, 스키마 | AGENTS.md, 시스템 프롬프트, 스킬 문서 |
| **Sensor** (feedback: 행동 후에 관측·교정) | 테스트, 린터, 타입 체커, 구조 테스트 | LLM 리뷰어, LLM-as-judge |

가이드만 있으면 규칙이 실제로 지켜졌는지 알 수 없다. 센서만 있으면 같은 실수를
계속 반복한다. 둘을 짝지어야 한다. 비용이 싸고 결과가 결정적인 computational
쪽을 먼저 쓰고, 그걸로 잡을 수 없는 부분만 inferential로 넘긴다.

### 6. 두 종류의 하네스

같은 용어가 두 맥락에서 쓰이니 구분해 둔다.

- **빌더 쪽 하네스**: 에이전트 제품을 만드는 사람이 설계한다. 루프, 도구,
  컨텍스트 관리, 샌드박스가 해당한다. 이 폴더의 주 대상이다.
- **사용자 쪽 하네스**: Claude Code나 Codex 같은 코딩 에이전트를 쓰는 사람이
  저장소에 구축한다. AGENTS.md, 린터, 구조 테스트, CI가 여기에 속한다. OpenAI
  글과 Böckeler 글이 주로 이쪽을 다룬다.

원리는 같다. 에이전트가 볼 수 있게 하고, 기계로 강제하고, 결과를 측정한다.

## 핵심 원칙 5가지

1. **실수는 하네스 개선 요청서다.** 같은 실수가 두 번 나오면 프롬프트 한 줄,
   도구, 검사 중 하나를 추가해 구조적으로 막는다. (Hashimoto)
2. **에이전트가 볼 수 없는 것은 존재하지 않는다.** Slack 스레드나 사람 머릿속에
   있는 지식은 에이전트에게 없는 것과 같다. 저장소나 컨텍스트로 가져와야 한다.
   (OpenAI)
3. **문서로 부탁하지 말고 기계로 강제한다.** "레이어 규칙을 지켜라"라고 쓰는
   대신 위반 시 실패하는 구조 테스트를 만든다. 에러 메시지에는 고치는 방법까지
   적는다. (OpenAI)
4. **가장 단순한 것부터 시작한다.** 고정된 워크플로우로 충분하면 자율 에이전트를
   쓰지 않는다. 복잡도는 측정으로 필요가 확인됐을 때만 더한다. (Anthropic,
   *Building effective agents*)
5. **하네스는 모델 약점에 대한 가정의 묶음이다.** 모델이 바뀌면 eval로 다시
   검증하고, 더 이상 필요 없어진 비계(scaffolding)는 걷어낸다. 하네스는 계속
   두꺼워지기만 해서는 안 된다.

## 어떻게 사용하는가? (How)

지금 운영 중이거나 만들 에이전트에 아래 질문을 던져본다. 답이 "아니오"인 항목이
학습 우선순위다.

- [ ] 에이전트가 끝났다고 말할 때, 그게 사실인지 코드로 확인하는 단계가 있는가?
      → [05](./05-verification-and-evals.md)
- [ ] 프롬프트를 한 줄 고쳤을 때 좋아졌는지 나빠졌는지 숫자로 비교할 수 있는가?
      → [05](./05-verification-and-evals.md)
- [ ] 에이전트가 악성 문서를 읽고 지시를 따르면, 최악의 피해는 무엇인가? →
      [06](./06-guardrails-and-permissions.md)
- [ ] 30턴짜리 작업에서 20턴째 컨텍스트에 무엇이 들어 있는지 설명할 수 있는가? →
      [03](./03-context-engineering.md)
- [ ] 어제 실패한 실행 하나를 골라 모든 모델 입출력과 도구 호출을 다시 볼 수
      있는가? → [08](./08-observability-and-cost.md)
- [ ] 프로세스가 15턴째 죽으면 처음부터 다시 하는가? →
      [07](./07-state-and-recovery.md)
- [ ] 같은 실수가 반복될 때 그걸 막는 장치를 추가하는 절차가 있는가? → 위 원칙 1

## 참고 자료 (References)

- [Harness engineering: leveraging Codex in an agent-first
  world](https://openai.com/index/harness-engineering/) — Ryan Lopopolo, OpenAI,
  2026-02-11
- [My AI Adoption Journey](https://mitchellh.com/writing/my-ai-adoption-journey)
  — Mitchell Hashimoto, 2026-02-05
- [The Anatomy of an Agent
  Harness](https://www.langchain.com/blog/the-anatomy-of-an-agent-harness) —
  Vivek Trivedy, LangChain, 2026-03-10
- [Harness engineering for coding agent
  users](https://martinfowler.com/articles/harness-engineering.html) — Birgitta
  Böckeler, 2026-04-02
- [Building effective
  agents](https://www.anthropic.com/engineering/building-effective-agents) —
  Anthropic

## 관련 문서

- [README — 학습 순서](./README.md)
- [02. 에이전트 루프](./02-agent-loop.md)
- [10. 프로덕션 체크리스트](./10-production-checklist.md)
