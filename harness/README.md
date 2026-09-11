---
tags: [harness-engineering, ai-agent, llmops, production]
level: intermediate
last_updated: 2026-09-12
---

# Harness Engineering 학습 노트

> Agent = Model + Harness. 하네스 엔지니어링은 모델을 뺀 나머지 전부를 설계해
> 에이전트를 운영할 수 있는 수준으로 끌어올리는 일이다. 루프, 도구, 컨텍스트,
> 가드레일, 검증, 관측이 모두 여기에 들어간다.

## 한 문단 요약

같은 모델도 하네스가 바뀌면 성능이 크게 달라진다. LangChain은 모델을 그대로 둔
채 하네스만 고쳐서 Terminal Bench 2.0 순위를 Top 30 밖에서 Top 5로 올렸다.
이 수치는 2026-02-17 발표 당시의 특정 모델·벤치마크 결과이며 현재 순위를
뜻하지 않는다. [공식 실험 보고](https://www.langchain.com/blog/improving-deep-agents-with-harness-engineering)

실패 원인을 모델 능력만으로 설명하지 말고, 컨텍스트·검증·권한·관측까지 함께
진단해야 한다. 하네스가 모델의 모든 한계를 해결하는 것은 아니다.

## 최신 내용부터 읽기

**2026-09-12 공개 1차 자료 확인 기준**으로 보강했다. 최신 발표와 적용 판단은
[11. 최신 동향과 적용 판단](./11-current-trends.md)에 모았다.

- 9월 8일: 서브에이전트의 isolated / fork 선택과 컨텍스트 전달 계약
- 7월 MCP 사양·8월 로드맵: 세션 제거, 추가 입력, Tasks 확장과 버전 호환
- 장기 실행: 세션 저장·하네스·실행 환경의 수명 분리와 부수효과 복구
- 도구 검색·코드 기반 호출, 보호된 eval을 이용한 하네스 개선 루프

제품별 사례, 사양, 연구용 구현, 이 노트의 권고를 구분했다. Python 예제는
각 개념을 보여주는 학습용 코드이며 통합된 운영 시스템이 아니다. 실제 사내
엔드포인트 연결·샌드박스·장애 복구 검증은 별도로 수행해야 한다.

## 가장 신경 써야 할 것 (우선순위)

| 순위 | 질문 | 문서 |
|---|---|---|
| 1 | 에이전트가 자기 결과를 기계적으로 확인할 수 있는가? | [05](./05-verification-and-evals.md) |
| 2 | 하네스를 바꿨을 때 좋아졌는지 숫자로 말할 수 있는가? | [05](./05-verification-and-evals.md) |
| 3 | 최악의 경우 무엇까지 망가질 수 있는가? (blast radius) | [06](./06-guardrails-and-permissions.md) |
| 4 | 매 턴 모델이 보는 토큰이 필요한 만큼만 들어가 있는가? | [03](./03-context-engineering.md) |
| 5 | 실패한 실행 하나를 처음부터 끝까지 다시 볼 수 있는가? | [08](./08-observability-and-cost.md) |

1~2번이 없으면 나머지 개선은 전부 감(感)에 의존하게 된다. 그래서 eval을 가장
먼저 만든다.

## 학습 순서

| # | 문서 | 핵심 질문 |
|---|---|---|
| 01 | [핵심 개념](./01-core-concept.md) | 하네스란 무엇이고, 왜 모델만큼 중요한가? |
| 02 | [에이전트 루프](./02-agent-loop.md) | 최소 하네스는 몇 줄인가? 루프는 언제 멈추는가? |
| 03 | [컨텍스트 엔지니어링](./03-context-engineering.md) | 유한한 컨텍스트에 무엇을 넣고 무엇을 뺄 것인가? |
| 04 | [도구 설계](./04-tool-design.md) | 모델이 잘 쓰는 도구는 사람용 API와 무엇이 다른가? |
| 05 | [검증과 평가](./05-verification-and-evals.md) | 비결정적 시스템의 품질을 어떻게 측정하는가? |
| 06 | [가드레일과 권한](./06-guardrails-and-permissions.md) | 프롬프트 인젝션을 전제로 피해를 어떻게 제한하는가? |
| 07 | [상태와 복구](./07-state-and-recovery.md) | 중간에 죽은 작업을 어떻게 이어서 하는가? |
| 08 | [관측과 비용](./08-observability-and-cost.md) | 무엇을 기록하고 어떤 지표를 보는가? |
| 09 | [멀티 에이전트](./09-multi-agent.md) | 언제 에이전트를 나누고, 언제 나누면 안 되는가? |
| 10 | [프로덕션 체크리스트](./10-production-checklist.md) | 배포 전에 무엇을 확인하는가? |
| 11 | [최신 동향과 적용 판단](./11-current-trends.md) | 최근 변화 중 무엇을 도입하고 검증할 것인가? |

## 5주 학습 계획 (제안)

| 주차 | 범위 | 손으로 해볼 것 |
|---|---|---|
| 1 | 01, 02 | `mini_harness.py`를 사내 OpenAI 호환 엔드포인트에 연결해 돌려본다 |
| 2 | 03, 04 | 긴 작업으로 컨텍스트 한도를 일부러 넘겨 보고 compaction을 붙인다 |
| 3 | 05 | 실제 실패 사례로 태스크 20개짜리 eval을 만든다. 이후 모든 변경은 이 eval로 판단한다 |
| 4 | 06, 07, 08 | 권한 정책, 체크포인트, JSONL 트레이스를 붙인다 |
| 5 | 09, 10 | 체크리스트로 내 시스템을 점검하고, 멀티 에이전트가 정말 필요한지 따져본다 |

## 더 읽을거리

- [Harness engineering: leveraging Codex in an agent-first
  world](https://openai.com/index/harness-engineering/) — OpenAI, 2026-02
- [Harness engineering for coding agent
  users](https://martinfowler.com/articles/harness-engineering.html) — Birgitta
  Böckeler, 2026-04
- [The Anatomy of an Agent
  Harness](https://www.langchain.com/blog/the-anatomy-of-an-agent-harness) —
  LangChain, 2026-03
- [awesome-harness-engineering](https://github.com/ai-boost/awesome-harness-engineering)
  — 도구·패턴 모음
