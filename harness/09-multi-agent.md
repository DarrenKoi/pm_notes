---
tags: [harness-engineering, multi-agent, subagent, orchestration]
level: advanced
last_updated: 2026-09-12
---

# 09. 멀티 에이전트 (Multi-Agent)

> 에이전트를 여러 개로 나누는 이유는 역할극이 아니라 컨텍스트 격리와 병렬화다.
> 둘 다 필요 없다면 단일 에이전트가 거의 항상 낫다.

## 왜 필요한가? (Why)

두 회사가 정반대 결론을 냈고, 둘 다 자기 문제에서는 옳았다. 이 대립을 이해하는
것이 핵심이다.

- **Anthropic (Research 기능)**: 리드 에이전트가 계획을 세우고 서브에이전트 여러
  개가 병렬로 검색한다. 내부 평가에서 단일 에이전트보다 90% 이상 좋았다. 대신
  토큰을 많이 쓴다(일반 채팅 대비 에이전트는 약 4배, 멀티 에이전트는 약 15배).
- **Cognition ("Don't Build Multi-Agents")**: 병렬 에이전트는 서로의 결정을 보지
  못한다. 예를 들어 Flappy Bird 클론을 나눠 맡기면, 한 에이전트는 마리오풍
  배경을 만들고 다른 에이전트는 화풍이 전혀 다른 새를 만든다. 행동에는 암묵적인
  결정이 담기는데, 그 결정이 공유되지 않아 결과가 충돌한다.

차이를 만든 것은 작업의 성격이다. **읽기 위주이고 서로 독립적인 하위
작업**(조사, 검색)은 격리가 장점이 된다. **쓰기 위주이고 상태를 공유하는
작업**(하나의 코드베이스 수정)은 격리가 독이 된다.

## 핵심 개념 (What)

### 1. 판단 기준

| 질문 | 예 → 분할 고려 | 아니오 → 단일 에이전트 |
|---|---|---|
| 하위 작업들이 서로의 결과 없이 진행 가능한가? | 문서 50개를 각각 요약 | 설계 결정이 다음 작업에 영향 |
| 하위 작업이 주로 읽기와 탐색인가? | 로그, 문서, 코드 조사 | 같은 파일들을 수정 |
| 중간 과정이 커서 메인 컨텍스트를 오염시키는가? | 검색 결과 수만 토큰 → 요약 몇 줄만 필요 | 중간 과정 자체가 다음 판단에 필요 |
| 토큰 비용이 늘어나도 결과 가치가 충분한가? | 고가치 리서치 | 대량 반복 처리 |

### 2. 쓸 만한 패턴

| 패턴 | 구조 | 쓰임 |
|---|---|---|
| **서브에이전트를 도구로** | 메인이 `delegate(task)`를 호출하면 새 컨텍스트에서 실행하고 요약만 반환 | 탐색 결과로 메인 컨텍스트가 오염되는 것 방지. 가장 흔하고 안전 |
| **Orchestrator–workers** | 리드가 분해하고, 워커가 병렬 실행하고, 리드가 종합 | 독립적인 조사와 검색 |
| **생성자–검토자 분리** | 작성 에이전트와 검토 에이전트(또는 판정 프롬프트) 분리 | 자기 결과에 관대해지는 문제 완화 ([05](./05-verification-and-evals.md)) |
| **Initializer + worker 세션** | 첫 에이전트가 환경과 계획 파일을 만들고, 이후 세션이 순차 진행 | 컨텍스트를 넘는 긴 작업 ([07](./07-state-and-recovery.md)) |

### 3. 흔한 실패

- **모호한 위임**: "관련 자료 조사해줘"만 받은 서브에이전트는 중복 조사를 하거나
  엉뚱한 범위까지 파고든다. 위임 메시지에 목표, 범위, 제외 대상, 반환 형식, 멈출
  조건을 적는다
- **전언 게임**: 요약이 여러 단계를 거치며 정확한 경로, 수치, 에러 메시지가
  사라진다. 반환 형식에 근거(파일 경로, 원문 인용)를 필수로 넣는다
- **재귀 폭주**: 서브에이전트가 또 서브에이전트를 부른다. 서브에이전트의 도구
  목록에서 `delegate`를 뺀다
- **병렬 쓰기 충돌**: 두 에이전트가 같은 파일을 고친다. 쓰기는 한 에이전트로
  모으거나 작업 공간(git worktree 등)을 분리한다
- **비용 폭증**: 워커 수 × 턴 수 × 컨텍스트로 토큰이 늘어난다. 워커 수와 워커별
  예산에 상한을 둔다

### 4. Cognition이 제시한 원칙

- 컨텍스트를 공유한다. 메시지 한 줄이 아니라 전체 에이전트 트레이스를 넘긴다
- 행동에는 암묵적 결정이 담긴다. 충돌하는 결정은 나쁜 결과를 낳는다

두 원칙을 동시에 지키기 어렵다면 단일 에이전트가 답이다. 이후 Cognition도 실제로
작동하는 멀티 에이전트 구성을 찾았다는 후속 글을 냈다. 요점은 "절대 금지"가
아니라 "기본값은 단일, 분할은 근거가 있을 때"다.

## 어떻게 사용하는가? (How)

### 서브에이전트를 도구로 쓰기

[02](./02-agent-loop.md)의 `run_agent`를 재사용한다.

```python
DELEGATE_TEMPLATE = """{task}

Scope: {scope}
Do NOT: modify files, or investigate outside the scope.
Return (max 300 words):
- Findings, each with evidence (file path + line, or exact quote)
- What you verified and how
- Open questions"""


def make_delegate(client, model):
    def delegate(task: str, scope: str) -> str:
        """새 컨텍스트에서 조사하고 요약만 돌려준다.

        메인 컨텍스트에는 이 반환값만 들어간다.
        """
        # 서브에이전트의 도구 목록에는 delegate를 넣지 않는다 (재귀 폭주 방지)
        prompt = DELEGATE_TEMPLATE.format(task=task, scope=scope)
        answer, _ = run_agent(client, model, prompt)
        return answer
    return delegate
```

적용 순서:

1. 단일 에이전트로 eval 기준선을 잰다
2. 컨텍스트가 가장 많이 부푸는 탐색 단계 하나만 `delegate`로 뺀다
3. 성공률, 성공 1건당 토큰, 지연 시간을 기준선과 비교한다
4. 좋아졌을 때만 범위를 넓힌다

## 최신 패턴: isolated와 fork의 선택

LangChain은 2026-09-08 서브에이전트 컨텍스트 모드를 소개했다.
`isolated`는 새 컨텍스트에서 지정 태스크를 받고, `fork`는 부모의 대화·상태를
이어받는다. 부모는 자식의 최종 결과를 받는다.
[공식 발표](https://www.langchain.com/blog/organizing-context-in-a-multi-agent-harness)

| 상황 | 우선 비교할 모드 | 이유 |
|---|---|---|
| 독립적인 질문 조사 | isolated | 무관한 부모 이력 불필요 |
| 작성 결과의 독립 검증 | isolated | 작성자의 가설에 고정되는 편향 완화 |
| 부모가 분석한 문제의 구현 계속하기 | fork | 재탐색·인수인계 누락 감소 |
| 대화에서 장기 결정 추출 | fork | 대화 자체가 분석 대상 |

fork가 항상 더 싸지는 않다. 캐시 적중과 중복 탐색 감소가 실제로 일어나는지
측정한다. 부모 컨텍스트의 불필요한 정보도 상속될 수 있다.

### 위임 계약 예시

아래는 프레임워크 설정이 아니라 작업 요청에 담을 내용의 예다.

```yaml
objective: 추출 결과의 행 누락 여부 검증
context_mode: isolated
inputs: [sample-page, extracted-table, acceptance-criteria]
allowed_actions: 제공된 입력 읽기와 비교
forbidden_actions: 원본 수정, 외부 전송, 추가 위임
budget: 상위 작업이 배정한 범위
return: [판정, 행별 근거, 미검증 항목]
stop_when: 비교 완료 또는 입력 부족
```

이 계약을 프롬프트에 적는 것만으로 권한이 강제되지는 않는다. 위 `make_delegate`
예제는 [02](./02-agent-loop.md)의 도구를 그대로 사용하므로 임의 파일 읽기와 테스트
실행도 가능하다. 읽기 전용 조사에 쓰려면 도구·파일시스템·네트워크 권한을
실제로 제한해야 한다. [06](./06-guardrails-and-permissions.md)

### 통합 담당자를 정한다

다음은 이 노트의 운영 권고다. 워커가 끝났다는 보고는 전체 업무 완료가 아니다.
부모는 산출물 버전·검증 증거·충돌을 확인하고 통합 결과를 다시 검증한다.
worktree는 파일 편집을 분리하지만 DB·포트·외부 티켓 같은 공유 자원까지 격리하지
않는다. 워커마다 담당 경로와 외부 자원을 정하고, 마지막 쓰기와 통합 검증의
책임자를 한 곳에 둔다.

## 학습 체크리스트

- [ ] 내 업무 작업을 "독립·읽기 위주"와 "상태 공유·쓰기 위주"로 분류했다
- [ ] 단일 에이전트 기준선을 먼저 측정했다
- [ ] 서브에이전트 반환 형식에 근거(경로, 인용)를 필수로 넣었다
- [ ] 멀티 에이전트로 바꾼 뒤 성공 1건당 토큰이 몇 배가 됐는지 쟀다

## 참고 자료 (References)

- [How we built our multi-agent research
  system](https://www.anthropic.com/engineering/multi-agent-research-system) —
  Anthropic
- [Don't Build Multi-Agents](https://cognition.com/blog/dont-build-multi-agents)
  — Walden Yan, Cognition
- [Multi-Agents: What's Actually
  Working](https://cognition.com/blog/multi-agents-working) — Cognition, 후속 글
- [Building effective agents —
  Orchestrator-workers](https://www.anthropic.com/engineering/building-effective-agents)
  — Anthropic

## 관련 문서

- [03. 컨텍스트 엔지니어링 — 격리 전략](./03-context-engineering.md)
- [05. 검증과 평가](./05-verification-and-evals.md)
- [02. 에이전트 루프](./02-agent-loop.md)
