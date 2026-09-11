---
tags: [harness-engineering, evals, verification, llm-as-judge]
level: intermediate
last_updated: 2026-09-11
---

# 05. 검증과 평가 (Verification & Evals)

> 하네스 엔지니어링에서 가장 중요한 한 가지를 고르라면 이것이다. 루프 안에서는
> 에이전트가 자기 결과를 확인하게 하고, 루프 밖에서는 하네스 변경을 숫자로
> 판단한다.

## 왜 필요한가? (Why)

- 에이전트는 비결정적이다. 한 번 성공했다고 다음에도 성공한다는 보장이 없다.
- 프롬프트 한 줄, 도구 설명 하나, compaction 설정 하나가 다른 태스크를 몰래
  망가뜨린다. eval 없이 하는 하네스 개선은 한쪽을 고치면 다른 쪽이 터지는 두더지
  잡기가 된다.
- 모델을 교체할 때(예: 사내 모델 버전 업그레이드) "새 모델이 우리 업무에서 더
  나은가"에 답할 방법이 eval뿐이다.
- OpenAI 팀의 에이전트가 사람 QA 없이 몇 시간씩 작업할 수 있었던 것도 테스트,
  린터, UI 확인 같은 검증 루프를 에이전트가 직접 돌릴 수 있었기 때문이다.

## 핵심 개념 (What)

### 1. 두 층의 검증

| 층 | 언제 | 목적 | 예 |
|---|---|---|---|
| **In-loop 검증 (sensor)** | 실행 중, 매 작업마다 | 에이전트가 스스로 고치게 한다 | 테스트 실행, 린터, 스키마 검증, 결과 파일 존재 확인 |
| **Offline eval** | 하네스를 바꿀 때마다 | 변경이 개선인지 판정한다 | 태스크 셋 × 반복 실행 → 성공률 비교 |

### 2. In-loop 검증 설계

- **Computational 센서를 먼저 쓴다.** 테스트, 타입 체크, JSON Schema, 행 수
  비교처럼 결정적이고 빠르고 싸다.
- **Inferential 센서는 그다음이다.** 코드로 판정할 수 없는 품질(요약이 원문에
  충실한가)만 LLM 판정기에 맡긴다.
- **작성자와 판정자를 분리한다.** 같은 컨텍스트의 모델이 자기 결과를 검토하면
  관대해진다. 별도 프롬프트나 별도 호출로 판정한다 (evaluator-optimizer 패턴).
- **"완료" 선언을 믿지 않는다.** 루프가 끝나면 하네스가 검증을 실행하고,
  실패하면 결과를 붙여 루프를 다시 돌린다.

### 3. Offline eval 용어 (Anthropic)

| 용어 | 뜻 |
|---|---|
| Task | 입력과 성공 기준을 가진 테스트 케이스 1개 |
| Trial | 그 태스크를 한 번 실행한 것. 비결정적이라 여러 번 돌린다 |
| Transcript (trace) | 한 trial의 전체 모델 입출력과 도구 호출 |
| Outcome | 실행 후 환경의 최종 상태 |
| Grader | 성공 여부 판정기 (코드 / 모델 / 사람) |

Anthropic의 가장 강한 조언은 **가능하면 최종 상태(outcome)를 확인하라**는
것이다. "환불했습니다"라는 응답 문구가 아니라 환불 레코드가 실제로 DB에
생겼는지를 본다.

### 4. pass@k와 pass^k

| 지표 | 의미 | 쓰임 |
|---|---|---|
| pass@k | k번 중 **한 번이라도** 성공할 확률 | 여러 번 시도 후 사람이 고를 수 있는 작업 |
| pass^k | k번 **모두** 성공할 확률 | 매번 맞아야 하는 자동화 (일관성) |

pass@1이 80%라도 pass^5는 훨씬 낮을 수 있다. 무인 자동화라면 pass^k를 봐야 한다.

### 5. Eval 셋 만드는 법

- **실제 실패에서 시작한다.** 운영 중 실패한 사례를 태스크로 옮긴다. 가상의
  태스크 100개보다 실제 실패 20개가 낫다.
- **Capability eval과 regression eval을 나눈다.** 전자는 아직 못 하는 것(점수가
  낮아야 정상), 후자는 이미 되는 것(항상 거의 100%여야 함)이다.
- **환경을 격리한다.** trial마다 깨끗한 작업 폴더나 DB 스냅샷에서 시작한다. 이전
  trial의 흔적이 결과를 오염시키면 안 된다.
- **포화되면 교체한다.** 모든 태스크가 통과하면 그 eval은 더 이상 정보를 주지
  않는다.

### 6. 운영 중 평가

오프라인 eval만으로는 부족하다. 자동 eval, 운영 모니터링, A/B 테스트, 사용자
피드백, 트랜스크립트 리뷰가 각각 다른 실패를 잡는다. 최소한 **주 1회 실패
트랜스크립트 10건 읽기**를 루틴으로 둔다. 새 태스크 대부분이 여기서 나온다.

## 어떻게 사용하는가? (How)

### Step 1. 최소 eval 러너

```python
"""run_eval.py — 태스크 × 반복 실행으로 하네스를 채점한다.

실행: python run_eval.py
"""
import math
from dataclasses import dataclass
from typing import Callable


@dataclass
class Task:
    id: str
    prompt: str
    # 가능하면 응답 문구가 아니라 환경의 최종 상태를 확인한다
    check: Callable[[str], bool]


def pass_at_k(n: int, c: int, k: int) -> float:
    """n번 중 c번 성공했을 때, k번 중 한 번 이상 성공할 확률 (비편향 추정)."""
    return 1 - math.comb(n - c, k) / math.comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """k번 모두 성공할 확률."""
    return math.comb(c, k) / math.comb(n, k)


def evaluate(tasks: list[Task], run: Callable[[str], str],
             n: int = 5, k: int = 3) -> list[dict]:
    report = []
    for t in tasks:
        # trial마다 환경 초기화는 run 안에서 한다
        c = sum(bool(t.check(run(t.prompt))) for _ in range(n))
        report.append({"task": t.id, "success": f"{c}/{n}",
                       f"pass@{k}": round(pass_at_k(n, c, k), 2),
                       f"pass^{k}": round(pass_hat_k(n, c, k), 2)})
    return report


if __name__ == "__main__":
    assert pass_at_k(5, 0, 3) == 0 and pass_at_k(5, 5, 3) == 1
    assert pass_hat_k(5, 2, 3) == 0 and pass_hat_k(5, 5, 3) == 1

    import random
    tasks = [Task("sum", "2+2?", lambda out: out.strip() == "4")]
    # 실제로는 run_agent를 호출한다
    flaky_agent = lambda prompt: random.choice(["4", "4", "4", "five"])
    for row in evaluate(tasks, flaky_agent):
        print(row)
```

하네스를 바꾸기 전과 후에 같은 태스크 셋으로 돌리고, 결과를 표로 남긴다. 한 번에
한 가지만 바꿔야 원인을 알 수 있다.

### Step 2. LLM-as-judge 판정기

```python
JUDGE_PROMPT = """You grade whether an extracted table matches the source page.
Score each criterion PASS or FAIL, then give a final verdict.
1. Every row in the source appears in the output (no missing rows)
2. Numbers are copied exactly (no rounding, no unit changes)
3. No values that are not in the source
Return JSON:
{"criteria": [..], "verdict": "PASS"|"FAIL", "reason": "<one sentence>"}"""
```

판정기를 쓰기 전에 사람이 30~50건에 직접 라벨을 붙이고 판정기와 일치율을 잰다.
판정기도 하네스의 일부이므로 검증 대상이다. 점수(1~10)보다 기준별 PASS/FAIL이 더
안정적이다.

### Step 3. 루프 안에 검증 단계 넣기

```python
def run_with_verification(client, model, task, verify, max_rounds=3):
    """verify(answer) -> (ok, feedback). 실패하면 피드백을 붙여 다시 돌린다."""
    prompt = task
    for _ in range(max_rounds):
        answer, _ = run_agent(client, model, prompt)
        ok, feedback = verify(answer)
        if ok:
            return answer
        prompt = (f"{task}\n\nYour previous attempt failed verification:\n"
                  f"{feedback}\nFix it.")
    raise RuntimeError(
        f"verification failed after {max_rounds} rounds: {feedback}")
```

## 학습 체크리스트

- [ ] 내 업무에서 최종 상태를 코드로 확인할 수 있는 태스크 20개를 모았다
- [ ] 태스크별로 5회씩 돌려 pass@1과 pass^3을 기록했다
- [ ] 프롬프트를 한 가지 바꾸고 eval로 전후를 비교했다
- [ ] LLM 판정기와 사람 라벨의 일치율을 쟀다
- [ ] 실패 트랜스크립트를 읽고 새 태스크를 하나 이상 추가했다

## 참고 자료 (References)

- [Demystifying evals for AI
  agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
  — Anthropic, 2026-01
- [Building effective agents —
  Evaluator-optimizer](https://www.anthropic.com/engineering/building-effective-agents)
  — Anthropic
- [Harness engineering for coding agent
  users](https://martinfowler.com/articles/harness-engineering.html) — Böckeler,
  computational vs inferential 센서
- [Harness engineering](https://openai.com/index/harness-engineering/) — OpenAI,
  에이전트가 직접 돌리는 검증 루프

## 관련 문서

- [01. 핵심 개념 — Guides와 Sensors](./01-core-concept.md)
- [08. 관측과 비용](./08-observability-and-cost.md)
- [10. 프로덕션 체크리스트](./10-production-checklist.md)
