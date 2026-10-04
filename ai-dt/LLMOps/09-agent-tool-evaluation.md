---
tags: [evaluation, agent, tool-calling, trajectory, task-success]
level: advanced
last_updated: 2026-07-06
reviewed_on: 2026-10-04
review_status: partial
document_type: learning
---

> [!info] 검토 범위 — 2026-10-04
> [공통 적용 조건](./verified-conditions.md)과 [정리 기록](./organization-log.md)에 판본·출처·로컬 검증을 남겼다. 실제 judge 품질·사내 접속·운영 승인·Claude 협의·Obsidian 읽기 화면은 미확인이다.


# 09. Agent · Tool 평가

> Agent는 여러 스텝(도구 선택 → 호출 → 관찰 → 다음 결정)을 거치므로 최종 답만이 아니라 **경로(trajectory)**와 **도구 호출 정확도**를 평가해야 실패 원인을 짚는다.

## 왜 필요한가? (Why)

- Agent가 틀린 답을 냈을 때, 원인은 여러 층이다: **도구를 잘못 골랐나, 인자를 틀렸나, 결과를 잘못 해석했나, 너무 많은/적은 스텝을 밟았나.** 최종 정확도만 보면 이 층을 못 가른다.
- 사내 Agent(예: 검색 + 사내 API 호출)는 **잘못된 도구 호출이 부작용**을 낳을 수 있어(엉뚱한 조회·쓰기), tool-call 정확도를 독립적으로 재야 한다.
- Agent는 비결정적이라 **같은 입력에도 경로가 흔들린다.** 그래서 성공률을 **분포**로 본다(여러 번 실행).

## 핵심 개념 (What)

### 1) 평가의 3층
| 층 | 무엇을 보나 | 지표 |
|----|------------|------|
| **결과(outcome)** | 최종적으로 과제를 해결했나 | Task Success Rate |
| **경로(trajectory)** | 올바른 스텝들을 밟았나 | tool-call 정확도, 스텝 수, 정답 경로 일치 |
| **단계(step)** | 각 도구 호출이 옳았나 | tool 선택 정확도, 인자 정확도 |

### 2) Task Success — 무엇을 "성공"으로 정의할지
- **최종 상태 기반**: 원하는 결과 상태에 도달했나(예: 올바른 값 반환, 올바른 레코드 생성). 가능하면 **결정적 검증기(assertion)**로.
- **LLM-judge 기반**: 자유서술 과제는 rubric judge로. → [07](./07-llm-as-a-judge.md)

### 3) Trajectory 평가 방식
- **정확 경로 일치(exact match)**: 기대 도구 시퀀스와 비교(엄격, 대안 경로 불인정).
- **부분 점수**: 올바른 도구를 호출했나(순서 무관), 불필요한 호출 감점.
- **효율성**: 스텝 수·토큰·지연 — 스텝 증가의 비용을 과제별로 해석한다. 20스텝 자체가 실패를 뜻하지 않는다.

### 4) 흔한 실패 모드 (라벨링해두면 개선 방향이 보임)
잘못된 도구 선택 / 인자 오류(스키마 위반) / 무한 루프·과도한 스텝 / 도구 결과 오해석 / 조기 종료(포기).

## 어떻게 사용하는가? (How)

### 도구 호출 로그를 표준화 (평가의 입력)
평가하려면 Agent 실행이 **구조화된 trajectory**를 남겨야 한다. → [03](./03-tracing-observability.md)

```python
# 한 번의 Agent 실행이 남기는 trajectory
trajectory = {
    "id": "t001",
    "steps": [
        {"tool": "search_docs", "args": {"q": "오버레이 오차"}, "obs": "..."},
        {"tool": "get_spec",    "args": {"id": "SPEC-12"}, "obs": "..."},
    ],
    "final_answer": "...",
}
expected = {
    "must_call": ["search_docs", "get_spec"],       # 반드시 불러야 할 도구
    "forbidden": ["write_record"],                  # 부작용 도구 호출 금지
    "gold_answer": "...",
}
```

### Tool-call 정확도 (선택·인자)

```python
from jsonschema import Draft202012Validator, ValidationError

def tool_selection_score(traj, expected) -> float:
    called = {step["tool"] for step in traj["steps"]}
    must = set(expected["must_call"])
    if called & set(expected.get("forbidden", [])):
        return 0.0  # 사후 발견 점수이며 이미 일어난 부작용을 예방하지 않음
    return len(called & must) / len(must) if must else 1.0

def validate_args(tool, args, schemas) -> bool:
    if tool not in schemas:
        raise ValueError("도구 schema 미확인")
    schema = schemas[tool]
    Draft202012Validator.check_schema(schema)
    try:
        Draft202012Validator(schema).validate(args)
        return True
    except ValidationError:
        return False

def arg_validity_score(traj, schemas) -> float:
    steps = traj["steps"]
    if not steps:
        raise ValueError("도구 호출 없음: 인자 검증 해당 없음")
    return sum(validate_args(s["tool"], s["args"], schemas) for s in steps) / len(steps)
```

### Trajectory 일치 & 효율성

```python
def trajectory_match(traj, expected, order_sensitive=False) -> float:
    called = [step["tool"] for step in traj["steps"]]
    must = expected["must_call"]
    if order_sensitive:
        # 정확 경로 일치. 필수 부분순서 허용과는 다른 정책.
        return float(called == must)
    required = set(must)
    return len(set(called) & required)/len(required) if required else 1.0

def efficiency_penalty(traj, ideal_steps: int) -> float:
    if type(ideal_steps) is not int or ideal_steps < 0:
        raise ValueError("ideal_steps는 0 이상 정수")
    return max(0.0, 1.0 - max(0, len(traj["steps"])-ideal_steps)*0.1)
# 초과1step=0.1감점은 학습용 가중치이며 성공률/승인 기준이 아니다.
```

### Task Success — 결정적 검증기 우선, 없으면 judge

인자 schema 통과는 의미·권한·동작 성공 증거가 아니다. 금지 도구는 실행 시 ACL/승인으로 차단해야 하며 평가 로그는 사후 검출 자료다. 사내 실제 API/DB 상태 검증은 미확인이다.

```python
import math

def task_success(traj, expected, state_check=None, judge_fn=None) -> float:
    if state_check is not None:
        result = state_check(traj, expected)  # 최종 DB/도구 관찰 상태 계약을 직접 검증
        if type(result) is not bool:
            raise ValueError("결정적 검증 결과 미확인")
        return float(result)
    if judge_fn is None:
        raise ValueError("상태 검증기 또는 reference/rubric judge가 필요")
    result = float(judge_fn({"question": expected.get("task"),
        "reference": expected.get("gold_answer")}, traj["final_answer"])["score"])
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError("judge 품질 점수 미확인")
    return result  # 연속 품질 점수이며 성공 boolean이 아님
# SequenceMatcher의 문자열0.9는 숫자/부정/상태의 정확성을 보증하지 않아 제거했다.
```

### 비결정성 다루기 — 여러 번 실행해 분포로

```python
def success_rate(agent_fn, case, expected, runs=5, state_check=None) -> float:
    if type(runs) is not int or runs <= 0 or state_check is None:
        raise ValueError("양의 runs와 최종 상태 검증기 필요")
    scores = [task_success(agent_fn(case), expected, state_check=state_check) for _ in range(runs)]
    return sum(scores)/len(scores)  # 각 실행의 binary 성공 비율. pass@k/pass^k와 다름.
# 같은 입력 반복은 실행 신뢰성 측정이다. 실행 독립성·상관 실패도 따로 확인한다.
```

### Agent 평가 리포트에 함께 담을 것
- 카테고리별 **Task Success Rate**(±분산).
- **실패 모드 분포**(어떤 유형이 많은가) — 개선 우선순위.
- **평균 스텝 수·토큰·지연** — 품질과 비용의 균형. → [12](./12-monitoring-drift.md)

> 로컬엔 사내 API/DB가 없으므로 실제 도구는 **mock/녹화된 관찰(obs)**로 대체해 경로·인자 검증까지만 오프라인으로 돌린다.

## 관련 문서
- [03. 트레이싱 & 관측성](./03-tracing-observability.md) — trajectory는 span 트리에서 나온다
- [07. LLM-as-a-Judge](./07-llm-as-a-judge.md) — task success의 judge 채점
- [10. 안전성·가드레일 평가](./10-safety-hallucination-guardrails.md) — 금지 도구·부작용은 안전 이슈이기도

## 참고 자료 (References)
- [jsonschema 검증 공식 API](https://python-jsonschema.readthedocs.io/en/stable/validate/) — 2026-10-04, schema 오류와 instance 오류 구분.
- τ-bench(도구 사용 에이전트 벤치마크): https://arxiv.org/abs/2406.12045
