---
tags: [evaluation, agent, tool-calling, trajectory, task-success]
level: advanced
last_updated: 2026-07-06
---

# 09. Agent · Tool 평가

> Agent는 여러 스텝(도구 선택 → 호출 → 관찰 → 다음 결정)을 거치므로, 최종 답만이 아니라 **경로(trajectory)**와 **도구 호출 정확도**를 평가해야 실패 원인을 짚는다.

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
- **효율성**: 스텝 수·토큰·지연 — 맞았어도 20스텝이면 나쁘다.

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
def tool_selection_score(traj, expected) -> float:
    called = [s["tool"] for s in traj["steps"]]
    must = expected["must_call"]
    hit = len(set(called) & set(must)) / max(len(must), 1)     # 필요한 도구 호출률
    # 금지 도구를 부르면 강한 감점 (부작용 방지)
    if any(t in called for t in expected.get("forbidden", [])):
        return 0.0
    return hit

def arg_validity_score(traj, schemas) -> float:
    """각 도구 인자가 스키마를 지켰는지 (jsonschema 등으로 검증)"""
    ok = 0
    for s in traj["steps"]:
        ok += validate_args(s["tool"], s["args"], schemas)     # 1/0
    return ok / max(len(traj["steps"]), 1)
```

### Trajectory 일치 & 효율성

```python
def trajectory_match(traj, expected, order_sensitive=False) -> float:
    called = [s["tool"] for s in traj["steps"]]
    must = expected["must_call"]
    if order_sensitive:
        # 기대 시퀀스가 부분 순서로 등장하는가
        it = iter(called)
        return 1.0 if all(t in it for t in must) else 0.0
    return len(set(called) & set(must)) / len(must)

def efficiency_penalty(traj, ideal_steps: int) -> float:
    n = len(traj["steps"])
    return max(0.0, 1.0 - max(0, n - ideal_steps) * 0.1)       # 초과 스텝당 감점
```

### Task Success — 결정적 검증기 우선, 없으면 judge

```python
def task_success(traj, expected) -> float:
    # 1) 가능하면 결정적 검증 (정답 값/상태 비교)
    if "gold_answer" in expected:
        from difflib import SequenceMatcher
        r = SequenceMatcher(None, traj["final_answer"], expected["gold_answer"]).ratio()
        if r > 0.9: return 1.0
    # 2) 자유서술이면 LLM-judge (07번)
    return judge_pointwise({"question": expected.get("task"), "reference": expected.get("gold_answer")},
                           traj["final_answer"])["score"]
```

### 비결정성 다루기 — 여러 번 실행해 분포로

```python
def success_rate(agent_fn, case, expected, runs=5) -> float:
    scores = []
    for _ in range(runs):
        traj = agent_fn(case)                    # 온도>0이면 매번 경로가 다름
        scores.append(task_success(traj, expected))
    return sum(scores) / runs                    # pass@k, 평균 성공률
```

### Agent 평가 리포트에 함께 담을 것
- 카테고리별 **Task Success Rate**(±분산).
- **실패 모드 분포**(어떤 유형이 많은가) — 개선 우선순위.
- **평균 스텝 수·토큰·지연** — 품질과 비용의 균형. → [12](./12-monitoring-drift.md)

> 로컬엔 사내 API/DB가 없으므로, 실제 도구는 **mock/녹화된 관찰(obs)**로 대체해 경로·인자 검증까지만 오프라인으로 돌린다.

## 관련 문서
- [03. 트레이싱 & 관측성](./03-tracing-observability.md) — trajectory는 span 트리에서 나온다
- [07. LLM-as-a-Judge](./07-llm-as-a-judge.md) — task success의 judge 채점
- [10. 안전성·가드레일 평가](./10-safety-hallucination-guardrails.md) — 금지 도구·부작용은 안전 이슈이기도

## 참고 자료 (References)
- Agent trajectory 평가 개념(LangSmith agent eval): https://docs.smith.langchain.com/
- τ-bench(도구 사용 에이전트 벤치마크): https://arxiv.org/abs/2406.12045
