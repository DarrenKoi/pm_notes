---
tags: [llmops, lifecycle, mlops, overview]
level: beginner
last_updated: 2026-07-06
---

# 01. LLMOps 개요 및 라이프사이클

> LLM 앱을 "한 번 만들고 끝"이 아니라 **지속적으로 측정·개선·운영**하는 시스템으로 보는 관점을 잡는다. LLMOps의 심장은 평가 루프임을 이해한다.

## 왜 필요한가? (Why)

- 데모는 프롬프트 몇 줄로 되지만, **운영**은 다르다. 모델이 바뀌고(Kimi-K2.5 → Qwen 등), 프롬프트를 고치고, 검색 문서가 늘어날 때마다 "**정말 좋아졌는가**"를 답할 수 없으면 개선이 도박이 된다.
- LLM 출력은 **비결정적**이고 정답이 하나가 아니다. 그래서 전통적 소프트웨어의 "테스트 통과=OK"가 성립하지 않고, **점수 분포로 관리**해야 한다.
- 사내에서는 모델·프롬프트를 자주 교체한다. 교체할 때마다 회귀(regression)를 자동으로 잡아내는 **평가 파이프라인**이 없으면, 어제 잘 되던 케이스가 조용히 망가진다.

## 핵심 개념 (What)

### 1) LLMOps란
LLM 기반 애플리케이션을 **개발 → 배포 → 운영 → 개선**하는 전 과정을 자동화·표준화하는 실무 체계. MLOps의 하위/변형이지만 강조점이 다르다.

| 구분 | MLOps | LLMOps |
|------|-------|--------|
| 중심 자산 | 학습된 **모델 가중치** | **프롬프트·검색·도구·평가셋** (모델은 보통 그대로 사용) |
| 반복 대상 | 데이터 → 재학습 | 프롬프트/컨텍스트/파이프라인 개선 |
| 품질 판단 | accuracy/F1 등 명확한 지표 | **정답이 여럿** → LLM-as-Judge·rubric 필요 |
| 핵심 루프 | train-eval-deploy | **eval-improve** (거의 매 커밋마다 평가) |

### 2) LLM 앱 라이프사이클 (평가가 관통한다)

```
요구·데이터 → 프롬프트/파이프라인 설계 → [오프라인 평가] → 배포
                        ↑                         ↓
                   개선(프롬프트/검색/모델)  ← [온라인 평가·모니터링]
```

- **오프라인 평가**: 고정된 `eval_set.jsonl`로 배포 전 채점 → [04](./04-llm-evaluation-overview.md)
- **온라인 평가**: 실제 트래픽에서 A/B·피드백·모니터링 → [11](./11-online-eval-deployment.md), [12](./12-monitoring-drift.md)
- 둘 사이를 **개선 루프**가 잇는다. 이 루프의 회전 속도가 곧 LLMOps 성숙도다.

### 3) 3개의 관측 대상
운영 중 반드시 붙잡아야 할 3가지 — **품질**(맞았나), **비용**(토큰·요금), **지연**(latency). 세 축을 함께 봐야 "품질은 올랐는데 비용이 3배"를 잡아낸다. → [03. 트레이싱](./03-tracing-observability.md)

## 어떻게 사용하는가? (How)

### 최소 LLMOps 루프의 뼈대
프레임워크 없이도 "평가 가능한 앱"의 골격은 이 정도다. 핵심은 **앱 함수 하나 + 평가 함수 하나**를 분리하는 것.

```python
# app.py — 평가 대상 시스템 (프롬프트/검색/모델을 이 안에서 조립)
from openai import OpenAI
client = OpenAI(base_url="http://llm-gateway.internal/v1", api_key="EMPTY")
TARGET_MODEL = "Kimi-K2.5"

PROMPT_VERSION = "v3"   # 프롬프트도 버전으로 관리 → 02번 문서
def answer(question: str) -> str:
    msgs = [
        {"role": "system", "content": "너는 반도체 공정 QA 도우미다. 모르면 모른다고 답해."},
        {"role": "user", "content": question},
    ]
    r = client.chat.completions.create(model=TARGET_MODEL, messages=msgs, temperature=0)
    return r.choices[0].message.content
```

```python
# run_eval.py — 고정 데이터셋으로 매번 같은 방식으로 채점
import json
def load_set(path="eval_set.jsonl"):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]

def evaluate(answer_fn, dataset, score_fn):
    rows = []
    for c in dataset:
        pred = answer_fn(c["question"])
        rows.append({"id": c["id"], "pred": pred, "score": score_fn(c, pred)})
    avg = sum(r["score"] for r in rows) / len(rows)
    return avg, rows

# score_fn은 06(자동지표)·07(LLM-judge)에서 구현한다
```

> 이 뼈대만 갖춰도 "프롬프트 v3 vs v4", "Kimi vs Qwen"을 **같은 데이터·같은 채점**으로 비교할 수 있다. 이것이 LLMOps의 출발점이다.

### 성숙도 단계 (자가진단)
1. **Level 0** — 눈으로 확인(“돌려보니 되네”). 회귀 못 잡음.
2. **Level 1** — 고정 eval set + 자동 채점(오프라인). 배포 전 점수 비교.
3. **Level 2** — 트레이싱·비용·지연까지 기록, CI에 평가 게이트.
4. **Level 3** — 온라인 A/B + 피드백 + 드리프트 감지로 개선 루프 자동화.

## 관련 문서
- [02. 프롬프트 관리 & 버전 관리](./02-prompt-management-versioning.md) — 개선 루프에서 무엇을 버전으로 다룰지
- [04. LLM 평가 개요](./04-llm-evaluation-overview.md) — 라이프사이클을 관통하는 평가의 원리
- [13. Mini Project](./13-mini-project.md) — 위 뼈대를 실제 파이프라인으로 확장

## 참고 자료 (References)
- LLMOps 개념 개요(일반): "LLMOps = MLOps for LLM apps, eval-centric"
- OpenAI Evals(평가 루프 사고방식): https://github.com/openai/evals
