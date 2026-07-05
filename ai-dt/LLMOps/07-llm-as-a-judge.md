---
tags: [evaluation, llm-as-judge, rubric, bias, calibration]
level: advanced
last_updated: 2026-07-06
---

# 07. LLM-as-a-Judge

> 정답이 여럿이거나 없는 자유서술을 채점하기 위해 LLM을 판정자로 쓴다. rubric 설계, pointwise/pairwise, 그리고 반드시 다뤄야 할 **편향과 보정**을 익힌다.

## 왜 필요한가? (Why)

- 자유 QA·요약·설명은 표면 지표([06](./06-automatic-metrics.md))로 못 잡는다. "사실인가, 질문에 답했나, 근거가 있나"는 **의미 이해**가 필요하다.
- 인간 평가는 가장 정확하지만 수백 케이스를 매 커밋마다 볼 수 없다. LLM-judge는 **인간 평가의 근사치를 자동·대량**으로 낸다.
- 사내에서는 판정자도 **내부 모델(Kimi-K2.5)**이어야 한다(외부 API 차단). 외부 프레임워크의 기본 judge를 반드시 교체한다.

## 핵심 개념 (What)

### 1) 두 가지 판정 방식
- **Pointwise(절대 채점)**: 답 하나에 rubric 기준으로 점수(1~5 또는 0~1). 회귀 추적·대시보드에 적합.
- **Pairwise(상대 비교)**: A vs B 중 더 나은 것 선택. 두 프롬프트/모델 버전 비교에 강력(절대 점수보다 일관됨). → [02](./02-prompt-management-versioning.md), [11](./11-online-eval-deployment.md)

### 2) rubric이 전부다
judge의 신뢰성은 **명확한 채점 기준**에서 나온다. 좋은 rubric은:
- **차원 분리**: correctness / faithfulness / relevance 각각 따로 채점.
- **구체적 기준**: "5점=모든 사실이 근거와 일치, 3점=핵심은 맞으나 일부 근거 없음, 1점=근거와 모순".
- **근거 요구**: 점수와 함께 **이유(reasoning)**를 내게 해 검증 가능·안정적으로.
- **구조화 출력**: JSON으로 받아 파싱.

### 3) LLM-judge의 알려진 편향 (반드시 보정)
| 편향 | 내용 | 완화 |
|------|------|------|
| **위치 편향** | pairwise에서 앞/뒤 위치를 선호 | A/B 순서 바꿔 두 번, 일치할 때만 승자 인정 |
| **장황함 편향** | 긴 답을 더 좋게 봄 | rubric에 "길이 아닌 정확성" 명시, 길이 통제 |
| **자기 선호** | 같은 계열 모델 답을 선호 | judge와 대상 모델 분리, 인간 라벨로 검증 |
| **관대함** | 전반적으로 후하게 줌 | 낮은 점수 기준을 rubric에 구체화, 보정 |

### 4) 메타평가 — judge를 믿기 전에 검증
judge 점수와 **golden set 인간 라벨의 일치도**(정확도/상관/Cohen's kappa)를 먼저 잰다. 일치가 낮으면 rubric을 고친다. 이 단계 없이 judge를 배포하면 "그럴듯한 잘못된 점수"를 신뢰하게 된다.

## 어떻게 사용하는가? (How)

### Pointwise judge (rubric + 구조화 출력)

```python
import json
from openai import OpenAI
client = OpenAI(base_url="http://llm-gateway.internal/v1", api_key="EMPTY")
JUDGE = "Kimi-K2.5"

RUBRIC = """너는 엄격한 평가자다. 아래 답변을 기준별로 1~5로 채점하라.
- correctness: 사실이 정답과 일치하는가 (5=완전일치 … 1=모순)
- relevance: 질문에 직접 답하는가
- completeness: 핵심을 빠뜨리지 않았는가
길이가 길다고 후하게 주지 마라. 반드시 JSON만 출력: 
{"correctness":n,"relevance":n,"completeness":n,"reason":"..."}"""

def judge_pointwise(case, pred) -> dict:
    user = f"[질문]\n{case['question']}\n\n[정답]\n{case.get('reference','(없음)')}\n\n[평가할 답변]\n{pred}"
    out = client.chat.completions.create(
        model=JUDGE, temperature=0,
        messages=[{"role":"system","content":RUBRIC},{"role":"user","content":user}],
        response_format={"type":"json_object"},
    ).choices[0].message.content
    d = json.loads(out)
    d["score"] = (d["correctness"] + d["relevance"] + d["completeness"]) / 15  # 0~1 정규화
    return d

# run_eval의 scorer로: lambda c,p: judge_pointwise(c,p)["score"]
```

### Pairwise judge (위치 편향 보정 포함)
A/B를 **양쪽 순서로 두 번** 물어 일치할 때만 승자를 인정. 불일치는 무승부(tie).

```python
def _ask_which(question, a, b):
    prompt = f"""질문에 대한 두 답변 중 더 정확하고 근거 있는 것을 고르라.
길이가 길다고 선호하지 마라. "A" 또는 "B"만 출력.

[질문] {question}
[A] {a}
[B] {b}"""
    return client.chat.completions.create(
        model=JUDGE, temperature=0,
        messages=[{"role":"user","content":prompt}]).choices[0].message.content.strip().upper()[:1]

def judge_pairwise(question, ans_a, ans_b) -> str:
    r1 = _ask_which(question, ans_a, ans_b)          # A=ans_a, B=ans_b
    r2 = _ask_which(question, ans_b, ans_a)          # 순서 뒤집기
    # r1이 A이고 r2가 B이면 둘 다 ans_a를 택함 → 위치 편향 아님
    if r1 == "A" and r2 == "B": return "a"
    if r1 == "B" and r2 == "A": return "b"
    return "tie"                                     # 순서 뒤집으니 뒤집힘 → 신뢰 못 함
```

### 메타평가 — judge vs 인간 라벨

```python
def meta_eval(judge_fn, golden):
    """golden: [{...case..., 'human': 1/0}] — 인간이 pass/fail 라벨한 소량 셋"""
    agree = 0
    for c in golden:
        judged = 1 if judge_fn(c, c["pred"])["score"] >= 0.6 else 0
        agree += (judged == c["human"])
    acc = agree / len(golden)
    print(f"judge-인간 일치도: {acc:.2f}")   # 0.8+ 목표. 낮으면 rubric 수정
    return acc
```

### 비용·안정성 팁
- judge는 **온도 0**으로 고정(재현성).
- CI에서는 자동 지표로 **1차 필터**하고, 애매한 케이스만 judge로 보내 비용 절감.
- 중요한 판정은 **self-consistency**(같은 판정 3회 다수결)로 분산을 줄인다.

## 관련 문서
- [04. 평가 개요](./04-llm-evaluation-overview.md) — reference-free 채점의 위치
- [06. 자동 평가 지표](./06-automatic-metrics.md) — judge 앞단의 저비용 필터
- [08. RAG 평가](./08-rag-evaluation.md) — faithfulness를 judge로 구현

## 참고 자료 (References)
- MT-Bench / "LLM-as-a-Judge" 논문: https://arxiv.org/abs/2306.05685
- 위치 편향·완화 논의: 위 논문 §4 (position bias, verbosity bias)
