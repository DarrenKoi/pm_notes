---
tags: [evaluation, llm-as-judge, rubric, bias, calibration]
level: advanced
last_updated: 2026-07-06
reviewed_on: 2026-10-04
review_status: partial
document_type: learning
---

> [!info] 검토 범위 — 2026-10-04
> [공통 적용 조건](./verified-conditions.md)과 [정리 기록](./organization-log.md)에 판본·출처·로컬 검증을 남겼다. 실제 judge 품질·사내 접속·운영 승인·Claude 협의·Obsidian 읽기 화면은 미확인이다.


# 07. LLM-as-a-Judge

> 정답이 여럿이거나 없는 자유서술을 채점하기 위해 LLM을 판정자로 쓴다. rubric 설계, pointwise/pairwise, 반드시 다뤄야 할 **편향과 보정**을 익힌다.

## 왜 필요한가? (Why)

- 자유 QA·요약·설명은 표면 지표([06](./06-automatic-metrics.md))로 못 잡는다. "사실인가, 질문에 답했나, 근거가 있나"는 **의미 이해**가 필요하다.
- 전문가 평가는 rubric·검수자 일치도 확인이 필요하며 수백 케이스를 매 커밋마다 볼 수 없다. LLM-judge는 **인간 평가의 근사치를 자동·대량**으로 낸다.
- 외부 API가 제한된 환경에서는 승인된 내부 판정자를 구성한다. 실제 사내 차단 정책·Kimi alias 가용성은 미확인이다. 외부 프레임워크의 기본 judge를 반드시 교체한다.

## 핵심 개념 (What)

### 1) 두 가지 판정 방식
- **Pointwise(절대 채점)**: 답 하나에 rubric 기준으로 점수(1~5 또는 0~1). 회귀 추적·대시보드에 적합.
- **Pairwise(상대 비교)**: A vs B 중 더 나은 것 선택. 두 버전 비교에 사용할 수 있으나 pointwise보다 항상 일관된다는 보장은 없다. → [02](./02-prompt-management-versioning.md), [11](./11-online-eval-deployment.md)

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
import os
from openai import OpenAI
client = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"])
JUDGE = os.environ["JUDGE_MODEL"]
import json, math
from pydantic import BaseModel, ConfigDict, Field

class JudgeResult(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    correctness: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    reason: str = Field(min_length=1)

RUBRIC = """너는 엄격한 평가자다. 아래 답변을 기준별로 1~5로 채점하라.
- correctness: 사실이 정답과 일치하는가 (5=완전일치 … 1=모순)
- relevance: 질문에 직접 답하는가
- completeness: 핵심을 빠뜨리지 않았는가
길이가 길다고 후하게 주지 마라. 반드시 JSON만 출력: 
{"correctness":n,"relevance":n,"completeness":n,"reason":"..."}"""

def judge_pointwise(case, pred) -> dict:
    if any(not isinstance(text, str) or not text.strip()
           for text in (case.get("question"), case.get("reference"), pred)):
        raise ValueError("이 rubric은 질문·검증된 reference·텍스트 답변이 필요")
    user = f"[질문]\n{case['question']}\n\n[정답]\n{case.get('reference','(없음)')}\n\n[평가할 답변]\n{pred}"
    out = client.chat.completions.create(
        model=JUDGE, temperature=0,
        messages=[{"role":"system","content":RUBRIC},{"role":"user","content":user}],
        response_format={"type":"json_object"},
    ).choices[0].message.content
    if not isinstance(out, str):
        raise ValueError("judge 텍스트 결과 없음")
    d = JudgeResult.model_validate_json(out).model_dump()
    d["score"] = (d["correctness"] + d["relevance"] + d["completeness"] - 3) / 12  # 1점씩=0, 5점씩=1
    return d

# run_eval의 scorer로: lambda c,p: judge_pointwise(c,p)["score"]
```

이 pointwise rubric의 correctness는 reference가 필요하다. reference-free relevance/faithfulness에는 다른 입력과 rubric을 정의한다. JSON schema 통과와 judge의 설명은 사실성·근거의 실제 검증을 대신하지 않는다.

### Pairwise judge (위치 편향 보정 포함)
A/B를 **양쪽 순서로 두 번** 물어 같은 원래 답을 선택할 때만 승자를 인정한다. 둘 다 TIE면 무승부, 순서 불일치는 disagreement로 구분한다. 형식 오류는 미확인 오류이며 승률 분모에 정상 판정으로 넣지 않는다.

```python
def _ask_which(question, a, b):
    prompt = f"""질문에 대한 두 답변 중 더 정확하고 근거 있는 것을 고르라.
길이로 선호하지 마라. A, B, TIE 중 한 단어만 출력.
[질문] {question}
[A] {a}
[B] {b}"""
    out = client.chat.completions.create(
        model=JUDGE, temperature=0,
        messages=[{"role": "user", "content": prompt}]).choices[0].message.content
    if not isinstance(out, str) or out.strip().upper() not in {"A", "B", "TIE"}:
        raise ValueError("pairwise 출력 형식 미확인")
    return out.strip().upper()

def judge_pairwise(question, ans_a, ans_b) -> str:
    if any(not isinstance(x, str) or not x.strip() for x in (question, ans_a, ans_b)):
        raise ValueError("질문·두 답변 필요")
    first = _ask_which(question, ans_a, ans_b)
    second = _ask_which(question, ans_b, ans_a)
    if first == "A" and second == "B": return "a"
    if first == "B" and second == "A": return "b"
    if first == second == "TIE": return "tie"
    return "disagreement"  # 순서를 바꿔도 일치했다고 편향이 없다는 증거는 아님
```

### 메타평가 — judge vs 인간 라벨

```python
def meta_eval(judge_fn, golden, threshold=0.6):
    """golden: pred와 전문가 human(정수0/1)을 포함. calibration과 holdout 분리."""
    golden = list(golden)
    if not golden or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("평가셋/threshold 오류")
    agreement = 0
    for case in golden:
        human = case.get("human")
        if type(human) is not int or human not in (0, 1):
            raise ValueError("검수된 human 0/1 라벨 필요")
        score = float(judge_fn(case, case["pred"])["score"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("judge 점수 미확인/범위 오류")
        agreement += int(score >= threshold) == human
    accuracy = agreement / len(golden)
    print(f"judge-인간 이진 일치도: {accuracy:.2f} (n={len(golden)})")
    return accuracy
# 0.6/0.8 목표는 가상 시작값이며 배포 품질 보장 기준이 아니다.
```

### 비용·안정성 팁
- judge는 **온도 0**과 모델/서빙/프롬프트 판본을 기록한다. 온도 0도 완전한 재현성을 보장하지 않는다.
- CI에서는 자동 지표로 **1차 필터**하고 애매한 케이스만 judge로 보내 비용 절감.
- 중요한 판정은 **self-consistency**(같은 판정 3회 다수결)을 조사할 수 있지만 독립 표본/편향 제거·정확도 개선이 보장되지 않는다.

## 관련 문서
- [04. 평가 개요](./04-llm-evaluation-overview.md) — reference-free 채점의 위치
- [06. 자동 평가 지표](./06-automatic-metrics.md) — judge 앞단의 저비용 필터
- [08. RAG 평가](./08-rag-evaluation.md) — faithfulness를 judge로 구현

## 참고 자료 (References)
- MT-Bench / "LLM-as-a-Judge" 논문: https://arxiv.org/abs/2306.05685
- 위치 편향·완화 논의: 위 논문 §4 (position bias, verbosity bias)
