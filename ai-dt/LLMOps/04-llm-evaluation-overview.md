---
tags: [evaluation, offline, online, paradigms]
level: beginner
last_updated: 2026-07-06
reviewed_on: 2026-10-04
review_status: partial
document_type: learning
category_major: "AI·DT"
category_middle: "LLM 평가·운영"
category_minor: "평가 데이터·지표"
note_kind: "학습"
classified_on: "2026-10-05"
---

> [!info] 검토 범위 — 2026-10-04
> 공식·일차 근거와 로컬 검증은 [공통 적용 조건](./verified-conditions.md), 변경·미확인은 [정리 기록](./organization-log.md)에 있다. 실제 사내 접속·모델 품질·운영 승인과 Claude 협의는 미확인이다. 원래17개를 개별 검토했다. 실제 운영·읽기 화면 검증은 미완료다.


# 04. LLM 평가(Evaluation) 개요

> LLM 평가가 왜 전통적 테스트와 다른지, 어떤 평가 축(offline/online, reference 유무, 자동/인간)이 있는지 지도를 그린다. 이후 06~10번이 각 축의 구현이다.

## 왜 필요한가? (Why)

- LLM 출력은 **정답이 하나가 아니다.** "포토 공정을 설명하라"의 정답은 무수히 많다. 문자열 일치만으로 자유서술의 사실성을 평가하기 어렵다. 분류·형식 테스트에는 일치 검사가 유용하다.
- 그래서 "테스트 통과/실패"가 아니라 **점수 분포**로 관리한다. 목표는 100점이 아니라 **"이번 변경이 회귀 없이 개선인가"**를 신뢰성 있게 답하는 것.
- 사내에서 모델·프롬프트·검색을 자주 바꾸므로, **변경마다 같은 잣대로 재는 평가 체계**가 없으면 개선이 감(感)에 의존한다.

## 핵심 개념 (What)

### 1) 평가를 나누는 3개의 축

**축 A — 언제 평가하나**
- **오프라인(offline)**: 고정 `eval_set.jsonl`으로 배포 전 채점. 재현 가능, 회귀 감지. → 06~10번의 주무대.
- **온라인(online)**: 실제 트래픽에서 A/B·피드백·모니터링. 진짜 사용자 반응. → [11](./11-online-eval-deployment.md), [12](./12-monitoring-drift.md)

**축 B — 정답(reference)이 있나**
- **Reference-based**: 정답이 있음 → exact match, BLEU/ROUGE, 임베딩 유사도. → [06](./06-automatic-metrics.md)
- **Reference-free**: 검증된 reference 없이 평가하되 rubric·질문·검색 문맥 등 과제별 입력이 필요 → LLM-as-Judge(rubric), faithfulness. → [07](./07-llm-as-a-judge.md), [08](./08-rag-evaluation.md)

**축 C — 누가 채점하나**
- **자동 지표** (규칙/통계): 싸고 빠름, 의미 파악은 약함.
- **LLM-as-Judge**: 의미까지 채점, 편향 주의·보정 필요.
- **인간(human)**: 전문가·rubric·일치도 검증이 필요하며 비용이 큼 → 소량 golden 라벨·보정용.

### 2) 무엇을 재는가 — 품질의 하위 차원
"좋은 답"은 단일 축이 아니다. 최소 이렇게 쪼갠다.

| 차원 | 질문 |
|------|------|
| 정확성(correctness) | 사실이 맞는가 |
| 충실성(faithfulness) | 주어진 문서에 근거하는가(RAG 환각 여부) |
| 관련성(relevance) | 질문에 답하는가 |
| 완결성(completeness) | 빠뜨린 게 없는가 |
| 형식(format) | 요구 포맷/스키마를 지키는가 |
| 안전성(safety) | 유해·기밀 누출이 없는가 → [10](./10-safety-hallucination-guardrails.md) |

전체 평균 한 숫자보다 **차원별 점수**가 개선 방향을 알려준다.

### 3) 평가의 신뢰성 = 메타평가
"내 채점기가 믿을 만한가?"도 검증 대상이다. LLM-judge 점수를 소량의 **인간 라벨과의 일치도 또는 상관관계(서로 다른 측정값)**로 맞춰본다. → [07](./07-llm-as-a-judge.md)

## 어떻게 사용하는가? (How)

### 평가 하네스(harness)의 표준 형태
어떤 지표를 쓰든 구조는 같다: **데이터셋 × 시스템 × 채점기 → 리포트**.

```python
import json, statistics, math

def load_set(path="eval_set.jsonl"):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]

def run_eval(system_fn, scorers: dict, dataset):
    """scorers: {"correctness": fn, "faithfulness": fn, ...}"""
    dataset = list(dataset)
    if not dataset or not scorers:
        raise ValueError("평가셋과 채점기가 필요")
    ids = [c["id"] for c in dataset]
    if len(set(ids)) != len(ids):
        raise ValueError("중복 case id")
    rows = []
    for c in dataset:
        pred = system_fn(c["question"])
        row = {"id": c["id"], "category": c.get("meta", {}).get("category")}
        for name, fn in scorers.items():
            value = float(fn(c, pred))
            if not math.isfinite(value):
                raise ValueError(f"{name}: 점수 미확인/비유한 값")
            row[name] = value  # 범위·방향은 지표별 정의. cosine은 [-1, 1]
        rows.append(row)
    return rows

def report(rows, scorers):
    if not rows:
        raise ValueError("빈 리포트")
    print("=== 차원별 평균 ===")
    for name in scorers:
        vals = [r[name] for r in rows]
        print(f"{name:14s} {statistics.mean(vals):.3f}")
    # 카테고리별로도 쪼개 본다 (전체 평균이 숨기는 회귀 발견)
    cats = {r["category"] for r in rows}
    for cat in cats:
        sub = [r for r in rows if r["category"] == cat]
        for name in scorers:
            m = statistics.mean(r[name] for r in sub)
            print(f"  [{cat}] {name}={m:.3f} (n={len(sub)})")
```

### 두 시스템을 같은 잣대로 비교 (A/B의 오프라인 버전)

```python
def find_regressions(rows_a, rows_b, metric="correctness", margin=0.1):
    # 높은 값이 좋은 지표 전용. latency/cost의 방향은 별도 정의한다.
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("허용 하락폭 오류")
    a = {row["id"]: row for row in rows_a}
    b = {row["id"]: row for row in rows_b}
    if not a or len(a) != len(rows_a) or len(b) != len(rows_b) or a.keys() != b.keys():
        raise ValueError("빈 결과·중복·서로 다른 case 집합")
    for table in (a, b):
        if any(not math.isfinite(row[metric]) for row in table.values()):
            raise ValueError("비유한 비교 점수")
    return [case_id for case_id in a if b[case_id][metric] < a[case_id][metric] - margin]

# system_v3/system_v4, scorers, dataset을 준비한 뒤 run_eval을 각각 호출한다.
# regressed = find_regressions(rows_a, rows_b)  # id로 비교하므로 행 순서에 영향 없음
```

### 평가 설계 체크리스트
- [ ] eval set이 **실제 사용 분포**를 대표하는가(카테고리 커버리지). → [05](./05-eval-dataset-construction.md)
- [ ] 지표가 **품질 차원별로** 분리돼 있는가.
- [ ] 채점기 자체를 **인간 라벨로 검증**했는가(메타평가).
- [ ] 결과가 **재현 조건**을 기록했는가(고정 데이터·모델/서빙/seed/채점기 버전·반복 분산). 온도 0도 완전한 결정성 보장은 아니다.

## 관련 문서
- [05. 평가 데이터셋 구축](./05-eval-dataset-construction.md) — 무엇으로 재는가
- [06. 자동 평가 지표](./06-automatic-metrics.md) — reference-based 채점
- [07. LLM-as-a-Judge](./07-llm-as-a-judge.md) — reference-free 채점

## 참고 자료 (References)
- MT-Bench / Chatbot Arena(LLM 평가 관점): https://arxiv.org/abs/2306.05685
- Holistic Evaluation of Language Models(HELM, 다차원 평가): https://crfm.stanford.edu/helm/
