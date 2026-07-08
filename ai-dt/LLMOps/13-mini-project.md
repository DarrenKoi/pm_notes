---
tags: [llmops, evaluation, mini-project, capstone]
level: advanced
last_updated: 2026-07-06
---

# 13. Mini Project — 사내 RAG/Agent 평가 파이프라인 구축

> 앞의 12편을 하나로 엮는 캡스톤. **"사내 RAG 챗봇을 평가 가능한 시스템으로 만들기"**를 주제로 요구사항 → 구현 → 테스트 → 발표/피드백까지 한 사이클을 돈다.

## 왜 필요한가? (Why)

- 지식은 따로 배우면 흩어진다. 데이터셋·자동지표·LLM-judge·RAG지표·안전·CI게이트·모니터링을 **한 파이프라인**으로 통합해야 실무 감각이 생긴다.
- 산출물(`eval_set.jsonl` + `run_eval.py` + CI gate + 리포트)은 그대로 **실제 사내 프로젝트의 출발점**이 된다.

## 핵심 개념 (What) — 프로젝트 골격

```
llmops-eval/
├── eval_set.jsonl          # 05: 데이터셋 (golden + 합성 + 거절 케이스)
├── prompts/                # 02: 버전 관리되는 프롬프트
├── system.py               # 평가 대상 RAG (검색+생성)
├── scorers/
│   ├── automatic.py        # 06: exact/rouge/임베딩
│   ├── judge.py            # 07: pointwise/pairwise
│   ├── rag.py              # 08: recall/faithfulness/relevance
│   └── safety.py           # 10: 거절/누출/인젝션
├── run_eval.py             # 04: 하네스 (데이터×시스템×채점기→리포트)
├── ci_eval.py              # 11: regression gate
├── tracing.py              # 03: Phoenix 계측 설정 (OpenInference)
├── release_manifest.json   # 14: 배포 아티팩트 버전 조합
├── incidents/              # 15: 사고 기록과 postmortem
└── report.md               # 결과·개선안 (발표용)
```

## 어떻게 진행하는가? (How)

### 1단계 — 주제 선정 & 요구사항 정의
- **대상 시스템**: 사내 공정 문서 RAG QA(예: 특정 공정 카테고리 하나로 스코프 축소).
- **성공 기준(측정 가능하게)**: 예 — faithfulness ≥ 0.85, 거절 정확도 ≥ 0.9, 누출률 0, p95 ≤ 3s.
- **평가 축 선정**: 검색·생성·안전을 최소 하나씩 포함. → [04](./04-llm-evaluation-overview.md)

### 2단계 — 평가 데이터셋 구축 (→ [05](./05-eval-dataset-construction.md))
- DRM 문서를 VLM(Qwen3-VL)으로 텍스트화 → context 확보.
- context에서 합성 QA 생성 + 사람 검수로 golden 50건.
- **거절 케이스**(문서에 없는 질문) 10~20% 포함.
- `eval_set.jsonl` 완성, 카테고리·난이도 균형 확인.

```python
# 최소 데이터셋 자가검증
import json
ds = [json.loads(l) for l in open("eval_set.jsonl", encoding="utf-8")]
assert all("question" in c and "meta" in c for c in ds)
refuse = [c for c in ds if not c["meta"].get("answerable", True)]
print(f"총 {len(ds)}건 / 거절 케이스 {len(refuse)}건({len(refuse)/len(ds):.0%})")
```

### 3단계 — 평가 대상 시스템 구현 (→ [02](./02-prompt-management-versioning.md))
- 버전 관리되는 프롬프트로 RAG 답변 함수 `answer(question) -> (pred, contexts, retrieved_ids)`.
- **Arize Phoenix** 계측 설정(`tracing.py`) — OpenAI 클라이언트 자동 계측으로 검색·생성 span 기록. → [03](./03-tracing-observability.md)

```python
# tracing.py — 프로젝트 시작 시 1회 호출
import os
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://phoenix.internal:6006"

from openinference.instrumentation.openai import OpenAIInstrumentor
from phoenix.otel import register
tracer_provider = register()
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
```

### 4단계 — 채점기 조립 (→ [06](./06-automatic-metrics.md)/[07](./07-llm-as-a-judge.md)/[08](./08-rag-evaluation.md)/[10](./10-safety-hallucination-guardrails.md))

```python
from scorers.automatic import cosine_sim
from scorers.judge import judge_pointwise
from scorers.rag import faithfulness, recall_at_k
from scorers.safety import refusal_correctness, leak_scan

def score_case(case, out):
    pred, contexts, retrieved_ids = out
    row = {
        "semantic":     cosine_sim(case, pred),
        "judge":        judge_pointwise(case, pred)["score"],
        "faithfulness": faithfulness(pred, contexts),
        "recall@5":     recall_at_k(retrieved_ids, case["meta"].get("gold_ids", []), 5),
        "refusal_ok":   refusal_correctness(case, pred),
        "leak":         0.0 if leak_scan(pred)["leak"] else 1.0,   # 누출 없으면 1
    }
    return row
```

### 5단계 — 하네스 실행 & 리포트 (→ [04](./04-llm-evaluation-overview.md))

```python
def main():
    ds = [json.loads(l) for l in open("eval_set.jsonl", encoding="utf-8")]
    rows = [ {"id": c["id"], "category": c["meta"]["category"], **score_case(c, answer_full(c["question"]))}
             for c in ds ]
    dims = ["semantic","judge","faithfulness","recall@5","refusal_ok","leak"]
    for d in dims:
        print(f"{d:14s} {sum(r[d] for r in rows)/len(rows):.3f}")
    # 카테고리별·실패 케이스 상위 N도 함께 출력 → 개선 우선순위
```

### 6단계 — 두 버전 비교 & CI 게이트 (→ [11](./11-online-eval-deployment.md))
- 프롬프트 v_old vs v_new를 같은 데이터로 채점, 회귀 케이스 추출.
- `ci_eval.py`로 regression gate + 안전 하드컷 구성.

### 6-1단계 — Release manifest & 운영 승인 (→ [14](./14-artifact-lineage-governance.md))
- prompt/model/index/tool/eval/rubric/guardrail 버전을 `release_manifest.json`에 고정.
- 위험 등급과 승인자를 적고, rollback 대상도 이전 아티팩트 조합으로 명시.
- trace 로그에 `release_id`를 남겨 [12](./12-monitoring-drift.md) 대시보드에서 버전별로 분리한다.

### 7단계 — 사고 시뮬레이션 & 피드백 (→ [15](./15-incident-response-postmortem.md))
- synthetic 실패 1건을 만들어 incident record와 postmortem을 작성한다.
- 사고 케이스를 `redteam` 또는 `production-mined` eval case로 승격한다.
- 새 케이스가 CI에서 재현되고, 패치 후 통과하는지 확인한다.

### 8단계 — 발표 & 피드백
발표 덱(또는 `report.md`)에 담을 것:
1. **문제·목표**(성공 기준을 숫자로).
2. **데이터셋**(규모·커버리지·거절 비율).
3. **결과 표**(차원별·카테고리별 점수, v_old vs v_new).
4. **실패 분석**(대표 실패 3~5개 + 원인: 검색 vs 생성 vs 안전).
5. **개선안 & 다음 스텝**(무엇을 바꾸면 어느 지표가 오를지 가설).
6. **운영 준비도**(manifest, rollback, incident runbook, 남은 risk).

## 평가 루브릭 (이 미니프로젝트 자체 채점)
| 항목 | 확인 |
|------|------|
| 데이터셋이 실사용을 대표하고 거절 케이스 포함 | ☐ |
| 검색·생성·안전 지표를 모두 측정 | ☐ |
| judge를 golden 라벨로 메타평가(일치도 보고) | ☐ |
| 두 버전을 같은 잣대로 비교, 회귀 케이스 식별 | ☐ |
| CI regression gate + 안전 하드컷 동작 | ☐ |
| 실패 분석에서 검색/생성 원인 분리 | ☐ |
| Phoenix 계측으로 trace가 수집되고 UI에서 조회 가능 | ☐ |
| release manifest에 아티팩트 버전과 rollback 대상 명시 | ☐ |
| incident/postmortem을 통해 실패 케이스를 eval set에 편입 | ☐ |

## 관련 문서
- [01. LLMOps 개요](./01-llmops-overview-lifecycle.md) — 전체 라이프사이클
- [04~10] — 각 단계의 이론·구현
- [11. 온라인 평가 & 배포](./11-online-eval-deployment.md) / [12. 모니터링](./12-monitoring-drift.md) — 확장 방향
- [14. 아티팩트 계보와 거버넌스](./14-artifact-lineage-governance.md) / [15. Incident Response](./15-incident-response-postmortem.md) — 운영 준비도 보강

## 참고 자료 (References)
- 앞 문서 12편의 References 종합
- OpenAI Evals(파이프라인 사고): https://github.com/openai/evals
