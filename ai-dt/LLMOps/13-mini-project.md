---
tags: [llmops, evaluation, mini-project, capstone]
level: advanced
last_updated: 2026-07-06
reviewed_on: 2026-10-04
review_status: partial
document_type: learning
---

> [!info] 검토 범위 — 2026-10-04
> [공통 적용 조건](./verified-conditions.md)과 [정리 기록](./organization-log.md)에 판본·일차 근거·로컬 검증을 기록했다. 실제 회사 운영·모델/서버 품질·조직 승인·Claude 협의·읽기 화면은 미확인이다.


# 13. Mini Project — 사내 RAG/Agent 평가 파이프라인 구축

> 앞의 12편을 하나로 엮는 캡스톤. **"사내 RAG 챗봇을 평가 가능한 시스템으로 만들기"**를 주제로 요구사항 → 구현 → 테스트 → 발표/피드백까지 한 사이클을 돈다.

## 왜 필요한가? (Why)

- 지식은 따로 배우면 흩어진다. 데이터셋·자동지표·LLM-judge·RAG지표·안전·CI게이트·모니터링을 **한 파이프라인**으로 통합해야 실무 감각이 생긴다.
- 아래는 학습용 산출물 구성안이다. 해당 examples 패키지는 아직 이 폴더에 없고 기존 실제 프로젝트를 구현/검증한 문서도 아니다.

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

파일 트리는 만들려는 프로젝트의 구성도이며 실행 파일 목록이 아니다. 코드 블록은 선행 함수/입력을 준비한 조립 예제다. 실제 서비스·운영 승인·대표 평가셋은 미확인이다.

### 1단계 — 주제 선정 & 요구사항 정의
- **대상 시스템**: 사내 공정 문서 RAG QA(예: 특정 공정 카테고리 하나로 스코프 축소).
- **성공 기준(제안값, 조직 승인 전)**: 예 — faithfulness ≥ 0.85, 거절 정확도 ≥ 0.9, 누출률 0, p95 ≤ 3s.
- **평가 축 선정**: 검색·생성·안전을 최소 하나씩 포함. → [04](./04-llm-evaluation-overview.md)

### 2단계 — 평가 데이터셋 구축 (→ [05](./05-eval-dataset-construction.md))
- 승인된 export/화면·VLM 후보 방식으로 근거 확보; 원문 숫자/표/페이지를 검수.
- context에서 합성 QA 생성 + 사람 검수로 시작 규모 예시 golden 50건.
- **거절 케이스**(문서에 없는 질문) 시작 비율 예시10~20% 포함; 실제 위험/사용 분포에 맞춰 근거 기록.
- `eval_set.jsonl` 완성, 카테고리·난이도 균형 확인.

```python
import json

def validate_dataset(cases):
    if not cases: raise ValueError("빈 평가셋")
    ids = []
    for case in cases:
        if not isinstance(case.get("id"),str) or not case["id"] or not isinstance(case.get("question"),str) or not case["question"].strip():
            raise ValueError("id/질문 필요")
        meta = case.get("meta",{})
        if not isinstance(meta.get("category"),str) or not meta["category"] or type(meta.get("answerable")) is not bool:
            raise ValueError("category/검수한 answerable 필요")
        if not isinstance(case.get("contexts"),list): raise ValueError("검색 문맥 목록 필요")
        ids.append(case["id"])
    if len(set(ids)) != len(ids): raise ValueError("중복 id")
    return cases

def load_dataset(path="eval_set.jsonl"):
    with open(path,encoding="utf-8") as file:
        cases = [json.loads(line) for line in file if line.strip()]
    return validate_dataset(cases)
# load_dataset() 후 검수/출처·실사용 대표성 확인. 필드 통과는 golden 승인 증거가 아님.
```

### 3단계 — 평가 대상 시스템 구현 (→ [02](./02-prompt-management-versioning.md))
- 버전 관리되는 프롬프트로 RAG 답변 함수 `answer(question) -> (pred, contexts, retrieved_ids)`.
- **Arize Phoenix** 계측 설정(`tracing.py`) — OpenAI 자동 계측은 LLM 호출만 기록; 검색 span은 직접 생성. → [03](./03-tracing-observability.md)

```python
# 03의 초기화 함수를 앱 시작 시 한 번만 호출한다. collector/network 수신은 미검증.
import os
from phoenix.otel import register
from openinference.instrumentation import TraceConfig
from openinference.instrumentation.openai import OpenAIInstrumentor

def setup_tracing():
    provider = register(endpoint=os.environ["PHOENIX_COLLECTOR_ENDPOINT"],
                        batch=True, auto_instrument=False, verbose=False)
    config = TraceConfig(hide_inputs=True, hide_outputs=True,
        hide_llm_invocation_parameters=True, hide_llm_tools=True,
        hide_embedding_vectors=True, hide_embeddings_text=True)
    OpenAIInstrumentor().instrument(tracer_provider=provider, config=config)
    return provider  # 종료 시 shutdown(); 중복 instrument 금지
```

### 4단계 — 채점기 조립 (→ [06](./06-automatic-metrics.md)/[07](./07-llm-as-a-judge.md)/[08](./08-rag-evaluation.md)/[10](./10-safety-hallucination-guardrails.md))

```python
import math
# 선행06/07/08/10 함수들을 준비해 아래 mapping에 전달한다.
# semantic=cosine_sim, judge=judge_pointwise, faithfulness=faithfulness,
# recall@5=recall_at_k, refusal_ok=refusal_correctness, leak_scan_hit=leak_scan

def score_case(case, out, scorers):
    pred, contexts, retrieved_ids = out
    row, status = {}, {}
    for name in ("semantic","judge","faithfulness","recall@5","refusal_ok","leak_scan_hit"):
        try:
            if name in ("semantic","judge"):
                if not case.get("reference"): raise ValueError("reference 미확인")
                value = scorers[name](case,pred)
                if name == "judge": value = value["score"]
            elif name == "faithfulness":
                if case["meta"]["answerable"] is False:
                    row[name],status[name] = None,"not_applicable_refusal"; continue
                value = scorers[name](pred,contexts)
            elif name == "recall@5":
                gold = case["meta"].get("gold_ids")
                if not gold: raise ValueError("gold id 미확인")
                value = scorers[name](retrieved_ids,gold,5)
            elif name == "refusal_ok": value = scorers[name](case,pred)
            else: value = float(scorers[name](pred)["leak"])  # 1=스캔 hit, 실제 누출 확정 아님
            value = float(value)
            if not math.isfinite(value): raise ValueError("비유한 점수")
            row[name],status[name] = value,"measured"
        except Exception as exc:
            row[name],status[name] = None,"unconfirmed:"+type(exc).__name__
    return {**row,"metric_status":status}
# 실패/미확인을0 또는 안전1로 대체하지 않는다. 오류 원문/기밀은 출력하지 않는다.
```

### 5단계 — 하네스 실행 & 리포트 (→ [04](./04-llm-evaluation-overview.md))

```python
def main(dataset, system_fn, scorers):
    validate_dataset(dataset)
    rows = [{"id":case["id"],"category":case["meta"]["category"],
             **score_case(case,system_fn(case["question"]),scorers)} for case in dataset]
    summary = {}
    for name in rows[0]["metric_status"]:
        values = [row[name] for row in rows if row["metric_status"][name] == "measured"]
        summary[name] = {"mean":sum(values)/len(values) if values else None,
            "n_measured":len(values), "n_total":len(rows),
            "n_unconfirmed":sum(row["metric_status"][name].startswith("unconfirmed") for row in rows),
            "n_not_applicable":sum(row["metric_status"][name].startswith("not_applicable") for row in rows)}
    return rows,summary
# 같은 적용 subset/coverage로 비교. unknown을 뺀 평균만으로 CI 통과로 보지 않음.
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
- 사고 후보의 answerable/reference/출처를 사람이 검수한 뒤 redteam 또는 production-mined 평가셋으로 승격한다.
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
- [04. 하네스](./04-llm-evaluation-overview.md)·[05. 데이터](./05-eval-dataset-construction.md)·[06. 자동지표](./06-automatic-metrics.md)·[07. Judge](./07-llm-as-a-judge.md)·[08. RAG](./08-rag-evaluation.md)·[09. Agent](./09-agent-tool-evaluation.md)·[10. 안전](./10-safety-hallucination-guardrails.md) — 각 단계의 서로 다른 역할
- [11. 온라인 평가 & 배포](./11-online-eval-deployment.md) / [12. 모니터링](./12-monitoring-drift.md) — 확장 방향
- [14. 아티팩트 계보와 거버넌스](./14-artifact-lineage-governance.md) / [15. Incident Response](./15-incident-response-postmortem.md) — 운영 준비도 보강

## 참고 자료 (References)
- 앞 문서 12편의 References 종합
- OpenAI Evals(파이프라인 사고): https://github.com/openai/evals
