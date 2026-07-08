---
tags: [llmops, monitoring, drift, cost, latency, feedback-loop]
level: advanced
last_updated: 2026-07-06
---

# 12. 모니터링 및 드리프트

> 배포는 끝이 아니라 시작이다. production에서 품질·비용·지연을 지속 관찰하고, **입력 분포·모델 행동의 변화(drift)**를 감지해 다시 평가 루프로 돌린다.

## 왜 필요한가? (Why)

- 세상은 변한다. 사용자 질문 주제가 바뀌고(신규 공정·용어) 사내 문서가 갱신되고 모델이 교체된다. **어제의 좋은 시스템이 오늘 조용히 나빠진다.**
- 오프라인 eval set은 고정이라 **새로운 실패**를 못 잡는다. production 로그에서 이상을 캐 **eval set을 살아있게** 유지해야 한다. → [05](./05-eval-dataset-construction.md)
- 비용·지연은 방치하면 새어나간다. 품질과 **함께** 봐야 지속 가능하다.

## 핵심 개념 (What)

### 1) 무엇을 상시 모니터링하나
| 축 | 지표 |
|----|------|
| **품질(proxy)** | 👎 비율, 재질문율, 사후 샘플 채점(judge) 점수 추이 |
| **비용** | 요청당 토큰, 일 총비용, 프롬프트 버전별 비교 |
| **지연** | p50/p95 latency, 단계별(retrieve/llm) 분해 |
| **안전** | 인젝션·누출·거절 트리거 발생률 → [10](./10-safety-hallucination-guardrails.md) |
| **드리프트** | 입력 분포·출력 특성의 시간적 변화 |

### 2) 드리프트의 종류
- **입력(데이터) 드리프트**: 질문 주제·길이·언어 분포가 학습/평가 시점과 달라짐.
- **행동(모델) 드리프트**: 같은 질문에 답이 달라짐(모델 교체·미묘한 게이트웨이 변경).
- **성능 드리프트**: 위 둘의 결과로 품질 지표가 서서히 하락.

### 3) Feedback loop — 관찰을 개선으로
production에서 캔 **낮은 점수·👎·새 주제** 케이스를 라벨링해 eval set에 추가 → 다음 개선의 회귀 테스트가 된다. 이 순환이 LLMOps Level 3. → [01](./01-llmops-overview-lifecycle.md)

### 4) SLO와 알림은 품질·비용·안전을 나눠 둔다
운영 지표는 대시보드에만 있으면 늦다. 아래처럼 "언제 대응할지"를 정해둔다.

| SLO | 경고 | 사고 전환 |
|---|---|---|
| p95 latency | baseline 대비 30% 상승 | 2시간 이상 지속 또는 canary에서만 급등 |
| avg tokens/request | baseline 대비 30% 상승 | token 폭증으로 쿼터/비용 위험 |
| thumbs-down rate | baseline 대비 50% 상승 | 특정 release_id에서만 반복 |
| leak_rate | 0 초과 | 즉시 SEV-1 후보 |
| injection_resistance | 0.95 미만 | 배포 차단, red team 보강 |
| faithfulness sample | baseline 대비 0.05 하락 | SEV-2 후보, 검색/생성 분리 진단 |

## 어떻게 사용하는가? (How)

### Arize Phoenix로 production 모니터링

사내에서는 **Arize Phoenix**를 self-host하여 production 모니터링을 수행한다. Phoenix는 [03번](./03-tracing-observability.md)에서 설정한 트레이싱 데이터를 기반으로 대시보드, 평가, 드리프트 감지를 제공한다.

```python
# ── Phoenix에 production trace 전송 (03번에서 이미 설정) ─────────
import os
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://phoenix.internal:6006"

from openinference.instrumentation.openai import OpenAIInstrumentor
from phoenix.otel import register
tracer_provider = register()
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
```

```python
# ── Phoenix 클라이언트로 trace 데이터 조회 ───────────────────────
import phoenix as px

client = px.Client(endpoint="http://phoenix.internal:6006")

# 최근 trace를 DataFrame으로 가져와 분석
traces_df = client.get_spans_dataframe()

# 특정 기간의 trace만 필터링
import pandas as pd
recent = traces_df[traces_df["start_time"] > pd.Timestamp.now() - pd.Timedelta(days=1)]
print(f"최근 24시간 span 수: {len(recent)}")
print(f"평균 latency: {recent['latency_ms'].mean():.0f}ms")
```

```python
# ── Phoenix Evaluations: production 샘플에 사후 채점 결과 연결 ───
from phoenix.evals import run_evals, llm_classify
import phoenix as px

client = px.Client(endpoint="http://phoenix.internal:6006")

# 사후 채점 결과를 Phoenix에 업로드하면 trace와 연결되어 UI에서 확인 가능
from phoenix.trace import SpanEvaluations
eval_results = SpanEvaluations(
    eval_name="faithfulness",
    dataframe=scored_df,         # span_id, score, label 컬럼 포함
)
client.log_evaluations(eval_results)
# → Phoenix UI의 Evaluations 탭에서 점수 분포·추이를 시각화
```

> Phoenix UI에서 확인 가능한 모니터링 항목:
> - **Traces 탭**: 개별 요청의 워터폴 뷰, 입출력 확인, 에러 span 필터링
> - **Evaluations 탭**: 사후 채점 점수 분포, 시간별 추이 차트
> - **Embeddings 탭**: 입력 임베딩의 클러스터·드리프트 시각화 (아래 드리프트 감지 참고)

### trace 로그를 집계해 일별 지표 (03번 로그 위에서)
Phoenix 대시보드와 별개로, 커스텀 알림·SLO 판정을 위해 직접 집계하는 코드도 유지한다.

```python
import statistics
from collections import defaultdict

def daily_metrics(spans: list[dict]) -> dict:
    """spans: 03번 트레이서가 남긴 llm_call.usage 등 레코드"""
    lat = [s["latency_ms"] for s in spans if "latency_ms" in s]
    tok = [s.get("prompt_tokens",0)+s.get("completion_tokens",0) for s in spans if "prompt_tokens" in s]
    fb  = [s["user_feedback"] for s in spans if "user_feedback" in s]  # 1/0
    lat.sort()
    return {
        "n": len(spans),
        "p50_ms": lat[len(lat)//2] if lat else None,
        "p95_ms": lat[int(len(lat)*0.95)] if lat else None,
        "avg_tokens": round(statistics.mean(tok),1) if tok else None,
        "thumbs_down_rate": round(1 - statistics.mean(fb),3) if fb else None,
    }
```

### 입력 드리프트 감지

#### 방법 1: Phoenix Embeddings 탭 활용 (권장)
Phoenix는 임베딩 시각화와 드리프트 감지를 기본 제공한다. trace에 임베딩 벡터가 포함되어 있으면 **Embeddings 탭**에서 UMAP 클러스터링과 기간별 분포 비교를 바로 확인할 수 있다.

```python
# Phoenix에서 임베딩 드리프트를 프로그래밍 방식으로 조회
import phoenix as px

client = px.Client(endpoint="http://phoenix.internal:6006")

# 기간별 임베딩 데이터 추출
embeddings_df = client.get_spans_dataframe(
    filter_condition="span_kind == 'EMBEDDING'"
)
# Phoenix UI의 Embeddings 탭에서 reference vs current 기간을 선택하면
# 클러스터 이동·신규 주제 유입을 시각적으로 확인할 수 있다.
```

#### 방법 2: 커스텀 드리프트 점수 (임베딩 중심 거리)
Phoenix와 별도로, 기준 기간(reference)과 최근(current) 질문 임베딩의 **중심 거리**로 주제 이동을 근사한다. 알림 자동화에 활용.

```python
import numpy as np
def embed(texts):
    from openai import OpenAI
    c = OpenAI(base_url="http://llm-gateway.internal/v1", api_key="EMPTY")
    return np.array([d.embedding for d in c.embeddings.create(model="BGE-M3", input=texts).data])

def drift_score(ref_questions, cur_questions) -> float:
    ref_c = embed(ref_questions).mean(axis=0)
    cur_c = embed(cur_questions).mean(axis=0)
    dist = float(np.linalg.norm(ref_c - cur_c))
    return dist        # 임계 초과 시 "새 주제 유입" 알림 → eval set 갱신 트리거
```

### 새로운 실패 케이스 자동 수집 (feedback → eval set)

```python
def mine_failures(spans, judge_sample_rate=0.1):
    """👎 이거나 사후 채점 낮은 케이스를 개선 후보로 추출"""
    candidates = []
    for s in spans:
        bad_feedback = s.get("user_feedback") == 0
        low_judged = s.get("eval_score", 1.0) < 0.5      # 03번에서 샘플 채점한 값
        if bad_feedback or low_judged:
            candidates.append({"question": s.get("input"), "pred": s.get("output"),
                               "reason": "thumbs_down" if bad_feedback else "low_score"})
    return candidates
# 이 후보를 사람이 검수 → reference 라벨 → eval_set.jsonl에 추가 (05번)
```

### 카나리/버전별 이상 감지 (간단한 규칙 알림)

```python
def alert(cur: dict, base: dict) -> list[str]:
    alerts = []
    if cur["thumbs_down_rate"] and cur["thumbs_down_rate"] > base["thumbs_down_rate"] * 1.5:
        alerts.append("👎 비율 급증")
    if cur["p95_ms"] and cur["p95_ms"] > base["p95_ms"] * 1.3:
        alerts.append("p95 지연 악화")
    if cur["avg_tokens"] and cur["avg_tokens"] > base["avg_tokens"] * 1.3:
        alerts.append("토큰/비용 급증")
    if cur.get("leak_rate", 0) > 0:
        alerts.append("SEV-1 후보: 누출 감지")
    if cur.get("faithfulness", 1.0) < base.get("faithfulness", 1.0) - 0.05:
        alerts.append("SEV-2 후보: faithfulness 하락")
    return alerts     # 알림 → 롤백 검토 (11번)
```

### 대시보드 구성 — Phoenix + 커스텀 알림 병행

**Phoenix UI가 기본 대시보드 역할**을 하고, SLO 알림과 커스텀 집계는 위 코드로 보완한다.

| 패널 | Phoenix 제공 | 커스텀 보완 |
|------|-------------|------------|
| 품질 추이(👎·사후 judge) | Evaluations 탭 — 점수 분포·시간 추이 | `mine_failures`로 개선 후보 추출 |
| 비용(일 토큰·버전별) | Traces 탭 — 토큰 사용량 집계 | `daily_metrics`로 SLO 초과 알림 |
| 지연(p50/p95·단계별) | Traces 워터폴 — 단계별 latency 분해 | `alert`로 p95 임계 알림 |
| 드리프트·신규 실패 수 | Embeddings 탭 — 클러스터 이동 시각화 | `drift_score`로 수치 알림 |
| release_id별 안전 트리거 | Traces 필터 — release_id별 조회 | `alert`로 SEV-1/SEV-2 판정 |

> 로컬에서는 `px.launch_app()`으로 Phoenix를 띄워 위 패널을 바로 확인할 수 있다. production에서는 사내 Phoenix 서버에 집계하고, 알림은 커스텀 스크립트(cron)로 발송한다. → [03](./03-tracing-observability.md)

## 관련 문서
- [03. 트레이싱 & 관측성](./03-tracing-observability.md) — 모니터링의 원천 로그
- [05. 평가 데이터셋 구축](./05-eval-dataset-construction.md) — feedback loop로 데이터셋 갱신
- [11. 온라인 평가 & 배포](./11-online-eval-deployment.md) — 이상 감지 시 롤백
- [14. 아티팩트 계보와 거버넌스](./14-artifact-lineage-governance.md) — release_id별 지표 분리
- [15. Incident Response](./15-incident-response-postmortem.md) — 알림을 사고 대응으로 전환하는 기준

## 참고 자료 (References)
- Arize Phoenix(사내 사용 LLM 모니터링): https://docs.arize.com/phoenix
- Phoenix Evaluations 가이드: https://docs.arize.com/phoenix/evaluation
- Data drift 개념: https://en.wikipedia.org/wiki/Concept_drift
- Production ML monitoring(일반): "monitor inputs, outputs, and business metrics"
