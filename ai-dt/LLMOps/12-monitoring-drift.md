---
tags: [llmops, monitoring, drift, cost, latency, feedback-loop]
level: advanced
last_updated: 2026-07-06
---

# 12. 모니터링 및 드리프트

> 배포는 끝이 아니라 시작이다. production에서 품질·비용·지연을 지속 관찰하고, **입력 분포·모델 행동의 변화(drift)**를 감지해 다시 평가 루프로 돌린다.

## 왜 필요한가? (Why)

- 세상은 변한다. 사용자 질문 주제가 바뀌고(신규 공정·용어), 사내 문서가 갱신되고, 모델이 교체된다. **어제의 좋은 시스템이 오늘 조용히 나빠진다.**
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

## 어떻게 사용하는가? (How)

### trace 로그를 집계해 일별 지표 (03번 로그 위에서)

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

### 입력 드리프트 감지 (임베딩 분포 이동)
기준 기간(reference)과 최근(current) 질문 임베딩의 **중심 거리**로 주제 이동을 근사.

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
    return alerts     # 알림 → 롤백 검토 (11번)
```

### 대시보드에 두는 4개 패널
1. 품질 추이(👎·사후 judge) 2. 비용(일 토큰·버전별) 3. 지연(p50/p95·단계별) 4. 드리프트·신규 실패 수.

> 로컬엔 OpenSearch/실트래픽이 없으므로 **집계·드리프트 계산 로직까지만** 검증하고, 적재·대시보드·알림은 사내에서 붙인다. → [03](./03-tracing-observability.md)

## 관련 문서
- [03. 트레이싱 & 관측성](./03-tracing-observability.md) — 모니터링의 원천 로그
- [05. 평가 데이터셋 구축](./05-eval-dataset-construction.md) — feedback loop로 데이터셋 갱신
- [11. 온라인 평가 & 배포](./11-online-eval-deployment.md) — 이상 감지 시 롤백

## 참고 자료 (References)
- Data drift 개념: https://en.wikipedia.org/wiki/Concept_drift
- Production ML monitoring(일반): "monitor inputs, outputs, and business metrics"
