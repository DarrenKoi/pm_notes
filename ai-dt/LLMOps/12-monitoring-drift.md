---
tags: [llmops, monitoring, drift, cost, latency, feedback-loop]
level: advanced
last_updated: 2026-07-06
reviewed_on: 2026-10-04
review_status: partial
document_type: learning
---

> [!info] 검토 범위 — 2026-10-04
> [공통 적용 조건](./verified-conditions.md)과 [정리 기록](./organization-log.md)에 판본·일차 근거·로컬 검증을 기록했다. 실제 회사 운영·모델/서버 품질·조직 승인·Claude 협의·읽기 화면은 미확인이다.


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
production에서 캔 **낮은 점수·👎·새 주제** 케이스를 라벨링해 eval set에 추가 → 다음 개선의 회귀 테스트가 된다. Level 3는 이 노트의 학습용 분류다. 검수와 대표성 확인 없이 로그를 정답셋으로 승격하지 않는다. → [01](./01-llmops-overview-lifecycle.md)

### 4) SLO와 알림은 품질·비용·안전을 나눠 둔다
운영 지표는 대시보드에만 있으면 늦다. 아래 숫자·SEV 전환은 제안값이며 실제 조직 SLO/승인 기준은 미확인이다. 표본 수·coverage·기간·release를 함께 본다.

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

Phoenix 자체 호스팅은 후보 구성이다. 실제 회사 채택·collector 수신·server/UI는 미확인이다. [03](./03-tracing-observability.md)의 계측과 client/server 설치는 별도이며, raw 입력·embedding 숨김을 유지한다.

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

```python
import os
from datetime import datetime, timezone, timedelta
import pandas as pd
from phoenix.client import Client

phoenix_client = Client(base_url=os.environ["PHOENIX_ENDPOINT"],
                        api_key=os.environ["PHOENIX_API_KEY"])

def recent_spans(client=phoenix_client):
    end = datetime.now(timezone.utc)
    frame = client.spans.get_spans_dataframe(start_time=end-timedelta(days=1),
        end_time=end, project_identifier=os.environ["PHOENIX_PROJECT"], limit=1000)
    if frame.empty:
        return frame, None  # 미수집을 0ms로 보지 않음
    start = pd.to_datetime(frame["start_time"], utc=True, errors="coerce")
    finish = pd.to_datetime(frame["end_time"], utc=True, errors="coerce")
    latency = (finish-start).dt.total_seconds()*1000
    if latency.isna().any() or (latency < 0).any():
        raise ValueError("span 시각 계약 확인 필요")
    return frame.assign(latency_ms=latency), float(latency.mean())
# 건수는 span 수다. 요청 집계와 다르며 limit 도달은 전체 하루 자료의 증거가 아님.
```

```python
import numpy as np

def upload_scores(scored_df, client=phoenix_client):
    # 실제로 조회한 span_id와 검수한 score가 있어야 한다. 원문 설명은 보내지 않는다.
    required = {"span_id", "score"}
    if scored_df.empty or not required <= set(scored_df.columns):
        raise ValueError("span id/score 자료 필요")
    if not scored_df["span_id"].map(lambda value: isinstance(value,str) and bool(value.strip())).all() or scored_df["span_id"].duplicated().any():
        raise ValueError("span id 누락/중복")
    if not scored_df["score"].map(lambda value: isinstance(value,(int,float,np.number)) and not isinstance(value,(bool,np.bool_))).all():
        raise ValueError("숫자 score 필요")
    values = pd.to_numeric(scored_df["score"], errors="coerce")
    if not np.isfinite(values).all() or not values.between(0,1).all():
        raise ValueError("faithfulness 점수 미확인")
    scores = scored_df[["span_id", "score"]].copy()
    scores["score"] = values.astype(float)
    return client.spans.log_span_annotations_dataframe(
        dataframe=scores, annotation_name="faithfulness",
        annotator_kind="LLM", sync=True)
# 서버 쓰기: 승인된 테스트 프로젝트/권한으로만 호출. 로컬 검증은 HTTP MockTransport만 사용.
```

> 워터폴·token/latency·사후 평가 조회는 관측 요구사항이다. 실제 화면명·집계 필드·권한·저장 성공은 해당 server 판본에서 확인해야 한다. SDK 모의 HTTP 검증이 UI/운영 검증을 대신하지 않는다.

### trace 로그를 집계해 일별 지표 (03번 로그 위에서)
Phoenix 대시보드와 별개로, 커스텀 알림/SLO 판정을 위해 요청별 정규화 레코드를 집계한다. 03의 usage/timer 로그를 그대로 각각 요청으로 세지 않는다. root 요청 지연·요청별 전체 LLM token 합·피드백을 request_id로 조인하고 release/기간을 먼저 분리한 자료가 필요하다. 이 조인/운영 파이프라인은 미구현이며 아래 함수는 입력 계약에 맞는 집계만 수행한다.

```python
import math, statistics
import numpy as np

def finite_nonnegative(value):
    if value is None:
        return None
    if type(value) not in (int,float) or not math.isfinite(value) or value < 0:
        raise ValueError("비유한/음수 관측치")
    return value

def daily_metrics(spans: list[dict]) -> dict:
    """spans는 raw span이 아닌 요청당1개 정규화 레코드. 미수집 None 보존."""
    ids = [s["request_id"] for s in spans]
    if any(not isinstance(i,str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("요청 id 누락/중복: 먼저 조인해야 함")
    latency, tokens, feedback = [], [], []
    for record in spans:
        value = finite_nonnegative(record.get("latency_ms"))
        if value is not None: latency.append(value)
        counts = (record.get("prompt_tokens"),record.get("completion_tokens"))
        if any(v is not None and (type(v) is not int or v < 0) for v in counts):
            raise ValueError("token 정수 계약 오류")
        if all(v is not None for v in counts):
            tokens.append(sum(counts))
        fb = record.get("user_feedback")
        if fb is not None:
            if type(fb) is not int or fb not in (0,1): raise ValueError("feedback 계약 오류")
            feedback.append(fb)
    return {"n_requests":len(spans), "n_latency":len(latency),
        "n_token_usage":len(tokens), "n_feedback":len(feedback),
        "p50_ms":float(np.percentile(latency,50,method="linear")) if latency else None,
        "p95_ms":float(np.percentile(latency,95,method="linear")) if latency else None,
        "avg_tokens":statistics.mean(tokens) if tokens else None,
        "thumbs_down_rate":1-statistics.mean(feedback) if feedback else None}
# 관측된 부분의 평균이다. 분모/미수집 수 없이 전체 요청 성능으로 해석하지 않음.
```

### 입력 드리프트 감지

#### 방법 1: Phoenix Embeddings 탭 활용 (조건부)
기존 노트의 “기본 Embeddings 탭에서 즉시 UMAP/드리프트” 주장은 현재 server/UI에서 확인하지 못했다. 다음은 embedding 종류 span을 조회하는 예제이며 벡터 시각화/주제 판정을 수행하지 않는다. 03의 privacy 설정으로 raw embedding은 수집되지 않는다. 원문/벡터 수집을 켜는 변경은 별도 권한·정보 정책 검토가 필요하다.

```python
from phoenix.client.types.spans import SpanQuery  # client3.5.0의 실제 import 경로

def embedding_spans(client=phoenix_client):
    query = SpanQuery().where("span_kind == 'EMBEDDING'")
    return client.spans.get_spans_dataframe(query=query,
        project_identifier=os.environ["PHOENIX_PROJECT"], limit=1000)
# embedding span 조회이며 vector 추출/UMAP/드리프트 검정은 아님.
```

#### 방법 2: 커스텀 드리프트 점수 (임베딩 중심 거리)
Phoenix와 별도로, 기준 기간(reference)과 최근(current) 질문 임베딩의 **중심 거리**로 주제 이동을 근사한다. 알림 자동화에 활용.

```python
# README의 검증된 embed/EMBED_MODEL 사용. 이름을 재정의하지 않는다.
def drift_score(ref_questions, cur_questions) -> float:
    reference = np.asarray(embed(ref_questions, model=EMBED_MODEL),dtype=float)
    current = np.asarray(embed(cur_questions, model=EMBED_MODEL),dtype=float)
    if reference.shape[1] != current.shape[1]:
        raise ValueError("embedding 모델/차원 계약 불일치")
    distance = float(np.linalg.norm(reference.mean(axis=0)-current.mean(axis=0)))
    if not np.isfinite(distance): raise ValueError("중심 거리 미확인")
    return distance
# 같은 모델/revision/전처리로 비교. 중심이 같은 다른 분포는 거리0이며 신규 주제 탐지 보장 아님.
```

### 새로운 실패 케이스 자동 수집 (feedback → eval set)

```python
def mine_failures(spans):
    """명시된 실패 또는 채점 미확인 후보. raw input/output을 자동 복사하지 않음."""
    candidates = []
    for record in spans:
        score = record.get("eval_score")
        if score is not None and (type(score) not in (int,float) or not math.isfinite(score)):
            raise ValueError("eval_score 계약 오류")
        reason = ("thumbs_down" if record.get("user_feedback") == 0 else
                  "score_unconfirmed" if score is None else "low_score" if score < 0.5 else None)
        if reason:
            candidates.append({"request_id":record.get("request_id"),
                "data_ref":record.get("data_ref"), "reason":reason, "review_status":"unreviewed"})
    return candidates
# 검수 전 golden/reference가 아니다. 출처 pointer의 승인된 마스킹 자료를 확인.
```

### 카나리/버전별 이상 감지 (간단한 규칙 알림)

```python
def alert(cur: dict, base: dict) -> dict:
    messages, unknown = [], []
    for key, factor, message in [("thumbs_down_rate",1.5,"👎 비율 급증"),
            ("p95_ms",1.3,"p95 지연 악화"),("avg_tokens",1.3,"토큰 증가")]:
        now, old = finite_nonnegative(cur.get(key)), finite_nonnegative(base.get(key))
        if now is None or old is None or old == 0:
            unknown.append(key)  # zero baseline의 상대 증가율도 미정의
        elif now > old*factor: messages.append(message)
    leak = finite_nonnegative(cur.get("leak_rate"))
    if leak is None: unknown.append("leak_rate")
    elif leak > 0: messages.append("SEV-1 후보: 누출 감지")
    now, old = cur.get("faithfulness"), base.get("faithfulness")
    if now is None or old is None: unknown.append("faithfulness")
    elif finite_nonnegative(now) < finite_nonnegative(old)-0.05:
        messages.append("SEV-2 후보: faithfulness 하락")
    return {"alerts":messages, "unknown":unknown}
# 제안 임계값. 알림 전송/rollback/조직 SEV 확정은 수행하지 않음.
```

### 대시보드 구성 — Phoenix + 커스텀 알림 병행

다음은 대시보드 요구사항이다. 실제 화면명/설정/데이터 수집은 미확인이며 직접 집계의 입력 계약과 coverage를 대조해야 한다.

| 패널 | 확인할 Phoenix 구성 | 커스텀 보완 |
|------|-------------|------------|
| 품질 추이(👎·사후 judge) | Evaluations 탭 — 점수 분포·시간 추이 | `mine_failures`로 개선 후보 추출 |
| 비용(일 토큰·버전별) | Traces 탭 — 토큰 사용량 집계 | `daily_metrics`로 SLO 초과 알림 |
| 지연(p50/p95·단계별) | Traces 워터폴 — 단계별 latency 분해 | `alert`로 p95 임계 알림 |
| 드리프트·신규 실패 수 | 별도 embedding 화면/벡터 수집 필요 여부 미확인 | `drift_score`로 수치 알림 |
| release_id별 안전 트리거 | Traces 필터 — release_id별 조회 | `alert`로 미확인/SEV 후보 구분 |

> server/UI·수집/저장·예약 알림 구성은 별도 실행이 필요하다. 이번 검증에서는 원격 전송/서버 시작/알림 발송을 수행하지 않았다. → [03](./03-tracing-observability.md)

## 관련 문서
- [03. 트레이싱 & 관측성](./03-tracing-observability.md) — 모니터링의 원천 로그
- [05. 평가 데이터셋 구축](./05-eval-dataset-construction.md) — feedback loop로 데이터셋 갱신
- [11. 온라인 평가 & 배포](./11-online-eval-deployment.md) — 이상 감지 시 롤백
- [14. 아티팩트 계보와 거버넌스](./14-artifact-lineage-governance.md) — release_id별 지표 분리
- [15. Incident Response](./15-incident-response-postmortem.md) — 알림을 사고 대응으로 전환하는 기준

## 참고 자료 (References)
- Arize Phoenix(자체 호스팅 후보, 사내 사용 미확인): https://docs.arize.com/phoenix
- Phoenix Evaluations 가이드: https://docs.arize.com/phoenix/evaluation
- 중심 거리는 작성자의 단순 proxy다. 실제 통계적 분포 검정/성능 저하 판정과 구분한다.
- [NumPy percentile 공식 API](https://numpy.org/doc/stable/reference/generated/numpy.percentile.html) — 2026-10-04, linear 분위수 정의.
- Production ML monitoring(일반): "monitor inputs, outputs, and business metrics"
