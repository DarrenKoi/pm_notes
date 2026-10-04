---
tags: [evaluation, online, ab-test, canary, ci, regression-gate]
level: advanced
last_updated: 2026-07-06
reviewed_on: 2026-10-04
review_status: partial
document_type: learning
---

> [!info] 검토 범위 — 2026-10-04
> [공통 적용 조건](./verified-conditions.md)과 [정리 기록](./organization-log.md)에 판본·출처·로컬 검증을 남겼다. 실제 judge 품질·사내 접속·운영 승인·Claude 협의·Obsidian 읽기 화면은 미확인이다.


# 11. 온라인 평가 및 배포

> 오프라인 점수가 좋아도 실사용에서 나빠질 수 있다. 배포 전 **CI regression gate**로 회귀를 막고 배포 시 **canary·A/B**로 실제 트래픽에서 검증한다.

## 왜 필요한가? (Why)

- 오프라인 eval set은 과거의 스냅샷이다. 실제 사용자는 **데이터셋에 없는 질문**을 하고 진짜 만족도는 트래픽에서만 드러난다.
- 프롬프트·모델·검색을 바꿀 때마다 손으로 확인하면 반드시 회귀가 샌다. **자동 게이트**가 배포의 안전벨트다.
- 전면 배포는 위험하다. **일부 트래픽(canary)**에 먼저 태워 지표를 보고 확대/롤백하는 게 표준.

## 핵심 개념 (What)

### 1) 오프라인 → 온라인 연결
| 단계 | 장소 | 판단 |
|------|------|------|
| **Regression gate** | CI(배포 전) | 고정 eval set 점수 ≥ baseline, 안전 게이트 통과 |
| **Canary** | 배포 초기 | 소량 트래픽에서 지표 이상 없나 |
| **A/B** | 운영 | 신·구 버전을 나눠 실사용 지표 비교 |
| **Full rollout / rollback** | 운영 | 좋으면 확대, 나쁘면 되돌림 |

### 2) CI Regression Gate
PR마다 `eval_set.jsonl`을 돌려 **핵심 지표가 승인된 허용 하락폭을 넘으면 실패 상태를 반환**. 안전 지표([10](./10-safety-hallucination-guardrails.md))는 하드 컷(무조건 통과 요구). 실제 머지 차단에는 CI 실행과 필수 status check/보호 규칙 설정이 필요하다. Level 2는 이 노트의 학습용 분류다. → [01](./01-llmops-overview-lifecycle.md)

배포 단위는 코드 커밋이 아니라 [14](./14-artifact-lineage-governance.md)의 **release manifest**다. gate는 `prompt/model/index/tool/eval/rubric/guardrail` 버전 조합을 입력으로 받아야 한다.

### 3) 온라인 지표 (proxy)
실시간 정답은 없으므로 **간접 신호**를 본다: 사용자 피드백(👍/👎), 재질문·이탈률, 응답 채택률, latency/cost, 안전 트리거 발생률. → [12](./12-monitoring-drift.md)

### 4) A/B의 통계
버전 간 차이가 **우연이 아닌지** 검정한다(표본 수, 유의수준). 소량 표본에서 "좋아 보임"에 속지 않기.

## 어떻게 사용하는가? (How)

### CI Regression Gate (배포 전 자동 채점)

```python
import json, math
THRESH = {"correctness": -0.02, "faithfulness": -0.02}  # 제안 허용 하락폭

def gate(current: dict, baseline: dict, manifest: dict) -> bool:
    if not manifest.get("release_id") or not all(manifest.get("artifacts", {}).get(k)
        for k in ("prompt", "model", "index", "tool", "eval", "rubric", "guardrail")):
        raise ValueError("release 아티팩트 조합 미확인")
    if current.get("safety_pass") is not True:
        return False
    for key, min_delta in THRESH.items():
        for report in (current, baseline):
            value = report.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{key} 점수 미확인/계약 오류")
        delta = current[key] - baseline[key]
        if delta < min_delta and not math.isclose(delta, min_delta, rel_tol=0, abs_tol=1e-12):
            return False  # 경계의 부동소수점 오차만 허용
    return True
# 같은 eval/rubric/judge 판본·coverage로 baseline/current를 계산해야 비교 가능.
# 이 함수는 품질 회귀 gate다. immutable snapshot/서명된 승인·운영 배포 권한 검증은 별도.
# baseline/manifest JSON 파일은 context manager로 읽고 system_fn·scorers·데이터를 준비한다.
# run_eval은 04, safety_gate는 10에서 가져온다. False/None/예외는 실패 종료로 연결한다.
# 아래 workflow는 개념 fragment이며 이 문서만으로 실행 CI가 완성되지는 않는다.
```

```yaml
# .github/workflows/eval.yml (개념 — 사내 CI에 맞게)
jobs:
  llm-eval:
    steps:
      - run: python ci_eval.py     # 실제 스크립트의 실패 종료와 required status check 설정 필요
```

### Canary 라우팅 (트래픽 일부만 신버전)

```python
import hashlib
def route(user_id: str, canary_pct: int = 5) -> str:
    if not isinstance(user_id, str) or not user_id or type(canary_pct) is not int or not 0 <= canary_pct <= 100:
        raise ValueError("사용자 id/0~100 비율 필요")
    # 사용자 단위로 안정적 분배(같은 유저는 항상 같은 버전 → 경험 일관성)
    bucket = int(hashlib.sha256(user_id.encode()).hexdigest(), 16) % 100
    return "v_new" if bucket < canary_pct else "v_stable"
```

### A/B 지표 수집 & 검정

```python
from math import sqrt, erfc

def ab_compare(fb_a: list[int], fb_b: list[int]) -> dict:
    """독립 사용자 단위의 사전 정의된 binary 지표. 반복 응답은 독립 표본이 아님."""
    if not fb_a or not fb_b or any(type(v) is not int or v not in (0,1) for v in fb_a+fb_b):
        raise ValueError("각 그룹에 binary 관측치 필요")
    na, nb = len(fb_a), len(fb_b)
    pa, pb = sum(fb_a)/na, sum(fb_b)/nb
    pooled = (sum(fb_a)+sum(fb_b))/(na+nb)
    result = {"p_stable": pa, "p_new": pb, "delta": pb-pa,
              "n_stable": na, "n_new": nb, "z": None, "p_value": None, "significant": None}
    # 근사 적합성의 시작 검사. 독립성/랜덤화/편향·검정력까지 보장하지 않음.
    if min(na*pooled, na*(1-pooled), nb*pooled, nb*(1-pooled)) < 5:
        return {**result, "reason": "normal_approximation_unconfirmed"}
    se = sqrt(pooled*(1-pooled)*(1/na+1/nb))
    z = (pb-pa)/se
    p_value = erfc(abs(z)/sqrt(2))
    return {**result, "z": z, "p_value": p_value, "significant": p_value < 0.05}
# 양측 alpha=0.05 차이 검정. 유의하지 않음은 동등/비열등·안전의 증거가 아니다.
```

### 배포 결정 규칙
- canary 확대에는 사전 정의한 비열등 허용폭·표본/검정력·안전/지연 기준이 필요하다. 차이 검정의 p≥0.05만으로 “나쁘지 않음”을 입증하거나 확대하지 않는다.
- 안전 트리거 급증 or 👎 유의 상승 → **즉시 롤백**(승인된 전체 release 조합 복원; 프롬프트 변경만이 원인이 아닐 수 있음. 프롬프트 관리 — [02](./02-prompt-management-versioning.md)).
- SEV-1/SEV-2 사고가 의심되면 canary 확대가 아니라 [15. Incident Response](./15-incident-response-postmortem.md)로 전환.

> 로컬엔 실트래픽이 없으므로 **regression gate 로직까지만** 검증하고 canary/A/B는 사내 서빙에서 붙인다.

## 관련 문서
- [01. LLMOps 개요](./01-llmops-overview-lifecycle.md) — 오프라인/온라인 평가의 큰 그림
- [10. 안전성·가드레일 평가](./10-safety-hallucination-guardrails.md) — CI 하드 컷 대상
- [12. 모니터링 & 드리프트](./12-monitoring-drift.md) — 배포 후 지속 관찰
- [14. 아티팩트 계보와 거버넌스](./14-artifact-lineage-governance.md) — release manifest와 승인 기준
- [15. Incident Response](./15-incident-response-postmortem.md) — 이상 감지 후 롤백·postmortem 절차

## 참고 자료 (References)
- Canary release(개념): https://martinfowler.com/bliki/CanaryRelease.html
- [NIST 두 비율 검정](https://www.itl.nist.gov/div898/handbook/prc/section3/prc33.htm) — 2026-10-04, 독립 표본·정규 근사 조건.
- [GitHub protected branches](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches) — required status check와 bypass 조건; 실제 저장소 정책 미확인.
