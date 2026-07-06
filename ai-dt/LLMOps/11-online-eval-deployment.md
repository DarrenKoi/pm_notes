---
tags: [evaluation, online, ab-test, canary, ci, regression-gate]
level: advanced
last_updated: 2026-07-06
---

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
PR마다 `eval_set.jsonl`을 돌려 **핵심 지표가 baseline 이하로 떨어지면 머지 차단**. 안전 지표([10](./10-safety-hallucination-guardrails.md))는 하드 컷(무조건 통과 요구). 이게 LLMOps Level 2의 핵심. → [01](./01-llmops-overview-lifecycle.md)

배포 단위는 코드 커밋이 아니라 [14](./14-artifact-lineage-governance.md)의 **release manifest**다. gate는 `prompt/model/index/tool/eval/rubric/guardrail` 버전 조합을 입력으로 받아야 한다.

### 3) 온라인 지표 (proxy)
실시간 정답은 없으므로 **간접 신호**를 본다: 사용자 피드백(👍/👎), 재질문·이탈률, 응답 채택률, latency/cost, 안전 트리거 발생률. → [12](./12-monitoring-drift.md)

### 4) A/B의 통계
버전 간 차이가 **우연이 아닌지** 검정한다(표본 수, 유의수준). 소량 표본에서 "좋아 보임"에 속지 않기.

## 어떻게 사용하는가? (How)

### CI Regression Gate (배포 전 자동 채점)

```python
# ci_eval.py — CI에서 실행, 실패하면 exit(1)로 머지 차단
import json, sys, statistics

BASELINE = json.load(open("baseline_scores.json"))   # 직전 승인 버전 점수
MANIFEST = json.load(open("release_manifest.json"))  # 14번 문서의 manifest
THRESH = {"correctness": -0.02, "faithfulness": -0.02}  # 허용 하락폭

def gate(current: dict) -> bool:
    ok = True
    for k, min_delta in THRESH.items():
        delta = current[k] - BASELINE[k]
        status = "OK" if delta >= min_delta else "REGRESSION"
        print(f"{k}: {current[k]:.3f} (Δ{delta:+.3f}) {status}")
        ok &= delta >= min_delta
    # 안전은 하드 컷 (10번 safety_gate)
    ok &= current["safety_pass"]
    print("release:", MANIFEST["release_id"], "prompt:", MANIFEST["artifacts"]["prompt"])
    return ok

if __name__ == "__main__":
    rows = run_eval(system_v_new, scorers, load_set())   # 04번 하네스
    current = {k: statistics.mean(r[k] for r in rows) for k in scorers}
    current["safety_pass"] = safety_gate(safety_report())
    sys.exit(0 if gate(current) else 1)
```

```yaml
# .github/workflows/eval.yml (개념 — 사내 CI에 맞게)
jobs:
  llm-eval:
    steps:
      - run: python ci_eval.py     # REGRESSION이면 비정상 종료 → 머지 차단
```

### Canary 라우팅 (트래픽 일부만 신버전)

```python
import hashlib
def route(user_id: str, canary_pct: int = 5) -> str:
    # 사용자 단위로 안정적 분배(같은 유저는 항상 같은 버전 → 경험 일관성)
    bucket = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 100
    return "v_new" if bucket < canary_pct else "v_stable"
```

### A/B 지표 수집 & 검정

```python
from math import sqrt
def ab_compare(fb_a: list[int], fb_b: list[int]) -> dict:
    """fb: 각 응답의 👍=1/👎=0 리스트"""
    pa, pb = sum(fb_a)/len(fb_a), sum(fb_b)/len(fb_b)
    # 두 비율 차이의 z-검정(근사)
    p = (sum(fb_a)+sum(fb_b))/(len(fb_a)+len(fb_b))
    se = sqrt(p*(1-p)*(1/len(fb_a)+1/len(fb_b))) or 1e-9
    z = (pb - pa) / se
    return {"p_stable": round(pa,3), "p_new": round(pb,3), "z": round(z,2),
            "significant": abs(z) > 1.96}   # 95% 신뢰
```

### 배포 결정 규칙
- canary 지표(피드백·안전·latency)가 stable 대비 **유의하게 나쁘지 않으면** 확대.
- 안전 트리거 급증 or 👎 유의 상승 → **즉시 롤백**(프롬프트 버전 되돌리기 — [02](./02-prompt-management-versioning.md)).
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
- A/B 검정 기초: 두 비율 z-검정
