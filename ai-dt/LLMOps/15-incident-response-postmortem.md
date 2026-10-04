---
tags: [llmops, incident-response, postmortem, rollback, feedback-loop]
level: advanced
last_updated: 2026-07-06
reviewed_on: 2026-10-04
review_status: partial
document_type: learning
---

> [!info] 검토 범위 — 2026-10-04
> [공통 적용 조건](./verified-conditions.md)과 [정리 기록](./organization-log.md)에 판본·일차 근거·로컬 검증을 기록했다. 실제 회사 운영·모델/서버 품질·조직 승인·Claude 협의·읽기 화면은 미확인이다.


# 15. Incident Response와 Postmortem

> LLM 시스템의 장애는 서버 다운만이 아니다. **틀린 답, 근거 없는 답, 기밀 누출, 과도한 비용, 도구 오작동**도 모두 운영 사고다. 사고를 eval set과 가드레일로 되돌리는 절차를 만든다.

## 왜 필요한가? (Why)

- LLM 품질 사고는 조용히 발생한다. 예외가 터지지 않아도 사용자는 틀린 답을 받고 모델은 계속 200 OK를 반환한다.
- "나쁜 답 하나"를 고치는 것보다 중요한 일은 그 실패를 **재현 가능한 평가 케이스**로 바꾸는 것이다. 그래야 같은 사고가 다음 배포에서 다시 잡힌다.
- Agent/Tool 시스템은 읽기 도구와 쓰기 도구가 섞이므로, 사고 대응에 권한 차단·도구 비활성화·인덱스 롤백이 포함돼야 한다.

## 핵심 개념 (What)

### 1) LLM 사고 유형

| 유형 | 예시 | 우선 조치 |
|---|---|---|
| Quality incident | 정답 문서가 있는데 엉뚱한 답 | trace 확보, 검색/생성 원인 분리 |
| Hallucination | 문서에 없는 조건을 확신 있게 생성 | faithfulness gate 강화, 거절 케이스 추가 |
| Safety incident | PII/기밀/시스템 프롬프트 노출 | 응답 차단, 로그 보존, 보안 보고 |
| Tool incident | 잘못된 도구 선택·인자·쓰기 실행 | 도구 disable, 권한/스키마 점검 |
| Cost/latency incident | 토큰 폭증, p95 급등 | prompt/index/streaming 변경점 확인 |
| Evaluation incident | judge rubric 변경으로 점수 왜곡 | judge-human agreement 재측정 |

### 2) 심각도 기준

다음 SEV 분류/30분 대응/3회 편입 기준은 제안 운영 규칙이며 조직 승인·실제 대응 실적은 미확인이다.

| 등급 | 기준 | 예 |
|---|---|---|
| SEV-1 | 기밀·개인정보 유출, 쓰기 도구 오작동, 안전 정책 중대 위반 | 내부 문서 원문 노출, 잘못된 레코드 생성 |
| SEV-2 | 넓은 범위의 품질 회귀·환각, canary 중 안전 트리거 급증 | 특정 공정 카테고리 정답률 급락 |
| SEV-3 | 일부 케이스의 품질 저하, 비용·지연 악화 | p95 30% 상승, thumbs-down 증가 |
| SEV-4 | 문서/대시보드/평가셋 불일치 | manifest 누락, baseline 이름 오류 |

### 3) 사고 대응의 원칙

- 먼저 **확산을 멈춘다**: canary 중지, prompt rollback, tool disable, index rollback.
- 다음 **증거를 보존한다**: trace_id, release_id, prompt/model/index/tool/eval 버전, 입력/출력 마스킹본.
- 마지막으로 **eval에 편입한다**: 사고 케이스를 `production-mined` → 사람 검수 → golden/redteam set으로 승격한다.

## 어떻게 사용하는가? (How)

### Incident record 표준 포맷

아래2026-07-06 기록은 가상 학습 시나리오이며 실제 회사 사고 이력이 아니다. 영향/기밀 노출 없음/원인 후보를 확인된 사실로 읽지 않는다.

```yaml
incident_id: llm-2026-07-06-001
severity: SEV-2
detected_at: 2026-07-06T10:30:00+09:00
detected_by: monitoring
release_id: rag-photo-2026-07-06-01
trace_ids: [abc123def456]
symptom: "photo 카테고리 faithfulness 샘플 점수 급락"
impact:
  users: "canary 5% traffic"
  data: "기밀 노출 없음"
immediate_action:
  - "canary stopped"
  - "prompt rolled back to recipe_qa_system:v3"
root_cause_candidate:
  - "v4 prompt removed explicit citation requirement"
follow_up_eval_cases:
  - "prod_mined_20260706_001"
owner: process-ai-team
status: open
```

### 30분 대응 루프

1. **Detect**: [12](./12-monitoring-drift.md)의 알림, 사용자 신고, red team 실패.
2. **Classify**: SEV 등급과 사고 유형을 정한다.
3. **Contain**: canary 중지, feature flag off, tool disable, prompt/index rollback.
4. **Preserve**: trace와 manifest를 보존한다. 원문 로그는 민감정보 마스킹본으로 공유한다.
5. **Diagnose**: [08](./08-rag-evaluation.md) 기준으로 검색 실패인지 생성 실패인지 분리한다. Agent면 [09](./09-agent-tool-evaluation.md)의 trajectory를 본다.
6. **Patch**: 프롬프트·검색·가드레일·도구 스키마 중 최소 변경으로 수정한다.
7. **Regression**: 사고 케이스를 포함해 [11](./11-online-eval-deployment.md) gate를 다시 실행한다.

### 사고 케이스를 eval set으로 승격

```python
def incident_to_eval_case(incident, trace):
    ids = incident.get("follow_up_eval_cases")
    if not isinstance(ids,list) or not ids or not isinstance(ids[0],str) or not ids[0]:
        raise ValueError("유일 후보 id 준비 필요")
    # trace는 승인된 마스킹 자료/참조. 원래 raw input/output을 자동 복사하지 않음.
    return {"id":ids[0], "question":trace.get("sanitized_input"),
        "reference":None, "contexts":trace.get("sanitized_retrieved_contexts"),
        "meta":{"category":trace.get("category","unknown"),
            "source":f"incident:{incident['incident_id']}","data_ref":trace.get("data_ref"),
            "answerable":None,"review_status":"unreviewed","dataset_tier":"candidate",
            "severity":incident["severity"],"release_id":incident["release_id"]}}
# 후보는eval_set이 아니다. question/contexts/answerable/reference/유일id·권한/출처를검수해승격한다.
```

승격 규칙(제안):

- SEV-1/SEV-2는 검수 후 `golden` 또는 `redteam`에 반드시 편입한다.
- SEV-3은 같은 유형이 3회 이상 반복되면 편입한다.
- 단순 사용자 취향 문제는 `feedback-mined`에 두고 대표성이 확인될 때만 golden으로 승격한다.

### Postmortem 템플릿

```markdown
# Postmortem: llm-2026-07-06-001

## Summary
- 무엇이 잘못됐는가:
- 영향 범위:
- 현재 상태:

## Timeline
| 시간 | 이벤트 |
|---|---|

## Root Cause
- 검색:
- 생성:
- 가드레일:
- 배포/거버넌스:

## What Caught It / What Missed It
- 잡은 신호:
- 놓친 게이트:

## Actions
| 액션 | 소유자 | 기한 | 검증 |
|---|---|---|---|
| 사고 케이스를 redteam set에 추가 | app | D+1 | CI에서 재현 |
| prompt v4 citation rule 복구 | app | D+1 | faithfulness >= baseline |
| alert threshold 조정 | platform | D+3 | staging replay |
```

### 종료 조건

- [ ] 사용자 영향이 멈췄다(canary 중지/롤백/패치).
- [ ] 사고 trace와 release manifest가 보존됐다.
- [ ] 재현 케이스가 eval/redteam set에 들어갔다.
- [ ] 같은 실패를 막는 regression gate가 추가됐다.
- [ ] postmortem 액션에 소유자와 기한이 있다.

## 관련 문서

- [03. 트레이싱 & 관측성](./03-tracing-observability.md) — 사고 증거의 원천
- [10. 안전성·환각·가드레일 평가](./10-safety-hallucination-guardrails.md) — 안전 사고 분류와 red team
- [11. 온라인 평가 및 배포](./11-online-eval-deployment.md) — rollback/canary control
- [12. 모니터링 및 드리프트](./12-monitoring-drift.md) — 사고 탐지와 feedback loop
- [14. 아티팩트 계보와 거버넌스](./14-artifact-lineage-governance.md) — release manifest와 승인 기준

## 참고 자료 (References)

- OWASP Top 10 for LLM Applications 2025: https://genai.owasp.org/llm-top-10/
- [Google SRE postmortem 원문](https://sre.google/sre-book/postmortem-culture/) — 2026-10-04, 비난보다 원인/재발 방지 액션과 검증 기록.
- NIST AI RMF Generative AI Profile(NIST AI 600-1): https://doi.org/10.6028/NIST.AI.600-1
