---
tags: [llmops, governance, artifact-lineage, release, risk-management]
level: advanced
last_updated: 2026-07-06
reviewed_on: 2026-10-04
review_status: partial
document_type: learning
---

> [!info] 검토 범위 — 2026-10-04
> [공통 적용 조건](./verified-conditions.md)과 [정리 기록](./organization-log.md)에 판본·일차 근거·로컬 검증을 기록했다. 실제 회사 운영·모델/서버 품질·조직 승인·Claude 협의·읽기 화면은 미확인이다.


# 14. 아티팩트 계보와 거버넌스

> LLM 앱의 배포 단위는 모델 하나가 아니라 **프롬프트·모델·검색 인덱스·도구 스키마·평가셋·가드레일**의 조합이다. 이 조합을 추적하지 못하면 점수도, 장애도, 롤백도 설명할 수 없다.

## 왜 필요한가? (Why)

- LLM 앱의 품질은 여러 자산이 동시에 만든다. 프롬프트만 바뀌어도, 인덱스만 갱신돼도, 도구 인자 스키마만 바뀌어도 결과가 달라진다.
- 평가 리포트가 의미 있으려면 **어떤 버전 조합을 평가했는지**가 고정돼야 한다. "v4가 0.82점"이라는 말은 `prompt=v4`, `model=Kimi-K2.5`, `retriever=bge-m3-202607`, `eval_set=golden-20260706`까지 함께 있어야 재현 조건을 조사할 수 있다. mutable model alias·온도0·manifest만으로 동일 출력이 보장되지는 않는다.
- 사내 배포에서는 품질뿐 아니라 보안·기밀·책임 소재가 중요하다. 승인자, 위험 등급, 롤백 경로가 없으면 운영 문서로 부족하다.

## 핵심 개념 (What)

### 1) LLMOps 아티팩트 목록

| 아티팩트 | 예시 | 왜 버전이 필요한가 |
|---|---|---|
| Prompt | `recipe_qa_system:v4` | 출력 스타일·거절 정책·근거 강제 방식이 바뀜 |
| Model | `Kimi-K2.5`, `Qwen3-VL-30B` | 서빙 id뿐 아니라 revision/가중치·tokenizer·런타임 조건 기록 |
| Retriever / Index | `bge-m3-photo-20260706` | 문서 추가·청킹 변경·임베딩 변경이 검색 품질을 바꿈 |
| Tool schema | `search_docs@2`, `write_record@1` | agent가 호출 가능한 행동과 부작용 범위가 바뀜 |
| Eval set | `golden-20260706`, `redteam-20260706` | 점수 비교의 기준선 |
| Judge rubric | `faithfulness-rubric:v2` | 채점 기준 자체가 바뀌면 점수 분포도 바뀜 |
| Guardrail policy | `pii-policy:v3` | 안전 게이트와 허용/차단 범위 |

### 2) Release manifest

배포 가능한 단위는 코드 커밋 하나가 아니라 아래 같은 manifest다.

```yaml
release_id: rag-photo-learning-example
owner: example-team
risk_level: medium
status: draft
approvals: []
code_commit: unconfirmed
worktree_dirty: unconfirmed
artifacts:
  prompt: recipe_qa_system:v4
  model: target-model@immutable-snapshot-example
  judge_model: judge-model@immutable-snapshot-example
  embedding_model: embedding@immutable-snapshot-example
  index: photo-index:snapshot-example
  tool: search_docs:schema-v2
  eval: golden:dataset-hash-example
  redteam_set: redteam:dataset-hash-example
  rubric: judge-rubric:v2
  guardrail: pii-policy:v3
gates:
  correctness: ">= baseline - 0.02"
  faithfulness: ">= 0.85"
  refusal_correctness: ">= 0.90"
  leak_rate: "== 0"
rollback:
  release_id: previously-approved-release-example
  manifest_ref: approved_release_manifest.json
  manifest_sha256: unconfirmed
```

위 YAML은 가상 draft이며 승인된 배포가 아니다. snapshot/hash/승인 증거를 실제 값으로 확인하고, rollback manifest가 코드·모든 아티팩트 조합을 복원하는지 점검한다.

이 manifest를 [11. 온라인 평가 및 배포](./11-online-eval-deployment.md)의 CI gate 입력으로 쓰면 평가 리포트와 배포가 같은 대상을 가리킨다.

### 3) NIST AI RMF식 운영 관점

NIST AI RMF1.0의 위험 관리 기능은 `Govern / Map / Measure / Manage`로 나눠 다룬다. 다음 LLMOps 산출물 매핑은 작성자의 학습용 해석이며 NIST 인증/조직 정책이 아니다.

| RMF 관점 | LLMOps 산출물 |
|---|---|
| Govern | 소유자, 승인자, 위험 등급, 사용 금지 범위 |
| Map | 사용 사례, 데이터 출처, 사용자, 영향 범위, 실패 시 피해 |
| Measure | offline eval, red team, latency/cost, judge-human agreement |
| Manage | canary, rollback, incident runbook, eval set 갱신 |

## 어떻게 사용하는가? (How)

### 평가 리포트에 manifest 붙이기

```python
import json, subprocess, copy
from pathlib import Path

def git_sha(cwd=".") -> str:
    return subprocess.check_output(["git","rev-parse","HEAD"],cwd=cwd,text=True).strip()

def write_manifest(manifest: dict, path="release_manifest.json", cwd="."):
    draft = copy.deepcopy(manifest)
    if not draft.get("release_id") or not draft.get("owner"):
        raise ValueError("release id/owner 필요")
    if not all(draft.get("artifacts",{}).get(key) for key in
               ("prompt","model","index","tool","eval","rubric","guardrail")):
        raise ValueError("필수 아티팩트 조합 미확인")
    rollback = draft.get("rollback",{})
    if not rollback.get("release_id") or not rollback.get("manifest_ref"):
        raise ValueError("이전 전체 manifest 참조 필요")
    draft.update(code_commit=git_sha(cwd),status="draft",approvals=[])
    dirty = subprocess.check_output(["git","status","--porcelain","--untracked-files=normal","--","."],cwd=cwd,text=True)
    draft["worktree_dirty"] = bool(dirty.strip())
    # hash는 현재 dirty/untracked 본문을 나타내지 않는다.승인 전 실제 immutable snapshot 증거 필요.
    serialized = json.dumps(draft,ensure_ascii=False,indent=2,allow_nan=False)
    with Path(path).open("x",encoding="utf-8") as file:  # 기존 기록 덮어쓰기 방지
        file.write(serialized)
    return draft
# repo/Git 읽기만 수행; commit/push 하지 않음. 실습 산출 경로를 명시한다.
```

### 변경 유형별 승인 기준

다음 역할·gate는 제안 절차이며 실제 회사 승인 체계를 확인하지 않았다. 로컬 manifest 파일/승인자 문자열은 인증된 승인 증거가 아니다.

| 변경 | 필수 검증 | 승인자 |
|---|---|---|
| 프롬프트 문구 수정 | offline eval + 안전 게이트 | 서비스 오너 |
| 모델 교체 | offline eval + judge 메타평가 + canary | 서비스 오너 + 플랫폼 오너 |
| 인덱스 재구축 | retrieval recall@k + 대표 질의 smoke | 검색/데이터 오너 |
| 도구 추가/쓰기 권한 추가 | tool-call eval + 보안 리뷰 + 부작용 롤백 | 서비스 오너 + 보안 |
| 가드레일 완화 | red team 전체 재실행 | 보안 + 책임자 |

### 위험 등록부(risk register)

```markdown
| Risk ID | 설명 | 영향 | 완화 | 상태 | 소유자 |
|---|---|---:|---|---|---|
| R-001 | 문서에 없는 공정 조건을 확신 있게 답함 | High | faithfulness gate, refusal cases 20% | Open | QA lead |
| R-002 | 검색 도구가 권한 없는 문서를 반환 | High | ACL 필터, trace audit, red team | Unconfirmed | platform |
| R-003 | 프롬프트 변경으로 JSON 출력 포맷 깨짐 | Medium | schema validation, CI gate | Open | app |
```

표는 가상 예시이며 ACL 완화 효과를 실측한 기록이 아니다. 기존 R-002의 Mitigated 단정은 근거 없는 상태여서 Unconfirmed로 표시했다.

위험 등록부는 **배포 차단 여부를 결정하는 테이블**이다. `High` 위험이 `Open`이면 canary 이상으로 확대하지 않는다.

### Release review checklist

- [ ] manifest에 prompt/model/index/tool/eval/rubric/guardrail 버전이 모두 적혀 있다.
- [ ] [04](./04-llm-evaluation-overview.md) 하네스 결과가 baseline과 비교돼 있다.
- [ ] [10](./10-safety-hallucination-guardrails.md) 안전 게이트가 통과했다.
- [ ] [03](./03-tracing-observability.md) trace에서 manifest의 `release_id`를 찾을 수 있다.
- [ ] rollback 대상이 "이전 코드"가 아니라 이전 **아티팩트 조합**으로 명시돼 있다.
- [ ] 운영 중 이상 발생 시 [15. Incident Response](./15-incident-response-postmortem.md) 절차로 전환한다.

## 관련 문서

- [02. 프롬프트 관리 & 버전 관리](./02-prompt-management-versioning.md) — prompt artifact 관리
- [03. 트레이싱 & 관측성](./03-tracing-observability.md) — trace에 manifest 식별자 남기기
- [11. 온라인 평가 및 배포](./11-online-eval-deployment.md) — manifest 기반 release gate
- [15. Incident Response](./15-incident-response-postmortem.md) — 운영 사고 후 재발 방지

## 참고 자료 (References)

- NIST AI RMF 1.0: https://www.nist.gov/itl/ai-risk-management-framework
- NIST AI RMF Generative AI Profile(NIST AI 600-1): https://doi.org/10.6028/NIST.AI.600-1
- OpenAI Evals(평가 실행과 registry 사고): https://github.com/openai/evals
