---
tags: [llmops, prompt-management, versioning, registry]
level: beginner
last_updated: 2026-07-06
reviewed_on: 2026-10-04
review_status: partial
document_type: learning
category_major: "AI·DT"
category_middle: "LLM 평가·운영"
category_minor: "운영 기초·관측"
note_kind: "학습"
classified_on: "2026-10-05"
---

> [!info] 검토 범위 — 2026-10-04
> 공식·일차 근거와 로컬 검증은 [공통 적용 조건](./verified-conditions.md), 변경·미확인은 [정리 기록](./organization-log.md)에 있다. 실제 사내 접속·모델 품질·운영 승인과 Claude 협의는 미확인이다. 원래17개를 개별 검토했다. 실제 운영·읽기 화면 검증은 미완료다.


# 02. 프롬프트 관리 및 버전 관리

> 프롬프트를 "코드 안에 흩어진 문자열"이 아니라 **버전이 붙은 자산(artifact)**으로 다룬다. 어떤 프롬프트가 어떤 점수를 냈는지 추적할 수 있게 한다.

## 왜 필요한가? (Why)

- 프롬프트 한 줄을 바꾸면 출력 전체가 바뀐다. 그런데 코드 곳곳에 f-string으로 박혀 있으면 **"언제 무엇을 왜 바꿨는지"**를 잃어버린다.
- 평가([04](./04-llm-evaluation-overview.md))는 "**프롬프트 버전 → 점수**"의 매핑이 있어야 의미가 있다. 버전이 없으면 점수가 올라도/내려도 원인을 못 짚는다.
- 사내에서는 프롬프트에 **공정 용어·안전 지침·출력 포맷**이 촘촘히 들어간다. 이런 프롬프트는 사실상 "설정 파일"이므로 코드와 분리해 리뷰·롤백이 가능해야 한다.

## 핵심 개념 (What)

### 1) 프롬프트를 코드처럼 (Prompt as Code)
- **분리**: 프롬프트 텍스트를 소스코드에서 떼어 `prompts/` 디렉토리나 registry로.
- **버전**: `v1, v2, ...` 또는 semver. 커밋과 함께 git으로 이력 관리.
- **파라미터화**: 변수 슬롯(`{domain}`, `{question}`)을 분리해 텍스트 재사용.
- **메타데이터**: 작성자·목적·대상 모델·마지막 평가 점수를 함께 저장.

### 2) 프롬프트 registry의 최소 스키마

| 필드 | 의미 |
|------|------|
| `name` | 논리적 이름 (예: `recipe_qa_system`) |
| `version` | `v3` — 링크와 롤백의 기준 |
| `template` | 변수 슬롯이 있는 본문 |
| `model` | 이 버전이 검증된 대상 모델 |
| `eval_score` | 최근 오프라인 평가 점수(회귀 감시용) |

### 3) 프롬프트 변경 = 실험
프롬프트를 바꾸는 것은 "실험"이다. 그래서 **바꾸기 전 baseline 점수를 기록**하고, 바꾼 뒤 같은 eval set으로 재채점해 **회귀 여부**를 확인한 뒤에만 승격(promote)한다.

## 어떻게 사용하는가? (How)

### 파일 기반 프롬프트 registry (가장 단순·실용적)

```
prompts/
├── recipe_qa_system/
│   ├── v3.txt
│   ├── v4.txt
│   └── meta.json      # {"active": "v4", "v4": {"eval_score": 0.82, "model": "Kimi-K2.5"}}
```

```python
import json, re, math
from pathlib import Path

class PromptRegistry:
    def __init__(self, root="prompts"):
        self.root = Path(root)

    def get(self, name: str, version: str | None = None) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("name은 단일 식별자여야 함")
        meta = json.loads((self.root / name / "meta.json").read_text(encoding="utf-8"))
        version = meta["active"] if version is None else version
        if not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", version):
            raise ValueError("version은 단일 식별자여야 함")
        return (self.root / name / f"{version}.txt").read_text(encoding="utf-8")

    def render(self, name: str, version=None, **kwargs) -> str:
        # 빠진 슬롯은 KeyError. JSON 리터럴 중괄호는 {{ }}로 작성한다.
        return self.get(name, version).format_map(kwargs)

reg = PromptRegistry()  # 위 prompts/ 파일을 먼저 준비한다
system_prompt = reg.render("recipe_qa_system", domain="포토")
```

템플릿은 신뢰하는 로컬 파일을 사용한다. 이 검사는 이름/버전의 경로 성분을 제한하며 symlink·저장소 권한을 격리하는 sandbox는 아니다. `safe_substitute`는 누락 변수를 남기므로 오류 탐지 용도로 쓰지 않는다. Python `format_map`의 누락 키 오류와 리터럴 `{{}}` 규칙을 사용한다.

### 프롬프트 버전을 평가와 묶기
프롬프트를 바꿀 때마다 아래를 돌려 **버전별 점수 테이블**을 남긴다. (채점기는 [06](./06-automatic-metrics.md)/[07](./07-llm-as-a-judge.md))

```python
def eval_prompt_version(version: str, dataset, score_fn, call_llm, reg: PromptRegistry) -> float:
    # call_llm(system_prompt, question)와 채점기를 호출자가 명시적으로 전달한다.
    dataset = list(dataset)
    if not dataset:
        raise ValueError("빈 평가셋")
    scores = []
    for case in dataset:
        system = reg.render("recipe_qa_system", version=version, domain="포토")
        value = float(score_fn(case, call_llm(system, case["question"])))
        if not math.isfinite(value):
            raise ValueError("유한 점수 필요")
        scores.append(value)
    return sum(scores) / len(scores)

# 입력 파일·dataset·score_fn·call_llm을 준비한 뒤 실행:
# for version in ["v3", "v4"]:
#     print(version, eval_prompt_version(version, dataset, score_fn, call_llm, reg))
# 원래 노트의 v3=0.78/v4=0.82는 가상 출력이며 개선 실측 근거가 아니다.
```

### 프롬프트 회귀 방지 체크리스트
- [ ] 새 버전은 **기존 버전을 삭제하지 않고 추가**한다(롤백 가능).
- [ ] 승격 전 **같은 eval set**으로 baseline과 비교한다.
- [ ] 카테고리별 점수를 본다(전체 평균이 올라도 특정 공정 카테고리가 떨어질 수 있음).
- [ ] 출력 **포맷 계약**(JSON 스키마 등)이 깨지지 않는지 별도 체크.

> 프레임워크(LangSmith/Langfuse 등)의 Prompt Hub도 같은 개념을 SaaS로 제공하지만, 외부 SaaS 사용이 제한된 환경에서는 **git + 파일 registry**를 선택할 수 있다. 실제 회사 정책과 제품 선택은 미확인이다.

## 관련 문서
- [01. LLMOps 개요](./01-llmops-overview-lifecycle.md) — 개선 루프에서 프롬프트의 위치
- [03. 트레이싱 & 관측성](./03-tracing-observability.md) — 실행마다 프롬프트 버전을 로그에 남기기
- [07. LLM-as-a-Judge](./07-llm-as-a-judge.md) — 프롬프트 버전 비교를 pairwise로

## 참고 자료 (References)
- Prompt versioning 개념(일반): "treat prompts as versioned config, not inline strings"
- [Python 문자열/템플릿 공식 문서](https://docs.python.org/3/library/string.html) — 2026-10-04 확인; 누락 변수와 리터럴 중괄호 규칙.
