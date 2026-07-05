---
tags: [llmops, prompt-management, versioning, registry]
level: beginner
last_updated: 2026-07-06
---

# 02. 프롬프트 관리 및 버전 관리

> 프롬프트를 "코드 안에 흩어진 문자열"이 아니라 **버전이 붙은 자산(artifact)**으로 다룬다. 어떤 프롬프트가 어떤 점수를 냈는지 추적 가능하게 만든다.

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
import json, string
from pathlib import Path

class PromptRegistry:
    def __init__(self, root="prompts"):
        self.root = Path(root)

    def get(self, name: str, version: str | None = None) -> str:
        meta = json.loads((self.root / name / "meta.json").read_text(encoding="utf-8"))
        version = version or meta["active"]           # 지정 없으면 active 버전
        return (self.root / name / f"{version}.txt").read_text(encoding="utf-8")

    def render(self, name: str, version=None, **kwargs) -> str:
        # {slot} 형태 변수 치환. 안전하게 누락 슬롯 방지
        return string.Template(self._to_dollar(self.get(name, version))).safe_substitute(**kwargs)

    @staticmethod
    def _to_dollar(t: str) -> str:      # {x} → $x (Template 문법)
        return t.replace("{", "${").replace("}", "}")

reg = PromptRegistry()
system_prompt = reg.render("recipe_qa_system", domain="포토")
```

### 프롬프트 버전을 평가와 묶기
프롬프트를 바꿀 때마다 아래를 돌려 **버전별 점수 테이블**을 남긴다. (채점기는 [06](./06-automatic-metrics.md)/[07](./07-llm-as-a-judge.md))

```python
def eval_prompt_version(version: str, dataset, score_fn) -> float:
    reg = PromptRegistry()
    def answer(q):
        sys = reg.render("recipe_qa_system", version=version, domain="포토")
        # ... client.chat.completions.create(system=sys, user=q) ...
        return call_llm(sys, q)
    return sum(score_fn(c, answer(c["question"])) for c in dataset) / len(dataset)

for v in ["v3", "v4"]:
    print(v, round(eval_prompt_version(v, dataset, score_fn), 3))
# v3 0.78
# v4 0.82   ← 회귀 없이 개선 확인 후 meta.json active를 v4로 승격
```

### 프롬프트 회귀 방지 체크리스트
- [ ] 새 버전은 **기존 버전을 삭제하지 않고 추가**한다(롤백 가능).
- [ ] 승격 전 **같은 eval set**으로 baseline과 비교한다.
- [ ] 카테고리별 점수를 본다(전체 평균이 올라도 특정 공정 카테고리가 떨어질 수 있음).
- [ ] 출력 **포맷 계약**(JSON 스키마 등)이 깨지지 않는지 별도 체크.

> 프레임워크(LangSmith/Langfuse 등)의 Prompt Hub도 같은 개념을 SaaS로 제공하지만, 사내에서는 외부 SaaS가 막혀 있으므로 **git + 파일 registry**가 가장 확실하다.

## 관련 문서
- [01. LLMOps 개요](./01-llmops-overview-lifecycle.md) — 개선 루프에서 프롬프트의 위치
- [03. 트레이싱 & 관측성](./03-tracing-observability.md) — 실행마다 프롬프트 버전을 로그에 남기기
- [07. LLM-as-a-Judge](./07-llm-as-a-judge.md) — 프롬프트 버전 비교를 pairwise로

## 참고 자료 (References)
- Prompt versioning 개념(일반): "treat prompts as versioned config, not inline strings"
- `string.Template` 안전 치환: 표준 라이브러리 문서
