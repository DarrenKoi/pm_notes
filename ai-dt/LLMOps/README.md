---
tags: [llmops, evaluation]
reviewed_on: 2026-10-04
review_status: partial
document_type: index
category_major: "AI·DT"
category_middle: "LLM 평가·운영"
category_minor: "커리큘럼 안내"
note_kind: "목차"
classified_on: "2026-10-05"
---

> [!info] 검토 범위 — 2026-10-04
> 공식·일차 근거와 로컬 검증은 [공통 적용 조건](./verified-conditions.md), 변경·미확인은 [정리 기록](./organization-log.md)에 있다. 실제 사내 접속·모델 품질·운영 승인과 Claude 협의는 미확인이다. 원래17개를 개별 검토했다. 실제 운영·읽기 화면 검증은 미완료다.


# LLMOps & 평가(Evaluation) 학습 노트

> LLM 애플리케이션을 **운영 가능한 시스템**으로 만들기 위한 LLMOps와, 그 품질을 **정해진 조건에서 측정**하는 평가(Evaluation)를 `study_list.txt` 커리큘럼(기초 → 평가 기초 → 심화 평가 → 안전·운영 → Mini Project → 거버넌스·사고대응)을 따라 단계별로 정리한 실습 노트입니다.

---

## 🎯 이 노트의 방향

- **언어**: 한국어 (기술 용어는 영어 병기)
- **코드**: 학습용 함수와 조립 예제. 선행 블록·입력 파일·승인된 endpoint가 필요합니다. 01~15의 로컬 fixture 검증과 실제 모델 실행은 구분합니다. 접속·판본·출처는 [공통 적용 조건](./verified-conditions.md)에서 확인합니다.
- **깊이**: 커리큘럼의 각 라인을 별도 문서로 다루고 실무에서 바로 쓰는 기법(LLM-as-a-Judge rubric·편향 보정, RAG 검색/생성 지표, RAGAS 계열 지표를 사내 판정 모델로 계산, agent trajectory 평가, CI regression gate, production 모니터링·드리프트, release manifest, incident postmortem)을 포함합니다.

> [!warning] 사례와 실제 환경을 구분
> 기존 노트의 외부 API 차단·DRM 99%·Phoenix 사내 채택·Kimi/Qwen/BGE 서빙 이름은 확인되지 않은 시나리오다. 승인된 데이터·export 방식·접속 정책을 확인한 환경에서만 실습한다. 호환 endpoint라도 JSON mode·vision·embedding·usage 지원은 따로 확인한다.
> 실제 DB·트래픽·회사 권한은 이번 로컬 검증에 포함하지 않았다.

## LLMOps vs MLOps 한 줄 메모

- MLOps는 모델·데이터의 개발과 운영을 포괄합니다. 이 노트의 LLMOps는 프롬프트·검색·도구·평가·가드레일을 중심으로 설명하지만 fine-tuning·모델 배포도 포함할 수 있습니다.
- 그래서 LLMOps의 심장은 **평가(Evaluation) 루프**입니다. "바꿨더니 좋아졌는가"를 같은 조건의 지표·사람 검토·운영 결과로 판단합니다. 수치만으로 모든 품질을 증명할 수는 없습니다. → [04. 평가 개요](./04-llm-evaluation-overview.md)

---

## 📚 목차 (학습 순서)

### 1. LLMOps 기초와 라이프사이클
| 문서 | 내용 |
|------|------|
| [커리큘럼 커버리지 점검](./curriculum-coverage.md) | `study_list.txt` 항목별 생성 문서 매핑과 보강 포인트 |
| [01. LLMOps 개요 & 라이프사이클](./01-llmops-overview-lifecycle.md) | LLMOps 정의, MLOps와의 차이, LLM 앱 라이프사이클, 평가 중심 루프 |
| [02. 프롬프트 관리 & 버전 관리](./02-prompt-management-versioning.md) | 프롬프트를 코드처럼, registry/버전/템플릿, 회귀 방지 |
| [03. 트레이싱 & 관측성](./03-tracing-observability.md) | span/trace 개념, 토큰·비용·지연 추적, **Arize Phoenix** 계측·대시보드 |

### 2. LLM 평가 기초
| 문서 | 내용 |
|------|------|
| [04. LLM 평가 개요](./04-llm-evaluation-overview.md) | 왜 어려운가, offline/online, reference-based vs free, human-in-the-loop |
| [05. 평가 데이터셋 구축](./05-eval-dataset-construction.md) | golden set, 합성 데이터 생성, 라벨링 스키마, `eval_set.jsonl` |
| [06. 자동 평가 지표](./06-automatic-metrics.md) | exact/regex match, BLEU/ROUGE, BERTScore, BGE-M3 임베딩 유사도 |

### 3. LLM/RAG/Agent 심화 평가
| 문서 | 내용 |
|------|------|
| [07. LLM-as-a-Judge](./07-llm-as-a-judge.md) | rubric 설계, pointwise/pairwise, 편향(위치·장황함)과 보정, 사내 판정 모델 |
| [08. RAG 평가](./08-rag-evaluation.md) | 검색 지표(recall/precision/MRR/nDCG), 생성 지표(faithfulness/relevance), RAGAS 사내화 |
| [09. Agent · Tool 평가](./09-agent-tool-evaluation.md) | trajectory, tool-call 정확도, task success, 다단계 실패 원인 분해 |

### 4. 안전성 · 운영 · 배포
| 문서 | 내용 |
|------|------|
| [10. 안전성·환각·가드레일 평가](./10-safety-hallucination-guardrails.md) | hallucination, toxicity, PII/기밀 누출, jailbreak, red teaming |
| [11. 온라인 평가 & 배포](./11-online-eval-deployment.md) | A/B, canary, 오프라인 regression gate를 CI에 연결 |
| [12. 모니터링 & 드리프트](./12-monitoring-drift.md) | production 지표(**Phoenix** 대시보드), cost/latency, 데이터·행동 드리프트, feedback loop |

### 5. 실전 Mini Project
| 문서 | 내용 |
|------|------|
| [13. Mini Project 가이드](./13-mini-project.md) | 사내 RAG/Agent 평가 파이프라인: 요구사항 → 구현 → 테스트 → 발표/피드백 |

### 6. 거버넌스 · 사고대응
| 문서 | 내용 |
|------|------|
| [14. 아티팩트 계보와 거버넌스](./14-artifact-lineage-governance.md) | prompt/model/index/tool/eval/guardrail 버전 조합, release manifest, 승인 기준 |
| [15. Incident Response와 Postmortem](./15-incident-response-postmortem.md) | LLM 품질·안전·도구 사고 대응, 롤백, 사고 케이스의 eval set 편입 |

---

## 🧰 공통 개발 환경

```bash
# 별도 실습 환경의 확인 판본. 전체 의존성/OS 호환 lock은 아님.
python -m pip install openai==3.24.0 numpy==2.5.3 rouge-score==0.1.2 sacrebleu==2.6.0 jsonschema==4.26.0
python -m pip install arize-phoenix-otel==0.17.2 openinference-instrumentation-openai==0.1.63
python -m pip install arize-phoenix-client==3.5.0 pandas==3.0.6
# 08의 Ragas는 별도 Python3.12 legacy 환경: 공통 적용 조건 참고.
# Phoenix server·실제 운영·DeepEval은 별도 검증 대상.
```

### LLM(판정) / 임베딩 클라이언트 — 모든 문서 공통 보일러플레이트

평가에서 LLM은 두 역할로 쓰입니다. ① **평가 대상(under test)** 시스템, ② **판정자(judge)**. 두 역할의 모델·권한·서빙 설정을 별도로 기록합니다. 같은 gateway를 사용할 수도 있지만 독립된 judge가 항상 보장되는 것은 아닙니다.

```python
import os
import numpy as np
from openai import OpenAI

# 승인된 gateway 값을 환경변수로 설정한다. 공개 API 사용 시 해당 정책을 확인한다.
client = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"])
JUDGE_MODEL = os.environ["JUDGE_MODEL"]
TARGET_MODEL = os.environ["LLM_MODEL"]
VLM_MODEL = os.environ["VLM_MODEL"]
EMBED_MODEL = os.environ["EMBEDDING_MODEL"]

def chat(model: str, messages: list[dict], temperature: float = 0.0) -> str:
    r = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
    if not r.choices or not isinstance(r.choices[0].message.content, str):
        raise ValueError("텍스트 응답 없음: 거부/tool/서빙 계약을 별도로 확인")
    return r.choices[0].message.content

def embed(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError("비어 있지 않은 문자열 목록 필요")
    r = client.embeddings.create(model=model, input=texts, encoding_format="float")
    data = sorted(r.data, key=lambda d: d.index)
    if [d.index for d in data] != list(range(len(texts))):
        raise ValueError("임베딩 누락/중복 index")
    a = np.asarray([d.embedding for d in data], dtype=float)
    if a.ndim != 2 or a.shape[1] == 0 or not np.isfinite(a).all():
        raise ValueError("임베딩 차원/유한값 확인 실패")
    return a.tolist()
```

> 같은 API 형태는 인증·출력 schema·tool/vision/usage의 동일한 지원을 보장하지 않는다. 프레임워크마다 별도 adapter와 판본 검증이 필요하다. 온도 0도 서버의 완전한 결정성을 보장하지 않는다. → [07. LLM-as-a-Judge](./07-llm-as-a-judge.md)

### 평가 데이터 표준 포맷 (`eval_set.jsonl`)

```jsonl
{"id": "q001", "question": "...", "reference": "...", "contexts": ["..."], "meta": {"category": "recipe"}}
```

한 줄 = 한 케이스. `contexts`는 해당 실행에서 실제 검색한 문맥이며 정답 근거는 별도 `meta.gold_contexts`/`gold_ids`로 보관합니다. 13은 실제 검색과 gold를 분리하고, 15의 사고 후보는 사람 검수 전 평가 정답으로 사용하지 않습니다. → [05. 평가 데이터셋 구축](./05-eval-dataset-construction.md)

---

## 📖 참고 자료 (References)
- OpenAI 호환 클라이언트: `OpenAI(base_url=..., api_key=...)` + 호출 시 `model=...`
- RAGAS (RAG 평가 지표): https://docs.ragas.io/
- DeepEval (LLM 평가 프레임워크): https://docs.confident-ai.com/
- Arize Phoenix(자체 호스팅 후보, 사내 사용 미확인): https://docs.arize.com/phoenix
- OpenAI Evals(개념 참고): https://github.com/openai/evals
- OpenTelemetry GenAI Semantic Conventions: https://github.com/open-telemetry/semantic-conventions-genai
- OWASP Top 10 for LLM Applications 2025: https://genai.owasp.org/llm-top-10/
- NIST AI RMF Generative AI Profile: https://doi.org/10.6028/NIST.AI.600-1
- LLM-as-a-Judge 원 논문(MT-Bench): https://arxiv.org/abs/2306.05685
