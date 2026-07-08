# LLMOps & 평가(Evaluation) 학습 노트

> LLM 애플리케이션을 **운영 가능한 시스템**으로 만들기 위한 LLMOps와, 그 품질을 **숫자로 증명**하는 평가(Evaluation)를 `study_list.txt` 커리큘럼(기초 → 평가 기초 → 심화 평가 → 안전·운영 → Mini Project → 거버넌스·사고대응)을 따라 단계별로 정리한 실습 노트입니다.

---

## 🎯 이 노트의 방향

- **언어**: 한국어 (기술 용어는 영어 병기)
- **코드**: 실행 가능한 완전한 예제. 모든 LLM/임베딩 호출은 **공개 OpenAI API**와 **사내 OpenAI 호환 엔드포인트**(Kimi-K2.5 / Qwen3-VL / BGE-M3)를 **병기**합니다.
- **깊이**: 커리큘럼의 각 라인을 별도 문서로 다루고 실무에서 바로 쓰는 기법(LLM-as-a-Judge rubric·편향 보정, RAG 검색/생성 지표, RAGAS 계열 지표를 사내 판정 모델로 계산, agent trajectory 평가, CI regression gate, production 모니터링·드리프트, release manifest, incident postmortem)을 포함합니다.

> ⚠️ **사내 환경 주의**
> - 외부 LLM API(OpenAI/Anthropic/Google)는 방화벽으로 **차단**됩니다. 평가에서 흔히 쓰는 프레임워크(RAGAS, DeepEval 등)는 기본값이 OpenAI 판정 모델이므로 **반드시 사내 엔드포인트로 판정/임베딩 모델을 교체**해야 합니다.
> - **DB(OpenSearch/Elasticsearch)와 실제 트래픽은 로컬 개발 환경에 없습니다.** 로컬에서는 import·문법·소규모 오프라인 평가까지만 검증하고, 온라인/모니터링 코드는 구조만 확인합니다.
> - 사내 문서 99%는 DRM 보호 → 평가 데이터셋도 **스크린샷 + VLM(Qwen3-VL) 파이프라인**으로 만든 텍스트를 기준으로 구성합니다.

## LLMOps vs MLOps 한 줄 메모

- MLOps는 **모델 학습·배포·재학습** 중심. LLMOps는 대개 모델을 직접 학습하지 않고 **프롬프트·검색·도구·평가·가드레일**을 반복 개선하는 것이 핵심입니다.
- 그래서 LLMOps의 심장은 **평가(Evaluation) 루프**입니다. "바꿨더니 좋아졌는가"를 매번 숫자로 답할 수 있어야 운영이 성립합니다. → [04. 평가 개요](./04-llm-evaluation-overview.md)

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
# 핵심 패키지 (버전은 예시, 실제로는 사내 미러/프록시 사용)
pip install openai                        # 사내 OpenAI 호환 클라이언트
pip install pandas numpy scikit-learn     # 데이터셋·지표 계산
pip install evaluate rouge-score          # BLEU/ROUGE 등 참조 기반 지표
pip install arize-phoenix openinference-instrumentation-openai  # LLM 관측성 (트레이싱·모니터링)
# 선택: 평가 프레임워크 (판정/임베딩 모델은 반드시 사내 엔드포인트로 교체)
pip install ragas deepeval
```

### LLM(판정) / 임베딩 클라이언트 — 모든 문서 공통 보일러플레이트

평가에서 LLM은 두 역할로 쓰입니다. ① **평가 대상(under test)** 시스템, ② **판정자(judge)**. 둘 다 같은 사내 엔드포인트를 쓰되 모델/온도만 다르게 둡니다.

```python
# ── 공개 OpenAI API (커리큘럼 표준, 사내에서는 차단됨) ─────────────────
from openai import OpenAI
client = OpenAI()                                   # OPENAI_API_KEY 사용
# judge_model = "gpt-4o"; embed_model = "text-embedding-3-small"

# ── 사내 OpenAI 호환 엔드포인트 (실제 사용) ──────────────────────────
from openai import OpenAI
client = OpenAI(base_url="http://llm-gateway.internal/v1", api_key="EMPTY")

JUDGE_MODEL = "Kimi-K2.5"          # 판정용 텍스트 LLM (온도 0 권장)
TARGET_MODEL = "Kimi-K2.5"         # 평가 대상 (실제 서비스가 쓰는 모델)
VLM_MODEL = "Qwen3-VL-30B"         # DRM 문서/이미지 정답 추출용
EMBED_MODEL = "BGE-M3"             # 임베딩 유사도 지표용

def chat(model, messages, temperature=0.0):
    r = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
    return r.choices[0].message.content

def embed(texts, model=EMBED_MODEL):
    r = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in r.data]
```

> `base_url`/`model`만 바꾸면 공개 예제 코드와 사내 코드가 **동일**합니다. 평가 프레임워크(RAGAS/DeepEval)도 내부적으로 이 클라이언트를 주입해 판정 모델을 사내 모델로 강제합니다. → [07. LLM-as-a-Judge](./07-llm-as-a-judge.md)

### 평가 데이터 표준 포맷 (`eval_set.jsonl`)

```jsonl
{"id": "q001", "question": "...", "reference": "...", "contexts": ["..."], "meta": {"category": "recipe"}}
```

한 줄 = 한 케이스. 전 문서가 이 포맷을 공유합니다. → [05. 평가 데이터셋 구축](./05-eval-dataset-construction.md)

---

## 📖 참고 자료 (References)
- OpenAI 호환 클라이언트: `OpenAI(base_url=..., api_key=..., model=...)`
- RAGAS (RAG 평가 지표): https://docs.ragas.io/
- DeepEval (LLM 평가 프레임워크): https://docs.confident-ai.com/
- Arize Phoenix(사내 사용 LLM observability): https://docs.arize.com/phoenix
- OpenAI Evals(개념 참고): https://github.com/openai/evals
- OpenTelemetry GenAI Semantic Conventions: https://github.com/open-telemetry/semantic-conventions-genai
- OWASP Top 10 for LLM Applications 2025: https://genai.owasp.org/llm-top-10/
- NIST AI RMF Generative AI Profile: https://doi.org/10.6028/NIST.AI.600-1
- LLM-as-a-Judge 원 논문(MT-Bench): https://arxiv.org/abs/2306.05685
