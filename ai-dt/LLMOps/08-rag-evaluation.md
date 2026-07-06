---
tags: [evaluation, rag, retrieval-metrics, faithfulness, ragas]
level: advanced
last_updated: 2026-07-06
---

# 08. RAG 평가

> RAG는 **검색(retrieval)**과 **생성(generation)** 두 단계라, 평가도 두 축으로 나눠야 원인을 짚는다. 검색 지표(recall/precision/MRR/nDCG)와 생성 지표(faithfulness/answer relevance)를 사내 모델로 계산한다.

## 왜 필요한가? (Why)

- RAG 답이 틀렸을 때 원인은 둘 중 하나다: **검색이 근거를 못 가져왔거나(retrieval)**, 근거는 맞는데 **생성이 왜곡했거나(generation)**. 한 숫자(최종 정확도)만 보면 어디를 고칠지 모른다.
- 사내 RAG는 DRM 문서 기반이라 **환각(문서에 없는 말)**이 특히 위험하다. faithfulness를 별도로 재야 한다.
- RAGAS 같은 표준 지표는 훌륭하지만 기본 judge가 OpenAI다. **사내 엔드포인트로 교체**해야 쓸 수 있다.

## 핵심 개념 (What)

### 1) 두 축으로 분해

```
질문 → [검색] → contexts → [생성] → 답변
         ↑ 검색 지표              ↑ 생성 지표
   recall/precision/MRR/nDCG   faithfulness/answer relevance
```

### 2) 검색 지표 (gold context가 필요 — [05](./05-eval-dataset-construction.md))
| 지표 | 의미 |
|------|------|
| **Recall@k** | 정답 문서가 상위 k에 들어왔나 (놓치면 생성이 답할 수 없음 — RAG에서 가장 중요) |
| **Precision@k** | 상위 k 중 관련 문서 비율 (잡음이 많으면 생성이 흔들림) |
| **MRR** | 첫 정답 문서의 순위 역수 (정답을 얼마나 위로 올렸나) |
| **nDCG@k** | 순위와 관련도를 함께 반영한 랭킹 품질 |
| **Context Precision/Recall** (RAGAS) | gold 없이 LLM으로 근거-질문 관련성 판정 |

### 3) 생성 지표 (LLM-judge 기반 — [07](./07-llm-as-a-judge.md))
- **Faithfulness(충실성)**: 답의 각 주장이 **검색된 context로 뒷받침**되는가. 환각 탐지의 핵심.
- **Answer Relevance**: 답이 **질문에 답하는가**(장황·회피 감점).
- **Context Utilization**: 가져온 근거를 실제로 **활용**했는가.

### 4) 진단 규칙 (어디를 고칠까)
- Recall 낮음 → 검색 문제: 청킹·임베딩·top_k·하이브리드 검색 개선.
- Recall 높은데 faithfulness 낮음 → 생성/프롬프트 문제: 근거 강제·인용 요구.

## 어떻게 사용하는가? (How)

### 검색 지표 — gold context로 계산

```python
import numpy as np

def recall_at_k(retrieved_ids, gold_ids, k) -> float:
    top = set(retrieved_ids[:k])
    return len(top & set(gold_ids)) / max(len(gold_ids), 1)

def mrr(retrieved_ids, gold_ids) -> float:
    for i, rid in enumerate(retrieved_ids, 1):
        if rid in gold_ids:
            return 1.0 / i
    return 0.0

def ndcg_at_k(retrieved_ids, gold_ids, k) -> float:
    rel = [1 if rid in gold_ids else 0 for rid in retrieved_ids[:k]]
    dcg = sum(r / np.log2(i + 2) for i, r in enumerate(rel))
    ideal = sorted(rel, reverse=True)
    idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal)) or 1.0
    return dcg / idcg

# 케이스마다 retriever가 낸 문서 id 리스트를 gold와 비교
```

### Faithfulness — 사내 judge로 (RAGAS 개념을 직접 구현)
답을 문장 단위로 쪼개, 각 문장이 context로 뒷받침되는지 판정 → 뒷받침 비율.

```python
import json
from openai import OpenAI
client = OpenAI(base_url="http://llm-gateway.internal/v1", api_key="EMPTY")
JUDGE = "Kimi-K2.5"

def faithfulness(answer: str, contexts: list[str]) -> float:
    ctx = "\n---\n".join(contexts)
    prompt = f"""아래 '문맥'만을 근거로 '답변'의 각 주장을 검증하라.
답변을 개별 주장으로 나누고, 각 주장이 문맥으로 뒷받침되면 1, 아니면 0.
JSON만: {{"claims":[{{"claim":"...","supported":0또는1}}]}}

[문맥]
{ctx}

[답변]
{answer}"""
    out = client.chat.completions.create(
        model=JUDGE, temperature=0,
        messages=[{"role":"user","content":prompt}],
        response_format={"type":"json_object"}).choices[0].message.content
    claims = json.loads(out)["claims"]
    return sum(c["supported"] for c in claims) / max(len(claims), 1)
```

### Answer Relevance — 질문 역생성 방식
"이 답변에서 원래 질문을 복원"하게 하고 복원된 질문과 실제 질문의 임베딩 유사도로 관련성 측정(RAGAS 방식).

```python
def answer_relevance(question: str, answer: str, n=3) -> float:
    gen = client.chat.completions.create(
        model=JUDGE, temperature=0.3,
        messages=[{"role":"user","content":f"다음 답변이 답하려는 질문을 {n}개 추정해 한 줄씩:\n{answer}"}]
    ).choices[0].message.content.strip().split("\n")
    embs = client.embeddings.create(model="BGE-M3", input=[question] + gen).data
    q = np.array(embs[0].embedding); gs = [np.array(e.embedding) for e in embs[1:]]
    sims = [float(q @ g / (np.linalg.norm(q)*np.linalg.norm(g))) for g in gs]
    return sum(sims) / len(sims)
```

### RAGAS를 사내 모델로 (프레임워크 사용 시)
RAGAS는 judge/embeddings를 주입할 수 있으므로 사내 엔드포인트로 교체한다.

```python
# pip install ragas langchain-openai
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import faithfulness as rg_faith, answer_relevancy, context_recall

judge = ChatOpenAI(model="Kimi-K2.5", base_url="http://llm-gateway.internal/v1", api_key="EMPTY", temperature=0)
emb = OpenAIEmbeddings(model="BGE-M3", base_url="http://llm-gateway.internal/v1", api_key="EMPTY")
# result = evaluate(dataset, metrics=[rg_faith, answer_relevancy, context_recall], llm=judge, embeddings=emb)
```

> 로컬엔 벡터DB가 없으므로, 검색 지표는 **저장된 retrieved_ids 로그**로 오프라인 계산하고 생성 지표는 소규모 케이스로 검증한다. → [03](./03-tracing-observability.md)

## 관련 문서
- [05. 평가 데이터셋 구축](./05-eval-dataset-construction.md) — gold context 준비
- [07. LLM-as-a-Judge](./07-llm-as-a-judge.md) — faithfulness judge의 편향·보정
- [09. Agent · Tool 평가](./09-agent-tool-evaluation.md) — 검색이 도구 호출로 확장될 때

## 참고 자료 (References)
- RAGAS 지표 설명: https://docs.ragas.io/en/stable/concepts/metrics/
- nDCG 정의: https://en.wikipedia.org/wiki/Discounted_cumulative_gain
