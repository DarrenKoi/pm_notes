---
tags: [evaluation, metrics, bleu, rouge, bertscore, embedding-similarity]
level: intermediate
last_updated: 2026-07-06
---

# 06. 자동 평가 지표 (Reference-based)

> 정답(reference)이 있을 때 쓰는 규칙·통계·임베딩 기반 지표. 싸고 빠르고 재현 가능하다. 언제 믿을 수 있고 언제 무너지는지 경계를 안다.

## 왜 필요한가? (Why)

- LLM-as-Judge([07](./07-llm-as-a-judge.md))는 강력하지만 **느리고 비싸고 편향**이 있다. CI에서 매 커밋마다 수천 케이스를 돌리려면 자동 지표가 1차 방어선이다.
- 형식·추출형 과제(분류, JSON 필드, 수치)는 애초에 **정답이 딱 떨어지므로** 정확 일치가 최선이다.
- 자동 지표는 **회귀 감지**에 탁월하다. 절대 점수가 완벽하지 않아도, 같은 잣대로 두 버전을 비교하면 방향은 정확하다.

## 핵심 개념 (What)

### 1) 지표 스펙트럼 — 엄격 → 관대
| 지표 | 무엇을 보나 | 강점 | 약점 |
|------|------------|------|------|
| **Exact / Regex match** | 문자 완전/패턴 일치 | 분류·추출·형식에 완벽 | 표현이 다르면 0점 |
| **BLEU** | n-gram 정밀도(생성이 정답과 겹치나) | 기계번역 표준 | 짧은 답·의역에 취약 |
| **ROUGE** | n-gram 재현율(정답을 얼마나 담았나) | 요약 평가 표준 | 의미 아닌 표면 |
| **BERTScore** | 임베딩 토큰 정렬 유사도 | 의역 견딤 | 모델 의존, 사실성 못 봄 |
| **임베딩 유사도** | 문장 임베딩 코사인 | 의미 근접, 사내 BGE-M3로 계산 | "그럴듯한 오답" 못 거름 |

### 2) 핵심 경고: 표면 지표는 의미를 모른다
BLEU/ROUGE가 높아도 **사실이 틀릴 수 있고**, 낮아도 **정답을 다르게 말한 것**일 수 있다. 그래서 자유서술형에서는 임베딩 유사도·LLM-judge와 **병행**한다. 반대로 분류/추출은 exact match가 진리다.

### 3) 지표 선택 규칙 (task → metric)
- 분류·yes/no·수치 추출 → **exact / regex**
- 요약 → **ROUGE + LLM-judge**
- 번역·패러프레이즈 → **BLEU/BERTScore**
- 자유 QA → **임베딩 유사도 + LLM-judge(faithfulness)**

## 어떻게 사용하는가? (How)

### Exact / 정규화 일치 (추출·형식 과제)
채점 전 **정규화**(소문자·공백·문장부호 정리)가 점수를 좌우한다.

```python
import re
def normalize(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[.,!?;:'\"()]", "", s)
    return s

def exact_match(case, pred) -> float:
    return 1.0 if normalize(pred) == normalize(case["reference"]) else 0.0

def contains_match(case, pred) -> float:      # 정답이 답 안에 포함되면 인정
    return 1.0 if normalize(case["reference"]) in normalize(pred) else 0.0
```

### BLEU / ROUGE (표준 라이브러리)

```python
# pip install evaluate rouge-score sacrebleu
import evaluate
rouge = evaluate.load("rouge")
bleu = evaluate.load("sacrebleu")

def rougeL(case, pred) -> float:
    return rouge.compute(predictions=[pred], references=[case["reference"]])["rougeL"]

def bleu_score(case, pred) -> float:
    return bleu.compute(predictions=[pred], references=[[case["reference"]]])["score"] / 100
```

### 임베딩 유사도 (사내 BGE-M3)
표면이 달라도 의미가 가까우면 높은 점수. 사내 임베딩 모델로 계산한다.

```python
import numpy as np
from openai import OpenAI
client = OpenAI(base_url="http://llm-gateway.internal/v1", api_key="EMPTY")

def _embed(texts):
    r = client.embeddings.create(model="BGE-M3", input=texts)
    return np.array([d.embedding for d in r.data])

def cosine_sim(case, pred) -> float:
    a, b = _embed([case["reference"], pred])
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

# 임계값으로 pass/fail 이진화도 가능
def semantic_pass(case, pred, th=0.75) -> float:
    return 1.0 if cosine_sim(case, pred) >= th else 0.0
```

> 임계값(0.75 등)은 **golden set에서 인간 라벨과의 일치가 최대가 되도록** 튜닝한다. 임의로 정하면 안 된다. → [04](./04-llm-evaluation-overview.md) 메타평가

### 여러 지표를 한 번에 → run_eval에 연결

```python
scorers = {
    "exact": contains_match,
    "rougeL": rougeL,
    "semantic": cosine_sim,
}
# run_eval(system_fn, scorers, dataset)  → [04번 하네스]
```

### BERTScore (의역에 강건, 필요 시)

```python
# pip install bert-score  (모델은 사내 미러의 한국어 지원 모델 지정)
from bert_score import score
def bertscore(case, pred) -> float:
    P, R, F1 = score([pred], [case["reference"]], lang="ko", rescale_with_baseline=False)
    return float(F1.mean())
```

## 관련 문서
- [05. 평가 데이터셋 구축](./05-eval-dataset-construction.md) — reference/contexts 준비
- [07. LLM-as-a-Judge](./07-llm-as-a-judge.md) — 자동 지표가 못 보는 의미·사실성 채점
- [08. RAG 평가](./08-rag-evaluation.md) — 임베딩 유사도를 검색 지표로 확장

## 참고 자료 (References)
- Hugging Face `evaluate`: https://huggingface.co/docs/evaluate
- BERTScore 논문: https://arxiv.org/abs/1904.09675
- BGE-M3(사내 임베딩): https://huggingface.co/BAAI/bge-m3
