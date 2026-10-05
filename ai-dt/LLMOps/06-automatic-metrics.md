---
tags: [evaluation, metrics, bleu, rouge, bertscore, embedding-similarity]
level: intermediate
last_updated: 2026-07-06
reviewed_on: 2026-10-04
review_status: partial
document_type: learning
category_major: "AI·DT"
category_middle: "LLM 평가·운영"
category_minor: "평가 데이터·지표"
note_kind: "학습"
classified_on: "2026-10-05"
---

> [!info] 검토 범위 — 2026-10-04
> 공식·일차 근거와 로컬 검증은 [공통 적용 조건](./verified-conditions.md), 변경·미확인은 [정리 기록](./organization-log.md)에 있다. 실제 사내 접속·모델 품질·운영 승인과 Claude 협의는 미확인이다. 원래17개를 개별 검토했다. 실제 운영·읽기 화면 검증은 미완료다.


# 06. 자동 평가 지표 (Reference-based)

> 정답(reference)이 있을 때 쓰는 규칙·통계·임베딩 기반 지표. 싸고 빠르고 재현 가능하다. 언제 믿을 수 있고 언제 무너지는지 경계를 안다.

## 왜 필요한가? (Why)

- LLM-as-Judge([07](./07-llm-as-a-judge.md))는 강력하지만 **느리고 비싸고 편향**이 있다. CI에서 매 커밋마다 수천 케이스를 돌리려면 자동 지표가 1차 방어선이다.
- 형식·추출형 과제(분류, JSON 필드, 수치)는 애초에 **정답이 딱 떨어지므로** 정확 일치가 최선이다.
- 자동 지표는 **회귀 감지**에 탁월하다. 절대 점수가 완벽하지 않아도 같은 잣대로 두 버전을 비교하면 방향도 데이터·정규화·채점기 편향에 영향받으므로 실제 실패 사례와 함께 해석한다.

## 핵심 개념 (What)

### 1) 지표 스펙트럼 — 엄격 → 관대
| 지표 | 무엇을 보나 | 강점 | 약점 |
|------|------------|------|------|
| **Exact / Regex match** | 문자 완전/패턴 일치 | 계약이 명확한 분류·추출·형식에 유용 | 표현이 다르면 0점 |
| **BLEU** | n-gram 정밀도(생성이 정답과 겹치나) | 번역의 n-gram 기반 지표 | 짧은 답·의역에 취약 |
| **ROUGE** | n-gram 재현율(정답을 얼마나 담았나) | 요약의 표면 중첩 지표 | 의미 아닌 표면 |
| **BERTScore** | 임베딩 토큰 정렬 유사도 | 의역 견딤 | 모델 의존, 사실성 못 봄 |
| **임베딩 유사도** | 문장 임베딩 코사인 | 의미 근접, 사내 BGE-M3로 계산 | "그럴듯한 오답" 못 거름 |

### 2) 핵심 경고: 표면 지표는 의미를 모른다
BLEU/ROUGE가 높아도 **사실이 틀릴 수 있고**, 낮아도 **정답을 다르게 말한 것**일 수 있다. 그래서 자유서술형에서는 임베딩 유사도·LLM-judge와 **병행**한다. 반대로 분류/추출은 정규화·수치 오차·단위 계약을 정의한 일치 검사를 사용한다.

### 3) 지표 선택 규칙 (task → metric)
- 분류·yes/no·수치 추출 → **exact / regex**
- 요약 → **ROUGE + LLM-judge**
- 번역·패러프레이즈 → **BLEU/BERTScore**
- 자유 QA → **임베딩 유사도 + LLM-judge(faithfulness)**

## 어떻게 사용하는가? (How)

### Exact / 정규화 일치 (추출·형식 과제)
채점 전 **정규화**(Unicode·대소문자·공백 처리)가 점수를 좌우한다.

```python
import re, unicodedata

def normalize(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("문자열 필요")
    # 소수점/부호/단위를 보존한다. NFKC·casefold도 과제별 계약으로 선택한다.
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).strip().casefold())

def reference_text(case) -> str:
    reference = case.get("reference")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("정답 없음: reference-based 점수 미확인")
    return reference

def exact_match(case, pred) -> float:
    return float(normalize(pred) == normalize(reference_text(case)))

def contains_match(case, pred) -> float:
    # 부분문자열 포함률. “정답이 아님” 같은 부정도 통과할 수 있어 사실성 지표가 아님.
    return float(normalize(reference_text(case)) in normalize(pred))
```

### BLEU / ROUGE (표준 라이브러리)

```python
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU

class WhitespaceTokenizer:
    def tokenize(self, text):
        return normalize(text).split()  # 한글을 버리지 않음. 형태소 분석은 아님.

rouge = rouge_scorer.RougeScorer(["rougeL"], tokenizer=WhitespaceTokenizer())
bleu = BLEU(tokenize="intl", smooth_method="exp", effective_order=False)

def rougeL(case, pred) -> float:
    return rouge.score(reference_text(case), pred)["rougeL"].fmeasure

def bleu_score(case, pred) -> float:
    result = bleu.corpus_score([pred], [[reference_text(case)]])
    return result.score / 100
# BLEU report에 str(bleu.get_signature())도 기록한다. 짧은 답은 4-gram 부재로 0일 수 있음.
```

ROUGE-L 예제는 F1이며 모든 ROUGE 점수가 재현율인 것은 아니다. 공백 토큰화와 BLEU intl은 한국어 형태소 평가를 대신하지 않는다. SacreBLEU의 ko-mecab은 별도 한국어 의존성/사전 검증이 필요하다. `evaluate.load`는 지표 코드 다운로드가 발생할 수 있어 이 예제는 로컬 설치 패키지를 직접 사용한다.

### 임베딩 유사도 (사내 BGE-M3)
표면이 달라도 의미가 가까우면 높은 점수. 사내 임베딩 모델로 계산한다.

```python
import numpy as np
# README의 client·embed·EMBED_MODEL을 먼저 정의한다.

def _embed(texts):
    return np.asarray(embed(texts, model=EMBED_MODEL), dtype=float)

def cosine_sim(case, pred) -> float:
    a, b = _embed([reference_text(case), pred])
    norms = np.linalg.norm([a, b], axis=1)
    if not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("영벡터/비유한 임베딩: 유사도 미확인")
    return float(np.clip((a/norms[0]) @ (b/norms[1]), -1.0, 1.0))

def semantic_pass(case, pred, th=0.75) -> float:
    if not np.isfinite(th) or not -1 <= th <= 1:
        raise ValueError("cosine 임계값 범위 오류")
    return float(cosine_sim(case, pred) >= th)
```

> cosine 범위는 [-1, 1]이며 정답 확률이 아니다. 0.75는 예시다. 별도 calibration set에서 사람 라벨과 맞춰 임계값을 정하고, 사용하지 않은 평가셋에서 다시 측정한다. 같은 golden set에 맞추고 그 점수를 일반화하면 누수 위험이 있다. → [04](./04-llm-evaluation-overview.md) 메타평가

### 여러 지표를 한 번에 → run_eval에 연결

```python
scorers = {
    "exact": exact_match,
    "contains": contains_match,
    "rougeL": rougeL,
    "semantic": cosine_sim,
}
# run_eval(system_fn, scorers, dataset)  → [04번 하네스]
```

### BERTScore (의역에 강건, 필요 시)

```python
# 선택 실습: bert-score/torch·승인된 로컬 모델·tokenizer snapshot이 별도로 필요.
import os
from pathlib import Path
from bert_score import score

def bertscore(case, pred) -> float:
    model_path = Path(os.environ["BERTSCORE_MODEL_PATH"])
    if not model_path.is_dir():
        raise ValueError("로컬 모델 디렉터리 필요")
    layers = int(os.environ["BERTSCORE_NUM_LAYERS"])
    if layers <= 0:
        raise ValueError("양의 layer 수 필요")
    _, _, f1 = score([pred], [reference_text(case)], model_type=str(model_path),
                     num_layers=layers, rescale_with_baseline=False, device="cpu")
    return float(f1.mean())
# 모델/layer/tokenizer/hash/설정을 기록한다. 실제 가중치 실행은 이번 검증에서 제외.
```

## 관련 문서
- [05. 평가 데이터셋 구축](./05-eval-dataset-construction.md) — reference/contexts 준비
- [07. LLM-as-a-Judge](./07-llm-as-a-judge.md) — 자동 지표가 못 보는 의미·사실성 채점
- [08. RAG 평가](./08-rag-evaluation.md) — 임베딩 유사도를 검색 지표로 확장

## 참고 자료 (References)
- Hugging Face `evaluate`: https://huggingface.co/docs/evaluate
- BERTScore 논문: https://arxiv.org/abs/1904.09675
- [SacreBLEU 공식 구현](https://github.com/mjpost/sacrebleu)·[ROUGE 공식 구현](https://github.com/google-research/google-research/tree/master/rouge) — 2026-10-04 확인, 확인 판본은 공통 안내 참고.
- [BERTScore 공식 구현](https://github.com/Tiiiger/bert_score) — 로컬 모델/layer 지정; 가중치 실행 미확인.
- BGE-M3(공개 모델 카드, 실제 사내 서빙 미확인): https://huggingface.co/BAAI/bge-m3
