---
tags: [dataset, pretraining, fine-tuning, sft, chat-template, deduplication, evaluation]
level: intermediate
last_updated: 2026-10-09
type: learning
category_major: "AI·DT"
category_middle: "AI 기초"
category_minor: "파운데이션 모델"
note_kind: "학습"
---

# LLM 학습 데이터셋은 어떻게 준비하는가?

> "LLM을 만든다"는 말은 사전학습, 지시 미세조정, 선호 학습 중 하나를 뜻하고, 단계마다 필요한 데이터의 모양과 양이 다르다. 어느 단계든 평가셋을 먼저 떼어 놓고, 중복과 누수를 막고, 모델이 요구하는 형식에 정확히 맞추는 일이 핵심이다.

> [!info] 범위
> 공개 논문과 일반적인 실무 절차를 정리했다. 데이터 양의 수치는 규모 감각을 위한 것이며 모델·작업에 따라 크게 다르다. 특정 기업의 데이터 소스나 필터 규칙은 다루지 않는다.

## 왜 필요한가? (Why)

- [Foundation LLM은 어떻게 만들어지는가?](./how-foundation-llms-are-built.md)는 데이터 정제를 파이프라인의 한 칸으로 소개한다. 이 문서는 그 칸 안에서 실제로 어떤 파일을 어떤 순서로 만드는지를 다룬다.
- 사내에서 현실적으로 하게 되는 일은 사전학습이 아니라 **공개 모델의 미세조정**이다. 이때 결과를 좌우하는 것은 학습 코드보다 데이터셋이다.
- 데이터 문제는 학습이 끝난 뒤에야 드러나고 원인을 찾기 어렵다. 평가셋 누수나 형식 불일치는 "점수는 높은데 실제로는 못 쓰는 모델"을 만든다.

## 핵심 개념 (What)

### 1. 단계마다 데이터가 다르다

| 단계 | 목적 | 데이터 모양 | 규모 감각 | 누가 만드는가 |
|------|------|-------------|-----------|---------------|
| 사전학습 | 언어와 지식 전반을 익힌다 | 라벨 없는 원문 텍스트 | 수천억 토큰 이상 (GPT-3는 3,000억 토큰 학습) | 수집·정제 파이프라인 |
| 지시 미세조정 (SFT) | 지시를 따르고 원하는 형식으로 답한다 | 지시-응답 쌍, 대화 | 수백~수십만 건 | 사람 작성, 로그, 합성 |
| 선호 학습 | 더 나은 답을 고르게 한다 | 같은 질문에 좋은 답/나쁜 답 쌍 | 수천~수만 건 | 사람 비교 평가 |
| 평가 | 위 단계의 결과를 잰다 | 질문과 채점 기준 | 수십~수천 건 | 사람이 직접 |

사전학습은 정답 라벨이 필요 없다. 텍스트 자체가 "다음 토큰"이라는 정답을 담고 있기 때문이다([Attention은 LLM 안에서 어떻게 쓰이는가?](./how-attention-powers-llm.md)의 causal mask 참고). 그래서 양으로 승부할 수 있고, 반대로 미세조정 데이터는 양보다 한 건 한 건의 정확도가 중요하다.

### 2. 공통 준비 절차

```text
0. 평가셋 정의     무엇을 잘하면 성공인지 먼저 정한다
1. 수집           원천 확보, 사용 권한·라이선스 확인
2. 텍스트 추출     PDF·HTML·이미지에서 본문만 꺼낸다
3. 정규화         유니코드, 공백, 줄바꿈, 깨진 문자
4. 중복 제거       완전 일치 → 거의 같은 문서(near-duplicate)
5. 품질 필터       너무 짧음, 반복, 잘린 문장, 기계 생성 스팸
6. 민감 정보 처리   개인정보, 기밀, 자격 증명
7. 분할           train / eval — 원천 문서 단위로 나눈다
8. 형식 변환       모델의 chat template에 맞춘다
9. 토큰화·길이 검사 컨텍스트 한도를 넘는 샘플 확인
10. 버전 기록      어떤 원천에서 어떤 규칙으로 몇 건이 남았는지
```

0번이 맨 앞인 것이 의도다. 평가셋 없이 데이터를 모으면 "충분한가", "이 필터가 도움이 됐는가"에 답할 수 없다.

### 3. 단계별 주의점

**추출**: 원천의 형식이 깨지면 뒤의 모든 단계가 오염된다. 표가 한 줄로 뭉개지거나, 머리글·쪽번호가 본문에 섞이거나, 2단 편집이 좌우로 섞여 읽히는 경우가 흔하다. DRM 문서를 스크린샷과 VLM으로 추출하는 경우에는 VLM이 없는 내용을 지어내는 오류가 추가되므로 표본 검수가 필요하다.

**중복 제거**: 완전 일치는 해시로 잡는다. 문제는 공백·날짜·머리글만 다른 거의 같은 문서다. 대규모에서는 MinHash 같은 근사 기법을 쓴다. 중복이 남으면 모델이 그 문장을 통째로 외우고, 학습과 평가 양쪽에 걸치면 평가 점수가 부풀려진다.

**분할과 누수**: 행 단위로 무작위 분할하면 같은 문서에서 나온 질문들이 train과 eval에 나뉘어 들어간다. 모델은 문서를 외워서 맞히게 되고 점수는 실제보다 높게 나온다. **원천 문서(또는 장비, 사용자, 기간) 단위로 나눈다.**

**형식**: 대화형 모델은 학습 때 본 특수 토큰 구조를 기대한다. 모델마다 역할 구분 토큰이 다르므로 손으로 문자열을 조립하지 말고 그 모델 토크나이저의 chat template을 쓴다. 형식이 어긋나면 학습은 돌아가지만 결과가 조용히 나빠진다. SFT에서는 보통 응답 부분의 토큰에만 손실을 계산한다.

### 4. 파일 형식

한 줄에 한 건인 JSONL이 사실상 표준이다.

```jsonl
{"text": "사전학습용 원문 한 덩어리 ..."}
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"prompt": "...", "chosen": "더 나은 답", "rejected": "덜 좋은 답"}
```

첫 줄이 사전학습, 둘째 줄이 SFT, 셋째 줄이 선호 학습 형태다.

### 5. 미세조정 데이터의 품질 기준

- **실제 사용과 같은 분포**: 서비스에서 들어올 질문의 종류·길이·말투와 비슷해야 한다. 쉬운 사례만 모으면 어려운 사례에서 무너진다.
- **답이 정확하다**: 틀린 답 한 건은 맞는 답 여러 건의 효과를 깎는다. 건수를 늘리기 전에 검수한다.
- **답의 형식이 일관된다**: 같은 종류의 질문에 어떤 답은 표, 어떤 답은 문장이면 모델도 흔들린다.
- **"모른다"와 거절 사례가 있다**: 답할 근거가 없는 질문에 근거 없음을 말하는 샘플이 없으면, 모델은 항상 뭔가를 지어내도록 학습된다.
- **다양하다**: 같은 틀에 단어만 바꾼 샘플 1,000건은 서로 다른 100건보다 못하다.
- **합성 데이터는 검수한다**: 더 큰 모델로 샘플을 생성하는 방법은 널리 쓰이지만, 생성 모델의 오류와 말버릇이 그대로 옮겨 온다.

### 6. 미세조정 전에 먼저 물어볼 것

| 원하는 것 | 먼저 시도할 방법 |
|-----------|------------------|
| 최신·사내 지식으로 답하게 하기 | RAG (지식은 프롬프트로 넣는다) |
| 출력 형식 맞추기 | 프롬프트 지시, 예시, 구조화 출력 |
| 특정 작업의 판단 습관·말투·도메인 표기를 익히게 하기 | 미세조정 |
| 작은 모델로 큰 모델의 특정 작업을 대체하기 | 미세조정 |

지식을 미세조정으로 넣으려 하면 데이터가 많이 들고 갱신할 때마다 다시 학습해야 한다. 미세조정은 "무엇을 아는가"보다 "어떻게 행동하는가"를 바꾸는 데 적합하다.

## 어떻게 사용하는가? (How)

### 최소 준비 스크립트

정규화, 중복 제거, 품질 필터, 문서 단위 분할, chat 형식 저장까지 표준 라이브러리만으로 하는 예제다.

```python
import hashlib
import json
import re
import tempfile
import unicodedata
from pathlib import Path

SYSTEM = "너는 계측 장비 문서를 근거로 답하는 조수다. 근거가 없으면 모른다고 답한다."

raw = [
    {"doc": "manual-a", "q": "측정 레시피란 무엇인가요?", "a": "측정 위치, 배율, 알고리즘 같은 측정 조건을 묶어 저장한 설정입니다."},
    {"doc": "manual-a", "q": "측정  레시피란 무엇인가요? ", "a": "측정 위치, 배율, 알고리즘 같은 측정 조건을 묶어 저장한 설정입니다."},  # 공백만 다른 중복
    {"doc": "manual-a", "q": "레시피는 어디에 저장되나요?", "a": "문서에 저장 위치에 대한 내용이 없어 답할 수 없습니다."},  # 모름 사례
    {"doc": "manual-b", "q": "정렬이 실패하면 어떻게 하나요?", "a": "네."},  # 답이 너무 짧다
    {"doc": "manual-b", "q": "정렬 마크는 왜 필요한가요?", "a": "웨이퍼의 위치와 회전을 보정해 매번 같은 지점을 측정하기 위해서입니다."},
    {"doc": "manual-d", "q": "배율을 높이면 무엇이 달라지나요?", "a": "시야가 좁아지고 한 픽셀이 나타내는 실제 길이가 짧아집니다."},
]


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip()


def is_eval(doc: str, ratio: float = 0.2) -> bool:
    """Split by source document, deterministically, so one document never lands on both sides."""
    return int(hashlib.sha1(doc.encode()).hexdigest(), 16) % 100 < ratio * 100


seen, train, evals, dropped = set(), [], [], {"duplicate": 0, "too_short": 0}
for r in raw:
    q, a = normalize(r["q"]), normalize(r["a"])
    key = hashlib.sha1(f"{q}\n{a}".encode()).hexdigest()
    if key in seen:
        dropped["duplicate"] += 1
        continue
    seen.add(key)
    if len(a) < 10:
        dropped["too_short"] += 1
        continue
    sample = {"doc": r["doc"], "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ]}
    (evals if is_eval(r["doc"]) else train).append(sample)

out = Path(tempfile.mkdtemp())
for name, rows in (("train", train), ("eval", evals)):
    with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)

print(f"train {len(train)}건, eval {len(evals)}건, 제외 {dropped}")
print("train 문서:", sorted({s["doc"] for s in train}), "/ eval 문서:", sorted({s["doc"] for s in evals}))

assert dropped == {"duplicate": 1, "too_short": 1}
assert not {s["doc"] for s in train} & {s["doc"] for s in evals}  # 문서 단위 누수 없음
```

출력:

```text
train 3건, eval 1건, 제외 {'duplicate': 1, 'too_short': 1}
train 문서: ['manual-a', 'manual-b'] / eval 문서: ['manual-d']
```

마지막 `assert`가 이 스크립트에서 가장 중요한 줄이다. 건수가 늘고 규칙이 복잡해져도 "같은 문서가 양쪽에 없다"는 검사는 남겨 둔다.

### 모델 형식으로 바꾸기

저장한 `messages`를 실제 학습 입력으로 바꿀 때는 그 모델의 토크나이저에 맡긴다.

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("모델-경로")  # 미세조정할 모델과 같은 토크나이저
text = tok.apply_chat_template(sample["messages"], tokenize=False)
# template이 이미 BOS 같은 특수 토큰을 넣었으므로 다시 붙이지 않는다
n_tokens = len(tok(text, add_special_tokens=False)["input_ids"])  # 컨텍스트 한도를 넘는지 확인
```

학습 전에 변환된 `text`를 몇 건 직접 출력해 눈으로 읽어 본다. 역할 토큰이 두 번 들어갔거나 응답 끝 토큰이 빠진 문제는 이 단계에서만 쉽게 잡힌다.

### 시작 순서

1. 실제 업무에서 나온 질문 50~100건으로 **평가셋**을 먼저 만들고 채점 기준을 정한다.
2. 미세조정 없이 프롬프트와 RAG만으로 그 평가셋 점수를 잰다. 이것이 기준선이다.
3. 기준선에서 틀린 유형을 분류한다. 그 유형을 겨냥해 학습 샘플을 수백 건 만든다.
4. 미세조정 후 같은 평가셋으로 다시 잰다. 오르지 않았으면 건수를 늘리기 전에 샘플 품질과 형식을 의심한다.

### 실무와 연결하기

- **DRM 문서**: 스크린샷과 VLM으로 뽑은 텍스트는 추출 오류율을 먼저 재야 학습 데이터로 쓸 수 있다. 추출 결과 일부를 원문과 대조해 표·수치가 맞는지 확인한다.
- **기밀**: 학습 데이터에 들어간 내용은 모델 출력으로 재현될 수 있다. 구체적인 장비 데이터와 내부 시스템 상세는 학습 데이터에 넣기 전에 접근 범위와 함께 판단한다.
- **장비·제품 단위 분할**: 장비 모델이나 벤더별로 문서가 나뉜다면, 처음 보는 장비에서도 통하는지 보려면 평가셋을 장비 단위로 떼어 놓는다.

## 참고 자료 (References)

- Brown et al., *Language Models are Few-Shot Learners* (GPT-3, 2020) — 학습 데이터 구성과 중복 문제 서술: <https://arxiv.org/abs/2005.14165>
- Ouyang et al., *Training language models to follow instructions with human feedback* (InstructGPT, 2022) — SFT·선호 데이터 수집 절차: <https://arxiv.org/abs/2203.02155>
- Broder, *On the resemblance and containment of documents* (MinHash, 1997): <https://ieeexplore.ieee.org/document/666900>
- Hugging Face Transformers, *Chat templates*: <https://huggingface.co/docs/transformers/chat_templating>

## 관련 문서

- [Foundation LLM은 어떻게 만들어지는가?](./how-foundation-llms-are-built.md) - 데이터 이후의 토크나이저, 사전학습, post-training
- [Attention은 LLM 안에서 어떻게 쓰이는가?](./how-attention-powers-llm.md) - 원문 텍스트가 그대로 학습 문제가 되는 이유
- [LLM은 어떻게 서빙되는가? Prefill, Cache, 양자화](./llm-serving-prefill-cache-quantization.md) - 학습 이후 서비스 단계
