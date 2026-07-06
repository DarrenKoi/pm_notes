---
tags: [evaluation, dataset, golden-set, synthetic-data, labeling]
level: intermediate
last_updated: 2026-07-06
---

# 05. 평가 데이터셋 구축

> 평가의 품질은 데이터셋의 품질을 넘지 못한다. 실제 사용 분포를 대표하는 `eval_set.jsonl`을 어떻게 만들고 정답을 어떻게 라벨링하는지 다룬다.

## 왜 필요한가? (Why)

- 아무리 좋은 지표도 **엉뚱한 데이터**로 재면 무의미하다. "우리 사용자가 실제로 묻는 질문"을 대표하지 못하면 점수는 허상이다.
- 사내 문서 99%가 DRM 보호라 정답(reference)을 손으로 긁어오기 어렵다. → **VLM(Qwen3-VL) 파이프라인**으로 정답 근거를 추출해야 한다.
- 초기에는 데이터가 없다. **합성 데이터(synthetic)**로 부트스트랩하되, 사람이 검수한 **golden set**을 반드시 섞어야 신뢰가 생긴다.

## 핵심 개념 (What)

### 1) 데이터셋의 3계층
| 계층 | 규모 | 출처 | 용도 |
|------|------|------|------|
| **Golden set** | 소(50~200) | 사람이 직접 검수 | 채점기 검증(메타평가), 최종 판단 |
| **Silver set** | 중 | 합성 + 부분 검수 | 회귀 테스트 주력 |
| **Production-mined** | 대 | 실제 로그에서 샘플 | 실사용 분포 반영, 드리프트 감지 → [12](./12-monitoring-drift.md) |

### 2) 표준 스키마 (`eval_set.jsonl`)
전 문서가 공유하는 한 줄 = 한 케이스 포맷.

```jsonl
{"id": "q001", "question": "포토 공정에서 오버레이 오차의 주요 원인은?", "reference": "정답 텍스트(근거 문서 기반)", "contexts": ["근거 청크1", "근거 청크2"], "meta": {"category": "photo", "difficulty": "med", "source": "doc_1234 p.5"}}
```

- `reference`: reference-based 지표용 정답. 없으면 reference-free만 사용.
- `contexts`: RAG 평가에서 **검색 정답(gold context)**. faithfulness/recall 계산에 필요. → [08](./08-rag-evaluation.md)
- `meta.category`: 카테고리별 점수 분해에 필수. → [04](./04-llm-evaluation-overview.md)

### 3) 커버리지 설계
데이터셋은 **의도적으로** 균형을 맞춘다: 카테고리(공정별), 난이도(easy/med/hard), 유형(사실질의/요약/추론/거절해야 하는 질문). "거절해야 하는 질문"(문서에 없는 것)을 반드시 포함해 **환각**을 잡는다. → [10](./10-safety-hallucination-guardrails.md)

## 어떻게 사용하는가? (How)

### DRM 문서 → 정답 근거 추출 (VLM 파이프라인)
텍스트 복사가 막힌 문서는 스크린샷을 VLM에 넣어 텍스트/근거를 뽑는다.

```python
import base64
from openai import OpenAI
client = OpenAI(base_url="http://llm-gateway.internal/v1", api_key="EMPTY")

def extract_from_image(png_path: str) -> str:
    b64 = base64.b64encode(open(png_path, "rb").read()).decode()
    r = client.chat.completions.create(
        model="Qwen3-VL-30B",              # 정확도 우선 → 무거운 VLM
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "이 페이지의 본문을 표/수치 포함해 정확히 텍스트로 옮겨라."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
        temperature=0,
    )
    return r.choices[0].message.content
# → 추출된 본문을 context로, 여기서 Q/A를 만들면 정답에 출처가 붙는다
```

### 합성 QA 생성 (context에서 질문·정답 역생성)
문서 청크를 주고 "이 청크로 답할 수 있는 질문과 정답"을 LLM에게 만들게 한다. 초기 데이터 부트스트랩의 핵심 기법.

```python
import json
def synth_qa(context: str, n=3) -> list[dict]:
    prompt = f"""다음 문서만 근거로, 사실을 묻는 질문 {n}개와 정답을 만들어라.
문서에 없는 내용은 절대 만들지 마라. JSON 리스트로: [{{"question":..,"reference":..}}]

문서:
{context}"""
    out = client.chat.completions.create(
        model="Kimi-K2.5", temperature=0.3,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},   # 구조화 출력
    ).choices[0].message.content
    items = json.loads(out).get("items", json.loads(out))
    for it in items:
        it["contexts"] = [context]      # gold context 부착
    return items
```

> ⚠️ 합성 데이터의 함정: 같은 계열 LLM이 만든 질문은 **그 LLM이 쉽게 맞히는 쪽으로 편향**된다. 반드시 사람이 표본 검수해 golden으로 승격하고 실사용 로그에서 캔 질문을 섞는다.

### 사람 검수 라벨링 스키마
검수자는 각 케이스에 최소 이 라벨을 단다.

```python
label = {
    "id": "q001",
    "reference_ok": True,       # 정답이 실제로 맞는가
    "answerable": True,         # 문서로 답 가능한가(불가면 거절 정답)
    "category": "photo",
    "reviewer": "dy",
}
```

### 데이터셋 위생(hygiene) 체크리스트
- [ ] **중복 제거**: 임베딩 유사도로 near-duplicate 질문 병합. → [06](./06-automatic-metrics.md)
- [ ] **누수 방지**: eval set 질문이 프롬프트 예시(few-shot)에 들어가 있지 않은가.
- [ ] **거절 케이스 포함**: 문서에 없는 질문 10~20%.
- [ ] **출처 기록**: `meta.source`로 정답의 근거 페이지를 남겨 재검증 가능하게.

## 관련 문서
- [04. LLM 평가 개요](./04-llm-evaluation-overview.md) — 이 데이터로 무엇을 재는가
- [08. RAG 평가](./08-rag-evaluation.md) — `contexts`(gold context) 사용처
- [13. Mini Project](./13-mini-project.md) — 실제 사내 데이터셋 구축 절차

## 참고 자료 (References)
- RAGAS Testset Generation(합성 평가셋): https://docs.ragas.io/en/stable/concepts/testset_generation.html
- 데이터셋 near-duplicate 제거: 임베딩 코사인 유사도 임계값 방식
