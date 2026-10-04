---
tags: [evaluation, dataset, golden-set, synthetic-data, labeling]
level: intermediate
last_updated: 2026-07-06
reviewed_on: 2026-10-04
review_status: partial
document_type: learning
---

> [!info] 검토 범위 — 2026-10-04
> 공식·일차 근거와 로컬 검증은 [공통 적용 조건](./verified-conditions.md), 변경·미확인은 [정리 기록](./organization-log.md)에 있다. 실제 사내 접속·모델 품질·운영 승인과 Claude 협의는 미확인이다. 원래17개를 개별 검토했다. 실제 운영·읽기 화면 검증은 미완료다.


# 05. 평가 데이터셋 구축

> 평가의 품질은 데이터셋의 품질을 넘지 못한다. 실제 사용 분포를 대표하는 `eval_set.jsonl`을 어떻게 만들고 정답을 어떻게 라벨링하는지 다룬다.

## 왜 필요한가? (Why)

- 아무리 좋은 지표도 **엉뚱한 데이터**로 재면 무의미하다. "우리 사용자가 실제로 묻는 질문"을 대표하지 못하면 점수는 허상이다.
- 복사 제한 문서는 승인된 export 또는 화면·VLM 후보 경로를 검토한다. 원래 “사내 DRM 99%” 주장은 미확인이다. VLM 추출은 숫자·단위·표·출처를 사람이 원문과 대조해야 한다.
- 초기에는 데이터가 없다. **합성 데이터(synthetic)**로 부트스트랩하되, 사람이 검수한 **golden set**을 반드시 섞어야 신뢰가 생긴다.

## 핵심 개념 (What)

### 1) 데이터셋의 3계층
| 계층 | 규모 | 출처 | 용도 |
|------|------|------|------|
| **Golden set** | 시작 규모 예시(50~200) | 사람이 직접 검수 | 채점기 검증(메타평가), 최종 판단 |
| **Silver set** | 중 | 합성 + 부분 검수 | 회귀 테스트 주력 |
| **Production-mined** | 대 | 실제 로그에서 샘플 | 실사용 분포 반영, 드리프트 감지 → [12](./12-monitoring-drift.md) |

### 2) 표준 스키마 (`eval_set.jsonl`)
전 문서가 공유하는 한 줄 = 한 케이스 포맷.

```jsonl
{"id": "q001", "question": "포토 공정에서 오버레이 오차의 주요 원인은?", "reference": "정답 텍스트(근거 문서 기반)", "contexts": ["근거 청크1", "근거 청크2"], "meta": {"category": "photo", "difficulty": "med", "source": "doc_1234 p.5"}}
```

- `reference`: reference-based 지표용 정답. 없으면 reference-free만 사용.
- `contexts`: **해당 실행에서 실제 검색해 생성기에 제공한 문맥**. faithfulness에 사용한다. 정답 근거는 `meta.gold_contexts`, 검색 정답 id는 `meta.gold_ids`로 별도 보관한다. 합성 생성용 문맥과 실제 검색 결과를 섞지 않는다. → [08](./08-rag-evaluation.md)
- `meta.category`: 카테고리별 점수 분해에 필수. → [04](./04-llm-evaluation-overview.md)

### 3) 커버리지 설계
데이터셋은 **의도적으로** 균형을 맞춘다: 카테고리(공정별), 난이도(easy/med/hard), 유형(사실질의/요약/추론/거절해야 하는 질문). "거절해야 하는 질문"(문서에 없는 것)을 반드시 포함해 **환각**을 잡는다. → [10](./10-safety-hallucination-guardrails.md)

## 어떻게 사용하는가? (How)

### DRM 문서 → 정답 근거 추출 (VLM 파이프라인)
권한·기밀 처리·허용된 화면 수집이 확인된 문서만 사용한다. DRM을 우회하는 절차가 아니다. 추출용 VLM의 이름/출력 품질은 미확인이다.

```python
import base64
import os
from openai import OpenAI
client = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"])
TARGET_MODEL = os.environ["LLM_MODEL"]
VLM_MODEL = os.environ["VLM_MODEL"]

def extract_from_image(png_path: str) -> str:
    with open(png_path, "rb") as file:
        b64 = base64.b64encode(file.read()).decode()
    r = client.chat.completions.create(
        model=VLM_MODEL,  # 승인된 vision 서빙 id. 모델 크기는 정확도 보장이 아님
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "이 페이지의 본문을 표/수치 포함해 정확히 텍스트로 옮겨라."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
        temperature=0,
    )
    if not r.choices or not isinstance(r.choices[0].message.content, str):
        raise ValueError("텍스트 추출 결과 없음")
    return r.choices[0].message.content
# → 추출된 본문을 context로, 여기서 Q/A를 만들면 정답에 출처가 붙는다
```

### 합성 QA 생성 (context에서 질문·정답 역생성)
문서 청크를 주고 "이 청크로 답할 수 있는 질문과 정답"을 LLM에게 만들게 한다. 초기 데이터 부트스트랩의 핵심 기법.

```python
import json
JUDGE_MODEL = os.environ["JUDGE_MODEL"]

def synth_qa(context: str, n: int = 3) -> list[dict]:
    if not isinstance(context, str) or not context.strip() or type(n) is not int or n <= 0:
        raise ValueError("문맥과 양의 정수 n 필요")
    prompt = f"""다음 문서만 근거로, 사실 질문 {n}개와 정답을 만들어라.
문서 밖 사실은 만들지 마라. JSON object: {{"items": [{{"question": "...", "reference": "..."}}]}}
문서:
{context}"""
    r = client.chat.completions.create(
        model=JUDGE_MODEL, temperature=0.3,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},  # schema/사실성 보장이 아님
    )
    if not r.choices or not isinstance(r.choices[0].message.content, str):
        raise ValueError("합성 결과 없음")
    payload = json.loads(r.choices[0].message.content)
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list) or len(items) != n:
        raise ValueError("items 목록/개수 오류")
    output = []
    for item in items:
        if not isinstance(item, dict) or any(
            not isinstance(item.get(key), str) or not item[key].strip()
            for key in ("question", "reference")
        ):
            raise ValueError("질문·정답 문자열 오류")
        output.append({"question": item["question"], "reference": item["reference"],
                       "contexts": [],  # 실제 검색 실행에서 채움
                       "meta": {"gold_contexts": [context], "review_status": "unreviewed"}})
    return output
# 저장 전에 유일 id·category·source를 부여하고 검수한다. 아직 golden set이 아니다.
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
- [ ] **중복 후보 탐지**: 임베딩 유사도 후 사람이 의미·수치·조건 차이를 확인한다. 다른 정답 조건은 자동 병합하지 않는다. → [06](./06-automatic-metrics.md)
- [ ] **누수 방지**: eval set 질문이 프롬프트 예시(few-shot)에 들어가 있지 않은가.
- [ ] **거절 케이스 포함**: 10~20%는 시작 비율 예시. 실제 위험·사용 분포에 맞춰 정하고 수치의 근거를 기록한다.
- [ ] **출처 기록**: `meta.source`로 정답의 근거 페이지를 남겨 재검증 가능하게.

## 관련 문서
- [04. LLM 평가 개요](./04-llm-evaluation-overview.md) — 이 데이터로 무엇을 재는가
- [08. RAG 평가](./08-rag-evaluation.md) — 실제 `contexts`와 gold 근거의 구분
- [13. Mini Project](./13-mini-project.md) — 실제 사내 데이터셋 구축 절차

## 참고 자료 (References)
- 합성 QA 예제는 작성자의 축약 구현이며 특정 Ragas API를 호출하지 않는다. [Ragas metrics](https://docs.ragas.io/en/stable/concepts/metrics/)의 입력 계약은 실제 검색 결과와 reference를 구분한다(2026-10-04 확인).
- 데이터셋 near-duplicate 제거: 임베딩 코사인 유사도 임계값 방식
