---
type: learning
tags: [unsloth, finetuning]
reviewed_on: 2026-10-04
review_status: partial
category_major: "AI·DT"
category_middle: "모델 학습"
category_minor: "sLLM 파인튜닝"
note_kind: "학습"
classified_on: "2026-10-05"
---

# 데이터셋과 Chat Template 가이드

> 데이터 형식·chat template·label masking을 학습과 추론에서 일치시켜야 한다. 실패 빈도를 비교한 로컬 통계는 없다.

> [!info] 검토 범위 · 2026-10-04
> 공식 문서와 코드 예제를 대조한 학습 자료다. GPU 학습·모델 다운로드·export·serving은 실행하지 않았다. [적용 조건](./verified-conditions.md)과 [정리 기록](./organization-log.md)에 버전 경계와 미확인을 남겼다.

## 왜 이 문서가 중요한가?

Unsloth 공식 가이드는 데이터셋 형식과 chat template 일치를 반복해서 강조한다. 실제로 로컬 파인튜닝이 망가지는 대표 원인은 다음 두 가지다.

- 데이터가 일관되지 않다
- 학습 template과 추론 template이 다르다

## 권장 데이터 형식

대화형 JSONL을 사용하는 예다. 아래는 **가독성을 위해 펼친 JSON 객체**이며 실제 JSONL은 객체 하나를 한 물리적 줄에 기록한다.

```json
{"conversations":[
  {"role":"system","content":"You are a concise internal support assistant."},
  {"role":"user","content":"Summarize the incident in 3 bullets."},
  {"role":"assistant","content":"- Impact: ...\n- Root cause: ...\n- Next action: ..."}
]}
```

system role이 불필요하면 생략해도 되지만, 학습/추론 전체에서 같은 규칙을 유지해야 한다.

## 최소 데이터 수에 대한 현실적 기준

다음 구간은 원래 학습 메모의 계획 예시다. 검증된 최소 표본 수나 base/instruct 전환의 통계적 기준은 아니다.

- 50 ~ 100개: 파이프라인 점검용
- 100 ~ 300개: 좁은 형식 학습은 가능
- 500 ~ 3,000개: 실전적인 작은 태스크 튜닝 시작점
- 3,000개 이상: base 모델 파인튜닝도 진지하게 고려 가능

데이터 수보다 더 중요한 것은 예시 간의 일관성과 오류 밀도다.

## 데이터셋 구성 원칙

### 1. 정상 사례만 넣지 않는다

다음 유형을 함께 넣어야 한다.

- 정상 요청
- 모호한 요청
- 불완전한 입력
- 거절해야 하는 요청
- 형식 오류를 유도하는 요청

### 2. 원하는 출력 형식을 강하게 고정한다

태스크가 요구하는 형식을 예제로 보여준다. JSON 객체의 key 순서는 JSON schema의 의미적 유효성과 별개이며, 외부 시스템이 직렬화 순서를 요구할 때만 추가 조건으로 다룬다:

- JSON key 순서
- bullet 수
- heading 형식
- refusal 문구
- tool-call envelope

### 3. 너무 긴 답을 남발하지 않는다

student가 작을수록 장황한 답변보다 짧고 규격화된 답변에 더 안정적으로 수렴한다.

## synthetic data 만드는 방법

질문처럼 이미 로컬 API 모델이 있다면, 그 모델을 teacher로 쓰는 것이 좋다.

### 추천 절차

1. 사람이 gold examples 50 ~ 200개 작성(계획 예시) 후 원본별 train/eval/test 분리
2. train 원본만 teacher로 paraphrase / edge case 생성
3. 자동 필터로 중복 제거
4. 사람이 샘플 검수
5. 평가 원본과 같은 파생 그룹이 train에 섞이지 않았는지 확인

### teacher 프롬프트 예시

```text
You are generating training data for a small instruction-tuned model.
Keep the output short, deterministic, and schema-consistent.
Create 5 variations of the following example without changing the policy.
```

핵심은 teacher가 "좋아 보이는 답"이 아니라 "student가 배우기 쉬운 답"을 생성하도록 제한하는 것이다.

## train / eval 분리

eval set을 마지막에 급히 만드는 방식은 좋지 않다. 초기에 따로 떼어두는 편이 낫다.

권장 예:

- train: 80 ~ 90%
- eval: 10 ~ 20%

가능하면 실제 배포 입력과 비슷한 holdout set을 별도로 유지한다.

## chat template 맞추기

Unsloth chat templates 문서의 핵심은 "학습 형식과 추론 형식을 하나로 맞추라"는 점이다.

### 반드시 고정할 항목

- 모델별 template (`llama`, `chatml`, `gemma`, `qwen` 등)
- system role 사용 여부
- BOS / EOS 처리
- generation prompt 추가 방식
- multi-turn 구조

### 대표적인 실패 패턴

- 학습은 ChatML인데 추론은 Llama template 사용
- 학습 데이터에는 system prompt가 있는데 배포 프롬프트에는 없음
- export 후 serving layer가 EOS를 다르게 처리

## Unsloth에서 template 적용 예시

```python
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

text = tokenizer.apply_chat_template(
    conversations,
    tokenize=False,
    add_generation_prompt=False,
)
```

위 snippet의 `tokenizer`와 `conversations`는 앞서 로딩·검증한 객체를 전제한다. `llama-3.1`은 해당 형식의 예이며 이름만 비슷한 모든 모델에 적용하지 않는다. system role 지원, special token ID와 tokenizer의 실제 template을 확인한다. 렌더링한 문자열을 다시 tokenize한다면 BOS/EOS 중복을 피하도록 `add_special_tokens=False` 조건을 확인한다.

## response-only training 메모

모든 프롬프트 전체를 loss에 넣을 수도 있고, assistant 답변 부분만 loss에 반영할 수도 있다. 일반적으로 형식 튜닝에는 response-only 방식이 유리할 때가 많다.

다만 label masking이 잘못되면 학습 loss가 비정상적으로 낮거나 0에 가까워질 수 있으므로, 학습 전 일부 샘플을 직접 디코딩하고 실제 `labels != -100`인 토큰이 있는지 확인한다. response-only는 delimiter/template 지원에 따라 달라지며 아래 레시피의 preformatted text 학습은 response-only masking을 구현하지 않는다.

## 데이터 품질 체크리스트

- role 순서가 일관적인가
- assistant 답이 요청과 정확히 대응하는가
- 중복 샘플이 많은가
- 지나치게 긴 답이 섞여 있는가
- schema가 조금씩 흔들리는가
- 금지/거절/예외 사례가 포함되어 있는가
- eval set이 train set과 사실상 중복되지 않는가

## 참고 자료

- Unsloth Datasets Guide: <https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/datasets-guide>
- Unsloth Chat Templates: <https://docs.unsloth.ai/basics/chat-templates>
- Unsloth Fine-Tuning Guide: <https://unsloth.ai/docs/get-started/fine-tuning-guide>
