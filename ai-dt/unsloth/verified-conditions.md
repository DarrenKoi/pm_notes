---
type: learning
tags: [unsloth, finetuning, verification]
aliases: [Unsloth 적용 조건]
reviewed_on: 2026-10-04
review_status: partial
category_major: "AI·DT"
category_middle: "모델 학습"
category_minor: "sLLM 파인튜닝"
note_kind: "검토 기록"
classified_on: "2026-10-05"
---

# Unsloth 적용 조건

## 이 문서의 역할

[개요](./unsloth-overview.md)는 목적과 trade-off, [워크플로우](./local-sllm-finetuning-workflow.md)는 실험 순서, [데이터셋 가이드](./dataset-and-chat-template-guide.md)는 형식·분리·mask, [레시피](./training-and-deployment-recipe.md)는 구체 설정과 export 예제를 담당한다. 반복되는 버전·메모리·평가 조건은 여기에 모은다. 원래 문서들의 고유 teacher/student 시나리오와 예제는 유지한다.

## 확인한 근거 · 2026-10-04

| 항목 | 일차 자료와 적용 조건 |
|---|---|
| 설치 방식 | [설치 안내](https://unsloth.ai/docs/get-started/install)는 Desktop, Studio, Core를 구분한다. 이 묶음의 Python 예제는 Core를 다룸 |
| Core 설치 | [pip/가상 환경 안내](https://unsloth.ai/docs/get-started/install/pip-install)를 따른다. 기존 environment의 의존성이 바뀔 수 있어 실제 해결된 versions/lock을 기록해야 함 |
| 하드웨어 | [requirements](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements)는 Studio의 Mac/MLX 지원과 Core의 별도 안내를 함께 담음. CUDA Core 코드의 Mac 실행 가능성을 일반화하지 않음 |
| 메모리 예시 | 같은 requirements 표의 3B/7B/8B/14B QLoRA 최소 예시는 원문 수치를 보존. 모델·길이·batch·backend의 실측과 성공 보장 아님 |
| 모델 선택 | [선택 가이드](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/what-model-should-i-use)의 300/1000 rows 구간은 안내 수준. 고유 데이터 다양성·태스크·모델을 무시한 임계값으로 사용하지 않음 |
| 성능 | [TRL 통합 문서](https://huggingface.co/docs/trl/en/unsloth_integration)의 up to 수치는 vendor/framework 문서의 비교 주장. 이 저장소에서 측정한 속도·VRAM·정확도 아님 |
| TRL API | [SFT v0.23.1](https://huggingface.co/docs/trl/v0.23.1/en/sft_trainer)의 `processing_class`/`max_length`를 예제 기준으로 사용. Unsloth/torch/transformers까지 검증한 조합을 의미하지 않음 |
| template | [Unsloth template](https://unsloth.ai/docs/basics/chat-templates), [Transformers template](https://huggingface.co/docs/transformers/main/en/chat_templating): 모델의 실제 tokenizer/template를 사용하고 special token 중복을 피함 |
| export | [새 GGUF 문서](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf): export 뒤에도 동일 template/EOS/BOS와 runtime 결과를 확인. 이전 running-and-saving-models 경로는 Page Not Found여서 갱신 |

위 동적 문서들은 패키지의 정확한 설치 버전과 다르다. “현재 설치되는 최신 조합이 검증됐다”는 주장을 하지 않는다. 로컬은 Darwin arm64, Python 3.14이며 unsloth/torch/trl/transformers/datasets/peft가 설치되어 있지 않다. 이 환경에서 학습을 시도하지 않았다.

## 데이터·평가 계약

- JSONL은 한 물리적 줄당 하나의 JSON 객체다. 문서의 여러 줄 JSON은 설명용 pretty print다. `conversations`와 각 message의 role/content는 실제 tokenizer가 허용하는 형식으로 검증한다.
- 원본 seed·문서·사용자 사건별 그룹을 먼저 분리한다. 같은 원본의 paraphrase를 train과 eval에 나누면 평가 누출이 될 수 있다. 변형 생성은 train 그룹에서 수행하고 실제 배포 분포의 별도 평가 세트를 유지한다.
- 표본 수·train/eval 비율·LoRA 초기값은 실험 계획이다. 이 값만으로 품질을 예측하지 않는다. 작은 세트의 높은 점수는 불확실성을 줄여서 보고하면 안 된다.
- teacher가 생성과 judge를 함께 맡으면 자기 표현을 선호할 수 있다. 사람이 검토한 독립 평가 기준·사례를 유지하고 baseline과 동일 조건으로 비교한다.
- JSON key 순서와 JSON schema 유효성은 다른 계약이다. 사실성·거절 기준·출처 근거도 형식 통과와 따로 평가한다.

## 학습과 배포 계약

레시피의 preformatted `text`는 prompt와 assistant 답변 전체를 포함한다. response-only 목적이면 지원되는 template/masking 방식을 별도로 적용하고 일부 labels를 직접 확인한다. truncation 때문에 정답 부분이 없어지거나 모든 label이 -100이면 의미 있는 학습을 기대할 수 없다.

단일 GPU에서 예제의 batch 2 × accumulation 8은 optimizer update당 16개 microbatch sample의 출발점이다. 이는 GPU 한 번에 16개를 올린다는 뜻이 아니며 분산 학습·packing·마지막 불완전 batch에서는 해석 조건이 달라진다. warmup/learning rate는 실제 optimizer step 수와 함께 확인한다.

adapter는 base model/revision, tokenizer와 함께 사용한다. merged weights와 GGUF는 다른 산출물이다. 파일 저장 성공만으로 runtime 호환·권한·품질이 검증된 것은 아니다. base → adapter → merged → quantized 단계별로 동일한 평가 사례를 비교한다.

## 검증 경계

[정리 기록](./organization-log.md)에 개별 문서와 로컬 정적/형식 검증을 남겼다. 실제 GPU·dependency 설치·학습·GGUF 변환·serving 및 Claude 협의는 미완료다. 공식 자료끼리 지원 범위가 다르게 읽히는 부분은 설치 방식별로 표시하고 미확인으로 남긴다.
