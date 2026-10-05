---
type: review-log
tags: [unsloth, documentation-review]
reviewed_on: 2026-10-04
review_status: partial
category_major: "AI·DT"
category_middle: "문서 관리"
category_minor: "정리·검증 기록"
note_kind: "관리 기록"
classified_on: "2026-10-05"
---

# Unsloth 문서 정리 기록

## 범위와 결정

원래 Markdown 5개를 전체 읽고 같은 경로에 유지했다. README의 원래 `last_updated: 2026-03-11`은 보존했고 나머지 원문에 없던 작성일을 추측으로 추가하지 않았다. 이 주제에는 실행 파일·첨부가 없다. 이동·삭제·다른 주제 통합과 신규 cross link는 없다.

개요, teacher/student 워크플로우, 데이터 형식, 학습/export 예제는 용도가 달라 유지했다. 반복되는 수치·버전·하드웨어·평가 한계를 [적용 조건](./verified-conditions.md)에 대표 설명으로 모았다. 구체 hyperparameter/target module 예제는 학습 레시피를 기준으로 읽게 하고 각 문서의 고유 시나리오를 보존했다.

| 원래 문서 | 개별 결과 |
|---|---|
| [README](./README.md) | 목적·4개 문서와 읽기 순서 유지. 모든 미지원 경우를 RAG+fine-tuning으로 해결한다는 오해 수정. 검토 metadata와 조건 안내 추가 |
| [개요](./unsloth-overview.md) | 속도·VRAM up to 수치는 공식 비교 주장이며 로컬 측정 아님. 최소 메모리·데이터 rows 기준을 보장 임계값으로 쓰지 않음. teacher/student 자원이 실제 확보되었다는 표현과 최적성 단정 수정 |
| [워크플로우](./local-sllm-finetuning-workflow.md) | “현재 가장 실용적/안전” 단정 대신 비교 실험. 원본 그룹을 먼저 분리하고 train에서 synthetic 확장. teacher 자기 평가 편향, 초기값/표본 수의 계획 조건, runtime import 지원 보완 |
| [데이터셋/template](./dataset-and-chat-template-guide.md) | pretty JSON과 실제 한 줄 JSONL 구분. JSON key 순서와 schema 의미 구분. 원본·파생 평가 누출, 모델별 role/BOS/EOS, 재tokenize 시 special token 중복, labels masking 확인 조건 보완 |
| [레시피](./training-and-deployment-recipe.md) | Core/Studio/Desktop 구분, Mac 범위 한정. TRL v0.23.1의 processing_class/max_length와 Unsloth 길이 인자 구분. Unsloth 우선 import, 타입 표기 추가. 미사용 get_peft_model 길이 인자 제거. 실제 eval/response-only가 없는 예제임을 명시하고 adapter/merged/GGUF 검증 단계 구분 |

## 근거 · 확인 2026-10-04

[적용 조건](./verified-conditions.md)의 표에 공식 Unsloth 설치·pip·requirements·모델 선택·template·GGUF, Hugging Face TRL 통합·v0.23.1 SFT, Transformers template 자료를 연결했다. 다음 사항을 실제 페이지 내용과 대조했다.

- 오래된 `installing-+-updating`는 열람 실패하여 공식 install/pip-install 경로로 갱신했다. 이전 GGUF `running-and-saving-models` 경로는 Page Not Found라서 현재 `inference-and-deployment` 문서로 수정했다.
- requirements는 Studio Mac 지원과 Core의 MLX 진행 안내를 동시에 포함한다. 이 차이를 단일 “Mac 지원/미지원”으로 정리하지 않았고 설치 방식별 조건·미확인으로 남겼다.
- 모델 선택 페이지의 300/1000 rows 안내는 확인했지만 실험으로 검증된 보편 임계값이라는 의미를 부여하지 않았다. 데이터 가이드의 다른 수치들도 원래 계획 예시로 보존한다.
- [공식 Unsloth main 소스](https://github.com/unslothai/unsloth/blob/main/unsloth/models/llama.py)의 get_peft_model에서 max_seq_length는 미사용으로 표시된다. 해당 예제 인자를 제거했다. main과 설치 배포 버전의 일치는 미확인이다.
- [TRL v0.23.1 SFT](https://huggingface.co/docs/trl/v0.23.1/en/sft_trainer)는 예제 API 인자의 기준이다. TRL/Unsloth/torch 전체 호환 lock을 검증한 것은 아니다. 기존 dtype auto는 공식 TRL 통합 예시에 있어 유지했으며 전체 학습 실행 가능성은 주장하지 않는다.

## Claude 협의

`HERDR_ENV=1`에서 `herdr pane current --current`가 `pane_not_found`다. 전용 Claude 연결을 확보하지 못했고 다른 pane은 제어하지 않았다. 의견을 받았다고 기록하지 않는다. 공개 자료로 확인한 수정은 진행했으며 제품별 지원 범위가 다른 부분과 다섯 문서 전체 합병 여부는 보류했다.

## 세 차례 검증

1. **목록·내용:** 원문 snapshot의 5개 경로와 현재 문서를 대조했다. teacher/student 도식, 단계별 실험, JSON·teacher prompt·template 예제, SFT·LoRA·GGUF 코드의 용도를 유지했다. 수정·대표 설명 연결은 위 개별 결과에 기록했다. 현재는 새 조건 안내와 기록을 포함해 7개다.
2. **근거·로컬 실행:** Python fence 5개 AST, bash 2개 구문, JSON 1개 파싱 통과. pretty JSON을 한 물리적 JSONL 줄로 변환 후 roundtrip하여 답변의 내장 newline이 보존됨을 확인했다. 실제 format_batch 함수를 가짜 tokenizer로 실행해 대화 두 개와 빈 batch 처리를 확인했다. TRL 인자 구분은 AST로 확인했다. 이는 실제 tokenizer·GPU 학습 검증이 아니다. Darwin arm64/Python 3.14 환경에서 unsloth/torch/trl/transformers/datasets/peft가 없어 설치·학습·다운로드·export는 수행하지 않았다.
3. **링크·메타데이터·읽기:** YAML 중복 key/검토일·relative links/anchor·Obsidian properties와 읽기 탐색 결과를 아래 최종 확인에 남긴다.

## 남은 미확인

GPU별 실제 VRAM/속도/성능, 정확한 dependency lock, 모델 접근·실제 template token IDs·response-only labels, GGUF 변환/양자화 품질·Ollama/vLLM serving, 제품별 Core/Studio 지원 차이, 전용 Claude 협의는 미완료다. source main·동적 docs가 이후 바뀔 수 있으므로 “최신 검증된 실행 레시피”라고 단정하지 않는다.

## 최종 확인

현재 Markdown 7개의 YAML 중복 key·검토일, 상대 링크·anchor 검사가 통과했다. 새 참조 오류와 기존 참조 오류는 각각 0개다. Obsidian `pm_notes` vault에서 7개 문서의 properties를 확인했고 README의 적용 조건 링크로 이동하여 `ai-dt/unsloth/verified-conditions` 경로, 한국어 aliases, 공식 근거 표와 데이터 조건이 읽기 콘텐츠에 표시됨을 확인했다. 모든 아래쪽 화면이나 실제 GPU 예제를 검증한 것은 아니다.

원문 5개 경로가 모두 남아 있다. “가장 안전/실용” 등의 절 제목만 조건부 실험 표현으로 수정했고 다른 실제 절 제목과 고유 예제의 역할을 유지했다. 기록 작성 후 로컬 검사를 다시 실행한다. 실행 코드·첨부를 수정하지 않았고 커밋·push는 하지 않았다.
