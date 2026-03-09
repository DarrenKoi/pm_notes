---
tags: [vlm, vllm, private-cloud, requests, h200]
level: beginner
last_updated: 2026-03-10
status: active
---

# VLM Cloud Notes

> 이 폴더는 모델을 직접 준비한 뒤, H200 cloud 터미널에서 `vLLM`을 띄우고 Python `requests`로 확인하는 최소 문서만 남긴다.

## 이 폴더의 원칙

- 모델 다운로드 도구는 다루지 않는다.
- 모델은 직접 받아서 cloud에 둔다.
- HTTP 확인은 Python `requests` 예제로 통일한다.
- 별도 web wrapper 문서는 두지 않는다.

## 지금 남기는 문서

| 문서 | 역할 |
|---|---|
| [README](./README.md) | 전체 흐름, 모델 shortlist, 반입 메모 |
| [ui-vlm-models.md](./ui-vlm-models.md) | 모델별 간단 비교 |
| [private-cloud-vllm-next-steps.md](./private-cloud-vllm-next-steps.md) | H200 상태 확인, `vllm serve`, `requests` smoke test |
| [local-pc-vllm-image-guide.md](./local-pc-vllm-image-guide.md) | 로컬 PC나 다른 서버에서 이미지 전송 |

## 추천 흐름

1. 모델은 `UI-Venus-1.5-8B` 또는 `MAI-UI-8B`부터 시작한다.
2. 모델 폴더를 cloud의 `/data/models/<model-name>` 아래에 둔다.
3. H200 상태를 먼저 확인한다.
4. `vllm serve`를 띄운다.
5. `/v1/models`와 `/v1/chat/completions`는 Python `requests`로 확인한다.
6. 외부 PC에서는 [send_image_to_vllm.py](./send_image_to_vllm.py)나 직접 `requests` 코드로 호출한다.

## 간단 shortlist

| 우선순위 | 모델 | Repo ID | cloud 시작점 | 메모 |
|---|---|---|---|---|
| 1 | `UI-Venus-1.5-8B` | `inclusionAI/UI-Venus-1.5-8B` | 단일 GPU | 가장 무난한 첫 `vLLM` 후보 |
| 2 | `MAI-UI-8B` | `Tongyi-MAI/MAI-UI-8B` | 단일 GPU | baseline 비교용 |
| 3 | `UI-Venus-1.5-30B-A3B` | `inclusionAI/UI-Venus-1.5-30B-A3B` | 2 GPU | `8B`가 안정화된 뒤 확장 |
| 4 | `UGround-V1-7B` | `osunlp/UGround-V1-7B` | 단일 GPU | grounding 비교용 |
| 5 | `UI-TARS-1.5-7B` | `ByteDance-Seed/UI-TARS-1.5-7B` | 별도 runtime 검토 | first path로는 비권장 |
| 6 | `GUI-Actor-7B-Qwen2.5-VL` | `microsoft/GUI-Actor-7B-Qwen2.5-VL` | 별도 runtime 검토 | dedicated runtime 성격이 강함 |
| 7 | `OmniParser-v2.0` | `microsoft/OmniParser-v2.0` | parser stage | direct VLM 대체가 아니다 |

## 모델 폴더 반입 메모

cloud에 올리기 전에 아래 파일이 빠지지 않았는지 본다.

- `config.json`
- `tokenizer_config.json`
- `preprocessor_config.json`
- `generation_config.json`
- `model.safetensors` 또는 shard 전체
- shard 구조면 `model.safetensors.index.json`

권장 경로:

```text
/data/models/
  UI-Venus-1.5-8B/
  MAI-UI-8B/
  UI-Venus-1.5-30B-A3B/
```

cloud에 폴더를 올린 뒤에는 아래 정도만 확인하면 충분하다.

```bash
MODEL_DIR=/data/models/UI-Venus-1.5-8B

find "$MODEL_DIR" -maxdepth 1 | sort
du -sh "$MODEL_DIR"
```

web 업로드만 가능하면 임시 경로를 따로 둔다.

```bash
UPLOAD_DIR=~/uploads/vlm
MODEL_ROOT=/data/models

mkdir -p "$UPLOAD_DIR" "$MODEL_ROOT"
mv "$UPLOAD_DIR/UI-Venus-1.5-8B" "$MODEL_ROOT/"
```

## Runtime 선택

| 경로 | 언제 쓰나 |
|---|---|
| direct `vLLM` | `UI-Venus`, `MAI-UI`, `UGround`를 빨리 띄울 때 |
| dedicated runtime | `UI-TARS`, `GUI-Actor`처럼 repo runtime 요구가 강할 때 |
| parser path | `OmniParser`로 요소 목록을 먼저 뽑고 싶을 때 |

첫 성공 경로는 direct `vLLM`으로 고정하는 편이 가장 단순하다.

## 다음 문서

1. [UI 특화 VLM 모델 메모](./ui-vlm-models.md)
2. [Private Cloud에서 `vLLM` 시작](./private-cloud-vllm-next-steps.md)
3. [로컬 PC에서 `requests`로 이미지 보내기](./local-pc-vllm-image-guide.md)

## 관련 문서

- [위로: 개발 환경](../README.md)
- [루트 README](../../README.md)
