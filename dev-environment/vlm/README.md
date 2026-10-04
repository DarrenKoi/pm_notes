---
reviewed_on: 2026-10-04
review_status: partial
document_type: learning
tags: [vlm, vllm, private-cloud, requests, h200]
level: beginner
last_updated: 2026-03-10
---

# 사내 VLM 실험 목차

> [!info] 2026-10-04 검토 범위
> UI grounding과 문서 OCR의 용도를 구분했다. 공개 모델 카드의 지원 경로 확인과 사내 GPU 실측은 다르다. 아래 모델별 품질 우열·GPU 수·메모리 설정은 미실측 가설이며 고정 버전 환경과 샘플로 평가해야 한다.

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
| [PPT/PDF OCR 목차](./read_ppt/README.md) | PPT/PDF 슬라이드 이미지 → 구조화된 텍스트 추출 (모델 비교 + 배포 + 프롬프트) |

## 추천 흐름

1. 모델은 `UI-Venus-1.5-8B` 또는 `MAI-UI-8B`부터 시작한다.
2. 모델 폴더를 cloud의 `/data/models/<model-name>` 아래에 둔다.
3. H200 상태를 먼저 확인한다.
4. `vllm serve`를 띄운다.
5. `/v1/models`와 `/v1/chat/completions`는 Python `requests`로 확인한다.
6. 외부 PC에서는 [send_image_to_vllm.py](./send_image_to_vllm.py)나 직접 `requests` 코드로 호출한다.

## 모델 선택의 대표 문서

모델 ID·목적·후보 목록은 [UI 모델 비교](./ui-vlm-models.md)에 모았다. 이 목차는 준비→서빙→클라이언트 호출 흐름을 안내한다. GPU 수와 추천 순위는 사내 측정 결과가 아니다.

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

아래는 파일 목록·용량의 1차 점검이다. shard index, checksum, processor, chat template와 고정 revision까지 확인해야 완전성을 판단할 수 있다.

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
| 전용 실행 코드 필요 | `UI-TARS`, `GUI-Actor`처럼 repo 실행 방식 확인이 먼저일 때 |
| parser path | `OmniParser`로 요소 목록을 먼저 뽑고 싶을 때 |

첫 성공 경로는 direct `vLLM`으로 고정하는 편이 가장 단순하다.

## 다음 문서

1. [UI 특화 VLM 모델 메모](./ui-vlm-models.md)
2. [Private Cloud에서 `vLLM` 시작](./private-cloud-vllm-next-steps.md)
3. [로컬 PC에서 `requests`로 이미지 보내기](./local-pc-vllm-image-guide.md)
4. [PPT/PDF 슬라이드 → 텍스트 추출 VLM 리서치](./read_ppt/vlm-for-ppt-pdf-extraction.md)

## 관련 문서

- [위로: 개발 환경](../README.md)
