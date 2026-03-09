---
tags: [vlm, huggingface, private-cloud, gui-agent]
level: intermediate
last_updated: 2026-03-09
status: in-progress
---

# Hugging Face 다운로드 shortlist for private cloud

> 기준일: 2026-03-09. 아래 내용은 Hugging Face 공식 페이지를 직접 확인해서 정리했다.
>
> `UV-Venus`라고 적었지만, Hugging Face 공식 모델명은 `UI-Venus`다.

## TL;DR

- `H200 x2`를 가장 잘 쓰려면 1순위는 `UI-Venus-1.5-30B-A3B`
- 공개 Hugging Face 기준으로 `MAI-UI`는 현재 `2B`, `8B`만 바로 다운로드 가능
- 공개 Hugging Face 기준으로 `UI-TARS-1.5`는 현재 `UI-TARS-1.5-7B`가 바로 다운로드 가능
- `GUI-Actor`는 가중치만 받으면 끝이 아니라, HF 모델 카드 예시대로 GitHub runtime 코드도 같이 써야 한다
- `OmniParser v2`는 단독 에이전트가 아니라 파서 컴포넌트다. 보통 위 에이전트 모델과 같이 둔다

## 내가 먼저 받을 것

1. [inclusionAI/UI-Venus-1.5-30B-A3B](https://huggingface.co/inclusionAI/UI-Venus-1.5-30B-A3B)
2. [Tongyi-MAI/MAI-UI-8B](https://huggingface.co/Tongyi-MAI/MAI-UI-8B)
3. [ByteDance-Seed/UI-TARS-1.5-7B](https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B)
4. [microsoft/GUI-Actor-7B-Qwen2.5-VL](https://huggingface.co/microsoft/GUI-Actor-7B-Qwen2.5-VL)
5. [microsoft/GUI-Actor-Verifier-2B](https://huggingface.co/microsoft/GUI-Actor-Verifier-2B)
6. [microsoft/OmniParser-v2.0](https://huggingface.co/microsoft/OmniParser-v2.0)
7. [inclusionAI/UI-Venus-1.5-8B](https://huggingface.co/inclusionAI/UI-Venus-1.5-8B)
8. [Tongyi-MAI/MAI-UI-2B](https://huggingface.co/Tongyi-MAI/MAI-UI-2B)

## 모델별 다운로드 위치

| 관심 모델 | 추천 repo | 다운로드 링크 | 파일 목록 | 비고 |
|---|---|---|---|---|
| UI-TARS-1.5 | [ByteDance-Seed/UI-TARS-1.5-7B](https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B) | [Model card](https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B) | [Files](https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B/tree/main) | 공개 HF 체크포인트는 현재 1.5-7B |
| MAI-UI | [Tongyi-MAI/MAI-UI-8B](https://huggingface.co/Tongyi-MAI/MAI-UI-8B) | [Model card](https://huggingface.co/Tongyi-MAI/MAI-UI-8B) | [Files](https://huggingface.co/Tongyi-MAI/MAI-UI-8B/tree/main) | `H200 x2`에서도 우선 8B부터 권장 |
| MAI-UI 경량 | [Tongyi-MAI/MAI-UI-2B](https://huggingface.co/Tongyi-MAI/MAI-UI-2B) | [Model card](https://huggingface.co/Tongyi-MAI/MAI-UI-2B) | [Files](https://huggingface.co/Tongyi-MAI/MAI-UI-2B/tree/main) | 스모크 테스트용 |
| UI-Venus | [inclusionAI/UI-Venus-1.5-30B-A3B](https://huggingface.co/inclusionAI/UI-Venus-1.5-30B-A3B) | [Repo](https://huggingface.co/inclusionAI/UI-Venus-1.5-30B-A3B) | [Files](https://huggingface.co/inclusionAI/UI-Venus-1.5-30B-A3B/tree/main) | `H200 x2`라면 이 변형이 메인 |
| UI-Venus baseline | [inclusionAI/UI-Venus-1.5-8B](https://huggingface.co/inclusionAI/UI-Venus-1.5-8B) | [Model card](https://huggingface.co/inclusionAI/UI-Venus-1.5-8B) | [Files](https://huggingface.co/inclusionAI/UI-Venus-1.5-8B/tree/main) | 비교 기준용 |
| UI-Venus 경량 | [inclusionAI/UI-Venus-1.5-2B](https://huggingface.co/inclusionAI/UI-Venus-1.5-2B) | [Model card](https://huggingface.co/inclusionAI/UI-Venus-1.5-2B) | [Files](https://huggingface.co/inclusionAI/UI-Venus-1.5-2B/tree/main) | 빠른 확인용 |
| GUI-Actor | [microsoft/GUI-Actor-7B-Qwen2.5-VL](https://huggingface.co/microsoft/GUI-Actor-7B-Qwen2.5-VL) | [Model card](https://huggingface.co/microsoft/GUI-Actor-7B-Qwen2.5-VL) | [Files](https://huggingface.co/microsoft/GUI-Actor-7B-Qwen2.5-VL/tree/main) | HF usage 예시에 GitHub runtime 코드가 필요 |
| GUI-Actor verifier | [microsoft/GUI-Actor-Verifier-2B](https://huggingface.co/microsoft/GUI-Actor-Verifier-2B) | [Model card](https://huggingface.co/microsoft/GUI-Actor-Verifier-2B) | [Files](https://huggingface.co/microsoft/GUI-Actor-Verifier-2B/tree/main) | HF 카드 기준 성능 향상용 옵션 |
| OmniParser V2 | [microsoft/OmniParser-v2.0](https://huggingface.co/microsoft/OmniParser-v2.0) | [Model card](https://huggingface.co/microsoft/OmniParser-v2.0) | [Files](https://huggingface.co/microsoft/OmniParser-v2.0/tree/main) | 파서 컴포넌트 |

## 바로 실행할 다운로드 명령

```bash
# UI-Venus
huggingface-cli download inclusionAI/UI-Venus-1.5-30B-A3B --local-dir ./models/UI-Venus-1.5-30B-A3B
huggingface-cli download inclusionAI/UI-Venus-1.5-8B --local-dir ./models/UI-Venus-1.5-8B
huggingface-cli download inclusionAI/UI-Venus-1.5-2B --local-dir ./models/UI-Venus-1.5-2B

# MAI-UI
huggingface-cli download Tongyi-MAI/MAI-UI-8B --local-dir ./models/MAI-UI-8B
huggingface-cli download Tongyi-MAI/MAI-UI-2B --local-dir ./models/MAI-UI-2B

# UI-TARS-1.5
huggingface-cli download ByteDance-Seed/UI-TARS-1.5-7B --local-dir ./models/UI-TARS-1.5-7B

# GUI-Actor
huggingface-cli download microsoft/GUI-Actor-7B-Qwen2.5-VL --local-dir ./models/GUI-Actor-7B-Qwen2.5-VL
huggingface-cli download microsoft/GUI-Actor-Verifier-2B --local-dir ./models/GUI-Actor-Verifier-2B

# OmniParser V2
huggingface-cli download microsoft/OmniParser-v2.0 --local-dir ./models/OmniParser-v2.0
```

## 모델별 메모

### 1. UI-Venus

- 공식 family collection: [UI-Venus collection](https://huggingface.co/collections/inclusionAI/ui-venus)
- Hugging Face에서 확인된 1.5 계열 공개 repo:
  - [UI-Venus-1.5-30B-A3B](https://huggingface.co/inclusionAI/UI-Venus-1.5-30B-A3B)
  - [UI-Venus-1.5-8B](https://huggingface.co/inclusionAI/UI-Venus-1.5-8B)
  - [UI-Venus-1.5-2B](https://huggingface.co/inclusionAI/UI-Venus-1.5-2B)
- 공식 모델 카드에서 `vLLM` quick start가 제공된다. 2B/8B 페이지에서 직접 확인했다
- 라이선스는 `apache-2.0`
- `H200 x2`라면 `30B-A3B`를 메인으로 두고, `8B`를 baseline으로 같이 받는 구성이 실용적
- 브라우저로 개별 다운로드할 때는 `Files` 탭에서 가중치 shard와 tokenizer/config를 빠짐없이 받아야 한다. 큰 모델은 CLI가 훨씬 안전하다
- 단일 파일이 필요한 빠른 확인용 direct link:
  - [UI-Venus-1.5-2B model.safetensors](https://huggingface.co/inclusionAI/UI-Venus-1.5-2B/resolve/main/model.safetensors)

### 2. MAI-UI

- 공식 family collection: [MAI-UI collection](https://huggingface.co/collections/Tongyi-MAI/mai-ui)
- Hugging Face 공식 컬렉션 기준, 2026-03-09 현재 바로 받을 수 있는 공개 repo는:
  - [MAI-UI-8B](https://huggingface.co/Tongyi-MAI/MAI-UI-8B)
  - [MAI-UI-2B](https://huggingface.co/Tongyi-MAI/MAI-UI-2B)
- 모델 카드 본문은 `2B`, `8B`, `32B`, `235B-A22B` family를 언급하지만, 공식 HF 컬렉션에는 현재 `2B`, `8B`만 보인다
- `MAI-UI-8B` 파일 트리는 약 `17.6 GB`, shard 4개
- `MAI-UI-2B`는 약 `4.27 GB`, 단일 `model.safetensors`
- 공식 모델 카드에서 `vLLM` quick start가 제공된다
- 라이선스는 `apache-2.0`
- 단일 파일이 필요한 빠른 확인용 direct link:
  - [MAI-UI-2B model.safetensors](https://huggingface.co/Tongyi-MAI/MAI-UI-2B/resolve/main/model.safetensors)

### 3. UI-TARS-1.5

- 공개 Hugging Face 다운로드 repo: [ByteDance-Seed/UI-TARS-1.5-7B](https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B)
- 공식 모델 카드에는 top-performing `UI-TARS-1.5`는 early research access라고 적혀 있고, 공개 다운로드 체크포인트로는 `UI-TARS-1.5-7B`가 확인된다
- repo 크기는 약 `33.2 GB`, shard 7개
- 라이선스는 `apache-2.0`
- 관련 코드/앱 링크는 HF 모델 카드에 같이 적혀 있다:
  - Code: `https://github.com/bytedance/UI-TARS`
  - Application: `https://github.com/bytedance/UI-TARS-desktop`
- 즉, private cloud에서 쓸 때는 HF weights + GitHub runtime 쪽을 같이 검토하는 편이 안전하다

### 4. GUI-Actor

- 메인 repo: [microsoft/GUI-Actor-7B-Qwen2.5-VL](https://huggingface.co/microsoft/GUI-Actor-7B-Qwen2.5-VL)
- 권장 추가 repo: [microsoft/GUI-Actor-Verifier-2B](https://huggingface.co/microsoft/GUI-Actor-Verifier-2B)
- HF 모델 카드 usage 예시는 아래 Python import를 사용한다:
  - `from gui_actor.constants import chat_template`
  - `from gui_actor.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer`
- 즉, HF 가중치만으로 끝나는 형태가 아니라, HF 모델 카드에 연결된 GitHub runtime 코드가 사실상 같이 필요하다
- GUI-Actor 7B repo는 약 `16.8 GB`, shard 4개
- grounding 성능은 verifier 포함 시 더 좋아진다고 HF 카드에 적혀 있으므로, 비교 실험을 할 거면 verifier도 같이 받는 편이 낫다
- 라이선스는 `mit`

### 5. OmniParser V2

- 루트 repo: [microsoft/OmniParser-v2.0](https://huggingface.co/microsoft/OmniParser-v2.0)
- 하위 폴더:
  - [icon_detect](https://huggingface.co/microsoft/OmniParser-v2.0/tree/main/icon_detect)
  - [icon_caption](https://huggingface.co/microsoft/OmniParser-v2.0/tree/main/icon_caption)
- 직접 받을 파일:
  - [icon_detect/model.pt](https://huggingface.co/microsoft/OmniParser-v2.0/resolve/main/icon_detect/model.pt)
  - [icon_detect/model.yaml](https://huggingface.co/microsoft/OmniParser-v2.0/resolve/main/icon_detect/model.yaml)
  - [icon_detect/train_args.yaml](https://huggingface.co/microsoft/OmniParser-v2.0/resolve/main/icon_detect/train_args.yaml)
  - [icon_caption/config.json](https://huggingface.co/microsoft/OmniParser-v2.0/resolve/main/icon_caption/config.json)
  - [icon_caption/generation_config.json](https://huggingface.co/microsoft/OmniParser-v2.0/resolve/main/icon_caption/generation_config.json)
  - [icon_caption/model.safetensors](https://huggingface.co/microsoft/OmniParser-v2.0/resolve/main/icon_caption/model.safetensors)
- HF 쪽 문서 기준 파일 크기:
  - `icon_detect/model.pt`: 약 `40.6 MB`
  - `icon_caption/model.safetensors`: 약 `1.08 GB`
- Hugging Face 쪽 README/Space 문서 기준 다운로드 예시는 아래와 같다:

```bash
for f in icon_detect/{train_args.yaml,model.pt,model.yaml} \
         icon_caption/{config.json,generation_config.json,model.safetensors}; do
  huggingface-cli download microsoft/OmniParser-v2.0 "$f" --local-dir weights
done

mv weights/icon_caption weights/icon_caption_florence
```

- 중요한 라이선스 메모:
  - repo 메타데이터는 `mit`로 보이지만, 모델 카드 본문은 하위 컴포넌트 라이선스를 별도로 명시한다
  - `icon_detect`는 AGPL
  - `icon_caption`은 MIT
- private cloud에서 사내 공용 서비스로 붙일 계획이면, 특히 `icon_detect`의 AGPL 조건은 보안/법무 검토가 필요하다

## 브라우저로 수동 다운로드할 때 꼭 받아야 하는 것

### 공통

- `config.json`
- `tokenizer_config.json`
- `tokenizer.json` 또는 `vocab/merges` 계열
- `preprocessor_config.json`
- `model.safetensors.index.json`가 있으면 그 index와 모든 `model-0000x-of-0000y.safetensors`

### 예외

- `MAI-UI-2B`, `UI-Venus-1.5-2B`처럼 단일 `model.safetensors` 구조면 shard 전체를 챙길 필요는 없다
- `OmniParser-v2.0`는 일반 LLM repo처럼 통째로 받는 것보다, 위에 적은 `icon_detect`, `icon_caption` 필수 파일만 받아도 된다

## private cloud 기준 추천 조합

### 최소 실전 세트

- [inclusionAI/UI-Venus-1.5-30B-A3B](https://huggingface.co/inclusionAI/UI-Venus-1.5-30B-A3B)
- [Tongyi-MAI/MAI-UI-8B](https://huggingface.co/Tongyi-MAI/MAI-UI-8B)
- [ByteDance-Seed/UI-TARS-1.5-7B](https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B)
- [microsoft/OmniParser-v2.0](https://huggingface.co/microsoft/OmniParser-v2.0)

### grounding 비교 세트

- [inclusionAI/UI-Venus-1.5-8B](https://huggingface.co/inclusionAI/UI-Venus-1.5-8B)
- [Tongyi-MAI/MAI-UI-8B](https://huggingface.co/Tongyi-MAI/MAI-UI-8B)
- [microsoft/GUI-Actor-7B-Qwen2.5-VL](https://huggingface.co/microsoft/GUI-Actor-7B-Qwen2.5-VL)
- [microsoft/GUI-Actor-Verifier-2B](https://huggingface.co/microsoft/GUI-Actor-Verifier-2B)

### 빠른 smoke test 세트

- [inclusionAI/UI-Venus-1.5-2B](https://huggingface.co/inclusionAI/UI-Venus-1.5-2B)
- [Tongyi-MAI/MAI-UI-2B](https://huggingface.co/Tongyi-MAI/MAI-UI-2B)
- [microsoft/OmniParser-v2.0](https://huggingface.co/microsoft/OmniParser-v2.0)

## 확인한 Hugging Face 페이지

- [ByteDance-Seed/UI-TARS-1.5-7B](https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B)
- [Tongyi-MAI/MAI-UI-8B](https://huggingface.co/Tongyi-MAI/MAI-UI-8B)
- [Tongyi-MAI/MAI-UI-2B](https://huggingface.co/Tongyi-MAI/MAI-UI-2B)
- [MAI-UI collection](https://huggingface.co/collections/Tongyi-MAI/mai-ui)
- [inclusionAI/UI-Venus-1.5-8B](https://huggingface.co/inclusionAI/UI-Venus-1.5-8B)
- [inclusionAI/UI-Venus-1.5-2B](https://huggingface.co/inclusionAI/UI-Venus-1.5-2B)
- [UI-Venus collection](https://huggingface.co/collections/inclusionAI/ui-venus)
- [microsoft/GUI-Actor-7B-Qwen2.5-VL](https://huggingface.co/microsoft/GUI-Actor-7B-Qwen2.5-VL)
- [microsoft/GUI-Actor-Verifier-2B](https://huggingface.co/microsoft/GUI-Actor-Verifier-2B)
- [microsoft/OmniParser-v2.0](https://huggingface.co/microsoft/OmniParser-v2.0)
