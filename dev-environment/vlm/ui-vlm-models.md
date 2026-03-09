---
tags: [vlm, ui-grounding, gui-agent, models]
level: beginner
last_updated: 2026-03-10
status: active
---

# UI 특화 VLM 모델 메모

> cloud terminal에서 바로 실험할 모델만 짧게 정리한다.

## 가장 먼저 볼 모델

| 모델 | Repo ID | GPU 가이드 | 추천도 | 메모 |
|---|---|---|---|---|
| `UI-Venus-1.5-8B` | `inclusionAI/UI-Venus-1.5-8B` | 1 GPU | 높음 | 첫 bring-up 기본값 |
| `MAI-UI-8B` | `Tongyi-MAI/MAI-UI-8B` | 1 GPU | 높음 | `UI-Venus` 비교용 baseline |
| `UI-Venus-1.5-30B-A3B` | `inclusionAI/UI-Venus-1.5-30B-A3B` | 2 GPU | 중간 | `8B` 다음 확장 |

## direct `vLLM` 대안

| 모델 | Repo ID | GPU 가이드 | 메모 |
|---|---|---|---|
| `UGround-V1-7B` | `osunlp/UGround-V1-7B` | 1 GPU | grounding 비교용 |
| `MAI-UI-2B` | `Tongyi-MAI/MAI-UI-2B` | 1 GPU | 아주 가벼운 smoke test용 |

## 별도 runtime 쪽 모델

| 모델 | Repo ID | 메모 |
|---|---|---|
| `UI-TARS-1.5-7B` | `ByteDance-Seed/UI-TARS-1.5-7B` | action-heavy agent 실험용 |
| `GUI-Actor-7B-Qwen2.5-VL` | `microsoft/GUI-Actor-7B-Qwen2.5-VL` | dedicated runtime 성격이 강함 |
| `OmniParser-v2.0` | `microsoft/OmniParser-v2.0` | parser stage, direct VLM 아님 |

## 단순 선택 규칙

### 첫 성공이 목표일 때

- `UI-Venus-1.5-8B`
- `MAI-UI-8B`

### H200 2장을 활용하고 싶을 때

- `UI-Venus-1.5-8B`로 먼저 프롬프트와 출력 형식을 고정한다.
- 그 다음 `UI-Venus-1.5-30B-A3B`로 확장한다.

### action model이 꼭 필요할 때

- `UI-TARS`나 `GUI-Actor`를 쓰되, direct `vLLM` bring-up과는 분리해서 본다.

### parser가 먼저 필요할 때

- `OmniParser`를 별도 단계로 둔다.
- direct VLM path와 섞지 않는 편이 디버깅이 쉽다.

## 모델 폴더 메모

cloud에서는 repo id보다 짧은 폴더 이름이 편하다.

```text
/data/models/
  UI-Venus-1.5-8B/
  MAI-UI-8B/
  UI-Venus-1.5-30B-A3B/
  UGround-V1-7B/
```

## 관련 문서

- [VLM Cloud Notes](./README.md)
- [Private Cloud에서 `vLLM` 시작](./private-cloud-vllm-next-steps.md)
