---
reviewed_on: 2026-10-04
review_status: partial
document_type: learning
tags: [vlm, ui-grounding, gui-agent, models]
level: beginner
last_updated: 2026-03-10
category_major: "개발 환경"
category_middle: "비전 모델 실행"
category_minor: "UI 모델 선택"
note_kind: "학습"
classified_on: "2026-10-05"
---

# UI 특화 VLM 모델 메모

> [!info] 2026-10-04 검토 범위
> UI grounding과 문서 OCR의 용도를 구분했다. 공개 모델 카드의 지원 경로 확인과 사내 GPU 실측은 다르다. 아래 모델별 품질 우열·GPU 수·메모리 설정은 미실측 가설이며 고정 버전 환경과 샘플로 평가해야 한다.

> cloud terminal에서 바로 실험할 모델만 짧게 정리한다.

## 초기 실험 후보 (성능 순위 미검증)

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

## 전용 실행 코드가 필요한 모델

| 모델 | Repo ID | 메모 |
|---|---|---|
| `UI-TARS-1.5-7B` | `ByteDance-Seed/UI-TARS-1.5-7B` | action-heavy agent 실험용, repo 실행 방식 확인 필요 |
| `GUI-Actor-7B-Qwen2.5-VL` | `microsoft/GUI-Actor-7B-Qwen2.5-VL` | repo 전용 실행 코드가 필요할 수 있음 |
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

## 2026-10-04 확인한 지원 조건

`UI-Venus-1.5-8B`와 `MAI-UI-8B` 개발자 모델 카드는 vLLM **0.11.0 이상**, Transformers **4.57.0 이상**의 서빙 예제를 제공한다. 버전 하한을 충족해도 드라이버·CUDA·template·메모리 조합의 성공은 별도 확인한다. 다른 후보의 현재 호환성과 추천도는 미확인이다. GPU 장수는 파라미터 수만으로 정해지지 않는다.

- [UI-Venus 개발자 모델 카드](https://huggingface.co/inclusionAI/UI-Venus-1.5-8B)
- [MAI-UI 개발자 모델 카드](https://huggingface.co/Tongyi-MAI/MAI-UI-8B)
