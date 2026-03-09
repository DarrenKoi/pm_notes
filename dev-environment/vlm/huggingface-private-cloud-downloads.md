---
tags: [vlm, huggingface, private-cloud, gui-agent]
level: intermediate
last_updated: 2026-03-09
status: in-progress
---

# Hugging Face 다운로드 shortlist for private cloud

> 목표: `H200 x2`가 있는 private cloud에 무엇을 먼저 반입할지 빠르게 결정한다.

## 이 문서의 역할

- 이 문서는 `무엇을 받을지`만 정한다.
- 실제 다운로드/전송 절차는 [오프라인 다운로드 & 폐쇄망 전송 가이드](./offline-download-guide.md)를 따른다.
- 전체 모델 비교는 [UI 특화 VLM 모델 카탈로그](./ui-vlm-models.md)를 참고한다.

## 먼저 결론

- 첫 실전 후보는 `UI-Venus-1.5-8B`와 `MAI-UI-8B`다.
- `H200 x2`를 제대로 활용하려면 다음 단계에서 `UI-Venus-1.5-30B-A3B`를 추가한다.
- `UI-TARS-1.5-7B`는 에이전트 비교용으로 좋지만, pure `vLLM`만으로 끝나지 않을 수 있다.
- `GUI-Actor`는 전용 runtime 검토가 필요하다.
- `OmniParser V2`는 단독 에이전트가 아니라 parser 컴포넌트다.

## 추천 세트

### 1. 빠른 첫 실험 세트

가장 적은 시행착오로 첫 성공 경로를 확인할 때:

| 우선순위 | Repo ID | 용도 | 대략 용량 |
|---|---|---|---|
| 1 | `inclusionAI/UI-Venus-1.5-8B` | 기본 `vLLM` smoke test | ~18GB |
| 2 | `Tongyi-MAI/MAI-UI-8B` | 비교용 baseline | ~16GB |
| 3 | `microsoft/OmniParser-v2.0` | parser pipeline 비교 | ~1GB+ |

### 2. 최고 성능 비교 세트

정확도 기준으로 바로 상위권 조합을 보고 싶을 때:

| 우선순위 | Repo ID | 용도 | 대략 용량 |
|---|---|---|---|
| 1 | `inclusionAI/UI-Venus-1.5-30B-A3B` | `H200 x2` 메인 후보 | ~60GB |
| 2 | `inclusionAI/UI-Venus-1.5-8B` | 단일 GPU baseline | ~18GB |
| 3 | `Tongyi-MAI/MAI-UI-8B` | 대안 비교 | ~16GB |

### 3. 에이전트 비교 세트

그라운딩뿐 아니라 액션/추론형 모델도 같이 보려면:

| 우선순위 | Repo ID | 용도 | 대략 용량 |
|---|---|---|---|
| 1 | `ByteDance-Seed/UI-TARS-1.5-7B` | 에이전트 비교 | ~33GB |
| 2 | `microsoft/GUI-Actor-7B-Qwen2.5-VL` | runtime 분리형 비교 | ~17GB |
| 3 | `microsoft/GUI-Actor-Verifier-2B` | verifier 옵션 | 추가 수 GB |

### 4. 빠른 스모크 테스트 세트

작은 모델로 경로만 먼저 확인할 때:

| 우선순위 | Repo ID | 용도 | 대략 용량 |
|---|---|---|---|
| 1 | `inclusionAI/UI-Venus-1.5-2B` | 최소 `UI-Venus` 테스트 | ~4GB |
| 2 | `Tongyi-MAI/MAI-UI-2B` | 최소 `MAI-UI` 테스트 | ~4GB |
| 3 | `microsoft/OmniParser-v2.0` | parser 확인 | ~1GB+ |

## 모델별 판단 메모

| 모델 | 지금 받는 이유 | 주의할 점 |
|---|---|---|
| `UI-Venus-1.5-8B` | 가장 무난한 첫 `vLLM` 후보 | 기본 baseline으로 같이 두기 좋다 |
| `UI-Venus-1.5-30B-A3B` | `H200 x2` 자원을 살리기 좋다 | 처음부터 이 모델만 받기보다 `8B`도 같이 두는 편이 디버깅이 쉽다 |
| `MAI-UI-8B` | `UI-Venus`와 상위권 비교 가능 | private cloud 첫 A/B 비교 대상으로 적합 |
| `UI-TARS-1.5-7B` | 액션 중심 비교에 좋다 | GitHub runtime/app 확인이 필요할 수 있다 |
| `GUI-Actor-7B` | 좌표 없는 그라운딩 실험용 | HF weights만으로 끝나는 흐름이 아니다 |
| `OmniParser-v2.0` | parser + 내부 LLM 조합 검토용 | `icon_detect` 라이선스 검토가 필요하다 |

## 실제 다운로드 명령

이 문서에서는 세트별 명령만 둔다. 상세 다운로드 절차와 전송 절차는 [오프라인 다운로드 & 폐쇄망 전송 가이드](./offline-download-guide.md)에 있다.

### 빠른 첫 실험 세트

```bash
huggingface-cli download inclusionAI/UI-Venus-1.5-8B \
  --local-dir ./models/UI-Venus-1.5-8B

huggingface-cli download Tongyi-MAI/MAI-UI-8B \
  --local-dir ./models/MAI-UI-8B

huggingface-cli download microsoft/OmniParser-v2.0 \
  --local-dir ./models/OmniParser-v2.0
```

### 최고 성능 비교 세트

```bash
huggingface-cli download inclusionAI/UI-Venus-1.5-30B-A3B \
  --local-dir ./models/UI-Venus-1.5-30B-A3B

huggingface-cli download inclusionAI/UI-Venus-1.5-8B \
  --local-dir ./models/UI-Venus-1.5-8B

huggingface-cli download Tongyi-MAI/MAI-UI-8B \
  --local-dir ./models/MAI-UI-8B
```

### 에이전트 비교 세트

```bash
huggingface-cli download ByteDance-Seed/UI-TARS-1.5-7B \
  --local-dir ./models/UI-TARS-1.5-7B

huggingface-cli download microsoft/GUI-Actor-7B-Qwen2.5-VL \
  --local-dir ./models/GUI-Actor-7B-Qwen2.5-VL

huggingface-cli download microsoft/GUI-Actor-Verifier-2B \
  --local-dir ./models/GUI-Actor-Verifier-2B
```

## 브라우저 수동 다운로드 시 최소 확인 항목

- `config.json`
- `tokenizer_config.json`
- `tokenizer.json` 또는 vocab 파일
- `preprocessor_config.json`
- `generation_config.json`
- `model.safetensors` 또는 모든 shard 파일
- shard 구조면 `model.safetensors.index.json`

`OmniParser-v2.0`는 일반 LLM repo와 다르게 필요한 하위 파일만 선별해서 받는 편이 낫다. 자세한 내용은 [OmniParser V2 설치 및 Cloud API 패턴](./omniparser-cloud-api-guide.md)에서 정리한다.

## 다음 문서

1. [오프라인 다운로드 & 폐쇄망 전송 가이드](./offline-download-guide.md)
2. [모델 서빙 가이드](./serving-guide.md)

## 관련 문서

- [이전: VLM 가이드 인덱스](./README.md)
- [참고: UI 특화 VLM 모델 카탈로그](./ui-vlm-models.md)
