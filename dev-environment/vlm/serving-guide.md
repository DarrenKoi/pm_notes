---
tags: [vlm, vllm, sglang, serving, openai-api]
level: intermediate
last_updated: 2026-03-09
status: in-progress
---

# 모델 서빙 가이드

> 목표: private cloud에 모델 파일을 올린 뒤, 어떤 서빙 트랙으로 갈지 먼저 결정한다.

## 이 문서의 역할

- 이 문서는 `어떤 서빙 방식이 맞는지`를 고르는 문서다.
- 정확한 `vLLM` smoke test 절차는 [Private Cloud에서 모델 다운로드 후 다음 단계](./private-cloud-vllm-next-steps.md)에 둔다.
- Flask API나 OmniParser wrapper 예시는 구현 문서에서만 다룬다.

## 가장 먼저 판단할 것

| 질문 | 추천 경로 |
|---|---|
| `UI-Venus` 또는 `MAI-UI`를 바로 OpenAI 호환 API로 쓰고 싶은가 | `vLLM direct path` |
| 멀티턴/특수 추론 서버가 더 중요한가 | `SGLang` 검토 |
| 모델 카드가 전용 Python runtime을 요구하는가 | `dedicated runtime path` |
| 스크린샷을 먼저 요소 목록으로 파싱하고 싶은가 | `OmniParser path` |

## 트랙별 정리

### 1. `vLLM direct path`

가장 기본이고, 지금 폴더의 기본 추천 경로다.

잘 맞는 모델:

- `UI-Venus-1.5-8B`
- `UI-Venus-1.5-30B-A3B`
- `MAI-UI-8B`
- `MAI-UI-2B`

장점:

- OpenAI 호환 API를 바로 제공한다
- private cloud 운영 문서가 단순해진다
- 첫 smoke test가 가장 쉽다

다음 문서:

- [Private Cloud에서 모델 다운로드 후 다음 단계](./private-cloud-vllm-next-steps.md)

### 2. `SGLang` path

멀티턴 상호작용이나 별도 최적화 포인트가 중요할 때 검토한다.

장점:

- 멀티턴 파이프라인에서 유리할 수 있다
- 일부 시나리오에서 지연시간 관리가 쉽다

주의:

- 이 폴더의 기준 문서는 `vLLM` 우선이다
- 첫 성공 경로로는 `SGLang`보다 `vLLM`이 단순하다

### 3. dedicated runtime path

HF weights만 `vllm serve /path/to/model`로 끝나지 않는 모델들이다.

대표 예:

- `GUI-Actor`
- 일부 `UI-TARS` 사용 시나리오

이 경우 확인할 것:

- Hugging Face 카드의 `Usage` 예시
- 연결된 GitHub runtime 코드
- custom prompt/post-processing 요구사항

운영 원칙:

- `vLLM` 트랙과 별도 문서/별도 프로세스로 보는 편이 안전하다
- weights와 runtime 코드를 같이 버전 고정하는 편이 낫다

### 4. `OmniParser path`

`OmniParser`는 독립 VLM 서빙보다 parser stage에 가깝다.

추천 구조:

1. `OmniParser`가 스크린샷을 파싱한다
2. 내부 LLM 또는 VLM이 파싱 결과를 해석한다
3. 필요하면 좌표/액션을 후처리한다

다음 문서:

- [OmniParser V2 설치 및 Cloud API 패턴](./omniparser-cloud-api-guide.md)

## 오프라인 환경 준비 메모

private cloud가 외부 인터넷에 직접 못 나가는 경우, 패키지 준비 방법도 미리 정해야 한다.

### 방법 1. wheel 다운로드

```bash
mkdir vllm-packages
pip download vllm -d ./vllm-packages/
```

### 방법 2. conda-pack

```bash
conda create -n vlm-serve python=3.11
conda activate vlm-serve
pip install vllm
conda pack -n vlm-serve -o vlm-serve.tar.gz
```

운영 기준:

- 첫 실험은 `vLLM`만 먼저 준비
- `SGLang`, `OmniParser`, 전용 runtime은 성공 경로가 분리된 뒤 추가

## 추천 흐름

1. [Hugging Face 다운로드 shortlist](./huggingface-private-cloud-downloads.md)에서 세트를 정한다.
2. [오프라인 다운로드 & 폐쇄망 전송 가이드](./offline-download-guide.md)대로 모델을 반입한다.
3. `UI-Venus-1.5-8B` 또는 `MAI-UI-8B`면 `vLLM direct path`로 간다.
4. 첫 API 응답이 나온 뒤에만 `30B`, `UI-TARS`, `GUI-Actor`, `OmniParser`로 확장한다.

## 트랙 선택 요약

| 대상 | 1차 추천 | 이유 |
|---|---|---|
| `UI-Venus` | `vLLM` | 가장 단순한 첫 성공 경로 |
| `MAI-UI` | `vLLM` | baseline 비교가 쉽다 |
| `UI-TARS` | 상황별 | runtime/app 요구를 같이 확인해야 한다 |
| `GUI-Actor` | dedicated runtime | 모델 카드 예시가 전용 코드 의존 |
| `OmniParser` | parser service | 독립 parser 컴포넌트로 보는 편이 맞다 |

## 다음 문서

1. [Private Cloud에서 모델 다운로드 후 다음 단계](./private-cloud-vllm-next-steps.md)
2. [UI-Venus-1.5-8B Cloud API Guide with Flask Blueprint](./ui-venus-flask-blueprint-cloud-guide.md)
3. [OmniParser V2 설치 및 Cloud API 패턴](./omniparser-cloud-api-guide.md)

## 관련 문서

- [이전: 오프라인 다운로드 & 폐쇄망 전송 가이드](./offline-download-guide.md)
- [위로: VLM 가이드 인덱스](./README.md)
