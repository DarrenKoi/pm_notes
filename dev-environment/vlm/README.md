---
tags: [vlm, ui-automation, gui-agent, offline-deployment]
level: intermediate
last_updated: 2026-03-09
status: in-progress
---

# UI 특화 VLM 설치 및 운영 가이드

> 폐쇄망 환경에서 UI 특화 VLM을 선택하고, 다운로드하고, private cloud에 올려서, 실제 API로 쓰기까지의 순서를 정리한 인덱스

## 이 폴더를 읽는 순서

### Step 1. 먼저 어떤 모델을 받을지 결정

- [Hugging Face 다운로드 shortlist](./huggingface-private-cloud-downloads.md)
- [UI 특화 VLM 모델 카탈로그](./ui-vlm-models.md)

먼저 `shortlist`를 읽고, 세부 비교가 필요할 때만 `카탈로그`를 참고하는 방식이 가장 빠르다.

### Step 2. 외부 PC에서 모델 다운로드

- [오프라인 다운로드 & 폐쇄망 전송 가이드](./offline-download-guide.md)

이 문서는 `huggingface-cli`, 무결성 확인, USB/외장하드 복사, 클라우드 업로드까지 포함한 기본 워크플로우의 기준 문서다.

### Step 3. 내부 클라우드로 업로드

- 기본 경로: [오프라인 다운로드 & 폐쇄망 전송 가이드](./offline-download-guide.md)
- SSH/SCP/rsync 불가 시: [SSH 없이 Private Cloud에 대용량 VLM 파일 올리기](./private-cloud-upload-without-ssh.md)

### Step 4. 어떤 서빙 경로로 갈지 결정

- [모델 서빙 가이드](./serving-guide.md)

이 문서는 `vLLM`, `SGLang`, 전용 runtime, parser pipeline 중 어떤 트랙이 맞는지 고르는 문서다.

### Step 5. local path 기준으로 첫 서빙 성공

- [Private Cloud에서 모델 다운로드 후 다음 단계](./private-cloud-vllm-next-steps.md)

`UI-Venus` 또는 `MAI-UI`를 private cloud 로컬 경로에서 `vLLM`으로 띄우고 `/v1/models`, `/v1/chat/completions`까지 확인하는 실전 체크리스트다.

### Step 6. 필요하면 서비스 레이어 추가

- Flask wrapper 예시: [UI-Venus-1.5-8B Cloud API Guide with Flask Blueprint](./ui-venus-flask-blueprint-cloud-guide.md)
- Parser API 예시: [OmniParser V2 설치 및 Cloud API 패턴](./omniparser-cloud-api-guide.md)

이 두 문서는 기본 서빙이 끝난 뒤에 읽는 구현 예시다. 모델 다운로드나 기본 `vLLM` 명령은 여기서 다시 설명하지 않는다.

## 문서 역할 정리

| 문서 | 역할 | 먼저 읽어야 하나 |
|---|---|---|
| [README](./README.md) | 전체 순서 안내 | 예 |
| [huggingface-private-cloud-downloads.md](./huggingface-private-cloud-downloads.md) | 무엇을 받을지 결정 | 예 |
| [ui-vlm-models.md](./ui-vlm-models.md) | 전체 모델 레퍼런스 | 필요할 때 |
| [offline-download-guide.md](./offline-download-guide.md) | 다운로드/전송 기준 문서 | 예 |
| [private-cloud-upload-without-ssh.md](./private-cloud-upload-without-ssh.md) | SSH 없는 업로드 대안 | 조건부 |
| [serving-guide.md](./serving-guide.md) | 서빙 트랙 선택 | 예 |
| [private-cloud-vllm-next-steps.md](./private-cloud-vllm-next-steps.md) | 첫 `vLLM` 성공 절차 | 예 |
| [ui-venus-flask-blueprint-cloud-guide.md](./ui-venus-flask-blueprint-cloud-guide.md) | Flask API 예시 구현 | 선택 |
| [omniparser-cloud-api-guide.md](./omniparser-cloud-api-guide.md) | OmniParser API 예시 구현 | 선택 |

## 중복 정리 원칙

- 모델 선택은 `shortlist`와 `카탈로그`에서만 관리한다.
- 다운로드와 전송 명령은 `offline-download-guide.md`를 기준으로 둔다.
- 첫 `vLLM` smoke test 명령은 `private-cloud-vllm-next-steps.md`를 기준으로 둔다.
- Flask/OmniParser 문서는 구현 예시만 다루고, 앞단 워크플로우는 링크로만 연결한다.

## 빠른 시작

가장 빨리 성공 경로를 확인하려면 아래 순서만 따라가면 된다.

1. [Hugging Face 다운로드 shortlist](./huggingface-private-cloud-downloads.md)에서 `빠른 첫 실험 세트`를 고른다.
2. [오프라인 다운로드 & 폐쇄망 전송 가이드](./offline-download-guide.md)로 외부 PC에서 다운로드하고 private cloud에 올린다.
3. [모델 서빙 가이드](./serving-guide.md)에서 `vLLM direct path`가 맞는지 확인한다.
4. [Private Cloud에서 모델 다운로드 후 다음 단계](./private-cloud-vllm-next-steps.md)대로 `UI-Venus-1.5-8B` 또는 `MAI-UI-8B`를 띄운다.
5. 필요하면 Flask 또는 OmniParser 예시 문서로 넘어간다.

## 관련 문서

- [위로: 개발 환경](../README.md)
- [루트 README](../../README.md)
