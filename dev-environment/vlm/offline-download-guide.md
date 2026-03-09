---
tags: [vlm, huggingface, offline, firewalled, download]
level: beginner
last_updated: 2026-03-09
status: in-progress
---

# 오프라인 다운로드 & 폐쇄망 전송 가이드

> 외부 인터넷이 되는 PC에서 모델을 받고, USB/외장하드로 옮긴 뒤, private cloud 로컬 경로까지 올리는 기본 워크플로우

## 이 문서의 역할

- 이 문서는 `어떻게 다운로드하고 옮길지`만 다룬다.
- 어떤 모델을 받을지는 [Hugging Face 다운로드 shortlist](./huggingface-private-cloud-downloads.md)에서 먼저 정한다.
- SSH/SCP/rsync가 막혀 있으면 [SSH 없이 Private Cloud에 대용량 VLM 파일 올리기](./private-cloud-upload-without-ssh.md)를 본다.

## 전체 흐름

```text
[외부 PC]
1. 다운로드 대상 결정
2. huggingface-cli 설치
3. 모델 다운로드
4. 파일 무결성 확인
5. USB/외장하드로 복사

                |
                v

[폐쇄망 내부]
6. 내부 PC로 반입
7. private cloud 업로드
8. 클라우드에서 최종 확인
```

## Step 1. 다운로드 대상 확정

먼저 [Hugging Face 다운로드 shortlist](./huggingface-private-cloud-downloads.md)에서 받을 세트를 정한다.

권장 기준:

- 첫 성공이 목표면 `UI-Venus-1.5-8B`, `MAI-UI-8B`
- `H200 x2` 성능까지 보려면 `UI-Venus-1.5-30B-A3B` 추가
- parser 비교가 필요하면 `OmniParser-v2.0` 추가

## Step 2. 외부 PC에 `huggingface-cli` 설치

```bash
pip install "huggingface_hub[cli]"
huggingface-cli --version
```

메모:

- 이 폴더의 기본 대상은 퍼블릭 모델이라 보통 로그인 없이 가능하다.
- gated 모델을 받을 때만 `huggingface-cli login`이 필요하다.

## Step 3. 저장 디렉터리 준비

```bash
mkdir -p ~/vlm-downloads/models
cd ~/vlm-downloads
```

폴더를 모델별로 분리해 두는 편이 좋다.

```text
~/vlm-downloads/
  models/
    UI-Venus-1.5-8B/
    MAI-UI-8B/
    OmniParser-v2.0/
```

## Step 4. 모델 다운로드

실제 대상은 `shortlist` 문서의 repo id를 사용한다.

### 기본 형식

```bash
huggingface-cli download <repo-id> --local-dir ./models/<folder-name>
```

### 예시

```bash
huggingface-cli download inclusionAI/UI-Venus-1.5-8B \
  --local-dir ./models/UI-Venus-1.5-8B

huggingface-cli download Tongyi-MAI/MAI-UI-8B \
  --local-dir ./models/MAI-UI-8B
```

### 일부 파일만 먼저 받는 경우

```bash
huggingface-cli download inclusionAI/UI-Venus-1.5-8B \
  --include "*.json" "*.txt" "*.py" \
  --local-dir ./models/UI-Venus-1.5-8B
```

대부분의 첫 반입에서는 필요한 파일을 추측해서 줄이기보다, 모델 폴더를 가능한 한 그대로 보관하는 편이 안전하다.

## Step 5. 다운로드 상태 확인

```bash
ls -lh ./models/UI-Venus-1.5-8B
find ./models/UI-Venus-1.5-8B -maxdepth 1 | sort
find ./models -name "*.safetensors" -size 0 -print
```

최소 확인 포인트:

- `config.json`
- `tokenizer_config.json`
- `preprocessor_config.json`
- `generation_config.json`
- `model.safetensors` 또는 shard 파일 전체
- shard 구조면 `model.safetensors.index.json`

## Step 6. 체크섬 저장

반입 전후 비교를 위해 체크섬 파일을 같이 저장하는 편이 좋다.

```bash
cd ./models/UI-Venus-1.5-8B
sha256sum *.safetensors > checksums.sha256
cd -
```

모델이 shard 구조면 모든 shard가 `checksums.sha256`에 포함되어야 한다.

## Step 7. USB/외장하드로 복사

```bash
cp -r ./models /Volumes/USB_DRIVE/vlm-models
```

대용량일 때는 `rsync`가 더 낫다.

```bash
rsync -avh --progress ./models/ /Volumes/USB_DRIVE/vlm-models/
```

## Step 8. 폐쇄망 내부에서 private cloud로 업로드

### SSH/SCP/rsync 사용 가능할 때

```bash
scp -r /path/to/vlm-models/ user@cloud-server:/data/models/
```

이어받기가 필요하면 `rsync`가 낫다.

```bash
rsync -avhP /path/to/vlm-models/ user@cloud-server:/data/models/
```

### SSH/SCP/rsync가 막혀 있을 때

`code-server` drag-and-drop 대신 [SSH 없이 Private Cloud에 대용량 VLM 파일 올리기](./private-cloud-upload-without-ssh.md) 문서의 resumable upload 방식을 따른다.

## Step 9. 클라우드에서 최종 확인

```bash
ls -lh /data/models/UI-Venus-1.5-8B
du -sh /data/models/*
```

체크섬도 다시 확인한다.

```bash
cd /data/models/UI-Venus-1.5-8B
sha256sum -c checksums.sha256
```

모든 파일이 `OK`로 나오면 반입은 끝난다.

## 자주 막히는 지점

### 다운로드가 중간에 끊김

같은 명령을 다시 실행하면 된다. `huggingface-cli`는 기본적으로 resume를 지원한다.

```bash
huggingface-cli download inclusionAI/UI-Venus-1.5-8B \
  --local-dir ./models/UI-Venus-1.5-8B
```

### USB 용량이 부족함

한 번에 전체 세트를 옮기지 말고, 아래처럼 우선순위를 나눈다.

1. `UI-Venus-1.5-8B` 또는 `MAI-UI-8B`
2. `OmniParser-v2.0`
3. 대형 비교 모델

### 일부 파일만 누락된 것 같음

아래 파일이 빠졌는지 먼저 본다.

- `config.json`
- `tokenizer_config.json`
- `preprocessor_config.json`
- `generation_config.json`
- 모든 shard 파일
- `model.safetensors.index.json`

## 다음 문서

1. [모델 서빙 가이드](./serving-guide.md)
2. [Private Cloud에서 모델 다운로드 후 다음 단계](./private-cloud-vllm-next-steps.md)

## 관련 문서

- [이전: Hugging Face 다운로드 shortlist](./huggingface-private-cloud-downloads.md)
- [대안 업로드: SSH 없이 Private Cloud에 대용량 VLM 파일 올리기](./private-cloud-upload-without-ssh.md)
- [위로: VLM 가이드 인덱스](./README.md)
