---
tags: [vlm, huggingface, offline, firewalled, download]
level: beginner
last_updated: 2026-03-04
status: in-progress
---

# 오프라인 다운로드 & 폐쇄망 전송 가이드

> 외부 PC에서 HuggingFace 모델을 다운로드하고 폐쇄망 내부 Linux 클라우드로 전송하는 전체 워크플로우

## 왜 필요한가? (Why)

- 회사 폐쇄망에서는 외부 인터넷 접근 불가
- 모델 파일을 외부에서 다운로드 → 물리적 매체로 반입 → 내부 클라우드에 업로드하는 과정이 필요
- 모델 파일은 수 GB~수십 GB 규모이므로 효율적인 다운로드 및 전송 방법이 중요

## 전체 워크플로우

```
[외부 PC (Mac)]                    [폐쇄망 내부]

1. huggingface-cli 설치
2. 모델 다운로드
3. USB/외장하드에 복사     ──────→  4. 내부 PC에 복사
                                   5. scp/rsync로 클라우드 업로드
                                   6. vLLM으로 모델 서빙
```

---

## Step 1: 외부 PC에 huggingface-cli 설치

```bash
# pip으로 설치 (Python 3.8+)
pip install huggingface_hub[cli]

# 설치 확인
huggingface-cli --version

# 로그인 없이 사용 가능 — 이 가이드의 모든 모델은 퍼블릭
# 로그인이 필요한 경우(Gated 모델)에만 아래 실행
# huggingface-cli login
```

> 이 가이드에 포함된 모든 UI 특화 VLM은 **퍼블릭 모델**이므로 로그인 불필요

---

## Step 2: 모델 다운로드

### 방법 1: huggingface-cli (추천)

```bash
# 기본 사용법
huggingface-cli download <모델경로> --local-dir <저장경로>

# 다운로드 중 끊겨도 이어받기 가능 (자동 resume)
```

### 방법 2: git lfs clone (대안)

```bash
# Git LFS 설치 필요
brew install git-lfs  # macOS
git lfs install

# 모델 클론 (전체 이력 포함, 더 느림)
git clone https://huggingface.co/inclusionAI/UI-Venus-1.5-8B
```

### 방법 3: 브라우저 직접 다운로드

HuggingFace 웹사이트에서 개별 파일을 직접 다운로드할 수도 있음:
1. `https://huggingface.co/<모델경로>/tree/main` 접속
2. 각 파일 옆 다운로드 버튼 클릭
3. 대용량 모델은 파일이 여러 개로 분할되어 있으므로 **모든 파일을 빠짐없이** 다운로드해야 함

> 파일 수가 많은 대형 모델은 huggingface-cli가 가장 안정적

---

## Step 3: 모델별 다운로드 명령어 & 예상 용량

### Tier 1: 최고 성능

```bash
# UI-Venus-1.5-8B (~18GB)
huggingface-cli download inclusionAI/UI-Venus-1.5-8B \
    --local-dir ./models/UI-Venus-1.5-8B

# UI-Venus-1.5-30B-A3B MoE (~60GB 예상)
huggingface-cli download inclusionAI/UI-Venus-1.5-30B-A3B \
    --local-dir ./models/UI-Venus-1.5-30B-A3B

# UI-Venus-1.5-2B (~4GB)
huggingface-cli download inclusionAI/UI-Venus-1.5-2B \
    --local-dir ./models/UI-Venus-1.5-2B

# MAI-UI-8B (~16GB)
huggingface-cli download Tongyi-MAI/MAI-UI-8B \
    --local-dir ./models/MAI-UI-8B

# MAI-UI-2B (~4GB)
huggingface-cli download Tongyi-MAI/MAI-UI-2B \
    --local-dir ./models/MAI-UI-2B
```

### Tier 2: 강력한 대안

```bash
# GUI-Actor-7B (~14GB)
huggingface-cli download microsoft/GUI-Actor-7B-Qwen2.5-VL \
    --local-dir ./models/GUI-Actor-7B

# UI-TARS-1.5-7B (~14GB)
huggingface-cli download ByteDance-Seed/UI-TARS-1.5-7B \
    --local-dir ./models/UI-TARS-1.5-7B

# UI-TARS-72B-DPO (~144GB) — H200 x2로 실행 가능
huggingface-cli download ByteDance-Seed/UI-TARS-72B-DPO \
    --local-dir ./models/UI-TARS-72B-DPO

# UGround-V1-7B (~14GB)
huggingface-cli download osunlp/UGround-V1-7B \
    --local-dir ./models/UGround-V1-7B
```

### Tier 3: 경량 모델

```bash
# ShowUI-2B (~4GB)
huggingface-cli download showlab/ShowUI-2B \
    --local-dir ./models/ShowUI-2B

# SeeClick (~20GB, Qwen-VL 기반)
huggingface-cli download cckevinn/SeeClick \
    --local-dir ./models/SeeClick

# ZonUI-3B (~6GB)
huggingface-cli download zonghanHZH/ZonUI-3B \
    --local-dir ./models/ZonUI-3B

# Aria-UI-base (~8GB 예상, MoE)
huggingface-cli download Aria-UI/Aria-UI-base \
    --local-dir ./models/Aria-UI-base
```

### 파싱 파이프라인

```bash
# OmniParser V2 (~1GB, YOLO + Florence-2)
huggingface-cli download microsoft/OmniParser-v2.0 \
    --local-dir ./models/OmniParser-v2.0
```

### 참고: 기타 모델

```bash
# OS-Atlas-7B (~14GB)
huggingface-cli download OS-Copilot/OS-Atlas-Base-7B \
    --local-dir ./models/OS-Atlas-Base-7B

# CogAgent-9B (~18GB)
huggingface-cli download THUDM/cogagent-9b-20241220 \
    --local-dir ./models/CogAgent-9B
```

---

## Step 4: 예상 총 용량

| 모델 | 예상 용량 |
|------|-----------|
| UI-Venus-1.5-8B | ~18GB |
| UI-Venus-1.5-30B-A3B | ~60GB |
| UI-Venus-1.5-2B | ~4GB |
| MAI-UI-8B | ~16GB |
| MAI-UI-2B | ~4GB |
| GUI-Actor-7B | ~14GB |
| UI-TARS-1.5-7B | ~14GB |
| UI-TARS-72B-DPO | ~144GB |
| UGround-V1-7B | ~14GB |
| ShowUI-2B | ~4GB |
| SeeClick | ~20GB |
| ZonUI-3B | ~6GB |
| Aria-UI-base | ~8GB |
| OmniParser V2 | ~1GB |
| OS-Atlas-7B | ~14GB |
| CogAgent-9B | ~18GB |
| **합계** | **~360GB** |

> 전부 다운로드할 필요 없음. 우선 Tier 1(UI-Venus-8B, MAI-UI-8B)부터 시작 추천

### 우선 다운로드 추천 (최소 세트, ~50GB)

```bash
# 최우선: 최고 성능 비교
huggingface-cli download inclusionAI/UI-Venus-1.5-8B --local-dir ./models/UI-Venus-1.5-8B
huggingface-cli download Tongyi-MAI/MAI-UI-8B --local-dir ./models/MAI-UI-8B

# 에이전트 비교
huggingface-cli download ByteDance-Seed/UI-TARS-1.5-7B --local-dir ./models/UI-TARS-1.5-7B

# 파싱 파이프라인 비교
huggingface-cli download microsoft/OmniParser-v2.0 --local-dir ./models/OmniParser-v2.0
```

---

## Step 5: 무결성 확인

다운로드 완료 후 파일 무결성 확인:

```bash
# 각 모델 폴더에 들어가서 파일 목록 확인
ls -lh ./models/UI-Venus-1.5-8B/

# safetensors 파일이 정상적으로 다운로드되었는지 확인
# 파일 크기가 0이거나 비정상적으로 작으면 재다운로드 필요
find ./models/ -name "*.safetensors" -size 0 -print

# SHA256 체크섬 확인 (HuggingFace 페이지에서 해시 확인 가능)
sha256sum ./models/UI-Venus-1.5-8B/*.safetensors
```

---

## Step 6: 폐쇄망으로 전송

### USB/외장 하드 디스크 사용

```bash
# 외부 PC에서 → USB로 복사
cp -r ./models/ /Volumes/USB_DRIVE/vlm-models/

# 또는 rsync로 진행률 표시와 함께 복사 (대용량 권장)
rsync -avh --progress ./models/ /Volumes/USB_DRIVE/vlm-models/
```

### 폐쇄망 내부 PC에서 → 클라우드로 업로드

```bash
# scp로 클라우드에 업로드
scp -r /path/to/vlm-models/ user@cloud-server:/data/models/

# 또는 rsync (이어받기 가능, 대용량 권장)
rsync -avhP /path/to/vlm-models/ user@cloud-server:/data/models/
```

> `rsync -avhP`에서 `-P`는 진행률 표시 + 중단 시 이어받기 지원

### 클라우드에서 파일 확인

```bash
# SSH 접속
ssh user@cloud-server

# 모델 파일 확인
ls -lh /data/models/UI-Venus-1.5-8B/

# 디스크 사용량 확인
du -sh /data/models/*
```

---

## Step 7: 전송 후 무결성 재확인

```bash
# 외부 PC에서 체크섬 생성
cd ./models/UI-Venus-1.5-8B/
sha256sum *.safetensors > checksums.sha256

# 체크섬 파일도 함께 전송

# 클라우드에서 검증
cd /data/models/UI-Venus-1.5-8B/
sha256sum -c checksums.sha256
# 모든 파일에 "OK" 표시되면 성공
```

---

## 트러블슈팅

### 다운로드가 중간에 끊긴 경우

```bash
# huggingface-cli는 자동으로 이어받기 지원
# 같은 명령어를 다시 실행하면 됨
huggingface-cli download inclusionAI/UI-Venus-1.5-8B \
    --local-dir ./models/UI-Venus-1.5-8B
```

### 특정 파일만 다운로드하고 싶은 경우

```bash
# config.json, tokenizer 등 설정 파일만 먼저 다운로드
huggingface-cli download inclusionAI/UI-Venus-1.5-8B \
    --include "*.json" "*.txt" "*.py" \
    --local-dir ./models/UI-Venus-1.5-8B

# 모델 가중치만 다운로드
huggingface-cli download inclusionAI/UI-Venus-1.5-8B \
    --include "*.safetensors" \
    --local-dir ./models/UI-Venus-1.5-8B
```

### USB 용량이 부족한 경우

```bash
# 여러 번 나누어 전송
# 1차: Tier 1 모델만 (~40GB)
# 2차: Tier 2 모델 (~50GB)
# 3차: 나머지
```

## 참고 자료

- [HuggingFace CLI 문서](https://huggingface.co/docs/huggingface_hub/guides/cli)
- [Git LFS 설치 가이드](https://git-lfs.com/)

## 관련 문서

- [이전: UI VLM 모델 카탈로그](./ui-vlm-models.md)
- [다음: 모델 서빙 가이드](./serving-guide.md)
- [위로: VLM 가이드 인덱스](./README.md)
