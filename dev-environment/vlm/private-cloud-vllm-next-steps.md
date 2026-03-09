---
tags: [vlm, vllm, private-cloud, h200, requests]
level: beginner
last_updated: 2026-03-10
status: active
---

# Private Cloud에서 `vLLM` 시작

> 상황: 모델 폴더가 이미 cloud에 있고, H200 GPU를 할당받았다.

## Step 1. H200 상태 확인

먼저 GPU가 실제로 보이는지 확인한다.

```bash
nvidia-smi -L
nvidia-smi
```

자주 보는 요약 정보:

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv
```

실시간으로 보고 싶으면:

```bash
watch -n 2 nvidia-smi
```

2 GPU 연결 상태까지 보려면:

```bash
nvidia-smi topo -m
```

PyTorch에서 잡히는지도 본다.

```bash
python - <<'PY'
import torch

print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
for idx in range(torch.cuda.device_count()):
    print(idx, torch.cuda.get_device_name(idx))
PY
```

실행 중 프로세스까지 보고 싶으면:

```bash
nvidia-smi pmon -c 1
```

## Step 2. 패키지와 모델 경로 확인

```bash
vllm --version
python -c "import requests, transformers; print('requests', requests.__version__); print('transformers', transformers.__version__)"
```

모델 폴더도 확인한다.

```bash
MODEL_DIR=/data/models/UI-Venus-1.5-8B

find "$MODEL_DIR" -maxdepth 1 | sort
du -sh "$MODEL_DIR"
```

## Step 3. 오프라인용 환경 변수

```bash
export HF_HUB_OFFLINE=1
export VLLM_API_KEY='change-this-internal-key'
```

설명:

- `export VLLM_API_KEY=...` 는 현재 shell 세션에 환경 변수를 넣는다는 뜻이다.
- 같은 터미널에서 시작한 `vllm serve` 와 Python `requests` 예제는 이 값을 그대로 읽는다.
- 새 터미널을 열면 보통 다시 `export` 해야 한다.
- `bash`, `zsh` 모두 같은 방식으로 동작한다.

## Step 4. 첫 모델 실행

아래 두 명령은 같은 포트 `8000`을 쓰지만, 동시에 실행하는 예시가 아니다.

- `UI-Venus-1.5-8B`를 먼저 띄울 때의 예시
- 또는 `MAI-UI-8B`를 띄울 때의 예시

즉, 둘 중 하나만 `8000`에서 실행한다고 생각하면 된다.

### `UI-Venus-1.5-8B`

```bash
MODEL_DIR=/data/models/UI-Venus-1.5-8B

CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_DIR" \
  --served-model-name ui-venus-8b \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --api-key "$VLLM_API_KEY" \
  --generation-config vllm \
  --limit-mm-per-prompt '{"image":1}'
```

### `MAI-UI-8B`

```bash
MODEL_DIR=/data/models/MAI-UI-8B

CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_DIR" \
  --served-model-name mai-ui-8b \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --api-key "$VLLM_API_KEY" \
  --generation-config vllm \
  --limit-mm-per-prompt '{"image":1}'
```

## Step 5. 포트와 GPU 선택 기준

### 모델 1개만 실행할 때

가장 단순한 방법이다.

- `CUDA_VISIBLE_DEVICES=0` 이면 물리 GPU 0만 사용한다.
- `--port 8000` 이면 서버는 `8000` 포트 하나만 연다.

### 모델 2개를 각각 다른 GPU에 실행할 때

이 경우는 각 프로세스가 서로 다른 GPU와 포트를 가져야 한다.

```bash
export VLLM_API_KEY='change-this-internal-key'

CUDA_VISIBLE_DEVICES=0 vllm serve /data/models/UI-Venus-1.5-8B \
  --served-model-name ui-venus-8b \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --api-key "$VLLM_API_KEY" \
  --generation-config vllm \
  --limit-mm-per-prompt '{"image":1}'

CUDA_VISIBLE_DEVICES=1 vllm serve /data/models/MAI-UI-8B \
  --served-model-name mai-ui-8b \
  --host 0.0.0.0 \
  --port 8001 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --api-key "$VLLM_API_KEY" \
  --generation-config vllm \
  --limit-mm-per-prompt '{"image":1}'
```

이 구조에서는:

- `http://127.0.0.1:8000/v1` 는 `ui-venus-8b`
- `http://127.0.0.1:8001/v1` 는 `mai-ui-8b`

### `--tensor-parallel-size` 는 언제 쓰나

`--tensor-parallel-size 2` 는 모델 1개를 GPU 2장에 나눠서 실행할 때 쓴다.

- 목적은 큰 모델 1개를 여러 GPU가 함께 처리하게 하는 것이다.
- "GPU 하나에 모델 하나"와는 다른 개념이다.
- 메모리만 정확히 반으로 쪼갠다고 보기보다, 모델 계산과 weight를 여러 GPU에 shard해서 돌린다고 이해하는 편이 맞다.

즉:

- 작은 모델 2개를 각각 돌릴 때: `CUDA_VISIBLE_DEVICES=0` 과 `CUDA_VISIBLE_DEVICES=1` 을 따로 사용
- 큰 모델 1개를 2 GPU로 돌릴 때: `CUDA_VISIBLE_DEVICES=0,1` 과 `--tensor-parallel-size 2` 를 같이 사용

### 프로세스 안에서 GPU 번호가 다르게 보일 수 있음

예를 들어 `CUDA_VISIBLE_DEVICES=1` 로 실행하면, 그 프로세스는 물리 GPU 1만 보게 된다.

그래서:

- 운영 관점의 실제 GPU 번호는 `nvidia-smi` 로 확인
- 프로세스 내부에서는 그 GPU가 `cuda:0` 처럼 보일 수 있음

## Step 6. 포트와 프로세스 확인

```bash
ss -ltnp | rg "8000|8001"
ps -ef | rg "vllm serve"
```

## Step 7. `requests`로 모델 목록 확인

```bash
python - <<'PY'
import os
import requests

base_url = "http://127.0.0.1:8000/v1"
headers = {"Authorization": f"Bearer {os.environ['VLLM_API_KEY']}"}

response = requests.get(f"{base_url}/models", headers=headers, timeout=30)
response.raise_for_status()
print(response.json())
PY
```

정상이면 `ui-venus-8b` 또는 `mai-ui-8b`가 보인다.

`8001` 에 `MAI-UI`를 띄운 상태라면 `base_url`을 `http://127.0.0.1:8001/v1`로 바꾼다.

## Step 8. 첫 이미지 요청 보내기

`sample-screen.png` 같은 로컬 파일이 cloud에 있다고 가정한다.

```bash
python - <<'PY'
import base64
import os
from pathlib import Path

import requests

image_path = Path("./sample-screen.png")
image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")

payload = {
    "model": "ui-venus-8b",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Identify the main visible controls. Answer briefly.",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}",
                    },
                },
            ],
        }
    ],
    "temperature": 0,
    "max_tokens": 200,
}

response = requests.post(
    "http://127.0.0.1:8000/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {os.environ['VLLM_API_KEY']}",
        "Content-Type": "application/json",
    },
    json=payload,
    timeout=180,
)
response.raise_for_status()
print(response.json()["choices"][0]["message"]["content"])
PY
```

`MAI-UI-8B`를 띄운 상태면 payload의 `model`만 `mai-ui-8b`로 바꾼다.

`8001` 에 `MAI-UI`를 띄운 상태면 요청 URL도 `http://127.0.0.1:8001/v1/chat/completions` 로 바꾼다.

## Step 9. H200 2장으로 확장

`UI-Venus-1.5-30B-A3B`는 single GPU first path가 끝난 뒤에 본다.

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve /data/models/UI-Venus-1.5-30B-A3B \
  --served-model-name ui-venus-30b \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --api-key "$VLLM_API_KEY" \
  --generation-config vllm \
  --limit-mm-per-prompt '{"image":1}'
```

이 명령은 GPU 0과 1이 함께 모델 1개를 담당하는 경우다.

- `CUDA_VISIBLE_DEVICES=0,1`: 이 프로세스가 두 GPU를 모두 본다.
- `--tensor-parallel-size 2`: 두 GPU에 모델을 분산해서 실행한다.
- 이 상태는 "VLM 두 개를 동시에 띄운다"가 아니라 "VLM 하나를 GPU 두 장으로 띄운다"에 가깝다.

확장 전에 다시 GPU 요약을 본다.

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
```

## 자주 막히는 지점

### GPU는 보이는데 로딩이 실패할 때

- 모델 폴더에 shard가 모두 있는지 본다.
- `transformers`가 너무 낮지 않은지 본다.
- `--trust-remote-code`가 빠지지 않았는지 본다.

### 메모리가 빠듯할 때

- 먼저 `8B` 모델로 다시 확인한다.
- `--gpu-memory-utilization 0.85`로 낮춰 본다.
- 2 GPU 모델이면 `--tensor-parallel-size 2`를 확인한다.

### endpoint 응답이 없을 때

- `ss -ltnp | rg 8000`
- `ps -ef | rg "vllm serve"`
- `requests` 예제의 `base_url`과 `VLLM_API_KEY`

## 관련 문서

- [VLM Cloud Notes](./README.md)
- [UI 특화 VLM 모델 메모](./ui-vlm-models.md)
- [로컬 PC에서 `requests`로 이미지 보내기](./local-pc-vllm-image-guide.md)
