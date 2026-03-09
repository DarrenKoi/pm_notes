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

## Step 4. 첫 모델 실행

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

## Step 5. 포트와 프로세스 확인

```bash
ss -ltnp | rg 8000
ps -ef | rg "vllm serve"
```

## Step 6. `requests`로 모델 목록 확인

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

## Step 7. 첫 이미지 요청 보내기

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

## Step 8. H200 2장으로 확장

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
