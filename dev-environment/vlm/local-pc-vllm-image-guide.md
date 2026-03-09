---
tags: [vlm, vllm, requests, local-client]
level: beginner
last_updated: 2026-03-10
status: active
---

# 로컬 PC에서 `requests`로 이미지 보내기

> 상황: cloud의 `vLLM`은 이미 실행 중이고, 로컬 PC에서 스크린샷을 보내고 싶다.

## 가장 쉬운 방법

이 폴더의 [send_image_to_vllm.py](./send_image_to_vllm.py) 스크립트를 쓴다.

이 스크립트는:

- `requests`로 `/v1/models`를 조회한다.
- 모델 이름이 없으면 첫 모델을 자동 선택한다.
- 로컬 이미지 파일을 base64 data URL로 바꿔 `/v1/chat/completions`로 보낸다.

## Step 1. 모델 목록 확인

```bash
python3 dev-environment/vlm/send_image_to_vllm.py \
  --base-url "http://your-cloud-host:8000" \
  --list-models
```

proxy 경로를 쓰는 환경이면 `http://host/proxy/8000`처럼 넣어도 된다.

## Step 2. 이미지 한 장 보내기

```bash
python3 dev-environment/vlm/send_image_to_vllm.py \
  --base-url "http://your-cloud-host:8000" \
  --image "/Users/yourname/Desktop/screen.png" \
  --prompt "Read this screen and tell me the main visible UI elements."
```

## Step 3. 모델 이름을 고정하고 싶을 때

```bash
python3 dev-environment/vlm/send_image_to_vllm.py \
  --base-url "http://your-cloud-host:8000" \
  --model "ui-venus-8b" \
  --image "/Users/yourname/Desktop/screen.png" \
  --prompt "Identify the settings button."
```

## Step 4. API key가 있을 때

```bash
export VLLM_API_KEY="your-key"
python3 dev-environment/vlm/send_image_to_vllm.py \
  --base-url "http://your-cloud-host:8000" \
  --image "/Users/yourname/Desktop/screen.png"
```

## 직접 `requests` 코드를 쓰고 싶을 때

```python
import base64
from pathlib import Path

import requests

image_path = Path("/Users/yourname/Desktop/screen.png")
image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")

payload = {
    "model": "ui-venus-8b",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the main UI briefly."},
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
    "http://your-cloud-host:8000/v1/chat/completions",
    headers={"Authorization": "Bearer your-key"},
    json=payload,
    timeout=180,
)
response.raise_for_status()
print(response.json()["choices"][0]["message"]["content"])
```

## 자주 막히는 지점

### `/v1/models`부터 확인하고 싶을 때

`--list-models`가 가장 빠른 확인 방법이다.

### 인증서 문제가 있을 때

임시로 `--insecure`를 붙여 테스트할 수 있다.

```bash
python3 dev-environment/vlm/send_image_to_vllm.py \
  --base-url "https://your-cloud-host:8000" \
  --image "/Users/yourname/Desktop/screen.png" \
  --insecure
```

### 404 또는 502가 날 때

- cloud에서 `vLLM`이 실제로 떠 있는지
- proxy가 `/v1/*`를 통과시키는지
- `--base-url`에 포트나 proxy 경로가 맞는지

## 관련 문서

- [스크립트: send_image_to_vllm.py](./send_image_to_vllm.py)
- [Private Cloud에서 `vLLM` 시작](./private-cloud-vllm-next-steps.md)
- [VLM Cloud Notes](./README.md)
