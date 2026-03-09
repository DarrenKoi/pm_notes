---
tags: [vlm, ui-venus, vllm, flask, blueprint, private-cloud]
level: intermediate
last_updated: 2026-03-09
status: in-progress
---

# UI-Venus-1.5-8B Cloud API Guide with Flask Blueprint

> 목표: `UI-Venus-1.5-8B` 모델 파일을 이미 private cloud에 올려둔 상태에서,
> cloud 안의 `vLLM`이 모델을 실행하고, 기존 Flask 서버는 **blueprint 기반 API**로 이미지를 받아서 `vLLM`에 전달하고,
> local PC는 HTTP로 Flask API를 호출하는 구조를 만든다.

## 권장 구조

```text
[Local PC]
  Python requests
        |
        v
[Cloud Flask Server]
  Blueprint: /api/ui-venus
        |
        v
[Cloud vLLM Server]
  UI-Venus-1.5-8B
  http://127.0.0.1:8000/v1
```

이 구조가 좋은 이유:

- `vLLM`을 외부에 직접 열지 않아도 된다
- Flask에서 인증, 로깅, 요청 제한을 넣기 쉽다
- 추후 `MAI-UI`, `UI-TARS`를 blueprint 단위로 분리하기 쉽다

## 공식 문서 기준

- `UI-Venus-1.5-8B` Hugging Face 카드에는 `vllm>=0.11.0`, `transformers>=4.57.0`가 필요하다고 적혀 있다
- 같은 카드의 quick start는 `vLLM` OpenAI-compatible server로 서빙하는 예시를 제공한다
- `vLLM` 공식 문서는 이미지 입력을 Chat Completions API의 OpenAI Vision 형식으로 받는다

참고:

- UI-Venus HF card: https://huggingface.co/inclusionAI/UI-Venus-1.5-8B
- vLLM OpenAI-compatible server: https://docs.vllm.ai/en/stable/serving/openai_compatible_server/
- vLLM multimodal inputs: https://docs.vllm.ai/en/stable/features/multimodal_inputs.html

## Step 1. Cloud에서 모델 폴더 확인

예시 경로:

```bash
MODEL_DIR=/data/models/UI-Venus-1.5-8B

ls -lh "$MODEL_DIR"
find "$MODEL_DIR" -maxdepth 1 \( -name '*.json' -o -name '*.safetensors' \) | sort
```

최소 확인 파일:

- `config.json`
- `tokenizer_config.json`
- `preprocessor_config.json`
- `generation_config.json`
- `model.safetensors.index.json`
- 모든 `model-0000x-of-0000y.safetensors`

## Step 2. Cloud에서 vLLM 실행

`UI-Venus` 모델 카드는 repo id로 예시를 보여주지만, private cloud에서는 **local path로 띄우는 것이 안전하다**. 이 부분은 vLLM의 일반적인 local path 서빙 기능에 따른 적용이다.

```bash
export HF_HUB_OFFLINE=1

CUDA_VISIBLE_DEVICES=0 vllm serve /data/models/UI-Venus-1.5-8B \
  --served-model-name ui-venus-8b \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --api-key change-this-vllm-key \
  --generation-config vllm \
  --limit-mm-per-prompt '{"image":1}'
```

운영 포인트:

- `--host 127.0.0.1`: vLLM을 외부에 직접 노출하지 않음
- `--api-key`: Flask에서 내부 호출할 때 사용
- `--generation-config vllm`: HF의 generation config가 sampling 기본값을 덮어쓰는 것을 피함
- `--limit-mm-per-prompt '{"image":1}'`: 한 요청당 이미지 1장 기준

## Step 3. vLLM 단독 확인

```bash
curl -s http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer change-this-vllm-key" | python -m json.tool
```

정상이면 `ui-venus-8b`가 보여야 한다.

## Step 4. Flask는 Blueprint 기반으로 붙인다

권장 구조:

```text
your_flask_app/
  app/
    __init__.py
    blueprints/
      ui_venus.py
    services/
      ui_venus_vllm.py
  run.py
```

## Step 5. App factory에서 blueprint 등록

`app/__init__.py`

```python
from flask import Flask

from app.blueprints.ui_venus import bp as ui_venus_bp


def create_app() -> Flask:
    app = Flask(__name__)

    app.config.update(
        UI_VENUS_VLLM_BASE_URL="http://127.0.0.1:8000/v1",
        UI_VENUS_VLLM_API_KEY="change-this-vllm-key",
        UI_VENUS_MODEL_NAME="ui-venus-8b",
        UI_VENUS_MAX_TOKENS=300,
        MAX_CONTENT_LENGTH=10 * 1024 * 1024,
    )

    app.register_blueprint(ui_venus_bp)
    return app
```

`run.py`

```python
from app import create_app

app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
```

## Step 6. vLLM 호출 로직은 service로 분리

`app/services/ui_venus_vllm.py`

```python
import base64
from typing import Any

import requests


def analyze_image_with_ui_venus(
    image_bytes: bytes,
    instruction: str,
    base_url: str,
    api_key: str,
    model_name: str,
    max_tokens: int = 300,
    mime_type: str = "image/png",
) -> dict[str, Any]:
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": instruction,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_b64}",
                        },
                    },
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }

    response = requests.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=180,
    )
    response.raise_for_status()
    return response.json()
```

왜 service로 분리하는가:

- blueprint는 HTTP 입출력만 담당
- vLLM 호출부를 테스트하기 쉽다
- 나중에 다른 모델 service를 추가하기 쉽다

## Step 7. Blueprint에서 업로드 파일을 받아 service에 전달

`app/blueprints/ui_venus.py`

```python
from flask import Blueprint, current_app, jsonify, request
from werkzeug.datastructures import FileStorage

from app.services.ui_venus_vllm import analyze_image_with_ui_venus

bp = Blueprint("ui_venus", __name__, url_prefix="/api/ui-venus")


@bp.post("/analyze")
def analyze() -> tuple[dict, int]:
    uploaded_file: FileStorage | None = request.files.get("image")
    instruction = request.form.get(
        "instruction",
        "Identify the target UI element and answer briefly.",
    )

    if uploaded_file is None or uploaded_file.filename == "":
        return jsonify({"error": "image file is required"}), 400

    image_bytes = uploaded_file.read()
    mime_type = uploaded_file.mimetype or "image/png"

    result = analyze_image_with_ui_venus(
        image_bytes=image_bytes,
        instruction=instruction,
        base_url=current_app.config["UI_VENUS_VLLM_BASE_URL"],
        api_key=current_app.config["UI_VENUS_VLLM_API_KEY"],
        model_name=current_app.config["UI_VENUS_MODEL_NAME"],
        max_tokens=current_app.config["UI_VENUS_MAX_TOKENS"],
        mime_type=mime_type,
    )

    content = result["choices"][0]["message"]["content"]

    return jsonify(
        {
            "model": current_app.config["UI_VENUS_MODEL_NAME"],
            "instruction": instruction,
            "result": content,
            "raw_response": result,
        }
    ), 200
```

## Step 8. Local PC에서 Flask API 호출

이제 local PC는 Flask만 호출하면 된다.

```python
import requests

url = "http://your-cloud-host:5050/api/ui-venus/analyze"

with open("screen.png", "rb") as f:
    response = requests.post(
        url,
        files={"image": ("screen.png", f, "image/png")},
        data={"instruction": "Find the settings button and answer briefly."},
        timeout=180,
    )

response.raise_for_status()
print(response.json())
```

## Step 9. curl로도 확인 가능

```bash
curl -X POST http://your-cloud-host:5050/api/ui-venus/analyze \
  -F "image=@./screen.png" \
  -F "instruction=Find the settings button and answer briefly."
```

## Step 10. 운영 시 추천사항

### 1. vLLM은 localhost에만 바인딩

- Flask만 외부에 노출
- 보안과 운영이 단순해진다

### 2. Flask 응답은 raw와 parsed를 같이 주는 편이 좋다

초기에는 아래 둘 다 남기는 편이 디버깅에 좋다:

- `result`: 사람이 바로 읽는 응답
- `raw_response`: vLLM 원문

### 3. instruction을 정형화하라

처음에는 자유문장보다 아래처럼 형식을 고정하는 편이 낫다:

```text
You are looking at a UI screenshot.
Identify the best target for the following instruction:
"Open settings"
Answer in one short sentence.
```

### 4. 이후에는 좌표 포맷을 강제

UI 자동화를 붙일 거면 나중에는 응답 형식을 JSON으로 강제하는 편이 좋다. 예:

```text
Return JSON only:
{"target_text":"...", "target_description":"...", "bbox":[x1,y1,x2,y2]}
```

단, 첫 단계에서는 **자유 응답으로 성공 여부를 먼저 확인**하는 편이 낫다.

## 내가 추천하는 실제 순서

1. Cloud에서 `vllm serve`로 `UI-Venus-1.5-8B` 실행
2. Cloud에서 `/v1/models` 확인
3. Flask app factory에 blueprint 등록
4. `/api/ui-venus/analyze` endpoint 추가
5. local PC에서 `requests.post(... files=...)`로 이미지 업로드
6. 응답이 안정되면 JSON 포맷 강제, 인증, 로깅 추가

## 다음에 바로 붙일 수 있는 확장

- `/api/ui-venus/health`
- `/api/ui-venus/models`
- API key 인증
- 업로드 이미지 파일 저장 여부 선택
- request id / access log / latency log

## 참고 자료

- UI-Venus-1.5-8B HF card: https://huggingface.co/inclusionAI/UI-Venus-1.5-8B
- vLLM OpenAI-compatible server: https://docs.vllm.ai/en/stable/serving/openai_compatible_server/
- vLLM multimodal inputs: https://docs.vllm.ai/en/stable/features/multimodal_inputs.html
