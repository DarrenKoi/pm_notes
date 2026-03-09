---
tags: [vlm, ui-venus, vllm, flask, blueprint, private-cloud]
level: intermediate
last_updated: 2026-03-09
status: in-progress
---

# UI-Venus-1.5-8B Cloud API Guide with Flask Blueprint

> 목표: `UI-Venus-1.5-8B`를 private cloud의 `vLLM`으로 실행한 뒤, Flask blueprint 기반 API로 감싸서 외부 클라이언트는 Flask만 호출하게 만든다.

## 이 문서의 전제

이 문서는 구현 예시 문서다. 아래 단계는 이미 끝났다고 가정한다.

- 모델 반입: [오프라인 다운로드 & 폐쇄망 전송 가이드](./offline-download-guide.md)
- 서빙 트랙 선택: [모델 서빙 가이드](./serving-guide.md)
- 기본 `vLLM` smoke test 성공: [Private Cloud에서 모델 다운로드 후 다음 단계](./private-cloud-vllm-next-steps.md)

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
  http://127.0.0.1:8000/v1
```

이 구조가 좋은 이유:

- `vLLM`을 외부에 직접 열지 않아도 된다
- 인증, 로깅, 요청 제한을 Flask에서 처리하기 쉽다
- 이후 `MAI-UI`나 다른 모델도 blueprint 단위로 확장하기 쉽다

## Step 1. 전제 상태 확인

아래가 이미 되는 상태여야 한다.

```bash
curl -s http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer change-this-vllm-key"
```

정상이면 `ui-venus-8b`가 보인다.

## Step 2. Flask 구조를 나눈다

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

원칙:

- blueprint는 HTTP 입출력만 담당
- `vLLM` 호출은 service로 분리

## Step 3. App factory에서 설정과 blueprint 등록

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

## Step 4. `vLLM` 호출 로직은 service로 둔다

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
                    {"type": "text", "text": instruction},
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

## Step 5. Blueprint에서 업로드를 받아 service에 전달

`app/blueprints/ui_venus.py`

```python
from flask import Blueprint, current_app, jsonify, request
from werkzeug.datastructures import FileStorage

from app.services.ui_venus_vllm import analyze_image_with_ui_venus

bp = Blueprint("ui_venus", __name__, url_prefix="/api/ui-venus")


@bp.post("/analyze")
def analyze():
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

## Step 6. Local PC에서 Flask API 호출

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

`curl`로도 확인 가능하다.

```bash
curl -X POST http://your-cloud-host:5050/api/ui-venus/analyze \
  -F "image=@./screen.png" \
  -F "instruction=Find the settings button and answer briefly."
```

## 운영 메모

- `vLLM`은 `127.0.0.1`에만 바인딩하는 편이 안전하다.
- 초기에는 `result`와 `raw_response`를 같이 반환하는 편이 디버깅에 유리하다.
- 첫 단계는 자유 응답으로 성공 여부만 보고, 이후에 좌표 JSON 형식을 강제하는 편이 낫다.

## 추천 순서

1. Cloud에서 `vLLM`이 단독으로 응답하는지 확인
2. Flask app factory에 blueprint 등록
3. `/api/ui-venus/analyze` endpoint 추가
4. local PC에서 업로드 요청 확인
5. 응답이 안정되면 인증, 로깅, JSON 포맷 강제 추가

## 관련 문서

- [이전: Private Cloud에서 모델 다운로드 후 다음 단계](./private-cloud-vllm-next-steps.md)
- [위로: VLM 가이드 인덱스](./README.md)
