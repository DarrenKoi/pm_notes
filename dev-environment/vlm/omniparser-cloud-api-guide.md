---
tags: [vlm, omniparser, fastapi, private-cloud, requests]
level: intermediate
last_updated: 2026-03-09
status: in-progress
---

# OmniParser V2 설치 및 Cloud API 패턴

> 목적: private cloud 안에서 `OmniParser V2`를 돌리고, 로컬 PC 또는 다른 내부 서비스가 Python `requests`로 스크린샷 이미지를 업로드해서 결과를 받는 구조를 정리한다.

## 먼저 답

- 지금까지 적어둔 문서는 `OmniParser` 다운로드와 `gradio_demo.py` 실행 수준까지였다
- **cloud에서 HTTP request로 이미지를 보내는 운영 패턴**은 이 문서에서 따로 정리한다
- 권장 구조는 `OmniParser API`와 `LLM API`를 분리하는 것이다

## 권장 아키텍처

```text
[client python requests]
        |
        v
[internal OmniParser API]
  - image upload
  - parse screen elements
  - return JSON
        |
        +----> [optional internal LLM API via vLLM/OpenAI-compatible]
```

이렇게 나누는 이유:

- `OmniParser`는 parser stage이고, LLM은 reasoning stage다
- parser와 LLM을 분리하면 디버깅이 쉽다
- parser 결과만 저장해서 재현 테스트하기 쉽다

## Step 1. Cloud에 OmniParser 소스와 weights 준비

공식 소스:

- GitHub: https://github.com/microsoft/OmniParser
- Hugging Face weights: https://huggingface.co/microsoft/OmniParser-v2.0

GitHub README 기준 quick start:

```bash
git clone https://github.com/microsoft/OmniParser.git
cd OmniParser
git lfs install

conda create -n "omni" python==3.12
conda activate omni
pip install -r requirements.txt
```

## Step 2. Hugging Face weights를 Cloud 경로에 배치

Hugging Face 쪽 문서에서 확인되는 필수 파일:

- `icon_detect/model.pt`
- `icon_detect/model.yaml`
- `icon_detect/train_args.yaml`
- `icon_caption/config.json`
- `icon_caption/generation_config.json`
- `icon_caption/model.safetensors`

다운로드 예시:

```bash
mkdir -p /data/models/OmniParser-v2.0
cd /data/models/OmniParser-v2.0

for f in icon_detect/{train_args.yaml,model.pt,model.yaml} \
         icon_caption/{config.json,generation_config.json,model.safetensors}; do
  huggingface-cli download microsoft/OmniParser-v2.0 "$f" --local-dir .
done

mv icon_caption icon_caption_florence
```

배치 후 예상 구조:

```text
/data/models/OmniParser-v2.0/
  icon_detect/
    model.pt
    model.yaml
    train_args.yaml
  icon_caption_florence/
    config.json
    generation_config.json
    model.safetensors
```

## Step 3. 먼저 공식 데모로 smoke test

공식 repo는 production REST server보다 `gradio_demo.py` 중심으로 문서화되어 있다. 따라서 처음에는 데모가 뜨는지부터 확인하는 편이 맞다.

```bash
cd /data/OmniParser
conda activate omni

ln -s /data/models/OmniParser-v2.0 weights
python gradio_demo.py
```

이 단계에서 확인할 것:

- 업로드한 스크린샷이 열리는지
- bounding box와 caption이 나오는지
- GPU 메모리 사용량이 정상인지

## Step 4. Cloud용으로는 Gradio가 아니라 내부 API를 하나 두기

공식 repo는 데모/UI 중심이다. private cloud에서 외부 클라이언트가 이미지를 보내려면 보통 내부 HTTP API를 하나 더 두는 편이 낫다.

권장 방식:

1. cloud 내부에 `FastAPI` 같은 얇은 wrapper 서비스 생성
2. `POST /parse`로 이미지 업로드
3. 서버가 이미지 저장 후 OmniParser 호출
4. JSON으로 요소 목록 반환

## Step 5. 추천 API 형태

### `POST /parse`

입력:

- multipart form image file
- 선택 입력: `instruction`

출력 예시:

```json
{
  "image_size": [1920, 1080],
  "elements": [
    {
      "type": "icon",
      "text": "Settings",
      "bbox": [1710, 36, 1752, 78]
    },
    {
      "type": "button",
      "text": "Save",
      "bbox": [1450, 980, 1530, 1024]
    }
  ]
}
```

### `POST /analyze`

입력:

- multipart form image file
- `instruction`

서버 내부 처리:

1. OmniParser로 element list 생성
2. 그 결과를 내부 LLM API에 전달
3. 클릭 대상이나 좌표를 응답

출력 예시:

```json
{
  "instruction": "open settings",
  "chosen_element": {
    "text": "Settings",
    "bbox": [1710, 36, 1752, 78]
  },
  "reasoning_model": "ui-venus-8b"
}
```

## Step 6. Python client에서 이미지 보내기

이게 당신이 실제로 쓰게 될 기본 패턴이다.

### parser만 호출

```python
import requests

url = "http://your-cloud-host:9000/parse"

with open("screen.png", "rb") as f:
    resp = requests.post(
        url,
        files={"image": ("screen.png", f, "image/png")},
        timeout=120,
    )

resp.raise_for_status()
data = resp.json()
print(data)
```

### parser + LLM 분석까지 한 번에 호출

```python
import requests

url = "http://your-cloud-host:9000/analyze"

with open("screen.png", "rb") as f:
    resp = requests.post(
        url,
        files={"image": ("screen.png", f, "image/png")},
        data={"instruction": "Open settings and tell me which element to click."},
        timeout=180,
    )

resp.raise_for_status()
data = resp.json()
print(data)
```

## Step 7. Server에서 LLM과 연결하는 방법

가장 단순한 구조는:

- `OmniParser`는 같은 서버에서 local inference
- LLM은 내부 `vLLM` OpenAI-compatible endpoint로 호출

예:

```python
from openai import OpenAI

llm = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="internal-key",
)

prompt = f"""
You are given parsed UI elements from a screenshot.
Instruction: {instruction}
Elements:
{elements}

Return the single best target element.
"""

resp = llm.chat.completions.create(
    model="ui-venus-8b",
    messages=[{"role": "user", "content": prompt}],
    temperature=0,
    max_tokens=300,
)
```

실무 팁:

- 처음에는 `elements`만 LLM에 넣어라
- 나중에 필요하면 원본 이미지도 멀티모달로 같이 넣어라
- parser 결과 JSON을 로그로 남겨야 디버깅이 쉽다

## Step 8. FastAPI wrapper 예시

아래 코드는 **운영 패턴 예시**다. upstream repo가 stable REST server를 제공하는 것은 아니므로, 실제 OmniParser 호출부는 repo에 vendor한 코드 기준으로 맞춰야 한다.

```python
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, Form, UploadFile
from openai import OpenAI

app = FastAPI()

LLM = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="internal-key",
)


def parse_with_omniparser(image_path: str) -> dict:
    """
    Replace this stub with the actual OmniParser call from your vendored repo.
    The return value should be a JSON-serializable dict containing UI elements.
    """
    raise NotImplementedError


@app.post("/parse")
async def parse(image: UploadFile = File(...)):
    suffix = Path(image.filename or "upload.png").suffix or ".png"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name

    result = parse_with_omniparser(tmp_path)
    return result


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    instruction: str = Form(...),
):
    suffix = Path(image.filename or "upload.png").suffix or ".png"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name

    parsed = parse_with_omniparser(tmp_path)

    prompt = f"""
Instruction: {instruction}
Parsed screen elements:
{parsed}

Return the best UI target.
"""

    completion = LLM.chat.completions.create(
        model="ui-venus-8b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=300,
    )

    return {
        "instruction": instruction,
        "parsed": parsed,
        "llm_output": completion.choices[0].message.content,
    }
```

실행 예시:

```bash
uvicorn app:app --host 0.0.0.0 --port 9000
```

## Step 9. 운영 시 주의할 점

- `OmniParser`는 `vLLM serve`처럼 바로 OpenAI-compatible server로 뜨는 모델이 아니다
- 따라서 parser service와 LLM service를 분리하는 쪽이 유지보수에 유리하다
- 요청당 이미지 업로드는 `multipart/form-data`가 가장 단순하다
- 원본 이미지는 object storage에 저장하고, API에는 URL만 넘기는 구조도 가능하지만 초기 구축은 파일 업로드가 더 단순하다
- `icon_detect`는 AGPL, `icon_caption`은 MIT이므로 사내 공용 서비스라면 라이선스 검토가 필요하다

## 내가 추천하는 실제 순서

1. `gradio_demo.py`로 smoke test
2. FastAPI wrapper로 `POST /parse`만 먼저 구현
3. Python `requests` client로 이미지 업로드 확인
4. 그다음 `/analyze`에서 내부 `vLLM` LLM 호출 연결
5. 마지막에 reverse proxy, auth, logging 추가

## 참고 자료

- OmniParser GitHub: https://github.com/microsoft/OmniParser
- OmniParser Hugging Face: https://huggingface.co/microsoft/OmniParser-v2.0
- vLLM OpenAI-compatible server docs: https://docs.vllm.ai/en/stable/serving/openai_compatible_server/
- Hugging Face Hub environment variables: https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables
