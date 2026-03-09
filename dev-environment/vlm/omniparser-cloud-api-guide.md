---
tags: [vlm, omniparser, fastapi, private-cloud, requests]
level: intermediate
last_updated: 2026-03-09
status: in-progress
---

# OmniParser V2 설치 및 Cloud API 패턴

> 목표: private cloud 안에서 `OmniParser V2`를 parser service로 두고, 다른 내부 서비스나 로컬 PC가 HTTP로 이미지를 보내 결과를 받는 구조를 정리한다.

## 이 문서의 전제

이 문서는 구현 예시 문서다. 아래는 이미 끝났다고 가정한다.

- 모델 선정: [Hugging Face 다운로드 shortlist](./huggingface-private-cloud-downloads.md)
- 반입: [오프라인 다운로드 & 폐쇄망 전송 가이드](./offline-download-guide.md)
- 서빙 트랙 판단: [모델 서빙 가이드](./serving-guide.md)

## 먼저 답

- `OmniParser`는 `vLLM`처럼 바로 OpenAI 호환 API로 띄우는 모델이 아니다.
- production에서는 `parser service`와 `LLM service`를 분리하는 편이 가장 단순하다.
- 처음에는 `gradio_demo.py`로 smoke test를 끝낸 뒤, 그 다음에 얇은 HTTP wrapper를 올리는 편이 낫다.

## 권장 구조

```text
[client python requests]
        |
        v
[internal OmniParser API]
  - image upload
  - parse screen elements
  - return JSON
        |
        +----> [optional internal LLM API]
```

## Step 1. 준비물 확인

필수 준비물:

- OmniParser GitHub 소스
- `microsoft/OmniParser-v2.0` weights
- private cloud에서 돌릴 Python 환경

필수 파일은 보통 아래 정도로 본다.

- `icon_detect/model.pt`
- `icon_detect/model.yaml`
- `icon_detect/train_args.yaml`
- `icon_caption/config.json`
- `icon_caption/generation_config.json`
- `icon_caption/model.safetensors`

## Step 2. 먼저 공식 데모로 smoke test

```bash
cd /data/OmniParser
ln -s /data/models/OmniParser-v2.0 weights
python gradio_demo.py
```

이 단계에서 확인할 것:

- 스크린샷 업로드가 되는지
- bounding box와 caption이 나오는지
- GPU 메모리 사용량이 정상인지

이 단계가 실패하면 wrapper부터 만들지 않는 편이 낫다.

## Step 3. Cloud용으로는 얇은 API wrapper를 둔다

권장 방식:

1. `POST /parse`로 이미지 업로드
2. 서버가 OmniParser를 호출
3. JSON 요소 목록 반환

필요하면 나중에:

4. `POST /analyze`에서 내부 LLM까지 연결

## Step 4. 추천 API 형태

### `POST /parse`

입력:

- `multipart/form-data` image file
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
    }
  ]
}
```

### `POST /analyze`

서버 내부 처리:

1. OmniParser로 element list 생성
2. 내부 LLM API에 전달
3. 클릭 후보를 응답

## Step 5. FastAPI wrapper 예시

아래 코드는 운영 패턴 예시다. 실제 OmniParser 호출부는 vendored repo 구조에 맞게 바꿔야 한다.

```python
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, Form, UploadFile
from openai import OpenAI

app = FastAPI()

llm = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="internal-key",
)


def parse_with_omniparser(image_path: str) -> dict:
    """
    Replace this stub with the actual OmniParser call from your local repo.
    """
    raise NotImplementedError


@app.post("/parse")
async def parse(image: UploadFile = File(...)):
    suffix = Path(image.filename or "upload.png").suffix or ".png"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name

    return parse_with_omniparser(tmp_path)


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

    completion = llm.chat.completions.create(
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

실행:

```bash
uvicorn app:app --host 0.0.0.0 --port 9000
```

## Step 6. Python client에서 이미지 보내기

### parser만 호출

```python
import requests

url = "http://your-cloud-host:9000/parse"

with open("screen.png", "rb") as f:
    response = requests.post(
        url,
        files={"image": ("screen.png", f, "image/png")},
        timeout=120,
    )

response.raise_for_status()
print(response.json())
```

### parser + LLM 분석까지 호출

```python
import requests

url = "http://your-cloud-host:9000/analyze"

with open("screen.png", "rb") as f:
    response = requests.post(
        url,
        files={"image": ("screen.png", f, "image/png")},
        data={"instruction": "Open settings and tell me which element to click."},
        timeout=180,
    )

response.raise_for_status()
print(response.json())
```

## 운영 메모

- parser 결과 JSON은 로그로 남기는 편이 디버깅에 유리하다.
- `OmniParser`와 내부 LLM은 별도 프로세스로 두는 편이 안전하다.
- 초기 구축은 object storage보다 `multipart/form-data` 업로드가 단순하다.
- `icon_detect`는 AGPL, `icon_caption`은 MIT이므로 사내 공용 서비스면 라이선스 검토가 필요하다.

## 추천 순서

1. `gradio_demo.py`로 smoke test
2. `POST /parse`만 먼저 구현
3. Python `requests`로 이미지 업로드 확인
4. 그 다음 `/analyze`에서 내부 LLM 연결
5. 마지막에 auth, logging, reverse proxy 추가

## 관련 문서

- [이전: 모델 서빙 가이드](./serving-guide.md)
- [참고: Private Cloud에서 모델 다운로드 후 다음 단계](./private-cloud-vllm-next-steps.md)
- [위로: VLM 가이드 인덱스](./README.md)
