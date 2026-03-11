---
tags: [vlm, ocr, install, private-cloud, huggingface, vllm, transformers]
level: intermediate
last_updated: 2026-03-11
status: draft
---

# Private Cloud 설치 가이드: PaddleOCR-VL-1.5 + GOT-OCR-2.0-hf

기준 문서: [vlm-for-ppt-pdf-extraction.md](./vlm-for-ppt-pdf-extraction.md)

이 문서는 아래 제약을 전제로 한다.

- 외부 API 사용 불가
- 모델은 Hugging Face에서 승인된 다운로드 머신으로만 받음
- 런타임은 사내 private cloud
- `vllm`, `transformers` 는 이미 있거나 내부 패키지 미러에서 설치 가능
- Docker 사용 불가
- H200 GPU는 다른 모델과 shared usage

## 먼저 고를 2개

초기 설치 대상으로는 아래 2개가 가장 낫다.

1. `PaddlePaddle/PaddleOCR-VL-1.5`
2. `stepfun-ai/GOT-OCR-2.0-hf`

선정 이유는 단순하다.

- `PaddleOCR-VL-1.5` 는 현재 가장 강한 기본 1차 문서 파서다.
- `GOT-OCR-2.0-hf` 는 설치가 단순하고, crop OCR / 작은 글씨 재인식 / 특정 bbox OCR 에 강하다.
- 둘 다 Apache-2.0 이라 사내 도입 시 법무 부담이 상대적으로 낮다.
- 둘 다 7B+ 급 대형 VLM보다 훨씬 가볍다.

이번 1차 설치에서는 `MinerU2.5` 는 뺐다.

- 정확도는 좋지만 초기 운영 난이도가 더 높다.
- AGPL-3.0 라이선스 검토가 필요하다.
- "처음 2개만 올려서 빨리 안정화" 목표에는 `PaddleOCR-VL-1.5 + GOT-OCR-2.0-hf` 가 더 현실적이다.

## 권장 역할 분담

- `PaddleOCR-VL-1.5`: PPT/PDF 페이지 전체 구조 파싱, reading order, 표/차트/수식/텍스트 추출
- `GOT-OCR-2.0-hf`: low-confidence 영역, 각주, 작은 텍스트, chart label, crop 재인식

즉, 두 개를 같은 계층의 "대체재"로 보지 말고 아래처럼 쓰는 편이 좋다.

```text
page image
  -> PaddleOCR-VL-1.5
  -> suspicious bbox / small text only
  -> GOT-OCR-2.0-hf
```

## 운영 원칙

- 런타임 노드에서는 public internet 접근을 막는다.
- 모델은 반드시 로컬 디렉터리 경로로만 로드한다.
- `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` 를 기본으로 둔다.
- H200 shared 환경에서는 `PaddleOCR-VL-1.5` 만 상시 서비스화하고, `GOT-OCR-2.0-hf` 는 필요할 때만 on-demand worker 로 실행한다.

## 추천 디렉터리

```text
/srv/ocr/
  models/
    PaddleOCR-VL-1.5/
    PP-DocLayoutV3/
    GOT-OCR-2.0-hf/
  wheelhouse/
  samples/
  output/
```

## 1. 다운로드 머신에서 모델 받기

인터넷이 되는 승인된 머신에서만 수행한다.

```bash
python -m pip install -U "huggingface_hub[cli]" hf_xet

export HF_HOME="$PWD/.hf_home"
mkdir -p ./artifacts

huggingface-cli download PaddlePaddle/PaddleOCR-VL-1.5 \
  --local-dir ./artifacts/PaddleOCR-VL-1.5

huggingface-cli download PaddlePaddle/PP-DocLayoutV3 \
  --local-dir ./artifacts/PP-DocLayoutV3

huggingface-cli download stepfun-ai/GOT-OCR-2.0-hf \
  --local-dir ./artifacts/GOT-OCR-2.0-hf

tar -C ./artifacts -czf ocr-model-bundle-20260311.tar.gz \
  PaddleOCR-VL-1.5 \
  PP-DocLayoutV3 \
  GOT-OCR-2.0-hf
```

private cloud 로 옮긴 뒤:

```bash
mkdir -p /srv/ocr/models
tar -C /srv/ocr/models -xzf ocr-model-bundle-20260311.tar.gz
```

## 2. 패키지까지 완전 오프라인으로 가져와야 할 때

private cloud 에서 public PyPI 접근도 막혀 있으면 wheelhouse 를 같이 만든다.

이미 내부 미러가 있으면 이 절은 건너뛰어도 된다.

connected machine:

```bash
mkdir -p ./wheelhouse

python -m pip download -d ./wheelhouse \
  "transformers>=5.3.0" \
  accelerate \
  pillow \
  sentencepiece

python -m pip download -d ./wheelhouse \
  "paddleocr[doc-parser]"

python -m pip download -d ./wheelhouse \
  --index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/ \
  paddlepaddle-gpu==3.2.1

tar -C ./wheelhouse -czf ocr-wheelhouse-20260311.tar.gz .
```

private cloud:

```bash
mkdir -p /srv/ocr/wheelhouse
tar -C /srv/ocr/wheelhouse -xzf ocr-wheelhouse-20260311.tar.gz
```

`vllm` 은 GPU/CUDA 조합에 따라 wheel 선택이 까다롭기 때문에 이 문서에서는 wheelhouse 범위에서 제외했다.

- 이미 사내에서 검증된 `vllm` env 를 재사용하는 것을 권장한다.
- 새로 offline wheel 을 만들려면 해당 클러스터의 CUDA, PyTorch, 드라이버 조합 기준으로 별도 검증이 필요하다.

## 3. 공통 오프라인 환경 변수

모든 런타임 세션에서 아래를 기본으로 둔다.

```bash
export HF_HOME=/srv/ocr/.hf
export TRANSFORMERS_CACHE=/srv/ocr/.hf/transformers
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
```

## 4. PaddleOCR-VL-1.5 설치

### 4.1 권장 구조

`PaddleOCR-VL-1.5` 는 아래처럼 나누는 것이 가장 안정적이다.

- client env: `paddleocr` + layout detection
- server env: `vllm` + local VLM weights

이유:

- PaddleOCR 문서도 `vllm` 과 `paddlepaddle` 을 separate venv 로 두는 쪽을 권장한다.
- shared H200 에서 server 와 client 튜닝 포인트를 분리하기 쉽다.

### 4.2 client env

내부 패키지 미러가 있으면 그 미러를 쓰고, 없으면 `/srv/ocr/wheelhouse` 를 쓴다.

```bash
python -m venv /srv/ocr/.venv_paddle_client
source /srv/ocr/.venv_paddle_client/bin/activate

python -m pip install --no-index --find-links=/srv/ocr/wheelhouse \
  paddlepaddle-gpu==3.2.1 \
  "paddleocr[doc-parser]"
```

만약 `wheelhouse` 없이 내부 미러를 쓰는 환경이면 위 명령만 미러 URL 기준으로 바꾸면 된다.

### 4.3 server env

`vllm` 이 이미 검증된 사내 공용 env 에 있으면 그 env 를 재사용해도 된다.

권장 방식:

```bash
source /path/to/your/validated-vllm-env/bin/activate
python -m pip install --no-index --find-links=/srv/ocr/wheelhouse \
  "paddleocr[doc-parser]"
```

주의:

- `paddleocr genai_server` 가 추가 server deps 를 요구할 수 있다.
- 이미 사내 env 에서 `vllm` serving 이 정상 동작하면 그 env 를 그대로 쓰는 편이 더 쉽다.
- 별도 `vllm` env 를 새로 만드는 경우는 이 문서 범위를 넘어선다. CUDA/PyTorch 호환성 검증을 먼저 끝내고 진행한다.

### 4.4 vLLM backend 설정 파일

H200 shared 환경에서는 처음부터 공격적으로 메모리를 잡지 않는 편이 낫다.

`/srv/ocr/vllm_config.yaml`

```yaml
gpu-memory-utilization: 0.3
max-num-seqs: 32
```

시작점 권장값:

- 다른 모델과 많이 공유 중이면 `0.25`
- 한 GPU 를 상대적으로 여유 있게 쓸 수 있으면 `0.3` 또는 `0.35`

### 4.5 PaddleOCR-VL-1.5 vLLM server 실행

```bash
source /path/to/your/validated-vllm-env/bin/activate

export HF_HOME=/srv/ocr/.hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

paddleocr genai_server \
  --model_dir /srv/ocr/models/PaddleOCR-VL-1.5 \
  --model_name PaddleOCR-VL-1.5-0.9B \
  --backend vllm \
  --port 8118 \
  --backend_config /srv/ocr/vllm_config.yaml
```

여기서 `1.5-0.9B` 는 버전이 `1.5` 이고 파라미터 규모가 `0.9B` 라는 뜻이다.

이 방식의 장점:

- 런타임에서 Hugging Face 접근이 필요 없다.
- `model_dir` 를 명시하므로 public network fallback 을 막기 쉽다.
- PaddleOCR 공식 파이프라인을 그대로 쓸 수 있다.

### 4.6 PaddleOCR client 예제

```python
from paddleocr import PaddleOCRVL

pipeline = PaddleOCRVL(
    pipeline_version="v1.5",
    vl_rec_backend="vllm-server",
    vl_rec_server_url="http://127.0.0.1:8118/v1",
    layout_detection_model_name="PP-DocLayoutV3",
    layout_detection_model_dir="/srv/ocr/models/PP-DocLayoutV3",
    vl_rec_max_concurrency=2,
)

results = pipeline.predict("/srv/ocr/samples/sample-slide.png")

for i, res in enumerate(results):
    res.save_to_json(save_path=f"/srv/ocr/output/paddle_{i}.json")
    res.save_to_markdown(save_path=f"/srv/ocr/output/paddle_{i}.md")
```

포인트:

- `layout_detection_model_dir` 를 꼭 준다. 안 주면 기본 모델을 다시 받으려 할 수 있다.
- shared GPU 환경이면 `vl_rec_max_concurrency` 는 낮게 시작한다.

## 5. GOT-OCR-2.0-hf 설치

`GOT-OCR-2.0-hf` 는 상시 서비스보다는 on-demand worker 가 낫다.

- 메인 파서는 아니다.
- 특정 crop 재확인용으로 쓸 때 가장 value 가 높다.
- shared GPU 환경에서 항상 띄워둘 이유가 적다.

### 5.1 env

```bash
python -m venv /srv/ocr/.venv_got
source /srv/ocr/.venv_got/bin/activate

python -m pip install --no-index --find-links=/srv/ocr/wheelhouse \
  "transformers>=5.3.0" \
  accelerate \
  pillow \
  sentencepiece
```

### 5.2 plain OCR smoke test

```python
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "/srv/ocr/models/GOT-OCR-2.0-hf"

model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
).to(device).eval()

processor = AutoProcessor.from_pretrained(
    model_path,
    use_fast=True,
    local_files_only=True,
)

image = Image.open("/srv/ocr/samples/sample-crop.png").convert("RGB")
inputs = processor(image, return_tensors="pt").to(device)

generate_ids = model.generate(
    **inputs,
    do_sample=False,
    tokenizer=processor.tokenizer,
    stop_strings="<|im_end|>",
    max_new_tokens=1024,
)

text = processor.decode(
    generate_ids[0, inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
)

print(text)
```

### 5.3 patch OCR 예제

가로로 긴 슬라이드, 작은 텍스트, box 합쳐짐이 의심되는 경우:

```python
inputs = processor(
    image,
    return_tensors="pt",
    format=True,
    crop_to_patches=True,
    max_patches=3,
).to(device)
```

### 5.4 specific region OCR 예제

특정 bbox 만 다시 읽고 싶을 때:

```python
inputs = processor(
    image,
    return_tensors="pt",
    box=[x1, y1, x2, y2],
).to(device)
```

## 6. H200 shared GPU 운영 팁

- `PaddleOCR-VL-1.5` 만 상시 프로세스로 두고 `GOT-OCR-2.0-hf` 는 batch worker 로 실행한다.
- 처음에는 `gpu-memory-utilization: 0.3` 이하로 시작한다.
- `tensor_parallel_size=1` 수준의 단일 GPU 운영부터 시작한다.
- OCR 는 prefix caching 이 큰 이득이 적으므로 기본적으로 보수적으로 둔다.
- cold start 줄이려면 모델 디렉터리는 NFS 보다 local NVMe 쪽이 낫다.
- PDF 대량 처리 전에 작은 샘플 20~50장으로 먼저 latency / VRAM / 누락률을 확인한다.

## 7. 추천 시작 방식

처음부터 복잡하게 가지 말고 아래 순서로 올리면 된다.

1. `PaddleOCR-VL-1.5` 만 먼저 서비스화
2. PPT/PDF 샘플 20~50장으로 구조 추출 검증
3. low-confidence 영역 규칙 정의
4. 그 뒤 `GOT-OCR-2.0-hf` 를 hotspot recrop worker 로 연결

이 순서가 좋은 이유:

- 운영 복잡도를 낮춘다.
- shared GPU 환경에서 상시 점유 프로세스를 최소화한다.
- 나중에 MinerU 같은 더 무거운 대안을 넣더라도 교체 포인트가 명확하다.

## 8. 간단 결론

지금 private cloud 에 먼저 올릴 2개는 아래가 맞다.

- `PaddlePaddle/PaddleOCR-VL-1.5`
- `stepfun-ai/GOT-OCR-2.0-hf`

실무 기준으로는:

- `PaddleOCR-VL-1.5` 를 메인 문서 파서
- `GOT-OCR-2.0-hf` 를 재인식 전용 worker

로 두는 것이 가장 안전하고, no-Docker + offline + shared H200 조건에도 잘 맞는다.

## Sources

- PaddleOCR-VL-1.5 model card: https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5
- PaddleOCR-VL usage tutorial: https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html
- PP-DocLayoutV3 model card: https://huggingface.co/PaddlePaddle/PP-DocLayoutV3
- GOT-OCR-2.0-hf model card: https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf
- Transformers GOT-OCR2 docs: https://huggingface.co/docs/transformers/main/en/model_doc/got_ocr2
- vLLM PaddleOCR-VL recipe: https://docs.vllm.ai/projects/recipes/en/latest/PaddlePaddle/PaddleOCR-VL.html
