---
tags: [vlm, vllm, sglang, serving, openai-api]
level: intermediate
last_updated: 2026-03-04
status: in-progress
---

# 모델 서빙 가이드

> 폐쇄망 Linux 클라우드(H200 x2)에서 UI 특화 VLM을 서빙하고 OpenAI 호환 API로 접근하는 방법

## 왜 필요한가? (Why)

- 다운로드한 VLM 모델을 실제로 실행하여 GUI 자동화에 활용해야 함
- OpenAI 호환 API로 서빙하면 기존 내부 LLM 인프라와 동일한 방식으로 사용 가능
- H200 x2 (280GB VRAM)으로 어떤 모델이든 실행 가능

## 핵심 개념 (What)

### 서빙 프레임워크 비교

| 프레임워크 | 장점 | 단점 | 추천 용도 |
|------------|------|------|-----------|
| **vLLM** | 최고 처리량, PagedAttention, OpenAI API 호환 | 설정이 다소 복잡 | **프로덕션 서빙 (추천)** |
| **SGLang** | 멀티턴 최적화, KV Cache 재사용, 안정적 지연시간 | vLLM보다 약간 낮은 피크 처리량 | 복잡한 멀티턴 파이프라인 |
| **transformers** | 가장 간단, Python 직접 통합 | 배치 최적화 없음, 낮은 처리량 | 개발/디버깅/프로토타이핑 |

> **추천**: 대부분의 경우 vLLM이 최적. 멀티턴 상호작용이 많으면 SGLang 고려.

---

## 어떻게 사용하는가? (How)

### 사전 준비: 폐쇄망 Python 환경 구성

폐쇄망이므로 pip 패키지도 외부에서 다운로드 후 전송해야 함:

```bash
# === 외부 PC에서 (인터넷 접근 가능) ===

# 방법 1: pip wheel로 오프라인 패키지 준비
mkdir vllm-packages
pip download vllm -d ./vllm-packages/
pip download sglang[all] -d ./sglang-packages/

# USB로 전송 후 클라우드에서 설치
# pip install --no-index --find-links=./vllm-packages/ vllm

# 방법 2: conda-pack으로 전체 환경 패키징
conda create -n vlm-serve python=3.11
conda activate vlm-serve
pip install vllm
conda pack -n vlm-serve -o vlm-serve.tar.gz
# 약 5-10GB, USB로 전송 후 클라우드에서 풀기
```

```bash
# === 클라우드에서 ===

# conda-pack 환경 풀기
mkdir -p ~/envs/vlm-serve
tar -xzf vlm-serve.tar.gz -C ~/envs/vlm-serve
source ~/envs/vlm-serve/bin/activate
conda-unpack  # 경로 수정

# 또는 pip wheel로 설치
pip install --no-index --find-links=./vllm-packages/ vllm
```

---

### vLLM으로 모델 서빙 (추천)

#### 기본 서빙 (단일 GPU)

```bash
# UI-Venus-1.5-8B 서빙
vllm serve /data/models/UI-Venus-1.5-8B \
    --served-model-name ui-venus-8b \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --trust-remote-code

# MAI-UI-8B 서빙
vllm serve /data/models/MAI-UI-8B \
    --served-model-name mai-ui-8b \
    --host 0.0.0.0 \
    --port 8001 \
    --dtype bfloat16 \
    --trust-remote-code
```

> H200 x2 (280GB)이므로 여러 모델을 동시에 각각 다른 포트에서 서빙 가능

#### 2-GPU Tensor Parallel (대형 모델)

```bash
# UI-TARS-72B-DPO (144GB, 2개 GPU에 분산)
vllm serve /data/models/UI-TARS-72B-DPO \
    --served-model-name ui-tars-72b \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --tensor-parallel-size 2 \
    --trust-remote-code

# UI-Venus-1.5-30B-A3B (MoE, 2개 GPU)
vllm serve /data/models/UI-Venus-1.5-30B-A3B \
    --served-model-name ui-venus-30b \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --tensor-parallel-size 2 \
    --trust-remote-code
```

#### 여러 모델 동시 서빙 (A/B 비교)

```bash
# GPU 0에 UI-Venus-8B, GPU 1에 MAI-UI-8B
CUDA_VISIBLE_DEVICES=0 vllm serve /data/models/UI-Venus-1.5-8B \
    --served-model-name ui-venus-8b \
    --port 8000 --dtype bfloat16 --trust-remote-code &

CUDA_VISIBLE_DEVICES=1 vllm serve /data/models/MAI-UI-8B \
    --served-model-name mai-ui-8b \
    --port 8001 --dtype bfloat16 --trust-remote-code &

# 두 모델에 같은 스크린샷을 보내서 결과 비교 가능
```

#### vLLM 버전 요구사항

> 현재 클라우드에 설치된 버전: **vLLM v0.16.0** — 모든 모델 호환 확인됨

| 모델 | 최소 vLLM 버전 | v0.16.0 호환 |
|------|----------------|--------------|
| UI-Venus-1.5-* | v0.11.0+ | ✅ |
| UI-TARS-1.5-7B | v0.6.1+ | ✅ |
| MAI-UI-* | v0.6.1+ | ✅ |
| GUI-Actor-7B | v0.6.1+ | ✅ |
| ShowUI-2B | v0.6.1+ | ✅ |
| UGround-V1-* | v0.6.1+ | ✅ |

---

### SGLang으로 모델 서빙 (멀티턴 최적화)

```bash
# SGLang 서버 시작
python -m sglang.launch_server \
    --model-path /data/models/UI-Venus-1.5-8B \
    --host 0.0.0.0 \
    --port 30000 \
    --tp 1

# 2-GPU tensor parallel
python -m sglang.launch_server \
    --model-path /data/models/UI-TARS-72B-DPO \
    --host 0.0.0.0 \
    --port 30000 \
    --tp 2
```

---

### transformers로 직접 추론 (프로토타이핑)

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch
from PIL import Image

# ShowUI-2B 예시 (가장 간단한 사용법)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/data/models/ShowUI-2B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(
    "/data/models/ShowUI-2B",
    min_pixels=256 * 28 * 28,
    max_pixels=1344 * 28 * 28
)

# 스크린샷 로드 및 그라운딩 요청
image = Image.open("screenshot.png")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Click on the 'File' menu"}
        ]
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=128)
result = processor.decode(output[0], skip_special_tokens=True)
print(result)
```

```python
# SeeClick 예시 (Qwen-VL 기반)
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "/data/models/SeeClick",
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    "/data/models/SeeClick",
    device_map="cuda",
    trust_remote_code=True,
    bf16=True
).eval()
```

---

### OmniParser V2 설정 (파싱 파이프라인)

OmniParser V2는 독립 VLM이 아닌 **UI 파싱 도구**로, 스크린샷을 구조화된 요소 목록으로 변환:

```bash
# GitHub 코드 클론 (외부에서 다운로드 후 전송)
git clone https://github.com/microsoft/OmniParser
cd OmniParser

# 의존성 설치 — 기본 pip 설정이 사내 Nexus로 되어 있으므로
# 공식 PyPI URL을 명시적으로 지정하고 trusted-host 설정 필요
pip install -r requirements.txt \
    -i https://pypi.org/simple/ \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org

# 모델 가중치 경로 설정 (이미 다운로드한 파일 사용)
# weights/ 폴더에 OmniParser-v2.0 파일을 배치
ln -s /data/models/OmniParser-v2.0 weights/

# Gradio 데모 실행
python gradio_demo.py
```

**파이프라인 활용 예시** — OmniParser + 내부 LLM:

```python
# 1. OmniParser로 스크린샷 파싱
from omniparser import OmniParser

parser = OmniParser(model_path="/data/models/OmniParser-v2.0")
elements = parser.parse("screenshot.png")
# 결과: [{"label": "File Menu", "bbox": [10, 5, 50, 25], "type": "button"}, ...]

# 2. 파싱 결과를 내부 LLM(Kimi-K2.5)에 전달
import openai

client = openai.OpenAI(
    base_url="http://internal-llm-server:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="kimi-k2.5",
    messages=[{
        "role": "user",
        "content": f"""다음은 화면의 UI 요소 목록입니다:
{elements}

'File' 메뉴를 클릭하려면 어떤 좌표를 클릭해야 합니까?"""
    }]
)
```

---

### OpenAI 호환 API 호출 (vLLM 서빙 후)

vLLM으로 서빙한 모델은 OpenAI API와 동일한 인터페이스로 호출:

```python
import openai
import base64

# vLLM 서버에 연결
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # 로컬 서빙이므로 API 키 불필요
)

# 스크린샷을 base64로 인코딩
with open("screenshot.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# 그라운딩 요청
response = client.chat.completions.create(
    model="ui-venus-8b",  # --served-model-name과 일치
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                },
                {
                    "type": "text",
                    "text": "Click on the 'Run Synthesis' button"
                }
            ]
        }
    ],
    max_tokens=256
)

print(response.choices[0].message.content)
# 예상 출력: 클릭 좌표 또는 액션 지시
```

### curl로 직접 호출

```bash
# 모델 목록 확인
curl http://localhost:8000/v1/models

# 텍스트 전용 요청
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "ui-venus-8b",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 64
    }'
```

---

## 모델별 서빙 명령어 요약

| 모델 | 서빙 명령어 | GPU 수 |
|------|------------|--------|
| UI-Venus-1.5-8B | `vllm serve /data/models/UI-Venus-1.5-8B --dtype bf16 --trust-remote-code` | 1 |
| UI-Venus-1.5-30B-A3B | `vllm serve ... --tensor-parallel-size 2 --trust-remote-code` | 2 |
| MAI-UI-8B | `vllm serve /data/models/MAI-UI-8B --dtype bf16 --trust-remote-code` | 1 |
| GUI-Actor-7B | `vllm serve /data/models/GUI-Actor-7B --dtype bf16 --trust-remote-code` | 1 |
| UI-TARS-1.5-7B | `vllm serve /data/models/UI-TARS-1.5-7B --dtype bf16 --trust-remote-code` | 1 |
| UI-TARS-72B-DPO | `vllm serve ... --tensor-parallel-size 2 --trust-remote-code` | 2 |
| UGround-V1-7B | `vllm serve /data/models/UGround-V1-7B --dtype bf16 --trust-remote-code` | 1 |
| ShowUI-2B | `vllm serve /data/models/ShowUI-2B --dtype bf16 --trust-remote-code` | 1 |
| SeeClick | `transformers` 직접 사용 (Qwen-VL 기반, vLLM 호환성 확인 필요) | 1 |
| ZonUI-3B | `vllm serve /data/models/ZonUI-3B --dtype bf16 --trust-remote-code` | 1 |

---

## 트러블슈팅

### `trust-remote-code` 오류

대부분의 UI VLM은 커스텀 코드를 포함하므로 `--trust-remote-code` 필수:
```bash
vllm serve /data/models/UI-Venus-1.5-8B --trust-remote-code
```

### GPU 메모리 부족 (OOM)

```bash
# GPU 메모리 사용률 제한 (기본 90%)
vllm serve ... --gpu-memory-utilization 0.85

# 최대 모델 길이 제한
vllm serve ... --max-model-len 4096
```

### 모델 로딩 시간이 긴 경우

```bash
# safetensors 형식이면 자동으로 빠른 로딩
# 모델이 .bin 형식이면 safetensors로 변환 가능
python -c "
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('/data/models/MODEL_NAME')
model.save_pretrained('/data/models/MODEL_NAME', safe_serialization=True)
"
```

## 참고 자료

- [vLLM 공식 문서](https://docs.vllm.ai/)
- [vLLM Vision Language Models 가이드](https://docs.vllm.ai/en/stable/examples/offline_inference/vision_language/)
- [SGLang 공식 문서](https://sgl-project.github.io/)
- [OmniParser GitHub](https://github.com/microsoft/OmniParser)

## 관련 문서

- [이전: 오프라인 다운로드 가이드](./offline-download-guide.md)
- [위로: VLM 가이드 인덱스](./README.md)
