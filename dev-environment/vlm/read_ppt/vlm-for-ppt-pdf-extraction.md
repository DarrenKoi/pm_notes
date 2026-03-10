---
tags: [vlm, ocr, ppt, pdf, document-extraction, vllm, h200, structured-output]
level: intermediate
last_updated: 2026-03-10
status: in-progress
---

# PPT/PDF 슬라이드 이미지 → 구조화된 텍스트 추출을 위한 VLM 리서치

> DRM-protected PPT/PDF를 이미지로 캡처한 뒤, VLM으로 텍스트·테이블·다이어그램을 JSON 또는 plain text로 추출하기 위한 모델 비교와 실전 배포 가이드.

## 왜 필요한가? (Why)

- **DRM → 이미지 캡처 → VLM이 유일한 경로**: 사내 문서의 99%가 DRM 보호 상태이므로 프로그래매틱 파싱이 불가능하다
- **PPT 슬라이드의 특수성**: 텍스트 박스, 다이어그램, 표, 이미지, 화살표 흐름도가 혼합된 비정형 레이아웃
- **목표**: 슬라이드별 **JSON**(title, text_boxes, tables, images, notes) 또는 **plain text** 추출
- **기존 문서와의 차이점**:
  - `dev-environment/vlm/` → UI 특화 VLM 배포 문서 (GUI Agent용)
  - `ai-dt/rag/token_strategy/when_drm/` → RAG 파이프라인 관점의 스크린샷+VLM 파이프라인
  - **이 문서** → "PPT/PDF 슬라이드 → 구조화된 텍스트 추출"에 최적인 **모델 비교 + 배포 + 프롬프트**에 집중

## 핵심 개념 (What) — H200 2GPU 모델 비교표

하드웨어: **H200 2GPU** (2 × 141GB = **282GB VRAM**)

### 대형 범용 VLM (문서 이해 + 의미 추론)

| 모델 | 파라미터 | VRAM (BF16) | 2×H200 적합 | 문서 추출 성능 | 비고 |
|------|---------|-------------|-------------|--------------|------|
| **Qwen2.5-VL-72B-Instruct** | 73B | ~144GB | **O** (TP=2) | **최고** — DocVQA 96.4, OCRBench 888, CC-OCR 1위 | 문서 이해 1순위 추천. FP8(~72GB)도 가능 |
| **InternVL3-78B** | 78.4B | ~160GB | **O** (TP=2) | SOTA급 — DocVQA 95.4, OCRBench 906 | Qwen2.5-72B LLM 백본, 다국어 강점 |

### 중형 VLM (단일 GPU 가능)

| 모델 | 파라미터 | VRAM (BF16) | 2×H200 적합 | 문서 추출 성능 | 비고 |
|------|---------|-------------|-------------|--------------|------|
| Qwen3-VL-32B | 33B | ~64GB | O (1GPU) | DocVQA 96.5, OCRBench 875 | FP8(~32GB) 가능. 속도 vs 72B 트레이드오프 |
| Qwen3-VL-8B-Instruct | 8.8B | ~18GB | O (1GPU) | DocVQA 96, **OCRBench 896-905** | **사내 배포됨**. 8B임에도 OCRBench가 32B보다 높음 |

### OCR/문서 특화 소형 모델 (레이아웃 분석 전문)

| 모델 | 파라미터 | VRAM (BF16) | 2×H200 적합 | 문서 추출 성능 | 비고 |
|------|---------|-------------|-------------|--------------|------|
| **dots.ocr-1.5** | 3B | ~7GB | O (1GPU) | olmOCR-bench 79.1, OmniDocBench 83-88 | rednote-hilab. 레이아웃+OCR+차트→SVG를 통합한 end-to-end 모델 |
| **olmOCR-2-7B** | ~8B | ~16GB | O (1GPU) | **olmOCR-bench 82.4** (문서 파싱 SOTA) | Qwen2.5-VL-7B 파인튜닝 + GRPO RL. FP8(~8GB) 가능 |

### 도구/파이프라인

| 도구 | 유형 | 메모 |
|------|------|------|
| **MinerU 2.5** | 파이프라인 (VLM 아님) | 내부에 MinerU2.5-1.2B VLM 포함. 2단계 구조: (1) 다운샘플 썸네일로 레이아웃 분석 (2) 원본 해상도로 세밀 인식. 109개 언어 지원. VRAM 6-25GB. **DRM 해제된 PDF에만 사용 가능** |

### 사내 API로 이미 제공되는 모델

아래 모델은 **사내 API 서버에서 이미 서빙 중**이므로, H200에 직접 배포할 필요 없이 API 호출로 바로 사용 가능하다:

| 모델 | 용도 |
|------|------|
| **Qwen3-VL-30B-Instruct** | 고성능 VLM (복잡한 슬라이드 추출) |
| **Qwen3-VL-8B-Instruct** | 경량 VLM (대량 처리, 빠른 추출) |
| **Kimi-K2.5** | 텍스트 LLM (후처리, 요약, 메타데이터 보강) |
| **GLM-4.7** | 텍스트 LLM (대안) |

→ H200 GPU는 **사내 API에 없는 모델** (Qwen2.5-VL-72B, InternVL3-78B, dots.ocr-1.5, olmOCR-2 등)을 직접 배포할 때만 필요하다.

### 모델 선택 가이드

```
빠른 대량 처리 (단순 텍스트 위주 슬라이드)
  → Qwen3-VL-8B (사내 API, GPU 불필요)

복잡한 슬라이드 (테이블, 다이어그램 포함)
  → Qwen3-VL-30B (사내 API, GPU 불필요)

레이아웃 정밀 분석 (표, bounding box 필요)
  → dots.ocr-1.5 (H200에 배포, 1GPU, 초경량)

최고 품질 구조화 추출
  → Qwen2.5-VL-72B-Instruct (H200에 배포, TP=2)

2단계 파이프라인 (권장)
  → 1단계 dots.ocr-1.5/olmOCR-2 (H200) + 2단계 Qwen2.5-VL-72B (H200)
  → 또는 1단계 dots.ocr-1.5 (H200) + 2단계 Qwen3-VL-30B (사내 API)
```

## 어떻게 사용하는가? (How)

### 1. vLLM 배포 커맨드

모든 명령의 공통 전제:

```bash
export HF_HUB_OFFLINE=1
export VLLM_API_KEY='change-this-internal-key'
```

#### Qwen2.5-VL-72B-Instruct (TP=2, 2GPU 필수)

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve /data/models/Qwen2.5-VL-72B-Instruct \
  --served-model-name qwen25-vl-72b \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --trust-remote-code \
  --api-key "$VLLM_API_KEY" \
  --limit-mm-per-prompt '{"image":4}'
```

- `--max-model-len 32768`: 슬라이드 이미지(고해상도) + JSON 출력을 고려한 설정. 메모리 부족 시 `16384`로 낮춘다
- `--limit-mm-per-prompt '{"image":4}'`: 한 요청에 슬라이드 최대 4장. 메모리 상태에 따라 조절
- BF16 기준 ~144GB → 2×H200(282GB)에서 KV cache 여유 충분
- FP8 사용 시 ~72GB → 단일 H200에도 가능하지만, KV cache 여유를 위해 TP=2 권장

#### InternVL3-78B (TP=2, 2GPU 필수)

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve /data/models/InternVL3-78B \
  --served-model-name internvl3-78b \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --trust-remote-code \
  --api-key "$VLLM_API_KEY" \
  --limit-mm-per-prompt '{"image":4}'
```

#### dots.ocr-1.5 (1GPU, 초경량)

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /data/models/dots.ocr-1.5 \
  --served-model-name dots-ocr \
  --host 0.0.0.0 \
  --port 8001 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 16384 \
  --trust-remote-code \
  --api-key "$VLLM_API_KEY" \
  --limit-mm-per-prompt '{"image":1}'
```

- ~7GB만 사용하므로 다른 모델과 **동시 배포 가능**
- 예: GPU 0에 dots.ocr(8001), GPU 0,1에 Qwen2.5-VL-72B(8000) → dots.ocr은 워낙 작아서 같은 GPU에 올려도 됨
  - 단, 이 경우 Qwen2.5-VL-72B를 먼저 띄우고, dots.ocr은 남는 VRAM에 올린다

#### olmOCR-2-7B (1GPU)

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve /data/models/olmOCR-2-7B \
  --served-model-name olmocr-7b \
  --host 0.0.0.0 \
  --port 8002 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 16384 \
  --trust-remote-code \
  --api-key "$VLLM_API_KEY" \
  --limit-mm-per-prompt '{"image":1}'
```

#### 2단계 파이프라인 동시 배포 (권장 구성)

```
GPU 0 + GPU 1: Qwen2.5-VL-72B (TP=2) → port 8000
GPU 0 여유분:   dots.ocr-1.5           → port 8001  (별도 프로세스)
```

> dots.ocr이 ~7GB만 쓰므로, TP=2로 잡힌 72B 모델의 GPU 0 여유 VRAM에 함께 올릴 수 있다.
> 만약 VRAM이 부족하면 72B 모델을 내리고 dots.ocr만 따로 돌린 뒤, 결과를 저장하고 모델을 교체하는 **순차 실행** 방식도 가능하다.

### 2. JSON 추출 프롬프트 예시

#### PPT 슬라이드용 JSON 스키마

```json
{
  "slide_number": 1,
  "title": "슬라이드 제목",
  "text_boxes": [
    {"text": "본문 텍스트 내용", "type": "bullet|paragraph|label"}
  ],
  "tables": [
    {
      "caption": "테이블 설명",
      "headers": ["열1", "열2", "열3"],
      "rows": [
        ["값1", "값2", "값3"]
      ]
    }
  ],
  "diagrams": [
    {"description": "화살표 흐름도: A → B → C", "type": "flowchart|chart|diagram"}
  ],
  "images": [
    {"description": "사진/그림 설명", "location": "우측 상단"}
  ],
  "notes": "기타 특이사항"
}
```

#### 방법 A: vLLM `--guided-json` (Structured Output)

vLLM은 `guided_json` 파라미터로 출력을 JSON 스키마에 강제할 수 있다.

요청 시 `extra_body`에 JSON 스키마를 넘긴다:

```python
import base64
import json
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="change-this-internal-key",
)

# JSON 스키마 정의
slide_schema = {
    "type": "object",
    "properties": {
        "slide_number": {"type": "integer"},
        "title": {"type": "string"},
        "text_boxes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "type": {"type": "string", "enum": ["bullet", "paragraph", "label"]}
                },
                "required": ["text", "type"]
            }
        },
        "tables": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "caption": {"type": "string"},
                    "headers": {"type": "array", "items": {"type": "string"}},
                    "rows": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}}
                },
                "required": ["headers", "rows"]
            }
        },
        "diagrams": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "type": {"type": "string"}
                },
                "required": ["description"]
            }
        },
        "images": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "location": {"type": "string"}
                },
                "required": ["description"]
            }
        },
        "notes": {"type": "string"}
    },
    "required": ["title", "text_boxes"]
}

# 이미지 인코딩
with open("slide_001.png", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="qwen25-vl-72b",
    messages=[
        {
            "role": "system",
            "content": "You are a document extraction expert. Extract all content from the slide image into the specified JSON structure. Be thorough and accurate."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Extract all content from this PPT slide into JSON format. Include every text box, table, diagram, and image description."
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                }
            ]
        }
    ],
    max_tokens=4096,
    temperature=0.0,
    extra_body={
        "guided_json": json.dumps(slide_schema)
    }
)

result = json.loads(response.choices[0].message.content)
print(json.dumps(result, ensure_ascii=False, indent=2))
```

> **주의**: `guided_json`은 vLLM의 guided decoding 기능이다. 모델이 JSON 스키마를 벗어나는 토큰을 생성하지 못하도록 logit masking을 적용한다. 속도가 약간 느려질 수 있지만 출력 형식이 보장된다.

#### 방법 B: 프롬프트 내 JSON 스키마 명시 (Fallback)

`guided_json`이 지원되지 않거나 호환 이슈가 있을 때 사용한다:

```python
PROMPT_WITH_SCHEMA = """이 PPT 슬라이드 이미지에서 모든 내용을 추출하세요.

반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트는 포함하지 마세요.

{
  "title": "슬라이드 제목",
  "text_boxes": [
    {"text": "텍스트 내용", "type": "bullet|paragraph|label"}
  ],
  "tables": [
    {"caption": "설명", "headers": ["열1"], "rows": [["값1"]]}
  ],
  "diagrams": [
    {"description": "다이어그램 설명", "type": "flowchart|chart|diagram"}
  ],
  "images": [
    {"description": "이미지 설명", "location": "위치"}
  ],
  "notes": "기타 사항"
}

규칙:
1. 모든 텍스트 박스를 빠짐없이 추출
2. 테이블은 모든 셀 데이터 포함
3. 다이어그램/흐름도는 "A → B → C" 형태로 기술
4. 한국어/영어 원문 그대로 유지
5. 빈 항목은 빈 배열 []로"""
```

#### 방법 C: Plain Text 추출 (간단 버전)

JSON이 필요 없고 텍스트만 추출하면 될 때:

```python
PLAIN_TEXT_PROMPT = """이 PPT 슬라이드의 모든 텍스트를 Markdown 형식으로 추출하세요.

규칙:
- 슬라이드 제목은 ## 헤더로
- bullet point는 - 로
- 테이블은 Markdown 테이블로
- 다이어그램은 [Figure: 설명] 으로
- 도형 안 텍스트도 빠짐없이 추출
- 읽을 수 없는 부분은 [불명확]으로 표시"""
```

### 3. 2단계 전략 (권장)

#### 왜 2단계인가?

| 관점 | OCR 특화 모델 (dots.ocr) | 대형 VLM (Qwen2.5-VL-72B) |
|------|-------------------------|--------------------------|
| **레이아웃 분석** | **최고** — bounding box, 텍스트 위치 정밀 | 보통 — 대략적 구조만 파악 |
| **OCR 정확도** | **최고** — 작은 글씨, 회전 텍스트에 강함 | 좋음 — 대부분 정확하지만 특수 케이스에서 밀림 |
| **의미적 이해** | **약함** — "이 도표가 뭘 말하는지" 모름 | **최고** — 문맥 파악, 요약, 구조화 |
| **JSON 구조화** | 제한적 | **강함** — 복잡한 스키마도 따름 |

→ **결론**: 각자의 강점을 결합하면 최고 품질

#### 파이프라인 구조

```
슬라이드 이미지
      │
      ▼
┌─────────────────────┐
│ 1단계: dots.ocr-1.5 │  ← port 8001
│                     │
│ - 레이아웃 분석      │
│ - OCR (텍스트 위치)  │
│ - 테이블 구조 감지    │
│ - bounding box 출력  │
└──────────┬──────────┘
           │  raw OCR text + layout info
           ▼
┌──────────────────────────┐
│ 2단계: Qwen2.5-VL-72B   │  ← port 8000
│                          │
│ - 원본 이미지 + OCR 결과  │
│ - 의미적 이해/보정        │
│ - 구조화된 JSON 생성      │
│ - 다이어그램 해석         │
└──────────┬───────────────┘
           │  structured JSON
           ▼
      최종 결과
```

#### 2단계 파이프라인 코드

```python
import base64
import json
from openai import OpenAI

# 두 모델에 대한 클라이언트 (같은 서버, 다른 포트)
ocr_client = OpenAI(
    base_url="http://127.0.0.1:8001/v1",
    api_key="change-this-internal-key",
)
vlm_client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="change-this-internal-key",
)


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def stage1_ocr(image_path: str) -> str:
    """1단계: dots.ocr로 레이아웃 + OCR 추출"""
    img_b64 = encode_image(image_path)

    response = ocr_client.chat.completions.create(
        model="dots-ocr",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract all text from this document image with layout information. Include bounding boxes if possible. Preserve the reading order."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                    }
                ]
            }
        ],
        max_tokens=4096,
        temperature=0.0,
    )
    return response.choices[0].message.content


def stage2_structure(image_path: str, ocr_text: str) -> dict:
    """2단계: 대형 VLM으로 의미 이해 + JSON 구조화"""
    img_b64 = encode_image(image_path)

    prompt = f"""아래는 이 PPT 슬라이드 이미지에서 OCR 모델이 추출한 raw text입니다.

--- OCR 결과 ---
{ocr_text}
--- OCR 결과 끝 ---

위 OCR 결과와 원본 이미지를 함께 참고하여, 슬라이드 내용을 아래 JSON 구조로 정리하세요.
OCR이 놓친 내용이 있으면 이미지에서 직접 보완하세요.
OCR 오류가 있으면 이미지를 기준으로 교정하세요.

JSON 형식:
{{
  "title": "슬라이드 제목",
  "text_boxes": [{{"text": "내용", "type": "bullet|paragraph|label"}}],
  "tables": [{{"caption": "설명", "headers": ["열"], "rows": [["값"]]}}],
  "diagrams": [{{"description": "설명", "type": "flowchart|chart|diagram"}}],
  "images": [{{"description": "설명", "location": "위치"}}],
  "notes": "기타"
}}"""

    response = vlm_client.chat.completions.create(
        model="qwen25-vl-72b",
        messages=[
            {
                "role": "system",
                "content": "You are a document extraction expert. Always respond with valid JSON only."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                ]
            }
        ],
        max_tokens=4096,
        temperature=0.0,
    )

    raw = response.choices[0].message.content
    # JSON 블록 추출 (```json ... ``` 감싸기 대응)
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0]
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0]

    return json.loads(raw)


def extract_slide(image_path: str) -> dict:
    """2단계 파이프라인으로 슬라이드 추출"""
    # Stage 1: OCR
    ocr_text = stage1_ocr(image_path)

    # Stage 2: 구조화
    structured = stage2_structure(image_path, ocr_text)

    return structured


# 사용 예시
if __name__ == "__main__":
    result = extract_slide("slide_001.png")
    print(json.dumps(result, ensure_ascii=False, indent=2))
```

#### 대량 처리 (비동기)

```python
import asyncio
from openai import AsyncOpenAI
from pathlib import Path

async_ocr = AsyncOpenAI(base_url="http://127.0.0.1:8001/v1", api_key="change-this-internal-key")
async_vlm = AsyncOpenAI(base_url="http://127.0.0.1:8000/v1", api_key="change-this-internal-key")


async def extract_slide_async(image_path: str, semaphore: asyncio.Semaphore) -> dict:
    """비동기 2단계 추출 (동시 요청 제한 포함)"""
    async with semaphore:
        img_b64 = encode_image(image_path)

        # Stage 1
        ocr_resp = await async_ocr.chat.completions.create(
            model="dots-ocr",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "Extract all text with layout from this slide."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
            ]}],
            max_tokens=4096, temperature=0.0,
        )
        ocr_text = ocr_resp.choices[0].message.content

        # Stage 2
        structured_resp = await async_vlm.chat.completions.create(
            model="qwen25-vl-72b",
            messages=[
                {"role": "system", "content": "Document extraction expert. Respond with valid JSON only."},
                {"role": "user", "content": [
                    {"type": "text", "text": f"OCR result:\n{ocr_text}\n\nStructure this into JSON with title, text_boxes, tables, diagrams, images, notes."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                ]}
            ],
            max_tokens=4096, temperature=0.0,
        )
        raw = structured_resp.choices[0].message.content
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        return {"source": image_path, "data": json.loads(raw)}


async def batch_extract(image_dir: str, max_concurrent: int = 3) -> list[dict]:
    """폴더 내 모든 슬라이드 일괄 추출"""
    paths = sorted(Path(image_dir).glob("*.png"))
    sem = asyncio.Semaphore(max_concurrent)
    tasks = [extract_slide_async(str(p), sem) for p in paths]
    return await asyncio.gather(*tasks)
```

### 4. 단일 모델 전략 (간단 버전)

2단계가 과하다면, **Qwen2.5-VL-72B 하나로 직접 추출**하는 것도 충분히 실용적이다:

```python
def extract_slide_single(image_path: str) -> dict:
    """단일 모델로 바로 JSON 추출"""
    img_b64 = encode_image(image_path)

    response = vlm_client.chat.completions.create(
        model="qwen25-vl-72b",
        messages=[
            {
                "role": "system",
                "content": "당신은 PPT 슬라이드에서 모든 내용을 정확하게 추출하는 전문가입니다. 항상 유효한 JSON으로만 응답하세요."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT_WITH_SCHEMA  # 위에서 정의한 JSON 스키마 포함 프롬프트
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                    }
                ]
            }
        ],
        max_tokens=4096,
        temperature=0.0,
    )

    raw = response.choices[0].message.content
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0]
    return json.loads(raw)
```

## 벤치마크 요약

### 문서 이해/OCR 벤치마크

| 벤치마크 | 측정 내용 | 비고 |
|----------|----------|------|
| **DocVQA** | 문서 이미지 QA 정확도 | 가장 널리 쓰이는 문서 VLM 벤치 |
| **CC-OCR** | 다국어 OCR 정확도 (중/영/일/한 등) | 실무 OCR 성능 반영 |
| **olmOCR-bench** | 문서 레이아웃 + OCR 종합 | Allen AI 제안, 실전 문서 파싱 특화 |
| **ChartQA** | 차트/그래프 이해 | PPT 차트 추출 시 참고 |
| **TextVQA** | 이미지 내 텍스트 인식 QA | 사진 속 텍스트 인식 능력 |

### 주요 모델 벤치마크 점수 (참고용)

| 모델 | 파라미터 | VRAM BF16 | VRAM FP8 | DocVQA | OCRBench | olmOCR-bench | 비고 |
|------|---------|-----------|----------|--------|----------|-------------|------|
| **Qwen2.5-VL-72B** | 73B | 144GB | 72GB | **96.4** | 888 | — | CC-OCR 1위, 문서 이해 최강 |
| **InternVL3-78B** | 78.4B | ~160GB | AWQ 가능 | 95.4 | **906** | — | 다국어 강점 |
| Qwen3-VL-32B | 33B | ~64GB | ~32GB | **96.5** | 875 | — | 단일 GPU 가능 |
| Qwen3-VL-8B | 8.8B | ~18GB | ~9GB | 96.0 | **896-905** | — | 사내 배포됨. 8B임에도 OCR 최강급 |
| **olmOCR-2-7B** | ~8B | ~16GB | ~8GB | — | — | **82.4** | 문서 파싱 SOTA |
| dots.ocr-1.5 | 3B | ~7GB | — | — | — | 79.1 | end-to-end OCR+레이아웃+SVG |
| MinerU 2.5 | 1.2B (VLM) | 6-25GB | — | — | — | 75.8 | 파이프라인 도구 |

> **주목**: Qwen3-VL-8B의 OCRBench 점수(896-905)가 32B(875)보다 높다. 이는 8B 모델만으로도 OCR 정확도가 충분할 수 있음을 시사한다. 다만 의미적 이해/구조화 능력은 대형 모델이 우위.

> 벤치마크 점수는 모델 릴리즈 시점의 공개 수치. 실제 PPT 슬라이드에서의 성능은 프롬프트와 이미지 품질에 크게 좌우된다. 반드시 자체 샘플로 비교 평가할 것.

## 참고 자료 (References)

### 모델
- [Qwen2.5-VL GitHub](https://github.com/QwenLM/Qwen2.5-VL) — 문서 이해 성능 벤치마크 포함
- [Qwen2.5-VL-72B HuggingFace](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct) — FP8 버전: `RedHatAI/Qwen2.5-VL-72B-Instruct-FP8-dynamic`
- [InternVL3 GitHub](https://github.com/OpenGVLab/InternVL) — 멀티모달 벤치마크 최상위권
- [dots.ocr GitHub](https://github.com/rednote-hilab/dots.ocr) / [HuggingFace](https://huggingface.co/rednote-hilab/dots.ocr-1.5) — end-to-end OCR+레이아웃 통합 모델
- [olmOCR-2 Blog](https://allenai.org/blog/olmocr-2) / [GitHub](https://github.com/allenai/olmocr) — Allen AI 문서 파싱 프로젝트. FP8: `allenai/olmOCR-2-7B-1025-FP8`
- [MinerU GitHub](https://github.com/opendatalab/MinerU) — PDF→Markdown 파이프라인
- [Qwen3-VL GitHub](https://github.com/QwenLM/Qwen3-VL) — Qwen3-VL-8B/32B 벤치마크

### vLLM 배포
- [vLLM Structured Output Docs](https://docs.vllm.ai/en/latest/features/structured_outputs/) — `guided_json` 사용법
- [vLLM OpenAI Compatible Server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
- [vLLM Qwen2.5-VL Recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen2.5-VL.html)

### guided_json 주의사항
- 범용 VLM (Qwen2.5-VL-72B, Qwen3-VL 등)에서는 `guided_json`이 잘 동작
- OCR 특화 모델 (olmOCR-2, dots.ocr)은 특정 출력 형식(Markdown, layout JSON)으로 학습되어 있어, 다른 JSON 스키마를 `guided_json`으로 강제하면 품질 저하 가능성이 있음

## 관련 문서

- [스크린샷 + VLM 기반 추출 파이프라인](../../../ai-dt/rag/token_strategy/when_drm/screenshot-vlm-pipeline.md) — RAG 파이프라인 관점
- [VLM Cloud Notes (README)](../README.md) — UI 특화 VLM 배포 기본 문서
- [Private Cloud에서 vLLM 시작](../private-cloud-vllm-next-steps.md) — H200 환경 셋업
- [로컬 PC에서 이미지 전송](../local-pc-vllm-image-guide.md) — requests 기반 이미지 전송
