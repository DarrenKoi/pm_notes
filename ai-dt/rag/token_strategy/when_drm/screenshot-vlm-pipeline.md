---
tags: [rag, drm, vlm, screenshot, gpt-4o, claude, pipeline]
level: intermediate
last_updated: 2026-02-12
status: complete
---

# Phase 1: 스크린샷 + VLM 기반 추출 파이프라인

> DRM 문서를 스크린샷으로 캡처한 후, Vision Language Model(VLM)을 통해 구조화된 텍스트로 변환하는 파이프라인.

## 왜 필요한가? (Why)

- DRM 파일은 프로그래매틱 파싱 불가 → 화면 캡처가 유일한 입력 경로
- VLM(GPT-4o, Claude 등)은 이미지에서 텍스트, 테이블, 다이어그램을 인식하여 구조화 가능
- 수작업 타이핑 대비 압도적인 속도와 정확도

## 핵심 개념 (What)

### 파이프라인 개요

```
DRM 문서 (뷰어에서 열람)
        │
        ▼
  ┌─────────────┐
  │ 스크린샷 캡처  │  ← 자동화 또는 수동
  └──────┬──────┘
         │  PNG/JPG 이미지들
         ▼
  ┌─────────────┐
  │ 이미지 전처리  │  ← 해상도 보정, 크롭, 정렬
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  VLM 추출    │  ← 문서 유형별 프롬프트
  └──────┬──────┘
         │  구조화된 텍스트 (Markdown)
         ▼
  ┌─────────────┐
  │  후처리/검증  │  ← 품질 검사, 오류 보정
  └──────┬──────┘
         │
         ▼
     청킹 파이프라인
```

### VLM 선택 가이드

| VLM | 장점 | 단점 | 비용 (입력 이미지) |
|-----|------|------|--------------------|
| **GPT-4o** | 테이블/차트 인식 우수, 한국어 지원 | 가격 높음 | ~$0.01-0.03/이미지 |
| **Claude Sonnet 4.5** | 긴 문맥 처리, 정확한 추출 | 이미지 해상도 제한 | ~$0.01-0.02/이미지 |
| **Gemini 2.0 Flash** | 빠르고 저렴, 대량 처리 적합 | 복잡한 레이아웃에서 정확도 ↓ | ~$0.002-0.005/이미지 |
| **Qwen2.5-VL (로컬)** | 무료, 데이터 유출 걱정 없음 | GPU 필요, 정확도 상대적 ↓ | 무료 (GPU 비용) |

> **사내 보안 고려**: DRM 문서의 내용을 외부 API로 전송하는 것이 허용되는지 반드시 확인.
> 보안이 중요하면 **로컬 VLM(Qwen2.5-VL 등)** 또는 **사내 GPU 서버** 활용 검토.

## 어떻게 사용하는가? (How)

### Step 1: 스크린샷 캡처 자동화

#### macOS 자동화 (AppleScript + Python)

```python
import subprocess
import time
import pyautogui
from pathlib import Path

def capture_document_pages(
    output_dir: str,
    num_pages: int,
    delay: float = 1.0,
    scroll_amount: int = 5,
) -> list[str]:
    """DRM 뷰어 화면을 페이지별로 캡처"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    captured = []

    for i in range(num_pages):
        time.sleep(delay)  # 페이지 로딩 대기

        # 전체 화면 캡처
        filename = f"page_{i+1:03d}.png"
        filepath = output_path / filename
        screenshot = pyautogui.screenshot()
        screenshot.save(str(filepath))
        captured.append(str(filepath))

        # 다음 페이지로 이동
        pyautogui.press("pagedown")  # 또는 right, down 등 뷰어에 맞게

    return captured
```

#### Windows 자동화

```python
import win32gui
import win32ui
import win32con
from PIL import Image

def capture_window(window_title: str, output_path: str) -> str:
    """특정 윈도우를 캡처 (DRM 뷰어 창)"""
    hwnd = win32gui.FindWindow(None, window_title)
    if not hwnd:
        raise ValueError(f"Window not found: {window_title}")

    # 윈도우 영역 가져오기
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width = right - left
    height = bottom - top

    # 캡처
    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()

    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
    saveDC.SelectObject(saveBitMap)
    saveDC.BitBlt((0, 0), (width, height), mfcDC, (0, 0), win32con.SRCCOPY)

    saveBitMap.SaveBitmapFile(saveDC, output_path)

    # 정리
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)

    return output_path
```

#### 스크린샷 품질 팁

| 항목 | 권장 설정 | 이유 |
|------|-----------|------|
| **해상도** | 원본 해상도 또는 2x 스케일 | VLM 인식률 향상 |
| **형식** | PNG (무손실) | JPG 압축 아티팩트가 OCR 방해 |
| **뷰어 설정** | 확대 100% 이상, 단일 페이지 뷰 | 글자 크기 확보 |
| **UI 제거** | 문서 영역만 크롭 | 뷰어 UI가 노이즈로 작용 |
| **다크모드** | 비활성화 | 밝은 배경이 인식률 높음 |

### Step 2: 이미지 전처리

```python
from PIL import Image, ImageEnhance, ImageFilter

def preprocess_screenshot(
    image_path: str,
    crop_region: tuple = None,  # (left, top, right, bottom)
    target_width: int = 2048,
) -> Image.Image:
    """스크린샷 전처리: 크롭, 리사이즈, 선명도 보정"""
    img = Image.open(image_path)

    # 1. 뷰어 UI 영역 크롭 (문서 본문만)
    if crop_region:
        img = img.crop(crop_region)

    # 2. 적절한 해상도로 리사이즈 (너무 크면 API 비용 ↑, 너무 작으면 인식 ↓)
    if img.width > target_width:
        ratio = target_width / img.width
        img = img.resize(
            (target_width, int(img.height * ratio)),
            Image.LANCZOS
        )

    # 3. 선명도 보정 (스크린 캡처 시 약간 흐려질 수 있음)
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.3)

    # 4. 대비 보정 (텍스트-배경 구분 향상)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)

    return img


def batch_preprocess(
    image_dir: str,
    crop_region: tuple = None,
) -> list[str]:
    """폴더 내 모든 스크린샷 일괄 전처리"""
    from pathlib import Path

    input_dir = Path(image_dir)
    output_dir = input_dir / "preprocessed"
    output_dir.mkdir(exist_ok=True)

    processed = []
    for img_path in sorted(input_dir.glob("*.png")):
        img = preprocess_screenshot(str(img_path), crop_region)
        output_path = output_dir / img_path.name
        img.save(str(output_path), "PNG")
        processed.append(str(output_path))

    return processed
```

### Step 3: VLM 추출 — 문서 유형별 프롬프트

**프롬프트 설계가 VLM 추출 품질의 80%를 결정한다.**

#### 공통 시스템 프롬프트

```python
SYSTEM_PROMPT = """당신은 문서 이미지에서 텍스트와 구조를 정확하게 추출하는 전문가입니다.

추출 규칙:
1. 모든 텍스트를 빠짐없이 추출합니다.
2. 문서의 구조(제목, 본문, 리스트, 테이블)를 Markdown 형식으로 보존합니다.
3. 테이블은 반드시 Markdown 테이블 형식으로 변환합니다.
4. 다이어그램이나 차트는 [Figure: 내용 설명] 형태로 텍스트 설명합니다.
5. 읽을 수 없는 부분은 [불명확] 으로 표시합니다.
6. 페이지 번호, 헤더/푸터 등 반복 요소는 제외합니다.
7. 한국어와 영어가 혼합된 경우 원문 그대로 유지합니다.
"""
```

#### PowerPoint 전용 프롬프트

```python
PPTX_PROMPT = """이 이미지는 PowerPoint 슬라이드입니다.

다음 구조로 추출하세요:

## 슬라이드 제목
(제목 텍스트)

### 본문
(bullet point, 텍스트 등)

### 테이블
(테이블이 있으면 Markdown 테이블로)

### 다이어그램/차트
(시각적 요소의 텍스트 설명)

### 핵심 키워드
(이 슬라이드의 핵심 키워드 3-5개)

중요:
- bullet point의 계층 구조(들여쓰기)를 보존하세요.
- 도형 안의 텍스트도 빠짐없이 추출하세요.
- 화살표나 흐름도는 "A → B → C" 형태로 표현하세요.
"""
```

#### Excel 전용 프롬프트

```python
XLSX_PROMPT = """이 이미지는 Excel 스프레드시트입니다.

다음 규칙으로 추출하세요:

1. 테이블 전체를 Markdown 테이블 형식으로 변환하세요.
2. 헤더 행을 정확히 식별하세요.
3. 병합된 셀은 해당 값을 모든 관련 셀에 표시하세요.
4. 숫자 데이터는 정확하게 기록하세요 (단위 포함).
5. 시트 이름이 보이면 ## 시트이름 형태로 시작하세요.
6. 빈 셀은 빈 칸으로 두세요.
7. 차트가 있으면 데이터 포인트를 텍스트로 설명하세요.

형식:
## [시트 이름 또는 테이블 제목]

| 헤더1 | 헤더2 | 헤더3 |
|-------|-------|-------|
| 값1   | 값2   | 값3   |
"""
```

#### Word 전용 프롬프트

```python
DOCX_PROMPT = """이 이미지는 Word 문서의 한 페이지입니다.

다음 규칙으로 추출하세요:

1. 제목/소제목은 Markdown 헤더(#, ##, ###)로 변환하세요.
2. 본문 텍스트는 문단 단위로 보존하세요.
3. 번호 목록은 1. 2. 3. 형태, 불릿 목록은 - 형태로 변환하세요.
4. 테이블은 Markdown 테이블로 변환하세요.
5. 강조(볼드, 이탤릭)는 Markdown 서식(**볼드**, *이탤릭*)으로 보존하세요.
6. 각주가 있으면 [^1] 형태로 표시하세요.
7. 이미지/그림은 [Figure: 설명] 으로 표시하세요.
"""
```

### Step 4: VLM API 호출 구현

```python
import base64
import asyncio
from pathlib import Path
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def extract_single_image(
    image_path: str,
    doc_type: str = "general",
    model: str = "gpt-4o",
) -> dict:
    """단일 이미지에서 VLM으로 텍스트 추출"""

    # 문서 유형별 프롬프트 선택
    prompts = {
        "pptx": PPTX_PROMPT,
        "xlsx": XLSX_PROMPT,
        "docx": DOCX_PROMPT,
        "general": "이 문서 이미지의 모든 텍스트와 구조를 Markdown으로 추출하세요.",
    }
    user_prompt = prompts.get(doc_type, prompts["general"])

    # 이미지 인코딩
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{img_b64}",
                    "detail": "high",  # 고해상도 모드
                }}
            ]}
        ],
        max_tokens=4096,
        temperature=0.0,  # 결정적 출력
    )

    return {
        "source_image": image_path,
        "extracted_text": response.choices[0].message.content,
        "model": model,
        "tokens_used": response.usage.total_tokens,
    }
```

### Step 5: 대량 처리 — 비동기 배치

```python
import asyncio
from tqdm.asyncio import tqdm_asyncio

async def extract_document_batch(
    image_paths: list[str],
    doc_type: str = "general",
    model: str = "gpt-4o",
    max_concurrent: int = 5,  # API 동시 요청 제한
) -> list[dict]:
    """여러 이미지를 비동기로 병렬 추출"""

    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_extract(path):
        async with semaphore:
            return await extract_single_image(path, doc_type, model)

    tasks = [limited_extract(p) for p in image_paths]
    results = await tqdm_asyncio.gather(*tasks, desc="VLM 추출 중")

    # 페이지 순서 보장
    results.sort(key=lambda x: x["source_image"])
    return results


async def extract_full_document(
    image_dir: str,
    doc_type: str,
    model: str = "gpt-4o",
) -> dict:
    """전체 문서(여러 페이지) 추출"""
    image_paths = sorted(Path(image_dir).glob("*.png"))

    results = await extract_document_batch(
        [str(p) for p in image_paths],
        doc_type=doc_type,
        model=model,
    )

    # 전체 문서 텍스트 조합
    full_text = "\n\n---\n\n".join(
        f"<!-- Page {i+1} -->\n{r['extracted_text']}"
        for i, r in enumerate(results)
    )

    total_tokens = sum(r["tokens_used"] for r in results)

    return {
        "full_text": full_text,
        "page_results": results,
        "total_pages": len(results),
        "total_tokens": total_tokens,
        "estimated_cost_usd": total_tokens * 0.000005,  # 대략적 추정
    }
```

### Step 6: 추출 결과 검증

VLM 추출은 100% 정확하지 않다. 자동 품질 검사 적용.

```python
def validate_extraction(result: dict) -> dict:
    """추출 결과 품질 검사"""
    text = result["extracted_text"]
    issues = []

    # 1. 텍스트 최소 길이 확인
    if len(text.strip()) < 20:
        issues.append("WARNING: 추출된 텍스트가 너무 짧음 - 빈 페이지이거나 추출 실패")

    # 2. [불명확] 마커 확인
    unclear_count = text.count("[불명확]")
    if unclear_count > 3:
        issues.append(f"WARNING: 불명확한 부분 {unclear_count}건 - 이미지 품질 확인 필요")

    # 3. 테이블 구조 검증 (파이프 문자 일관성)
    lines = text.split("\n")
    table_lines = [l for l in lines if l.strip().startswith("|")]
    if table_lines:
        pipe_counts = [l.count("|") for l in table_lines]
        if len(set(pipe_counts)) > 2:  # 헤더 구분선 포함하여 2가지까지 허용
            issues.append("WARNING: 테이블 열 수가 일관되지 않음 - 수동 확인 필요")

    # 4. 반복 패턴 감지 (VLM hallucination)
    for i in range(len(lines) - 2):
        if lines[i] == lines[i+1] == lines[i+2] and lines[i].strip():
            issues.append(f"WARNING: 반복 텍스트 감지 (line {i+1}) - hallucination 의심")
            break

    return {
        **result,
        "quality_issues": issues,
        "quality_score": max(0, 100 - len(issues) * 20),
    }
```

## 비용 최적화 전략

### 티어별 모델 사용

모든 페이지에 GPT-4o를 쓰면 비용이 폭증한다. **문서 복잡도에 따라 모델을 분리**:

```python
async def smart_extract(image_path: str, doc_type: str) -> dict:
    """복잡도에 따라 VLM 모델 자동 선택"""

    # 1차: 저비용 모델로 빠른 추출 시도
    result = await extract_single_image(
        image_path, doc_type, model="gemini-2.0-flash"
    )

    # 품질 검사
    validated = validate_extraction(result)

    # 품질 낮으면 고비용 모델로 재추출
    if validated["quality_score"] < 60:
        result = await extract_single_image(
            image_path, doc_type, model="gpt-4o"
        )
        result["fallback"] = True

    return result
```

### 비용 추정 테이블

100페이지 문서 기준 대략적 추정:

| 전략 | 모델 | 추정 비용 | 처리 시간 |
|------|------|-----------|-----------|
| 전부 GPT-4o | GPT-4o | ~$2-3 | ~5분 |
| 전부 Gemini Flash | Gemini 2.0 Flash | ~$0.3-0.5 | ~2분 |
| 티어별 분리 | Flash + GPT-4o 폴백 | ~$0.5-1.0 | ~3분 |
| 로컬 VLM | Qwen2.5-VL 72B | $0 (GPU 비용) | ~10-15분 |

## 로컬 VLM 파이프라인 (보안 우선)

사내 문서를 외부 API로 보내지 못하는 경우:

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch

class LocalVLMExtractor:
    """로컬 GPU에서 실행하는 VLM 추출기"""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def extract(self, image_path: str, prompt: str) -> str:
        image = Image.open(image_path)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]}
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image],
            return_tensors="pt", padding=True
        ).to(self.model.device)

        output_ids = self.model.generate(
            **inputs, max_new_tokens=4096, temperature=0.0
        )
        output_text = self.processor.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]

        return output_text
```

## 참고 자료 (References)

- [GPT-4o Vision Guide](https://platform.openai.com/docs/guides/vision)
- [Claude Vision Documentation](https://docs.anthropic.com/en/docs/build-with-claude/vision)
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
- [pyautogui Documentation](https://pyautogui.readthedocs.io/)

## 관련 문서

- [VLM 추출 결과 청킹 전략](./vlm-chunking-strategy.md)
- [DRM 해제 후 하이브리드 전략](./post-drm-hybrid.md)
- [청킹 방법론 총론](../overview-chunking-methods.md)
