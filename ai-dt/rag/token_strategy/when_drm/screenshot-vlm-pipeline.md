---
tags: [rag, drm, vlm, screenshot, qwen3-vl, kimi-k2, pipeline]
level: intermediate
last_updated: 2026-02-12
reviewed_on: 2026-10-04
review_status: partial
document_type: learning_proposal
---

# Phase 1: 스크린샷 + VLM 기반 추출 파이프라인

> 승인된 화면 이미지에서 텍스트·구조를 추출하는 학습 제안이다. DRM 보호 해제나 실제 뷰어 조작을 수행한 기록은 아니다.

> [!warning] 검토 범위 — 2026-10-04
> 원래 2026-02-12 제안의 사내 방화벽·모델 배포·과금·성능은 미확인이다. 공식 모델 카드와 API·Pillow 문서를 확인하고 생성 이미지/모의 HTTP로 예제를 검증한다. 실제 화면 취득·DRM 권한·VLM 품질·GPU 성능은 검증하지 않았다. Herdr 현재 pane 연결 실패로 Claude와의 모델 선택/정책 협의는 보류한다.

읽기 순서: [입력 조건과 전체 흐름](./README.md) → 이 문서 → [추출 결과 청킹](./vlm-chunking-strategy.md). 화면만으로 원본 수식·숨은 셀·잘린 영역을 복원할 수 없다. 페이지/문서 식별 정보는 이미지 밖의 manifest로 보존하고 모르는 페이지 번호는 `None`으로 남긴다.

## 왜 필요한가? (Why)

- 파서 접근이 제한된 경우 승인된 화면 이미지가 입력 후보가 된다. 파일 확장자나 파싱 실패만으로 DRM 여부/권한/유일 경로를 판단할 수 없다.
- VLM(Qwen3-VL 등)은 이미지에서 텍스트, 테이블, 다이어그램을 인식하여 구조화 가능
- 반복 전사 작업을 줄이는 것이 목적이다. 속도·정확도는 대표 자료의 원문 대조 평가로 확인해야 한다.

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

### 사내 VLM 환경

> **2026-02-12 당시 제안의 환경 가정**: 외부 API 차단/사내 모델 사용 전제. 현재 네트워크·서비스·배포 모델 목록을 확인하지 않았으며 모델 공개와 사내 제공은 별개다.

#### 사용 가능한 모델

| 모델 | 유형 | 특징 | 용도 |
|------|------|------|------|
| **Qwen3-VL-8B-Instruct** | VLM | 공식 공개 모델 확인 | 전사 후보; 사내 alias/속도/품질 미확인 |
| **Qwen3-VL-30B-A3B-Instruct** | VLM | 공식 모델명 확인; 기존 `Qwen3-VL-30B`는 미확인 alias | 복잡한 문서 후보; 8B 대비 우위 미확인 |
| **Kimi-K2.5** | 네이티브 멀티모달 | 공식 카드에서 시각 입력 지원 확인 | 텍스트 후처리 용도로만 사용한다는 것은 원래 역할 제안 |

공식 공개 모델명과 실제 API의 served model name은 같다고 가정하지 않는다. 라이선스·배포 버전·시각 입력 형식·출력 한도는 실제 제공 서비스별로 확인한다.

#### 모델 선택 전략

아래는 당시 역할 분담 제안이다. 괄호의 속도/충분한 품질/정확도는 측정 결과가 아니며 실제 선택은 협의·평가 대기다.

```
단순 텍스트 위주 문서 (Word, 단순 PPT)
  → Qwen3-VL-8B-Instruct (빠르고 충분한 품질)

테이블/다이어그램/차트 포함 문서 (Excel, 복잡한 PPT, 기술 보고서)
  → Qwen3-VL-30B (정확도 우선)

추출 후 메타데이터 보강, 요약, 키워드 추출
  → Kimi-K2.5 (텍스트 LLM으로 충분)
```

## 어떻게 사용하는가? (How)

### Step 1: 스크린샷 캡처 자동화

#### macOS 자동화 (AppleScript + Python)

원래 제목을 유지한다. 아래는 Python 콜백 예시이며 AppleScript 구현은 포함하지 않는다. 기존 `sleep`+전체 화면+`pagedown` 루프를 창/문서/페이지 확인 콜백으로 바꾸었다. PyAutoGUI `region`은 `(left, top, width, height)`이고 Pillow crop은 `(left, top, right, bottom)`이므로 혼동하지 않는다.

```python
from pathlib import Path
from collections.abc import Callable
from PIL import Image


def capture_document_pages(
    output_dir: str,
    page_numbers: list[int],
    capture: Callable[[], Image.Image],
    prepare_page: Callable[[int], None],
) -> list[dict]:
    """승인된 대상의 페이지를 준비·확인한 뒤 캡처. 실제 페이지 번호는 호출자가 제공."""
    if any(type(n) is not int or n < 1 for n in page_numbers):
        raise ValueError("positive page numbers required")
    if len(set(page_numbers)) != len(page_numbers):
        raise ValueError("duplicate page numbers")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)  # 기존 캡처 덮어쓰기 방지
    records = []
    for n in page_numbers:
        prepare_page(n)  # 창/문서/실제 페이지/완전 로딩 확인; sleep만으로 보장하지 않음
        with capture() as image:
            target = output / f"page_{n:05d}.png"
            image.save(target, format="PNG")
        records.append({"source_image": str(target), "page_number": n})
    return records

# 실제 승인된 화면에서만 연결한다. 이 문서 검증에서는 생성 이미지 콜백만 사용.
# import pyautogui
# capture = lambda: pyautogui.screenshot(region=(left, top, width, height))
# prepare_page는 뷰어별 이동/확인 함수다. 마지막 페이지 뒤에 추가 이동하지 않는다.
```

#### Windows 자동화

기존 Win32 `GetWindowDC`/`BitBlt`/`SaveBitmapFile` 예시는 BMP를 `.png`로 저장할 수 있었고 bitmap 해제/예외 정리가 빠졌다. 여기서는 Pillow의 명시 PNG 저장으로 대체했다. Win32 구현을 유지할 경우 선택한 bitmap을 원래 객체로 복구한 뒤 `DeleteObject`하고 DC 수명/`ReleaseDC`를 예외 경로까지 관리해야 한다. 실제 Windows·보호 창에서는 실행하지 않았다.

```python
from pathlib import Path
from PIL import ImageGrab


def capture_window(hwnd: int, output_path: str) -> str:
    """Windows Pillow12.3 예시. 호출자가 확인한 창 핸들을 사용한다."""
    if type(hwnd) is not int or hwnd <= 0:
        raise ValueError("verified window handle required")
    target = Path(output_path)
    if target.suffix.lower() != ".png":
        raise ValueError("PNG output required")
    # Windows의 window 인자 지원은 Pillow11.2.1부터. 실제 DRM 창 동작은 미검증.
    with ImageGrab.grab(window=hwnd) as image:
        with target.open("xb") as output:  # 기존 파일 보존
            image.save(output, format="PNG")
    return str(target)
```

#### 스크린샷 품질 팁

아래 설정은 원래 실험 후보다. 2x 보간 확대나 밝은 배경이 항상 인식률을 높인다는 근거는 없다. 실제 글자 크기·클리핑·원본 해상도와 서버 이미지 처리 조건을 대조한다.

| 항목 | 권장 설정 | 이유 |
|------|-----------|------|
| **해상도** | 원본 해상도 또는 2x 스케일 | 원본 글자 보존; 2x 인식률 향상 미확인 |
| **형식** | PNG (무손실) | 손실 압축의 영향은 자료별 평가 |
| **뷰어 설정** | 확대 100% 이상, 단일 페이지 뷰 | 글자 크기 확보 |
| **UI 제거** | 문서 영역만 크롭 | 뷰어 UI가 노이즈로 작용 |
| **다크모드** | 비활성화 | 밝은/어두운 배경의 품질 비교 필요 |

### Step 2: 이미지 전처리

```python
from pathlib import Path
from PIL import Image, ImageEnhance


def preprocess_screenshot(
    image_path: str,
    crop_region: tuple[int, int, int, int] | None = None,
    target_width: int = 2048,
    sharpness: float = 1.0,
    contrast: float = 1.0,
) -> Image.Image:
    """원본을 보존하고 독립 이미지를 반환. 보정은 평가 후 명시적으로 선택."""
    if type(target_width) is not int or target_width < 1:
        raise ValueError("positive target_width required")
    if sharpness < 0 or contrast < 0:
        raise ValueError("nonnegative enhancement factors required")
    with Image.open(image_path) as source:
        source.load()
        if crop_region is not None:
            if len(crop_region) != 4 or any(type(v) is not int for v in crop_region):
                raise ValueError("integer crop box required")
            left, top, right, bottom = crop_region
            if not (0 <= left < right <= source.width and 0 <= top < bottom <= source.height):
                raise ValueError("crop outside image")
            image = source.crop(crop_region)
        else:
            image = source.copy()
    if image.width > target_width:
        size = (target_width, max(1, round(image.height * target_width / image.width)))
        resized = image.resize(size, Image.Resampling.LANCZOS)
        image.close()
        image = resized
    if sharpness != 1.0:
        enhanced = ImageEnhance.Sharpness(image).enhance(sharpness)
        image.close()
        image = enhanced
    if contrast != 1.0:
        enhanced = ImageEnhance.Contrast(image).enhance(contrast)
        image.close()
        image = enhanced
    return image  # 호출자가 with 또는 close로 해제


def batch_preprocess(records: list[dict], output_dir: str, crop_region=None) -> list[dict]:
    """manifest 순서·실제 페이지 번호 보존. 파일명 정렬을 페이지 순서로 간주하지 않음."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    processed = []
    for index, record in enumerate(records):
        target = output / f"image_{index:05d}.png"
        with preprocess_screenshot(record["source_image"], crop_region) as image:
            image.save(target, "PNG")
        processed.append({**record, "original_image": record["source_image"],
                          "source_image": str(target), "crop_region": crop_region})
    return processed
```

### Step 3: VLM 추출 — 문서 유형별 프롬프트

프롬프트는 추출 요구를 명시하지만 원래 “품질의 80%” 수치는 출처/평가가 없어 철회한다. 원문 전사와 생성 요약을 분리하고, 반복 헤더/개정 정보는 검토 전 제거하지 않는다.

#### 공통 시스템 프롬프트

```python
SYSTEM_PROMPT = """당신은 문서 이미지에서 텍스트와 구조를 정확하게 추출하는 전문가입니다.

추출 규칙:
1. 보이는 텍스트를 추출하되 읽을 수 없거나 잘린 부분은 추측하지 않습니다.
2. 문서의 구조(제목, 본문, 리스트, 테이블)를 Markdown 형식으로 보존합니다.
3. 표는 가능한 경우 Markdown으로 변환하고 복잡한 병합 구조와 불명확한 범위를 별도로 기록합니다.
4. 다이어그램이나 차트는 [Figure: 내용 설명] 형태로 텍스트 설명합니다.
5. 읽을 수 없는 부분은 [불명확] 으로 표시합니다.
6. 페이지 번호, 헤더/푸터, 개정 정보도 보존합니다. 제거 여부는 별도 검토합니다.
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
(생성 요약: 원문 전사와 구분하여 핵심 키워드 3-5개 제안)

중요:
- bullet point의 계층 구조(들여쓰기)를 보존하세요.
- 도형 안의 텍스트도 빠짐없이 추출하세요.
- 보이는 화살표 방향만 "A → B → C" 형태로 표현하고 불분명한 연결은 [불명확]으로 기록하세요.
"""
```

#### Excel 전용 프롬프트

```python
XLSX_PROMPT = """이 이미지는 Excel 스프레드시트입니다.

다음 규칙으로 추출하세요:

1. 화면에 보이는 테이블 범위만 Markdown 테이블 형식으로 변환하세요.
2. 헤더로 보이는 행을 표시하고 확실하지 않으면 [불명확]으로 기록하세요.
3. 병합된 셀의 표시 값과 보이는 범위를 기록하고 다른 셀에 임의로 복제하지 마세요.
4. 보이는 숫자·표시 형식·단위를 전사하고 원래 수식/정밀도/숨은 값은 추측하지 마세요.
5. 시트 이름이 보이면 ## 시트이름 형태로 시작하세요.
6. 빈 것으로 확인되는 셀은 빈 칸, 잘리거나 읽을 수 없는 셀은 [불명확]으로 표시하세요.
7. 차트의 보이는 축/라벨/수치만 전사하고 표시되지 않은 원시 데이터 포인트를 만들지 마세요.

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
6. 각주 표식과 실제로 보이는 각주 본문을 함께 기록하세요. 본문이 안 보이면 [불명확]으로 표시하세요.
7. 이미지/그림은 [Figure: 설명] 으로 표시하세요.
"""
```

### Step 4: VLM API 호출 구현

사내 API가 OpenAI-compatible이라는 것은 당시 가정이다. vLLM 공식 서버는 해당 인터페이스를 제공하지만 모든 모델·배포·파라미터 지원이 동일하지 않다. 확인된 내부 endpoint/키로 생성한 `AsyncOpenAI`를 호출자에게서 받는다. 키는 문서에 저장하지 않는다. 이미지 `detail` 지원과 `temperature=0` 재현성을 가정하지 않는다. 아래는 OpenAI Python3.24.0 모의 요청으로 검사했다.

```python
import base64
import io
from PIL import Image
from openai import AsyncOpenAI


async def extract_single_image(
    record: dict,
    client: AsyncOpenAI,
    model: str,
    doc_type: str = "general",
) -> dict:
    """확인된 내부 client/model을 주입. 이미지를 실제 PNG로 재인코딩."""
    prompts = {"pptx": PPTX_PROMPT, "xlsx": XLSX_PROMPT, "docx": DOCX_PROMPT,
               "general": "보이는 텍스트와 구조만 Markdown으로 추출하세요."}
    if doc_type not in prompts or not model:
        raise ValueError("supported doc_type and verified model required")
    page = record.get("page_number")
    if page is not None and (type(page) is not int or page < 1):
        raise ValueError("page_number must be positive or None (unknown)")
    with Image.open(record["source_image"]) as image, io.BytesIO() as output:
        with image.convert("RGB") as converted:
            converted.save(output, format="PNG")
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": prompts[doc_type]},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
            ]},
        ],
        max_tokens=4096,  # 배포 서버 지원/한도 확인 필요
        temperature=0.0,  # 재현성 보장 아님
    )
    if not response.choices:
        raise ValueError("no completion choice")
    choice = response.choices[0]
    if choice.finish_reason != "stop":
        raise ValueError(f"incomplete/unsupported completion: {choice.finish_reason}")
    content = choice.message.content
    if not isinstance(content, str) or not content.strip():
        raise ValueError("missing extraction text")
    return {**record, "extracted_text": content, "model": model,
            "tokens_used": response.usage.total_tokens if response.usage else None,
            "review_status": "unverified"}
```

### Step 5: 대량 처리 — 비동기 배치

```python
import asyncio
from openai import AsyncOpenAI


async def extract_document_batch(
    records: list[dict], client: AsyncOpenAI, model: str,
    doc_type: str = "general", max_concurrent: int = 5,
) -> list[dict]:
    """명시 manifest 순서 보존. 실패 시 부분 결과를 전체 성공으로 반환하지 않음."""
    if type(max_concurrent) is not int or max_concurrent < 1:
        raise ValueError("positive concurrency required")
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited(record: dict) -> dict:
        async with semaphore:
            return await extract_single_image(record, client, model, doc_type)

    tasks = [asyncio.create_task(limited(record)) for record in records]
    try:
        return await asyncio.gather(*tasks)  # 완료 순서와 관계없이 입력 순서
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def extract_full_document(
    records: list[dict], client: AsyncOpenAI, model: str, doc_type: str = "general",
) -> dict:
    results = await extract_document_batch(records, client, model, doc_type)
    text = "\n\n---\n\n".join(
        f"<!-- source page: {r.get('page_number') if r.get('page_number') is not None else 'unknown'} -->\n"
        + r["extracted_text"] for r in results
    )
    known = [r["tokens_used"] for r in results if r["tokens_used"] is not None]
    return {"full_text": text, "page_results": results, "total_images": len(results),
            "known_tokens_sum": sum(known),
            "total_tokens": sum(known) if len(known) == len(results) else None,
            "review_status": "unverified"}
```

### Step 6: 추출 결과 검증

VLM 결과의 정확도는 원본 대조 없이 알 수 없다. 원래 길이20/불명확3/반복3 규칙은 검토 신호로 보존하고 임의 `100 - 문제수×20` 점수/60점 재시도는 제거했다. 서로 다른 표를 합쳐 파이프 개수를 세는 검사는 오탐이 있어 제거했다. 신호가 없더라도 `unverified`이며 Markdown 표 렌더링·셀 내용 대조는 별도다.

```python
def validate_extraction(result: dict) -> dict:
    """휴리스틱 검토 신호. 원문 대조 없는 정확도/신뢰도 점수가 아님."""
    text = result["extracted_text"]
    if not isinstance(text, str):
        raise ValueError("text required")
    issues = []
    if len(text.strip()) < 20:
        issues.append("short_text: 빈 페이지/짧은 원문/추출 누락을 구별하려면 원본 확인")
    if text.count("[불명확]") > 3:
        issues.append("unclear_markers: 원본과 전처리 확인")
    lines = text.splitlines()
    if any(a == b == c and a.strip() for a, b, c in zip(lines, lines[1:], lines[2:])):
        issues.append("repeated_lines: 실제 반복인지 생성 오류인지 원본 확인")
    # 서로 다른 표/escaped pipe/코드 블록을 raw pipe 수로 검증하지 않는다.
    # Markdown 렌더링 검사와 원본 표의 셀·병합·숫자 대조를 별도로 수행한다.
    return {**result, "quality_issues": issues, "review_status": "unverified"}
```

## 처리 속도 최적화 전략

### 티어별 모델 사용

모델별 처리 시간/GPU 부하는 배포·이미지 크기·출력·동시성에 따라 달라진다. **원래 티어 분리는 평가 후보**다. 모델명만으로 정확도 우위를 정하지 않고, 명시 정책이 `None`을 반환하면 재시도 결정을 미확인으로 보존한다. 두 번째 추출도 검토하고 첫 결과를 지우지 않는다.

```python
from collections.abc import Callable
from openai import AsyncOpenAI


async def smart_extract(
    record: dict, doc_type: str, client: AsyncOpenAI,
    first_model: str, second_model: str,
    should_retry: Callable[[dict], bool | None],
) -> dict:
    """모델/재시도 정책은 평가·합의 후 주입. 두 시도의 검토 결과를 모두 보존."""
    first = validate_extraction(await extract_single_image(record, client, first_model, doc_type))
    attempts = [first]
    decision = should_retry(first)
    if decision is not None and type(decision) is not bool:
        raise ValueError("retry policy must return True, False, or None (unknown)")
    if decision is True:
        second = validate_extraction(await extract_single_image(record, client, second_model, doc_type))
        attempts.append(second)
    return {"attempts": attempts, "selected_result": attempts[-1],
            "retry_decision": decision, "review_status": "unverified"}
```

### 처리 시간 추정

2026-02-12 당시 100페이지 가정의 미검증 추정치. 하드웨어/모델 revision/이미지·출력 길이/측정 방법이 없어 운영 예상으로 사용할 수 없다:

| 전략 | 모델 | 추정 처리 시간 | GPU 부하 |
|------|------|----------------|----------|
| 전부 8B | Qwen3-VL-8B-Instruct | ~3-5분 | 낮음 |
| 전부 30B | Qwen3-VL-30B | ~10-15분 | 높음 |
| 티어별 분리 | 8B + 30B 폴백 | ~5-8분 | 중간 |

> **미확인**: 당시 “사내 API 별도 과금 없음” 가정은 현재 계약/내부 배부 비용을 확인하지 않았다. `usage=None`은 0토큰이 아니며 호출/전처리/재시도/인프라 비용은 별도로 평가한다.

## 참고 자료 (References)

확인일 **2026-10-04**. 아래 공개 자료는 모델/API 동작 설명의 근거이며 사내 배포·권한·품질의 근거는 아니다.

- [Qwen3-VL-8B 공식 카드](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), [30B-A3B 공식 카드](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct) — 공개 모델명/시각 입력. 기존 Qwen2.5 저장소를 Qwen3 근거로 연결한 오류 수정.
- [Kimi-K2.5 공식 카드](https://huggingface.co/moonshotai/Kimi-K2.5) — 네이티브 멀티모달. 기존 Kimi-K2 저장소/텍스트 전용 분류 수정.
- [vLLM OpenAI 호환 서버](https://docs.vllm.ai/en/latest/serving/online_serving/) — 실제 서버 버전/모델 지원 확인 필요.
- [OpenAI Python 공식 SDK](https://github.com/openai/openai-python) — async client/응답 형식. 로컬 검증 버전3.24.0.
- [Pillow Image](https://pillow.readthedocs.io/en/stable/reference/Image.html), [ImageEnhance](https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html), [ImageGrab](https://pillow.readthedocs.io/en/stable/reference/ImageGrab.html) — 로컬/공식 문서12.3.0; Windows `window` 인자11.2.1 도입.
- [PyAutoGUI screenshot](https://pyautogui.readthedocs.io/en/latest/screenshot.html) — region/PIL image; 실제 화면 취득 미실행.
- [Microsoft DeleteObject](https://learn.microsoft.com/en-us/windows/win32/api/wingdi/nf-wingdi-deleteobject) — 선택 중인 GDI 객체 해제 제약; 기존 구현을 직접 실행한 근거는 아님.

## 관련 문서

- [VLM 추출 결과 청킹 전략](./vlm-chunking-strategy.md)
- [DRM 해제 후 하이브리드 전략](./post-drm-hybrid.md)
- [청킹 방법론 총론](../overview-chunking-methods.md)
