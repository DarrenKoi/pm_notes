---
tags: [vlm, ocr, ppt, pdf, document-extraction, structured-output, company-api]
level: intermediate
last_updated: 2026-03-11
status: in-progress
---

# PPT/PDF 슬라이드 이미지 -> 구조화 텍스트 추출을 위한 OCR-first 파이프라인

> 전제: 대형 모델은 사내 API로만 사용하고, 로컬 GPU에는 OCR/문서 특화 모델과 필요 시 중형 모델만 둔다. 목표는 DRM 때문에 이미지로 캡처한 슬라이드에서 텍스트, 표, 레이아웃, 다이어그램 단서를 최대한 손실 없이 뽑아낸 뒤 사내 API 모델이 최종 JSON을 정리하게 하는 것이다.

## 왜 이 구성이 맞는가?

- **이미지 기반 추출에서는 OCR이 1차 병목**이다. 대형 VLM이 강한 것은 의미 해석과 정리이지, 작은 글씨/표 셀/읽기 순서 복원 자체가 아니다.
- **슬라이드는 일반 문서보다 난도가 높다**. 텍스트 박스, 표, 차트, 도형, 캡션, 화살표, 작은 주석이 섞여 있어서 단일 범용 VLM에만 맡기면 누락이 생기기 쉽다.
- **사내 API의 대형 모델은 비싼 후처리기로 쓰는 편이 낫다**. 먼저 OCR 특화 모델이 구조와 원문을 뽑고, 그 다음 큰 모델이 JSON 정리와 의미 보강을 맡게 하는 편이 안정적이다.
- **이 문서는 Qwen2.5-VL-72B, InternVL3-78B 로컬 배포를 추천하지 않는다**. 당신의 운영 방침에 맞춰 제외한다.

## 핵심 원칙

1. **1차 추출은 OCR/문서 파서가 맡는다.**
2. **애매한 영역만 crop OCR로 다시 읽는다.**
3. **사내 API 대형 모델은 최종 정리와 의미 해석만 맡는다.**
4. **모든 단계에서 bbox, reading order, confidence, page id를 보존한다.**

## 권장 아키텍처

```text
slide image / pdf page
  -> render + deskew + split
  -> primary OCR parser
       - PaddleOCR-VL-1.5 or MinerU2.5 or dots.ocr
  -> hotspot recrop OCR
       - GOT-OCR-2.0-hf or dots.ocr grounding
  -> optional local normalizer
       - granite-docling-258M or olmOCR-2
  -> company API large model
       - JSON normalization
       - diagram/chart interpretation
       - error correction with image + OCR evidence
  -> schema validation + quality checks
```

## 추천 모델

### 1. 로컬 OCR/문서 특화 모델

| 모델 | 권장 역할 | 강점 | 주의점 | Hugging Face |
|------|-----------|------|--------|--------------|
| **PaddleOCR-VL-1.5** | 기본 1차 파서 | 0.9B 급 초경량인데도 문서 파싱 성능이 매우 높고, 텍스트 spotting, 표, 차트, 수식, seal까지 폭넓게 다룸 | PaddleOCR 생태계 기준 사용이 가장 편함. 범용 OpenAI API 서버처럼 다루기보다는 공식 파이프라인에 맞추는 편이 낫다 | https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5 |
| **MinerU2.5-2509-1.2B** | 고해상도 페이지 파서 | 1.2B 2단계 구조라서 전체 레이아웃과 세밀한 crop 인식을 분리해 dense PDF, 작은 글씨, 수식, 표에 강함 | 문서 페이지 중심 모델이다. 슬라이드 이미지에는 강하지만 일반 사진 OCR 용도로는 과하다 | https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B |
| **dots.ocr** | 다국어 레이아웃 파서 | 1.7B 단일 VLM으로 layout + content + reading order를 함께 뽑기 좋고, multilingual 슬라이드에 강함 | 모델 카드 기준 pictures 파싱과 매우 복잡한 표/수식은 아직 한계가 있다 | https://huggingface.co/rednote-hilab/dots.ocr |
| **olmOCR-2-7B-1025** | 영어 중심 PDF/문서 OCR 보강 | 8B 급 OCR 특화 모델로 표, 수식, 난이도 높은 OCR 보강에 강함. vLLM 기반 대량 처리 툴킷도 갖춤 | **중요**: 이 모델은 Qwen2.5-VL-7B 기반 파인튜닝이다. "Qwen 계열 자체를 완전히 배제"하려면 제외해야 한다 | https://huggingface.co/allenai/olmOCR-2-7B-1025 |
| **GOT-OCR-2.0-hf** | hotspot crop OCR | 작은 영역 재인식, 특이 aspect ratio, patch OCR, 특정 bbox OCR에 적합 | HF 구현은 plain text 중심이라 whole-page structure parser로 쓰기보다 crop 재인식용이 더 적합하다 | https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf |
| **granite-docling-258M** | 초경량 구조화 보조 | 매우 작고 Docling 통합이 좋아서 경량 structured conversion, bbox-guided region inference에 유리 | 범용 이미지 해석용 모델이 아니라 문서 변환 파이프라인의 한 부품으로 보는 편이 맞다 | https://huggingface.co/ibm-granite/granite-docling-258M |

### 2. 사내 API 쪽에서 맡길 모델

| 모델 계층 | 맡길 일 |
|-----------|---------|
| **중형 멀티모달 모델** | OCR 결과를 바탕으로 JSON 정규화, 누락 점검, 간단한 다이어그램 설명. 사내 API에 `Qwen3-VL-8B-Instruct` 급 모델이 있으면 여기에 두고, 어려운 페이지만 큰 모델로 올린다 |
| **대형 멀티모달 모델** | 애매한 표 구조 해석, 차트/흐름도 의미 해석, OCR 충돌 조정, 최종 문서 품질 보강 |
| **텍스트 LLM** | 슬라이드 간 메타데이터 정리, 제목 정제, 섹션 분류, 요약 |

## 내 기준 추천 순위

### 가장 추천하는 조합

1. **PaddleOCR-VL-1.5 + GOT-OCR-2.0-hf + 사내 API 대형 모델**
2. **MinerU2.5 + GOT-OCR-2.0-hf + 사내 API 대형 모델**
3. **dots.ocr + GOT-OCR-2.0-hf + 사내 API 대형 모델**

### 조건별 선택

- **가장 무난한 기본값**: `PaddleOCR-VL-1.5`
- **dense PDF / 작은 글씨 / 수식 / 표 비중 높음**: `MinerU2.5`
- **다국어 + reading order 중요**: `dots.ocr`
- **영문 인쇄 PDF 대량 처리**: `olmOCR-2-7B-1025`
- **특정 영역 재확인용**: `GOT-OCR-2.0-hf`
- **매우 작은 로컬 정규화 모델이 필요함**: `granite-docling-258M`

## Qwen2.5/InternVL3를 빼고도 강한 이유

- 로컬에서 가장 중요한 일은 **문자 인식, 구조 복원, reading order 보존**이다.
- 이 부분은 범용 초대형 VLM보다 **문서 특화 모델**이 더 잘하는 경우가 많다.
- 큰 모델은 OCR의 1차 엔진이 아니라 **해석기**로 쓰는 편이 비용과 품질 면에서 더 낫다.
- 따라서 이 문서의 추천 파이프라인은:

```text
문자와 구조는 작은/중형 특화 모델이 뽑고
의미 해석과 최종 JSON 정리는 사내 API의 큰 모델이 맡는다
```

## 강한 파이프라인 설계안

### A. 기본 파이프라인

```text
render page image
  -> PaddleOCR-VL-1.5
  -> low-confidence region detect
  -> GOT-OCR-2.0-hf recrop
  -> company API large model
  -> final JSON
```

이 구성이 가장 균형이 좋다.

- 전체 구조와 원문 추출은 `PaddleOCR-VL-1.5`
- 작은 글씨, 주석, 깨진 영역은 `GOT-OCR-2.0-hf`
- 표/도형 의미 해석과 JSON 정리는 사내 API

### B. 문서 품질을 더 올리는 파이프라인

```text
render page image
  -> MinerU2.5 or dots.ocr
  -> hotspot crop OCR with GOT-OCR-2.0-hf
  -> optional local normalizer (granite-docling-258M or olmOCR-2)
  -> company API large model
  -> schema validator
```

이 구성이 더 무겁지만, 아래 상황에서 좋다.

- 표가 많다
- 주석이 작다
- 슬라이드마다 레이아웃이 자주 바뀐다
- 한/영 혼합, 세로 읽기, 박스형 배치가 많다

### C. Qwen 계열을 완전히 피하고 싶을 때

```text
PaddleOCR-VL-1.5
  + MinerU2.5
  + GOT-OCR-2.0-hf
  + granite-docling-258M
  + company API large model
```

`olmOCR-2`는 Qwen2.5-VL-7B 기반이므로 이 조합에서는 제외한다.

## 단계별 구현 포인트

### 1. 렌더링/전처리

- PDF는 페이지당 PNG로 렌더링하고 원본 해상도를 보존한다
- PPT 스크린샷은 가능하면 여백 포함 전체 캡처를 유지한다
- deskew, orientation correction, contrast normalization은 OCR 전에 수행한다
- 페이지 id, slide number, source path를 처음부터 메타데이터로 붙인다

### 2. 1차 OCR 결과 저장

반드시 아래 정보를 함께 저장한다.

```json
{
  "page_id": 1,
  "blocks": [
    {
      "bbox": [0, 0, 100, 100],
      "type": "title|text|table|figure|formula|chart",
      "text": "원문",
      "confidence": 0.97,
      "source_model": "paddleocr-vl-1.5"
    }
  ]
}
```

### 3. hotspot 재인식

재인식 후보는 아래 조건으로 뽑는다.

- confidence가 낮다
- 텍스트 길이에 비해 bbox가 비정상적으로 작다
- 표 셀 병합이 의심된다
- OCR 결과에 `[UNK]`, 반복 문자, 깨진 숫자가 많다
- 차트 레이블이나 각주처럼 작은 글씨가 있다

이 단계는 `GOT-OCR-2.0-hf` 또는 `dots.ocr` grounding prompt가 잘 맞는다.

### 4. 사내 API 대형 모델에 넘길 입력

큰 모델에는 **이미지 하나만** 넘기지 말고 아래 세 가지를 함께 넘긴다.

1. 원본 이미지
2. 1차 OCR block JSON
3. hotspot 재인식 결과

이렇게 해야 큰 모델이 "이미지 직접 보기"와 "OCR 증거"를 함께 사용해 더 안정적으로 정리한다.

## 최종 JSON 스키마 예시

```json
{
  "slide_number": 1,
  "title": "슬라이드 제목",
  "sections": [
    {
      "heading": "섹션 제목",
      "items": ["불릿1", "불릿2"]
    }
  ],
  "tables": [
    {
      "caption": "표 설명",
      "headers": ["열1", "열2"],
      "rows": [["값1", "값2"]]
    }
  ],
  "figures": [
    {
      "type": "chart|diagram|image",
      "description": "도식 설명",
      "evidence_text": ["라벨A", "라벨B"]
    }
  ],
  "ocr_warnings": [
    "우측 하단 6pt 텍스트 재검토 필요"
  ]
}
```

## 오케스트레이션 예시

아래 코드는 특정 OCR 라이브러리에 종속되지 않는 파이프라인 골격이다.

```python
import json
from openai import OpenAI


company_client = OpenAI(
    base_url="https://your-company-api.example/v1",
    api_key="YOUR_COMPANY_API_KEY",
)


def primary_ocr(image_path: str) -> dict:
    # PaddleOCR-VL-1.5 / MinerU2.5 / dots.ocr 중 하나로 구현
    raise NotImplementedError


def rerun_hotspots(image_path: str, ocr_result: dict) -> list[dict]:
    # GOT-OCR-2.0-hf 또는 crop OCR 재시도 로직
    return []


def build_prompt(ocr_result: dict, hotspot_result: list[dict]) -> str:
    return f"""
You are a document extraction system.

Use the OCR evidence below as the primary source of truth.
Use the image only to fix OCR mistakes, recover reading order,
and interpret diagrams, charts, and ambiguous layout.

OCR blocks:
{json.dumps(ocr_result, ensure_ascii=False)}

Hotspot recrop OCR:
{json.dumps(hotspot_result, ensure_ascii=False)}

Return valid JSON only.
"""


def finalize_with_company_api(image_b64: str, ocr_result: dict, hotspot_result: list[dict]) -> dict:
    response = company_client.chat.completions.create(
        model="company-large-vlm",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_prompt(ocr_result, hotspot_result)},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ],
            }
        ],
        temperature=0.0,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)
```

핵심은 단순하다.

- OCR 모델은 **원문과 구조를 뽑는다**
- 큰 모델은 **정리와 해석만 한다**

## 실전 운영 팁

- **한 모델로 끝내려 하지 말 것**: whole-page OCR과 crop OCR은 역할이 다르다
- **표/차트는 별도 재검토 규칙을 둘 것**: 전체 OCR 결과만으로는 셀 병합과 축 라벨이 자주 깨진다
- **이미지와 OCR 둘 다 보존할 것**: 나중에 품질 이슈를 추적하기 쉽다
- **confidence 기반 재시도 루프를 넣을 것**: 모든 페이지에 큰 모델을 쓰기보다, 문제 페이지에만 escalate 한다
- **벤치마크 점수보다 샘플셋 비교가 중요**: 사내 슬라이드 캡처 품질, 폰트, DRM 워터마크가 성능을 크게 바꾼다

## 빠른 결론

- 가장 먼저 볼 모델: **PaddleOCR-VL-1.5**
- dense 문서까지 포함하면: **MinerU2.5**
- 다국어 레이아웃이 중요하면: **dots.ocr**
- 작은 영역 재인식은: **GOT-OCR-2.0-hf**
- 영문 인쇄 PDF OCR 특화 보강은: **olmOCR-2-7B-1025**
- 초경량 구조화 보조는: **granite-docling-258M**

당신의 조건에서는 **"OCR specialist + crop OCR + company API large model"**이 가장 강한 설계다.

## 참고 링크

### Hugging Face 모델 카드

- [PaddleOCR-VL-1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5)
- [MinerU2.5-2509-1.2B](https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B)
- [dots.ocr](https://huggingface.co/rednote-hilab/dots.ocr)
- [olmOCR-2-7B-1025](https://huggingface.co/allenai/olmOCR-2-7B-1025)
- [olmOCR-2-7B-1025-FP8](https://huggingface.co/allenai/olmOCR-2-7B-1025-FP8)
- [GOT-OCR-2.0-hf](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf)
- [granite-docling-258M](https://huggingface.co/ibm-granite/granite-docling-258M)

### 참고 문서

- [vLLM: PaddleOCR-VL Usage Guide](https://docs.vllm.ai/en/latest/models/supported_models.html)
- [PaddleOCR 공식 문서](https://www.paddleocr.ai/)
- [MinerU GitHub](https://github.com/opendatalab/MinerU)
- [olmOCR GitHub](https://github.com/allenai/olmocr)
- [Docling 문서](https://docling-project.github.io/docling/)

## 관련 문서

- [스크린샷 + VLM 기반 추출 파이프라인](../../../ai-dt/rag/token_strategy/when_drm/screenshot-vlm-pipeline.md)
- [VLM Cloud Notes (README)](../README.md)
- [Private Cloud에서 vLLM 시작](../private-cloud-vllm-next-steps.md)
- [로컬 PC에서 이미지 전송](../local-pc-vllm-image-guide.md)
