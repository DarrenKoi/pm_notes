---
reviewed_on: 2026-10-04
review_status: partial
document_type: learning
tags: [ocr, document-extraction, vlm]
category_major: "개발 환경"
category_middle: "비전 모델 실행"
category_minor: "문서 OCR·추출"
note_kind: "목차"
classified_on: "2026-10-05"
---

# PPT/PDF 이미지 OCR 읽기 순서

UI 버튼의 위치를 찾는 grounding과 페이지의 문자·표·읽기 순서를 복원하는 문서 OCR은 목표가 다르다. 여기서는 후자를 다룬다.

1. [OCR-first 설계](./vlm-for-ppt-pdf-extraction.md): 페이지 파서, 영역 재인식, 사내 API 후처리의 역할과 JSON 골격.
2. [1.5 모델 설치 예제](./install.md): 승인된 반입 머신, 오프라인 패키지, 공유 GPU 환경의 후보 구성.
3. [상위 VLM 목차](../README.md): 일반 서빙과 이미지 HTTP 호출로 돌아간다.

설계 문서는 후보 비교와 인터페이스를, 설치 문서는 특정 버전·환경의 실행 예제를 다룬다. 기존 추천과 미실측 수치는 현재 성능 보장이 아니다. PaddleOCR-VL 1.5 모델 카드는 확인일에 1.6 후속 모델을 안내하지만 설치 예제를 검증 없이 1.6으로 바꾸지 않았다.

## 확인과 남은 미확인

확인일 **2026-10-04**. 개발자 모델 카드에서 1.5의 0.9B 및 `pipeline_version="v1.5"` 경로와 CUDA 12.6용 PaddlePaddle 3.2.1·PaddleOCR 3.4.0 이상 예제를 확인했다. GOT의 crop/patch 입력 사용 경로도 모델 카드와 대조했다. 사내 no-Docker 환경, 패키지 전체 lockfile, H200 공유 서비스와 OCR 정확도는 미검증이다.

confidence를 제공하지 않는 모델은 `unknown`을 유지한다. 임의 수치로 실패 영역을 결정하지 않는다. JSON object 응답은 문법 형식이지 스키마·원문 정확성 보장이 아니다. 사용자가 권한을 가진 이미지와 문서만 처리한다.

- [PaddleOCR-VL 1.5 개발자 카드](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5)
- [GOT-OCR HF 개발자 카드](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf)
- [Hub CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli): `hf download`와 `--revision`.
