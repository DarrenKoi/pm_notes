---
tags: [vlm, ui-automation, gui-agent, offline-deployment]
level: intermediate
last_updated: 2026-03-04
status: in-progress
---

# UI 특화 VLM 설치 및 운영 가이드

> 폐쇄망 환경에서 UI 특화 VLM을 다운로드, 전송, 배포하여 GUI 자동화에 활용하는 방법

## 왜 필요한가? (Why)

- 복잡한 엔지니어링 도구(EDA, CAD 등)의 GUI 자동화를 위해 UI 특화 VLM이 필요
- 일반 VLM(Qwen2.5-VL, InternVL 등)은 전문 소프트웨어 UI 인식률이 **2% 미만** (ScreenSpot-Pro 기준)
- 회사 환경은 **폐쇄망**이므로 외부에서 모델을 다운로드 후 내부로 전송해야 함
- H200 x2 (280GB VRAM) 보유 → 모든 크기의 모델 실행 가능

## 모델 비교 요약

> ScreenSpot-Pro: 전문 소프트웨어(Vivado, AutoCAD, MATLAB, VS Code 등) UI에서의 그라운딩 정확도 벤치마크

| 모델 | 파라미터 | ScreenSpot-Pro | VRAM (fp16) | 용도 |
|------|----------|----------------|-------------|------|
| UI-Venus-1.5-30B-A3B | 30B (3B active) | 69.6% | ~24-32GB | 최고 정확도 (MoE) |
| UI-Venus-1.5-8B | 9B | 68.4% | ~18GB | 단일 GPU 최강 |
| MAI-UI-8B | 8B | 65.8% | ~16GB | 강력한 대안 |
| MAI-UI-2B | 2B | 57.4% | ~4GB | 가장 작은 고성능 |
| GUI-Actor-7B | 7B | 44.6% | ~14GB | 좌표 없는 그라운딩 |
| OmniParser V2 + LLM | YOLO+Florence-2 | 39.6% | ~4-6GB | 파싱 파이프라인 |
| UI-TARS-72B-DPO | 72B | 38.1% | ~144GB | 대형 에이전트 |
| UI-TARS-1.5-7B | 7B | ~35% | ~14GB | 균형잡힌 에이전트 |
| UGround-V1-7B | 7B | 31.1% | ~14GB | 범용 그라운딩 |
| SeeClick | 9.6B | 초기 기준 | ~16GB | 선구적 모델 |
| ShowUI-2B | 2B | 표준 해상도 강점 | ~8GB | 경량 에이전트 |
| ZonUI-3B | 3B | 표준 해상도 강점 | ~6GB | 초효율 학습 모델 |

> 모든 모델이 HuggingFace 로그인 없이 다운로드 가능

## 하위 문서

| 주제 | 설명 | 상태 |
|------|------|------|
| [UI VLM 모델 카탈로그](./ui-vlm-models.md) | 모든 UI 특화 VLM 상세 정보 및 다운로드 명령어 | 🟡 진행 중 |
| [HF private cloud 다운로드 shortlist](./huggingface-private-cloud-downloads.md) | H200 x2 기준으로 먼저 받을 Hugging Face repo와 다운로드 링크 | 🟡 진행 중 |
| [오프라인 다운로드 가이드](./offline-download-guide.md) | HuggingFace 다운로드 → 폐쇄망 전송 워크플로우 | 🟡 진행 중 |
| [모델 서빙 가이드](./serving-guide.md) | vLLM/SGLang으로 모델 서빙 및 API 구성 | 🟡 진행 중 |

## 학습 순서

```
1. UI VLM 모델 카탈로그 확인 → 어떤 모델들이 있는지 파악
   ↓
2. 오프라인 다운로드 가이드 → 외부에서 모델 다운로드 & 전송
   ↓
3. 모델 서빙 가이드 → 클라우드에서 모델 실행 & API 구성
   ↓
4. 실제 엔지니어링 도구 UI로 벤치마크 → 최적 모델 선정
```

## 관련 문서

- [위로: 개발 환경](../README.md)
- [루트 README](../../README.md)
