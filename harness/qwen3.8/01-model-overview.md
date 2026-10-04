# Qwen3.8-27B 모델 개요

> 2026-08-14 공개 (Apache 2.0). Qwen 오픈 모델 계열 중 "가장 유능한 세대(Qwen3.8)"의 소형 dense 멤버.

## 기본 스펙

| 항목 | 값 |
|---|---|
| 파라미터 | 27B (dense, vision encoder 포함 시 ~28B) |
| 아키텍처 | Qwen3.5 기반. 64 레이어 하이브리드: 16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN)) |
| 컨텍스트 | **262,144 토큰 네이티브**, YaRN으로 최대 1M 확장 |
| 권장 출력 할당 | reasoning 최대 262K + 최종 응답 최대 131K |
| 모달리티 | 텍스트 + 이미지 + 비디오 (네이티브 멀티모달) |
| 라이선스 | Apache 2.0 (상업적 셀프호스팅 제약 없음) |
| 양자화 체크포인트 | FP8 공식 공개 (block size 128 fine-grained, 원본과 성능 거의 동일) |
| 서빙 호환 | HF Transformers, vLLM, SGLang, TokenSpeed (각각 공식 Qwen3.8 레시피 있음) |
| MTP | Multi-Token Prediction으로 학습됨 (speculative decoding에 유리) |

## 특징

- **Gated DeltaNet(선형 어텐션) + Gated Attention 혼합**: 긴 컨텍스트에서 메모리 효율이 좋고, KV 캐시 부담이 순수 어텐션 대비 낮다.
- **thinking 모드 기본 ON**: `enable_thinking=True`가 기본. `reasoning_effort`(xhigh/medium/low)로 추론 깊이 조절.
- **preserve_thinking**: 이전 턴의 thinking 내용을 컨텍스트에 유지하는 옵션(기본 ON). 결정 일관성에 도움이 되지만 컨텍스트/KV 비용 증가.
- **유연한 thinking 제어**: 요청 단위로 on/off 가능.

## 벤치마크 (Qwen 발표 기준, 27B 기준 동급 비교)

| 벤치마크 | Qwen3.8-27B | Qwen3.6-27B | Qwen3.7-Plus(호스팅) |
|---|---|---|---|
| Terminal Bench 2.1 | **73.0** | 63.4 | 64.0 |
| SWE-bench Pro | **61.7** | 53.5 | 57.6 |
| QwenSWEBench | **79.0** | 49.3 | 59.2 |
| DeepSWE 1.1 | **42.2** | 13.3 | 14.2 |
| LiveCodeBench v6 | **90.3** | 83.9 | 89.6 |
| GPQA Diamond | 89.2 | 87.8 | 90.3 |
| OSWorld-Verified (computer use) | **84.3** | 63.9 | 73.3 |
| AndroidWorld (mobile use) | **81.9** | 70.3 | 81.0 |
| OmniDocBench 1.5 | 91.1 | 89.4 | 91.4 |
| IFBench | **79.5** | 69.1 | 79.1 |

**해석**: 코딩·에이전트·오피스 자동화에 최적화된 모델. Qwen은 코딩 벤치마크를 Claude Code 하네스로 평가했음 — 즉, 에이전트형(툴 루프) 사용을 전제로 설계됨.

## 적합한 워크로드

- 셀프호스팅 코딩/터미널 에이전트 (하드웨어 예산 내 최대 dense 모델)
- 컴퓨터/브라우저/모바일 사용 에이전트 (비전이 같은 모델에 통합됨)
- 문서·차트 이해 (OmniDocBench 91.1, CharXiv RQ 83.7)
- 저장소 규모 코드, 시간 단위 비디오 등 262K~1M 컨텍스트 작업

## 출처

- https://github.com/AlibabaCloud-Official/Qwen3.8-27B
- https://huggingface.co/unsloth/Qwen3.8-27B
- https://huggingface.co/Qwen/Qwen3.8-27B-FP8
- https://www.qwencloud.com/models/qwen3.8-27b
- https://ai-tldr.dev/models/qwen3-8-27b/
- https://aireleasetracker.com/model/qwen/qwen3.8-27b
