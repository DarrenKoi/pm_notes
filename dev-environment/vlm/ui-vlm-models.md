---
tags: [vlm, ui-grounding, gui-agent, screenspot-pro]
level: intermediate
last_updated: 2026-03-04
status: in-progress
---

# UI 특화 VLM 모델 카탈로그

> GUI 자동화를 위한 UI 특화 VLM 전체 목록. 이 문서는 레퍼런스 문서이며, 실제 읽는 순서는 [VLM 가이드 인덱스](./README.md)를 따른다.

## 이 문서의 역할

- 어떤 모델들이 있는지 전체 지형도를 보는 문서다.
- 바로 다운로드할 대상을 고를 때는 [Hugging Face 다운로드 shortlist](./huggingface-private-cloud-downloads.md)를 먼저 본다.
- 실제 다운로드/전송 절차는 [오프라인 다운로드 & 폐쇄망 전송 가이드](./offline-download-guide.md)를 따른다.

## 왜 UI 특화 모델이 필요한가? (Why)

- 일반 VLM(Qwen2.5-VL, GPT-4o 등)은 **ScreenSpot-Pro에서 0.8~2% 수준** — 전문 소프트웨어 UI를 거의 인식하지 못함
- 엔지니어링 도구(EDA, CAD 등)는 4K 해상도에 작은 아이콘, 텍스트 없는 버튼, 복잡한 메뉴 구조가 특징
- UI 특화 모델은 이런 복잡한 UI에서의 **요소 위치 파악(그라운딩)**, **액션 예측**, **다단계 조작**에 최적화됨

## 핵심 벤치마크: ScreenSpot-Pro

ScreenSpot-Pro는 전문 소프트웨어 UI에서의 그라운딩 정확도를 측정하는 벤치마크:

- **개발 도구**: VS Code, PyCharm, Android Studio, VMware
- **크리에이티브**: Photoshop, Premiere, Illustrator, Blender, DaVinci Resolve
- **CAD/EDA**: AutoCAD, SolidWorks, Inventor, **Vivado**, **Quartus**
- **과학/공학**: MATLAB, Stata, EViews
- **오피스**: Word, Excel, PowerPoint

> 타겟 영역이 화면의 평균 0.07%만 차지 — 정밀한 UI 인식 능력이 핵심

---

## Tier 1: 최고 성능 (ScreenSpot-Pro 60%+)

### UI-Venus-1.5-8B

| 항목 | 내용 |
|------|------|
| **개발사** | inclusionAI (Ant Group) |
| **HuggingFace** | `inclusionAI/UI-Venus-1.5-8B` |
| **파라미터** | 9B |
| **VRAM** | ~18GB (fp16) |
| **ScreenSpot-Pro** | 68.4% (ZoomIn 적용 시) |
| **베이스 모델** | Qwen3-VL 계열 |
| **라이선스** | Apache 2.0 |
| **HF 로그인** | 불필요 |

**특징**:
- 현재 단일 GPU에서 실행 가능한 모델 중 **최고 성능**
- 모바일, 데스크톱, 웹 크로스 플랫폼 지원
- 4단계 학습: Mid-training → Offline-RL → Online-RL → Model Merge
- ZoomIn 전략으로 작은 UI 요소 인식률 대폭 향상

**같은 계열 모델**:

| 변형 | HuggingFace 경로 | 비고 |
|------|-------------------|------|
| 2B | `inclusionAI/UI-Venus-1.5-2B` | 경량 버전 |
| 8B | `inclusionAI/UI-Venus-1.5-8B` | 단일 GPU 최적 |
| 30B-A3B (MoE) | `inclusionAI/UI-Venus-1.5-30B-A3B` | 최고 성능, MoE (활성 3B) |

**다운로드** (외부 PC에서):
```bash
# 전체 모델 다운로드
huggingface-cli download inclusionAI/UI-Venus-1.5-8B --local-dir ./UI-Venus-1.5-8B

# 30B MoE 버전 (활성 파라미터 3B, 메모리 효율적)
huggingface-cli download inclusionAI/UI-Venus-1.5-30B-A3B --local-dir ./UI-Venus-1.5-30B-A3B
```

**GitHub**: https://github.com/inclusionAI/UI-Venus

---

### MAI-UI-8B

| 항목 | 내용 |
|------|------|
| **개발사** | Tongyi (Alibaba) |
| **HuggingFace** | `Tongyi-MAI/MAI-UI-8B` |
| **파라미터** | 8B |
| **VRAM** | ~16GB (fp16) |
| **ScreenSpot-Pro** | 65.8% |
| **라이선스** | Apache 2.0 |
| **HF 로그인** | 불필요 |

**특징**:
- UI-Venus와 함께 ScreenSpot-Pro 최상위권
- GGUF 양자화 버전 제공: `Mungert/MAI-UI-8B-GGUF` (4-bit: ~5GB)
- Ollama에서도 실행 가능: `ollama pull maternion/mai-ui:8b`

**같은 계열 모델**:

| 변형 | HuggingFace 경로 | ScreenSpot-Pro |
|------|-------------------|----------------|
| 2B | `Tongyi-MAI/MAI-UI-2B` | 57.4% |
| 8B | `Tongyi-MAI/MAI-UI-8B` | 65.8% |

**다운로드**:
```bash
huggingface-cli download Tongyi-MAI/MAI-UI-8B --local-dir ./MAI-UI-8B
huggingface-cli download Tongyi-MAI/MAI-UI-2B --local-dir ./MAI-UI-2B
```

**GitHub**: https://github.com/Tongyi-MAI/MAI-UI

---

## Tier 2: 강력한 대안 (ScreenSpot-Pro 30~50%)

### GUI-Actor-7B (Microsoft, NeurIPS 2025)

| 항목 | 내용 |
|------|------|
| **개발사** | Microsoft |
| **HuggingFace** | `microsoft/GUI-Actor-7B-Qwen2.5-VL` |
| **파라미터** | 7B |
| **VRAM** | ~14GB (bf16) |
| **ScreenSpot-Pro** | 44.6% (검증기 포함 시 47.7%) |
| **HF 로그인** | 불필요 |

**특징**:
- **좌표 없는(Coordinate-free) 그라운딩** — `<ACTOR>` 토큰이 시각 패치에 직접 어텐션
- 좌표를 텍스트로 생성하지 않으므로 해상도 변화에 강건
- 한 번의 forward pass로 여러 후보 영역 생성
- UI-TARS-72B(38.1%)를 7B 규모로 능가

**변형**:

| 변형 | HuggingFace 경로 |
|------|-------------------|
| Qwen2.5-VL 백본 | `microsoft/GUI-Actor-7B-Qwen2.5-VL` |
| Qwen2-VL 백본 | `microsoft/GUI-Actor-7B-Qwen2-VL` |

**다운로드**:
```bash
huggingface-cli download microsoft/GUI-Actor-7B-Qwen2.5-VL --local-dir ./GUI-Actor-7B
```

**GitHub**: https://github.com/microsoft/GUI-Actor

---

### UI-TARS-1.5-7B (ByteDance)

| 항목 | 내용 |
|------|------|
| **개발사** | ByteDance Seed |
| **HuggingFace** | `ByteDance-Seed/UI-TARS-1.5-7B` |
| **파라미터** | 7B |
| **VRAM** | ~14GB (bf16) |
| **ScreenSpot-Pro** | ~35% |
| **라이선스** | Apache 2.0 |
| **HF 로그인** | 불필요 |

**특징**:
- **think-before-act 추론** — 강화학습(RL) 기반 사고 과정
- 그라운딩 + 추론 + 액션 예측을 하나의 모델에서 수행
- OSWorld, WebVoyager, AndroidWorld 등 에이전트 벤치마크 강세
- 게임 플레이까지 확장 (v1.5 신규)

**같은 계열 모델**:

| 변형 | HuggingFace 경로 | 비고 |
|------|-------------------|------|
| 1.5-7B | `ByteDance-Seed/UI-TARS-1.5-7B` | 최신, 추론 강화 |
| 7B-DPO | `ByteDance-Seed/UI-TARS-7B-DPO` | v1, DPO 학습 |
| 7B-SFT | `ByteDance-Seed/UI-TARS-7B-SFT` | v1, SFT 기본 |
| 2B-SFT | `ByteDance-Seed/UI-TARS-2B-SFT` | 가장 작은 버전 |
| 72B-DPO | `ByteDance-Seed/UI-TARS-72B-DPO` | 대형 (H200 x2로 실행 가능) |
| 7B-GGUF | `bytedance-research/UI-TARS-7B-gguf` | llama.cpp용 양자화 |

**다운로드**:
```bash
huggingface-cli download ByteDance-Seed/UI-TARS-1.5-7B --local-dir ./UI-TARS-1.5-7B
huggingface-cli download ByteDance-Seed/UI-TARS-72B-DPO --local-dir ./UI-TARS-72B-DPO
```

**GitHub**: https://github.com/bytedance/UI-TARS

---

### UGround-V1-7B (OSU NLP, ICLR 2025 Oral)

| 항목 | 내용 |
|------|------|
| **개발사** | OSU NLP Group |
| **HuggingFace** | `osunlp/UGround-V1-7B` |
| **파라미터** | 7B (Qwen2-VL 기반) |
| **VRAM** | ~14GB (bf16) |
| **ScreenSpot-Pro** | 31.1% |
| **HF 로그인** | 불필요 |

**특징**:
- 범용 GUI 비주얼 그라운딩에 초점
- 모바일, 데스크톱, 웹 크로스 플랫폼
- vLLM 배포 지원 (temperature=0으로 일관된 좌표 출력)

**변형**: `osunlp/UGround-V1-2B`, `osunlp/UGround-V1-7B`, `osunlp/UGround-V1-72B`

**다운로드**:
```bash
huggingface-cli download osunlp/UGround-V1-7B --local-dir ./UGround-V1-7B
```

**GitHub**: https://github.com/OSU-NLP-Group/UGround

---

## Tier 3: 경량 모델 (2~3B)

### ShowUI-2B (CVPR 2025)

| 항목 | 내용 |
|------|------|
| **HuggingFace** | `showlab/ShowUI-2B` |
| **파라미터** | 2B (Qwen2-VL-2B 기반) |
| **VRAM** | ~8-10GB (fp16), INT8 ~5GB |
| **베이스** | Qwen2-VL-2B-Instruct |
| **HF 로그인** | 불필요 |

**특징**:
- UI-Guided Token Selection — 인터랙티브 컴포넌트 기반 동적 UI 그래프 구축
- 환각 액션 90% 감소, SeeClick 대비 5배 빠름
- 정규화된 [x, y] 좌표 반환
- CPU에서도 INT8으로 추론 가능

**다운로드**:
```bash
huggingface-cli download showlab/ShowUI-2B --local-dir ./ShowUI-2B
```

**GitHub**: https://github.com/showlab/ShowUI

---

### SeeClick (ACL 2024)

| 항목 | 내용 |
|------|------|
| **HuggingFace** | `cckevinn/SeeClick` |
| **파라미터** | 9.6B (Qwen-VL 기반) |
| **VRAM** | ~16GB |
| **HF 로그인** | 불필요 |

**특징**:
- **GUI 그라운딩의 선구적 모델** — ScreenSpot 벤치마크를 최초 제안
- 모바일, 데스크톱, 웹 크로스 플랫폼 사전학습
- 후속 모델들(ShowUI, UGround, UI-Venus 등)의 기반이 됨

**주의**: 최신 모델(UI-Venus, MAI-UI)에 비해 복잡한 UI에서의 성능은 낮음. 하지만 기초 비교용 베이스라인으로 가치 있음.

**다운로드**:
```bash
huggingface-cli download cckevinn/SeeClick --local-dir ./SeeClick
```

**GitHub**: https://github.com/njucckevin/SeeClick

---

### ZonUI-3B (WACV 2026)

| 항목 | 내용 |
|------|------|
| **HuggingFace** | `zonghanHZH/ZonUI-3B` |
| **파라미터** | 3B |
| **VRAM** | ~6GB (fp16) |
| **HF 로그인** | 불필요 |

**특징**:
- 단 24K 샘플로 학습 — 극한의 데이터 효율성
- ScreenSpot-v2: 86.4% (OS-Atlas-7B 84.1%, UI-TARS-2B 84.7% 능가)
- 단일 RTX 4090에서 학습 가능

**다운로드**:
```bash
huggingface-cli download zonghanHZH/ZonUI-3B --local-dir ./ZonUI-3B
```

**GitHub**: https://github.com/Han1018/ZonUI-3B

---

### Aria-UI (MoE 아키텍처)

| 항목 | 내용 |
|------|------|
| **HuggingFace** | `Aria-UI/Aria-UI-base` |
| **파라미터** | 3.9B 활성 (MoE) |
| **HF 로그인** | 불필요 |

**특징**:
- MoE 아키텍처로 효율적 추론
- 순수 비전 접근 (보조 입력 불필요)
- 초고해상도, 가변 종횡비 지원
- AndroidWorld 1위 (44.8%), OSWorld 3위 (15.2%)

**다운로드**:
```bash
huggingface-cli download Aria-UI/Aria-UI-base --local-dir ./Aria-UI-base
```

**GitHub**: https://github.com/AriaUI/Aria-UI

---

## 파싱 파이프라인: OmniParser V2

> 독립 VLM이 아닌 **UI 파싱 도구** — 스크린샷을 구조화된 요소 목록으로 변환하여 어떤 LLM과든 결합 가능

| 항목 | 내용 |
|------|------|
| **개발사** | Microsoft |
| **HuggingFace** | `microsoft/OmniParser-v2.0` |
| **구성** | Fine-tuned YOLOv8 (요소 탐지) + Florence-2 (아이콘 캡셔닝) + OCR |
| **VRAM** | ~4-6GB (두 모델 모두 경량) |
| **ScreenSpot-Pro** | 39.6% (GPT-4o와 결합 시) |
| **HF 로그인** | 불필요 |

**작동 방식**:
```
스크린샷 → OmniParser V2 → 구조화된 요소 목록 (바운딩 박스 + 라벨) → LLM → 액션
```

**장점**:
- 기존 내부 LLM(Kimi-K2.5)과 결합 가능 — 추가 VLM 배포 불필요
- V1 대비 60% 지연시간 감소
- 작은/밀집된 UI 요소 탐지 크게 개선

**주의**: YOLOv8 탐지 모듈은 AGPL 라이선스, Florence-2 캡셔닝은 MIT 라이선스

**다운로드**:
```bash
huggingface-cli download microsoft/OmniParser-v2.0 --local-dir ./OmniParser-v2.0
```

**GitHub**: https://github.com/microsoft/OmniParser

---

## 기타 참고 모델

### OS-Atlas-7B (ICLR 2025 Spotlight)

| 항목 | 내용 |
|------|------|
| **HuggingFace** | `OS-Copilot/OS-Atlas-Base-7B` (그라운딩), `OS-Copilot/OS-Atlas-Pro-7B` (액션) |
| **파라미터** | 7B (Qwen2-VL-7B 기반) |
| **ScreenSpot-Pro** | 18.9% |

- 1300만+ GUI 요소로 학습 (최대 규모 오픈소스 GUI 그라운딩 코퍼스)
- 크로스 플랫폼 (Windows, Linux, macOS, Android, Web)

### Aguvis (ICML 2025)

| 항목 | 내용 |
|------|------|
| **HuggingFace** | `ranpox/aguvis` 컬렉션 |
| **파라미터** | Qwen2-VL 기반 다양한 크기 |

- 최초의 순수 비전 자율 GUI 에이전트 (텍스트 주석 불필요)
- 2단계 학습: 그라운딩 → 계획/추론

### CogAgent-9B

| 항목 | 내용 |
|------|------|
| **HuggingFace** | `THUDM/cogagent-9b-20241220` |
| **파라미터** | 9B (GLM-4V-9B 기반) |
| **VRAM** | ~29GB (bf16) |

- 중국어 + 영어 이중 언어 GUI 상호작용
- 에이전트 실행 모델 (채팅 모델 아님)

---

## 모델 선택 가이드

### 복잡한 엔지니어링 도구 UI 용 (최우선 추천)

1. **UI-Venus-1.5-8B** — 단일 GPU 최고 성능 (ScreenSpot-Pro 68.4%)
2. **UI-Venus-1.5-30B-A3B** — 절대 최고 성능, H200 x2로 충분히 실행 가능
3. **MAI-UI-8B** — UI-Venus와 비슷한 수준의 대안

### 에이전트 (다단계 GUI 조작) 용

1. **UI-TARS-1.5-7B** — think-before-act 추론, 에이전트 벤치마크 강세
2. **UI-Venus-1.5-8B** — 그라운딩 + 에이전트 통합

### 기초 비교/벤치마킹 용

- **SeeClick** — 선구적 모델, 베이스라인 비교 용도
- **ShowUI-2B** — 경량 모델 기준점
- **OmniParser V2** — 기존 LLM 활용 파이프라인 비교

### 핵심 인사이트

> 전문 소프트웨어 UI의 핵심 난이도는 **4K 해상도에서 화면의 0.07%만 차지하는 작은 아이콘/버튼**을 인식하는 것. ZoomIn 전략(UI-Venus)이나 파싱 파이프라인(OmniParser)을 사용하는 모델이 전체 화면을 한 번에 처리하는 모델보다 월등히 우수.

## 참고 자료

- [ScreenSpot-Pro 리더보드](https://gui-agent.github.io/grounding-leaderboard/)
- [ScreenSpot-Pro 논문](https://arxiv.org/abs/2504.07981)
- [UI-Venus 기술 보고서](https://arxiv.org/abs/2602.09082)

## 관련 문서

- [이전: VLM 가이드 인덱스](./README.md)
- [다음: 오프라인 다운로드 가이드](./offline-download-guide.md)
- [위로: 개발 환경](../README.md)
