---
category_major: "에이전트 하네스"
category_middle: "모델별 적용"
category_minor: "Qwen 모델 검토·운영"
note_kind: "학습"
classified_on: "2026-10-05"
---
# 서빙 설정 (vLLM / SGLang / 양자화)

## 프레임워크 선택

| 프레임워크 | 언제 쓰나 | 특징 |
|---|---|---|
| **vLLM** | 프로덕션 API, 동시 요청 많음 | 텐서 병렬 + continuous batching, 동시 부하에서 2~4배 효율 |
| **SGLang** | 프로덕션, 에이전트/구조화 출력 | Qwen 공식 레시피 제공, reasoning parser 지원 |
| **TokenSpeed** | Qwen3.8 전용 최적화 | Qwen이 공식 레시피 발행한 3종 중 하나 |
| Ollama/llama.cpp | 개인·소규모, 빠른 도입 | 편의성 우선, 동시성엔 불리 |

## vLLM 기본 서빙 커맨드 (Qwen 계열 권장 패턴)

```bash
vllm serve Qwen/Qwen3.8-27B-FP8 \
  --served-model-name qwen3.8-27b \
  --trust-remote-code \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.92 \
  --enable-auto-tool-choice \
  --tool-call-parser <parser> \
  --reasoning-parser qwen3 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --kv-cache-dtype fp8
```

포인트:

- **`--enable-prefix-caching`**: 동일 시스템 프롬프트/긴 공통 컨텍스트를 재사용하는 서비스(회사 내부 봇 등)에서 프리필 비용을 크게 절감. **가장 비용 효과가 큰 단일 옵션.**
- **`--kv-cache-dtype fp8`**: KV 캐시 메모리를 절반으로 줄여 컨텍스트 여유 확대. 품질 손실은 미미.
- **`--max-num-seqs` 제한**: 소형 GPU에서 동시성을 무작정 올리면 KV 캐시가 넘쳐 OOM/스래싱. 워크로드에 맞게 튜닝.
- **`--tool-call-parser`를 반드시 명시**: vLLM은 템플릿을 자동 감지하지 않는다. 미지정 시 툴콜이 텍스트로 새어 나온다.

### 툴콜 파서 선택 (커뮤니티 검증)

- 긴 컨텍스트 에이전트 작업에는 **`qwen3_xml`** 이 regex 기반 `qwen3_coder`보다 안정적이라는 커뮤니티 테스트 다수 (C 기반 XML 파서, malformed XML 자동 치유, 특수문자 강건).
- Qwen3(구세대) 계열은 `--tool-call-parser hermes`가 공식 권장. 자기 버전의 모델 카드 권장값을 먼저 확인하고, 장기 에이전트에서 불안정하면 `qwen3_xml`로 전환 테스트.

## 커스텀 chat template 이슈

커뮤니티 보고에 따르면 Qwen3.x 세대 공식 Jinja 템플릿에는 엣지 케이스가 있다:

- 툴콜 중 thinking 태그가 제대로 닫히지 않음
- 히스토리의 thinking 블록이 컨텍스트에 새어 나감 (reasoning leakage)

vLLM은 템플릿을 자동 감지하지 않으므로, 문제가 관찰되면 검증된 커스텀 템플릿을 `--chat-template`으로 명시적으로 지정하는 것이 해법이다.

## 양자화

| 형식 | 상황 | 비고 |
|---|---|---|
| **FP8 (공식 체크포인트)** | 첫 선택지 | block 128 fine-grained, 원본과 성능 거의 동일. vLLM/SGLang 네이티브 지원 |
| BF16 원본 | 검증 기준선 | VRAM ~55GB+ (28B 파라미터) |
| AWQ/GPTQ INT4 | VRAM이 절대적으로 부족할 때 | Qwen 계열 dense 모델은 4bit에서 대부분 2% 미만 열화 |
| Q3 이하 (GGUF) | 비권장 | MoE/dense 모두 급격한 품질 하락. 커뮤니티 테스트에서 일관된 결과 |

**주의**: 커뮤니티 SFT 파생 모델(Claude 증류 등)은 장기 컨텍스트(65K+)에서 출력 포맷이 drift하는 사례가 보고됨. 프로덕션에는 공식/공식 양자화 체크포인트를 쓰는 것이 안전.

## Ollama 계열로 돌릴 때의 운영 환경변수

```
OLLAMA_KEEP_ALIVE=-1          # 유휴 언로드 방지 (기본 5분 — 서버 배포에선 필수)
OLLAMA_NUM_PARALLEL=4         # 동시 요청 수 (VRAM 여유에 맞게)
OLLAMA_FLASH_ATTENTION=1      # Ampere+ GPU에서 20-30% 속도 향상
OLLAMA_GPU_OVERHEAD=512       # OOM 예방용 여유 VRAM
```

## 컨텍스트/VRAM 산수

- 27B FP8 가중치 ≈ 28GB + KV 캐시. 262K 풀 컨텍스트를 쓰려면 상당한 KV 여유가 필요.
- 실전 접근: 워크로드의 실제 P95 컨텍스트를 측정해 `--max-model-len`을 그에 맞게 제한하면 KV 캐시 절약 → 동시성/속도 향상. 1M 확장(YaRN)은 정말 필요할 때만 — static YaRN은 짧은 입력에서도 성능을 깎는다.

## 출처

- https://huggingface.co/Qwen/Qwen3.8-27B-FP8
- https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix (파서/템플릿 실전 검증)
- https://qwen.readthedocs.io/en/latest/deployment/vllm.html
- https://www.promptquorum.com/power-local-llm/qwen-local-deployment-complete-guide-2026 (Ollama 환경변수)
- https://baeseokjae.github.io/posts/qwen-3-32b-local-guide-2026 (KV 캐시/VRAM 산수)
