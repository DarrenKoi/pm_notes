---
tags: [llm-serving, prefill, decode, kv-cache, prefix-cache, quantization, inference]
level: intermediate
last_updated: 2026-10-09
type: learning
category_major: "AI·DT"
category_middle: "AI 기초"
category_minor: "파운데이션 모델"
note_kind: "학습"
---

# LLM은 어떻게 서빙되는가? Prefill, Cache, 양자화

> 한 요청은 프롬프트를 한꺼번에 읽는 prefill과 토큰을 하나씩 만드는 decode로 나뉜다. cache는 이미 한 계산을 재사용해 두 구간을 줄이고, 양자화는 가중치를 적은 비트로 저장해 같은 GPU에 더 큰 모델을 올린다.

**시각화**: [llm-serving-prefill-cache-quantization.html](./llm-serving-prefill-cache-quantization.html) — 프롬프트·출력 길이, cache 적중률, 비트 수를 바꿔 가며 볼 수 있다.

> [!info] 범위
> 공개된 decoder-only Transformer 서빙의 일반 원리를 다룬다. 속도·메모리 예시는 계산식을 보여 주려는 가정값이며 측정값이 아니다. 특정 엔진의 옵션 이름과 기본값은 버전마다 다르므로 사용하는 엔진의 문서로 확인한다.

## 왜 필요한가? (Why)

- 사내 LLM은 OpenAI 호환 API 뒤에 서빙 엔진과 GPU가 있다. 응답이 느리거나 동시 사용자가 늘 때 무엇을 바꿔야 하는지는 이 세 개념으로 대부분 설명된다.
- 프롬프트를 어떤 순서로 조립하느냐에 따라 같은 요청의 대기 시간이 달라진다. 이것은 서버 설정이 아니라 호출하는 쪽에서 정하는 부분이다.
- 모델을 직접 올릴 때 "8B 모델이 이 GPU에 들어가는가"는 양자화 비트 수와 KV cache 크기로 계산할 수 있다.

## 핵심 개념 (What)

### 1. Prefill과 decode

[Attention은 LLM 안에서 어떻게 쓰이는가?](./how-attention-powers-llm.md)에서 본 forward pass를 서빙 관점에서 다시 나누면 두 구간이다.

| 구간 | 하는 일 | 처리 방식 | 사용자가 느끼는 지표 |
|------|---------|-----------|----------------------|
| **Prefill** | 프롬프트 전체를 읽고 각 토큰의 Key/Value를 계산해 저장한다. 끝나면 첫 토큰이 나온다 | 프롬프트 토큰 전부를 한 번에 병렬 계산 | TTFT (첫 토큰까지 시간) |
| **Decode** | 토큰을 하나 만들고, 그 토큰의 K/V를 cache에 추가하고, 다음 토큰으로 넘어간다 | 한 걸음에 토큰 한 개, 순차 | 초당 출력 토큰 수 |

- prefill은 병렬이라 토큰당 시간이 매우 짧다. decode는 순차라 토큰당 시간이 훨씬 길다. 그래서 **입력 1,000토큰보다 출력 100토큰이 더 오래 걸리는** 일이 흔하고, API 가격표에서 출력 토큰이 더 비싼 이유이기도 하다.
- 응답 시간을 줄이려면 TTFT가 문제인지(프롬프트가 길다), 전체 시간이 문제인지(출력이 길다)부터 나눠 본다.

> **용어 주의**: "prefill"은 다른 뜻으로도 쓰인다. 모델 응답의 첫 부분을 호출자가 미리 채워 넣어 출력 형식을 유도하는 프롬프트 기법(예: 응답을 `{`로 시작시켜 JSON 유도)도 prefill이라고 부른다. 이 문서의 prefill은 서빙 구간을 뜻한다.

### 2. KV cache: 한 요청 안에서의 재사용

causal mask 때문에 앞 토큰의 계산 결과는 뒤에 무엇이 오든 변하지 않는다. 그래서 각 층에서 계산한 Key와 Value를 저장해 두면, 새 토큰을 만들 때 앞 토큰들을 다시 계산할 필요가 없다.

- **없으면**: 토큰을 하나 만들 때마다 지금까지의 전체 시퀀스를 다시 통과시킨다. 계산량이 출력 길이의 제곱으로 늘어난다.
- **있으면**: 새 토큰 한 개만 계산하고, attention은 저장된 K/V를 읽기만 한다.

대가는 메모리다.

```text
KV cache 크기 ≈ 2(K와 V) × 층 수 × 토큰 수 × K/V 벡터 크기 × 값 하나의 바이트 수
```

32층, 벡터 4,096차원, 16비트 모델이면 토큰당 약 0.5MB이고, 8,000토큰 컨텍스트 하나가 약 4GB를 쓴다. 동시 요청 수만큼 곱해진다. **GPU 메모리는 가중치만이 아니라 "가중치 + 동시 요청들의 KV cache"로 채워진다.** 컨텍스트 길이 제한과 동시 처리 수가 서로 맞바꾸는 관계인 이유다. (최근 모델은 K/V 쪽 head 수를 줄이는 구조로 이 크기를 낮추기도 한다.)

### 3. Prefix cache: 요청 사이의 재사용

같은 원리를 요청 경계 너머로 확장한 것이다. 이전 요청의 KV cache를 보관했다가, 새 요청의 프롬프트가 **첫 토큰부터 동일한 구간**만큼 그대로 쓴다. prompt caching, prefix caching이라고도 부른다.

- 재사용되는 만큼 prefill을 건너뛰므로 **TTFT가 줄어든다.** decode 속도는 그대로다.
- **앞에서부터 연속으로 같은 구간만** 재사용된다. 중간의 한 토큰이 다르면 그 뒤는 전부 새로 계산한다. 각 토큰의 K/V가 자기 앞의 모든 토큰에 의존하기 때문이다.
- 그래서 프롬프트 조립 순서가 중요하다.

```text
좋은 순서                        나쁜 순서
[시스템 지침]      고정           [현재 시각]        매번 바뀜 → 뒤가 전부 무효
[도구 정의]        고정           [시스템 지침]
[참고 문서]        세션 동안 고정   [참고 문서]
[대화 이력]        뒤에 추가만     [질문]
[이번 질문]        매번 바뀜
```

멀티턴 대화는 매 턴 이전 대화가 그대로 앞에 오므로 자연스럽게 적중한다. 대화 이력을 중간에서 수정·요약하면 그 지점부터 무효가 된다.

### 4. 이름이 비슷한 다른 cache

| 이름 | 저장하는 것 | 일치 조건 | 위치 |
|------|-------------|-----------|------|
| KV cache | 한 요청의 토큰별 K/V | - (항상 사용) | 서빙 엔진 |
| Prefix cache | 이전 요청들의 K/V | 토큰 단위로 앞부분 완전 일치 | 서빙 엔진 |
| 응답 cache | 완성된 답변 텍스트 | 같은 질문(문자열 일치 또는 의미 유사) | 애플리케이션 |

응답 cache는 모델을 아예 호출하지 않는 것이라 성격이 다르다. 유사 질문에 이전 답을 돌려주므로 답이 틀릴 위험을 따로 관리해야 한다.

### 5. 양자화(Quantization)

학습된 가중치는 보통 16비트 실수(fp16/bf16)다. 양자화는 이것을 8비트나 4비트 정수로 바꿔 저장한다.

```text
원래 값 w  →  정수 q = round(w / scale)  →  쓸 때는 q × scale 로 복원
scale은 가중치 묶음(행, 채널, 블록)마다 따로 정한다
```

**얻는 것**

- 메모리: 80억 파라미터 모델이 16비트로 약 16GB, 8비트로 약 8GB, 4비트로 약 4~5GB다(scale 같은 부가 정보 때문에 정확히 절반·1/4보다 조금 크다).
- 속도: decode는 매 토큰마다 가중치 전체를 메모리에서 읽어야 해서, 읽을 양이 줄면 빨라지는 경우가 많다. 다만 하드웨어와 엔진이 그 정수 형식을 빠르게 처리할 수 있을 때만 그렇다.

**잃는 것**: 반올림 오차만큼 출력이 달라진다. 일반적으로 8비트는 차이가 거의 없고, 4비트는 작은 손실로 널리 쓰이며, 그 아래로 갈수록 손실이 빠르게 커진다고 알려져 있다. 모델과 작업마다 다르므로 자기 작업의 평가셋으로 직접 확인해야 한다.

**진행 방식**

| 방식 | 내용 | 비용 |
|------|------|------|
| PTQ (학습 후 양자화) | 학습이 끝난 가중치를 변환한다. 서빙용 양자화는 대부분 이 방식이다 | 분~시간 단위, 재학습 없음 |
| QAT (양자화 인지 학습) | 학습 중에 양자화를 흉내 내어 모델이 오차에 적응하게 한다 | 학습을 다시 해야 함 |

PTQ도 단순 반올림에서 끝나지 않는다. 널리 쓰이는 기법은 **보정 데이터(calibration data)** 수백 개를 모델에 흘려 보내 실제 활성값을 관찰하고, 출력에 영향이 큰 가중치의 오차를 줄이는 방향으로 반올림한다.

- **GPTQ**: 층별로 가중치를 차례로 양자화하면서, 생긴 오차를 아직 양자화하지 않은 가중치에 보정해 넣는다.
- **AWQ**: 활성값이 큰 채널의 가중치를 보호하도록 scale을 조정한다.
- **GGUF 양자화** (`Q4_K_M` 등): llama.cpp 계열 엔진용 파일 형식. 층의 중요도에 따라 비트 수를 섞어 쓴다.
- **bitsandbytes 4비트**: 모델을 불러올 때 즉석에서 양자화한다. 미세조정(QLoRA)에 주로 쓴다.

**서비스까지의 흐름**

```text
학습된 가중치(16비트)
 → 서빙 엔진과 하드웨어가 지원하는 형식을 고른다
 → 보정 데이터 준비 (실제 사용과 비슷한 프롬프트 수백 개)
 → 양자화 실행
 → 원본과 같은 평가셋으로 비교 (품질, 속도, 메모리)
 → 통과하면 배포
```

형식은 엔진을 따라간다. 양자화 형식마다 지원하는 엔진이 다르기 때문에 "어떤 엔진으로 서빙하는가"를 먼저 정하고 형식을 고른다.

### 6. 배칭

GPU는 여러 요청을 묶어 한 번에 계산할 때 효율이 높다. 서빙 엔진은 요청이 끝나길 기다리지 않고 decode 걸음마다 진행 중인 요청들을 묶었다 풀었다 하는 **continuous batching**을 쓴다. 동시 요청이 늘면 개별 요청은 조금 느려지지만 전체 처리량은 크게 늘어난다. 한 요청의 속도와 서버 전체의 처리량은 다른 지표다.

## 어떻게 사용하는가? (How)

### KV cache 메모리와 양자화 오차 계산해 보기

```python
import numpy as np


def kv_cache_gb(tokens, layers=32, kv_dim=4096, bytes_per_value=2):
    """K and V, per layer, per token."""
    return 2 * layers * tokens * kv_dim * bytes_per_value / 2**30


def quantize(W, bits):
    """Symmetric uniform quantization with one scale per row, then dequantize."""
    qmax = 2 ** (bits - 1) - 1
    scale = np.abs(W).max(axis=1, keepdims=True) / qmax
    return np.clip(np.round(W / scale), -qmax, qmax) * scale


print(f"8,000토큰 컨텍스트 1개의 KV cache: {kv_cache_gb(8000):.1f} GB")

rng = np.random.default_rng(0)
W = rng.normal(0, 0.02, size=(256, 256)).astype(np.float32)  # 가중치 행렬 하나
x = rng.normal(size=256).astype(np.float32)                   # 이 층에 들어오는 입력

errors = {}
for bits in (8, 4, 3, 2):
    errors[bits] = np.linalg.norm(W @ x - quantize(W, bits) @ x) / np.linalg.norm(W @ x)
    print(f"{bits}비트: 층 출력 오차 {errors[bits]:.1%}")

assert errors[8] < errors[4] < errors[3] < errors[2]  # 비트를 줄일수록 오차가 커진다
assert errors[8] < 0.01
```

출력:

```text
8,000토큰 컨텍스트 1개의 KV cache: 3.9 GB
8비트: 층 출력 오차 0.7%
4비트: 층 출력 오차 13.2%
3비트: 층 출력 오차 31.7%
2비트: 층 출력 오차 82.2%
```

이것은 보정 없는 단순 반올림을 층 하나에 적용한 결과다. 4비트 아래에서 오차가 급격히 커지는 모양을 볼 수 있다. GPTQ·AWQ 같은 기법이 필요한 이유가 이 오차를 줄이는 것이다.

### 프롬프트를 cache에 유리하게 조립하기

```python
from openai import OpenAI

client = OpenAI(base_url="http://internal-llm.example/v1", api_key="unused")  # 사내 엔드포인트로 교체

SYSTEM = "너는 반도체 계측 장비 문서를 읽고 답하는 조수다. ..."   # 고정: 글자 하나도 바꾸지 않는다
MANUAL = open("manual_excerpt.txt", encoding="utf-8").read()   # 세션 동안 고정


def ask(question: str, history: list[dict]) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"참고 문서:\n{MANUAL}"},
        *history,                                   # 뒤에 추가만 한다
        {"role": "user", "content": question},      # 바뀌는 내용은 맨 뒤
    ]
    r = client.chat.completions.create(model="kimi-k2.5", messages=messages, temperature=0)
    return r.choices[0].message.content
```

피해야 할 것:

- 시스템 프롬프트에 현재 시각, 요청 ID, 사용자 이름을 넣는다 → 맨 앞이 매번 달라져 전부 무효
- 문서 여러 개를 요청마다 다른 순서로 넣는다 → 정렬 기준을 고정한다
- JSON을 키 순서가 보장되지 않는 방식으로 직렬화해 넣는다

prefix cache가 켜져 있는지, 얼마나 적중했는지는 서빙 엔진 설정과 응답의 usage 필드(지원하는 경우)로 확인한다. 엔진마다 다르므로 운영 담당자에게 확인이 필요하다.

### 실무와 연결하기

- **RAG**: 검색 결과가 매번 달라지면 그 지점부터 cache가 무효다. 고정 지침을 검색 결과보다 앞에 두면 최소한 지침 부분은 재사용된다.
- **VLM 추출 파이프라인**: 같은 지시문으로 스크린샷만 바꿔 반복 호출하는 구조라면, 지시문을 이미지 앞에 두는 편이 유리하다. 출력을 짧은 구조화 형식으로 제한하면 decode 시간이 줄어 전체 처리량이 오른다.
- **모델 크기 선택**: 가벼운 VLM과 무거운 VLM 중 고를 때, 무거운 모델을 4비트로 올리는 것과 가벼운 모델을 16비트로 올리는 것이 비슷한 메모리를 쓸 수 있다. 어느 쪽이 나은지는 자기 작업의 평가셋으로만 판단할 수 있다.

## 참고 자료 (References)

- Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers* (2022): <https://arxiv.org/abs/2210.17323>
- Lin et al., *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration* (2023): <https://arxiv.org/abs/2306.00978>
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (2023): <https://arxiv.org/abs/2305.14314>
- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM, 2023): <https://arxiv.org/abs/2309.06180>

## 관련 문서

- [Attention은 LLM 안에서 어떻게 쓰이는가?](./how-attention-powers-llm.md) - causal mask와 생성 루프. KV cache가 가능한 이유
- [LLM 학습 데이터셋은 어떻게 준비하는가?](./llm-dataset-preparation.md) - 서빙 이전, 학습 단계의 데이터
- [Foundation LLM은 어떻게 만들어지는가?](./how-foundation-llms-are-built.md) - 전체 제작 파이프라인
