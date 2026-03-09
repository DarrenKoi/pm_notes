---
tags: [vlm, vllm, private-cloud, serving, offline]
level: intermediate
last_updated: 2026-03-09
status: in-progress
---

# Private Cloud에서 모델 다운로드 후 다음 단계

> 상황: 모델 폴더를 이미 private cloud 로컬 경로에 올려둔 상태
>
> 목표: local path 기준으로 `vLLM` 서버를 띄우고 첫 요청까지 확인한다.

## 이 문서의 역할

- 이 문서는 `첫 vLLM 성공 절차`의 기준 문서다.
- 어떤 모델을 받을지 고르는 일은 [Hugging Face 다운로드 shortlist](./huggingface-private-cloud-downloads.md)에서 끝낸다.
- 서빙 방식 선택은 [모델 서빙 가이드](./serving-guide.md)에서 끝낸다.

## 시작 모델

처음에는 아래 둘 중 하나로 시작하는 편이 가장 안전하다.

- `UI-Venus-1.5-8B`
- `MAI-UI-8B`

둘 다 성공한 뒤에만 `UI-Venus-1.5-30B-A3B`나 다른 트랙으로 넓히는 편이 디버깅이 쉽다.

## Step 1. 버전과 GPU 상태 확인

```bash
vllm --version
python -c "import transformers; print(transformers.__version__)"
nvidia-smi
```

확인 포인트:

- `vllm`이 너무 오래된 버전이면 먼저 정리한다
- `transformers`가 낮으면 모델 로딩 단계에서 깨질 수 있다
- `H200` 두 장이 모두 보이는지 확인한다

## Step 2. 모델 디렉터리 확인

```bash
MODEL_DIR=/data/models/UI-Venus-1.5-8B

ls -lh "$MODEL_DIR"
find "$MODEL_DIR" -maxdepth 1 | sort
```

최소 확인 파일:

- `config.json`
- `tokenizer_config.json`
- `preprocessor_config.json`
- `generation_config.json`
- `model.safetensors` 또는 shard 파일 전체
- shard 구조면 `model.safetensors.index.json`

private cloud에서는 repo id보다 **로컬 절대경로**로 서빙하는 편이 안전하다.

## Step 3. 오프라인 안전 설정

```bash
export HF_HUB_OFFLINE=1
export VLLM_API_KEY='change-this-internal-key'
```

메모:

- local path를 써도 내부적으로 Hub 확인을 시도하는 경우가 있다
- private cloud 기본값으로 `HF_HUB_OFFLINE=1`을 두는 편이 낫다

## Step 4. 첫 모델 실행

### `UI-Venus-1.5-8B`

```bash
MODEL_DIR=/data/models/UI-Venus-1.5-8B

CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_DIR" \
  --served-model-name ui-venus-8b \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --api-key "$VLLM_API_KEY" \
  --generation-config vllm \
  --limit-mm-per-prompt '{"image":1}'
```

### `MAI-UI-8B`

```bash
MODEL_DIR=/data/models/MAI-UI-8B

CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_DIR" \
  --served-model-name mai-ui-8b \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --api-key "$VLLM_API_KEY" \
  --generation-config vllm \
  --limit-mm-per-prompt '{"image":1}'
```

핵심 옵션:

- `--served-model-name`: 내부 API에서 고정된 이름으로 쓰기 쉽다
- `--trust-remote-code`: UI 계열 모델에서 자주 필요하다
- `--generation-config vllm`: HF 기본 generation 설정이 sampling을 덮어쓰는 문제를 줄인다
- `--limit-mm-per-prompt '{"image":1}'`: 첫 실험은 이미지 1장 기준이 가장 단순하다

## Step 5. 서버 상태 확인

```bash
curl -s http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer ${VLLM_API_KEY}" | python -m json.tool
curl -s http://127.0.0.1:8000/metrics | head
```

정상이라면:

- `/v1/models`에 `ui-venus-8b` 또는 `mai-ui-8b`가 보인다
- `/metrics`가 응답한다

## Step 6. 첫 이미지 요청 보내기

```bash
IMG_PATH=./sample-screen.png
IMG_B64=$(base64 -w 0 "$IMG_PATH")
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${VLLM_API_KEY}" \
  -d @- <<JSON
{
  "model": "ui-venus-8b",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Identify the most likely button to open settings. Answer briefly."
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/png;base64,${IMG_B64}"
          }
        }
      ]
    }
  ],
  "temperature": 0,
  "max_tokens": 200
}
JSON
```

첫 요청은 좌표 JSON을 강제하기보다, 짧은 자연어 응답으로 성공 여부만 확인하는 편이 낫다.

## Step 7. 잘 되면 다음 확장

### 여러 모델 비교

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /data/models/UI-Venus-1.5-8B \
  --served-model-name ui-venus-8b \
  --port 8000 \
  --dtype bfloat16 \
  --trust-remote-code \
  --api-key "$VLLM_API_KEY" \
  --generation-config vllm \
  --limit-mm-per-prompt '{"image":1}'

CUDA_VISIBLE_DEVICES=1 vllm serve /data/models/MAI-UI-8B \
  --served-model-name mai-ui-8b \
  --port 8001 \
  --dtype bfloat16 \
  --trust-remote-code \
  --api-key "$VLLM_API_KEY" \
  --generation-config vllm \
  --limit-mm-per-prompt '{"image":1}'
```

### `H200 x2`로 `UI-Venus-1.5-30B-A3B`

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve /data/models/UI-Venus-1.5-30B-A3B \
  --served-model-name ui-venus-30b \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --api-key "$VLLM_API_KEY" \
  --generation-config vllm \
  --limit-mm-per-prompt '{"image":1}'
```

## 자주 막히는 지점

### 시작 직후 로딩 실패

먼저 볼 것:

```bash
find "$MODEL_DIR" -maxdepth 1 | sort
```

가능한 원인:

- `transformers` 버전 부족
- custom code 로딩 실패
- shard 일부 누락

### OOM 또는 KV cache 부족

먼저 시도할 것:

- `8B` 모델부터 시작
- `--tensor-parallel-size 2`
- `--max-model-len 4096`
- `--gpu-memory-utilization 0.85`

### 응답 형식이 들쭉날쭉함

첫 단계에서는:

- `temperature=0`
- 이미지 1장만 사용
- 자연어 한 문장 응답으로 테스트

이게 안정화된 뒤 좌표 JSON이나 강한 포맷 제약을 추가한다.

## 이 문서 다음에 볼 것

- Flask API wrapper가 필요하면 [UI-Venus-1.5-8B Cloud API Guide with Flask Blueprint](./ui-venus-flask-blueprint-cloud-guide.md)
- parser pipeline이 필요하면 [OmniParser V2 설치 및 Cloud API 패턴](./omniparser-cloud-api-guide.md)
- `GUI-Actor`, `UI-TARS`, `OmniParser`는 `vLLM direct path`와 별도 트랙으로 본다

## 관련 문서

- [이전: 모델 서빙 가이드](./serving-guide.md)
- [위로: VLM 가이드 인덱스](./README.md)
