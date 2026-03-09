---
tags: [vlm, vllm, private-cloud, serving, offline]
level: intermediate
last_updated: 2026-03-09
status: in-progress
---

# Private Cloud에서 모델 다운로드 후 다음 단계

> 상황: Hugging Face에서 모델을 이미 받아서 private cloud 안의 로컬 디렉터리에 올려둔 상태
>
> 목표: local path 기준으로 `vLLM` 서버를 띄우고, 첫 요청까지 확인한다.

## 먼저 결론

- `vLLM`로 가장 먼저 붙여볼 모델은 `UI-Venus-1.5-8B` 또는 `MAI-UI-8B`
- `H200 x2`를 제대로 쓰려면 다음 단계에서 `UI-Venus-1.5-30B-A3B`를 `tensor parallel 2`로 올리면 된다
- `UI-TARS-1.5-7B`는 `qwen2_5_vl` 계열이라 `vLLM` 시도는 가능하지만, Hugging Face 카드 자체는 GitHub runtime/app 쪽을 같이 가리킨다
- `GUI-Actor`는 `vLLM-first` 모델로 보기 어렵다. Hugging Face usage 예시가 `gui_actor` Python 모듈을 직접 import한다
- `OmniParser v2`는 `vLLM` 모델 서버가 아니라 파서 컴포넌트다

## 어떤 모델부터 올릴까

| 모델 | 바로 `vLLM` 권장 | 이유 |
|---|---|---|
| `UI-Venus-1.5-8B` | 예 | HF 카드에 `vLLM` 배포 예시가 있다 |
| `UI-Venus-1.5-30B-A3B` | 예 | HF family가 `vLLM` 기준으로 정리되어 있고 `H200 x2`에 잘 맞는다 |
| `MAI-UI-8B` | 예 | HF 카드에 `vLLM` 배포 예시가 있다 |
| `MAI-UI-2B` | 예 | 스모크 테스트용으로 가장 빠르다 |
| `UI-TARS-1.5-7B` | 조건부 | HF 카드는 GitHub runtime/app를 같이 안내한다. `vLLM`은 시도 가능하지만 프롬프트와 후처리를 더 봐야 한다 |
| `GUI-Actor-7B-Qwen2.5-VL` | 아니오 | HF usage가 `gui_actor` 전용 코드 의존 |
| `OmniParser-v2.0` | 아니오 | 파서용 별도 컴포넌트 |

## Step 1. 버전과 GPU 상태 확인

`UI-Venus`와 `MAI-UI` Hugging Face 카드 기준으로 `vllm>=0.11.0`, `transformers>=4.57.0`가 필요하다.

```bash
vllm --version
python -c "import transformers; print(transformers.__version__)"
nvidia-smi
```

확인 포인트:

- `vllm`가 너무 낮으면 먼저 업그레이드
- `transformers`가 낮으면 model load 단계에서 실패할 수 있다
- `H200` 2장이 모두 보이는지 확인

## Step 2. 모델 디렉터리 구조 확인

예시는 `/data/models` 기준으로 적는다.

```bash
MODEL_DIR=/data/models/UI-Venus-1.5-8B

ls -lh "$MODEL_DIR"
find "$MODEL_DIR" -maxdepth 1 \( -name '*.json' -o -name '*.safetensors' \) | sort
```

최소 확인 파일:

- `config.json`
- `tokenizer_config.json`
- `preprocessor_config.json`
- `model.safetensors` 또는 `model-00001-of-0000x.safetensors`
- shard 구조라면 `model.safetensors.index.json`

이 단계에서 중요한 점:

- private cloud에서는 Hugging Face repo id 대신 **로컬 절대경로**로 서빙하는 것이 안전하다
- 폴더 하나에 모델 하나만 두는 편이 운영이 쉽다

## Step 3. 오프라인 안전 설정

Hugging Face 공식 문서 기준으로 `HF_HUB_OFFLINE=1`을 주면 Hub로 HTTP 호출을 막을 수 있다.

```bash
export HF_HUB_OFFLINE=1
export VLLM_API_KEY='change-this-internal-key'
```

운영 메모:

- 모델을 local path로 주더라도 custom runtime이나 tokenizer 로딩 중 Hub 체크를 시도하는 경우가 있다
- private cloud에서 외부 egress를 막아둘 거면 이 값을 기본값으로 두는 편이 낫다

## 꼭 전부 저장해야 하나

처음에는 **예**로 보는 편이 맞다. private cloud에 첫 반입할 때는 Hugging Face에서 받은 모델 폴더를 가능한 한 그대로 보관하는 것이 안전하다.

왜냐하면:

- shard 모델은 파일 하나만 빠져도 로딩이 실패한다
- `config`, `tokenizer`, `processor`, `generation_config` 중 하나가 빠져도 추론이나 전처리가 깨질 수 있다
- 나중에 `vLLM`이 아니라 `transformers` 또는 모델 전용 runtime으로 바꿀 때 추가 파일이 필요할 수 있다

### 일반적인 `vLLM` 모델에서 보통 꼭 필요한 것

- `config.json`
- `tokenizer_config.json`
- `tokenizer.json` 또는 vocab 파일들
- `preprocessor_config.json`
- `generation_config.json`
- `model.safetensors` 또는 모든 shard 파일
- `model.safetensors.index.json`가 있으면 그 index 파일

### 처음에는 같이 보관하는 편이 좋은 것

- `README.md`
- `LICENSE` 또는 라이선스 관련 파일
- special tokens, chat template, processor 관련 추가 JSON
- custom code용 Python 파일이 포함된 경우 그 파일들 전체

### 나중에 검증 후 줄여도 되는 것

- 예제 이미지
- 샘플 노트북
- 문서성 파일
- `.gitattributes`

### 예외

- `OmniParser-v2.0`는 전체 repo를 꼭 그대로 둘 필요는 없다. 이전 문서에 정리한 `icon_detect`, `icon_caption` 필수 파일만 있으면 된다
- `GUI-Actor`는 HF 가중치만 보관해서 끝나는 구조가 아니다. GitHub runtime 코드까지 같이 버전 고정해서 보관하는 편이 안전하다

운영 팁:

- 처음 반입: 원본 전체 보관
- 첫 서빙 성공 후: 별도 `serving-bundle` 폴더를 만들어 실제 사용 파일만 추려도 된다
- 단, 원본 복사본은 지우지 말고 남겨두는 편이 좋다

## Step 4. 첫 번째 모델부터 띄우기

처음에는 `UI-Venus-1.5-8B`나 `MAI-UI-8B`로 시작하는 편이 좋다.

### 4-1. 단일 GPU 예시

```bash
MODEL_DIR=/data/models/UI-Venus-1.5-8B

CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_DIR" \
  --served-model-name ui-venus-8b \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --generation-config vllm \
  --limit-mm-per-prompt '{"image":1}'
```

`MAI-UI-8B` 예시:

```bash
MODEL_DIR=/data/models/MAI-UI-8B

CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_DIR" \
  --served-model-name mai-ui-8b \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --generation-config vllm \
  --limit-mm-per-prompt '{"image":1}'
```

왜 이 옵션을 쓰는가:

- `--served-model-name`: 내부 API에서 stable model id로 쓰기 쉽다
- `--trust-remote-code`: model card 예시 기준 필요
- `--generation-config vllm`: vLLM 공식 문서 기준, HF `generation_config.json`의 추천값이 기본 sampling을 덮어쓰는 것을 막는다
- `--limit-mm-per-prompt '{"image":1}'`: 멀티모달 요청 한 장 기준으로 명시해두면 동작이 분명하다

### 4-2. 2 GPU 예시

`H200 x2`라면 이 단계부터 `UI-Venus-1.5-30B-A3B`를 붙여볼 수 있다.

```bash
MODEL_DIR=/data/models/UI-Venus-1.5-30B-A3B

CUDA_VISIBLE_DEVICES=0,1 vllm serve "$MODEL_DIR" \
  --served-model-name ui-venus-30b \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --generation-config vllm \
  --limit-mm-per-prompt '{"image":1}'
```

## Step 5. 서버가 떴는지 확인

vLLM Quickstart 기준으로 `/v1/models`가 가장 먼저 확인할 엔드포인트다.

```bash
curl -s http://127.0.0.1:8000/v1/models | python -m json.tool
```

모니터링 확인:

```bash
curl -s http://127.0.0.1:8000/metrics | head
```

정상이라면:

- `ui-venus-8b` 또는 `mai-ui-8b`가 model list에 보인다
- `/metrics`에서 vLLM Prometheus metrics가 출력된다

## Step 6. 첫 번째 이미지 요청 보내기

vLLM 공식 문서 기준, 멀티모달은 OpenAI Vision style chat completion으로 보낸다.

테스트용 스크린샷 하나를 준비한다:

```bash
IMG_PATH=./sample-screen.png
IMG_B64=$(base64 -w 0 "$IMG_PATH")
```

그다음 요청:

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

실전 메모:

- grounding 결과 포맷은 모델마다 다르다
- 좌표/박스 포맷은 각 모델 card 또는 GitHub 예시 프롬프트를 따라야 한다
- 첫 요청은 “정답 형식 강제”보다 “짧게 설명해라” 수준으로 보내는 편이 디버깅이 쉽다

## Step 7. 잘 되면 운영 형태로 바꾸기

vLLM Quickstart 기준으로 서버는 기본적으로 한 프로세스가 한 모델을 호스팅한다. 여러 모델을 같이 운영하려면 포트를 나눠서 여러 프로세스를 띄우면 된다.

예:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /data/models/UI-Venus-1.5-8B \
  --served-model-name ui-venus-8b \
  --port 8000 \
  --dtype bfloat16 \
  --trust-remote-code \
  --generation-config vllm \
  --limit-mm-per-prompt '{"image":1}'

CUDA_VISIBLE_DEVICES=1 vllm serve /data/models/MAI-UI-8B \
  --served-model-name mai-ui-8b \
  --port 8001 \
  --dtype bfloat16 \
  --trust-remote-code \
  --generation-config vllm \
  --limit-mm-per-prompt '{"image":1}'
```

추천 순서:

1. 단일 모델 smoke test
2. `/v1/chat/completions` 성공 확인
3. 두 번째 모델을 다른 포트에 올려 A/B 비교
4. 그 다음에만 reverse proxy, auth, systemd 같은 운영 레이어 추가

## Step 8. 자주 막히는 지점

### 8-1. 시작하자마자 로딩 실패

가능한 원인:

- `transformers` 버전 부족
- 모델 구조는 맞지만 custom code 로딩이 막힘
- model directory에 shard 일부가 누락됨

먼저 볼 것:

```bash
find "$MODEL_DIR" -maxdepth 1 | sort
```

### 8-2. OOM 또는 KV cache 부족

먼저 시도할 것:

- `8B` 모델부터 시작
- `--tensor-parallel-size 2` 사용
- `--max-model-len 4096` 추가
- `--gpu-memory-utilization 0.85`로 내려보기

## Step 9. 모델별 다음 액션

### UI-Venus

- 가장 먼저 `vLLM` smoke test할 모델
- `8B`로 먼저 확인한 뒤 `30B-A3B`로 올리면 된다

### MAI-UI

- `8B`가 가장 실용적
- `2B`는 파이프라인 확인용으로 좋다

### UI-TARS-1.5-7B

- 이 모델은 Hugging Face 태그상 `qwen2_5_vl` 계열이다
- vLLM 공식 supported models는 `Qwen2.5-VL` 아키텍처를 지원한다
- 따라서 **vLLM 시도는 가능하다**는 추론은 가능하다
- 다만 Hugging Face 카드 자체는 `UI-TARS` GitHub 코드와 desktop app을 같이 가리키므로, private cloud 실전 운영은 별도 runtime 검토가 필요하다

### GUI-Actor

- Hugging Face usage 예시가 `gui_actor.constants`, `gui_actor.modeling_qwen25vl`를 직접 import한다
- 즉, 일반적인 `vllm serve /path/to/model`로 끝나는 흐름이 아니다
- 이 모델은 전용 runtime 기준으로 분리해서 보는 편이 맞다

### OmniParser V2

- vLLM로 띄우는 대상이 아니다
- 스크린샷을 UI element list로 바꾸는 parser stage로 두고, 위 agent 모델과 조합해야 한다

## 내가 추천하는 실제 순서

1. `MAI-UI-8B` 또는 `UI-Venus-1.5-8B`를 `/data/models/...`에서 local path로 띄운다
2. `/v1/models`와 `/v1/chat/completions`를 확인한다
3. 같은 요청을 두 모델에 각각 보내서 응답 형식을 비교한다
4. 그다음 `UI-Venus-1.5-30B-A3B`를 `tensor-parallel-size 2`로 올린다
5. `UI-TARS`, `GUI-Actor`, `OmniParser`는 별도 runtime 트랙으로 나눈다

## 참고 문서

- vLLM Quickstart: https://docs.vllm.ai/en/stable/getting_started/quickstart/
- vLLM OpenAI-Compatible Server: https://docs.vllm.ai/en/stable/serving/openai_compatible_server/
- vLLM Supported Models: https://docs.vllm.ai/en/latest/models/supported_models.html
- vLLM Multimodal Inputs: https://docs.vllm.ai/en/stable/features/multimodal_inputs.html
- vLLM CLI serve options: https://docs.vllm.ai/en/latest/cli/serve/
- Hugging Face hub environment variables: https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables
- UI-Venus-1.5-8B HF card: https://huggingface.co/inclusionAI/UI-Venus-1.5-8B
- MAI-UI-8B HF card: https://huggingface.co/Tongyi-MAI/MAI-UI-8B
- UI-TARS-1.5-7B HF card: https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B
- GUI-Actor-7B-Qwen2.5-VL HF card: https://huggingface.co/microsoft/GUI-Actor-7B-Qwen2.5-VL
