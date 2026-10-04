# 샘플링 파라미터와 모드 전환

## thinking vs instruct(비-thinking) 모드

Qwen3.8은 **thinking 모드가 기본**이다. thinking이 켜지면 응답 앞에 추론 블록(사고 과정)이 붙고, 정확도가 올라가지만 출력 토큰이 수 배로 늘어난다.

| 상황 | thinking | 이유 |
|---|---|---|
| 복잡한 추론, 수학, 버그 디버깅, 아키텍처 결정, 에이전트 계획 | ON | 체인 오브 소트가 품질을 크게 끌어올림 |
| 일반 채팅, 요약, 단순 코드 생성, RAG 문서 Q&A, 창작 | OFF | 지연만 늘고 품질 이득 거의 없음 |

**실무 팁**: 루틴한 요청은 기본으로 thinking을 끄고, 필요한 요청에만 켜는 운영이 지연/비용 면에서 가장 효율적이다.

## 모드별 공식 권장 샘플링 값

| 파라미터 | thinking 모드 | instruct 모드 | Qwen3.8 권장(모델 카드) |
|---|---|---|---|
| Temperature | **0.6** | **0.7** | thinking: **1.0** |
| TopP | **0.95** | **0.8** | thinking: **0.95** |
| TopK | **20** | **20** | **20** |
| MinP | 0 | 0 | 0 |
| presence_penalty | 0~2 (양자화 모델은 **1.5** 권장) | 동일 | — |
| max_tokens | 32,768 이상 | 8,192 정도 | reasoning 262K + 응답 131K 여유 할당 |

### 절대 규칙

1. **thinking 모드에서 greedy decoding(temp=0, top_k=1) 금지.** 성능이 크게 하락하고 무한 반복에 빠진다.
2. **top_k=1은 greedy와 동일** — 결정론이 필요하면 temperature를 0으로 하지 말고 모드 자체를 재검토.
3. **출력 길이를 충분히**: max_tokens가 작으면 추론 블록만으로 예산이 소진되어 본문이 잘린다. "답이 이상하다"고 느껴지면 먼저 잘림 여부(`finish_reason=length`)를 확인.
4. 반복이 관찰되면 presence_penalty를 0~2 범위에서 올리되, 너무 높으면 언어 혼용(mixed language)과 품질 저하가 생긴다.

## Qwen3.8 전용 추론 제어

```python
completion = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B-FP8",
    messages=messages,
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": True,    # 기본값
            "preserve_thinking": True,  # 기본값
        },
    },
    reasoning_effort="medium",  # xhigh(기본) | medium | low
    stream=True,
)
```

- `reasoning_effort`: xhigh는 정밀 분석이 필요한 복잡 과제, medium은 정확도-속도 균형, low는 빠른 응답. **비용 최적화의 1차 수단.**
- `preserve_thinking`: 멀티턴에서 이전 턴의 thinking을 컨텍스트에 유지. 결정 일관성에 유리하지만 KV/컨텍스트 비용이 큼 — 일상 대화에서는 끄는 것이 효율적.
- 멀티턴에서는 **히스토리에 thinking 내용을 넣지 말고 최종 답변만** 넣는 것이 Qwen 공식 권장(preserve_thinking을 명시적으로 켠 경우 제외).

## vLLM에서 모드 전환 (비-thinking 예시)

```json
{
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "max_tokens": 8192,
  "presence_penalty": 1.5,
  "chat_template_kwargs": {"enable_thinking": false}
}
```

## 출처

- https://huggingface.co/Qwen/Qwen3.8-27B-FP8 (모델 카드, reasoning_effort/preserve_thinking)
- https://ai-tldr.dev/models/qwen3-8-27b/ (thinking 1.0/0.95/20, instruct 0.7/0.8/20)
- https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html
- https://huggingface.co/Qwen/Qwen3-32B-GGUF (presence_penalty 1.5, 양자화 모델 반복 억제)
- https://jan.ai/post/qwen3-settings (실전 모드별 세팅 정리)
