# 툴 콜링 & 에이전트 운영 노하우

Qwen3.8-27B는 코딩/에이전트 벤치마크(Terminal Bench 73, SWE-bench Pro 61.7, OSWorld 84.3)가 강점이다. 에이전트 하네스에서 쓸 때가 이 모델의 최대 효율 구간.

## 기본 구성 (vLLM)

```bash
vllm serve Qwen/Qwen3.8-27B-FP8 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3
```

- Qwen 공식 문서는 **Hermes-style tool use**를 Qwen3 계열의 권장 포맷으로 명시. 툴은 JSON Schema로 기술.
- 장기 에이전트/긴 컨텍스트에서는 커뮤니티 검증상 `qwen3_xml` 파서가 더 안정적 (C 기반 XML 파서, malformed XML 자동 치유).

## thinking 모드 + 툴콜 주의사항

- **thinking 모드에서는 `tool_choice`가 `"auto"`/`"none"`만 지원**된다. 특정 툴 강제(`{"type":"function",...}`)나 `"required"`는 thinking을 끄고 써야 한다.
- **ReAct 등 stopword 기반 툴 템플릿 금지** (Qwen 공식 경고): reasoning 모델은 thought 섹션에서 stopword를 출력할 수 있어 오동작한다. 내장 Hermes/XML 템플릿을 쓸 것.
- 툴 결과를 모델에 돌려줄 때 `tool_call_id`를 정확히 매핑할 것.

## 툴 스키마 작성 팁 (소형 모델일수록 중요)

- `name`, `description`, `parameters`의 description을 **명확하고 구체적으로**. 모델이 description만 보고 툴을 고르고 인자를 추출한다.
- **툴 개수는 20개 이하로 제한** (QwenCloud 권장). 그 이상이면 라우팅 계층(시맨틱 검색, 키워드 필터, 경량 LLM 라우터)으로 사전 필터링.
- 선택 파라미터는 `{"type": ["string", "null"]}`로 nullable 처리 (strict schema 스타일).

## 프로덕션 에이전트 수비책

QwenCloud 공식 가이드 요약:

1. **프로토콜 준수를 100% 가정하지 마라** — malformed 툴콜에 대한 자체 파싱/재시도 로직 필수. 프로토콜이 깨질 때를 대비한 카운터미저(에러 메시지를 모델에 되돌려 수정 유도) 구현.
2. **최소 권한 원칙**: 기본은 read-only 툴. 코드 실행, 파일 삭제, 금전 이체 같은 위험 연산은 모델에 직접 노출 금지.
3. **쓰기 연산엔 휴먼 확인**: 이메일 발송, 데이터 수정 등 되돌릴 수 없는 동작은 사용자 확인 단계 삽입.
4. **스텝별 타임아웃 + 폴백 응답**: "지금 정보를 가져올 수 없습니다. 나중에 다시 시도해 주세요" 같은 명확한 실패 응답.
5. **평가셋 구축**: 실제 시나리오를 반영한 데이터셋으로 툴 선택 정확도, 인자 추출 정확도, 엔드투엔드 성공률을 추적. 툴 선택이 틀리면 모델을 올리기 전에 **description/시스템 프롬프트를 먼저 다듬을 것.**
6. **병렬 툴콜**: 독립적인 다중 호출이 필요한 서비스는 `parallel_tool_calls=true` 활용.

## 최종 응답 생성 시 함정

툴 결과를 요약해서 자연어로 답할 때는 `tool_choice` 파라미터를 **제거**해야 한다. 남아 있으면 API가 자연어 대신 툴콜 정보를 다시 반환한다.

## 스트리밍

- `stream=True` 사용 시 툴콜 인자 델타를 조인(aggregation)한 뒤 JSON 파싱할 것. 청크 단위 파싱은 실패한다.

## 출처

- https://qwen.readthedocs.io/en/latest/framework/function_call.html (Qwen 공식 — Hermes 권장, ReAct 경고, vLLM 예제)
- https://docs.qwencloud.com/developer-guides/tool-calling/function-calling (tool_choice/thinking 제약, 툴 20개 제한, 수비책)
- https://docs.vllm.ai/en/stable/features/tool_calling/ (strict schema, 파서)
- https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix (qwen3_xml 안정성 검증)
