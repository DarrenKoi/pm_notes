---
tags: [llm, harness-engineering, evaluation, agents, mcp, openai]
level: intermediate
last_updated: 2026-04-14
reviewed_on: 2026-10-04
review_status: partial
type: learning
category_major: "AI·DT"
category_middle: "에이전트 개발"
category_minor: "하네스 개념"
note_kind: "학습"
classified_on: "2026-10-05"
---

# Harness Engineering for LLM

> LLM 애플리케이션을 “실험 가능하고(재현성) 안전하며(거버넌스) 운영 가능한(관측/배포)” 상태로 만드는 엔지니어링 방법론.

> [!info] 검토 범위 · 2026-10-04
> 프로토콜 사양과 SDK 버전은 별개다. [버전·실행 조건](./version-and-execution-notes.md)과 [정리 기록](./organization-log.md)을 먼저 확인한다. 원래 작성일은 보존했으며 실제 API·원격 서버 실행은 미검증이다.

## 왜 필요한가? (Why)
- 프롬프트 품질만으로는 프로덕션 안정성을 보장할 수 없다. 같은 입력에서도 모델/툴/외부 API 상태에 따라 결과가 달라진다.
- 도구 호출(tool use), RAG, 멀티에이전트가 붙으면 실패 지점이 급격히 늘어난다.
- 실행·평가·관측을 서로 다른 책임으로 설계한다. 작은 앱에서는 같은 프로그램에 함께 구현할 수도 있으며 별도 서비스 세 개를 만드는 것이 필수는 아니다.

## 핵심 개념 (What)

### 1) Harness Engineering 정의
LLM 시스템에서 하네스는 다음을 표준화한다.
- **입출력 계약(Contract)**: 프롬프트 템플릿, structured output schema, tool schema
- **실행 오케스트레이션**: 모델 호출, tool routing, retries, timeout, fallback
- **평가 루프**: 오프라인 회귀 테스트 + 온라인 품질 모니터링
- **관측/거버넌스**: trace, token/cost, 안전 정책, human approval

### 2) 구성 요소 체크리스트
- Prompt registry (버전/실험 태그)
- Model gateway (모델 라우팅/폴백)
- Tool adapter (사내 API, DB, SaaS, MCP)
- State store (세션/작업 상태)
- Eval dataset + grader (rule-based + LLM judge + human audit)
- Observability (OpenTelemetry 기반 trace/metrics/logs)
- Policy engine (PII/보안/승인 정책)
- CI/CD gate (평가 기준 미달 시 배포 차단)

### 3) 공식 자료에서 읽어낸 설계 관점
아래는 이 학습 노트의 설계 해석이다. OpenAI가 동일한 체크리스트나 복잡도별 평가 비율을 표준으로 정했다는 의미는 아니다:
- 에이전트 복잡도(single call → workflow → multi-agent)가 올라갈수록 **평가 자동화** 비중을 높여야 한다.
- 툴/함수 호출은 자연어 프롬프트가 아니라 **명시적 schema 기반 계약**으로 운영해야 안정적이다.
- 운영 품질은 정답률만이 아니라 **latency/cost/safety**를 함께 봐야 한다.
- 내부 엔지니어링 사례(Codex 활용)에서도 코드 이해/리팩터링/테스트 자동화처럼 “작업을 반복 가능한 파이프라인으로 바꾸는 것”이 핵심이다.

## 어떻게 사용하는가? (How)

### A. 구축 방법론 (권장 순서)
1. **Task contract 고정**
   - 입력 타입, 출력 포맷(JSON schema), 실패 기준 정의
2. **MVP 런타임 하네스**
   - 모델 1개 + 툴 1~2개 + timeout/retry/fallback 적용
3. **평가 하네스 연결**
   - 대표 시나리오 eval dataset 구성
   - 변경마다 자동 평가(회귀 테스트)
4. **관측 하네스 연결**
   - trace ID로 model call/tool call 연결
   - token/cost/error rate 대시보드화
5. **거버넌스/승인 추가**
   - 고위험 액션은 human-in-the-loop
   - 외부 connector/MCP는 allowlist 우선

### B. LLM API 연결 패턴

#### 패턴 1) Direct Tool Calling
- 앱이 모델에 tools schema 전달
- 모델이 tool call 생성
- 앱이 실제 도구 실행 후 결과를 다시 모델에 전달
- 장점: 구현 단순 / 단점: 도구 증가 시 오케스트레이션 복잡

#### 패턴 2) Hosted Tool + Custom Tool Hybrid
- 웹 검색/파일 검색 같은 hosted tool + 사내 API custom tool 혼합
- 장점: 빠른 개발 / 단점: 도메인 정책 통합 필요

#### 패턴 3) MCP 기반 Tool 표준화
- 도구를 MCP 서버로 분리해 모델/프레임워크 독립성 확보
- 장점: 재사용성/확장성 / 단점: 인증/권한/신뢰 체계 설계 필요

### C. 최소 Python 예시 (OpenAI Responses API + tool schema)

이 코드는 tool schema를 보내는 **첫 요청까지만** 구현한다. 도구가 실행되거나 최종 날씨 답변이 완성되는 예제가 아니다. 모델 ID `gpt-4.1`은 기존 예제를 유지했으며 실제 API/계정 접근은 검증하지 않았다. 환경 변수 `OPENAI_API_KEY`가 필요하다. 로컬 SDK `openai==2.30.0`의 함수 schema 타입과 공식 Responses 가이드에 대조했다.
```python
from openai import OpenAI

client = OpenAI()

tools = [{
    "type": "function",
    "name": "get_weather",
    "description": "Get weather by city",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    },
    "strict": True,
}]

resp = client.responses.create(
    model="gpt-4.1",
    input="서울 날씨 알려줘",
    tools=tools,
)

# 1) tool call 추출
# 2) 실제 함수 실행
# 3) 함수 결과를 다시 responses.create에 전달
# 4) 최종 응답 + trace/metrics 저장
```

### D. 운영 지표 선택
- Quality: task success rate, groundedness, format-valid rate
- Reliability: tool success rate, timeout rate, retry rate
- Efficiency: latency p50/p95, token usage, cost per task
- Safety: policy violation rate, blocked action count, manual escalation rate

## 도구 결과를 돌려주는 계약

모델의 `function_call`에서 `arguments`를 JSON으로 읽고 이름·입력·권한을 확인한 뒤 앱이 함수를 실행한다. 결과를 `function_call_output`으로 보내며 **`call_id`**가 원래 호출과 일치해야 한다. 응답을 이어 갈 때 이전 output 또는 지원되는 conversation 연결 방식을 보존한다. 한 응답에 여러 호출이 올 수 있고 후속 응답에서도 추가 호출이 올 수 있으므로 한 번의 if문으로 완료를 가정하지 않는다. 호출 횟수·시간 제한을 정하고 부작용 있는 작업의 재시도는 중복 실행 조건을 별도로 다룬다. 근거: [공식 function calling 가이드](https://developers.openai.com/api/docs/guides/function-calling), 확인 2026-10-04.

`strict: true`는 지원되는 schema의 형식 준수에 관한 설정이다. 현재 예제의 `required`와 `additionalProperties: false`는 그 조건을 맞추지만, 내용의 사실성·인가·도구 실행 성공을 보장하지 않는다. 운영 지표는 과제별 분모·성공 기준을 정의하여 선택한다. 작은 시스템에 모든 체크리스트 구성 요소를 도입해야 하는 것은 아니다.

## 참고 자료 (References)

### OpenAI (공식)
- 기존 Building agents 링크 (2026-10-04 열람 실패·미확인): https://developers.openai.com/resources/guides-and-library/building-agents/
- Evaluation best practices: https://developers.openai.com/api/docs/guides/evaluation-best-practices
- Evaluation getting started: https://developers.openai.com/api/docs/guides/evals
- Trace grading: https://developers.openai.com/api/docs/guides/trace-grading
- Function calling lifecycle: https://developers.openai.com/api/docs/guides/function-calling
- Tools (Web search): https://developers.openai.com/api/docs/guides/tools-web-search
- Tools (File search): https://platform.openai.com/docs/guides/tools-file-search/
- Connectors & MCP: https://developers.openai.com/api/docs/guides/tools-connectors-mcp
- How OpenAI uses Codex (OpenAI 작성): https://cdn.openai.com/pdf/6a2631dc-783e-479b-b1a4-af0cfbd38630/how-openai-uses-codex.pdf
- o3-mini system card (tool harness 언급): https://cdn.openai.com/o3-mini-system-card.pdf

Codex 활용 PDF와 o3-mini system card는 원래 참고 목록을 보존한 과거 자료다. 이 검토에서 두 PDF의 개별 주장을 재검증하지 않았으며 현재 제품 동작의 근거로 사용하지 않는다.

### 표준/생태계
- Model Context Protocol spec: https://modelcontextprotocol.io/specification/2026-07-28
- OpenTelemetry GenAI semantic conventions: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/
- NIST AI RMF GenAI Profile: https://doi.org/10.6028/NIST.AI.600-1
- LangSmith evaluation concepts: https://docs.langchain.com/langsmith/evaluation-concepts
- Promptfoo CI/CD eval: https://www.promptfoo.dev/docs/integrations/ci-cd/

## 관련 문서
- [MCP 기본](./mcp-basics.md)
- [AI/DT 루트](../README.md)
