---
tags: [llm, harness-engineering, evaluation, agents, mcp, openai]
level: intermediate
last_updated: 2026-04-14
status: complete
---

# Harness Engineering for LLM

> LLM 애플리케이션을 “실험 가능하고(재현성) 안전하며(거버넌스) 운영 가능한(관측/배포)” 상태로 만드는 엔지니어링 방법론.

## 왜 필요한가? (Why)
- 프롬프트 품질만으로는 프로덕션 안정성을 보장할 수 없다. 같은 입력에서도 모델/툴/외부 API 상태에 따라 결과가 달라진다.
- 도구 호출(tool use), RAG, 멀티에이전트가 붙으면 실패 지점이 급격히 늘어난다.
- 따라서 실행 하네스(runtime harness) + 평가 하네스(eval harness) + 관측 하네스(observability harness)를 분리/연동해야 한다.

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

### 3) OpenAI가 강조하는 하네스 관점 요약
OpenAI 문서/가이드에서 일관되게 보이는 포인트:
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

### D. 운영 지표(필수)
- Quality: task success rate, groundedness, format-valid rate
- Reliability: tool success rate, timeout rate, retry rate
- Efficiency: latency p50/p95, token usage, cost per task
- Safety: policy violation rate, blocked action count, manual escalation rate

## 참고 자료 (References)

### OpenAI (공식)
- Building effective agents: https://developers.openai.com/resources/guides-and-library/building-agents/
- Evaluation best practices: https://developers.openai.com/api/docs/guides/evaluation-best-practices
- Evaluation getting started: https://platform.openai.com/docs/guides/evaluation-getting-started
- Trace grading: https://developers.openai.com/api/docs/guides/trace-grading
- Function calling lifecycle: https://platform.openai.com/docs/guides/function-calling/lifecycle
- Tools (Web search): https://developers.openai.com/api/docs/guides/tools-web-search
- Tools (File search): https://platform.openai.com/docs/guides/tools-file-search/
- Connectors & MCP: https://developers.openai.com/api/docs/guides/tools-connectors-mcp
- How OpenAI uses Codex (OpenAI 작성): https://cdn.openai.com/pdf/6a2631dc-783e-479b-b1a4-af0cfbd38630/how-openai-uses-codex.pdf
- o3-mini system card (tool harness 언급): https://cdn.openai.com/o3-mini-system-card.pdf

### 표준/생태계
- Model Context Protocol spec: https://modelcontextprotocol.io/specification/2025-06-18
- OpenTelemetry GenAI semantic conventions: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/
- NIST AI RMF GenAI Profile: https://doi.org/10.6028/NIST.AI.600-1
- LangSmith evaluation concepts: https://docs.langchain.com/langsmith/evaluation-concepts
- Promptfoo CI/CD eval: https://www.promptfoo.dev/docs/integrations/ci-cd/

## 관련 문서
- [MCP 기본](./mcp-basics.md)
- [AI/DT 루트](../README.md)
