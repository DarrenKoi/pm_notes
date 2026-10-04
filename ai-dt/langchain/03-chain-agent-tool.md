---
tags: [langchain, agent, tool, tool-calling, structured-output]
level: intermediate
last_updated: 2026-07-06
reviewed_on: 2026-10-04
review_status: partial
document_type: learning_note
---

# 03. Chain, Agent, Tool 정의 및 사용

> Chain(정해진 흐름), Tool(모델이 부를 수 있는 함수), Agent(도구를 스스로 골라 부르는 루프)의 차이를 이해하고 tool calling 방식으로 구현한다.


> [!info] 적용 조건과 실행 순서
> 2026-10-04 개별 검토. [공통 적용 조건](./verified-conditions.md)의 판본·설정·검증 경계를 먼저 확인한다. 같은 문서의 코드 조각은 위에서 아래로 이어 실행하며 개념 조각은 별도로 표시한다. 이전 문서의 vs/chunks/embeddings 등은 관련 절의 선행 예제가 필요하다. 공개·사내 API/실제 데이터·운영 실행은 미확인이며 예제 출력은 보장이 아니다.

## 왜 필요한가? (Why)

- **Chain**은 흐름이 고정된 파이프라인이다. 하지만 "질문에 따라 계산기를 쓸지, 검색을 할지"처럼 **분기가 데이터에 달린** 문제는 RunnableBranch 등 결정적 분기도 가능하다. 동적 LLM 도구 선택이 필요한지 별도로 판단한다.
- **Tool**은 LLM에게 "이런 함수를 쓸 수 있어"라고 알려주고, **Agent**는 LLM이 스스로 어떤 Tool을 언제 부를지 결정하는 루프다.
- 사내에서 "장비 상태 조회 API", "사양 DB 검색" 같은 기능을 LLM이 필요할 때 호출하게 하려면 Tool/Agent가 핵심이다.

## 핵심 개념 (What)

### Chain vs Agent
| | Chain | Agent |
|---|-------|-------|
| 흐름 | 고정 (설계자가 결정) | 동적 (LLM이 결정) |
| 예측성 | 높음 | 낮음(유연) |
| 비용 | 단계·모델·토큰 수에 달림 | 반복 도구/model 호출로 증가 가능 |
| 언제 | 단계가 정해진 작업 | 도구 선택이 입력마다 다른 작업 |

> 원칙: **가능하면 Chain, 꼭 필요할 때만 Agent.** 이 노트는 단계 제어가 중요한 경우 [LangGraph](./05-langgraph-overview-state-machine.md)로 흐름을 **제어된 상태 머신**으로 만드는 쪽을 선택하는 학습 기준을 제안한다. 실제 업계 선호를 검증한 통계는 아니다.

### Tool calling의 원리
현대 LLM은 "함수 스키마(이름/설명/파라미터)"를 받으면, 직접 실행하는 대신 **"이 함수를 이 인자로 부르라"는 JSON**(`tool_calls`)을 반환한다. 실행은 우리 코드가 한다. LangChain은 `@tool` + `bind_tools`로 이 과정을 표준화한다.

## 어떻게 사용하는가? (How)

### 1) Tool 정의
```python
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """두 정수 a와 b를 곱한다."""   # ← docstring이 LLM에게 전달되는 설명
    return a * b

@tool
def get_equipment_status(eqp_id: str) -> str:
    """설비 ID로 현재 상태(RUN/IDLE/DOWN)를 조회한다."""
    # 실제로는 사내 API 호출 → 04번 문서 참고
    return f"{eqp_id}: RUN (가상 fixture; 실제 설비 조회 아님)"

print(multiply.name, multiply.args)   # 스키마 확인
```

### 2) 모델에 Tool 바인딩 (`bind_tools`)
```python
import os
from langchain_openai import ChatOpenAI

# 공개: llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# 사내:
llm = ChatOpenAI(model=os.environ["LLM_MODEL"], base_url=os.environ["LLM_BASE_URL"],
                 api_key=os.environ["LLM_API_KEY"], temperature=0)

llm_with_tools = llm.bind_tools([multiply, get_equipment_status])

ai = llm_with_tools.invoke("EQP-102 상태 알려줘")
print(ai.tool_calls)   # [{'name': 'get_equipment_status', 'args': {'eqp_id': 'EQP-102'}, 'id': ...}]
```

### 3) 수동 Agent 루프 (원리 이해용)
Agent는 결국 "모델 호출 → tool_calls 실행 → 결과를 다시 모델에 전달" 루프다.
```python
from langchain_core.messages import HumanMessage, ToolMessage

tools = {"multiply": multiply, "get_equipment_status": get_equipment_status}
messages = [HumanMessage("EQP-102 상태 확인하고, 12 곱하기 7도 알려줘")]

for _ in range(8):  # 최대8 model 호출; 이 상한은 timeout/권한 검사를 대체하지 않음
    ai = llm_with_tools.invoke(messages)
    messages.append(ai)
    if not ai.tool_calls:
        break                        # 도구 호출이 없으면 최종 답변
    for call in ai.tool_calls:
        if call["name"] not in tools:
            raise ValueError("등록하지 않은 tool 호출")
        result = tools[call["name"]].invoke(call["args"])
        messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

else:
    raise RuntimeError("도구 호출 상한 도달; 완료로 취급하지 않음")
print(ai.content)
```

### 4) v1 API: `create_agent` (LangChain)
위 루프를 직접 짤 필요 없이, LangChain의 `create_agent`를 쓴다. `create_agent`는 모델·도구·시스템 프롬프트를 조합하는 고수준 Agent harness다. create_agent도 LangGraph 기반으로 checkpointer/middleware/HITL을 지원한다. 노드/엣지를 직접 설계하려면 [08번](./08-langgraph-agent-scenario.md)의 LangGraph로 확장한다.
```python
from langchain.agents import create_agent

agent = create_agent(
    model=llm,
    tools=[multiply, get_equipment_status],
    system_prompt="너는 도구를 사용해 사실을 확인한 뒤 한국어로 답한다.",
)
out = agent.invoke({"messages": [{"role": "user", "content": "EQP-102 상태와 12*7 알려줘"}]})
print(out["messages"][-1].content)
```

### 5) 구조화 출력 (`with_structured_output`)
지원 방식은 provider-native JSON schema/tool calling/JSON mode 등에 따라 달라진다. Pydantic은 구조를 검사하지만 파싱/거부/검증 실패와 잘못된 사실은 여전히 가능하다. 호환 gateway 예시는 method="function_calling"을 명시하며 해당 기능 지원은 미확인이다.
```python
from pydantic import BaseModel, Field

class DefectReport(BaseModel):
    """웨이퍼 결함 리포트 요약."""
    defect_type: str = Field(description="결함 유형")
    severity: int = Field(ge=1, le=5, description="심각도 1-5")
    action: str = Field(description="권장 조치")

structured_llm = llm.with_structured_output(DefectReport, method="function_calling")
report = structured_llm.invoke("스크래치가 3개 발견됨, 재작업 필요, 심각도 중간")
print(report.defect_type, report.severity, report.action)   # Pydantic 구조 검증 성공 시; 사실성/실업무 심각도는 별도
```

## 관련 문서
- [02. LCEL 실습](./02-lcel.md) — Chain 구성 기초
- [04. 외부 API 연동 Agent](./04-external-api-agent.md) — Tool을 실 API로 확장
- [08. 시나리오 기반 Agent 구축](./08-langgraph-agent-scenario.md) — LangGraph로 견고한 Agent

## 참고 자료 (References)
- Tool calling: https://docs.langchain.com/oss/python/langchain/tools
- `bind_tools`, `with_structured_output` (ChatOpenAI 통합 문서)
- Agents / `create_agent`: https://docs.langchain.com/oss/python/langchain/agents
