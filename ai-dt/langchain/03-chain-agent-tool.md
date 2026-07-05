---
tags: [langchain, agent, tool, tool-calling, structured-output]
level: intermediate
last_updated: 2026-07-06
---

# 03. Chain, Agent, Tool 정의 및 사용

> Chain(정해진 흐름), Tool(모델이 부를 수 있는 함수), Agent(도구를 스스로 골라 부르는 루프)의 차이를 이해하고 최신 tool calling 방식으로 구현한다.

## 왜 필요한가? (Why)

- **Chain**은 흐름이 고정된 파이프라인이다. 하지만 "질문에 따라 계산기를 쓸지, 검색을 할지"처럼 **분기가 데이터에 달린** 문제는 Chain만으론 부족하다.
- **Tool**은 LLM에게 "이런 함수를 쓸 수 있어"라고 알려주고, **Agent**는 LLM이 스스로 어떤 Tool을 언제 부를지 결정하는 루프다.
- 사내에서 "장비 상태 조회 API", "사양 DB 검색" 같은 기능을 LLM이 필요할 때 호출하게 하려면 Tool/Agent가 핵심이다.

## 핵심 개념 (What)

### Chain vs Agent
| | Chain | Agent |
|---|-------|-------|
| 흐름 | 고정 (설계자가 결정) | 동적 (LLM이 결정) |
| 예측성 | 높음 | 낮음(유연) |
| 비용 | 낮음 | 높음(반복 호출) |
| 언제 | 단계가 정해진 작업 | 도구 선택이 입력마다 다른 작업 |

> 원칙: **가능하면 Chain, 꼭 필요할 때만 Agent.** 최신 실무에서는 자유로운 Agent보다 [LangGraph](./05-langgraph-overview-state-machine.md)로 흐름을 **제어된 상태 머신**으로 만드는 쪽을 선호한다.

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
    return f"{eqp_id}: RUN"

print(multiply.name, multiply.args)   # 스키마 확인
```

### 2) 모델에 Tool 바인딩 (`bind_tools`)
```python
from langchain_openai import ChatOpenAI

# 공개: llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# 사내:
llm = ChatOpenAI(model="Kimi-K2.5", base_url="http://llm-gateway.internal/v1",
                 api_key="EMPTY", temperature=0)

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

while True:
    ai = llm_with_tools.invoke(messages)
    messages.append(ai)
    if not ai.tool_calls:
        break                        # 도구 호출이 없으면 최종 답변
    for call in ai.tool_calls:
        result = tools[call["name"]].invoke(call["args"])
        messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

print(ai.content)
```

### 4) 최신 권장: `create_agent` (LangChain)
위 루프를 직접 짤 필요 없이, LangChain의 `create_agent`를 쓴다. `create_agent`는 모델·도구·시스템 프롬프트를 조합하는 고수준 Agent harness다. 상태 머신, HITL, 장시간 실행 제어가 필요하면 [08번](./08-langgraph-agent-scenario.md)의 LangGraph로 확장한다.
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
Tool calling과 같은 메커니즘으로, 자유 텍스트 대신 **Pydantic 스키마에 맞는 객체**를 강제한다. 파싱 실패가 없어 실무에서 매우 유용하다.
```python
from pydantic import BaseModel, Field

class DefectReport(BaseModel):
    """웨이퍼 결함 리포트 요약."""
    defect_type: str = Field(description="결함 유형")
    severity: int = Field(description="심각도 1-5")
    action: str = Field(description="권장 조치")

structured_llm = llm.with_structured_output(DefectReport)
report = structured_llm.invoke("스크래치가 3개 발견됨, 재작업 필요, 심각도 중간")
print(report.defect_type, report.severity, report.action)   # 타입 보장된 객체
```

## 관련 문서
- [02. LCEL 실습](./02-lcel.md) — Chain 구성 기초
- [04. 외부 API 연동 Agent](./04-external-api-agent.md) — Tool을 실 API로 확장
- [08. 시나리오 기반 Agent 구축](./08-langgraph-agent-scenario.md) — LangGraph로 견고한 Agent

## 참고 자료 (References)
- Tool calling: https://docs.langchain.com/oss/python/concepts/tool-calling
- `bind_tools`, `with_structured_output` (ChatOpenAI 통합 문서)
- Agents / `create_agent`: https://docs.langchain.com/oss/python/langchain/agents
