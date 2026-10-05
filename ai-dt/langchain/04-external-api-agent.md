---
tags: [langchain, agent, tool, external-api, integration]
level: intermediate
last_updated: 2026-07-06
reviewed_on: 2026-10-04
review_status: partial
document_type: learning_note
category_major: "AI·DT"
category_middle: "에이전트 개발"
category_minor: "LangChain 기초"
note_kind: "학습"
classified_on: "2026-10-05"
---

# 04. 외부 API 연동 기능 수행 Agent 설계

> LLM이 실제 외부/사내 REST API를 호출해 실시간 데이터를 가져오게 하는 Tool을 설계하고, 인증·에러·재시도·타임아웃 등 운영 문제를 다룬다.


> [!info] 적용 조건과 실행 순서
> 2026-10-04 개별 검토. [공통 적용 조건](./verified-conditions.md)의 판본·설정·검증 경계를 먼저 확인한다. 같은 문서의 코드 조각은 위에서 아래로 이어 실행하며 개념 조각은 별도로 표시한다. 이전 문서의 vs/chunks/embeddings 등은 관련 절의 선행 예제가 필요하다. 공개·사내 API/실제 데이터·운영 실행은 미확인이며 예제 출력은 보장이 아니다.

## 왜 필요한가? (Why)

- LLM은 학습 시점 이후의 사실이나 사내 시스템 데이터를 모른다. **실시간 조회 API**를 Tool로 붙이면 "지금 EQP-102 상태"처럼 조회 응답의 시각/출처를 확인할 수 있다. API 지연·캐시·오류이면 최신 사실을 보장하지 않는다.
- 실 API 연동은 데모 코드와 달리 **실패**한다: 타임아웃, 401, rate limit, 잘못된 인자. 이걸 Tool 안에서 견고하게 처리해야 Agent가 무너지지 않는다.
- 사내 활용: 설비 MES 조회, 사양(Spec) DB 검색, 레시피 파라미터 lookup 등.

## 핵심 개념 (What)

### 좋은 API Tool의 조건
1. **명확한 docstring** — LLM이 언제 이 도구를 쓸지 판단하는 근거.
2. **좁은 입력 스키마** — Pydantic으로 인자 검증 (LLM이 이상한 값을 넣는 걸 방지).
3. **실패를 값으로 반환** — 예외를 던져 Agent를 죽이기보다, "에러: ..." 문자열을 돌려주면 모델에 실패를 전달할 수 있다. 재시도/우회·agent 생존을 보장하지 않으며 인증/쓰기 오류에는 별도 정책이 필요하다.
4. **부작용(쓰기) 도구는 승인 게이트** — 삭제/변경 API는 human-in-the-loop([08번](./08-langgraph-agent-scenario.md)).

## 어떻게 사용하는가? (How)

### 1) 읽기 전용 API Tool (인자 검증 포함)

api.example.com은 서비스가 없는 placeholder다. MES_BASE_URL/MES_TOKEN은 승인된 실제 설정에서 읽는다. 아래 GET에만 제한 재시도를 적용하며 쓰기의 중복 실행/Retry-After/전체 deadline은 구현하지 않는다.
```python
import httpx
import math
from pydantic import BaseModel, Field
from langchain_core.tools import tool

class WeatherArgs(BaseModel):
    city: str = Field(min_length=1, max_length=80, description="도시 이름, 예: Seoul")

@tool(args_schema=WeatherArgs)
def get_weather(city: str) -> str:
    """도시의 현재 날씨를 조회한다."""
    try:
        r = httpx.get("https://api.example.com/weather",
                      params={"q": city}, timeout=5.0)
        r.raise_for_status()
        d = r.json()
        if not isinstance(d, dict) or not isinstance(d.get("temp"), (int, float)) or isinstance(d.get("temp"), bool) or not math.isfinite(d["temp"]) or not isinstance(d.get("desc"), str):
            raise ValueError("날씨 응답 schema 불일치")
        return f"{city}: {d['temp']}°C, {d['desc']}"
    except httpx.TimeoutException:
        return "에러: 날씨 API 응답 시간 초과. 잠시 후 재시도 권장."
    except httpx.HTTPStatusError as e:
        return f"에러: API {e.response.status_code} 응답."
    except httpx.RequestError:
        return "에러: 날씨 API 연결 실패."
    except ValueError:
        return "에러: 날씨 API JSON/schema 불일치."
```

### 2) 사내 인증이 필요한 API Tool
```python
import os
import os, httpx
from urllib.parse import quote
from langchain_core.tools import tool

INTERNAL_BASE = os.environ["MES_BASE_URL"]
TOKEN = os.environ["MES_TOKEN"]     # 비밀은 코드가 아니라 환경변수/시크릿 매니저에서

@tool
def get_equipment_status(eqp_id: str) -> str:
    """설비 ID로 현재 상태(RUN/IDLE/DOWN)와 최근 알람을 조회한다."""
    try:
        r = httpx.get(f"{INTERNAL_BASE}/equipment/{quote(eqp_id, safe='')}",
                      headers={"Authorization": f"Bearer {TOKEN}"}, timeout=5.0)
        r.raise_for_status()
        d = r.json()
        if not isinstance(d, dict) or d.get("state") not in {"RUN", "IDLE", "DOWN"}:
            raise ValueError("설비 상태 미확인/schema 불일치")
        return f"{eqp_id}: {d['state']}, 최근알람={d.get('last_alarm','미관측')}"
    except Exception as e:
        return f"에러: {eqp_id} 조회 실패 ({type(e).__name__})."
```

### 3) 재시도 데코레이터 (일시적 오류 대비)
```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

def retryable(exc: BaseException) -> bool:
    return isinstance(exc, httpx.TransportError) or (
        isinstance(exc, httpx.HTTPStatusError)
        and exc.response.status_code in {429, 500, 502, 503, 504})

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8),
       retry=retry_if_exception(retryable), reraise=True)
def _call_spec_api(part_no: str) -> dict:
    r = httpx.get(f"{INTERNAL_BASE}/spec/{quote(part_no, safe='')}",
                  headers={"Authorization": f"Bearer {TOKEN}"}, timeout=5.0)
    r.raise_for_status()
    return r.json()

@tool
def get_spec(part_no: str) -> str:
    """부품 번호로 사양(치수/공차)을 조회한다."""
    try:
        d = _call_spec_api(part_no)
        return f"{part_no}: {d['dim']} ±{d['tol']}"
    except Exception:
        return f"에러: {part_no} 사양 조회 실패(재시도는 일시적 transport/지정 HTTP 상태에만 적용)."
```

### 4) Agent에 연결
```python
import os
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

# 사내:
llm = ChatOpenAI(model=os.environ["LLM_MODEL"], base_url=os.environ["LLM_BASE_URL"],
                 api_key=os.environ["LLM_API_KEY"], temperature=0)

agent = create_agent(
    model=llm,
    tools=[get_weather, get_equipment_status, get_spec],
    system_prompt="필요할 때만 도구를 호출하고, 도구 오류는 사용자에게 명확히 설명한다.",
)
out = agent.invoke({"messages": [
    {"role": "user", "content": "EQP-102 상태 확인하고, 문제 있으면 부품 PN-77 사양도 알려줘"}
]})
print(out["messages"][-1].content)
```

### 5) 추가 패턴: MCP로 도구를 표준화 (참고)
개별 API를 매번 `@tool`로 감싸는 대신, **MCP(Model Context Protocol)** 서버로 감싸면 여러 에이전트/앱이 같은 도구를 재사용한다. LangChain은 `langchain-mcp-adapters`로 MCP 도구를 LangChain Tool로 변환할 수 있다. (이 폴더 범위 밖 — 개념만 인지)

## 설계 체크리스트
- [ ] docstring이 "언제 쓰는지"를 명확히 말하는가
- [ ] 입력 스키마(Pydantic)로 인자를 검증하는가
- [ ] 타임아웃을 설정했는가 (기본 무한 대기 금지)
- [ ] 예외를 "에러 문자열"로 변환해 모델에 실패를 전달하고 종료/재시도 정책을 구분하는가
- [ ] 비밀 값은 환경변수/시크릿에서 오는가
- [ ] 쓰기 API는 승인 게이트가 있는가

## 관련 문서
- [03. Chain · Agent · Tool](./03-chain-agent-tool.md) — Tool/Agent 기본
- [08. 시나리오 기반 Agent 구축](./08-langgraph-agent-scenario.md) — 승인 게이트(HITL)

## 참고 자료 (References)
- Tool calling 개념: https://docs.langchain.com/oss/python/langchain/tools
- httpx: https://www.python-httpx.org/ · tenacity: https://tenacity.readthedocs.io/
