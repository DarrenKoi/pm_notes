---
tags: [langchain, openai-api, architecture, basics]
level: beginner
last_updated: 2026-07-06
reviewed_on: 2026-10-04
review_status: partial
document_type: learning_note
---

# 01. OpenAI API 및 LangChain 구조 이해

> LLM 호출이 실제로 어떻게 일어나는지(원시 API), 그리고 LangChain이 그 위에 어떤 추상화를 얹어 개발을 단순화하는지를 이해한다.


> [!info] 적용 조건과 실행 순서
> 2026-10-04 개별 검토. [공통 적용 조건](./verified-conditions.md)의 판본·설정·검증 경계를 먼저 확인한다. 같은 문서의 코드 조각은 위에서 아래로 이어 실행하며 개념 조각은 별도로 표시한다. 이전 문서의 vs/chunks/embeddings 등은 관련 절의 선행 예제가 필요하다. 공개·사내 API/실제 데이터·운영 실행은 미확인이며 예제 출력은 보장이 아니다.

## 왜 필요한가? (Why)

- LLM 앱을 만들 때 매번 원시 HTTP 요청을 다루면 **프롬프트 관리, 모델 교체, 스트리밍, 도구 호출**을 직접 구현해야 한다.
- LangChain은 이 반복 작업을 **표준 인터페이스(Runnable)**로 묶어, 모델·프롬프트·검색기·파서를 레고처럼 조립한다.
- 사내 환경에서는 모델이 자주 바뀐다(Kimi-K2.5 → Qwen 등). 표준 입력/출력은 교체 비용을 줄인다. 인증·tool/JSON/vision 지원·context·stream·토큰 계산의 호환성을 따로 확인해야 한다.

## 핵심 개념 (What)

### 1) 원시 LLM 호출은 무엇인가
LLM API의 본질은 "메시지 리스트 → 다음 메시지" 함수다. 메시지는 `role`(system/user/assistant/tool)과 `content`로 구성된다.

```python
import os
# ── 공개 OpenAI SDK (원시 호출) ──
from openai import OpenAI
client = OpenAI()  # OPENAI_API_KEY 사용
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "너는 반도체 공정 도우미다."},
        {"role": "user", "content": "포토 공정 한 줄 요약해줘."},
    ],
)
print(resp.choices[0].message.content)

# ── 사내 OpenAI 호환 엔드포인트 (원시 호출) ──
from openai import OpenAI
client = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"])
resp = client.chat.completions.create(
    model=os.environ["LLM_MODEL"],
    messages=[
        {"role": "system", "content": "너는 반도체 공정 도우미다."},
        {"role": "user", "content": "포토 공정 한 줄 요약해줘."},
    ],
)
print(resp.choices[0].message.content)
```

> 핵심: 사내 엔드포인트도 **동일한 OpenAI SDK**를 쓰되 접속 설정을 바꾼다. 같은 SDK 사용과 모든 OpenAI 기능 호환은 다르다.

### 2) LangChain 패키지 구조
LangChain은 하나의 거대한 라이브러리가 아니라 **여러 패키지**로 쪼개져 있다.

| 패키지 | 역할 |
|--------|------|
| `langchain-core` | 추상화의 뿌리 — Runnable, Message, Prompt, Output Parser 인터페이스 |
| `langchain-openai` | OpenAI(호환) 모델 구현체 — `ChatOpenAI`, `OpenAIEmbeddings` |
| `langchain` | v1 Agent 구성(create_agent 등); legacy retriever/indexing은 langchain-classic |
| `langchain-community` | 서드파티 로더/벡터스토어 (PDF, 웹 로더, FAISS 래퍼 등) |
| `langgraph` | 상태 머신 기반 워크플로우/에이전트 (2번 모듈) |

### 3) 3대 추상화
- **Chat Model** (`ChatOpenAI`): 메시지 리스트를 받아 `AIMessage`를 반환.
- **Prompt Template**: 변수를 끼워 넣어 메시지를 생성.
- **Output Parser**: 모델 출력(문자열)을 원하는 타입(str, JSON, Pydantic)으로 변환.

이 셋은 모두 **Runnable**이라 `|`(파이프)로 연결된다 → [02. LCEL](./02-lcel.md).

## 어떻게 사용하는가? (How)

### Chat Model 기본 사용
```python
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# 공개: llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# 사내:
llm = ChatOpenAI(
    model=os.environ["LLM_MODEL"],
    base_url=os.environ["LLM_BASE_URL"],
    api_key=os.environ["LLM_API_KEY"],
    temperature=0,
)

messages = [
    SystemMessage("너는 간결하게 답하는 도우미다."),
    HumanMessage("LangChain이 뭐야?"),
]
ai = llm.invoke(messages)
print(ai.content)          # 문자열/내용 블록일 수 있음; 업무 스키마 별도 검증
print(ai.usage_metadata)   # 토큰 사용량 (응답이 제공할 때; 미관측은 None)
```

### Prompt Template
```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 {domain} 전문가다. 한국어로 답해."),
    ("human", "{question}"),
])

# 템플릿도 Runnable → invoke로 실제 메시지 생성
msgs = prompt.invoke({"domain": "반도체", "question": "EUV가 뭐야?"})
print(msgs.messages)
```

### 추가 패턴: `init_chat_model`로 모델 provider 추상화
LangChain v1은 provider 문자열로 모델을 초기화하는 헬퍼를 제공한다. 사내에서는 여전히 `ChatOpenAI` + `base_url`이 지원 기능을 직접 확인하기 쉽지만, 코드 이식성 측면에서 알아둘 만하다.

```python
from langchain.chat_models import init_chat_model
# 공개 예: model = init_chat_model("gpt-4o-mini", model_provider="openai")
# 사내: base_url이 필요하므로 configurable로 감싸거나 ChatOpenAI 직접 사용 권장
```

### 스트리밍 (청크 단위 출력)
```python
for chunk in llm.stream("LangChain을 3문장으로 설명해줘."):
    print(chunk.content, end="", flush=True)
```

## 관련 문서
- [02. LCEL 실습](./02-lcel.md) — 여기서 만든 Model/Prompt를 파이프로 연결
- [03. Chain · Agent · Tool](./03-chain-agent-tool.md)

## 참고 자료 (References)
- Providers & Models: https://docs.langchain.com/oss/python/langchain/models
- Chat models 개념: https://docs.langchain.com/oss/python/langchain/models
- OpenAI 호환 엔드포인트 연결: `ChatOpenAI(base_url=..., api_key=..., model=...)`
