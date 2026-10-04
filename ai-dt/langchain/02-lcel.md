---
tags: [langchain, lcel, runnable, chains]
level: beginner
last_updated: 2026-07-06
reviewed_on: 2026-10-04
review_status: partial
document_type: learning_note
---

# 02. LCEL (LangChain Expression Language) 실습

> `|` 파이프 하나로 모델·프롬프트·파서를 조립하는 LCEL의 Runnable 프로토콜을 이해하고, 병렬·조건·폴백까지 구성한다.


> [!info] 적용 조건과 실행 순서
> 2026-10-04 개별 검토. [공통 적용 조건](./verified-conditions.md)의 판본·설정·검증 경계를 먼저 확인한다. 같은 문서의 코드 조각은 위에서 아래로 이어 실행하며 개념 조각은 별도로 표시한다. 이전 문서의 vs/chunks/embeddings 등은 관련 절의 선행 예제가 필요하다. 공개·사내 API/실제 데이터·운영 실행은 미확인이며 예제 출력은 보장이 아니다.

## 왜 필요한가? (Why)

- 예전 LangChain은 `LLMChain`, `SequentialChain` 등 **클래스마다 사용법이 달랐다**. LCEL은 이를 **단일 인터페이스(Runnable)**로 통일했다.
- 공통 실행 인터페이스를 제공한다. 기본 async가 thread 실행일 수 있고, 중간 Runnable이 streaming을 막을 수 있다. 실제 병렬성·chunk·취소·rate limit은 구현과 provider에 달린다.
- 사내 배치 처리(수백 개 웨이퍼 로그 요약 등)에서 `batch`/`abatch`로 손쉽게 병렬화할 수 있다.

## 핵심 개념 (What)

### Runnable 프로토콜
LangChain의 모든 조립 가능한 객체는 `Runnable`이다. 공통 메서드:

| 메서드 | 설명 |
|--------|------|
| `invoke(input)` | 입력 1개 → 출력 1개 (동기) |
| `stream(input)` | 출력을 청크로 스트리밍 |
| `batch(inputs)` | 입력 리스트를 **병렬** 처리 |
| `ainvoke / astream / abatch` | 위의 async 버전 |

### 파이프 연산자 `|`
`chain = prompt | llm | parser` 는 "prompt의 출력을 llm에 넣고, llm의 출력을 parser에 넣는" 합성이다. 왼쪽 출력 타입 = 오른쪽 입력 타입이면 연결된다.

### 주요 조립용 Runnable
- `RunnablePassthrough` — 입력을 그대로 통과 (RAG에서 질문 유지에 필수)
- `RunnableParallel` (dict) — 여러 브랜치를 **동시에** 실행
- `RunnableLambda` — 임의 파이썬 함수를 체인에 삽입
- `RunnableBranch` — 조건 분기

## 어떻게 사용하는가? (How)

### 기본 체인: prompt → llm → parser
```python
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 공개: llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# 사내:
llm = ChatOpenAI(model=os.environ["LLM_MODEL"], base_url=os.environ["LLM_BASE_URL"],
                 api_key=os.environ["LLM_API_KEY"], temperature=0)

prompt = ChatPromptTemplate.from_template("{topic}을(를) 한 문장으로 설명해줘.")
parser = StrOutputParser()          # AIMessage → str

chain = prompt | llm | parser       # ← LCEL
print(chain.invoke({"topic": "임베딩"}))
```

### 스트리밍 & 배치 (구현 조건 확인)
```python
# 스트리밍
for tok in chain.stream({"topic": "RAG"}):
    print(tok, end="", flush=True)

# 배치 (여러 입력 병렬 처리)
results = chain.batch([{"topic": "FAISS"}, {"topic": "LangGraph"}],
                      config={"max_concurrency": 2})
```

### 병렬 브랜치: `RunnableParallel`
한 입력으로 요약·번역·키워드를 **동시에** 뽑기.
```python
from langchain_core.runnables import RunnableParallel

summarize = ChatPromptTemplate.from_template("요약: {text}") | llm | parser
keywords  = ChatPromptTemplate.from_template("키워드 3개: {text}") | llm | parser

parallel = RunnableParallel(summary=summarize, keywords=keywords)
out = parallel.invoke({"text": "긴 공정 리포트 ..."})
print(out["summary"], out["keywords"])
```

### `RunnablePassthrough` + `RunnableLambda` (RAG 준비)
```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

def retrieve(q: str) -> str:
    return "문서A, 문서B ..."   # 실제로는 retriever

rag_prompt = ChatPromptTemplate.from_template(
    "컨텍스트:\n{context}\n\n질문: {question}\n답:"
)

chain = (
    {"context": RunnableLambda(retrieve), "question": RunnablePassthrough()}
    | rag_prompt | llm | parser
)
print(chain.invoke("포토 공정이 뭐야?"))   # 입력 문자열이 question으로도, retrieve 입력으로도 흐름
```

### 구조화 출력 파서 (JSON)
```python
from langchain_core.output_parsers import JsonOutputParser
json_chain = ChatPromptTemplate.from_template(
    '아래를 JSON으로: {text}. 키는 title, tags.'
) | llm | JsonOutputParser()
print(json_chain.invoke({"text": "EUV 노광 개요"}))   # 유효 JSON이면 dict 반환; 필드/타입 검증과 내용의 사실성은 별도
```

### 추가 패턴: 폴백(fallback)과 설정 가능한 필드
```python
import os
# 모델 장애 시 대체 모델로 폴백 (사내: 무거운 모델 → 가벼운 모델)
robust_llm = llm.with_fallbacks([
    ChatOpenAI(model=os.environ["FALLBACK_MODEL"], base_url=os.environ["LLM_BASE_URL"],
               api_key=os.environ["LLM_API_KEY"]),
])

# 런타임에 temperature 등을 바꿀 수 있게 노출
from langchain_core.runnables import ConfigurableField
tunable = llm.configurable_fields(
    temperature=ConfigurableField(id="temperature")
)
tunable.invoke("창의적으로 답해", config={"configurable": {"temperature": 0.9}})
```

## 관련 문서
- [01. OpenAI API & LangChain 구조](./01-openai-api-and-langchain-structure.md)
- [03. Chain · Agent · Tool](./03-chain-agent-tool.md) — Runnable 위에 도구 호출을 얹기
- [11. RAG 질의응답 흐름](./11-rag-qa-flow.md) — LCEL로 RAG chain 구성

## 참고 자료 (References)
- LCEL / Runnable 인터페이스: https://reference.langchain.com/python/langchain-core/runnables/base
- `RunnableParallel`, `RunnablePassthrough` 개념 문서
