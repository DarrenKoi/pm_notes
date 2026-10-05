---
tags: [rag, langchain, langgraph]
reviewed_on: 2026-10-04
review_status: partial
document_type: learning
category_major: "AI·DT"
category_middle: "RAG"
category_minor: "LangChain·Tool Calling"
note_kind: "학습"
classified_on: "2026-10-05"
---

# LangChain + LangGraph 기초 사용법

> [!info] 검토 — 2026-10-04
> 공식 근거·설치 판본·검증 결과는 [RAG 정리 기록](../organization-log.md)에 있다. 실제 모델·서버·회사 권한은 미확인이다. 함수/객체를 명시적으로 조립하는 학습 예제이며 자동 API 접속은 하지 않는다.


## 1) LangChain vs LangGraph

### LangChain이 잘하는 것

- PromptTemplate, OutputParser, Retriever, Tool 같은 **구성 요소(component)** 조합
- LLM 호출을 파이프라인 형태로 빠르게 작성
- 공통 인터페이스로 provider 교체를 돕지만 tool/JSON/vision/usage·인증 지원은 별도 확인

### LangGraph가 잘하는 것

- 노드/엣지 기반 **상태 중심(stateful) 제어 흐름**
- 분기, 반복, 재시도, human-in-the-loop 같은 복잡한 워크플로우
- 대화 상태/중간 결과를 그래프 state에 보관

### 실무에서의 조합

- LangChain: "무엇을 실행할지"(프롬프트, 검색, 툴)
- LangGraph: "어떤 순서/조건으로 실행할지"(라우팅, 루프, 종료 조건)

---

## 2) 최소 설치

```bash
# 검증한 실습 판본; 최신/운영 lock 또는 전체 OS 호환성 보장이 아님.
python -m pip install langchain==1.4.3 langgraph==1.2.12 langchain-openai==1.6.7
python -m pip install langchain-community==0.4.2 langchain-text-splitters==1.1.3 faiss-cpu==1.15.1
```

---

## 3) LangChain 최소 예제 (Chain)

```python
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def make_llm():
    # 명시 호출할 때 객체 생성; invoke는 실습자가 승인된 환경에서 실행한다.
    return ChatOpenAI(model=os.environ["LLM_MODEL"], temperature=0)

def build_chain(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "너는 친절한 기술 튜터다."), ("human", "{question}")
    ])
    return prompt | llm | StrOutputParser()

def ask_chain(chain, question: str) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("비어 있지 않은 question 필요")
    answer = chain.invoke({"question": question})
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("텍스트 답변 없음")
    return answer

# chain = build_chain(make_llm())
# print(ask_chain(chain, "RAG가 뭔지 3줄로 설명해줘"))
```

핵심은 `prompt | llm | parser`처럼 LCEL로 구성 요소를 연결하는 점이다. temperature=0은 완전한 재현성을 보장하지 않는다. 판본/모델 alias·응답 반복 분산을 따로 기록한다.

---

## 4) LangGraph 최소 예제 (Graph)

```python
from typing import TypedDict, Required
from langgraph.graph import StateGraph, START, END

class MyState(TypedDict, total=False):
    question: Required[str]
    answer: str

def build_graph(llm):
    def answer_node(state: MyState):
        question = state.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("비어 있지 않은 question 필요")
        response = llm.invoke(f"질문에 간단히 답해줘: {question}")
        if not isinstance(response.content, str) or not response.content.strip():
            raise ValueError("텍스트 답변 없음; tool/refusal 계약은 별도")
        return {"answer": response.content}  # 노드는 변경한 key만 반환할 수 있음
    builder = StateGraph(MyState)
    builder.add_node("answer", answer_node)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    return builder.compile()

# app = build_graph(make_llm())
# print(app.invoke({"question": "LangGraph의 장점은?"})["answer"])
```

---

위 최소 그래프는 한 번 답하고 종료한다. 체크포인터·도구·검색·승인·재시도·관측 설정은 제공하지 않는다. TypedDict는 런타임 입력 검증을 대신하지 않는다. 아래는 확장 후보이며 구현 완료가 아니다.

## 5) 어떤 기능까지 확장 가능한가?

1. **멀티 스텝 에이전트**
   - 계획(Plan) → 도구 실행(Act) → 검증(Check) 루프
2. **조건부 라우팅**
   - 질문 유형(정의형/분석형/코드형)별 다른 노드로 분기
3. **고급 RAG**
   - Query Rewrite, Multi-Query, Re-ranking, Self-RAG/CRAG
4. **Human-in-the-loop**
   - 특정 신뢰도 이하일 때 승인 요청 노드로 이동
5. **장기 메모리/세션 관리**
   - 사용자별 컨텍스트 저장소 연결
6. **외부 시스템 통합**
   - 검색 API, 사내 DB, 티켓 시스템, MCP 서버
7. **평가/관측성**
   - 실행 추적(trace), 노드별 latency/token/cost 모니터링

---

## 6) 설계 체크리스트

- 상태(state)에 무엇을 저장할지 먼저 정의했는가?
- 종료 조건(END)과 최대 반복 횟수를 정의했는가?
- 실패 시 fallback 경로(재시도/간단 답변)를 마련했는가?
- Tool 결과 검증 및 권한 범위를 제한했는가?
- RAG의 chunk/index/retriever 하이퍼파라미터를 측정 기반으로 조정하는가?

이 체크리스트를 먼저 고정하면, 이후 RAG/Tool Calling 확장이 훨씬 쉬워진다.
