---
tags: [rag, qa, lcel, crag, self-rag, agentic-rag, citation]
level: advanced
last_updated: 2026-07-06
---

# 11. RAG 기반 질의 응답 흐름 구성

> retriever와 LLM을 연결해 "검색 → 프롬프트 주입 → 생성" RAG chain을 만들고, 인용·CRAG/Self-RAG·Agentic RAG 같은 최신 흐름으로 확장한다.

## 왜 필요한가? (Why)

- 임베딩·검색을 갖췄으면, 이제 검색 결과를 **프롬프트에 넣어 근거 기반 답변**을 생성해야 완성이다.
- 단순 RAG는 "검색이 틀리면 답도 틀린다". 최신 흐름(CRAG/Self-RAG)은 검색 품질을 **스스로 평가·보정**해 견고성을 높인다.
- 사내: 근거 문서를 **인용**해 신뢰성을 확보하고, 환각을 검증하는 단계가 실무 도입의 필수 조건.

## 핵심 개념 (What)

### 기본 RAG chain (LCEL)
`{context: retriever, question: passthrough} | prompt | llm | parser`. context는 검색 문서를 문자열로 합친 것.

### 최신 RAG 변형
| 패턴 | 아이디어 |
|------|----------|
| **Naive RAG** | 한 번 검색 → 한 번 생성 |
| **CRAG** (Corrective RAG) | 검색 문서의 관련성을 평가 → 부족하면 질문 재작성/웹검색 후 재검색 |
| **Self-RAG** | 생성 후 "근거에 부합하는가"를 스스로 채점, 미달이면 재생성 |
| **Agentic RAG** | 검색을 도구로 보고, Agent가 "검색할지/몇 번 할지"를 스스로 결정 |

## 어떻게 사용하는가? (How)

### 1) 기본 RAG chain (LCEL)
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="Kimi-K2.5", base_url="http://llm-gateway.internal/v1",
                 api_key="EMPTY", temperature=0)
retriever = vs.as_retriever(search_kwargs={"k": 4})

def format_docs(docs):
    return "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))

prompt = ChatPromptTemplate.from_template(
    "아래 컨텍스트만 근거로 한국어로 답해. 모르면 모른다고 해.\n\n"
    "컨텍스트:\n{context}\n\n질문: {question}\n답:"
)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)
print(rag_chain.invoke("포토 공정 결함의 흔한 원인은?"))
```

### 2) 인용(citation) 포함 — 근거 추적
문서와 답을 함께 반환해 어떤 문서를 근거로 썼는지 보여준다.
```python
from langchain_core.runnables import RunnableParallel

rag_with_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(
    answer=(lambda x: {"context": format_docs(x["context"]), "question": x["question"]})
           | prompt | llm | StrOutputParser()
)
out = rag_with_source.invoke("온도과열 알람 대응법?")
print(out["answer"])
for d in out["context"]:
    print("근거:", d.metadata.get("source"))
```

### 3) CRAG: 검색 품질 평가 후 보정 (LangGraph)
```python
from typing import List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

class Grade(BaseModel):
    binary_score: str = Field(description="관련 있으면 'yes' 없으면 'no'")

grader = llm.with_structured_output(Grade)

class State(TypedDict):
    question: str
    documents: List
    web_needed: str
    generation: str

def retrieve(s):
    return {"documents": retriever.invoke(s["question"])}

def grade(s):                      # 각 문서 관련성 채점
    kept, need_web = [], "no"
    for d in s["documents"]:
        g = grader.invoke(f"질문:{s['question']}\n문서:{d.page_content}")
        if g.binary_score == "yes": kept.append(d)
        else: need_web = "yes"
    return {"documents": kept, "web_needed": need_web}

def transform_or_search(s):        # 부족하면 질문 재작성(여기선 간단화)
    q2 = llm.invoke(f"검색이 잘 되도록 질문을 다시 써: {s['question']}").content
    return {"question": q2, "documents": s["documents"]}

def generate(s):
    ctx = format_docs(s["documents"])
    ans = (prompt | llm | StrOutputParser()).invoke(
        {"context": ctx, "question": s["question"]})
    return {"generation": ans}

def decide(s) -> str:
    return "transform" if s["web_needed"] == "yes" else "generate"

b = StateGraph(State)
for name, fn in [("retrieve", retrieve), ("grade", grade),
                 ("transform", transform_or_search), ("generate", generate)]:
    b.add_node(name, fn)
b.add_edge(START, "retrieve")
b.add_edge("retrieve", "grade")
b.add_conditional_edges("grade", decide,
                        {"transform": "transform", "generate": "generate"})
b.add_edge("transform", "retrieve")     # 재작성 후 재검색 루프
b.add_edge("generate", END)
crag = b.compile()

print(crag.invoke({"question": "EQP-102 온도 알람 조치"})["generation"])
```

### 4) Agentic RAG (검색을 도구로)
retriever를 도구로 만들어 Agent가 검색 여부·횟수를 스스로 결정.
```python
from langchain.agents import create_agent
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever, name="search_docs",
    description="사내 공정/설비 문서에서 관련 내용을 검색한다.")

agent = create_agent(
    model=llm,
    tools=[retriever_tool],
    system_prompt="답변 전 필요한 경우 search_docs로 근거 문서를 검색한다.",
)
out = agent.invoke({
    "messages": [{"role": "user", "content": "포토 공정 결함 원인과 조치를 문서 근거로 알려줘"}]
})
print(out["messages"][-1].content)
```

### 5) 환각 검증(Self-RAG의 핵심 아이디어)
생성 답이 컨텍스트에 근거하는지 별도 LLM 호출로 채점하고, 미달이면 재생성/보류.
```python
class Faithful(BaseModel):
    grounded: str = Field(description="답이 컨텍스트에 근거하면 'yes'")

checker = llm.with_structured_output(Faithful)
verdict = checker.invoke(f"컨텍스트:{ctx}\n답:{answer}\n답이 컨텍스트에 근거하나?")
if verdict.grounded == "no":
    answer = "확실한 근거를 찾지 못했습니다. 원문 확인이 필요합니다."
```

## 관련 문서
- [09. 문서 임베딩 & FAISS](./09-document-embedding-faiss.md) · [10. Retriever 튜닝](./10-retriever-tuning.md)
- [07. Condition/Branching/Tool workflow](./07-condition-branching-tool-workflow.md) — CRAG 루프의 기반
- [12. PDF·웹·내부 지식 적용](./12-rag-document-sources.md)

## 참고 자료 (References)
- RAG 튜토리얼: https://docs.langchain.com/oss/python/
- CRAG: Yan et al., "Corrective Retrieval-Augmented Generation" (2024)
- Self-RAG: Asai et al., "Self-RAG" (2023)
