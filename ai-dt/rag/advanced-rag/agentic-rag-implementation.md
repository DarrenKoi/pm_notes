# Agentic RAG 구현

> StateGraph로 retrieve → grade → generate/rewrite 조건 분기를 구현하여, LLM이 검색 전략을 자율 판단하는 RAG 시스템을 만든다

---
tags: [agentic-rag, langgraph, stategraph, grade-documents, corrective-rag]
level: advanced
last_updated: 2026-07-16
---

## 왜 필요한가? (Why)

Naive RAG(검색 → 생성)의 한계:

| 문제 | 원인 | 결과 |
|------|------|------|
| 노이즈 컨텍스트 | 관련 없는 문서가 컨텍스트에 포함 | 부정확한 답변 |
| 검색 실패 | 쿼리가 문서 용어와 불일치 | "모르겠습니다" 응답 |
| 품질 편차 | 검색 결과에 대한 검증 없음 | 답변 신뢰도 불안정 |

**Agentic RAG**는 LLM이 **검색 결과를 판별**하고, 관련 문서가 없으면 **쿼리를 재작성하여 재검색**하는 자율 루프를 구성한다. 핵심은 LangGraph의 `StateGraph`로 **조건 분기(conditional edges)**를 구현하는 것이다.

## 핵심 개념 (What)

### 그래프 흐름

```
START
  → retrieve (벡터스토어 검색)
    → grade_documents (LLM으로 관련성 판별)
      ├─ [관련 문서 있음] → generate (답변 생성) → END
      └─ [관련 문서 없음] → rewrite_query (쿼리 재작성) → retrieve (재검색)
```

### 노드 역할 정리

| 노드 | 역할 | 입력 | 출력 |
|------|------|------|------|
| `retrieve` | 벡터스토어에서 질문으로 검색 | `question` | `documents` |
| `grade_documents` | 각 문서의 관련성을 LLM으로 판별 | `question`, `documents` | `documents` (필터링됨) |
| `generate` | 관련 문서 기반 최종 답변 생성 | `question`, `documents` | `generation` |
| `rewrite_query` | 검색 실패 시 질문을 재작성 | `question` | `question` (재작성됨) |

### 조건 분기 함수

| 함수 | 판단 기준 | 반환값 |
|------|----------|--------|
| `route_question` | `documents` 리스트가 비어있는지 | `"generate"` 또는 `"rewrite"` |

## 어떻게 사용하는가? (How)

### 1단계: RAGState 정의

StateGraph의 모든 노드가 공유하는 상태 구조를 `TypedDict`로 정의한다.

```python
from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages
from langchain_core.documents import Document

class RAGState(TypedDict):
    messages:   Annotated[list, add_messages]  # 대화 이력 (누적)
    question:   str                             # 현재 질문 (재작성 시 업데이트)
    documents:  List[Document]                  # 검색·필터링된 문서
    generation: str                             # 최종 생성 답변
```

**핵심 포인트:**

| 필드 | 리듀서 | 설명 |
|------|--------|------|
| `messages` | `add_messages` | 메시지 히스토리 누적 (덮어쓰기 방지) |
| `question` | 없음 (덮어쓰기) | rewrite 시 업데이트됨 |
| `documents` | 없음 (덮어쓰기) | grade 필터링 결과로 교체 |
| `generation` | 없음 (덮어쓰기) | 최종 답변 |

`add_messages` 리듀서가 없으면 `messages`가 매 노드 실행마다 덮어씌워져 대화 이력이 소실된다.

### 2단계: GradeDocuments 구조화 출력 스키마

LLM 출력을 `"yes"` / `"no"` 로만 강제한다.

```python
from typing import Literal
from pydantic import BaseModel, Field

class GradeDocuments(BaseModel):
    """문서-질문 관련성 판별 결과"""
    score: Literal["yes", "no"] = Field(
        description="문서가 질문에 관련이 있으면 'yes', 없으면 'no'"
    )

grader = llm.with_structured_output(GradeDocuments)
```

**`with_structured_output()` 동작 원리:**

```
LLM 호출 → JSON 출력 강제 → Pydantic 모델로 파싱 → score="yes" 또는 "no"
```

`Literal["yes", "no"]`로 타입을 제한하면 LLM이 반드시 이 두 값 중 하나만 반환한다. "maybe", "partially" 같은 모호한 답변을 원천 차단한다.

`Field(description=...)`은 LLM에게 전달되는 채점 기준이다. 도메인에 맞게 구체적으로 작성할수록 판별 정확도가 향상된다:

```python
# PM 도메인
description="문서가 질문에 관련이 있으면 'yes', 없으면 'no'"

# 반도체 공정 도메인
description="문서가 반도체 공정 질문에 관련이 있으면 'yes', 없으면 'no'"
```

### 3단계: 노드 함수 구현

#### retrieve 노드

```python
def retrieve_node(state: RAGState) -> dict:
    """벡터 스토어에서 관련 문서를 검색한다."""
    question = state["question"]
    docs = retriever.invoke(question)
    return {"documents": docs}
```

`retriever`는 이전 단계([Advanced RAG 파이프라인](./advanced-rag-pipeline.md))에서 구성한 ChromaDB 기반 retriever를 사용한다. 재검색 시에도 동일한 노드를 재사용한다 — `question`이 `rewrite_query`에 의해 업데이트되어 있으므로 새로운 검색어로 동작한다.

#### grade_documents 노드

```python
def grade_documents_node(state: RAGState) -> dict:
    """각 검색 문서의 관련성을 LLM으로 판별하고 필터링한다."""
    question = state["question"]
    docs     = state["documents"]
    filtered = []

    for doc in docs:
        prompt = (
            f"질문: {question}\n\n"
            f"문서 내용:\n{doc.page_content}\n\n"
            "이 문서가 질문에 관련이 있나요? 'yes' 또는 'no'로만 답하세요."
        )
        result = grader.invoke(prompt)
        if result.score.lower() == "yes":
            filtered.append(doc)

    print(f"  [grade] {len(docs)}개 검색 → {len(filtered)}개 관련 문서")
    return {"documents": filtered}
```

**핵심 동작:** 검색된 문서를 **개별적으로** 판별한다. 3개 검색 → 1개만 관련 → 1개만 남기고 2개 제거. 모든 문서가 `"no"`면 `documents=[]`가 되어 조건 분기에서 `rewrite` 경로로 이동한다.

#### generate 노드

```python
def generate_node(state: RAGState) -> dict:
    """관련 문서를 기반으로 최종 답변을 생성한다."""
    question = state["question"]
    context  = "\n\n".join(
        f"[출처: {os.path.basename(doc.metadata.get('source', ''))}]\n{doc.page_content}"
        for doc in state["documents"]
    )
    prompt = (
        f"아래 문서를 참고하여 질문에 정확하고 상세하게 답변하세요.\n\n"
        f"문서:\n{context}\n\n"
        f"질문: {question}"
    )
    result = llm.invoke(prompt)
    return {
        "generation": result.content,
        "messages":   [result],
    }
```

`context`에 출처 정보(`[출처: 파일명]`)를 포함하면 LLM이 답변에 근거를 명시할 수 있다.

#### rewrite_query 노드

```python
def rewrite_node(state: RAGState) -> dict:
    """검색 결과가 없을 때 질문을 재작성한다."""
    question = state["question"]
    prompt = (
        f"다음 질문을 벡터 검색에 더 적합하도록 재작성하세요.\n"
        f"- 핵심 키워드를 명확히 포함하세요\n"
        f"- 모호한 표현을 구체적으로 바꾸세요\n\n"
        f"원래 질문: {question}\n\n"
        f"재작성된 질문:"
    )
    result = llm.invoke(prompt)
    new_question = result.content.strip()
    print(f"  [rewrite] '{question}' → '{new_question}'")
    return {"question": new_question}
```

**도메인 특화 재작성 프롬프트:**

```python
# 반도체 공정 도메인
prompt = (
    f"다음 반도체 공정 관련 질문을 벡터 검색에 더 적합하도록 재작성하세요.\n"
    f"- 공정명(Etch, CVD, CMP, Particle 등) 키워드를 명확히 포함하세요\n"
    f"- 장비명·파라미터명을 구체적으로 바꾸세요\n\n"
    f"원래 질문: {question}\n\n"
    f"재작성된 질문:"
)
```

도메인 키워드를 명시적으로 안내하면 재작성 품질이 향상된다.

### 4단계: 조건 분기 함수

```python
def route_question(state: RAGState) -> str:
    """관련 문서 유무에 따라 generate 또는 rewrite 경로를 선택한다."""
    if state["documents"]:
        print("  [route] 관련 문서 있음 → generate")
        return "generate"
    print("  [route] 관련 문서 없음 → rewrite")
    return "rewrite"
```

반환값 `"generate"` / `"rewrite"`는 `add_conditional_edges`의 `path_map` 키와 **정확히 일치**해야 한다. 불일치 시 런타임 에러가 발생한다.

### 5단계: StateGraph 조립 및 컴파일

```python
from langgraph.graph import StateGraph, START, END

workflow = StateGraph(RAGState)

# 노드 등록
workflow.add_node("retrieve",        retrieve_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("generate",        generate_node)
workflow.add_node("rewrite_query",   rewrite_node)

# 엣지 정의
workflow.add_edge(START,             "retrieve")
workflow.add_edge("retrieve",        "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    route_question,
    {"generate": "generate", "rewrite": "rewrite_query"},
)
workflow.add_edge("rewrite_query",   "retrieve")  # 재시도 루프
workflow.add_edge("generate",        END)

rag_agent = workflow.compile()
```

**그래프 구조 해설:**

```
add_edge(START, "retrieve")
→ 시작하면 무조건 retrieve 실행

add_edge("retrieve", "grade_documents")
→ 검색 후 무조건 판별 실행

add_conditional_edges("grade_documents", route_question, {...})
→ 판별 결과에 따라 분기:
  - "generate" → generate 노드로
  - "rewrite"  → rewrite_query 노드로

add_edge("rewrite_query", "retrieve")
→ 재작성 후 다시 검색 → 재시도 루프 형성

add_edge("generate", END)
→ 답변 생성 후 종료
```

#### 재시도 루프 주의사항

`rewrite_query → retrieve → grade_documents → rewrite_query → ...` 루프가 무한 반복될 수 있다. 프로덕션 환경에서는 **최대 재시도 횟수(max_retry)** 를 추가해야 한다:

```python
class RAGState(TypedDict):
    messages:    Annotated[list, add_messages]
    question:    str
    documents:   List[Document]
    generation:  str
    retry_count: int  # 재시도 카운터 추가

def route_question(state: RAGState) -> str:
    if state["documents"]:
        return "generate"
    if state.get("retry_count", 0) >= 2:
        return "generate"  # 최대 재시도 초과 시 강제 생성
    return "rewrite"

def rewrite_node(state: RAGState) -> dict:
    # ... 기존 코드 ...
    return {
        "question": new_question,
        "retry_count": state.get("retry_count", 0) + 1,
    }
```

### 6단계: 실행 및 테스트

```python
def ask_rag(question: str) -> str:
    """Agentic RAG에 질문을 전달하고 최종 답변을 반환한다."""
    initial_state = {
        "messages":   [],
        "question":   question,
        "documents":  [],
        "generation": "",
    }
    result = rag_agent.invoke(initial_state)

    docs_used = result.get("documents", [])
    answer    = result.get("generation", "(생성 실패)")

    print(f"[사용된 문서 {len(docs_used)}개]")
    for doc in docs_used:
        src = os.path.basename(doc.metadata.get("source", ""))
        print(f"  - {src}")

    print(f"\n[답변]\n{answer}")
    return answer
```

**테스트 시나리오 설계:**

| 질문 유형 | 기대 동작 | 검증 포인트 |
|----------|----------|------------|
| 직접 매칭 | retrieve → grade 통과 → generate | 올바른 문서 검색 확인 |
| 간접 표현 | retrieve → grade 실패 → rewrite → 재검색 → generate | rewrite 동작 확인 |
| 복합 질문 | retrieve → 일부 grade 통과 → generate | 부분 필터링 확인 |

```python
# 테스트 1: 직접 매칭 (PM 도메인)
ask_rag("프로젝트 리스크 관리 절차를 설명해주세요")
# 기대: 리스크_관리_절차서.md 검색 → grade 통과 → generate

# 테스트 2: 간접 표현 → rewrite 트리거
ask_rag("스프린트 회고 미팅")
# 기대: grade 실패 가능 → rewrite → "애자일 스프린트 회고 미팅의 목적" → 재검색

# 테스트 3: 반도체 공정 도메인
ask_rag("Particle 불량 발생 시 조치 방법은?")
# 기대: Particle_불량_조치_가이드.md + 증착공정_트러블슈팅.md 검색
```

### 7단계: 그래프 시각화 (Jupyter 환경)

```python
from IPython.display import Image, display

try:
    display(Image(rag_agent.get_graph().draw_mermaid_png()))
except Exception as e:
    print(f"시각화 건너뜀: {e}")
```

## 검색 품질 검증 패턴

일괄 테스트로 검색 정확도를 정량 확인하는 패턴:

```python
test_questions = [
    ("리스크 관리 절차",     "리스크_관리_절차서.md"),
    ("스프린트 회고 미팅",    "애자일_스크럼_가이드.md"),
    ("품질 검수 기준",       "품질_검수_체크리스트.md"),
]

for question, expected_src in test_questions:
    result = rag_agent.invoke({
        "messages": [], "question": question,
        "documents": [], "generation": "",
    })
    actual_sources = list({
        os.path.basename(d.metadata.get("source", ""))
        for d in result.get("documents", [])
    })
    hit = "O" if any(expected_src in s for s in actual_sources) else "X"
    print(f"{hit} {question:28s} 기대: {expected_src:25s} 실제: {', '.join(actual_sources)}")
```

기대 출처와 실제 검색 출처를 비교하여 `O` / `X`로 판정한다. `X`가 발생하면 `chunk_size`, `k`, 프롬프트를 조정한다.

## Agentic RAG가 Naive RAG보다 나은 이유 — 실제 비교 결과

워크숍 실험 결과:

| 질문 | 방법 | 사용 문서 수 | 답변 길이 |
|------|------|------------|----------|
| Particle 불량 조치 | Naive RAG | 3 | 998자 |
| | Agentic RAG | 3 | 1,578자 |
| CMP Pad 교체 주기 | Naive RAG | 3 | 476자 |
| | Agentic RAG | **1** | 647자 |
| Etch RF Power 이상 | Naive RAG | 3 | 784자 |
| | Agentic RAG | **1** | 912자 |

**핵심 관찰:**
- Agentic RAG는 CMP/Etch 질문에서 **관련 없는 2개 문서를 제거**하고 1개만 사용했다
- 관련 문서만 사용했음에도 답변이 **더 길고 상세**했다 — 노이즈 제거로 LLM이 관련 정보에 집중
- Naive RAG는 3개 문서를 모두 컨텍스트에 넣어 노이즈가 답변 품질을 저하시켰다

## 관련 문서

- [Advanced RAG 파이프라인](./advanced-rag-pipeline.md) — 벡터스토어 구축 (이 문서의 사전 단계)
- [RAG 확장 기법](./rag-extensions.md) — HyDE, MemorySaver 등 고급 확장
- [멀티에이전트 RAG 통합](./multi-agent-rag-integration.md) — Supervisor 패턴 통합
- [LangGraph 기초](../langgraph/langgraph-basics.md) — StateGraph 개념 복습

## 참고 자료 (References)

- [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph/)
- [Corrective RAG 논문](https://arxiv.org/abs/2401.15884)
- [LangChain Structured Output 가이드](https://python.langchain.com/docs/how_to/structured_output/)
