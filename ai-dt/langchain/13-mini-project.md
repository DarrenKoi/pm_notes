---
tags: [project, rag, agent, capstone, evaluation]
level: advanced
last_updated: 2026-07-06
reviewed_on: 2026-10-04
review_status: partial
document_type: learning_note
category_major: "AI·DT"
category_middle: "에이전트 개발"
category_minor: "통합 프로젝트"
note_kind: "실습"
classified_on: "2026-10-05"
---

# 13. 실전 Mini Project — 주제 선정부터 발표까지

> 앞선 12개 문서의 기술을 하나로 엮어 실제로 동작하는 미니 프로젝트를 완성한다. 주제 선정 → 요구사항 정의 → 구현 → 테스트 → 발표/피드백의 전 과정을 가이드한다.


> [!info] 적용 조건과 실행 순서
> 2026-10-04 개별 검토. [공통 적용 조건](./verified-conditions.md)의 판본·설정·검증 경계를 먼저 확인한다. 같은 문서의 코드 조각은 위에서 아래로 이어 실행하며 개념 조각은 별도로 표시한다. 이전 문서의 vs/chunks/embeddings 등은 관련 절의 선행 예제가 필요하다. 공개·사내 API/실제 데이터·운영 실행은 미확인이며 예제 출력은 보장이 아니다.

## 왜 필요한가? (Why)

- 조각 지식(LCEL·Agent·LangGraph·RAG)은 **하나의 완성품**을 만들어봐야 체득된다.
- 실무 도입 전, 작은 범위로 "검색→답변→검증→UI"까지 관통하는 경험이 리스크를 줄인다.
- 사내 환경(외부 API 차단, DRM 문서, 내부 LLM)에서 **끝까지 돌아가는** 파이프라인을 검증하는 것이 목표.

## 1단계: 주제 선정 (Scoping)

### 좋은 미니 프로젝트 조건
- **범위가 작다**: 1~2주, 문서 수십~수백 개 규모.
- **평가 가능**: 정답을 알 수 있는 질문 20~30개를 만들 수 있다.
- **사용자 가치**: 누군가의 반복 작업을 실제로 줄인다.

### 추천 주제 (사내 맥락)
| 주제 | 한 줄 | 쓰는 기술 |
|------|-------|-----------|
| **설비 매뉴얼 Q&A 봇** | 매뉴얼 근거로 알람·조치 질문 답변 | RAG(09~12) + 인용 |
| **이상 대응 어시스턴트** | 상태조회 API + 매뉴얼 검색 결합 | Agent(03/04) + RAG |
| **사양/레시피 검색기** | 자연어로 파트/사양 조회 | 하이브리드 검색(10) |
| **회의록 요약·액션 추출** | 문서 요약 + 구조화 출력 | LCEL(02) + `with_structured_output` |

> 권장 1순위: **설비 매뉴얼 Q&A 봇** — 12개 문서를 골고루 쓰고 평가가 쉽다.

## 2단계: 요구사항 정의 (Requirements)

예시(매뉴얼 Q&A 봇):
- **입력**: 자연어 질문. **출력**: 한국어 답변 + 근거 문서(출처/페이지).
- **기능**: (1) DRM 매뉴얼 VLM OCR 인덱싱, (2) 하이브리드 검색, (3) 근거 없으면 "모름", (4) 대화 메모리.
- **비기능**: 답변 5초 이내, 근거 인용 필수, 외부 API 미사용.
- **성공 기준**: 평가셋 30문항 중 정답률 ≥ 80%, 환각(근거 없는 단정) 0건.

## 3단계: 구현 (Implementation)

골격은 dense MMR 검색과 질문당 생성만 구현한다. 요구사항의 하이브리드·대화 메모리·검증 gate·5초/SLO는 아직 없다. 직접 생성·신뢰한 FAISS index만 load하며 모델·차원·정규화/metadata 계약을9/12절과 맞춘다. 일반 source 목록은 정확한 claim citation의 증명이 아니다.
```python
import os
# app.py — 설비 매뉴얼 Q&A 봇 (사내 엔드포인트)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

GATEWAY = os.environ["LLM_BASE_URL"]
llm = ChatOpenAI(model=os.environ["LLM_MODEL"], base_url=GATEWAY, api_key=os.environ["LLM_API_KEY"], temperature=0)
emb = OpenAIEmbeddings(model=os.environ["EMBEDDING_MODEL"],
                              check_embedding_ctx_length=False, encoding_format="float", base_url=GATEWAY, api_key=os.environ["LLM_API_KEY"])

# (사전 단계에서 12번 파이프라인으로 만든 인덱스를 로드)
vs = FAISS.load_local("faiss_kb", emb, allow_dangerous_deserialization=True, normalize_L2=True)
retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20})

def format_docs(docs):
    return "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))

prompt = ChatPromptTemplate.from_template(
    "너는 설비 매뉴얼 도우미다. 아래 컨텍스트만 근거로 한국어로 답하고, "
    "근거가 없으면 '매뉴얼에서 찾지 못했습니다'라고 답해.\n\n"
    "컨텍스트:\n{context}\n\n질문: {question}\n답:")

chain = RunnableParallel(
    {"docs": retriever, "question": RunnablePassthrough()}
).assign(
    answer=lambda x: (prompt | llm | StrOutputParser()).invoke(
        {"context": format_docs(x["docs"]), "question": x["question"]}))

def ask(q: str):
    out = chain.invoke(q)
    sources = [f"{d.metadata.get('source')} p.{d.metadata.get('page')}" for d in out["docs"]]
    return out["answer"], sources

if __name__ == "__main__":
    ans, src = ask("EQP-102 온도과열 알람이 뜨면?")
    print(ans, "\n근거:", src)
```
> 확장: 단순 도구 선택은 [03번](./03-chain-agent-tool.md)의 `create_agent`로 붙이고, 메모리·승인·조건 분기가 필요하면 [06번](./06-multistep-conversation-flow.md)의 checkpointer와 [08번](./08-langgraph-agent-scenario.md)의 LangGraph 흐름으로 승격한다.

### 간단한 UI (선택)
```python
# pip install gradio  (사내 허용 시)
import gradio as gr
def chat(msg, history):
    ans, src = ask(msg)  # history를 사용하지 않는 단일 질문 UI; 대화 기억 미구현
    return f"{ans}\n\n📎 근거: {', '.join(src)}"
gr.ChatInterface(chat).launch()
```

## 4단계: 테스트 (Evaluation)

아래 코드는 문자열 포함률만 측정하며 정답률/충실성/검색 적중을 구현하지 않는다. 부정문도 키워드를 포함하면 통과한다. 30문항/80%/5초/환각0건은 원래 제안 목표이며 실측 성과나 무환각 보장이 아니다. 빈 평가셋은 성공0%로 해석하지 않고 미측정으로 표시한다. 핵심 지표:
- **정답률(Correctness)**: LLM-as-judge 또는 키워드 매칭.
- **충실성(Faithfulness)**: 답이 근거에 있는가(환각 여부, [11번](./11-rag-qa-flow.md) 검증).
- **검색 적중(Retrieval hit)**: 정답 문서가 검색 상위에 들어왔는가.

```python
eval_set = [
    {"q": "온도과열 알람 대응 순서는?", "must_include": ["쿨링", "재기동"]},
    # ... 30문항
]
hit = 0
for e in eval_set:
    ans, _ = ask(e["q"])
    if all(kw in ans for kw in e["must_include"]):
        hit += 1
if not eval_set:
    print("미측정: 평가셋 없음")
else:
    print(f"키워드 포함률: {hit}/{len(eval_set)} = {hit/len(eval_set):.0%}")
```
> LLM-as-judge: 채점을 `llm.with_structured_output(Score)`로 자동화하면 자유형 답변도 평가 가능. 단, 채점 LLM도 사내 모델을 사용.

### 흔한 실패와 처방
| 증상 | 원인 | 처방 |
|------|------|------|
| 엉뚱한 답 | 검색 실패 | 청킹 재조정, 하이브리드 검색([10](./10-retriever-tuning.md)) |
| 근거 없이 단정 | 환각 | "모르면 모른다" 프롬프트 + 충실성 검증 |
| 느림 | k·리랭킹 과다 | k 축소, HNSW 인덱스([09](./09-document-embedding-faiss.md)) |
| OCR 깨짐 | VLM 라우팅 | 원본/누락률 대조와8B/30B 후보 측정으로([12](./12-rag-document-sources.md)) |

## 5단계: 발표 및 피드백 (Presentation)

### 발표 구성(10분)
1. **문제**: 어떤 반복 작업을 줄이려 했나 (1장)
2. **접근**: 파이프라인 다이어그램 (검색→생성→검증) (1장)
3. **데모**: 실제 질의응답 + 근거 인용 (라이브)
4. **평가**: 정답률/환각률 수치 (1장)
5. **한계·다음 단계**: 데이터 확장, 운영 DB(OpenSearch) 이관, MCP화 (1장)

### 피드백 반영 루프
- 발표에서 나온 반례 질문을 **평가셋에 추가** → 재튜닝 → 재측정.
- "왜 이 답이 나왔나"를 근거 문서로 설명할 수 있어야 신뢰를 얻는다.

## 완성 체크리스트
- [ ] 인덱싱: 소스(DRM 포함) → Document → 청킹 → 벡터스토어
- [ ] 검색: 하이브리드/MMR 적용, 결과 눈으로 확인
- [ ] 생성: 근거 기반, "모름" 처리, 인용 표시
- [ ] 검증: 원문 대조·충실성 체크의 오탐/미탐 확인
- [ ] 평가: 정답셋 30문항 자동 채점 ≥ 목표치
- [ ] 발표: 데모 + 수치 + 한계/다음 단계

## 관련 문서
- 전체 커리큘럼: [README](./README.md)
- 핵심 의존: [02. LCEL](./02-lcel.md), [08. Agent 시나리오](./08-langgraph-agent-scenario.md), [09~12 RAG](./09-document-embedding-faiss.md)

## 참고 자료 (References)
- RAG 평가 개념: LLM-as-judge, faithfulness/answer-relevance
- LangChain 튜토리얼: https://docs.langchain.com/oss/python/
