# Advanced RAG 파이프라인

> 문서 로딩부터 벡터스토어 구축, Retriever 구성까지 — RAG 시스템의 기반을 단계별로 구현한다

---
tags: [rag, pipeline, chromadb, embedding, text-splitting]
level: intermediate
last_updated: 2026-07-16
---

## 왜 필요한가? (Why)

RAG 시스템의 **답변 품질은 검색 품질에 의존**한다. 문서를 어떻게 분할하고, 어떤 임베딩을 사용하며, 벡터스토어를 어떻게 구성하느냐에 따라 검색 정확도가 크게 달라진다. 이 문서는 Agentic RAG의 토대가 되는 **문서 파이프라인**을 체계적으로 구성하는 방법을 다룬다.

## 핵심 개념 (What)

### 전체 파이프라인 흐름

```
.md 파일
  → DirectoryLoader (문서 로딩)
    → RecursiveCharacterTextSplitter (청크 분할)
      → OpenAIEmbeddings (벡터 변환)
        → ChromaDB (벡터스토어 저장)
          → Retriever (검색 인터페이스)
```

| 단계 | 구성요소 | 역할 |
|------|---------|------|
| 로딩 | `DirectoryLoader` + `TextLoader` | 디렉토리에서 마크다운 파일 일괄 로드 |
| 분할 | `RecursiveCharacterTextSplitter` | 마크다운 구조를 존중하면서 청크 분할 |
| 임베딩 | `OpenAIEmbeddings` | 텍스트를 벡터로 변환 |
| 저장 | `ChromaDB` | 로컬 파일 기반 벡터 DB에 영속 저장 |
| 검색 | `as_retriever()` | 유사도 기반 top-k 검색 인터페이스 |

## 어떻게 사용하는가? (How)

### 1단계: 환경 설정

```python
# 필요 패키지 설치
# pip install langchain langchain-openai langchain-community langchain-text-splitters langchain-chroma chromadb python-dotenv

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv(override=True)

llm = ChatOpenAI(model="gpt-4.1", temperature=0)
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
```

`temperature=0`으로 설정하는 이유: RAG 답변은 문서 근거 기반이므로 **일관성(결정적 출력)**이 중요하다. 창의적 변형보다 정확한 재현이 우선이다.

### 2단계: 문서 로딩 — DirectoryLoader

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader

DOCS_PATH = "sample_data/pm/pm_docs/"

raw_docs = DirectoryLoader(
    DOCS_PATH,
    glob="*.md",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
).load()

# 로딩 결과 확인
for doc in raw_docs:
    src = os.path.basename(doc.metadata.get("source", ""))
    print(f"  {src}  ({len(doc.page_content):,} chars)")
```

**핵심 파라미터:**

| 파라미터 | 설명 |
|---------|------|
| `glob="*.md"` | 마크다운 파일만 선택적 로드 |
| `loader_cls=TextLoader` | 원문 그대로 보존 (파싱 변환 없음) |
| `encoding="utf-8"` | 한글 문서 깨짐 방지 — **필수** |

### 3단계: 문서 분할 — RecursiveCharacterTextSplitter

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n## ", "\n### ", "\n\n", "\n", " "],
)
splits = splitter.split_documents(raw_docs)

print(f"원본 {len(raw_docs)}개 → 분할 {len(splits)}개 청크")
print(f"평균 청크 길이: {sum(len(s.page_content) for s in splits) / len(splits):.0f} chars")
```

**분할 전략 상세:**

| 파라미터 | 값 | 근거 |
|---------|---|------|
| `chunk_size` | 500 | 검색 정확도와 컨텍스트 길이의 균형점 |
| `chunk_overlap` | 50 | 청크 경계에서 정보 손실 최소화 |
| `separators` | 마크다운 헤더 우선 | 의미 단위 분할 → 검색 품질 향상 |

**separators 동작 원리:**

```
우선순위:  "\n## " > "\n### " > "\n\n" > "\n" > " "

문서 내용:
## 1. 리스크 식별        ← "\n## " 기준으로 먼저 분할 시도
리스크를 찾아내는...
### 1.1 브레인스토밍     ← chunk_size 초과 시 "\n### " 기준으로 세분화
팀 전체가 참여하여...
```

마크다운 헤더(`## `, `### `)를 최우선 분할 지점으로 설정하면 **의미 단위가 보존**되어 검색 시 관련 청크를 정확히 반환할 확률이 높아진다.

#### chunk_size 선택 가이드

| chunk_size | 장점 | 단점 | 적합한 경우 |
|-----------|------|------|------------|
| 200~300 | 검색 정밀도 높음 | 컨텍스트 부족 가능 | FAQ, 정의 문서 |
| 400~600 | 정밀도-컨텍스트 균형 | - | 일반 기술 문서 (권장) |
| 800~1000 | 풍부한 컨텍스트 | 노이즈 증가 | 장문 보고서, 논문 |

### 4단계: 벡터스토어 생성 — ChromaDB

```python
import shutil
from pathlib import Path
from langchain_chroma import Chroma

CHROMA_DIR = "./pm_chroma_db"

# 재실행 시 중복 방지: 기존 DB 삭제
if Path(CHROMA_DIR).exists():
    shutil.rmtree(CHROMA_DIR)

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    collection_name="pm_docs",
    persist_directory=CHROMA_DIR,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
```

**ChromaDB 핵심 포인트:**

| 항목 | 설명 |
|------|------|
| `persist_directory` | 로컬 폴더 기반 영속 저장 — 서버 불필요 |
| `collection_name` | 논리적 문서 그룹 구분 |
| `search_kwargs={"k": 3}` | 상위 3개 유사 청크 반환 |
| 재실행 시 | `shutil.rmtree()`으로 기존 DB 삭제 후 재생성 |

#### 기존 DB 로드 (재사용 시)

```python
persist_path = Path(CHROMA_DIR)
if persist_path.exists() and any(persist_path.iterdir()):
    vectorstore = Chroma(
        collection_name="pm_docs",
        embedding_function=embedding,
        persist_directory=str(persist_path),
    )
else:
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        collection_name="pm_docs",
        persist_directory=str(persist_path),
    )
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
```

### 5단계: 검색 동작 검증

```python
test_query = "리스크 관리 절차"
test_docs = retriever.invoke(test_query)

print(f"쿼리: '{test_query}' → {len(test_docs)}개 청크 반환")
for i, doc in enumerate(test_docs, 1):
    src = os.path.basename(doc.metadata.get("source", ""))
    print(f"\n[청크 {i}] 출처: {src}")
    print(doc.page_content[:200])
```

검색 결과가 0개면 임베딩 또는 인덱싱 과정에 문제가 있다 — 벡터스토어 생성 단계를 재실행한다.

## 파이프라인 파라미터 튜닝 가이드

검색 품질이 만족스럽지 않을 때 조정할 수 있는 주요 파라미터:

| 문제 현상 | 조정 파라미터 | 방향 |
|----------|-------------|------|
| 관련 문서를 못 찾음 | `k` 값 증가 | 3 → 5 |
| 관련 없는 문서가 섞임 | `chunk_size` 감소 | 500 → 300 |
| 컨텍스트가 잘려서 불완전 | `chunk_size` 증가 | 500 → 800 |
| 청크 경계에서 정보 손실 | `chunk_overlap` 증가 | 50 → 100 |
| 검색 결과 다양성 부족 | 검색 유형 변경 | `search_type="mmr"` |

### MMR(Maximal Marginal Relevance) 검색

기본 유사도 검색은 비슷한 청크가 중복 반환될 수 있다. MMR은 **관련성과 다양성을 균형** 있게 고려한다:

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10},
)
```

| 파라미터 | 설명 |
|---------|------|
| `fetch_k=10` | 후보군 10개를 먼저 검색 |
| `k=3` | 후보군에서 다양성을 고려해 3개 선택 |

## 도메인별 적용 예시

### PM(프로젝트 관리) 도메인

```python
DOCS_PATH = "sample_data/pm/pm_docs/"
# 대상: 리스크 관리 절차서, 애자일 스크럼 가이드, 품질 검수 체크리스트 등
```

### 반도체 공정 도메인

```python
DOCS_PATH = "sample_data/semi/semi_docs/"
# 대상: 식각공정 매뉴얼, 증착공정 트러블슈팅, CMP 장비 스펙 등
```

파이프라인 코드는 동일하되, **separators를 도메인 문서 구조에 맞게 조정**하는 것이 핵심이다. 예를 들어 반도체 공정 문서에 표(`|`)가 많다면 표 행 단위 분할을 추가할 수 있다.

## 관련 문서

- [Agentic RAG 구현](./agentic-rag-implementation.md) — 이 파이프라인 위에 조건 분기 그래프를 구축
- [토큰 전략 (문서 분할 상세)](../token_strategy/README.md) — PDF/PPTX/XLSX 등 다양한 포맷의 분할 전략
- [LangGraph 기초](../langgraph/langgraph-basics.md) — StateGraph 기본 개념

## 참고 자료 (References)

- [LangChain Text Splitters 공식 문서](https://python.langchain.com/docs/how_to/#text-splitters)
- [ChromaDB 공식 문서](https://docs.trychroma.com/)
- [OpenAI Embeddings 가이드](https://platform.openai.com/docs/guides/embeddings)
