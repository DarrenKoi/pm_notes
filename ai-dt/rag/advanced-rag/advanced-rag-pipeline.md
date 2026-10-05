---
tags: [rag, pipeline, chromadb, embedding, text-splitting]
level: intermediate
last_updated: 2026-07-16
reviewed_on: 2026-10-04
review_status: partial
document_type: learning
category_major: "AI·DT"
category_middle: "RAG"
category_minor: "고급 RAG·멀티에이전트"
note_kind: "학습"
classified_on: "2026-10-05"
---

# Advanced RAG 파이프라인

> [!info] 검토 — 2026-10-04
> 공식 근거·설치 판본·로컬 검증은 [RAG 정리 기록](../organization-log.md)에 있다. 서비스 구현 완료를 뜻하지 않는 학습 예제다. 시리즈의 문서별 검토 범위와 대기는 목차/정리 기록을 따른다. 실제 사내 데이터·모델 품질·회사 권한은 미확인이다.


> 문서 로딩부터 벡터스토어 구축, Retriever 구성까지 — RAG 시스템의 기반을 단계별로 구현한다


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
| 분할 | `RecursiveCharacterTextSplitter` | 문자 구분자 우선 분할; Markdown AST/표/fence 의미 보존은 별도 |
| 임베딩 | `OpenAIEmbeddings` | 텍스트를 벡터로 변환 |
| 저장 | `ChromaDB` | 로컬 파일 기반 벡터 DB에 영속 저장 |
| 검색 | `as_retriever()` | 유사도 기반 top-k 검색 인터페이스 |

## 어떻게 사용하는가? (How)

### 1단계: 환경 설정

```python
# 별도 임시 Python3.14.2의 확인 판본; 최신/운영 lock을 뜻하지 않음.
# python -m pip install langchain-chroma==1.1.0 chromadb==1.5.9 langchain-community==0.4.2
# python -m pip install langchain-openai==1.6.7 langchain-text-splitters==1.1.3 python-dotenv==1.2.4
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def make_components(env_file=None):
    if env_file is not None:
        load_dotenv(env_file, override=False)  # 기존 환경의 인증/주소를 자동 덮어쓰지 않음
    llm = ChatOpenAI(model=os.environ["LLM_MODEL"], temperature=0)
    embedding = OpenAIEmbeddings(model=os.environ["EMBEDDING_MODEL"])
    return llm, embedding

# 승인된 provider의 키/base_url·model 지원은 별도 확인. 함수 정의만으로 API 요청하지 않음.
```

temperature=0은 출력 변동을 줄이려는 설정이며 결정성/정확도를 보장하지 않는다. 같은 데이터·모델/revision·검색/프롬프트 설정과 반복 분산을 기록한다. 원래 GPT-4.1/text-embedding-3-small은 예시이며 실제 승인된 모델은 환경변수/명시 객체로 받는다.

### 2단계: 문서 로딩 — DirectoryLoader

```python
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader

def load_documents(docs_path: str, recursive: bool = False):
    if type(recursive) is not bool:
        raise ValueError("recursive bool 필요")
    docs = DirectoryLoader(docs_path, glob="*.md", recursive=recursive,
        loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"},
        silent_errors=False, load_hidden=False).load()
    docs = sorted(docs, key=lambda d: str(d.metadata.get("source", "")))
    if not docs or any(not d.page_content.strip() for d in docs):
        raise ValueError("비어 있지 않은 Markdown 입력 필요")
    return docs

def loading_stats(docs):
    return [{"source_name": Path(d.metadata["source"]).name,
             "characters": len(d.page_content)} for d in docs]

# raw_docs = load_documents("sample_data/pm/pm_docs/", recursive=False)  # 준비할 실습 입력
```

**핵심 파라미터:**

| 파라미터 | 설명 |
|---------|------|
| `glob="*.md"` | 기본은 해당 디렉터리만. 하위 폴더는 recursive=True를 명시 |
| `loader_cls=TextLoader` | 원문 그대로 보존 (파싱 변환 없음) |
| `encoding="utf-8"` | 이 예제는 UTF-8 파일 계약. cp949 등은 실제 인코딩을 확인해 설정; 자동 추측 아님 |

### 3단계: 문서 분할 — RecursiveCharacterTextSplitter

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(raw_docs):
    if not raw_docs or any(not d.page_content.strip() for d in raw_docs):
        raise ValueError("빈/공백 문서 분할 불가")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""])
    splits = splitter.split_documents(raw_docs)
    if not splits or any(not s.page_content.strip() for s in splits):
        raise ValueError("사용할 청크 없음")
    return splits

def chunk_stats(splits):
    return {"count": len(splits),
            "mean_characters": sum(len(s.page_content) for s in splits)/len(splits) if splits else None}

# splits = split_documents(raw_docs)
```

**분할 전략 상세:**

| 파라미터 | 값 | 근거 |
|---------|---|------|
| `chunk_size` | 500문자 | 학습용 제안값; 정답/검색 표본으로 측정 |
| `chunk_overlap` | 목표50문자 | 정확한 overlap·의미 보존을 보장하지 않음 |
| `separators` | 헤더/공백 우선, 마지막 "" | 공백 없는 긴 문자열도 분할. 구조/품질은 검증 필요 |

**separators 동작 원리:**

```
우선순위:  "\n## " > "\n### " > "\n\n" > "\n" > " "

문서 내용:
## 1. 리스크 식별        ← "\n## " 기준으로 먼저 분할 시도
리스크를 찾아내는...
### 1.1 브레인스토밍     ← chunk_size 초과 시 "\n### " 기준으로 세분화
팀 전체가 참여하여...
```

헤더 구분자 우선은 분할 지점의 휴리스틱이다. 기본 length_function=len이므로 문자 수이지 token 수가 아니다. 마지막 빈 문자열 구분자가 있어야 긴 비공백 문자열도500문자 이하로 잘린다. 표/코드 fence·제목과 근거의 의미 보존은 보장하지 않는다. 검증용 질문과 기대 근거로 비교한다.

#### chunk_size 선택 가이드

| chunk_size | 장점 | 단점 | 적합한 경우 |
|-----------|------|------|------------|
| 200~300문자 | 작은 검색 단위 후보 | 컨텍스트 부족 가능 | FAQ/정의에서 비교할 후보 |
| 400~600문자 | 중간 크기 후보 | 데이터마다 효과 다름 | 기술 문서에서 비교할 후보 |
| 800~1000문자 | 더 긴 문맥 포함 가능 | 노이즈/입력 예산 증가 가능 | 보고서/논문에서 비교할 후보 |

### 4단계: 벡터스토어 생성 — ChromaDB

```python
import hashlib
import json
from pathlib import Path
from langchain_chroma import Chroma
from chromadb.config import Settings

def build_new_store(splits, embedding, persist_directory: str, embedding_id: str,
                    collection_name: str = "pm_docs"):
    if not splits or any(not d.page_content.strip() for d in splits):
        raise ValueError("비어 있지 않은 청크 목록 필요")
    if any(not isinstance(x, str) or not x.strip() for x in (embedding_id, collection_name)):
        raise ValueError("확인한 embedding id/revision과 collection 필요")
    records = [[d.metadata.get("source", ""), i, d.page_content] for i,d in enumerate(splits)]
    encoded = [json.dumps(r, ensure_ascii=False).encode() for r in records]
    ids = [hashlib.sha256(r).hexdigest() for r in encoded]
    manifest = {"schema": 1, "embedding_id": embedding_id, "collection": collection_name,
                "count": len(splits), "chunks_sha256": hashlib.sha256(b"\n".join(encoded)).hexdigest()}
    target = Path(persist_directory)
    target.mkdir(parents=True, exist_ok=False)  # 기존 디렉터리는 비어 있어도 보호; 자동 삭제 없음
    store = Chroma.from_documents(splits, embedding, ids=ids, collection_name=collection_name,
        persist_directory=str(target), client_settings=Settings(anonymized_telemetry=False))
    if len(store.get()["ids"]) != len(splits):
        raise ValueError("저장된 청크 수 불일치")
    # 성공한 저장만 manifest 작성. 실패한 새 디렉터리는 조사 후 사람이 처리한다.
    with (target/"pipeline-manifest.json").open("x", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return store

# vectorstore = build_new_store(splits, embedding, "새_실습_DB_경로", "확인한_embedding_revision")
# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
```

**ChromaDB 핵심 포인트:**

| 항목 | 설명 |
|------|------|
| `persist_directory` | 로컬 폴더 기반 영속 저장 — 서버 불필요 |
| `collection_name` | 논리적 문서 그룹 구분 |
| `search_kwargs={"k": 3}` | 상위 3개 유사 청크 반환 |
| 재실행 시 | 새 경로에서 생성하거나 manifest 계약을 대조해 재사용; 기존 DB 자동 삭제/업데이트 없음 |

기존 DB 삭제는 중복 방지 정책이 아니다. 새 경로 생성은 기존 자료를 보존하며 실패한 새 DB도 자동 삭제하지 않는다. manifest는 선언한 embedding id/collection/청크 수·입력 digest를 기록한다. 모델 alias의 실제 revision/차원 동일성이나 DB 무결성을 증명하지 않는다. 입력이 변경되면 새 DB를 만들고 비교 후 전환한다. 업데이트/삭제/동시 재색인·서버 인증/백업은 구현하지 않는다.

#### 기존 DB 로드 (재사용 시)

```python
def open_existing_store(persist_directory: str, embedding, expected_embedding_id: str):
    target = Path(persist_directory)
    with (target/"pipeline-manifest.json").open(encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("schema") != 1 or manifest.get("embedding_id") != expected_embedding_id:
        raise ValueError("manifest 판본/embedding 계약 확인 실패")
    store = Chroma(collection_name=manifest["collection"], embedding_function=embedding,
        persist_directory=str(target), create_collection_if_not_exists=False,
        client_settings=Settings(anonymized_telemetry=False))
    if type(manifest.get("count")) is not int or manifest["count"] <= 0 or len(store.get()["ids"]) != manifest["count"]:
        raise ValueError("빈/변경된 collection; 자동 재색인하지 않음")
    return store

# 재사용은 위 성공 manifest가 있는 DB만. 실제 model/revision·차원 동일성은 사용자가 확인한다.
# vectorstore = open_existing_store(확인한_DB_경로, embedding, 확인한_embedding_revision)
```

### 5단계: 검색 동작 검증

```python
def inspect_search(retriever, query: str):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("검색어 필요")
    docs = retriever.invoke(query)
    # 이 요약은 검색 동작 점검이지 정답성/관련성 판정이 아니다. 원문 자동 출력 없음.
    return {"count": len(docs), "sources": [Path(d.metadata.get("source", "미확인")).name for d in docs]}

# inspect_search(retriever, "리스크 관리 절차")
```

검색0개는 collection/필터/데이터/검색 조건을 먼저 확인할 상태다. embedding 오류라고 단정하거나 기존 DB를 삭제하지 않는다. 반환 개수·출처만으로 관련성/정답성을 확인할 수 없으며 실제 기대 근거·답변 검수를 추가한다.

## 파이프라인 파라미터 튜닝 가이드

검색 품질이 만족스럽지 않을 때 조정할 수 있는 주요 파라미터:

| 문제 현상 | 조정 파라미터 | 방향 |
|----------|-------------|------|
| 관련 문서를 못 찾음 | 검색/query·필터·embedding와 k 검토 | 3→5는 비교 후보, recall/노이즈 확인 |
| 관련 없는 문서가 섞임 | chunk/query·reranking 비교 | 500→300도 품질 악화 가능; 측정 필요 |
| 컨텍스트가 잘려서 불완전 | `chunk_size` 증가 | 500 → 800 |
| 청크 경계에서 정보 손실 | `chunk_overlap` 증가 | 50 → 100 |
| 검색 결과 다양성 부족 | 검색 유형 변경 | `search_type="mmr"` |

### MMR(Maximal Marginal Relevance) 검색

기본 유사도 검색은 비슷한 청크가 중복 반환될 수 있다. MMR은 **관련성과 다양성을 균형** 있게 고려한다:

```python
def mmr_retriever(vectorstore):
    return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10})
```

| 파라미터 | 설명 |
|---------|------|
| `fetch_k=10` | 후보군 10개를 먼저 검색 |
| `k=3` | 후보군에서 다양성을 고려해 3개 선택 |

## 도메인별 적용 예시

### PM(프로젝트 관리) 도메인

```python
DOCS_PATH = "sample_data/pm/pm_docs/"
# 준비할 승인된 입력: 리스크 관리 절차서, 애자일 스크럼 가이드, 품질 검수 체크리스트 등
```

### 반도체 공정 도메인

```python
DOCS_PATH = "sample_data/semi/semi_docs/"
# 준비할 승인된 입력: 식각공정 매뉴얼, 증착공정 트러블슈팅, CMP 장비 스펙 등
```

파이프라인 코드는 동일하되, **separators를 도메인 문서 구조에 맞게 조정**하는 것이 핵심이다. 표(`|`)를 행 단위로 나누면 제목·열 이름·단위·주석 관계가 깨질 수 있다. 도메인별 parser/청크의 근거 연결을 검토하고 단순 구분자 추가의 효과를 실측한다.

## 관련 문서

- [Agentic RAG 구현](./agentic-rag-implementation.md) — 이 파이프라인 위에 조건 분기 그래프를 구축
- [토큰 전략 (문서 분할 상세)](../token_strategy/README.md) — PDF/PPTX/XLSX 등 다양한 포맷의 분할 전략
- [LangGraph 기초](../langgraph/langgraph-basics.md) — StateGraph 기본 개념

## 참고 자료 (References)

- [LangChain Text Splitters 공식 문서](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter)
- [ChromaDB 공식 문서](https://docs.trychroma.com/), [LangChain Chroma](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma)
- [DirectoryLoader 공식 소스](https://raw.githubusercontent.com/langchain-ai/langchain-community/main/libs/community/langchain_community/document_loaders/directory.py)
- [OpenAI Embeddings 가이드](https://platform.openai.com/docs/guides/embeddings)
