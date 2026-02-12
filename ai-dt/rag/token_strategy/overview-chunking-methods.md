---
tags: [rag, chunking, tokenization, strategy]
level: intermediate
last_updated: 2026-02-12
status: complete
---

# 청킹 방법론 총론 (Overview of Chunking Methods)

> RAG 시스템에서 사용되는 주요 문서 분할(chunking) 방법론을 비교 정리한다.

## 왜 필요한가? (Why)

- 청크가 너무 크면: 검색 정확도 ↓, 불필요한 정보 포함, 토큰 비용 ↑
- 청크가 너무 작으면: 맥락 손실, 의미 불완전, 검색 노이즈 ↑
- 문서 유형에 맞지 않는 청킹: 테이블이 잘리거나, 슬라이드 맥락이 분리됨

**목표**: 검색 시 의미적으로 완결된(self-contained) 청크를 반환하는 것

## 핵심 개념 (What)

### 1. Fixed-Size Chunking (고정 크기 분할)

가장 단순한 방법. 일정한 토큰/문자 수로 기계적으로 분할.

```python
from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=1000,       # 청크당 최대 문자 수
    chunk_overlap=200,     # 인접 청크 간 겹침
    separator="\n"
)
chunks = splitter.split_text(document_text)
```

| 장점 | 단점 |
|------|------|
| 구현이 단순 | 문장/문단 중간에서 잘림 |
| 일관된 청크 크기 | 의미 단위 무시 |
| 빠른 처리 속도 | 테이블, 리스트 구조 파괴 |

**적합한 경우**: 구조가 없는 평문 텍스트, 빠른 프로토타이핑

### 2. Recursive Character Splitting (재귀적 문자 분할)

LangChain 기본 전략. 여러 구분자를 계층적으로 시도.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]  # 우선순위 순서
)
chunks = splitter.split_text(document_text)
```

**동작 방식**:
1. `\n\n` (문단 구분)으로 먼저 시도
2. 청크가 너무 크면 `\n` (줄바꿈)으로 재분할
3. 그래도 크면 `. ` (문장 구분)으로 재분할
4. 최후에 공백/글자 단위로 분할

| 장점 | 단점 |
|------|------|
| 문단/문장 경계 존중 | 시맨틱 의미를 고려하진 않음 |
| 고정 크기보다 자연스러운 분할 | 구분자 설정에 의존 |
| LangChain 기본 지원 | 테이블 등 구조 데이터에 부적합 |

**적합한 경우**: 일반 텍스트 문서, 보고서, 기사

### 3. Semantic Chunking (의미 기반 분할)

문장 간 임베딩 유사도를 계산하여, 의미가 크게 바뀌는 지점에서 분할.

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",  # or "standard_deviation", "interquartile"
    breakpoint_threshold_amount=95
)
chunks = splitter.split_text(document_text)
```

**동작 방식**:
1. 문장 단위로 분리
2. 각 문장의 임베딩 벡터 계산
3. 인접 문장 간 코사인 유사도 측정
4. 유사도가 급격히 떨어지는 지점에서 분할

| 장점 | 단점 |
|------|------|
| 의미적으로 응집된 청크 생성 | 임베딩 계산 비용 (API 호출) |
| 토픽 전환 지점을 잘 감지 | 처리 속도 느림 |
| 구조 없는 문서에도 효과적 | 청크 크기 불균일 |

**적합한 경우**: 토픽이 자주 바뀌는 문서, 고품질 검색이 필요한 경우

### 4. Document Structure-Based Chunking (구조 기반 분할)

문서의 고유 구조(헤더, 섹션, 슬라이드)를 활용하여 분할.

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = splitter.split_text(markdown_text)
# 각 청크에 헤더 메타데이터가 자동으로 포함됨
```

| 장점 | 단점 |
|------|------|
| 문서 의도에 맞는 자연스러운 분할 | 구조가 없는 문서에 적용 불가 |
| 메타데이터(섹션명) 자동 보존 | 문서 유형별 파서 필요 |
| 검색 시 컨텍스트 품질 높음 | 섹션 크기 편차가 클 수 있음 |

**적합한 경우**: 구조화된 문서 (Word, HTML, Markdown)

### 5. Agentic Chunking (에이전트 기반 분할)

LLM이 직접 문서를 읽고 의미 단위로 분할 결정.

```python
# 개념적 코드 - LLM에게 청킹을 위임
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

prompt = """다음 문서를 의미적으로 완결된 단위로 분할하세요.
각 청크는 독립적으로 이해 가능해야 합니다.
청크 경계를 [SPLIT] 마커로 표시하세요.

문서:
{document}"""

response = llm.invoke(prompt.format(document=text))
chunks = response.content.split("[SPLIT]")
```

| 장점 | 단점 |
|------|------|
| 가장 높은 품질의 의미 분할 | LLM API 비용 매우 높음 |
| 복잡한 문서도 잘 처리 | 처리 속도 매우 느림 |
| 유연한 분할 기준 적용 가능 | 결과 재현성 낮음 |

**적합한 경우**: 소량의 고가치 문서, 품질이 최우선인 경우

### 6. Late Chunking

2024년 Jina AI에서 제안한 방법. 먼저 문서 전체를 임베딩한 후, 토큰 수준에서 청크로 분할.

**핵심 아이디어**: 기존 방식은 "먼저 자르고 → 임베딩"이지만, Late Chunking은 "먼저 임베딩 → 나중에 자른다". 이렇게 하면 각 청크의 임베딩이 문서 전체 컨텍스트를 반영한다.

```python
# Jina Embeddings API를 통한 Late Chunking (개념적)
# 일반 임베딩 모델에서는 직접 구현이 필요

from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("jinaai/jina-embeddings-v2-base-en")
tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v2-base-en")

# 1. 전체 문서를 한 번에 인코딩 (컨텍스트 보존)
inputs = tokenizer(full_document, return_tensors="pt")
outputs = model(**inputs)
token_embeddings = outputs.last_hidden_state  # 모든 토큰의 임베딩

# 2. 사후에 토큰 임베딩을 청크 단위로 평균 풀링
chunk_boundaries = find_chunk_boundaries(full_document)  # 문장/문단 경계
chunk_embeddings = []
for start, end in chunk_boundaries:
    chunk_emb = token_embeddings[0, start:end, :].mean(dim=0)
    chunk_embeddings.append(chunk_emb)
```

| 장점 | 단점 |
|------|------|
| 전체 문맥이 각 청크에 반영됨 | Long-context 모델 필요 |
| 기존 청킹 대비 검색 정확도 ↑ | 구현 복잡도 높음 |
| 짧은 청크에서도 맥락 손실 없음 | 모든 임베딩 모델에서 사용 불가 |

**적합한 경우**: 긴 문서에서 세밀한 검색이 필요한 경우

## 방법론 비교 요약

| 방법 | 품질 | 속도 | 비용 | 구현 난이도 | 추천 문서 유형 |
|------|------|------|------|-------------|---------------|
| Fixed-Size | ⭐⭐ | ⭐⭐⭐⭐⭐ | 무료 | 쉬움 | 평문 텍스트 |
| Recursive | ⭐⭐⭐ | ⭐⭐⭐⭐ | 무료 | 쉬움 | 일반 문서 |
| Semantic | ⭐⭐⭐⭐ | ⭐⭐ | 중간 | 중간 | 토픽 혼합 문서 |
| Structure-Based | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 무료 | 중간 | 구조화된 문서 |
| Agentic | ⭐⭐⭐⭐⭐ | ⭐ | 높음 | 어려움 | 고가치 소량 문서 |
| Late Chunking | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 중간 | 어려움 | 긴 문서 |

## 실무 권장 전략

### 엔지니어링 문서에 대한 권장 조합

```
PDF 보고서     → Structure-Based + Semantic (하이브리드)
PowerPoint     → Slide-Based (슬라이드 단위) + 메타데이터 보강
Excel          → Table-Aware Chunking (테이블 단위)
Word           → Header-Based + Recursive (계층적)
스캔 문서       → OCR → Layout Analysis → Structure-Based
```

자세한 전략은 각 문서 유형별 파일을 참고.

## 참고 자료 (References)

- [LangChain Text Splitters](https://python.langchain.com/docs/how_to/#text-splitters)
- [Unstructured.io](https://unstructured.io/) - 다양한 문서 형식 파싱
- [Jina AI Late Chunking](https://jina.ai/news/late-chunking-in-long-context-embedding-models/)
- [Greg Kamradt - Chunking Strategies](https://www.youtube.com/watch?v=8OJC21T2SL4)
- [Pinecone Chunking Guide](https://www.pinecone.io/learn/chunking-strategies/)

## 관련 문서

- [PDF 토큰화 전략](./pdf-tokenization.md)
- [PowerPoint 토큰화 전략](./pptx-tokenization.md)
- [Excel 토큰화 전략](./xlsx-tokenization.md)
- [Word 토큰화 전략](./docx-tokenization.md)
