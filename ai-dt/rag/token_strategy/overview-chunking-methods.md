---
tags: [rag, chunking, tokenization, strategy]
level: intermediate
last_updated: 2026-02-12
reviewed_on: 2026-10-04
review_status: partial
document_type: learning_note
aliases: [문서 청킹 방법 비교]
category_major: "AI·DT"
category_middle: "RAG"
category_minor: "문서 추출·청킹"
note_kind: "학습"
classified_on: "2026-10-05"
---

# 청킹 방법론 총론 (Overview of Chunking Methods)

> RAG 시스템에서 사용되는 주요 문서 분할(chunking) 방법론을 비교 정리한다.

> [!warning] 단위·판본·검증 조건 — 2026-10-04
> 문자 len과 모델 토큰 수는 다르다. 1000/200·percentile95는 원문의 비교용 설정이며 업무 최적값이 아니다. langchain-text-splitters1.1.3·langchain-experimental0.4.2·numpy2.5.3의 로컬 동작을 확인했다. experimental은 공식 유지보수 종료 상태이므로 기존 인터페이스 검증용이며 새 운영 채택을 권고하지 않는다. 실제 임베딩/LLM·Jina 모델/서버·검색 품질은 검증하지 않았다. 모델 자동 다운로드/외부 요청을 실행하지 않는다.

## 왜 필요한가? (Why)

- 큰 청크는 다른 주제를 섞거나 입력 한도/비용을 늘릴 수 있다. 검색 정확도가 항상 단조롭게 낮아지는 것은 아니다.
- 작은 청크는 관계/근거를 분리할 수 있다. 제목·부모 문서/인접 자료 연결과 실제 평가로 확인한다.
- 문서 유형에 맞지 않는 청킹: 테이블이 잘리거나, 슬라이드 맥락이 분리됨

**목표**: 검색 시 의미적으로 완결된(self-contained) 청크를 반환하는 것

## 핵심 개념 (What)

### 1. Fixed-Size Chunking (고정 크기 분할)

고정 문자/토큰 창은 기준을 명시해야 한다. 아래는 len 기준 문자 창이며 토큰 창이 아니다. 원문 CharacterTextSplitter는 separator 단위를 합치므로 구분자가 없는 긴 입력을1000이하로 자르는 보장이 없다. strict 문자 창은 RecursiveCharacterTextSplitter의 빈 구분자로 비교하고, 기존 줄바꿈 기반 예제도 별도 결과로 유지한다:

```python
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter


def character_chunks(document_text: str) -> dict[str, list[str]]:
    if not isinstance(document_text, str):
        raise TypeError("문자열 입력이 필요합니다")
    strict = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=[""],
        length_function=len, strip_whitespace=False,
    )
    paragraph = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separator="\n", length_function=len,
    )
    return {"fixed_characters": strict.split_text(document_text),
            "separator_groups": paragraph.split_text(document_text)}
```

| 장점 | 단점 |
|------|------|
| 구현이 단순 | 문장/문단 중간에서 잘림 |
| strict 창에서는 상한/겹침을 정의; 마지막 조각은 짧을 수 있음 | 의미 단위 무시 |
| 모델 추론 없이 분할; 실제 지연은 측정 대기 | 테이블, 리스트 구조 파괴 |

**적합한 경우**: 구조가 없는 평문 텍스트, 빠른 프로토타이핑

### 2. Recursive Character Splitting (재귀적 문자 분할)

공식 안내의 일반 텍스트 시작 후보다. 여러 구분자를 순서대로 시도하며 모델/API가 항상 이 전략을 자동 적용하는 것은 아니다. overlap200은 목표 겹침이며 모든 경계에 정확히200이 보장되지 않는다.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter


def recursive_chunks(document_text: str) -> list[str]:
    if not isinstance(document_text, str):
        raise TypeError("문자열 입력이 필요합니다")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(document_text)
```

**동작 방식**:
1. `\n\n` (문단 구분)으로 먼저 시도
2. 청크가 너무 크면 `\n` (줄바꿈)으로 재분할
3. 그래도 크면 `. ` (문장 구분)으로 재분할
4. 최후에 공백/글자 단위로 분할

| 장점 | 단점 |
|------|------|
| 문단/문장 경계 존중 | 시맨틱 의미를 고려하진 않음 |
| 구분자 경계를 우선하는 분할 | 구분자 설정에 의존 |
| LangChain 기본 지원 | 테이블 등 구조 데이터에 부적합 |

**적합한 경우**: 일반 텍스트 문서, 보고서, 기사

### 3. Semantic Chunking (의미 기반 분할)

이 구현은 정규식으로 나눈 문장과 주변 buffer(기본1)를 묶어 임베딩하고 인접 cosine distance가 선택한 분포 임계값을 넘는 곳에서 나눈다. percentile95는 거리 분포의95분위이지 정답률95%가 아니다. 한국어 종결/표/코드의 분리·출력 길이와 의미 응집을 별도로 평가해야 한다. 기존 experimental0.4.2의 동작을 검증한 예제이며 유지보수 종료 후 대체 패키지 선택은 Claude 협의 대기다.

```python
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker


def semantic_chunks(document_text: str, embeddings: Embeddings) -> list[str]:
    if not isinstance(document_text, str):
        raise TypeError("문자열 입력이 필요합니다")
    if not document_text.strip():
        return []
    splitter = SemanticChunker(
        embeddings, breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
    )
    return splitter.split_text(document_text)

# caller가 검증한 embeddings를 전달한다. BAAI/bge-m3는 원문의 로컬 모델 후보다.
# HuggingFaceEmbeddings(model_name=...) 초기화만으로 로컬파일/오프라인 준비가 보장되지는 않는다.
```

**동작 방식**:
1. 문장 단위로 분리
2. 문장+주변 buffer 묶음의 임베딩 계산
3. 인접 묶음의 cosine distance 계산
4. 선택한 임계값보다 큰 거리의 경계에서 분할

| 장점 | 단점 |
|------|------|
| 의미 변화 기반 후보 생성; 응집은 실측 | 임베딩 추론 비용(로컬/외부 배치 조건에 따름) |
| 토픽 경계 탐지 비교 가능 | 추론 배치/모델에 따른 지연 |
| 구조 없는 문서에도 적용 후보 | 청크 크기 불균일 |

**적합한 경우**: 토픽이 자주 바뀌는 문서, 고품질 검색이 필요한 경우

### 4. Document Structure-Based Chunking (구조 기반 분할)

헤더·섹션·슬라이드를 추출한 구조를 활용한다. 아래 API는 Markdown만 처리하며 Word/HTML 파서가 아니다. Header metadata와 출처를 유지한 뒤 큰 section을 문자 재분할한다. strip_headers=False로 헤더 본문을 보존하지만 이 splitter는 공백/줄바꿈을 정규화하므로 원문 byte 보존 파서는 아니다. OCR/변환으로 헤더가 잘못 추출되면 분할도 영향을 받는다.

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def markdown_chunks(markdown_text: str, source: str) -> list[Document]:
    if not isinstance(markdown_text, str) or not isinstance(source, str) or not source.strip():
        raise ValueError("Markdown 문자열과 출처가 필요합니다")
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    sections = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False,
    ).split_text(markdown_text)
    for section in sections:
        section.metadata["source"] = source
    return RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200,
    ).split_documents(sections)
```

| 장점 | 단점 |
|------|------|
| 문서 의도에 맞는 자연스러운 분할 | 구조가 없는 문서에 적용 불가 |
| 메타데이터(섹션명) 자동 보존 | 문서 유형별 파서 필요 |
| 제목/출처를 검색 결과와 함께 전달 가능 | 섹션 크기 편차가 클 수 있음 |

**적합한 경우**: 구조화된 문서 (Word, HTML, Markdown)

### 5. Agentic Chunking (에이전트 기반 분할)

LLM이 경계를 제안하는 방식이다. 원문의 ChatOpenAI·Kimi-K2.5·사내 OpenAI-compatible endpoint/키는 검증되지 않은 후보 설정이다. 특정 모델/서버가 실제 설치되었거나 호환된다고 단정하지 않는다. 예제는 명시 callback으로 경계를 받으며 모델 응답으로 원문을 다시 쓰지 않는다. 토큰 길이/재시도·승인된 배치·prompt injection·LLM 비용/품질은 별도 계약이다.

```python
from collections.abc import Callable


def agentic_chunks(text: str, propose_boundaries: Callable[[str], list[int]]) -> list[str]:
    if not isinstance(text, str):
        raise TypeError("원문 문자열이 필요합니다")
    if not text:
        return []
    # callback은 원문 문자 인덱스의 끝 경계를 반환한다. 모델 호출/정책은 caller가 준비한다.
    ends = propose_boundaries(text)
    if not isinstance(ends, list) or not ends or any(type(x) is not int for x in ends):
        raise ValueError("끝 경계 정수 list가 필요합니다")
    if ends[-1] != len(text) or any(b <= a for a, b in zip([0] + ends, ends)):
        raise ValueError("증가하는 경계로 원문 전체를 덮어야 합니다")
    return [text[a:b] for a, b in zip([0] + ends, ends)]

# 원문의 [SPLIT] 답변 문자열 방식은 모델이 원문을 다시 쓰거나 마커를 잘못 쓰면 손실한다.
# 제안 경계를 검증하고 원문에서 직접 slice한다. 좋은 의미 경계인지는 별도 평가한다.
```

| 장점 | 단점 |
|------|------|
| LLM 기준으로 경계 제안 가능; 품질 우위 미확인 | 모델 추론/토큰·운영 비용 추가 |
| 복잡한 구조에 시도 가능; 누락/환각 검사 필요 | 모델/문서 길이/호출 수에 따른 지연 |
| 유연한 분할 기준 적용 가능 | 결과 재현성 낮음 |

**적합한 경우**: 소량의 고가치 문서, 품질이 최우선인 경우

### 6. Late Chunking

Jina 저자들의2024-08-22 소개와2024-09-07 논문(v3:2025-07-07)에서 확인한 방법이다. 문서를 먼저 하나의 문서 벡터로 평균한 뒤 쪼개는 것이 아니다. 지원 context 길이 안의 전체 token을 transformer로 인코딩한 **token hidden states**에서 청크 경계를 적용하고 각 범위를 mean pooling한다.

**핵심 아이디어**: 일반 방식은 청크별 독립 인코딩/풀링, late 방식은 full-context token 인코딩→경계별 풀링이다. 모델 최대 길이를 넘어 잘린 문서는 전체 문맥을 반영하지 못한다. 원문의 find_chunk_boundaries가 문자 인덱스인지 token 인덱스인지 정의되지 않은 문제를 바로잡아, 아래 함수는 준비된 token 범위만 받는다. tokenizer offset·padding/special token 제외·모델의 정규화/metric을 검증해야 한다.

```python
import numpy as np


def late_mean_pool(token_embeddings: np.ndarray, token_spans: list[tuple[int, int]],
                   content_mask: list[int]) -> list[np.ndarray]:
    # 한 번의 full-context encoder 출력 T x D를 받는다. tokenizer/encoder는 호출하지 않는다.
    raw = np.asarray(token_embeddings)
    if not np.isrealobj(raw):
        raise ValueError("실수 token embedding이 필요합니다")
    values = np.asarray(raw, dtype=float)
    if values.ndim != 2 or not all(values.shape) or not np.isfinite(values).all():
        raise ValueError("유한한 T x D 행렬이 필요합니다")
    if len(content_mask) != len(values) or any(type(x) is not int or x not in (0, 1) for x in content_mask):
        raise ValueError("token마다 0/1 content mask가 필요합니다")
    mask = np.asarray(content_mask, dtype=bool)
    result = []
    for start, end in token_spans:
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(values):
            raise ValueError("반열린 token 인덱스 범위를 확인하세요")
        selected = values[start:end][mask[start:end]]
        if not len(selected):
            raise ValueError("내용 token이 없는 범위입니다")
        pooled = selected.mean(axis=0)
        if not np.isfinite(pooled).all():
            raise ValueError("pooling 결과도 유한해야 합니다")
        result.append(pooled)
    return result

# 원문의 jinaai/jina-embeddings-v2-base-en + AutoTokenizer/AutoModel은 후보 흐름이다.
# 모델 revision/입력한도/eval·inference 모드·pooling/정규화 정책을 먼저 검증한다.
# 문자/문장 경계를 tokenizer offset mapping으로 동일 encoder token 인덱스에 대응시킨다.
# content_mask는 attention padding과 special token을 함께 제외한다.
```

| 장점 | 단점 |
|------|------|
| 전체 문맥이 각 청크에 반영됨 | Long-context 모델 필요 |
| 논문 평가에서 개선 보고; 본 corpus 우위 미확인 | 구현 복잡도 높음 |
| 짧은 청크 표현에 주변 문맥 포함 가능; 손실0 보장 아님 | 모든 임베딩 모델에서 사용 불가 |

**적합한 경우**: 긴 문서에서 세밀한 검색이 필요한 경우

## 방법론 비교 요약

| 방법 | 분할/출력 기준 | 추가 계산과 조건 | 비교할 입력 |
|---|---|---|---|
| Fixed-Size | 문자/토큰 창을 명시 | 로컬 분할도 운영 자원 비용 존재; 원문 separator 방식과 strict 창 구분 | 평문 기준선 |
| Recursive | 구분자 순서·문자 길이 | 문자 한도를 토큰 한도로 읽지 않음 | 일반 보고서/기사 |
| Semantic | 문장 buffer 임베딩 거리/임계값 | 추론·불균일 길이·한국어 문장 분리 검증 | 토픽 혼합 문서 |
| Structure-Based | 추출된 헤더/섹션 metadata | 형식 파서/큰 section 후분할 필요 | 구조화된 문서 |
| Agentic | 모델이 제안한 경계 | 원문 slice/누락·비용·재현성 평가 | 소량 고가치 후보 |
| Late Chunking | full-context token states 경계별 pooling | 지원 모델/토큰 대응·context 한도/메모리 검증 | 긴 문맥 관계 후보 |

원문의 별점/무료 표는 공통 평가 세트와 측정 근거가 없어 우열 근거로 제거했다. 여섯 방법의 목적과 비교 축은 유지한다. 분할 기준과 모델 계산/실행 자원 비용을 나누어 평가한다.

## 실무 권장 전략

### 엔지니어링 문서에 대한 권장 조합

```
PDF 보고서     → Structure-Based + Semantic (하이브리드)
PowerPoint     → Slide-Based (슬라이드 단위) + 메타데이터 보강
Excel          → Table-Aware Chunking (테이블 단위)
Word           → Header-Based + Recursive (계층적)
스캔 문서       → OCR → Layout Analysis → Structure-Based
```

위 조합은 원문 제안으로 보존한 비교 후보다. 파서가 실제 읽는 텍스트/표/그림과 읽기 순서·출처를 먼저 확인한다. 공통 질의/근거 평가에서 추출 누락·chunk 크기 분포·recall·답변 근거·지연/비용을 비교하며 최적 크기/모델은 협의/실측 전 미확인이다. 자세한 전략은 각 문서 유형별 파일을 참고한다.

## 참고 자료 (References)

- [LangChain Text Splitters](https://docs.langchain.com/oss/python/integrations/splitters/index)
- [Unstructured.io](https://unstructured.io/) - 다양한 문서 형식 파싱
- [Jina AI Late Chunking](https://jina.ai/news/late-chunking-in-long-context-embedding-models/)
- [Greg Kamradt - Chunking Strategies](https://www.youtube.com/watch?v=8OJC21T2SL4)
- [Pinecone Chunking Guide](https://www.pinecone.io/learn/chunking-strategies/)

출처 확인일 **2026-10-04**. rolling 문서와 실제 설치판본을 구분한다. 자료 소개 링크만으로 모델 품질/지원/사내 운영을 확인하지 않는다. Greg/Pinecone 링크는 원문의 추가 읽기이며 이번 API 수정 근거로 삼지 않았다.

- [Character](https://docs.langchain.com/oss/python/integrations/splitters/character_text_splitter) · [Recursive](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter) · [Markdown](https://docs.langchain.com/oss/python/integrations/splitters/markdown_header_metadata_splitter)
- [Semantic 일차 소스](https://raw.githubusercontent.com/langchain-ai/langchain-experimental/main/libs/experimental/langchain_experimental/text_splitter.py) · [experimental 유지보수 종료](https://github.com/langchain-ai/langchain-experimental/issues/87)
- [Late Chunking 논문v3](https://arxiv.org/abs/2409.04701v3) — token encoding/pooling 순서 및 평가 범위 근거

## 관련 문서

- [PDF 토큰화 전략](./pdf-tokenization.md)
- [PowerPoint 토큰화 전략](./pptx-tokenization.md)
- [Excel 토큰화 전략](./xlsx-tokenization.md)
- [Word 토큰화 전략](./docx-tokenization.md)
