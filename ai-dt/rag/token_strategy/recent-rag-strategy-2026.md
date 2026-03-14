---
tags: [rag, tokenization, chunking, embedding, hybrid-search, enterprise]
level: advanced
last_updated: 2026-03-14
status: complete
---

# 최근 RAG 전략 정리 (2026) - 사내 구축 관점

> 2024년 하반기부터 2026년 3월 14일까지 공개된 모델 카드, 공식 문서, 연구 글을 기준으로, **사내 구축형 RAG**에서 바로 적용 가능한 토큰화/청킹/임베딩 전략을 정리한다.

## 전제: 이 저장소 기준으로 추정한 사내 조건

이 문서는 아래 조건을 전제로 작성했다.

- OpenSearch 기반 검색을 이미 사용하거나 검토 중이다.
- 외부 SaaS에 원문을 보내기 어렵고, **로컬 또는 사내 API(OpenAI-compatible)** 를 선호한다.
- 사내에서 **`bge-m3` embedding API를 이미 제공** 하며, 현재 기본 임베딩 모델로 사용 중이다.
- 문서는 한국어/영어가 섞인 엔지니어링 문서(PDF, PPTX, XLSX, DOCX)가 많다.
- 정확한 코드, SKU, 모델명, 문서번호 같은 **lexical match** 가 중요하다.
- DRM/스캔 문서 같은 비정형 입력도 일부 존재한다.

이 가정이 다르면 권장안도 달라진다. 특히 공개 클라우드 사용 가능 여부와 GPU 여유가 가장 큰 분기점이다.

## 한 줄 결론

지금 사내 RAG에서 가장 현실적인 기본선은 다음 조합이다.

1. **구조 기반 청킹 + 토큰 수 기준 보정**
2. **사내 `bge-m3` API + BM25 하이브리드 검색**
3. **상위 후보에 대한 reranker 추가**
4. 긴 문서에만 **Contextual chunking** 또는 **Late chunking** 을 선택적으로 적용

반대로, 아래 두 가지는 기본선으로 두지 않는 편이 낫다.

- 모든 문서에 LLM 기반 agentic chunking 적용
- dense embedding 하나로 SKU/문서번호/버전 문자열까지 해결하려는 접근

## 최근 흐름에서 실제로 바뀐 점

### 1. 청킹은 이제 "작게 자르기"보다 "문맥을 남기며 자르기"가 핵심

2024년 이후의 핵심 변화는 단순 fixed-size chunking 자체가 아니라, **청크가 잃어버리는 문맥을 어떻게 복구할지** 에 있다.

- Anthropic의 **Contextual Retrieval**(2024-09-19)은 각 청크 앞에 50~100 토큰 정도의 짧은 문맥 설명을 덧붙이는 방식을 제안했다.
- Jina의 **Late Chunking**(2024-08-22)은 문서를 먼저 긴 컨텍스트로 인코딩한 뒤, 나중에 chunk boundary별로 pooling 해서 chunk embedding을 만든다.

즉 최근 전략은 다음 둘 중 하나다.

- 청크를 만들기 전에 문서 구조를 최대한 보존한다.
- 청크를 만든 뒤에도 상위 문맥을 prefix나 long-context embedding으로 다시 주입한다.

### 2. embedding 모델은 "문장 임베딩"보다 "검색용 retrieval model" 중심으로 이동

최근 임베딩 모델은 단순 sentence similarity보다 아래 속성이 중요해졌다.

- **instruction-aware**: query instruction을 붙일수록 성능이 좋아짐
- **long context**: 8K~32K 문서 입력 지원
- **multilingual**: 한국어/영어 혼합 처리
- **MRL(Matryoshka)**: 저장 비용에 맞춰 embedding dimension을 줄일 수 있음
- **dense + sparse + rerank 조합 지원**

기업 환경에서는 이 변화가 중요하다. 이유는 벡터 품질보다도, **저장비용, 검색지연, 라이선스, 배포 방식** 이 실제 제약이기 때문이다.

### 3. hybrid + rerank가 사실상 기본값이 됨

OpenSearch 공식 문서 기준으로도 hybrid search는 이제 부가 기능이 아니라 기본 전략에 가깝다.

- `normalization-processor`: OpenSearch 2.10 도입
- `hybrid` query: OpenSearch 2.11부터 공식 흐름
- `score-ranker-processor`(RRF): OpenSearch 2.19 도입
- `rerank` processor: OpenSearch 2.12+

사내 문서 검색에서는 dense 단독보다 다음 조합이 안정적이다.

`BM25(or Nori 기반 lexical) + dense embedding + reranker`

## 토큰화/청킹 전략 권장안

### 권장 1. 문자 수가 아니라 "embedding 모델 tokenizer 기준 토큰 수"로 자르기

이유:

- embedding 모델은 최대 입력 길이를 넘기면 truncation이 발생한다.
- OpenSearch 공식 문서도 긴 문서를 embedding 전에 분할하라고 명시한다.
- 같은 500자라도 한국어, 영어, 코드 스니펫, 표는 실제 토큰 수가 크게 다르다.

실무 기준:

- `multilingual-e5-large-instruct` 같은 **512 토큰 계열**: 청크 본문 250~350 tokens
- `bge-m3` 같은 **8K 계열**: 청크 본문 300~500 tokens
- `Qwen3-Embedding-*` 같은 **32K 계열**:
  - 일반 전략: 300~600 tokens
  - late chunking 전략: 전체 문서를 길게 인코딩한 뒤 200~400 token 단위로 후처리

### 권장 2. 문서 구조를 먼저 자르고, 토큰 제한은 두 번째 단계에서 맞추기

이 저장소의 기존 방향과도 일치한다.

- PDF/DOCX: 제목, 섹션, 표, 리스트 단위로 먼저 분리
- PPTX: 슬라이드 단위, 필요 시 슬라이드 내부 객체 단위 분리
- XLSX: 시트 요약 + 테이블/행 그룹 단위

그 다음에만 토큰 기준 보정을 건다.

가장 실용적인 형태는 다음과 같다.

1. 문서 구조 단위 분할
2. 각 단위를 embedding tokenizer로 길이 측정
3. 초과 시 recursive split 또는 paragraph split
4. 부모 메타데이터 유지

예:

```text
[문서명]
[섹션 경로: 3.2 Inverter Fault Handling]
[페이지/슬라이드 번호]
[본문 chunk]
```

이 prefix 메타데이터는 짧지만 retrieval 성능에 크게 기여한다.

### 권장 3. overlap은 작게, 대신 부모 컨텍스트를 메타데이터로 유지

최근 전략은 overlap을 무조건 크게 두지 않는다.

- OpenSearch `text_chunking` 문서는 overlap을 0~0.2 범위로 권장한다.
- overlap을 크게 잡으면 index 크기와 검색 노이즈가 늘어난다.

사내 문서에서는 다음이 더 낫다.

- overlap: 10~15% 정도만 사용
- 대신 metadata에 `document_title`, `section_path`, `page`, `slide`, `table_name` 보존
- 표/슬라이드처럼 독립 의미가 약한 경우에는 **context prefix** 추가

### 권장 4. 한국어 + 영문 + 식별자 문자열은 "필드 분리"로 해결

dense embedding 하나로는 다음 항목이 불안정하다.

- 장비 모델명
- 에러 코드
- 도면 번호
- SKU
- 버전 문자열
- 약어/사내 용어

따라서 인덱스 필드를 분리하는 것이 좋다.

- `content_semantic`: 자연어 중심 정제 본문
- `content_lexical`: 원문 보존 본문
- `content_exact`: 식별자 정규화 필드(keyword or exact)
- `metadata.*`: 문서/섹션/페이지/작성일/작성자

한국어 검색은 기존처럼 `nori` 계열 analyzer를 유지하고, 식별자 계열은 `keyword` 또는 별도 exact 필드로 관리하는 쪽이 안전하다.

### 권장 5. 긴 보고서에는 Contextual chunking 또는 Late chunking만 선택 적용

두 방법은 비슷해 보이지만 적용 지점이 다르다.

#### Contextual chunking

- 각 chunk 앞에 "이 청크가 문서 전체에서 무엇을 의미하는지" 짧게 붙인다.
- 구현이 간단하다.
- 현재 사내 OpenAI-compatible LLM이 있으면 바로 적용 가능하다.
- Anthropic은 Contextual Embeddings + Contextual BM25 조합이 retrieval failure를 더 줄였다고 보고했다.

추천 대상:

- 문단만 보면 주어가 빠지는 재무/정책/설계 보고서
- 섹션 제목이 의미를 많이 좌우하는 문서

#### Late chunking

- 전체 문서를 길게 embedding한 뒤, chunk boundary별로 pooling 한다.
- long-context embedding 모델이 필요하다.
- 구현 복잡도가 더 높다.
- 긴 문서에서 작은 청크가 상위 문맥을 잃는 문제에 강하다.

추천 대상:

- 긴 PDF/DOCX 보고서
- 앞 문단을 알아야 의미가 생기는 기술 문서

비추천 대상:

- PPTX, 표 중심 문서, OCR 품질이 불안정한 문서

## embedding 모델 선택 가이드

### 옵션 A. `BAAI/bge-m3` - 가장 무난한 사내 기본선

장점:

- MIT License
- 100+ languages
- 최대 8192 tokens
- dense / sparse / multi-vector 기능을 한 모델 계열에서 다룸
- query instruction 없이도 시작 가능

언제 좋은가:

- 한국어/영어 혼합 문서가 많다.
- dense만이 아니라 sparse/lexical 확장성도 보고 싶다.
- 한 모델로 실험 폭을 넓히고 싶다.

주의:

- sparse와 multi-vector까지 제대로 쓰려면 파이프라인 복잡도가 올라간다.
- OpenSearch에 바로 붙일 때는 dense만 먼저 쓰고, lexical은 BM25로 유지하는 것이 현실적이다.

추천 사용 방식:

- 현재 회사 표준이 이미 `bge-m3` API라면, **모델 교체보다 chunking/hybrid/rerank 개선이 우선** 이다.
- 1차 배포: 사내 `bge-m3` API + OpenSearch BM25 hybrid + reranker
- 2차 실험: 일부 데이터셋에서만 sparse 또는 ColBERT 계열 비교

### 옵션 B. `Qwen/Qwen3-Embedding-0.6B` 또는 `Qwen/Qwen3-Embedding-4B` - 최근형 long-context dense baseline

장점:

- Apache-2.0
- 32K context
- instruction-aware
- MRL 지원
- 공식 Qwen 시리즈 reranker와 짝을 맞추기 쉽다

언제 좋은가:

- 긴 문서 비중이 높다.
- late chunking 또는 긴 context retrieval 실험을 하고 싶다.
- index 크기와 latency를 맞추기 위해 embedding dimension을 줄이는 실험이 필요하다.
- 이미 쓰고 있는 `bge-m3` 대비 **긴 문서군에서만 추가 이득이 있는지 비교** 하고 싶다.

권장 선택:

- 보수적 시작: `Qwen3-Embedding-0.6B`
- 성능 우선, GPU 여유 있음: `Qwen3-Embedding-4B`

같이 볼 모델:

- `Qwen/Qwen3-Reranker-0.6B`

실무 포인트:

- Qwen 팀은 instruction 사용 시 다수 작업에서 1~5% 개선을 관측했다고 밝힌다.
- multilingual 환경에서는 instruction을 영어로 쓰는 것을 권장한다.

### 옵션 C. `intfloat/multilingual-e5-large-instruct` - 보수적이고 안정적인 baseline

장점:

- MIT License
- 100개 언어 지원
- instruction 기반 retrieval가 명확하다
- 현재도 비교군으로 쓰기 좋다

언제 좋은가:

- 실험군이 너무 많아지는 것을 막고 싶다.
- 먼저 안정적 baseline을 만들고 싶다.
- chunk를 짧게 유지하는 운영 정책이 가능하다.

주의:

- 입력 길이가 512 토큰 계열이라 긴 문서 대응력은 최근 long-context 모델보다 제한적이다.
- chunk sizing을 더 엄격하게 해야 한다.

### 옵션 D. Jina 최신 long-context 계열 - 성능 실험용, 기본선은 아님

2026-02-18 공개된 `jina-embeddings-v5-text-small`은 32K context와 Matryoshka를 제공한다. 다만 공식 페이지 기준 라이선스가 `CC-BY-NC-4.0` 이므로, **일반적인 사내 상용 시스템의 기본 후보로 두기 어렵다**.

정리하면:

- 상용/사내 기본선: Qwen3, BGE-M3, E5
- 비교 실험용: Jina 계열

## 사내 구축용 추천 조합

### 시나리오 1. 지금 가장 현실적인 기본안

- Chunking: 구조 기반 + token-aware 보정
- Embedding: **사내 `bge-m3` API 고정**
- Lexical: OpenSearch BM25 + 한국어 analyzer
- Fusion: OpenSearch hybrid + RRF
- Rerank: Qwen3 또는 cross-encoder 계열 reranker

이 구성이 좋은 이유:

- 이미 제공 중인 사내 표준 모델을 그대로 쓰므로 도입 마찰이 가장 적다.
- 구현 난이도와 성능의 균형이 좋다.
- SKU/코드/약어와 의미 검색을 동시에 잡는다.
- 현재 저장소의 OpenSearch 중심 구조와 가장 잘 맞는다.

### 시나리오 2. 긴 보고서가 많고 GPU가 어느 정도 있는 경우

- Chunking: 구조 기반
- Long-doc strategy: contextual chunking 먼저 적용
- 그 다음 후보 실험: late chunking
- Embedding: 기본은 사내 `bge-m3`, 비교 실험은 `Qwen/Qwen3-Embedding-4B`
- Rerank: 반드시 추가

이 경우 핵심은 chunk size를 키우는 것이 아니라, **문서 상위 문맥을 각 chunk에 어떻게 전달할지** 다.

### 시나리오 3. 보안 제약이 강하고 빠른 1차 구축이 필요한 경우

- Chunking: 기존 문서 유형별 구조 청킹 유지
- Embedding: 사내 `bge-m3` API
- Retrieval: OpenSearch BM25 + dense hybrid
- 개선 포인트: ambiguity가 큰 문서에만 contextual prefix 추가

장점:

- 도입 리스크가 낮다.
- 짧은 chunk 위주 운영이 명확하다.
- 평가셋을 빨리 만들 수 있다.

## 구현 우선순위

### 1단계. 먼저 바꿔야 하는 것

1. 문서별 구조 청킹을 기본값으로 고정
2. embedding tokenizer 기준 token length 측정 추가
3. exact match용 필드 분리
4. hybrid 검색과 reranker를 기본 파이프라인으로 고정

### 2단계. 그 다음 추가할 것

1. section/title/page prefix 자동 부착
2. ambiguous chunk에 contextual prefix 생성
3. query instruction template 정립

예:

```text
Represent this query for retrieving relevant internal engineering documents:
{user_query}
```

### 3단계. 충분히 평가한 뒤 할 것

1. late chunking
2. dense+sparse 통합 모델 실험
3. multi-vector / late interaction retrieval

## 평가 기준

사내 환경에서는 공개 벤치마크보다 **내부 질의셋** 이 더 중요하다.

최소한 아래는 측정하는 편이 좋다.

- Recall@10
- nDCG@10
- MRR@10
- answer grounding success rate
- query latency(P50/P95)
- index size
- chunk 수 증가율

권장 방식:

- 실제 사내 질문 100~200개 수집
- 각 질문에 정답 문서/섹션 표시
- 모델 교체보다 먼저 chunking 전략을 고정해서 비교

## 최종 권장안

사내 조건에서 가장 먼저 구현할 전략은 아래다.

1. **구조 기반 청킹**
2. **토큰 수 기준 chunk 보정**
3. **사내 `bge-m3` API + BM25 hybrid**
4. **reranker 추가**
5. 긴 문서에만 **contextual prefix**

그리고 아래 순서로 확장하는 것이 가장 안전하다.

1. **사내 `bge-m3` API를 baseline으로 고정**
2. `Qwen3-Embedding-0.6B`로 long-context dense 비교
3. 특정 긴 문서군에 late chunking 적용

즉 현재 조건에서는 "어떤 embedding 모델을 쓸까"보다, 이미 제공되는 `bge-m3`를 기준으로 아래를 먼저 최적화하는 편이 더 실용적이다.

- chunk boundary
- context prefix
- lexical field 설계
- hybrid fusion
- reranker

즉, "최신 전략"의 핵심은 무조건 더 큰 모델이 아니다. 사내 RAG에서는 오히려 아래가 성능을 더 크게 바꾼다.

- chunk boundary 품질
- lexical 필드 분리
- hybrid fusion
- reranking
- chunk contextualization

## 참고 자료

- Anthropic, *Introducing Contextual Retrieval* (2024-09-19)  
  https://www.anthropic.com/research/contextual-retrieval
- Jina AI, *Late Chunking in Long-Context Embedding Models* (2024-08-22)  
  https://jina.ai/news/late-chunking-in-long-context-embedding-models/
- OpenSearch Docs, *Text chunking processor*  
  https://docs.opensearch.org/latest/ingest-pipelines/processors/text-chunking/
- OpenSearch Docs, *Text embedding processor*  
  https://docs.opensearch.org/latest/ingest-pipelines/processors/text-embedding/
- OpenSearch Docs, *Hybrid search*  
  https://docs.opensearch.org/latest/vector-search/ai-search/hybrid-search/index/
- OpenSearch Docs, *Score ranker processor*  
  https://docs.opensearch.org/latest/search-plugins/search-pipelines/score-ranker-processor/
- Hugging Face Model Card, `BAAI/bge-m3`  
  https://huggingface.co/BAAI/bge-m3
- Hugging Face Model Card, `Qwen/Qwen3-Embedding-0.6B`  
  https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- Hugging Face Model Card, `Qwen/Qwen3-Embedding-4B`  
  https://huggingface.co/Qwen/Qwen3-Embedding-4B
- Hugging Face Model Card, `Qwen/Qwen3-Reranker-0.6B`  
  https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
- Hugging Face Model Card, `intfloat/multilingual-e5-large-instruct`  
  https://huggingface.co/intfloat/multilingual-e5-large-instruct
- Jina AI Model Page, `jina-embeddings-v5-text-small` (2026-02-18)  
  https://jina.ai/models/jina-embeddings-v5-text-small

## 관련 문서

- [문서 토큰화 전략 README](./README.md)
- [청킹 방법론 총론](./overview-chunking-methods.md)
- [PDF 토큰화 전략](./pdf-tokenization.md)
- [OpenSearch 하이브리드 검색](../opensearch/hybrid-search.md)
