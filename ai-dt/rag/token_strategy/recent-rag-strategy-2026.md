---
tags: [rag, tokenization, chunking, embedding, hybrid-search, enterprise]
level: advanced
last_updated: 2026-03-14
reviewed_on: 2026-10-04
review_status: partial
document_type: historical_strategy
aliases: [2026년 3월 사내 RAG 전략 메모]
---

# 최근 RAG 전략 정리 (2026) - 사내 구축 관점


> [!info] 먼저 읽을 검토 결과 — 2026-10-04
> 이 문서는 **2026-03-14의 사내 구축 제안 기록**이다. 아래 원문은 당시 판단을 보존하며 현재 운영 사실·최신 전략·승인된 구현 계획을 뜻하지 않는다. 공식 모델 카드/제품 API와 연구의 주장 범위를 대조한 결과를 먼저 읽는다. 사내 API·검색 서버·모델/토크나이저·업무 평가 자료는 실행하지 않았다. HERDR_ENV=1에서 현재 pane_not_found로 Claude 협의가 불가능해 모델 선택/우위·운영 정책·완전 문서 통합 판단은 보류했다.

## 목적과 읽는 방법

문서 추출·구조 청킹·모델 입력 길이·lexical/dense 결합·재정렬 중 무엇을 비교할지 정리한 제안이다. 원문에 있는 BGE-M3 제공, OpenSearch 사용, 외부 전송 제약은 **작성자의 당시 추정**이다. 확인된 사내 서비스 명세가 없으므로 그대로 배포하지 않는다. 제목의 최근/2026은 검증일 현재 최신 보장이 아니다.

[청킹 총론](./overview-chunking-methods.md)은 공통 알고리즘의 대표 학습 문서, 이 메모는 당시 사내 조건에서의 선택 이유/단계/평가 제안이다. 형식별 문서는 입력에서 실제 추출되는 자료와 출처를 다룬다. 역할이 다른 유사 설명을 보존하고, 협의가 필요한 완전 통합은 보류한다.

## 근거로 확인한 모델 범위

다음은 확인일 현재 게시된 공식 카드/페이지의 **명시값**이다. 모델 revision·서빙 설정·실제 한국어 품질·속도·메모리·사내 사용 적합성은 확인하지 않았다. 라이선스 표시와 특정 배포/서비스 이용 조건의 판단은 구분한다.

| 후보와 근거 | 카드의 입력/차원·라이선스 표시 | 적용 전에 확인할 조건 |
|-------------|--------------------------------|-----------------------|
| [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | 8192 tokens·dense 1024차원·MIT·100+ 언어; dense/sparse/multi-vector 기능 | 사내 API가 실제 노출하는 모드/토크나이저/최대 길이; BM25와 learned sparse는 다른 방식 |
| [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | 32K context·최대1024/MRL·Apache-2.0 | query instruction과 pooling/정규화·실제 서버의 입력 제한/출력차원 |
| [Qwen3-Embedding-4B 공식 원문 카드](https://huggingface.co/Qwen/Qwen3-Embedding-4B/raw/main/README.md) | 32K context·최대2560/MRL·Apache-2.0 | 4B가 사내 자료에서 더 좋거나 사용 가능한 GPU에 맞는지는 미측정 |
| [multilingual-e5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct) | 최대512 tokens·MIT·100 언어 안내 | query에 Instruct/Query 형식; 문서 instruction은 불필요하다는 카드 안내 |
| [jina-embeddings-v5-text-small](https://jina.ai/models/jina-embeddings-v5-text-small/) | 2026-02-18 공개 표시·32K·최대1024/MRL·CC-BY-NC-4.0 | 가중치/서비스·계약/사용 목적별 조건을 별도로 확인; 사내라는 이유로 허용/금지를 단정하지 않음 |

Qwen 카드의 instruction 1~5%와 영어 instruction 권고는 **제공자의 평가/권고**이며 한국어 사내 자료의 향상 보장은 아니다. BGE-M3는 query instruction이 필요 없다는 카드 안내이므로 원문의 generic instruction을 모든 모델에 붙이지 않는다. [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)는 query/document 쌍을 평가하는 별도 후보이며 OpenSearch rerank processor와 바로 호환된다고 확인한 것은 아니다. 확인일 2026-10-04.

## 원문 기술 주장에 대한 정정과 적용 조건

### 토큰 길이와 overlap

원문 250~350/300~500/300~600/200~400 tokens와 overlap10~15%는 **미검증 실험 시작값**이다. 공개 모델 최대 길이가 권장 청크 크기나 실제 서버 허용 길이와 같지는 않다. prefix·instruction·special token을 포함한 실제 입력을 해당 모델 토크나이저와 서버 설정으로 확인한다. 초과 입력은 구현에 따라 잘리거나 거부될 수 있으므로 truncation을 조용히 성공으로 취급하지 않는다. 임베딩과 생성 모델의 입력 한도/토크나이저를 따로 확인한다.

[OpenSearch text_chunking](https://docs.opensearch.org/latest/ingest-pipelines/processors/text-chunking/)의 fixed_token_length 기본 standard tokenizer는 단어 토크나이저다. 이를 BGE/Qwen/E5 subword token 수로 간주하지 않는다. rolling 문서는 overlap_rate 유효 범위0~0.5와 권고0~0.2를 구분한다. 또한 max_chunk_limit 초과분이 마지막 청크에 합쳐질 수 있어 token_limit만으로 모든 청크가 모델 한도에 맞는다고 보장하지 않는다. [text_embedding](https://docs.opensearch.org/latest/ingest-pipelines/processors/text-embedding/)의 문서 분할 안내는 특정 사내 모델의 최적 크기를 증명하지 않는다. 확인일2026-10-04; 실제 ingest simulation/서버 검증 미실행.

### Contextual Retrieval과 Late Chunking

[Anthropic 원문](https://www.anthropic.com/engineering/contextual-retrieval)(2024-09-19)은 청크별 문맥 설명을 생성해 임베딩과 BM25 색인 입력에 붙이는 방법을 제안한다. 50~100 tokens는 연구에서 설명한 일반적 길이이며 강제 규격이 아니다. 제공자 실험의 retrieval failure는 1−Recall@20이므로 사내 Recall@10·답변 정확도·항상 개선과 같은 뜻으로 해석하지 않는다. 생성 prefix는 원문과 별도 보존하고 출처/환각/비용을 검사한다. 제목을 metadata에 저장만 한 것과 실제 색인 입력에 넣는 것도 다르다.

[Jina 원문](https://jina.ai/news/late-chunking-in-long-context-embedding-models/)(2024-08-22)의 Late Chunking은 transformer의 **token-level 표현을 얻은 뒤 각 청크 범위로 pooling**하는 방식이다. 전체 문서의 단일 최종 벡터를 나누는 방식이 아니다. 일반 embeddings API가 한 벡터만 반환하면 그 출력만으로 구현할 수 없다. 모델 길이·token 경계/offset·padding/special token·원문 출처를 확인해야 한다.

Qwen의 공식 기본 예제는 last-token pooling이다. **32K라는 이유만으로 Jina의 chunk별 mean pooling이 검증되었다고 볼 수 없다**. 모델별 hidden state/attention/pooling 적합성·서버 노출·동등성·품질을 먼저 실험해야 하며 이 문서에서는 Qwen Late Chunking을 구현/지원 확인으로 표시하지 않는다. PPTX/표/OCR 입력의 비추천도 원문의 실험 우선순위 판단이며 기술적으로 불가능하다는 뜻이 아니다. 확인일2026-10-04.

### Hybrid·RRF·reranker

| 공식 기능 | 확인한 판본/도입 안내 | 원문 해석 보완 |
|-----------|----------------------|----------------|
| [normalization-processor](https://docs.opensearch.org/latest/search-plugins/search-pipelines/normalization-processor/) | 도입2.10 | 서로 다른 점수 척도를 정규화/결합; RRF와 다른 후보 |
| [hybrid query 2.11 문서](https://docs.opensearch.org/2.11/query-dsl/compound/hybrid/) | 2.11 문서에 기능 존재; 현재 유지보수 종료 안내 | 당시 흐름의 근거이며 사내 판본/설정·현행 운영 지원을 확인해야 함 |
| [score-ranker-processor](https://docs.opensearch.org/latest/search-plugins/search-pipelines/score-ranker-processor/) | 도입2.19·RRF | 순위를 융합하는 query/fetch 사이 processor; 학습 reranker가 아님 |
| [rerank processor](https://docs.opensearch.org/latest/search-plugins/search-pipelines/rerank-processor/) | ml_opensearch2.12·by_field2.18 | 결과 재정렬 단계; 모델/connector·대상 필드/지원 유형 설정 필요 |

원문의 “사실상 기본값”, “반드시 추가”, “안정적/최상 균형”은 작성자의 권고이며 공식 문서가 모든 자료에서의 우위를 증명한 것은 아니다. 한국어 Nori는 analyzer 후보, BM25는 점수 함수다. keyword exact는 문자열/normalizer 조건에 따른 매칭이며 본문 BM25만으로 모든 SKU/문서번호 exact 처리가 보장되지 않는다. 의미/원문/exact 필드를 나누는 것은 설계 후보로 보존하고 원 식별자와 정규화 규칙/ACL/문서 버전/중복 ID를 함께 검증한다. 확인일2026-10-04.

## 현재 적용할 때의 확인 순서

1. 승인된 입력·사내 API/서버 판본·모델/revision/토크나이저·차원/길이·권한과 원문 보존 조건을 확인한다. 확인 전에는 원문 BGE-M3 고정 배포안을 운영 사실로 삼지 않는다.
2. 형식별 추출 누락과 구조/출처를 먼저 검사한다. 같은 질의/관련 근거/분할/모델로 lexical/dense/fusion/rerank를 한 단계씩 비교하고 효과와 지연/실패를 기록한다.
3. 원문 100~200개 질문은 초기 평가 계획이다. 대표성·질의 중복/학습 유출·복수 정답·관련도 등급·개정본/ACL·근거 없는 질문을 확인한다. Recall@10/nDCG@10/MRR@10과 답변 근거성 정의를 별도로 정한다.
4. 생성 prefix·Late Chunking·MRL/sparse/multi-vector는 기능과 품질을 검증한 후보만 추가한다. 차원을 변경하면 기존 벡터와 섞지 말고 동일 모델/차원/정규화 계약과 재색인을 확인한다. 이 순서는 검증 체크이며 당시 권고를 새 승인 계획으로 바꾸지 않는다.

사내 최종 모델/우선순위·비용/운영 정책·평가 채택은 Claude 협의와 실측이 없어 보류다. 로컬에서는 원문 보존/두 text 예시·링크·메타데이터만 검사하며 모델/서버 품질을 실행한 것으로 보고하지 않는다.

## 2026-03-14 전략 메모 원문

> [!quote] 작성 당시 기록
> 아래 본문 전체는 원문 그대로다. 현재 읽을 때는 위 검토 결과를 우선한다. “현재”, “이미 제공”, “가장 현실적”, “최신”, “반드시”는 2026-03-14 당시 가정/권고이며 현재 운영의 증거가 아니다.

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
