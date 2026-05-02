---
tags: [llm, rag, normalization, retrieval, metadata, glossary, embeddings]
level: advanced
last_updated: 2026-05-02
---

# LLM과 RAG에서의 정규화

> RAG에서 정규화는 검색 성능만의 문제가 아니다. LLM이 어떤 사실을 어떤 출처와 어떤 개념으로 이해해야 하는지를 안정화하는 작업이다.

## 왜 RAG에 정규화가 필요한가?

RAG는 문서를 검색해서 LLM의 컨텍스트에 넣는 구조다. 이때 데이터가 정규화되어 있지 않으면 다음 문제가 생긴다.

- 같은 문서가 여러 버전으로 중복 검색된다.
- 같은 용어가 약어, 한국어, 영문 full name으로 흩어진다.
- 오래된 사실과 최신 사실이 함께 들어온다.
- chunk의 출처, 페이지, 섹션을 추적할 수 없다.
- embedding 검색은 의미적으로 비슷한 문서를 찾지만, 업무적으로 정확한 객체를 놓친다.
- LLM이 충돌하는 근거를 보고 그럴듯한 평균 답변을 만든다.

RAG의 품질은 embedding 모델만으로 결정되지 않는다. 문서, 메타데이터, 용어, 식별자, 검색 점수를 정규화해야 안정적인 답변이 나온다.

## RAG 파이프라인의 정규화 지점

```text
원천 수집
  -> 파일/문서 ID 정규화
  -> 텍스트 추출 정규화
  -> chunk 정규화
  -> 메타데이터 정규화
  -> 용어/엔터티 정규화
  -> embedding 생성
  -> 검색 인덱스 저장
  -> 사용자 쿼리 정규화
  -> hybrid retrieval
  -> rerank/context assembly
  -> LLM 답변 생성
```

## 1. 문서 ID와 버전 정규화

문서가 바뀔 때마다 새로운 ID가 생기면 중복이 폭발한다. 반대로 버전을 무시하면 최신성과 감사 추적이 깨진다.

권장 필드:

```json
{
  "source_doc_id": "manual_abc_100",
  "source_uri": "s3://kb/manual/abc_100.pdf",
  "source_hash": "sha256:...",
  "version": "2026-05-02",
  "ingested_at": "2026-05-02T10:00:00+09:00",
  "valid_from": "2026-05-02",
  "valid_to": null,
  "is_latest": true
}
```

검색 시 최신 문서만 필요한지, 과거 시점 재현이 필요한지에 따라 필터가 달라진다.

## 2. Chunk 정규화

Chunk는 RAG의 최소 검색 단위다. Chunk ID가 안정적이지 않으면 재색인, 삭제, 평가가 어려워진다.

```json
{
  "chunk_id": "manual_abc_100:p003:s02:c01",
  "source_doc_id": "manual_abc_100",
  "page": 3,
  "section_path": ["설치", "전원 연결"],
  "chunk_index": 1,
  "text": "...",
  "prev_chunk_id": "manual_abc_100:p003:s01:c03",
  "next_chunk_id": "manual_abc_100:p003:s02:c02"
}
```

정규화 원칙:

- `chunk_id`는 원천 문서, 위치, 순서를 반영한다.
- 문서 파서가 바뀌어도 가능한 한 안정적으로 유지한다.
- parent document와 section 정보를 보존한다.
- chunk text만 저장하지 말고 구조 정보를 함께 저장한다.

## 3. 메타데이터 정규화

LLM/RAG에서 메타데이터는 검색 필터이자 답변 근거다.

| 메타데이터 | 이유 |
|------------|------|
| `doc_type` | 매뉴얼, 회의록, 정책, FAQ 구분 |
| `source` | 출처 신뢰도와 citation |
| `created_at` / `updated_at` | 최신성 판단 |
| `version` | 문서 충돌 방지 |
| `language` | 다국어 검색과 답변 언어 제어 |
| `owner_team` | 책임 부서와 접근 권한 |
| `security_level` | 민감정보 필터링 |
| `entity_ids` | 장비, 고객, 프로젝트 등 객체 필터링 |
| `canonical_terms` | 용어 기반 query expansion |

메타데이터 값은 자유 텍스트가 아니라 코드나 canonical value로 관리한다.

```text
나쁜 예:
  doc_type: "manual", "Manual", "매뉴얼", "사용설명서"

좋은 예:
  doc_type: "MANUAL"
  doc_type_label: "매뉴얼"
```

## 4. 용어와 엔터티 정규화

LLM은 약어와 동의어를 어느 정도 이해하지만, 업무 시스템에서는 "어느 ID의 객체인가"가 중요하다.

```json
{
  "canonical_id": "term:cvd",
  "canonical_label": "Chemical Vapor Deposition",
  "aliases": ["CVD", "화학기상증착", "chemical vapor deposition"],
  "category": "process",
  "definition": "기체 원료의 화학 반응으로 박막을 형성하는 공정"
}
```

RAG 활용:

```text
사용자 질문:
  "CVD 온도 조건 알려줘"

정규화:
  CVD -> term:cvd -> Chemical Vapor Deposition

검색:
  text match: CVD
  synonym/canonical match: Chemical Vapor Deposition, 화학기상증착
  metadata filter: canonical_terms contains term:cvd
```

이 방식은 약어가 많은 반도체, 제조, 의료, 법무, 금융 도메인에서 특히 중요하다.

## 5. Query normalization

사용자 질문도 정규화 대상이다.

정규화 항목:

- 오탈자 보정
- 약어 확장
- 날짜 표현 변환: "지난달" -> 절대 기간
- 단위 변환: "5k" -> `5000`
- 객체 식별: "ABC 장비" -> `equipment:abc`
- 권한/테넌트 필터 추가
- 검색 의도 분류: 정의, 절차, 비교, 장애 대응

예시:

```json
{
  "raw_query": "CVD 장비 지난달 알람 원인",
  "normalized_query": "Chemical Vapor Deposition equipment alarm root cause",
  "filters": {
    "entity_ids": ["equipment:cvd"],
    "date_range": {
      "from": "2026-04-01",
      "to": "2026-04-30"
    }
  },
  "intent": "root_cause_analysis"
}
```

LLM을 query normalizer로 사용할 수 있지만, 중요한 필터는 검증 가능한 코드와 규칙으로 후처리하는 편이 안전하다.

## 6. Embedding과 벡터 정규화

Embedding에서도 정규화라는 말이 쓰인다.

- 텍스트 전처리: 불필요한 header/footer 제거
- 의미 단위 chunking: 제목, 표, 목록을 보존
- embedding model/version 통일
- vector dimension 관리
- cosine similarity를 위한 L2 normalization 여부 관리

주의할 점:

- 원문을 과도하게 소문자화하거나 기호를 제거하면 코드, 모델명, 약어가 손상될 수 있다.
- embedding 입력용 정제 텍스트와 citation용 원문 텍스트를 분리할 수 있다.

```json
{
  "raw_text": "CVD-2000 장비의 Alarm A-17은 ...",
  "embedding_text": "CVD-2000 장비 Alarm A-17 원인 조치 ...",
  "embedding_model": "text-embedding-...",
  "embedding_version": "2026-05-02",
  "embedding_dimension": 1536
}
```

## 7. Retrieval score normalization

RAG에서는 BM25, vector similarity, recency, authority, reranker 점수가 함께 쓰인다. 이 점수들은 스케일이 다르므로 그대로 더하면 안 된다.

```text
final_score =
  0.35 * normalized_bm25
  + 0.45 * normalized_vector_score
  + 0.10 * normalized_recency
  + 0.10 * authority_score
```

OpenSearch의 `normalization-processor`처럼 검색엔진 레벨에서 처리할 수도 있고, 애플리케이션 reranker에서 처리할 수도 있다.

중요한 것은 점수 정규화와 데이터 모델 정규화를 구분하되, 둘 다 "비교 가능한 좌표계"를 만든다는 공통점이 있다는 점이다.

## 8. LLM을 정규화에 활용하는 방법

LLM은 정규화 작업 자체에도 유용하다.

| 작업 | LLM 역할 | 검증 방법 |
|------|----------|-----------|
| 용어 추출 | 문서에서 후보 용어, 약어, 정의 추출 | 사람 검토, 사전 중복 확인 |
| 엔터티 매칭 | "ABC 장비", "ABC-100"을 같은 객체 후보로 연결 | ID 규칙, fuzzy match, source 확인 |
| 스키마 추론 | 비정형 문서에서 필드 후보 추출 | JSON Schema, 샘플 검증 |
| 문서 분류 | 매뉴얼/정책/장애보고서 분류 | confidence threshold, 샘플 평가 |
| 쿼리 해석 | 자연어 질문을 검색 필터로 변환 | 허용 필드 whitelist, 날짜 검증 |
| 충돌 감지 | 서로 다른 문서의 상반된 주장 탐지 | 최신성, 권위, 원천 우선순위 |

LLM 출력은 바로 원천 데이터로 쓰지 말고 candidate로 다룬다. 정규화된 데이터는 반복성과 감사 가능성이 중요하므로, 사람이 승인하거나 규칙 기반 검증을 통과해야 한다.

## 9. RAG 정규화 체크리스트

- [ ] 문서 ID와 chunk ID가 안정적인가?
- [ ] 중복 문서와 오래된 버전을 식별할 수 있는가?
- [ ] chunk에 source, page, section, version이 있는가?
- [ ] 용어 alias가 canonical term으로 연결되는가?
- [ ] 객체명이 실제 entity ID로 연결되는가?
- [ ] query normalization 결과를 로깅하고 재현할 수 있는가?
- [ ] BM25, vector, recency, authority 점수를 정규화해서 결합하는가?
- [ ] LLM이 만든 정규화 결과를 검증하는 절차가 있는가?
- [ ] 답변에 사용된 근거가 chunk ID와 source로 역추적되는가?

## 참고 자료

- [OpenSearch Normalization Processor](https://docs.opensearch.org/2.15/search-plugins/search-pipelines/normalization-processor/)
- [MongoDB Vector Search Overview](https://www.mongodb.com/docs/atlas/atlas-search/vector-search/)
- [Redis Vector Search Concepts](https://redis.io/docs/latest/develop/ai/search-and-query/vectors/)
- [OpenSearch RAG 파이프라인 연동](../../rag/opensearch/rag-integration.md)
- [보조 용어 사전 DB](../../rag/auxiliary-glossary-db.md)
