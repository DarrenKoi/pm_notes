---
tags: [llm, rag, normalization, retrieval, metadata, glossary, embeddings]
level: advanced
last_updated: 2026-05-02
reviewed_on: 2026-10-04
review_status: reviewed_with_limits
document_type: learning_note
category_major: "AI·DT"
category_middle: "데이터 엔지니어링"
category_minor: "정규화·모델링"
note_kind: "학습"
classified_on: "2026-10-05"
---

# LLM과 RAG에서의 정규화

> RAG에서 정규화는 검색 성능만의 문제가 아니다. LLM이 어떤 사실을 어떤 출처와 어떤 개념으로 이해해야 하는지를 안정화하는 작업이다.

> [!info] 문서의 역할과 검토일 — 2026-10-04
> 이 장은 RAG 데이터/조회 계약의 학습용 설계다. 정규형 이론과 다른 값·메타데이터·검색 변환을 구분한다. [03 검색](./03-opensearch-normalization.md)·[04 저장](./04-mongodb-normalization.md)·[05 캐시](./05-redis-normalization.md)의 확인 판본/조건을 따른다. 실제 검색엔진·모델·회사 데이터·ACL은 미실행이다.

## 왜 RAG에 정규화가 필요한가?

RAG는 문서를 검색해서 LLM의 컨텍스트에 넣는 구조다. 이때 데이터가 정규화되어 있지 않으면 다음 문제가 생긴다.

- 같은 문서가 여러 버전으로 중복 검색된다.
- 같은 용어가 약어, 한국어, 영문 full name으로 흩어진다.
- 오래된 사실과 최신 사실이 함께 들어온다.
- chunk의 출처, 페이지, 섹션을 추적할 수 없다.
- embedding 검색은 의미적으로 비슷한 문서를 찾지만, 업무적으로 정확한 객체를 놓친다.
- LLM이 충돌하는 근거를 보고 그럴듯한 평균 답변을 만든다.

RAG의 품질은 embedding 모델만으로 결정되지 않는다. 문서/메타데이터/용어/식별자의 일관성은 점검 대상이지만 답변 정확성의 충분조건은 아니다. 검색·근거 충돌·생성 모델/프롬프트를 별도 평가한다. 원래 RAG 논문(2020)은 외부 검색 근거와 parametric model 결합을 제안하며 온톨로지 사용을 모든 RAG의 필수 조건으로 정하지 않는다.

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

source_doc_id는 논리 문서, version/source_hash는 특정 판본의 식별이다. 위 URI/hash는 가상 예다. ingested_at은 수집 시각이며 발행/업무 유효 시각과 다르다. valid_to:null은 이 예에서 종료 미정이지 무조건 현재 승인/최신이라는 뜻은 아니다. is_latest는 승인된 manifest/원천 버전 계약으로 갱신해야 한다. 과거 시점 조회와 최신 조회를 구분하고 삭제·재수집·중복 event·동시 변경을 검사한다.

## 2. Chunk 정규화

이 파이프라인은 chunk를 검색 단위로 사용한다. 문서/문장/parent 등을 검색하는 다른 설계도 있다. Chunk ID가 안정적이지 않으면 재색인, 삭제, 평가가 어려워진다.

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
- 같은 원문 버전과 parser/chunker 계약에서 재현 가능한 ID를 만든다. 아래 위치 ID만으로 버전 간 충돌을 막지는 못하므로 저장 키를 `(source_doc_id, version, chunking_revision, chunk_id)` 등으로 구분한다. parser 변경 시 위치가 달라지면 과거 citation 보존/새 ID와 폐기 정책을 정한다.
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

필터에 쓰는 값은 타입/코드 사전을 정한다. title/원문 정의 같은 자유 텍스트를 모두 코드로 바꾸라는 규칙은 아니다. owner_team/security_level이라는 필드만으로 권한 검사가 실행되지 않는다. 사용자/테넌트 권한은 신뢰한 인증 주체로부터 조회 시 강제하고 citation/응답 cache에서도 재검사한다. 누락 권한·버전은 기본 허용/최신으로 바꾸지 않는다.

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

이 예제의 canonical_terms contains term:cvd는 **ID 계약**이다. 03/04의 canonical_terms 표기 문자열과 동일 의미로 혼용하지 않는다. 같은 인덱스에서 함께 사용하려면 canonical_term_ids(ID)와 canonical_terms(label)를 분리하는 등의 명세와 변환을 정한다. alias는 여러 개념에 대응할 수 있으므로 도메인/출처로 후보를 검토하고 미확정을 보존한다. LLM 연결 후보는 실제 장비 ID의 증명이 아니다.

## 5. Query normalization

사용자 질문도 정규화 대상이다.

정규화 항목:

- 오탈자 보정
- 약어 확장
- 날짜 표현 변환: "지난달" -> 절대 기간
- 단위 변환: "5k"는 개수/통화/온도 등 문맥과 단위를 확인한 경우만 변환하고 미확정은 보존
- 객체 식별: "ABC 장비" -> `equipment:abc`
- 권한/테넌트 필터는 LLM 출력과 별개로 서버의 인증/권한 주체에서 강제
- 검색 의도 분류: 정의, 절차, 비교, 장애 대응

예시:

```json
{
  "raw_query": "CVD 장비 지난달 알람 원인",
  "normalized_query": "Chemical Vapor Deposition equipment alarm root cause",
  "filters": {
    "entity_ids": ["equipment:cvd"],
    "date_range": {
      "from": "2026-04-01T00:00:00+09:00",
      "to_exclusive": "2026-05-01T00:00:00+09:00"
    }
  },
  "intent": "root_cause_analysis"
}
```

위 예는 기준 시각2026-05-02·Asia/Seoul에서 지난달을 `[2026-04-01 00:00, 2026-05-01 00:00)`로 해석한 후보다. 실제 장비 equipment:cvd와 intent는 조회로 확인해야 하며 CVD 공정 용어만으로 장비 한 대를 확정하지 않는다. 종료일 자정 때문에4월30일 낮을 제외하는 오류를 막도록 반열린 기간/시간대를 명시한다. LLM 후보를 허용 필드/타입·달력·단위·객체 존재로 검증하고 권한 필터를 대체하게 하지 않는다.

## 6. Embedding과 벡터 정규화

Embedding에서도 정규화라는 말이 쓰인다.

- 텍스트 전처리: 불필요한 header/footer 제거
- 의미 단위 chunking: 제목, 표, 목록을 보존
- embedding model/version 통일
- vector dimension 관리
- cosine/L2/dot-product의 인덱스 계약에 맞춰 L2 normalization 여부 관리 — cosine 자체가 벡터 길이로 나누므로 언제나 선행 정규화가 필수인 것은 아님

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

embedding_model의 placeholder와1536차원은 가상의 메타데이터 예시이며 특정 모델 결과가 아니다. 실제 모델/revision·차원·전처리와 index/query를 일치시킨다. 0벡터는 cosine/L2 정규화가 정의되지 않고, nonfinite 값·차원 불일치는 검사해서 거부한다. 정규화 여부만으로 다른 모델 공간을 호환으로 만들 수 없다.

## 7. Retrieval score normalization

RAG에서는 BM25, vector similarity, recency, authority, reranker 점수가 함께 쓰인다. 이 점수들은 스케일이 다르므로 그대로 더하면 안 된다.

```text
final_score =
  0.35 * normalized_bm25
  + 0.45 * normalized_vector_score
  + 0.10 * normalized_recency
  + 0.10 * authority_score
```

위 계수는 학습용 가중합이고 평가로 정한 최적값이 아니다. 누락/동일 점수의 min-max 분모0·낮을수록 가까운 거리/높을수록 좋은 similarity의 방향을 명시한다. unknown 값을0으로 조용히 바꾸지 않는다. OpenSearch2.15 normalization-processor의 지원 query 점수 결합은 [03](./03-opensearch-normalization.md)을 따른다. 위 recency/authority4항을 해당 processor가 그대로 구현한다는 뜻이 아니다. 앱의 별도 scoring/reranking 구현과 평가가 필요하다.

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

## 검토 결과

2026-10-04: 원래 파이프라인/ID/chunk/용어/query/embedding/score/LLM 후보 예제와 절·작성일을 보존했다. 날짜 경계·인증 주체·모호성·버전·벡터 조건과 과도한 보장을 정정했다. raw query/근거에는 민감 정보가 있을 수 있으므로 재현 로그의 범위·접근·보존/가림 정책을 정한다. 실제 로그·모델·검색·권한은 미실행이다. Claude pane_not_found로 완전 통합·업무별 계약은 보류했다. [정리 기록](../organization-log.md)을 따른다.

## 참고 자료

제품 문서는 각 적용 장의 확인 판본과 실행 범위를 따른다. 아래 RAG 원 논문을2026-10-04 확인했다.

- [OpenSearch Normalization Processor](https://docs.opensearch.org/2.15/search-plugins/search-pipelines/normalization-processor/)
- [MongoDB Vector Search Overview](https://www.mongodb.com/docs/vector-search/)
- [Redis Vector Search Concepts](https://redis.io/docs/latest/develop/ai/search-and-query/vectors/)
- [같은 주제의 OpenSearch 적용](./03-opensearch-normalization.md)
- [RAG 원 논문, Lewis et al., 2020](https://arxiv.org/abs/2005.11401)

- [NIST 연구, 기체 전구체/CVD 박막 증착, 2019; 2026-10-04 확인](https://www.nist.gov/publications/apparatus-characterizing-gas-phase-chemical-precursor-delivery-thin-film-deposition) — 용어 배경이며 가상 장비의 실제 조치 근거가 아님.
