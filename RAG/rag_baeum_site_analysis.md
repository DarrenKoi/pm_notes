# rag.baeum.ai.kr — 사이트 종합 분석

> RAG 파이프라인의 모든 단계(Loader → Parser → Embedding → Pre-Retrieval → Retrieval → Post-Retrieval → Generation)를 전수 실험으로 비교하고, 최적 조합을 도출·공개하는 **한국어 RAG 벤치마크 & 학습 플랫폼**.
> 수집일: 2026-07-09 · 수집 방식: 로그인 세션에서 전체 라우트 순회

---

## 1. 개요 — 사이트 정체성

**"BAEUM.AI RAG Bench"** 는 동일한 한국어 데이터셋(300 Q&A · 5개 도메인)에 대해 RAG의 각 단계를 전수(Cartesian) 실험으로 채점하고, 단계별 winner를 결합해 최적 파이프라인을 도출한 결과를 시각화한 사이트다.

핵심 성격은 세 가지가 결합되어 있다.

- **벤치마크 대시보드** — 426개 구성의 실험 결과를 인터랙티브 차트/표로 탐색
- **커뮤니티 리더보드** — 사용자가 자기 파이프라인 결과를 제출(API·JSON 업로드)해 순위 등재
- **교육 콘텐츠** — RAG 개념·방법론·평가·용어집을 담은 학습 섹션

### 실험 규모 (대시보드 핵심 지표)

| 지표 | 값 | 부연 |
|---|---|---|
| 총 실행 수 | **426** | 조합 384 + 축별 42 |
| 최고 JUDGE | **4.07** / 5.0 | judge_mean |
| 최고 MRR | **0.787** | hit@5 0.897 |
| 최고 정확도 | **82.7%** | majority-O (4지표 중 ≥2개 ≥4) |
| 최적 구성 | `run #baseline__dense__no_rerank` 계열 | 아래 최종 1위 파이프라인 |

### 데이터셋 구성

300개 질문 · 5개 도메인 각 60문항: **finance / public / medical / law / commerce**. 실제 PDF 코퍼스(3,166 청크) 기반이며, 각 질문에 reference 정답과 근거 문서(파일·페이지·문단)가 매핑되어 있다.

---

## 2. 정보 구조 (IA) · 네비게이션

라우팅은 해시 기반 SPA(`#/route`). 상단 탭 3개 + 좌측 사이드바(컨텍스트별로 바뀜) 구조.

### 상단 글로벌 네비게이션
- **대시보드** — 벤치마크 분석 뷰 묶음
- **리더보드** — 384 조합 전체 테이블 + 제출
- **학습** — 교육 콘텐츠

### 대시보드 사이드바
```
대시보드
 ├─ 개요                          #/overview
 └─ 벡터스토어 벤치 [신규]         #/vectorbench
NAIVE RAG
 ├─ 파이프라인 단계별 비교         #/pipeline-naive
 ├─ RAG 결과 보기                 #/drilldown-naive
 └─ LLM-as-a-Judge                #/judges
ADVANCED RAG
 ├─ 파이프라인 단계별 비교         #/pipeline-advanced
 ├─ RAG 결과 보기                 #/drilldown-advanced
 └─ 조합 탐색                     #/matrix
바로가기
 └─ RAG 처음이세요? →             (학습 개념 페이지로 연결)
```

### 학습 사이드바
```
학습
 ├─ RAG 개념        #learn/concept
 ├─ RAG 방법론      #learn/methodologies
 ├─ RAG 평가        #learn/evaluation
 ├─ 흔한 실수       #learn/pitfalls
 ├─ 자주 묻는 질문  #learn/faq
 └─ 용어집          #learn/glossary
데이터 보기
 ├─ 실험 결과 보기 →        (리더보드로)
 └─ RAG 결과 보기 (Naive) → (drilldown으로)
```

전체 라우트 15종을 모두 순회했다.

---

## 3. UI / UX 요약

- **레이아웃**: 좌측 고정 사이드바(접기 가능) + 상단 바(로고 "RAG", 탭 3개, 테마 토글) + 넓은 메인 영역. 라이트/다크 테마 토글 제공.
- **색상 언어**: 오렌지(주황) = 강조·주력 지표(MRR, 주력 계열), 그린 = 보조 지표(Judge), 진한 셀 = 우수 값. 히트맵·박스플롯·산점도·Pareto 곡선 등 데이터 시각화가 매우 풍부.
- **인터랙션 패턴**:
  - 파이프라인 단계 카드를 **클릭하면 해당 단계 전체 실험이 아래에 펼쳐지는** 아코디언식 드릴다운.
  - 도메인 필터 탭(전체/finance/public/medical/law/commerce)로 지표를 도메인별로 전환.
  - 산점도 범례(모델 계열) 클릭 토글로 시리즈 on/off.
  - 리더보드는 컬럼 정렬 + 좌측 다중 필터(단계별 드롭다운).
  - 학습 개념 페이지는 8단계 스텝을 넘기며 "미리보기 / 코드 / 언제 어떤 옵션을?" 탭으로 학습.
- **정보 밀도**: 매우 높음. 각 차트마다 "읽는 법", "핵심 발견", "INSIGHT" 같은 해설 블록을 곁들여 데이터→시사점 해석을 유도.
- **톤**: 실무자 대상. "직관과 다른 결과", "전수탐색을 한 이유" 등 인사이트 중심의 카피.

---

## 4. 대시보드 페이지별 상세

### 4-1. 개요 (`#/overview`)

전체 실험을 한 화면에 요약하는 랜딩 대시보드.

**최종 1위 파이프라인 (단계별 winner)**

| 단계 | 후보 수 | Winner |
|---|---|---|
| LOADER | 7종 | PyMuPDFLoader |
| PARSER | 42종 | RecursiveChar 300/50 |
| EMBEDDING | 39종 | embeddinggemma-300m |
| PRE-RETRIEVAL | 10종 | HyDE |
| RETRIEVAL | 7종 | Hybrid d0.3:s0.7 |
| POST-RETRIEVAL | 25종 | jina-m0 (jina-reranker-m0) |
| GENERATION | 46종 | GPT-5.4 |

**조합 상위 (Pre×Retrieval×Post 384 중 Accuracy 상위)**

| # | Pre-Retrieval | Retrieval | Post-Retrieval | MRR | Judge | Accuracy |
|---|---|---|---|---|---|---|
| 1 | query2doc | hybrid_7_3 | jina-reranker-m0 | 0.763 | 4.067 | **82.7%** |
| 2 | hyde_rrf | hybrid_5_5 | qwen3-reranker-0.6b | 0.724 | 3.997 | 82.0% |
| 3 | baseline | hybrid_7_3 | jina-reranker-m0 | 0.777 | 4.004 | 81.7% |
| 4 | query2doc | hybrid_7_3 | bge-reranker-large | 0.694 | 3.991 | 81.3% |
| 5 | query2doc | hybrid_5_5 | jina-reranker-m0 | 0.773 | 4.036 | 81.3% |

**주요 분석(인사이트) 블록**
- **작은 한국어 reranker가 SOTA 압도** — `dragonkue/bge-v2-m3-ko`(0.6B)가 6.7배 큰 Qwen3-Reranker-4B를 +1.83pp MRR 능가. 크기보다 한국어 정렬이 핵심.
- **비용은 Generator에 집중** — 파이프라인 비용 구조 Generator 70% · Judge 25% · Retrieval 5%.
- **파이프라인 최적화 > 모델 업그레이드** — 동일 GPT-5.4에서 파이프라인만 최적화 → 0.827로, GPT-5.4-pro(10배 비쌈) 업그레이드보다 +6.0pp 높음.
- **MRR ≠ 답변 품질** — retrieval MRR과 judge_mean이 강하게 발산하는 경우 존재. reranker를 MRR만으로 고르면 위험.

**누적 개선 (naive → cartesian winner)**

| 단계 | MRR | JUDGE | ACCURACY |
|---|---|---|---|
| 0. naive (dense, no rerank) | 0.682 | 3.850 | 73.0% |
| + Hybrid retrieval | 0.717 | 3.869 | 75.3% |
| + Reranker (bge-v2-m3-ko) | 0.770 | 3.916 | 77.0% |
| Cartesian winner (judge) | 0.763 | 4.067 | **82.7%** |

→ 총 +9.7pp accuracy 개선. 단일 단계 최대 기여는 조합 최적화(+5.7pp). **투자 우선순위: Reranker → Retrieval → Pre-Retrieval.**

**단계별 임팩트 (선택이 평균에 주는 표준편차)**
- Post-Retrieval σ 1.54pp (best 79.0% vs worst 74.0%) — 가장 큰 레버
- Retrieval σ 0.99pp
- Pre-Retrieval σ 0.54pp — 가장 작음

기타 시각화: 가중치 공개여부별(Open/Closed) 정확도 분포 산점도, 비용-정확도 Pareto frontier, Embedding×도메인 히트맵, Open vs Closed 박스플롯, Pre×Post 상호작용 히트맵.

**Open vs Closed Top 5**
- Open: gpt-oss_120b 74.0% / gpt-oss_20b 72.7% / deepseek-v4-flash 70.7%
- Closed: gpt-5.4 78.7% / gpt-5.4-pro 76.7% / grok-4.20 75.7%
- Closed 평균 71% vs Open 평균 66% (격차 +5.6pp, 최고점 +4.7pp)이나 분포 겹침이 커 단독 판단 불가.

---

### 4-2. 벡터스토어 벤치 (`#/vectorbench`) [신규 · 완료]

17종 벡터 검색 시스템을 **단일 서버**에서 동일 조건으로 비교. recall보다 **자원·확장성·운영 안정성**이 1차 목적(더미 벡터 1536차원, 1K→10M 규모 게이트 방식).

**성능 전용 스코어보드 (6지표 가중, 만점 45)**

| # | System | Baseline | Tuned | 카테고리 |
|---|---|---|---|---|
| 🥇 | faiss_gpu | 41 | 41 | 라이브러리 |
| 🥇 | faiss_cpu | 41 | 41 | 라이브러리 |
| 🥉 | lancedb | 40 | 40 | 임베디드/파일 |
| 4 | annoy | 39 | 39 | 라이브러리 |
| 5 | hnswlib | 34 | 34 | 라이브러리 |
| 6 | weaviate | 33 | 35▲ | 전용 벡터 DB |
| 7 | scann | 32 | 32 | 라이브러리 |
| 8 | redis | 29 | 29 | 인메모리 |
| 9 | milvus | 28 | 34▲ | 전용 벡터 DB |
| 9 | qdrant | 28 | 39▲ | 전용 벡터 DB |
| 9 | chroma | 28 | 28 | 임베디드 |
| 9 | vespa | 28 | 28 | 검색엔진 |
| 13 | clickhouse | 22 | 22 | 분석 DB |
| 14 | elasticsearch | 21 | 21 | 검색엔진 |
| 15 | opensearch | 20 | 20 | 검색엔진 |
| 16 | pgvector | 17 | 21▲ | RDB 확장 |
| 17 | duckdb | 16 | 16 | 분석 DB |

**핵심 시사점**
- **10M 고동시성 검색 생존은 faiss_gpu · lancedb 둘뿐** (faiss_cpu는 5M↑ 탈락).
- 전용 벡터 DB가 자동 우위 아님 — 라이브러리/임베디드가 상위 독식. qdrant·milvus는 baseline 공동 9위(1M 지연·느린 빌드가 발목).
- **튜닝 탄력성**: qdrant는 binary 양자화 하나로 QPS 38→1742(45배) → #10→#5(+11).
- Phase C 노브 법칙: HNSW의 M은 빌드시간 지배, ef는 런타임 QPS↔지연. 양자화가 빌드시간을 극적으로 단축.
- **용도별 추천**: GPU 보유·최고 처리량 → faiss_gpu / GPU 없는 단일서버 범용 → lancedb / ≤3M 인메모리 최고속 → faiss_cpu / 운영기능+1M → qdrant+binary / 분산 → milvus+IVF_SQ8 / RDB 통합 → pgvector.
- **본 RAG 평가의 기본 retrieval backend = FAISS CPU** (3,166 청크 소규모라 충분).

포함 시각화: 단계별 검색 건전성(P/△/F/·) 게이트, 10개 지표별 규모-성능 라인차트, 전체 히트맵, 동시성 확장 효율표, 튜닝 전후 순위 반전, 자원 효율 Pareto, Phase C OFAT 튜닝표(qdrant/milvus/pgvector/weaviate).

---

### 4-3. Naive RAG — 파이프라인 단계별 비교 (`#/pipeline-naive`)

단순 RAG의 4단계(Loader·Parser·Embedding·Generation). 카드 클릭 시 해당 단계 전체 비교가 펼쳐짐. 도메인 필터 제공.

**LOADER (7종)** — 동일 PDF 코퍼스 MRR/Hit@5

| Loader | MRR | Hit@5 |
|---|---|---|
| pymupdf | 0.649 | 0.763 |
| pdfplumber | 0.647 | 0.770 |
| pymupdf4llm | 0.639 | 0.773 |
| pdfminer | 0.630 | 0.753 |
| docling | 0.624 | 0.737 |
| pypdf | 0.620 | 0.747 |
| opendataloader | 0.599 | 0.753 |

**PARSER / Chunker (42종)** — character·semantic 통합 비교. 상위: Chonkie Slumber(GPT-5.4) MRR 0.711, LlamaIndex SemanticSplitter 0.708, Kiwi+Recursive 500/100 0.703. Hit@5 최상위는 LC Recursive 300/50 (0.813). 즉 **MRR은 semantic 계열, Hit@5는 character recursive 계열**이 강함.

**EMBEDDING (39종)** — 로컬+상용 API, 768~5376차원. 상위:

| Embedding | 차원 | 유형 | MRR | Hit@5 |
|---|---|---|---|---|
| voyage-3-large | 1024 | 클로즈 | 0.687 | 0.763 |
| kure-v1 | 1024 | 오픈 | 0.669 | 0.737 |
| pixie-rune-v1 | 1024 | 오픈 | 0.658 | 0.743 |
| gemini-embed-001 | 3072 | 클로즈 | 0.652 | 0.740 |
| snowflake-arctic-ko | 1024 | 오픈 | 0.652 | 0.737 |
| cohere-embed-v4 | 1536 | 클로즈 | 0.652 | 0.753 |

추가로 **정밀도·양자화 강건성** 분석 포함: dtype(fp32→bf16→fp16)은 거의 무손실이나 Gemma 계열 fp16은 NaN(오버플로). 정수 양자화는 Q8~Q4 안전, Q3부터 하락, Q2 붕괴. 권고: **bf16 기본 → Q8/Q4 안전, Q3 이하·Gemma fp16 금지.** (harrier-27b는 과거 fp16-NaN 버그로 저평가됐다가 0.618로 교정.)

**GENERATION (46종)** — 동일 context에서 답변 품질·비용 비교. FAMILY/TIER 필터 + 비용-정확도 산점도. 상위:

| Model | Family | Hosting | Accuracy | $/1K tok |
|---|---|---|---|---|
| gpt-5.4 | openai | API | 78.7% | $0.001381 |
| gpt-5.4-pro | openai | API | 76.7% | $0.016309 |
| grok-4.20 | xai | API | 75.7% | $0.003095 |
| gemini-3-flash-preview | google | API | 74.0% | $0.000476 |
| gpt-oss_120b | openai-oss | self-host | 74.0% | $0 |
| kimi-k2.5 | moonshot | API | 74.0% | $0.002143 |

→ 무료 self-host `gpt-oss_120b`가 74.0%로 유료 상위권과 동급.

---

### 4-4. RAG 결과 보기 — Naive drilldown (`#/drilldown-naive`)

실제 300 Q&A 전체를 한 건씩 열람. Reference pipeline = `C_bge-reranker-v2-m3-ko`.

- 좌측: 도메인 필터 + qid/본문 검색 + 질문 셀렉터(1/300 페이징)
- 모드 탭: **Reference·검색 결과** / **LLM 모델 비교**
- 한 질문에 대해: QUESTION → REFERENCE(정답, 근거 문서·페이지·문단) → RETRIEVED CHUNKS(TOP-5) → GENERATED ANSWER(GPT-5.4) → JUDGE SCORES(similarity/correctness/completeness/faithfulness 각 1–5) → judge_mean / ACCURACY(O·X) / RULE(≥2 of 4 ≥4)
- 주의: 실측 벤치는 retrieval metric만 저장 → 질문별 top-K chunk 본문은 미기록("NO PER-QUERY CHUNK LOG"). 일부 질문은 "모델별 답변 데이터 없음".

예시 Q0000(finance, 시중/지방/인터넷은행 인가 요건): judge_mean 4.25, Accuracy O.

---

### 4-5. LLM-as-a-Judge (`#/judges`)

**20종 judge LLM(클로즈 9 + 오픈 11)이 동일 46종 generator 답변을 채점한 결과**를 비교 — 어떤 채점기가 정답 합의에 가장 가까운지.

**정확도(O-rate) 산정법**: 4지표(similarity·correctness·completeness·faithfulness) 각 1–5점 → 각 ≥4점 통과 → 4개 중 ≥2개 통과 시 그 답변 O → O-RATE = O 수 / 300. (allganize 방법론, majority vote.)

**판단모델 신뢰도 — 정답(클로즈 9종 합의) 대비 편차 (낮을수록 정확)**

| # | 유형 | Judge | 편차 |
|---|---|---|---|
| 1 | 클로즈 | claude-sonnet-4-6 | 0.9pp |
| 2 | 클로즈 | gpt-5.5 | 0.9pp |
| 3 | 클로즈 | gemini-3.1-flash-lite | 1.1pp |
| 4 | **오픈** | qwen3.5_35b-a3b-q4_K_M | **1.6pp** |
| 5 | 오픈 | qwen3.5_122b-a10b-q4_K | 1.7pp |
| 6 | 클로즈 | gpt-5.4 | 1.8pp |
| … | | | |
| 20 | 오픈 | qwen3-next-80b | 16.5pp |

**핵심 발견**
- 오픈웨이트 최상위(qwen3.5_35b-a3b)가 **gpt-5.4보다 정답 편차가 작음** — 비용 0의 오픈 judge가 GPT-5.4급 신뢰도.
- **판단모델은 크기 무관** — 80B·100B·120B 대형 오픈이 35B-a3b보다 정답에서 멀어짐. judge는 크기가 아니라 정렬(alignment)이 핵심.
- self-host judge 추천: qwen3.5_122b-a10b-q4(균형), supergemma4-26b(일치도), nemotron-3-super-120b(GPT-5.4 1:1). 경량 절충은 qwen3.5_35b-a3b.

포함: 오픈×클로즈 judge Pearson 상관 매트릭스, generator 46종 × judge 20종 전체 점수 매트릭스(Closed Avg/Open Avg 컬럼).

---

### 4-6. Advanced RAG — 파이프라인 단계별 비교 (`#/pipeline-advanced`)

검색 최적화 3단계(Pre-Retrieval·Retrieval·Post-Retrieval). 앞단(Loader/Parser/Embedding)·Generator는 고정 조건 표기.

**PRE-RETRIEVAL (10종)** — query rewriting/expansion

| 기법 | MRR | Hit@5 |
|---|---|---|
| multi_query_para | 0.719 | 0.810 |
| baseline | 0.717 | 0.803 |
| hyde_rrf | 0.716 | 0.807 |
| hyde | 0.712 | 0.810 |
| decompose | 0.711 | 0.813 |
| query_expansion | 0.708 | 0.810 |
| query2doc | 0.699 | 0.803 |
| query_rewrite | 0.695 | 0.803 |
| multi_query_angle | 0.643 | 0.787 |
| step_back | 0.603 | 0.757 |

**RETRIEVAL (7종)** — Dense/Sparse/Hybrid 가중치

| 기법 | MRR | Hit@5 |
|---|---|---|
| Hybrid d0.3:s0.7 | 0.717 | 0.803 |
| Hybrid d0.5:s0.5 | 0.714 | 0.800 |
| Hybrid d0.7:s0.3 | 0.705 | 0.803 |
| Dense (vector only) | 0.682 | 0.813 |
| BM25 (KIWI) | 0.678 | 0.773 |
| Hybrid·WS d0.5:s0.5 | 0.650 | 0.743 |
| BM25 (whitespace) | 0.534 | 0.627 |

**POST-RETRIEVAL — Reranker (25종)** — 메트릭별 winner가 3개로 갈림:
- MRR 1위: `dragonkue/bge-v2-m3-ko` (0.864, 단독 평가)
- Judge 1위: `Dongjin-kr/ko-reranker` (4.38, 조합 평가)
- Cartesian 384 종합: `jina-reranker-m0`

MRR 상위: shoxa-mir-bge-v2-m3-ko 0.770 · bge-reranker-v2-m3-ko 0.770 · jina-reranker-m0 0.763 · qwen3-reranker-4b 0.751. 하위 극단: naver-xprovence 계열은 MRR 0.05~0.25(랭킹 실패)이나 judge는 3.7~3.97(content pruning 용도라 top-5 안에는 정답 유지).

---

### 4-7. RAG 결과 보기 — Advanced drilldown (`#/drilldown-advanced`)

Naive drilldown과 동일한 300 Q&A 열람 구조. 차이점은 모드 탭이 **Reference·검색 결과 / Pre-Retrieval / Retrieval / Post-Retrieval** 로 구성되어, 같은 질문에 대해 각 단계 기법별 결과를 비교할 수 있다는 점.

---

### 4-8. 조합 탐색 (`#/matrix`)

**"기법을 따로 튜닝해도 될까, 조합을 통째로 봐야 할까?"** — Pre 8 × Retrieval 6 × Reranker 8 = **384 전수 조합**(Generator GPT-5.4 고정)으로 축 영향력과 상호작용(interaction)을 분석. 지표 토글: Judge평균/MRR/Hit@5/Accuracy.

- **① 어느 축이 결정적인가** (Judge 변동폭): Reranker 0.150(가장 결정적) > Retrieval 0.069 > Pre-Retrieval 0.064. Reranker는 나머지 2축의 약 2.2배 레버.
- **② 축 독립성**: Reranker를 jina-m0로 고정 시, 대부분 검색기의 최적 Pre는 query_expansion이나 **hybrid_7_3에서는 query2doc, hybrid_ws_5_5에서는 multi_query_para로 역전** → 축이 완전 독립적이지 않고 상호작용 존재.
- **결정적 증거**: 축별 1등 조립(query_expansion+hybrid_5_5+jina-m0 = 4.06) < 전수탐색 최고(query2doc+hybrid_7_3+jina-m0 = 4.07). **단계별 튜닝만으론 최적 조합을 놓치므로 전수탐색이 필요했다는 직접 증거.**
- ③ 근거: jina-m0 고정 Pre×Retrieval 단면 히트맵 + 8개 Reranker별 단면 나란히 비교(focus 전환).

---

### 4-9. 리더보드 (`#board/leaderboard`)

384개 파이프라인 조합 전체를 나열한 대형 정렬 테이블 + **커뮤니티 제출** 기능.

- 컬럼: 제출 · LOADER · PARSER · EMBEDDING · PRE-RETRIEVAL · RETRIEVAL · POST-RETRIEVAL · GENERATOR · MRR · Hit@5 · JUDGE MEAN · ACCURACY (정렬 가능)
- 좌측 필터: 7단계 각각 드롭다운으로 조합 좁히기
- JUDGE 토글: **Qwen3.6 35B-A3B(오픈) / GPT-5.4(클로즈)** — 채점 모델별 순위 분리
- 상단 액션: **+결과 등록하기** · 평가 템플릿 ZIP · baseline JSON

**결과 등록하기 (제출 플로우)**
- BAEUM.AI RAG Bench(300 Q&A)에 자체 파이프라인 결과를 제출 → judge 채점(GPT-5.4 또는 오픈 Qwen3.6-35B-A3B) 완료 시 리더보드 등재
- 시작: 평가 템플릿 ZIP 다운로드 → `examples/01_use_baseline_retriever.py` 실행 → baseline retrieval API 호출 + 본인 LLM 답변
- **Retrieval API**: `curl -X POST https://rag.baeum.ai.kr/api/retrieve -H 'Content-Type: application/json' -d '{"query":"...","top_k":5,"retriever":"hybrid_3_7"}'`
- 폼 필드: 제출 이름* · 메모 · 7단계 구성 · 평가모델(JUDGE)* · 결과 JSON 업로드*
- 클로즈 judge는 정확하나 유료, 오픈 judge는 무료. 리더보드는 평가모델별로 구분 등재.

리더보드 1위(GPT-5.4 judge 기준): query2doc + hybrid_7_3 + jina-reranker-m0, Judge 4.215, Accuracy 85.7%.

---

## 5. 학습 섹션 (교육 콘텐츠)

### 5-1. RAG 개념 (`#learn/concept`)
"RAG란 무엇인가" — 8단계 인터랙티브 튜토리얼: ① 문서 로드(Loader) ② 청크 분할(Parser) ③ 임베딩 ④ 검색 전 변환(Pre-Retriever) ⑤ 검색(Retriever) ⑥ 재정렬(Reranker) ⑦ 답변 생성(Generation) ⑧ RAG 방법론(Naive→Advanced→Modular→Agentic). 각 단계마다 **미리보기 / 코드 / 언제 어떤 옵션을?** 탭.

### 5-2. RAG 방법론 (`#learn/methodologies`)
**25개 RAG 방법론**을 논문·난이도·장단점·구현팁·흐름도와 함께 상세 카드로 제공: Naive, Advanced, Modular, Self-RAG, CRAG, Agentic, GraphRAG, RAPTOR, HyDE, FLARE, HippoRAG, Contextual, Adaptive-RAG, LongRAG, RankRAG, Iter-RetGen, REPLUG, RA-DIT, VisRAG, ColPali, MemoRAG, Auto-RAG, HtmlRAG, Search-o1. (예: Naive RAG = Lewis et al., NeurIPS 2020, DPR+BART, Open NQ EM 44.5.)

### 5-3. RAG 평가 (`#learn/evaluation`)
6개 섹션: ① 메트릭 분류(Retrieval/Generation/Composite) ② 평가 패러다임(Human/LLM-judge/Reference-based·free) ③ LLM-as-judge 운영 ④ 벤치마크·데이터셋 ⑤ 자동 평가 도구(Ragas·TruLens·DeepEval·LangSmith·Phoenix) ⑥ 평가 신뢰성(표본·신뢰구간·multi-seed).
- Retrieval 메트릭: MRR, Hit@K, Recall@K, nDCG@K, File@K
- Generation 메트릭: Similarity, Correctness, Completeness, Faithfulness, Answer/Context relevancy
- Composite: judge_mean, Accuracy(majority-O), G-Eval, RAGAS score

### 5-4. 흔한 실수 (`#learn/pitfalls`)
난이도(high/med/low)별 10개 함정 + 증상·원인·처방·BAD/GOOD 코드:
- (HIGH) chunk size 과대 → 300~500자 + overlap 10–20%
- (HIGH) Reranker 생략 → cross-encoder 추가 시 MRR +0.04~0.06
- (HIGH) 한국어에 영어 임베딩만 → 다국어/한국어 모델 + BM25-KIWI 하이브리드
- (HIGH) eval set이 작거나 쉬움 → 300+ 샘플, 직설/추론/다단계/도메인외 혼합
- (MED) Hybrid 5:5 고정, LLM-judge만 신뢰, 인덱스 미재빌드, context 과다(lost in the middle), 메타데이터 필터 부재
- (LOW) Pre-retriever 무지성 사용(특히 Decompose)

### 5-5. 자주 묻는 질문 (`#learn/faq`)
카테고리별 **24개 Q&A**: 기초 3 · 청킹 2 · 검색 4 · 쿼리 변환 2 · 재정렬 2 · 평가 4 · 운영 5 · 디버깅 2. (예: RAG vs fine-tuning, Long-context와의 비교, chunk_size 결정법, Hybrid 3:7의 근거, Vector DB 선택, HyDE 비용, citation 구현, latency 최적화 등.)

### 5-6. 용어집 (`#learn/glossary`)
**116개 용어**를 카테고리(metric/method/eval/concept/llm/prompt)별로 정의 + 검색. MRR·BM25-KIWI·HyDE·RRF·ColBERT·HNSW·MoE·GGUF·vLLM·DPO·CoT·ReAct 등 RAG·LLM·프롬프트·평가 전반 커버.

---

## 6. RAG 관련 핵심 결론 (사이트가 주장하는 것)

1. **Reranker(Post-Retrieval)가 가장 큰 레버** — 단일 단계 최대 accuracy 기여(+5.7pp 조합 최적화 포함). 투자 우선순위 Reranker → Retrieval → Pre.
2. **한국어에선 정렬 > 크기** — 작은 한국어 특화 reranker/judge가 훨씬 큰 범용 모델을 능가.
3. **파이프라인 최적화가 모델 업그레이드보다 효율적** — 같은 모델로 +6.0pp, 10배 비용 회피.
4. **비용은 Generator에 집중**(70%) — retrieval/reranker는 거의 공짜라 절감은 generator부터.
5. **검색 품질(MRR) ≠ 답변 품질(judge)** — 두 축이 발산. reranker·모델 선택은 majority-O accuracy까지 함께 봐야.
6. **오픈웨이트로 충분** — generator·judge·embedding·vectorstore 모두 무료 오픈 모델이 유료 상위권과 경쟁 가능.
7. **전수탐색이 필요한 이유** — 축이 상호작용하므로 단계별 1등을 조립하면 진짜 최적을 놓친다.

---

## 부록 — 전체 라우트 맵

| 라우트 | 페이지 | 유형 |
|---|---|---|
| `#/overview` | 개요 대시보드 | 분석 |
| `#/vectorbench` | 벡터스토어 벤치 | 분석 |
| `#/pipeline-naive` | Naive 단계별 비교 | 인터랙티브 |
| `#/drilldown-naive` | Naive 결과 보기 (300 Q&A) | 드릴다운 |
| `#/judges` | LLM-as-a-Judge | 분석 |
| `#/pipeline-advanced` | Advanced 단계별 비교 | 인터랙티브 |
| `#/drilldown-advanced` | Advanced 결과 보기 | 드릴다운 |
| `#/matrix` | 조합 탐색 (384) | 분석 |
| `#board/leaderboard` | 리더보드 + 제출 | 테이블/폼 |
| `#learn/concept` | RAG 개념 (8단계) | 교육 |
| `#learn/methodologies` | RAG 방법론 (25종) | 교육 |
| `#learn/evaluation` | RAG 평가 (6섹션) | 교육 |
| `#learn/pitfalls` | 흔한 실수 (10) | 교육 |
| `#learn/faq` | 자주 묻는 질문 (24) | 교육 |
| `#learn/glossary` | 용어집 (116) | 교육 |
