---
tags: [langchain, langgraph, compatibility, sources]
reviewed_on: 2026-10-04
review_status: partial
document_type: reference
category_major: "AI·DT"
category_middle: "에이전트 개발"
category_minor: "커리큘럼 안내"
note_kind: "검토 기록"
classified_on: "2026-10-05"
---

# 공통 적용 조건과 근거

## 읽기와 설정

[README](./README.md)의 01→04(모델/조립/도구/API), 05→08(상태/기억/분기/승인), 09→12(인덱스/검색/생성/소스), 13(조립/평가) 순서로 읽는다. curriculum-coverage는 `study_list.txt`의 13개 항목을 매핑한 목록 점검이며 실행 완료 증거가 아니다. 사내 사례는 학습 시나리오이며 실제 연결/설비 상태/DRM 비율/모델 alias를 확인하지 않았다.

접속 설정은 문서에 반복된 내부 URL/더미 키를 제거하고 환경변수로 통일했다. 값을 출력하거나 저장소에 넣지 않는다. 공개 OpenAI 예시와 사내 예시는 해당 환경에서 하나를 선택한다. 실제 통신은 이번 검증에 포함하지 않았다.

| 설정 | 용도/조건 |
|---|---|
| `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | 승인된 chat endpoint·인증·서빙 model id. 원래 Kimi-K2.5 이름은 시나리오의 alias 예시 |
| `EMBEDDING_MODEL` | 같은 gateway의 dense embedding id. 별도 gateway라면 base/key도 분리해야 함 |
| `FALLBACK_MODEL` | 02절 fallback 후보. 기능/출력 계약·장애 영역이 같은지 확인 |
| `VLM_MODEL` | 12절 vision alias. 원래 Qwen3-VL-30B 이름과 공개 model id의 동일성 미확인 |
| `MES_BASE_URL`, `MES_TOKEN` | 승인된 조회 API 설정. 토큰 권한/만료·응답 schema 확인 |

OpenAI 호환은 모든 tool/JSON schema/vision/stream/인증·토큰 계산의 호환을 뜻하지 않는다. embedding 예제는 raw text/float encoding을 보내며 모델별 입력 한도는 호출자가 확인한다. 이는 사내 BGE endpoint가 지원된다는 증거가 아니다. 500/50 chunk_size/overlap은 문자 기반 시작값이며 BGE tokenizer의 token 수가 아니다. 실제 모델·판본·차원·정규화가 바뀌면 기존 index와 재임베딩 조건을 검토한다.

## 확인 판본과 실행 경계

2026-10-04 임시 환경 Python3.14.2에서 langchain1.4.3/core1.6.6/openai integration1.6.7/community0.4.2/classic1.0.8, langgraph1.2.12/checkpoint-sqlite3.1.1, faiss-cpu1.15.1, OpenAI SDK3.24.0, Pydantic2.13.5, httpx0.28.1/httpx2.13.1을 사용했다. SQLRecordManager import에는 SQLAlchemy asyncio extra/greenlet이 필요했다. 상위 패키지 pin은 전체 OS/ABI/의존성 lock이 아니며 다른 사내 환경 호환은 미확인이다.

로컬 Runnable·graph·checkpointer·FAISS는 실제 라이브러리로 실행했다. LLM/embedding 값은 검증용 fixture로 제공했다. 별도로 실제 SDK와 integration의 MockTransport 요청·SSE·tool/structured schema·raw embedding 직렬화, HTTP 응답/오류·제한 재시도를 실행했다. 실제 socket·서버 인증/모델 응답·PDF/웹 다운로드·OCR 정확도·운영 승인/부하·Gradio 화면을 검증한 것은 아니다.

## 대표 판단

- LangChain v1의 agent는 create_agent를 사용하고 legacy retriever/indexing은 langchain_classic 경로로 수정했다. create_agent도 checkpoint/HITL을 지원한다. 직접 node/edge가 필요한 예제를 커스텀 LangGraph로 구분한다.
- Runnable 공통 메서드는 실제 streaming/native async·병렬 성능 보장이 아니다. batch 동시성을 제한하고 provider rate limit/취소·중간 parser의 blocking을 따로 확인한다.
- Pydantic 구조 검증은 사실성 보증이 아니다. severity 1~5와 yes/no는 실제 타입/범위로 제한하고 거부/파싱/검증 실패를 다룬다.
- trim은 호출 입력을 줄이고 기존 checkpoint 상태를 지우지 않는다. thread id는 인증 수단이 아니다. SQLite graph는 connection context 밖에서 사용하지 않는다.
- 승인 예제는 메시지 기록만 하며 실제 조치 API를 호출하지 않는다. interrupt 뒤 재개는 해당 node를 다시 실행하므로 그 앞의 부작용은 중복 실행 조건을 검토한다.
- Flat은 주어진 거리의 exact 최근접이지 의미/답 정확도100%가 아니다. 예제의 unit L2와 cosine 순위 관계·HNSW metric/정규화·Flat 삭제를 구분한다. load_local의 pickle 허용은 직접 생성한 신뢰 파일에만 적용한다.
- EnsembleRetriever는 순위 RRF를 결합한다. LLMChainFilter는 필터이며 점수 기반 reranker가 아니다. 관련 문서 반환은 실제 사용한 근거나 정확한 claim citation의 증거가 아니다.
- CRAG 예제는 재작성 최대2회인 축약 흐름이며 논문의 웹검색/전체 evaluator가 아니다. Self-RAG 논문은 reflection-token 학습·추론 방법이고 별도 LLM judge 예제와 다르다.
- langchain-experimental은 공식 유지보수 중단/archived 상태다. SemanticChunker 예시는 기존 학습 맥락만 보존했고 신규 운영 채택/대체 설계는 보류했다.
- 13절의 keyword 포함률은 정답률이 아니다. 30문항/80%/5초/환각0건은 제안 목표이며 구현/실측 성과가 아니다. 오류 문자열도 자동 복구·agent 생존을 보장하지 않는다.

## 공식·일차 자료

확인일은 모두 2026-10-04. rolling/main 문서는 설치 판본과 따로 대조했다. 다음 근거를 벗어난 사내 성능·가용성은 미확인이다.

| 자료 | 확인한 주장 |
|---|---|
| [LangChain v1 migration](https://docs.langchain.com/oss/python/migrate/langchain-v1) | create_agent 이관·system_prompt·classic namespace |
| [Models](https://docs.langchain.com/oss/python/langchain/models), [ChatOpenAI](https://docs.langchain.com/oss/python/integrations/chat/openai) | 내용/usage·tool/structured output·호환 조건 |
| [Runnable reference](https://reference.langchain.com/python/langchain-core/runnables/base) | 기본 async/thread/batch·실제 구현별 차이 |
| [Agents](https://docs.langchain.com/oss/python/langchain/agents), [Tools](https://docs.langchain.com/oss/python/langchain/tools), [Structured output](https://docs.langchain.com/oss/python/langchain/structured-output) | agent runtime·tool 실행·schema validation과 오류 |
| [Graph API](https://docs.langchain.com/oss/python/langgraph/use-graph-api) | 상태/reducer/분기·superstep·stream mode |
| [Checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers), [Persistence](https://docs.langchain.com/oss/python/langgraph/persistence), [Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts) | 저장 수명·thread·interrupt/replay 조건 |
| [FAISS index wiki](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes), [거리](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances) | exact/approximate·HNSW·unit L2/cosine |
| [LangChain community FAISS 구현](https://github.com/langchain-ai/langchain-community/blob/main/libs/community/langchain_community/vectorstores/faiss.py) | wrapper 기본값·점수/정규화·pickle load/delete |
| [OpenAIEmbeddings 구현](https://github.com/langchain-ai/langchain/blob/master/libs/partners/openai/langchain_openai/embeddings/base.py) | non-OpenAI raw text/float encoding 옵션 |
| [BGE-M3 model card](https://huggingface.co/BAAI/bge-m3) | dense1024·learned sparse/multi-vector; BM25와 구분 |
| [CRAG 논문](https://arxiv.org/abs/2401.15884), [Self-RAG 논문](https://arxiv.org/abs/2310.11511) | 원 방법과 축약 구현 구분 |
| [Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval) | 문맥 prefix 전략; 사내 개선폭으로 일반화하지 않음 |
| [HTTPX exceptions](https://www.python-httpx.org/exceptions/), [Tenacity](https://tenacity.readthedocs.io/en/latest/) | 오류 종류·선택적 재시도·상한/재발생 |
| [Document loaders](https://docs.langchain.com/oss/python/integrations/document_loaders/), [Qwen3-VL-30B-A3B card](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct) | 로더·vision model 후보; 실제 사내 alias/OCR 미확인 |
| [experimental sunset issue87](https://github.com/langchain-ai/langchain-experimental/issues/87) | 2026-05-22 유지보수 중단 발표·2026-05-26 repo archive |

## 남은 미확인

실제 모델/endpoint·tool schema/JSON/vision/usage 지원·Korean BM25 tokenization·검색/인용 품질·DRM/권한/export·OCR 정확도·PDF/웹 실문서·영속 DB/승인/동시성/쓰기업무·Gradio/UI·Claude 협의·Obsidian 읽기 화면을 확인하지 않았다. [정리 기록](./organization-log.md)에 문서별 수정과 검증 결과를 남겼다.
