---
tags: [rag, documentation-review]
reviewed_on: 2026-10-04
review_status: partial
document_type: maintenance_record
category_major: "AI·DT"
category_middle: "문서 관리"
category_minor: "정리·검증 기록"
note_kind: "관리 기록"
classified_on: "2026-10-05"
---

# RAG 정리 기록

## 범위와 현재 상태

원래 Markdown38개·11,542줄을 발견했다. 추가 AGENTS.md/CLAUDE.md/CONTEXT.md는 이 주제 아래 없었다. ai-dt 안의 다른 주제와 독립성을 유지한다. 새 README는 기존 하위 주제의 목차/읽기 순서를 제공한다. 원문 개별 기술 검토는 총38개(대화 메모리1·조립 예제3·LangGraph4·고급 RAG5·Milvus3·OpenSearch기초/클라이언트3·BM25/벡터/하이브리드 검색3·성능/Settings2·핸들러1·RAG연동1·OpenSearch대화메모리1·청킹목차/총론2·DOCX1·XLSX1·PDF1·PPTX1·2026전략1·DRM목차1·화면추출1·VLM청킹1·입력전환1)이며 미검토 원문은 없다. 사내/모델/정책·Claude·전체 읽기 화면은 미완료다. 본 기록과 새 README를 포함한 현재 검토 문서는40개다. 실행 코드·첨부·비Markdown·기존 사용자 변경을 수정하지 않았고 이동/삭제/커밋/푸시가 없다.

## 문서별 검토 결과

| 원래 문서 | 결과 |
|---|---|
| [advanced-rag/README.md](./advanced-rag/README.md) | 역할/읽기 순서·품질 가설과 비용·schema/모델 적용 조건; partial/2026-10-04 |
| [advanced-rag/advanced-rag-pipeline.md](./advanced-rag/advanced-rag-pipeline.md) | 명시 생성/재사용·기존 DB 보존·manifest/모델 계약·빈 입력/문자 fallback; partial/2026-10-04 |
| [advanced-rag/agentic-rag-implementation.md](./advanced-rag/agentic-rag-implementation.md) | 원래 의도/검색어 분리·유한 재검색/abstain·schema unknown/명시 조립·워크숍 미확인 보존; partial/2026-10-04 |
| [advanced-rag/multi-agent-rag-integration.md](./advanced-rag/multi-agent-rag-integration.md) | 명시 도구/skill 등록·실제 task 반환/stream·완료건수·SQLite read-only/authorizer·초안/권한 한계; partial/2026-10-04 |
| [advanced-rag/rag-extensions.md](./advanced-rag/rag-extensions.md) | HyDE 검색어/의도 분리·이력 실제 전달/검색 해소와 구분·SQLite context·비용/워크숍 미확인; partial/2026-10-04 |
| [langchain-langgraph/README.md](./langchain-langgraph/README.md) | 실행 조건/읽기 순서·메타데이터 이동/기존 주제 밖 MCP 참조를 기술 이름으로 유지; partial/2026-10-04 |
| [langchain-langgraph/langchain-langgraph-basics.md](./langchain-langgraph/langchain-langgraph-basics.md) | 설치 판본/LCEL parser·입력 검사/부분 state 갱신·최소 예제와 확장 후보 구분; partial/2026-10-04 |
| [langchain-langgraph/rag-tool-calling-playbook.md](./langchain-langgraph/rag-tool-calling-playbook.md) | 누락 tool route·실제 ToolNode 결과 루프/호출 예산·허용 도구/가상 일정·문자 청킹/빈 검색; partial/2026-10-04 |
| [langgraph/README.md](./langgraph/README.md) | frontmatter/읽기 순서·역할·SQLite 판본/외부 렌더 조건; partial/2026-10-04 |
| [langgraph/langgraph-advanced.md](./langgraph/langgraph-advanced.md) | 명시 interrupt/승인·거절·unknown/executor 분리·가상 subgraph/SQLite context·replay/stream mode; partial/2026-10-04 |
| [langgraph/langgraph-basics.md](./langgraph/langgraph-basics.md) | LCEL branch/parallel·상태/문서 타입·reducer/정확한 분류·unknown/로컬 Mermaid; partial/2026-10-04 |
| [langgraph/langgraph-rag.md](./langgraph/langgraph-rag.md) | 논문 범위·원래 질문/검색어 분리·정확yes/no·빈 근거 abstain·유한 재생성/재검색; partial/2026-10-04 |
| [llm-conversation-memory.md](./llm-conversation-memory.md) | 상태 reducer/요약·후보 schema/검색 결과·공식 근거·로컬 fixture 검토; partial |
| [milvus/README.md](./milvus/README.md) | 읽기 순서·직접 SDK/래퍼 schema 역할·판본/명시 호출 조건; partial/2026-10-04 |
| [milvus/milvus-basics.md](./milvus/milvus-basics.md) | 자동 삭제 제거·일관 MilvusClient/schema/category·정확 메트릭·FLAT fixture·hybrid 별도 전제; partial/2026-10-04 |
| [milvus/milvus-rag-integration.md](./milvus/milvus-rag-integration.md) | 문자/PDF 적재·최초 add/이후 stable ID upsert·래퍼 검색·유한 승인 폴백/unknown/abstain; partial/2026-10-04 |
| [opensearch/README.md](./opensearch/README.md) | 학습/운영가정·11문서 읽기 순서/역할·깨진 주제 밖 코드 참조 제거; partial/2026-10-04 |
| [opensearch/conversation-memory-opensearch.md](./opensearch/conversation-memory-opensearch.md) | 완전JSON/Faiss조건·효율적user필터/should필수·dense배치/미완료요약·자료/권한/미확인manager 구분; partial/2026-10-04 |
| [opensearch/hybrid-search.md](./opensearch/hybrid-search.md) | normalization/hybrid/RRF 판본·정규화/순위/재순위 역할·명시 후보/가중치·원문/identity 보존; partial/2026-10-04 |
| [opensearch/keyword-search-bm25.md](./opensearch/keyword-search-bm25.md) | Nori 설치·BM25 3.0 점수·구문/필터/강조 조건·create 충돌/부분 검색 실패; partial/2026-10-04 |
| [opensearch/opensearch-basics.md](./opensearch/opensearch-basics.md) | 현재 라이선스/거버넌스·node.roles/샤드 조건·유효 JSON·명시 실습/기존 index/ID 보존; partial/2026-10-04 |
| [opensearch/opensearch-handler.md](./opensearch/opensearch-handler.md) | 역사적 로컬API/기본값 미확인·SDK 호환/순수함수 오해·toy mapping/ID·명시 import 경로; partial/2026-10-04 |
| [opensearch/performance-optimization.md](./opensearch/performance-optimization.md) | OSS/managed 수치 조건·heap/native/cache·refresh/fsync/flush·force merge·REST 구문; partial/2026-10-04 |
| [opensearch/python-client.md](./opensearch/python-client.md) | pool_maxsize/async 설정 구분·bulk 항목 실패/바이트·scan 해제·async 오류/취소 close; partial/2026-10-04 |
| [opensearch/rag-integration.md](./opensearch/rag-integration.md) | community sunset/판본·실제 import/검색 방식·원질문/근거/생성 분리·가중RRF identity; partial/2026-10-04 |
| [opensearch/settings/README.md](./opensearch/settings/README.md) | mapping/doc_values·새 index template/alias·rollover OR·ISM index 수명/부트스트랩 순서·기존 자원 보존; partial/2026-10-04 |
| [opensearch/vector-search-knn.md](./opensearch/vector-search-knn.md) | 엔진/필터 판본·metric/점수/recall·embedding 계약/제공 벡터·기존 데이터 보존; partial/2026-10-04 |
| [token_strategy/README.md](./token_strategy/README.md) | 추출/청킹/토큰화 구분·11개 목차/읽기 순서·방법론 대표와 형식별 역할; partial/2026-10-04 |
| [token_strategy/docx-tokenization.md](./token_strategy/docx-tokenization.md) | 실제본문순서/Heading·표투영/중첩/생략셀·문자분할/source·Mammoth중복/경고·미확인추출범위; partial/2026-10-04 |
| [token_strategy/overview-chunking-methods.md](./token_strategy/overview-chunking-methods.md) | 실제 splitter 판본/단위·separator 상한/semantic buffer·헤더출처·원문 경계/late token pooling·근거없는품질표수정; partial/2026-10-04 |
| [token_strategy/pdf-tokenization.md](./token_strategy/pdf-tokenization.md) | 실제 표/OCR 기능·혼합 페이지/50자 판별 한계·볼드 플래그·페이지 Markdown/SDK PNG·후처리 보존; partial/2026-10-04 |
| [token_strategy/pptx-tokenization.md](./token_strategy/pptx-tokenization.md) | 재귀 그룹/동일 제목 본문·표/병합·노트 unknown·빈 슬라이드 번호·별도 Vision/이웃 맥락·변환 실패 처리; partial/2026-10-04 |
| [token_strategy/recent-rag-strategy-2026.md](./token_strategy/recent-rag-strategy-2026.md) | 역사 원문 정확 보존; 모델 명시값/서버 조건·Late pooling·standard tokenizer/청크 limit·fusion/rerank 구분; partial/2026-10-04 |
| [token_strategy/when_drm/README.md](./token_strategy/when_drm/README.md) | 읽기 순서/취득·청킹·전환 역할; 당시99%/유일 주장 미확인·권한/입력unknown 구분·도식 보존; partial/2026-10-04 |
| [token_strategy/when_drm/post-drm-hybrid.md](./token_strategy/when_drm/post-drm-hybrid.md) | unknown/명시 승인 경로·대표 파서 주입/출처·비교값 비정답·실제 생성 파일/SDK mock 검증; partial/2026-10-04 |
| [token_strategy/when_drm/screenshot-vlm-pipeline.md](./token_strategy/when_drm/screenshot-vlm-pipeline.md) | 모델 분류/API·PNG/manifest 순서·응답/unknown·휴리스틱 정정; 생성 자료/SDK mock 검증; partial/2026-10-04 |
| [token_strategy/when_drm/vlm-chunking-strategy.md](./token_strategy/when_drm/vlm-chunking-strategy.md) | Markdown parser/문자 범위·페이지/empty/unknown·원문 보존·생성 schema 분리; partial/2026-10-04 |
| [token_strategy/xlsx-tokenization.md](./token_strategy/xlsx-tokenization.md) | 원수식/캐시unknown·좌표/실제행·0/False/NA·시트중복제거·병합투영/영역한계·통계/LLM구분; partial/2026-10-04 |

## 판단과 Claude 협의

HERDR_ENV=1을 확인한 뒤 current pane 연결을 읽기 전용 재확인했다. sandbox에서는 PermissionDenied, 허용된 재확인에서는 pane_not_found였다. 작업 전용 Claude pane을 확보하지 못했고 다른 프로젝트 pane은 제어하지 않았다. Claude 의견은 없다. 같은 주제의 완전 중복 통합·문서 분할/재분류와 기술 선택/임계값은 협의 불가로 보류한다. 기존 명확한 폴더 역할을 안내하고, 공식 근거·실제 구현으로 확인된 오류만 수정했다.

## 대화 메모리 변경

- 3계층은 설명용 설계 분류, 요약은 정보 손실을 보장하지 않는 압축임을 명시했다. 논문의 검색 점수를 min-max 정규화 후 가중합으로 바로잡았다. 기존 곱셈식은 논문 식이 아니다.
- MessagesState의 reducer는 최근 목록 반환만으로 과거를 삭제하지 않는다. RemoveMessage로 제거하고 텍스트 Human/AI 대화의 시작 경계를 유지했다. tool 대화는 별도 계약 필요로 실패시킨다. 삭제는 현재 state 기준이며 이전 checkpoint·로그·DB의 삭제를 대신하지 않는다.
- 모델 추출은 미검수 후보다. 사용자 진술만 입력하고 JSON 문자열 리스트를 검증했다. 기존 후보를 무조건 덮어쓰거나 모순을 최신으로 자동 해결하지 않는다. 실제 source/time·확인·정정/삭제 시스템은 미구현이다.
- graph/DB 자동 접속 코드를 함수·callback 조립으로 바꿨다. InMemorySaver의 프로세스 수명과 동기 순차 실행을 명시했다. 사용자/thread 키는 인증·소유권 검증을 대신하지 않는다.
- Milvus search의 한 query 결과는 nested hits이며 fact는 entity에 있다. filter_params로 사용자 값을 바인딩하고 반환 소유권도 확인한다. collection schema/index/load·동일 모델/차원은 사전 조건이다. 저장 id/UTC/candidate와 유한 벡터·점수를 검사한다. 실제 서버·SDK 실행은 하지 않았다.
- Mem0 OSS/managed와 Zep 서비스/별도 Graphiti를 구분했다. 회사 채택·최신성·운영 적합성을 단정하지 않았다.

## 검증 1 — 목록·고유 내용

원래38개 경로가 모두 남아 있고 미수정37개 SHA-256을 최초 목록과 비교했다. 대화 메모리의 기존 제목·fence 수(4 Python 포함)와 요약/추출/중요도/아키텍처/그래프/벡터 DB/프레임워크/트레이드오프 맥락을 보존했다. 업무 기록을 현재 사실로 다시 쓰지 않았다. 이 단계에서38개 전체 기술 검토를 완료로 계산하지 않는다.

## 검증 2 — 근거와 로컬 실행

확인일 2026-10-04. 공식 rolling 자료와 설치 판본을 구분하며 최신이라는 뜻은 아니다.

| 일차 근거 | 확인 내용 |
|---|---|
| [LangGraph memory](https://docs.langchain.com/oss/python/langgraph/add-memory) | thread/장기 저장 구분·RemoveMessage·대화 경계·checkpointer 수명 |
| [Generative Agents 논문](https://arxiv.org/html/2304.03442v2) | 정규화한 recency/importance/relevance의 가중합·context 예산 |
| [Milvus search2.6](https://milvus.io/api-reference/pymilvus/v2.6.x/MilvusClient/Vector/search.md), [filter templating](https://milvus.io/docs/filtering-templating.md) | nested hits/entity·filter_params |
| [Mem0 add](https://docs.mem0.ai/core-concepts/memory-operations/add), [Zep 저장소](https://github.com/getzep/zep) | OSS/managed 구분·서비스/Graphiti 역할. 설치·배포·가격·회사 적합성은 미확인 |

임시 Python 환경의 실제 LangGraph1.2.12/LangChain-core1.6.6에서 StateGraph와 InMemorySaver를 실행했다. 네트워크 모델 대신 fake invoke를 사용했다. Python AST4 통과, 실제 reducer/RemoveMessage와 반복 요약/남은 메시지/서로 다른 thread·후보 중복 방지, 잘못된 JSON/후보 schema의 저장 거부, tool 대화 trim 거부를 확인했다. Milvus 부분은 공식 반환 형태의 DB/embedding callback fixture로 nested entity·사용자 값 바인딩·다른 소유자 거부·UTC·candidate·차원/NaN·importance/top_k 경계를 확인했다. 실제 pymilvus·서버 필터 실행·DB 격리·모델 품질/인젝션 내성·인증은 검증하지 않았다.

## 검증 3 — 참조·메타데이터·Obsidian

현재 검토3개만 상대 링크·앵커·첨부와 YAML key/검토일/type/tags를 검사하고 pm_notes vault CLI에서 조회한다. 다른37개는 기존 오류가 있더라도 아직 검토 대기다. vault 경로는 /Users/daeyoung/Codes/pm_notes로 확인되어 있으며 다른 vault는 수정하지 않는다. 이전 읽기 화면 연결 timeout 이후 UI 탐색/렌더링은 미완료다. 최종 결과는 아래에 추가한다.


### 현재 부분 검증 결과

검토3개 상대 링크·앵커·첨부 오류0, YAML/CLI properties3개 통과. 원래38개 경로·미수정37개 SHA-256·ai-dt 비Markdown18개 보존과 원문 제목/fence 비교 통과. AST4·실제 StateGraph/실패 입력 fixture를 재검증했다. git diff --check는 실제 git 경로 ai-dt/rag 범위에서 통과했다.

CLI vault info=path는 /Users/daeyoung/Codes/pm_notes를 반환했다. vault 색인의 실제 경로는 ai-dt/rag였다. 대문자 RAG 조회는 파일 없음으로 실패했으며, 색인 경로와 로컬 samefile을 대조한 후 소문자 경로로 재조회했다. 파일을 이동/이름 변경하거나 다른 vault를 수정하지 않았다.

CLI open 후 active leaf preview 상태만으로 본문 렌더링을 통과시키지 않았다. 최초 screenshot은 이전 회귀 문서를 보여줘 실패 증거로 처리했다. 현재 leaf 표시 후 RAG README의 실제 screenshot에서 한국어 제목/검토 callout/읽기 순서 표·properties를 확인했고 DOM에서 같은 주제 링크들을 확인했다. 대화 메모리 링크 클릭은 실제 ai-dt/rag/llm-conversation-memory.md로 이동했지만 본문 DOM/screenshot은 이전 목차 화면이어서 해당 본문 렌더링은 미완료다. 정리 기록 본문의 화면 확인도 대기다. 외부 서비스·실제 모델 품질·DB 인증/격리·Claude 협의는 미확인이다.


## LangChain·LangGraph 조립 예제 추가 검토

### 변경과 판단

원래3개 경로·코드 밖 제목·fence 수를 보존했다. README의 H1 뒤 YAML을 실제 frontmatter로 옮기고 원래 last_updated=2026-04-08을 유지했다. 모델 id는 환경변수/명시 객체로 받으며 네트워크 invoke를 자동 실행하지 않는다. 실행 입력 파일과 조립 순서(3절 tool 정의 후2절 router 함수 호출)를 설명했다. 기존 MCP 주제 밖 링크는 기술 이름/역할 설명으로 유지하고 새 교차 링크는 만들지 않았다.

원래 tool route는 목적지가 없었고 bind_tools만으로는 도구가 실행되지 않았다. 실제 StateGraph의 tool_answer 경로와 ToolNode의 결과 메시지→모델 루프를 연결했다. 허용 도구·schema·호출 예산/recursion_limit을 검사하며 오류는 전파한다. 일정 데이터는 가상 읽기 예제다. 예약을 실제 수행하는 구현/승인·권한·timeout·관측·저장은 없다. 키워드 라우터는 의도/최신 근거를 보장하지 않고, 검색이 비면 미확인 답변으로 종료한다. sources는 검색 metadata 목록이며 답변별 인용/근거 타당성 검증이 아니다.

청킹800/120은 기본 len의 문자 단위/목표 overlap이다. FAISS는 프로세스 안 검색 예제이며 모델/차원·데이터 갱신·권한 필터·운영 품질은 별도다. 20문서/10실패 사례는 과제 제안값이다. HERDR_ENV=1 current pane 재확인은 pane_not_found였다. Claude 의견이 없으며 통합/재분류·임계값·실제 도구 선택은 계속 보류한다.

### 검증 1 — 원문 보존

추가3개의 원문 제목/fence 수와 경로를 보존했다. chain·최소 graph·라우터·TextLoader/FAISS·일정 tool·운영 확장/실습 과제의 고유 역할을 유지했다. 원문38개 중4개 개별 검토이며 다른34개는 미수정/대기다. 이동/삭제/코드 파일/첨부 변경·커밋/푸시는 없다.

### 검증 2 — 근거와 실행

확인일2026-10-04. Python3.14.2 임시 환경: langchain1.4.3, langgraph1.2.12, langchain-core1.6.6, langchain-openai1.6.7, community0.4.2, text-splitters1.1.3, faiss-cpu1.15.1, OpenAI SDK3.24.0. 확인 판본은 최신/보안 인증·전체Python3.11+/OS 호환 lock이 아니다.

| 일차 자료 | 확인 내용 |
|---|---|
| [Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api) | state key 갱신·조건부 경로·종료/recursion limit |
| [Workflow/ToolNode](https://docs.langchain.com/oss/python/langgraph/workflows-agents) | tool 호출과 결과 메시지·허용 tool node의 역할 |
| [ChatOpenAI](https://docs.langchain.com/oss/python/integrations/chat/openai) | 객체/도구 schema 연결·provider 계약은 별도 |
| [Recursive splitter](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter) | 기본 문자 길이·목표 overlap |
| [공식 FAISS adapter 소스](https://raw.githubusercontent.com/langchain-ai/langchain-community/main/libs/community/langchain_community/vectorstores/faiss.py) | from_documents/retriever/metadata·신뢰하지 않는 pickle 로드 조건. rolling 소스와 설치 판본 구분 |

FAISS 통합 안내 URL은 web 도구에서 Internal Error로 조회하지 못해 공식 구현과 설치 소스/실행으로 대조했다. 실제 TextLoader·RecursiveCharacterTextSplitter·FAISS와 fake embedding으로 임시 한국어 파일의 문자 크기/source 보존을 확인했다. 실제 LCEL/StrOutputParser·StateGraph·ToolNode로3경로, ToolMessage id/결과, 빈 파일/검색, 잘못된 질문·날짜·허용되지 않은 도구·반복 호출 예산 초과를 검증했다. 추가 AST7·bash1, 실제 ChatOpenAI/OpenAI SDK의 HTTP MockTransport4요청으로 chain/graph·tool schema/결과 왕복을 확인했다. 원격 socket/실제 모델·서비스 일정·사용자 권한·한국어 품질·운영 fallback 검증은 아니다.

### 검증 3 — 현재6개

추가3개와 기존 검토3개의 상대 참조·앵커·첨부/YAML/CLI를 대조한다. 아직34개 전체 메타데이터/링크를 검토한 것으로 계산하지 않는다. 읽기 화면 결과는 아래에 덧붙인다.


### 조립 예제 단계 최종 부분 검증 결과

현재6개 상대 링크·앵커·첨부 오류0, 중복 없는 YAML/검토일/type/tags와 CLI properties6개 통과. 원래38경로·미수정34개 SHA-256·ai-dt 비Markdown18개 보존. 추가3개 제목/fence·AST7·bash1·실제 framework/실패 입력·SDK HTTP MockTransport4요청 검증 및 git diff --check 통과. 기존 대화 메모리의 AST4/StateGraph fixture 결과와 합쳐 현재 AST11이다.

Obsidian CLI open·revealLeaf·setActiveLeaf로 LangChain·LangGraph 목차를 선택했고 active view의 path/preview는 확인했다. 그러나 본문 DOM은 breadcrumb만 있고 screenshot은 이전 대화 메모리의 빈 본문이어서 새 목차/입문/플레이북 읽기 화면을 검증 완료로 표시하지 않는다. dev:errors는 No errors captured를 반환했다. 오류가 없다는 보증이나 문서 렌더링 성공 증거는 아니다. 다른 vault·설정·plugin을 수정하지 않았다. 실제 서버/모델/회사 권한·Claude 의견·새3개와 이전 memory/log의 읽기 화면은 미완료다. RAG 원문4개만 개별 검토했고34개는 대기다.

검증 결과를 기록하는 임시 스크립트에 문자열 인용부호 문법 오류가 있어 실행되지 않았다. 문서/API 실패와 구분하여 해당 보조 스크립트를 고친 후 목록·참조·동작 검증을 재실행했다.


## LangGraph 시리즈 추가 검토

### 변경과 Claude 협의

원래4개 frontmatter를 파일 맨 앞으로 옮기고 last_updated=2026-01-31은 유지했다. 원래 경로·코드 밖 제목과 fence 수를 보존했고 README에 실행 조건 절만 추가했다. 이동/삭제/코드 파일·첨부 변경은 없다. 기본 그래프, Corrective RAG 기반 학습 루프, 승인/하위 그래프/저장/stream의 고유 맥락을 유지했다. 같은 RAG 주제 안 기존 조립 입문을 설치 안내로 참조하고 새 주제 밖 링크는 만들지 않았다. 기존 AI/DT 전체 목차는 이름/역할 설명으로 남겼다.

HERDR_ENV=1 current pane은 다시 pane_not_found였다. Claude 의견은 없다. 완전 중복 통합/분류와 운영 threshold·외부 검색/승인·DB 선택은 협의가 필요한 결정으로 계속 보류한다. 공식 문서와 실제 API로 확인한 동작 오류만 수정했다.

- 기초: LCEL도 branch/parallel을 지원하므로 선형만 가능하다는 구분을 바로잡았다. Document→텍스트 상태의 타입 계약과 reducer의 새 부분만 반환하는 누적을 설명했다. 분류 출력은 technical/general만 허용하고 unknown을 general로 바꾸지 않는다. Mermaid 텍스트 생성은 외부 PNG 서비스 호출과 분리했다.
- RAG: 원문 질문을 검색어 재작성으로 덮어쓰지 않는다. 관련/근거 판정은 정확 yes/no만 받는다. 빈 근거로 생성하지 않고 재검색·재생성 각각의 유한 한도 후 abstain한다. judge_supported는 모델 판정 상태이며 정답 인증이 아니다. 논문의 지식 정제/외부 검색 전체 구현이나 실제 웹 폴백이 아니다. 디버깅은 raw 내용 대신 노드/수량/상태를 반환한다.
- 고급: 동적 interrupt와 명시 boolean resume를 사용하며 거절/미확인 전에 executor가 실행되지 않는다. LLM의 결과 문장과 실제 작업 수행을 구분했다. node 재실행/부작용·plan 재검토·thread 소유권/인증/멱등성은 별도 조건이다. subgraph는 가상 검색/앞50문자 축약이며 실제 요약 품질을 보장하지 않는다.
- SQLite: from_conn_string context manager 안에서 실제 saver/graph를 사용한다. question 전용 builder를 분리해 이전 query/plan 스키마와 혼용하지 않는다. 프로세스 메모리와 파일 저장 수명을 구분하고 운영용이라고 단정하지 않는다. checkpoint replay/새 checkpoint 갱신은 외부 부작용 롤백이 아니다.
- Streaming: updates/values/messages와 v2 callback events를 구분한다. notebook의 기존 loop와 script asyncio.run을 구분하고 수집되지 않은 텍스트 stream은 오류다. rolling의 v3 event API와 설치판본 예제를 자동 동일시하지 않는다.

### 검증 1 — 목록과 고유 내용

추가4개 원문 경로/제목/fence 수 보존(README는 새 조건 절 추가), 기존 GraphState/분류/reducer/RAG 노드/분기/승인/subgraph/checkpointer/stream의 역할을 대조했다. 현재 원문8개 검토·30개 대기이며 신규2개 포함 검토10개다. 다른30개와 비Markdown 바이트 보존을 확인한다. 커밋/푸시는 하지 않았다.

### 검증 2 — 공식 근거와 실제 실행

확인일2026-10-04. 임시 Python3.14.2/langgraph1.2.12/langchain-core1.6.6/langgraph-checkpoint-sqlite3.1.1에서 실행했다. 이 판본은 최신/운영 호환성/모델 품질 보증이 아니다. 나머지 공통 환경은 앞선 조립 예제 기록을 따른다.

| 일차 자료 | 대조 내용 |
|---|---|
| [Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api) | 상태/reducer·갱신·조건부 route·TypedDict와 검증 |
| [Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts) | checkpoint/thread·Command resume·node 재실행과 명시 응답 |
| [Persistence](https://docs.langchain.com/oss/python/langgraph/persistence) | checkpoint history/replay·저장 backend 수명 |
| [Subgraphs](https://docs.langchain.com/oss/python/langgraph/use-subgraphs) | 공유 key 또는 wrapper 입출력 변환 |
| [Streaming](https://docs.langchain.com/oss/python/langgraph/streaming) | stream mode/이벤트·판본별 API 구분 |
| [Corrective RAG 원 논문](https://arxiv.org/abs/2401.15884) | retrieval evaluator/지식 정제·외부 검색. 현재 축약 예제와 범위 구분 |
| [FastAPI dependencies](https://fastapi.tiangolo.com/tutorial/dependencies/) | sample_docs의 Depends/DI 개념. 근거 데이터로 실제 운영 품질을 증명하지 않음 |

추가 AST11 통과. 실제 StateGraph/reducer/Document 텍스트, 두 정확한 분류와 unknown 실패/로컬 Mermaid, 재작성·재생성 한도/원래 의도/빈 근거 abstain/yes 부분문자열 거부/updates 안전한 출력, 실제 interrupt payload/검토 전 미실행/승인·거절·미확인, 공유 key subgraph, 임시 SQLite 파일의 context 수명/재열기/history/replay count, 실제 GenericFakeChatModel의 sync updates와 async v2 stream을 검증했다. 모델/검색 결과는 fixture다. 실제 원격 모델·운영 검색 정확도/논문 성능 재현·권한·side effect·운영 DB 동시성·token latency/UI는 검증하지 않았다.

### 검증 3 — 현재10개 참조와 앱

현재10개만 링크·앵커·첨부·unique YAML/검토일/type/tags·대상 vault CLI를 검사한다. 다른30개를 검토 완료로 계산하지 않는다. 원격/Claude·읽기 화면의 미완료는 결과와 별도로 유지한다. 앱의 새 파일 선택과 실제 본문이 어긋난 이전 관찰 이후 새4개 화면 검증은 미완료다. 새 plugin/설정이나 다른 vault를 수정하지 않는다.


### LangGraph 단계 최종 부분 검증 결과

현재10개의 상대 링크·앵커·첨부 오류0, 중복 없는 YAML/검토일/type/tags·CLI properties10개 통과. 원래38개 경로·미수정30개 SHA-256·ai-dt 비Markdown18개 보존. 추가4개 원문 제목/fence 수(README 조건 절 추가 기록)·AST11·실제 StateGraph/SQLite/async 실행 및 실패 입력을 재검증했다. wrapper를 우회한 문자열 승인도 실제 interrupt node에서 거부하고 executor가 실행되지 않았으며, chat stream 이벤트가 없는 callback은 빈 성공으로 반환하지 않았다. 현재 검토8개 원문의 Python AST는22개(기존4+7+추가11)다. git diff --check 통과. 실제 모델/회사 서버/권한·Claude 의견·새4개의 읽기 화면은 미완료다.


## 고급 RAG 목차·파이프라인 추가 검토

### 변경과 Claude 협의

원래2개 경로·기존 제목과 fence 수를 보존했다. README에 문서별 역할/실습 조건 절을 추가하고 H1 뒤 YAML을 실제 frontmatter로 옮겼다. last_updated=2026-07-16은 유지한다. Agentic/HyDE/메모리/다중 에이전트는 목적에 따라 추가하는 설계이며 품질 향상을 보장하지 않는다. 다른3개 고급 문서는 아직 개별 검토 대기다. PM/반도체 입력 경로와 예제 맥락을 유지했고 실제 회사 자료가 준비되었다고 주장하지 않는다.

파이프라인의 기존 폴더 자동 삭제를 제거했다. 새 경로 생성과 manifest를 가진 기존 DB 재사용을 분리하고, 기존 폴더·모델 id 불일치·미완성 manifest·저장 개수 불일치는 거부한다. manifest의 선언 모델 이름/개수는 실제 revision·내용 무결성·권한 인증을 증명하지 않는다. 실패한 새 경로는 조사하도록 남기고 자동 삭제/재색인하지 않는다. stable id와 입력 청크 digest를 기록했다. 원격 객체 생성/환경 로딩은 명시 함수이며 기존 환경을 덮어쓰지 않는다. 실질 API 호출과 실제 provider/모델 지원은 별도다.

DirectoryLoader 재귀·숨김/UTF-8/오류 처리와 입력 준비를 명시하고 빈 문서/청크를 거부한다. 빈 통계 평균은 None이다. 문자 splitter의 끝 fallback을 추가해 구분자 없는 긴 문자열도 분할한다. 500/50은 실습 제안값이고 토큰 제한이나 Markdown AST/표/fence 의미 보존이 아니다. 검색 진단은 원문 대신 개수/파일명이며 빈 검색이 DB 삭제 근거가 되지 않는다.

HERDR_ENV=1 current pane 확인은 pane_not_found였다. 작업 전용 Claude 연결/의견은 없고 다른 pane을 제어하지 않았다. 완전 중복 통합/문서 재분류·운영 임계값/모델/검색 엔진 채택은 협의 필요로 보류한다. 공식 API와 로컬 실행에서 확인한 오류만 수정했다.

### 검증 1 — 목록과 고유 내용

추가2개 기존 제목/fence 수와 환경/로딩/분할/embedding/DB/검색/MMR/PM·반도체 예제 맥락을 대조했다. README의 역할 조건 절만 새로 추가했다. 원래38개 경로를 보존했고 현재 원문10개 검토·28개 대기다. 신규 목차/기록 포함 검토12개이며 대기 문서를 완료로 계산하지 않는다. 이동/삭제/실행 코드/첨부/비Markdown 변경·커밋/푸시는 없다.

### 검증 2 — 출처와 실제 실행

확인일2026-10-04. 새 임시 Python3.14.2 환경의 chromadb1.5.9/langchain-chroma1.1.0/community0.4.2/text-splitters1.1.3/langchain-openai1.6.7/core1.6.6/python-dotenv1.2.4/OpenAI SDK3.24.0을 확인했다. 최신/운영 lock/전체 OS 호환성 보증이 아니다. 첫 임시 환경 생성은 사용자 uv cache 권한으로 실패해 /private/tmp의 독립 cache로 다시 생성했다. 저장소 코드/기존 환경은 변경하지 않았다.

| 일차 자료 | 확인 내용 |
|---|---|
| [Chroma 통합](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma) | persist_directory/from_documents/get/retriever·MMR |
| [Recursive splitter](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter) | 문자 길이/구분자 fallback·목표 overlap |
| [DirectoryLoader 공식 소스](https://raw.githubusercontent.com/langchain-ai/langchain-community/main/libs/community/langchain_community/document_loaders/directory.py) | glob/recursive·hidden/loader_kwargs·silent_errors. rolling 소스와 설치 판본 구분 |
| [Chroma 공식 cookbook clients](https://cookbook.chromadb.dev/core/clients/) | 로컬 파일 client와 원격 HTTP 역할 |
| [Structured output](https://docs.langchain.com/oss/python/langchain/structured-output) | schema/검증과 실제 내용 타당성을 구분. 다른3개 문서 구현 검증은 대기 |

DirectoryLoader 안내/Chroma persistent-client 일부 URL은 web 조회에서 Internal Error여서 확인 성공 근거로 쓰지 않았다. 추가 Python AST9 통과. 실제 DirectoryLoader에서 최상위/재귀·숨김 파일/UTF-8/source·빈 입력, 실제 splitter의 공백 없는1600문자와 빈 평균, 기존 환경을 유지하는 명시 factory를 확인했다. 실제 Chroma에 fixture embedding으로 저장·검색/MMR·stable id·manifest를 검사했다. 재실행 거부 전후 모든 DB 파일 바이트가 같고, 별도 새 Python 프로세스에서 실제 영속 DB를 재열어 같은 id/검색을 확인했다. 미완성/모델 id·개수 불일치도 재생성 없이 거부했다. 실제 LangChain OpenAIEmbeddings/OpenAI SDK는 HTTP MockTransport2요청으로 payload/검색을 검증했다. Python socket 호출을 막았으며 실제 외부 모델/회사 데이터/모델 품질/원격 인증·DB 운영 동시성/보안·native 네트워크 전체 감시는 검증하지 않았다.

### 검증 3 — 참조·메타데이터·앱

현재12개만 상대 링크·앵커·첨부/중복 없는 YAML·검토일/type/tags·대상 pm_notes vault CLI를 검사한다. 기존 RAG 목차 읽기 화면은 확인했으나 이후 active path와 본문 화면이 어긋난 상태여서 새2개 읽기 화면은 미완료다. app/plugin/settings/다른 vault를 수정하지 않는다. 최종 부분 검사 결과는 아래에 추가한다.


### 고급 파이프라인 단계 최종 부분 검증 결과

현재12개 상대 링크·앵커·첨부 오류0, unique YAML/검토일/type/tags·pm_notes CLI properties12개 통과. 원래38경로/미수정28개 SHA-256·ai-dt 비Markdown18개 보존, 추가2개 기존 제목/fence·AST9·실제 DB 영속/새 프로세스 재열기·기존 DB 전 바이트 보존/실패 입력과 SDK mock2요청을 재검증했다. 현재 원문 Python AST31개다. git diff --check 통과.

고급 README CLI open/eval은 실제 해당 경로/preview·새 내용 텍스트를 반환했으나 h1/h2 DOM 목록이 비었고 실제 screenshot은 이전 LangChain·LangGraph README를 표시했다. 새 고급 문서의 탐색/렌더링 성공으로 계산하지 않는다. screenshot에는 이전 조립 README의 한국어 제목/검토 callout/frontmatter가 보였으므로 그 목차의 현재 읽기 화면만 확인했다. 새2개와 나머지 미관찰 본문/실제 모델·회사 자료/Claude 협의는 미완료다. 다른 vault/settings/plugin은 수정하지 않았다.


## Agentic RAG 개별 추가 검토

### 변경·판단·Claude 협의

원래1개 경로/코드 밖 제목·15 Python fence와 PM/반도체 시나리오를 보존했다. YAML을 실제 frontmatter로 옮기고 원래 last_updated=2026-07-16을 유지했다. 환경/판본은 앞선 조립·파이프라인 기록과 구분해 명시한다. 추가2개 확장/멀티에이전트는 아직 대기이며 이전 global/node 조립을 이 예제와 자동 혼용하지 않는다.

- 원래 question과 검색 query를 분리해 retrieve만 재작성 검색어를 쓰고 grade/generate는 원래 의도를 사용한다. prepare가 매 질문의 문서/답변/카운터를 초기화한다. 기존2회 재작성 제안 한도를 기본 builder부터 적용하고 빈 근거는 강제 생성 대신 abstain한다. recursion_limit은 별도 오류 방어이며 운영 최적 한도/timeout·취소/비용은 미확인이다.
- retriever/llm/grader는 명시 인자로 주입하고 객체/모델 호출·테스트를 자동 실행하지 않는다. 후속 클래스/함수 재정의로 다른 스키마를 덮는 부분 구현을 없앴다. builder는 뒤 abstain_node 정의 후 호출한다. 직접/간접/반도체 질문 역할을 유지한다.
- Literal/Pydantic은 허용된 판정 schema만 검증하고 실제 관련성/모델 지원을 보장하지 않는다. 잘못된 결과·unknown/검증을 우회한 객체·지원하지 않는 content block은 실패시키며 no/정상 답변으로 자동 바꾸지 않는다. 문서 안 지시와 근거를 구분하도록 prompt를 명시했지만 인젝션 내성 증명은 아니다.
- Human/AI 메시지의 reducer 누적과 실제 prompt에서 과거 이력을 소비하는 것을 구분했다. 기본 노드는 과거 messages를 읽지 않으며 checkpointer만 붙여 멀티턴 이해가 구현됐다고 주장하지 않는다. 파일명/출처는 후보 표시이며 인용 정확성·동명 충돌/문서 id/권한 검증이 아니다.
- 원문 워크숍 숫자998/1,578·476/647·784/912자를 보존하고 원자료/모델·프롬프트/실행 trace 부재로 과거 미확인 기록을 표시했다. 길이/문서 개수만으로 품질 우위나 제거 문서의 실제 비관련성/인과를 증명하지 않는다. 파일명 정확 일치 smoke와 전체 Recall@k/정답 평가를 구분했다. raw 질문/답변의 자동 로그와 외부 Mermaid PNG 요청을 제거하고 로컬 Mermaid 문자열을 반환한다.

HERDR_ENV=1 current pane 재확인은 pane_not_found였다. 작업 전용 Claude 연결/의견은 없고 다른 pane은 제어하지 않았다. 완전 중복 통합/분류·재시도 최적값·운영 채택/근거 판정 선택은 협의 필요로 보류한다. 명확한 실행 오류/의도 손실·과거 기록의 미검증 상태만 공식 계약과 실행 근거에 따라 수정했다.

### 검증 1 — 목록과 고유 내용

추가1개 기존 코드 밖 제목/fence 수, 상태·schema·retrieve/grade/generate/rewrite·조건 분기·조립·제한·시나리오·시각화·평가/워크숍 숫자를 대조했다. 원문38경로 중11개 검토/27개 대기, 신규2개 포함13개다. 대기 문서/비Markdown 원래 bytes 보존을 확인한다. 이동/삭제/코드 파일/첨부 수정·커밋/푸시는 없다.

### 검증 2 — 공식 근거와 실제 실행

확인일2026-10-04. [Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)의 상태/reducer/조건부 route·compile/recursion limit과 [Structured output](https://docs.langchain.com/oss/python/langchain/structured-output)의 schema/오류 계약을 대조했다. [CRAG 원 논문](https://arxiv.org/abs/2401.15884)의 지식 정제·외부 검색 전체를 이 축약 예제가 구현/재현하지 않음을 명시했다. rolling 자료와 확인 판본은 다르며 최신/운영 호환성으로 단정하지 않는다.

Python3.14.2/langgraph1.2.12/core1.6.6/Pydantic2.13.5 임시 환경에서 AST15와 실제 StateGraph를 실행했다. 검색·grade·모델은 fixture이며 Python socket을 막았다. 직접 매칭·한 차례 재검색·원래 질문/검색어 보존, 빈 검색/전부no에서3차 검색·2차 재작성 후 생성 없이 abstain, 한도0, 빈/unknown·schema 우회/잘못된 객체·content block·bool counter 거부를 확인했다. 파일명의 정확 일치/부분 일치 거부와 로컬 Mermaid를 검증했다. 실제 InMemorySaver/reducer에서 같은 thread의 메시지4개/다른 thread2개와 매 질문 카운터 초기화를 확인했지만 기본 prompt가 과거 대화를 읽지 않는 것도 확인했다. 실제 원격 structured output·모델 품질/회사 자료·논문 성능/주장된 워크숍·보안/운영 저장·동시성은 검증하지 않았다.

### 검증 3 — 현재13개

현재13개만 상대 링크·앵커·첨부/unique YAML·검토일/type/tags·pm_notes CLI properties를 검사한다. 다른27개 기술 검토/화면을 완료로 계산하지 않는다. 이전 앱의 path/preview와 실제 screenshot이 어긋나므로 새 본문 읽기 화면도 미완료로 유지한다. 앱 설정/plugin/다른 vault를 수정하지 않았다.


### Agentic 단계 최종 부분 검증 결과

현재13개 상대 링크·앵커·첨부 오류0, unique YAML/검토일/type/tags·CLI properties13개 통과. 원래38경로·미수정27개 SHA-256·ai-dt 비Markdown18개 보존, AST15/실제 graph·오류 입력/reducer·원문 제목/fence·워크숍 숫자 보존을재검증했다. 현재 원문 Python AST46개다. git diff --check 통과. 외부 모델/회사 자료·Claude 의견/새 본문 읽기 화면은 미완료다.

CLI 첫 검사에서 앞서 동작했던 properties가 Command not found로 실패했다. help properties가 공식 명령을 다시 나열하고 vault info=path가 /Users/daeyoung/Codes/pm_notes, version이1.13.7을 반환한 후 재검사13개는 통과했다. 실패를 앱 연결의 일시 불일치로 관찰했으며 원인은 확정하지 않았다. plugin/설정/다른 vault는 변경하지 않았다.


## RAG 확장 개별 추가 검토

### 변경·판단·Claude 협의

원래 경로/제목·12 Python fence·PM/반도체/HyDE·메모리·비교/조합 맥락과 워크숍 숫자를 보존했다. H1 뒤 YAML을 실제 frontmatter로 옮기고 last_updated=2026-07-16을 유지했다. 앞선 Agentic 기본 정의를 먼저 실행하는 조립 순서를 명시하고 해당 state/schema/node를 재사용한다. 일부 클래스/node 재정의와 미정의 globals의 자동 호출은 제거했다. 멀티에이전트 문서는 아직 대기다.

HyDE는 가상 문서를 query로만 쓰고 원래 question을 유지한다. grade/rewrite/generate가 가상 답변을 사용자 의도로 판단하지 않는다. 첫 HyDE 이후 실패는 기본 한도 내 rewrite/abstain이다. 임베딩 밀도 차이가 검색 실패의 확정 원인/HyDE의 개선 보장이라는 설명을 고쳤다. 원문1.5배는 가상 공정 문구이며 실제 기준이 아니다. 논문 encoder/실제 corpus 검색과 임의 provider/embedding의 축약 예제를 구분했다.

checkpoint가 저장되었다고 generator가 과거 이력을 읽는 것은 아니다. 확장 generator는 SystemMessage와 현재 Human까지의 실제 messages를 모델에 전달한다. 이력의 AI 답변도 검증된 사실이 아님을 밝힌다. retrieve/grade/HyDE는 현재 question만 사용하므로 후속 지시 대상의 검색어 해소·기억의 정정/권한/압축·토큰 예산은 구현되지 않았다. 빈 현재 근거는 이력만으로 생성하지 않는다. 실제 의미 이해/검색 품질을 보장하지 않는다. InMemorySaver 객체를 재사용해야 RAM checkpoint가 유지되고 thread id는 인증이 아니다. SQLite는 파일 경로/context manager 수명 안에서 사용하며 운영 적합성/동시성은 별도다. add_messages의 새 id 누적/같은 id 갱신·빈 리스트는 삭제가 아님을 설명했다.

비교 예제는 문자 수/문서 수/status 진단이며 정답/근거·비용/latency 평가가 아니다. 원문 PM/반도체 숫자와656 chars·비용3~5배/지연2~4배는 과거 미확인으로 보존했다. 길이나 문서 수 감소로 정확성/노이즈 제거 인과를 입증하지 않는다. 원문7/8회는 검색 결과3개를 LLM 호출3회로 잘못 센 식이어서 기록으로 설명하고 현재 정적 경로는 grade3+generate1=4/HyDE 추가1=5로 고쳤다. 모델 호출 수는 grade 청크 합+rewrite+성공 generate+선택 HyDE이며 벡터 검색/embedding과 구분한다. 실제 비용/latency/효과를 최신 사실로 주장하지 않는다.

HERDR_ENV=1 current pane은 pane_not_found였다. 작업 전용 Claude 연결/의견은 없고 다른 pane은 제어하지 않았다. 완전 중복 통합/분류·HyDE/운영 저장/이력 해소 정책·채택/임계값은 협의 필요로 보류한다. 명확한 질문 손실/미사용 메시지·context/API/호출 수 오류는 공식 계약·로컬 실행으로 고쳤다.

### 검증 1 — 목록과 고유 내용

추가1개 원문 경로/제목/fence와 질문/문서명·656·PM/반도체의 모든 숫자/워크숍 해석 맥락을 대조했다. 원문38경로 중12개 검토·26개 대기, 신규2개 포함14개다. 대기 문서/비Markdown bytes를 별도 검사한다. 이동/삭제/실행 코드/첨부 변경·커밋/푸시는 없다.

### 검증 2 — 일차 근거와 실제 실행

확인일2026-10-04. [HyDE 원 논문](https://arxiv.org/abs/2212.10496)의 생성 가상 문서→encoder→실제 corpus 검색과 예제/논문 성능 범위를 구분했다. [Persistence](https://docs.langchain.com/oss/python/langgraph/persistence), [Memory](https://docs.langchain.com/oss/python/langgraph/add-memory)로 checkpoint/thread와 store·메시지 활용/보관/저장 수명을 대조했다. rolling 문서와 설치 판본은 별개이며 최신/운영 lock이 아니다.

Python3.14.2/langgraph1.2.12/core1.6.6/checkpoint-sqlite3.1.1의 실제 StateGraph/InMemorySaver/SqliteSaver·reducer/FakeMessagesListChatModel에서 AST12와 실행을 검증했다. HyDE query만 갱신/원래 intent 기준 grade·generate, 재검색/한도2 후 abstain/생성 없음, 같은 thread의4메시지/다른 thread2개·System/Human/AI/현재Human 전달, 실제 SQLite context 재열기/새 workflow에서도 이력 유지·질문/카운터 초기화를 확인했다. 기본 RAGState를 재정의하지 않으며 empty merge/same-id update·Naive 빈 근거 미생성·비교 진단/invalid bool·thread/의도 거부도 확인했다. Python socket을 막고 모델/검색은 fixture를 썼다. 실제 provider·HyDE 논문 성능/워크숍 재현·회사 자료/권한·대화의 검색어 해소·운영 저장/동시성·품질은 미검증이다.

### 검증 3 — 현재14개

현재14개만 상대 링크·앵커·첨부·unique YAML/검토일/type/tags와 pm_notes CLI를 검사한다. 다른26개 내용/화면을 완료로 계산하지 않는다. 앱의 선택 경로와 실제 screenshot이 어긋났으므로 새 확장 본문 읽기 화면은 미완료다. 다른 vault/plugin/settings를 수정하지 않는다.


### 확장 단계 최종 부분 검증 결과·읽기 화면 복구

현재14개 상대 링크·앵커·첨부 오류0, unique YAML/검토일/type/tags·pm_notes CLI properties14개 통과. 원래38경로·미수정26개 SHA-256·ai-dt 비Markdown18개 보존, AST12/실제 extension graph·메시지 prompt/SQLite 재열기/실패 입력을 재검증했고 기본 Agentic AST15/실행도 통과했다. 현재 원문 Python AST58개다. git diff --check 통과.

CUA 문서 재확인 후 Obsidian 앱에 연결되었다. 실제 pm_notes 창에서 advanced-rag/README의 비교/기술 표·관련 링크·역할 절과 상단의 현재 검토 callout/metadata를 AX·screenshot으로 확인했다. 목차의 RAG 확장 링크 클릭→실제 rag-extensions 경로/읽기 모드, 한국어 제목/검토 callout/metadata·HyDE 설명/도메인별 Python syntax 표시를 AX·screenshot으로 확인했다. 확장 callout의 Agentic 링크 클릭→실제 선행 문서 경로/제목·현재 검토 안내/metadata와 graph 흐름을 확인했다. CLI로 파이프라인을 연 뒤 실제 앱 경로/본문과 screenshot도 확인했다. 화면에서 파이프라인 callout의 이전 검토 범위(Agentic·확장 대기)가 남은 것을 발견해 목차/기록을 따르는 안내로 고치고 실제 화면에서 수정된 안내를 재확인했다. 파이프라인의 정리 기록 링크 클릭→RAG 기록의 실제 경로/제목·검토12개/대기26개와 개별 결과 표를 확인했다.

따라서 고급 목차/파이프라인/Agentic/확장·RAG 기록의 관찰한 영역과 위 내부 링크 탐색은 이제 확인되었다. 문서 전체 모든 스크롤 영역/다른 미관찰 문서·모델/회사 자료/Claude 의견을 통과시킨 것은 아니다. 과거 CLI 화면 불일치의 원인은 확정하지 않았다. 앱 설정/plugin/다른 vault는 수정하지 않았다.


## 멀티에이전트 RAG 개별 추가 검토

### 변경·판단·Claude 협의

원래 경로/기존 제목과19 Python fence·PM/반도체 wrapping·Excel/SQL/RAG/SKILL/보고서 맥락을 보존했다. YAML을 실제 frontmatter로 옮기고 last_updated=2026-07-16은 유지한다. 실행 조건 절을 추가했다. SKILL 파일 전체 예시의 yaml fence는 Markdown으로 바로잡았으며 메타데이터와 본문은 유지했다. 임시 환경만 만들고 저장소 코드/첨부/기존 환경은 변경하지 않았다.

- 단일 agent도 여러 도구를 사용할 수 있고 multi-agent가 필수라는 일반화를 제거했다. create_agent의 공유 여부는 wrapper의 입출력에 달렸으며 별도 호출과 보안 격리는 다르다. source 표시는 근거 후보이고 실제 인용/정답·권한 인증은 아니다. PM/공정 분석은 외부 근거 없는 미확인 초안으로 표시해 실제 수치/원인/정책 판단을 주장하지 않는다.
- RAG/분석/공정/보고서와 supervisor는 명시 인자/factory로 조립한다. 기본 question/query/status 계약을 사용하며 미정의 globals/자동 모델 invoke·raw 질문/답변 print를 제거했다. raw graph/trace에 민감내용이 포함될 수 있어 자동 전송/보관하지 않는다. 보고서는 초안 반환이며 파일 저장/발송/등급 인증이 아니다.
- 여기의 SKILL parser는 unique YAML/body·필수 name/description/allowed-tools와 실제 제공 도구 이름의 정확 일치를 검사한다. runtime agent name과 SKILL name의 매핑을 명시했다. 파일만 넣으면 등록/권한 강제·progressive disclosure가 된다는 설명을 바로잡았다. 실제 native skills/SkillsMiddleware는 별도이며 eager body 주입은 점진 공개를 구현하지 않는다. 미등록/추가/누락·중복 도구는 거부한다. 미구현 create_chart를 완료로 주장하지 않는다.
- Deep Agents의 business tools 외 기본 계획/가상 파일 도구·general-purpose subagent가 추가될 수 있음을 밝힌다. tools=[]는 전체 allowlist가 아니다. StateBackend만 사용하고 host filesystem/sandbox를 연결하지 않는다. native permissions·추가 도구 제한/채택은 별도 검증/Claude 협의로 보류한다.
- Excel 집계는 DataFrame 사본·필수 값/유한 비음수/진행률·월 범위/정수 건수를 확인한다. completed_tasks 합계는 완료건수이며 분모 없는 완료율이 아니다. StrictInt로 month의 bool/0을 거부하고 없는 값/NaN을0으로 바꾸지 않는다. 실제 열/단위/기간·중복 행·조직 집계는 미확인이다.
- SQL은 실제 파일 URI mode=ro·query_only와 authorizer를 사용해 승인된 테이블의 SELECT/제한한 집계 함수만 허용한다. 쓰기/임시 생성/ATTACH/PRAGMA/비승인 테이블·함수/재귀/다중 문장을 거부한다. DB 연결은 호출마다 close하고 승인된 schema만 안전하게 인용한다. 반환200행·실습 연산 한도는 운영 최적값이 아니다. JSON 비지원 BLOB/비유한 값은 실패하고 raw 오류/경로를 노출하지 않는다. 행별/사용자별·민감 열 권한/필드 길이·메모리/운영 동시성·timeout은 미구현이다.
- 마지막 stream chunk를 무조건 최종 답변으로 출력하지 않고 실제 values의 마지막 AIMessage·도구 호출 없는 텍스트만 반환한다. unknown message 계약은 거부한다. 라우팅 기대 이름을 runtime task 이름으로 맞추고 task 요청/반환 id를 검사한다. 요청/반환은 도구 성공/정답·순차 실행 증명이 아니다. 키워드 우위/정확도50% 주장은 출처가 없어 미확인으로 설명했다. CLI dev/up/build 역할을 공식 안내와 대조했으며 serve 하나로 운영 배포된다고 주장하지 않는다.

HERDR_ENV=1 current pane은 pane_not_found였다. 작업 전용 Claude 연결/의견은 없고 다른 pane은 제어하지 않았다. 완전 중복 통합/분류·운영 subagent/도구 제한 정책/배포·채택/임계값은 협의 필요로 보류했다. 공식 계약·실제 package/실행으로 확인한 동작 오류만 수정했다.

### 검증 1 — 목록과 고유 내용

추가1개의 기존 경로/제목·fence 수와 PM/공정 질문/agent 역할·SKILL3개/보고서1개·도구/시나리오·확장 맥락을 대조했다. 새 실행 조건 절만 추가하고 SKILL 템플릿 fence 언어를 고쳤다. 원래38경로 중13개 검토/25개 대기이며 신규2개 포함15개다. 대기 원문/비Markdown bytes를 검사한다. 이동/삭제/실행 코드/첨부 변경·커밋/푸시는 없다.

### 검증 2 — 근거와 실제 실행

확인일2026-10-04. 별도 임시 Python3.14.2/deepagents0.7.21/langchain1.4.3/langgraph1.2.12/core1.6.6/pandas3.0.6/PyYAML6.0.3/OpenAI SDK3.24.0을 확인했다. 최신/운영 lock/OS 호환성 보증이 아니다.

| 일차 자료 | 대조 내용 |
|---|---|
| [LangChain subagents](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents) | wrapping 입출력/도구 설명·context 역할 |
| [Deep Agents subagents](https://docs.langchain.com/oss/python/deepagents/subagents) | 명시 등록/model/tools·task와 기본 subagent·context 설정 |
| [Deep Agents skills](https://docs.langchain.com/oss/python/deepagents/skills) | native skills 로딩/점진 공개·실제 권한 설정과 eager prompt 구분 |
| [Python sqlite3](https://docs.python.org/3/library/sqlite3.html) | URI read-only·authorizer/progress·connection close. rolling Python3.14 문서와 실제3.14.2 구분 |
| [LangGraph CLI](https://docs.langchain.com/langsmith/cli) | dev/up/build 명령과 역할. 서버/배포/라이선스 실제 검증은 안 함 |

pandas 공식 aggregate 페이지는 web 도구에서 Internal Error여서 조회 성공 근거로 쓰지 않았다. 실제 설치 pandas 집계/실행으로 확인했다. Deep Agents graph.py의 StateBackend·default tools/GP subagent·등록/permission 계약도 설치 판본에서 대조했다. 잘못 추정한 profiles.py/_profiles.py 경로 검색은 파일 없음으로 실패하여 실제 profiles/harness 경로와 graph import를 확인했다.

AST19 통과. 실제 pandas의 예산 합120/진행률 평균30·월 완료건수15/필터·잘못된 month/빈 필터·NaN/소수 건수 거부·입력 DataFrame 보존을 확인했다. 첫 fixture는 pandas3의 int 열에1.5를 바로 넣어 TypeError로 준비 실패했다. 문서 오류와 구분해 fixture 열을 명시 float로 준비하고 재검증했다. 실제 SQLite에서 공백/#/? 포함 경로의 URI·승인 schema/정수 집계·200행 truncation, 쓰기/ATTACH/비승인/함수/재귀/다중 문장/BLOB 거부 및 DB 전체 바이트 보존을 확인했다. SKILL 중복/누락/extra/unknown 도구 거부·body 등록/원본 list 보존과 보고서 미확인 초안을 확인했다.

실제 create_agent에서 closure RAG 도구→일치 ToolMessage id→최종 응답을 실행했다. 실제 Deep Agents에서 부모 task 호출→별도 자식 모델→월 집계 도구→자식 최종 결과→부모 task 반환/최종 응답을 실행하고 task 요청/반환 id를 확인했다. 실제 values stream/Overwrite·최종 AI와 중간 ToolMessage 실패도 검증했다. 모델은 bind_tools 가능한 FakeMessagesListChatModel이고 토큰 수는 fixture 상수이며 비용/토큰 평가가 아니다. Python socket을 막았다. 실제 provider/의도별 라우팅 정확성/회사 Excel·DB/권한·native skills/permission sandbox·운영/배포·보고서 품질은 미검증이다.

### 검증 3 — 현재15개

현재15개만 상대 링크·앵커·첨부·unique YAML/검토일/type/tags·pm_notes CLI를 검사한다. 다른25개를 완료로 계산하지 않는다. 새 멀티에이전트 본문의 실제 읽기 화면은 추가 확인하며 이전 복구된 앱 상태를 새 본문 성공 증거로 대신하지 않는다. 다른 vault/plugin/settings는 수정하지 않는다.


### 멀티에이전트 단계 최종 부분 검증 결과

현재15개 상대 링크·앵커·첨부 오류0, unique YAML/검토일/type/tags·pm_notes CLI properties15개 통과. 원래38경로·미수정25개 SHA-256·ai-dt 비Markdown18개 보존, 추가 AST19/실제 pandas·SQLite DB바이트/오류 입력·create_agent/Deep Agents task·stream/skill 등록을재검증했다. 현재 원문 Python AST77개다. git diff --check 통과.

CLI open 뒤 CUA가 실제 multi-agent-rag-integration 경로/읽기 모드·한국어 제목/현재 callout·metadata·소스별 역할 표를 반환했고 실제 screenshot도 같은 문서의 제목/검토 안내/날짜·type/tags를 표시했다. 외부 프레임에 과거 기록 텍스트가 남은 AX 부분은 새 본문 성공 근거로 사용하지 않았다. 코드 영역 확인을 위한 scroll은 windowNotFoundAtPosition 오류로 실패해 아래 코드 영역/문서 전체 스크롤 관찰은 미완료다. 다른 vault/app 설정/plugin을 수정하지 않았다. 실제 모델/회사 자료·native skills/권한/운영·Claude 의견은 미확인이다.


## Milvus 시리즈 — 3문서 추가 검토

검증 기준2026-10-04. README와 기초/연동 원문3개를 읽고 원래 경로·절 제목·9+7 Python fence·Docker/PDF/partition/메타데이터·dense/sparse·MMR·Corrective RAG 맥락과 역사적2026-01-31 작성일을 보존했다. 두 설명 문서는 직접 SDK와 LangChain wrapper 역할이 달라 합치지 않았다. 파일 이동/삭제·코드 파일/첨부/비Markdown 수정이 없다.

### 수정과 근거

- 전용 벡터 DB 필수·제품별 프로덕션 등급·Chroma 메모리 전용/Cloud 없음 단정을 제거했다. 제품과 실제 배포 구성을 구분한다. [Chroma client](https://docs.trychroma.com/reference/python/client)·[Cloud](https://docs.trychroma.com/cloud/getting-started), [FAISS I/O](https://github.com/facebookresearch/faiss/wiki/Index-IO%2C-cloning-and-hyper-parameter-tuning)를 확인했다. 다른 제품의 세부 분산/운영 적합성은 미확인이다.
- [Milvus 아키텍처](https://milvus.io/docs/architecture_overview.md) 확인 페이지는v3.0.x다. Streaming/Query/Data의 역할과 WAL/객체/메타데이터 저장을 설명하고 이전 Index Node/Pulsar 고정 구성을 모든 판본에 적용하지 않는다. [Compose 설치](https://milvus.io/docs/install_standalone-docker-compose.md)의v3.0.2 명시 파일을 새 실습 디렉터리에서 사용하는 예시로 갱신했다. Docker 실제 기동/다운로드·서버 호환성·기존 데이터 업그레이드는 미확인이다.
- [Metric](https://milvus.io/docs/metric.md)의 L2 제곱 거리·IP/정규화·COSINE 방향과 점수 방향을 바로잡았다. FLAT 전수 비교는 의미 정답 정확도100%가 아니다. HNSW M16/efConstruction256/ef64는 예시 설정이며 nprobe와 섞지 않는다. [VARCHAR](https://milvus.io/docs/string.md)는 바이트 최대 길이 계약이다. 같은 모델/차원/전처리를 요구하고 영벡터/NaN/bool·UTF-8 바이트 초과를 거부한다.
- 단일 MilvusClient API와 id/text/source/category/embedding schema를 연결했다. 기존 Collection은 거부하고 자동 삭제하지 않는다. 안정된 PK upsert·nested entity 결과·index/load를 명시했다. 실제3차원 fixture는 의미 임베딩이 아니다. [Lite 제한표](https://milvus.io/docs/milvus_lite.md)는 FLAT/partition 제약을 제시한다. 설치된Lite3.2.1에서 확인한 FLAT 동작만 주장하며 서버/다른 판본의 지원을 대신하지 않는다.
- Hybrid 예제는 앞 단일 embedding Collection에 적용하지 않는다. 별도 dense/sparse schema·인코더·각 필드 인덱스를 전제로 request/WeightedRanker descriptor를 구성한다. [Reranking](https://milvus.io/docs/reranking.md)의 메트릭별 변환과 결합을 설명했다.0.7/0.3의 품질 우위·실제 hybrid 서버 검색은 미확인이다.
- [LangChain Milvus 통합](https://docs.langchain.com/oss/python/integrations/vectorstores/milvus), 실제 설치langchain-milvus0.4.0의 소스/실행과 대조했다. `VectorStoreRetriever`, `langchain_text_splitters` import·자체 schema·drop_old=False·명시 URI/embedding을 설명했다. 최초 `add_documents`/이후 ID 지정 `upsert`로 분기해 중복 적재 경로를 제거했다. 개정으로 사라진 청크 삭제/manifest enforcement/동시 writer는 미구현이다.
- LCEL/실제 LangGraph는 빈 근거 abstain·정확yes/no·unknown 실패·선택적 승인 웹 callback 최대1회/재판별로 구성했다. 회사 질문을 Tavily로 자동 전송하는 전역 호출을 제거했다. [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)와 실제StateGraph1.2.12를 대조했다. 논문 전체 CRAG·provider/실제 모델·인용/품질·timeout/권한 검증은 별도다.

### Claude 협의

HERDR_ENV=1 확인 후 허용된 current pane 조회도 pane_not_found였다. Claude pane을 만들거나 다른 pane을 제어하지 않았으며 Claude 의견이 없다. 전면 중복 통합·제품 채택·부하/품질 임계값·하이브리드 구성 선택은 보류한다. 근거/실행으로 확인한 오류와 조건만 수정했다.

### 검증 1 — 목록과 고유 내용

원문3개 제목/fence 비교·현재38개 원래 경로와 미수정22개 SHA-256·ai-dt 비Markdown18개 보존을 검사한다. 이전 난수와 파괴적 삭제 코드는 안전한 결정적 fixture/기존 Collection 거부로 바꾸고 원래 용도를 설명했다. partition/hybrid는 실행 가능한 준비된 서버 전제와 descriptor 역할을 남겼다. 다른 주제 통합/새 교차 링크가 없다.

### 검증 2 — 실제 실행

격리 임시Python3.14.2/pymilvus3.0.2·milvus-lite3.2.1·langchain-milvus0.4.0/LangGraph1.2.12·pypdf6.19.0에서 Python AST16·bash 구문3 통과. 임시 Lite DB의 create/upsert 재실행/PK2건 유지·필터/cosine/nested entity·기존 이름 거부·차원/영벡터/NaN/bool/바이트 초과/중복배치 거부를 확인했다. SDK 연결은127.0.0.1 임시 gRPC만 사용했다. 새 프로세스 재열기에서 load 전 query가 실패해 load 조건을 문서/검증에 반영한 후 저장 데이터2건 보존을 확인했다.

실제 래퍼에서 최초 ID upsert가 Collection을 생성하지 않아 실패했다. 설치 소스와 대조해 최초add/이후upsert를 분리하고 재실행해2건 유지·metadata/source 필터·similarity/MMR/LCEL을 확인했다. 실제PDF loader는 텍스트fixture/blank PDF 실패를 확인했다. 실제 그래프는 fixed callback/model로yes/no/unknown·빈 질문/문서·미승인 폴백·승인1회/다시no→abstain·잘못된 웹 반환을 검사했다. hybrid는 실제SDK request/ranker 생성만 확인했다. 외부 모델/API·Docker/분산·HNSW/hybrid 서버·운영/복구/인증/실제검색품질을 검증하지 않았다. 오류를 수정한 뒤 전체 임시 검증을 다시 통과했다.

### 검증 3 — 참조/메타데이터/읽기 화면

확인된vault=pm_notes 경로 /Users/daeyoung/Codes/pm_notes·앱1.13.7을 사용했다. CLIopen과 실제native 읽기 화면을 대조해 MilvusREADME의제목/검토callout/목차와 읽기 링크→기초 문서의정확경로/본문·메타데이터를 확인했다. 기초scroll 호출은 windowNotFoundAtPosition 오류였으나 후속 실제screenshot은 이동한메타데이터검색 Python tail·Docker bash/설치조건을 표시해 그 범위의코드 렌더를 확인했다. 연동 문서도 실제경로/H1/작성일·검토일/type와callout screenshot을 확인했다. 연동 아래모든코드·전체문서의모든스크롤영역은 아직 화면검증하지 않았다. CLIproperties/uniqueYAML·상대링크/앵커/첨부·범위diff 검사는 아래 최종 결과에 남긴다.


### Milvus 단계 최종 검사 결과

현재 검토18개 상대링크/앵커/첨부 오류0·uniqueYAML/검토일/type/tags·실제CLIproperties18개 통과. 원래38개경로/미수정22개SHA-256·비Markdown18개 보존 및 git diff --check 통과. 원문16개를 개별검토했고 다른22개는대기다. 전체RAG PythonAST는93개(이전77+Milvus16)다. vault경로/읽기화면은 위 관찰범위만 통과이며 다른문서의 미완료를 지우지 않는다.


## OpenSearch 목차·기초·Python 클라이언트 — 3문서 추가 검토

원래11문서 중3개를 추가 검토했다. 검색3·운영/설정/핸들러3·RAG/메모리2, 총8개는 여전히 대기다. 역사적2026-02-05/07/12 작성일과 모든 원래 경로·12 Python fence·학습/운영/벌크/scan/async 예제 맥락을 보존했다. README의 `실무 적용(Production &100GB+)` 분류를 `운영 조건과 응용 예제`로 바꿔 운영 검증으로 오해하지 않게 했으며11문서의 읽기 순서를 보완했다. 기초와클라이언트의역할은다르다. 완전 중복 통합/제품 선택은협의가필요해보류했다.

### 수정 내용과 일차 근거 — 확인2026-10-04

- [OpenSearch FAQ](https://opensearch.org/faq/)·[Foundation](https://opensearch.org/foundation/)로OSS7.10.2 출발/프로젝트와재단·관리형서비스의차이를설명했다. `AWS+커뮤니티`만으로현재거버넌스를단정하지않는다. [Elastic 라이선스 FAQ](https://www.elastic.co/pricing/faq/licensing)의2024 AGPLv3 선택추가를반영하고해당소스/배포ELv2조건을구분했다. 회사사용의법적적합성을판정하지않았다.
- [node.roles 설정](https://docs.opensearch.org/latest/install-and-configure/configuring-opensearch/configuration-system/)·[클러스터 구성](https://docs.opensearch.org/latest/tuning-your-cluster/index/)으로cluster_manager/data/ingest/전용coordinating을설명했다. 기존Master Boolean을현재3.x예제로복사하지않는다. 원래그림은manager/data겸임개념도임을명시했다. [Split](https://docs.opensearch.org/latest/api-reference/index-apis/split/)의새인덱스조건과Primary 수의단순put_settings변경불가를구분하고Replica/단일노드yellow와고가용성을설명했다.10~50GB샤드는검증된보편권장으로두지않았다.
- 주석/ellipsis가있던JSON을유효JSON으로바꾸고3차원구조예시와1536차원mapping을구분했다. [k-NN mapping](https://docs.opensearch.org/latest/mappings/supported-field-types/knn-vector/)의차원/engine조건과[Breaking changes](https://docs.opensearch.org/latest/breaking-changes/)의3.x NMSLIB deprecated/이전API호환조건을확인했다.기초예제의Faiss명시mapping은서버수용/검색품질검증과다르다.한국어analyzer출력은설명용이며_analyze로별도확인한다.
- [Docker 안내](https://docs.opensearch.org/latest/install-and-configure/install-opensearch/docker/)의demo보안2.12이후초기비밀번호조건과무보안실습의demo설치생략을대조했다.고정2.11.0은원래판본으로설명하고실행은명시판본변수/새실습폴더·동일Dashboards판본·127.0.0.1포트로한정했다. 실제서버/Docker/이미지다운로드·운영보안은미확인이다.
- [Python client](https://docs.opensearch.org/latest/clients/python-low-level/)·[Requests 연결 소스](https://github.com/opensearch-project/opensearch-py/blob/main/opensearchpy/connection/http_requests.py)와설치SDK3.2.0에서sync `pool_maxsize`와async `maxsize`를구분했다.기존Requests `maxsize=25`는풀설정이아니었다.TLS/CA/authfactory와무인증loopback실습을분리하고admin/admin상수·자동접속/출력·무분별한재시도를제거했다.기본retry3과예제retry0·timeout뒤반영여부미확정을설명했다.회사인증·SigV4·TLS실제handshake는검증하지않았다.
- [helpers 소스](https://github.com/opensearch-project/opensearch-py/blob/main/opensearchpy/helpers/actions.py)와실제SDK에서bulk부분실패/기본예외·parallel generator소진·실제bytes/단일문서한도·createID충돌·ISO UTCtimestamp·실패status집계를확인했다. raw문서오류출력을없앴다.100GB·100,000건/4threads/2000chunk·5~15MB/1KB당5000건수치를측정권장으로단정하지않는다.
- [페이지네이션](https://docs.opensearch.org/latest/search-plugins/searching-data/paginate/)의from/size·scroll·search_after/PIT조건을구분했다. scan`size`는설치helper설명상shard별배치이며모든문서를한목록에담지않는다. callback오류에서generator.close로scroll해제경로를실행하고실제query/source필터를대표함수로모았다. async성공/실패/취소finallyclose·오류를빈검색결과로바꾸지않는패턴을설명했다.실제scroll/PIT의서버동작·refresh설정복원/동시writer·부하/복구는미검증이다.

### Claude 협의

HERDR_ENV=1 확인후current pane읽기조회도pane_not_found였다.작업전용Claude와연결할수없었고다른pane/vault를제어하지않았다.Claude의견은없다.문서분할/완전중복통합·데이터별engine/샤드/재시도/성능임계값선택은보류하고공식근거·설치SDK와실제실행으로확인한오류만수정했다.

### 검증1 — 목록·고유 내용

3원문스냅샷과절제목/fence·대표예제역할을대조했다.README분류명1개변경을명시했고그밖의절제목/모든12 Python fence를유지했다.새코드실행파일/이동/삭제/첨부변경은없다.README의잘못된../../../Codes링크2개는존재하지않는ai-dt/Codes로향하며주제독립성을어겨일반문자경로/역할설명으로바꿨다.기존코드주제자체는수정하지않았다.현재38원문경로·미수정19개SHA-256·비Markdown18개보존을검사한다.

### 검증2 — 실제SDK/로컬mock REST

Python3.14.2/opensearch-py3.2.0/aiohttp3.14.3·PyYAML6.0.3의격리임시환경에서PythonAST12·JSON2/YAML1/bash3구문통과.실제RequestsHttpConnection의preparedrequest/gzip직렬화에모의REST를공급한28요청으로info/health·index생성/기존거부·create/409·get/update·bulk99성공1실패/재실행100실패·parallel20성공1실패·관리replica요청·scan3건/최신scroll해제·callback예외초기scroll해제·404/500/connection오류분류를검증했다.pooladapter25·CA/SSL옵션·retry0와차원/영벡터/NaN/bool·과대UTF-8문서/NaN·잘못된CA/포트거부를확인했다.

실제AsyncOpenSearch/aiohttp를127.0.0.1 임시mock REST서버에연결해정상/404/실행중취소각경로에서session이닫히는것과실제POST/query직렬화를확인했다.실제OpenSearch서버/검색엔진/인덱스mapping수용·TLS검증/권한·100GB부하·시스템설정은실행하지않았다.모의응답을서버증거로세지않는다.본문가독성보완후전체검증을다시통과했다.

### 검증3 — 메타데이터/참조/읽기 화면

vault=pm_notes의실제경로 /Users/daeyoung/Codes/pm_notes·앱1.13.7을CLI로확인했다.읽기화면에서OpenSearchREADME H1/callout/aliases와목차를확인하고Python클라이언트링크를실제클릭해정확ai-dt/rag/opensearch/python-client.md·역사적작성일/검토일/type/H1/card를대조했다.키보드Next(PageDown)가실제읽기화면을이동해설정표/소스·TLSfactory코드구문강조/줄바꿈screenshot을확인했다.이전좌표scroll오류와달리확인된키보드경로이며앱설정/파일을UI로편집하지않았다.기초문서도정확경로/H1/작성일/검토callout·라이선스표AX와top screenshot을확인했다.전체아래모든코드/각서버운영은아직화면/실행검증하지않았다.전체21개uniqueYAML/CLIproperties·링크/앵커/첨부·최종diff결과는아래에남긴다.


OpenSearch 기본 3문서 단계의 최종 검증: 검토한 RAG 21개 상대 링크·앵커·첨부 오류 0개, 고유 YAML과 검토일/type/tags 검사 통과, 실제 Obsidian CLI properties 21개 통과. 원문 38경로 유지, 당시 대기 원문19개와 ai-dt 비Markdown18개 SHA-256 동일, diff 공백 검사 통과였다.

## OpenSearch BM25 문서 추가 검토 — 2026-10-04

원문 keyword-search-bm25.md 1개를 추가 검토했다. 현재 원문20개 검토/18개 대기, 새 목차/기록을 포함한 검사 대상22개다. OpenSearch는 원문11개 중4개 검토/7개 대기다. 모든 원래 경로·실제 Markdown 절제목·Python fence11개·3개 articles와2개 API 실습 문장을 보존했다. 기존 2026-02-05 작성일을 보존하고 검토일·학습 문서 유형·별칭/partial을 추가했다. 문서 이동/통합·코드 파일 수정은 없다. 벡터 문서와 BM25는 후보 생성 방식이 다르므로 역할을 구분했으며 완전 중복 통합은 협의 대기다.

### 수정·근거

- [공식 Keyword search](https://docs.opensearch.org/latest/search-plugins/keyword-search/)에서 OpenSearch3.0 기본 BM25 구현 변경을 확인했다. 전통/Legacy 식과 3.0+ 식을 구분하고 raw score 임계값·하이브리드 합산을 판본 사이에 그대로 쓰지 않도록 설명했다. [Similarity](https://docs.opensearch.org/latest/im-plugin/similarity/)의 k1=1.2/b=0.75는 기본값이며 실측 최적값이 아니다. 서버별 `_explain`과 한국어 순위는 실행하지 않았다.
- [추가 플러그인](https://docs.opensearch.org/latest/install-and-configure/additional-plugins/index/)·[설치 안내](https://docs.opensearch.org/latest/install-and-configure/plugins/)에 따라 Nori를 기본 내장으로 단정하지 않는다. [내장 분석기](https://docs.opensearch.org/latest/analyzers/supported-analyzers/index/)에서 simple/whitespace/keyword 차이를 대조했다. SKU의 정확 일치는 keyword/term 조건으로 설명한다. 분석도와 한국어 출력은 실측이 아닌 개념 예시로 표시했다.
- [Nori 일차 소스](https://github.com/opensearch-project/OpenSearch/blob/main/plugins/analysis-nori/src/main/java/org/opensearch/index/analysis/NoriTokenizerFactory.java)를 읽고 mixed/사전 옵션을 대조했다. 없는 userdict_ko.txt 의존성을 inline 실습 사전으로 바꿨으며 파일 사전 맥락과 배타 조건을 설명했다. 기존 POS 제거 목록은 검증되지 않은 실습 후보로 보존한다. 실제 서버/사전 수용은 미확인이다.
- [bool](https://docs.opensearch.org/latest/query-dsl/compound/bool/)·[phrase](https://docs.opensearch.org/latest/query-dsl/full-text/match-phrase/)·[prefix](https://docs.opensearch.org/latest/query-dsl/full-text/match-phrase-prefix/)·[Analyze](https://docs.opensearch.org/latest/api-reference/analyze-apis/)로 should 기본값/필터·slop·prefix의 적용 조건을 보완했다. 기본 exact phrase 예제는 slop0을 명시한다. 별도 analyzer 설정은 고유 변수명으로 구분해 앞 절 mapping을 덮어쓰지 않는다.
- [Highlight](https://docs.opensearch.org/latest/search-plugins/searching-data/highlight/)의 html encoder와 bool 강조 한계를 설명했다. raw HTML/Markdown을 무검증 렌더링하는 예제로 취급하지 않는다. 강조 출력은 개념 결과다.

각 자료 확인일은2026-10-04이며 rolling latest/main과 고정 서버 버전을 구분한다. SDK 실행 검증은 opensearch-py3.2.0이다. 예제에서 암묵적 접속/자동 index 삭제를 제거하고 준비된 client/index를 전달한다. 안정 ID의 create 벌크는 성공/실패 수를 반환하며 재실행 충돌이 기존 원문을 보존한다. 검색 시간 초과/부분 shard 실패/완료 상태 누락을 빈 결과로 바꾸지 않는다. caller가 client를 닫고 필요한 mapping을 확인한다.

### Claude 협의

HERDR_ENV=1 조건에서 현재 pane을 다시 조회했지만 pane_not_found였다. Claude 의견은 없고 다른 pane을 제어하지 않았다. 분류/중복 통합·한국어 품사 제거 정책/검색 품질 임계값 결정은 보류했다. 공식 근거와 SDK 실행으로 확인되는 오류만 수정했다.

### 검증1·2·3

1. 원문 스냅샷과 실제 절제목·11 Python fence 및 고유 문장5개를 대조해 보존했다. 첫 검사기의 코드 주석을 절제목으로 오인한 부분을 수정해 실제 Markdown 절만 비교했다. 파일 이동/삭제/첨부 변경은 없다.
2. Python AST11개 통과. 실제 SDK3.2.0의 RequestsHttpConnection과 helpers에 모의 REST를 공급한23요청으로 새 index/기존 거부, bulk3성공/재실행3충돌, 각 전문/구문/복합/강조/분석 요청 직렬화, 안정 ID 문서 보존, 생성자 무통신, timeout/failed shard/완료 상태 누락 거부를 확인했다. 잘못된 query/size/bool/field/중복 ID/UTF-8 512bytes 초과 ID는 요청 전에 거부한다. 검사기에서 기본 urllib3 연결과 Requests mock이 맞지 않던 부분을 RequestsHttpConnection으로 고쳐 재검증했다. Nori tokenize와 BM25 순위·서버 mapping·HTML 화면 안전성은 모의 응답으로 증명하지 않는다.
3. CLI로 정확한 ai-dt/rag/opensearch/keyword-search-bm25.md를 열었다. CUA 첫 연결 cgWindowNotFound 후 확인된 md.obsidian 앱 ID로 재연결했다. 실제 읽기 화면의 경로/H1·별칭·역사적 작성일/검토일·callout을 AX와 screenshot으로 확인하고 Next(PageDown)로 수식·적용 조건·쿼리 표/분석도 아래 영역을 확인했다. 전체 아래쪽 코드 모두를 시각 확인한 것은 아니다. 마지막22개 링크/YAML/CLI·보존/diff 검사는 아래에 기록한다.

BM25 단계 최종 검사: 검토22문서의 상대 링크/앵커/첨부 오류0개, 고유 YAML/검토일/type/tags 검사 통과, 실제 CLI properties22개 통과. 원문38경로와 대기18개·비Markdown18개 SHA-256 보존. diff 공백 검사에서 발견한 함수 선언의 trailing whitespace1개를 수정하고 diff 검사·AST11/실제SDK 모의23요청 전체를 재통과했다. 누적 RAG Python AST 검토116개이며 실제 서버 검증과 구분한다.


## OpenSearch 벡터 검색 문서 추가 검토 — 2026-10-04

원문 vector-search-knn.md 1개를 추가 검토했다. 현재 원문21개 검토/17개 대기, 새 목차/기록 포함23개다. OpenSearch 원문11개 중5개 검토/6개 대기다. 원래 절제목·Python fence8개/JSON2개·샘플 문장7개·2026-02-05 작성일과 모든 경로를 보존했다. 검색 방식별 고유 목적을 설명했으며 파일 이동/통합·실행 코드/첨부 변경은 없다. 미평가 수치/출력은 설명용으로 구분했다.

### 수정·근거와 판단

- [설치 플러그인](https://docs.opensearch.org/latest/install-and-configure/plugins/)의 standard/minimal 제공 차이로 k-NN이 모든 설치에 자동 존재한다고 단정하지 않는다. [Methods/engines](https://docs.opensearch.org/latest/mappings/supported-field-types/knn-methods-engines/)와 breaking changes로 NMSLIB deprecated를 신규 기본 권장에서 제외한다. 제거되었다고 쓰지 않았다. Faiss cosine2.19+·Lucene innerproduct2.13+와 HNSW 파라미터 생성 시점/판본 차이를 표시했다. Faiss 라이브러리 GPU 기능을 OpenSearch GPU 지원으로 확대하지 않는다. 본문 Faiss3.x는 검증 조건을 명시한 학습 예제이며 회사/운영 엔진 채택 결정은 아니다.
- [Spaces3.0](https://docs.opensearch.org/3.0/field-types/supported-field-types/knn-spaces/)·[rolling Spaces](https://docs.opensearch.org/latest/mappings/supported-field-types/knn-spaces/)와 [SpaceType 일차 소스](https://github.com/opensearch-project/k-NN/blob/main/src/main/java/org/opensearch/knn/index/SpaceType.java)를 대조했다. l2는 제곱 거리, cosine 영벡터는 불가, innerproduct는 벡터 크기의 영향과 부호별 score 변환을 가진다. 현재 cosine 표/SpaceType의(1+cos)/2는 원래 식을 단순 오류로 삭제하지 않고 적용 조건을 보완했다. [과거2.3 ANN 자료](https://docs.opensearch.org/docs/2.3/search-plugins/knn/approximate-knn/)의 NMSLIB/Faiss 변환1/(1+d)와 다르므로 판본/실행 경로별 raw score를 보편 임계값으로 쓰지 않는다. 실제 서버의 ANN/script 점수는 미확인이다. 잘못된3.0 raw-source 경로와 2.19 score-script fetch 실패는 확인 증거로 쓰지 않았다.
- [ANN](https://docs.opensearch.org/latest/vector-search/vector-search-techniques/approximate-knn/)·[Exact score script](https://docs.opensearch.org/latest/vector-search/vector-search-techniques/knn-score-script/)로 정확 최근접과 의미적 정답100%를 구분한다. field mapping 자체는 검색 경로를 정하지 않으며 knn_score의 filtered exact와 ANN query를 분리했다. size는 최종 반환 조건, k는 엔진/샤드 후보 조건이다. current ANN 일반 설명에 있는 모든 필터 post-filter 문장은 전용 필터 문서보다 범위가 넓어 그대로 일반화하지 않는다.
- [효율적 필터](https://docs.opensearch.org/latest/vector-search/filter-search-knn/efficient-knn-filtering/)·[필터 비교](https://docs.opensearch.org/latest/vector-search/filter-search-knn/index/)에서 Faiss HNSW2.9+/IVF2.10+·Lucene2.4+ 조건을 확인했다. category mapping/샘플 적재를 보완하고 knn 내부 filter로 예제를 명시했다. 기존 bool 외부 filter는 ANN 뒤라 결과 수가 줄 수 있다는 맥락을 보존한다. top-level score cutoff와 knn radial min_score를 구분한다.0.7/0.75는 평가되지 않아 기본None으로 두고 실제 정책은 보류했다.
- [OpenAI 임베딩](https://developers.openai.com/api/docs/guides/embeddings)의 ada1536/v3 기본1536·3072와 dimensions 조건, [MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)384·[MPNet](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)768·[E5](https://huggingface.co/intfloat/multilingual-e5-large)1024·[BGE zh](https://huggingface.co/BAAI/bge-large-zh-v1.5)1024 모델 카드를 확인했다. 생략 ID는 전체repo와 연결하며 prefix/pooling/revision/정규화 계약을 설명한다. 모델 다운로드/실추론·한국어 정확도는 검증하지 않았다.
- [Force merge](https://docs.opensearch.org/latest/api-reference/index-apis/force-merge/)와 [Warmup](https://docs.opensearch.org/latest/vector-search/api/knn/)으로 쓰기 완료·임시 디스크/백그라운드 지속·캐시 용량/새 segment 조건을 확인했다. 기존 refresh/replica/ef_search/force merge는 미평가 운영 제안으로 보존하고 자동 실행을 제거했다. 영벡터 대신 유효 대표 질의 벡터를 전달하며 검색 워밍업과 별도 native Warmup API를 구분한다.

자료 확인일은2026-10-04이며 rolling main/latest를 고정 배포 증거로 삼지 않는다. 기존 index 자동 삭제/암묵적 연결·고정 ada 모델 호출을 제거했다. 별도 embedding callback과 dimension/모델·전처리 식별 계약을 전달하고 mapping을 명시 검사한다. 제공 embedding을 무시하던 class를 수정했다. 안정 create ID 충돌이 기존 원문을 보존하고 항목 실패 수를 반환한다. 결과의 generated id/score가 원문 키에 덮이지 않도록 명시 필드만 반환하고 vector는 제외한다. 검증 flag는 권한/서버판본/전처리 동일성 증명이 아니며 변경 후 재검사가 필요하다.

### Claude 협의

HERDR_ENV=1 조건에서 현재 pane 조회는 다시 pane_not_found였다. 작업 전용 Claude 의견은 없고 다른 pane/세션은 제어하지 않았다. 문서 통합·한국어 품질/임계값·새 엔진 도입/운영 선택은 보류했다. 공식 조건과 실제 SDK 검증으로 확인되는 오류를 수정하고 판본별 점수/일반 ANN 필터 설명의 차이는 미확인 경계와 함께 기록했다.

### 세 단계 검증

1. 스냅샷과 실제 절제목·8 Python/2 JSON fence·고유 샘플 텍스트7개를 대조했다. 모두 보존했으며 그림/embedding 차원표와 운영 설정 후보 맥락을 남겼다. 새 doc 경로/크로스 링크/파일 이동/삭제는 없다.
2. Python AST8/JSON2 통과. 실제 opensearch-py3.2.0 RequestsHttpConnection/helpers에 모의 REST를 제공한25요청으로 새 index/기존 거부·create bulk5성공/재실행5충돌·category/source 내부 필터·exact script 요청·대표 벡터 워밍업·mapping/동일 차원 다른 모델 계약 거부·제공 embedding 무호출 사용·source id/score 충돌 방지/응답 vector 제외를 확인했다. 시간 초과/failed shard/완료 상태 누락과 잘못된차원/영벡터/NaN/Inf/bool/k·중복ID를 거부한다. OpenAI adapter는 모의 객체에서 요청 없이 생성되고 명시 v3 dimensions가 전달되며 ada dimensions는 거부된다. 이는 실제 서버/공급자 실행·ANN 성능·사전/권한 증거가 아니다. 본문 가독성 보완 후 동일 검사 재실행을 완료한다.
3. 정확 pm_notes vault와 ai-dt/rag/opensearch/vector-search-knn.md를 CLI로 열었다. Obsidian1.13.7 실제 읽기 화면의 H1·별칭·작성일/검토일·callout을 AX/screenshot으로 확인했다. Next(PageDown)로 엔진/점수 표·exact/ANN JSON 구문강조와 Python dimension/vector/영벡터/k/응답 검사 함수의 줄바꿈/구문강조 screenshot을 확인했다. stale 바깥 AX container의 BM25 텍스트는 현재 문서 증거로 삼지 않았다. 전체 모든 아래 코드·실제 서버 검색은 화면/실행 검증하지 않았다.23개 참조·YAML·CLI·보존/diff 검사는 아래에 기록한다.

벡터 검색 단계 최종 검증: 검토23문서의 상대 링크/앵커/첨부 오류0개, 고유 YAML/검토일/type/tags와 실제 CLI properties23개 통과. 원문38경로·대기17개와 ai-dt 비Markdown18개 SHA-256 보존. 가독성 보완 후 AST8/JSON2·실제SDK 모의25요청 전체 재실행과 diff 공백 검사 통과. 누적 RAG Python AST 검토124개이며 실제 서버/공급자/검색 품질 검증과 구분한다.


## OpenSearch 하이브리드 검색 추가 검토 — 2026-10-04

원문1개 추가 검토. RAG원문22개 검토/16개 대기, 목차/기록 포함24개다. OpenSearch11개 중6개 검토/5개 대기다. 원래 모든 절과7 Python fence·RAG/벡터DB/SKU 고유 샘플3개·2026-02-05 작성일을 보존했다. Search Pipeline 절 제목의 잘못된 도입 조건만 수정했다. 공통 embedding/차원·완료 상태 검사와 VectorStore 적재 정의는 같은 OpenSearch 주제의 벡터 문서를 대표로 연결하고 수동 실습 준비 조건을 표시했다. 다른 폴더로 이동/통합하거나 코드/첨부를 변경하지 않았다.

### 수정과 일차 근거

- [Hybrid 개요](https://docs.opensearch.org/latest/vector-search/ai-search/hybrid-search/index/)·[normalization processor](https://docs.opensearch.org/latest/search-plugins/search-pipelines/normalization-processor/)·[native RRF](https://docs.opensearch.org/latest/vector-search/ai-search/hybrid-search/rrf/)로 normalization2.10+, hybrid 질의2.11+, score-ranker2.19+를 구분했다. [hybrid DSL](https://docs.opensearch.org/latest/query-dsl/compound/hybrid/)의 rescore 절 도입일2.18을 전체 hybrid 질의 도입일로 해석하지 않았다. 원래0.7/0.3은 미평가 예시이며 후보 분포/누락/정규화 조건에 따라 평가한다. bool should raw score 합산은 native normalization과 같은 실행 경로가 아니다.
- [RRF 원 논문](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)의 rank1·미등장 기여0·점수 크기를 버리는 합산과60 실험 조건을 대조했다. 원문 A/B 동점과 C/E·D/F 동점을 명시하고 index/id로 결정한다. 순차 두 요청을 병렬이라 쓴 주석을 수정했다. 같은 id의 다른 index를 구분하고 목록 내 중복을 거부하며 요청 사이 원문 변경을 오류로 표시한다. PIT/동시 writer의 전체 일관성은 구현하지 않았다.
- [Pipeline 조회](https://docs.opensearch.org/latest/search-plugins/search-pipelines/retrieving-search-pipeline/)에 따라 새 UUID 이름을 GET 확인 후 명시 PUT한다. 권한/연결 오류를 없는 자원으로 처리하지 않는다. GET→PUT는 원자적 생성 보장이 아니므로 동시 관리자가 없는 독립 실습 조건과 생성 이름의 별도 관리가 필요하다. 고정 운영 pipeline/index 자동 덮어쓰기·삭제를 제거했다.
- [script_score](https://docs.opensearch.org/latest/query-dsl/specialized/script-score/)와 앞 벡터 문서의 knn_score 근거에 맞춰 벡터 score 전체에 boost를 적용한다. raw BM25/벡터 척도 차이와 전수 script 비용은 남는다. 가중치 bool/NaN/범위/합을 거부하며 정규화로 몰래 보정하지 않는다.
- [CrossEncoder 공식 목록](https://www.sbert.net/docs/cross_encoder/pretrained_models.html)·[모델 카드](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)에서 모델 ID의 잘못된 L-6 철자를 L6으로 수정했다. English/MS MARCO 모델이며 한국어 정확도를 확인하지 않았다. callback을 명시 제공하고 다운로드/실모델 호출을 자동 실행하지 않는다. 빈 후보·점수 개수/비유한 값 검사·원문 복사·별도 rerank_score를 보완했다.

확인일2026-10-04. rolling 공식 자료는 특정 운영 서버/플러그인 실행 증거가 아니다. SKU 순위·가중치·후보 수는 설명용이며 keyword SKU mapping/term 구현이나 실측 품질이 아니다. 검색만으로 전체 RAG 정답을 보장하지 않는다. class의 사용되지 않는 vector_weight 인자를 제거하고 unknown mode를 오류로 처리하며 keyword 모드에서 embedding을 호출하지 않는다. 공통 예제의 후보 제한1~100과 기본2*k/2*initial_k 적용 조건을 표시했다.

### Claude 협의

HERDR_ENV=1에서 현재 pane 조회가 pane_not_found였다. 작업 전용 Claude 의견은 얻지 못했고 다른 pane을 제어하지 않았다. 완전 문서 통합·운영 결합 방식·한국어 모델/품질/임계값 선택은 보류했다. 공식 도입 조건·순위 산술과 실제 SDK에서 확인할 수 있는 수정만 진행했다.

### 세 단계 검증

1. 편집 전 snapshot과 fence를 제거한 실제 Markdown 절제목을 대조했다. 도입 조건 제목 변경 외 모든 절·Python7개·고유 원문 샘플3개를 유지했다. 새 이동/삭제/크로스 링크는 없다.
2. Python AST7와 실제 대표 벡터 문서8블록을 함께 준비해 실행했다. 실제 opensearch-py3.2.0 RequestsHttpConnection/helpers에 모의 REST를 제공한38요청으로 index 기존 거부·Nori mapping 요청 형태·pipeline GET404/PUT/기존 거부/403전파·질의 순서/실제 search_pipeline 파라미터·전체 score boost·RRF 후보/identity/동점/중복/원문 변경·create bulk3성공/재실행 충돌을 검증했다. rerank 후보6개/원래 initial20→후보40·빈 후보 무호출/점수 길이·NaN/문자 거부·원문 미변경을 확인했다. 모드·가중치·후보 수·부분 검색 실패/완료 상태 누락을 거부한다. 보완 후 동일38요청 재검사 통과. 실제 서버의 mapping 수락/Nori/normalization/native RRF·분산 후보·실모델/한국어 품질은 미확인이다.
3. CLI로 정확 pm_notes vault의 hybrid-search.md를 열고 Obsidian1.13.7 경로/H1·원작성일/검토일·별칭/callout screenshot을 확인했다. Next4 이동으로 RRF 산술/동점 예제와 재순위 설명을 screenshot, 도입 조건 표를 AX로 확인했다. stale 바깥 AX container의 이전 벡터 텍스트는 증거로 삼지 않았다. 모든 아래 코드 화면은 아직 검증하지 않았다. 전체24개 참조/YAML/CLI·보존/diff 결과는 후속 기록한다.


### 하이브리드 추가 후 전체 재검사

검토 문서24개의 상대 링크/앵커/첨부 오류0·unique YAML·정확 vault CLI properties24가 통과했다. 원래38경로·미검토16개 SHA·ai-dt 비Markdown18개 SHA가 보존되었다. ai-dt git diff --check 통과. 본문 후보 제한 보완 후 hybrid AST7/실제SDK 모의REST38요청 재검사도 통과했다. 실제 서버·모델 품질과 모든 아래 코드 읽기 화면은 미완료다.


## OpenSearch 성능 문서 추가 검토 — 2026-10-04

원문1개 추가: RAG원문23개 검토/15개 대기, 현재25개다. OpenSearch11개 중7개 검토/4개 대기다. 원래21개 절·7개 fence·100GB/샤드/16GB 사례·2026-02-07 작성일을 보존했다. REST Console 요청을 json/bash 대신 http로 표시하고 ...를 유효 match_all로 바꾸었으며 dynamic cluster 설정 예시1개를 추가했다. 경로 이동/문서 삭제·실행 코드/첨부 변경이 없다.

### 근거·중요 판단

[Amazon OpenSearch Service shard guide](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/bp-sharding.html)의 검색10~30GiB/쓰기30~50GiB를 managed 출발점으로 구분했다. 원문100GB→120~150GB·3~5샤드·최소RAM·SSD 필수·5~10MB는 부하 평가 없는 가정이다.5샤드 분배 산술을24~30GB로 수정했으며 비용/fan-out/복구 측정을 설명했다. [시스템 설정](https://docs.opensearch.org/latest/install-and-configure/configuring-opensearch/configuration-system/)의 heap 약50%와 [Vector settings](https://docs.opensearch.org/latest/vector-search/settings/)의 native 한도 기본50%를 구분했다. 남은 RAM은 모두 OS cache가 아니며 native/기타가 함께 쓴다.16GB/heap8GB 예시는 native4GB 한도이고 사전 할당은 아니다. 고정30~32GB 상한은 대상 JDK 압축 포인터 검증 없이 확정하지 않는다. [Methods/engines](https://docs.opensearch.org/latest/mappings/supported-field-types/knn-methods-engines/)의 float native HNSW 근사는 특정 구조 추정이고 전체 RAM이 아니다.

[Index settings](https://docs.opensearch.org/latest/install-and-configure/configuring-opensearch/index-settings/)로 refresh 유휴 동작·원래값/미설정 복원, request/async fsync·응답한 쓰기 손실·Lucene flush 차이를 수정했다. [Indexing-only tuning](https://docs.opensearch.org/latest/tuning-your-cluster/performance/)의 replica0/5~15MiB 후보는 해당 부하 평가 출발점이며 실제100GB 환경의 채택 증거가 아니다. [Force Merge](https://docs.opensearch.org/latest/api-reference/index-apis/force-merge/)로 writer 완료/큰 segment·임시 약2배 디스크/연결 종료 후 지속 조건을 보완했다. 프로세스 재시작과 OS cache 소멸을 동일시하지 않는다. 확인일2026-10-04, rolling 공식 자료와 실제 배포 판본은 구분한다.

### Claude 협의·미확인

HERDR_ENV=1에서 current pane은 pane_not_found였다. Claude 의견은 없고 다른 pane을 제어하지 않았다. 실제 샤드/heap/RAM·native60%·손실 허용 정책/운영 최적화 선택과 완전 문서 통합은 보류했다. 공식 조건·잘못된 메모리 분모·산술/REST 구문만 수정했다. 모든 예시는 서버에 보내지 않았다. 실제 용량·GC·native사용·복구·성능은 미확인이다.

### 세 단계 검증

1. snapshot 대비21개 Markdown 절 일치·원래7fence/고유100GB·16GB 사례·작성일을 보존했다. 현재8fence는 dynamic cluster REST 예시1개 추가다.
2. HTTP 요청행7개·JSON 본문6개·YAML1개와 shard120/150GB 분배·16GB native 한도 산술 통과. 실제 SDK/서버 요청0이며 운영 설정/모델·성능을 실행하지 않았다.
3. 정확 pm_notes vault의 performance-optimization.md를 CLI로 열고 Obsidian1.13.7 경로/H1/작성일·검토일/alias/callout screenshot과 첫 shard 설명 AX를 확인했다. 기존 app binding cgWindowNotFound 후 bundle md.obsidian 재연결로 현재 창을 확인했다. 전체 아래 코드 화면은 미완료다. 전체25개 참조/YAML/CLI·보존/diff는 후속 기록한다.


### 성능 문서 추가 후 전체 재검사

현재 검토25개 상대 링크/앵커/첨부 오류0·unique YAML·정확 vault CLI properties25 통과. 원래38경로·미검토15개 SHA·ai-dt 비Markdown18개 SHA를 보존했다. ai-dt git diff --check 통과. 실제 운영 설정·성능 검증은 실행하지 않았으며 모든 아래 코드 읽기 화면은 미완료다.


## OpenSearch Settings 추가 검토 — 2026-10-04

원문1개 추가. RAG원문24개 검토/14개 대기, 목차/기록 포함26개다. OpenSearch11개 중8개 검토/3개 대기이며 OpenSearch 목차의 오래된3/8 진행 수치를8/3으로 갱신했다. 원래15절·JSON/REST5와Python1 총6fence·notes names/ID/category/price/payload/raw_message 사례·2026-02-12 작성일을 유지했다. 공식 근거 절과 읽기 순서·별칭을 추가했다. 파일 이동/삭제·다른 주제 링크/내용 통합·실행 코드/첨부 변경이 없다.

### 수정과 일차 근거

[Index parameter](https://docs.opensearch.org/latest/mappings/mapping-parameters/index-parameter/)·[enabled](https://docs.opensearch.org/latest/mappings/mapping-parameters/enabled/)·[ignore_above](https://docs.opensearch.org/latest/mappings/mapping-parameters/ignore-above/)로 저장 전용 타입/doc_values·object 파싱 생략/원문 잔존과 keyword256 초과 미색인을 구분했다. [_id](https://docs.opensearch.org/latest/mappings/metadata-fields/id/) metadata를 일반 keyword 필드 강제와 구분했다. 문자열 *_id 동적 template을 명시하고 숫자/object ID를 이름만으로 keyword에 강제하지 않는다.

[Index template API](https://docs.opensearch.org/latest/api-reference/index-apis/create-index-template/)로 신규 기본값/기존 소급 변경 없음·최고priority/같은priority 겹침·수동version과 create=true를 확인했다. notes-read를 template alias에 넣어 새 세대 검색 누락을 바로잡았다. [Rollover](https://docs.opensearch.org/latest/api-reference/index-apis/rollover/)로 조건 OR·index 생성 age/primary 크기·dry_run과 실제 생성 성공의 차이·쓰기 alias 전환을 구분했다. 시간순 신규 쓰기와 과거 문서 수정의 concrete index/id/routing 경로를 분리했다.

[ISM operations](https://docs.opensearch.org/latest/im-plugin/ism/policies-operations/)·[Policies](https://docs.opensearch.org/latest/im-plugin/ism/policies/)·[Examples](https://docs.opensearch.org/latest/im-plugin/ism/policies-examples/)·[API](https://docs.opensearch.org/latest/im-plugin/ism/api/)로 rollover1d/10gb OR·주기 검사·index 전체 삭제·creation age/min_rollover_age/min_state_age를 구분했다. 원문의14d는 문서별 최소 보존/rollover 후14일 보장이 아니며 늦게 적재한 문서도 index와 함께 삭제된다. actions 완료 후 전이라 rollover 실패/지연이면 삭제 전이도 막힐 수 있다. ism_template은 신규 자동 연결이며 정책을 부트스트랩 이후 만들면 첫 index 연결이 보장되지 않는다. Python 순서를 정책→template→index로 수정하고 REST와 같은 전체 body/ism_template/읽기 alias를 공유한다. copy_alias 기본false에 기대지 않고 template에서 읽기 alias를 부착한다. 최신 any_of는 판본 조건이 있어 새 예제에 도입하지 않았다.

확인일2026-10-04. OpenSearch3.x+ISM 제공/권한 가정, 로컬SDK3.2.0 검증이다. 고정 namespace 자원/권한 오류를 없는 것으로 처리하거나 기존 template/policy를 자동 덮어쓰지 않는다. GET→PUT policy 생성은 원자적 create-only가 아니므로 독립 실습/동시 관리자 없음 조건이다. 여러 요청은 transaction이 아니며 중간 실패 시 남은 이름/응답을 개별 확인한다. 자동 삭제/롤백·실제 rollover는 실행하지 않는다. 삭제 수명/운영 선택·완전 문서 통합은 보류했다.

### Claude 협의

HERDR_ENV=1에서 current pane 조회가 pane_not_found였다. 작업 전용 Claude 의견은 얻지 못했고 다른 pane을 제어하지 않았다. 보존 정책/조건·운영 API/UI 선택은 협의 대기다. 공식 조건·원문 누락 alias/정책 순서·실제 SDK로 확인되는 오류만 수정했다.

### 세 단계 검증

1. snapshot 대비15개 절을 보존하고 근거 절1개 추가,6개 fence/고유필드·notes naming/원작성일 유지. 대표 용도는 필드/신규 index 수명 설계이며 검색/성능 문서와 다르다.
2. Python AST1·JSON5 통과. REST와Python complete body는 description 설명 보완 외 일치. 실제 opensearch-py3.2.0 RequestsHttpConnection에 모의 REST17요청으로 policy→template→bootstrap 순서·실제 create=true/dry_run 파라미터·ism_template 패턴/read alias·deepcopy 계획 격리·기존 namespace/template/policy와403이 쓰기를 막는 것을 확인했다. 처음 fixture가 Boolean 직렬화 대소문자를 잘못 가정한 오류를 수정 후 통과했다. 실제 서버 mapping/template 합성·alias 세대/ISM 연결·scheduler/삭제/권한/디스크는 검증하지 않았다.
3. CLI로 정확 pm_notes vault의 ai-dt/rag/opensearch/settings/README.md를 열고 Obsidian1.13.7 경로/H1/2026-02-12 작성일과2026-10-04검토일/alias/callout screenshot, 첫 타입 설명 AX를 확인했다. stale 바깥 container의 이전 성능 텍스트는 증거로 삼지 않았다. 모든 아래 예제 화면은 미완료다. 전체26개 링크/YAML/CLI·보존/diff 검사는 후속 기록한다.


### Settings 추가 후 전체 재검사

검토26개 상대 링크/앵커/첨부 오류0·unique YAML·정확 vault CLI properties26 통과. 원래38경로·미검토14개 SHA·ai-dt 비Markdown18개 SHA 보존. ai-dt git diff --check와 Settings AST1/JSON5/실제SDK 모의REST17요청 재검사 통과. 실제 서버·ISM 동작/보존/삭제와 아래 모든 예제 읽기 화면은 미확인이다.


## OpenSearch 공유 핸들러 추가 검토 — 2026-10-04

원문1개 추가: RAG원문25개 검토/13개 대기, 현재27개. OpenSearch11개 중9개 검토/2개 대기다. 원래15절·API 표 전체·Python7개·Hello/1000개 Doc/3차원 벡터·폴더 구조/기본값·2026-02-12 작성일을 보존했다. 경로 이동/파일 통합·실행 코드/첨부 변경은 없다. 과거 로컬 구현 설명에 미확인 경계를 표시하며 독립 코드 주제를 근거로 읽거나 import하지 않았다.

[SDK compatibility](https://github.com/opensearch-project/opensearch-py/blob/main/COMPATIBILITY.md)로 ES7.x 전체 무조건 호환을 제거하고 OpenSearch 판본별 조건/삭제된 기능을 구분했다. [공식 Python client](https://docs.opensearch.org/latest/clients/python-low-level/)는 인증·명시client/종료 조건 근거이며 로컬 wrapper 기본값 증거는 아니다. admin/verifyFalse 표는 과거 기록으로 남기고 실행 예제는 환경값/명시CA/TLS를 사용한다. 네트워크 CRUD를 순수 함수로 부른 오류를 수정했다. [Python module path](https://docs.python.org/3/tutorial/modules.html#the-module-search-path)로 경로 우선순위/실제 import 위치를 설명하고 필연적 namespace 충돌 주장은 미확인으로 바꿨다. 상위 폴더 추측 대신 caller가 확인한 절대 root/패키지 파일을 검사한다.

[Term](https://docs.opensearch.org/latest/query-dsl/term/term/)·[Bool](https://docs.opensearch.org/latest/query-dsl/compound/bool/)·[Index document](https://docs.opensearch.org/latest/api-reference/document-apis/index-document/)와 같은 주제의 벡터 문서 근거로 keyword 토큰/should/raw score·안정ID와 create-only 차이를 설명했다. 원문 my-index의 embedding/content 누락을3차원 Faiss/OpenSearch3.x toy schema와 적재 fixture로 맞췄다. 동일 벡터1000개는 품질 평가가 아니다. 기존 index 거부·명시 대상/수명·bulk id_field를 표시했지만 실제 wrapper 시그니처/반환/설정 병합·부수효과·충돌/부분 실패 계약은 미확인이다. 잘못된 다른 최상위 코드 상대 링크는 이름 맥락으로만 보존했다. 확인일2026-10-04; 로컬 SDK3.2.0 조건과 wrapper 실제성을 구분한다.

HERDR_ENV=1의 current pane은 pane_not_found였다. Claude 의견/다른 pane 제어는 없으며 로컬 모듈 통합/packaging/현재 운영 기본값 선택은 보류했다. 공식 호환성·표준 Python 경로·원문 내부의 schema 불일치만 수정했다.

1. snapshot 대비15절·원문API 표와 고유 예제를 보존했다. 새 cross link/이동/삭제 없음.
2. Python AST7 통과. 실제 stdlib로 임시 빈 패키지 파일 존재/절대경로·중복 sys.path 삽입 방지·relative/문자/bool/없는경로/패키지누락 거부를 확인하고 sys.path를 복구했다. wrapper는 import/실행하지 않았고 그 빈 파일을 실제API 증거로 삼지 않았다. 서버/SDK 요청0·bulk/검색/매핑 수락·모델 품질 미확인.
3. 정확 pm_notes vault의 opensearch-handler.md를 CLI로 열고 Obsidian1.13.7 경로/H1·원작성일/검토일/별칭·미확인 warning callout screenshot과 패키지 설명 AX를 확인했다. stale 이전 Settings container 텍스트는 증거로 쓰지 않았다. 모든 아래 예제 화면은 미완료. 전체27개 링크/YAML/CLI·보존/diff는 후속 기록한다.


### 핸들러 추가 후 전체 재검사

현재27개 상대링크/앵커/첨부 오류0·unique YAML·정확 vault CLI properties27 통과. 원래38경로/미검토13개SHA/ai-dt 비Markdown18개SHA 보존. ai-dt git diff --check 통과. 로컬wrapper 실제성/서버/모든 아래 예제 화면은 미확인으로 남긴다.


## OpenSearch RAG 연동 추가 검토 — 2026-10-04

원문1개 추가: RAG원문26개 검토/12개 대기, 현재28개. OpenSearch11개 중10개 검토/대화메모리1개 대기다. 원래13절·Python4/bash1·100GB 질의/OpenSearch 튜닝·OpenAI 모델 준비 선택/VectorStore/Retriever/Graph/Ensemble 맥락·2026-02-07 작성일을 보존했다. 코드의 깨진 join 문자열을 수정하고 명시 준비 함수로 조립했다. 고유 학습Document3개/동일corpus metadata id를 제공한다. 파일 이동/문서삭제/다른 주제 링크·실행코드/첨부 변경 없음.

[공식 provider](https://docs.langchain.com/oss/python/integrations/providers/opensearch)·[일차 래퍼 소스](https://raw.githubusercontent.com/langchain-ai/langchain-community/master/libs/community/langchain_community/vectorstores/opensearch_vector_search.py)와 실제community0.4.2를 대조했다. import는 langchain_community.vectorstores이며 원문의 [별도PyPI](https://pypi.org/pypi/langchain-opensearch/json)0.0.2/core<0.4 의존성을 같은 현재API로 취급하지 않는다. [공식sunset](https://github.com/langchain-ai/langchain-community/issues/674)으로 유지보수 종료/archived 상태를 표시했다. 오래된 provider 안내가 있다는 사실만으로 새 도입을 권고하지 않는다. 실제 설치판본을 고정했으며 새 패키지 선택/마이그레이션은 보류했다. 첫 문서 URL404는 확인 근거로 삼지 않았다.

Retriever search_type similarity/mmr/threshold와 래퍼 내부 approximate_search/script_scoring/painless_scripting/hybrid_search를 구분했다. native hybrid는 pipeline/query_text·별도필드 조건이 있고 as_retriever의 hybrid는 거부된다. Faiss3.x·명시 HTTPS/CA/client 수명·기존 index거부와 stable id를 설명한다. GET/생성/적재는 원자 transaction이 아니라 동시 writer 없는 독립실습 조건이다. 래퍼가 timeout/failed shard를 검사하지 않고 hits만 변환하는 경로를 그대로 신뢰하지 않는다. 이 서버 실패 계약의 구현/운영 검증은 미완료로 명시했다. MMR/필터/모델·100GB 성능을 무조건 보장하지 않는다.

[LangGraph API](https://docs.langchain.com/oss/python/langgraph/graph-api)로 StateGraph의 고정 search→generate를 구성한다. 원래 messages를 답변으로 덮던 구조 대신 query/documents/context/answer를 나눠 원질문과 출처를 유지한다. 빈 근거에서는 generation을 호출하지 않는다. f-string 더미를 LLM이라고 부르지 않고 명시callback을 전달한다. 자율 agentic 선택/재검색은 구현하지 않은 확장맥락이다. [Ensemble](https://reference.langchain.com/python/langchain-classic/retrievers/ensemble/EnsembleRetriever)의 classic import와 weightedRRF/c60/id_key를 확인했다.0.3/0.7은 raw score 비율이 아니며 metadata id가 없으면 page_content identity가 된다. 같은원문/권한/corpus revision을 양 branch에 적용해야 한다. 로컬BM25 공백 분석은 Nori와 다르다. 확인일2026-10-04이며 rolling/main과 고정 SDK 실행증거를 구분한다.

HERDR_ENV=1에서 current pane은 pane_not_found였다. Claude 의견/다른pane제어 없음. 유지보수 종료 패키지 대체/100GB 운영 선택·완전 문서 통합은 보류했다. 공식API·실제SDK/그래프로 검증 가능한 오류만 수정했다.

1. snapshot 대비13절/4Python/1bash·질의/고유 역할/작성일 보존. 새 이동/삭제/cross link 없음.
2. Python AST4/bash-n1 통과. 실제langchain-community0.4.2/core1.6.6/classic1.0.8·LangGraph1.2.12/opensearch-py3.2.0/rank-bm250.2.2를 임시환경에서 실행했다. 실제래퍼/RequestsHttpConnection 모의REST12요청으로 factory시I/O없음·fresh index/차원3Faiss mapping/bulk3개 안정ID/metadata·기존index/빈/중복ID거부·efficient_filter의metadata.category 요청·unsupported hybrid거부를 확인했다. 실제StateGraph+fixturegeneration으로 원질문/근거보존·전체context join·빈근거시 generation무호출/빈질의/빈답변거부, 실제BM25/Ensemble과 동일text다른id2개 유지가 통과했다. 첫 fixture가 래퍼의 GET index 조회를 누락한 오류를 추가 후 통과했다. native server/plugin/mapping수락·ANN/필터/부분응답·실모델/100GB성능은 검증하지 않았다. 실제원문/모델을 외부로 보내지 않았고 client/asyncclient를 종료했다.
3. 정확pm_notes vault의 rag-integration.md를 CLI로 열고 Obsidian1.13.7 경로/H1/작성일/검토일/alias·판본/sunset callout screenshot·Retriever 개념AX를 확인했다. stale 이전handler container텍스트는 증거로 쓰지 않았다. 모든 아래코드 읽기 화면은 미완료. 전체28개 링크/YAML/CLI·보존/diff는 후속기록한다.


### RAG 연동 추가 후 전체 재검사

검토28개 상대 링크/앵커/첨부 오류0·unique YAML·정확 vault CLI properties28 통과. 원래38경로·미검토12개 SHA·ai-dt 비Markdown18개 SHA 보존. ai-dt git diff --check와 실제 래퍼/SDK 모의REST12요청·StateGraph·BM25/Ensemble 재검사 통과. 실제 서버/부분응답 계약·모델/품질과 모든 아래 코드 읽기 화면은 미확인이다.


## OpenSearch 대화 메모리 추가 검토 — 2026-10-04

원문1개 추가. RAG 원문27개 검토/11개 대기, 현재29개. OpenSearch 원문11개 전체를 개별 검토했다. 원래17절·JSON3/Python5/text흐름1의9fence·원문 모든 mapping 필드/3계층명·20메시지/3세션/topK·BGE-M3/Qwen3/Kimi2 배치 맥락·user-001/session-002와 기존 사용자 프로필 예시·2026-02-08 작성일을 보존했다. 코드 중 반복 API는 본 문서의 명시 함수로 조립하고 호출은 caller에게 둔다. 다른 주제 코드 링크는 이름 맥락으로만 남겼다. 문서 이동/삭제/폴더간 통합·실행 코드/첨부 변경은 없다.

[엔진/metric](https://docs.opensearch.org/latest/mappings/supported-field-types/knn-methods-engines/)·[3.0 변경](https://docs.opensearch.org/latest/breaking-changes/)·[필터 방식](https://docs.opensearch.org/latest/vector-search/filter-search-knn/index/)·[효율적 필터](https://docs.opensearch.org/latest/vector-search/filter-search-knn/efficient-knn-filtering/)로 NMSLIB는 과거 선택/3.0 deprecated, 새 예제는 Faiss/HNSW/cosine2.19+ 조건을 명시했다. JSON의 ...를 완전 mapping으로 바꾸고 user filter를 knn 내부에 둔다. 원문의 bool 바깥 필터는 ANN 후 결합이며 항상 먼저 user로 좁혀 k개를 보장한다는 설명을 수정했다. 실제 기존 index 전환/운영 엔진 선택은 보류했다.

[bool](https://docs.opensearch.org/latest/query-dsl/compound/bool/)로 filter+should의 default minimum_should_match0을 확인하여1을 명시하고 vector branch에도 동일 user filter를 적용했다. raw bool score 혼합은 native hybrid/RRF가 아니며 품질 향상은 가설이다. [삭제API](https://docs.opensearch.org/latest/api-reference/document-apis/delete-by-query/)는 자동 TTL이 아니라 요청 기반·충돌/부분성공이며 요약 저장 성공만으로 원문 삭제 정책이 완성되지 않는다. 삭제 실행 없음.

[BGE-M3 카드](https://huggingface.co/BAAI/bge-m3)로 dense1024와 sparse/multivector 구분을 확인했다. [HTTPX](https://www.python-httpx.org/quickstart/) 기반 호출에 timeout/raise_for_status·batch index 정렬/개수/유한벡터/영벡터 검사를 추가했다. 모델 자체와 OpenAI-compatible 서버 계약을 구분하고 locality/한국어 품질을 보장하지 않는다. [Qwen3](https://github.com/QwenLM/Qwen3)·[Kimi K2](https://github.com/MoonshotAI/Kimi-K2)로 이름/공식 후보를 구분하지만 실제 checkpoint/서버/라이선스 판정은 미확인이다. 요약은 미완료/빈 출력을 완료된 저장자료로 쓰지 않는다. 무료 비용/충분한 품질 주장을 미확인으로 수정했다. vLLM 일반서빙자료는 특정모델지원 증거로 쓰지 않았다. 접근 불가/redirect 첫 URL은 근거로 세지 않았다.

MemoryManager는 원문의 제안 API로 남기고 실제 import/존재를 주장하지 않는다. 검증한 조립 함수는 원래 질문·role/출처 필드를 자료로 보존하고 사용자/최근session 불일치를 거부한다. 요약/추출fact는 검증된 사실/신뢰system지시가 아니며 provenance·승인/철회·토큰 예산/주입 방어·부분검색 응답 검사는 별도 구현/평가 대기다. 흐름도는 개념 설계이고 실제 자동승격 구현이 아니다. 확인일2026-10-04, 서버3.x/SDK3.2.0·HTTPX0.28.1, 실제 모델판본 미확인.

HERDR_ENV=1 확인 후 sandbox PermissionDenied였고 허용된 current pane 재확인은 pane_not_found였다. 작업 전용 Claude 의견을 얻지 못했고 다른 pane을 제어하지 않았다. 모델/보존·메모리 승인·운영 엔진 교체·완전 문서 통합은 협의 보류다. 공식 근거와 원문 내부 불일치/실제 라이브러리로 확인할 수 있는 오류만 수정했다.

1. snapshot 대비 fence 밖17절·9fence·고유필드/계층/모델/질의/표시 예제를 유지했다. 최초 verifier가 code comment를 heading으로 센 오류를 fence 밖으로 수정했다. 실제 원문 제목 손실은 없었다.
2. 완전 JSON3/Python AST5 통과. 실제 HTTPX0.28.1 MockTransport10요청으로 embedding batch 순서 복원·잘못된 개수/중복/boolindex/영벡터/503거부·완료 요약과 잘림/빈 요약 거부·잘못된 입력 collection 사전 거부를 검증했다. 실제 opensearch-py3.2.0 RequestsHttpConnection 모의REST2검색 body 직렬화로 inline user필터/minimum_should_match1/양branch user 조건을 확인했다. 실제 조립 함수에서 원질문/자료 role/출처/입력 불변·다른 사용자/미확인 소유자/다른 최근session 거부 통과. SDK/HTTP client 종료. 실제 서버 query 수락/ANN/권한/부분응답·모델/요약품질·MemoryManager/저장/삭제 실행은 없음.
3. 정확 pm_notes vault의 conversation-memory-opensearch.md를 CLI로 열었다. CUA 첫 cgWindowNotFound 이후 앱 목록/동일 md.obsidian 재연결로 Obsidian1.13.7 대상경로/H1·작성일/검토일·alias/callout screenshot, Next2의첫 JSON/보존 설명 AX를 확인했다. 모든 아래 코드 화면은 미완료다. 전체29개 링크/YAML/CLI·SHA/diff 재검사는 후속 기록한다.


### 대화 메모리 추가 후 전체 재검사

검토29개 상대 링크/앵커/첨부 오류0·unique YAML·정확 vault CLI properties29 통과. 원래38경로/미검토11개SHA/ai-dt 비Markdown18개SHA 보존. ai-dt git diff --check와 HTTPX모의10요청·SDK모의2검색/조립fixture 재검사 통과. 실제서버/모델/manager/아래모든코드화면은 미확인이다.


## 문서 추출·청킹 목차와 총론 추가 검토 — 2026-10-04

원문2개 추가. RAG 원문29개 검토/9개 대기, 현재31개. token_strategy 원문11개 중2개 검토/9개 대기다. 목차는 원래6절/전체11링크/용어·형식별 맥락을 유지하고 읽기 순서를 제공한다. 총론은 원래14절/6Python/text조합1·6방법/1000·200·95 비교설정·Word/PPTX/XLSX/PDF/스캔 조합·BGE-M3/Kimi-K2.5/Jina 모델 후보와2026-02-12작성일을 보존했다. 코드 반복은 같은 문서의 명시 함수/입력 계약으로 조립했다. 별점/무료/품질우위 표만 미측정 비교조건 표로 바꾸었다. 파일 이동/삭제·형식별 고유내용 제거·폴더간통합/새링크·실행코드/첨부 변경 없음.

[splitter개요](https://docs.langchain.com/oss/python/integrations/splitters/index)·[Character](https://docs.langchain.com/oss/python/integrations/splitters/character_text_splitter)·[Recursive](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter)로 새 import/길이기준·구분자 순서를 확인했다. chunk_size1000/overlap200은 문자 비교설정이며 token한도/모든경계 겹침보장이 아니다. CharacterTextSplitter는 구분자가없는 긴 입력을 그대로 반환할 수 있어 기존separator 예제와 strict 문자창을 함께 제공했다. [Markdown](https://docs.langchain.com/oss/python/integrations/splitters/markdown_header_metadata_splitter)로 Headermetadata/strip_headers=False·큰section후분할/출처 전달을 확인했다. 문서 형식 파서/원문byte보존이 아니라 Markdown 정규화분할이며 Word/HTML 추출과 구분한다.

[Semantic일차소스](https://raw.githubusercontent.com/langchain-ai/langchain-experimental/main/libs/experimental/langchain_experimental/text_splitter.py)와 실제experimental0.4.2로 문장주변buffer1→embedding→cosine distance→percentile95 초과경계를 대조했다.95는 정답률이 아니다. 한국어 정규식/추론품질·길이는 미확인. [공식sunset](https://github.com/langchain-ai/langchain-experimental/issues/87)에서 즉시 유지보수 종료/2026-05-26 archive를 확인하고 기존판본 재현검증과 새운영도입을 구분했다. HuggingFaceEmbeddings 모델 이름은 원문 후보로 유지하며 자동다운로드/오프라인 준비 보장을 제거했다.

[Late논문v3](https://arxiv.org/abs/2409.04701v3)·[저자소개](https://jina.ai/news/late-chunking-in-long-context-embedding-models/)로 full-context token encoding 뒤 청크범위 mean pooling임을 확인했다. 이미 평균한 문서벡터를 쪼개는 방식이 아니다. 원문의 find_chunk_boundaries 문자/token 혼동을 명시token범위/내용mask로 수정했다. 원문 Jina 모델 후보는 이름/흐름으로 보존하고 actual encoder/tokenizer/revision·truncation·metric/정규화/한국어품질은 미확인이다. Agentic은 caller callback에 경계만 제안하게 하며 원문slice/전체coverage를 확인한다. 특정LLM/사내endpoint호환·의미적최고품질은 미확인이다. 실제모델/서버를 호출하지 않았고 모델경계 제안품질도 검증하지 않았다.

확인일2026-10-04. 설치판본langchain-text-splitters1.1.3/experimental0.4.2/core1.6.6/numpy2.5.3·Python3.14.2, rolling원문과 로컬판본을 구분한다. Greg/Pinecone/Unstructured 추가읽기는 기존맥락이며 이번API수정 근거로 쓰지 않았다. 모든방법 품질/지연/비용은 공통평가세트전 미확인이다. 형식별 상세9개·회사DRM계획은 검토대기다.

HERDR_ENV=1의 current pane재확인은 pane_not_found였다. Claude의견/다른pane제어없음. 모델·크기·운영대체패키지/실무최적조합·문서완전통합은 협의보류. 공식API·실제라이브러리/논문으로 확인된 오해만수정했다.

1. snapshot 대비목차6/총론14 원래절·Python6/text1·형식별11링크/모델후보/고유조합·작성일보존. 별점은 출처없는주장으로만제거하고 여섯방법/비교목적은보존했다.
2. AST6/실제splitters1.1.3·SemanticChunker0.4.2와fixture Embeddings·NumPy2.5.3을 실행했다. 구분자없는2101문자에서 separator상한초과와 strict상한/200겹침원문재조립·빈입력·recursive상한·Semanticbuffer/95threshold/빈입력무추론·Header/sourcemetadata·동일원문경계slice/누락·중복·bool경계거부·latepoolmask/token범위/NaN거부 통과. 실제tokenizer/encoder/LLM/network/검색품질 실행없음.
3. 정확pm_notes vault의token_strategy/README를CLI로열고 Obsidian1.13.7 경로/H1/meta/callout screenshot/읽기순서AX를확인했다. 목차의청킹총론link를실제클릭해 같은경로overview로이동/H1/meta/callout screenshot·Next2첫코드/recursive코드AX 확인. stale이전OpenSearch 바깥container는증거로쓰지않았다. 모든아래semantic/agentic/late코드화면은미완료다. 전체31개링크/YAML/CLI·보존/diff는후속기록한다.


### 청킹 목차/총론 추가 후 전체 재검사

검토31개 상대 링크/앵커/첨부 오류0·unique YAML·정확 vault CLI properties31 통과. 원래38경로/미검토9개SHA/ai-dt 비Markdown18개SHA 보존. ai-dt git diff --check와 실제splitter/Semanticfixture/NumPy/원문경계 재검사 통과. 실제모델/tokenizer/검색품질과 모든아래코드화면은미확인이다.


## Word 추출·구조 청킹 추가 검토 — 2026-10-04

원문1개 추가. RAG 원문30개 검토/8개 대기, 현재32개. token_strategy 원문11개 중3개 검토/8개 대기다. 원래15절/7Python/text도식2·다섯방법/1500·200·splitlevel2·기술보고서/SOP/회의/제안서 맥락·펌프RPM3000/1500-2500 예시·2026-02-12작성일을 보존했다. 같은 문서 코드반복을 명시함수/Markdown표공유로 조립했다. 파일이동/삭제·다른주제 통합/새링크·실행코드/첨부 변경없음.

[Document API](https://python-docx.readthedocs.io/en/latest/api/document.html)·[표](https://python-docx.readthedocs.io/en/latest/user/tables.html)·[스타일](https://python-docx.readthedocs.io/en/latest/user/styles-using.html)와 실제python-docx1.2.0을 대조했다. private body XML 순회대신 iter_inner_content로 Paragraph/Table 본문순서를 읽는다. 정확 내장Heading1..9와 현지화표시/사용자스타일을 구분하고 출처block번호를 남긴다. 생략셀grid before/after·중첩표의row/column을 보존하며 병합값 반복은 grid투영 한계다. pipe/backslash/줄바꿈/HTML문자표시를 처리했다. 머리글/각주/이미지/캡션 연결·수정추적/numbering/run서식·완전layout/페이지를 모두 추출한다고 주장하지 않는다.

[Markdown분할](https://docs.langchain.com/oss/python/integrations/splitters/markdown_header_metadata_splitter)·[recursive](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter)와 기존확인판본1.1.3으로 큰section 빈문자fallback/크기검사/overlap clamp·headermetadata/source전달을 수정했다. source/block_numbers는 section 범위이고 정확subchunk페이지/좌표가 아니다. 큰표의 문자분할/짧은section자동병합은 미완료로 구분한다. 원문text는 유지하고 retrieval_text에만prefix를 생성해 반복누적/사용자입력mutation을 피한다.

[Mammoth공식](https://github.com/mwilliamson/python-mammoth)·실제1.11.0/BeautifulSoup4.15.0으로 convert_to_html/messages·sanitize없음을 확인했다. 기존find_all이 parenttable/list와자식paragraph를 중복추출하던 오류를 수정했다. 원HTML/경고·검색용text를 함께 반환하도록 제안함수를수정했으며 HTML은 비교자료이지신뢰렌더입력이 아니다. 중첩list/rowspan/colspan/이미지등은text투영에서완전재현되지않는다.

[Unstructured DOCX](https://docs.unstructured.io/open-source/core-functionality/partitioning#partition-docx)·[chunking](https://docs.unstructured.io/open-source/core-functionality/chunking)으로 Title분류/soft·hard크기/작은section결합·orig_elements를 구분했다. 원문의infer_table_structure=True는DOCX공식예제조건으로설명하지않고제거했다. 함수안에import를두고 실제판본/partition실행은미확인이다. 원문의도구별별점/최우선/무료·품질향상은평가근거가없어조건표/비교가설로수정했다. Pandoc은원문후보/미확인이다.

확인일2026-10-04. 임시환경만python-docx1.2.0/Mammoth1.11.0추가설치;repo실행환경/코드변경없음. HERDR_ENV=1현재pane은pane_not_found였다. Claude의견/다른pane제어없으며 운영도구/스타일계약·완전문서통합은협의보류다. 공식API와실제fixture로확인가능한오류만수정했다.

1. snapshot대비원래15절/9fence·고유예시/작성일을유지했다. 도구별미측정별점만조건표로대체하고다섯대안/원문도식전체맥락보존.
2. AST7통과. 임시생성DOCX와실제라이브러리로 본문前/표/뒤순서·중첩/생략셀/병합값투영·custom Heading Mystery미분류보존·pipe/개행/HTMLescape·300문자section/1500Markdown상한·상위header리셋/source/block번호·max_size10 overlapclamp·metadata deepcopy/원문불변·prefix재호출동일·빈DOCX·실제Mammoth변환/messages/parent-child중복방지통과. 입력DOCXbytes동일. 생략셀fixture만실제라이브러리XML을수정해grid검증했고repo/업무파일은안건드렸다. 실제Word화면/업무문서/Unstructured/임베딩·검색품질미확인.
3. 정확pm_notes vault의docx-tokenization.md를CLI로열고 Obsidian1.13.7 경로/H1/작성일/검토일/alias/callout screenshot·Next2구조도식/Heading/사용방법AX를확인했다. stale총론바깥container는증거로쓰지않았다. 모든아래코드읽기화면은미완료다. 전체32개링크/YAML/CLI·SHA/diff검사는후속기록한다.


### DOCX 추가 후 전체 재검사

검토32개 상대 링크/앵커/첨부 오류0·unique YAML·정확 vault CLI properties32 통과. 원래38경로/미검토8개SHA/ai-dt 비Markdown18개SHA 보존. ai-dt git diff --check와 실제DOCX/Mammoth/분할fixture 재검사 통과. 실제Word/업무문서·Unstructured/검색품질/모든아래코드화면은미확인이다.


## Excel 추출·행 청킹 추가 검토 — 2026-10-04

원문1개 추가. RAG 원문31개 검토/7개 대기, 현재33개. token_strategy11개 중4개 검토/7개 대기. 원래18절·Python8/text예시1·다섯대안/네청킹전략·20행/100행 threshold/30그룹·Pump-A/RPM3000/온도25.5/정상·2026-02-12작성일 보존. 동일문서의셀/표시함수만공유하여반복설명을조립했다. 별점/최상검색효과만측정전가설/조건표로수정했다. 이동/삭제/폴더간통합/새링크·실행코드/첨부변경없음.

[load_workbook](https://openpyxl.readthedocs.io/en/stable/api/openpyxl.reader.excel.html)·[수식캐시](https://openpyxl.readthedocs.io/en/stable/tutorial.html)·[병합셀](https://openpyxl.readthedocs.io/en/stable/editing_worksheets.html)와실제openpyxl3.1.5로data_only가계산아님을확인했다. 원수식/타입/numberformat·원셀좌표와저장캐시/상태를별도로보존한다. 캐시None은없음/빈결과구분미확인,존재해도최신성/재계산미확인이다. 캐시0은present_unverified로보존한다. 빈행제거후잘못된rowidx+2를실제row_number/명시row_numbers로수정했다. row_range는양끝범위이며사이빈행이있을수있다. blank/duplicateheader는columnletter로구분한다. 첫비어있지않은행header/한시트한표는caller확인조건이다.

원문의0/False를빈값으로잃는region변환을is None으로수정했다. pipe/backslash/줄바꿈/HTML표시를처리하고추가열을조용히잘라내지않는다. 전체worksheet물질화·타입→표시문자열/Excel서식재현/hidden정보·첨부/여러표완전탐지를미확인/한계로구분한다. 병합해제/워크북셀값변경대신별도표시배열에anchor를반복하고anchor/range/projection을함께반환한다. 원파일병합/수식은보존하고반복값을원래셀사실/상대참조계산으로보지않는다. region은빈행세로분리만이며원문의빈열감지주장을제한했다.

[pandas read_excel](https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html)·[ExcelFile](https://pandas.pydata.org/docs/reference/api/pandas.ExcelFile.html)와실제3.0.3으로context close/engine·원수식추출기준header행/keep_default_naFalse·3.x문자dtype요약을대조했다. summary는통계만이며LLM호출아니다. literalNA/N/A를임의결측으로취급하지않고요약dtype/캐시/반올림/사이빈행·헤더변환과원자료의차이를표시했다. 원문의대형시트마다전체시트groups를재추가하는오류를현재sheet만조립하는흐름으로수정했다. 100/30/20은모델한도/최적값이아니다. 추출자료와요약/표시청크를구분하며토큰예산/큰셀은별도조건이다.

[Unstructured partition](https://docs.unstructured.io/open-source/core-functionality/partitioning#partition-xlsx)은후보로유지하며import/실행/판본미확인이다. None metadata slicing을예제로실행하지않으며출처/원HTML확인을요구한다. 확인일2026-10-04. 공식rolling표시판본openpyxl3.1.3/pandas3.0.6과설치3.1.5/3.0.3차이를표시했다. 임시환경에만두패키지추가,repo환경수정없음.

HERDR_ENV=1현재pane은pane_not_found였다. Claude의견/다른pane제어없음. header/병합의운영해석·도구/threshold·완전문서통합은협의보류다. 공식근거·실제fixture로확인된오류만수정했다.

1. snapshot대비18절/9fence·고유예시/설정/작성일보존. 반환계약변경(병합rows/ranges/projection,좌표보강표시)을본문에설명했다.
2. AST8/실제openpyxl3.1.5·pandas3.0.3 fixture 통과. leadingblank/header2·중간빈행의원row3/5·0/False·빈/중복헤더·수식캐시None/원수식보존·별도 임시ZIP XML에syntheticcache0주입해present_unverified검증·Markdown특수문자·NA/N/A문자요약·대형2시트101행각4groups/소형1시트2rows중복0·병합배열투영/원bytes·병합range불변·region2..3/5..5·badgroupsize거부통과. 실제Excel계산/캐시freshness·Unstructured/업무문서/검색품질은실행하지않았다. 모든workbook/ExcelFile종료.
3. 정확pm_notes vault의xlsx-tokenization.md를CLI로열고 Obsidian1.13.7 경로/H1/작성일/검토일/alias/callout screenshot·Next2전략표/추출가정AX확인. staleDOCX바깥container는증거로삼지않았다. 모든아래코드화면은미완료다. 전체33개링크/YAML/CLI·SHA/diff검사는후속기록한다.


### XLSX 추가 후 전체 재검사

검토33개 상대 링크/앵커/첨부 오류0·unique YAML·정확 vault CLI properties33 통과. 원래38경로/미검토7개SHA/ai-dt 비Markdown18개SHA 보존. ai-dt git diff --check와 실제XLSX/openpyxl/pandasfixture 재검사 통과. 실제Excel계산/업무문서·Unstructured/검색품질/모든아래코드화면은미확인이다.


## PDF 추출·구조 청킹 추가 검토 — 2026-10-04

원문1개 추가. RAG 원문32개 검토/6개 대기, 현재34개. token_strategy 원문11개 중5개 검토/6개 대기. 원래15절·Python7/textdiagram1·다섯대안/엔지니어링조립·report.pdf·1500/1000/200·2배렌더링/4096·당시Qwen3-VL후보·2026-02-12작성일보존. 50자분류는미검증기존휴리스틱으로설명하고전체파일자동판별에서제거했다. 첫페이지빈텍스트/혼합·OCR레이어의조건을추가했다. 파일이동/삭제/새크로스링크·첨부/실행코드변경없음.

[PyMuPDF Page](https://pymupdf.readthedocs.io/en/latest/page.html)·[OCR](https://pymupdf.readthedocs.io/en/latest/recipes-ocr.html)·[flags](https://pymupdf.readthedocs.io/en/latest/vars.html)의표탐지/별도TesseractOCR/볼드16을확인했다. 열린페이지수명내table.extract값을보존하고document는context로닫는다. sort는읽기순서보장아니며원본imagebytes/표텍스트중복을표시했다. 기존PyMuPDF 표/OCR없음·무조건무료/별점우위를공식기능/조건으로수정했다. [공식라이선스](https://pymupdf.readthedocs.io/en/latest/about.html#license-and-copyright)의AGPL/상용선택을기록하고사내사용적합성판단은미확인이다.

[PyMuPDF4LLM API](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/api.html)와실제1.28.2로page_chunks·OCR옵션·1-based metadata.page_number를대조했다. 로컬fixture첫실행에서잘못가정한page키가KeyError였고공식page_number로수정한후재검증통과했다. langchain_text_splitters1.1.3의Header+Recursive1500/200·페이지출처를유지한다. 이미지만있는페이지는OCRfalse/자리표시일수있으며변환원형/품질보장아니다. pip는임시환경에만PyMuPDF/PyMuPDF4LLM/layout1.28.2와공식의존성설치,repo환경변경없음.

[Unstructured partition](https://docs.unstructured.io/open-source/core-functionality/partitioning#partition-pdf)·[chunking](https://docs.unstructured.io/open-source/core-functionality/chunking)으로fast/ocr_only/hi_res/auto·언어데이터·Title추정·soft/hard/작은섹션병합·TableChunk/원요소메타데이터를설명했다. import는함수호출시로분리하고실제파서/판본/모델은미실행이다. 원문후처리processed_chunks를버리고originalelements를다시청킹하던오류를같은kept요소→chunker→list[dict]으로수정했다. 헤더/푸터는기본보존하고명시drop_headers만적용,table_html None/원요소metadata/페이지unknown을유지한다. 주입fixture는실제UnstructuredAPI의증거아니다.

[이미지입력](https://developers.openai.com/api/docs/guides/images-vision)과실제OpenAI3.24.0+HTTPX모의전송으로PNGdataurl/max_tokens4096/client모델주입을확인했다. page_num0-based/인용page+1·범위/타입·emptychoice/length/빈content를성공으로취급하지않는다. 응답stop은내용정확성증거아니다. Qwen3-VL-30B는기존후보별칭으로보존하며서버배포/지원/품질미확인이다. client는호출자가종료한다.

Azure4.0공식개요·GoogleDocumentAI·AWS Textract개요는기능범위만확인했다. 한국어/우수정확도/단일페이지과금·사내방화벽을현재사실로단정하지않으며실제API/언어/지역/가격·사내연결을미확인표시했다. 확인일2026-10-04. 기존회사환경문장은2026-02-12가정으로보존했다.

HERDR_ENV=1현재pane은pane_not_found였다. Claude의견/다른pane제어없음. 도구우위/임계값·업무OCR/VLM운영·완전문서통합은협의보류,공식근거/fixture로확인된오류만수정했다.

1. snapshot대비15절/8fence·다섯방법/고유설정/작성일보존. mixed/OCR조건과반환list[dict]/tableHTML/원요소계약을본문에명시했다.
2. AST7/실제PyMuPDF/PyMuPDF4LLM1.28.2·splitters1.1.3·OpenAI3.24.0 fixture통과. 생성PDF디지털1+이미지1·Pump-A3000/표·Boldbit·빈image텍스트·페이지1/2·Markdown페이지/1500·원PDFbytes보존, PNG800x1000·SDKmock4·length/None/blank/잘못된인덱스거부,kept필터6→4·tableHTML/원요소deepcopy·unknownpage/HTMLNone·empty/잘못된strategy거부통과. OCR/실제Unstructured/실제서버/업무자료/검색품질은미실행.
3. 정확pm_notes vault의PDF를CLI로열고 Obsidian1.13.7 H1/경로/meta/alias/card screenshot·Next4로유형표/diagram/추출코드/볼드/표/OCR근거·Unstructured코드/Title설정AX와screenshot확인. 아래Markdown/Vision/종합예제전체화면은미완료다. 전체34개링크/YAML/CLI·SHA/diff검사는후속기록한다.


PDF 추가 후 전체34개 상대 링크/앵커/첨부 오류0·unique YAML/검토일/type/tags·정확 pm_notes vault CLI properties34 통과. 원래38경로·미검토6개 SHA·ai-dt 비Markdown18개 SHA 보존, git diff --check -- ai-dt 통과. PDF API 키 수정 후 실제 fixture/AST7 재실행 통과. 기존 앱 설정·첨부·실행 코드 변경은 이번 작업에 없다. 전체 저장소 최종 diff/보호 항목 감사는 아직 대기다.


## PPTX 추출·슬라이드 청킹 추가 검토 — 2026-10-04

원문 1개 추가. RAG 원문 33개 검토/5개 대기, 현재 35개. token_strategy 원문 11개 중 6개 검토/5개 대기다. 원래 15절·Python 6/text diagram 1·다섯 방법·1500/100·50자 후보·2배 이미지/2048·발표자 노트·전체/전후 제목 맥락과 2026-02-12 작성일을 보존했다. 같은 문서의 표 표시 함수/슬라이드 추출을 공유하고, 품질/비용 별점만 미측정 후보/조건표로 수정했다. 문서 이동/삭제·새 크로스 링크·첨부/실행 코드 변경은 없다.

[Shapes API](https://python-pptx.readthedocs.io/en/latest/api/shapes.html)·[표/병합](https://python-pptx.readthedocs.io/en/latest/user/table.html)와 실제 python-pptx 1.0.2로 그룹을 재귀 방문하고 shape 경로/좌표/병합 grid를 보존했다. 문자열이 제목과 같다는 이유로 다른 본문을 버리던 오류를 실제 title shape 제외로 수정했다. z-order는 읽기 순서가 아니며 그룹 bbox EMU는 변환을 펼친 페이지 좌표가 아니다. 한 행 표를 버리지 않고 pipe/줄바꿈을 표시 처리한다. 첫 행 헤더 가정과 병합 표시 한계를 명시했다.

[Slide/Notes API](https://python-pptx.readthedocs.io/en/latest/api/slides.html)·[노트 안내](https://python-pptx.readthedocs.io/en/latest/user/notes.html)로 has_notes_slide 검사와 notes_text_frame None을 확인했다. notes_status absent/present/missing_placeholder를 구분한다. 노트 본문 placeholder만 읽으며 마스터/다른 노트 도형·SmartArt/애니메이션/미디어 등은 미추출이다. [차트 API](https://python-pptx.readthedocs.io/en/latest/api/chart.html)는 존재하므로 표의 python-pptx 차트 불가 단정을 수정했다. 예제에서는 차트·그림을 content_not_extracted로 기록하고 내용/값은 추출하지 않는다. 공식 rolling 문서 표시 버전 1.0.0과 실제 설치 1.0.2를 구분한다. 임시 환경에만 공식 PyPI 패키지와 의존성을 설치했다.

빈/이미지 전용 슬라이드도 번호 기록을 유지한다. 한 슬라이드는 조립 단위이며 문자/토큰 한도를 강제하지 않으므로 이후 분할/실제 모델 길이 검사가 필요하다. 원문 hybrid는 빈 슬라이드를 제외한 목록 enumerate 위치로 이미지를 연결하고 본문을 덮어썼다. 선택한 원 slide_num과 호출자가 대조한 images_by_slide만 사용하고 원 text/노트와 vision_text를 구분했다. 50자 기준은 미검증 원문 후보로 남기고 자동 선택에서 제거했다. 메타데이터 이웃 제목은 필터링된 청크 배열 대신 원 PPTX에서 찾고 없는 이웃 None/제목 없는 실제 이웃 빈 문자열을 구분한다. deepcopy로 입력을 보존한다. 첫 슬라이드 전체 제목 가정/검색 개선은 미확인이다.

[LibreOffice 매개변수](https://help.libreoffice.org/latest/en-US/text/shared/guide/start_parameters.html)의 convert-to/outdir/UserInstallation을 확인했다. 기존 고정 /tmp/추측한 확장자·오래된 출력 재사용 대신 새 임시 출력/프로필·Path.stem·timeout60·PDF 존재/크기 검사를 사용한다. 렌더링 PDF 페이지를 원 슬라이드 번호로 추정하지 않고 source_slide_num None/mapping_status unverified로 반환한다. 숨김 슬라이드·순서·폰트·노트/차트 품질은 원본과 대조해야 한다. 현재 PATH에 libreoffice가 없어 실제 변환/판본 미확인이다. 명령 대역은 실제 변환 증거가 아니다.

[Unstructured PPTX](https://docs.unstructured.io/open-source/core-functionality/partitioning#partition-pptx)·[chunking](https://docs.unstructured.io/open-source/core-functionality/chunking)의 Title 추정/작은 섹션 병합/페이지 경계를 설명했다. include_page_breaks만으로 한 슬라이드 한 청크를 보장하지 않으며 multipage_sections=False와 orig_elements를 확인한다. 파서 import/실행·노트 포함·표 HTML·판본은 미확인이다. [이미지 입력](https://developers.openai.com/api/docs/guides/images-vision)과 실제 SDK3.24.0 모의 HTTP로 PNG data URL/2048·확인된 client/모델 주입·절단/빈 응답 거부를 검증했다. 기존 Qwen3-VL-30B는 당시 후보별칭으로 보존하며 실제 배포/지원/품질은 미확인이다. 확인일 2026-10-04.

HERDR_ENV=1 현재 pane은 pane_not_found였다. Claude 의견을 받지 못했고 다른 pane을 제어하지 않았다. 완전 문서 통합·도구 우위/50자 기준·업무 VLM 운영 판단은 협의 보류다. 공식 API와 실제 fixture로 확인된 오류만 수정했다.

1. snapshot 대비 원래 15절/7fence·다섯 대안/고유 설정/작성일을 보존했다. 빈 슬라이드 포함 반환·원문/Vision 별도 필드·변환 페이지의 대응 미확인 계약을 본문에 설명했다.
2. AST6/실제 python-pptx1.0.2의 생성 5슬라이드 fixture 통과. 중첩 그룹 depth3의 표·동일 제목 본문·병합 origin/spanned·노트 본문/없는 placeholder·그림 전용 빈 text·차트 미추출·한 행 표/pipe/줄바꿈·원파일 bytes 보존, 선택 slide1/2 정확 대응·원문/노트 보존·잘못된 번호/이미지 대응 거부, 필터링된 1/3/5 이웃·빈 프레젠테이션 통과. fixture 작성 중 GroupShapes.add_table이 없는 것을 확인하여 공개 add_group_shape([기존 table shape])로 fixture를 구성했다. 문서 추출 API에는 이 잘못된 호출이 없다. 실제 SDK mock4·length/None/blank/번호 거부 통과. subprocess 대역으로 명령 인자/별도 임시 출력/프로필·실제 생성 PDF 2페이지 PNG/임시 폴더 정리·미생성 PDF·timeout/exit 실패를 검사했다. 실제 LibreOffice/Unstructured/VLM/업무 문서 품질은 실행하지 않았다.
3. 정확 pm_notes vault에서 CLI로 PPTX를 열었다. Obsidian1.13.7의 경로/H1/meta/alias/card screenshot·Next3 청킹 단위 표/첫 재귀 그룹 코드 screenshot을 확인했다. 바깥 container의 이전 PDF 텍스트는 증거로 삼지 않았다. 아래 표/노트/변환/하이브리드 전체 화면 검증은 미완료다. 전체35개 링크/YAML/CLI·SHA/diff 결과는 후속 기록한다.


PPTX 추가 후 현재35개 상대 링크/앵커/첨부 오류0·unique YAML/검토일/type/tags·정확 pm_notes vault CLI properties35 통과. 원래38경로·미검토5개 SHA·ai-dt 비Markdown18개 SHA 보존, git diff --check -- ai-dt 통과. 전체 저장소 최종 범위/보호 항목 감사와 아래 모든 읽기 화면 검증은 아직 대기다.


## 2026 전략 메모 추가 검토 — 2026-10-04

원문 1개 추가. RAG 원문34개 검토/4개 대기, 현재36개. token_strategy 원문11개 중7개 검토/4개 대기다. 원래32절과 text 예시2개를 포함한 H1 이후 본문 전체를 정확히 보존했다. 2026-03-14 사내 구축 제안 기록으로 분류하고 앞에 현재 검토/정정·모델 범위·적용 조건을 추가했다. 당시 회사 BGE-M3 제공/OpenSearch/외부 전송 제약, 모델 선택/시나리오/우선순위·고유 입력 예시·수치 계획을 현재 운영 사실로 고쳐 쓰지 않았다. 문서 이동/삭제·새 크로스 링크·첨부/실행 코드 변경은 없다.

[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3), [Qwen0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), [Qwen4B 공식 raw 카드](https://huggingface.co/Qwen/Qwen3-Embedding-4B/raw/main/README.md), [E5 instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct), [Jina v5 small](https://jina.ai/models/jina-embeddings-v5-text-small/)와 [Qwen reranker](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)를 확인했다. Qwen4B 웹 카드는 도구 Internal Error였으나 같은 공식 저장소 raw README로 확인했다. 입력 길이/차원/MRL·라이선스 표시는 카드의 명시값이며 모델 revision·실제 사내 server/API 모드·최대 길이·한국어 품질/GPU/속도는 미확인이다. BGE의 query instruction 불필요, Qwen/E5의 query instruction/document 구분을 설명했다. Qwen instruction1~5%/영어 권고는 제공자의 평가 범위이며 사내 개선 보장이 아니다. Jina 공개일2026-02-18/32K/CC-BY-NC표시는 확인했으나 가중치/서비스 계약·특정 사용 허용/금지를 단정하지 않았다.

[Anthropic Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval)의2024-09-19/50~100·문맥을 embeddings와BM25입력에붙이는동작·평가1−Recall@20, [Jina Late Chunking](https://jina.ai/news/late-chunking-in-long-context-embedding-models/)의2024-08-22/token-level후chunkpooling을확인했다. 최종문서벡터분할이아니며 일반API단일벡터만으로구현할수없다. Qwen공식기본last-tokenpooling을Jina meanpooling과구분하고32K만으로late지원/품질을확인하지않았다. OCR/표/PPTX비추천은당시우선순위판단으로보존한다.

[OpenSearch text_chunking](https://docs.opensearch.org/latest/ingest-pipelines/processors/text-chunking/)·[text_embedding](https://docs.opensearch.org/latest/ingest-pipelines/processors/text-embedding/)을대조했다. standard는단어토크나이저이며모델subword한도와같지않다. overlap유효0~0.5/권고0~0.2·max_chunk_limit초과분의마지막청크병합을명시했다. 원문250~350/300~500/300~600/200~400·10~15%는미검증시작값이며prefix/instruction/specialtoken을포함한실제서버입력검사가필요하다. 초과입력은truncation/거부등실제구현조건으로읽는다.

[normalization](https://docs.opensearch.org/latest/search-plugins/search-pipelines/normalization-processor/)도입2.10, [hybrid2.11문서](https://docs.opensearch.org/2.11/query-dsl/compound/hybrid/)의기능존재/유지보수종료, [score-ranker](https://docs.opensearch.org/latest/search-plugins/search-pipelines/score-ranker-processor/)도입2.19/RRF, [rerank](https://docs.opensearch.org/latest/search-plugins/search-pipelines/rerank-processor/)ml2.12/by_field2.18을확인했다. 기능존재가모든자료의기본값/우위는아니며fusion과모델재정렬을구분한다. Qwenreranker가프로세서에바로호환됨을확인하지않았다. 원문lexical/semantic/exact필드·metadata설계와100~200질의/평가기준은당시제안으로보존하고조건/근거를덧붙였다. 확인일2026-10-04. 모든모델/서버/업무성능·가격/운영적합성은미실행/미확인이다.

HERDR_ENV=1 현재 pane은 pane_not_found였다. Claude 의견/다른 pane 제어 없음. 모델 선택·우위/운영 정책·완전 통합 결정은 협의 보류하며 공식 명시값과 원문의 과잉 일반화만 구분했다.

1. snapshot 대비 H1 이후 역사 본문 전체 정확 보존·원래32절/text예시2·작성일/고유 계획 유지 검사 통과.
2. 공식 모델 카드/제품 기능/연구 방법·평가 범위/날짜를 직접 대조했다. 실행 코드가 없으므로 모델/서버/토크나이저를 실행한 것으로 보고하지 않는다. text 예시는 형식/원문 보존 검사만 통과했다.
3. 정확pm_notes vault에서 전략 메모를CLI로열고 Obsidian1.13.7 경로/H1/meta/작성일/검토일/역사분류/card screenshot, Next2 모델조건표/토큰길이·Late/RRF표 AX와토큰조건screenshot 확인. 바깥이전PPTXcontainer는증거에서제외했다. 원문전체의아래화면은미완료다. 전체36개링크/YAML/CLI·SHA/diff결과는후속기록한다.


전략 메모 추가 후 현재36개 상대 링크/앵커/첨부 오류0·unique YAML/검토일/type/tags·정확 pm_notes vault CLI properties36 통과. 원래38경로·미검토4개 SHA·ai-dt 비Markdown18개 SHA 보존, git diff --check -- ai-dt 통과. 전체 저장소 최종 감사·원문 전체 아래 읽기 화면·사내 평가/Claude는 미완료다.


## DRM 목차 추가 검토 — 2026-10-04

원문1개 추가. RAG 원문35개 검토/3개 대기, 현재37개. token_strategy11개 중8개 검토/3개 대기다. 원래 Why 이후 본문 전체와 도식2개·99%/Phase1/2/향후 해제 가정을 정확 보존하고 앞에 검토 범위/현재 적용 조건·읽기 순서를 추가했다. 99%와 “스크린샷+VLM이 유일”은 당시 가정/미확인으로 구분했다. 파일 이동/삭제·새 크로스 링크·첨부/실행 코드 변경은 없다.

[Microsoft PowerPoint IRM](https://support.microsoft.com/en-us/powerpoint/restrict-access-to-presentations-with-information-rights-management-in-powerpoint)·[사용 권한](https://learn.microsoft.com/en-us/purview/rights-management-usage-rights)의 열람·복사/추출·인쇄/내보내기 구분을 확인했다. Microsoft365/지원 Office 제품의 사례이며 모든 DRM/사내 운영·캡처 승인에 일반화하지 않는다. 파일 확장자/파서 실패만으로 DRM 판정·경로 자동 승인을 하지 않으며 unknown을 보존한다. 승인된 파일/화면 입력 경로를 구분하고 실제 제품/버전·정책·사내 비율·해제/캡처/검색 재사용은 미확인으로 남겼다. 화면 입력은 보이는 내용이며 원 수식·숨김 셀·노트/차트 원데이터를 모두 추출했다는 뜻이 아니다. 확인일2026-10-04.

화면 취득/구조화·VLM 결과 청킹·향후 승인된 입력 전환은 역할이 달라 기존 세 문서와 고유 맥락을 보존했다. 공통 청킹의 대표 문서는 총론으로 안내하며 하위 세 개의 개별 기술 검토는 대기다. Herdr 현재 pane_not_found는 이번 연속 작업에서 HERDR_ENV=1로 확인했으며 Claude 의견/다른 pane 제어가 없었다. 완전 통합과 업무 경로 선택은 협의 보류다.

1. snapshot 대비 Why 이후 본문/도식2개·기존 고유 가정·작성일 정확 보존 검사 통과.
2. 공식 권한 사례/적용 제품 범위를 직접 대조했다. 실제 DRM·정책·Office/뷰어·캡처/VLM/파서를 실행하지 않았다. 실행 예제가 없으므로 원문/도식 보존 검증과 실제 운영 검증을 구분한다.
3. 정확pm_notes vault에서 CLI로DRM README를열고 Obsidian1.13.7 경로/H1/meta/작성일/검토일/card screenshot·Next2읽기순서/당시계획/도식·목차AX와역사구분screenshot 확인했다. 바깥이전전략container는증거에서제외했다. 전체아래도식/관련링크의읽기화면은미완료다. 전체37개 링크/YAML/CLI·SHA/diff 결과는 후속 기록한다.


DRM 목차 추가 후 현재37개 상대 링크/앵커/첨부 오류0·unique YAML/검토일/type/tags·정확 pm_notes vault CLI properties37 통과. 원래38경로·미검토3개 SHA·ai-dt 비Markdown18개 SHA 보존, git diff --check -- ai-dt 통과. 전체 저장소 최종 감사·전체 아래 읽기 화면·사내 DRM/모델 실행·Claude는 미완료다.


## 화면 이미지 추출 상세 추가 검토 — 2026-10-04

원문1개 추가: RAG36/38원문, 현재38개, 대기2개; token_strategy9/11개. 원래 제목/절·그림2개·문서별 프롬프트·8B/30B/Kimi 역할·100페이지 시간 추정·캡처/전처리/API/배치/검사/재시도11개 Python 예제의 목적을 대조했다. 실제 오류가 있는 코드는 교체했고 원래 속도/품질/방화벽/사내 제공/과금은2026-02-12 제안의 미검증 가정으로 보존했다. 이동/삭제/업무 자료/첨부/실행 코드 변경은 없다.

[Qwen8B](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)·[30B-A3B](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct)·[Kimi2.5](https://huggingface.co/moonshotai/Kimi-K2.5) 공식 카드로 모델명/이미지 입력과 Kimi 네이티브 멀티모달을 확인했다. Qwen2.5/Kimi2 저장소를 최신 모델 근거로 연결한 오류를 고쳤다. [vLLM online serving](https://docs.vllm.ai/en/latest/serving/online_serving/)은 기존 URL의 실제 리디렉션 목적지로 확인했다. 실제 사내 alias/서버 버전/모델 지원/인식 품질/속도는 미확인이다. temperature0을 결정적 출력으로 단정하지 않는다. prompt80% 수치의 근거가 없어 철회했다.

[Pillow12.3 Image](https://pillow.readthedocs.io/en/stable/reference/Image.html)·[Enhance](https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html)·[Grab](https://pillow.readthedocs.io/en/stable/reference/ImageGrab.html)·[PyAutoGUI region](https://pyautogui.readthedocs.io/en/latest/screenshot.html)·[SDK](https://github.com/openai/openai-python)·[GDI DeleteObject](https://learn.microsoft.com/en-us/windows/win32/api/wingdi/nf-wingdi-deleteobject)을 직접 대조했다. Windowswindow 인자11.2.1 도입을 확인했다. BMP를png로 저장하던 SaveBitmapFile 예제와 bitmap/DC정리 누락을 설명하고 Pillow 명시PNG 예제로 교체했다. 실제Win32/DRM캡처를 검증한 것으로 보고하지 않는다. 기존 전체 화면/고정sleep/마지막뒤pagedown을 명시 준비·취득 콜백으로 대체했다. 기존 scroll_amount 미사용/AppleScript미구현을 구분한다. 신규출력폴더·xb로 기존 자료를 보호한다. 원본 이미지·실제페이지/unknown·crop을 보존하고 lexicographicfilename정렬/인위페이지번호를 제거했다. 원래2048/1.3/1.2는 보정 후보이며 기본보정1.0·원본보존·범위검사·높이최소1·이미지닫기를 적용했다.

표 병합값 무조건 복제·수식/차트 숨은 값 추측·반복 헤더 삭제 대신 보이는 범위/미확명시·원문과생성키워드분리로 수정했다. 입력PNG/JPG/BMP를 실제PNG로재인코딩하고 unknown문서유형/잘린응답/없거나빈텍스트를 전체성공으로 반환하지 않는다. usage없음은None,knownsum과전체unknown분리. 입력manifest순서·동시성양수·실패시남은task취소/회수. 휴리스틱20/3/3은 검토신호로 유지하고 임의100−문제×20/60점 및서로다른표rawpipe오탐을 제거했다. 무신호도unverified. 재시도정책True/False/None과두시도의검사/출처를보존하며 큰모델우위/자동품질보증을단정하지않는다.

HERDR_ENV=1 현재 pane 재확인은 pane_not_found; Claude 의견/다른 pane 제어 없음. 모델 선택·재시도 정책·품질 판단·완전 통합 결정은 협의 보류다.

1. 원래 Markdown절/13fence(11Python+2그림)와고유단계/입력유형/역사추정치 대조 통과. 첫 검사에는 Python주석을 Markdown제목으로 세는 테스트 오류가 있었고 fence제외로 수정한 뒤 통과했다.
2. Python3.14.2/Pillow12.3/OpenAI3.24/HTTPX0.28.1 임시환경에서11예제AST+실행, 생성3000×1JPEG의2048×1변환·잘못된crop·원본SHA·생성capture2/10/10000순·기존폴더보존·mockwindow실제PNG/xb·실제SDKMockTransportPNG/응답length/null/emptychoices·unknownpage/usage·동시최대2·invalidconcurrency·unknown유형·retryNone/True양쪽검사·서로다른표/escapedpipe무오탐 통과. 실제화면/Windows/보호뷰어/모델/GPU추론은 미실행.
3. exactpm_notes CLIopen과 Obsidian1.13.7 경로/H1/6메타/검토card screenshot, Next2 모델분류/환경조건/역사전략 AX 확인. 아래 코드전체읽기화면은 미완료. 전체38개 링크/YAML/CLI/대기2SHA/nonMD18/diff 검사는 후속 기록한다.


화면 추출 상세 추가 후 현재38개 상대 링크/앵커/첨부 오류0·unique YAML/검토일/type/tags·정확pm_notes CLI properties38 통과. 원래38경로·미검토2개 SHA·ai-dt 비Markdown18개 SHA 보존, git diff --check -- ai-dt 통과. 전체 저장소 감사·전체 아래 읽기 화면·사내 DRM/모델 실행·Claude 협의는 미완료다.


## VLM 추출 결과 청킹 상세 추가 검토 — 2026-10-04

원문1개 추가: RAG37/38원문·현재39개·대기1개, token_strategy10/11개. 원래 절·전략3개 도식·page/section/element/type별후보·boundary/context/LLM보강7예제 목적을 대조했다. 원래일2026-02-12를 보존한다. 일반파서100%/VLM90~95%·95~99%는 출처/평가가 없어 미확인으로 수정하고 원문유형만으로최적을 단정하지 않는다. 파서 출력과 원래 읽기/페이지 구조를 구분한다. 파일 이동/삭제/첨부/실행 코드 수정은 없다.

[markdown-it-py 공식 문서](https://markdown-it-py.readthedocs.io/en/latest/using.html) CommonMark/table/token.map, [LangChain Markdown 대안](https://docs.langchain.com/oss/python/integrations/splitters/markdown_header_metadata_splitter) 제목메타/strip_headers, [SDK](https://github.com/openai/openai-python), [Kimi2.5](https://huggingface.co/moonshotai/Kimi-K2.5) 멀티모달을 대조했다. 확인일2026-10-04. markdown-it-py4.2.0/mdurl0.1.2는 임시환경 공식PyPI설치, 저장소환경/Obsidian플러그인변경없음. 2000/1500은문자hardlimit이며토큰/표/fence구조보장아님을 설명한다. 실제서버/모델/검색평가 미실행.

결과순서로만들던page=i+1/total_pages/qualitydefault100 대신 실제manifestpage/image/empty/None/legacyunverified를보존했다. 정규식heading/잘못된하위제목상속 대신 실제Markdownheading과문자slice·청크별기여source_spans를 기록한다. 길이초과문단/구분자무시를고치고공백포함원문을정확보존한다. 표/코드fence중간에서문자hardlimit이자르는제약을명시한다. 원래임의pipe행삭제/inlinefigure텍스트손실 대신 parser top-level표와독립Figure문단범위만분리하고나머지그대로보존,표제목앞줄추측None. nested/비표준/inline은일반text에남긴다. page+table이중적재기본예제를한번씩보존으로정정한다.

마침표없는줄+다음첫줄자동이동은원문/출처를변형하므로 함수이름을보존하고검토후보만표시한다. slider는페이지아닌청크이웃이며본문과context별도/deepcopy/context0/토큰예산구분. LLM보강의1000자묵시절단/metadata.update원본덮어쓰기를제거하고fullinput·확인model주입·metadata.generated에분리한다. JSONschema/finish/중복키/필드타입/has_dataNone/languageunknown를검사한다. 원래 Python fence 안에같은길이의fence를삽입해코드를조기에닫던렌더오류도수정했다.

HERDR_ENV=1 현재 pane재확인은pane_not_found. Claude 의견/다른pane제어없음. 최적전략·자동연결/토큰예산정책·완전통합은협의보류. 근거있는오류/내용보존만정정했다.

1. 원래Markdown절·전략도식1개/7Python예제목적을대조했다. 기존fence9개중하나는조기종료된prompt잔여fragment였으며정상8개(7Python+diagram1)로복구했다. 첫검사에서원래잘못된fence수동일assert가실패했고정상화검사로수정했다.
2. Python3.14.2/markdown-it-py4.2/OpenAI3.24/HTTPX0.28.1에서7AST/정의실행·생성4이미지의실제10/None/20/10000/빈추출·2000/1500기본값과실측67문자최대·구분자포함원문재조립·모든source_char범위·fence안가짜heading/table·Setext제목·leadingpipe없는표·멀티라인Figure/inline보존·deepcopy/context0/negative·boundary후보만보존·doc_typeunknown거부·전체1500자SDK모의전달·JSON완료/키덮어쓰기/has_data문자열/keywords형식/중복/배열/length거부통과. 빈이미지가0길이source_span에섞이는실제오류를발견하고 a<b검사를추가한뒤재검증통과. 실제VLM/임베딩/검색정확도미실행.
3. exactpm_notes CLIopen 후 오래된CUAhandle은cgWindowNotFound로실패;현재md.obsidian재연결하여1.13.7/경로/H1/6meta/card screenshot, Next2구조비교표AX/전략도식/첫예제 screenshot을확인했다. 아래전체코드/마지막JSON예제읽기화면은대기. 전체39개링크/YAML/CLI/대기1SHA/nonMD18/diff는후속기록한다.


VLM 청킹 추가 후 현재39개 상대 링크/앵커/첨부 오류0·unique YAML/검토일/type/tags·exactpm_notes CLI properties39 통과. 원래38경로·미검토1개 SHA·ai-dt 비Markdown18개 SHA 보존, git diff --check -- ai-dt 통과. 전체 저장소 감사·전체 아래 읽기 화면·실제VLM/검색품질·Claude협의는 미완료다.


## 입력 전환 하이브리드 상세 추가 검토 — 2026-10-04

RAG 원문38개 모두 개별 검토, 새 README/본 기록 포함40개. token_strategy원문11개 모두 검토. 원래절·도식3개·5Python예제의gate/추출기/라우터/3입력사례/비교 목적을대조했다. Phase1현재100%/점진해제/이상적Phase3/비용절감은2026-02-12당시가정/계획으로보존했다. 현재운영사실로고쳐쓰지않는다. 파일 이동/삭제/첨부/실행 코드 수정 없음.

[Python difflib](https://docs.python.org/3/library/difflib.html)·[asyncio to_thread](https://docs.python.org/3/library/asyncio-task.html)·[DOCX본문/표API](https://python-docx.readthedocs.io/en/latest/api/document.html)·[XLSX data_only](https://openpyxl.readthedocs.io/en/stable/api/openpyxl.reader.excel.html)를확인했다. Python공식페이지표시3.14.8/로컬3.14.2, openpyxl공식stable표시3.1.3/로컬3.1.5차이를구분했다. 형식별추가일차자료/조건은같은topic대표PDF/PPTX/XLSX/DOCX문서와화면파이프라인에있다. 검증일2026-10-04;현재최신/사내판본을단정하지않음.

폴더부분문자열/파서성공으로해제·모든예외로DRM활성확정하는오류를없애고조회API미구현은UNKNOWN. DRM정보와route/파일·이미지승인True/False/None을분리한다. 승인None/False는실행하지않으며파서예외는DRM확정/자동capture로전환하지않는다. 화면폴더정렬/결과i+1을manifest실제page/image로정정한다. 기존중복파싱함수의PPTX그룹/동일title본문/notesNone·XLSX삭제후행번호/data_only·DOCX표누락/heading상속·PDF빈페이지/수명문제는같은주제대표예제함수를주입하도록정리해수정근거/예제를한곳에서읽게했다. 원래추출단위·누락/본문/원소metadata를보존하고DOCX/XLSX/PPTX모르는page는None(PPTXslide_num별도),PDF실page유지. 동기파서는to_thread,순차batch실패전파/thread강제취소미보장. SDKclient/model명시주입·import시실제작업없음. 사내status조회/권한시스템을구현한것은아니다.

비교에서DRM/승인상태를임의로NO_DRM/ACTIVE로바꾸던로직을제거했다. 양쪽승인조건을검사하고route만replace,직접파싱은groundtruth아님/문자열유사도는정확도아님을설명한다. autojunk=False/반복조건과문자수·양쪽records유지·빈한쪽ratioNone을명시한다. 동일원본판본/범위·읽기순서/수치/표/누락평가는별도대기. 기존입력3사례와로드맵도식은맥락을보존한다.

HERDR_ENV=1현재pane재확인은pane_not_found;Claude의견/다른pane제어없음. 운영전환·최적경로/범위정렬·완전중복통합/정책결정은협의보류. 검증가능한중복예제오류는기존대표함수주입으로수정했다.

1. snapshot대비원래Markdown절/도식3개정확보존·5Python예제목적/DRM3상태+unknown·3입력사례·원래계획대조통과.
2. Python3.14.2/Pillow12.3/SDK3.24/HTTPX0.28/python-pptx1.0.2/docx1.2/openpyxl3.1.5/PyMuPDF1.28.2/textsplitters1.1.3에서5AST/runtime와실제대표문서함수주입으로생성PPTX본문+빈slide2·DOCX본문/표·XLSXheader2/row3·5/0·False·PDFtext+빈page2추출통과. 생성imageJPEG/실제SDKmock으로manifestpage10·unknownDRM·양쪽compare/approvalNoneFalse1거부·routeNone거부·missingfile전파/noVLMfallback·빈양쪽유사도None·잘못된record거부·deepcopy·모든생성원본SHA불변통과. 첫실행에서PDF를화면prompt_type으로직접전달하는통합오류를찾아general로매핑하고metadata는pdf로유지했다. missingPDF예외는PyMuPDF전용타입이어서테스트catch를정정한뒤통과. 실제DRM/viewer/model/search업무자료는미실행.
3. exactpm_notes CLIopen·Obsidian1.13.7경로/H1/6meta/card screenshot,Next3상태/명시경로설명·첫gate예제screenshot확인. 아래전체코드/비교/로드맵읽기화면은미완료. 전체40개링크/YAML/CLI/원래38경로/nonMD18/diff후속기록.


입력 전환 상세 추가 후 RAG 원문38개 모두 문서별 검토행이 존재함을 원래 목록과 대조했다. 현재40개 상대 링크/앵커/첨부 오류0·unique YAML/검토일/type/tags·exactpm_notes CLI properties40 통과. 원래38경로와 ai-dt 비Markdown18개 SHA 보존, git diff --check -- ai-dt 통과. 전체 저장소 최종 감사·남은 읽기 화면·실제DRM/모델·Claude협의는 미완료다.


## token_strategy 읽기 화면 추가 검증 — 2026-10-04

확인된 pm_notes vault/Obsidian1.13.7에서11개를 빠른 전환기의 정확한 기존경로로 열고 실제 읽기 모드·검토일·상단경로를 확인했다. 각 문서를 PageDown으로 끝까지 탐색하며 매화면의 native 접근성 상태를 수집했다. 일반 Markdown parser의 고유 절 제목 수와 관측한 제목을 대조했고 모두 일치했다. 코드/표/역사구분·하단참조가 읽기 상태에 나타났다. 대표 실제 screenshot은 VLM청킹/화면추출/DRM목차·XLSX/PPTX·전략·청킹목차의 하단과 전환안 비교코드/참조에서 확인했다. DOCX/PDF/총론은 이번 전체절AX+기존상단screenshot을 함께 근거로 사용한다. 모든 행의 모든 픽셀을 시각 검수한 것은 아니며 실제 모델/정책/업무 품질의 증거로 사용하지 않는다.

| 문서 | 탐색 상태 수 | 원문 고유절 = 관측 고유절 |
|---|---:|---:|
| [token_strategy/when_drm/vlm-chunking-strategy.md](./token_strategy/when_drm/vlm-chunking-strategy.md) | 18 | 15 |
| [token_strategy/when_drm/screenshot-vlm-pipeline.md](./token_strategy/when_drm/screenshot-vlm-pipeline.md) | 21 | 26 |
| [token_strategy/when_drm/post-drm-hybrid.md](./token_strategy/when_drm/post-drm-hybrid.md) | 17 | 14 |
| [token_strategy/when_drm/README.md](./token_strategy/when_drm/README.md) | 6 | 10 |
| [token_strategy/docx-tokenization.md](./token_strategy/docx-tokenization.md) | 16 | 15 |
| [token_strategy/xlsx-tokenization.md](./token_strategy/xlsx-tokenization.md) | 16 | 18 |
| [token_strategy/pdf-tokenization.md](./token_strategy/pdf-tokenization.md) | 16 | 16 |
| [token_strategy/pptx-tokenization.md](./token_strategy/pptx-tokenization.md) | 17 | 16 |
| [token_strategy/overview-chunking-methods.md](./token_strategy/overview-chunking-methods.md) | 14 | 14 |
| [token_strategy/recent-rag-strategy-2026.md](./token_strategy/recent-rag-strategy-2026.md) | 23 | 40 |
| [token_strategy/README.md](./token_strategy/README.md) | 6 | 8 |


이전의 “아래 읽기 화면 미완료” 기록은 당시 상태이며 이 추가 검증으로 위11개 문서의 전체절 탐색을 보완했다. 최초 오래된 app handle은 inactive 오류였고 현재 md.obsidian 재연결로 복구했다. End/super+End는 이동하지 않았고 PageDown은 이동했다. token_strategy/README는 fuzzy 목록의 다른 README가 함께 표시되어 자동 guard를 멈췄으며 실제 첫 정확경로를 확인하고 열어 breadcrumb가 when_drm 아닌 token_strategy임을 재검증했다. 새 문서 생성/설정 수정/다른vault 접근은 없다. Claude의견 없음; 새 설계 판단 없이 기존 읽기 검증을 수행했다. RAG 나머지29개 읽기 감사와 저장소 전체 최종 감사는 계속한다.


위11개 절수 대조의 첫 시도는 frontmatter의 closing separator를 Setext 제목으로 해석해 한 절을 더 세었다. 실제 Obsidian처럼 YAML을 별도로 제외한 뒤 모든 절수 대조가 통과했다. 기록 추가 후 현재40개 링크/앵커/첨부 오류0·uniqueYAML·exactpm_notes CLI properties40·원래38경로/비Markdown18개 보존·git diff --check -- ai-dt 통과.


## OpenSearch·Milvus 전체 절 읽기 추가 검증 — 2026-10-04

확인된 pm_notes vault/Obsidian 1.13.7에서 OpenSearch 11개와 Milvus 본문 2개를 정확한 기존 경로로 열었다. 실제 breadcrumb·읽기 모드·검토일을 확인하고 PageDown으로 끝까지 이동하며 매 화면의 native 접근성 상태를 수집했다. frontmatter를 제외한 원문의 고유 절 제목 수와 본문에서 관측한 제목 수가 모두 일치했다. 코드·참조와 대표 하단 screenshot도 확인했다. 모든 행의 모든 픽셀을 검수한 것은 아니며 실제 검색 서버·모델·업무 품질의 검증은 아니다.

| 문서 | 탐색 상태 수 | 원문 고유 절 = 관측 고유 절 |
|---|---:|---:|
| [opensearch/opensearch-basics.md](./opensearch/opensearch-basics.md) | 16 | 23 |
| [opensearch/python-client.md](./opensearch/python-client.md) | 11 | 15 |
| [opensearch/keyword-search-bm25.md](./opensearch/keyword-search-bm25.md) | 27 | 24 |
| [opensearch/vector-search-knn.md](./opensearch/vector-search-knn.md) | 23 | 24 |
| [opensearch/hybrid-search.md](./opensearch/hybrid-search.md) | 22 | 23 |
| [opensearch/performance-optimization.md](./opensearch/performance-optimization.md) | 10 | 21 |
| [opensearch/opensearch-handler.md](./opensearch/opensearch-handler.md) | 14 | 15 |
| [opensearch/rag-integration.md](./opensearch/rag-integration.md) | 12 | 13 |
| [opensearch/conversation-memory-opensearch.md](./opensearch/conversation-memory-opensearch.md) | 18 | 17 |
| [opensearch/settings/README.md](./opensearch/settings/README.md) | 15 | 16 |
| [opensearch/README.md](./opensearch/README.md) | 5 | 7 |
| [milvus/milvus-basics.md](./milvus/milvus-basics.md) | 15 | 25 |
| [milvus/milvus-rag-integration.md](./milvus/milvus-rag-integration.md) | 14 | 18 |


BM25 화면의 링크 미리보기에는 본문 밖 다른 문서의 제목 2개가 포함돼 있었다. 백링크·상태 표시 뒤에 나타난 미리보기 subtree를 제외하고 본문 24개 절로 대조했다. 실제 본문이 바뀌었다고 판정하지 않았다. 기존 아래 화면 미확인 기록은 당시 상태이며 위 13개의 전체 절 탐색으로 보완했다.

이후 Milvus 목차를 확인하려던 화면 제어는 cgWindowNotFound 오류가 발생했다. 앱 재연결과 CLI의 exact vault 기존 문서 open은 시도했으며 CLI는 성공했지만 native 화면 연결은 복구되지 않았다. 따라서 나머지 16개 RAG 문서의 전체 절 화면 탐색은 미완료다. 새 문서 생성이나 다른 vault 변경은 없다. Claude 협의 결과 없음; 기존 문서의 검증 작업이며 새 설계 판단은 내리지 않았다.


추가 기록 후 3단계 재검증: 원래 문서38개·현재40개 목록과 검토 결과 대조 통과, ai-dt 비Markdown18개 SHA 보존. 이번 수정은 검증 기록만 추가해 실행 예제는 바뀌지 않았다. 현재40개 링크·앵커·첨부 오류0, unique YAML·검토일·type·tags 및 exact pm_notes CLI properties40 통과, git diff --check -- ai-dt 통과. 화면 제어가 복구되지 않아 나머지16개 읽기 검증은 미완료로 유지한다.
