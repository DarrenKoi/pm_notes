---
tags: [ai-dt, organization, verification]
aliases: [AI DT 문서 정리 기록]
reviewed_on: 2026-10-04
review_status: reviewed_with_limits
document_type: organization_log
---

# AI/DT 문서 정리 기록

기준일 **2026-10-04**. 원본 Markdown178개/43,734행을 목록화하고 임시 원본 스냅샷을 저장했다. 최상위 ai-dt 안의 주제 하위 폴더끼리도 독립적이라는 CLAUDE 규칙을 확인했다. 한 하위 주제를 탐색·수정·검증한 뒤 다음 주제로 진행한다. 실행 코드·비밀·첨부·사용자 변경을 보존하며 커밋/푸시는 하지 않는다.

## 발견한 문서 분포와 현재 범위

| 하위 주제 | 원본 Markdown | 상태 |
|---|---:|---|
| 루트 README/CLAUDE | 2 | 원문2개 검토; 목차 대소문자·실제13개 하위 주제·현재 규칙 설명 정정 |
| LLMOps | 17 | 원래17개 개별 검토; 현재19개 링크/YAML/CLI·로컬 fixture 통과, 화면/사내실행/Claude 미확인 |
| ai-coding-dictionary | 8 | 개별 검토 완료; 제품 실제 실행·Claude 협의 미확인 |
| ai-terms-and-technologies | 11 | 개별 검토 완료; 실제 모델·사내 실행·Claude 협의 미확인 |
| data-handling | 32 | 원문32개 모두 개별 검토; 현재33개 참조/YAML/CLI·로컬 검증 통과, 일부 화면/장비/Claude 미확인 |
| foundation model | 5 | 개별 검토 완료; 실제 통합 실행·Claude 협의 미확인 |
| langchain | 15 | 원래15개 모두 개별 검토;현재17개 링크/YAML/CLI 확인·fixture 실행, 화면/사내실행/Claude 미확인 |
| llm_question | 9 | 개별 검토 완료; 사내 운영·Claude 협의 미확인 |
| mcp | 4 | 개별 검토 완료; 실제 통합 실행·Claude 협의 미확인 |
| ml-dl | 19 | 원래19개 모두 개별 검토; 전체29개 링크/메타데이터/CLI 확인, 일부 읽기 화면·외부 실행 미완료 |
| openwiki | 1 | 개별 검토 완료; 실제 통합 실행·Claude 협의 미확인 |
| rag | 38 | 원래38개 모두 개별 검토. 새 목차/기록 포함40개 참조·YAML·CLI 통과, RAG README 읽기 화면 확인; 추가 문서 화면/서버/모델/Claude 미확인 |
| roadmap | 12 | 개별 검토 완료; 실제 사업·조직 승인·Claude 협의 미확인 |
| unsloth | 5 | 개별 검토 완료; GPU 학습·배포·Claude 협의 미확인 |

## 최초 수정과 판단

- 루트 README의 llmops 소문자 링크를 실제 LLMOps 경로로 맞췄다. 플랫폼별 대소문자 해석에 의존하지 않는다. 학습/업무 기록·주제 독립성·검토 범위를 추가했다. 다른 폴더로 문서를 이동하지 않았다.
- CLAUDE의 순수 Markdown/실행 자료 없음 단정은 binary-reverse-engineering/scripts 등의 실제 분포와 맞지 않아 모듈별 확인으로 정정했다. 공통 서비스 실행을 지시하지 않는다. llm_question의 CONTEXT_CODEX 위치도 실제 경로로 맞췄다.
- 루트 지침의 기존 상위 CLAUDE 참조는 기존 규칙 문맥으로 보존했다. 신규 내용 통합·주제 사이 크로스 링크는 만들지 않는다. 목차는 기존 주제의 시작 위치만 안내한다.
- llm_question CONTEXT_CODEX는 당시 사내 출제 컨텍스트다. 45분 작성+5분 제출 등의 의미를 외부 사실 검증 대상과 구분하고 원문 맥락을 보존한다. 대회 출제·참가 평가·서비스 호출을 실제 실행하지 않는다.

## 협의와 검증 경계

앞 주제의 연결 확인에서 HERDR_ENV=1이었지만 Herdr current pane은 pane_not_found였다. 현재 ai-dt의 Claude 의견은 아직 없고, 연결 시도/협의가 필요한 선택은 실제 증거와 함께 하위 주제 기록에 남긴다. 임의 다른 pane·vault를 수정하지 않는다. 현재는 목록화·루트 규칙 검토 단계이며 개별 기술 주장·예제·링크·Obsidian 전체 검증을 완료했다고 주장하지 않는다.

## Foundation model 하위 주제 검토

[하위 정리 기록](./foundation%20model/organization-log.md)에 원래 5개 문서의 개별 결과와 세 단계 검증을 남겼다. 경로·제목·고유 예제를 유지했고 새 정리 기록을 포함한 6개 링크·YAML·CLI 확인 및 README → Attention 읽기 탐색이 통과했다. 공개 일차 자료 11건과 attention 수치 계산을 확인했다. Claude 협의·실제 모델 실행은 미확인이다. 다른 하위 주제는 아직 검토 중이다.

## OpenWiki 하위 주제 검토

[하위 정리 기록](./openwiki/organization-log.md)에 원래 README의 개별 검토와 공식 자료 5건 대조를 남겼다. Node 요구 조건, host/독립 실행 인증, 생성 범위와 CI history 조건을 수정했다. 원래 절과 예제를 유지하고 bash 10개 구문·YAML·링크 검사를 통과했다. Obsidian properties 2개와 README → 해당 정리 기록 읽기 탐색을 확인했다. 실제 OpenWiki CLI/API/CI·Claude 협의는 미완료다.

## MCP 하위 주제 검토

[하위 정리 기록](./mcp/organization-log.md)에 원래 문서 4개의 개별 검토와 버전·API 변경을 남겼다. 신규 버전 안내를 포함한 6개 문서의 YAML·링크·CLI properties와 README → 버전 안내 읽기 탐색을 확인했다. Python fence 10개, bash 3개, JSON 1개와 산술·가짜 client 흐름·실제 OpenAI SDK의 로컬 요청 직렬화 검증이 통과했다. 신규 참조 오류는 없으며 기존 형제 주제 참조 6개는 유지했다. 실제 MCP/HTTP/API·신규 namespace 실행·Claude 협의는 미완료다.

## Unsloth 하위 주제 검토

[하위 정리 기록](./unsloth/organization-log.md)에 원래 문서 5개의 개별 검토와 설치·데이터·TRL API·export 조건을 남겼다. 대표 조건 안내를 포함한 7개 문서의 YAML·링크·CLI properties와 README → 조건 안내 읽기 탐색을 확인했다. Python AST 5개, bash 2개, JSON roundtrip과 가짜 tokenizer를 이용한 실제 함수 실행이 통과했다. 신규 참조 오류는 없다. 실제 GPU/모델 다운로드·dependency lock·serving·Claude 협의는 미완료다.

## AI Coding Dictionary 하위 주제 검토

[하위 정리 기록](./ai-coding-dictionary/organization-log.md)에 원래 8개 문서의 개별 검토를 남겼다. 저자 비유·제품별 공식 계약·가상 사내 사례를 구분하고 학습·요청 수·cache 과금·실패 진단·subagent/지침 로딩 일반화를 수정했다. 111개 원래 절 제목과 text 도식 7개를 유지했다. 현재 10개 YAML·CLI properties와 README → 적용 조건 읽기 탐색, softmax 로컬 수치 검증이 통과했다. 신규 참조 오류 0개, 기존 형제 참조 10개 유지. 공식/일차 자료와 원문 18건을 대조했고 실제 제품 실행·책 인용·Claude 협의는 미확인이다.

## AI 입문 용어/기술 하위 주제 검토

[하위 정리 기록](./ai-terms-and-technologies/organization-log.md)에 원문 11개의 개별 검토와 상세 문서/연속 요약의 역할을 남겼다. GPU 계산 가정·RAG/학습/평가 일반화·토큰/추론 조건을 수정하고 보안 사건은 당시 발표와 별도 업데이트로 구분했다. 현재 13개 YAML·CLI properties, README → 적용 조건 읽기 탐색과 GPU 가정 수치 검산이 통과했다. 신규 참조 오류 0개, 기존 형제 참조 5개 유지. 원래 text fence 40개를 유지했고 공식/일차 자료 27건을 대조했다. 실제 모델/검색/학습·정부 원문/최종 사건 조사·사내 적용·Claude 협의는 미확인이다.

## LLM 출제 기록 하위 주제 검토

llm_question 원문 9개를 보존하고 90분 초안·50분 개정·3파트 문맥의 차이를 목차와 공통 검토 안내에 정리했다. 기존 입력 메모 상대 링크 1건을 복구했다. 원래 본문과 첨부 2개 보존, 점수 합산 4개, 현재 13개 YAML·링크·CLI properties와 README → 검토 안내 읽기 탐색을 확인했다. 새 참조 오류 0개. 공식 모델 카드 3건을 대조했고 실제 사내 모델·운영 승인·평가 타당성·Claude 협의는 미확인이다.

## 로드맵 업무 기록 하위 주제 검토

roadmap 원문 12개를 보존하고 보고서 설계·speaker notes·ADR·회의 준비·인계 역할을 목차와 공통 검토 안내에 구분했다. 표의 적용 칸 14개·간트 행 19개 및 희귀 고장 지표 산술을 검산했다. 공식/일차 자료 3건 대조, 첨부 8개 보존, 현재 15개 YAML·CLI properties·상대 참조를 확인했다. UI에서 slug 앵커의 절 이동 실패를 발견해 14곳을 제목 fragment로 복구하고 실제 Chapter 4 이동을 재검증했다. 새 참조 오류 0개, 기존 상위 참조 3회 유지. 실제 일정·목표·baseline·fallback 선택·조직 승인·Claude 협의는 미확인이다.

## ML/DL 데이터 처리 부분 검토

원래 루트 README와 데이터 처리 4개를 검토하고 로딩·EDA·피처·Pipeline의 역할을 구분했다. 공식 자료 15건·Python AST 70개·실행 fixture 18종과 원래 제목/fence 보존을 확인했다. 새 상대 참조 오류 0개, 현재 9개 CLI properties 및 실제 읽기 화면 탐색이 통과했다. 다른 14개 ML/DL 문서는 아직 개별 검토하지 않았다. Claude 연결이 불가능해 모델별 인코딩 최적화·예제 재설계·완전 중복 통합은 보류했다. 전체 ai-dt 원래 문서 178개 중 현재 62개 검토, 116개 대기다.

클래식 ML 워크플로우 1개 추가 검토: R²/불균형 F1·분할/누수·pandas 수동 fold 오류를 공식 근거와 실행 7종으로 수정했다. AST 13개·원래 제목/fence 보존·현재 3개 CLI/상대 참조·실제 읽기 탐색 확인. 현재 ai-dt 원래 문서 63개 검토, 115개 대기다.

모델 평가 원문 1개 추가 검토: API/지표/수익 입력·폐기 penalty를 근거와 실행으로 수정, 원래20절/16fence 보존, 15AST/8종/8PNG 통과, 새 참조 오류0·CLI4/실제 읽기 탐색 확인. ai-dt 원문 64개 검토, 114개 대기.

회귀 원문1개 추가 검토: 분할/누수·규제 CV·stopping·잔차 조건 수정, 원래18절/10fence 보존, 실제 scikit-learn 4모델/6검증/6PNG 통과, CLI5·새 참조0·실제 읽기 표 복구 확인. XGBoost import는 OpenMP 부재로 미검증. ai-dt 원문65개 검토, 113개 대기.

분류 원문1개 추가 검토: 분할/stopping/가중/LightGBM bagging 조건 수정,16절/7fence 보존·AST6/검증9/PNG1·새 참조0/CLI6 확인. native 두 모델 import 실패와 실제 읽기 화면 timeout은 미확인으로 구분. ai-dt 원문66개 검토,112개 대기.

군집 원문1개 추가 검토: TSNE API/DBSCAN 예외·밀도/거리/표시 해석 수정,15절/15fence 보존·AST14/검증9/PNG8·새 참조0/CLI7 확인. 실제 읽기 화면은 CUA timeout으로 미완료. ai-dt 원문67개 검토,111개 대기.

튜닝 원문 추가 검토와 클래식 ML 전체 대조: 6개120절/75fence·AST69·새 참조0·CLI8 확인. 전체 튜닝 예산/native 실행 및 분류·군집·튜닝 읽기 화면은 미확인. ai-dt 원문68개 검토,110개 대기.

PyTorch 기초 개별 검토: 공식2.14 근거·AST8·원래20epoch CPU 학습/복원 포함8종·새 참조0·CLI3 확인. GPU/worker/변환·읽기 화면은 미확인. ai-dt 원문69개 검토,109개 대기.

학습 루프 원문 추가 검토: 원래14절/8fence·AST8·CPU fixture9종·PNG2·새 참조0·CLI4 확인. CIFAR/GPU/worker/RNG 동일성·읽기 화면은 미확인. ai-dt 원문70개 검토,108개 대기.

CNN 원문 추가 검토: 22절/13fence·AST11·CPU/torchvision fixture7종·새 참조0·CLI5 확인. CIFAR/weights/50epoch/GPU/worker·읽기 화면은 미확인. ai-dt 원문71개 검토,107개 대기.

시퀀스 원문 추가 검토: 19절/10fence·AST7·CPU 검증7종·한글 PNG1 재검증·새 참조0·CLI6 확인. 시간 분할/packing 복원을 검증했고 50epoch/실제 센서/읽기 화면은 미확인. ai-dt 원문72개 검토,106개 대기.

전이학습 추가 검토와 딥러닝 전체 대조: 원래5개85절/48fence·AST40·전이학습CPU7종/공식HF sourceAST2·새 참조0·CLI7 확인. 실제 pretrained/CIFAR/BERT/전체예산/읽기·Claude 협의는 미확인. ai-dt 원문73개 검토,105개 대기.

저장/로딩 원문 추가 검토: 29절/16fence·AST14·CPU9종/ONNX18 ORT 수치 비교·새 참조0·CLI3 확인. GPU/판본변경/장애복구/읽기·Claude 협의는 미확인. ai-dt 원문74개 검토,104개 대기.

FastAPI 서빙 원문 추가 검토: 17절/17fence·AST11·CPU TestClient9종/bash2/YAML1·새 참조0·CLI4 확인. 실제 socket/Docker/GPU/부하/CDN/읽기·Claude 협의는 미확인. ai-dt 원문75개 검토,103개 대기.


실험 추적 원문 추가 검토 및 ML/DL 전체 대조: 원래19개/9,572행 모두 원래 경로·fence 수 보존, Python AST214개 오류0·uniqueYAML29·새 참조 오류0·Obsidian CLI properties29개 확인. 추적은 실제 SQL/CPU6종·세 모델 원래 설정/CV5·Pipeline·validation 선택/test1회·registry2versions/alias·CSV/JSONL 경계 검증을 수행했다. 상세 공식 근거/판본은 해당 정리 기록에 있다. MLflow UI/Lightning/원격 서비스/GPU/운영/일부 Obsidian 읽기·Claude 협의는 미확인. ai-dt 원문76개 검토,102개 대기다.


## LangChain 커리큘럼 개별 검토

원래15개/1,828행·193절/fence64개·study_list13항목을 대조하고 원래 경로와 fence/텍스트 bytes를 보존했다. v1/classic import·agent/HITL·SQLite 수명·trim 연결·unknown 분기·empty RAG/rewrite상한·FAISS 정규화/신뢰 파일·필터와rerank/논문과축약/keyword 포함률과정답률을 구분했다. 공통 접속/판본/근거를 같은 주제의 대표 안내로 모았다. AST63·로컬 라이브러리 fixture10종·실제SDK/HTTP MockTransport4종/요청13개·현재17개 YAML/링크 오류0/CLI 확인. 공식 experimental sunset/archive도 확인했다. 실제 endpoint/모델/토큰/OCR/PDF/운영/Gradio·Claude 협의·CUA 읽기창timeout은 미확인으로 해당 정리 기록에 남겼다. ai-dt 원문91개 검토,87개 대기다.


## LLMOps 기초 부분 검토

원래17개/2,270행·study_list15항목을 읽고 README/커버리지/01~06(8개)을 개별 수정·검증했다. 나머지07~15(9개)는 아직 수정·출처/실행 검증 대기다. 공통 설정/근거를 대표 안내로 모았고 원래 절/fence/경로·미수정9개·비Markdown18개를 보존했다. Python AST19개·실제SDK MockTransport요청21개·실제OTel/OpenInference 하위span3개와 원문 숨김을 포함한8검증 묶음이 통과했다. 수정/신규10개 YAML/링크 오류0/ObsidianCLI 확인. 실제 모델/endpoint/OCR/BERTScore가중치/Phoenix서버·Claude협의·CUA읽기창124초timeout은 미확인으로 구분했다. ai-dt 원문99개 검토,79개 대기다.


LLMOps 심화07~11의5개 추가 검토: strict judge schema/정규화·pairwise 무승부/순서불일치/형식오류, nDCG ideal전체gold·검색지표/주장unknown, 실제JSONSchema·최종상태/성공률, 인젝션 falsepass/가짜승인/안전None, CI필수check와 A-B근사/비열등 조건을 수정했다. AST21·SDK3.24 모의38요청과 별도Python3.12/Ragas0.3.1의3지표/actuallegacyadapter/SDK2.54 모의8요청·NaN무주장 실패 검증이 통과했다. Ragas v1import/Pillow/Python3.14async 실패도 기록했다. 현재15개 YAML/링크0/CLI 확인·미수정4개/비Markdown18개 보존. 실제judge/endpoint/CI/DB/트래픽/운영정책·Claude·읽기화면은 미확인이다. ai-dt 원문104개 검토,74개 대기.


## LLMOps 운영 및 전체 개별 검토

12~15의4개를 추가 처리했다. Phoenix client3.5 자원/UTC·span와 요청 집계/coverage·unknown, 명시적 scorer 조립과 미확인·적용 제외, 일관된7아티팩트/draft·전체 rollback·기존 파일 보존, 가상 사고 candidate/검수 전 golden 금지를 수정했다. 추가 AST14·실제 Phoenix SDK HTTP MockTransport6요청과 실패 입력 fixture 통과. 원래17경로·study_list·비Markdown18개 보존, 전체19개 링크/앵커/첨부 오류0·unique YAML/CLI properties 통과. 전체AST54·bash2·YAML3 구문과 최초/심화 SDK 모의 요청을 재실행했다. 실제 운영/인증/모델 품질·Claude 협의·읽기 화면 및 협의가 필요한 완전 중복 통합은 미확인/보류다. ai-dt 원문108개를 개별 검토했고 RAG38개·datahandling32개, 총70개는 대기다. 다른 주제로 이동/통합하거나 새 교차 링크를 만들지 않았다.


## RAG 대화 메모리 부분 검토

원래38개/11,542줄 목록과 기존 하위 목차를 확인했다. 원문 대화 메모리1개에서 MessagesState 삭제 reducer·반복 요약/대화 경계·미검수 후보 JSON·동기 실행/메모리 수명, Milvus nested entity/소유자 값 바인딩·collection/차원·UTC 조건, 논문 검색 점수 가중합과 OSS/managed 구분을 수정했다. 실제 LangGraph1.2.12/core1.6.6 StateGraph와 fake 모델·DB callback fixture/AST4 통과. Milvus 실제 SDK/server·모델 품질은 미검증이다. 새 RAG 목차/정리 기록 포함3개 참조 오류0·YAML/CLI 통과, 원래38경로·미수정37개·비Markdown18개 보존. CLI의 실제 소문자 vault 경로/로컬 samefile 확인 후 README 읽기 screenshot과 내부 링크를 검증했다. 대화 메모리 본문/정리 기록 화면은 아직 미완료다. Herdr pane_not_found로 완전 중복 통합 협의는 보류했다. ai-dt 원문109개 검토/69개 대기(RAG37·datahandling32)다.


## RAG LangChain·LangGraph 조립 예제 추가 검토

원문3개를 추가 처리했다. README YAML을 frontmatter로 옮기고 판본/실행 조건을 기록했다. 최소 LCEL·StateGraph는 함수/부분 state 갱신/입력 계약으로 설명하고 확장 후보와 구현을 구분했다. 플레이북의 누락 tool route를 실제 ToolNode 결과 루프와 연결하고 호출 예산·허용 목록·오류를 검사했다. 가상 일정과 최신 근거 미보장·문자 청킹·빈 검색을 명시했다. 실제 FAISS/TextLoader·LCEL/graph/tool과 모의 모델/embedding, SDK HTTP MockTransport4요청/실패 입력 통과. 추가AST7·bash1, 현재6개 참조 오류0·unique YAML/CLI와 원래38경로·미수정34개/비Markdown18개 보존 통과. 읽기 화면은 파일 선택/본문이 어긋나고 빈 화면이 관찰돼 새3개 검증은 미완료다. dev:errors에는 기록이 없었고 앱 설정을 수정하지 않았다. Herdr pane_not_found로 Claude 협의/완전 중복 통합은 보류다. ai-dt 원문112개 개별 검토/66개 대기(RAG34·datahandling32)다.


## RAG LangGraph 시리즈 추가 검토

원문4개를 추가 처리했다. frontmatter/읽기 순서·LCEL 분기/병렬·Document 상태와 reducer·분류 unknown을 수정하고, RAG의 원래 질문 보존/정확 yes-no·빈 근거 abstain·유한 재작성/재생성을 설명했다. 승인 자동 재개를 실제 interrupt/명시 응답과 executor callback으로 분리했다. 가상 subgraph·SQLite context manager 수명/독립 스키마/replay와 부작용 롤백 구분·stream mode/v2 이벤트를 보완했다. 실제 StateGraph·임시 SQLite 재열기/history/replay·async GenericFakeChatModel 및 잘못된 승인/미수집 stream 실패 검증·AST11 통과. 현재10개 참조 오류0/unique YAML/CLI·원래38경로/미수정30개/비Markdown18개 보존·git diff --check 통과. 실제 모델/운영/회사 권한·Claude 협의·추가4개 읽기 화면은 미완료다. ai-dt 원문116개 개별 검토/62개 대기(RAG30·datahandling32)다.


## RAG 고급 목차·파이프라인 추가 검토

원문2개를 추가 처리했다. 모델/schema와 품질 보증을 구분하고 고급 문서 역할·실습 조건을 명시했다. Chroma 자동 삭제를 없애고 새 경로 생성/manifest 기반 재사용·기존 폴더/불일치 거부를 분리했다. DirectoryLoader 재귀/숨김/UTF-8·빈 입력과 평균 unknown·문자 fallback을 보완했다. 원래 경로/제목/fence·PM/반도체 맥락을 보존했다. 추가 AST9·실제 Chroma1.5.9/langchain-chroma1.1.0 저장/MMR·DB 모든 바이트 보존·새 Python 프로세스 재열기·실제 SDK HTTP MockTransport2요청/실패 입력 통과. Claude pane_not_found로 통합/채택/임계값 협의는 보류이며 실제 모델/회사 데이터·읽기 화면은 미완료다. ai-dt 원문118개 검토/60개 대기(RAG28·datahandling32)다.


## RAG Agentic 구현 추가 검토

원문1개를 추가 처리했다. 원래 질문/검색어 분리·입력 초기화/명시 조립, 기본2회 재작성 한도·빈 근거 abstain, 정확 schema/unknown·텍스트 응답 검사를 보완했다. reducer 저장과 멀티턴 prompt 활용·출처 후보와 인용 정확성·파일명 smoke와정답 평가를 구분했다. 과거 워크숍 수치/문서 역할/제목·15 Python fence를 보존하고 재현 미확인과 길이/문서 수의 한계를 남겼다. 실제 StateGraph·InMemorySaver/reducer·실패 입력·로컬 Mermaid/AST15 통과. 원문11개 검토·현재13개 검사 범위이며 ai-dt 원문119개 검토/59개 대기(RAG27·datahandling32)다. 원격 모델/회사 데이터/Claude 협의·새 본문 읽기 화면은 미완료다.


## RAG 확장 추가 검토

원문1개의 HyDE query만 갱신·기본 state/node 명시 재사용, 실제 messages를 generator에 전달/현재 질문만 검색하는 한계, InMemorySaver 재사용·SQLite 파일/context 수명·reducer 계약을 보완했다. 워크숍의656 chars·PM/반도체 모든 수치·비용/지연 배수는 미확인 과거 기록으로 보존하고 품질 우위/원인 단정을 제거했다. 검색 결과를 LLM 호출로 센 식을 구분하고 실제 비용/정답 평가 조건을 남겼다. 추가 AST12·실제 StateGraph·checkpoint/reducer·FakeMessagesListChatModel/SQLite 재열기/실패 입력 통과. ai-dt 원문120개 검토/58개 대기(RAG26·datahandling32), RAG 현재14개 검사 범위다. 실제 모델/회사 자료/이력 검색어 해소·Claude/새 본문 UI는 미완료다.


## RAG 멀티에이전트 통합 추가 검토

원문1개를 추가 처리해19 Python fence/PM·공정·Excel/SQL/RAG/보고서·SKILL 역할을 보존했다. 명시 factory/도구·unique YAML 등록과 native skills/권한·기본 추가 도구를 구분하고 외부 근거 없는 분석/보고서는 미확인 초안으로 표시했다. 완료건수·읽기 SQLite mode=ro/authorizer·실제 task 요청/반환·values 최종 AI 검사를 보완했다. 추가 AST19·실제 pandas/SQLite 실패 입력·DB바이트 보존·create_agent/Deep Agents0.7.21의 task→child business tool→result→parent flow와 fake 모델/stream 검증 통과. 실제 provider/회사 권한·라우팅 품질/native skills/운영·Claude 협의는 미확인이다. ai-dt 원문121개 검토/57개 대기(RAG25·datahandling32), RAG 현재15개 검사 범위다.


## RAG Milvus 시리즈 추가 검토

원문3개를 추가 처리했다. 직접 SDK/래퍼 schema를 구분하고 자동삭제·미정의category·HNSW/nprobe 불일치·제곱L2/VARCHAR바이트·제품별 단정을 수정했다. stable ID와 최초add/이후upsert·PDF 문자분할·빈근거/unknown·승인된1회폴백/재판별을 보완했다. 추가AST16/bash3·실제pymilvus3.0.2/Lite3.2.1 저장/재열기·실제langchain-milvus0.4.0 similarity/MMR/filter/LCEL·PDF/StateGraph 실패입력 통과. 실제실행이 발견한load/최초upsert 조건을 수정·재검증했다. Docker/분산/HNSW/hybrid서버·실제모델/품질/운영/Claude는 미확인이다. Milvus목차→기초링크/코드일부와연동H1/card/meta 실제읽기 화면확인; 전체아래코드는미완료다. ai-dt 원문124개검토/54개대기(RAG22·datahandling32), RAG현재18개검사범위다.


## RAG OpenSearch 목차·기초·클라이언트 추가 검토

원문3개를추가검토했다.현재라이선스/재단·node.roles/샤드·유효JSON/벡터차원·명시판본의loopback실습·기존index/ID보존을수정했다.syncpool_maxsize/asyncmaxsize·bulk부분실패/바이트·scan예외해제·asyncfinallyclose와오류를빈결과로바꾸지않는계약을보완했다.추가AST12/JSON2/YAML1/bash3·실제SDK3.2.0 Requests모의28요청과실제aiohttp임시REST서버의정상/404/취소close를검증했다.실제OpenSearch/Docker/mapping/검색품질/TLS/부하·Claude는미확인이다.목차→클라이언트실제링크및H1/meta/card·키보드PageDown후TLS코드화면·기초top/표AX확인.기존깨진Codes참조2개는주제독립적인문자설명으로남겼다.ai-dt원문127개검토/51개대기(RAG19·datahandling32),RAG현재21개검사범위다.


## RAG OpenSearch BM25 추가 검토

원문1개를 추가 검토해 Nori 추가 플러그인 조건·OpenSearch3.0 BM25 배율 변경·keyword 정확 일치/phrase/should/highlight를 근거에 따라 수정했다. 기존 절과 Python 예제11개·고유 문장5개·작성일을 보존했다. 안정 create ID/성공실패 집계·자동 index 삭제 제거·partial timeout 상태 전파를 SDK3.2.0의 모의 REST23요청으로 확인했다. 실제 서버/Nori/검색 품질은 미확인이다. Obsidian 정확 경로/읽기 H1/meta/callout·PageDown 수식/표 AX를 확인했다. Herdr pane_not_found로 Claude 의견과 구조/품질 정책은 대기다. ai-dt원문128개 검토/50개 대기(RAG18·datahandling32), RAG현재22개 검사 범위다.


## RAG OpenSearch 벡터 검색 추가 검토

원문1개 추가 검토. 엔진별 판본·metric/score/exact/recall·필터 시점·모델 차원/전처리 계약·force merge/워밍업 조건을 일차 근거로 보완했다. 원래 절·8 Python/2 JSON fence·7개 고유 문장·작성일을 보존했다. SDK3.2.0 모의 REST25요청으로 기존 데이터 보존/embedding 계약/제공 벡터/응답 충돌·부분 실패 처리를 확인했다. Obsidian 정확경로 H1/meta/card·Next 이동 엔진/점수 표/JSON·Python 검사 함수 screenshot을 확인했다. 실제 서버/ANN/script 점수·검색 품질/Claude는 미확인이다. ai-dt원문129개 검토/49개 대기(RAG17·datahandling32), RAG현재23개 검사 범위다.


## RAG OpenSearch 하이브리드 검색 추가 검토

원문1개 추가. normalization2.10/hybrid2.11/native RRF2.19 조건과 가중 점수/순위/재순위 역할을 구분하고 후보 제한·identity/동점/원문 보존·명시 pipeline/모델·unknown mode를 수정했다. 원래 절·Python7·고유 샘플3개·작성일을 보존했다. 대표 벡터 정의와 실제SDK3.2 모의REST38요청을 함께 실행했고 RRF 산술/기존 데이터/부분 실패/재순위 원문 미변경이 통과했다. 정확 Obsidian 경로/H1/meta/card와 Next4 RRF예제 screenshot/도입 조건 표 AX 확인. 실제 서버/모델 품질·모든 아래 코드 화면·Claude는 미확인이다. ai-dt원문130개 검토/48개 대기(RAG16·datahandling32), RAG현재24개 검사 범위다.


## RAG OpenSearch 성능 설계 추가 검토

원문1개 추가. managed/OSS shard 기준·heap/native/cache 분모·미평가 용량·refresh/fsync/flush·force merge 조건과 REST 구문을 일차 근거로 수정했다. 원래21절/7예제/사례·작성일 보존, dynamic cluster 예시1개 추가. REST행7/JSON6/YAML1·산술 통과, 실제 서버 요청0. 정확 Obsidian 경로/H1/meta/card screenshot과 첫 shard 설명 AX 확인. 실제 성능·아래 모든 코드 화면·Claude는 미확인. ai-dt원문131개 검토/47개 대기(RAG15·datahandling32), RAG현재25개 검사 범위다.


## RAG OpenSearch Settings 추가 검토

원문1개 추가. 타입/doc_values/_id·template 우선순위/新index·read alias·rollover OR·ISM index 생성 age/문서별 보존 차이와 policy→template→bootstrap 순서를 수정했다. 원래15절·6예제/고유필드·작성일 보존. AST1/JSON5·실제SDK3.2 모의REST17요청·기존 자원/403전파/계획 deepcopy 통과. 정확 Obsidian 경로/H1/meta/card screenshot/타입 설명 AX 확인. 실제 서버/ISM/삭제·모든 아래 예제 화면·Claude 미확인. ai-dt원문132개 검토/46개 대기(RAG14·datahandling32), RAG현재26개 검사 범위다.


## RAG OpenSearch 핸들러 추가 검토

원문1개 추가. 원래15절/API표/7Python/고유예제·작성일을 보존하고 로컬 wrapper 현재성을 미확인으로 구분했다. 공식SDK호환·네트워크 부수효과·toy schema·ID와 명시경로 조건을 수정했다. AST7/stdlib 경로 fixture 통과, wrapper/서버 실행0. 정확 Obsidian 경로/H1/meta/warning screenshot 확인. ai-dt원문133개 검토/45개 대기(RAG13·datahandling32), RAG현재27개 검사 범위다. Claude/실제wrapper/모든 아래 코드 화면 미확인.


## RAG OpenSearch 연동 추가 검토

원문1개 추가. 실제import/legacy판본·community sunset·Retriever/internalsearch 구분·원질문/근거/생성분리·weightedRRF/identity를 수정했다. 원래13절/4Python/bash1/100GB질의·작성일 보존. 실제래퍼SDK모의REST12요청/StateGraph/로컬BM25·Ensemble identity 검증 통과. 정확Obsidian 경로/H1/meta/sunsetcallout screenshot 확인. 실제server/부분응답/모델품질·아래모든코드화면·Claude미확인. ai-dt원문134개 검토/44개 대기(RAG12·datahandling32), RAG현재28개 검사범위다.


## RAG OpenSearch 대화 메모리 추가 검토

원문1개 추가. 완전JSON/Faiss조건·효율적user필터/minimum_should_match·dense배치/요약완료/소유자경계·미확인manager/운영선택을 수정했다. 원래17절/9예제·고유계층/필드/프로필예시/작성일 보존. HTTPX모의10요청+실제SDK모의2검색/조립fixture 통과. 정확Obsidian 경로/H1/meta/callout screenshot/첫JSON AX 확인. 실제server/모델/manager/모든 아래 코드 화면·Claude미확인. ai-dt원문135개 검토/43개 대기(RAG11·datahandling32), RAG현재29개 검사범위.


## RAG 추출·청킹 목차/총론 추가 검토

원문2개 추가. 추출/분할/토큰화구분·현재import/실제splitter길이/상한/buffer/headermetadata·experimental sunset·agentic원문경계/latetokenpool조건·미측정별점표를수정했다. 목차6절/총론14절·6Python/고유방법/모델후보·작성일보존. 실제splitters/Semanticfixture/NumPy/원문경계검증통과. 정확Obsidian목차→총론실제link클릭/H1/meta/callout/첫코드AX확인. 실제model/tokenizer/품질/모든아래코드화면·Claude미확인. ai-dt원문137개검토/41개대기(RAG9·datahandling32), RAG현재31개검사범위.


## RAG DOCX 추출/청킹 추가 검토

원문1개 추가. 실제python-docx/Mammoth·본문순서/표투영/중첩/생략셀·문자fallback/크기/source·prefix불변·HTML중복/변환경고를수정했다. 원래15절/9예제·고유맥락/작성일보존. 실제임시DOCXfixture·AST7통과, Obsidian경로/H1/meta/card screenshot/구조/사용법AX확인. 실제Word/업무자료/Unstructured/모델/아래코드화면·Claude미확인. ai-dt원문138개검토/40개대기(RAG8·datahandling32), RAG현재32개검사범위.


## RAG XLSX 추출/행 청킹 추가 검토

원문1개 추가. 원수식/캐시unknown·실제행좌표/0/False/NA·대형시트중복제거·통계/LLM구분·원병합유지/표시투영을수정했다. 원래18절/9예제·고유맥락/작성일보존. 실제임시XLSX/openpyxl/pandasfixture·AST8통과, Obsidian정확경로/H1/meta/card screenshot/전략표AX확인. 실제Excel계산/업무자료/Unstructured/모델/아래코드화면·Claude미확인. ai-dt원문139개검토/39개대기(RAG7·datahandling32), RAG현재33개검사범위.


## RAG PDF 추출/구조 청킹 추가 검토

원문1개 추가. 공식표/OCR기능·혼합페이지/50자판별한계·볼드비트·페이지Markdown·SDK PNG·후처리/원요소/HTMLunknown보존을수정했다. 원래15절/8예제·고유설정/작성일보존. 실제임시PDF/PyMuPDF/4LLMfixture·SDKmock4·AST7통과. Obsidian정확경로/H1/meta/card screenshot·Next4표/diagram/첫코드/UnstructuredAX확인. 실제OCR/Unstructured/업무자료/모델/아래전체화면·Claude미확인. ai-dt원문140개검토/38개대기(RAG6·datahandling32), RAG현재34개검사범위.


## RAG PPTX 추출/슬라이드 청킹 추가 검토

원문 1개 추가. 재귀 그룹/동일 제목 본문·표/병합·노트 None·빈 슬라이드 번호·원문/Vision 별도 보존·원 PPTX 이웃 맥락·새 임시 변환/대응unknown을 수정했다. 원래15절/7예제·고유 설정/작성일 보존. 실제 생성PPTX/SDKmock4/생성PDF렌더링·AST6 통과, 실제 LibreOffice/Unstructured/VLM 미실행. Obsidian 정확경로/H1/meta/card screenshot·Next3 첫 그룹 코드 screenshot 확인; 아래 전체 화면·Claude 미확인. ai-dt 원문141개 검토/37개 대기(RAG5·datahandling32), RAG 현재35개 검사 범위.


## RAG 2026 전략 메모 추가 검토

2026-03-14 당시 제안 원문 전체/32절/text예시2를 정확 보존하고 검증일2026-10-04 모델 명시값·입력/서빙 조건·Late pooling·OpenSearch tokenizer/max_chunk_limit·fusion/rerank 정정을 앞에 추가했다. 실제 사내 API/모델·품질/Claude는 미확인이다. Obsidian 경로/H1/meta/역사분류/card screenshot·모델표/토큰조건 screenshot 확인, 아래 원문전체 화면 대기. ai-dt 원문142개 검토/36개 대기(RAG4·datahandling32), RAG 현재36개 검사 범위.


## RAG DRM 목차 추가 검토

원래Why이후본문/도식2개를정확보존하고당시99%/유일/해제가정과현재승인된입력/권한unknown·취득/청킹/향후전환역할·읽기순서를추가했다. Microsoft공식IRM/권한사례대조,실제사내제품/운영/Claude미확인. Obsidian정확경로/H1/meta/card·역사구분screenshot 확인. ai-dt원문143개검토/35개대기(RAG3·datahandling32), RAG현재37개검사범위.


## RAG 화면 추출 상세 추가 검토

공식 모델/Pillow/vLLM/SDK 기준으로 모델 분류·PNG·페이지 순서·응답/unknown·휴리스틱과 재시도 예제를 정정했다. 원래 절/그림2개·11Python목적·역사 가정/추정치는 보존했다. 생성 이미지/실제SDK 모의통신 검증 통과, 실제화면/DRM/모델 추론 미실행. Obsidian H1/meta/card screenshot/모델표AX 확인, 아래전체화면 미완료. ai-dt144원문 검토/34대기(RAG2·datahandling32). Claude 협의는 pane_not_found로 보류.


## RAG VLM 결과 청킹 상세 추가 검토

실제Markdownparser/문자범위와manifest로 원문·출처/unknown/빈추출을보존하고 자동문장이동·page+table중복·LLMmetadata덮어쓰기를 정정했다. 절/전략도식1개·7Python목적과기존작성일보존;조기닫힌fence복구. 7AST/로컬텍스트·SDKmock검증통과, 실제모델/검색품질미실행. ObsidianH1/meta/card/첫예제 screenshot확인;아래전체읽기대기. ai-dt145원문검토/33대기(RAG1·datahandling32), Claude협의pane_not_found보류.


## RAG 입력 전환 상세 추가 검토

마지막원문1개를검토했다. 원래절/도식3개/5Python목적/당시계획보존, DRMUNKNOWN과명시입력승인/route분리·동일topic대표파서주입·출처/empty/None·문자열비교비정답정정. 실제4형식생성파일과SDKmock통합검증통과(PDFpromptgeneral통합오류수정), 실제DRM/모델미실행. ObsidianH1/meta/card/gate예제screenshot확인;아래전체읽기대기. ai-dt146원문검토/32대기(datahandling32), RAG38원문개별검토/현재40. Claude협의pane_not_found보류.


## RAG token_strategy 전체절 읽기 보완

11개 문서를 exactpm_notes vault의 읽기 모드에서 PageDown으로 끝까지 탐색하고 실제 nativeAX의 고유절을 원문 parser와 대조했다. 8~40절 모두 일치, 코드/표/역사구분·하단참조와 대표 screenshot 확인. 모든 행의 pixel/실제모델·사내정책 검증은 아니다. RAG 나머지29개 읽기 감사 대기. 원문 개별검토146/178·datahandling32대기는 변동없다.


## RAG 검색 문서 전체 절 읽기 보완

OpenSearch 11개·Milvus 본문 2개를 pm_notes vault의 읽기 모드에서 끝까지 탐색해 원문의 고유 절 수와 대조했다. 7~25절 모두 일치했다. BM25의 본문 밖 링크 미리보기 2개 제목을 제외했다. 대표 하단 screenshot 확인은 실제 서버·모델 검증이 아니다. 현재 native 창 연결 cgWindowNotFound로 RAG 나머지16개 전체절 화면 검증은 미완료다. CLI는 기존 exact vault 문서 open에 성공했다. 원문 개별검토146/178·data-handling32 대기는 변동 없다.


## data-handling 첫 문서 검토

원래32개 Markdown/비Markdown4개를 목록화하고 목차·Task 의존성·이벤트3개를 검토했다. Airflow2.10.5를 대조 판본으로 정해 DAG간논리날짜·leaf 판정·skip/exit·worker 로컬 파일·retry jitter·Dataset event와 run의 비일대일·3.x SDK/API·웹훅 HTTP 오류 처리를 정정했다. 기존 절과 fence 수, 작성일·네 주제 목차, 미수정29개/비Markdown4개 SHA 보존. Python AST20·순수 함수1·HTTPX mock4·bash exit3 통과. 수정/신규4개 상대 참조 오류0·unique YAML/CLI properties4·diff 검사 통과. 실제 Airflow scheduler/provider/MinIO/사내환경·Native 읽기 화면은 미확인, Claude 연결pane_not_found로 중복 통합/서비스 재전송 정책 협의 보류다. ai-dt 원문149/178검토, 남은29개는 data-handling이며 my-task/루트/전체 최종감사는 계속 대기다.


## Airflow 커리큘럼 첫 검토 추가

목차·01·02·08의 4개를 추가 검토했다. 사내 권한 가정과 제품 기능을 구분하고 env/backend Connection 공급·2.10.5 예제 판본·3.x API·설명용 코드·Git Sync의 역할을 정정했다. 08의 H1 이후 원문 전체는 역사 시나리오로 정확히 보존했다. 실제 임시 Airflow 2.10.5/Python 3.12.12에서 예제 DAG 8개 발견·Task ID/edge 대조와 외부 접속 없는 PythonOperator 함수 1건을 실행했다. AST 38개·env loader 누락 4건 통과. 실제 scheduler·사내 endpoint·인증/CA/TLS·업무 파일은 실행하지 않았다.

원래 경로·고유 절·작성일, 미수정 25개/비Markdown 4개 SHA 보존. 현재 검토 8개 상대 참조 오류0·unique YAML/CLI properties 통과. Native 창 연결이 복구돼 현재 8개 전체 절 탐색·원문 제목 수 대조·대표 screenshot 확인을 마쳤다. 본문 밖 미리보기 제목은 제외했다. Claude pane_not_found로 사내 정책·구현 통합·완전 중복 통합은 보류했다. ai-dt 원문153/178개 검토, 남은25개는 data-handling이다. my-task·루트·전체 최종 감사는 대기다.


## data-handling 커리큘럼 03~07 추가 검토

5개 원문을 추가 검토해 카테고리·원래 절·예제 수·작성일·경로와 08 역사 본문을 보존했다. 스케줄/구간·XComArg·DB transaction·S3 copy/delete·Bash Python 선택·secret 주입·CA/constraints·DAG/Parquet 테스트·Celery queue/Pod·callback 적용 조건을 공식 판본과 소스로 정정했다. AST64·실제 DagBag11·URI callable1·KST template/bash 구문1·Parquet 내용 pytest1·DAG 정상 graph1와 음성 fixture2 검증 통과. 사내 운영·외부 접속·Kubernetes provider import·Claude 통합 협의는 미확인이다. ai-dt 원문158/178개 검토, 남은20개는 data-handling이다. my-task·루트·전체 최종 감사는 대기다.


## data-handling MinIO 연결 사례 추가 검토

원문1개 추가 검토. 기존 절·fence28·작성일·경로를 보존하고 `/tmp` 분산 handoff·TLS/CA·Task 성공/멱등성·Windows AST/실제 import·tasks test 부작용·관찰성 한계를 정정했다. key basename 덮어쓰기·부분 context·payload 식별자 덮어쓰기·임시 분석 파일 충돌을 수정하고 local fixture로 검증했다. AST15·실제 Parquet 값·S3Hook 대역2·DagBag2/helper대역·관찰성 context4/실패·Handler 유실 음성 fixture 통과. 실제 provider/MinIO/server/Windows와 Claude 계약·정책 협의는 미확인. Obsidian 읽기40상태/본문56절 source 대조 통과. ai-dt 원문159/178개 검토, 남은19개는 data-handling 정규화9·역공학10개다. my-task·루트·전체 최종 감사는 대기다.


## data-handling 정규화 입문 추가 검토

목차·핵심 정의·설계 절차 3개를 추가 검토했다. 모든 원래 절·text 예제·작성일·경로를 보존했다. 후보키/슈퍼키·FD·NULL·이메일·현재 주소/거래 스냅샷과 업무 키 조건을 근거에 따라 정정했다. SQLite3.50.4 문서 SQL/무결성 음성3건과 FD closure/3NF·BCNF·무손실/종속성 비보존 반례 검증 통과. 실제 업무 모델·제품 동기화·Claude 통합 협의는 미확인. ai-dt 원문162/178개 검토, 남은16개는 정규화6·역공학10개다. my-task·루트·전체 최종 감사는 대기다.


## data-handling OpenSearch 정규화 적용 추가 검토

원문1개를 추가 검토했다. 기존8개 본문/절/작성일을 보존하고 HTTP/JSON 표기·이메일/trim·nested/join·projection/chunk/벡터 조건·pipeline 적용 단계를 보완했다. 공식3.4/2.15 판본과 유지보수 종료 표시를 구분했다. 본문10개 파싱·배열 AND/OR 반례·weights 음성4건/shape 계약은 통과했으며 실제 서버는 미실행이다. Claude 통합/업무 계약 협의는 pane_not_found로 보류했다. ai-dt 원문163/178개 검토, 남은15개는 정규화5·역공학10개다. my-task·루트·전체 최종 감사는 대기다.


## data-handling MongoDB/Redis 정규화 적용 추가 검토

원문2개를 추가 검토했다. 모든 원래 절/작성일/경로와 고유 주문/snapshot/glossary/key/alias 예제를 보존했다. 이메일 case·unique/validator/원자성·Vector Search/index·8.0 JSON/Search/7.4fieldTTL·Cluster/key/동기화 조건을 공식 근거로 보완했다. MongoDB JSON8/JS2와 DB대역3·Draft4제한subset 정상/음성5·empty/glossary 반례, Redisbash7/대역10/JSON/인자/메일/chunk/timestamp 검증 통과. 실제 서버/검색/업무 계약·Claude 협의는 미확인. ai-dt 원문165/178개 검토, 남은13개는 정규화3·역공학10개다. my-task·루트·전체 최종 감사는 대기다.


## data-handling 정규화 시리즈 전체 개별 검토

RAG/온톨로지/통합3개 추가로 정규화9개 전체를 검토했다. 모든 원래 절/작성일/경로·고유 예제 맥락을 보존하고 판본/chunk/날짜/권한·ID/label·OWL/SHACL/RDF·BGE-M3차원·알람키/매뉴얼resolver/cache 문맥을 근거에 따라 보완했다. JSON11/AST1·cache 순수함수 정상/변경7/음성33·날짜경계·매뉴얼2판본/SQLite복합키FK·cosine음성3 통과. 실제 모델/서비스/장비·Claude 통합 계약 협의는 미확인. ai-dt 원문168/178개 검토, 남은10개는 data-handling역공학이다. my-task·루트·전체 최종 감사는 대기다.


## data-handling 역공학 실행 도구 README 검토

소스와 실제 합성 selftest에 맞춰 명령10개/검사28개·JSON/exit/인자 오류 계약·입력/출력 파일 역할을 정정했다. 고유 실습 명령과절/경로를 보존했으며 실행 코드 변경은 없다. 실제 장비 포맷/Claude 통합 협의는 미확인이다. ai-dt 원문169/178개 검토, 남은9개는 역공학 문서다. my-task·루트·전체 최종 감사는 대기다.


## data-handling 역공학 법률 참고 검토

공식미국조문/2021CJEU판결/SEMI목록과EU공식검색색인 범위를 구분해 법률 일반화와EDA/이미지허용 보장을 정정했다. 원래 절/작성일/맥락을 보존했다.2009지침 직접 조회와실제 관할/계약·Claude협의는 미확인이다. ai-dt 원문170/178개 검토,남은8개는역공학 문서다. my-task·루트·전체 최종 감사는 대기다.


## data-handling 역공학 개념과 작업 파이프라인 검토

개념/작업 계약/runbook/task4개를 추가 검토해 entropy/stride/count/EOF/배열 범위와 후보확정·단위·unknown·입출력/JSON/exit·Kaitai writer 조건을 정정했다. 모든 원래 절/작성일/phase/task·byte/findings 맥락을 보존하고 실행 코드는 수정하지 않았다. AST2/bash9와 실제도구 반례를 확인했으며 실제장비/reader/Claude통합은 미확인이다. ai-dt 원문174/178개 검토,남은4개는 역공학 목차/도구총람/벤더포맷/좌표 문서다. my-task·루트·전체 최종 감사는 대기다.


## data-handling 역공학 목차와 좌표/recipe 검토

목차/좌표2개를 추가 검토했다. 원래 절·작성일·고유 도식/목차를 보존하고 지원/확정 보장·ASCII/직렬화·단위/좌표계·TLV 부모 경계를 정정했다. bash6와 실제 serial/TLV 반례,31개 YAML/상대참조/CLI properties를 통과했다. native AX와 screenshot의 현재 문서가 일치하지 않아 새 두 문서와 부모 기록의 읽기 확인은 미완료다. Claude 통합/실제 계약은 pane_not_found로 보류했다. ai-dt 원문176/178개 검토,남은2개는 역공학 도구총람/벤더포맷이다. my-task·루트·전체 최종 감사는 대기다.


## data-handling 전체 원문 개별 검토 결과

도구 총람/벤더 포맷2개를 추가해 data-handling원문32개 전체를 개별 검토했다. binwalk/unblob 설치·rawDEFLATE·가격/지원/휴리스틱·FEI/Zeiss tag/None·SEMI명칭/실제판본·공개부재 단정을 근거에 따라 보완했다. PythonAST8/bash1/ksy·순수 계산/압축·fake metadata/tail 반례와33개 YAML/상대참조/CLI properties를 통과했다. 실제 장비·외부 reader/compiler/GUI·Claude완전통합/회사계약은 미확인이다. native/CLI 렌더링이 이전 문서를 유지해 새4개/부모와 기존 AI 일부 전체읽기 재검증은 미완료다. ai-dt 원문178/178개 개별 검토, 원문 검토 대기는0개다. my-task·루트·전체 최종 감사와 읽기 검증은 대기다.


## 최종 목차·참조 대조 — 2026-10-05

원문178개 모두 개별 검토 결과를 해당 하위 주제의 정리 기록에 남겼다. 현재213개 Markdown을 다시 대조한다. 루트 목차에 빠져 있던 ML/DL, roadmap, llm_question의 README 입구를 추가했다. 내용은 주제 간 이동·통합하지 않았다. 위의 중간 진행 수치는 당시 작업 기록으로 유지하며 현재 미검토 원문 수를 뜻하지 않는다.

용어 사전에서 이번에 추가한 GitHub 방식 절 참조292곳은 제목·도식·예제 내용을 유지한 채 상대 파일 링크와 절 이름 안내로 복구했다. 정확한 제목 fragment로 이전에 복구·실제 탐색을 확인한 roadmap14곳은 유지했다. 모든 Markdown 렌더러의 절 이동을 인증하는 것은 아니다. 신규 참조 오류와 metadata, 원문 경로·비Markdown 보호는 전역 최종 검사에 다시 포함한다.

기술 근거·예제 실행의 기존 확인일2026-10-04는 유지한다. 이번 목차·참조 최종 대조일은2026-10-05이며 기술 정보를 하루 뒤에 모두 재검증한 것으로 표시하지 않는다. 실제 모델/서버/GPU/회사 계약·미확인 독립 주장과 Claude 협의가 필요한 추가 물리 통합, 일부 전체 읽기 화면은 각 하위 주제에 남긴 경계를 따른다.

최종 재검사 결과:213개 Markdown·고유 YAML 검토 속성·새 참조 오류0개를 확인했다. 기존 상위 지침/문맥 참조4회는 원문대로 유지했다. 주제 하위 폴더 사이에 신규 내용 참조0개이며 전역 원문178개 경로도 모두 존재한다. exact pm_notes vault의 전체418개 metadata 확인에 이 폴더213개가 포함된다. 신규 용어 사전 링크 수정을 포함한 diff whitespace와 원본 비Markdown/원문 보존 검사가 통과했다. 읽기 화면·Claude 협의·실환경의 한계는 위 기록대로 남는다.
