---
tags: [langchain, documentation-review]
reviewed_on: 2026-10-04
review_status: partial
document_type: organization_log
category_major: "AI·DT"
category_middle: "문서 관리"
category_minor: "정리·검증 기록"
note_kind: "관리 기록"
classified_on: "2026-10-05"
---

# LangChain 커리큘럼 정리 기록

## 범위와 구성 결정

원래 Markdown15개/1,828행·study_list.txt의 4영역/13항목을 모두 읽고 대조했다. 01~04 모델·도구,05~08 그래프,09~12 RAG,13 조립의 기존 순서를 유지했다. README는 학습 순서와 판본·대표 적용 조건/정리 기록으로 연결한다. curriculum-coverage는 목록 점검, study_list는 원래 커리큘럼,13절은 제안 요구사항/골격으로 구분했다. 작성일2026-07-06과 검토일2026-10-04를 분리했다.

같은 접속 설정/판본·검증 경계의 반복 설명은 verified-conditions에 대표로 모으고 각 노트에서 참조한다. 코드의 반복 client 선언은 독립 실습 맥락을 위해 유지하며 같은 주제를 더 설명하는 01/02/03과09/10/11/12의 역할을 구분했다. 파일 이동/삭제·다른 주제로의 내용 통합/신규 링크·실행 코드/첨부 변경은 없다. 완전한 문서 재분할/대표 코드 모듈 추출은 Claude 협의가 불가해 보류했다.

## 문서별 개별 검토

| 문서 | 변경/판단 |
|---|---|
| README | 실행 가능한 완전/최신·현재 사내 차단 단정을 조건으로 수정. 환경변수·확인된 상위 pin/선택 dependency와 읽기 순서 제공. 같은 SDK와 모든 기능 호환을 구분 |
| 01 | 원시 SDK/Model/Prompt 흐름 보존. v1 package 역할·usage 미관측/내용 블록·stream 청크·provider 초기화 조건 명시 |
| 02 | Runnable의 자동 streaming/native async/병렬 보장 제거. batch max_concurrency2·JSON parser의 구조/사실성 차이·fallback 조건 설명 |
| 03 | 결정적 분기도 Chain으로 가능함을 설명. 수동 loop 최대8호출·unknown tool 실패·severity1~5 검증. method=function_calling을 명시하고 구조화 출력의 실패/사실성 경계 표시 |
| 04 | placeholder 날씨 API·설비 fixture/실업무 연결 구분. 날씨 transport/HTTP/JSON/schema 처리, 설비 UNKNOWN 거부·없는 알람 미관측 유지, URL segment 인코딩, spec 인증 누락 복구. GET transport/429/지정5xx만 최대3회 재시도;401/JSON 오류는 재시도 안 함 |
| 05 | node/update/reducer/graph 예제 유지. START/END 일반화 수정·stream_mode=updates 명시 |
| 06 | SQLite connection이 닫힌 graph를 뒤에서 조회하던 오류를 sqlite_graph로 분리. 메모리 graph 수명 유지·새 trim function을 새 graph에 연결. trim input과 저장 상태·thread 인증·실제 tokenizer 조건 설명 |
| 07 | 미관측/비문자 감정을 긍정으로 기본 처리하지 않고 unknown node로 분기. 가상 설비 출력·입력 선언·recursion_limit의 superstep 실패 조건 표시 |
| 08 | deprecated create_react_agent/prompt를 v1 create_agent/system_prompt로 이관. create_agent도 checkpoint/HITL 지원. 승인 메시지만 기록하는 예제임을 표시·interrupt replay/실제 부작용·인증/binding 조건 구분 |
| 09 | Flat exact 의미와100% 의미 정확도를 분리. raw dense embedding·unit L2 정규화/save-load 설정·실제 반환 차원·별도 HNSW L2 인덱스 유지. HNSW metric을 생성자에 전달해 내부 metric 불일치 방지. Flat에만 add/delete 실행·pickle 허용은 직접 생성/신뢰 파일만 |
| 10 | legacy retriever import를 classic으로 이관. RRF 순위 가중치·BM25 learned sparse 차이·LLMChainFilter가 rerank가 아닌 필터임을 설명. long_text 누락 복구. experimental sunset/archived 상태·신규 운영 채택 보류 표시 |
| 11 | retriever Tool import 이관·yes/no Literal·선행 ctx/answer/out 변수 복구. empty retrieval을 실패로 처리하고 rewrite최대2회/빈 근거 보류. CRAG 논문 전체와 축약·Self-RAG reflection-token 방법과 별도 judge를 구분. 반환 source 목록과 claim citation 검증 구분 |
| 12 | DRM99%/유일 경로·30B 정확성 단정은 출처 없이 미확인으로 표시. vision alias와 공개 Qwen3-VL-30B-A3B id를 구분. 페이지1/2/10 숫자 정렬·실제 page 번호/image_path/미검토 OCR metadata·빈 OCR 실패. indexing import classic/실제 upsert 미구현·같은 embedding 계약 명시 |
| 13 | 목표30문항/80%/5초/환각0건을 실측과 분리. skeleton은 dense MMR/단일 질문만 구현; hybrid/기억/gate 미구현. 신뢰 index/normalize 계약 통일·keyword 포함률과 정답률을 구분·빈 평가셋 미측정 처리. UI history 미사용 표시 |
| curriculum-coverage/study_list | 매핑13/13을 점검. 이전 점검 이력은 현재 사실로 고쳐 쓰지 않고 원래 기록으로 표시. study_list.txt의 bytes는 보존 |

## 근거와 Claude 협의

2026-10-04 공식 migration/models/tools/agent/structured output·Runnable·graph/checkpointer/interrupt·FAISS wiki/공식 wrapper 구현·embedding 구현·BGE/Qwen model card·CRAG/Self-RAG 원논문·Contextual Retrieval 발표·HTTPX/Tenacity·experimental issue87을 대조했다. 판본/주장별 링크는 verified-conditions에 있다. FAISS 옛 통합 링크는404여서 공식 구현과 설치 판본으로 대조했다. Gradio 문서 fetch는 실패했고 실제 UI/API 호환은 미확인이다. 불필요한 플러그인을 추가하지 않았다.

HERDR_ENV=1에서 current pane 조회가 pane_not_found였다. 작업 전용 Claude pane을 연결할 수 없어 분할/통합 재설계·사내 모델/승인 정책·experimental 대체 선택을 보류했다. 다른 pane을 제어하거나 standalone Claude를 실행하지 않았고 Claude 의견을 만들지 않았다. 직접 재현한 오류와 공식 계약 수정은 진행했다.

## 세 단계 검증

1. 원래 Markdown15개·193절·fence64개와 고유 예제를 대조했다. 모든 원래 파일·각 파일의 원래 fence 수를 보존했다. 제목 일부는 최신/무료/자동 보장 대신 조건/필터/후보/버전 표기로 수정했으며 내용 문맥은 위 표에 남겼다. study_list.txt 초기 해시 동일. 현재17개 문서에 개별 결과·대표 조건/목차·기록이 있다. 파일 이동/첨부 손실은 없다.
2. Python AST63개 오류0·실제 라이브러리/fixture 검증10종 통과. LCEL pipe/parallel/batch/configurable/JSON parser, tool schema/multiply84/수동 loop/create_agent/Pydantic 범위, 메모리 thread/SQLite close-reopen/trim 새 graph/get_state, ToolNode loop/recursion 초과 실패, create_agent memory/interrupt approve·reject, normalized Flat/HNSW/save-load/MMR/add-delete, BM25+vector RRF/classic imports/LLMChainFilter/MultiQuery/SemanticChunker, RAG/source·empty retrieval2rewrite 후 보류/agent 검색/faithfulness globals, mini denseMMR/keyword metric, 감정 unknown 보존을 실행했다. 값은 LLM/embedding fixture다. 실제 모델 토큰 계산/LLM 품질 증거가 아니다.

   별도 실제 OpenAI3.24/ChatOpenAI1.6.7/httpx2 MockTransport 검증4종/요청13개 통과: chat usage/SSE2청크/toolcall/Pydantic/raw text·float embedding 직렬화,01절 원시/사내 SDK·Model/Prompt/init import/stream, HTTPX Response 날씨 schema/timeout/connect/401/JSON 오류·MES encoded id/UNKNOWN/미관측 알람/auth·spec4011회/429·500·timeout3회/JSON1회·spec 인증,12절 vision image message 직렬화/숫자 page1·2·10/image_path/metadata. 재시도 대기 타이머는 검증에서 wait_none으로 생략했다. OCR용 파일은 가짜 bytes로 메시지 포맷만 검증했고 실제 이미지/OCR은 아니다. 실제 HTTP socket/모델/DB 서버/PDF/웹/Gradio/사내 설비·부하·인증/승인 실행은 없다. 모든 임시 index/checkpoint/file/package는 저장소 밖에 생성했다.
3. 최종 상대 링크/앵커·unique YAML·확인된 pm_notes vault의 CLI 속성을 검사한다. 실제 읽기 창은 2026-10-04 cua.getApp('Obsidian')을 timeout10초로 재시도했으나17.5초 뒤 timeout/kernel reset이었다. 화면 탐색·렌더링은 미완료이며 CLI 성공과 구분한다. 현재 최종 링크/속성 결과는 아래 확정 항목에 남긴다.

## 남은 미확인

실제 공개/사내 모델/API·schema/vision/usage/tokenizer·Korean BM25/검색/인용/판정 품질·DRM/권한/export/OCR/PDF·운영 쓰기/승인/영속 DB/동시성/재시도 deadline·Gradio·experimental 대체 설계·Claude 협의·Obsidian 읽기 화면은 미확인이다. 기존 고유 예제는 보존했고 성공/최신/실무 완성으로 단정하지 않는다.

## 최종 확정 검증

현재17개 상대 링크/앵커 오류0(기존 오류도0)·unique YAML17개·bash1 syntax·pm_notes Obsidian CLI properties17개 검토일 확인·diff 공백 검사 통과. ai-dt 비 Markdown18개 초기 해시 동일. 마지막 weather import/math·비문자 sentiment·자연 page정렬·LLMChainFilter 변수명·trim graph 수정 후 두 실행 검증을 재실행해 모두 통과했다. CUA 읽기 창 timeout은 해결하지 못해 화면 검증은 미완료다.
