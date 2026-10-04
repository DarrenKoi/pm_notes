---
tags: [llmops, evaluation, review]
reviewed_on: 2026-10-04
review_status: partial
document_type: maintenance_record
---

# LLMOps 정리 기록

## 범위와 보존

원래 17개 Markdown(01~15·README·커버리지)과 study_list.txt를 전부 읽었다. 현재 원래17개(01~15·README·커버리지)의 개별 수정·검증을 마쳤다. 최초8개·심화5개·운영4개 단계의 검증은 아래에 나눠 기록했다. 새 문서2개는 공통 적용 조건과 본 기록이다. 원래 경로·목차·15단계 예제 흐름을 유지했고 이동/삭제/실행 코드 변경은 없다. 완전한 중복 통합·분할은 협의 연결 실패로 보류했다. 공통 접속/근거/검증 경계만 대표 안내로 모았다.

## 개별 검토 결과

| 원래 문서 | 변경 및 검토 결과 |
|---|---|
| [README.md](./README.md) | 읽기 순서 유지; 환경변수/응답·embedding 검사; 사내 상태와 실제 품질 미확인 구분; partial/2026-10-04 |
| [curriculum-coverage.md](./curriculum-coverage.md) | study_list 15항목 대조; 14/15가 목록 밖이라는 오류 수정; 완료 주장 대신 검토 상태 구분; partial/2026-10-04 |
| [01-llmops-overview-lifecycle.md](./01-llmops-overview-lifecycle.md) | 운영 개념/공인 아닌 maturity 수준 구분; null 응답·빈 데이터·유한 점수 검사; partial/2026-10-04 |
| [02-prompt-management-versioning.md](./02-prompt-management-versioning.md) | 누락 슬롯 실패/JSON 중괄호·경로 성분 검사; callback/빈 점수/가상 출력 표시; partial/2026-10-04 |
| [03-tracing-observability.md](./03-tracing-observability.md) | JSON 타이머와 OTel span 구분; privacy 설정/요청 context 복원/unknown usage·cost/collector와 server 구분; partial/2026-10-04 |
| [04-llm-evaluation-overview.md](./04-llm-evaluation-overview.md) | 과제별 점수 범위/인간 라벨 조건; 전체·카테고리 지표 보고와 ID별 A/B 비교; partial/2026-10-04 |
| [05-eval-dataset-construction.md](./05-eval-dataset-construction.md) | 실제 검색/gold 근거 구분; 합성 object·items 검증/미검수 표시; VLM·DRM 조건; partial/2026-10-04 |
| [06-automatic-metrics.md](./06-automatic-metrics.md) | 소수점/부호·빈 reference/한글 ROUGE·BLEU signature/cosine 입력 계약; BERTScore 선택 조건; partial/2026-10-04 |
| [07-llm-as-a-judge.md](./07-llm-as-a-judge.md) | strict pointwise schema/score1..5 정규화·reference 계약; pairwise TIE/disagreement/형식 오류 구분; meta_eval 미확인; partial/2026-10-04 |
| [08-rag-evaluation.md](./08-rag-evaluation.md) | nDCG ideal/Recall/P@k/RR 입력; claims/schema/역질문 근사; Ragas legacy판본/실제 framework·SDK 검증; partial/2026-10-04 |
| [09-agent-tool-evaluation.md](./09-agent-tool-evaluation.md) | JSONSchema 실제 검증; 정확 경로/최종 상태·binary 성공과 pass@k 구분; 금지 로그와 실행권한 구분; partial/2026-10-04 |
| [10-safety-hallucination-guardrails.md](./10-safety-hallucination-guardrails.md) | answerable/누출 판정 unknown; 인젝션 검증 callback/가짜 승인·누락 시간; 안전 True/False/None 구분; partial/2026-10-04 |
| [11-online-eval-deployment.md](./11-online-eval-deployment.md) | manifest/finite gate/부동소수점 경계; CI required check; canary 비율/A-B 독립·소표본·비열등 조건; partial/2026-10-04 |
| [12-monitoring-drift.md](./12-monitoring-drift.md) | Phoenix3.5 resources/UTC·span와 요청 구분/집계 coverage·unknown/annotation·중심 거리·후보 포인터; partial/2026-10-04 |
| [13-mini-project.md](./13-mini-project.md) | 실제 callback 계약/데이터 schema·unknown·적용 제외/의심 scan 방향·미구현 패키지 제거; partial/2026-10-04 |
| [14-artifact-lineage-governance.md](./14-artifact-lineage-governance.md) | 일관된7아티팩트·전체 롤백/draft·전체HEAD·dirty·기존 파일 보호/NIST와 조직 정책 구분; partial/2026-10-04 |
| [15-incident-response-postmortem.md](./15-incident-response-postmortem.md) | 가상 사고·비난 없는 후속 조치/sanitized candidate·answerable/reference unknown·검수 전 golden 금지; partial/2026-10-04 |

## 결정과 Claude 협의

HERDR_ENV=1에서 herdr pane current --current가 pane_not_found를 반환했다. 관련 없는 다른 프로젝트 pane을 사용하거나 제어하지 않았다. 작업 전용 Claude pane을 확보하지 못해 의견 전달/협의 결과는 없다. 공식 근거/실제 설치 소스/로컬 fixture로 명백한 입력·API·지표 오류만 수정하고, 분류 재설계·같은 폴더 완전 중복 통합·golden 규모·judge/모델/안전 임계값·운영 도구 선택은 보류했다. 과거 업무 기록을 현재 사실로 다시 쓰지 않았다(이 묶음은 학습 문서다).

## 검증 1 — 최초 단계 목록과 고유 내용

8개 모두 원래 읽기 절과 fence 수를 비교했다. 기본 앱/하네스·registry·trace/usage/RAG·QA 생성/라벨·정규화/BLEU/ROUGE/embedding/BERTScore의 고유 예제 역할을 유지했다. 17원문 존재와 study_list 원문 보존을 확인한다. 전체 LLMOps 정리 완료로 계산하지 않는다.

## 검증 2 — 최초 단계 근거와 실행

[공통 적용 조건](./verified-conditions.md)에 공식 근거·확인일·판본을 기록했다. 임시 환경에서 다음 8검증 묶음이 통과했다. 실패 입력도 실제 호출 경로를 실행했으며 원격 socket/실제 모델을 사용하지 않았다.

- inventory8 original paths/headings/fencecounts retained; Python AST19
- README real SDK MockTransport chat/null content/embedding reversed indexes/missing duplicate NaN ragged blank
- 01 actual SDK answer; harness empty/nonfinite guard and cosine negative range
- 02 actual temporary registry JSON braces/missing slot/path components/active/version/harness callbacks/empty/nonfinite
- 04 all-metric category report/empty duplicate nonfinite/id-aligned shuffled regression/missing rows
- 05 actual SDK json_object/request/object shape/items count/field validation/context distinction/unknown content/VLM serialized fixture bytes(no OCR)
- 06 decimal/sign preserved/NFKC contract/empty reference/Korean whitespace ROUGE/F1/BLEU signature/cosine -1 zero thresholds; BERTScore AST only
- 03 real OTel/OpenInference/SDK nested3spans/redacted messages/tokens/context reset/None usage & cost/approved rates/exception type only; register network and Phoenix UI excluded

BLEU signature: `nrefs:1|case:mixed|eff:no|tok:intl|smooth:exp|version:2.6.0`. 최초 검증 보조 스크립트의 제목 집계가 fenced 코드 주석까지 포함하는 오류와 SDK 클래스 patch 방식이 계측기 요청 메서드를 가리는 오류를 고쳐 재실행했다. 문서 API 실패를 성공으로 바꾸거나 실제 서버 검증으로 계산하지 않았다.

## 검증 3 — 최초 단계 탐색과 메타데이터

현재 수정/신규10개 상대 링크·앵커·첨부 참조와 YAML/CLI properties를 확인한 뒤 결과를 아래에 덧붙인다. Obsidian vault는 pm_notes이고 대상 경로는 /Users/daeyoung/Codes/pm_notes다. 읽기 화면 getApp 연결은 timeout(124초)으로 실패해 탐색/렌더링은 미완료다. 다른 vault는 수정하지 않았다.

## 최초 단계에서 남긴 검토 과제

다음은 최초 단계 당시의 대기 상태다. 07~15의 후속 처리 결과는 아래 추가 검토와 현재 개별 결과 표를 기준으로 읽는다.

07~15를 읽으며 pairwise 출력 첫 글자 처리·nDCG ideal ranking·trajectory 상태 검증·인젝션 성공 기준·CI 필수 check/A-B 추론·Phoenix client/집계·manifest/사고 라벨의 미확인 처리에 후속 검토가 필요함을 발견했다. 아직 수정/실행 검증한 것으로 표시하지 않는다. 실제 모델·사내 gateway·DRM 비율/승인/export·VLM/OCR·BERTScore 가중치·운영 모니터링·Claude 협의·Obsidian 읽기 화면은 미확인이다.

### 최종 부분 검증 결과

수정/신규10개 상대 링크·앵커·첨부 참조 오류0, unique YAML10개와 reviewed_on/type/tags 확인, Obsidian CLI properties10개 통과. 원래17개 경로 존재, 미수정9개와 ai-dt 비Markdown18개(LLMOps study_list 포함)의 SHA-256 보존을 확인했다. Python AST19개·SDK 모의 요청21개와 8검증 묶음을 재실행했고, 비유한 A/B 비교 점수와 음수 token usage도 오류로 처리했다. git diff --check는 통과했다. 읽기 화면 탐색·Phoenix collector/서버 수신·Claude 협의는 미완료다.

## 심화 07~11 추가 검토와 재검증

원래5개를 추가 수정했다. 위 최초 단계의 judge/nDCG/injection/CI 과제는 공식 근거·실행으로 처리했고, 12~15의 Phoenix 집계·Mini Project·manifest/사고 라벨은 대기다. 원래 절·fence 수·경로를 보존했고 실행 코드/첨부를 수정하지 않았다. HERDR_ENV=1 current pane 재확인은 pane_not_found였으며 Claude 협의 결과는 없다. 모델 선택·threshold/조직 정책·완전 중복 통합은 보류했다.

### 실행 결과

- inventory5 originalpaths/heads/fencecounts + AST21
- 07 actualSDK strictschema 1..5/bool/string/null/reference/normalizedendpoints; swappedA/B/TIE/disagreement/invalidfulloutput; metaeval empty unknown NaN
- 08 binaryRecall/precision kslots/MRR/idealDCG missinggold duplicate/noanswerunknown; strictclaim schema/emptyclaims; actualSDK relevance n/embedding indexes/zero
- 09 actualjsonschema valid/invalid/missing/empty; forbidden/emptyrequired/exactsequence; finalstate truefalseunknown & judgeNaN/runs vs passk
- 10 missinganswerable/refusalheuristic/strict0or1judge; leakyrefusal and markerabsent attacknotpass; fakeapproval/latencyNoneNaN/safety truefalseunknown
- 11 manifest/gate finite/boundary tolerance/None vsTrue; SHAstablebucket pct; A/B binary/empty/normalapprox/allzero unknown + z/p formula
- actual Ragas 3metrics EvaluationDataset/SingleTurnSample default 1/1/1
- unsupported claim givesfaithfulness0 notcontextrecall0
- emptyclaims NaN rejected and missingreference/context/input rejected
- actual Langchain adapters plus OpenAI2.54 Async/Sync SDK MockTransport full3metrics; rawtextfloatembeddings; no provider sockets/modelquality

추가 AST21개·SDK3.24 MockTransport 요청38개, Ragas3지표/23generation fixture와 실제 legacy adapter/SDK2.54 요청8개 통과. 실제 모델 출력·품질/한국어 타당성·원격 인증·DB/도구 상태·CI/사용자 트래픽 검증은 아니다.

Ragas 환경을 검증하며 v1 community에서 vertexai import 실패, legacy 조합에서 Pillow 누락, Python3.14/nest_asyncio timeout 실행 실패를 발견했다. 기존 Python3.12.12와 Ragas0.3.1/LC0.3.30/community0.3.31/core0.3.86/openai integration0.3.35/SDK2.54.0/Pillow12.3.0 별도 환경에서 실제 metrics와 async/SDK 모의 실행을 확인했다. NaN 무주장은 미확인 오류로 재검증했다. 보조 검증 스크립트의 else0 문법 오타도 고쳐 실행했으며 문서 실패를 성공으로 표시하지 않았다.

### 현재 검증 3

수정/신규15개 YAML·상대 참조·CLI를 아래 결과로 확정한다. 읽기 화면은 이전 getApp124초 timeout으로 아직 검증하지 못했다. 다른 vault/pane을 제어하지 않았다. 목표 전체 완료로 계산하지 않는다.

### 심화 단계 최종 부분 검증 결과

수정/신규15개 상대 링크·앵커·첨부 참조 오류0, unique YAML15개/Obsidian CLI properties15개 통과. 원래17개 경로와 미수정12~15의4개·비Markdown18개 SHA-256 보존, 추가5개의 원래 절/fence 수와 AST21개 통과. Ragas3.12 전체3지표/async·실제 adapter/SDK MockTransport도 재검증했다. README/계측 설치 안내 bash2개와 CI YAML fragment1개 파싱 및 git diff --check도 통과했다. 실제 CI 실행은 아니다. 읽기 화면은 미완료다.


## 운영 12~15 추가 검토

원래4개 경로와 고유 예제 역할을 보존했다. 12의 `Phoenix Embeddings 탭 활용 (권장)` 제목만 `(조건부)`로 바꿨다. 실제 탭·회사 채택을 확인하지 못했기 때문이다. 나머지 코드 밖 제목과 fence 수는 보존했다. 11 gate 주석은 품질 회귀 검사와 인증된 승인/배포 권한을 구분하도록 보완했다. 이동/삭제/코드 파일 변경·커밋·푸시는 없다.

### 검증 1 — 목록과 내용

원래17개와 study_list의6영역15항목은 모두 남아 있다. 요청 지표·embedding drift proxy·실패 후보·dashboard, 조립 실습/scorer, 계보/위험 register/전체 rollback, 사고 YAML/postmortem/candidate의 고유 맥락을 유지했다. 운영의 공통 접속·출처는 적용 조건으로 모았고 완전 중복 통합은 Claude 연결 실패로 보류했다.

### 검증 2 — 근거와 실행

공식 Phoenix/Numpy/NIST/Google SRE 자료와 실제 설치 소스를 대조했다. 추가 AST14, Phoenix client3.5.0 HTTP MockTransport6요청, 요청 집계 누락/중복·p50/p95·잘못된 시간·점수, pointer 실패 후보/unknown 경보/동일 중심 다른 분포, 실제 계측기 초기화(register mock), 조립 JSONL/schema/callback/미확인·적용 제외, manifest YAML/전체HEAD/scoped dirty/draft/기존 파일 보호/NaN 사전 직렬화 실패, 사고 후보 unknown·raw 입력 미복사·검수 전 평가셋 거부를 확인했다. 파일 생성은 임시 디렉터리에서만 했다.

보조 테스트의 custom HTTP client에 기본 인증 header가 없던 문제와 from-import 이후 잘못된 register patch 대상을 고쳤다. 후자는 한 번 provider 초기화만 수행했고 span/모델 호출·원격 export 검증으로 계산하지 않았다. 최종 검증은 올바른 namespace를 mock 처리했다. 실제 서버 인증·collector·운영 경보·회사 승인·실제 judge/모델 품질은 미확인이다.

### 검증 3 — 전체 LLMOps

전체19개(원래17+신규2)의 링크·앵커·첨부·중복 YAML key·review metadata와 대상 vault CLI properties를 최종 대조한다. 원래17경로와 ai-dt 비Markdown18개 보존, Python AST54(19+21+14), bash2/CI YAML fragment 및 운영 YAML2의 구문 검증을 확인한다. 실제 workflow 실행과 화면 탐색/렌더링은 검증하지 않았다. 이전 getApp124초 timeout 이후 읽기 화면은 미완료이며 다른 vault를 수정하지 않았다. Claude 전용 pane은 확보하지 못해 협의 의견이 없다. 운영 threshold/정책·도구 선택·완전 중복 통합은 보류다.


### 전체 최종 부분 검증 결과

전체19개의 상대 링크·앵커·첨부 오류0, 중복 없는 YAML19개/검토일·type·tags/Obsidian CLI properties19개 통과. 원래17개 경로와 ai-dt 비Markdown18개 SHA-256 보존, 세 단계 원래 코드 밖 제목·fence 비교(12 제목 조건부 변경은 위 기록), Python AST54·bash2·YAML3 파싱 통과. 최초21·심화38 SDK 모의 요청과 운영6 Phoenix 모의 요청 및 실패 입력 검증을 재실행했다. git diff --check 통과. Ragas의 별도 Python3.12 검증 결과는 심화 단계 기록을 따른다. 원문17개 모두 개별 검토했지만 실제 서비스/모델/승인·Claude 협의·Obsidian 읽기 화면은 미완료이므로 review_status는 partial이다.
