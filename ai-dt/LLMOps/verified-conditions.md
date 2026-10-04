---
tags: [llmops, evaluation, review]
reviewed_on: 2026-10-04
review_status: partial
document_type: reference
---

# 공통 적용 조건과 근거

## 목적과 읽기 순서

평가는 “좋아졌다”는 주장을 정해진 입력·rubric·실행 조건으로 비교하기 위한 루프다. [README](./README.md)의 01→02→03으로 운영 자산/관측성을, 04→05→06으로 지표/데이터/하네스를 익힌다. 07~11은 judge·RAG·agent·안전·배포의 심화 평가다. 12~15는 관측·조립 실습·계보·사고 후보 검수를 다룬다. 같은 설명을 완전히 중복 통합하거나 재설계하는 결정은 Claude 협의 연결 실패로 보류했다. 공통 접속·판본·검증 경계만 여기로 모았다.

## 실행 전제

문서의 Python 블록은 위에서부터 같은 namespace에서 실행한다. 다른 문서 함수는 자동 import되지 않는다. 06은 README의 client/embed/EMBED_MODEL과 앞선 정규화 함수가 필요하며 04 하네스에는 system_fn·scorers·dataset을 전달한다. prompts/파일과 eval_set.jsonl은 예시 입력으로 실습자가 준비한다. 실행 모듈을 새로 만들거나 기존 실행 코드를 수정하지 않았다.

| 환경변수 | 용도 |
|---|---|
| LLM_BASE_URL, LLM_API_KEY | 승인된 chat/embedding gateway와 인증. 값을 출력/커밋하지 않음 |
| LLM_MODEL, JUDGE_MODEL | 대상 모델과 판정 모델의 서빙 id. 같은 모델이면 독립 judge라는 가정 금지 |
| EMBEDDING_MODEL | dense embedding id. 별도 gateway라면 client도 분리 |
| VLM_MODEL | 허용된 이미지 처리의 vision id |
| PHOENIX_COLLECTOR_ENDPOINT | collector의 실제 프로토콜/인증/수집 주소 |
| PHOENIX_ENDPOINT, PHOENIX_API_KEY, PHOENIX_PROJECT | 12의 승인된 Phoenix REST 접속·인증·project 식별자. collector 주소와 별개 |
| BERTSCORE_MODEL_PATH, BERTSCORE_NUM_LAYERS | 선택 BERTScore의 로컬 모델 snapshot과 평가 layer 수 |

원래 Kimi-K2.5/Qwen3-VL-30B/BGE-M3는 사내 alias 예시이며 실제 모델·가용성·권한은 미확인이다. base URL 교체만으로 tool/JSON/vision/embedding/usage 호환이 보장되지 않는다. 외부 API 차단·DRM 99%·사내 Phoenix 채택은 원래 시나리오의 미확인 주장이다. 온도 0은 완전한 결정성 보장이 아니며 모델/서빙 판본과 반복 분산을 기록한다.

## 확인 판본과 지표 계약

2026-10-04 별도 임시 Python3.14.2 환경: openai3.24.0, numpy2.5.3, rouge-score0.1.2, sacrebleu2.6.0, arize-phoenix-otel0.17.2, arize-phoenix-client3.5.0, openinference-instrumentation0.1.70/openai0.1.63, opentelemetry-sdk1.45.0, httpx2.13.1, pandas3.0.6. 확인 판본은 최신이라는 뜻이나 사내 호환 lock이 아니다. Phoenix server·BERTScore 가중치·실제 endpoint는 실행하지 않았다. Ragas0.3.1은 별도 Python3.12.12 환경에서 legacy adapter·실제 지표 실행을 검증했다. 기본 LangChain v1 환경에서는 제거된 vertexai import, Python3.14/legacy 환경에서는 nest_asyncio의 task timeout 오류가 발생했다. 확인된 조합은 langchain0.3.30/community0.3.31/core0.3.86/openai integration0.3.35, OpenAI SDK2.54.0, Pillow12.3.0이다. rolling stable 문서는 newer factory를 설명하므로 0.3.1 API 계약과 혼용하지 않는다. 이 예제용 구판 조합의 유지보수/보안·운영 채택 여부는 따로 검토한다.

- 정규화는 NFKC/casefold/공백만 사용하고 소수점·부호·단위를 보존한다. 대소문자가 중요한 분류에는 계약을 바꾼다. 빈 reference는 0점/통과가 아닌 미확인 오류다.
- contains는 부분문자열 포함률이고 사실성 지표가 아니다. ROUGE-L 예제는 공백 토큰의 F1이다. BLEU는 intl/exp smoothing/4-gram 설정을 기록하고 0~100을 0~1로 환산한다. 이 한국어 토큰화는 형태소 기반 성능 평가와 다르다.
- cosine은 [-1,1]이고 정답 확률이 아니다. 영벡터/누락/비유한/불균일 차원은 점수 미확인으로 실패시킨다. 0.75 임계값은 예시이며 calibration과 최종 holdout을 구분한다.
- contexts는 해당 실행의 실제 검색 문맥이다. gold_contexts/gold_ids는 정답 근거로 별도 저장한다. 합성 QA는 검수·id/category/source 부여 전까지 golden이 아니다. VLM 추출은 숫자/표/단위/페이지를 원문과 대조한다.
- JSON 타이머 로그는 OTel span이 아니다. 예제의 실제 OTel parent-child 구조와 SDK span은 로컬 exporter에서 확인했다. TraceConfig는 raw input/output·호출 파라미터·도구/embedding을 숨기며 직접 추가한 attribute와 예외의 민감정보는 별도 관리해야 한다. 비용 단가/usage가 없으면 null이며 무료/0토큰으로 간주하지 않는다.

## 공식·일차 자료

확인일 모두 2026-10-04. rolling/main 자료는 설치 판본·실제 동작과 별도로 대조했다.

| 자료 | 확인 내용 |
|---|---|
| [Python string](https://docs.python.org/3/library/string.html) | safe_substitute 누락 변수, format 리터럴 중괄호; 템플릿 검증에 적용 |
| [Phoenix OTel setup](https://arize.com/docs/phoenix/tracing/how-to-tracing/setup-tracing/setup-using-phoenix-otel) | SDK register/계측/collector와 server 구성의 구분 |
| [OpenInference TraceConfig 구현](https://raw.githubusercontent.com/Arize-ai/openinference/main/python/openinference-instrumentation/src/openinference/instrumentation/config.py) | 실제 설치 클래스의 원문 숨김 옵션과 대조 |
| [OTel GenAI span 명세](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md) | Development 상태, request/response model·usage·prompt version·stream 필드. OpenInference 실제 속성과 자동 동일시하지 않음 |
| [Ragas metrics](https://docs.ragas.io/en/stable/concepts/metrics/), [Faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/) | 입력 질문·응답·검색 문맥과 검증 reference의 역할 |
| [ROUGE 공식 구현](https://github.com/google-research/google-research/tree/master/rouge) | tokenizer 교체와 precision/recall/F1 |
| [SacreBLEU 공식 구현](https://github.com/mjpost/sacrebleu) | tokenizer/smoothing/signature·한국어 선택 의존성 |
| [Evaluate 공식 구현](https://github.com/huggingface/evaluate) | 지표 load 경로와 로컬 직접 계산의 구분 |
| [BERTScore 논문](https://arxiv.org/abs/1904.09675), [공식 구현](https://github.com/Tiiiger/bert_score) | token alignment·모델/layer 조건; 사실성 보증으로 쓰지 않음 |
| [BGE-M3 모델 카드](https://huggingface.co/BAAI/bge-m3) | 공개 임베딩 모델의 설명; 실제 사내 alias 성능 증거 아님 |
| [MT-Bench 논문](https://arxiv.org/abs/2306.05685) | judge 편향/메타평가가 필요하다는 근거. 07의 schema/교환 순서 로컬 검증과 실제 judge 품질은 구분 |

## 검증 경계

최초 단계의8개 원래 문서 경로·읽기 절·fence 수와 고유 예제 맥락을 보존했다. 추가5개는 아래 심화 검증을 참고한다. Python AST19개, 실제 SDK MockTransport 요청21개 및 프롬프트/하네스/합성 schema/자동 지표/OTel fixture를 확인했다. HTTP 요청 payload 검증은 실제 회사 endpoint 응답·OCR 정확도·모델 품질 검증이 아니다. BERTScore는 문법만 확인했다. Phoenix register의 설치 signature/소스와 privacy option은 대조했지만 수신/인증/flush/서버 UI·운영 모니터링은 미검증이다.

Obsidian pm_notes vault의 실제 경로는 이 저장소로 확인되어 있다. 이번 읽기창 getApp 연결은 timeout(124초)으로 실패했으며 화면 탐색/렌더링은 미완료다. CLI 메타데이터와 상대 링크 검증 결과는 [정리 기록](./organization-log.md)에 따로 남긴다. Herdr current pane은 HERDR_ENV=1에서도 pane_not_found여서 작업 전용 Claude pane을 확보하지 못했다. 협의한 의견은 없으며 재분류/중복 통합·운영 임계값 결정은 보류한다.

## 심화 평가 적용 조건 — 07~11

07의 pointwise correctness에는 검증된 reference가 필요하다. JSON parser/schema와 judge가 적은 이유는 사실성 증거가 아니다. 점수1~5 세 차원을 (합계-3)/12로 환산한다. pairwise의 TIE와 순서 불일치/형식 오류는 다르게 보고하며 응답 첫 글자를 임의로 승자로 받지 않는다. 메타평가 threshold/0.8목표/3회 다수결은 제안값이고 calibration/holdout·독립성 검증이 필요하다.

08의 검색 예제는 binary relevance와 중복 없는 id를 사용한다. Precision@k는 빈 슬롯을 비관련으로 세며 Recall@k는 고유 gold 문서 총수로 나눈다. 한 query의 reciprocal rank를 평균해야 MRR이다. nDCG의 ideal은 실제 검색에서 누락한 정답도 포함한다. gold/주장/필수 입력이 없으면 정상0점이 아닌 미확인이다. 직접 구현한 역질문 cosine은 Ragas ResponseRelevancy의 noncommittal penalty 등을 생략한 근사 예제다. Ragas LLMContextRecall에는 reference가 필요하며 모든 context 지표가 reference-free인 것은 아니다.

09의 schema/도구/경로 점수는 최종 상태 성공과 다르다. state_check는 명시적 boolean을 반환하고 미확인은 오류다. 금지 도구 로그는 이미 일어난 부작용을 방지하지 않는다. 여러 실행의 binary 성공 비율은 pass@k/pass^k가 아니다.

10의 refusal 정규식/PII 스캔은 휴리스틱이다. 거절 문구가 있어도 누출할 수 있고 마커 없는 기밀도 있다. 인젝션 평가는 공격 목표·허용 행동·도구 기록을 확인하는 검증 callback이 필요하다. answerable 누락을 True로 바꾸지 않으며 안전/시간 미확인은 None이고 배포는 is True만 통과시킨다. 로그의 human_approved=True도 인증된 실행 승인을 대신하지 않는다. 3개 공격/0.95·0.9기준은 학습용 제안값이다.

11의 gate는 같은 eval/rubric/judge·coverage의 baseline/current를 비교해야 한다. manifest 필수 항목은 14에서 설명하며 실제 승인 증빙은 운영 구성이 필요하다. YAML은 workflow fragment이고 Python 함수는 로컬 gate 로직이다. CI 실행·exit code와 보호 규칙/필수 check·bypass 조건을 갖추지 않으면 머지가 자동 차단되지 않는다. A/B는 독립 사용자·binary 관측·랜덤화/편향·표본 조건을 확인한다. 작은/퇴화 표본의 유의성은 미확인이다. 양측 차이 검정에서 p≥0.05는 비열등성/안전 증거가 아니며 canary 확대 기준을 따로 정의한다.

### 추가 일차 근거

확인일 모두 2026-10-04.

| 근거 | 적용 |
|---|---|
| [Stanford IR 교재](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html) | 검색 recall·rank·ideal DCG 정의 |
| [Ragas0.3.1 LLM API](https://docs.ragas.io/en/v0.3.1/references/llms/), [evaluate API](https://docs.ragas.io/en/v0.3.1/references/evaluate/) | legacy wrapper·metric 입력·raise_exceptions. NaN도 호출자 확인 |
| [Ragas 모델 사용자화](https://docs.ragas.io/en/stable/howtos/customizations/customize_models/), [ContextRecall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/) | rolling factory 변화와 reference 계약 구분 |
| [jsonschema 공식 API](https://python-jsonschema.readthedocs.io/en/stable/validate/), [τ-bench 원 논문](https://arxiv.org/abs/2406.12045) | instance/schema 오류·최종 상태와 반복 신뢰성 측정 |
| [OWASP 2025](https://genai.owasp.org/llm-top-10/), [Red teaming 원 논문](https://arxiv.org/abs/2202.03286) | 위험 분류와 적대 평가. 제안 gate의 완전성 인증 아님 |
| [NIST 두 비율 검정](https://www.itl.nist.gov/div898/handbook/prc/section3/prc33.htm) | 독립 표본·정규 근사·양측 검정 |
| [GitHub 보호 규칙](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches), [CanaryRelease 저자 설명](https://martinfowler.com/bliki/CanaryRelease.html) | required checks·bypass와 점진 배포/A-B 목적 구분 |

추가 Python AST21·SDK3.24 모의 요청38개를 실행했다. 별도 Ragas 환경의 실제 metrics/async framework·LangChain legacy adapters·SDK2.54 모의 요청8개도 통과했다. 실제 모델·서비스/도구/DB 상태·CI/배포·사용자 트래픽·운영 SLO/안전 정책은 검증하지 않았다.


## 운영 적용 조건 — 12~15

12는 Phoenix client3.5.0의 `Client.spans` API를 사용한다. UTC 조회 구간과 limit을 기록하고, 조회된 span 수를 요청 수로 부르지 않는다. 요청 지표는 request_id별로 root latency·전체 token·feedback을 먼저 결합한 한 행을 입력받는다. 결합 파이프라인은 제공하지 않는다. 미수집 값은 None이며 측정 표본 수를 함께 보고한다. annotation은 실제 span_id/유한0~1 점수가 필요하다. embedding query는 drift 검정이나 UI 탭 검증이 아니다. 중심 거리0도 분포 동일성을 증명하지 않는다. dashboard/SLO·경보 비율은 제안값이다.

13은 scorer callback을 명시적으로 전달하는 조립 예제다. 존재하지 않는 scorer 패키지를 자동 import하지 않는다. reference·gold 누락과 거절에 적용되지 않는 faithfulness를 구분하며 전체/측정/미확인 표본 수를 함께 보고한다. 평균만으로 CI 통과를 판단하지 않는다. leak_scan_hit=1은 의심 패턴 적중이며 안전1점이 아니다. 원문 폴더 구조와 목표50건/0.85/3초는 구현 완료나 실측 결과가 아니다.

14의 manifest는 prompt/model/index/tool/eval/rubric/guardrail 조합과 완전한 이전 release 참조를 기록하는 draft 예제다. 전체 git HEAD와 topic 범위 dirty 상태를 기록해도 미커밋 내용의 snapshot을 대신하지 않는다. 기존 파일은 exclusive 생성으로 보호하며 직렬화 오류를 파일 생성 전에 검사한다. 함수는 승인을 발급하지 않는다. NIST 기능별 대응표는 작성자의 적용 설명이고 인증·회사 정책이 아니다.

15의 사고 예시는 가상 학습 자료다. 사고 입력은 sanitized 필드/참조로 candidate만 만들며 reference와 answerable은 None이다. 출처·사용 권한·정답·실제 문맥·답변 가능성을 사람 검수하기 전 golden으로 승격하지 않는다. SEV/시간 기준·승인 역할도 제안값이다.

### 운영 일차 근거

확인일 모두 2026-10-04. rolling 문서는 실제 설치 소스와 대조했다.

| 근거 | 확인 내용 |
|---|---|
| [Phoenix SDK](https://arize.com/docs/phoenix/sdk-api-reference), [span 추출](https://arize.com/docs/phoenix/tracing/how-to-tracing/importing-and-exporting-traces/extract-data-from-spans), [annotation](https://arize.com/docs/phoenix/tracing/how-to-tracing/feedback-and-annotations/evaluating-phoenix-traces) | client 자원·query·annotation. 3.5.0 설치 소스의 `phoenix.client.types.spans.SpanQuery`와 signature 확인; server/UI 기능은 미확인 |
| [NumPy percentile](https://numpy.org/doc/stable/reference/generated/numpy.percentile.html) | 예제 p50/p95의 linear 방법 명시 |
| [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework), [GenAI Profile](https://doi.org/10.6028/NIST.AI.600-1) | 자발적 위험관리·Govern/Map/Measure/Manage; 예제 조직 승인과 구분 |
| [Google SRE postmortem](https://sre.google/sre-book/postmortem-culture/) | 비난보다 원인·후속 조치·학습을 기록하는 운영 원칙 |

추가 Python AST14·Phoenix HTTP MockTransport6요청, 요청 집계/누락·중복·잘못된 시간/점수, scorer 조립/빈 JSONL, manifest 임시 파일/기존 파일 보존/draft/직렬화 실패, 사고 candidate의 평가셋 편입 거부를 확인했다. OTel 계측기 초기화는 실제 구현을 사용하고 register는 mock 처리했다. 실제 REST 인증·collector 수신·UI·경보·회사 승인·모델 품질은 검증하지 않았다.
