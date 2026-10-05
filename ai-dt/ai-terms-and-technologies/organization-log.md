---
type: review-log
tags: [ai, documentation-review]
reviewed_on: 2026-10-04
review_status: partial
category_major: "AI·DT"
category_middle: "문서 관리"
category_minor: "정리·검증 기록"
note_kind: "관리 기록"
classified_on: "2026-10-05"
---

# AI 입문 문서 정리 기록

## 범위와 분류

원래 Markdown 11개를 전체 읽고 같은 경로에 유지했다. 원래 last_updated 2026-07-28을 보존하고 확인일은 별도로 기록했다. 실행 코드·첨부는 없고 다른 주제로 이동·삭제·내용 통합·신규 cross link는 없다.

01~09는 상세 예제/점검표의 대표 문서, all-in-one은 처음 연속해서 읽는 요약, README는 목차/읽기 순서다. 유사 문서의 목적 차이를 명시하고 통합본의 9개 섹션에서 상세 문서로 연결했다. 반복 출처·GPU/토큰/평가 가정은 [근거와 적용 조건](./verified-conditions.md)에 모았다. 고유 문장·도식·사내 설명 사례는 원래 역할을 유지하고 오류/보장 표현만 조건부로 수정했다.

| 원래 문서 | 개별 검토 결과 |
|---|---|
| [README](./README.md) | 9개 목차/읽기 순서 유지. 개별 문서와 요약의 목적 구분, 최근 사례를 2026년 6~7월 사례로 한정 |
| [01 모델](./01-ai-model-families.md) | H200 141GB 공식 사양 확인. 140GB/80%는 decimal 단순 가정, GPU 표는 실제 fit/처리량 아님. MoE 전체 weights 상주/offload와 활성 연산/속도 구분. reasoning·증류 효과 보장 표현 한정 |
| [02 검색·온톨로지](./02-vector-embedding-and-rag.md) | RAG가 필수·오류 대부분 검색 때문이라는 일반화 수정. chunk 전체 포함 필수 대신 맥락 복구 방법 보완. reranker 후보/추가 latency·평가 조건, 성능 가산식을 비유로 구분. 문서 권한/유효성·온톨로지 예제 유지 |
| [03 prompting](./03-zero-shot-and-few-shot.md) | zero/one/few-shot 3개 고객 예제·제조 사례 유지. 이 노트의 in-context prompting과 다른 few-shot learning 범위 구분 |
| [04 에이전트](./04-vibe-coding-harness-agent-orchestration.md) | 모델 생성/하네스 실행 구분 유지. 5×3 연동은 개별 구성 가정, MCP가 모든 호환/시스템 구현을 없애지 않음. 하네스 4개 공식 자료 확인, 능력 향상 시계열 미측정 표시 |
| [05 리터러시](./05-ai-literacy-bias-and-slop.md) | 다섯 능력·작성 절차는 학습 정리. 해로운 사회/운영 bias와 통계/inductive bias 의미를 구분. 장비/집단별 평가와 민감 속성 대리 정보 조건 유지 |
| [06 보안](./06-jailbreak-and-ai-security.md) | OWASP의 extraction 시도와 실제 유출 결과 구분. 정부 지시는 제공자 성명의 외국 국적자 범위로 한정. 6~7월 발표를 원문과 대조하고 7월 28일 OpenAI prototype 조치를 별도 추가. 당시 기록을 현재 접근권/법적 사실로 덮어쓰지 않음 |
| [07 AX/인프라](./07-ax-and-ai-data-centers.md) | AX/다섯 범주는 노트의 정리이며 표준 단계가 아님. IEA 2025 보고서 링크/확인일 추가; 시설별 전력/냉각·시나리오와 현재 모든 시설 수치를 구분 |
| [08 토큰/환각](./08-token-context-window-and-hallucination.md) | 한국어 항상 더 많음/영어 학습량 원인 단정 제거. 토큰 분할은 가상 예시. 한도·실제 input 누락·긴 입력 활용 구분. sampling/greedy·API 지원/과금·정확성 조건 보완 |
| [09 선택/평가](./09-prompt-rag-finetuning-and-evaluation.md) | fine-tuning도 지식/행동을 학습, weights는 ACL 집행 아님. LoRA 적용 layer 고정·저랭크 학습 구분. 30~100건은 초기 계획이며 보장 아님, 독립 holdout/회귀 범위 보완. 사내 과제 대부분 해결·6개월 임계값 근거 미확인 |
| [통합본](./all-in-one.md) | 9개 상세 문서에 연결하고 같은 기술 조건 정정. 연속 읽기 도식·비교·최종 원칙 유지. 상세 예제와 같은 역할로 중복 확장하지 않음 |

## 근거와 협의

공통 근거에 공식 자료·원논문 27건의 범위와 확인일을 연결했다. 기능 문서는 동적 자료이며 로컬 제품 버전·모든 계정에서 재현했다고 주장하지 않는다. 사건 자료는 각 조직의 당시 공개 입장/조치이고 정부 원문·최종 조사·독립 원인 판정을 대신하지 않는다.

HERDR_ENV=1에서 herdr pane current --current가 pane_not_found다. 전용 Claude 의견을 얻지 못했고 다른 pane은 제어하지 않았다. 통합본 전면 폐지/전면 합병·교육 범주 재설계·사건 원인의 독립 판정은 협의/추가 근거 대기로 보류했다. 공개 근거로 확정할 수 있는 설명 조건·날짜·참조는 수정했다.

## 세 차례 검증

1. **목록·내용:** 원문 snapshot 11개 경로, 절 제목·text fence 목록을 대조한다. 조건부로 바꾼 제목은 02의 RAG 필수, 04의 현재 하네스, 06의 최근 타임라인, 07의 다섯 구성, 08의 한국어 일반화이며 고유 예제와 나머지 절은 유지한다. 현재 근거/기록을 포함해 13개다.
2. **근거·로컬:** 공식 H200 141GB, 연구 조건과 하네스 지원, 당시 사건/업데이트를 확인했다. 본문 가정에서 3B/70B/400B의 BF16/4-bit weights와 GPU 개수 ceil을 수치로 재검산한다. 실행 언어 코드는 없고 text 예시는 도식·가상 prompt이다. GPU·tokenizer·모델/RAG·judge 실험·위험한 보안 payload는 실행하지 않는다.
3. **링크·메타데이터·Obsidian:** YAML·상대 링크/anchor·CLI properties와 읽기 탐색 결과를 최종 확인에 기록한다.

## 남은 미확인

실제 모델/학습·검색/평가 품질, GPU fit·처리량·권한·토큰 실측, 사내 적용과 정량 효과, 정부 원문·현재 법적 상태·후속 최종 사건 분석, Claude 협의는 미확인이다. 기존 형제 주제 참조는 새로 만들지 않고 유지한다.

## 최종 확인

원문 11개 경로가 모두 남아 있다. 원래 절 215개 중 조건부 설명으로 수정한 제목 5개를 위에 기록했고 다른 절 제목을 유지했다. text fence 40개를 모두 유지했고 6개는 가중치 예산·MoE 상주 가정·비산술 성능 도식·온도 조건 표현을 수정했다. 나머지 34개 fence는 원문 그대로다. 원래 날짜·YAML 중복 key/검토일 13개가 통과했다. H200 가정의 3B/70B/400B BF16·4-bit 용량/ceil GPU 장수가 각각 (1,1), (2,1), (8,2)로 재검산됐다. ai-dt 원래 non-Markdown 18개 해시는 초기 snapshot과 동일하다.

링크 검사 신규 문제 0개이고 기존 형제 참조 5개는 대상이 존재하는 기존 cross link로 유지했다. 확인된 pm_notes vault에서 13개 CLI properties가 통과했다. 읽기 화면의 README에서 원래 날짜·검토일·한국어 callout·9개 읽기 순서 표를 확인했다. 첫 링크 클릭은 화면 변화만 있어서 다시 최신 상태로 링크를 눌렀고, 적용 조건의 올바른 ai-dt/ai-terms-and-technologies 경로·한국어 alias·27개 근거 표가 노출됨을 확인했다. 모든 아래쪽 화면·모든 표 링크 클릭·실제 모델 실행을 검증한 것은 아니다.

최종 기록 이후 내용/metadata·링크·diff 공백 검사를 재실행한다. 커밋·push는 하지 않았다.
