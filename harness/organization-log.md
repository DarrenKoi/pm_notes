---
tags: [harness-engineering, organization-log]
reviewed_on: 2026-10-04
review_status: partial
document_type: maintenance
---

# harness 정리 기록 — 2026-10-04

## 범위와 결정

기존 Markdown 21개(공통 학습 12, 사용자 작성 중인 Qwen 9)를 개별 검토했다.
삭제·이동·폴더 간 통합 없이 공통 문서 12개에 검토 메타데이터/안내를 추가했다.
신규 3개는 현재 적용 조건, Qwen 원문 보충, 이 기록이다. 코드·첨부는 수정하지 않았다.
고유 실습과 기존 참고 자료는 유지했다. 10은 앞 장의 점검용 요약이므로 중복 삭제하지
않고, 11은 발표 시점에 따른 사례라는 용도를 유지한다. 과거 last_updated는 남기고
이번 reviewed_on을 별도로 기록했다. 새 링크는 harness 안에서만 연결한다.

Qwen 9개는 작업 시작 시 untracked 사용자 변경이었다. 직접 덮어쓰지 않고
[보충 문서](./qwen3.8/review-notes.md)에 검토 결과를 남겼다. 내용의 진위 확인과
기존 변경 보호를 동시에 충족하기 위한 결정이다. 03·04·07의 경계값 수정은
로컬에서 실패 시나리오를 재현할 수 있는 단순 오류로 판단했다.

## 근거와 확인 범위

[현재 적용 조건](./review-notes.md)의 공식 MCP/Tasks·LangGraph·OpenTelemetry·Anthropic
자료와 [Qwen 공식 카드 대조](./qwen3.8/review-notes.md)를 사용했다.
확인일은 2026-10-04. main 소스·웹 모델 카드는 버전 불변 자료가 아니므로
릴리스 지원과 같은 것으로 기록하지 않는다. 전체 과거 참고 자료 재검증과
논문/성능 수치 재현은 미완료다.

## Claude 협의 결과

HERDR_ENV=1은 확인했으나 `herdr pane current --current`가 pane_not_found를 반환했다.
현재 caller를 찾을 수 없어 작업 전용 Claude pane을 만들거나 연결하지 않았다.
다른 주제의 기존 Claude pane은 제어하지 않았다. Claude 의견은 받지 못했다.
모델별 파서·SDK 전체 호환·논문 일반화·더 큰 분할/통합 판단은 보류한다.
확정된 로컬 오류·원문 보호·출처 연결·목차 개선은 독립적으로 진행했다.

## 원래 문서별 결과

| 문서 | 결과와 한계 |
|---|---|
| [README.md](./README.md) | 기존 01~11 읽기 순서 유지; 현재 적용 조건과 Qwen 보충을 연결. 과거 벤치마크 순위는 당시 결과로 보존. |
| [01-core-concept.md](./01-core-concept.md) | 개념·구성·고유 예제 유지. 모델 입출력의 단순화와 외부 사례 수치의 일반화는 보충 문서에서 적용 한계 명시. |
| [02-agent-loop.md](./02-agent-loop.md) | SDK 재시도가 충분하다는 주석 수정. 가짜 클라이언트의 잘못된 도구/JSON과 최대 턴 종료 실행 통과. 실 API는 미실행. |
| [03-context-engineering.md](./03-context-engineering.md) | keep_last=0 버그 수정, 음수/불리언 거부. 연구 범위와 캐시 구현 조건 구분. compaction 모델 호출은 미실행. |
| [04-tool-design.md](./04-tool-design.md) | Schema와 페이지 함수의 cursor/limit 범위 일치, 끝 페이지 안내 추가. 경계값 실습 통과; 실제 검색 서버는 미실행. |
| [05-verification-and-evals.md](./05-verification-and-evals.md) | 조합식 실습 통과; 공식 Anthropic 평가 개념 대조. judge 보정과 운영 표본 검증은 미완료. |
| [06-guardrails-and-permissions.md](./06-guardrails-and-permissions.md) | 권한 예제·Docker 명령 보존. 문자열 차단과 실제 경계 차이 확인. Docker/보안 격리 실행은 미완료. |
| [07-state-and-recovery.md](./07-state-and-recovery.md) | 재시도 시도 횟수 검증 추가; SDK 충분성 일반화 수정. 체크포인트 왕복·일시 오류·멱등성 예제 통과. 실 외부 장애는 미실행. |
| [08-observability-and-cost.md](./08-observability-and-cost.md) | JSONL·예산 예제 보존. 공식 OTel 문서 이동 확인. 트레이스 성공/예외와 집계 실습 통과; OTel 수집은 미실행. |
| [09-multi-agent.md](./09-multi-agent.md) | 격리/상속 적용 조건과 위임 예제 유지. 공유 파일 권한 격리 구현과 성능 재현은 미확인. |
| [10-production-checklist.md](./10-production-checklist.md) | 앞 장의 적용 점검이라는 다른 용도여서 유지. 표본 수·시점 권고는 프로젝트별 제안임을 보충. |
| [11-current-trends.md](./11-current-trends.md) | 2026-07-28 코어/Tasks와 SDK receiver 세대 차이 재확인. 09-12 원래 확인 기록 유지. 다른 모든 발표/로드맵의 재검증은 미완료. |
| [qwen3.8/README.md](./qwen3.8/README.md) | 원문 보존. 사용자 학습 목차의 실제 사내 배포 모델과 최신 상태는 미확인. |
| [qwen3.8/01-model-overview.md](./qwen3.8/01-model-overview.md) | 원문 보존. 공식 FP8 카드의 컨텍스트·벤치마크 하네스 차이를 review-notes에 기록. 라이선스 해석은 보류. |
| [qwen3.8/02-sampling-and-modes.md](./qwen3.8/02-sampling-and-modes.md) | 원문 보존. thinking/instruct의 temperature와 presence penalty 차이를 공식 카드로 보충. |
| [qwen3.8/03-serving-setup.md](./qwen3.8/03-serving-setup.md) | 원문 보존. parser 자리표시자·엔진별 설정과 처리량/양자화 보장을 미확인으로 기록. |
| [qwen3.8/04-prompting-playbook.md](./qwen3.8/04-prompting-playbook.md) | 원문 보존. few-shot 개수는 실험 시작점, Schema는 사실성 보장이 아님을 보충. |
| [qwen3.8/05-tool-calling-agentic.md](./qwen3.8/05-tool-calling-agentic.md) | 원문 보존. API별 tool_choice와 파서 계약 혼동 가능성 기록; 확정 엔진 설정은 협의 보류. |
| [qwen3.8/06-rag-long-context.md](./qwen3.8/06-rag-long-context.md) | 원문 보존. 카드의 native context/YaRN 조건 보충. 타 모델 논문의 일반화는 협의 보류. |
| [qwen3.8/07-limits-and-mitigation.md](./qwen3.8/07-limits-and-mitigation.md) | 원문 보존. 모드별 penalty와 처리량 일반화 미확인 보충. |
| [qwen3.8/08-checklist.md](./qwen3.8/08-checklist.md) | 원문 보존. keep-alive 등 환경별 조건, 운영 실측 미완료 보충. |

## 신규 문서 검토

- [현재 적용 조건](./review-notes.md): 확인한 사양과 실습/미확인 운영 범위를 분리했다.
- [Qwen 보충](./qwen3.8/review-notes.md): 공식 FP8 카드 적용 조건과 사용자 원문 차이를 구분했다.
- 이 기록: 문서 수·변경 경계·협의 부재를 기록했으며 저장소 전체 완료 기록이 아니다.

## 검증 1 — 목록과 고유 내용

21개 원래 경로가 모두 남아 있다. Qwen 9개는 시작 시 텍스트/해시와 동일하다.
공통 문서의 예제·체크리스트·참고 자료는 그대로 두고 오류와 일반화만 한정 수정했다.
새로 통합하거나 이동한 문서가 없어 다른 문서의 참조 갱신은 필요하지 않았다.
비 Markdown 파일이 시작 시 해시와 같아야 폴더 검증을 통과한다.

## 검증 2 — 기술과 로컬 예제

임시 폴더에서 모델 없이 다음을 실행했다: 02 대본 루프·잘못된 JSON·최대 턴,
03 결과 치환 0/1/3/6·음수·불리언, 04 페이지 0/마지막/빈 결과·잘못된 범위,
05 조합식, 07 한국어 체크포인트 왕복·일시 오류 재시도·0회 거부·멱등성,
08 JSONL 성공/예외 트레이스·토큰 집계. 모두 통과했고 Python 코드 블록 20개를
AST 파싱했다. 실제 모델·GPU·도구 서버·Docker·외부 쓰기·OTel collector는 실행하지 않았다.
검사 스크립트는 /private/tmp에 두어 운영 코드/새 테스트 모듈을 추가하지 않았다.

## 검증 3 — 링크·메타데이터·앱

폴더 링크 검사·Obsidian CLI/읽기 화면 결과는 아래 최종 확인에 추가한다.

## 남은 미확인

전체 참고 자료의 당시/현재 차이, 모든 SDK 지원 버전, 논문 및 수치 재현,
실 운영 격리·장애 복구와 Claude 협의가 남아 있다. 이 폴더는 구조/로컬 검증을
마친 뒤에도 review_status=partial로 유지하며 미확인을 최신 사실로 바꾸지 않는다.

## 최종 확인 — 2026-10-04

- 목록 21 → 24, 원래 경로 누락 0, 보호 Qwen 원문 차이 0, 비 Markdown 변경 0.
- 링크·앵커 검사 24개: 새 오류 0, 기존 오류 0. git diff --check 통과.
- Obsidian 1.13.7, 대상 pm_notes: CLI properties 24개 성공. 편집/신규 15개 검토일
  정상, 보호 원문 9개는 frontmatter 없는 상태를 그대로 확인했다.
- 실제 읽기 화면에서 harness/README → harness/review-notes 링크를 클릭했고
  정확한 폴더·한국어 본문·날짜·태그·alias·표를 확인했다. 전체 24개 화면을 하나씩
  육안 검증한 것은 아니다. 초기 display name 연결 실패는 앱 ID 재연결로 회복했다.
- 로컬 실습과 문법 검증 결과는 위 검증 2와 같다. 외부 운영·Claude 협의는 미완료다.
