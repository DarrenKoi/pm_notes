---
tags: [harness-engineering, production, checklist]
level: intermediate
last_updated: 2026-09-11
---

# 10. 프로덕션 체크리스트

> 01~09의 내용을 배포 전 점검표로 압축했다. "반드시"는 없으면 배포하지 않을
> 항목이고, "운영하며"는 첫 한두 달 안에 채울 항목이다.

## 왜 필요한가? (Why)

데모에서 잘 되던 에이전트가 운영에서 무너지는 이유는 대부분 정해져 있다. 무한
루프, 컨텍스트 폭발, 권한 과다, 재현 불가능한 실패, 모르게 늘어난 비용이다.
체크리스트는 이 뻔한 실패를 배포 전에 걸러내기 위한 것이다.

## 핵심 개념 (What)

### 반드시 (배포 전)

**루프** ([02](./02-agent-loop.md))
- [ ] 최대 턴 수, 실행 단위 토큰·시간 예산이 코드로 강제된다
- [ ] 도구 에러와 잘못된 인자가 예외로 루프를 죽이지 않고 교정 메시지로 돌아간다
- [ ] 가짜 클라이언트로 종료 조건 테스트가 있다

**검증과 평가** ([05](./05-verification-and-evals.md))
- [ ] 실제 업무 기반 태스크가 최소 20개 있고, 태스크당 여러 번 실행해 성공률을
      잰다
- [ ] 판정은 가능하면 응답 문구가 아니라 최종 상태를 확인한다
- [ ] "완료" 선언 뒤 하네스가 검증을 실행한다
- [ ] 현재 하네스 버전의 eval 기준선 점수가 기록돼 있다

**안전** ([06](./06-guardrails-and-permissions.md))
- [ ] 치명적 삼중주(사적 데이터 + 비신뢰 입력 + 외부 통신) 중 최소 하나가 끊겨
      있다
- [ ] 코드 실행은 샌드박스(파일시스템 + 네트워크 격리) 안에서만 일어난다
- [ ] 에이전트 전용 계정이 최소 권한으로 동작하고, 비밀값이 컨텍스트에 들어가지
      않는다
- [ ] 되돌릴 수 없는 행동은 사람 승인을 거친다. 목록에 없는 도구의 기본값은
      거부다

**관측** ([08](./08-observability-and-cost.md))
- [ ] 실행마다 전체 트레이스(모델 입출력, 도구 호출, 토큰, 종료 이유)가 남는다
- [ ] 트레이스에 하네스 버전이 기록된다
- [ ] 실행 중인 에이전트를 즉시 멈출 수단(kill switch)이 있다

### 운영하며 (첫 1~2개월)

**컨텍스트** ([03](./03-context-engineering.md))
- [ ] 시스템 프롬프트 앞부분이 고정돼 캐시가 적중한다. 적중률을 지표로 본다
- [ ] 긴 작업에 도구 결과 지우기나 compaction이 적용되고, 그 효과를 eval로
      확인했다
- [ ] AGENTS.md나 시스템 프롬프트가 목차 수준이고 세부 내용은 필요할 때 읽힌다

**도구** ([04](./04-tool-design.md))
- [ ] 도구 결과에 limit, 페이지네이션, 잘림 안내가 있다
- [ ] 모든 도구에 부수효과 등급(read-only / reversible / irreversible)이 붙어
      있다
- [ ] 도구별 에러율을 보고, 가장 높은 도구의 설명과 반환값을 개선했다

**상태와 복구** ([07](./07-state-and-recovery.md))
- [ ] 매 턴 체크포인트가 저장되고, 프로세스를 죽였다 살려도 이어서 진행된다
- [ ] 외부 부수효과가 있는 도구에 멱등성 키가 있다
- [ ] 승인 대기 중에는 프로세스가 종료되고, 승인 후 재개된다

**운영 루틴**
- [ ] 주 1회 실패 트랜스크립트 10건을 읽고, 반복 실패는 eval 태스크로 추가한다
- [ ] 같은 실수가 두 번 나오면 프롬프트, 도구, 검사 중 하나로 구조적으로 막는다
- [ ] 모델을 교체하거나 버전을 올릴 때 eval을 다시 돌리고, 불필요해진 하네스
      장치를 걷어낸다
- [ ] 성공 1건당 비용과 p95 토큰 사용량을 주간 단위로 본다

## 어떻게 사용하는가? (How)

### 위험 신호 (Red Flags)

아래 말이 나오면 해당 문서로 돌아간다.

| 이런 말이 나오면 | 빠진 것 | 문서 |
|---|---|---|
| "프롬프트 고쳤더니 좋아진 것 같아요" | eval | [05](./05-verification-and-evals.md) |
| "시스템 프롬프트에 하지 말라고 써놨어요" | 기계적 강제 (권한, 샌드박스) | [06](./06-guardrails-and-permissions.md) |
| "가끔 이상하게 동작하는데 재현이 안 돼요" | 트레이스, 하네스 버전 | [08](./08-observability-and-cost.md) |
| "긴 작업은 후반에 엉뚱한 걸 해요" | 컨텍스트 관리 | [03](./03-context-engineering.md) |
| "중간에 죽으면 처음부터 다시 돌려요" | 체크포인트, 멱등성 | [07](./07-state-and-recovery.md) |
| "역할별로 에이전트 5개를 만들었어요" | 분할 근거 측정 | [09](./09-multi-agent.md) |
| "도구를 40개 붙였어요" | 도구 통합, 점진적 공개 | [04](./04-tool-design.md) |

### 사용법

1. 새 에이전트를 설계할 때 "반드시" 섹션을 요구사항으로 쓴다
2. 배포 리뷰에서 체크되지 않은 항목은 담당자와 기한을 붙인다
3. 분기마다 "운영하며" 섹션을 다시 점검한다

## 참고 자료 (References)

- [Harness engineering](https://openai.com/index/harness-engineering/) — OpenAI
- [Harness engineering for coding agent
  users](https://martinfowler.com/articles/harness-engineering.html) — Birgitta
  Böckeler
- [12-Factor Agents](https://github.com/humanlayer/12-factor-agents) —
  HumanLayer
- [Demystifying evals for AI
  agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
  — Anthropic
- [OWASP Top 10 for Agentic Applications for
  2026](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/)
  — OWASP

## 관련 문서

- [README — 학습 순서와 우선순위](./README.md)
- [01. 핵심 개념](./01-core-concept.md)
