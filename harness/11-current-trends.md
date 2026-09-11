---
tags: [harness-engineering, trends, mcp, context-fork, evals]
level: advanced
last_updated: 2026-09-12
---

# 11. 최신 동향과 적용 판단 — 2026-09-12 확인

> 최근 하네스 설계는 컨텍스트 전달 방식, 실행 환경의 수명, 프로토콜 버전,
> 하네스 자체의 개선 루프까지 다룬다. 새 기능의 존재와 내 환경에서의 효과는 별개다.

## 왜 필요한가? (Why)

2025년의 좋은 패턴도 모델과 런타임이 바뀌면 다시 판단해야 한다. 예를 들어
서브에이전트를 항상 빈 컨텍스트에서 시작하면 이미 읽은 파일을 다시 읽는다.
반대로 모든 이력을 물려주면 독립적인 검토에 작성자의 가정까지 전달된다.

이 장은 **2026-09-12에 확인한 공개 1차 자료**의 보충 노트다. 아래 날짜는
발표일·사양 버전이며, 확인일과 구분한다. 사례의 성능 수치는 해당 실험의 결과이고,
사내 모델이나 이 폴더의 예제를 실행해 얻은 결과가 아니다.

## 핵심 개념 (What)

### 1. 최근 변화 지도

| 발표일 / 버전 | 확인된 변화 | 자료의 성격 | 읽을 위치 |
|---|---|---|---|
| 2026-09-08 | 서브에이전트에 `isolated` / `fork` 컨텍스트 모드 도입 | LangChain 구현 사례 | [09](./09-multi-agent.md), 아래 2절 |
| 2026-08-22 | 에이전트 메시징·HTTP·신원·SDK 중심의 MCP 후속 로드맵 | 향후 방향, 전체 구현 완료 아님 | 아래 4절 |
| 2026-07-28 | MCP 세션·초기화 제거, 요청별 메타데이터, Tasks 확장 분리 | 발표된 코어 사양 개정 | 아래 4절 |
| 2026-04-08 | 세션 저장, 하네스, 실행 환경의 수명 분리 | Anthropic 운영 아키텍처 사례 | 아래 3절 |
| 2026-03-24 | planner–generator–evaluator와 모델 발전에 따른 하네스 단순화 | Anthropic 실험 사례 | 아래 5절 |
| 2026-02-05 | 실행 자원 차이가 에이전트 벤치마크에 영향 | Anthropic 실험 보고 | [05](./05-verification-and-evals.md) |
| 2025-11-24 | 도구 검색·코드 기반 도구 호출·사용 예시 | 당시 베타로 발표된 기능 | [04](./04-tool-design.md) |

각 행의 공식 출처는 해당 절 또는 하단 참고 자료에 연결했다. 오래된 발표를
새 기능처럼 소개하지 않으며, 공급자별 API 지원 상태는 적용 시 다시 확인한다.

### 2. 서브에이전트: 격리와 이력 상속을 선택한다

LangChain은 **독립 조사·검증에는 isolated, 진행 중 작업을 이어받는 워커에는
fork**를 선택하는 기준을 제시했다. fork는 부모의 대화·상태를 전달하고 최종
응답을 부모에게 반환한다. 중복 탐색 감소와 프롬프트 캐시 활용 가능성이 있지만
항상 더 저렴하다는 뜻은 아니다.
[공식 발표, 2026-09-08](https://www.langchain.com/blog/organizing-context-in-a-multi-agent-harness)

**이 노트의 적용 판단:** 모드를 역할 이름만으로 결정하지 말고, 필요한 증거와
편향 가능성을 보고 선택한다. 컨텍스트를 상속하더라도 파일 쓰기 권한과 실행
예산은 별도로 제한한다. 상세 계약은 [09](./09-multi-agent.md)에 정리했다.

### 3. 장기 실행: 세션 저장과 실행 컨테이너의 수명을 분리한다

Anthropic의 Managed Agents 사례는 세션, 하네스, 샌드박스를 분리한다.
모델이 다음 행동을 판단하는 과정과 그 행동을 수행하는 환경을 별개로 운영하고,
복구 가능한 이력 저장도 특정 컨텍스트 관리 기법에 묶지 않는다.
[공식 아키텍처 설명, 2026-04-08](https://www.anthropic.com/engineering/managed-agents)

**이 노트의 적용 설계 예:** 사내 문서 추출 작업을 다음 세 수명으로 나눈다.

| 단위 | 보관할 것 | 종료·장애 때 처리 |
|---|---|---|
| 업무 실행(run) | 목표, 승인 기록, 작업 목록, 검증 결과 | 프로세스가 죽어도 유지 |
| 하네스 워커 | 현재 모델 호출, 남은 예산, 다음 행동 | 체크포인트에서 다른 워커가 재개 |
| 실행 환경 | 작업 파일, 임시 패키지, 생성 프로세스 | 필요한 산출물을 보관한 뒤 폐기·재구성 |

컨테이너를 없애기 전에 산출물을 저장하지 않으면 대화만 복구되고 작업 파일은
사라진다. 반대로 컨테이너만 살려두면 어떤 행동이 승인·완료됐는지 설명할 수 없다.
작은 시스템은 먼저 [07](./07-state-and-recovery.md)의 파일·DB 저장으로 시작하고,
독립적인 확장이나 복구가 필요해질 때 실행 단위를 분리한다. 외부 관리형 서비스
도입을 전제하지 않는다.

### 4. MCP: 2025 예제와 2026 사양을 섞지 않는다

공식 8월 로드맵은 `2026-07-28`을 발표된 사양 릴리스로 명시한다.
다음은 그 개정에서 확인한 핵심 차이다.
[공식 변경 기록](https://modelcontextprotocol.io/specification/2026-07-28/changelog)

| 항목 | 2026-07-28 변경 | 하네스에서 확인할 것 |
|---|---|---|
| 초기화·세션 | 프로토콜 세션과 초기화 핸드셰이크 제거 | 구버전 어댑터와 혼용하지 않기 |
| 기능 확인 | `server/discover`, 요청별 버전·capability 메타데이터 | 서버와 클라이언트 지원 버전 기록 |
| 추가 입력 | MRTR와 `resultType: input_required` | 추가 입력을 업무 완료로 오인하지 않기 |
| 장기 작업 | Tasks를 공식 확장으로 이동 | 코어 버전과 확장 버전을 따로 확인 |
| 연결 복구 | SSE 재전달·재개 제거, 새 요청으로 재발행 | 업무 멱등성 키를 요청 ID와 분리 |

Tasks 확장은 2025의 `tasks/result`, `tasks/list` 흐름과 다르며,
`tasks/get` 폴링과 `tasks/update` 입력 전달을 사용한다.
공식 저장소는 **`2026-07-28`을 Stable, `draft`를 Development**로 구분한다.
안정 버전이 있다는 사실을 모든 SDK의 지원으로 해석하지 않는다.
[공식 확장 버전 목록](https://github.com/modelcontextprotocol/ext-tasks)
확장 문서와 SDK의 지원 버전도 별도로 확인한다.
[Tasks 확장 명세](https://tasks.extensions.modelcontextprotocol.io/specification/2026-07-28/tasks)

8월 로드맵의 에이전트 신원·위임, 이벤트 전달, 전송 방식 통합은 후속 작업의
방향이다. 발표된 기능과 앞으로 성숙시킬 기능을 구분해야 한다.
[공식 로드맵, 2026-08-22](https://blog.modelcontextprotocol.io/posts/mcp-roadmap/)

**이 노트의 이행 순서:** 사용 중인 SDK·서버 버전을 적고, 지원하는 조합으로
읽기 도구 한 개를 먼저 연결한다. 그다음 추가 입력, 장기 작업, 취소, 연결 단절을
각각 검증한다. 문서 날짜만 보고 기존 서버의 핸드셰이크를 삭제하지 않는다.

### 5. 하네스 개선에도 별도의 바깥 루프를 둔다

Anthropic의 장기 앱 개발 실험은 planner–generator–evaluator 구조를 사용했고,
모델이 개선되면서 일부 컨텍스트 리셋 장치를 제거할 수 있었다고 설명한다.
이는 역할 수를 늘리는 처방보다 **기존 장치가 여전히 필요한지 측정하는 태도**를
보여준다.
[공식 실험 보고, 2026-03-24](https://www.anthropic.com/engineering/harness-design-long-running-apps)

LangChain의 `better-harness`는 바깥 에이전트가 허용된 하네스 파일을 바꾸고
eval로 후보를 비교하는 **연구용 구현**이다. 저장소는 train/holdout 분리가
강한 샌드박스 경계가 아니라고 명시한다. 운영 격리가 완성된 제품으로 보지 않는다.
[공식 예제 저장소](https://github.com/langchain-ai/deepagents/blob/main/examples/better-harness/README.md)

```text
안쪽 루프: 목표 → 모델 → 도구 → 검증 → 결과
바깥 루프: 실패 수집 → 변경 가설 → 후보 하네스 → 고정된 평가 → 채택/폐기
```

**이 노트의 운영 권고:** 바깥 에이전트가 정답·판정기·권한 정책까지 바꾸게 하면
실제 능력 향상 없이 점수만 높일 수 있다. 수정 가능한 프롬프트·도구 설명의 범위를
정하고, 별도 평가 프로세스가 보호된 테스트와 안전 회귀를 실행하게 한다.
holdout 점수를 반복적으로 보고 선택하면 그 셋에도 간접 과적합할 수 있으므로,
최종 확인용 미공개 셋을 따로 유지한다.

## 어떻게 사용하는가? (How)

### Step 1. 유행이 아니라 병목에 연결한다

다음은 위 사례를 바탕으로 한 **실무 적용 제안**이다.

| 관측한 문제 | 먼저 시험할 변경 | 함께 확인할 부작용 |
|---|---|---|
| 워커마다 같은 파일을 다시 읽는다 | 필요한 워커만 fork | 캐시 미적중 비용, 불필요한 정보 상속 |
| 도구 정의가 입력 대부분을 차지한다 | 도구 검색과 지연 로딩 | 필요한 도구를 검색하지 못하는 실패 |
| 원시 로그 처리에 모델 턴을 낭비한다 | 코드에서 필터·집계 후 결과 전달 | 누락·잘림, 실패를 0건으로 표시하는 오류 |
| 컨테이너 장애 뒤 작업을 잃는다 | 이력·산출물·실행 환경 수명 분리 | 재개 후 중복 쓰기 |
| 모델 교체 후 루프가 길어진다 | 모델별 하네스 조합 비교 | 특정 모델·평가 셋에만 맞춘 최적화 |
| 개선을 반복해도 운영 품질이 그대로다 | 보호된 평가와 실패 유형별 비교 | 평가 오염, 잘못된 판정기 |

### Step 2. 사내 문서 추출 예시로 한 번 연결한다

아래는 구현 완료 사례가 아니라 설계 연습이다. 실제 문서나 내부 주소는 필요 없다.

1. 샘플 문서와 정답 표를 고정하고, 누락 행·수치 변형·처리 실패를 구분한다.
2. 단일 에이전트로 기준선을 측정한다. 모델·하네스·데이터·실행 자원 버전을 남긴다.
3. 원문을 모두 모델에 넣기 전에 코드로 페이지 메타데이터를 만들고, 필요한
   페이지·영역만 전달한다. 모든 생략에는 원문 참조를 남긴다.
4. 독립 문서 처리는 isolated 워커로 시험한다. 이미 분석한 특정 문서의 후속
   수정만 fork 후보로 둔다. 검토자는 정답 기준과 산출물을 직접 확인한다.
5. 결과는 임시 위치에 쓰고 검증한다. 최종 반영 권한과 승인 범위는 하네스가 확인한다.
6. 작업 중 프로세스를 종료해 본다. 재개 뒤 완료된 문서를 재처리하거나 결과를
   중복 등록하지 않는지 확인한다.

### Step 3. 변경마다 짧은 실험 기록을 남긴다

```yaml
# 설계 기록 예시 — SDK 설정 파일이 아니다.
experiment: deferred-tool-loading
hypothesis: 필요한 도구만 공개하면 입력 토큰을 줄일 수 있다
baseline: 고정된 모델과 전체 도구 목록
candidate: 같은 모델과 검색 기반 도구 공개
fixed: [task_set, grader, resource_limits, total_budget]
measure: [verified_success, tool_selection_errors, total_tokens, p95_latency]
adopt_if: 보호된 평가와 안전 회귀를 만족하며 운영 목적 지표가 개선됨
rollback_if: 필수 도구 누락 또는 권한 회귀
verification: 미실행
```

한 번에 여러 장치를 추가하지 않는다. 모델만 교체한 경우, 하네스만 바꾼 경우를
따로 비교하고, [10](./10-production-checklist.md)으로 배포 조건을 확인한다.

## 참고 자료 (References)

- [Organizing Context in a Multi-Agent Harness](https://www.langchain.com/blog/organizing-context-in-a-multi-agent-harness) — LangChain, 2026-09-08
- [The New MCP Roadmap](https://blog.modelcontextprotocol.io/posts/mcp-roadmap/) — MCP, 2026-08-22
- [MCP Key Changes](https://modelcontextprotocol.io/specification/2026-07-28/changelog) — 사양 버전 2026-07-28
- [Scaling Managed Agents](https://www.anthropic.com/engineering/managed-agents) — Anthropic, 2026-04-08
- [Harness design for long-running application development](https://www.anthropic.com/engineering/harness-design-long-running-apps) — Anthropic, 2026-03-24
- [Quantifying infrastructure noise in agentic coding evals](https://www.anthropic.com/engineering/infrastructure-noise) — Anthropic, 2026-02-05
- [Introducing advanced tool use](https://www.anthropic.com/engineering/advanced-tool-use) — Anthropic, 2025-11-24
- [better-harness](https://github.com/langchain-ai/deepagents/blob/main/examples/better-harness/README.md) — 연구용 구현, 2026-09-12 확인; `main`은 이후 바뀔 수 있음

## 관련 문서

- [README — 학습 순서](./README.md)
- [04. 도구 설계](./04-tool-design.md)
- [05. 검증과 평가](./05-verification-and-evals.md)
- [09. 멀티 에이전트](./09-multi-agent.md)
- [10. 프로덕션 체크리스트](./10-production-checklist.md)
