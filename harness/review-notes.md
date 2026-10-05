---
tags: [harness-engineering, verification, learning]
aliases: [하네스 적용 조건]
reviewed_on: 2026-10-04
review_status: partial
document_type: learning
category_major: "에이전트 하네스"
category_middle: "하네스 설계"
category_minor: "적용 조건"
note_kind: "검토 기록"
classified_on: "2026-10-05"
---

# 하네스 노트를 현재 환경에 적용하는 방법

이 폴더는 모델 밖의 실행 루프·컨텍스트·도구·권한·평가·복구를 설명한다.
01~09는 개념과 작은 실습, 10은 배포 점검, 11은 날짜가 있는 외부 사례다.
같은 주제를 다시 언급해도 목적이 다르므로 각 장을 유지했다.
[읽기 순서](./README.md), [문서별 검토 기록](./organization-log.md).

## 사양과 구현을 구분한다

| 대상 | 2026-10-04 확인 결과 | 적용 조건 |
|---|---|---|
| MCP 코어 | `2026-07-28` 변경 기록에서 세션·초기화 제거와 요청별 메타데이터 확인 | 설치된 서버·클라이언트가 지원하는 버전을 먼저 확인 |
| Tasks | 안정 스키마 `2026-07-28`, 개발 스키마 `draft`; SDK receiver 지원과 별개 | core·extension·SDK 각각 기록하고 단절·취소·중복 쓰기를 시험 |
| LangGraph | 공식 main 소스에 `sync`/`async`/`exit` durability 선택지 존재 | 릴리스 버전 고정 검증은 미완료; 저장소 main과 설치 패키지를 혼동하지 않기 |
| OpenTelemetry GenAI | 기존 semantic-conventions 페이지는 새 저장소로 이동했다고 공지 | 08의 JSONL은 학습용 형식; SDK 계측·OTLP 전송·완전한 규약 준수 구현이 아님 |

근거: [MCP 변경 기록](https://modelcontextprotocol.io/specification/2026-07-28/changelog),
[Tasks 공식 저장소](https://github.com/modelcontextprotocol/ext-tasks),
[LangGraph 공식 소스](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/pregel/main.py),
[OpenTelemetry 이동 공지](https://opentelemetry.io/docs/specs/semconv/gen-ai/),
[GenAI 새 저장소](https://github.com/open-telemetry/semantic-conventions-genai).
동적 main 자료는 확인일의 스냅샷 설명이며 릴리스 고정 계약이 아니다.

## 실습을 실행할 때

Python 3.10 이상 문법을 사용한다. 02의 본 실행에는 `openai`와 지원 엔드포인트가
필요하지만, 가짜 클라이언트 실습은 모델 호출 없이 동작한다. Markdown의 코드
블록을 파일로 저장하는 예제이며 저장소에 운영 앱이 구현됐다는 뜻은 아니다.

- **03**: `keep_last=0`은 모든 도구 결과를 치환한다. 음수·불리언은 거부한다.
  삭제 가능한 읽기 결과만 선택하는 운영 정책은 별도로 필요하다.
- **04**: cursor는 0 이상, limit은 1~100 정수다. 끝을 넘은 페이지는 안내를 반환한다.
  고정 배열의 offset 예제여서 변경 중인 검색 결과에 안정적인 커서를 보장하지 않는다.
- **05**: 조합식은 `0 ≤ c ≤ n`, `1 ≤ k ≤ n`에서 사용한다. 시도 간 상관과
  태스크 분포가 있으면 추정치를 운영 신뢰도로 바로 해석하지 않는다.
- **06**: 문자열 차단과 도구 이름 목록은 경로·계정·네트워크 경계를 대신하지 않는다.
  `read_file`도 접근 가능한 경로를 별도로 제한해야 한다.
- **07**: 파일 교체의 원자적 가시성과 디스크 내구성·다중 워커·외부 멱등성은 다르다.
  `retry`는 양수 시도 횟수가 필요하고 호출별 timeout·총 deadline은 구현하지 않는다.
  `cap`은 지수 항의 상한이며 jitter가 추가돼 전체 대기시간이 cap보다 길 수 있다.
- **08**: 호출 후 토큰 차감은 선결제 상한이 아니다. 에러 문자열 반환은 예외 기반
  tracer에서 성공으로 기록될 수 있다. 별도 업무 상태를 계측한다.
- **09**: 위임은 컨텍스트 전달 예제다. 권한·파일·프로세스 격리는 구현하지 않는다.

평가에서는 task, 반복 trial, grader, 실제 환경 outcome을 구분하고 판정기를
사람의 라벨과 대조한다. [Anthropic 평가 설명](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
(2026-01-09 발표, 2026-10-04 확인).

## 미확인과 사례 수치

캐시 적중은 서버의 토큰화·모델·캐시 키 조건에 달려 있다. JSON 순서를 고정했다고
적중이 보장되지는 않는다. 모델 평가 순위, Manus 입력/출력 비율, 멀티 에이전트
개선율과 비용 배수는 인용된 실험의 수치다. 이번 정리에서 재현하지 않았다.
전체 참고 자료의 재검증, 사내 엔드포인트·Docker 격리·장애 복구·실제 OTel 수집은
미완료다. 20개 eval·5주 계획은 학습 제안이며 충분한 표본이나 일정의 보장이 아니다.

> [!warning] 협의 보류
> Herdr의 현재 pane을 찾지 못해 전용 Claude 검토 pane을 연결할 수 없었다.
> 출처 해석이 더 필요한 모델 서빙 설정·논문 일반화·문서 재분할은 보류했다.
> Claude 의견이나 운영 검증을 수행한 것으로 기록하지 않는다.
