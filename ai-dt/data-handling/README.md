---
tags: [data-handling, index]
reviewed_on: 2026-10-04
review_status: partial
document_type: topic_index
aliases: [데이터 처리 학습 목차]
category_major: "AI·DT"
category_middle: "데이터 엔지니어링"
category_minor: "데이터 처리 안내"
note_kind: "목차"
classified_on: "2026-10-05"
---

# 데이터 처리 학습 목차

데이터 구조를 정하고, 처리 단계를 연결하며, 문서화되지 않은 입력 형식을 분석하는 학습 문서 모음이다. 이 폴더 안에서 주제별로 읽고 실행 자료는 각 모듈에 둔다.

> [!info] 검토 진행 중 — 2026-10-04
> 원래 Markdown 32개와 실행 자료 4개를 목록화했다. 원문32개 전체를 개별 검토하고 목차·의존성·이벤트·Airflow9개·MinIO·정규화9개·역공학10개의 역할과 적용 조건을 설명했다. 개별 문서의 reviewed_with_limits는 실제 장비/외부 도구/회사 계약을 모두 검증했다는 뜻이 아니다. Claude 협의가 필요한 완전 통합과 실제 읽기 화면의 최종 재검증은 일부 미완료다. 회사 관리형 환경은 확인된 일반 제품 기능과 구분해 읽는다. 범위·출처·미확인은 [정리 기록](./organization-log.md)에 남긴다.

## 카테고리와 읽기 순서

| 목적 | 시작 문서 | 이어서 읽기 |
|---|---|---|
| DAG 작성과 배포를 차례로 배우기 | [Airflow 기초부터 고급까지](./airflow-basic-to-advanced/README.md) | 모듈 목차의 01~08 순서 |
| Python 처리 단계의 선후 관계 정하기 | [Task 의존성](./task-dependencies.md) | [이벤트 기반 실행](./event-driven-execution.md)에서 DAG 시작 조건 구분 |
| MinIO와 분석 파이프라인의 연결 예시 보기 | [Airflow + MinIO](./airflow-minio-tutorial.md) | 위 의존성·이벤트 문서와 설치 버전/파일 전달 조건 대조 |
| 저장·검색·캐시·근거의 데이터 의미 정리하기 | [데이터 정규화](./normalization/README.md) | 모듈 목차의 핵심 개념 → 모델링 → 저장소별 적용 → 통합 표 |
| 알 수 없는 binary 입력을 조사하기 | [Binary 역공학](./binary-reverse-engineering/README.md) | 개념 입문 → 작업 계약 → runbook → 도구 참고; 원본 보존과 승인 범위 확인 |

## 유사 문서의 역할

Airflow 커리큘럼은 단계별 학습 과정이고, Task 의존성 문서는 실행 순서 패턴 참고, 이벤트 문서는 시작/대기 조건 참고다. MinIO 문서는 저장소를 연결하는 상황별 예시다. 겹치는 Operator 설명의 완전 통합은 Claude 협의가 필요해 보류했으며 고유 예제와 맥락을 그대로 보존한다.

정규화 시리즈와 역공학 runbook은 목적이 다르다. 전자는 모델링·의미 일관성 학습, 후자는 증거를 수집해 형식 가설을 검증하는 작업 절차다. runbook의 실행 명령은 이 문서 정리 작업에서 실제 장비 파일을 조사하라는 지시가 아니다.
