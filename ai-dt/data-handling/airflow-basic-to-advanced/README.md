---
tags: [airflow, tutorial, python, dag, orchestration]
level: beginner-to-advanced
last_updated: 2026-05-02
reviewed_on: 2026-10-04
review_status: partial
document_type: topic_index
category_major: "AI·DT"
category_middle: "데이터 엔지니어링"
category_minor: "Airflow 파이프라인"
note_kind: "목차"
classified_on: "2026-10-05"
---

# Airflow 기초부터 고급까지

> [!info] 판본·검증 범위 — 2026-10-04
> 예제는 **Airflow 2.10.5** 학습용이다. 3.x public API는 `airflow.sdk`와 별도 provider 경로를 확인한다. 확인 당시 공식 3.x 페이지는 3.3.2를 표시했으며 회사 설치 버전이나 최신 보증이 아니다. 사내 Git Sync·권한·executor 조건은 확인되지 않은 시나리오다. 실행 예제의 로컬 구문/파싱 검증과 실제 scheduler·worker·업무 서버 검증을 구분한다.

목차의 8개 학습 단계를 보존한다. 목차·01~08을 모두 개별 검토했으며, 실제 사내 운영 조건과 Claude 협의가 필요한 통합 결정은 미확인으로 구분했다. [이 주제 정리 기록](../organization-log.md)에 근거와 남은 확인을 기록한다.


> 회사가 관리하는 Airflow 서버에 여러 Python 파일을 올려 안정적으로 실행하기 위한 독립 튜토리얼

## 이 튜토리얼의 목표

이 문서는 "로컬에서는 Python 파일들이 잘 실행되는데, 회사 Airflow 서버에 올리려면 무엇을 바꿔야 하는가?"라는 상황을 기준으로 작성했다.

Airflow를 처음 쓰는 사람도 다음 순서로 따라가면 된다.

1. Airflow가 무엇을 해주는지 이해한다.
2. 기존 Python 파일 하나를 Airflow Task로 실행한다.
3. 여러 Python 파일을 순서대로 연결한다.
4. 실패, 재시도, 스케줄, 날짜/시간 파티션을 다룬다.
5. Task 간 데이터 전달과 저장소 설계를 정리한다.
6. 로컬 환경과 Airflow 서버 환경 차이를 해결한다.
7. 운영 환경에서 필요한 고급 기능과 장애 대응 방법을 익힌다.
8. Connection 접근이 불가능하고 Bitbucket Git Sync로 배포하는 회사 조건에 맞춰 운영한다.

## 학습 순서

| 순서 | 문서 | 핵심 내용 |
|------|------|----------|
| 1 | [Airflow 기본 개념](./01-basic-concepts.md) | DAG, Task, Operator, Scheduler, Worker, XCom, Connection과 대체 방식 |
| 2 | [첫 번째 DAG 만들기](./02-first-dag-python-file.md) | 기존 Python 파일을 BashOperator/PythonOperator로 실행 |
| 3 | [의존성, 스케줄, 재시도](./03-dependencies-scheduling-retry.md) | `task1 >> task2`, `schedule`, `catchup`, hourly 인자, retry, timeout |
| 4 | [데이터와 상태 관리](./04-data-and-state.md) | XCom, 파일 저장소, 멱등성, 날짜/시간 파티션, 코드 기반 secret |
| 5 | [패키지와 실행 환경](./05-packages-and-environments.md) | 로컬과 서버 패키지 차이, venv, ExternalPython, 컨테이너 |
| 6 | [로컬 개발과 테스트](./06-local-development-and-testing.md) | DAG 파싱 테스트, Task 테스트, requirements 정리, 배포 체크 |
| 7 | [고급 운영 패턴](./07-advanced-operations.md) | Sensor, Dataset, Dynamic Task Mapping, pool, backfill, 장애 대응 |
| 8 | [Bitbucket Git Sync와 코드 기반 Secret 운영](./08-bitbucket-git-sync-and-code-secrets.md) | Connection 접근 불가, Bitbucket 배포, 코드 기반 secret, Git Sync 주의점 |

## 예제 기준

예제의 import와 사용법을 Airflow 2.10.5 판본으로 대조한다. 회사별 설치 분포는 조사하지 않았으며 2.x가 대부분이라는 근거는 없다.

```python
from airflow import DAG
from airflow.decorators import dag, task
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
```

Airflow 3에서는 DAG/Task/Asset의 public API가 `airflow.sdk`로 바뀌고 일부 Operator는 provider로 이동했다. 실행 전에 core/Python/provider 버전을 확인하고 해당 판본의 문서를 사용한다.

## 회사 관리형 Airflow에서의 현실

다음은 관리형 환경을 설명하기 위한 역할 예시다. 실제 회사의 권한 분리는 미확인이다.

| 역할 | 보통 가능한 일 |
|------|---------------|
| 일반 사용자 | Bitbucket repository에 DAG push, 수동 실행, 로그 확인 |
| Airflow 운영팀 | Worker 이미지 변경, 패키지 설치, provider 추가, executor 설정 변경 |
| 인프라/플랫폼팀 | Kubernetes, Docker registry, secret backend, network policy 관리 |

따라서 처음부터 "내 로컬 환경 그대로 Airflow에서 돌리겠다"라고 접근하면 막힐 수 있다. 먼저 Airflow 서버에서 무엇이 허용되는지 확인하고, 그 범위 안에서 실행 방식을 선택해야 한다.

원문에서 가정한 Connection/Variable UI 접근 제한과 Bitbucket Git Sync 환경에 해당한다면, [08. Bitbucket Git Sync와 코드 기반 Secret 운영](./08-bitbucket-git-sync-and-code-secrets.md)을 먼저 읽고 운영팀의 환경변수/secret backend 또는 승인된 외부 파일 주입 가능 여부를 확인한다. UI 권한 제한만으로 Git에 비밀을 저장해야 한다고 결론 내리지 않는다. 매시간 실행하는 Python 파일은 `--date {{ ds }}`만 넘기지 말고 `--start-ts`, `--end-ts`로 처리 구간을 넘기는 방식을 기본으로 한다.

## 추천 진행 방식

처음부터 고급 기능을 쓰지 말고 아래 순서로 간다.

1. Python 파일 하나를 `BashOperator`로 실행한다.
2. 필요한 패키지가 서버에 있는지 확인한다.
3. 여러 파일을 `step1 >> step2 >> step3`로 연결한다.
4. Task 간 파일 전달을 로컬 디스크가 아니라 공유 저장소로 바꾼다.
5. 패키지 충돌이 확인되면 `PythonVirtualenvOperator`, `ExternalPythonOperator`, 컨테이너 중 하나로 격리한다.
6. 운영 전 retry, timeout, `max_active_runs`, alert, log 확인 절차를 정리한다.

## 공식 문서

- [Apache Airflow PythonOperator / PythonVirtualenvOperator / ExternalPythonOperator](https://airflow.apache.org/docs/apache-airflow/2.10.5/howto/operator/python.html)
- [Apache Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/2.10.5/best-practices.html)
- [Apache Airflow Dependencies and Providers](https://airflow.apache.org/docs/apache-airflow/2.10.5/installation/dependencies.html)
- [Apache Airflow Modules Management](https://airflow.apache.org/docs/apache-airflow/2.10.5/administration-and-deployment/modules_management.html)

- [Airflow 3 public interface](https://airflow.apache.org/docs/apache-airflow/stable/public-airflow-interface.html).
- [2.10.5 Connection 저장 방식](https://airflow.apache.org/docs/apache-airflow/2.10.5/howto/connection.html).


### 로컬 검증 결과 — 2026-10-04

실제 Airflow 2.10.5의 DagBag에서 예제 DAG 8개를 발견하고 Task ID와 의존성을 대조했다. Python 예제 38개가 구문 검사를 통과했다. 외부 접속이 없는 PythonOperator 함수 1건을 실행했고, 환경변수 loader의 필수 값 누락 4건을 검사했다.

검증 환경은 저장소 밖의 임시 Python 3.12.12 환경이다. 문서 예제를 임시 파일로 추출하고 로컬 helper를 사용했다. 실제 scheduler와 worker, 업무 파일, FTP·MinIO·API, 인증과 TLS 연결은 실행하지 않았다. DAG 발견은 업무 처리의 성공을 보장하지 않는다.