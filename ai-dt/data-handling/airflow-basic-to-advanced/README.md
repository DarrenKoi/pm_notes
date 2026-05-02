---
tags: [airflow, tutorial, python, dag, orchestration]
level: beginner-to-advanced
last_updated: 2026-05-02
---

# Airflow 기초부터 고급까지

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

대부분의 회사 Airflow는 아직 Airflow 2.x를 많이 사용하므로, 예제는 Airflow 2.x 스타일을 기본으로 한다.

```python
from airflow import DAG
from airflow.decorators import dag, task
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
```

Airflow 3.x에서는 일부 public API와 provider import 경로가 달라질 수 있다. 회사 서버의 Airflow 버전을 먼저 확인하고, import error가 나면 사내 버전에 맞는 공식 문서를 확인한다.

## 회사 관리형 Airflow에서의 현실

회사 Airflow는 보통 다음 권한이 분리되어 있다.

| 역할 | 보통 가능한 일 |
|------|---------------|
| 일반 사용자 | Bitbucket repository에 DAG push, 수동 실행, 로그 확인 |
| Airflow 운영팀 | Worker 이미지 변경, 패키지 설치, provider 추가, executor 설정 변경 |
| 인프라/플랫폼팀 | Kubernetes, Docker registry, secret backend, network policy 관리 |

따라서 처음부터 "내 로컬 환경 그대로 Airflow에서 돌리겠다"라고 접근하면 막힐 수 있다. 먼저 Airflow 서버에서 무엇이 허용되는지 확인하고, 그 범위 안에서 실행 방식을 선택해야 한다.

현재 사내 조건처럼 Connection/Variable 접근이 불가능하고 Bitbucket Git Sync만 허용된다면, [08. Bitbucket Git Sync와 코드 기반 Secret 운영](./08-bitbucket-git-sync-and-code-secrets.md)을 먼저 읽고 secret과 배포 방식을 그 조건에 맞춘다. 매시간 실행하는 Python 파일은 `--date {{ ds }}`만 넘기지 말고 `--start-ts`, `--end-ts`로 처리 구간을 넘기는 방식을 기본으로 한다.

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
- [Apache Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [Apache Airflow Dependencies and Providers](https://airflow.apache.org/docs/apache-airflow/stable/installation/dependencies.html)
- [Apache Airflow Modules Management](https://airflow.apache.org/docs/apache-airflow/stable/administration-and-deployment/modules_management.html)
