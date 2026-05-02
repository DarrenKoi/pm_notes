---
tags: [airflow, dag, operator, scheduler, worker]
level: beginner
last_updated: 2026-05-02
status: complete
---

# 01. Airflow 기본 개념

## Airflow는 무엇인가

Airflow는 Python 코드를 실행하는 서버라기보다, 작업을 **언제**, **어떤 순서로**, **실패하면 어떻게**, **어디까지 성공했는지** 관리하는 오케스트레이션 도구다.

예를 들어 아래 작업이 있다고 하자.

```text
1. download.py     원천 데이터 다운로드
2. preprocess.py   전처리
3. analyze.py      분석
4. report.py       리포트 생성
```

터미널에서는 이렇게 실행할 수 있다.

```bash
python download.py && python preprocess.py && python analyze.py && python report.py
```

하지만 운영 환경에서는 다음 문제가 생긴다.

- 어느 단계에서 실패했는지 찾기 어렵다.
- 실패한 단계만 다시 실행하기 어렵다.
- 매일 정해진 시간에 실행해야 한다.
- 동시에 여러 날짜를 처리하면 결과가 섞일 수 있다.
- 로그가 파일마다 흩어진다.
- 패키지 버전이 서버마다 다를 수 있다.

Airflow는 이 문제를 DAG와 Task 단위로 해결한다.

## 핵심 구조

```text
Airflow Web UI
  - DAG 목록 확인
  - 실행/중지
  - Task별 로그 확인

Scheduler
  - DAG 파일을 주기적으로 읽음
  - 실행할 Task를 결정

Metadata DB
  - DAG run, Task 상태, Connection, Variable 저장

Executor / Worker
  - 실제 Task 실행
```

중요한 점은 Python 코드가 Web UI에서 실행되는 것이 아니라 **Worker에서 실행**된다는 것이다. 패키지 설치 여부, 파일 경로, 네트워크 권한도 Worker 기준으로 봐야 한다.

## DAG

DAG는 Directed Acyclic Graph의 약자다. 순환이 없는 작업 그래프라는 뜻이다.

```text
download -> preprocess -> analyze -> report
```

Airflow에서는 보통 `dags/` 폴더에 있는 `.py` 파일 하나가 DAG 하나를 정의한다.

```python
from datetime import datetime

from airflow import DAG


with DAG(
    dag_id="my_first_pipeline",
    start_date=datetime(2026, 5, 1),
    schedule="@daily",
    catchup=False,
) as dag:
    ...
```

`dag_id`는 Airflow UI에 표시되는 이름이다. 회사 서버 안에서 유일해야 한다.

## Task

Task는 DAG 안에서 실행되는 최소 작업 단위다.

예를 들어 다음은 세 개의 Task다.

```text
download
preprocess
analyze
```

Task는 성공, 실패, 재시도, 건너뜀 같은 상태를 가진다.

| 상태 | 의미 |
|------|------|
| `success` | 정상 완료 |
| `failed` | 실패 |
| `upstream_failed` | 앞 Task가 실패해서 실행되지 않음 |
| `skipped` | 조건에 의해 건너뜀 |
| `running` | 실행 중 |
| `queued` | 실행 대기 중 |

## Operator

Operator는 Task가 "무엇을 실행할지" 정하는 템플릿이다.

| Operator | 용도 |
|----------|------|
| `BashOperator` | shell command 실행 |
| `PythonOperator` | Python 함수 실행 |
| `@task` | Python 함수를 Task로 만드는 TaskFlow 방식 |
| `PythonVirtualenvOperator` | 실행 시점에 별도 virtualenv 생성 후 Python 함수 실행 |
| `ExternalPythonOperator` | 미리 만들어진 다른 Python 환경에서 실행 |
| `KubernetesPodOperator` | Kubernetes Pod에서 컨테이너 실행 |
| `DockerOperator` | Docker 컨테이너 실행 |
| Sensor 계열 | 파일, S3 object, DB 상태 등을 기다림 |

처음에는 `BashOperator` 또는 `PythonOperator`만 알아도 충분하다.

## DAG Run과 Task Instance

DAG Run은 특정 날짜/시각에 실행된 DAG의 한 번의 실행 기록이다.

예:

```text
daily_sales_pipeline
  - 2026-05-01 실행
  - 2026-05-02 실행
  - 2026-05-03 실행
```

Task Instance는 특정 DAG Run 안에서 특정 Task가 실행된 기록이다.

```text
2026-05-02 DAG Run의 preprocess Task
```

Airflow UI에서 실패한 Task만 다시 실행할 수 있는 이유가 이 구조 때문이다.

## 스케줄과 처리 구간

Airflow에서 가장 헷갈리는 부분은 "지금 실행"과 "어느 데이터 구간을 처리"가 다를 수 있다는 점이다.

예를 들어 매일 오전 6시에 전날 데이터를 처리한다면, 2026-05-02 06:00 실행은 보통 2026-05-01 데이터를 처리한다.

먼저 구분해야 한다.

| 목적 | 날짜/시간 인자가 필요한가 |
|------|--------------------------|
| 단순히 Python 파일을 매시간 실행하고 성공/실패만 Web UI에서 보고 싶음 | 필수 아님 |
| Python 파일이 항상 "현재 최신 데이터"만 처리함 | 필수 아님. 단, 재실행 재현성은 낮음 |
| 특정 날짜/시간 데이터만 정확히 처리해야 함 | 필요 |
| 실패한 10시 작업을 다시 실행하면 10시 데이터만 다시 처리해야 함 | 필요 |
| backfill로 과거 구간을 다시 처리해야 함 | 필요 |

즉, Airflow의 Web UI 추적을 위해 `ds`가 필요한 것은 아니다. Airflow는 인자 없이 실행해도 DAG Run과 Task 성공/실패를 추적한다. `ds`, `data_interval_start`, `data_interval_end` 같은 값은 Python 파일이 **어느 데이터 구간을 처리해야 하는지 알아야 할 때** 넘긴다.

daily 작업이면 날짜만 인자로 받아도 충분할 수 있다.

```python
bash_command="python job.py --date {{ ds }}"
```

`{{ ds }}`는 `YYYY-MM-DD` 형식 문자열이다.

하지만 매시간 실행하는 작업이면 날짜만으로는 부족하다. 같은 날짜 안에 24번 실행되기 때문이다. hourly 작업은 시간 정보까지 넘긴다.

```python
bash_command=(
    "python job.py "
    "--start-ts '{{ data_interval_start }}' "
    "--end-ts '{{ data_interval_end }}'"
)
```

또는 날짜와 시간을 나눠서 넘길 수 있다.

```python
bash_command=(
    "python job.py "
    "--date {{ ds }} "
    "--hour {{ data_interval_start.strftime('%H') }}"
)
```

권장 기준:

| 실행 주기 | 권장 인자 |
|----------|----------|
| daily | `--date {{ ds }}` |
| hourly | `--start-ts '{{ data_interval_start }}' --end-ts '{{ data_interval_end }}'` |
| hourly, 파일 경로가 날짜/시간 파티션 | `--date {{ ds }} --hour {{ data_interval_start.strftime('%H') }}` |
| manual rerun이 잦음 | `--start-ts`, `--end-ts`처럼 처리 구간을 명확히 전달 |

중요한 데이터 파티션을 코드 안에서 `datetime.now()`로 정하면 재실행할 때 다른 구간을 처리할 수 있다. Airflow에서는 실행 컨텍스트에서 받은 처리 구간을 인자로 넘기는 구조가 안전하다.

## XCom

XCom은 Task 간 작은 데이터를 주고받는 기능이다.

좋은 사용:

```text
"s3://bucket/daily/dt=2026-05-02/output.parquet"
```

나쁜 사용:

```text
DataFrame 전체
대용량 JSON
이미지/엑셀/압축 파일 내용
```

큰 데이터는 MinIO/S3/NAS/DB 같은 외부 저장소에 저장하고, XCom에는 경로만 넣는다.

## Connection과 코드 기반 secret

Connection은 DB, S3, MinIO, API 같은 외부 시스템 접속 정보를 Airflow에 저장하는 기능이다.

일반적으로는 코드에 비밀번호를 넣지 않고 Connection을 쓴다.

나쁜 예:

```python
PASSWORD = "my-password"
```

좋은 예:

```python
from airflow.hooks.base import BaseHook


def run():
    conn = BaseHook.get_connection("warehouse_db")
    print(conn.host)
```

Connection 등록 권한이 없으면 운영팀에 요청해야 한다.

하지만 현재 사내 환경처럼 사용자가 Connection/Variable에 접근할 수 없고 Bitbucket Git Sync로 DAG만 배포한다면, 현실적으로 `secrets.py` 같은 코드 파일에 secret을 넣어야 할 수 있다. 이 경우 Bitbucket private repository 자체를 secret 저장소처럼 관리해야 한다.

```python
# dags/company_job/secrets.py

MINIO = {
    "endpoint": "minio.company.internal:9000",
    "access_key": "REPLACE_WITH_REAL_ACCESS_KEY",
    "secret_key": "REPLACE_WITH_REAL_SECRET_KEY",
    "bucket": "company-data",
}
```

이 방식에서는 다음 원칙을 지킨다.

- secret을 여러 파일에 흩뿌리지 않고 `secrets.py`에 모은다.
- secret 값을 로그에 출력하지 않는다.
- secret이 들어간 Bitbucket repository 접근 권한을 최소화한다.
- secret이 노출되면 파일 삭제가 아니라 key rotation으로 대응한다.
- 자세한 운영 방식은 [08. Bitbucket Git Sync와 코드 기반 Secret 운영](./08-bitbucket-git-sync-and-code-secrets.md)을 따른다.

## Variable

Variable은 DAG에서 사용하는 설정값을 Airflow에 저장하는 기능이다.

예:

- bucket name
- 기본 처리 경로
- feature flag
- 실행 모드

주의할 점은 DAG 파일 최상단에서 `Variable.get()`을 많이 호출하면 Scheduler가 DAG를 파싱할 때마다 DB를 조회할 수 있다는 것이다. 가능하면 Task 실행 시점에 읽거나 Jinja template을 사용한다.

## 회사 Airflow에서 가장 먼저 확인할 것

| 항목 | 확인 이유 |
|------|----------|
| Airflow 버전 | 예제 코드 import 경로와 기능 차이 |
| Python 버전 | 로컬 패키지와 호환성 확인 |
| Executor | Task가 같은 서버에서 도는지, 여러 Worker에 분산되는지 확인 |
| DAG 배포 방식 | Bitbucket Git Sync 대상 repository, branch, sync interval 확인 |
| Worker 패키지 목록 | `ModuleNotFoundError` 예방 |
| 사내 PyPI/Nexus | 추가 패키지 설치 가능 여부 |
| Connection/Variable 권한 | 현재 환경에서는 접근 불가로 가정하고 코드 기반 secret 사용 |
| 공유 저장소 | Task 간 파일 전달 방식 결정 |
| Kubernetes/Docker 허용 여부 | 패키지 격리 전략 결정 |

## 기본 원칙

- DAG 파일은 가볍게 유지한다.
- 실제 무거운 처리는 Task 내부에서 한다.
- 실패해야 할 상황에서는 예외를 발생시킨다.
- 처리 날짜 또는 처리 시간 구간은 Airflow에서 받은 값을 사용한다.
- 큰 데이터는 XCom이 아니라 외부 저장소로 전달한다.
- 현재 환경에서는 secret을 `secrets.py`에 모으고 Bitbucket 접근 권한을 엄격히 관리한다.
- 로컬 디스크에 의존하지 않는다.
- 패키지 버전 차이를 빨리 확인한다.

## 다음 단계

다음 문서에서는 기존 Python 파일 하나를 Airflow에서 실행하는 가장 작은 DAG를 만든다.

- [02. 첫 번째 DAG 만들기](./02-first-dag-python-file.md)
