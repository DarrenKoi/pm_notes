---
tags: [airflow, dependency, schedule, retry, timeout]
level: beginner-intermediate
last_updated: 2026-05-02
status: complete
---

# 03. 의존성, 스케줄, 재시도

## 목표

여러 Python 파일을 순서대로 실행하고, 운영에 필요한 기본 설정을 붙인다.

예제 흐름:

```text
download.py -> preprocess.py -> analyze.py -> report.py
```

Airflow에서는 이 순서를 Task 의존성으로 표현한다.

```python
download >> preprocess >> analyze >> report
```

기본적으로 앞 Task가 성공해야 뒤 Task가 실행된다. 앞 Task가 실패하면 뒤 Task는 실행되지 않고 `upstream_failed` 상태가 된다.

## 여러 스크립트 순차 실행

```python
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator


BASE_DIR = "/opt/airflow/dags/company_jobs"

default_args = {
    "owner": "data-team",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="daily_company_pipeline",
    default_args=default_args,
    start_date=datetime(2026, 5, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["company", "daily"],
) as dag:
    download = BashOperator(
        task_id="download",
        bash_command=f"set -euo pipefail; python {BASE_DIR}/download.py --date {{{{ ds }}}}",
        execution_timeout=timedelta(minutes=30),
    )

    preprocess = BashOperator(
        task_id="preprocess",
        bash_command=f"set -euo pipefail; python {BASE_DIR}/preprocess.py --date {{{{ ds }}}}",
        execution_timeout=timedelta(hours=1),
    )

    analyze = BashOperator(
        task_id="analyze",
        bash_command=f"set -euo pipefail; python {BASE_DIR}/analyze.py --date {{{{ ds }}}}",
        execution_timeout=timedelta(hours=2),
    )

    report = BashOperator(
        task_id="report",
        bash_command=f"set -euo pipefail; python {BASE_DIR}/report.py --date {{{{ ds }}}}",
        execution_timeout=timedelta(minutes=20),
    )

    download >> preprocess >> analyze >> report
```

## 중요한 DAG 설정

| 설정 | 의미 | 권장 시작값 |
|------|------|-------------|
| `dag_id` | DAG 이름 | 팀/업무가 드러나게 작성 |
| `start_date` | 스케줄 계산 시작점 | 과거의 고정 날짜 |
| `schedule` | 실행 주기 | 처음에는 `None`, 운영 시 cron 또는 preset |
| `catchup` | 과거 미실행 구간 자동 실행 여부 | 처음에는 `False` |
| `max_active_runs` | 같은 DAG의 동시 실행 개수 | 순차 배치면 `1` |
| `tags` | UI 필터용 태그 | 팀명, 시스템명 |

## schedule 예시

| 값 | 의미 |
|----|------|
| `None` | 자동 실행 없음, 수동 실행만 |
| `"@daily"` | 매일 |
| `"@hourly"` | 매시간 |
| `"0 6 * * *"` | 매일 06:00 |
| `"0 6 * * 1-5"` | 평일 06:00 |

cron을 쓸 때는 Airflow 서버의 timezone 설정을 확인한다. 회사 서버가 UTC로 설정되어 있으면 한국 시간과 9시간 차이가 난다.

## catchup 이해

`catchup=True`이면 `start_date`부터 현재까지 실행되지 않은 모든 스케줄을 한꺼번에 만들 수 있다.

예:

```python
start_date=datetime(2026, 1, 1)
schedule="@daily"
catchup=True
```

2026-05-02에 처음 DAG를 켜면 2026-01-01부터 2026-05-01까지 많은 DAG Run이 생길 수 있다.

처음 운영할 때는 보통 `catchup=False`로 시작한다. 과거 데이터를 의도적으로 재처리할 때만 backfill이나 수동 실행을 사용한다.

## retry

일시적인 네트워크 오류, DB connection 오류, API timeout은 재시도로 해결되는 경우가 있다.

```python
default_args = {
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}
```

Task별로 다르게 줄 수도 있다.

```python
download = BashOperator(
    task_id="download",
    bash_command="python download.py --date {{ ds }}",
    retries=3,
    retry_delay=timedelta(minutes=10),
)
```

주의:

- 재시도를 켜려면 Task가 재실행 가능해야 한다.
- 같은 데이터를 중복 insert하는 코드라면 retry가 오히려 문제를 키운다.
- output path를 날짜별로 고정하고 overwrite/upsert 전략을 정해야 한다.

## timeout

Task가 무한정 실행되는 것을 막기 위해 `execution_timeout`을 둔다.

```python
from datetime import timedelta

analyze = BashOperator(
    task_id="analyze",
    bash_command="python analyze.py --date {{ ds }}",
    execution_timeout=timedelta(hours=2),
)
```

timeout은 업무 특성에 맞게 정한다.

| 작업 | timeout 예시 |
|------|-------------|
| API 호출 | 10~30분 |
| 전처리 | 30분~2시간 |
| 모델 학습 | 별도 협의 |
| 리포트 생성 | 10~30분 |

## trigger_rule

기본값은 `all_success`다. upstream Task가 모두 성공해야 실행된다.

대부분의 순차 배치는 기본값을 그대로 쓰면 된다.

특정 상황에서는 trigger rule을 바꾼다.

| 값 | 의미 | 사용 예 |
|----|------|--------|
| `all_success` | 앞 Task가 모두 성공해야 실행 | 일반 처리 |
| `all_done` | 성공/실패와 상관없이 앞 Task가 끝나면 실행 | cleanup |
| `one_failed` | 앞 Task 중 하나라도 실패하면 실행 | 실패 알림 |
| `none_failed` | 실패가 없으면 실행, skipped는 허용 | branch 합치기 |

cleanup 예:

```python
cleanup = BashOperator(
    task_id="cleanup",
    bash_command="python cleanup.py --date {{ ds }}",
    trigger_rule="all_done",
)

[download, preprocess, analyze] >> cleanup
```

## 병렬 실행

순차 실행만 필요한 것은 아니다.

```text
            -> analyze_a ->
download -> preprocess -> merge -> report
            -> analyze_b ->
```

Airflow에서는 이렇게 쓴다.

```python
download >> preprocess
preprocess >> [analyze_a, analyze_b] >> merge >> report
```

주의:

- 병렬 Task가 같은 output 파일을 쓰면 안 된다.
- DB table에 동시에 write하면 lock이나 중복 문제가 생길 수 있다.
- 회사 Airflow의 worker slot/pool 제한을 확인해야 한다.

## TaskFlow API로 의존성 표현

Python 함수 중심 코드라면 `@task`가 더 읽기 좋을 수 있다.

```python
from datetime import datetime

from airflow.decorators import dag, task


@dag(
    dag_id="daily_company_taskflow",
    start_date=datetime(2026, 5, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
)
def pipeline():
    @task
    def download(run_date: str) -> str:
        return f"s3://raw/company/dt={run_date}/data.parquet"

    @task
    def preprocess(run_date: str, raw_path: str) -> str:
        return f"s3://clean/company/dt={run_date}/data.parquet"

    @task
    def analyze(run_date: str, clean_path: str) -> str:
        return f"s3://result/company/dt={run_date}/result.json"

    @task
    def report(run_date: str, result_path: str) -> None:
        print(f"create report for {run_date}: {result_path}")

    raw_path = download("{{ ds }}")
    clean_path = preprocess("{{ ds }}", raw_path)
    result_path = analyze("{{ ds }}", clean_path)
    report("{{ ds }}", result_path)


pipeline()
```

함수의 반환값은 XCom을 통해 다음 Task로 전달된다. 반환값은 작은 문자열이나 dict 정도로 유지한다.

## 운영용 기본값 예시

```python
default_args = {
    "owner": "data-platform",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="service_domain_job_name",
    default_args=default_args,
    start_date=datetime(2026, 5, 1),
    schedule="0 6 * * *",
    catchup=False,
    max_active_runs=1,
    tags=["service-domain", "daily"],
) as dag:
    ...
```

## 체크리스트

- `dag_id`가 회사 Airflow 안에서 중복되지 않는가
- 처음 테스트는 `schedule=None`으로 했는가
- 운영 전 `catchup` 의도를 확인했는가
- 순차 실행이 필요한 곳에 `>>`가 명확히 있는가
- retry를 켜도 중복 데이터가 생기지 않는가
- Task별 timeout이 있는가
- 처리 날짜를 `{{ ds }}` 등 Airflow 값으로 받고 있는가
- 병렬 Task가 같은 output에 쓰지 않는가

## 다음 단계

다음 문서에서는 Task 간 데이터 전달, XCom, 파일 저장소, 멱등성을 정리한다.

- [04. 데이터와 상태 관리](./04-data-and-state.md)
