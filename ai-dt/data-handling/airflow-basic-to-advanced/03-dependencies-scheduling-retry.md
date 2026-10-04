---
tags: [airflow, dependency, schedule, retry, timeout]
level: beginner-intermediate
last_updated: 2026-05-02
reviewed_on: 2026-10-04
review_status: partial
document_type: learning_note
---

# 03. 의존성, 스케줄, 재시도

> [!info] 판본·검증 범위 — 2026-10-04
> **Airflow 2.10.5** 예제다. 회사 설치 버전·executor·권한·Git Sync 조건은 미확인이다. 3.x로 그대로 복사하지 않고 해당 판본의 public API/provider 문서를 확인한다. 로컬 파싱과 실제 scheduler·worker·외부 시스템 검증을 구분한다.
> DAG 정의와 부분 설정 예제를 구분한다. 표시 시각, logical date, 처리 구간은 같은 개념이 아니다.


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

일반 Task의 기본 `all_success`에서는 upstream이 모두 성공해야 실행된다. 최종 실패는 downstream의 `upstream_failed`로 이어질 수 있고, retry 중에는 기다리며 skip은 별도 전파 규칙이 있다. branch/teardown/다른 trigger rule은 구분한다.

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
| `None` | 자동 스케줄 없음; UI/CLI/API 외부 trigger 가능 |
| `"@daily"` | 매일 |
| `"@hourly"` | 매시간 |
| `"0 6 * * *"` | 매일 06:00 |
| `"0 6 * * 1-5"` | 평일 06:00 |

cron은 DAG timezone을 기준으로 계산한다. aware `start_date`와 기본 timezone을 확인한다. Web UI의 표시 timezone은 브라우저 설정이므로 스케줄 기준과 따로 확인한다. UTC와 KST는 9시간 차이다.

## UTC 서버에서 한국 시간 기준으로 실행하기

회사 Airflow 서버가 UTC이고 업무 기준이 한국 시간(KST, `Asia/Seoul`)이면 `{{ ds }}`를 그대로 업무일로 쓰지 않는 것이 안전하다.

Airflow 2.10.5는 내부 datetime을 UTC로 저장하고 template datetime을 업무 timezone으로 자동 변환하지 않는다. `ds`는 logical date의 날짜 표현이며 실제 실행 시각이나 UI 표시 날짜가 아니다. 따라서 한국 업무일이 필요하면 명시적으로 KST로 변환해서 넘긴다.

timezone 선언 골격(아래 `...`는 Task 구현이 아님):

```python
from airflow import DAG
import pendulum

KST = pendulum.timezone("Asia/Seoul")

with DAG(
    dag_id="hourly_kst_pipeline",
    start_date=pendulum.datetime(2026, 5, 1, tz=KST),
    schedule="@hourly",
    catchup=False,
) as dag:
    ...
```

그리고 앞서 import한 `BashOperator`/`BASE_DIR`와 DAG context 안에서 다음 Task 설정을 추가한다. 아래는 독립 실행 파일이 아닌 부분 예제다.

```python
hourly_job = BashOperator(
    task_id="hourly_job",
    bash_command=(
        f"set -euo pipefail; python {BASE_DIR}/hourly_job.py "
        "--kst-date '{{ data_interval_start.in_timezone('Asia/Seoul').format('YYYY-MM-DD') }}' "
        "--kst-hour '{{ data_interval_start.in_timezone('Asia/Seoul').format('HH') }}' "
        "--kst-start '{{ data_interval_start.in_timezone('Asia/Seoul').to_iso8601_string() }}' "
        "--kst-end '{{ data_interval_end.in_timezone('Asia/Seoul').to_iso8601_string() }}'"
    ),
)
```

예를 들어 Airflow UI에서 UTC로 `2026-05-01 15:00`처럼 보이는 실행이 한국 시간으로는 `2026-05-02 00:00`일 수 있다. 이때 원본 `ds`를 그대로 쓰면 `2026-05-01`로 보일 수 있으므로, 한국 업무일 기준 파일 경로에는 `kst-date`를 사용한다.

```text
권장 output path:
s3://bucket/sales/dt=2026-05-02/hour=00/data.parquet

위험한 output path:
s3://bucket/sales/dt={{ ds }}/data.parquet
```

정리하면 다음과 같다.

| 필요 | 권장 값 |
|------|---------|
| Airflow UI에서 실행 추적만 필요 | 인자 없이 실행 가능 |
| UTC 기준 날짜가 필요 | `{{ ds }}` 사용 가능 |
| 한국 업무일이 필요 | `data_interval_start.in_timezone('Asia/Seoul').format('YYYY-MM-DD')` |
| 한국 업무시간이 필요 | `data_interval_start.in_timezone('Asia/Seoul').format('HH')` |
| 정확한 시간 범위 조회가 필요 | KST 변환된 `kst-start`, `kst-end` |

## hourly 작업의 인자 전달

매시간 실행하는 DAG라고 해서 항상 `ds`나 시간 인자를 넘겨야 하는 것은 아니다.

Airflow Web UI에서 성공/실패, 로그, 실행 시간을 추적하는 것만 목적이라면 인자 없이 실행해도 된다.

```python
hourly_job = BashOperator(
    task_id="hourly_job",
    bash_command=f"set -euo pipefail; python {BASE_DIR}/hourly_job.py",
)
```

이 방식은 가장 단순하다. 스크립트가 내부에서 "현재 처리할 대상"을 스스로 결정한다면 충분할 수 있다.

하지만 다음 상황이면 인자를 넘기는 것이 맞다.

- 실패한 특정 시간 구간만 다시 처리해야 한다.
- 과거 시간대를 backfill해야 한다.
- output path가 `dt=2026-05-02/hour=10`처럼 시간 파티션이다.
- DB/API 조회 조건이 `event_time >= start_ts and event_time < end_ts` 형태다.
- retry 시 "현재 시간"이 아니라 원래 실패한 시간 구간을 다시 처리해야 한다.

매시간 실행하는 DAG에서 `{{ ds }}`만 넘기면 같은 날짜의 24개 실행을 구분하기 어렵다. hourly 작업이 데이터 구간을 알아야 한다면 처리 구간의 시작과 끝을 넘기는 방식이 가장 명확하다.

```python
hourly_job = BashOperator(
    task_id="hourly_job",
    bash_command=(
        f"set -euo pipefail; python {BASE_DIR}/hourly_job.py "
        "--start-ts '{{ data_interval_start }}' "
        "--end-ts '{{ data_interval_end }}'"
    ),
)
```

Python 파일은 이렇게 받는다.

```python
import argparse


def main(start_ts: str, end_ts: str) -> None:
    print(f"process rows where {start_ts} <= event_time < {end_ts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-ts", required=True)
    parser.add_argument("--end-ts", required=True)
    args = parser.parse_args()

    main(start_ts=args.start_ts, end_ts=args.end_ts)
```

파일 경로가 날짜/시간 파티션이면 날짜와 시간을 나눠 넘겨도 된다.

```python
hourly_job = BashOperator(
    task_id="hourly_job",
    bash_command=(
        f"set -euo pipefail; python {BASE_DIR}/hourly_job.py "
        "--date {{ ds }} "
        "--hour {{ data_interval_start.strftime('%H') }}"
    ),
)
```

기준은 다음과 같다.

| 처리 단위 | 적합한 인자 |
|----------|-------------|
| 하루 단위 파일/집계 | `--date {{ ds }}` |
| 한 시간 단위 파일/집계 | `--date {{ ds }} --hour {{ data_interval_start.strftime('%H') }}` |
| DB/API에서 시간 범위 조회 | `--start-ts '{{ data_interval_start }}' --end-ts '{{ data_interval_end }}'` |
| 재실행과 backfill이 중요함 | `--start-ts`, `--end-ts` 권장 |

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

`catchup=False`의 의미는 "과거에 놓친 모든 스케줄 구간을 자동으로 따라잡지 않는다"는 뜻이다. 실패한 Task를 무시한다는 뜻은 아니다.

예를 들어 hourly DAG가 있고 Airflow가 10:00부터 13:30까지 멈췄다고 하자.

```python
schedule="@hourly"
catchup=False
```

이 경우 Scheduler가 다시 살아났을 때 10시, 11시, 12시 구간을 모두 만들지 않고 최신 구간 중심으로 DAG Run을 만든다. 이후에는 다음 정규 스케줄을 기다린다.

반대로 `catchup=True`이면 누락된 시간 구간별 DAG Run을 가능한 만큼 만든다.

```text
catchup=True
  -> 10:00 구간, 11:00 구간, 12:00 구간, 13:00 구간별 run을 생성 대상으로 삼음; 실제 시작/완료 순서는 보장하지 않음

catchup=False
  -> 과거 누락 구간 전체를 자동 backfill하지 않음
```

구분이 중요하다.

| 상황 | `catchup=False` 동작 |
|------|---------------------|
| DAG를 오랫동안 pause했다가 켬 | 과거 전체 기간을 자동 생성하지 않음 |
| Scheduler가 멈췄다가 복구됨 | 누락된 모든 interval을 자동으로 만들지 않음 |
| 이미 만들어진 DAG Run의 Task가 실패함 | 실패 상태로 남음. retry 또는 수동 clear 필요 |
| 특정 시간 데이터를 반드시 처리해야 함 | 수동 실행, backfill, 또는 `catchup=True` 검토 |

hourly 데이터에서 누락이 절대 있으면 안 된다면 `catchup=False`만 믿으면 안 된다. 수동 backfill 절차를 만들거나, `catchup=True`와 `max_active_runs`, pool을 조합해 과거 구간을 안전하게 따라잡도록 설계한다.

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
- output path를 처리 구간별로 고정하고 overwrite/upsert 전략을 정해야 한다.

## timeout

`execution_timeout`은 각 Task 실행 시도에 대한 제한이다. retry 전체 시간이나 외부 제출 job 취소를 보장하지 않는다. HTTP/DB client timeout과 외부 job 정리도 따로 설정한다.

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

`all_done` cleanup이 유일한 leaf이고 성공하면 중간 Task가 실패해도 DAG Run이 성공으로 계산될 수 있다. cleanup 성공과 업무 성공을 구분해 leaf 구조를 확인한다. `one_failed`는 upstream의 `failed`/`upstream_failed` 조건이며 skip만으로 실패 알림을 대신하지 않는다.

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

위 TaskFlow 함수는 경로 생성/print만 하는 인터페이스 예제이며 실제 파일을 읽거나 쓰지 않는다. 함수의 반환값은 XCom을 통해 다음 Task로 전달된다. 반환값은 작은 문자열이나 dict 정도로 유지한다.

## 운영용 기본값 예시

앞선 import를 전제로 한 선언 골격이다. Task가 없으며 이 설정만으로 업무가 실행되지 않는다. `max_active_runs=1`은 같은 DAG의 active run 제한이지 다른 DAG·외부 writer와의 전역 lock이 아니다.

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
- 처리 날짜 또는 처리 시간 구간을 Airflow 값으로 받고 있는가
- 병렬 Task가 같은 output에 쓰지 않는가

## 다음 단계

다음 문서에서는 Task 간 데이터 전달, XCom, 파일 저장소, 멱등성을 정리한다.

- [04. 데이터와 상태 관리](./04-data-and-state.md)


## 검증 근거 — 2026-10-04

- [2.10.5 Time Zones](https://airflow.apache.org/docs/apache-airflow/2.10.5/authoring-and-scheduling/timezone.html), [템플릿](https://airflow.apache.org/docs/apache-airflow/2.10.5/templates-ref.html).
- [DAG Run·leaf 상태·catchup](https://airflow.apache.org/docs/apache-airflow/2.10.5/core-concepts/dag-run.html), [Task timeout](https://airflow.apache.org/docs/apache-airflow/2.10.5/core-concepts/tasks.html).

미확인: 사내 배포·계정·리소스·네트워크 조건과 실제 운영 성공. 중복 예제의 계약 통합·회사 정책 결정은 Herdr `pane_not_found`로 Claude 협의를 보류한다. [주제 정리 기록](../organization-log.md)에 진행 결과를 남긴다.

로컬 확인: Python 3.12.12/Airflow 2.10.5 임시 환경에서 이 장의 Python 구문과 완성 DAG 정의를 검사했다. Kubernetes provider import·실제 venv job·외부 접속·scheduler 실행은 별도 미확인이다. 추가 실행 결과와 판본은 위 정리 기록을 읽는다.
