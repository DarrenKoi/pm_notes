---
tags: [airflow, xcom, storage, idempotency, connections]
level: intermediate
last_updated: 2026-05-02
status: complete
---

# 04. 데이터와 상태 관리

## 목표

Airflow에서 여러 Task가 데이터를 주고받을 때 어떤 방식이 안전한지 이해한다.

핵심 원칙은 간단하다.

```text
큰 데이터는 외부 저장소에 저장하고, Airflow에는 경로만 전달한다.
```

## 왜 로컬 파일 전달이 위험한가

아래 구조는 로컬에서 잘 동작한다.

```text
download.py -> /tmp/raw.csv 생성
preprocess.py -> /tmp/raw.csv 읽기
```

하지만 회사 Airflow에서는 실패할 수 있다.

이유:

- Task가 서로 다른 Worker에서 실행될 수 있다.
- KubernetesExecutor에서는 Task마다 다른 Pod에서 실행될 수 있다.
- `/tmp`는 Task 종료 후 사라질 수 있다.
- retry 시 이전 임시 파일이 남아 결과를 오염시킬 수 있다.

따라서 Task 간 파일 전달에는 공유 저장소를 사용한다.

| 저장소 | 사용 상황 |
|--------|----------|
| S3/MinIO | 객체 파일, parquet, json, csv |
| NAS/NFS | 사내 공유 파일 시스템 |
| DB/Data Warehouse | 정형 데이터 |
| Redis/Queue | 작은 이벤트성 메시지 |

## 좋은 데이터 흐름

```text
download
  -> s3://raw/company_job/dt=2026-05-02/data.parquet

preprocess
  -> s3://clean/company_job/dt=2026-05-02/data.parquet

analyze
  -> s3://result/company_job/dt=2026-05-02/result.json

report
  -> s3://report/company_job/dt=2026-05-02/report.xlsx
```

각 Task는 입력 경로와 출력 경로를 명시적으로 가진다.

```python
def preprocess(run_date: str, raw_path: str) -> str:
    clean_path = f"s3://clean/company_job/dt={run_date}/data.parquet"
    # raw_path에서 읽고 clean_path에 저장
    return clean_path
```

## XCom 사용 기준

XCom에 넣기 좋은 값:

- 파일 경로
- row count
- 처리 상태
- 작은 dict
- model version
- report URL

XCom에 넣지 말아야 할 값:

- DataFrame
- 대용량 JSON
- 파일 binary
- 이미지
- 모델 파일
- 수십 MB 이상의 문자열

TaskFlow 예:

```python
from airflow.decorators import task


@task
def extract(run_date: str) -> str:
    output_path = f"s3://raw/sales/dt={run_date}/data.parquet"
    return output_path


@task
def transform(run_date: str, input_path: str) -> str:
    output_path = f"s3://clean/sales/dt={run_date}/data.parquet"
    return output_path


raw_path = extract("{{ ds }}")
clean_path = transform("{{ ds }}", raw_path)
```

`raw_path`와 `clean_path`는 실제 데이터가 아니라 경로 문자열이다.

## 멱등성

운영 배치에서 가장 중요한 성질 중 하나는 멱등성이다.

멱등성이란 같은 입력으로 여러 번 실행해도 결과가 같다는 뜻이다.

Airflow에서는 다음 상황 때문에 멱등성이 필요하다.

- Task retry
- 실패 Task만 Clear 후 재실행
- 과거 날짜 backfill
- 운영자가 수동 재실행

## 나쁜 예

```python
def load_to_db(rows):
    for row in rows:
        insert(row)
```

같은 날짜 또는 같은 시간 구간을 재실행하면 중복 insert가 생길 수 있다.

## 좋은 예

```python
def load_to_db(run_date: str, rows):
    delete_partition(run_date)
    insert_partition(run_date, rows)
```

또는 DB가 지원하면 upsert를 사용한다.

```python
def load_to_db(rows):
    upsert_by_primary_key(rows)
```

파일 출력도 마찬가지다.

```python
output_path = f"s3://clean/sales/dt={run_date}/data.parquet"
```

같은 `run_date`는 같은 경로에 쓰고, 재실행 시 덮어쓰기 또는 atomic replace 정책을 사용한다.

## 임시 경로와 최종 경로

긴 작업은 임시 경로에 먼저 쓰고 성공하면 최종 경로로 이동하는 방식이 안전하다.

```text
s3://clean/sales/_tmp/run_id=.../data.parquet
  -> 성공 후
s3://clean/sales/dt=2026-05-02/data.parquet
```

이렇게 하면 중간 실패로 불완전한 파일이 최종 경로에 남는 문제를 줄일 수 있다.

## 날짜/시간 파티션

Airflow에서는 처리 날짜 또는 처리 시간 구간을 명시적으로 받는다.

단, 모든 작업이 날짜/시간 인자를 필요로 하는 것은 아니다. Python 파일이 "현재 상태를 확인하고 필요한 일을 수행"하는 운영성 작업이라면 인자 없이 실행하고 Airflow UI에서 성공/실패만 추적해도 된다.

인자 없이 실행해도 괜찮은 예:

- 매시간 health check
- queue에 쌓인 일을 가능한 만큼 처리
- 외부 시스템의 최신 상태를 동기화
- 항상 최신 파일 하나만 처리
- 실패해도 다음 시간에 다시 최신 상태를 보면 되는 작업

인자가 필요한 예:

- `2026-05-02 10:00~11:00` 데이터만 처리
- 특정 날짜/시간 파티션 파일 생성
- 실패한 시간 구간만 다시 실행
- 과거 데이터를 backfill
- 같은 입력으로 재실행하면 같은 결과가 나와야 하는 데이터 파이프라인

```python
def main(run_date: str) -> None:
    input_path = f"s3://raw/sales/dt={run_date}/data.parquet"
    output_path = f"s3://clean/sales/dt={run_date}/data.parquet"
```

DAG에서는 이렇게 넘긴다.

```python
BashOperator(
    task_id="preprocess",
    bash_command="python preprocess.py --date {{ ds }}",
)
```

매시간 실행이면 날짜만으로는 부족하다.

```python
def main(start_ts: str, end_ts: str) -> None:
    # 예: start_ts = "2026-05-02T10:00:00+09:00"
    # 예: end_ts = "2026-05-02T11:00:00+09:00"
    ...
```

DAG에서는 처리 구간을 넘긴다.

```python
BashOperator(
    task_id="preprocess_hourly",
    bash_command=(
        "python preprocess.py "
        "--start-ts '{{ data_interval_start }}' "
        "--end-ts '{{ data_interval_end }}'"
    ),
)
```

파일 저장 경로는 시간 파티션까지 포함한다.

```python
def make_hourly_output_path(run_date: str, run_hour: str) -> str:
    return f"s3://clean/sales/dt={run_date}/hour={run_hour}/data.parquet"
```

중요한 처리 기준에는 `datetime.now()`를 쓰지 않는다.

나쁜 예:

```python
from datetime import datetime

run_date = datetime.now().strftime("%Y-%m-%d")
```

좋은 예:

```python
def main(run_date: str) -> None:
    ...
```

hourly 작업에서는 이렇게 받는다.

```python
def main(start_ts: str, end_ts: str) -> None:
    ...
```

## Secret 관리

일반적인 Airflow 운영에서는 DB, S3, MinIO, API key를 Airflow Connection에 저장하고 `conn_id`로 참조한다.

```python
from airflow.hooks.base import BaseHook


def get_db_config():
    conn = BaseHook.get_connection("warehouse_db")
    return {
        "host": conn.host,
        "port": conn.port,
        "user": conn.login,
        "password": conn.password,
        "schema": conn.schema,
    }
```

S3/MinIO는 회사에 따라 AWS provider의 S3Hook을 쓰거나 boto3/minio client를 직접 쓴다. provider가 설치되어 있는지 먼저 확인해야 한다.

하지만 현재 사내 환경에서는 Airflow Connection/Variable 접근이 불가능하므로 이 방식을 사용할 수 없다. 대신 Bitbucket Git Sync repository 안에 `secrets.py`와 `config.py`를 두는 방식으로 정리한다.

```text
dags/
└── company_job/
    ├── __init__.py
    ├── config.py
    ├── secrets.py
    ├── clients.py
    └── jobs/
```

예:

```python
# dags/company_job/secrets.py

WAREHOUSE_DB = {
    "host": "warehouse.company.internal",
    "port": 5432,
    "database": "analytics",
    "user": "batch_user",
    "password": "REPLACE_WITH_REAL_PASSWORD",
}
```

```python
# dags/company_job/clients.py

def get_db_config() -> dict:
    from company_job.secrets import WAREHOUSE_DB

    return WAREHOUSE_DB
```

주의:

- secret은 `secrets.py` 한 곳에 모은다.
- secret 값을 로그에 출력하지 않는다.
- Bitbucket repository 접근 권한을 최소화한다.
- secret이 노출되면 key rotation을 한다.

자세한 내용은 [08. Bitbucket Git Sync와 코드 기반 Secret 운영](./08-bitbucket-git-sync-and-code-secrets.md)을 따른다.

## 설정값 관리

일반적인 Airflow에서는 자주 바뀌는 설정을 Variable로 둘 수 있다.

```python
from airflow.models import Variable


def run():
    bucket = Variable.get("company_data_bucket")
```

하지만 DAG 파일 최상단에서 `Variable.get()`을 호출하는 것은 피한다. Scheduler가 DAG 파일을 반복적으로 파싱할 때마다 DB 조회가 발생할 수 있다.

가능하면 Task 내부에서 읽는다.

```python
def run():
    from airflow.models import Variable

    bucket = Variable.get("company_data_bucket")
```

또는 template을 사용한다.

```python
bash_command="python job.py --bucket {{ var.value.company_data_bucket }}"
```

현재 환경에서는 Variable 접근이 불가능하므로 설정값은 `config.py`에 둔다.

```python
# dags/company_job/config.py

RAW_PREFIX = "raw/sales"
CLEAN_PREFIX = "clean/sales"
DEFAULT_RETRIES = 2
```

## 로그에 남기면 안 되는 것

Task 로그는 운영자와 여러 사용자가 볼 수 있다.

로그에 남기지 말아야 할 값:

- password
- token
- access key
- secret key
- 주민번호/개인정보
- 내부망 민감 URL 전체

로그에 남기면 좋은 값:

- run_date
- input path
- output path
- row count
- elapsed time
- package version
- query id
- request id

## 실패 처리를 명확히 하기

Airflow는 Python 함수가 예외 없이 끝나면 성공으로 본다. 결과가 잘못되었는데도 예외를 삼키면 뒤 Task가 계속 실행된다.

나쁜 예:

```python
def main():
    try:
        run_job()
    except Exception as e:
        print(e)
```

좋은 예:

```python
def main():
    try:
        run_job()
    except Exception:
        logger.exception("job failed")
        raise
```

BashOperator에서는 Python script가 non-zero exit code로 종료되어야 실패로 잡힌다.

```python
subprocess.run(command, check=True)
```

## 데이터 관리 체크리스트

- Task 간 큰 데이터는 외부 저장소에 저장하는가
- XCom에는 작은 값만 넣는가
- 출력 경로가 날짜/시간 파티션을 포함하는가
- 같은 처리 구간 재실행 시 중복이 생기지 않는가
- 실패 중간 산출물이 최종 경로에 남지 않는가
- 현재 환경에서는 secret을 `secrets.py`에 모으고 Bitbucket 권한을 제한했는가
- 로그에 민감정보를 찍지 않는가
- 코드가 실패를 삼키지 않고 예외를 발생시키는가

## 다음 단계

다음 문서에서는 로컬 Python 환경과 Airflow 서버 환경이 다를 때 패키지 버전을 어떻게 맞출지 다룬다.

- [05. 패키지와 실행 환경](./05-packages-and-environments.md)
