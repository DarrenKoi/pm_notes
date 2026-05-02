---
tags: [airflow, local-development, testing, ci, deployment]
level: intermediate
last_updated: 2026-05-02
---

# 06. 로컬 개발과 테스트

## 목표

Airflow DAG를 회사 서버에 올리기 전에 로컬에서 최대한 문제를 줄인다.

검증은 세 단계로 나눈다.

```text
1. 순수 Python 코드 테스트
2. DAG import 테스트
3. Airflow Task 실행 테스트
```

회사 관리형 Airflow에서는 로컬과 서버 환경이 완전히 같지 않을 수 있다. 그래도 로컬 검증을 해두면 syntax error, import error, 인자 누락, 멱등성 문제를 많이 줄일 수 있다.

현재 환경에서는 배포가 Bitbucket Git Sync로 이루어진다. 따라서 로컬 검증 후 Bitbucket에 push하고, Airflow가 해당 branch를 sync했는지 UI에서 확인하는 흐름을 기준으로 한다.

## 추천 프로젝트 구조

작은 프로젝트:

```text
airflow-project/
├── dags/
│   ├── daily_pipeline.py
│   └── jobs/
│       ├── __init__.py
│       ├── download.py
│       ├── preprocess.py
│       └── analyze.py
├── tests/
│   ├── test_download.py
│   └── test_dag_import.py
└── requirements-app.txt
```

현재처럼 Connection/Variable 접근이 불가능한 환경에서는 secret/config 파일을 DAG package 안에 둔다.

```text
airflow-project/
├── dags/
│   ├── hourly_pipeline.py
│   └── company_job/
│       ├── __init__.py
│       ├── config.py
│       ├── secrets.py
│       ├── clients.py
│       └── jobs/
│           ├── __init__.py
│           ├── download.py
│           └── preprocess.py
├── tests/
└── requirements-app.txt
```

조금 큰 프로젝트:

```text
airflow-project/
├── dags/
│   └── daily_pipeline.py
├── src/
│   └── company_jobs/
│       ├── __init__.py
│       ├── download.py
│       ├── preprocess.py
│       └── analyze.py
├── tests/
├── pyproject.toml
└── requirements-app.txt
```

두 번째 구조는 Python package로 관리하기 좋지만, 회사 Airflow에서 `src/` 패키지를 어떻게 배포할 수 있는지 확인해야 한다. Git Sync가 `dags/`만 sync하거나 `PYTHONPATH`가 repository root를 포함하지 않으면 `src/` import가 실패할 수 있다.

## Python 파일은 main 함수로 분리

나쁜 구조:

```python
# preprocess.py
import pandas as pd

df = pd.read_csv("/data/input.csv")
df.to_parquet("/data/output.parquet")
```

이 파일은 import하는 순간 실행된다. DAG에서 import하면 Scheduler가 파일을 읽을 때마다 작업이 실행될 수 있다.

좋은 구조:

```python
# preprocess.py
def main(run_date: str, input_path: str, output_path: str) -> None:
    import pandas as pd

    df = pd.read_csv(input_path)
    df.to_parquet(output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    main(
        run_date=args.date,
        input_path=args.input_path,
        output_path=args.output_path,
    )
```

이 구조는 로컬 CLI 실행과 Airflow 함수 호출을 모두 지원한다.

hourly 작업은 날짜만 받지 말고 시간 구간을 받는다.

```python
# hourly_preprocess.py
def main(start_ts: str, end_ts: str, output_path: str) -> None:
    print(f"process {start_ts} <= event_time < {end_ts}")
    print(f"output={output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start-ts", required=True)
    parser.add_argument("--end-ts", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    main(
        start_ts=args.start_ts,
        end_ts=args.end_ts,
        output_path=args.output_path,
    )
```

## 순수 Python 테스트

Airflow 없이 먼저 Python 함수 자체를 테스트한다.

```python
# tests/test_preprocess.py
from company_jobs.preprocess import main


def test_preprocess_creates_output(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.parquet"

    input_path.write_text("id,value\n1,10\n2,20\n", encoding="utf-8")

    main(
        run_date="2026-05-02",
        input_path=str(input_path),
        output_path=str(output_path),
    )

    assert output_path.exists()
```

실행:

```bash
pytest tests/
```

Airflow 없이 테스트할 수 있는 로직이 많을수록 유지보수가 쉬워진다.

## DAG import 테스트

DAG 파일이 import되는지 확인한다.

```bash
python dags/daily_pipeline.py
```

이 명령에서 에러가 나면 Airflow UI에서도 DAG가 보이지 않거나 import error가 발생할 가능성이 높다.

pytest로도 확인할 수 있다.

```python
# tests/test_dag_import.py
import importlib.util
from pathlib import Path


def test_dag_imports_without_error():
    dag_file = Path("dags/daily_pipeline.py")
    spec = importlib.util.spec_from_file_location("daily_pipeline", dag_file)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
```

## Airflow CLI 테스트

로컬에 Airflow가 설치되어 있으면 DAG 목록과 Task 실행을 테스트한다.

```bash
airflow dags list
airflow tasks list daily_pipeline
airflow tasks test daily_pipeline preprocess 2026-05-02
```

`airflow tasks test`는 특정 Task를 단독 실행한다. Scheduler가 없어도 Task 로직을 확인할 수 있어 유용하다.

단, 회사 서버와 로컬 패키지, Bitbucket sync branch, secret 파일 내용이 다르면 로컬 성공이 서버 성공을 보장하지는 않는다.

## Secret과 설정 mocking

순수 Python 테스트에서는 운영 secret 파일을 직접 사용하지 않도록 설계하는 것이 좋다.

좋은 구조:

```python
def run_query(db_config: dict, sql: str) -> list[dict]:
    ...
```

Airflow Task 또는 client factory에서만 `secrets.py`를 읽는다.

```python
def task_main():
    from company_job.clients import get_db_config

    db_config = get_db_config()
    run_query(db_config, "select 1")
```

이렇게 하면 `run_query()`는 Airflow와 운영 secret 없이 테스트할 수 있다.

테스트에서는 fake config를 넘긴다.

```python
def test_run_query_builds_sql():
    fake_db_config = {
        "host": "localhost",
        "user": "test",
        "password": "test",
        "database": "test",
    }
    result = run_query(fake_db_config, "select 1")
    assert result is not None
```

## requirements 관리

운영 실행 패키지:

```text
requirements-app.txt
```

테스트/개발 패키지:

```text
requirements-dev.txt
```

예:

```text
# requirements-app.txt
pandas==2.2.2
numpy==1.26.4
requests==2.32.3
pyarrow==16.1.0
```

```text
# requirements-dev.txt
-r requirements-app.txt
pytest==8.2.2
ruff==0.5.0
```

Airflow 서버에 올릴 패키지는 `requirements-app.txt` 기준으로 운영팀과 협의한다.

## lint

Python 코드 품질은 최소한 아래 정도를 확인한다.

```bash
ruff check .
```

Airflow DAG 전용 규칙을 쓰는 경우도 있다.

```bash
ruff check dags/ --select AIR
```

사용 가능한 rule은 ruff 버전에 따라 다를 수 있다.

## 배포 전 수동 체크리스트

코드:

- Python 파일이 `main()` 함수로 분리되어 있는가
- import하는 순간 작업이 실행되지 않는가
- 처리 날짜 또는 처리 시간 구간을 인자로 받는가
- 실패 시 예외를 발생시키는가
- 현재 환경에서는 secret이 `secrets.py` 한 곳에 모여 있고 로그에 찍히지 않는가

DAG:

- `dag_id`가 중복되지 않는가
- `schedule`과 `catchup` 의도가 맞는가
- Task 의존성이 명확한가
- timeout이 설정되어 있는가
- retry를 켜도 재실행 안전한가
- XCom에 큰 데이터를 넣지 않는가

환경:

- Airflow 서버 Python 버전을 확인했는가
- 필요한 패키지 버전을 확인했는가
- provider 설치 여부를 확인했는가
- Airflow가 sync하는 Bitbucket repository와 branch를 확인했는가
- `secrets.py`와 `config.py`가 Airflow에서 import 가능한 위치에 있는가
- Worker에서 접근 가능한 storage path를 확인했는가

## 배포 후 확인 순서

1. Bitbucket에 push한 commit이 Airflow Git Sync 대상 branch에 있는지 확인
2. Git Sync interval만큼 기다림
3. DAG가 UI에 보이는지 확인
4. Import Error가 없는지 확인
5. `schedule=None` 또는 pause 상태에서 수동 실행
6. 첫 Task 로그 확인
7. output path 생성 확인
8. 실패 Task만 Clear 후 재실행 테스트
9. 전체 DAG 재실행 시 중복 결과가 생기지 않는지 확인
10. 운영 스케줄 활성화

## 회사 Airflow에서 CLI가 없을 때

일반 사용자는 Airflow CLI 접근이 없을 수 있다. 이 경우 UI와 debug DAG로 확인한다.

대체 방법:

- DAG import error 화면 확인
- Task 로그 확인
- 환경 조사 DAG 실행
- 작은 smoke test DAG 실행
- 운영팀에 Scheduler/Worker 로그 요청

## Smoke test DAG

운영 DAG를 켜기 전에 간단한 smoke test를 만든다.

```python
from datetime import datetime

from airflow.decorators import dag, task


@dag(
    dag_id="company_smoke_test",
    start_date=datetime(2026, 5, 1),
    schedule=None,
    catchup=False,
    tags=["debug"],
)
def smoke_test():
    @task
    def check_imports() -> None:
        import pandas as pd
        import requests

        print(f"pandas={pd.__version__}")
        print(f"requests={requests.__version__}")

    @task
    def check_storage() -> None:
        print("write/read small test file or call storage health check here")

    check_imports() >> check_storage()


smoke_test()
```

조사가 끝나면 debug DAG는 제거하거나 pause한다.

## 다음 단계

다음 문서에서는 Sensor, Dataset, Dynamic Task Mapping, pool, backfill 같은 고급 운영 패턴을 다룬다.

- [07. 고급 운영 패턴](./07-advanced-operations.md)
