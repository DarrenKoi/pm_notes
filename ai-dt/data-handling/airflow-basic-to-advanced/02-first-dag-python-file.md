---
tags: [airflow, bashoperator, pythonoperator, first-dag]
level: beginner
last_updated: 2026-05-02
status: complete
---

# 02. 첫 번째 DAG 만들기

## 목표

기존 Python 파일 하나를 Airflow에서 실행한다.

처음에는 복잡하게 생각하지 말고 아래 흐름만 성공시키면 된다.

```text
내 Python 파일 -> Airflow DAG -> 수동 실행 -> 로그 확인
```

## 예제 Python 파일

먼저 기존 코드가 아래처럼 실행 가능하다고 가정한다.

```bash
python hello_job.py --date 2026-05-02
```

Airflow에 올릴 Python 파일은 실행 인자를 명확히 받는 형태가 좋다.

```python
# hello_job.py
import argparse


def main(run_date: str) -> None:
    print(f"Hello Airflow. run_date={run_date}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args()

    main(args.date)
```

이 구조가 중요한 이유:

- 로컬에서도 실행할 수 있다.
- Airflow에서도 같은 방식으로 실행할 수 있다.
- 처리 날짜를 명시적으로 받을 수 있다.
- 테스트하기 쉽다.

## 방식 A. BashOperator로 실행

기존 Python 파일을 거의 그대로 실행하려면 `BashOperator`가 가장 쉽다.

```python
# dags/hello_job_dag.py
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


SCRIPT_PATH = "/opt/airflow/dags/jobs/hello_job.py"


with DAG(
    dag_id="hello_job_dag",
    start_date=datetime(2026, 5, 1),
    schedule=None,
    catchup=False,
    tags=["tutorial"],
) as dag:
    run_hello = BashOperator(
        task_id="run_hello",
        bash_command=f"set -euo pipefail; python {SCRIPT_PATH} --date {{{{ ds }}}}",
    )
```

`schedule=None`은 자동 스케줄 없이 수동 실행만 하겠다는 뜻이다. 처음 테스트할 때는 이 설정이 안전하다.

`{{ ds }}`는 Airflow가 실행 날짜를 `YYYY-MM-DD` 형식으로 넣어주는 Jinja template이다.

`set -euo pipefail`은 shell command가 중간에 실패했을 때 Task가 정상 성공으로 표시되는 일을 줄이기 위한 방어 장치다.

## 방식 B. PythonOperator로 실행

Python 파일이 함수 형태로 정리되어 있다면 `PythonOperator`를 사용할 수 있다.

```text
dags/
├── hello_job_dag.py
└── jobs/
    ├── __init__.py
    └── hello_job.py
```

```python
# dags/jobs/hello_job.py
def main(run_date: str) -> None:
    print(f"Hello Airflow. run_date={run_date}")
```

```python
# dags/hello_job_dag.py
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from jobs.hello_job import main


with DAG(
    dag_id="hello_job_python_operator",
    start_date=datetime(2026, 5, 1),
    schedule=None,
    catchup=False,
    tags=["tutorial"],
) as dag:
    run_hello = PythonOperator(
        task_id="run_hello",
        python_callable=main,
        op_kwargs={"run_date": "{{ ds }}"},
    )
```

장점:

- shell command 문자열이 줄어든다.
- Python 함수 단위로 테스트하기 쉽다.
- XCom 반환값을 다루기 쉽다.

단점:

- Python import 경로가 Airflow 서버에서 맞아야 한다.
- DAG import 시점에 무거운 import가 실행되지 않도록 주의해야 한다.

## 방식 C. TaskFlow API로 실행

Airflow 2.x에서는 `@task` 방식도 많이 사용한다.

```python
from datetime import datetime

from airflow.decorators import dag, task


@dag(
    dag_id="hello_job_taskflow",
    start_date=datetime(2026, 5, 1),
    schedule=None,
    catchup=False,
    tags=["tutorial"],
)
def hello_pipeline():
    @task
    def run_hello(run_date: str) -> None:
        from jobs.hello_job import main

        main(run_date)

    run_hello("{{ ds }}")


hello_pipeline()
```

`@task` 내부에서 import하는 이유는 DAG 파싱을 가볍게 만들기 위해서다. `pandas`, `torch`, `tensorflow`처럼 import가 무거운 패키지는 Task 내부에서 import하는 것이 안전하다.

## DAG 업로드 후 확인할 것

회사 Airflow UI에서 다음을 확인한다.

1. DAG 목록에 `hello_job_dag`가 보이는가
2. Import Error가 없는가
3. 수동 실행이 가능한가
4. Task 로그에 `Hello Airflow`가 찍히는가
5. 실패 시 traceback이 로그에 보이는가

## DAG가 UI에 안 보일 때

가능한 원인:

- Python syntax error
- import error
- Airflow 버전과 맞지 않는 import 경로
- DAG 파일이 올바른 폴더에 없음
- `dag_id` 중복
- Scheduler가 아직 DAG 파일을 파싱하지 않음

먼저 Airflow UI의 import error 메뉴나 Scheduler 로그를 확인한다.

## 로컬에서 최소 검증

Airflow 서버에 올리기 전에 로컬에서 Python 파일 자체를 먼저 확인한다.

```bash
python hello_job.py --date 2026-05-02
```

Airflow가 로컬에 설치되어 있다면 DAG import도 확인한다.

```bash
python dags/hello_job_dag.py
```

이 명령이 아무 에러 없이 끝나야 한다. 단, 실제 Task 실행까지 검증하는 것은 아니다.

Airflow CLI가 있다면 Task 단위 테스트도 가능하다.

```bash
airflow tasks test hello_job_dag run_hello 2026-05-02
```

회사 관리형 Airflow에서는 CLI 접근 권한이 없을 수 있다. 그 경우 UI에서 수동 실행하고 로그를 확인한다.

## 첫 DAG에서 하지 말아야 할 것

- 처음부터 매일 자동 스케줄을 켜지 않는다.
- 여러 파일을 한 번에 모두 연결하지 않는다.
- 패키지 설치까지 동시에 해결하려고 하지 않는다.
- secret을 코드에 넣지 않는다.
- 로컬 절대 경로를 사용하지 않는다.

먼저 작은 Task 하나를 성공시키고, 그 다음에 단계를 늘린다.

## 다음 단계

다음 문서에서는 여러 Python 파일을 순서대로 연결하고, 실패/재시도/스케줄을 설정한다.

- [03. 의존성, 스케줄, 재시도](./03-dependencies-scheduling-retry.md)
