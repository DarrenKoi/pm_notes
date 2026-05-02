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
- 처리 날짜 또는 처리 시간 구간을 명시적으로 받을 수 있다.
- 테스트하기 쉽다.

## 방식 A. BashOperator로 실행

기존 Python 파일을 거의 그대로 실행하려면 `BashOperator`가 가장 쉽다.

스크립트가 인자를 필요로 하지 않고, Airflow Web UI에서 실행 성공/실패만 추적하면 되는 경우에는 인자 없이 실행해도 된다.

```python
# dags/simple_job_dag.py
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


SCRIPT_PATH = "/opt/airflow/dags/jobs/simple_job.py"


with DAG(
    dag_id="simple_job_dag",
    start_date=datetime(2026, 5, 1),
    schedule="@hourly",
    catchup=False,
    tags=["tutorial"],
) as dag:
    run_simple = BashOperator(
        task_id="run_simple",
        bash_command=f"set -euo pipefail; python {SCRIPT_PATH}",
    )
```

이 경우에도 Airflow UI에서는 매시간 DAG Run, Task 상태, 로그를 볼 수 있다.

다만 스크립트가 특정 날짜나 시간대 데이터를 처리해야 한다면 인자를 넘긴다.

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

`{{ ds }}`는 Airflow가 실행 날짜를 `YYYY-MM-DD` 형식으로 넣어주는 Jinja template이다. daily 작업에는 이 정도로 충분할 수 있다.

매시간 실행하는 파일이면 날짜만 넘기지 말고 시간 구간을 넘긴다.

```python
run_hourly = BashOperator(
    task_id="run_hourly",
    bash_command=(
        f"set -euo pipefail; python {SCRIPT_PATH} "
        "--start-ts '{{ data_interval_start }}' "
        "--end-ts '{{ data_interval_end }}'"
    ),
)
```

Python 파일도 hourly 인자를 받도록 만든다.

```python
# hourly_job.py
import argparse


def main(start_ts: str, end_ts: str) -> None:
    print(f"process interval: {start_ts} <= data < {end_ts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-ts", required=True)
    parser.add_argument("--end-ts", required=True)
    args = parser.parse_args()

    main(start_ts=args.start_ts, end_ts=args.end_ts)
```

`set -euo pipefail`은 shell command가 중간에 실패했을 때 Task가 정상 성공으로 표시되는 일을 줄이기 위한 방어 장치다.

각 옵션의 의미:

- `set`: 현재 shell 실행 옵션을 바꾼다.
- `-e`: command가 실패하면 즉시 shell을 종료한다.
- `-u`: 정의되지 않은 변수를 사용하면 에러로 처리한다.
- `-o pipefail`: pipe로 연결된 command 중 하나라도 실패하면 전체 command를 실패로 처리한다.

예를 들어 아래 command는 `python`이 실패해도 `tee`가 성공하면 전체 command가 성공처럼 보일 수 있다.

```bash
python hello_job.py --date 2026-05-02 | tee output.log
```

`pipefail`을 켜면 앞쪽의 `python` 실패도 Airflow Task 실패로 전달된다.

처음에는 아래 정도로 이해하면 된다.

```bash
set -euo pipefail
```

의미:

```text
shell script 안에서 실패를 조용히 넘기지 말고, 가능한 빨리 실패로 멈춰라.
```

단, 이것은 `BashOperator`처럼 shell command를 실행할 때 필요한 설정이다. `PythonOperator`는 shell을 거치지 않고 Python 함수를 직접 실행하므로 `set -euo pipefail`을 쓰지 않는다.

## 방식 B. PythonOperator로 실행

Python 파일이 함수 형태로 정리되어 있다면 `PythonOperator`를 사용할 수 있다.

`PythonOperator`는 "shell command 문자열을 실행"하는 방식이 아니라, Airflow Worker 안에서 Python 함수를 직접 호출하는 방식이다.

```text
BashOperator
  -> bash command 실행
  -> python hello_job.py --date 2026-05-02

PythonOperator
  -> Python 함수 직접 실행
  -> main(run_date="2026-05-02")
```

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

여기서 중요한 부분:

- `python_callable=main`: 실행할 Python 함수를 지정한다.
- `op_kwargs={"run_date": "{{ ds }}"}`: `main(run_date="...")`처럼 keyword argument를 넘긴다.
- `{{ ds }}`: Airflow가 Task 실행 시점에 날짜 문자열로 바꿔준다.

실제 실행 시점에는 아래처럼 호출되는 것과 비슷하다.

```python
main(run_date="2026-05-02")
```

`PythonOperator`를 쓰려면 기존 Python 파일을 CLI 전용 구조가 아니라, import 가능한 함수 구조로 만드는 것이 좋다.

```python
# jobs/hello_job.py
def main(run_date: str) -> None:
    print(f"Hello Airflow. run_date={run_date}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args()

    main(args.date)
```

이렇게 만들면 두 방식이 모두 가능하다.

로컬 CLI 실행:

```bash
python jobs/hello_job.py --date 2026-05-02
```

Airflow `PythonOperator` 실행:

```python
PythonOperator(
    task_id="run_hello",
    python_callable=main,
    op_kwargs={"run_date": "{{ ds }}"},
)
```

시간 구간이 필요한 작업도 `op_kwargs`로 넘길 수 있다.

```python
run_hourly = PythonOperator(
    task_id="run_hourly",
    python_callable=process_interval,
    op_kwargs={
        "start_ts": "{{ data_interval_start }}",
        "end_ts": "{{ data_interval_end }}",
    },
)
```

함수는 아래처럼 받는다.

```python
def process_interval(start_ts: str, end_ts: str) -> None:
    print(f"process interval: {start_ts} <= data < {end_ts}")
```

Airflow context를 함수 안에서 직접 읽을 수도 있다.

```python
from airflow.operators.python import get_current_context


def main() -> None:
    context = get_current_context()
    run_date = context["ds"]
    print(f"Hello Airflow. run_date={run_date}")
```

다만 처음에는 `get_current_context()`보다 `op_kwargs`로 필요한 값만 명시적으로 넘기는 방식이 더 이해하기 쉽다.

장점:

- shell command 문자열이 줄어든다.
- Python 함수 단위로 테스트하기 쉽다.
- XCom 반환값을 다루기 쉽다.
- Python 예외가 그대로 Task 실패로 기록된다.

단점:

- Python import 경로가 Airflow 서버에서 맞아야 한다.
- DAG import 시점에 무거운 import가 실행되지 않도록 주의해야 한다.
- 필요한 Python 패키지가 Airflow Worker 환경에 설치되어 있어야 한다.
- 큰 데이터 자체를 return 값으로 넘기면 XCom이 커지므로 피해야 한다.

## 방식 C. TaskFlow API로 실행

Airflow 2.x에서는 `@task` 방식도 많이 사용한다.

`@task`의 의미는 "이 Python 함수를 Airflow Task로 바꾼다"는 뜻이다.

중요한 점:

- `@task`가 붙은 함수는 DAG 안에서 Task가 된다.
- DAG 파일을 파싱할 때 함수 본문이 바로 실행되는 것은 아니다.
- DAG 안에서 `run_hello()`처럼 호출하면 실제 Python 실행이 아니라 Task 정의가 만들어진다.
- 실제 함수 본문은 Scheduler가 해당 Task를 Worker에 보냈을 때 실행된다.
- 함수의 `return` 값은 자동으로 XCom에 저장된다.
- 다른 `@task` 함수의 인자로 넘기면 Airflow가 Task 의존성을 자동으로 만든다.

일반 Python 관점과 Airflow 관점이 다르다.

```python
@task
def extract() -> str:
    print("this runs on worker")
    return "s3://bucket/raw/data.csv"


raw_path = extract()
```

위 코드에서 `raw_path = extract()`는 DAG 파싱 시점에 `print()`를 실행하지 않는다. `raw_path`는 실제 문자열이 아니라 "extract Task가 나중에 반환할 값"을 가리키는 Airflow 객체다.

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

TaskFlow API에서 의존성은 함수 호출처럼 표현할 수 있다.

```python
@task
def download() -> str:
    return "minio://bucket/raw.csv"


@task
def preprocess(raw_path: str) -> str:
    return raw_path.replace("raw", "clean")


@task
def report(clean_path: str) -> None:
    print(clean_path)


raw = download()
clean = preprocess(raw)
report(clean)
```

이 코드는 `download -> preprocess -> report` 순서로 실행된다. `raw`와 `clean`은 실제 파일 내용이 아니라 XCom으로 전달되는 작은 값이다.

기존 `.py` 파일을 그냥 실행만 하고 싶으면 `BashOperator`가 더 단순하다. Python 함수 단위로 DAG를 깔끔하게 구성하고 반환값을 다음 Task에 넘기고 싶으면 `@task`가 편하다.

## Bitbucket Git Sync 후 확인할 것

현재 사내 환경에서는 DAG 파일을 서버에 직접 업로드하지 않고 Bitbucket repository에 push한다. Airflow는 지정된 repository와 branch를 Git Sync로 읽는다.

배포 흐름:

```text
local edit
  -> git commit
  -> git push Bitbucket
  -> Airflow Git Sync
  -> Scheduler DAG parse
  -> Airflow UI 확인
```

회사 Airflow UI에서 다음을 확인한다.

1. DAG 목록에 `hello_job_dag`가 보이는가
2. Import Error가 없는가
3. 수동 실행이 가능한가
4. Task 로그에 `Hello Airflow`가 찍히는가
5. 실패 시 traceback이 로그에 보이는가

push 직후 바로 반영되지 않을 수 있다. Git Sync interval이 몇 분인지 운영팀에 확인한다.

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
- secret을 여러 코드 파일에 흩뿌리지 않는다. 현재 환경에서는 `secrets.py`에 모으고 로그에 찍지 않는다.
- 로컬 절대 경로를 사용하지 않는다.

먼저 작은 Task 하나를 성공시키고, 그 다음에 단계를 늘린다.

## 다음 단계

다음 문서에서는 여러 Python 파일을 순서대로 연결하고, 실패/재시도/스케줄을 설정한다.

- [03. 의존성, 스케줄, 재시도](./03-dependencies-scheduling-retry.md)
