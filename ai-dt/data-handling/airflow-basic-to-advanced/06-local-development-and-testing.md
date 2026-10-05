---
tags: [airflow, local-development, testing, ci, deployment]
level: intermediate
last_updated: 2026-05-02
reviewed_on: 2026-10-04
review_status: partial
document_type: learning_note
category_major: "AI·DT"
category_middle: "데이터 엔지니어링"
category_minor: "Airflow 파이프라인"
note_kind: "학습"
classified_on: "2026-10-05"
---

# 06. 로컬 개발과 테스트

> [!info] 판본·검증 범위 — 2026-10-04
> **Airflow 2.10.5** 예제다. 회사 설치 버전·executor·권한·Git Sync 조건은 미확인이다. 3.x로 그대로 복사하지 않고 해당 판본의 public API/provider 문서를 확인한다. 로컬 파싱과 실제 scheduler·worker·외부 시스템 검증을 구분한다.
> 실행 가능한 예제, 미구현 의사코드, 환경 점검용 stub을 구분한다. 외부 부작용이 있는 Task는 실제 테스트 대상으로 승인된 환경에서만 실행한다.


## 목표

Airflow DAG를 회사 서버에 올리기 전에 로컬에서 최대한 문제를 줄인다.

검증은 세 단계로 나눈다.

```text
1. 순수 Python 코드 테스트
2. DAG import 테스트
3. Airflow Task 실행 테스트
```

회사 관리형 Airflow에서는 로컬과 서버 환경이 완전히 같지 않을 수 있다. 그래도 로컬 검증을 해두면 syntax error, import error, 인자 누락, 멱등성 문제를 많이 줄일 수 있다.

원문이 가정한 배포 시나리오는 Bitbucket Git Sync이며 실제 회사 설정은 미확인이다. 따라서 로컬 검증 후 Bitbucket에 push하고, Airflow가 해당 branch를 sync했는지 UI에서 확인하는 흐름을 기준으로 한다.

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

아래는 원문의 코드 설정 비교 구조다. UI 권한 제한만으로 비밀을 DAG/Git package에 넣어야 한다고 결론 내리지 않는다. 승인된 env/backend 공급을 확인하고 `config.py`에는 비밀 없는 설정을 둔다.

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

`src/` 구조는 Python package로 관리하기 좋지만, 회사 Airflow에서 `src/` 패키지를 어떻게 배포할 수 있는지 확인해야 한다. Git Sync가 `dags/`만 sync하거나 `PYTHONPATH`가 repository root를 포함하지 않으면 `src/` import가 실패할 수 있다.

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
    df.to_parquet(output_path, index=False)


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

이 구조는 로컬 CLI 실행과 Airflow 함수 호출을 모두 지원한다. `run_date`는 인터페이스 예시일 뿐 이 구현이 CSV를 날짜로 필터링하지는 않는다. 입력이 해당 날짜 범위인지 호출자가 검증해야 한다. pandas와 Parquet engine(예: pyarrow)이 필요하다. hourly 예제도 print만 하며 실제 조회 구현이 아니다.

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

Airflow 없이 먼저 Python 함수 자체를 테스트한다. 아래는 `src/company_jobs/preprocess.py`를 import 가능한 package로 설치한 경우다. 작은 구조의 `jobs`와 섞지 않는다. 테스트도 pandas/Parquet engine이 필요하다.

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

    import pandas as pd

    actual = pd.read_parquet(output_path)
    assert actual.to_dict(orient="list") == {"id": [1, 2], "value": [10, 20]}
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
from airflow.models import DagBag


def test_dag_imports_without_error():
    bag = DagBag(dag_folder="dags", include_examples=False)
    assert not bag.import_errors, bag.import_errors
    dag = bag.dags.get("daily_pipeline")
    assert dag is not None, "예상 DAG가 발견되지 않음"
    assert set(dag.task_ids) == {"download", "preprocess", "analyze"}
    assert dag.get_task("download").downstream_task_ids == {"preprocess"}
    assert dag.get_task("preprocess").downstream_task_ids == {"analyze"}

```

위 DAG ID/Task 집합/edge는 작은 프로젝트의 계약 예시다. 자신의 DAG에 맞춰 명시하고 빈 DAG나 누락된 DAG도 실패시키도록 한다. scheduler·권한·배포 성공을 증명하는 테스트는 아니다.

## Airflow CLI 테스트

로컬에 Airflow가 설치되어 있으면 DAG 목록과 Task 실행을 테스트한다.

```bash
airflow dags list
airflow tasks list daily_pipeline
airflow tasks test daily_pipeline preprocess 2026-05-02
```

`airflow tasks test`는 의존성을 무시하고 특정 Task를 단독 실행하며 일반 Task Instance 실행 상태를 기록하지 않는다. 하지만 코드의 DB write/API 요청/파일 변경은 실제로 발생한다. 테스트용 입력·계정·출력 경로로 부작용을 격리한다. Scheduler가 없어도 Task 로직을 확인할 수 있어 유용하다.

단, 회사 서버와 로컬 패키지, Bitbucket sync branch, secret 파일 내용이 다르면 로컬 성공이 서버 성공을 보장하지는 않는다.

## Secret과 설정 mocking

순수 Python 테스트에서는 운영 secret 파일을 직접 사용하지 않도록 설계하는 것이 좋다.

좋은 구조:

```python
def run_query(db_config: dict, sql: str) -> list[dict]:
    ...
```

위 `run_query`의 `...`는 미구현 placeholder다. 아래 호출과 fake-config 검사는 독립 실행 가능한 unit test가 아니다. 쿼리 구현·client mock·기대 결과를 작성한 뒤 실행한다. fake host/password만 넘긴다고 네트워크가 mock되는 것은 아니다. Airflow Task/client factory에서 승인된 credential 공급을 읽고 순수 로직에 필요한 값만 전달한다.

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
- 비밀이 승인된 방식으로 공급되고 Git·로그·XCom에 노출되지 않는가

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
- helper package와 비밀 없는 설정은 import 가능하고 secret 주입은 실행 worker에 전달되는가
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

`check_storage`는 print만 하는 stub이므로 파일 쓰기/읽기·storage health·권한을 검증하지 않는다. 승인된 테스트 저장소의 작은 fixture를 실제 왕복시킨 뒤 확인 결과를 분리해 기록한다. 조사가 끝나면 debug DAG는 제거하거나 pause한다.

## 다음 단계

다음 문서에서는 Sensor, Dataset, Dynamic Task Mapping, pool, backfill 같은 고급 운영 패턴을 다룬다.

- [07. 고급 운영 패턴](./07-advanced-operations.md)


## 검증 근거 — 2026-10-04

- [2.10.5 DAG 테스트](https://airflow.apache.org/docs/apache-airflow/2.10.5/best-practices.html), [CLI tasks test](https://airflow.apache.org/docs/apache-airflow/2.10.5/cli-and-env-variables-ref.html#test).
- [module/package 배치](https://airflow.apache.org/docs/apache-airflow/2.10.5/administration-and-deployment/modules_management.html), [Connection](https://airflow.apache.org/docs/apache-airflow/2.10.5/howto/connection.html).
- requirements의 숫자는 원문 예시다. 로컬 재검증 환경의 설치 버전은 정리 기록에 별도로 남기며 회사 판본 증거로 사용하지 않는다.

미확인: 사내 배포·계정·리소스·네트워크 조건과 실제 운영 성공. 중복 예제의 계약 통합·회사 정책 결정은 Herdr `pane_not_found`로 Claude 협의를 보류한다. [주제 정리 기록](../organization-log.md)에 진행 결과를 남긴다.

로컬 확인: Python 3.12.12/Airflow 2.10.5 임시 환경에서 이 장의 Python 구문과 완성 DAG 정의를 검사했다. Kubernetes provider import·실제 venv job·외부 접속·scheduler 실행은 별도 미확인이다. 추가 실행 결과와 판본은 위 정리 기록을 읽는다.
