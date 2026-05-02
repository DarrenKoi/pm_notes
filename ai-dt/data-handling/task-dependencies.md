---
tags: [airflow, task-dependency, dag, sequential-execution]
level: beginner
last_updated: 2026-05-02
status: complete
---

# Airflow Task 의존성 — Python 코드 순차 실행 패턴

> 여러 Python 코드를 "앞 코드가 성공해야 뒤 코드가 실행"되도록 묶는 모든 방법

## 왜 필요한가? (Why)

### 단순 chaining의 한계
```bash
# Bash에서 흔히 쓰는 방식
python step1.py && python step2.py && python step3.py
```
- 중간 실패 시 어디서 멈췄는지 추적 어려움
- step2만 다시 돌리려면 수동
- 로그 분산, 재시도 없음, 모니터링 없음

### Airflow가 자동으로 보장하는 것
- **`trigger_rule="all_success"` (모든 Task의 기본값)** → upstream이 모두 성공해야만 downstream 실행
- upstream이 실패하면 downstream은 자동으로 `upstream_failed` 상태가 되어 **건너뛰어짐**
- 실패 Task만 골라서 **Clear & Re-run** 가능
- 재시도(`retries`), 타임아웃(`execution_timeout`), 알림(`on_failure_callback`)도 Task 단위 설정

> 핵심: **순차 실행은 별도 옵션이 아니라 의존성을 선언만 하면 자동으로 보장**된다.

---

## 핵심 개념 (What)

### Task 의존성을 표현하는 3가지 문법

| 문법 | 형태 | 권장 상황 |
|------|------|-----------|
| **TaskFlow API** | 함수 호출 체인 (`b(a())`) | 신규 DAG 기본. Pythonic, XCom 자동 처리 |
| **Shift 연산자** | `task_a >> task_b` | 전통 Operator(BashOperator 등) 사용 시 |
| **메서드** | `task_b.set_upstream(task_a)` | 동적으로 의존성을 만들 때 |

> 한 DAG 안에서는 **하나로 통일**한다. 섞이면 의존성 그래프가 한눈에 안 들어옴.

### `trigger_rule` 종류

| 값 | 의미 | 쓰임 |
|----|------|------|
| `all_success` (기본) | upstream 모두 성공 시 실행 | 일반적인 순차 실행 |
| `all_failed` | upstream 모두 실패 시 실행 | 실패 알림 / 복구 Task |
| `all_done` | upstream 결과 무관, 끝나기만 하면 | 정리(cleanup) Task |
| `one_success` | upstream 중 하나라도 성공 시 | 백업 경로 (A 실패 → B 시도) |
| `one_failed` | upstream 중 하나라도 실패 시 | 부분 실패 알림 |
| `none_failed` | 실패 없이 끝났을 때 (skip은 OK) | 분기 후 합치기 |

### "성공"의 정의
- Python 함수가 **예외를 발생시키지 않고 return**하면 성공
- `return` 값이 `None`이든 dict든 무관 — 예외 여부만 본다
- BashOperator는 **exit code 0**이 성공
- 따라서 "결과가 의도와 다르면" Task 안에서 **명시적으로 `raise`** 해야 다음 Task가 차단된다

---

## 어떻게 사용하는가? (How)

### 패턴 1. TaskFlow API — 함수 호출로 의존성 선언 (권장)

`@task`로 감싼 함수를 다른 함수의 인자로 넘기면, Airflow가 의존성을 자동 추론한다.

```python
from datetime import datetime
from airflow.decorators import dag, task


@dag(
    dag_id="sequential_python_pipeline",
    start_date=datetime(2026, 5, 1),
    schedule="@daily",
    catchup=False,
)
def pipeline():

    @task
    def step1_extract() -> dict:
        print("Extracting data...")
        return {"records": [1, 2, 3, 4, 5]}

    @task
    def step2_transform(data: dict) -> list[int]:
        print(f"Transforming {len(data['records'])} records")
        return [x * 10 for x in data["records"]]

    @task
    def step3_load(values: list[int]) -> None:
        print(f"Loading: {values}")

    # 의존성: step1 → step2 → step3 (함수 호출 순서로 자동 정의)
    raw = step1_extract()
    transformed = step2_transform(raw)
    step3_load(transformed)


pipeline()
```

**무엇이 보장되는가**:
- `step1_extract`가 예외 발생 → `step2_transform`은 `upstream_failed` (실행 안 됨)
- `step2_transform`이 예외 발생 → `step3_load`는 `upstream_failed`
- return 값은 자동으로 XCom에 저장되어 다음 Task의 인자로 전달

---

### 패턴 2. PythonOperator + `>>` 연산자 (전통 방식)

TaskFlow가 도입되기 전 표준. 외부 라이브러리 함수를 그대로 쓰거나, 의존성을 명시적으로 보고 싶을 때.

```python
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator


def extract():
    print("Extract")
    return [1, 2, 3]

def transform(**context):
    # XCom에서 직접 꺼냄
    data = context["ti"].xcom_pull(task_ids="extract")
    return [x * 10 for x in data]

def load(**context):
    data = context["ti"].xcom_pull(task_ids="transform")
    print(f"Load: {data}")


with DAG(
    dag_id="sequential_classic",
    start_date=datetime(2026, 5, 1),
    schedule="@daily",
    catchup=False,
) as dag:

    t1 = PythonOperator(task_id="extract", python_callable=extract)
    t2 = PythonOperator(task_id="transform", python_callable=transform)
    t3 = PythonOperator(task_id="load", python_callable=load)

    # 의존성 선언: t1 → t2 → t3
    t1 >> t2 >> t3
```

**`>>` 연산자 응용**:
```python
t1 >> [t2a, t2b] >> t3   # t1 끝나면 t2a, t2b 병렬 실행, 둘 다 끝나면 t3
t1 >> t2; t1 >> t3        # t1 끝나면 t2, t3 동시 실행 (fan-out)
```

---

### 패턴 3. `.py` 스크립트 파일을 순서대로 실행 (BashOperator)

이미 작성된 독립 스크립트들을 그대로 순차 실행하고 싶을 때.

```python
from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator


with DAG(
    dag_id="sequential_scripts",
    start_date=datetime(2026, 5, 1),
    schedule="@daily",
    catchup=False,
) as dag:

    s1 = BashOperator(
        task_id="run_step1",
        bash_command="set -euo pipefail; python /opt/scripts/step1_download.py --date {{ ds }}",
    )
    s2 = BashOperator(
        task_id="run_step2",
        bash_command="set -euo pipefail; python /opt/scripts/step2_clean.py --date {{ ds }}",
    )
    s3 = BashOperator(
        task_id="run_step3",
        bash_command="set -euo pipefail; python /opt/scripts/step3_upload.py --date {{ ds }}",
    )

    s1 >> s2 >> s3
```

**핵심 디테일**:
- `set -euo pipefail` 필수 — 안 쓰면 Python이 traceback 찍고 죽어도 exit code가 0으로 나가는 경우가 생김
- `{{ ds }}`는 Airflow Jinja 매크로 (`execution_date`의 `YYYY-MM-DD`)
- 각 스크립트 자체는 **인자로 날짜를 받아 동작하는 멱등한 형태**여야 재실행이 안전

---

### 패턴 4. 격리된 가상환경에서 순차 실행 (PythonVirtualenvOperator)

각 Task가 다른 패키지 버전을 요구할 때.

```python
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonVirtualenvOperator


def step1():
    import pandas as pd  # venv 안에서 import
    df = pd.DataFrame({"a": [1, 2, 3]})
    df.to_parquet("/tmp/step1_out.parquet")

def step2():
    import pandas as pd
    df = pd.read_parquet("/tmp/step1_out.parquet")
    print(df.describe())


with DAG(
    dag_id="sequential_venv",
    start_date=datetime(2026, 5, 1),
    schedule="@daily",
    catchup=False,
) as dag:

    t1 = PythonVirtualenvOperator(
        task_id="step1",
        python_callable=step1,
        requirements=["pandas==2.2.2", "pyarrow==15.0.0"],
        system_site_packages=False,
    )
    t2 = PythonVirtualenvOperator(
        task_id="step2",
        python_callable=step2,
        requirements=["pandas==2.2.2", "pyarrow==15.0.0"],
        system_site_packages=False,
    )

    t1 >> t2
```

> 주의: `python_callable` 안에서 쓰는 모든 import는 **함수 내부**에 둬야 한다. 외부 import는 venv에 없으므로 `NameError` 발생.

---

### 패턴 5. 여러 스크립트를 동적으로 chaining (반복문)

스크립트가 10개라 일일이 `>>`로 잇기 귀찮을 때.

```python
from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator


STEPS = [
    "01_download",
    "02_validate",
    "03_clean",
    "04_enrich",
    "05_aggregate",
    "06_upload",
]

with DAG(
    dag_id="dynamic_sequential",
    start_date=datetime(2026, 5, 1),
    schedule="@daily",
    catchup=False,
) as dag:

    tasks = [
        BashOperator(
            task_id=name,
            bash_command=f"set -euo pipefail; python /opt/scripts/{name}.py --date {{{{ ds }}}}",
        )
        for name in STEPS
    ]

    # 리스트의 인접한 두 Task를 순서대로 연결
    for upstream, downstream in zip(tasks, tasks[1:]):
        upstream >> downstream
```

또는 더 간결하게 `chain` 헬퍼:

```python
from airflow.models.baseoperator import chain

chain(*tasks)   # tasks[0] >> tasks[1] >> tasks[2] >> ...
```

---

### 패턴 6. 분기와 합치기 — 일부 실패해도 계속 진행

기본은 "하나라도 실패하면 다음은 안 돈다"이지만, **`trigger_rule`로 예외 처리**할 수 있다.

```python
from datetime import datetime
from airflow.decorators import dag, task
from airflow.utils.trigger_rule import TriggerRule


@dag(start_date=datetime(2026, 5, 1), schedule="@daily", catchup=False)
def with_cleanup():

    @task
    def main_work():
        # ... 본 작업 ...
        raise RuntimeError("Boom")  # 일부러 실패

    @task(trigger_rule=TriggerRule.ALL_DONE)
    def cleanup():
        """main_work의 성공/실패와 무관하게 항상 실행 (정리 작업)."""
        print("Cleaning up tmp files regardless of outcome")

    @task(trigger_rule=TriggerRule.ONE_FAILED)
    def alert_on_failure():
        """upstream 중 하나라도 실패 시에만 실행 (알림)."""
        print("Sending Slack alert")

    work = main_work()
    work >> cleanup()
    work >> alert_on_failure()


with_cleanup()
```

**자주 쓰는 조합**:
- `all_done` cleanup Task — 임시 파일 삭제, 락 해제
- `one_failed` 알림 Task — Slack/이메일 통보
- `none_failed_min_one_success` — 분기 후 합칠 때 ("하나라도 성공했고 실패는 없을 때")

---

### 패턴 7. 다른 DAG의 결과를 기다리기 (Cross-DAG)

Pipeline A가 끝나야 Pipeline B가 실행되어야 하는데, 두 DAG의 스케줄이 다르거나 별도 팀이 관리할 때.

#### 방법 A. `TriggerDagRunOperator` — A가 직접 B를 띄움
```python
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

trigger_b = TriggerDagRunOperator(
    task_id="trigger_pipeline_b",
    trigger_dag_id="pipeline_b",
    wait_for_completion=True,    # B가 끝날 때까지 이 Task가 기다림
    poke_interval=30,            # 30초마다 상태 체크
    reset_dag_run=True,          # 같은 날짜 재실행 허용
)

last_task_in_a >> trigger_b
```

#### 방법 B. `ExternalTaskSensor` — B가 A의 완료를 기다림
```python
from airflow.sensors.external_task import ExternalTaskSensor

wait_for_a = ExternalTaskSensor(
    task_id="wait_for_pipeline_a",
    external_dag_id="pipeline_a",
    external_task_id="final_task",   # None이면 DAG 전체
    timeout=3600,
    mode="reschedule",                # 슬롯 점유 안 함 (긴 대기 시 권장)
)

wait_for_a >> first_task_in_b
```

| 선택 기준 | A: TriggerDagRunOperator | B: ExternalTaskSensor |
|-----------|-------------------------|----------------------|
| 누가 주도? | 선행 DAG가 후속을 띄움 | 후속 DAG가 선행을 기다림 |
| 추천 상황 | 명확한 1:1 관계, 같은 팀 | 다수의 후속 DAG가 같은 선행을 공유 |

---

### 패턴 8. 실패 시 재시도 — 일시적 오류 자동 복구

순차 실행 중간에 네트워크 blip 같은 일시적 오류로 멈추는 걸 막는다.

```python
from datetime import timedelta
from airflow.decorators import task

@task(
    retries=3,
    retry_delay=timedelta(minutes=2),
    retry_exponential_backoff=True,   # 2분, 4분, 8분으로 점증
    max_retry_delay=timedelta(minutes=30),
)
def flaky_api_call():
    # ... requests.get(...) ...
    pass
```

DAG 전체 default로도 가능:
```python
default_args = {
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

@dag(default_args=default_args, ...)
def my_dag(): ...
```

> 재시도는 **upstream Task 입장에서 보면 마지막 시도가 성공하면 success로 간주**된다. 즉 downstream은 정상 실행됨.

---

## 의사결정 요약

```
"앞 Python 코드 성공해야 뒤가 실행" 시나리오
│
├─ 함수 단위로 나눠서 데이터 주고받기?
│  └─ 패턴 1 (TaskFlow API) ← 1순위 권장
│
├─ 이미 .py 스크립트가 따로 있고 그대로 쓰고 싶음?
│  └─ 패턴 3 (BashOperator) 또는 4 (PythonVirtualenvOperator)
│
├─ 스크립트 개수가 많음 (5개+)?
│  └─ 패턴 5 (chain 헬퍼)
│
├─ 일부 Task는 실패해도 정리/알림은 돌아야 함?
│  └─ 패턴 6 (trigger_rule)
│
├─ 다른 DAG와 연동?
│  └─ 패턴 7 (TriggerDagRunOperator / ExternalTaskSensor)
│
└─ 일시적 오류로 멈추는 것 방지?
   └─ 패턴 8 (retries + retry_delay)
```

---

## 자주 하는 실수

| 실수 | 결과 | 해결 |
|------|------|------|
| `def func(): ...` 만 작성하고 `func()` 호출 안 함 | DAG에 Task가 0개로 등록 | TaskFlow는 `func()`로 호출해야 Task 인스턴스 생성 |
| `t1 >> t2` 인데 의존성이 안 잡힘 | 두 Task가 병렬 실행됨 | `with DAG(...)` 블록 **안에서** Operator를 만들었는지 확인 |
| 함수 내부 import를 까먹고 외부에 둠 (PythonVirtualenvOperator) | `NameError` | 모든 import를 함수 안으로 이동 |
| 결과가 잘못됐는데 예외 안 던짐 | 다음 Task가 잘못된 데이터로 실행 | Task 끝부분에 검증 후 `raise ValueError(...)` |
| `BashOperator`에서 `set -e` 없이 파이프 사용 | Python 실패해도 exit 0 | `set -euo pipefail` 항상 prepend |
| XCom으로 큰 DataFrame 전달 | metadata DB 비대화, 성능 저하 | 파일 경로만 XCom으로 넘기고 데이터는 MinIO/디스크 |

---

## 참고 자료 (References)

- [Airflow: TaskFlow API](https://airflow.apache.org/docs/apache-airflow/stable/tutorial/taskflow.html)
- [Airflow: Tasks & Dependencies](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/tasks.html)
- [Trigger Rules](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/dags.html#trigger-rules)
- [Cross-DAG Dependencies](https://airflow.apache.org/docs/apache-airflow/stable/howto/operator/external_task_sensor.html)
- [chain / cross_downstream 헬퍼](https://airflow.apache.org/docs/apache-airflow/stable/_api/airflow/models/baseoperator/index.html#airflow.models.baseoperator.chain)

## 관련 문서
- [Airflow + MinIO 파이프라인 튜토리얼](./airflow-minio-tutorial.md) — 전체 파이프라인 구성
- [AI/DT 학습 노트](../README.md)
