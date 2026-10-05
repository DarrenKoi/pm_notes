---
tags: [airflow, task-dependency, dag, sequential-execution]
level: beginner
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

# Airflow Task 의존성 — Python 코드 순차 실행 패턴

> [!info] 검증 판본과 읽기 조건 — 2026-10-04
> Airflow **2.10.5** 공식 문서·해당 tag 소스와 대조했다. 아래 DAG 예제는 2.x 학습용이며 설치된 Airflow/provider에서 파싱·실행해야 한다. Airflow 3의 DAG/Task public API는 `airflow.sdk`이며 legacy import를 그대로 최신 예제로 읽지 않는다. 회사 서버 버전·executor·패키지·운영 권한은 미확인이다. Python 구문/모의 검증과 실제 scheduler 실행을 구분한다.
>
> 이 문서는 **Task 사이의 실행 순서**를 다룬다. DAG를 시작시키는 조건은 [이벤트 기반 실행](./event-driven-execution.md), 순서대로 배우는 과정은 [Airflow 커리큘럼](./airflow-basic-to-advanced/README.md)을 읽는다. 중복 문법을 독립적인 실전 패턴으로 유지했으며 완전 통합은 Claude 협의 대기다.


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
- **`trigger_rule="all_success"` (일반 Task의 기본값; teardown 등 예외 확인)** → upstream이 모두 성공해야만 downstream 실행
- 기본 `all_success`에서는 upstream 실패로 downstream이 `upstream_failed`가 될 수 있다. 이는 `skipped`와 다른 상태다. 분기 skip·다른 trigger rule·재시도 중인 상태를 별도로 본다
- 실패 Task만 골라서 **Clear & Re-run** 가능
- 재시도(`retries`), 타임아웃(`execution_timeout`), 알림(`on_failure_callback`)도 Task 단위 설정

> 핵심: **의존성은 한 DAG run 안의 선후 관계를 선언한다**. 실제 실행 시점은 trigger rule·상태·pool·executor 자원에 따라 달라지고, 서로 다른 run이나 외부 부작용까지 직렬화하지 않는다.

---

## 핵심 개념 (What)

### Task 의존성을 표현하는 3가지 문법

| 문법 | 형태 | 권장 상황 |
|------|------|-----------|
| **TaskFlow API** | 함수 호출 체인 (`b(a())`) | 신규 DAG 기본. Pythonic, XCom 자동 처리 |
| **Shift 연산자** | `task_a >> task_b` | 전통 Operator(BashOperator 등) 사용 시 |
| **메서드** | `task_b.set_upstream(task_a)` | 동적으로 의존성을 만들 때 |

> TaskFlow의 데이터 의존과 `>>`의 순서 의존은 한 DAG에서 함께 쓸 수 있다. 코드 스타일보다 실제 Graph에 필요한 edge가 있는지 확인한다.

### `trigger_rule` 종류

| 값 | 의미 | 쓰임 |
|----|------|------|
| `all_success` (기본) | upstream 모두 성공 시 실행 | 일반적인 순차 실행 |
| `all_failed` | upstream 모두 실패 시 실행 | 실패 알림 / 복구 Task |
| `all_done` | upstream 결과 무관, 끝나기만 하면 | 정리(cleanup) Task |
| `one_success` | upstream 중 하나라도 성공 시 | upstream 중 하나 성공 시 실행; A 실패 뒤 B 실행을 의미하지 않음 |
| `one_failed` | upstream 중 하나라도 실패 시 | 부분 실패 알림 |
| `none_failed` | 실패 없이 끝났을 때 (skip은 OK) | 분기 후 합치기 |

### "성공"의 정의
- 정상 Python callable은 예외 없이 끝나면 성공한다. `AirflowSkipException` 같은 명시적 skip·운영자가 바꾼 상태는 별도다
- `return` 값이 `None`이든 dict든 무관 — 예외 여부만 본다
- BashOperator는 exit code 0이면 성공, 기본 skip code 99는 `skipped`, 다른 nonzero는 실패다 (`skip_on_exit_code` 설정에 따라 달라짐)
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
t1 >> [t2a, t2b] >> t3   # t1 성공 후 t2a/t2b 실행 가능; 둘 다 성공 후 t3 (자원에 따라 실제 동시성은 다름)
t1 >> t2; t1 >> t3        # t1 성공 후 t2/t3 실행 가능 (fan-out; 동시 시작 보장은 아님)
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
- Python 단독 명령의 nonzero exit는 그대로 실패한다. 뒤에 성공 명령이 붙거나 pipeline 앞단에서 실패하면 최종 exit가 0이 될 수 있어 `set -euo pipefail`을 사용한다. bash의 조건문 등 `-e` 예외까지 제거하는 기능은 아니다
- `{{ ds }}`는 logical date의 `YYYY-MM-DD`다. hourly 처리 범위는 `data_interval_start/end`를 사용하며 `ds`만으로 run 고유 경로나 시간 구간을 만들지 않는다
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

> [!warning] 패턴 4의 적용 범위
> 함수 소스를 분리한 환경에서 실행하므로 필요한 import와 의존성을 함수 안에 명시한다. 위 `/tmp/step1_out.parquet` 전달은 **동일 파일시스템이 유지되는 제한된 실습**이다. Celery/Kubernetes 등에서는 다른 worker/pod에 배치돼 파일이 없을 수 있다. 운영에서는 공유 저장소의 run별 객체 URI를 전달하고 재시도·동시 run·쓰기 완료를 설계한다. 고정 pandas/pyarrow 버전은 당시 예시이며 현재 Python 호환성은 미확인이다.

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

> [!warning] Task 성공과 DAG run 성공은 다르다
> 위 패턴은 trigger rule 시연이며 실패 감지용 운영 DAG의 완성형이 아니다. 2.10.5의 일반 DAG run 판정은 leaf 상태를 사용한다. `main_work`가 실패해도 leaf인 cleanup·알림이 모두 성공하면 run이 성공으로 표시될 수 있다. [DAG run 상태](https://airflow.apache.org/docs/apache-airflow/2.10.5/core-concepts/dag-run.html)를 대조하고 원래 작업 실패가 최종 판정에 남는지 실제 scheduler에서 확인한다.

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
    reset_dag_run=False,         # 기존 run을 자동 clear하지 않음; 재실행 정책은 별도 결정
)

last_task_in_a >> trigger_b
```

`logical_date`를 생략하면 2.10.5 구현은 호출 시각을 사용한다. 같은 논리 날짜의 기존 run을 의도한 경우 날짜/run ID를 명시해야 한다. `reset_dag_run=True`는 기존 run을 clear해 재실행하며 conf를 새로 만들지 않는다. wait는 worker 슬롯을 점유할 수 있고 database isolation mode에서는 이 두 기능의 제약이 있다.

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

기본 sensor는 **현재 run과 같은 logical date**를 찾는다. 서로 다른 schedule이면 `execution_delta` 또는 `execution_date_fn`으로 대응 날짜를 정의해야 한다. 기본적으로 선행 실패를 곧바로 자신의 실패로 바꾸지 않고 timeout까지 기다릴 수 있으므로 `failed_states`/`skipped_states`를 목적에 맞게 확인한다.

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
    retry_exponential_backoff=True,   # 기본 지연을 바탕으로 증가; task별 hash jitter와 상한 적용
    max_retry_delay=timedelta(minutes=30),
)
def flaky_api_call():
    # ... requests.get(...) ...
    pass
```

DAG 전체 default로도 가능:
```python
from datetime import datetime, timedelta
from airflow.decorators import dag, task

default_args = {
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

@dag(default_args=default_args, start_date=datetime(2026, 5, 1), schedule=None, catchup=False)
def my_dag():
    @task
    def work() -> None:
        print("작업 예시")
    work()

my_dag()
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
| `t1 >> t2` 인데 의존성이 안 잡힘 | 두 Task가 병렬 실행됨 | context manager·`dag=`·연결을 통한 DAG 할당과 실제 edge를 확인; `with` 밖이라는 이유만으로 병렬이 되지 않음 |
| 함수 내부 import를 까먹고 외부에 둠 (PythonVirtualenvOperator) | `NameError` | 모든 import를 함수 안으로 이동 |
| 결과가 잘못됐는데 예외 안 던짐 | 다음 Task가 잘못된 데이터로 실행 | Task 끝부분에 검증 후 `raise ValueError(...)` |
| pipeline 앞단 실패가 최종 exit에 반영되지 않음 | Python 실패해도 exit 0 | `pipefail`/종료 코드와 조건문 동작을 확인 |
| XCom으로 큰 DataFrame 전달 | metadata DB 비대화, 성능 저하 | 파일 경로만 XCom으로 넘기고 데이터는 MinIO/디스크 |

---

## 참고 자료 (References)

- [Airflow: TaskFlow API](https://airflow.apache.org/docs/apache-airflow/2.10.5/tutorial/taskflow.html)
- [Airflow: Tasks & Dependencies](https://airflow.apache.org/docs/apache-airflow/2.10.5/core-concepts/tasks.html)
- [Trigger Rules](https://airflow.apache.org/docs/apache-airflow/2.10.5/core-concepts/dags.html#trigger-rules)
- [Cross-DAG Dependencies](https://airflow.apache.org/docs/apache-airflow/2.10.5/howto/operator/external_task_sensor.html)
- [chain / cross_downstream 헬퍼](https://airflow.apache.org/docs/apache-airflow/2.10.5/_api/airflow/models/baseoperator/index.html#airflow.models.baseoperator.chain)

## 관련 문서
- [Airflow + MinIO 파이프라인 튜토리얼](./airflow-minio-tutorial.md) — 전체 파이프라인 구성


### 추가 확인 근거 — 2026-10-04

- [Airflow 2.10.5 Best Practices](https://airflow.apache.org/docs/apache-airflow/2.10.5/best-practices.html): worker 간 로컬 파일 공유를 가정하지 않는다.
- [BashOperator 종료 상태](https://airflow.apache.org/docs/apache-airflow/2.10.5/howto/operator/bash.html), [템플릿 날짜](https://airflow.apache.org/docs/apache-airflow/2.10.5/templates-ref.html).
- [TriggerDagRunOperator 2.10.5 소스](https://github.com/apache/airflow/blob/2.10.5/airflow/operators/trigger_dagrun.py), [ExternalTaskSensor 소스](https://github.com/apache/airflow/blob/2.10.5/airflow/sensors/external_task.py), [retry 계산 소스](https://github.com/apache/airflow/blob/2.10.5/airflow/models/taskinstance.py).
- [Airflow 3 public interface](https://airflow.apache.org/docs/apache-airflow/stable/public-airflow-interface.html): 확인 당시 문서 표시는 3.3.2였다. 최신 보증이나 회사 설치 버전 확인은 아니다.
