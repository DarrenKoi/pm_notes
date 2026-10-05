---
tags: [airflow, sensor, dataset, dynamic-task-mapping, backfill, operations]
level: advanced
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

# 07. 고급 운영 패턴

> [!info] 판본·검증 범위 — 2026-10-04
> **Airflow 2.10.5** 예제다. 회사 설치 버전·executor·권한·Git Sync 조건은 미확인이다. 3.x로 그대로 복사하지 않고 해당 판본의 public API/provider 문서를 확인한다. 로컬 파싱과 실제 scheduler·worker·외부 시스템 검증을 구분한다.
> Dataset과 mapping의 코드 구성은 실제 데이터 생성·외부 호출·알림 전송의 증거가 아니다. 운영 권한과 부하 조건을 확인한다.


## 목표

기본 DAG를 운영에 올린 뒤 마주치는 고급 주제를 정리한다.

다루는 내용:

- Sensor
- Dataset
- Dynamic Task Mapping
- pool과 queue
- backfill
- SLA/알림
- 장애 대응
- 운영 체크리스트

## Sensor

Sensor는 어떤 조건이 만족될 때까지 기다리는 Task다.

예:

- 파일이 도착할 때까지 대기
- S3/MinIO object가 생길 때까지 대기
- DB에 특정 row가 생길 때까지 대기
- 외부 API 상태가 완료가 될 때까지 대기

파일 대기 예:

```python
from airflow.sensors.filesystem import FileSensor


wait_file = FileSensor(
    task_id="wait_for_input_file",
    filepath="/data/landing/{{ ds }}/done.flag",
    poke_interval=60,
    timeout=60 * 60 * 6,
    mode="reschedule",
)

wait_file >> preprocess
```

중요한 설정:

| 설정 | 의미 |
|------|------|
| `poke_interval` | 몇 초마다 확인할지 |
| `timeout` | 최대 대기 시간 |
| `mode="poke"` | Worker slot을 잡고 대기 |
| `mode="reschedule"` | 확인 후 slot을 반환하고 나중에 다시 확인 |

FileSensor는 worker가 해당 경로를 볼 수 있어야 한다. 위 `preprocess`는 별도 정의가 필요한 부분 예제다. 긴 대기에는 `mode="reschedule"`을 우선 고려한다. `poke`로 수시간 대기하면 Worker slot을 낭비할 수 있다.

## done.flag 패턴

데이터 파일 자체를 기다리는 것보다 완료 신호 파일을 기다리는 것이 안전하다.

```text
s3://raw/sales/dt=2026-05-02/data_001.parquet
s3://raw/sales/dt=2026-05-02/data_002.parquet
s3://raw/sales/dt=2026-05-02/_DONE
```

업스트림 시스템이 모든 파일을 쓴 뒤 `_DONE` 파일을 만들고, Airflow는 `_DONE`을 기다린다.

여러 object의 batch 완료 신호를 구분하는 패턴이다. S3 단일 object 게시와 전체 batch 완료는 다르다. producer가 전체 파일의 무결성을 확인한 후 신호를 게시하고 reader가 해당 run의 신호/파일 목록을 검증하는 계약이 필요하다. 오래된 `_DONE` 재사용은 잘못된 완료 판단을 만들 수 있다.

## Dataset

Airflow 2.4+에서는 Dataset으로 DAG 간 데이터 의존성을 표현할 수 있다.

Producer DAG:

```python
from datetime import datetime

from airflow import Dataset
from airflow.decorators import dag, task


RAW_SALES = Dataset("s3://raw/sales/")


@dag(
    dag_id="produce_raw_sales",
    start_date=datetime(2026, 5, 1),
    schedule="@daily",
    catchup=False,
)
def produce_raw_sales():
    @task(outlets=[RAW_SALES])
    def extract() -> None:
        print("write raw sales data")

    extract()


produce_raw_sales()
```

Consumer DAG:

```python
from datetime import datetime

from airflow import Dataset
from airflow.decorators import dag, task


RAW_SALES = Dataset("s3://raw/sales/")


@dag(
    dag_id="consume_raw_sales",
    start_date=datetime(2026, 5, 1),
    schedule=[RAW_SALES],
    catchup=False,
)
def consume_raw_sales():
    @task
    def transform() -> None:
        print("read raw sales data")

    transform()


consume_raw_sales()
```

위 producer/consumer는 print만 한다. Dataset URI는 의존성 식별자이며 파일을 직접 검사하지 않는다. outlet Task 성공이 event를 발생시키므로 실제 저장 완료 후 성공하도록 구현한다. 이 장은 2.10.5 Dataset API이며 3.x Asset API와 혼용하지 않는다. Dataset은 같은 Airflow 인스턴스 안에서 DAG 간 의존성을 표현하기 좋다. 다른 시스템의 외부 이벤트를 직접 받는 용도라면 Sensor, REST API trigger, message queue 연동을 검토한다.

## Dynamic Task Mapping

파일 목록이나 테이블 목록을 보고 Task를 동적으로 여러 개 만들고 싶을 때 사용한다.

예:

```python
from datetime import datetime

from airflow.decorators import dag, task


@dag(
    dag_id="dynamic_mapping_example",
    start_date=datetime(2026, 5, 1),
    schedule=None,
    catchup=False,
)
def dynamic_mapping_example():
    @task
    def list_targets() -> list[str]:
        return ["sales", "customer", "product"]

    @task
    def process_table(table_name: str) -> None:
        print(f"process {table_name}")

    process_table.expand(table_name=list_targets())


dynamic_mapping_example()
```

주의:

- scheduler가 runtime의 list/dict로 Task를 확장한다. 빈 입력은 skipped, 2.10.5 `max_map_length` 기본 1024는 설치 설정으로 확인한다.
- 너무 많은 Task를 한 번에 만들면 Scheduler와 UI가 느려질 수 있다.
- 수천 개 이상의 작은 Task보다 적당히 묶어서 처리하는 것이 나을 수 있다.
- pool로 동시 실행 개수를 제한한다.

## pool

pool은 여러 DAG/Run에 걸쳐 공유하는 slot을 제한한다. 기본1slot/Task이지만 `pool_slots` 가중치에 따라 Task 수와 slot 수가 다르다. deferred Task를 포함할지는 pool 설정으로 확인한다.

예:

- DB connection을 많이 쓰는 Task는 동시에 3개만
- 외부 API 호출은 동시에 5개만
- 무거운 분석 작업은 동시에 1개만

DAG에서 pool 지정:

```python
task = BashOperator(
    task_id="call_api",
    bash_command=(
        "python call_api.py "
        "--start-ts '{{ data_interval_start }}' "
        "--end-ts '{{ data_interval_end }}'"
    ),
    pool="external_api_pool",
)
```

pool 생성은 운영팀 권한일 수 있다.

## 병렬 실행 가능 여부 확인

병렬 실행 가능 여부는 DAG 코드 하나로만 결정되지 않는다. 아래 제한을 모두 통과해야 Task가 동시에 실행된다.

| 제한 | 의미 | 사용자가 확인 가능한 곳 |
|------|------|------------------------|
| Executor | Sequential/Local/Celery/Kubernetes 중 무엇인지 | UI Admin Config 또는 운영팀 문의 |
| Worker capacity | 실제 Worker가 동시에 몇 Task를 돌릴 수 있는지 | 운영팀 문의, Task가 queued에 머무는지 관찰 |
| `parallelism` | Airflow 전체에서 동시에 running 가능한 Task 수 | Admin Config 또는 운영팀 문의 |
| `max_active_tasks_per_dag` | DAG 하나에서 동시에 running 가능한 Task 수 | Admin Config, DAG 코드의 `max_active_tasks` |
| `max_active_runs_per_dag` | 같은 DAG의 active run 상한 | Admin Config, DAG 코드의 `max_active_runs` |
| Pool slots | 특정 pool에 묶인 Task의 동시 실행 수 | Admin -> Pools, Task Instance detail |
| Queue | 특정 Worker queue로 Task를 보낼지 | Task Instance detail, 운영팀 문의 |
| Task 의존성 | upstream이 끝나야 downstream 실행 | Graph/Grid view |

### UI에서 확인하는 순서

1. DAG Graph/Grid view에서 병렬이어야 하는 Task들이 서로 의존성 없이 나란히 있는지 확인한다.
2. 실행 시 여러 Task가 동시에 `running`이 되는지 본다.
3. 일부 Task가 `queued`에 오래 머물면 제한에 걸린 것이다.
4. Task Instance detail에서 `Pool`, `Pool Slots`, `Queue` 값을 확인한다.
5. 권한이 있으면 `Admin -> Pools`에서 해당 pool의 slots, occupied, queued 상태를 본다.
6. 권한이 있으면 `Admin -> Config`에서 `executor`, `parallelism`, `max_active_tasks_per_dag`, `max_active_runs_per_dag`를 확인한다.

`Admin -> Pools`나 `Admin -> Config` 메뉴가 보이지 않으면 일반 사용자 권한으로는 직접 확인하기 어렵다. 이 경우 운영팀에 값을 물어보거나, 아래처럼 작은 테스트 DAG로 실제 병렬 실행 여부를 확인한다.

### 병렬 테스트 DAG

Bitbucket Git Sync 환경에서는 아래 DAG를 별도 debug DAG로 올리고 수동 실행한다. 운영 부하를 줄이기 위해 sleep 시간과 Task 개수를 작게 둔다.

```python
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator


with DAG(
    dag_id="debug_parallel_capacity",
    start_date=datetime(2026, 5, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    max_active_tasks=3,
    tags=["debug"],
) as dag:
    t1 = BashOperator(
        task_id="sleep_1",
        bash_command="date; sleep 60; date",
        execution_timeout=timedelta(minutes=3),
    )

    t2 = BashOperator(
        task_id="sleep_2",
        bash_command="date; sleep 60; date",
        execution_timeout=timedelta(minutes=3),
    )

    t3 = BashOperator(
        task_id="sleep_3",
        bash_command="date; sleep 60; date",
        execution_timeout=timedelta(minutes=3),
    )

    [t1, t2, t3]
```

`max_active_tasks`는 Airflow 2.2+ 기준 DAG-level 설정이다. 회사 Airflow가 더 오래된 버전이면 같은 이름이 동작하지 않을 수 있으므로 운영팀에 버전과 DAG-level 동시성 설정명을 확인한다.

판단 기준:

| 결과 | 해석 |
|------|------|
| 세 Task가 동시에 `running` | 해당 시점의 sleep Task 3개 상태 중첩을 관찰; CPU/메모리/실제 job 용량 증거는 아님 |
| 하나만 `running`, 나머지는 `queued` | pool, worker, executor, DAG 동시성 제한 가능성 |
| 순서대로 하나씩 실행 | SequentialExecutor, pool slot 1, worker slot 부족, DAG 제한 가능성 |
| 계속 `queued` | worker/queue/pool 문제 가능성 |

테스트가 끝나면 debug DAG는 pause하거나 제거한다.

### 운영팀에 물어볼 항목

```text
확인 목적:
- hourly DAG에서 여러 Python Task를 병렬 실행할 수 있는지 확인

확인 요청:
- executor 종류: LocalExecutor / CeleryExecutor / KubernetesExecutor / etc.
- core.parallelism 값
- core.max_active_tasks_per_dag 값
- core.max_active_runs_per_dag 값
- default_pool slots 값
- 우리 DAG가 사용할 수 있는 pool 이름과 slots
- task queue 이름과 해당 queue를 듣는 Worker 수
- CeleryExecutor라면 worker_concurrency와 Worker 개수
- KubernetesExecutor라면 namespace quota / pod 동시 생성 제한
```

## queue

CeleryExecutor의 `queue`는 그 queue를 듣는 Celery worker로 routing하는 설정이다. 일반 KubernetesExecutor에서 `queue="high_memory"`가 고메모리 node를 선택한다는 뜻은 아니다. Kubernetes pod resource/node 배치는 해당 provider의 pod template/`executor_config`로 별도 설정한다. hybrid/multi-executor 구성은 설치 설정에 따라 다르다.

```python
task = BashOperator(
    task_id="heavy_job",
    bash_command=(
        "python heavy_job.py "
        "--start-ts '{{ data_interval_start }}' "
        "--end-ts '{{ data_interval_end }}'"
    ),
    queue="high_memory",
)
```

queue 이름과 Worker 구성이 회사마다 다르므로 운영팀에 확인한다.

## backfill

backfill은 과거 날짜 또는 시간 구간 데이터를 다시 처리하는 작업이다.

예:

```text
2026-04-01부터 2026-04-30까지 재처리
2026-05-02 10:00부터 2026-05-02 18:00까지 시간별 재처리
```

주의:

- output path가 날짜/시간 파티션으로 분리되어 있어야 한다.
- DB insert가 중복을 만들지 않아야 한다.
- `max_active_runs`와 pool로 동시 실행량을 제한해야 한다.
- 과거 데이터가 현재 코드와 호환되는지 확인해야 한다.
- 외부 API를 과거 구간만큼 대량 호출하면 rate limit에 걸릴 수 있다.

회사 관리형 Airflow에서는 CLI backfill 권한이 없을 수 있다. 그 경우 UI에서 날짜/시간 구간별 수동 실행하거나 운영팀에 요청한다.

## 수동 재실행

Airflow UI에서 실패한 Task를 Clear하면 해당 Task와 downstream Task를 다시 실행할 수 있다.

재실행 전에 확인할 것:

- 이 Task가 같은 처리 구간으로 다시 실행되어도 안전한가
- 이전 output을 지워야 하는가
- downstream까지 같이 재실행해야 하는가
- 외부 시스템에 중복 요청이 나가지 않는가

## 알림

실패를 UI에서만 확인하면 늦다. 운영 DAG에는 알림이 필요하다.

가능한 방식:

- Email
- Slack/Teams webhook
- 사내 메신저
- Airflow callback
- 외부 모니터링 시스템 연동

callback 예:

```python
def notify_failure(context):
    dag_id = context["dag"].dag_id
    task_id = context["task_instance"].task_id
    run_id = context["run_id"]
    print(f"FAILED dag={dag_id}, task={task_id}, run_id={run_id}")


default_args = {
    "on_failure_callback": notify_failure,
}
```

위 callback은 print만 하며 알림을 전송하지 않는다. 2.10.5 callback은 worker 실행에 따른 상태 변화에서 호출되며 UI/CLI의 단순 상태 변경은 호출하지 않는다. callback 자체 오류는 scheduler log에서 확인한다. 전달 실패/timeout/retry/중복 알림 계약도 별도 구현한다. 실제 운영에서는 print 대신 사내 알림 API를 호출한다. 단, 알림 함수 안에서도 secret을 로그에 남기지 않는다.

## 장기 실행 작업

Airflow는 작업을 시작하고 감시하는 데 좋지만, 매우 무거운 compute 자체를 Airflow Worker에서 직접 처리하는 것은 위험할 수 있다.

무거운 작업 예:

- 수시간 이상 모델 학습
- 대용량 Spark job
- GPU 작업
- 대량 파일 변환

이 경우 Airflow Task는 외부 compute job을 제출하고 상태를 감시하는 역할로 두는 것이 좋다.

```text
Airflow Task
  -> Spark job submit
  -> job id 저장
  -> 상태 polling
  -> 성공/실패 반영
```

## 장애 대응 흐름

Task 실패 시 확인 순서:

1. 실패 Task 로그 확인
2. Python traceback 또는 shell exit code 확인
3. 같은 Task만 재실행 가능한지 판단
4. 입력 데이터 존재 여부 확인
5. 패키지/import 오류인지 확인
6. 승인된 secret 주입, 계정 권한, 네트워크 오류인지 확인
7. Worker 리소스 부족인지 확인
8. upstream/downstream 영향 범위 확인
9. 재실행 또는 코드 수정 결정

## 흔한 장애

| 증상 | 가능 원인 | 대응 |
|------|----------|------|
| DAG가 UI에 안 보임 | import error, syntax error | import error 메뉴/Scheduler log 확인 |
| Task가 queued에 오래 있음 | worker slot 부족, pool 부족 | pool/queue/worker 상태 확인 |
| Task가 running에서 멈춤 | 외부 API hang, timeout 없음 | timeout 추가, 코드 timeout 설정 |
| `ModuleNotFoundError` | Worker 패키지 없음 | 운영팀 설치 요청, venv/container 사용 |
| 다음 Task가 파일 못 찾음 | 로컬 파일 전달 | 공유 저장소 사용 |
| 재실행 시 중복 데이터 | 멱등성 없음 | partition overwrite/upsert |
| 특정 날짜/시간만 실패 | 원천 데이터 문제 | 해당 구간 input 확인 |
| 모든 날짜 실패 | 코드/환경/권한 문제 | 최근 배포와 환경 변경 확인 |

## 운영 전 최종 체크리스트

DAG 설계:

- Task 의존성이 명확한가
- `schedule`, `catchup`, `max_active_runs`가 의도와 맞는가
- retry와 timeout이 설정되어 있는가
- pool/queue가 필요한 Task에 지정되어 있는가

데이터:

- 입력과 출력이 날짜/시간 파티션으로 분리되어 있는가
- 재실행해도 결과가 중복되지 않는가
- 큰 데이터가 XCom에 들어가지 않는가
- 중간 실패 산출물 처리 방식이 있는가

환경:

- Worker Python 버전과 패키지를 확인했는가
- provider 설치 여부를 확인했는가
- helper package/비밀 없는 설정이 배포되고 실행 worker에 승인된 secret이 공급되는가
- Worker가 필요한 저장소와 DB에 접근 가능한가

운영:

- 실패 알림이 있는가
- 로그에서 원인을 추적할 수 있는가
- 수동 재실행 절차가 있는가
- backfill 절차가 있는가
- 운영팀에 필요한 권한과 리소스를 요청했는가

## 마무리

Airflow 운영의 핵심은 DAG 문법보다 실행 환경과 재실행 가능성이다.

작은 DAG 하나를 안정적으로 만들고, 그 다음 패키지 격리, 공유 저장소, 알림, pool, backfill을 단계적으로 붙이는 방식이 가장 안전하다.


## 검증 근거 — 2026-10-04

- [2.10.5 Dataset](https://airflow.apache.org/docs/apache-airflow/2.10.5/authoring-and-scheduling/datasets.html), [Dynamic Task Mapping](https://airflow.apache.org/docs/apache-airflow/2.10.5/authoring-and-scheduling/dynamic-task-mapping.html), [pools](https://airflow.apache.org/docs/apache-airflow/2.10.5/administration-and-deployment/pools.html).
- [callbacks](https://airflow.apache.org/docs/apache-airflow/2.10.5/administration-and-deployment/logging-monitoring/callbacks.html).
- [공식 2.10.5 Celery 전송 소스](https://github.com/apache/airflow/blob/2.10.5/airflow/providers/celery/executors/celery_executor_utils.py)의 `apply_async(queue=queue)`와 [KubernetesExecutor 소스](https://github.com/apache/airflow/blob/2.10.5/airflow/providers/cncf/kubernetes/executors/kubernetes_executor.py)의 `execute_async`/`PodGenerator.from_obj(executor_config)`를 대조했다. provider 독립 release나 실제 cluster 실행 증거는 아니다. 병렬 sleep/실제 외부 알림은 실행하지 않았다.

미확인: 사내 배포·계정·리소스·네트워크 조건과 실제 운영 성공. 중복 예제의 계약 통합·회사 정책 결정은 Herdr `pane_not_found`로 Claude 협의를 보류한다. [주제 정리 기록](../organization-log.md)에 진행 결과를 남긴다.

로컬 확인: Python 3.12.12/Airflow 2.10.5 임시 환경에서 이 장의 Python 구문과 완성 DAG 정의를 검사했다. Kubernetes provider import·실제 venv job·외부 접속·scheduler 실행은 별도 미확인이다. 추가 실행 결과와 판본은 위 정리 기록을 읽는다.
