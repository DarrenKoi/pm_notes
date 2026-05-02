---
tags: [airflow, sensor, dataset, dynamic-task-mapping, backfill, operations]
level: advanced
last_updated: 2026-05-02
status: complete
---

# 07. 고급 운영 패턴

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

긴 대기에는 `mode="reschedule"`을 우선 고려한다. `poke`로 수시간 대기하면 Worker slot을 낭비할 수 있다.

## done.flag 패턴

데이터 파일 자체를 기다리는 것보다 완료 신호 파일을 기다리는 것이 안전하다.

```text
s3://raw/sales/dt=2026-05-02/data_001.parquet
s3://raw/sales/dt=2026-05-02/data_002.parquet
s3://raw/sales/dt=2026-05-02/_DONE
```

업스트림 시스템이 모든 파일을 쓴 뒤 `_DONE` 파일을 만들고, Airflow는 `_DONE`을 기다린다.

이 방식은 "파일이 보이지만 아직 쓰는 중"인 문제를 줄인다.

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

Dataset은 같은 Airflow 인스턴스 안에서 DAG 간 의존성을 표현하기 좋다. 다른 시스템의 외부 이벤트를 직접 받는 용도라면 Sensor, REST API trigger, message queue 연동을 검토한다.

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

- 너무 많은 Task를 한 번에 만들면 Scheduler와 UI가 느려질 수 있다.
- 수천 개 이상의 작은 Task보다 적당히 묶어서 처리하는 것이 나을 수 있다.
- pool로 동시 실행 개수를 제한한다.

## pool

pool은 특정 리소스를 사용하는 Task의 동시 실행 수를 제한한다.

예:

- DB connection을 많이 쓰는 Task는 동시에 3개만
- 외부 API 호출은 동시에 5개만
- 무거운 분석 작업은 동시에 1개만

DAG에서 pool 지정:

```python
task = BashOperator(
    task_id="call_api",
    bash_command="python call_api.py --date {{ ds }}",
    pool="external_api_pool",
)
```

pool 생성은 운영팀 권한일 수 있다.

## queue

CeleryExecutor나 KubernetesExecutor 환경에서는 queue를 통해 특정 Worker 그룹으로 Task를 보낼 수 있다.

```python
task = BashOperator(
    task_id="heavy_job",
    bash_command="python heavy_job.py --date {{ ds }}",
    queue="high_memory",
)
```

queue 이름과 Worker 구성이 회사마다 다르므로 운영팀에 확인한다.

## backfill

backfill은 과거 날짜 데이터를 다시 처리하는 작업이다.

예:

```text
2026-04-01부터 2026-04-30까지 재처리
```

주의:

- output path가 날짜 파티션으로 분리되어 있어야 한다.
- DB insert가 중복을 만들지 않아야 한다.
- `max_active_runs`와 pool로 동시 실행량을 제한해야 한다.
- 과거 데이터가 현재 코드와 호환되는지 확인해야 한다.
- 외부 API를 과거 날짜만큼 대량 호출하면 rate limit에 걸릴 수 있다.

회사 관리형 Airflow에서는 CLI backfill 권한이 없을 수 있다. 그 경우 UI에서 날짜별 수동 실행하거나 운영팀에 요청한다.

## 수동 재실행

Airflow UI에서 실패한 Task를 Clear하면 해당 Task와 downstream Task를 다시 실행할 수 있다.

재실행 전에 확인할 것:

- 이 Task가 같은 날짜로 다시 실행되어도 안전한가
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

실제 운영에서는 print 대신 사내 알림 API를 호출한다. 단, 알림 함수 안에서도 secret을 로그에 남기지 않는다.

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
6. Connection/권한 오류인지 확인
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
| 특정 날짜만 실패 | 원천 데이터 문제 | 해당 날짜 input 확인 |
| 모든 날짜 실패 | 코드/환경/권한 문제 | 최근 배포와 환경 변경 확인 |

## 운영 전 최종 체크리스트

DAG 설계:

- Task 의존성이 명확한가
- `schedule`, `catchup`, `max_active_runs`가 의도와 맞는가
- retry와 timeout이 설정되어 있는가
- pool/queue가 필요한 Task에 지정되어 있는가

데이터:

- 입력과 출력이 날짜 파티션으로 분리되어 있는가
- 재실행해도 결과가 중복되지 않는가
- 큰 데이터가 XCom에 들어가지 않는가
- 중간 실패 산출물 처리 방식이 있는가

환경:

- Worker Python 버전과 패키지를 확인했는가
- provider 설치 여부를 확인했는가
- Connection/Variable이 운영 환경에 있는가
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
