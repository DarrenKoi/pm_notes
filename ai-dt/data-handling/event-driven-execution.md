---
tags: [airflow, event-driven, sensor, dataset, trigger, webhook]
level: intermediate
last_updated: 2026-05-02
---

# Airflow Event-Driven Execution — 이벤트 기반 실행 패턴

> 시간 스케줄이 아니라 "특정 사건 발생 시" DAG/Task를 실행하는 모든 방법

## 왜 필요한가? (Why)

### 시간 스케줄(`@daily`, `0 2 * * *`)의 한계
- **데이터 도착 시점이 가변적** — 업스트림 시스템이 새벽 2시에 끝날 줄 알았는데 어제는 3시 30분에 끝났다면? Task는 빈 데이터로 돌고 실패
- **이른 실행 시 데이터 누락**, **늦은 실행 시 가치 지연**
- **폴링 낭비** — "혹시 왔나?" 1분마다 확인하면 99% 헛수고

### 이벤트 기반의 가치
- "**데이터가 실제로 도착했을 때**" 정확히 한 번 실행
- "다른 작업이 끝났을 때" 자동 연쇄 실행 (cross-team)
- "외부 시스템에서 webhook으로 신호가 왔을 때" 즉시 처리

---

## 핵심 개념 (What)

### 5가지 메커니즘

| 메커니즘 | 누가 주도? | 적합 시나리오 | 복잡도 |
|---------|-----------|--------------|--------|
| **1. Sensor** | Airflow가 폴링 | 파일/객체/쿼리 결과 대기 | ★ |
| **2. Dataset (Asset)** | Airflow 내부 pub/sub | DAG 간 의존 (같은 Airflow 내) | ★ |
| **3. Deferrable Operator** | Airflow + Triggerer | 장시간 대기, 슬롯 절약 | ★★ |
| **4. REST API trigger** | 외부 시스템이 push | webhook, CI/CD, 다른 서비스 | ★★ |
| **5. Message queue 연동** | Kafka/SQS/MinIO notification | 진정한 실시간 이벤트 | ★★★ |

---

## 어떻게 사용하는가? (How)

### 메커니즘 1. Sensor — 조건이 충족될 때까지 대기

Sensor는 "특정 조건이 참이 될 때까지 주기적으로 확인"하는 특수한 Operator. 조건이 만족되면 success → downstream 실행.

#### 1-A. 파일 도착 대기 (`FileSensor`)
```python
from airflow.sensors.filesystem import FileSensor

wait_file = FileSensor(
    task_id="wait_for_input_file",
    filepath="/data/landing/{{ ds }}/input.csv",
    poke_interval=60,         # 60초마다 확인
    timeout=60 * 60 * 6,      # 최대 6시간 대기 후 실패
    mode="reschedule",        # 슬롯 점유 안 함 (긴 대기 시 권장)
)

wait_file >> process_task
```

#### 1-B. MinIO/S3 객체 도착 대기 (`S3KeySensor`)
```python
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor

wait_object = S3KeySensor(
    task_id="wait_for_minio_object",
    bucket_name="raw-data",
    bucket_key="daily/{{ ds }}/done.flag",   # 와일드카드 가능: "daily/{{ ds }}/*.parquet"
    aws_conn_id="minio_default",
    poke_interval=120,
    timeout=60 * 60 * 12,
    mode="reschedule",
)

wait_object >> download_and_process
```

> **팁**: 데이터 파일 자체를 기다리지 말고, **업스트림이 작업 완료 후 만드는 `done.flag` 같은 sentinel 파일**을 기다리는 게 안전하다. 데이터 파일은 쓰기 도중에도 일부만 보일 수 있다.

#### 1-C. DB 쿼리 결과 대기 (`SqlSensor`)
```python
from airflow.providers.common.sql.sensors.sql import SqlSensor

wait_rows = SqlSensor(
    task_id="wait_for_today_data",
    conn_id="warehouse_db",
    sql="SELECT COUNT(*) FROM ingestion_log WHERE date = '{{ ds }}' AND status = 'DONE'",
    success=lambda x: x > 0,   # 1건 이상이면 success
    poke_interval=300,
    mode="reschedule",
)
```

#### `mode` 의 차이 — 매우 중요

| mode | 동작 | 슬롯 | 권장 |
|------|------|-----|------|
| `poke` (기본) | worker 슬롯을 점유한 채 sleep & 체크 | 점유 | 짧은 대기 (수 분) |
| `reschedule` | 체크 후 스스로 종료 → 다음 체크 시점에 재실행 | 해제 | 긴 대기 (수 시간) |

> 잘못된 mode는 worker pool을 통째로 마비시킨다. 1시간 이상 기다릴 가능성이 있으면 무조건 `reschedule`.

---

### 메커니즘 2. Dataset (Asset) — Producer/Consumer 자동 트리거

Airflow 2.4+ 도입, 3.0에서 **Asset**으로 개명. **Producer DAG가 데이터셋을 갱신하면 Consumer DAG가 자동으로 트리거된다** — 별도 sensor 불필요.

```python
from datetime import datetime
from airflow.decorators import dag, task
from airflow.datasets import Dataset

# 데이터셋 정의 — URI는 식별자일 뿐, 실제 위치와 일치할 필요는 없음
RAW_DATA = Dataset("s3://raw-data/daily/")
CLEAN_DATA = Dataset("s3://processed-data/cleaned/")


# Producer 1: 시간 스케줄로 도는 수집 DAG
@dag(start_date=datetime(2026, 5, 1), schedule="@hourly", catchup=False)
def ingest_pipeline():

    @task(outlets=[RAW_DATA])     # ← 이 Task가 RAW_DATA를 갱신함을 선언
    def ingest():
        # MinIO에 새 파일 업로드
        ...

    ingest()

ingest_pipeline()


# Producer 2: 정제 DAG — RAW_DATA가 갱신되면 자동 실행
@dag(start_date=datetime(2026, 5, 1), schedule=[RAW_DATA], catchup=False)
def cleaning_pipeline():

    @task(outlets=[CLEAN_DATA])   # 이 DAG도 CLEAN_DATA를 produce
    def clean():
        ...

    clean()

cleaning_pipeline()


# Consumer: CLEAN_DATA가 갱신되면 자동 실행
@dag(start_date=datetime(2026, 5, 1), schedule=[CLEAN_DATA], catchup=False)
def analytics_pipeline():

    @task
    def analyze():
        ...

    analyze()

analytics_pipeline()
```

**무엇이 좋은가**:
- Sensor 폴링 없음 — Airflow 내부에서 즉시 트리거
- 의존 관계가 **Datasets 탭에서 그래프로 시각화**됨
- Consumer 여러 개가 같은 Producer를 구독해도 **각자 독립적으로 트리거**

**여러 Dataset을 AND/OR로 조합** (Airflow 2.9+):
```python
from airflow.datasets import DatasetOrTimeSchedule
from airflow.timetables.datasets import DatasetOrTimeSchedule

@dag(schedule=(RAW_DATA & REFERENCE_DATA))   # 둘 다 갱신돼야 실행
def needs_both(): ...

@dag(schedule=(RAW_DATA | BACKUP_DATA))      # 하나라도 갱신되면 실행
def either_one(): ...
```

> 한계: 같은 Airflow 인스턴스 내부에서만 동작. 다른 클러스터/회사 간에는 메커니즘 4·5 필요.

---

### 메커니즘 3. Deferrable Operator — 슬롯 점유 없이 비동기 대기

긴 대기를 worker 슬롯 점유 없이 처리. **별도 `triggerer` 프로세스**가 비동기 I/O로 수천 개 이벤트를 동시 감시.

```python
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor

wait_async = S3KeySensor(
    task_id="wait_async",
    bucket_name="raw-data",
    bucket_key="daily/{{ ds }}/done.flag",
    aws_conn_id="minio_default",
    deferrable=True,         # ← 이 한 줄로 deferrable 모드
    poke_interval=30,
)
```

| 비교 | 일반 Sensor (`mode=reschedule`) | Deferrable Sensor |
|------|-------------------------------|-------------------|
| 워커 슬롯 | 체크 시점만 점유 | 거의 점유 안 함 |
| 폴링 단위 | reschedule 주기 (수 분~) | 수 초 가능 |
| 동시 감시 가능 수 | worker 수 제한 | 수천 개 (asyncio) |
| 운영 요구사항 | 없음 | `triggerer` 프로세스 필요 |

> 사내 Airflow가 triggerer를 띄워뒀는지 운영팀에 확인. 안 띄워져 있으면 deferrable Task는 무한 대기.

---

### 메커니즘 4. REST API Trigger — 외부에서 push

외부 시스템(CI/CD, 다른 서비스, 사용자)이 HTTP 요청으로 DAG를 즉시 실행.

#### Airflow REST API
```bash
# DAG run 생성
curl -X POST \
  -u "user:password" \
  -H "Content-Type: application/json" \
  -d '{
    "dag_run_id": "manual__2026-05-02T10:00:00",
    "conf": {
      "input_path": "s3://raw-data/special/2026-05-02.csv",
      "priority": "high"
    }
  }' \
  https://airflow.your-company.com/api/v1/dags/my_pipeline/dagRuns
```

DAG에서 `conf`를 받아 사용:
```python
from airflow.decorators import dag, task

@dag(start_date=datetime(2026, 5, 1), schedule=None, catchup=False)
def my_pipeline():

    @task
    def process(**context):
        conf = context["dag_run"].conf
        input_path = conf.get("input_path", "default")
        priority = conf.get("priority", "normal")
        print(f"Processing {input_path} with priority {priority}")

    process()

my_pipeline()
```

> `schedule=None` 으로 두면 **수동/API 트리거 전용 DAG**가 된다.

#### 활용 예시
- **GitHub Actions**: 배포 후 마이그레이션 DAG 자동 실행
- **사내 웹앱**: 사용자가 "분석 요청" 버튼 클릭 → API 호출 → 즉시 처리
- **MinIO bucket notification → Lambda/Function → API 호출**: 진정한 이벤트 기반

---

### 메커니즘 5. Message Queue / Object Storage 알림 연동

가장 강력한 패턴 — 외부 이벤트 발생 → 메시지 → Airflow.

#### 시나리오: MinIO 버킷에 파일 업로드 → 즉시 처리

```
[업로더]
   ↓ (s3:ObjectCreated:Put)
[MinIO bucket notification]
   ↓ (webhook 또는 큐)
[수신자: Lambda / Knative / 작은 FastAPI 서비스]
   ↓ (HTTP POST)
[Airflow REST API]
   ↓ (DAG run 생성, conf로 객체 키 전달)
[처리 DAG]
```

MinIO 측 설정 (개념):
```bash
# 버킷에 webhook 알림 설정
mc event add minio/raw-data arn:minio:sqs::primary:webhook \
  --event put --suffix .csv
```

수신자 (FastAPI 예시):
```python
from fastapi import FastAPI, Request
import httpx

app = FastAPI()
AIRFLOW_API = "https://airflow.your-company.com/api/v1"

@app.post("/minio-event")
async def handle(request: Request):
    payload = await request.json()
    for record in payload.get("Records", []):
        key = record["s3"]["object"]["key"]
        async with httpx.AsyncClient(auth=("user", "pw")) as client:
            await client.post(
                f"{AIRFLOW_API}/dags/process_uploaded_file/dagRuns",
                json={"conf": {"object_key": key, "bucket": "raw-data"}},
            )
    return {"ok": True}
```

#### Kafka 연동 — Long-running consumer DAG
별도 패턴: Airflow가 직접 메시지를 소비하기보다, **Kafka Connect / 별도 consumer가 메시지를 처리하고 마일스톤마다 Airflow API 호출** 하는 게 더 안정적이다. Airflow는 배치 오케스트레이션에 최적화되어 있지, 실시간 스트림 처리에는 부적합.

---

## 의사결정 가이드

```
"무언가 일어났을 때 실행하고 싶다"
│
├─ 그 "무언가"가 같은 Airflow 안의 다른 DAG?
│  └─ Dataset (Asset) ← 1순위, 가장 깔끔
│
├─ 파일/객체/DB row 가 도착하길 기다림?
│  ├─ 짧은 대기 (수 분) → Sensor (mode=poke)
│  ├─ 긴 대기 (수 시간) → Sensor (mode=reschedule)
│  └─ 매우 긴 대기 + 많은 동시 감시 → Deferrable Sensor
│
├─ 외부 시스템(웹앱, CI, 다른 서비스)이 시작 신호를 줌?
│  └─ REST API + schedule=None DAG
│
├─ 진짜 실시간, 객체 업로드 즉시 처리?
│  └─ MinIO/S3 notification → 수신자 → Airflow API
│
└─ 메시지 큐 (Kafka 등) 기반 스트림?
   └─ 별도 consumer + 마일스톤마다 Airflow API 호출 (Airflow는 배치용)
```

---

## 자주 하는 실수

| 실수 | 결과 | 해결 |
|------|------|------|
| 6시간 대기 Sensor를 `mode="poke"`로 둠 | worker pool 마비 | `mode="reschedule"` 또는 `deferrable=True` |
| 데이터 파일 자체를 sensor로 대기 | 쓰기 중인 파일을 잡아 처리 시작 | 업스트림이 만드는 `done.flag` sentinel 파일 대기 |
| Dataset 의존을 선언했는데 trigger 안 됨 | Producer가 `outlets=[...]` 누락 | Producer Task 데코레이터에 `outlets` 명시 확인 |
| API trigger DAG에 `schedule="@daily"` 둠 | 수동 트리거 외에 매일 자동 실행됨 | API 전용 DAG는 `schedule=None` |
| Airflow를 Kafka 실시간 consumer로 사용 | 메모리 누적, Task 무한 실행 | Airflow는 배치, 스트림은 별도 consumer |
| Deferrable Operator 썼는데 무한 대기 | triggerer 프로세스 미기동 | 운영팀에 triggerer 확인 |
| Sensor timeout 미설정 | 무한 대기로 자원 낭비 | `timeout` 항상 명시 (예: 12시간) |

---

## 참고 자료 (References)

- [Airflow Sensors](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/sensors.html)
- [Datasets (Data-aware scheduling)](https://airflow.apache.org/docs/apache-airflow/stable/authoring-and-scheduling/datasets.html)
- [Deferrable Operators & Triggers](https://airflow.apache.org/docs/apache-airflow/stable/authoring-and-scheduling/deferring.html)
- [Airflow REST API](https://airflow.apache.org/docs/apache-airflow/stable/stable-rest-api-ref.html)
- [MinIO Bucket Notifications](https://min.io/docs/minio/linux/administration/monitoring/bucket-notifications.html)
- [S3KeySensor Provider](https://airflow.apache.org/docs/apache-airflow-providers-amazon/stable/sensors/s3.html)

## 관련 문서
- [Airflow + MinIO 파이프라인 튜토리얼](./airflow-minio-tutorial.md)
- [Task 의존성 — Python 코드 순차 실행 패턴](./task-dependencies.md)
- [AI/DT 학습 노트](../README.md)
