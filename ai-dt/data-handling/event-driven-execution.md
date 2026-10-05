---
tags: [airflow, event-driven, sensor, dataset, trigger, webhook]
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

# Airflow Event-Driven Execution — 이벤트 기반 실행 패턴

> [!info] 검증 판본과 적용 조건 — 2026-10-04
> Airflow **2.10.5**의 Dataset·Sensor·deferral·REST API를 기준으로 대조했다. 아래 `airflow.decorators`/`airflow.datasets`와 `/api/v1`은 2.x 예제다. 3.x에서는 `airflow.sdk.Asset` 및 `/api/v2`·JWT 인증을 별도로 적용한다. 확인 당시 공식 3.x 문서 표시는 3.3.2였으며 회사 설치 버전·provider·triggerer·auth manager는 미확인이다.
>
> Task를 어떤 순서로 실행하는지는 [의존성 패턴](./task-dependencies.md), DAG 시작 조건은 이 문서가 담당한다. 시간 스케줄을 없애도 재시도·외부 중복 요청·업무 부작용의 exactly-once는 보장되지 않는다. 아래 서비스·명령은 학습 예시이며 실제 운영 배포를 검증하지 않았다.


> 시간 스케줄이 아니라 "특정 사건 발생 시" DAG/Task를 실행하는 모든 방법

## 왜 필요한가? (Why)

### 시간 스케줄(`@daily`, `0 2 * * *`)의 한계
- **데이터 도착 시점이 가변적** — 업스트림 시스템이 새벽 2시에 끝날 줄 알았는데 어제는 3시 30분에 끝났다면? Task는 빈 데이터로 돌고 실패
- **이른 실행 시 데이터 누락**, **늦은 실행 시 가치 지연**
- **폴링 비용** — 조건 확인 주기와 데이터 도착 빈도에 따라 불필요한 조회가 생긴다. 기존 99% 수치는 측정 근거가 없어 일반 사실로 사용하지 않는다

### 이벤트 기반의 가치
- 데이터 준비 신호를 시작 조건으로 사용할 수 있다. 이벤트 1개와 DAG run 1개가 항상 일대일이거나 부작용이 정확히 한 번이라는 뜻은 아니다
- "다른 작업이 끝났을 때" 자동 연쇄 실행 (cross-team)
- webhook 요청으로 run을 만들 수 있다. 큐·scheduler·worker 지연이 있어 즉시 완료를 보장하지 않는다

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
    bucket_key="daily/{{ ds }}/done.flag",   # wildcard_match=True일 때 glob 가능: "daily/{{ ds }}/*.parquet"
    aws_conn_id="minio_default",
    poke_interval=120,
    timeout=60 * 60 * 12,
    mode="reschedule",
)

wait_object >> download_and_process
```

> **완료 신호의 의미**: 로컬 파일은 쓰는 중에도 보일 수 있고, 객체 하나의 존재도 전체 batch 완료를 뜻하지 않는다. `done.flag`는 모든 필요한 객체·검증이 완료된 뒤 작성한다는 생산자 계약이 있을 때 유효하다. Amazon S3는 단일 key 업데이트의 원자성을 제공하므로 S3 객체가 부분적으로 보인다고 일반화하지 않는다. 사내 MinIO 버전·일관성·완료 계약은 미확인이다.

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

> 장시간 `poke`는 worker 슬롯을 점유한다. `reschedule`은 체크 사이 슬롯을 해제하되 scheduler 부담·검사 지연이 있다. 시간 하나로 무조건 선택하지 않고 polling 주기·동시 대기 수·provider의 deferral 지원을 함께 본다.

---

### 메커니즘 2. Dataset (Asset) — Producer/Consumer 자동 트리거

Dataset scheduling은 2.4에서 도입됐다. `outlets`를 선언한 **Task가 성공**하면 갱신 이벤트를 기록하고 소비 DAG의 scheduling 조건에 반영한다. skip/실패는 갱신 이벤트를 만들지 않는다. URI는 논리 식별자이며 Airflow가 실제 객체 내용을 자동 검증하지 않는다. 3.x public API에서는 `Asset`을 사용한다.

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
- 소비 DAG의 sensor 없이 갱신 이벤트를 scheduling에 반영한다. 실행 지연이나 데이터 검증이 사라지는 것은 아니다
- 의존 관계가 **Datasets 탭에서 그래프로 시각화**됨
- Consumer 여러 개가 같은 Producer를 구독해도 **각자 독립적으로 트리거**

**여러 Dataset을 AND/OR로 조합** (Airflow 2.9+):
```python
# 앞 예제의 RAW_DATA를 함께 사용한다. 시간표 결합은 별도 DatasetOrTimeSchedule 예제다.
REFERENCE_DATA = Dataset("s3://reference-data/current/")
BACKUP_DATA = Dataset("s3://backup-data/daily/")

@dag(schedule=(RAW_DATA & REFERENCE_DATA))   # 둘 다 갱신돼야 실행
def needs_both(): ...

@dag(schedule=(RAW_DATA | BACKUP_DATA))      # 하나라도 갱신되면 실행
def either_one(): ...
```

> 여러 dataset을 list/AND로 소비하면 마지막 소비 이후 각각 최소 한 번 갱신될 때 실행한다. 한 dataset의 여러 갱신이 하나의 소비 run으로 합쳐질 수 있다. URI/extra는 metadata DB의 평문이므로 자격 증명을 넣지 않는다. 다른 클러스터의 실제 객체 업로드를 이 선언만으로 감지하지 않는다.

---

### 메커니즘 3. Deferrable Operator — 슬롯 점유 없이 비동기 대기

긴 대기를 worker 슬롯 점유 없이 처리. **별도 `triggerer` 프로세스**가 비동기 trigger를 감시한다. 실제 감시 한도는 설정·provider·부하로 검증한다.

```python
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor

wait_async = S3KeySensor(
    task_id="wait_async",
    bucket_name="raw-data",
    bucket_key="daily/{{ ds }}/done.flag",
    aws_conn_id="minio_default",
    deferrable=True,         # ← 이 한 줄로 deferrable 모드
    poke_interval=30,
    timeout=60 * 60 * 12,    # provider 동작과 실제 timeout 처리 확인 필요
)
```

| 비교 | 일반 Sensor (`mode=reschedule`) | Deferrable Sensor |
|------|-------------------------------|-------------------|
| 워커 슬롯 | 체크 시점 점유 | deferred 동안 해제; 시작/재개 시 점유 |
| 폴링 단위 | reschedule 주기 (수 분~) | 수 초 가능 |
| 동시 감시 가능 수 | scheduler/worker 처리량·체크 주기에 영향 | triggerer capacity·I/O·provider에 영향 |
| 운영 요구사항 | scheduler/worker·대상 접근 권한 | 추가로 `triggerer`와 provider/비동기 의존성 필요 |

> triggerer와 필요한 비동기 의존성을 확인한다. 없으면 deferred 상태에서 진행하지 못할 수 있으며 timeout/실패 동작은 Operator별로 확인한다. deferred task의 pool 슬롯 점유 여부도 pool 설정에 따른다.

---

### 메커니즘 4. REST API Trigger — 외부에서 push

외부 시스템(CI/CD, 다른 서비스, 사용자)이 HTTP 요청으로 DAG를 즉시 실행.

#### Airflow REST API

아래는 **2.x `/api/v1`에서 Basic 인증 backend가 허용된 경우**의 예시다. 실제 주소·계정은 자리표시자다. Airflow 3 문서는 `/api/v2`와 Bearer JWT를 사용하며 token 발급은 설치된 auth manager에 따른다. run ID는 중복 요청을 구별할 수 있게 설계하되 API의 중복 거부가 외부 부작용의 exactly-once까지 보장하지 않는다.
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

수신자 (FastAPI **개념 예시; 배포 미완성**):

이 예시는 요청 인증·허용 bucket/key·이벤트 고유 ID·내구성 중복 제거·key 디코딩·복구 큐를 구현하지 않는다. 원래 고정 `raw-data`와 Basic 계정은 자리표시자다. 재전송/HTTP 충돌 정책 및 3.x 인증 전환은 Claude 협의·실제 사내 조건 확인 후 결정한다. HTTP 실패를 성공으로 응답하지 않도록 `raise_for_status()`를 추가했지만 운영 수신 서비스의 완료를 뜻하지 않는다.
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
            response = await client.post(
                f"{AIRFLOW_API}/dags/process_uploaded_file/dagRuns",
                json={"conf": {"object_key": key, "bucket": "raw-data"}},
            )
            response.raise_for_status()
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
| 객체 하나만으로 batch 전체 완료를 판단 | 다른 객체·검증이 아직 안 끝남 | 생산자 완료 계약·manifest 확인; 로컬 쓰기 중 파일과 S3 단일 key 원자성 구분 |
| Dataset 의존을 선언했는데 trigger 안 됨 | Producer가 `outlets=[...]` 누락 | Producer Task 데코레이터에 `outlets` 명시 확인 |
| API trigger DAG에 `schedule="@daily"` 둠 | 수동 트리거 외에 매일 자동 실행됨 | API 전용 DAG는 `schedule=None` |
| Airflow를 Kafka 실시간 consumer로 사용 | 메모리 누적, Task 무한 실행 | Airflow는 배치, 스트림은 별도 consumer |
| Deferrable Operator 썼는데 무한 대기 | triggerer 프로세스 미기동 | 운영팀에 triggerer 확인 |
| timeout 기본값을 확인하지 않음 | 의도보다 오래 기다림 | 기본값·DAG deadline과 provider timeout 동작 확인 후 명시 |

---

## 참고 자료 (References)

- [Airflow Sensors](https://airflow.apache.org/docs/apache-airflow/2.10.5/core-concepts/sensors.html)
- [Datasets (Data-aware scheduling)](https://airflow.apache.org/docs/apache-airflow/2.10.5/authoring-and-scheduling/datasets.html)
- [Deferrable Operators & Triggers](https://airflow.apache.org/docs/apache-airflow/2.10.5/authoring-and-scheduling/deferring.html)
- [Airflow REST API](https://airflow.apache.org/docs/apache-airflow/2.10.5/stable-rest-api-ref.html)
- [MinIO Bucket Notifications](https://min.io/docs/minio/linux/administration/monitoring/bucket-notifications.html)
- [S3KeySensor Provider](https://airflow.apache.org/docs/apache-airflow-providers-amazon/stable/sensors/s3.html)

## 관련 문서
- [Airflow + MinIO 파이프라인 튜토리얼](./airflow-minio-tutorial.md)
- [Task 의존성 — Python 코드 순차 실행 패턴](./task-dependencies.md)


### 추가 확인 근거 — 2026-10-04

- [조건식이 있는 Dataset 2.9.3](https://airflow.apache.org/docs/apache-airflow/2.9.3/authoring-and-scheduling/datasets.html), [Dataset 2.10.5](https://airflow.apache.org/docs/apache-airflow/2.10.5/authoring-and-scheduling/datasets.html), [deferral와 pool](https://airflow.apache.org/docs/apache-airflow/2.10.5/authoring-and-scheduling/deferring.html).
- [Airflow 3 public interface](https://airflow.apache.org/docs/apache-airflow/stable/public-airflow-interface.html), [3.3.2 API 인증](https://airflow.apache.org/docs/apache-airflow/3.3.2/security/api.html).
- [Amazon S3 일관성](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html), [HTTPX 종료 상태 검사](https://www.python-httpx.org/quickstart/).
- MinIO notification의 사내 버전/설정 및 Amazon provider의 정확한 설치 버전은 미확인이다. 이 문서의 `mc` 명령과 실제 이벤트 수신·scheduler·API 인증은 실행하지 않았다.
