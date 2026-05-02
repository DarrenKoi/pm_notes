---
tags: [airflow, orchestration, minio, data-pipeline, etl]
level: beginner
last_updated: 2026-05-02
status: in-progress
---

# Airflow + MinIO 파이프라인 튜토리얼

> 폐쇄망/사내 Airflow 환경에서 Python 스크립트를 순차 실행하여 MinIO의 파일을 다운로드·분석하는 ETL 파이프라인 구축 가이드

## 왜 필요한가? (Why)

### 단순 cron / 스크립트의 한계
- **의존성 관리 부재**: A 스크립트가 실패해도 B가 그대로 돌아 데이터 꼬임 발생
- **재실행/재처리 불편**: 특정 날짜의 데이터만 다시 돌리려면 수동 작업
- **모니터링 부재**: 어디서 실패했는지, 얼마나 걸렸는지 추적 불가
- **로그 분산**: 각 스크립트가 자기 로그를 따로 남겨서 한 번에 보기 어려움

### Airflow가 해결하는 것
- **DAG (Directed Acyclic Graph)**로 작업 간 순서·의존성 명시 → 앞 단계가 성공해야 다음 단계 실행
- **Scheduler**가 자동으로 정해진 시간에 실행, 실패 시 재시도(retry) 자동화
- **Web UI**로 각 Task의 성공/실패, 로그, 실행 시간 한눈에 확인
- **XCom / Connection / Variable**로 Task 간 데이터·설정 공유

### 사내 폐쇄망 환경에서의 실제 맥락
SK하이닉스처럼 외부 PyPI 접근이 차단된 환경에서는,
- Airflow Worker가 동작하는 컨테이너/VM에 **사내 Nexus/Artifactory**로부터 패키지를 설치하거나
- DAG 별로 독립된 **virtualenv**를 만들어 패키지를 격리하거나
- 미리 패키지가 설치된 **Docker 이미지**를 빌드해서 KubernetesPodOperator로 실행

해야 한다. 이 문서는 그 중에서도 가장 흔한 시나리오 — **이미 운영 중인 Airflow에 DAG만 올려서 사용** — 를 중심으로 한다.

---

## 핵심 개념 (What)

### Airflow 아키텍처

```
┌─────────────────────────────────────────────┐
│  Web Server (UI)                            │
│  - DAG 시각화, 로그 조회, 수동 트리거         │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│  Metadata DB (PostgreSQL/MySQL)             │
│  - DAG 상태, Task 실행 이력, Connection 등   │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│  Scheduler                                  │
│  - dags/ 폴더 파싱, 스케줄 판단,             │
│    실행 가능한 Task를 Executor에 큐잉        │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│  Executor (Local/Celery/Kubernetes)         │
│  - Worker에 Task 분배                       │
└──────────────────┬──────────────────────────┘
                   │
            ┌──────▼──────┐
            │   Worker    │ ← 실제 Python 코드 실행
            └─────────────┘
```

### 핵심 용어

| 용어 | 설명 |
|------|------|
| **DAG** | Task들의 의존 관계를 정의한 그래프. 보통 `dags/` 폴더의 `.py` 파일 하나가 DAG 하나 |
| **Task** | DAG 안의 실행 단위. Operator의 인스턴스 |
| **Operator** | "무엇을 할지"를 정의한 템플릿 (PythonOperator, BashOperator 등) |
| **Task Instance** | 특정 실행 시점(execution_date)의 Task |
| **Executor** | Task를 어디서/어떻게 실행할지 (Local, Celery, Kubernetes) |
| **XCom** | Task 간 작은 데이터를 주고받는 메커니즘 (cross-communication) |
| **Connection** | 외부 시스템(DB, S3, MinIO 등) 접속 정보를 안전하게 저장 |
| **Variable** | DAG에서 쓰는 설정값을 UI/CLI로 관리 |

### 자주 쓰는 Operator

| Operator | 용도 |
|----------|------|
| `PythonOperator` | 같은 Python 환경에서 함수 실행 (가장 기본) |
| `PythonVirtualenvOperator` | 일회성 가상환경을 만들어 격리 실행 |
| `ExternalPythonOperator` | 미리 만들어둔 가상환경을 재사용 |
| `BashOperator` | 셸 명령 실행 |
| `DockerOperator` | Docker 컨테이너로 실행 |
| `KubernetesPodOperator` | k8s Pod로 실행 (사내 클러스터 환경에서 격리에 유리) |
| `S3Hook` (provider) | S3 호환 스토리지(MinIO 포함) 접근 |

### TaskFlow API (Airflow 2.x 권장 방식)
데코레이터 기반으로 DAG를 더 파이썬답게 작성:

```python
from airflow.decorators import dag, task
from datetime import datetime

@dag(start_date=datetime(2026, 1, 1), schedule="@daily", catchup=False)
def my_pipeline():
    @task
    def extract():
        return {"records": [1, 2, 3]}

    @task
    def transform(data):
        return [x * 2 for x in data["records"]]

    @task
    def load(values):
        print(f"Loaded: {values}")

    load(transform(extract()))

my_pipeline()
```

`extract → transform → load` 의존성이 함수 호출 순서로 자동 정의된다.

---

## 어떻게 사용하는가? (How)

### 0. 사전 준비: 사내 Airflow 환경 파악

운영팀에 다음 정보를 확인한다:

- **Airflow 버전**: 2.x 권장 (TaskFlow API, dataset 등 기능 차이)
- **Executor 종류**: LocalExecutor / CeleryExecutor / KubernetesExecutor
- **DAG 배포 방식**: Git Sync / 공유 볼륨 업로드 / S3 Sync
- **패키지 설치 정책**: 사내 Nexus 사용 가능? `requirements.txt` 자동 설치 여부?
- **MinIO 엔드포인트, 버킷 권한, 액세스 키 발급 절차**

---

### 1. 폐쇄망에서 패키지 설치하기

#### 방법 A. 사내 PyPI 미러 (Nexus/Artifactory) 사용
운영 중인 Airflow Worker 이미지에서 사내 미러가 설정되어 있으면 가장 쉽다.

```bash
# pip.conf 또는 환경변수로 미러 지정
pip install minio pandas \
  --index-url https://nexus.your-company.com/repository/pypi/simple \
  --trusted-host nexus.your-company.com
```

> 운영팀이 이미지에 `pip.conf`를 미리 넣어둔 경우가 대부분이라, 보통 그냥 `pip install` 하면 된다.

#### 방법 B. PythonVirtualenvOperator (DAG별 격리)
글로벌 환경에 패키지를 추가할 권한이 없을 때 권장. **DAG가 실행될 때마다 가상환경을 만들고 패키지를 설치**한다.

```python
from airflow.operators.python import PythonVirtualenvOperator

def analyze():
    import pandas as pd
    from minio import Minio
    # 분석 로직
    df = pd.DataFrame({"a": [1, 2, 3]})
    print(df.describe())

analyze_task = PythonVirtualenvOperator(
    task_id="analyze",
    python_callable=analyze,
    requirements=["pandas==2.2.2", "minio==7.2.7"],
    system_site_packages=False,  # 시스템 패키지와 격리
    pip_install_options=[
        "--index-url", "https://nexus.your-company.com/repository/pypi/simple",
        "--trusted-host", "nexus.your-company.com",
    ],
)
```

> 주의: `python_callable` 안에서 사용하는 모든 import는 함수 **내부**에 둬야 한다. 외부 import는 가상환경에 없으므로 NameError가 발생한다.

#### 방법 C. ExternalPythonOperator (미리 만든 venv 재사용)
매 실행마다 설치하면 느리므로, Worker에 미리 가상환경을 만들어두고 재사용:

```bash
# Worker 호스트에서 한 번만
python -m venv /opt/airflow/venvs/data-pipeline
/opt/airflow/venvs/data-pipeline/bin/pip install pandas minio
```

```python
from airflow.operators.python import ExternalPythonOperator

analyze_task = ExternalPythonOperator(
    task_id="analyze",
    python="/opt/airflow/venvs/data-pipeline/bin/python",
    python_callable=analyze,
)
```

#### 방법 D. KubernetesPodOperator (사내 K8s + 사내 레지스트리)
이미지를 사내 Harbor 등에 미리 빌드해 올려두고 사용. 가장 깔끔한 격리.

```python
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator

analyze_task = KubernetesPodOperator(
    task_id="analyze",
    image="harbor.your-company.com/data/pipeline:1.0.0",
    cmds=["python", "/app/analyze.py"],
    name="analyze-pod",
    namespace="airflow",
)
```

---

### 2. MinIO 연결 설정

#### Connection으로 안전하게 등록 (권장)
액세스 키를 코드에 박지 말고 Airflow Connection으로 관리한다.

**Web UI에서**: Admin → Connections → `+`
- Connection Id: `minio_default`
- Connection Type: `Amazon Web Services` (S3 호환)
- AWS Access Key ID: `<MINIO_ACCESS_KEY>`
- AWS Secret Access Key: `<MINIO_SECRET_KEY>`
- Extra:
  ```json
  {
    "endpoint_url": "http://minio.your-company.com:9000",
    "region_name": "us-east-1"
  }
  ```

**CLI로**:
```bash
airflow connections add minio_default \
  --conn-type aws \
  --conn-login "$MINIO_ACCESS_KEY" \
  --conn-password "$MINIO_SECRET_KEY" \
  --conn-extra '{"endpoint_url": "http://minio.your-company.com:9000"}'
```

---

### 3. 순차 실행 파이프라인 만들기 (핵심)

> "앞 코드가 완벽히 끝나야 다음 코드가 실행" — Airflow의 **Task 의존성**으로 자연스럽게 보장된다.
> 기본적으로 Task는 **upstream이 success일 때만** 실행된다 (`trigger_rule="all_success"`가 기본값).

#### 폴더 구조
```
dags/
├── minio_analysis_pipeline.py     # DAG 정의
└── tasks/
    ├── __init__.py
    ├── download.py                # MinIO에서 파일 다운로드
    ├── preprocess.py              # 전처리
    └── analyze.py                 # 분석
```

#### `tasks/download.py`
```python
from pathlib import Path
from airflow.providers.amazon.aws.hooks.s3 import S3Hook


def download_from_minio(
    bucket: str,
    prefix: str,
    local_dir: str,
    conn_id: str = "minio_default",
) -> list[str]:
    """MinIO 버킷의 prefix 아래 파일을 모두 로컬에 내려받는다.

    Returns: 다운로드한 파일들의 로컬 경로 리스트 (XCom으로 다음 Task에 전달)
    """
    hook = S3Hook(aws_conn_id=conn_id)
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    keys = hook.list_keys(bucket_name=bucket, prefix=prefix) or []
    if not keys:
        raise ValueError(f"No objects under s3://{bucket}/{prefix}")

    downloaded = []
    for key in keys:
        if key.endswith("/"):
            continue
        local_path = str(Path(local_dir) / Path(key).name)
        hook.get_key(key, bucket_name=bucket).download_file(local_path)
        downloaded.append(local_path)

    print(f"Downloaded {len(downloaded)} files to {local_dir}")
    return downloaded
```

#### `tasks/preprocess.py`
```python
import pandas as pd
from pathlib import Path


def preprocess(file_paths: list[str], output_path: str) -> str:
    """다운로드한 CSV들을 합치고 정제한다."""
    if not file_paths:
        raise ValueError("No input files provided")

    dfs = [pd.read_csv(p) for p in file_paths]
    merged = pd.concat(dfs, ignore_index=True)

    # 결측치 제거, 중복 제거 등
    cleaned = merged.dropna().drop_duplicates()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(output_path, index=False)

    print(f"Preprocessed rows: {len(cleaned)} → {output_path}")
    return output_path
```

#### `tasks/analyze.py`
```python
import pandas as pd
from airflow.providers.amazon.aws.hooks.s3 import S3Hook


def analyze_and_upload(
    input_path: str,
    bucket: str,
    output_key: str,
    conn_id: str = "minio_default",
) -> dict:
    """분석 결과를 MinIO에 업로드하고 요약을 리턴."""
    df = pd.read_parquet(input_path)
    summary = df.describe().to_dict()

    # 결과 파일 작성
    result_path = "/tmp/analysis_result.csv"
    df.describe().to_csv(result_path)

    # MinIO 업로드
    hook = S3Hook(aws_conn_id=conn_id)
    hook.load_file(
        filename=result_path,
        key=output_key,
        bucket_name=bucket,
        replace=True,
    )

    print(f"Uploaded result to s3://{bucket}/{output_key}")
    return {"rows": len(df), "result_key": output_key}
```

#### `minio_analysis_pipeline.py` (DAG)
```python
from datetime import datetime, timedelta
from airflow.decorators import dag, task
from tasks.download import download_from_minio
from tasks.preprocess import preprocess
from tasks.analyze import analyze_and_upload


default_args = {
    "owner": "daeyoung",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "depends_on_past": False,  # 이전 날짜 실행과 독립
}


@dag(
    dag_id="minio_analysis_pipeline",
    description="MinIO에서 데이터 다운로드 → 전처리 → 분석 → 결과 업로드",
    start_date=datetime(2026, 5, 1),
    schedule="0 2 * * *",        # 매일 02:00
    catchup=False,                # 과거 실행 분 자동 채우지 않음
    max_active_runs=1,            # 동시에 1개 DAG run만 (데이터 꼬임 방지)
    default_args=default_args,
    tags=["minio", "etl"],
)
def minio_analysis_pipeline():

    # ds = execution_date를 'YYYY-MM-DD' 형식으로 받는 Airflow 매크로
    @task
    def download(ds=None) -> list[str]:
        return download_from_minio(
            bucket="raw-data",
            prefix=f"daily/{ds}/",
            local_dir=f"/tmp/raw/{ds}",
        )

    @task
    def clean(file_paths: list[str], ds=None) -> str:
        return preprocess(
            file_paths=file_paths,
            output_path=f"/tmp/clean/{ds}/cleaned.parquet",
        )

    @task
    def analyze(input_path: str, ds=None) -> dict:
        return analyze_and_upload(
            input_path=input_path,
            bucket="processed-data",
            output_key=f"reports/{ds}/summary.csv",
        )

    # 의존성: download → clean → analyze (TaskFlow가 자동 연결)
    files = download()
    cleaned = clean(files)
    analyze(cleaned)


minio_analysis_pipeline()
```

#### 무엇이 "데이터 꼬임"을 막는가?

| 메커니즘 | 효과 |
|----------|------|
| `trigger_rule="all_success"` (기본값) | upstream이 모두 성공해야만 다음 Task 실행 |
| `max_active_runs=1` | 같은 DAG의 두 run이 겹쳐서 같은 파일을 동시에 건드리는 상황 차단 |
| `retries=2` + `retry_delay` | 일시적 네트워크 오류 자동 복구 |
| `depends_on_past=True` (옵션) | 어제 run이 성공해야 오늘 run 실행 — 누적 처리에 유용 |
| `catchup=False` | 배포 시 과거 누락분이 한꺼번에 돌아 시스템 부하 일으키는 것 방지 |
| Task 단위의 멱등성(idempotency) | 같은 Task를 재실행해도 결과가 동일하도록 코드 설계 |

#### 멱등성 팁
- 출력 파일 경로에 `{{ ds }}` 같은 실행 날짜 포함 → 같은 날짜 재실행 시 덮어쓰기
- MinIO 업로드 시 `replace=True`
- 입력 파일을 삭제하지 않고 Task 시작 시 출력 디렉토리만 비움

---

### 4. DAG 배포 & 디버깅

#### 배포
운영팀이 정한 방식대로 — 보통은 Git push 후 사내 GitOps가 `dags/` 폴더에 동기화한다.

```bash
git add dags/minio_analysis_pipeline.py dags/tasks/
git commit -m "feat: add MinIO analysis pipeline DAG"
git push
```

배포 후 Web UI에서 DAG가 보이는지 확인 (수 초~수 분 소요).

#### 로컬에서 빠르게 검증
DAG 구문 오류는 운영 환경에 올리기 전에 잡는 게 좋다:

```bash
# 1. import 에러 검사
python dags/minio_analysis_pipeline.py

# 2. DAG 파싱 검사
airflow dags list-import-errors

# 3. 단일 Task만 격리 실행 (Connection 등 필요)
airflow tasks test minio_analysis_pipeline download 2026-05-02
```

#### Web UI에서 자주 보는 곳
- **Graph View**: 의존성 흐름 시각화
- **Grid View**: 날짜별 Task 실행 상태 매트릭스
- **Logs**: 각 Task Instance 로그 (실패 원인 1순위 확인 위치)
- **XCom 탭**: Task 간 주고받은 데이터 확인

---

### 5. 자주 만나는 문제 (Troubleshooting)

| 증상 | 원인 / 해결 |
|------|-------------|
| `ModuleNotFoundError: minio` | Worker에 패키지 미설치. 방법 A/B/C/D 중 선택 |
| `botocore endpoint url` 오류 | Connection의 Extra에 `endpoint_url` 누락 |
| MinIO SSL 오류 | Extra에 `"verify": false` 추가 (개발용), 운영은 사내 CA 설치 |
| DAG가 UI에 안 보임 | `dags_folder` 경로 확인, import 에러 확인 (`airflow dags list-import-errors`) |
| Task가 큐잉만 되고 실행 안 됨 | Worker가 죽었거나 슬롯 부족. Scheduler/Worker 로그 확인 |
| XCom이 너무 커서 실패 | XCom은 작은 메타데이터용. 큰 데이터는 MinIO 경로만 넘기고 데이터 자체는 파일로 |
| 같은 DAG 두 번 도는 문제 | `max_active_runs=1`, 또는 파일 단위 락(예: MinIO에 lock 객체) |

---

## 학습 ↔ 실무 연결

- **Recipe Setup 자동화**: 매일 새 레시피 파일이 MinIO에 떨어지면, 본 패턴 그대로 다운로드 → 검증 → DB 적재 파이프라인으로 확장 가능
- **SKEWNONO**: 모델 학습용 데이터 수집(MinIO) → 전처리 → 임베딩 → Milvus/OpenSearch 인덱싱을 단일 DAG로 묶어 일관된 재실행성 확보
- **LangGraph 연동**: 분석 단계에서 LangGraph 워크플로를 호출하면 RAG 파이프라인의 일일 인덱싱 잡으로도 활용 가능 → [LangGraph 기초](../rag/langgraph/langgraph-basics.md)

---

## 참고 자료 (References)

- [Apache Airflow 공식 문서](https://airflow.apache.org/docs/)
- [TaskFlow API 가이드](https://airflow.apache.org/docs/apache-airflow/stable/tutorial/taskflow.html)
- [PythonVirtualenvOperator](https://airflow.apache.org/docs/apache-airflow/stable/_api/airflow/operators/python/index.html#airflow.operators.python.PythonVirtualenvOperator)
- [Amazon Provider (S3Hook)](https://airflow.apache.org/docs/apache-airflow-providers-amazon/stable/connections/aws.html)
- [MinIO Python SDK](https://min.io/docs/minio/linux/developers/python/API.html)
- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)

## 관련 문서
- [AI/DT 학습 노트](../README.md)
- [LangGraph 기초](../rag/langgraph/langgraph-basics.md)
