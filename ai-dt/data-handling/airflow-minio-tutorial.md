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

#### 로컬(Windows)에서 검증하기

> Airflow는 `pwd`, `grp`, `fcntl` 같은 POSIX 전용 모듈을 사용하므로 **Windows Python에 직접 설치되지 않는다**. 따라서 로컬 검증은 (1) Airflow 없이 파싱만, (2) WSL2/Docker로 Linux 환경을 빌려서 진행한다.

##### 레벨 1 — Airflow 없이 파싱·import만 검사 (가장 가벼움)
운영 환경에 올리기 전 90%의 실수(오타, import 에러, 모듈 경로)는 이 단계에서 잡힌다. PowerShell/cmd에서:

```powershell
# 가상환경 만들고 최소 의존성만
python -m venv .venv
.venv\Scripts\activate
pip install "apache-airflow==2.9.*" --constraint https://raw.githubusercontent.com/apache/airflow/constraints-2.9.3/constraints-3.11.txt
# (Windows에선 일부 패키지 install이 실패할 수 있음 — 그땐 레벨 2로)

# DAG import만 시도 (스케줄러 없이도 동작)
python dags/minio_analysis_pipeline.py
```

import만이라도 통과하면 DAG 구조 자체는 유효하다. 더 가볍게는 **Airflow 설치 없이** Operator/decorator를 mock해서 import만 검증하는 방법도 있다.

##### 레벨 2 — WSL2 + `airflow standalone` (가벼우면서 UI까지)
PowerShell에서 한 번만:

```powershell
wsl --install -d Ubuntu-22.04
```

WSL Ubuntu 안에서:

```bash
sudo apt update && sudo apt install -y python3-venv
python3 -m venv ~/airflow-venv && source ~/airflow-venv/bin/activate

export AIRFLOW_HOME=~/airflow
pip install "apache-airflow==2.9.3" \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.3/constraints-3.10.txt"

# DAG 디렉토리를 Windows 쪽 repo로 심볼릭 링크
ln -s /mnt/c/Code/pm_notes/dags ~/airflow/dags

# scheduler + webserver를 한 프로세스로 (개발 전용)
airflow standalone
```

`http://localhost:8080`에서 Web UI 접속, 초기 admin 비밀번호는 콘솔에 출력된다. DAG가 Graph View에 뜨는지, Task 의존성이 의도대로 그려지는지 확인.

##### 레벨 3 — Docker Compose (운영과 가장 유사)
Docker Desktop이 깔려 있으면 공식 compose 파일로 한 방에 띄울 수 있다.

```powershell
mkdir airflow-local; cd airflow-local
curl -O https://airflow.apache.org/docs/apache-airflow/2.9.3/docker-compose.yaml
mkdir dags, logs, plugins, config
"AIRFLOW_UID=50000" | Out-File -Encoding ascii .env

# 초기화 1회
docker compose up airflow-init

# 기동
docker compose up -d
```

repo의 `dags/` 폴더를 `airflow-local/dags/`에 복사하거나 compose 파일의 `volumes:` 경로를 repo 경로로 바꾸면 된다. UI는 `http://localhost:8080` (계정: `airflow`/`airflow`).

> **주의**: compose 파일의 볼륨 매핑에 Windows 절대 경로를 쓸 땐 `C:\Code\pm_notes\dags`처럼 Docker Desktop이 인식 가능한 형태로. WSL2 백엔드면 `/c/Code/pm_notes/dags` 형식도 가능.

##### 레벨 4 — Astro CLI (개인 개발 가장 편함)
Astronomer가 만든 dev 도구. Docker Desktop 위에서 한 줄로 동작.

```powershell
winget install -e --id Astronomer.Astro
astro dev init       # Airflow 프로젝트 스캐폴딩 생성
# dags/, requirements.txt, packages.txt 등이 만들어짐
astro dev start      # 컨테이너 4개 (webserver/scheduler/triggerer/postgres) 기동
```

#### 단일 Task만 실행해서 디버깅
WSL/Docker 환경 안에서:

```bash
# 특정 Task 1개만 실행 (실제 Connection 사용)
airflow tasks test minio_analysis_pipeline download 2026-05-02

# DAG 전체를 scheduler 없이 한 번 돌려보기
airflow dags test minio_analysis_pipeline 2026-05-02

# 파싱 에러 일괄 확인
airflow dags list-import-errors
```

`tasks test`는 metadata DB에 결과를 기록하지 않으므로 부작용 없이 반복 실행할 수 있다 — 디버깅 1순위.

#### Windows ↔ Linux 환경 차이에서 자주 깨지는 것

| 항목 | Windows 로컬 | Linux 운영 | 대처 |
|------|-------------|-----------|------|
| 임시 디렉토리 | `C:\Users\..\AppData\Local\Temp` | `/tmp` | 코드에서 `tempfile.gettempdir()` 사용, DAG 파라미터로 경로 주입 |
| 경로 구분자 | `\` | `/` | `pathlib.Path`만 사용, 문자열 결합 금지 |
| 줄바꿈 | CRLF | LF | repo 루트에 `.gitattributes`로 `*.py text eol=lf` 강제 |
| 파일 권한 | 의미 없음 | `chmod +x` 필요 | `BashOperator`보다 `PythonOperator` 우선 |
| 셸 명령 | PowerShell/cmd | bash | `BashOperator`의 `bash_command`는 항상 `set -euo pipefail`로 시작 |
| 한글 인코딩 | cp949 기본 | UTF-8 | 파일 read/write 시 `encoding="utf-8"` 명시 |
| MinIO 엔드포인트 | `localhost:9000` (Docker) | `minio.your-company.com:9000` | Connection으로 추상화, 코드에 하드코딩 금지 |

> 이 차이들 때문에 "로컬에서 됐는데 운영에서 깨지는" 일이 발생한다. 가능하면 **레벨 3(Docker)** 이상으로 운영과 같은 OS·같은 Python 버전에서 한 번 더 검증하는 게 안전하다.

#### Web UI에서 자주 보는 곳
- **Graph View**: 의존성 흐름 시각화
- **Grid View**: 날짜별 Task 실행 상태 매트릭스
- **Logs**: 각 Task Instance 로그 (실패 원인 1순위 확인 위치)
- **XCom 탭**: Task 간 주고받은 데이터 확인

---

### 5. Logging & 관찰성 (Observability)

> 매니지드 Airflow에서 우리가 할 수 있는 건 **DAG 코드 안에서 로그를 잘 남기는 것**과 **UI에서 추적 가능한 형태로 신호를 노출하는 것** 두 가지다.

#### Airflow가 로그를 잡는 원리
- 각 Task Instance가 시작될 때 Airflow는 `airflow.task` 로거에 **파일 핸들러**를 붙인다 (재시도마다 별도 파일)
- Python 표준 `logging`으로 찍은 로그, `print()`, 그리고 unhandled exception은 모두 이 핸들러로 흘러간다
- UI의 Grid View → Task 클릭 → **Log** 탭이 이 파일을 그대로 렌더링한다 (재시도 별로 `Try 1, Try 2` 탭 분리)

#### 권장 패턴: 표준 logging 사용

```python
# tasks/download.py
import logging
from pathlib import Path
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

log = logging.getLogger(__name__)  # Airflow가 자동으로 task logger에 연결


def download_from_minio(bucket: str, prefix: str, local_dir: str, conn_id: str = "minio_default") -> list[str]:
    log.info("Start download bucket=%s prefix=%s", bucket, prefix)

    hook = S3Hook(aws_conn_id=conn_id)
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    keys = hook.list_keys(bucket_name=bucket, prefix=prefix) or []
    log.info("Discovered %d keys under %s", len(keys), prefix)

    if not keys:
        log.error("No objects under s3://%s/%s — upstream system may be late", bucket, prefix)
        raise ValueError(f"No objects under s3://{bucket}/{prefix}")

    downloaded = []
    for i, key in enumerate(keys, 1):
        if key.endswith("/"):
            continue
        local_path = str(Path(local_dir) / Path(key).name)
        try:
            hook.get_key(key, bucket_name=bucket).download_file(local_path)
            downloaded.append(local_path)
        except Exception:
            log.exception("Failed to download %s", key)  # traceback 자동 포함
            raise

        # 100개마다 한 번씩 진행상황 (수만 개일 때 로그 폭주 방지)
        if i % 100 == 0:
            log.info("Progress: %d/%d", i, len(keys))

    log.info("Done. Downloaded %d files to %s", len(downloaded), local_dir)
    return downloaded
```

핵심 규칙:
- `print()` 대신 **`logging`**: timestamp/level/모듈명이 자동으로 붙어 grep·필터링이 쉬움
- 에러는 `log.exception(...)`: 한 줄로 메시지 + 전체 traceback 동시 기록
- 반복문에서는 **batch로 요약** 로그 (수만 줄 로그는 UI 렌더가 느려지고 핵심을 가림)
- `%s`, `%d` 포맷 사용 (logging 모듈이 lazy evaluation — 로그 비활성화 시 문자열 연산 스킵)

#### "끝났다" vs "제대로 됐다"를 구분하기

UI의 초록불은 **exit code 0**만 보장한다. 의도한 결과가 나왔는지 확인하려면 Task 끝부분에 검증을 둔다:

```python
def preprocess(file_paths: list[str], output_path: str) -> str:
    # ... 처리 ...
    cleaned.to_parquet(output_path, index=False)

    # 검증: 핵심 가정이 맞는지
    if len(cleaned) == 0:
        raise ValueError("Preprocessed result is empty — upstream data quality issue?")

    expected_cols = {"timestamp", "lot_id", "value"}
    missing = expected_cols - set(cleaned.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    log.info("Validated: %d rows, %d cols", len(cleaned), len(cleaned.columns))
    return output_path
```

빨리 실패하는 게 늦게 이상한 결과를 만드는 것보다 훨씬 디버깅하기 쉽다.

#### XCom으로 핵심 지표 노출 (UI에서 한눈에)

함수 return 값이 자동으로 XCom에 저장돼 UI의 **XCom 탭**에서 보인다. 로그를 펴보지 않고도 "어제 run에서 몇 건 처리됐지?"를 즉답할 수 있다.

```python
@task
def analyze(input_path: str) -> dict:
    df = pd.read_parquet(input_path)
    summary = {
        "rows": len(df),
        "null_rate": float(df.isna().mean().mean()),
        "duration_sec": elapsed,
        "output_key": output_key,
    }
    log.info("Analysis summary: %s", summary)
    return summary  # → XCom에 저장됨
```

> 큰 데이터(수백 KB+)는 XCom에 넣지 말 것 — metadata DB에 쌓여 성능 저하. **요약 dict, 경로, 카운트만**.

#### Task Context의 logger 직접 쓰기

`@task` 함수 안에서 Airflow context에 접근하면 task logger를 명시적으로 쓸 수 있다:

```python
@task
def download(**context):
    log = context["ti"].log  # 또는 logging.getLogger("airflow.task")
    log.info("execution_date=%s try_number=%d", context["ds"], context["ti"].try_number)
```

대부분은 `logging.getLogger(__name__)`로 충분하지만, **재시도 횟수**나 **execution_date**처럼 task 메타데이터를 로그에 함께 찍고 싶을 때 유용하다.

#### 매니지드 환경에서 로그가 안 보일 때

| 증상 | 확인 |
|------|------|
| Log 탭에 `*** Falling back to local log` | 원격 스토리지(S3/MinIO) ship 실패. 운영팀에 권한·연결 문의 |
| 며칠 지난 run의 로그가 비어 있음 | UI 캐시 만료 — `remote_base_log_folder` 위치를 운영팀에 확인해 직접 다운로드 |
| Task가 **success**인데 코드가 안 돈 것 같음 | `print` 만 쓰고 있을 가능성. 표준 `logging` + 시작/종료 sentinel 로그 추가 |
| 로그가 너무 길어 UI 렌더가 느림 | 반복문 batch 요약, 큰 dict는 `repr` 대신 핵심 키만 |
| 운영팀이 로그 보존 N일이라고 함 | 중요 run의 핵심 메트릭은 `analyze` Task에서 **MinIO에 별도 JSON으로 저장**해두기 |

#### 운영팀에 한 번만 확인해두면 좋은 것
- `[logging] remote_logging`, `remote_base_log_folder`, `remote_log_conn_id` 설정값
- 로그 보존 기간 (예: 30일 후 삭제 정책)
- 로그를 ELK/Loki/Splunk 같은 사내 중앙 시스템에도 보내는지 — 그렇다면 검색이 훨씬 빨라짐
- `EXTRA_LOGGING_LEVEL`이나 DAG 단위 로그 레벨 override가 가능한지

#### 보강 옵션: OpenSearch에 로그/이벤트 직접 보내기

매니지드 Airflow의 로그 보존이 짧거나, DAG 간 비교·집계가 필요할 때 OpenSearch를 보조 저장소로 쓰는 게 유효하다. **단, 방식 선택이 결과를 가른다**.

##### 0순위: 인프라 레벨 ship이 이미 있는지 확인
사내에 OpenSearch가 있다면 운영팀이 이미 **Filebeat/Fluentd/Fluent Bit**로 Airflow의 task 로그를 인덱싱하고 있을 가능성이 매우 높다. 그렇다면:
- 우리는 DAG 코드를 바꿀 필요 없음
- 인덱스 패턴(예: `airflow-task-logs-*`)과 검색 권한만 받으면 끝
- Kibana/OpenSearch Dashboards에서 `dag_id:"minio_analysis_pipeline"` 같은 쿼리로 즉시 검색 가능

운영팀에 "Airflow 로그가 OpenSearch에 들어오고 있나요? 인덱스 이름 알려주세요" 한 줄 질문이 1순위다.

##### 그게 없을 때: 두 가지 패턴

| 방식 | 무엇을 보내나 | 장점 | 단점 |
|------|---------------|------|------|
| **A. Logging Handler로 모든 로그 라인** | 모든 `log.info/error` | Kibana에서 raw 로그 그대로 검색 | 매 라인 네트워크 호출 → 느림. 버퍼링 필수. 네트워크 blip 시 로그 손실 위험 |
| **B. 구조화 이벤트만 (권장)** | start/end/metric/error 같은 핵심 시점만 JSON 1건 인덱싱 | 가볍고 쿼리 가독성 좋음. raw 로그는 어차피 Airflow에 있음. 비용 적음 | "특정 줄에서 뭐 찍혔는지"는 여전히 Airflow UI를 봐야 함 |

대부분의 경우 **B**가 정답. raw 로그까지 OpenSearch에 다 넣고 싶으면 A를 쓰되 반드시 **버퍼링 + bulk flush** 형태로.

##### 패턴 B: 구조화 이벤트 인덱싱

```python
# tasks/observability.py
import logging
import os
import socket
import time
from contextlib import contextmanager
from datetime import datetime, timezone

from opensearchpy import OpenSearch
from airflow.hooks.base import BaseHook

log = logging.getLogger(__name__)


def _client() -> OpenSearch:
    """Connection 'opensearch_default'에서 host/auth 읽기."""
    conn = BaseHook.get_connection("opensearch_default")
    return OpenSearch(
        hosts=[{"host": conn.host, "port": conn.port or 9200}],
        http_auth=(conn.login, conn.password) if conn.login else None,
        use_ssl=conn.schema == "https",
        verify_certs=False,  # 사내 CA는 별도 설정
        timeout=10,
    )


def emit_event(event_type: str, payload: dict, context: dict | None = None) -> None:
    """OpenSearch에 1개 이벤트 인덱싱. 실패해도 Task는 망치지 않는다."""
    doc = {
        "@timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,            # "task_start" / "task_end" / "metric" / "error"
        "host": socket.gethostname(),
        "dag_id": (context or {}).get("dag", {}).dag_id if context else None,
        "task_id": (context or {}).get("task", {}).task_id if context else None,
        "run_id": (context or {}).get("run_id") if context else None,
        "execution_date": (context or {}).get("ds") if context else None,
        "try_number": (context or {}).get("ti", {}).try_number if context else None,
        **payload,
    }
    index = f"airflow-events-{datetime.now(timezone.utc):%Y.%m}"
    try:
        _client().index(index=index, body=doc)
    except Exception:
        # 관찰성 시스템이 본 작업을 깨면 안 된다 — 경고만 남기고 통과
        log.warning("Failed to emit event to OpenSearch", exc_info=True)


@contextmanager
def observed_task(name: str, **context):
    """Task를 감싸서 시작·종료·실패·소요시간을 자동 인덱싱."""
    start = time.perf_counter()
    emit_event("task_start", {"name": name}, context)
    try:
        yield
    except Exception as e:
        emit_event(
            "task_error",
            {"name": name, "error_type": type(e).__name__, "error_msg": str(e)},
            context,
        )
        raise
    finally:
        elapsed = time.perf_counter() - start
        emit_event("task_end", {"name": name, "duration_sec": round(elapsed, 3)}, context)
```

DAG에서 사용:

```python
from tasks.observability import observed_task, emit_event

@task
def analyze(input_path: str, **context) -> dict:
    with observed_task("analyze", **context):
        df = pd.read_parquet(input_path)

        # 핵심 메트릭은 별도 이벤트로
        emit_event("metric", {
            "metric": "input_rows",
            "value": len(df),
            "input_path": input_path,
        }, context)

        result = df.describe()
        # ... 업로드 ...
        return {"rows": len(df)}
```

이렇게 하면 OpenSearch에서:
- `event_type:task_error AND dag_id:"minio_analysis_pipeline"` → 최근 실패 모음
- `event_type:metric AND metric:"input_rows"` 시계열 → 일별 입력 행 수 추이
- `event_type:task_end` aggregation → 평균 처리 시간 / 95th percentile

##### 패턴 A를 굳이 쓴다면: 버퍼링 핸들러

```python
import atexit, logging, queue
from logging.handlers import QueueHandler, QueueListener
from opensearchpy import helpers

class OpenSearchBulkHandler(logging.Handler):
    def __init__(self, client, index_pattern, batch_size=100, flush_interval=5):
        super().__init__()
        self.buffer = []
        self.client = client
        self.index_pattern = index_pattern
        self.batch_size = batch_size

    def emit(self, record):
        self.buffer.append({
            "_index": datetime.utcnow().strftime(self.index_pattern),
            "_source": {
                "@timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                # context는 LogRecord.extra로 주입
            },
        })
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        try:
            helpers.bulk(self.client, self.buffer)
        except Exception:
            pass  # 로그가 본 작업을 깨면 안 됨
        finally:
            self.buffer.clear()

# Task 시작 시 attach, 종료 시 flush
handler = OpenSearchBulkHandler(_client(), "airflow-logs-%Y.%m.%d")
logging.getLogger().addHandler(handler)
atexit.register(handler.flush)
```

> 진짜 production이라면 `QueueHandler`/`QueueListener`로 별도 스레드에서 flush하도록 분리하는 게 안전 — 메인 스레드 block을 막을 수 있다.

##### 주의사항
- **opensearch-py 패키지 설치**: 폐쇄망이면 사내 PyPI 미러 사용 (앞서 본 방법 A/B/C/D)
- **인덱스 라이프사이클(ISM) 정책**: 인덱스가 무한정 커지지 않도록 운영팀과 협의해 ILM/ISM 설정 (예: 30일 후 close, 90일 후 delete)
- **PII/기밀**: Task 로그에 장비ID·사번·민감 데이터가 섞이지 않게 인덱싱 전 필터
- **인증 정보**: OpenSearch user/password는 반드시 Airflow Connection으로, 코드에 박지 말 것
- **순환 의존**: OpenSearch가 죽으면 우리 DAG도 같이 죽는 구조를 만들면 안 된다 — `try/except`로 관찰성 코드는 본 작업과 격리

---

### 6. 자주 만나는 문제 (Troubleshooting)

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
