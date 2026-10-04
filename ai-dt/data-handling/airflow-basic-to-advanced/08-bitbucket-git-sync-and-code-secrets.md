---
tags: [airflow, bitbucket, git-sync, secrets, company-env]
level: intermediate-advanced
last_updated: 2026-05-02
reviewed_on: 2026-10-04
review_status: partial
document_type: historical_scenario_with_review
---

# 08. Bitbucket Git Sync와 코드 기반 Secret 운영

> [!warning] 현재 사용 지침과 원문 시나리오 — 2026-10-04
> 아래 “2026-05-02 원문 시나리오”는 당시 문서가 가정한 권한 제한·코드 secret 운영안을 보존한 것이다. 실제 회사 정책·허용 방식은 검증되지 않았다. 원문의 “실제 secret은 private repository에 넣는다”와 `secure=False`를 현재 권장 방식으로 따르지 않는다. private Git도 비밀의 안전한 공급을 자동 보장하지 않는다. Claude/사내 정책 협의가 필요하나 현재 Herdr pane_not_found여서 미결정으로 남긴다.

## 현재 확인된 기능과 적용 조건

Airflow 2.10.5 Connection은 metadata DB 외에 환경변수와 외부 secret backend에서도 제공된다. env/backend 값은 UI의 Connection 목록에 나타나지 않을 수 있다. UI 관리 권한 제한과 Task 실행 시 credential 사용 가능 여부는 다른 문제다. 운영팀이 제공한 Connection ID/주입 방식·Task 접근 권한·유효 기간을 확인한다. [Connection 공식 설명](https://airflow.apache.org/docs/apache-airflow/2.10.5/howto/connection.html), [secret backend](https://airflow.apache.org/docs/apache-airflow/2.10.5/security/secrets/secrets-backend/index.html).

Git Sync는 DAG source를 배포하는 방식이다. secret 주입이나 requirements 설치를 그 자체로 보장하지 않는다. 운영팀 제공 환경변수/backend 또는 승인된 외부 파일을 사용하도록 구성할 수 있지만 이 회사에서 사용 가능하다는 사실은 미확인이다. [공식 Helm DAG 배포 예시](https://airflow.apache.org/docs/helm-chart/stable/manage-dag-files.html)는 제품 배포 옵션의 근거이며 사내 설치 증거는 아니다.

### 비밀 값 대신 실행 환경에서 읽는 최소 예시

아래는 **환경변수 주입이 승인돼 사용 가능한 경우**의 loader다. 실제 key를 source·문서에 적거나 출력하지 않는다. 누락은 빈 문자열/기본 key로 바꾸지 않고 실패한다. TLS를 사용하는 MinIO SDK client에 전달할 설정이며 실제 endpoint·CA·계정·권한은 별도 확인해야 한다.

```python
from collections.abc import Mapping
import os


def load_minio_settings(env: Mapping[str, str]) -> dict[str, str | bool]:
    names = {
        "endpoint": "COMPANY_MINIO_ENDPOINT",
        "access_key": "COMPANY_MINIO_ACCESS_KEY",
        "secret_key": "COMPANY_MINIO_SECRET_KEY",
        "bucket": "COMPANY_MINIO_BUCKET",
    }
    missing = [name for name in names.values() if not env.get(name)]
    if missing:
        raise ValueError("필수 MinIO 환경변수 누락: " + ", ".join(missing))
    return {**{key: env[name] for key, name in names.items()}, "secure": True}


# Task 내부에서 운영팀이 주입한 환경을 읽는다. 반환값은 print/log/XCom하지 않는다.
# settings = load_minio_settings(os.environ)
# Minio(endpoint=settings["endpoint"], access_key=settings["access_key"],
#       secret_key=settings["secret_key"], secure=settings["secure"])
```

[MinIO Python API](https://docs.min.io/aistor/developers/sdk/python/api/)의 `secure=True`는 HTTPS 사용이다. CA 신뢰·인증서 검증·권한/소유자 정책은 별도 조건이며 TLS를 끄는 것을 사내 기본으로 가정하지 않는다. 확인한 페이지는 AIStor 문서로 redirect됐고 기존 OSS/사내 버전은 미확인이다.

### 원래 예제의 실행 한계

원문08에는 날짜 전용 `main(run_date)`와 시간 구간용 `main(start_ts,end_ts)`가 같은 가상 download.py 이름으로 나타난다. 서로 다른 예제이므로 동시에 덮어써 실행하지 않는다. 단순 Bash 예제는 CLI entrypoint가 없는 함수만의 파일과 그대로 조합하면 일을 하지 않을 수 있다. 시간 문자열 slice 기반 경로도 timezone/DST/수동 run을 식별하는 완성형이 아니다. 주석 처리된 `put_object`는 실제 업로드가 아니다. 고유 예제를 보존하되 이 장을 완성된 운영 서비스나 secret 정책으로 사용하지 않는다. 실제 구현 계약 통합은 Claude 협의 대기다.



### 로컬 검증 결과 — 2026-10-04

역사 시나리오의 DAG 2개는 정의와 의존성 파싱만 통과했다. 새 환경변수 loader는 필수 값 4가지가 누락되면 실패했고, 오류 메시지에 비밀 값을 포함하지 않았다.

검증 환경은 저장소 밖의 임시 Python 3.12.12 환경이다. 문서 예제를 임시 파일로 추출하고 로컬 helper를 사용했다. 실제 scheduler와 worker, 업무 파일, FTP·MinIO·API, 인증과 TLS 연결은 실행하지 않았다. DAG 발견은 업무 처리의 성공을 보장하지 않는다.

## 2026-05-02 원문 시나리오 — 역사 보존, 현재 사실 아님

## 이 장의 전제

이 장은 다음 회사 환경을 기준으로 한다.

- Airflow 서버는 회사가 관리한다.
- 사용자는 Airflow Connection/Variable에 접근할 수 없다.
- DAG는 직접 서버에 업로드하는 것이 아니라 Bitbucket repository에 push한다.
- Airflow는 Git Sync 방식으로 지정된 Bitbucket repository와 branch를 주기적으로 읽는다.
- DB password, API key, MinIO/S3 access key 같은 secret은 Airflow Connection이 아니라 코드 또는 코드와 함께 배포되는 설정 파일에 있어야 한다.
- 여러 Python 파일을 매시간 실행한다. 단순 실행 추적만 필요하면 인자 없이 실행하고, 특정 데이터 구간을 처리해야 하면 시작/종료 시각을 인자로 넘긴다.

이 조건에서는 Airflow의 이상적인 secret 관리 방식과 다르게 운영해야 한다. 핵심은 **Bitbucket repository 자체를 secret 저장소처럼 취급**하는 것이다.

## 가장 중요한 주의

코드에 secret을 넣는 방식은 위험하다. 그래도 회사 정책과 권한 구조상 이 방식만 가능하다면 아래 통제가 최소한 필요하다.

- private Bitbucket repository만 사용한다.
- repository 접근 권한을 실제 운영자와 개발자 최소 인원으로 제한한다.
- branch permission을 걸어 force push, 무심사 merge를 막는다.
- secret이 들어간 repository를 GitHub, 개인 fork, 로컬 공개 백업으로 미러링하지 않는다.
- screenshot, 로그, 에러 메시지에 secret 값이 노출되지 않게 한다.
- secret이 한 번 commit되면 삭제해도 Git history에 남는다고 가정한다.
- secret 변경 시 key rotation 절차를 갖춘다.
- 가능하면 read-only 계정, 특정 bucket/table만 접근 가능한 계정처럼 권한을 줄인다.

문서 예제에는 실제 secret을 쓰지 않는다. 실제 값은 회사 내부 Bitbucket repository에만 넣는다.

## 권장 repository 구조

Airflow가 Bitbucket repository의 `dags/` 폴더를 DAG root로 인식한다고 가정한 구조다.

```text
airflow-dags-repo/
├── dags/
│   ├── hourly_sales_pipeline.py
│   └── company_sales/
│       ├── __init__.py
│       ├── config.py
│       ├── secrets.py
│       ├── clients.py
│       └── jobs/
│           ├── __init__.py
│           ├── download.py
│           ├── preprocess.py
│           └── report.py
├── tests/
│   └── test_dag_import.py
├── requirements-app.txt
└── README.md
```

중요한 점:

- DAG 파일은 `dags/` 바로 아래에 둔다.
- 공통 Python module은 `dags/company_sales/`처럼 DAG root 아래에 둔다.
- `src/` 구조는 Airflow의 `PYTHONPATH` 설정이 맞지 않으면 import 실패할 수 있다.
- secret은 여러 파일에 흩뿌리지 말고 `secrets.py` 같은 한 파일에 모은다.
- `requirements-app.txt`는 필요한 패키지를 문서화하는 용도다. Git Sync가 패키지를 자동 설치한다고 가정하면 안 된다.

Airflow가 repository root 전체를 DAG folder로 읽는지, `dags/` 하위만 읽는지는 회사 설정에 따라 다르다. import error가 나면 이 경로부터 확인한다.

## Secret 파일 예시

`secrets.py`에는 실제 운영 secret이 들어간다. 이 파일은 private Bitbucket repository에만 존재해야 한다.

```python
# dags/company_sales/secrets.py

MINIO = {
    "endpoint": "minio.company.internal:9000",
    "access_key": "REPLACE_WITH_REAL_ACCESS_KEY",
    "secret_key": "REPLACE_WITH_REAL_SECRET_KEY",
    "secure": False,
    "bucket": "sales-data",
}

WAREHOUSE_DB = {
    "host": "warehouse.company.internal",
    "port": 5432,
    "database": "analytics",
    "user": "sales_batch_user",
    "password": "REPLACE_WITH_REAL_PASSWORD",
}

REPORT_API = {
    "base_url": "https://report.company.internal",
    "token": "REPLACE_WITH_REAL_TOKEN",
}
```

권장 사항:

- secret 변수 이름은 명확하게 쓴다.
- secret 값을 print하지 않는다.
- DAG 파일 최상단에서 secret을 읽어 로그로 남기지 않는다.
- 가능하면 Task 실행 함수 내부에서 import한다.

## 일반 설정과 Secret 분리

secret이 아닌 설정은 `config.py`에 둔다.

```python
# dags/company_sales/config.py

JOB_NAME = "daily_sales"
RAW_PREFIX = "raw/sales"
CLEAN_PREFIX = "clean/sales"
REPORT_PREFIX = "report/sales"

DEFAULT_RETRIES = 2
```

secret과 일반 설정을 분리하면 변경 영향이 줄고 review가 쉬워진다.

## Client 생성 함수

secret을 직접 여러 job 파일에서 import하지 말고 client 생성 함수를 한 곳에 둔다.

```python
# dags/company_sales/clients.py

def make_minio_client():
    from minio import Minio

    from company_sales.secrets import MINIO

    return Minio(
        endpoint=MINIO["endpoint"],
        access_key=MINIO["access_key"],
        secret_key=MINIO["secret_key"],
        secure=MINIO["secure"],
    )


def get_minio_bucket() -> str:
    from company_sales.secrets import MINIO

    return MINIO["bucket"]
```

이렇게 하면 job 코드에서는 secret 구조를 몰라도 된다.

```python
# dags/company_sales/jobs/download.py

def main(run_date: str) -> str:
    from company_sales.clients import get_minio_bucket, make_minio_client

    client = make_minio_client()
    bucket = get_minio_bucket()
    output_key = f"raw/sales/dt={run_date}/data.csv"

    print(f"write minio://{bucket}/{output_key}")
    # client.put_object(...)
    return f"minio://{bucket}/{output_key}"
```

로그에는 bucket/key 정도만 남기고 access key, secret key는 남기지 않는다.

## hourly DAG 예시: 성공/실패 추적만 필요한 경우

스크립트가 내부에서 처리 대상을 스스로 결정하고, Airflow에서는 매시간 실행 여부와 로그만 확인하면 된다면 인자 없이 실행한다.

```python
# dags/hourly_simple_pipeline.py
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator


BASE_DIR = "/opt/airflow/dags/company_sales/jobs"


with DAG(
    dag_id="hourly_simple_pipeline",
    start_date=datetime(2026, 5, 1),
    schedule="@hourly",
    catchup=False,
    max_active_runs=1,
    tags=["sales", "git-sync"],
) as dag:
    download = BashOperator(
        task_id="download",
        bash_command=f"set -euo pipefail; python {BASE_DIR}/download.py",
        execution_timeout=timedelta(minutes=30),
    )

    preprocess = BashOperator(
        task_id="preprocess",
        bash_command=f"set -euo pipefail; python {BASE_DIR}/preprocess.py",
        execution_timeout=timedelta(hours=1),
    )

    report = BashOperator(
        task_id="report",
        bash_command=f"set -euo pipefail; python {BASE_DIR}/report.py",
        execution_timeout=timedelta(minutes=20),
    )

    download >> preprocess >> report
```

이 방식도 Airflow UI에서 각 Task의 성공/실패, 로그, 재시도 상태를 확인할 수 있다.

## hourly DAG 예시: 데이터 구간을 명시해야 하는 경우

```python
# dags/hourly_sales_pipeline.py
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from company_sales.config import DEFAULT_RETRIES


default_args = {
    "owner": "sales-data-team",
    "retries": DEFAULT_RETRIES,
    "retry_delay": timedelta(minutes=5),
}


def run_download(start_ts: str, end_ts: str) -> str:
    from company_sales.jobs.download import main

    return main(start_ts=start_ts, end_ts=end_ts)


def run_preprocess(start_ts: str, end_ts: str) -> str:
    from company_sales.jobs.preprocess import main

    return main(start_ts=start_ts, end_ts=end_ts)


def run_report(start_ts: str, end_ts: str) -> None:
    from company_sales.jobs.report import main

    main(start_ts=start_ts, end_ts=end_ts)


with DAG(
    dag_id="hourly_sales_pipeline",
    default_args=default_args,
    start_date=datetime(2026, 5, 1),
    schedule="@hourly",
    catchup=False,
    max_active_runs=1,
    tags=["sales", "git-sync"],
) as dag:
    download = PythonOperator(
        task_id="download",
        python_callable=run_download,
        op_kwargs={
            "start_ts": "{{ data_interval_start }}",
            "end_ts": "{{ data_interval_end }}",
        },
        execution_timeout=timedelta(minutes=30),
    )

    preprocess = PythonOperator(
        task_id="preprocess",
        python_callable=run_preprocess,
        op_kwargs={
            "start_ts": "{{ data_interval_start }}",
            "end_ts": "{{ data_interval_end }}",
        },
        execution_timeout=timedelta(hours=1),
    )

    report = PythonOperator(
        task_id="report",
        python_callable=run_report,
        op_kwargs={
            "start_ts": "{{ data_interval_start }}",
            "end_ts": "{{ data_interval_end }}",
        },
        execution_timeout=timedelta(minutes=20),
    )

    download >> preprocess >> report
```

DAG 파일에서는 secret을 직접 import하지 않는다. Task 함수 안에서 job module을 import하고, job module이 client를 통해 secret을 사용한다.

hourly job 파일은 처리 구간을 기준으로 동작한다.

```python
# dags/company_sales/jobs/download.py

def main(start_ts: str, end_ts: str) -> str:
    from company_sales.clients import get_minio_bucket, make_minio_client

    client = make_minio_client()
    bucket = get_minio_bucket()

    run_date = start_ts[:10]
    run_hour = start_ts[11:13]
    output_key = f"raw/sales/dt={run_date}/hour={run_hour}/data.csv"

    print(f"process interval: {start_ts} <= data < {end_ts}")
    print(f"write minio://{bucket}/{output_key}")
    # client.put_object(...)
    return f"minio://{bucket}/{output_key}"
```

날짜만 쓰는 `--date {{ ds }}` 방식은 daily DAG에는 괜찮지만, hourly DAG에서는 같은 날짜의 여러 실행을 구분하지 못하므로 피한다.

## Bitbucket Git Sync 배포 흐름

일반적인 흐름은 다음과 같다.

```text
local edit
  -> local test
  -> git commit
  -> git push Bitbucket
  -> Airflow Git Sync가 repo/branch를 pull
  -> Scheduler가 DAG parse
  -> UI에서 DAG 확인
  -> 수동 실행
  -> 로그 확인
```

배포 절차:

1. 로컬에서 Python 파일과 DAG를 수정한다.
2. `python dags/daily_sales_pipeline.py`로 DAG import를 확인한다.
3. 가능하면 `pytest`를 실행한다.
4. Bitbucket에 commit/push한다.
5. Airflow Git Sync interval만큼 기다린다.
6. Airflow UI에서 DAG import error를 확인한다.
7. 처음에는 수동 실행으로 검증한다.
8. 정상 확인 후 schedule을 활성화한다.

Git Sync interval은 회사 설정에 따라 다르다. push 직후 UI에 바로 반영되지 않을 수 있다.

## Git Sync에서 자주 생기는 문제

| 증상 | 원인 | 확인/해결 |
|------|------|-----------|
| push했는데 DAG가 안 바뀜 | Git Sync interval 지연 | 몇 분 대기 후 UI 새로고침 |
| 특정 branch 변경이 반영 안 됨 | Airflow가 다른 branch를 보고 있음 | 운영팀에 sync branch 확인 |
| DAG import error | import path 문제 | 공통 module을 `dags/` 아래로 이동 |
| `ModuleNotFoundError` | 패키지 미설치 또는 module 경로 오류 | Worker 패키지와 DAG root 확인 |
| secret 수정 후에도 이전 값 사용 | 다른 branch/commit을 sync 중 | UI에서 DAG code 또는 Git Sync commit 확인 |
| local에서는 되는데 서버에서 실패 | Python/package/OS 차이 | 환경 probe DAG로 Worker 환경 확인 |

## 패키지 설치에 대한 현실

Git Sync는 보통 repository 파일을 동기화할 뿐이다. `requirements-app.txt`를 push한다고 Airflow Worker에 자동 설치되는 것은 아니다.

패키지 문제는 별도로 판단한다.

| 상황 | 대응 |
|------|------|
| Worker에 이미 패키지가 있음 | 그대로 사용 |
| pure Python helper 코드 | DAG repo 안의 package로 포함 가능 |
| `pandas`, `numpy`, `pyarrow` 필요 | Worker 설치, venv, container 필요 |
| 외부 PyPI 차단 | 사내 Nexus/Artifactory 필요 |
| 설치 권한 없음 | 서버에 있는 패키지 버전에 맞춰 코드 작성 |

작은 내부 helper module은 repository에 포함해도 된다. 하지만 `numpy`, `pandas`, `pyarrow`처럼 compiled dependency가 있는 패키지를 DAG repo에 vendor로 넣는 방식은 피한다.

## Secret 변경 절차

secret이 코드에 들어가는 환경에서는 변경 절차가 중요하다.

1. 새 key/password를 발급한다.
2. Bitbucket에서 `secrets.py`를 수정한다.
3. commit message에는 secret 값을 쓰지 않는다.
4. push 후 Airflow가 새 commit을 sync했는지 확인한다.
5. smoke test DAG 또는 수동 실행으로 연결을 확인한다.
6. 기존 key/password를 폐기한다.

실수로 잘못된 secret을 public repository나 잘못된 branch에 push했다면 파일 삭제만으로 끝나지 않는다. Git history에 남아 있으므로 즉시 key rotation을 해야 한다.

## Review 기준

secret이 코드에 들어가는 repository에서는 code review 기준을 더 엄격하게 둔다.

확인할 것:

- secret 값이 로그에 출력되지 않는가
- secret 값이 exception message에 들어가지 않는가
- access key가 필요한 최소 권한만 가지는가
- 운영/개발 secret이 섞이지 않았는가
- Bitbucket repository 접근 권한이 제한되어 있는가
- `requirements-app.txt` 변경이 실제 Airflow 환경에 반영 가능한가
- DAG import 시점에 외부 시스템 접속을 시도하지 않는가

## 이 조건에서의 권장 원칙

- Connection을 못 쓰면 `secrets.py`를 한 곳에 모은다.
- secret을 쓰는 코드는 Task 실행 시점에 import한다.
- Bitbucket repository 접근 권한을 secret 접근 권한처럼 관리한다.
- Git Sync는 source 배포이며 package 설치가 아님을 기억한다.
- 처음 배포는 항상 `schedule=None` 또는 pause 상태에서 수동 실행한다.
- 패키지 문제와 credential 문제를 분리해서 디버깅한다.
- secret이 노출되면 코드 수정이 아니라 key rotation으로 대응한다.
