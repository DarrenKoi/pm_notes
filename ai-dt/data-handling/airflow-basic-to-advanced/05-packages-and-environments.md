---
tags: [airflow, python, dependency, virtualenv, external-python, kubernetes]
level: intermediate-advanced
last_updated: 2026-05-02
status: complete
---

# 05. 패키지와 실행 환경

## 목표

로컬에서는 잘 실행되는 Python 코드가 Airflow 서버에서는 실패하는 가장 흔한 이유는 실행 환경 차이다.

대표적인 오류:

```text
ModuleNotFoundError: No module named 'pandas'
ImportError: cannot import name ...
ValueError: numpy.dtype size changed
command not found
Permission denied
```

이 문서에서는 회사 관리형 Airflow에서 패키지와 Python 버전 문제를 해결하는 방법을 정리한다.

## 먼저 알아야 할 사실

`PythonOperator`, `@task`, `BashOperator`로 실행한 Python 코드는 기본적으로 **Airflow Worker의 Python 환경**을 사용한다.

즉, 로컬에 `pandas==2.2.2`가 설치되어 있어도 Airflow Worker에 `pandas==1.5.3`만 있으면 서버에서는 `1.5.3`으로 실행된다.

Airflow 서버에서 패키지 설치 권한이 없으면 사용자가 DAG만 올려서는 해결할 수 없는 문제가 많다.

## 환경 조사 DAG

처음에는 아래 DAG를 올려 Worker의 실제 환경을 확인한다. secret 값은 출력하지 않는다.

```python
from datetime import datetime

from airflow.decorators import dag, task


@dag(
    dag_id="airflow_env_probe",
    start_date=datetime(2026, 5, 1),
    schedule=None,
    catchup=False,
    tags=["debug"],
)
def airflow_env_probe():
    @task
    def print_runtime() -> None:
        import os
        import platform
        import subprocess
        import sys
        from importlib.metadata import PackageNotFoundError, version

        print("=== Python ===")
        print(f"executable={sys.executable}")
        print(f"version={sys.version}")
        print(f"platform={platform.platform()}")
        print(f"cwd={os.getcwd()}")

        print("=== Selected packages ===")
        package_names = [
            "apache-airflow",
            "pandas",
            "numpy",
            "requests",
            "boto3",
            "minio",
            "openpyxl",
            "pyarrow",
            "scikit-learn",
        ]
        for name in package_names:
            try:
                print(f"{name}=={version(name)}")
            except PackageNotFoundError:
                print(f"{name}: NOT INSTALLED")

        print("=== pip freeze first 200 lines ===")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=False,
        )
        print("\n".join(result.stdout.splitlines()[:200]))
        if result.stderr:
            print(result.stderr)

    print_runtime()


airflow_env_probe()
```

확인할 것:

- Python 버전
- `apache-airflow` 버전
- 내 코드에 필요한 패키지 설치 여부
- 패키지 버전
- provider 설치 여부
- Worker 실행 경로
- `pip freeze` 실행 가능 여부

조사가 끝나면 이 DAG는 비활성화하거나 삭제한다.

## 로컬 의존성 정리

로컬에서 모든 패키지를 그대로 freeze하면 개발 도구까지 섞인다. 운영 실행에 필요한 것만 별도 파일로 정리한다.

```text
requirements-app.txt
```

예:

```text
pandas==2.2.2
numpy==1.26.4
requests==2.32.3
openpyxl==3.1.5
pyarrow==16.1.0
```

개발 도구는 분리한다.

```text
requirements-dev.txt
```

예:

```text
pytest==8.2.2
ruff==0.5.0
mypy==1.10.1
```

## 전략 선택표

| 상황 | 권장 전략 |
|------|----------|
| 서버에 필요한 패키지가 모두 있음 | `PythonOperator`, `@task`, `BashOperator` |
| 패키지 한두 개만 추가 필요 | 운영팀에 Worker 이미지/requirements 추가 요청 |
| 특정 DAG만 다른 버전 필요 | `PythonVirtualenvOperator` |
| 매 실행마다 설치가 느림 | `ExternalPythonOperator` |
| 시스템 패키지, Java, CLI, GPU 필요 | 컨테이너 실행 |
| 회사가 Bitbucket Git Sync만 허용 | DAG repository에는 코드만 넣고, 서버 설치 버전에 맞춰 코드 수정 |
| 외부 PyPI 차단 | 사내 Nexus/Artifactory/wheelhouse 사용 |

## 전략 A. Airflow Worker 기본 환경 사용

가장 단순하다.

```python
from airflow.operators.python import PythonOperator


def run_job(run_date: str) -> None:
    import pandas as pd

    print(pd.__version__)
    print(run_date)


task = PythonOperator(
    task_id="run_job",
    python_callable=run_job,
    op_kwargs={"run_date": "{{ ds }}"},
)
```

장점:

- 빠르다.
- 단순하다.
- 운영팀 입장에서도 관리가 쉽다.

단점:

- 서버 패키지 버전에 종속된다.
- 다른 팀 DAG와 의존성 충돌이 생길 수 있다.
- 사용자가 직접 패키지를 바꾸기 어렵다.

## 전략 B. PythonVirtualenvOperator

Task 실행 시점에 별도 virtualenv를 만들고 패키지를 설치한다.

```python
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonVirtualenvOperator


def run_job(run_date: str) -> None:
    import pandas as pd

    print(f"pandas={pd.__version__}, run_date={run_date}")


with DAG(
    dag_id="virtualenv_example",
    start_date=datetime(2026, 5, 1),
    schedule=None,
    catchup=False,
) as dag:
    task = PythonVirtualenvOperator(
        task_id="run_job",
        python_callable=run_job,
        op_kwargs={"run_date": "{{ ds }}"},
        requirements=[
            "pandas==2.2.2",
            "numpy==1.26.4",
            "openpyxl==3.1.5",
        ],
        system_site_packages=False,
    )
```

중요한 제약:

- 함수 내부에서 필요한 import를 해야 한다.
- 함수 바깥의 global import나 global object에 기대면 안 된다.
- Airflow context 객체를 넘기는 데 제약이 있다.
- 매번 venv를 만들면 느릴 수 있다.
- Worker의 temp disk 공간이 필요하다.
- 회사망에서 패키지 저장소 접근이 가능해야 한다.

사내 PyPI/Nexus가 있다면 설치 옵션을 붙인다.

```python
task = PythonVirtualenvOperator(
    task_id="run_job",
    python_callable=run_job,
    requirements=["pandas==2.2.2", "openpyxl==3.1.5"],
    pip_install_options=[
        "--index-url",
        "https://nexus.your-company.com/repository/pypi/simple",
        "--trusted-host",
        "nexus.your-company.com",
    ],
)
```

현재 환경처럼 Connection 접근이 불가능하면 Airflow Connection 기반 package index 설정은 사용할 수 없다. 이 경우 운영팀이 Worker의 pip config를 설정해주거나, DAG 코드의 `pip_install_options`에 사내 Nexus 주소를 직접 넣어야 한다.

credential이 필요한 package index라면 코드에 넣는 순간 Bitbucket repository가 credential 저장소가 된다. 접근 권한과 key rotation 정책을 반드시 확인한다.

## 전략 C. ExternalPythonOperator

미리 만들어진 Python 환경을 사용한다.

운영팀이 Worker에 다음 venv를 만들어준다고 가정한다.

```text
/opt/airflow/venvs/company-data-jobs/bin/python
```

DAG:

```python
from datetime import datetime

from airflow import DAG
from airflow.operators.python import ExternalPythonOperator


def run_job(run_date: str) -> None:
    import pandas as pd

    print(f"python env uses pandas={pd.__version__}")
    print(f"run_date={run_date}")


with DAG(
    dag_id="external_python_example",
    start_date=datetime(2026, 5, 1),
    schedule=None,
    catchup=False,
) as dag:
    task = ExternalPythonOperator(
        task_id="run_job",
        python="/opt/airflow/venvs/company-data-jobs/bin/python",
        python_callable=run_job,
        op_kwargs={"run_date": "{{ ds }}"},
    )
```

장점:

- 매번 패키지를 설치하지 않아 빠르다.
- Airflow 기본 환경과 job 환경을 분리할 수 있다.

주의:

- 모든 Worker에 같은 경로와 같은 패키지가 있어야 한다.
- CeleryExecutor에서는 어떤 Worker에서 실행될지 모를 수 있다.
- KubernetesExecutor에서는 Pod 이미지 안에 venv가 있어야 한다.
- venv 업데이트 절차를 운영팀과 정해야 한다.

## 전략 D. 컨테이너 실행

가장 강한 격리 방식이다. 사내 Kubernetes 또는 Docker 실행이 허용되어야 한다.

```python
from datetime import datetime

from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator


with DAG(
    dag_id="container_job_example",
    start_date=datetime(2026, 5, 1),
    schedule=None,
    catchup=False,
) as dag:
    task = KubernetesPodOperator(
        task_id="run_container_job",
        name="company-data-job",
        namespace="airflow",
        image="harbor.your-company.com/data/company-job:1.0.0",
        cmds=["python", "-m", "company_jobs.analyze"],
        arguments=[
            "--start-ts",
            "{{ data_interval_start }}",
            "--end-ts",
            "{{ data_interval_end }}",
        ],
        get_logs=True,
    )
```

장점:

- Airflow 환경과 완전히 분리 가능
- system package, CLI, Java, ML library 포함 가능
- 재현성이 좋음

단점:

- 이미지 빌드/배포 권한 필요
- 사내 registry 필요
- Kubernetes provider 필요
- secret, volume, network policy 설정 필요

## Airflow provider

Airflow는 core package와 provider package가 분리되어 있다. S3, Kubernetes, JDBC, Slack 같은 연동은 provider 설치 여부에 따라 사용할 수 있다.

예:

```text
apache-airflow-providers-amazon
apache-airflow-providers-cncf-kubernetes
apache-airflow-providers-postgres
apache-airflow-providers-slack
```

provider가 없으면 import error가 난다.

```text
ModuleNotFoundError: No module named 'airflow.providers.amazon'
```

이 경우 사용자가 DAG에서 해결하는 것이 아니라 운영팀에 provider 설치를 요청해야 하는 경우가 많다.

## Airflow constraints

Airflow Worker 이미지를 직접 만들거나 운영팀이 패키지를 설치할 때는 Airflow 버전에 맞는 constraints를 사용하는 것이 안전하다.

예:

```bash
AIRFLOW_VERSION=2.10.5
PYTHON_VERSION=3.11
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

python -m pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"
python -m pip install -r requirements-app.txt --constraint "${CONSTRAINT_URL}"
```

폐쇄망에서는 constraints 파일과 wheel 파일을 사내 저장소에 미리 반입해야 한다.

## 운영팀에 요청할 내용

패키지 문제를 운영팀에 전달할 때는 아래 형식이 좋다.

```text
요청 목적:
- daily_customer_report DAG 실행 환경 준비

현재 확인된 Airflow 환경:
- Airflow version: 2.10.5
- Python version: 3.11
- Executor: CeleryExecutor

필요 패키지:
- pandas==2.2.2
- numpy==1.26.4
- pyarrow==16.1.0
- openpyxl==3.1.5

현재 서버 패키지:
- pandas==1.5.3
- numpy==1.24.4
- pyarrow: not installed

희망 방식:
- 1순위: /opt/airflow/venvs/company-data-jobs venv 생성 후 ExternalPythonOperator 사용
- 2순위: Worker image에 requirements-app.txt 반영
- 3순위: KubernetesPodOperator로 사내 이미지 실행

추가 확인 필요:
- 사내 PyPI/Nexus 주소
- venv가 모든 Worker에 동일하게 배포되는지
- MinIO/S3 connection id
- DAG에서 사용할 queue/pool
```

## 결정 기준

처음 시작할 때는 다음 순서로 판단한다.

```text
1. 서버 기본 환경으로 실행 가능한가?
   가능 -> PythonOperator/BashOperator 사용
   불가능 -> 2번

2. 특정 패키지만 추가하면 되는가?
   가능 -> 운영팀에 Worker 패키지 추가 요청
   불가능 -> 3번

3. DAG별 가상환경이 허용되는가?
   가능 -> PythonVirtualenvOperator 또는 ExternalPythonOperator
   불가능 -> 4번

4. 컨테이너 실행이 허용되는가?
   가능 -> KubernetesPodOperator/DockerOperator
   불가능 -> 서버 환경에 맞춰 코드 수정
```

## 체크리스트

- 로컬 Python 버전을 기록했는가
- Airflow Worker Python 버전을 확인했는가
- 로컬 `requirements-app.txt`를 만들었는가
- 서버에 설치된 패키지 버전을 확인했는가
- provider 설치 여부를 확인했는가
- 외부 PyPI 접근 가능 여부를 확인했는가
- 사내 Nexus/Artifactory 주소를 확인했는가
- venv/container 사용 권한을 확인했는가
- Worker가 여러 대일 때 환경이 모두 같은지 확인했는가
- Bitbucket Git Sync가 requirements를 자동 설치하지 않는다는 점을 확인했는가

## 다음 단계

다음 문서에서는 로컬에서 DAG와 Python 코드를 어떻게 테스트하고 배포 전에 점검할지 다룬다.

- [06. 로컬 개발과 테스트](./06-local-development-and-testing.md)
