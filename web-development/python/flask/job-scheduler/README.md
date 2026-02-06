# Python Job Scheduler

> Flask + APScheduler 기반의 Python 작업 스케줄러. 웹 대시보드에서 작업 상태 확인, 수동 실행, 로그 조회 가능.

## 왜 필요한가? (Why)

- Python 스크립트를 주기적으로 실행해야 하는 경우 (데이터 수집, 보고서 생성, 자동화 등)
- Windows Task Scheduler/cron 대신 웹 UI로 관리하고 싶은 경우
- 작업 실행 기록과 로그를 한 곳에서 확인하고 싶은 경우

## 핵심 개념 (What)

### 아키텍처

```
jobs/                         # 작업 폴더
├── {user}/
│   └── {task}/
│       ├── pyproject.toml    # uv 프로젝트 설정
│       ├── job.yaml          # 스케줄 설정
│       └── main.py           # 실행 스크립트

src/
├── app.py                    # Flask 웹 서버
├── job_manager.py            # 작업 관리 엔진
├── config.yaml               # 서버 설정
└── templates/                # 웹 UI
```

### 작업 구조

각 작업은 `jobs/{user}/{task}/` 디렉토리에 위치하며, 최소한 `job.yaml`과 실행 스크립트가 필요합니다.

**job.yaml 예시:**

```yaml
name: "My Task"
description: "무엇을 하는 작업인지 설명"

schedule:
  type: interval        # "interval" 또는 "cron"
  minutes: 30           # interval: seconds, minutes, hours
  # type: cron
  # hour: 9
  # minute: 0

entry_point: main.py    # 실행할 파일 (기본값: main.py)
timeout: 3600           # 타임아웃 초 (기본값: 3600)
```

### 스케줄 타입

| 타입 | 설명 | 옵션 예시 |
|------|------|-----------|
| `interval` | 일정 간격으로 반복 | `seconds: 30`, `minutes: 5`, `hours: 1` |
| `cron` | cron 표현식 | `hour: 9`, `minute: 0`, `day_of_week: "mon-fri"` |

## 어떻게 사용하는가? (How)

### 1. 설치

```bash
cd web-development/python/flask/job-scheduler
uv sync
```

### 2. 서버 실행

```bash
uv run src/app.py
```

http://localhost:5050 으로 접속

### 3. 작업 추가

```bash
# 1. 작업 디렉토리 생성
mkdir -p jobs/myname/my_task

# 2. job.yaml 작성
cat > jobs/myname/my_task/job.yaml << 'EOF'
name: "My Task"
description: "설명"
schedule:
  type: interval
  hours: 1
entry_point: main.py
timeout: 300
EOF

# 3. pyproject.toml 작성 (의존성이 있는 경우)
cat > jobs/myname/my_task/pyproject.toml << 'EOF'
[project]
name = "my-task"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["requests"]
EOF

# 4. 스크립트 작성
cat > jobs/myname/my_task/main.py << 'EOF'
def main():
    print("Hello from my task!")

if __name__ == "__main__":
    main()
EOF

# 5. 대시보드에서 Reload Jobs 클릭 또는 자동 스캔 대기 (기본 60초)
```

### 4. 설정 변경

`src/config.yaml`에서 서버 포트, 스캔 주기, 허용 사용자 등을 설정할 수 있습니다.

```yaml
server:
  port: 8080              # 포트 변경 (기본값: 5050)

scheduler:
  scan_interval_seconds: 30  # 스캔 주기 변경

security:
  allowed_users:
    - daeyoung            # 특정 사용자만 허용

log_retention:
  days: 7                 # 로그 7일 보존
```

### 5. 웹 대시보드 기능

- **Dashboard** (`/`): 전체 작업 목록, 상태, 다음 실행 시간, 활성화 토글, 수동 실행
- **Runs** (`/runs`): 실행 기록, 상태별 필터, 로그 조회 (모달)
- 30초 자동 새로고침
- 최근 24시간 실패 건수 배지 표시

## 참고 자료 (References)

- [APScheduler 공식 문서](https://apscheduler.readthedocs.io/)
- [Flask 공식 문서](https://flask.palletsprojects.com/)
- [uv - Python 패키지 매니저](https://docs.astral.sh/uv/)
