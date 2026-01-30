---
tags: [python, uv, package-manager, tooling]
level: beginner
last_updated: 2026-01-31
status: complete
---

# uv - 차세대 Python 패키지 매니저

> Rust로 작성된 초고속 Python 패키지 및 프로젝트 매니저로, pip/poetry/pyenv/virtualenv를 하나로 통합한 올인원 도구

## 왜 필요한가? (Why)

기존 Python 패키지 관리 생태계의 문제점:

- **도구 파편화**: pip, pip-tools, virtualenv, pyenv, poetry, pipx, twine 등 역할별로 다른 도구 필요
- **느린 속도**: pip의 의존성 해결(dependency resolution)은 대규모 프로젝트에서 수 분 소요
- **환경 관리 번거로움**: venv 생성 → activate → pip install 의 반복적 워크플로우
- **재현성 부족**: `requirements.txt`만으로는 완벽한 재현이 어려움

uv는 이 모든 문제를 **하나의 바이너리**로 해결하며, pip 대비 **10~100배 빠른 속도**를 제공한다.

## 핵심 개념 (What)

### uv가 대체하는 도구들

| 기존 도구 | 역할 | uv 대체 명령어 |
|-----------|------|----------------|
| pip | 패키지 설치 | `uv pip install` |
| pip-tools | 의존성 잠금 | `uv lock` |
| virtualenv | 가상환경 생성 | `uv venv` |
| pyenv | Python 버전 관리 | `uv python install` |
| pipx | CLI 도구 실행 | `uvx` / `uv tool` |
| poetry | 프로젝트 관리 | `uv init` / `uv add` |
| twine | 패키지 배포 | `uv publish` |

### 핵심 특징

- **Rust 기반**: 네이티브 바이너리로 Python 런타임 불필요
- **글로벌 캐시**: 패키지를 한 번만 다운로드하고 프로젝트 간 링크로 공유 (디스크 절약)
- **유니버설 락파일**: 플랫폼 독립적인 `uv.lock` 파일로 재현 가능한 빌드
- **pip 호환 인터페이스**: 기존 pip 명령어와 유사한 인터페이스 제공

## 어떻게 사용하는가? (How)

### 1. 설치

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Homebrew
brew install uv

# pip (기존 환경에서)
pip install uv
```

### 2. Python 버전 관리

```bash
# Python 설치
uv python install 3.12

# 설치된 Python 목록
uv python list

# 프로젝트별 Python 버전 고정
uv python pin 3.12
# → .python-version 파일 생성
```

### 3. 프로젝트 생성 및 관리

```bash
# 새 프로젝트 초기화
uv init my-project
cd my-project

# 의존성 추가
uv add fastapi
uv add uvicorn
uv add --dev pytest ruff

# 의존성 제거
uv remove flask

# 락파일 기반으로 동기화
uv sync
```

생성되는 파일 구조:

```
my-project/
├── pyproject.toml    # 프로젝트 설정 및 의존성 선언
├── uv.lock           # 정확한 버전 잠금 (커밋 대상)
├── .python-version   # Python 버전
└── .venv/            # 가상환경 (자동 생성)
```

### 4. 스크립트 실행

```bash
# 프로젝트 내 스크립트 실행 (자동으로 venv 활성화)
uv run python main.py
uv run pytest
uv run uvicorn app:app --reload

# 인라인 의존성이 있는 단일 파일 스크립트
uv run script.py
```

인라인 의존성 스크립트 예시:

```python
# /// script
# requires-python = ">=3.12"
# dependencies = ["requests", "rich"]
# ///

import requests
from rich import print

resp = requests.get("https://api.github.com")
print(resp.json())
```

```bash
uv run my_script.py  # 자동으로 requests, rich 설치 후 실행
```

### 5. CLI 도구 실행 (pipx 대체)

```bash
# 일회성 실행 (임시 환경)
uvx ruff check .
uvx black .

# 영구 설치
uv tool install ruff
uv tool install httpie
```

### 6. pip 호환 모드

기존 프로젝트에서 점진적으로 전환할 때 유용:

```bash
# 기존 pip 명령어와 동일한 인터페이스
uv pip install flask
uv pip install -r requirements.txt
uv pip freeze > requirements.txt

# 가상환경 생성
uv venv
source .venv/bin/activate
```

### 7. pyproject.toml 예시

```toml
[project]
name = "my-project"
version = "0.1.0"
description = "My awesome project"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn>=0.34.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "ruff>=0.9.0",
]
```

## pip vs uv 명령어 비교

| 작업 | pip | uv |
|------|-----|-----|
| 패키지 설치 | `pip install flask` | `uv add flask` |
| requirements 설치 | `pip install -r requirements.txt` | `uv pip install -r requirements.txt` |
| 가상환경 생성 | `python -m venv .venv` | `uv venv` |
| 프로젝트 초기화 | (없음) | `uv init` |
| 의존성 잠금 | `pip freeze` | `uv lock` |
| 스크립트 실행 | `python script.py` | `uv run script.py` |
| 도구 실행 | `pipx run ruff` | `uvx ruff` |

## 실무 팁

### FastAPI 프로젝트 시작 예시

```bash
uv init my-api
cd my-api
uv add fastapi uvicorn[standard] sqlalchemy
uv add --dev pytest httpx ruff
uv run uvicorn app:app --reload
```

### 기존 프로젝트 마이그레이션

```bash
# 기존 requirements.txt가 있는 프로젝트에서
cd existing-project
uv init
uv add $(cat requirements.txt | grep -v '^#' | tr '\n' ' ')
# 이후 uv.lock과 pyproject.toml로 관리
```

### CI/CD에서 활용

```yaml
# GitHub Actions 예시
- uses: astral-sh/setup-uv@v5
- run: uv sync
- run: uv run pytest
```

## 다른 도구와 비교

| 비교 항목 | pip | poetry | uv |
|-----------|-----|--------|-----|
| 속도 | 느림 | 보통 | 매우 빠름 (10-100x) |
| Python 버전 관리 | X | X | O |
| 락파일 | X | O | O |
| 가상환경 자동 관리 | X | O | O |
| CLI 도구 실행 | X | X | O |
| 단일 바이너리 | X | X | O |
| pip 호환 | - | X | O |

## 참고 자료

- [uv 공식 문서](https://docs.astral.sh/uv/)
- [GitHub: astral-sh/uv](https://github.com/astral-sh/uv)
- [Real Python: Managing Python Projects With uv](https://realpython.com/python-uv/)

## 관련 문서

- [FastAPI 관련 문서](./fastapi/) (uv로 프로젝트 셋업 시 활용)
