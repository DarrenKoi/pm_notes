---
tags: [python, uv, pip, migration, troubleshooting]
level: intermediate
last_updated: 2026-01-31
status: complete
---

# pip에서 uv로 마이그레이션 가이드

> 기존 pip 기반 프로젝트를 uv로 전환하는 단계별 가이드와 주요 트러블슈팅

## 왜 필요한가? (Why)

- 기존 프로젝트의 의존성 관리를 더 빠르고 재현 가능하게 개선
- `requirements.txt` → `pyproject.toml` + `uv.lock` 전환으로 의존성 충돌 방지
- 팀원 간 환경 불일치 문제 해결
- pip를 당장 버릴 필요 없이 **점진적 전환** 가능

## 단계별 마이그레이션 (How)

### Step 1: uv 설치

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 또는 Homebrew
brew install uv

# 설치 확인
uv --version
```

### Step 2: 현재 프로젝트 상태 파악

전환 전에 기존 의존성을 정확히 파악한다.

```bash
cd my-existing-project

# 현재 설치된 패키지 목록 추출 (아직 없다면)
pip freeze > requirements-freeze.txt

# 기존 requirements.txt 확인
cat requirements.txt
```

기존 프로젝트 구조 예시:

```
my-project/
├── requirements.txt          # 또는 requirements-dev.txt 등
├── setup.py                  # 또는 setup.cfg
├── venv/                     # 기존 가상환경
└── src/
```

### Step 3: pyproject.toml 생성 (uv init)

**경우 A: pyproject.toml이 없는 프로젝트**

```bash
# 기존 프로젝트 디렉토리에서 실행
uv init
```

이렇게 하면 기본 `pyproject.toml`이 생성된다. 기존 파일은 건드리지 않는다.

**경우 B: 이미 pyproject.toml이 있는 프로젝트 (setuptools/flit 등)**

이미 `pyproject.toml`에 `[project]` 섹션과 `dependencies`가 정의되어 있다면 uv가 바로 인식한다. 별도의 init이 필요 없다.

```bash
# 바로 락파일 생성 가능
uv lock
```

### Step 4: 의존성 옮기기

**방법 1: requirements.txt에서 한 번에 추가**

```bash
# requirements.txt의 패키지를 pyproject.toml에 추가
uv add $(grep -v '^\s*#\|^\s*$' requirements.txt | sed 's/==.*//g' | tr '\n' ' ')
```

> 주의: `==` 버전 고정을 제거하고 추가하는 방식. uv.lock이 정확한 버전을 잠그므로 pyproject.toml에는 유연한 버전 범위가 낫다.

**방법 2: 정확한 버전을 유지하고 싶다면**

```bash
# 버전 고정 그대로 추가
uv add $(grep -v '^\s*#\|^\s*$' requirements.txt | tr '\n' ' ')
```

**방법 3: 수동으로 pyproject.toml 편집**

패키지가 많지 않거나 정리가 필요한 경우 직접 편집이 가장 깔끔하다.

```toml
[project]
name = "my-project"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "fastapi>=0.115.0",
    "sqlalchemy>=2.0",
    "httpx>=0.27",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "ruff>=0.9",
]
```

그런 다음:

```bash
uv lock    # 락파일 생성
uv sync    # 새 .venv에 설치
```

### Step 5: dev 의존성 분리

기존에 `requirements-dev.txt`가 있었다면:

```bash
# dev 의존성은 --dev 플래그로 추가
uv add --dev pytest ruff mypy httpx
```

### Step 6: 가상환경 전환

```bash
# 기존 venv 삭제 (선택사항, 나중에 해도 됨)
rm -rf venv/    # 또는 .venv/

# uv가 자동으로 .venv를 생성하고 관리
uv sync
```

`uv sync` 실행 시 `.venv/`가 없으면 자동 생성된다.

### Step 7: 실행 방식 변경

```bash
# Before (pip)
source venv/bin/activate
python main.py
pytest

# After (uv) - activate 불필요
uv run python main.py
uv run pytest
```

### Step 8: CI/CD 업데이트

**GitHub Actions 예시**

```yaml
# Before
- uses: actions/setup-python@v5
  with:
    python-version: '3.12'
- run: pip install -r requirements.txt
- run: pytest

# After
- uses: astral-sh/setup-uv@v5
- run: uv sync
- run: uv run pytest
```

**Docker 예시**

```dockerfile
# Before
FROM python:3.12-slim
COPY requirements.txt .
RUN pip install -r requirements.txt

# After
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY . .
CMD ["uv", "run", "python", "main.py"]
```

### Step 9: Git 설정 업데이트

`.gitignore`에 추가:

```gitignore
.venv/
```

커밋 대상에 포함:

```
pyproject.toml    # 의존성 선언
uv.lock           # 정확한 버전 잠금 (반드시 커밋)
.python-version   # Python 버전 고정 (선택)
```

### Step 10: 기존 파일 정리

전환이 완료되고 안정적으로 동작하면:

```bash
# 더 이상 필요 없는 파일 제거
rm requirements.txt
rm requirements-dev.txt
rm setup.py          # pyproject.toml로 대체된 경우
rm setup.cfg         # pyproject.toml로 대체된 경우
```

> 팀 프로젝트라면 모든 팀원이 uv로 전환될 때까지 requirements.txt를 유지하는 것도 방법이다.

---

## 트러블슈팅 (Troubleshooting)

### 1. 의존성 충돌 (Resolution failed)

**증상:**
```
error: No solution found when resolving dependencies:
  ╰─▶ Because package-a==1.0 depends on numpy>=1.24 and package-b==2.0
      depends on numpy<1.24, we can conclude that ...
```

**원인:** 두 패키지가 요구하는 의존성 버전이 충돌

**해결:**
```bash
# 어떤 패키지가 충돌하는지 확인
uv tree

# 특정 패키지 버전을 명시적으로 지정
uv add "numpy>=1.24,<2.0"

# 또는 override로 강제 지정 (pyproject.toml)
[tool.uv]
override-dependencies = ["numpy==1.26.4"]
```

### 2. 프라이빗 PyPI / 사내 레지스트리

**증상:** 사내 패키지 설치 실패

**해결:** `pyproject.toml`에 인덱스 추가:

```toml
[tool.uv]
index-url = "https://pypi.company.com/simple/"
extra-index-url = ["https://pypi.org/simple/"]
```

또는 환경변수:

```bash
export UV_INDEX_URL="https://pypi.company.com/simple/"
export UV_EXTRA_INDEX_URL="https://pypi.org/simple/"
```

### 3. 시스템 의존성이 필요한 패키지 (빌드 실패)

**증상:**
```
error: Failed to build package-name
  ╰─▶ Building wheel failed
```

**원인:** C 확장이 필요한 패키지(예: psycopg2, lxml, pillow)가 시스템 라이브러리 없이 빌드 실패

**해결:**
```bash
# 바이너리 휠이 있는 패키지로 대체
uv add psycopg2-binary    # psycopg2 대신
uv add pillow              # 보통 휠 제공됨

# 또는 시스템 라이브러리 먼저 설치
# macOS
brew install libpq postgresql
# Ubuntu
sudo apt install libpq-dev
```

### 4. Python 버전 불일치

**증상:**
```
error: No interpreter found for Python >=3.12
```

**해결:**
```bash
# uv로 Python 설치
uv python install 3.12

# 프로젝트의 requires-python 확인/수정
# pyproject.toml에서:
requires-python = ">=3.10"    # 범위를 넓히거나

# 프로젝트에 버전 고정
uv python pin 3.12
```

### 5. uv.lock 충돌 (팀 협업 시)

**증상:** Git merge 시 `uv.lock` 충돌

**해결:**
```bash
# uv.lock 충돌 시 어느 한쪽을 선택한 뒤 재생성
git checkout --theirs uv.lock    # 또는 --ours
uv lock
```

`uv.lock`은 자동 생성 파일이므로 수동 편집하지 말고 항상 `uv lock`으로 재생성한다.

### 6. editable install / 로컬 패키지

**증상:** `pip install -e .` 처럼 개발 모드 설치가 필요

**해결:**
```bash
# uv는 프로젝트를 자동으로 editable로 설치
uv sync    # 현재 프로젝트가 editable로 설치됨

# 로컬 경로의 다른 패키지 추가
uv add --editable ../my-local-lib
```

### 7. 특정 플랫폼에서만 필요한 패키지

**증상:** Windows에서만 필요한 `pywin32` 같은 패키지

**해결:** pyproject.toml에서 환경 마커 사용:

```toml
dependencies = [
    "pywin32>=306; sys_platform == 'win32'",
]
```

### 8. pip install -e . 로 설치한 패키지가 import 안 됨

**증상:** `uv sync` 후 로컬 패키지 import 실패

**원인:** 기존 venv에 editable로 설치했던 것이 새 .venv에는 없음

**해결:**
```bash
# src layout인 경우 pyproject.toml에 빌드 시스템 설정 확인
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

# 그 후 uv sync하면 자동으로 editable 설치됨
uv sync
```

---

## 점진적 전환 전략

팀 프로젝트에서 한 번에 전환이 어려운 경우:

```
Phase 1: uv를 pip 대용으로만 사용
         uv pip install -r requirements.txt
         (기존 requirements.txt 유지)

Phase 2: pyproject.toml 도입, uv lock 사용
         requirements.txt는 uv로부터 자동 생성하여 병행
         uv export > requirements.txt

Phase 3: 완전 전환
         requirements.txt 제거
         CI/CD를 uv sync로 변경
         팀 전원 uv 사용
```

## 참고 자료

- [uv 공식 마이그레이션 가이드](https://docs.astral.sh/uv/guides/projects/)
- [uv pip 호환 인터페이스](https://docs.astral.sh/uv/pip/compatibility/)

## 관련 문서

- [uv 패키지 매니저 개요](./uv-package-manager.md)
