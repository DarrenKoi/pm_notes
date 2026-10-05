---
tags: [python, uv, pip, migration, troubleshooting]
level: intermediate
last_updated: 2026-01-31
reviewed_on: 2026-10-04
review_status: partial
document_type: learning_note
category_major: "웹 개발"
category_middle: "백엔드 개발"
category_minor: "Python 환경·패키지"
note_kind: "학습"
classified_on: "2026-10-05"
---

# pip에서 uv로 마이그레이션 가이드

> 기존 pip 기반 프로젝트를 uv로 전환하는 단계별 가이드와 주요 트러블슈팅

## 왜 필요한가? (Why)

- 기존 프로젝트의 의존성 관리를 더 빠르고 재현 가능하게 개선
- `requirements.txt` → `pyproject.toml` + `uv.lock`으로 선언과 해결 결과를 분리; 충돌을 자동으로 없애지는 않음
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
uv init --bare --vcs none --no-workspace
```

기존 파일을 백업하고 diff를 확인한다. 이 명령은 최소 pyproject.toml을 생성하는 용도다. 기존 setup.py/setup.cfg의 패키지·entry point·빌드 옵션을 자동 이전하지 않는다.

**경우 B: 이미 pyproject.toml이 있는 프로젝트 (setuptools/flit 등)**

이미 `pyproject.toml`에 `[project]` 섹션과 `dependencies`가 정의되어 있다면 uv가 바로 인식한다. 별도의 init이 필요 없다.

```bash
# 바로 락파일 생성 가능
uv lock
```

### Step 4: 의존성 옮기기

**방법 1: requirements 파일을 파서로 가져오기**

```bash
uv add -r requirements.txt
```

기존 핀·extras·환경 마커를 보존하며 시작한다. 셸 공백 분할이나 `sed`로 `==`를 제거하지 않는다. 인덱스·constraints·editable·로컬 경로 옵션은 이전 후 pyproject.toml과 lock의 실제 반영을 검토한다.

**방법 2: 의도적으로 버전 정책을 바꾸기**

```bash
# 해당 패키지의 호환성 검증을 한 뒤 명시적으로 범위를 바꾸는 예
uv add "httpx>=0.27,<1"
```

lockfile은 새 해결 결과를 고정한다. 기존 핀을 제거했을 때의 호환성을 대신 보증하지 않는다. 모든 프로젝트에 유연한 범위가 더 낫다고 일반화하지 않는다.

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
# 기존 환경은 비교가 끝날 때까지 보존
# 프로젝트의 .venv를 생성/동기화
uv sync
```

`uv sync` 실행 시 `.venv/`가 없으면 생성한다. 기존 `.venv`를 사용하면 기본 exact sync로 lock에 없는 패키지가 제거될 수 있다. 기존 환경 경로를 먼저 확인한다.

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
- run: uv sync --locked
- run: uv run --locked pytest
```

**Docker 예시**

```dockerfile
# Before
FROM python:3.12-slim
COPY requirements.txt .
RUN pip install -r requirements.txt

# After: 앱 예시 (패키지 빌드용 소스까지 복사한 뒤 sync)
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /bin/uv
WORKDIR /app
ENV UV_PYTHON_DOWNLOADS=never
COPY . .
RUN uv sync --locked --no-dev
CMD ["/app/.venv/bin/python", "main.py"]
```

`.dockerignore`에는 `.venv/`를 넣어 호스트 환경을 이미지에 복사하지 않는다. uv 버전은 로컬 확인 버전 예시이며 이미지 pull/build는 미실행이다. 재현성 요구에 따라 Python·uv 이미지 digest도 고정한다. 의존성만 먼저 설치하는 별도 캐시 단계가 필요하면 `--no-install-project`를 쓰고 소스를 복사한 뒤 최종 sync를 한다. 런타임의 uv run 재동기화로 dev 의존성이 다시 설치되지 않도록 환경의 Python을 직접 실행한다.

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

전환·빌드·테스트·팀 CI 검증을 마친 뒤 역할별로 정리한다. requirements는 외부 소비자가 필요하면 export로 유지하고, setup.py/setup.cfg의 패키징·entry point·도구 설정이 이전되었는지 확인한다. 파일명만 보고 일괄 삭제하지 않는다. 이번 문서 정리는 기존 프로젝트 파일을 삭제하지 않았다.

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

```

override는 상위 패키지의 선언을 덮어쓰므로 실행 호환성 검증이 필요하다. 다음은 **pyproject.toml** 조각이다.

```toml
[tool.uv]
override-dependencies = ["numpy==1.26.4"]
```

### 2. 프라이빗 PyPI / 사내 레지스트리

**증상:** 사내 패키지 설치 실패

**해결:** `pyproject.toml`에 인덱스 추가:

```toml
[[tool.uv.index]]
name = "internal"
url = "https://packages.example.invalid/simple/"
default = true
```

또는 환경변수:

```bash
export UV_DEFAULT_INDEX="https://packages.example.invalid/simple/"
```

위 주소는 교체가 필요한 비밀 없는 예시다. default=true는 기본 PyPI를 대체한다. 공용 fallback이 필요하면 허용 정책과 패키지별 source를 정하고 인덱스를 추가한다. 기본 first-index는 패키지 이름이 발견된 첫 인덱스에서 후보를 선택한다. pip의 설정·동작과 완전히 같지 않다. 자격증명은 문서·URL·Git에 넣지 않는다.

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
requires-python = ">=3.12"    # 실제 코드가 요구하는 범위; 오류 우회로 낮추지 않음

# 프로젝트에 버전 고정
uv python pin 3.12
```

### 5. uv.lock 충돌 (팀 협업 시)

**증상:** Git merge 시 `uv.lock` 충돌

**해결:**
```bash
# pyproject.toml·sources의 양쪽 의도를 먼저 병합하고 백업
# 충돌 표식 없는 유효 lock을 바탕으로 재해결한 뒤 버전 diff 확인
uv lock
```

`uv.lock`은 자동 생성 파일이므로 수동 편집하지 말고 항상 `uv lock`으로 재생성한다.

### 6. editable install / 로컬 패키지

**증상:** `pip install -e .` 처럼 개발 모드 설치가 필요

**해결:**
```bash
# build-system이 선언된 패키지 프로젝트를 기본 editable로 설치
uv sync

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

**해결:** 빌드 설정은 프로젝트의 실제 패키지 이름·레이아웃에 맞춰 검토한다.

```toml
# src layout인 경우 pyproject.toml에 빌드 시스템 설정 확인
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

```

빌드 시스템을 선언한 뒤:

```bash
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
         uv export --format requirements-txt --no-dev --no-emit-project -o requirements.txt

Phase 3: 완전 전환
         requirements.txt 제거
         CI/CD를 uv sync로 변경
         팀 전원 uv 사용
```

## 참고 자료

- [uv 공식 마이그레이션 가이드](https://docs.astral.sh/uv/guides/migration/pip-to-project/)
- [uv pip 호환 인터페이스](https://docs.astral.sh/uv/pip/compatibility/)

## 관련 문서

- [uv 패키지 매니저 개요](./uv-package-manager.md)

## 현재 검토 근거와 한계

확인일 **2026-10-04**, 로컬 CLI **uv 0.12.13**. [pip → project 공식 절차](https://docs.astral.sh/uv/guides/migration/pip-to-project/), [인덱스 선택](https://docs.astral.sh/uv/concepts/indexes/), [editable·exact sync](https://docs.astral.sh/uv/concepts/projects/sync/), [Docker의 소스 복사·실행 조건](https://docs.astral.sh/uv/guides/integration/docker/)를 대조했다. 실제 사내 인덱스·패키지의 빌드 및 Docker·Actions 실행은 미확인이다. 두 uv 문서는 역할이 다르다. 개요는 새 작업과 명령 개념을, 이 문서는 기존 환경의 보존·전환·회귀 검증을 담당한다. 공통 기본 설치 설명의 완전 통합은 Claude 협의 부재로 보류했다.
