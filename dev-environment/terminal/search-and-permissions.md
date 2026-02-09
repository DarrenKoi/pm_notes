---
tags: [terminal, zsh, grep, find, chmod, chown, xargs, permissions]
level: beginner
last_updated: 2026-02-09
status: complete
---

# 검색과 권한 (Search & Permissions)

> 파일/텍스트 검색, 파일 권한 관리, 명령어 조합을 위한 고급 명령어

## 왜 필요한가? (Why)

- 대규모 코드베이스에서 특정 패턴을 빠르게 찾는 능력은 디버깅과 코드 이해의 핵심
- 파일 권한 이해는 서버 운영, 보안 설정, 배포 시 필수
- `xargs`와 파이프 조합은 복잡한 일괄 작업을 자동화하는 강력한 도구

## 핵심 개념 (What)

### 파일 권한 체계 (rwx)

Unix/Linux 파일 시스템의 권한은 세 그룹으로 나뉜다:

```
-rwxr-xr--
│└┬┘└┬┘└┬┘
│ │  │  └─ 기타(Other): r-- (읽기만 가능)
│ │  └──── 그룹(Group): r-x (읽기+실행)
│ └─────── 소유자(Owner/User): rwx (모든 권한)
└───────── 파일 유형: - (일반 파일)
```

| 기호 | 권한 | 숫자 값 |
|------|------|---------|
| `r` | 읽기 (Read) | 4 |
| `w` | 쓰기 (Write) | 2 |
| `x` | 실행 (Execute) | 1 |
| `-` | 없음 | 0 |

#### 자주 사용하는 숫자 표기법

| 숫자 | 의미 | 사용 예시 |
|------|------|-----------|
| `755` | 소유자: rwx, 그룹: r-x, 기타: r-x | 실행 파일, 디렉토리 |
| `644` | 소유자: rw-, 그룹: r--, 기타: r-- | 일반 파일 |
| `700` | 소유자: rwx, 그룹: ---, 기타: --- | 비공개 스크립트 |
| `600` | 소유자: rw-, 그룹: ---, 기타: --- | 비밀 키 파일 (SSH key 등) |
| `777` | 모두: rwx | (보안상 가급적 사용 금지) |

### 정규 표현식 기초 메타문자

`grep`과 `find`에서 사용하는 기본 정규 표현식:

| 메타문자 | 의미 | 예시 |
|----------|------|------|
| `.` | 임의의 한 글자 | `a.c` → abc, aXc |
| `*` | 앞 문자 0회 이상 반복 | `ab*c` → ac, abc, abbc |
| `+` | 앞 문자 1회 이상 반복 (확장) | `ab+c` → abc, abbc |
| `^` | 줄의 시작 | `^Error` → Error로 시작하는 줄 |
| `$` | 줄의 끝 | `\.py$` → .py로 끝나는 줄 |
| `[]` | 문자 클래스 | `[0-9]` → 숫자 한 자리 |
| `\` | 이스케이프 | `\.` → 실제 점 문자 |

## 어떻게 사용하는가? (How)

### `grep` — 텍스트 검색 (Global Regular Expression Print)

기본 사용법:

```bash
grep "pattern" file.txt          # 파일에서 패턴 검색
grep "pattern" *.log             # 여러 파일에서 검색
```

#### 주요 옵션

| 옵션 | 설명 | 예시 |
|------|------|------|
| `-r` | 디렉토리 재귀 검색 | `grep -r "TODO" ./src` |
| `-n` | 줄 번호 표시 | `grep -n "error" app.log` |
| `-i` | 대소문자 무시 | `grep -i "error" app.log` |
| `-l` | 매칭된 파일 이름만 출력 | `grep -rl "import" ./src` |
| `-c` | 매칭 횟수만 출력 | `grep -c "ERROR" app.log` |
| `-v` | 매칭되지 않는 줄 출력 (반전) | `grep -v "^#" config.txt` |
| `-A n` | 매칭 후 n줄 함께 출력 (After) | `grep -A 3 "ERROR" app.log` |
| `-B n` | 매칭 전 n줄 함께 출력 (Before) | `grep -B 2 "ERROR" app.log` |
| `-E` | 확장 정규식 사용 (= `egrep`) | `grep -E "error|warn" app.log` |
| `--include` | 특정 파일 패턴만 대상 | `grep -r --include="*.py" "def "` |

**실무 예제**: 코드베이스 검색

```bash
# Python 파일에서 함수 정의 찾기
grep -rn --include="*.py" "def " ./src

# TODO 주석 찾기 (대소문자 무시)
grep -rni "todo\|fixme\|hack" ./src

# 주석이 아닌 줄에서 특정 import 찾기
grep -rn --include="*.py" "^from fastapi" ./src

# 특정 패턴 전후 컨텍스트와 함께 보기
grep -rn -C 3 "raise Exception" ./src
```

### `find` — 파일 검색

기본 구문: `find <경로> <조건> [<동작>]`

```bash
find . -name "*.py"              # 현재 디렉토리에서 .py 파일 찾기
find . -type f -name "*.md"      # 파일만 찾기 (디렉토리 제외)
find . -type d -name "src"       # 디렉토리만 찾기
```

#### 주요 필터

| 필터 | 설명 | 예시 |
|------|------|------|
| `-name` | 파일명 매칭 (대소문자 구분) | `-name "*.txt"` |
| `-iname` | 파일명 매칭 (대소문자 무시) | `-iname "readme*"` |
| `-type f` | 일반 파일만 | `-type f` |
| `-type d` | 디렉토리만 | `-type d` |
| `-size +10M` | 10MB 초과 파일 | `-size +10M` |
| `-size -1k` | 1KB 미만 파일 | `-size -1k` |
| `-mtime -7` | 7일 이내 수정된 파일 | `-mtime -7` |
| `-mtime +30` | 30일 이전 수정된 파일 | `-mtime +30` |
| `-empty` | 빈 파일/디렉토리 | `-empty` |
| `-maxdepth n` | 검색 깊이 제한 | `-maxdepth 2` |

**실무 예제**:

```bash
# 최근 24시간 내 수정된 Python 파일
find . -name "*.py" -mtime -1

# 10MB 이상의 큰 파일 찾기
find . -type f -size +10M

# 빈 디렉토리 찾기
find . -type d -empty

# node_modules 제외하고 검색
find . -name "*.js" -not -path "*/node_modules/*"

# 찾은 파일에 대해 명령 실행 (-exec)
find . -name "*.tmp" -exec rm {} \;
find . -name "*.py" -exec grep -l "import os" {} \;
```

### `chmod` — 파일 권한 변경 (Change Mode)

```bash
# 숫자 표기법
chmod 755 script.sh              # rwxr-xr-x (실행 가능하게)
chmod 644 config.txt             # rw-r--r-- (일반 파일)
chmod 600 ~/.ssh/id_rsa          # rw------- (SSH 키, 소유자만)

# 기호 표기법
chmod +x script.sh               # 실행 권한 추가
chmod u+w file.txt               # 소유자에게 쓰기 권한 추가
chmod go-w file.txt              # 그룹과 기타에서 쓰기 권한 제거
chmod -R 755 directory/          # 재귀적으로 권한 변경
```

기호 표기법 문법: `[ugoa][+-=][rwx]`

| 대상 | 의미 |
|------|------|
| `u` | 소유자 (user) |
| `g` | 그룹 (group) |
| `o` | 기타 (other) |
| `a` | 모두 (all) |

**실무 예제**: 스크립트 파일을 실행 가능하게 만들기

```bash
chmod +x deploy.sh
./deploy.sh                      # 이제 직접 실행 가능
```

### `chown` — 파일 소유자 변경 (Change Owner)

```bash
chown user file.txt              # 소유자 변경
chown user:group file.txt        # 소유자와 그룹 동시 변경
chown -R user:group directory/   # 재귀적으로 변경
```

> macOS에서 `chown`은 보통 `sudo`가 필요하다:
> ```bash
> sudo chown daeyoung:staff project/
> ```

### `xargs` — 표준 입력을 명령어 인자로 변환

`xargs`는 파이프로 받은 입력을 다른 명령어의 인자로 전달한다. `find`와 조합하면 강력한 일괄 처리가 가능하다.

기본 구문:

```bash
명령어1 | xargs 명령어2
# 명령어1의 출력 각 줄을 명령어2의 인자로 전달
```

```bash
# 찾은 파일 삭제
find . -name "*.tmp" | xargs rm

# 파일 목록에 대해 grep 실행
find . -name "*.py" | xargs grep "import os"

# 파일명에 공백이 있을 때 안전하게 처리
find . -name "*.txt" -print0 | xargs -0 wc -l

# 한 번에 하나씩 실행 (-I로 위치 지정)
find . -name "*.bak" | xargs -I {} mv {} {}.old
```

#### 주요 옵션

| 옵션 | 설명 | 예시 |
|------|------|------|
| `-I {}` | `{}`를 입력값으로 치환 | `xargs -I {} cp {} backup/` |
| `-0` | null 문자 구분 (`-print0`과 함께) | 공백 있는 파일명 처리 |
| `-n 1` | 한 번에 하나의 인자만 전달 | 각 파일별 개별 실행 |
| `-p` | 실행 전 확인 프롬프트 | 안전한 일괄 작업 |

**실무 파이프라인 예제**:

```bash
# 프로젝트에서 Python 파일의 총 라인 수
find . -name "*.py" -not -path "./.venv/*" | xargs wc -l

# 특정 패턴을 포함하는 파일 목록을 찾아 편집기로 열기
grep -rl "deprecated" ./src | xargs code

# Git에서 추적하지 않는 파일 삭제
git ls-files --others --exclude-standard | xargs rm

# 여러 파일의 첫 줄(헤더) 확인
find . -name "*.csv" | xargs -I {} head -1 {}
```

## macOS(BSD) vs Linux(GNU) 차이점

macOS의 명령어는 BSD 계열이고, 대부분의 Linux는 GNU 계열이다. 주요 차이점:

| 명령어 | macOS (BSD) | Linux (GNU) |
|--------|-------------|-------------|
| `grep` | 기본 설치, PCRE 미지원 | `-P` 옵션으로 PCRE 지원 |
| `sed` | `sed -i '' 's/a/b/'` (빈 문자열 필수) | `sed -i 's/a/b/'` |
| `find` | `-delete` 옵션 위치 중요 | 동일 |
| `xargs` | `-I` 뒤에 replace-str 필수 | 동일 |
| `date` | `date -v+1d` (상대 날짜) | `date -d "+1 day"` |
| `readlink` | `readlink` (절대 경로 미지원) | `readlink -f` |

GNU 버전이 필요하면 Homebrew로 설치할 수 있다:

```bash
brew install coreutils    # gls, gcp, gmv 등 (g 접두사)
brew install gnu-sed      # gsed
brew install grep         # ggrep (GNU grep)
```

## 참고 자료 (References)

- `man grep`, `man find`, `man chmod`, `man chown`, `man xargs`
- [Regular Expressions 101](https://regex101.com/) - 정규 표현식 테스트 도구

## 관련 문서

- 이전: [파일 내용 조회](./file-content.md)
- [터미널 명령어 목차](./README.md)
