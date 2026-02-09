---
tags: [terminal, zsh, navigation, ls, cd, pwd]
level: beginner
last_updated: 2026-02-09
status: complete
---

# 파일 시스템 탐색 (Navigation & Listing)

> 터미널에서 현재 위치를 확인하고, 디렉토리를 이동하고, 파일 목록을 조회하는 기본 명령어

## 왜 필요한가? (Why)

- 터미널 작업의 가장 기본은 "지금 어디에 있는지"와 "여기에 뭐가 있는지"를 아는 것
- GUI 파일 탐색기의 모든 기능을 CLI에서 더 빠르게 수행 가능
- 스크립트에서 경로 기반 작업의 기초

## 핵심 개념 (What)

### 경로(Path) 체계

| 구분 | 설명 | 예시 |
|------|------|------|
| 절대 경로(Absolute) | 루트(`/`)부터 시작하는 전체 경로 | `/Users/daeyoung/projects` |
| 상대 경로(Relative) | 현재 위치 기준 경로 | `./src/main.py`, `../config` |

### 특수 경로 기호

| 기호 | 의미 | 예시 |
|------|------|------|
| `~` | 홈 디렉토리 (`$HOME`) | `cd ~` → `/Users/daeyoung` |
| `.` | 현재 디렉토리 | `./run.sh` (현재 폴더의 스크립트 실행) |
| `..` | 상위 디렉토리 | `cd ../..` (두 단계 위로) |
| `/` | 루트 디렉토리 | `cd /` |
| `-` | 이전 디렉토리 (cd 전용) | `cd -` (직전 위치로 이동) |

## 어떻게 사용하는가? (How)

### `pwd` — 현재 위치 확인 (Print Working Directory)

```bash
pwd
# /Users/daeyoung/Codes/pm_notes
```

스크립트에서 현재 경로를 변수에 저장할 때 유용하다:

```bash
CURRENT_DIR=$(pwd)
echo "작업 디렉토리: $CURRENT_DIR"
```

### `cd` — 디렉토리 이동 (Change Directory)

```bash
# 절대 경로로 이동
cd /Users/daeyoung/Codes

# 상대 경로로 이동
cd ./src
cd ../config

# 홈 디렉토리로 이동
cd ~
cd          # 인자 없이도 홈으로 이동

# 직전 디렉토리로 복귀
cd -
```

**실무 예제**: 프로젝트 루트와 하위 폴더 사이를 왔다갔다 할 때

```bash
cd ~/Codes/pm_notes         # 프로젝트 폴더로 이동
cd web-development/python   # 하위 폴더로 이동
cd -                        # pm_notes로 복귀
```

### `ls` — 파일 목록 조회 (List)

기본 사용법:

```bash
ls              # 현재 디렉토리 파일 목록
ls /etc         # 특정 디렉토리 파일 목록
ls *.md         # 패턴 매칭 (모든 .md 파일)
```

#### 주요 옵션 비교표

| 옵션 | 설명 | 예시 |
|------|------|------|
| `-l` | 상세 정보 (권한, 크기, 날짜) | `ls -l` |
| `-a` | 숨김 파일 포함 (`.`으로 시작하는 파일) | `ls -a` |
| `-la` | 숨김 파일 + 상세 정보 (가장 많이 사용) | `ls -la` |
| `-lh` | 상세 정보 + 사람이 읽기 쉬운 크기 (KB, MB) | `ls -lh` |
| `-lt` | 상세 정보 + 수정 시간순 정렬 (최신 먼저) | `ls -lt` |
| `-lS` | 상세 정보 + 파일 크기순 정렬 (큰 것 먼저) | `ls -lS` |
| `-R` | 재귀적으로 하위 디렉토리까지 표시 | `ls -R` |
| `-1` | 한 줄에 하나씩 출력 | `ls -1` |

**`ls -la` 출력 읽는 법**:

```
drwxr-xr-x  5 daeyoung staff  160 Feb  9 10:30 src
-rw-r--r--  1 daeyoung staff 1234 Feb  9 09:15 README.md
│└─┬──┘└┬─┘    └──┬───┘ └─┬─┘ └─┬┘ └────┬─────┘ └───┬───┘
│  │    │         │       │     │       │           └─ 파일명
│  │    │         │       │     │       └─ 수정 일시
│  │    │         │       │     └─ 파일 크기 (바이트)
│  │    │         │       └─ 그룹
│  │    │         └─ 소유자
│  │    └─ 기타(other) 권한
│  └─ 그룹(group) 권한
└─ d=디렉토리, -=파일, l=심볼릭 링크
```

**실무 예제**: 최근 수정된 파일 확인

```bash
ls -lt | head -10    # 최근 수정된 파일 상위 10개
ls -la .*            # 숨김 파일만 보기
ls -lhS *.log        # 로그 파일을 크기순으로 보기
```

### `tree` — 디렉토리 트리 구조 표시

macOS에는 기본 설치되어 있지 않으므로 Homebrew로 설치한다:

```bash
brew install tree
```

```bash
tree                    # 현재 디렉토리 트리 출력
tree -L 2               # 깊이 2단계까지만
tree -L 2 -d            # 디렉토리만 표시
tree -I 'node_modules'  # 특정 패턴 제외
tree -I 'node_modules|.git|__pycache__'  # 여러 패턴 제외
```

**실무 예제**: 프로젝트 구조 파악

```bash
# README에 넣을 프로젝트 구조 확인
tree -L 3 -I 'node_modules|.git|__pycache__|.venv'
```

### `which` — 명령어 위치 확인

```bash
which python
# /Users/daeyoung/.pyenv/shims/python

which node
# /usr/local/bin/node

which code
# /usr/local/bin/code
```

여러 버전이 설치되어 있을 때 어떤 바이너리가 실행되는지 확인할 때 유용하다:

```bash
which -a python  # python 이름의 모든 실행 파일 경로 표시
```

## zsh 팁

### AUTO_CD

zsh에서는 디렉토리 이름만 입력해도 이동할 수 있다 (AUTO_CD 옵션):

```bash
# cd 없이 디렉토리 이동
~/Codes/pm_notes    # cd ~/Codes/pm_notes 와 동일
..                  # cd .. 와 동일
```

`~/.zshrc`에서 활성화:

```bash
setopt AUTO_CD
```

### 탭 완성 (Tab Completion)

```bash
cd ~/Co<Tab>          # ~/Codes/ 로 자동 완성
ls READ<Tab>          # README.md 로 자동 완성
```

### 와일드카드 Globbing

```bash
ls **/*.md            # 모든 하위 디렉토리의 .md 파일 (zsh 재귀 글로빙)
ls *.{py,js}          # .py 또는 .js 파일
ls file?.txt          # file1.txt, fileA.txt 등 (? = 한 글자)
```

## 참고 자료 (References)

- [Zsh 공식 문서](https://zsh.sourceforge.io/Doc/)
- `man ls`, `man cd`, `man pwd`

## 관련 문서

- [터미널 명령어 목차](./README.md)
- 다음: [파일 관리](./file-management.md)
