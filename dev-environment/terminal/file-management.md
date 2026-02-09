---
tags: [terminal, zsh, file-management, cp, mv, rm, mkdir, ln]
level: beginner
last_updated: 2026-02-09
status: complete
---

# 파일 관리 (File Management)

> 파일과 디렉토리를 생성, 복사, 이동, 삭제하는 명령어

## 왜 필요한가? (Why)

- 파일 시스템 조작은 모든 개발 작업의 기반
- 프로젝트 구조 설정, 배포 파일 준비, 백업 등 일상적으로 사용
- 스크립트에서 자동화할 때 필수

## 핵심 개념 (What)

### 재귀 옵션 (`-r`, `-R`)

많은 명령어에서 `-r` (recursive) 옵션은 디렉토리와 그 안의 모든 내용물을 포함한다는 의미이다. 디렉토리를 대상으로 할 때 거의 필수적으로 사용한다.

```bash
cp -r src/ backup/    # src 폴더 전체를 복사
rm -r old_project/    # old_project 폴더 전체를 삭제
```

## 어떻게 사용하는가? (How)

### `touch` — 빈 파일 생성 / 타임스탬프 갱신

```bash
touch newfile.txt           # 빈 파일 생성 (이미 존재하면 수정 시간만 갱신)
touch file1.txt file2.txt   # 여러 파일 동시 생성
```

**실무 예제**: 프로젝트 초기 파일 세팅

```bash
touch .gitignore .env.example README.md
```

### `mkdir` — 디렉토리 생성 (Make Directory)

```bash
mkdir new_folder             # 디렉토리 생성
mkdir -p path/to/deep/dir    # 중첩 디렉토리 한 번에 생성 (-p: parents)
mkdir -p src/{components,utils,hooks}  # 여러 하위 폴더 동시 생성
```

`-p` 옵션은 중간 경로가 없어도 자동으로 생성하고, 이미 존재해도 에러를 내지 않는다.

**실무 예제**: 프로젝트 구조 생성

```bash
mkdir -p my-app/{src/{components,pages,utils},public,tests}
```

### `cp` — 파일/디렉토리 복사 (Copy)

```bash
cp source.txt dest.txt           # 파일 복사
cp source.txt ~/backup/          # 다른 위치로 복사
cp -r src/ src_backup/           # 디렉토리 전체 복사 (-r 필수)
cp -i important.txt backup.txt   # 덮어쓰기 전 확인 (-i: interactive)
cp *.md docs/                    # 패턴 매칭으로 여러 파일 복사
```

#### 주요 옵션

| 옵션 | 설명 |
|------|------|
| `-r` | 디렉토리 재귀 복사 (하위 내용 포함) |
| `-i` | 덮어쓰기 전 확인 |
| `-v` | 복사 과정 출력 (verbose) |
| `-n` | 이미 존재하는 파일은 덮어쓰지 않음 |

### `mv` — 파일 이동 / 이름 변경 (Move)

```bash
mv old.txt new.txt               # 이름 변경
mv file.txt ~/Documents/         # 파일 이동
mv -i source.txt dest.txt        # 덮어쓰기 전 확인
mv *.log logs/                   # 여러 파일 이동
mv old_dir/ new_dir/             # 디렉토리 이름 변경 (-r 불필요)
```

`mv`는 복사+삭제가 아니라 파일 시스템의 참조만 변경하므로, 같은 파일 시스템 내에서는 파일 크기에 관계없이 즉시 완료된다.

### `rm` — 파일/디렉토리 삭제 (Remove)

```bash
rm file.txt                  # 파일 삭제
rm -r directory/             # 디렉토리 전체 삭제
rm -i important.txt          # 삭제 전 확인
rm *.tmp                     # 패턴 매칭 삭제
```

#### 주요 옵션

| 옵션 | 설명 |
|------|------|
| `-r` | 디렉토리 재귀 삭제 |
| `-i` | 삭제 전 확인 |
| `-f` | 경고 없이 강제 삭제 |
| `-v` | 삭제 과정 출력 |

#### `rm` 안전 가이드

`rm`은 **휴지통으로 보내는 것이 아니라 즉시 영구 삭제**한다. 복구가 불가능하므로 주의가 필요하다.

**위험한 패턴**:

```bash
# 절대 실행하지 말 것
rm -rf /                     # 시스템 전체 삭제
rm -rf ~                     # 홈 디렉토리 전체 삭제
rm -rf *                     # 현재 폴더 전체 삭제 (경로 확인 필수)
```

**안전한 대안**: `trash` 명령어 사용

```bash
brew install trash
trash file.txt               # 파일을 macOS 휴지통으로 이동
```

`~/.zshrc`에 alias 설정으로 안전하게 사용:

```bash
alias rm='rm -i'             # rm 실행 시 항상 확인 프롬프트
```

### `rmdir` — 빈 디렉토리 삭제

```bash
rmdir empty_folder           # 비어 있는 폴더만 삭제 가능
```

내용이 있으면 에러가 발생하므로 `rm -r`보다 안전하다. 실수로 데이터를 날릴 걱정이 없다.

### `ln` — 링크 생성 (Link)

#### 하드 링크 vs 심볼릭 링크 비교

| 구분 | 하드 링크 | 심볼릭 링크 (소프트 링크) |
|------|-----------|---------------------------|
| 생성 | `ln source link` | `ln -s source link` |
| 원본 삭제 시 | 여전히 접근 가능 | 깨진 링크(broken link)가 됨 |
| 디렉토리 링크 | 불가 | 가능 |
| 다른 파일 시스템 | 불가 | 가능 |
| 실무 사용 빈도 | 드묾 | 매우 자주 사용 |

```bash
# 심볼릭 링크 (가장 많이 사용)
ln -s /usr/local/bin/python3 /usr/local/bin/python

# 심볼릭 링크 확인
ls -la /usr/local/bin/python
# lrwxr-xr-x  1 root wheel  24 Jan 10 python -> /usr/local/bin/python3
```

**실무 예제**: 설정 파일 심볼릭 링크

```bash
# dotfiles 관리: 실제 파일은 git repo에 두고 홈 디렉토리에 링크
ln -s ~/Codes/dotfiles/.zshrc ~/.zshrc
ln -s ~/Codes/dotfiles/.gitconfig ~/.gitconfig
```

## 참고 자료 (References)

- `man cp`, `man mv`, `man rm`, `man ln`, `man mkdir`
- [trash CLI - GitHub](https://github.com/ali-rantakari/trash)

## 관련 문서

- 이전: [파일 시스템 탐색](./navigation-and-listing.md)
- 다음: [파일 내용 조회](./file-content.md)
- [터미널 명령어 목차](./README.md)
