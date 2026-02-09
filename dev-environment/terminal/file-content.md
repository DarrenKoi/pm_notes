---
tags: [terminal, zsh, cat, head, tail, less, pipe, redirect]
level: beginner
last_updated: 2026-02-09
status: complete
---

# 파일 내용 조회 (File Content)

> 파일 내용을 확인하고, 정렬/필터링하고, 비교하는 명령어

## 왜 필요한가? (Why)

- 로그 파일 분석, 설정 파일 확인 등 파일 내용을 빠르게 확인하는 작업은 매일 발생
- 파이프(`|`)와 리다이렉션(`>`)을 활용하면 복잡한 데이터 처리를 한 줄로 수행 가능
- 서버에서 실시간 로그 모니터링 시 필수

## 핵심 개념 (What)

### 표준 입출력 (Standard I/O)

| 스트림 | 이름 | 설명 | 번호 |
|--------|------|------|------|
| stdin | 표준 입력 | 키보드 입력 (기본값) | 0 |
| stdout | 표준 출력 | 화면 출력 (기본값) | 1 |
| stderr | 표준 에러 | 에러 메시지 출력 | 2 |

### 파이프 (`|`)

앞 명령어의 stdout을 뒤 명령어의 stdin으로 연결한다:

```bash
명령어1 | 명령어2 | 명령어3
# 명령어1의 출력 → 명령어2의 입력 → 명령어3의 입력
```

### 리다이렉션 (`>`, `>>`)

| 기호 | 설명 | 예시 |
|------|------|------|
| `>` | 출력을 파일에 쓰기 (덮어쓰기) | `echo "hello" > file.txt` |
| `>>` | 출력을 파일에 추가 (append) | `echo "world" >> file.txt` |
| `2>` | 에러 출력을 파일에 쓰기 | `cmd 2> error.log` |
| `2>&1` | 에러를 표준 출력에 합치기 | `cmd > all.log 2>&1` |
| `<` | 파일을 입력으로 사용 | `sort < names.txt` |

## 어떻게 사용하는가? (How)

### `cat` — 파일 내용 출력 (Concatenate)

```bash
cat file.txt                 # 파일 전체 출력
cat -n file.txt              # 줄 번호와 함께 출력
cat file1.txt file2.txt      # 여러 파일 연결 출력
cat file1.txt file2.txt > merged.txt  # 파일 합치기
```

짧은 파일 확인에 적합. 긴 파일은 `less`를 사용하는 것이 좋다.

### `head` — 파일 앞부분 출력

```bash
head file.txt                # 처음 10줄 출력 (기본값)
head -n 20 file.txt          # 처음 20줄 출력
head -n 1 *.csv              # 각 CSV 파일의 헤더(1줄) 확인
```

**실무 예제**: CSV 파일 구조 빠르게 확인

```bash
head -n 5 data.csv           # 헤더 + 샘플 데이터 확인
```

### `tail` — 파일 뒷부분 출력

```bash
tail file.txt                # 마지막 10줄 출력 (기본값)
tail -n 20 file.txt          # 마지막 20줄 출력
tail -n +5 file.txt          # 5번째 줄부터 끝까지 출력
```

#### `tail -f` — 실시간 파일 모니터링

```bash
tail -f app.log              # 파일에 새로 추가되는 내용을 실시간으로 출력
tail -f app.log | grep ERROR # 실시간으로 ERROR만 필터링
```

`Ctrl + C`로 종료. 서버 로그 모니터링 시 가장 많이 사용하는 패턴이다.

### `less` — 페이지 단위 파일 뷰어

```bash
less file.txt                # 파일을 페이지 단위로 탐색
less +F file.txt             # tail -f처럼 실시간 모니터링 모드로 시작
```

#### `less` 조작 키

| 키 | 동작 |
|----|------|
| `Space` / `f` | 다음 페이지 |
| `b` | 이전 페이지 |
| `j` / `↓` | 한 줄 아래 |
| `k` / `↑` | 한 줄 위 |
| `g` | 파일 맨 위로 |
| `G` | 파일 맨 아래로 |
| `/패턴` | 아래 방향 검색 |
| `?패턴` | 위 방향 검색 |
| `n` | 다음 검색 결과 |
| `N` | 이전 검색 결과 |
| `q` | 종료 |
| `&패턴` | 패턴과 매칭되는 줄만 표시 |

### `wc` — 글자/단어/줄 수 세기 (Word Count)

```bash
wc file.txt                  # 줄 수, 단어 수, 바이트 수 출력
wc -l file.txt               # 줄 수만 출력
wc -w file.txt               # 단어 수만 출력
wc -l *.py                   # 각 Python 파일의 줄 수
```

**실무 예제**: 코드 라인 수 확인

```bash
find . -name "*.py" | xargs wc -l | tail -1    # Python 코드 총 라인 수
```

### `sort` — 정렬

```bash
sort file.txt                # 알파벳순 정렬
sort -n numbers.txt          # 숫자순 정렬
sort -r file.txt             # 역순 정렬
sort -k2 data.txt            # 2번째 필드 기준 정렬
sort -t',' -k3 -n data.csv   # CSV에서 3번째 컬럼 기준 숫자 정렬
sort -u file.txt             # 정렬 + 중복 제거
```

### `uniq` — 연속 중복 제거

```bash
sort file.txt | uniq          # 정렬 후 중복 제거 (sort 필수)
sort file.txt | uniq -c       # 중복 횟수 카운트
sort file.txt | uniq -d       # 중복된 줄만 출력
```

> `uniq`은 **연속된** 중복만 제거하므로, 반드시 `sort`와 함께 사용해야 전체 중복을 제거할 수 있다.

### `diff` — 파일 비교

```bash
diff file1.txt file2.txt         # 두 파일의 차이점 출력
diff -u file1.txt file2.txt      # unified 형식 (git diff와 유사)
diff -y file1.txt file2.txt      # 나란히 비교 (side-by-side)
diff -r dir1/ dir2/              # 디렉토리 단위 비교
```

`-u` (unified) 형식이 가장 읽기 쉽고, `git diff`와 동일한 형식이다:

```diff
--- file1.txt
+++ file2.txt
@@ -1,3 +1,3 @@
 first line
-old content
+new content
 third line
```

## 실무 파이프라인 예제

### 로그 파일 분석

```bash
# 오늘 발생한 ERROR 로그의 종류별 횟수
grep "ERROR" app.log | grep "2026-02-09" | sort | uniq -c | sort -rn

# 가장 많이 접속한 IP 상위 10개
cat access.log | awk '{print $1}' | sort | uniq -c | sort -rn | head -10
```

### 데이터 처리

```bash
# CSV 파일에서 특정 컬럼 추출 후 정렬
cat data.csv | cut -d',' -f2 | sort -u > unique_names.txt

# 두 파일의 공통 줄 찾기 (파일이 정렬되어 있어야 함)
comm -12 <(sort file1.txt) <(sort file2.txt)
```

### 파일 크기 확인

```bash
# 현재 디렉토리에서 파일 크기 순으로 정렬
ls -lhS | head -20

# 큰 파일 찾기
du -sh * | sort -rh | head -10
```

## 참고 자료 (References)

- `man cat`, `man head`, `man tail`, `man less`, `man sort`, `man uniq`, `man diff`

## 관련 문서

- 이전: [파일 관리](./file-management.md)
- 다음: [검색과 권한](./search-and-permissions.md)
- [터미널 명령어 목차](./README.md)
