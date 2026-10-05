---
reviewed_on: 2026-10-04
review_status: partial
document_type: learning
tags: [terminal, zsh, macos, cli]
level: beginner
last_updated: 2026-02-09
category_major: "개발 환경"
category_middle: "터미널·협업"
category_minor: "파일·셸 명령"
note_kind: "목차"
classified_on: "2026-10-05"
---

# Mac 터미널(zsh) 필수 명령어

> [!info] 2026-10-04 검토 범위
> macOS 로컬 매뉴얼과 임시 파일 실습으로 핵심 명령의 조건을 확인했다. BSD/GNU 옵션과 설치 도구 버전은 각각의 `man`·`--help`를 확인한다. 운영 파일 삭제·권한 변경은 실행하지 않았다.

> macOS 터미널에서 자주 사용하는 기본 명령어를 체계적으로 정리한 시리즈

## 왜 필요한가? (Why)

- GUI 없이 파일 시스템을 빠르게 조작하는 능력은 개발자의 핵심 역량
- 서버 환경(SSH)에서는 터미널만 사용 가능하므로 CLI 숙달이 필수
- 스크립트 작성, CI/CD 파이프라인, 자동화의 기초

## 문서 목차

| # | 문서 | 설명 | 주요 명령어 |
|---|------|------|-------------|
| 1 | [파일 시스템 탐색](./navigation-and-listing.md) | 디렉토리 이동 및 파일 목록 조회 | `pwd`, `cd`, `ls`, `tree`, `which` |
| 2 | [파일 관리](./file-management.md) | 파일/디렉토리 생성, 복사, 이동, 삭제 | `touch`, `mkdir`, `cp`, `mv`, `rm`, `ln` |
| 3 | [파일 내용 조회](./file-content.md) | 파일 내용 확인, 정렬, 비교 | `cat`, `head`, `tail`, `less`, `wc`, `sort`, `uniq`, `diff` |
| 4 | [검색과 권한](./search-and-permissions.md) | 파일 검색, 텍스트 검색, 권한 관리 | `grep`, `find`, `chmod`, `chown`, `xargs` |

## 학습 순서

```
1. 파일 시스템 탐색 (Navigation)  ← 여기서 시작
   ↓
2. 파일 관리 (File Management)
   ↓
3. 파일 내용 조회 (File Content)
   ↓
4. 검색과 권한 (Search & Permissions)
```

## 도움말 활용 팁

모르는 명령어가 있으면 `man` (manual) 명령어로 공식 매뉴얼을 확인할 수 있다:

```bash
man ls        # ls 명령어의 전체 매뉴얼
man -k search # "search" 키워드로 관련 매뉴얼 검색
```

`man` 페이지는 `less`로 열리므로, `q`로 종료하고 `/키워드`로 검색할 수 있다.

## 관련 문서

- [개발 환경 인덱스](../README.md)
