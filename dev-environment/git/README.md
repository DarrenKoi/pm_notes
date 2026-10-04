---
tags: [git, gitlab, collaboration]
level: beginner
last_updated: 2026-09-30
---

# 회사에서 Git과 GitLab으로 협업하기

> fetch로 최신 이력을 확인하고, 작업 브랜치에서 수정한 뒤 GitLab Merge Request로 검토받는 실무 입문 노트.

## 왜 필요한가

여러 사람이 같은 프로젝트를 수정하면, 내 PC의 코드와 회사 서버의 코드가 달라집니다. Git은 변경 이력을 기록하고 서로의 작업을 합치는 도구입니다. GitLab은 Git 저장소를 서버에 보관하며 이슈, 코드 리뷰, 자동 검사 등을 함께 제공하는 협업 서비스입니다.

처음에는 **최신 코드 확인 → 작업 브랜치 생성 → 수정과 커밋 → push → Merge Request → 리뷰와 병합** 흐름을 익히면 됩니다. 아래에서는 기본 브랜치를 `main`, 원격 저장소를 `origin`으로 가정합니다. 회사에서 `develop` 등을 사용한다면 실제 대상 브랜치로 바꿉니다. 회사의 GitLab 버전과 프로젝트 설정에 따라 메뉴와 병합 조건은 달라질 수 있습니다.

## 핵심 개념

### 파일을 수정하는 곳과 이력을 저장하는 곳

```text
내 PC
작업 폴더 ── git add ──> 스테이징 영역 ── git commit ──> 로컬 저장소
                                                        │
                                                     git push
                                                        ↓
                                                   GitLab 저장소
```

| 용어 | 의미 |
|------|------|
| 작업 폴더 또는 작업 트리 | 편집기로 수정하는 실제 파일 |
| 스테이징 영역 | 다음 커밋에 넣을 변경을 골라 놓는 곳 |
| 커밋(commit) | 선택한 변경을 저장한 이력 단위. 고유 ID가 있음 |
| 브랜치(branch) | 특정 커밋을 가리키는 이름. 새 커밋을 만들면 그 이름이 앞으로 이동함 |
| `HEAD` | 보통 현재 작업 중인 브랜치를 가리킴 |
| `origin` | clone할 때 일반적으로 붙는 원격 저장소의 별명 |
| Merge Request 또는 MR | 내 브랜치의 변경을 대상 브랜치에 반영해 달라는 GitLab의 검토 요청 |

`commit`은 내 PC에 이력을 저장합니다. `push`해야 서버에 전달됩니다. **push했다고 `main`에 반영되는 것은 아닙니다.** 작업 브랜치에 push했다면 GitLab의 해당 작업 브랜치만 갱신됩니다.

### main과 origin/main은 다르다

| 이름 | 위치와 역할 |
|------|-------------|
| `main` | 내 PC의 로컬 브랜치 |
| `origin/main` | 마지막으로 fetch한 서버 `main`의 상태를 기록하는 내 PC의 참조 |
| GitLab의 `main` | 서버에 있는 실제 브랜치. 동료가 변경할 수 있음 |

`origin/main`은 서버를 실시간으로 보는 이름이 아닙니다. fetch하기 전에는 오래된 상태일 수 있습니다. 또한 fetch로 `origin/main`이 갱신되어도 내 `main`은 그대로입니다. [공식 fetch 문서](https://git-scm.com/docs/git-fetch)

예를 들어 동료가 새 커밋 `C`를 GitLab에 병합했다면 다음과 같습니다.

```text
fetch 전
내 main:         A──B
내 origin/main:  A──B
GitLab main:     A──B──C

git fetch origin 실행 후
내 main:         A──B       ← 그대로
내 origin/main:  A──B──C    ← 서버 상태 갱신
GitLab main:     A──B──C
```

### fetch와 pull 비교

`fetch`는 서버의 커밋과 브랜치 정보를 가져옵니다. 일반적인 `git fetch origin`은 현재 브랜치와 작업 파일을 바꾸지 않습니다. 먼저 정보를 받아서 상황을 살펴볼 때 사용합니다.

```bash
git fetch origin
git status
git log --oneline --graph --decorate --all -15
```

`pull`은 가져온 이력을 **현재 브랜치에 반영하는 단계까지** 수행합니다. 옵션으로 방식을 명시하면 동작을 이해하기 쉽습니다. [공식 pull 문서](https://git-scm.com/docs/git-pull)

| 명령어 | 동작 |
|--------|------|
| `git fetch origin` | 원격 정보를 가져옴 |
| `git pull --ff-only origin main` | 가져온 서버 `main`으로 현재 브랜치를 앞으로 이동할 수 있을 때만 갱신 |
| `git pull --no-rebase origin main` | 가져온 서버 `main`을 현재 브랜치에 merge |
| `git pull --rebase origin main` | 가져온 서버 `main` 위로 현재 브랜치의 커밋을 rebase |

**어느 브랜치에서 실행하는지가 중요합니다.** 작업 브랜치에서 `git pull --rebase origin main`을 실행하면 작업 브랜치가 바뀝니다. 로컬 `main`을 갱신하는 명령이 아닙니다. 인자 없는 `git pull`은 보통 현재 브랜치의 upstream을 사용하므로, 작업 브랜치에서는 그 원격 작업 브랜치를 가져옵니다.

처음에는 `fetch`와 `merge` 또는 `rebase`를 나누어 실행하면 두 단계를 눈으로 확인할 수 있습니다.

### merge는 갈라진 이력을 합친다

동료는 `main`에 `C`를 추가했고 나는 작업 브랜치에 `D`, `E`를 추가했다고 가정합니다.

```text
       C          ← origin/main
      /
A──B
      \
       D──E       ← feature/report
```

작업 브랜치에서 다음을 실행합니다. 먼저 커밋하거나 변경을 보관하여 `git status`가 깨끗한 상태여야 합니다.

```bash
git switch feature/report
git fetch origin
git merge origin/main
```

```text
A──B──C           ← origin/main
    \  \
     D──E──M      ← feature/report
```

`M`은 두 이력을 연결하는 병합 커밋입니다. 기존 `D`, `E`는 유지됩니다. **현재 작업 브랜치가 갱신되며 `origin/main`이나 서버 `main`은 바뀌지 않습니다.** [공식 merge 문서](https://git-scm.com/docs/git-merge)

항상 병합 커밋이 생기는 것은 아닙니다. 이력이 갈라지지 않았다면 브랜치 이름만 최신 커밋으로 이동할 수 있는데, 이를 fast-forward라고 합니다. `--ff-only`는 이런 이동만 허용하고, 이력이 갈라졌다면 중단합니다.

### rebase는 내 커밋을 새 출발점 위에 다시 적용한다

앞의 같은 출발 상태에서 다음을 실행합니다. merge 예제를 실행한 뒤 이어서 하는 것이 아니라, **두 방법 중 하나를 선택하는 예제**입니다.

```bash
git switch feature/report
git fetch origin
git rebase origin/main
```

```text
A──B──C           ← origin/main
       \
        D′──E′    ← feature/report
```

내 변경을 최신 `C` 위에 다시 적용합니다. 부모 커밋 등이 달라져 `D`, `E` 대신 새 ID의 `D′`, `E′`가 생깁니다. 위 그림은 충돌 없이 재적용되는 단순한 예입니다. [공식 rebase 문서](https://git-scm.com/docs/git-rebase)

| 비교 | merge | rebase |
|------|-------|--------|
| 기존 커밋 ID | 유지 | 다시 적용한 커밋은 변경 |
| 이력 모양 | 갈라지고 합쳐진 과정이 남음 | 단순한 경우 한 줄로 정리됨 |
| 처음 사용할 상황 | 공유 브랜치에 최신 변경을 합칠 때 | 내가 혼자 사용하는 브랜치를 최신 기준에 맞출 때 |

입문 단계의 권장 선택은 **팀 규칙이 없으면 merge로 협업을 시작하고, rebase는 내 전용 브랜치에서 연습하는 것**입니다. 이미 다른 사람이 기반으로 삼은 커밋을 rebase하면 상대도 이력을 맞춰야 합니다. 팀이 선형 이력을 요구한다면 팀의 rebase 절차를 따릅니다.

## 혼자 일할 때와 같이 일할 때

Git 명령의 동작은 같지만, **내 변경이 다른 사람의 작업에 영향을 주는지**에 따라 사용 방법이 달라집니다. 회사 프로젝트를 혼자 담당하더라도 배포 규칙, 보호 브랜치, 승인 절차는 그대로 따라야 합니다.

| 항목 | 혼자 작업하는 프로젝트 | 여러 사람이 작업하는 프로젝트 |
|------|------------------------|--------------------------------|
| 작업 시작 | 다른 PC나 웹에서 수정했다면 최신 상태 확인 | 동료의 변경을 받기 위해 시작할 때 fetch하고 대상 브랜치 갱신 |
| 브랜치 | 작은 개인 프로젝트는 main만으로도 가능. 실험이나 큰 변경은 작업 브랜치 사용 | 보통 작업별 브랜치를 만들고 MR로 반영 |
| 커밋 | 나중에 되돌리고 이해할 수 있도록 작업 단위로 기록 | 리뷰와 문제 추적이 쉽도록 한 가지 변경씩 기록 |
| merge | 내가 나눈 작업을 합칠 때 사용 | 동료의 변경을 내 작업에 가져오거나 MR을 병합할 때 사용 |
| rebase | 다른 사람이 사용하지 않는 내 커밋을 정리할 때 선택 가능 | 팀 규칙을 따르고, 다른 사람이 기반으로 삼은 커밋은 임의로 재작성하지 않음 |
| MR | 회사 규칙이 허용하면 생략 가능. 스스로 검토하거나 검사 결과를 모으는 용도로도 유용 | 변경 의도, 검사 결과, 리뷰 의견을 모으는 기본 협업 절차 |
| 충돌 | 내 브랜치끼리 또는 여러 PC의 작업 사이에서도 발생 가능 | 양쪽 변경의 의도를 확인하고 필요한 경우 작성자와 함께 해결 |
| 실수 복구 | 다른 작업이나 배포에서 사용 중인 이력인지 먼저 확인 | 이미 공유된 변경은 보통 revert로 취소하고 영향 범위를 알림 |

### 함께 일해도 각자 브랜치를 쓰는 경우

예를 들어 나는 `feature/report`, 동료는 `feature/login`에서 작업합니다. 서로 다른 브랜치에 push하므로 상대의 브랜치를 직접 덮어쓰지는 않습니다. 하지만 동료의 MR이 `main`에 먼저 병합되면, 내 MR을 병합하기 전에 그 변경을 가져와 호환성을 확인해야 할 수 있습니다.

이때 내 브랜치를 나만 사용하고 팀이 허용한다면 rebase를 선택할 수 있습니다. **팀 작업이라는 이유만으로 rebase가 금지되는 것은 아닙니다.** 내 브랜치의 커밋을 다른 사람이 가져가서 작업했는지가 판단 기준입니다. [공식 rebase 문서](https://git-scm.com/docs/git-rebase)

### 같은 브랜치를 두 사람이 공유하는 경우

둘 다 `feature/report`에 push한다면 상대가 올린 커밋을 먼저 확인해야 합니다. 상대가 먼저 push하여 내 push가 거절되었다면, 내 변경을 커밋하고 다음처럼 원격 작업 브랜치의 이력을 확인합니다.

```bash
git switch feature/report
git fetch origin
git log --oneline --left-right HEAD...origin/feature/report
```

여기서 상대의 작업을 가져올 대상은 `origin/main`이 아니라 **`origin/feature/report`**입니다. 팀이 merge를 허용한다면 `git merge origin/feature/report`로 합치고, 충돌 해결과 테스트 후 다시 push합니다. main의 최신 변경을 가져오는 작업은 별도입니다.

공유 브랜치의 rebase나 강제 push는 기존 커밋을 상대가 사용 중일 수 있으므로 임의로 하지 않습니다. 처음에는 각자 작업 브랜치를 사용하는 편이 조율하기 쉽습니다.

## 주의해야 할 점과 소통 방법

### 실수하기 쉬운 지점

- **명령을 실행하기 전 현재 브랜치를 확인합니다.** `git status`에서 브랜치와 남은 변경을 확인합니다. merge와 rebase는 현재 브랜치에 적용됩니다.
- **브랜치 이동이나 이력 통합 전에 변경을 저장합니다.** 커밋하거나 stash하여, 다른 작업과 섞이거나 충돌 해결 중 잃지 않도록 합니다.
- **push 거절의 원인을 먼저 확인합니다.** 동료의 새 커밋, 내 rebase, 권한 제한 등 원인이 다릅니다. 해결책으로 곧바로 `--force`를 사용하지 않습니다.
- **충돌 표시가 사라졌다고 해결이 끝난 것은 아닙니다.** 두 기능이 함께 동작하는지 테스트합니다. Git이 자동 병합해도 코드의 의미가 충돌할 수 있습니다.
- **비밀 정보와 불필요한 파일을 커밋하지 않습니다.** 토큰, 비밀번호, 로컬 설정, 생성 파일을 확인합니다. `.gitignore`는 이미 추적 중인 파일을 자동으로 제외하지 않습니다.
- **공유된 실수는 조용히 이력을 지우지 않습니다.** 영향받는 사람에게 알리고 복구 방법을 합의합니다. 비밀 정보가 올라갔다면 파일 삭제만으로 끝내지 않고 담당자에게 알려 자격 증명을 폐기·교체합니다.
- **병합과 배포를 구분합니다.** MR 병합 뒤 배포되는지는 프로젝트의 파이프라인 설정에 달려 있습니다. 실제 배포 여부도 확인합니다.

### 언제 무엇을 전달할까

| 시점 | 전달할 내용 | 예시 |
|------|-------------|------|
| 작업 시작 | 담당 범위, 브랜치, 겹칠 가능성이 있는 파일 | “보고서 다운로드를 `feature/report`에서 수정하겠습니다. 공통 API 응답 형식도 바꾸려는데 겹치는 작업이 있나요?” |
| 공통 규격 변경 전 | 무엇이 바뀌는지, 영향받는 호출부, 적용 순서 | “응답 필드명을 바꾸면 로그인 화면도 수정이 필요합니다. 기존 필드를 유지하는 기간과 병합 순서를 먼저 맞추겠습니다.” |
| 충돌 발생 | 파일과 충돌 내용, 양쪽 의도, 제안하는 최종 동작 | “검증 함수에서 충돌이 났습니다. 제 입력 검증과 동료의 오류 처리를 모두 유지하는 방향으로 해결해도 될까요?” |
| MR 리뷰 요청 | MR 링크, 변경 이유, 확인 결과, 집중해서 볼 부분 | “MR !42 리뷰 부탁드립니다. 다운로드 오류를 수정했고 로컬 테스트는 통과했습니다. 권한 처리 부분을 확인해 주세요.” |
| 리뷰 반영 | 반영한 내용과 재확인 결과 | “빈 입력 처리를 수정하고 테스트를 다시 실행했습니다. 변경 내용은 MR에 추가했습니다.” |
| 병합 또는 배포 후 | 반영 위치, 남은 조치, 배포 상태 | “MR !42가 main에 병합되었습니다. 배포는 아직이며, 다음 작업 전 최신 main을 받아 주세요.” |

진행 상황을 알릴 때는 “완료했습니다”만 쓰기보다 **코드 수정, 로컬 검사, MR 병합, 배포 중 어디까지 끝났는지** 구분합니다. 리뷰 의견은 해당 코드 줄의 MR 댓글에 남기고, 별도 대화에서 결정한 내용도 MR에 요약하면 나중에 이유를 찾기 쉽습니다.

### 리뷰 의견을 주고받는 방법

리뷰할 때는 사람보다 동작을 설명합니다. “이 코드가 이상합니다” 대신 “빈 값이 들어오면 여기서 예외가 발생할 수 있습니다. 입력 검증이 필요해 보입니다”처럼 조건과 영향을 적습니다. 반드시 고쳐야 하는 문제인지, 선택 가능한 제안인지도 구분합니다.

리뷰를 받는 쪽은 이해하지 못한 의견을 무조건 적용하기보다 예상 동작과 이유를 확인합니다. 반영했다면 무엇을 바꾸고 어떻게 확인했는지 답합니다. 동의하지 않는다면 근거와 대안을 제시하고 팀 규칙에 따라 결정합니다.

### 이력을 재작성해야 한다면 먼저 조율하기

다른 사람이 사용하는 브랜치의 rebase가 꼭 필요하다면 다음 내용을 먼저 합의합니다.

1. 대상 브랜치와 재작성 이유를 알립니다.
2. 상대가 이미 가져간 커밋과 아직 push하지 않은 작업이 있는지 확인합니다.
3. 작업 시간과 상대가 이후 이력을 맞추는 방법을 정합니다.
4. 재작성과 push가 끝나면 알리고, 상대의 로컬 작업 보존 여부도 확인합니다.

단순히 “rebase할게요”라고 알리는 것만으로 상대 작업이 보호되지는 않습니다. 입문 단계에서는 공유 브랜치의 이력을 유지하는 merge가 가능한지 먼저 검토합니다.

## 어떻게 사용하는가

### 처음 한 번 연결하기

GitLab 프로젝트의 저장소 복제 메뉴에서 회사가 허용한 SSH 또는 HTTPS 주소를 복사합니다. SSH 키 등록이나 HTTPS 인증은 회사 안내에 따릅니다. 아래 주소는 예시이므로 실제 복제 주소로 바꿉니다.

```bash
git clone <GitLab에서_복사한_저장소_URL>
cd <복제된_프로젝트_폴더>

# 이 저장소에서 사용할 커밋 작성자 정보
git config user.name "홍길동"
git config user.email "회사에서 지정한 이메일"

git remote -v
git branch -vv
git status
```

꺾쇠로 표시한 값은 그대로 실행하지 않고 실제 값으로 바꿉니다. `user.name`과 `user.email`은 커밋 작성자 정보이며 로그인 인증 정보가 아닙니다. 인증 토큰은 URL이나 저장소 파일에 넣지 않고 회사가 지정한 자격 증명 저장 방식을 사용합니다.

### 새 작업 시작하기

변경이 없는 상태에서 시작합니다. 변경이 있다면 먼저 커밋하거나 아래의 stash 절차로 보관합니다.

```bash
git status
git switch main
git fetch origin
git merge --ff-only origin/main
git switch -c feature/report
```

로컬 `main`을 최신 상태로 만든 다음 작업 브랜치를 만듭니다. `--ff-only`가 실패하면 로컬에도 서버에 없는 커밋이 있다는 뜻일 수 있으므로, 아래 명령으로 이력을 확인합니다. 곧바로 강제 초기화하지 않습니다.

```bash
git log --oneline --left-right main...origin/main
```

`<`는 로컬 `main`에만, `>`는 `origin/main`에만 있는 커밋입니다. 팀과 처리 방법을 확인하고 진행합니다.

### 수정하고 커밋하기

실제 파일을 편집한 뒤 프로젝트의 테스트나 실행 확인을 수행합니다. 아래 파일명은 수정한 파일명으로 바꿉니다.

```bash
git diff
git add README.md
git diff --staged
git commit -m "보고서 사용법 설명 추가"
```

`git add`는 실행한 시점의 파일 내용을 선택합니다. 이후 파일을 다시 편집했다면 추가 변경을 넣기 위해 다시 add해야 합니다. 커밋 전 `git diff --staged`에서 실제 포함될 내용과 비밀 정보 유무를 확인합니다. 하나의 커밋에는 설명할 수 있는 한 가지 변경을 담습니다.

### 작업 브랜치를 올리고 MR 만들기

```bash
git push -u origin feature/report
```

`-u`는 로컬 브랜치의 upstream을 `origin/feature/report`로 설정합니다. 이후 같은 브랜치에서는 보통 `git push`만으로 올릴 수 있습니다.

GitLab 프로젝트에서 Merge Request를 생성합니다. source는 `feature/report`, target은 팀이 정한 `main` 등입니다. 제목과 설명에 다음을 적습니다.

- 어떤 문제 때문에 무엇을 바꿨는지.
- 어떻게 확인했는지와 아직 확인하지 못한 부분.
- 관련 이슈가 있다면 이슈 번호.

검토자를 지정하고 변경 내용, 의견, 파이프라인 결과를 확인합니다. GitLab의 CI/CD 파이프라인은 프로젝트에 설정되어 있을 때 자동 빌드와 테스트 등을 실행합니다. 팀이 요구하는 승인과 검사가 끝나면 권한이 있는 사람이 병합합니다. MR은 리뷰와 검사를 모으는 협업 단위입니다. [GitLab MR 공식 문서](https://docs.gitlab.com/user/project/merge_requests/)

리뷰 수정은 같은 작업 브랜치에서 새 커밋을 만들고 push합니다. 기존 MR에 반영되므로 수정할 때마다 새 MR을 만들 필요가 없습니다.

### 작업 중 main이 갱신되었을 때

작업 브랜치의 변경을 먼저 커밋하고 최신 `main`을 가져옵니다. 팀에서 merge를 허용한다면 다음처럼 진행합니다.

```bash
git switch feature/report
git status
git fetch origin
git merge origin/main
# 충돌이 있다면 아래 절차로 해결한 뒤 진행
# 프로젝트 테스트와 실행 확인
git push origin feature/report
```

이것은 **main의 변경을 내 작업에 가져오는 병합**입니다. GitLab MR의 병합은 반대 방향인 **내 작업을 main에 반영하는 절차**입니다.

팀에서 rebase를 요구하고 혼자 쓰는 브랜치라면 위의 merge 대신 `git rebase origin/main`을 사용합니다. 아직 push하지 않은 커밋만 다시 적용했다면 보통 일반 push로 충분합니다. 이미 push한 커밋을 다시 적용했다면 일반 push가 거절될 수 있습니다.

팀이 이력 재작성을 허용하고 해당 브랜치를 나만 사용한다는 것을 확인한 경우에만 다음을 사용합니다.

```bash
git push --force-with-lease origin feature/report
```

`--force-with-lease`는 서버 브랜치가 예상한 상태와 일치할 때만 강제로 갱신합니다. 기본형은 로컬의 원격 추적 참조를 기준으로 하므로 편집기의 자동 fetch 등이 이 보호를 약화할 수 있습니다. 공유 브랜치를 안전하게 덮어쓰는 만능 옵션이 아니며, 거절되면 fetch 후 이력을 검토하고 팀과 조율합니다. `main`에는 사용하지 않습니다. [공식 push 문서](https://git-scm.com/docs/git-push)

### 충돌 해결하기

충돌 해결은 **한쪽을 승자로 고르는 작업이 아니라, 두 변경의 목적을 확인해 최종 파일을 만드는 작업**입니다. 사람은 최종 동작을 결정하고, Git에는 그 파일을 해결 결과로 등록합니다.

#### 1단계 어느 작업에서 멈췄는지 확인하기

예를 들어 내 `feature/report`에 최신 `main`을 가져오는 상황입니다. 시작 전 내 변경을 커밋하거나 stash합니다.

```bash
git switch feature/report
git status
git fetch origin
git merge origin/main
```

충돌이 나면 merge가 중간에서 멈춥니다. 다시 pull이나 rebase를 실행하지 않고 현재 상태부터 확인합니다.

```bash
git status
git diff --name-only --diff-filter=U
```

두 번째 명령은 아직 해결되지 않은 파일을 나열합니다. 예를 들어 `both modified: report.py`는 양쪽에서 수정한 파일입니다. 다른 브랜치를 만들거나 push해도 이 충돌이 저절로 해결되지는 않습니다.

#### 2단계 원래 코드와 양쪽 변경 의도 읽기

원래 `report.py`가 다음과 같았다고 가정합니다. 입력은 문자열 또는 `None`입니다.

```python
def normalize_title(title):
    return title.strip()
```

나는 빈 제목에 기본값을 넣었고, 동료는 `None`을 처리했습니다. merge 후 파일에는 다음과 같은 표시가 생길 수 있습니다.

```python
def normalize_title(title):
<<<<<<< HEAD
    cleaned = title.strip()
    return cleaned if cleaned else "제목 없음"
=======
    if title is None:
        return "제목 없음"
    return title.strip()
>>>>>>> origin/main
```

| 표시 | 이 merge 예제에서의 의미 |
|------|--------------------------|
| `<<<<<<< HEAD` 아래 | merge를 시작한 내 작업 브랜치의 내용 |
| `=======` | 두 내용을 나누는 구분선 |
| `>>>>>>> origin/main` 위 | 가져오는 `origin/main`의 내용 |

한 파일 안에 이 구간이 여러 개 있을 수 있습니다. 설정에 따라 `|||||||` 뒤에 공통 조상의 내용도 표시됩니다. 그것은 추가할 세 번째 코드가 아니라 비교 기준입니다.

변경 이유가 불명확하면 파일의 커밋 이력과 MR 설명을 읽거나 동료에게 묻습니다. 단순히 최근에 작성된 코드라는 이유로 선택하지 않습니다. [공식 merge 문서](https://git-scm.com/docs/git-merge)

#### 3단계 최종 동작을 합의하고 파일 직접 수정하기

예제의 요구사항은 다음 세 가지입니다.

- `None`이면 `"제목 없음"`을 반환한다.
- 빈 문자열이나 공백뿐인 문자열도 `"제목 없음"`을 반환한다.
- 정상 제목은 양끝 공백을 제거한다.

내 코드만 남기면 `None`에서 실패합니다. 동료 코드만 남기면 공백 입력에 빈 문자열을 반환합니다. 따라서 **두 의도를 함께 만족하는 코드**로 바꿉니다.

```python
def normalize_title(title):
    if title is None:
        return "제목 없음"
    cleaned = title.strip()
    return cleaned if cleaned else "제목 없음"
```

편집기에서 충돌 구간 전체를 위의 최종 코드로 교체하고 저장합니다. `<<<<<<<`, `=======`, `>>>>>>>` 표시 줄도 삭제합니다. 충돌 구간 바깥에서 자동으로 합쳐진 변경은 필요 없이 지우지 않습니다.

해결 방식은 상황에 따라 달라집니다.

| 상황 | 수정 방향 |
|------|-----------|
| 한쪽 변경이 현재 요구사항에 맞고 다른 쪽은 폐기하기로 합의 | 선택한 내용을 남기고 이유를 기록 |
| 서로 다른 기능을 추가했고 둘 다 필요 | 실행 순서와 중복을 확인하며 함께 반영 |
| 같은 설정값을 서로 다른 값으로 변경 | 담당자와 최종 값을 결정. 값을 두 개 나열하는 것으로 해결하지 않음 |
| 함수명이나 API 형식 변경과 기능 수정이 겹침 | 최종 인터페이스를 정하고 호출부까지 함께 수정 |

동료에게는 “충돌 났어요” 대신 다음처럼 전달합니다.

> `report.py`의 제목 처리에서 충돌이 났습니다. 제 변경은 빈 제목 기본값이고, 동료 변경은 None 처리입니다. 두 조건 모두 기본값을 반환하도록 합치려 합니다. None을 별도 오류로 처리하려던 의도가 있었나요?

#### 4단계 편집기의 선택 버튼을 사용할 때 주의하기

편집기에 Current, Incoming, Both 등의 선택 버튼이 보일 수 있습니다. 각 버튼이 실제로 어떤 코드를 남기는지 확인합니다. **Both는 두 코드를 붙일 뿐**, 실행 순서나 중복 함수, 앞쪽 return으로 뒤쪽 코드가 실행되지 않는 문제까지 해결하지는 않습니다.

특히 rebase에서는 Git의 ours가 새 기반과 이미 재적용한 이력, theirs가 지금 재적용하는 내 커밋 쪽입니다. 따라서 “Current는 무조건 내 작업”이라고 생각하면 잘못 선택할 수 있습니다. 화면의 실제 코드와 커밋을 읽고 최종 결과를 확인합니다. [공식 rebase 문서](https://git-scm.com/docs/git-rebase)

#### 5단계 결과를 검사하고 해결 완료로 등록하기

예제 파일을 저장한 뒤 프로젝트의 테스트를 수행합니다. Python이 설치된 환경에서는 다음 작은 검사로 세 요구사항을 확인할 수 있습니다. 프로젝트가 `python` 명령을 사용한다면 명령 이름을 바꿉니다.

```bash
python3 -c 'from report import normalize_title; assert normalize_title(None) == "제목 없음"; assert normalize_title("   ") == "제목 없음"; assert normalize_title("  보고서  ") == "보고서"'
```

성공하면 출력 없이 끝납니다. 오류가 나면 코드를 수정하고 다시 확인합니다. 실제 프로젝트에서는 관련 호출부와 기존 기능 검사도 필요합니다.

```bash
git diff --check
git add report.py
git diff --name-only --diff-filter=U
git diff --staged --check
git diff --staged
git status
```

`git add report.py`는 “이 파일의 현재 내용을 해결 결과로 사용하겠다”는 뜻입니다. **Git이 코드의 정확성을 검증했다는 뜻은 아닙니다.** 추가 편집을 했다면 다시 add합니다. 여러 충돌 파일이 있다면 각각 처리합니다.

`--diff-filter=U` 출력이 없어야 미해결 파일이 없는 상태입니다. `--check`는 변경 안의 충돌 표시나 공백 오류를 찾는 보조 검사이며, 기능 테스트를 대신하지 않습니다. staged diff에는 충돌 해결 외에 자동으로 합쳐진 변경도 포함될 수 있으므로 함께 검토합니다. [공식 diff 문서](https://git-scm.com/docs/git-diff)

#### 6단계 merge 또는 rebase 마무리하기

merge 중이라면 다음을 실행합니다.

```bash
git merge --continue
git status
```

커밋 메시지 편집기가 열리면 병합 목적을 확인하고 저장·종료합니다. merge가 끝난 후 관련 테스트를 다시 확인하고 `git push origin feature/report`로 올립니다. 일반적인 merge에는 강제 push가 필요하지 않습니다.

rebase 중이라면 다음을 실행합니다. 별도의 일반 커밋을 먼저 만들기보다 rebase를 계속합니다.

```bash
git rebase --continue
git status
```

다음 커밋에서 다시 충돌하면 파일 편집 → 검사 → add → continue를 반복합니다. 전체 rebase가 끝난 뒤 테스트하고 push합니다. 이미 push한 커밋을 재작성한 경우에는 앞의 강제 push 조건과 팀 규칙을 확인합니다.

MR 설명이나 댓글에는 충돌을 어떻게 해결했고 어떤 검사를 했는지 남깁니다.

> 제목 처리 충돌을 해결했습니다. None 처리와 공백 제목 기본값을 모두 유지했습니다. None, 공백, 정상 제목 검사를 통과했습니다. 관련 호출부도 리뷰 부탁드립니다.

#### 삭제와 수정이 겹치거나 텍스트가 아닌 파일에서 충돌한 경우

모든 충돌에 표시 줄이 생기는 것은 아닙니다. `git status`와 파일 내용을 함께 확인합니다.

| 충돌 종류 | 처리 방법 |
|-----------|-----------|
| 한쪽은 파일 삭제, 다른 쪽은 수정 | 파일이 여전히 필요한지 합의. 유지한다면 최종 파일을 저장하고 `git add <파일>`, 삭제한다면 필요한 변경의 이동 여부를 확인하고 `git rm <파일>` |
| 파일 이동 또는 이름 변경과 수정이 겹침 | 최종 경로를 정하고 내용을 반영. 옛 경로가 불필요하게 남았는지와 import·참조 경로 확인 |
| 양쪽에서 같은 이름의 새 파일 추가 | 두 파일의 목적을 확인하여 하나로 합치거나 이름을 나누고 참조 수정 |
| 이미지, Excel 등 바이너리 파일 | Git이 내부 내용을 텍스트처럼 합칠 수 없음. 사용할 버전을 합의하거나 해당 앱에서 수동으로 통합한 뒤 저장하고 add |
| 의존성 잠금 파일 | 선언 파일의 의존성부터 합의한 뒤 프로젝트가 정한 패키지 관리자와 버전으로 잠금 파일 갱신 및 검사 |

#### 방향을 모르겠다면 취소하고 다시 협의하기

merge나 rebase가 아직 진행 중일 때 해당 작업만 취소합니다.

```bash
git merge --abort     # merge 진행 중일 때만
# 또는
git rebase --abort    # rebase 진행 중일 때만
```

취소하면 해당 작업 시작 전 상태로 돌아갑니다. 해결 중 편집한 내용을 남기고 싶다면 먼저 별도로 복사하거나 기록합니다. 시작 전 미커밋 변경이 있었다면 merge 취소로 완전히 복구되지 않을 수도 있으므로, 처음에 작업 폴더를 깨끗하게 두어야 합니다.

완료된 병합을 취소하는 명령은 아닙니다. 이미 완료하거나 push했다면 공유 상태를 확인하고 별도 복구 방법을 정합니다. 단순히 다음으로 넘어가려고 `git rebase --skip`을 실행하면 현재 커밋의 변경이 제외될 수 있습니다.

### 아직 커밋하기 어려운 변경 잠시 보관하기

```bash
git stash push -u -m "보고서 작업 임시 보관"
# 브랜치 이동이나 갱신 수행
# 원래 작업 브랜치로 돌아온 뒤
git stash apply
git status
```

`-u`는 추적하지 않는 새 파일도 포함하며, 무시된 파일은 포함하지 않습니다. `apply`는 보관본을 유지하면서 변경을 복원합니다. 복원 시에도 충돌이 생길 수 있습니다. 변경이 잘 돌아왔는지 확인한 다음 필요 없어진 보관본만 삭제합니다.

```bash
git stash list
git stash drop 'stash@{0}'
```

### MR 병합 후 다음 작업 준비하기

작업 폴더가 깨끗한 상태에서 다음을 실행합니다.

```bash
git switch main
git fetch --prune origin
git merge --ff-only origin/main
git branch -d feature/report
```

일반적인 복제 설정에서 `--prune`은 서버에서 삭제된 브랜치의 원격 추적 참조를 정리합니다. 로컬 작업 브랜치는 별도로 삭제합니다. GitLab에서 squash 또는 rebase 방식으로 병합했다면 커밋 ID가 달라 `-d`가 거절될 수 있습니다. MR이 실제 병합되었는지, 로컬에 남은 작업은 없는지 확인하기 전에는 `-D`로 강제 삭제하지 않습니다.

## worktree로 여러 브랜치를 별도 폴더에서 작업하기

`git worktree`는 **하나의 로컬 저장소에 작업 폴더를 추가하여 여러 브랜치를 동시에 열어 두는 기능**입니다. 코드 전체를 별도 clone해서 관리하는 대신 커밋 이력과 브랜치 정보를 공유합니다. 각 폴더의 작업 파일, 스테이징 영역, 현재 브랜치는 별도로 유지됩니다. [공식 worktree 문서](https://git-scm.com/docs/git-worktree)

예를 들어 기능 개발 중 긴급 수정이 들어왔을 때, 원래 폴더의 미완성 변경을 그대로 두고 긴급 수정용 폴더에서 작업할 수 있습니다.

```text
project/         → feature/report   기능 개발 중
project-hotfix/  → fix/report-error 긴급 수정
     두 폴더는 같은 로컬 Git 이력을 공유
```

### 언제 사용하면 좋을까

- 기능 개발을 그대로 두고 긴급 수정을 해야 할 때.
- 동료의 브랜치를 별도 폴더에서 확인하거나 테스트할 때.
- 두 브랜치의 동작을 나란히 비교할 때.

한 번에 한 가지 작업만 한다면 기존 폴더에서 switch하는 것으로 충분합니다. worktree를 필수로 사용할 필요는 없습니다.

### 추가하고 사용하기

아래는 원래 저장소 폴더에서 실행합니다. `../project-hotfix`는 아직 존재하지 않는 새 폴더 경로이며 실제 환경에 맞게 바꿉니다.

```bash
git fetch origin
git worktree add -b fix/report-error ../project-hotfix origin/main
git worktree list

cd ../project-hotfix
git status
# 파일 수정 및 프로젝트 테스트
git add <수정한_파일>
git commit -m "보고서 오류 수정"
git push -u origin fix/report-error
```

`-b`는 새 브랜치를 만듭니다. 이미 존재하는 로컬 브랜치를 열고 싶다면 `git worktree add ../project-review feature/review`처럼 사용합니다. 같은 브랜치를 여러 worktree에 동시에 체크아웃하는 것은 기본적으로 제한됩니다.

push 후에는 일반 작업과 동일하게 GitLab MR로 검토받습니다. worktree 자체가 GitLab에 업로드되는 것은 아니며, push로 브랜치의 커밋이 전달됩니다.

### 주의점과 충돌의 관계

| 구분 | 기억할 점 |
|------|-----------|
| 미완성 변경 | 원래 폴더의 미커밋 변경은 다른 폴더에 자동으로 복사되지 않음 |
| 공유 정보 | 커밋 이력, 브랜치와 원격 추적 참조를 공유. 한 폴더에서 fetch하면 다른 폴더에서도 갱신된 참조를 볼 수 있음 |
| 파일 상태 | fetch했다고 다른 폴더의 작업 파일까지 자동 갱신되는 것은 아님 |
| 실행 환경 | 의존성 설치, 가상 환경, 로컬 설정은 새 폴더에서 별도 준비가 필요할 수 있음. 두 앱을 실행하면 포트도 조율 |
| 충돌 | 작업 공간을 나눠도 최종 merge나 rebase에서 같은 코드의 충돌은 발생 가능 |

충돌이 난 worktree 안에서 `git status`를 확인하고 앞의 해결 절차를 수행합니다. 다른 worktree에서 파일을 고쳐도 충돌 중인 폴더의 파일은 바뀌지 않습니다. 협업에서는 사용하는 폴더 이름보다 **브랜치 이름과 MR**을 기준으로 소통합니다.

### 작업 폴더 정리하기

MR 병합 여부와 남은 변경을 확인한 다음 원래 저장소 폴더로 돌아와 제거합니다.

```bash
cd ../project
git worktree list
git worktree remove ../project-hotfix
```

`project`는 원래 저장소 폴더 이름으로 바꿉니다. remove는 추가한 작업 폴더와 등록 정보를 제거하지만 브랜치를 삭제하지 않습니다. 미커밋 변경이나 추적하지 않는 파일이 있어 거절되면 먼저 보존 여부를 확인합니다. 로컬 설정처럼 무시된 파일도 필요하다면 제거 전에 별도로 보관합니다. 브랜치는 앞의 MR 병합 후 정리 절차에 따라 따로 삭제합니다.

## GitLab을 효과적으로 사용하는 팀 규칙

새로 GitLab을 도입한 팀이라면 복잡한 브랜치 체계보다 다음 합의부터 시작하는 것을 권장합니다. 이는 회사의 현재 정책을 확인한 결과가 아니라 입문용 제안입니다.

- 기본 브랜치와 MR 대상 브랜치를 명확히 정합니다.
- `main`은 보호하고 직접 push 대신 MR로 변경을 반영하도록 설정합니다. 보호 브랜치의 push 권한과 merge 권한은 별도로 설정할 수 있습니다. [GitLab 보호 브랜치 문서](https://docs.gitlab.com/user/project/repository/branches/protected/)
- 브랜치는 작업 하나당 하나, MR은 검토 가능한 크기로 유지합니다. 오래 유지할수록 다른 변경과 충돌할 가능성이 커집니다.
- 최신 변경을 가져올 때 merge를 쓸지 rebase를 쓸지 합의합니다.
- 승인, 파이프라인 성공, 의견 해결 중 무엇을 병합 조건으로 삼는지 정합니다. 실제 강제 기능은 GitLab 버전과 요금제, 설정에 따라 다릅니다.

GitLab에서 MR을 병합할 때의 설정도 로컬 Git 명령과 구분해야 합니다.

| GitLab 방식 또는 옵션 | 의미 |
|-----------------------|------|
| Merge commit | 병합 커밋으로 브랜치 이력을 연결 |
| Fast-forward merge | 대상 브랜치를 앞으로 이동. 대상 이력이 소스에 포함되어 있어야 함 |
| Semi-linear history | fast-forward 가능한 상태를 요구하면서 병합 커밋도 생성 |
| Squash | 작업의 여러 커밋을 하나의 변경 커밋으로 묶는 옵션 |

Squash와 rebase는 같은 작업이 아닙니다. 전자는 변경을 묶고, 후자는 커밋의 기반을 옮겨 다시 적용합니다. 로컬에서 merge가 가능해도 프로젝트의 병합 방식이나 검사 조건 때문에 GitLab MR이 막힐 수 있습니다. [GitLab 병합 방식 문서](https://docs.gitlab.com/user/project/merge_requests/methods/)

## 회사 저장소 없이 연습하기

아래는 Git Bash, macOS 또는 Linux 터미널에서 실행하는 로컬 연습입니다. **회사 저장소 밖에서 아직 존재하지 않는 `git-practice` 폴더를 만들어 실행합니다.** 서버 접속이나 push는 없습니다.

```bash
mkdir git-practice
cd git-practice
git init -b main
git config user.name "연습 사용자"
git config user.email "practice@example.invalid"

echo "공통 파일" > base.txt
git add base.txt
git commit -m "공통 출발점 생성"

git switch -c feature/practice
echo "내 작업" > feature.txt
git add feature.txt
git commit -m "작업 파일 추가"
git branch feature/rebase-practice

git switch main
echo "동료 작업" > team.txt
git add team.txt
git commit -m "동료 작업 가정"

# 같은 출발 상태에서 merge 결과 확인
git switch feature/practice
git merge main -m "최신 main 병합"
git log --oneline --graph --decorate --all

# 별도로 보관한 브랜치에서 rebase 결과 확인
git switch feature/rebase-practice
git rebase main
git log --oneline --graph --decorate --all
```

merge 브랜치에는 두 부모를 가진 병합 커밋이 생깁니다. rebase 브랜치에는 `main` 이후에 새 ID의 작업 커밋이 놓입니다. 양쪽 모두 `base.txt`, `feature.txt`, `team.txt`를 포함하는지 확인합니다. 이 실습의 `main`은 로컬 브랜치이며, 앞의 실무 예제에서는 fetch로 갱신한 `origin/main`을 기준으로 삼았습니다.

## 자주 헷갈리는 상황

| 상황 | 먼저 확인할 것 |
|------|----------------|
| fetch했는데 파일이 그대로 | 정상. 현재 브랜치에 반영하는 merge 또는 rebase는 별도 |
| push했는데 main에 코드가 없음 | 작업 브랜치에 push했는지, MR이 병합되었는지 확인 |
| push가 non-fast-forward로 거절됨 | 같은 원격 브랜치에 새 커밋이 있는지, 내가 이력을 재작성했는지 fetch와 log로 확인 |
| MR 파이프라인 실패 | 검사 로그를 읽고 원인을 수정. 무조건 rebase한다고 해결되지 않음 |
| MR에 예상 밖의 파일이 보임 | source와 target, 브랜치 출발점, 실제 커밋 내용 확인 |
| 실수한 커밋을 이미 팀이 사용 중 | 보통 이력을 삭제하는 reset보다 취소 커밋을 만드는 revert를 검토 |

기억할 순서는 **fetch는 가져오기, merge는 이력 합치기, rebase는 새 기반에 다시 적용하기, push는 서버에 전달하기, MR은 검토 후 반영하기**입니다.

## 참고 자료

- [Git fetch 공식 문서](https://git-scm.com/docs/git-fetch)
- [Git pull 공식 문서](https://git-scm.com/docs/git-pull)
- [Git merge 공식 문서](https://git-scm.com/docs/git-merge)
- [Git rebase 공식 문서](https://git-scm.com/docs/git-rebase)
- [Git push 공식 문서](https://git-scm.com/docs/git-push)
- [Git diff 공식 문서](https://git-scm.com/docs/git-diff)
- [Git worktree 공식 문서](https://git-scm.com/docs/git-worktree)
- [GitLab Merge Request 공식 문서](https://docs.gitlab.com/user/project/merge_requests/)
- [GitLab 병합 방식 공식 문서](https://docs.gitlab.com/user/project/merge_requests/methods/)
- [GitLab 보호 브랜치 공식 문서](https://docs.gitlab.com/user/project/repository/branches/protected/)
- [개발 환경 문서 목록](../README.md)
