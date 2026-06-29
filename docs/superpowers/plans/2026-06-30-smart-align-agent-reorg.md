# AIX_POC 재편 — Smart Align Agent 프로젝트화 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `my-task/AIX_POC/`를 방법론 표준(lectures) / 재사용 가이드(_가이드) / 프로젝트(프로젝트_smart_align_agent) 3축으로 재배치하고, 중복 바이너리·구버전 발표물을 삭제한다.

**Architecture:** 모든 이동은 `git mv`로 히스토리를 보존한다. 파일명은 유지하고 폴더만 옮긴다 — 프로젝트 내부 상대링크는 상대 구조가 보존되어 자동 유효하고, **경계를 넘는 링크만**(→source, →틀, →삭제될 06) 치환한다. ASCII 경로 접두사는 `perl -CSD -i -pe`로, 한글 포함 문자열은 Edit 도구로 정확히 치환한다.

**Tech Stack:** Markdown 문서, `git mv`, `perl`(UTF-8), Python3(링크 검증 스크립트).

## Global Constraints

- 작업 디렉터리 기준: `/Users/daeyoung/Codes/pm_notes/my-task/AIX_POC` (이하 `AIX_POC/`).
- `handoff/*` 는 **절대 수정하지 않는다** (과거 세션 기록·히스토리). source/06 옛 경로 인용이 남아도 둔다.
- 모든 파일 이동은 `git mv` (복사+삭제 금지). 삭제는 `git rm`.
- 커밋 메시지 말미에 항상:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_011hzFj4Vzvt4kkd3jLFSnxU
  ```
- 한글 파일명/패턴 치환은 `LANG=en_US.UTF-8 perl -CSD -i -pe` 또는 Edit 도구 사용.
- 새 콘텐츠 작성 금지 — 재배치·링크정리·인덱스 문서만. 미확정 수치 플레이스홀더 유지.

---

### Task 1: 방법론 표준 정리 — source→lectures/captures 흡수 + 바이너리 삭제

**Files:**
- Move: `AIX_POC/source/` → `AIX_POC/lectures/captures/`
- Delete: `AIX_POC/lectures/SKHY New AI Design Camp_V2.7 (배포)_A.pdf`, `AIX_POC/lectures/SKHY The New Design Camp_Template V2.7.pptx`
- Modify: `AIX_POC/lectures/README.md`

**Interfaces:**
- Produces: 디렉터리 `AIX_POC/lectures/captures/01..10-*.md` (이후 모든 `원문 NN` 인용의 새 타깃).

- [ ] **Step 1: source/ 를 lectures/captures/ 로 이동 (git mv)**

```bash
cd /Users/daeyoung/Codes/pm_notes/my-task/AIX_POC
git mv source lectures/captures
```

- [ ] **Step 2: 이동 검증**

Run: `ls lectures/captures/ | head && test -d source && echo "STILL EXISTS" || echo "source gone OK"`
Expected: `01-kakao-3step-pain-point.md … 10-check-and-review.md` 목록 + `source gone OK`

- [ ] **Step 3: 원본 바이너리 삭제 (git rm)**

```bash
cd /Users/daeyoung/Codes/pm_notes/my-task/AIX_POC
git rm "lectures/SKHY New AI Design Camp_V2.7 (배포)_A.pdf" "lectures/SKHY The New Design Camp_Template V2.7.pptx"
```

- [ ] **Step 4: lectures/README.md 의 source 참조를 captures 로 갱신 + 바이너리 삭제 명시**

`lectures/README.md` 에서 Edit 도구로:
- frontmatter 아래 요약줄 `` `source/`(캡처 10장 전사)보다 `` → `` `captures/`(구 source, 캡처 10장 전사)보다 ``
- `## `source/`(10캡처)와의 관계` → `## `captures/`(10캡처)와의 관계`
- `- `source/01~10` = 초기 캡처` → `- `captures/01~10` = 초기 캡처`
- "원본 PDF·PPTX는 용량이 커서 **전문(全文)을 md로 추출**해 보관한다. 아래 두 md가 원본을 대체하며, 원본 바이너리는 삭제 가능하다." → "원본 PDF·PPTX는 용량이 커서 **전문(全文)을 md로 추출**해 보관하고 **원본 바이너리는 삭제했다**. 아래 두 md가 원본을 대체한다."
- 표의 `| 원본 |` 열에 적힌 원본 파일명은 "(삭제됨)" 주석을 덧붙인다.

- [ ] **Step 5: lectures 내부에 깨진 source 링크 없는지 확인**

Run: `grep -rn "source/" lectures/README.md || echo "no source refs in lectures/README OK"`
Expected: `no source refs in lectures/README OK` (또는 frontmatter tag만 — tag는 무시)

- [ ] **Step 6: 커밋**

```bash
cd /Users/daeyoung/Codes/pm_notes/my-task/AIX_POC
git add -A
git commit -m "$(cat <<'EOF'
docs(aix-poc): source/ → lectures/captures 흡수 + 원본 바이너리 삭제

방법론 자료를 lectures 표준 아래로 통합. PDF/PPTX 원본은 md 추출본이
대체하므로 삭제(7.4MB 회수). lectures/README를 captures 참조로 갱신.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011hzFj4Vzvt4kkd3jLFSnxU
EOF
)"
```

---

### Task 2: 재사용 가이드 정리 — 01·02 틀 + 00-템플릿 → _가이드/

**Files:**
- Move: `AIX_POC/01-기획문서_AX서비스기획.md` → `AIX_POC/_가이드/01-기획문서_AX서비스기획.md`
- Move: `AIX_POC/02-기술문서_AI과제정의구현.md` → `AIX_POC/_가이드/02-기술문서_AI과제정의구현.md`
- Move: `AIX_POC/00-템플릿_AI과제발굴/` → `AIX_POC/_가이드/00-템플릿_AI과제발굴/`
- Modify (source 링크): `_가이드/01-기획문서_AX서비스기획.md`, `_가이드/02-기술문서_AI과제정의구현.md`, `_가이드/00-템플릿_AI과제발굴/*.md`, `_가이드/00-템플릿_AI과제발굴/_internal/SPEC.md`

**Interfaces:**
- Consumes: `lectures/captures/` (Task 1).
- Produces: `_가이드/01-기획문서_AX서비스기획.md`, `_가이드/02-기술문서_AI과제정의구현.md` (이후 프로젝트 문서의 "틀" 링크 타깃).
- Note: 01↔02↔00-템플릿 상호링크(`../01-…`, `../02-…`)는 셋이 함께 `_가이드/` 아래로 이동하므로 **자동 보존**된다 — 수정 불필요.

- [ ] **Step 1: _가이드/ 로 이동 (git mv, 폴더 자동 생성)**

```bash
cd /Users/daeyoung/Codes/pm_notes/my-task/AIX_POC
mkdir -p _가이드
git mv 01-기획문서_AX서비스기획.md _가이드/01-기획문서_AX서비스기획.md
git mv 02-기술문서_AI과제정의구현.md _가이드/02-기술문서_AI과제정의구현.md
git mv 00-템플릿_AI과제발굴 _가이드/00-템플릿_AI과제발굴
```

- [ ] **Step 2: 이동 검증**

Run: `ls _가이드/ && ls _가이드/00-템플릿_AI과제발굴/ | head -3`
Expected: `00-템플릿_AI과제발굴  01-기획문서_AX서비스기획.md  02-기술문서_AI과제정의구현.md` + 템플릿 파일 목록

- [ ] **Step 3: 01·02 의 source 링크 치환 (`./source/` → `../lectures/captures/`)**

```bash
cd /Users/daeyoung/Codes/pm_notes/my-task/AIX_POC
LANG=en_US.UTF-8 perl -CSD -i -pe 's{\]\(\./source/}{](../lectures/captures/}g' \
  _가이드/01-기획문서_AX서비스기획.md _가이드/02-기술문서_AI과제정의구현.md
```

- [ ] **Step 4: 00-템플릿/* 의 source 링크 치환 (`../source/` → `../../lectures/captures/`)**

```bash
cd /Users/daeyoung/Codes/pm_notes/my-task/AIX_POC
LANG=en_US.UTF-8 perl -CSD -i -pe 's{\]\(\.\./source/}{](../../lectures/captures/}g' \
  _가이드/00-템플릿_AI과제발굴/*.md _가이드/00-템플릿_AI과제발굴/_internal/*.md
```

- [ ] **Step 5: _가이드 내 남은 깨진 source 링크 검증**

Run: `grep -rn "](\./source/\|](\.\./source/" _가이드/ || echo "no broken source links in _가이드 OK"`
Expected: `no broken source links in _가이드 OK`

- [ ] **Step 6: 커밋**

```bash
cd /Users/daeyoung/Codes/pm_notes/my-task/AIX_POC
git add -A
git commit -m "$(cat <<'EOF'
docs(aix-poc): 방법론 틀(01·02) + 템플릿 키트(00) → _가이드/ 로 이동

재사용 가능한 가이드 계층을 한 폴더로 분리. source 인용 경로를
lectures/captures 로 치환(무손실). 01↔02↔00 상호링크는 함께 이동해 보존.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011hzFj4Vzvt4kkd3jLFSnxU
EOF
)"
```

---

### Task 3: 프로젝트화 — CD-SEM 일체 → 프로젝트_smart_align_agent/ + 경계 링크 치환

**Files:**
- Move: `AIX_POC/03-적용_CDSEM기획.md`, `04-적용_CDSEM기술.md`, `05-적용_CDSEM_PoC실험설계.md` → `AIX_POC/프로젝트_smart_align_agent/`
- Move: `AIX_POC/07-적용_CDSEM_실행기획/` → `AIX_POC/프로젝트_smart_align_agent/07-적용_CDSEM_실행기획/`
- Move: `AIX_POC/07-적용_CDSEM_실행기획_발표요약.md`, `07-적용_CDSEM_실행기획_발표요약.pptx`, `07-적용_CDSEM_실행기획_세트.pptx` → `AIX_POC/프로젝트_smart_align_agent/`
- Modify (source + 틀 + 06 링크): 위 이동한 .md 전부 + 07 폴더 .md 전부

**Interfaces:**
- Consumes: `lectures/captures/` (Task 1), `_가이드/01·02` (Task 2).
- Note: 프로젝트 내부 링크(03↔04↔05, 07폴더→`../03`·`../04`, 발표요약→`./03`·`./07-…`)는 상대 구조 보존으로 **자동 유효**. 치환 대상은 source / 틀(01·02) / 06 세 종류뿐.

- [ ] **Step 1: 프로젝트 폴더로 이동 (git mv)**

```bash
cd /Users/daeyoung/Codes/pm_notes/my-task/AIX_POC
mkdir -p 프로젝트_smart_align_agent
git mv 03-적용_CDSEM기획.md 프로젝트_smart_align_agent/03-적용_CDSEM기획.md
git mv 04-적용_CDSEM기술.md 프로젝트_smart_align_agent/04-적용_CDSEM기술.md
git mv 05-적용_CDSEM_PoC실험설계.md 프로젝트_smart_align_agent/05-적용_CDSEM_PoC실험설계.md
git mv 07-적용_CDSEM_실행기획 프로젝트_smart_align_agent/07-적용_CDSEM_실행기획
git mv 07-적용_CDSEM_실행기획_발표요약.md 프로젝트_smart_align_agent/07-적용_CDSEM_실행기획_발표요약.md
git mv 07-적용_CDSEM_실행기획_발표요약.pptx 프로젝트_smart_align_agent/07-적용_CDSEM_실행기획_발표요약.pptx
git mv 07-적용_CDSEM_실행기획_세트.pptx 프로젝트_smart_align_agent/07-적용_CDSEM_실행기획_세트.pptx
```

- [ ] **Step 2: 이동 검증**

Run: `ls 프로젝트_smart_align_agent/ && ls 프로젝트_smart_align_agent/07-적용_CDSEM_실행기획/ | head -3`
Expected: 03·04·05·07 파일 + 07 폴더 내부 파일 목록

- [ ] **Step 3: 03·04·05 의 source 링크 치환 (`./source/` → `../lectures/captures/`)**

```bash
cd /Users/daeyoung/Codes/pm_notes/my-task/AIX_POC/프로젝트_smart_align_agent
LANG=en_US.UTF-8 perl -CSD -i -pe 's{\]\(\./source/}{](../lectures/captures/}g' \
  03-적용_CDSEM기획.md 04-적용_CDSEM기술.md 05-적용_CDSEM_PoC실험설계.md
```

- [ ] **Step 4: 03·04 의 틀 링크 치환 (`](./01-` / `](./02-` → `](../_가이드/0N-`)**

03·04 의 project-root 위치에서 `./01-`·`./02-` 로 시작하는 링크는 방법론 틀뿐이다(내부 narrative 링크는 `./03·04·05·07`).

```bash
cd /Users/daeyoung/Codes/pm_notes/my-task/AIX_POC/프로젝트_smart_align_agent
LANG=en_US.UTF-8 perl -CSD -i -pe 's{\]\(\./01-기획문서}{](../_가이드/01-기획문서}g; s{\]\(\./02-기술문서}{](../_가이드/02-기술문서}g' \
  03-적용_CDSEM기획.md 04-적용_CDSEM기술.md
```

- [ ] **Step 5: 07 폴더의 source 링크 치환 (`../source/` → `../../lectures/captures/`)**

```bash
cd /Users/daeyoung/Codes/pm_notes/my-task/AIX_POC/프로젝트_smart_align_agent
LANG=en_US.UTF-8 perl -CSD -i -pe 's{\]\(\.\./source/}{](../../lectures/captures/}g' \
  07-적용_CDSEM_실행기획/*.md 07-적용_CDSEM_실행기획/_internal/*.md
```

- [ ] **Step 6: 07 폴더의 틀 링크 치환 (`](../01-` / `](../02-` → `](../../_가이드/0N-`)**

07 폴더에서 `../01-`·`../02-` 는 방법론 틀뿐(`../03·04` 는 프로젝트 narrative 라 유지, siblings 는 `./0N`).

```bash
cd /Users/daeyoung/Codes/pm_notes/my-task/AIX_POC/프로젝트_smart_align_agent
LANG=en_US.UTF-8 perl -CSD -i -pe 's{\]\(\.\./01-기획문서}{](../../_가이드/01-기획문서}g; s{\]\(\.\./02-기술문서}{](../../_가이드/02-기술문서}g' \
  07-적용_CDSEM_실행기획/*.md
```

- [ ] **Step 7: 07 폴더의 06(삭제예정) 참조 치환**

`07-적용_CDSEM_실행기획/00-README.md` (Edit 도구):
- `- 발표 장표: [06 Design Camp 장표](../06-적용_CDSEM_DesignCamp장표.md)` → `- 발표물: [실행기획 세트 PPTX](../07-적용_CDSEM_실행기획_세트.pptx) · [발표 요약](../07-적용_CDSEM_실행기획_발표요약.md)`

`07-적용_CDSEM_실행기획/11-review-산출물-발표.md:64` (Edit 도구):
- `기존 [06 Design Camp 발표 장표](../06-적용_CDSEM_DesignCamp장표.md) 세트와 연결해,` → `[실행기획 세트 PPTX](../07-적용_CDSEM_실행기획_세트.pptx)로 변환해,`

- [ ] **Step 8: 프로젝트 폴더 내 경계 링크 잔존 검증**

Run:
```bash
cd /Users/daeyoung/Codes/pm_notes/my-task/AIX_POC/프로젝트_smart_align_agent
grep -rn "](\./source/\|](\.\./source/\|](\./01-기획\|](\./02-기술\|](\.\./01-기획\|](\.\./02-기술\|06-적용_CDSEM\|DesignCamp장표" . || echo "all boundary links rewritten OK"
```
Expected: `all boundary links rewritten OK`

- [ ] **Step 9: 커밋**

```bash
cd /Users/daeyoung/Codes/pm_notes/my-task/AIX_POC
git add -A
git commit -m "$(cat <<'EOF'
docs(aix-poc): CD-SEM 산출물 일체 → 프로젝트_smart_align_agent/ 로 모음

Smart Align Agent 프로젝트 폴더 신설(03·04·05·07 세트·발표물). 경계 링크
(source→lectures/captures, 틀→_가이드, 06→07세트) 치환. 내부 링크는 보존.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011hzFj4Vzvt4kkd3jLFSnxU
EOF
)"
```

---

### Task 4: 구버전 발표물(06) 삭제 + AIX_POC/README 3축 재작성

**Files:**
- Delete: `AIX_POC/06-적용_CDSEM_DesignCamp장표.md`, `AIX_POC/06-적용_CDSEM_DesignCamp장표.pptx`
- Rewrite: `AIX_POC/README.md`

**Interfaces:**
- Consumes: Task 1~3 의 최종 폴더 구조.

- [ ] **Step 1: 06 장표 삭제 (git rm) — 07 세트가 상위 대체**

```bash
cd /Users/daeyoung/Codes/pm_notes/my-task/AIX_POC
git rm "06-적용_CDSEM_DesignCamp장표.md" "06-적용_CDSEM_DesignCamp장표.pptx"
```

- [ ] **Step 2: AIX_POC/README.md 를 3축 구조로 재작성**

`AIX_POC/README.md` 를 Write 로 재작성. 반드시 포함:
- 한 줄 요약: New AI Design Camp 방법론 표준 + 부문 AIX 프로젝트 저장소.
- **3축 구성 표**:
  - `lectures/` — **[표준]** New AI Design Camp V2.7 완본 덱 + 편집 Template + `captures/`(캡처 10장 전사). **팀 가이드의 기준점**.
  - `_가이드/` — **[가이드]** 재사용 방법론 틀: `01-기획문서_AX서비스기획.md`, `02-기술문서_AI과제정의구현.md` + `00-템플릿_AI과제발굴/`(도메인 중립 키트).
  - `프로젝트_smart_align_agent/` — **[프로젝트]** CD-SEM Align Fail 대응 자동화 = **Smart Align Agent**. Discovery·기술·PoC설계·실행기획 세트·발표물.
  - `tools/` — md→pptx 변환기. `handoff/` — 세션 인계 기록.
- **프로젝트 폴더 컨벤션** 안내: 앞으로 신규 과제는 `프로젝트_<이름>/` 폴더로 추가하고, `lectures`(표준)와 `_가이드`(틀)를 참조해 채운다.
- 핵심 흐름(Baseline 3원칙 → 4 Step → 12-Step Track A/B) 다이어그램은 기존 내용 유지.
- 기존 "원문 인덱스(source/)" 표는 경로를 `lectures/captures/` 로 갱신.

- [ ] **Step 3: README 에 깨진 링크 없는지 검증**

Run:
```bash
cd /Users/daeyoung/Codes/pm_notes/my-task/AIX_POC
grep -n "06-적용\|](\./source/\|](\./0[1-7]-적용\|](\./01-기획\|](\./02-기술\|](\./00-템플릿" README.md || echo "README links clean OK"
```
Expected: `README links clean OK` (모든 링크가 `lectures/`, `_가이드/`, `프로젝트_smart_align_agent/` 경로를 가리킴)

- [ ] **Step 4: 커밋**

```bash
cd /Users/daeyoung/Codes/pm_notes/my-task/AIX_POC
git add -A
git commit -m "$(cat <<'EOF'
docs(aix-poc): 06 구버전 장표 삭제 + README 3축 구조로 재작성

07 세트가 06을 상위 대체하므로 삭제. README를 표준(lectures)/가이드/
프로젝트 3축 + 프로젝트 폴더 컨벤션 안내로 재작성.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011hzFj4Vzvt4kkd3jLFSnxU
EOF
)"
```

---

### Task 5: AIX_POC/CLAUDE.md 를 새 구조로 갱신

**Files:**
- Rewrite: `AIX_POC/CLAUDE.md`

- [ ] **Step 1: CLAUDE.md 갱신**

`AIX_POC/CLAUDE.md` 를 Edit/Write 로 갱신. 반영할 변경:
- "현재 파일 현황" 표의 경로를 새 구조로(01·02→`_가이드/`, 03·04·05·07→`프로젝트_smart_align_agent/`).
- 콘텐츠 아키텍처 3계층 다이어그램에서 `source/01~10` → `lectures/captures/01~10`, "01·02 틀" → "`_가이드/01·02`", "03·04 사례" → "`프로젝트_smart_align_agent/03·04`".
- **새 규칙 추가**: "신규 과제는 `프로젝트_<이름>/` 폴더로 분리한다. 표준은 `lectures/`, 재사용 틀은 `_가이드/`. 적용 문서 작성 시 `_가이드/01`(기획)·`_가이드/02`(기술)를 먼저 읽고 미러링한다."
- `source/` 언급을 `lectures/captures/` 로 일괄 갱신. "캡처 1장=파일 1개, 불변 전사" 원칙은 유지.
- handoff/ 절은 그대로.

- [ ] **Step 2: 검증**

Run: `grep -n "source/\|^| `01-\|^| `03-" CLAUDE.md || echo "no stale top-level paths OK"`
Expected: 구 `source/`·루트 `01-`/`03-` 경로 표기가 남지 않음 (`lectures/captures`, `_가이드`, `프로젝트_` 로 대체)

- [ ] **Step 3: 커밋**

```bash
cd /Users/daeyoung/Codes/pm_notes/my-task/AIX_POC
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
docs(aix-poc): CLAUDE.md 를 3축 구조 + 프로젝트 폴더 규칙으로 갱신

경로(lectures/captures·_가이드·프로젝트_)·아키텍처 다이어그램 갱신 및
신규 과제 폴더 컨벤션 규칙 추가.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011hzFj4Vzvt4kkd3jLFSnxU
EOF
)"
```

---

### Task 6: 프로젝트 인덱스 README 신설

**Files:**
- Create: `AIX_POC/프로젝트_smart_align_agent/README.md`

- [ ] **Step 1: 프로젝트 README 작성**

`AIX_POC/프로젝트_smart_align_agent/README.md` 를 Write 로 작성. 포함:
- frontmatter(tags: [aix, cd-sem, smart-align-agent, itc], type: project-index, last_updated: 2026-06-30).
- 제목·한 줄 정의: **Smart Align Agent — CD-SEM Auto Recipe Creation 중 1차 PoC(Align Fail 대응 자동화)**. SEM 화면 판독+GUI 제어로 좌표 shift·align 실패·오인식을 실시간 재정합(Human-in-the-loop: ①검토·⑦승인은 사람).
- **문서 목록 표** (역할/링크): `03-적용_CDSEM기획.md`(Discovery)·`04-적용_CDSEM기술.md`(To-Be/KSF)·`05-적용_CDSEM_PoC실험설계.md`(KSF#1 검증)·`07-적용_CDSEM_실행기획/`(11단계 상세, 내년 실행 깊이)·`07-…_발표요약.md/.pptx`(단계당 1장)·`07-…_세트.pptx`(76장 상세).
- **참조 위치**: 방법론 틀 `../_가이드/01·02`, 표준 강의 `../lectures/`.
- 읽는 순서 가이드: 03 → 04 → 05 → 07 세트.

- [ ] **Step 2: README 링크 유효성 검증**

Run:
```bash
cd /Users/daeyoung/Codes/pm_notes/my-task/AIX_POC/프로젝트_smart_align_agent
for f in 03-적용_CDSEM기획.md 04-적용_CDSEM기술.md 05-적용_CDSEM_PoC실험설계.md 07-적용_CDSEM_실행기획/00-README.md ../_가이드/01-기획문서_AX서비스기획.md ../lectures/README.md; do test -e "$f" && echo "OK $f" || echo "MISSING $f"; done
```
Expected: 전부 `OK`

- [ ] **Step 3: 커밋**

```bash
cd /Users/daeyoung/Codes/pm_notes/my-task/AIX_POC
git add 프로젝트_smart_align_agent/README.md
git commit -m "$(cat <<'EOF'
docs(smart-align-agent): 프로젝트 인덱스 README 신설

Smart Align Agent 정의 + 문서 목록(03·04·05·07)·읽는 순서·참조(_가이드·
lectures) 안내. 향후 프로젝트 폴더 컨벤션의 첫 사례.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011hzFj4Vzvt4kkd3jLFSnxU
EOF
)"
```

---

### Task 7: my-task/CLAUDE.md 경로·명령 예시 갱신

**Files:**
- Modify: `/Users/daeyoung/Codes/pm_notes/my-task/CLAUDE.md`

- [ ] **Step 1: 폴더 표 + md2pptx 명령 예시 갱신**

`my-task/CLAUDE.md` 를 Edit 로:
- "무엇이 들어있나" 표의 `AIX_POC/` 설명에 3축(lectures 표준 / _가이드 틀 / 프로젝트_*) 한 줄 추가.
- md2pptx 명령 예시 경로 갱신:
  - `python tools/md2pptx.py 07-적용_CDSEM_실행기획_발표요약.md 발표요약.pptx` → `python tools/md2pptx.py 프로젝트_smart_align_agent/07-적용_CDSEM_실행기획_발표요약.md 발표요약.pptx`
  - `md2pptx_doc.py` "입력 생략 시 07 폴더 01~11 자동 수집" 설명은 도구 기본 경로에 의존하므로, 도구가 새 경로를 못 찾으면 인자를 명시하라는 주석 한 줄 추가(도구 코드는 이번 범위 밖).
- "콘텐츠 아키텍처" 다이어그램의 `source/01~10`→`lectures/captures/01~10`, `01·02`→`_가이드/01·02`, `03·04…07`→`프로젝트_smart_align_agent/…`.

- [ ] **Step 2: 검증**

Run: `grep -n "프로젝트_smart_align_agent\|lectures/captures\|_가이드" /Users/daeyoung/Codes/pm_notes/my-task/CLAUDE.md | head`
Expected: 새 경로들이 등장

- [ ] **Step 3: 커밋**

```bash
cd /Users/daeyoung/Codes/pm_notes
git add my-task/CLAUDE.md
git commit -m "$(cat <<'EOF'
docs(my-task): CLAUDE.md 를 AIX_POC 3축 구조에 맞춰 갱신

폴더 표·md2pptx 명령 예시 경로·콘텐츠 아키텍처 다이어그램을 새 구조
(lectures/captures·_가이드·프로젝트_smart_align_agent)로 정정.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011hzFj4Vzvt4kkd3jLFSnxU
EOF
)"
```

---

### Task 8: 전수 검증 — 링크 체커 + git 히스토리 확인

**Files:**
- Create(임시): `scratchpad/linkcheck.py` (커밋 안 함)

- [ ] **Step 1: 링크 검증 스크립트 작성**

`/private/tmp/claude-501/-Users-daeyoung-Codes-pm-notes-my-task/9b7717d0-53b1-4b61-b247-f432912968bb/scratchpad/linkcheck.py` 를 Write:

```python
import os, re, sys
ROOT = "/Users/daeyoung/Codes/pm_notes/my-task/AIX_POC"
SKIP = ("handoff/",)  # 과거 기록은 검사 제외
link_re = re.compile(r"\]\((?!https?://|#)([^)]+)\)")
bad = []
for dp, _, fs in os.walk(ROOT):
    for fn in fs:
        if not fn.endswith(".md"):
            continue
        path = os.path.join(dp, fn)
        rel = os.path.relpath(path, ROOT)
        if rel.startswith(SKIP):
            continue
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                for m in link_re.finditer(line):
                    target = m.group(1).split("#")[0].strip()
                    if not target:
                        continue
                    resolved = os.path.normpath(os.path.join(dp, target))
                    if not os.path.exists(resolved):
                        bad.append(f"{rel}:{i} -> {target}")
if bad:
    print("BROKEN LINKS:")
    print("\n".join(bad))
    sys.exit(1)
print("ALL LINKS OK")
```

- [ ] **Step 2: 링크 체커 실행 (handoff 제외 전 문서)**

Run: `python3 /private/tmp/claude-501/-Users-daeyoung-Codes-pm-notes-my-task/9b7717d0-53b1-4b61-b247-f432912968bb/scratchpad/linkcheck.py`
Expected: `ALL LINKS OK`
(BROKEN 출력 시 해당 파일을 Edit 로 고치고 재실행 — 깨끗해질 때까지 반복.)

- [ ] **Step 3: source/06 잔존 참조 전수 검색 (handoff 제외)**

Run:
```bash
cd /Users/daeyoung/Codes/pm_notes/my-task/AIX_POC
grep -rn "](\./source/\|](\.\./source/\|](\.\./\.\./source/\|06-적용_CDSEM\|DesignCamp장표" . --include="*.md" | grep -v "^./handoff/" || echo "no stale source/06 refs OK"
```
Expected: `no stale source/06 refs OK`

- [ ] **Step 4: git 히스토리(rename) 보존 확인**

Run: `cd /Users/daeyoung/Codes/pm_notes && git log --oneline -8 && git log --follow --oneline -3 -- "my-task/AIX_POC/프로젝트_smart_align_agent/03-적용_CDSEM기획.md" | head`
Expected: 최근 커밋들 + `--follow` 가 이동 이전 03 히스토리를 추적(이동 전 커밋 표시)

- [ ] **Step 5: 최종 구조 확인**

Run: `cd /Users/daeyoung/Codes/pm_notes/my-task/AIX_POC && find . -maxdepth 2 -type d -not -path '*/.*' | sort`
Expected: `lectures`, `lectures/captures`, `_가이드`, `_가이드/00-템플릿_AI과제발굴`, `프로젝트_smart_align_agent`, `프로젝트_smart_align_agent/07-적용_CDSEM_실행기획`, `tools`, `handoff` (구 `source` 없음)

- [ ] **Step 6: (필요 시) 잔여 수정 커밋**

링크 체커가 잡아낸 수정이 있었다면:
```bash
cd /Users/daeyoung/Codes/pm_notes/my-task/AIX_POC
git add -A
git commit -m "$(cat <<'EOF'
docs(aix-poc): 재편 후 잔여 깨진 링크 수정

전수 링크 검증에서 발견된 경로 정정.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011hzFj4Vzvt4kkd3jLFSnxU
EOF
)"
```

---

## Self-Review (작성자 체크)

**Spec coverage:**
- 이동(A) → Task 1·2·3. 삭제(B) → Task 1(바이너리)·4(06). 링크치환(C) → Task 1·2·3 + 검증 Task 8. 문서재작성(D) → Task 4(README)·5(CLAUDE)·6(프로젝트README)·7(my-task). 불변(E: handoff) → 모든 grep/링크체커에서 제외. ✅
- 검증 기준(spec) 5項 → Task 8 Step 2·3·4·5 가 각각 커버. ✅

**Placeholder scan:** 모든 perl/grep/git 명령에 실제 경로·패턴 기입. 문서 재작성 태스크는 "반드시 포함" 항목을 구체 명시(빈 TODO 없음). ✅

**Type/path consistency:** 폴더명 `_가이드`·`프로젝트_smart_align_agent`·`lectures/captures` 전 태스크 동일 표기. source 치환 깊이(01·02=`../`, 00-템플릿/07=`../../`, 03·04·05=`../`) 일관. ✅

**주의(실행 중 확인):** perl 치환 후 Step 단위 grep 검증이 항상 뒤따르므로, 패턴 미스매치(예: 링크 형식이 예상과 다른 경우)는 그 즉시 드러난다 — 해당 Step에서 Edit 로 보정 후 진행.
