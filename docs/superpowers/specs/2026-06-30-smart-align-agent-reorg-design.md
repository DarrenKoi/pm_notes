---
tags: [aix, reorg, smart-align-agent, design]
level: design-spec
last_updated: 2026-06-30
type: design-spec
---

# AIX_POC 폴더 재편 설계 — Smart Align Agent 프로젝트화

> `my-task/AIX_POC/`의 중복·산재를 정리한다. **lectures를 방법론 표준으로 고정**하고, 콘텐츠를 **3축(표준 / 가이드 / 프로젝트)** 으로 재배치하며, CD-SEM Align Fail 자동화 일체를 **Smart Align Agent** 프로젝트 폴더로 모은다.

## 왜 필요한가 (Why)

- 방법론 자료가 세 곳(`source/` 캡처 10장, `lectures/` 완본 덱, `01·02` 틀)에 흩어져 같은 내용이 중복된다.
- CD-SEM 산출물이 루트에 `03·04·05·06·07` + 폴더 + 발표물 3종으로 평면 나열돼 찾기 어렵다.
- 앞으로 다른 프로젝트도 **폴더 단위**로 관리하려면, lectures를 팀 가이드 기준점으로 세우고 프로젝트별 폴더 컨벤션을 지금 확립해야 한다.

## 목표 구조 (What)

```
AIX_POC/
├── README.md                     # 3축 안내 + "프로젝트별 폴더" 컨벤션으로 재작성
├── CLAUDE.md                     # 경로·규칙 갱신
├── lectures/                     ★ 방법론 표준 (팀 가이드 기준점)
│   ├── README.md                 #   captures 참조로 갱신, 바이너리 삭제 명시
│   ├── design-camp-deck-v2.7.md
│   ├── design-camp-template-v2.7.md
│   └── captures/                 #   구 source/ 10장 흡수 (파일명 유지)
│       └── 01..10-*.md
├── _가이드/                       ★ 재사용 방법론 틀 + 템플릿 키트
│   ├── 01-기획문서_AX서비스기획.md      # 구 루트 01 (파일명 유지)
│   ├── 02-기술문서_AI과제정의구현.md     # 구 루트 02 (파일명 유지)
│   └── 00-템플릿_AI과제발굴/            # 구 루트 00-템플릿 (그대로)
├── 프로젝트_smart_align_agent/    ★ CD-SEM 전부 = Smart Align Agent
│   ├── README.md                 #   프로젝트 인덱스 (신규)
│   ├── 03-적용_CDSEM기획.md         # 파일명 유지
│   ├── 04-적용_CDSEM기술.md
│   ├── 05-적용_CDSEM_PoC실험설계.md
│   ├── 07-적용_CDSEM_실행기획/        # 11파일 폴더째
│   ├── 07-적용_CDSEM_실행기획_발표요약.md / .pptx
│   └── 07-적용_CDSEM_실행기획_세트.pptx
└── tools/                        # 그대로
                                  # (구 handoff/ 는 삭제 — 사용자 지시 2026-06-30)
```

## 핵심 설계 원칙 (How)

1. **파일명 보존, 폴더만 이동** — 번호 재정렬(03→01 등)은 하지 않는다. 프로젝트 내부 상대링크(03↔04↔05, 07폴더→03/04)는 상대 구조가 보존되어 **자동으로 유효**하게 남는다. 깨지는 건 경계를 넘는 링크뿐.
2. **모든 이동은 `git mv`** — blame/log 히스토리 보존.
3. **source/ 는 삭제가 아니라 흡수** — `lectures/captures/`로 이동. ~40개 `원문 NN` 인용은 경로 앞부분만 기계 치환(무손실, 추적성 유지).

## 작업 항목

### A. 이동 (git mv)
| 원본 | 대상 |
|------|------|
| `source/` (10파일+) | `lectures/captures/` |
| `01-기획문서_AX서비스기획.md` | `_가이드/01-기획문서_AX서비스기획.md` |
| `02-기술문서_AI과제정의구현.md` | `_가이드/02-기술문서_AI과제정의구현.md` |
| `00-템플릿_AI과제발굴/` | `_가이드/00-템플릿_AI과제발굴/` |
| `03·04·05-적용_CDSEM*.md` | `프로젝트_smart_align_agent/` |
| `07-적용_CDSEM_실행기획/` | `프로젝트_smart_align_agent/07-적용_CDSEM_실행기획/` |
| `07-…_발표요약.md/.pptx`, `07-…_세트.pptx` | `프로젝트_smart_align_agent/` |

### B. 삭제
- `lectures/SKHY*.pdf`, `lectures/SKHY*.pptx` (원본 바이너리; md 추출본 보존)
- `06-적용_CDSEM_DesignCamp장표.md` + `.pptx` (07 세트가 상위 대체)

### C. 링크 치환 (상대 깊이별)
- `source/` → `lectures/captures/`
  - `_가이드/01·02` (1단계 깊이): `./source/NN` → `../lectures/captures/NN`
  - `_가이드/00-템플릿/*`: `../source/NN` → `../../lectures/captures/NN`
  - `프로젝트_/03·04·05`: `./source/NN` → `../lectures/captures/NN`
  - `프로젝트_/07.../*`: `../source/NN` → `../../lectures/captures/NN`
  - `lectures/README.md`: `./source/…` → `./captures/…`
- 틀 문서 경계 링크
  - `프로젝트_/03·04` → `./0N-…틀` → `../_가이드/0N-…`
  - `프로젝트_/07.../*` → `../0N-…틀` → `../../_가이드/0N-…`
- 삭제된 06 참조: `07/00-README.md`, `07/11-review`, `README.md`에서 06 링크 제거/07 세트로 대체
- `lectures/README.md`: "원본 바이너리 삭제 가능" → "삭제됨(md 추출본이 대체)"

### D. 문서 재작성
- `AIX_POC/README.md` — 3축 구성표 + 프로젝트별 폴더 컨벤션 안내
- `AIX_POC/CLAUDE.md` — 콘텐츠 아키텍처/경로/규칙을 새 구조로 갱신
- `프로젝트_smart_align_agent/README.md` — 신규 프로젝트 인덱스 (Smart Align Agent 정의 + 문서 목록)
- `my-task/CLAUDE.md` — 폴더 표·md2pptx 명령 예시 경로 갱신

### E. 삭제 (사용자 지시 2026-06-30)
- `handoff/*` — 과거 세션 기록. 과거 작업이라 보존 불필요 → `git rm`으로 삭제(폴더째 제거). 옛 경로 인용 수정도 불필요(파일 자체가 사라짐).

## 검증 (완료 기준)
- [ ] `grep -rn '](\.\?\./*source/'` → 0건 (모든 source 링크 치환됨)
- [ ] `grep -rln 'DesignCamp장표\|06-적용'` → 0건
- [ ] 이동한 모든 문서의 상대링크가 실제 파일을 가리킴 (링크 체커 통과)
- [ ] `git status`에 rename(R)으로 잡혀 히스토리 보존 확인
- [ ] 루트 README/CLAUDE/프로젝트 README가 새 구조와 일치

## 범위 밖 (YAGNI)
- 프로젝트 폴더 내부 파일 번호 재정렬(03→01) — 차후 별도 정리
- 신규 콘텐츠 작성 (이번은 재배치·정리만)
