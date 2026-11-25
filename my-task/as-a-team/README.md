---
tags: [my-task, as-a-team, index]
note_kind: "목차"
last_updated: 2026-11-25
---

# AX Team 승격 기획

> AX Part를 AX Team으로 승격해 달라는 요청의 기획 문서 폴더. 부문장 요청 ("팀이 되면 할 수 있는 일을 리스트로")에 대한 답이 중심.

## 문서

| 파일 | 내용 | 상태 |
|------|------|------|
| [01-team-proposal.md](./01-team-proposal.md) | AX Team 승격 기획안 (원본, 요약형). 2절이 "팀이 되면 추진할 수 있는 업무" 11가지 | 초안, 사용자 답변 반영 (2026-11-25) |
| [01-team-proposal-v2.md](./01-team-proposal-v2.md), `01-team-proposal-v2.docx` | 2번째 버전: 추진 업무 10가지 (가, 나)와 조직 구성만 담은 축약본. 배경, 일정, 요청 사항 제외 | 초안 (2026-11-26) |
| `01-team-proposal.docx` | 위 md로 만든 보고용 문서 | `build_docx.py`로 생성, 손으로 고치지 않음 |
| `build_docx.py` | docx 생성 래퍼 (변환기는 `arc-tf/deliverables/build_docx.py`) | `python3 as-a-team/build_docx.py` |
| [02-fde-research.md](./02-fde-research.md) | FDE (Forward Deployed Engineer) 조사와 AX Team 적용안. 기획안 업무 9번 (현업 배치 개발)의 근거와 운영안 | 초안, 운영 방향 반영 (2026-11-26) |
| `00-goal-brief.md` | 이 기획안을 만든 작업 지시서 (배경과 전제) | 참고 |

## 작업 기록 (`_work/`)

| 파일 | 내용 |
|------|------|
| `_work/01-team-proposal-detail.md` | 상세본 (항목별 근거, 파견 운영 원칙, 리스크, 추진 방식 비교). 질의 대비용 |
| `_work/open-questions.md` | **아침에 먼저 볼 것**, 가정과 확인할 질문 |
| `_work/progress.md` | 단계별 진행 기록, 반영하지 않은 지적과 이유 |
| `evidence_사내용.md` | 근거로 쓴 문장의 출처. 사내용 수치가 들어 있어 커밋하지 않음 (`*_사내용.md`는 gitignore, 로컬에만 있음) |
| `_work/map.md` | 지형도 (부문장의 판단 축, 파트와 팀의 차이, 모르는 것) |
| `_work/candidates.md`, `_work/list-v1.md` | 후보 35개와 1차로 좁힌 11개 |
| `_work/grilling.md` | 부문장 입장의 반론 15개와 답 |
| `_work/codex-log.md` | Codex와 상의한 3회 기록 (전문은 `codex-r1~r3.md`) |
| `_work/council-1.md`, `_work/council-2.md` | 카운슬 2회 전문 |

## 다시 만들기

```bash
python3 as-a-team/build_docx.py   # my-task/ 에서 실행. 01-team-proposal.docx 생성
```

## 휴대폰, 태블릿에서 읽기

[reading.html](./reading.html)을 브라우저로 열면 폴더 문서를 한 페이지로 읽을 수 있음 (저장소 루트에서 `python3 scripts/build_folder_html.py my-task/as-a-team`으로 생성, 손으로 고치지 않음). Artifact 게시는 사용자 확인 뒤에 함.
