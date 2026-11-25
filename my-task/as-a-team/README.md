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
| [01-team-proposal.md](./01-team-proposal.md) | AX Team 승격 기획안 (원본). 3절이 "팀이 되면 할 수 있는 일" 9가지 | 초안, `⟨확인⟩` 자리 채우기 전 |
| `01-team-proposal.docx` | 위 md로 만든 보고용 문서 | `build_docx.py`로 생성, 손으로 고치지 않음 |
| [build_docx.py](./build_docx.py) | docx 생성 래퍼 (변환기는 `arc-tf/deliverables/build_docx.py`) | `python as-a-team/build_docx.py` |
| [00-goal-brief.md](./00-goal-brief.md) | 이 기획안을 만든 작업 지시서 (배경과 전제) | 참고 |

## 작업 기록 (`_work/`)

| 파일 | 내용 |
|------|------|
| [open-questions.md](./_work/open-questions.md) | **아침에 먼저 볼 것**, 가정과 확인할 질문 |
| [progress.md](./_work/progress.md) | 단계별 진행 기록, 반영하지 않은 지적과 이유 |
| `evidence_사내용.md` | 근거로 쓴 문장의 출처. 사내용 수치가 들어 있어 커밋하지 않음 (`*_사내용.md`는 gitignore, 로컬에만 있음) |
| [map.md](./_work/map.md) | 지형도 (부문장의 판단 축, 파트와 팀의 차이, 모르는 것) |
| [candidates.md](./_work/candidates.md), [list-v1.md](./_work/list-v1.md) | 후보 35개와 1차로 좁힌 11개 |
| [grilling.md](./_work/grilling.md) | 부문장 입장의 반론 15개와 답 |
| [codex-log.md](./_work/codex-log.md) | Codex와 상의한 3회 기록 (전문은 `codex-r1~r3.md`) |
| [council-1.md](./_work/council-1.md), [council-2.md](./_work/council-2.md) | 카운슬 2회 전문 |

## 다시 만들기

```bash
python as-a-team/build_docx.py   # my-task/ 에서 실행. 01-team-proposal.docx 생성
```

## 휴대폰, 태블릿에서 읽기

게시하지 않음. `reading.html` 생성과 Artifact 게시는 사용자 확인 뒤에 함.
