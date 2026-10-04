---
title: 에이전트 문서 정리 기록
tags: [agents, maintenance]
document_type: maintenance
reviewed_on: 2026-10-04
status: partial
---

# 에이전트 문서 정리 기록

## 범위와 결정

기존 Markdown 7건을 검토했다. 운영 안내 3건은 한국어로 목적·사용법·적용 조건을 정리했고, 과거 설계·계획 4건은 원문을 보존했다. 파일 이동·삭제·중복 통합은 없다. 설계는 목적·범위, 계획은 구현 단계·예제를 설명하므로 별도 문서로 유지한다.

## 문서별 검토 결과

| 문서 | 결과 |
|---|---|
| [도메인 안내](agents/domain.md) | 영문을 한국어로 정리. 루트 규칙에 따라 문맥 탐색을 폴더별 순차 작업으로 명확히 함 |
| [이슈 추적](agents/issue-tracker.md) | 대상 저장소 명시, 본문 파일 전달, 기본 조회 30건 제한 설명. 현재 원격 운영 상태는 미확인 |
| [분류 라벨](agents/triage-labels.md) | 기존 다섯 라벨 유지. 판단 기준과 권한 경계 명시. 실제 원격 라벨은 미확인 |
| [6월 재편 설계](superpowers/specs/2026-06-30-smart-align-agent-reorg-design.md) | 과거 설계 표지 추가. 기존 본문 보존. 현재 구현 상태 재검증 안 함 |
| [6월 재편 계획](superpowers/plans/2026-06-30-smart-align-agent-reorg.md) | 당시 삭제·커밋 계획 실행 금지 안내. 예제·경로·체크박스 원문 보존 |
| [7월 리더 설계](superpowers/specs/2026-07-28-ai-terms-html-reader-design.md) | 과거 설계 표지 추가. 기존 본문 보존. 당시 기술 조건은 현재 보증이 아님 |
| [7월 리더 계획](superpowers/plans/2026-07-28-ai-terms-html-reader.md) | 영문 계약·코드 예제 원문 보존. 한국어 목차와 상태 안내 추가 |
| [README](README.md) | 운영 안내 읽기 순서와 설계/계획 역할을 구분한 목차 신설 |
| 이 기록 | 각 문서의 검토 범위와 미확인 상태를 기록 |

## 근거와 기술 확인

확인일 2026-10-04. GitHub CLI 공식 문서의 명령·옵션을 확인했다.

- [list](https://cli.github.com/manual/gh_issue_list): `--json`, `--jq`, `--limit` 기본 30, `--state`, `--repo`.
- [view](https://cli.github.com/manual/gh_issue_view): `--comments`, `--json`, `--repo`.
- [create](https://cli.github.com/manual/gh_issue_create)와 [comment](https://cli.github.com/manual/gh_issue_comment): `--body-file`.
- [edit](https://cli.github.com/manual/gh_issue_edit): `--add-label`, `--remove-label`.
- [close](https://cli.github.com/manual/gh_issue_close): `--reason`, `--comment`.

로컬 `gh --version`은 2.96.0(2026-07-02), 해당 여섯 명령의 `--help`와 대조했다. 네트워크 작성 예제는 문법 확인이며 원격에 실행하지 않았다. 과거 계획의 다른 기술 주장은 현재 사실로 수정하지 않고 미확인 상태를 붙였다.

## Claude 협의

`HERDR_ENV=1`이나 현재 호출 pane이 `pane_not_found`다. 목록의 다른 저장소 pane을 이용하지 않아 협의 결과 없음. 설계/계획 통합 여부를 더 바꾸는 결정, 영문 계획의 전면 번역, 과거 예제의 현대화는 보류했다. 역할 표지와 원문 보존, 현재 안내의 공식 문법 대조는 진행했다.

## 세 단계 검증

1. 기존 목록 7건을 표와 대조. 과거 4건은 첫 H1부터 끝까지 Git HEAD와 완전 일치 확인. 운영 3건의 기존 기능(문맥/ADR, 이슈 작업 종류, 다섯 라벨)을 보존했다.
2. 변경한 CLI 예제를 공식 문서 및 로컬 2.96.0 help와 대조. 예제는 원격 실행 안 함. 과거 구현 계획의 테스트·삭제·이동·커밋 명령 실행 안 함.
3. 폴더 내 Markdown 9건의 상대 링크·앵커·메타데이터 검사를 실행하여 신규 깨진 참조 0건 확인. 기존 과거 계획의 문장 속 치환 예시 링크는 현재 파일 링크로 해석할 수 없어 보존된 참조로 기록했다. 기존 과거 기록의 타 폴더 경로는 당시 맥락으로 보존하며 신규 크로스 링크는 만들지 않았다. Obsidian 1.13.7에서 CLI가 목차의 내부 링크 8건과 frontmatter를 인식했다. 실제 읽기 화면에서 목차·표·확인일을 표시하고 이슈 추적 링크로 이동하여 한국어 본문과 코드 블록 렌더링을 확인했다. Bash 예제 세 블록은 `bash -n` 통과. 기타 문서는 화면 전수 열람 대신 파일 검사로 검증했다.
