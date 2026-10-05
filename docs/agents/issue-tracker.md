---
title: GitHub 이슈 추적 운영 안내
tags: [agents, github, workflow]
aliases: [Issue tracker]
document_type: operations
reviewed_on: 2026-10-04
verification_status: partially-verified
category_major: "에이전트 운영 문서"
category_middle: "작업 운영"
category_minor: "이슈 추적"
note_kind: "운영 지침"
classified_on: "2026-10-05"
---

# GitHub 이슈 추적 운영 안내

> 문제, 요구사항과 결정의 근거를 이슈에 남겨 다음 작업자가 범위와 완료 조건을 찾게 한다.

## 목적과 적용 조건

기존 저장소 운영 안내는 `DarrenKoi/pm_notes`의 GitHub Issues를 사용한다고 명시했다. 이번 검토는 CLI 문법만 확인했으며 해당 원격의 현재 운영 상태·접근 권한·라벨 존재는 확인하지 않았다. 프로젝트의 이슈 추적기가 다르면 이 예제를 그대로 실행하지 않는다.

`gh`는 GitHub API를 호출한다. 읽기 작업과 작성 작업 모두 네트워크·인증·대상 저장소가 필요하다. 아래 예제는 `--repo`로 대상을 명시한다. 이 문서를 읽는 것만으로 이슈 작성이나 댓글 게시가 승인되는 것은 아니다.

## 읽고 범위를 확인하기

```bash
gh issue list --repo DarrenKoi/pm_notes --state open --limit 30 \
  --json number,title,body,labels,comments \
  --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'
gh issue view 123 --repo DarrenKoi/pm_notes --comments
```

`123`은 실제 이슈 번호로 바꾼다. `list`는 기본 30건만 반환하므로 위 결과를 전체 이슈 목록이라고 해석하지 않는다. 라벨로 좁히려면 `--label needs-triage`를 추가한다. [공식 list 문서](https://cli.github.com/manual/gh_issue_list), [view 문서](https://cli.github.com/manual/gh_issue_view), 확인일 2026-10-04.

## 작성하고 상태를 바꾸기

본문에 목적·영향 경로·재현 조건·완료 기준을 적는다. 여러 줄 본문은 임시 파일로 전달하면 셸 인용과 줄바꿈을 보존하기 쉽다.

```bash
cat > /tmp/pm-notes-issue-body.md <<'BODY'
문제: 재현 가능한 증상과 영향을 적는다.
범위: 변경할 모듈과 제외할 작업을 적는다.
완료 기준: 확인할 결과와 검증 방법을 적는다.
BODY
gh issue create --repo DarrenKoi/pm_notes \
  --title '구체적인 문제 제목' --body-file /tmp/pm-notes-issue-body.md
```

댓글도 `gh issue comment 123 --repo DarrenKoi/pm_notes --body-file /tmp/pm-notes-issue-body.md`로 전달할 수 있다. [공식 create 문서](https://cli.github.com/manual/gh_issue_create), [comment 문서](https://cli.github.com/manual/gh_issue_comment), 확인일 2026-10-04.

라벨과 종료 예시는 실제 대상·권한·판단 근거를 확인한 다음 실행한다.

```bash
gh issue edit 123 --repo DarrenKoi/pm_notes --add-label ready-for-agent --remove-label needs-triage
gh issue close 123 --repo DarrenKoi/pm_notes --reason completed --comment "완료 근거 요약"
```

[공식 edit 문서](https://cli.github.com/manual/gh_issue_edit), [close 문서](https://cli.github.com/manual/gh_issue_close), 확인일 2026-10-04. 실제 라벨 의미는 [분류 라벨 안내](triage-labels.md)를 따른다.

## 스킬과 연결

스킬의 "publish to the issue tracker"는 이 저장소 안내에서 GitHub 이슈 작성을 의미하고, "fetch the relevant ticket"은 해당 이슈 읽기를 의미한다. 실제 행동은 현재 사용자의 지시와 승인 범위에 따른다.

검증 환경: 로컬 `gh 2.96.0`의 help에서 `--json`, `--jq`, `--limit`, `--body-file`, `--repo`를 확인했다. 원격 API 호출·이슈 게시·댓글·라벨 변경·종료는 실행하지 않았다.
