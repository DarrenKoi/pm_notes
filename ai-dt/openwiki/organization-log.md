---
type: review-log
tags: [openwiki, documentation-review]
reviewed_on: 2026-10-04
review_status: partial
category_major: "AI·DT"
category_middle: "문서 관리"
category_minor: "정리·검증 기록"
note_kind: "관리 기록"
classified_on: "2026-10-05"
---

# OpenWiki 문서 정리 기록

## 범위와 개별 결과

[원래 README](./README.md) 1개를 전체 검토했다. 작성일 `2026-07-07`, 설치·초기화·갱신·provider별 설정·CI·운영 목적을 유지했다. 읽기 순서와 실행 방식 구분을 추가했다. 이동·통합·삭제·다른 주제 링크는 없다. 실행 코드·첨부도 없다.

## 근거와 수정 · 확인 2026-10-04

- [공식 package.json](https://github.com/langchain-ai/openwiki/blob/main/package.json): 확인한 main의 버전은 `0.7.0`, Node 조건은 `>=22.22.0`. Node 20 안내를 수정했다. npm 배포 일치·main commit SHA는 확인하지 못했으므로 최신 배포라고 단정하지 않는다.
- [공식 README](https://github.com/langchain-ai/openwiki/blob/main/README.md): host 연동과 자체 CLI의 인증을 구분했다. AGENTS 관리와 기존 CLAUDE 갱신, init 재실행의 생성 위키 교체를 보완했다. 오래된 provider 기본값을 수정하고 모델 ID는 사용 가능한 값으로 바꾸는 placeholder로 표시했다. gateway별 예제는 유지한다.
- [공식 CLI parser](https://github.com/langchain-ai/openwiki/blob/main/src/cli/commands.ts): 명령 구조를 대조했다. 실행 도움말과 모든 slash command의 실제 동작은 미검증이다.
- [GitHub CI 예시](https://github.com/langchain-ai/openwiki/blob/main/examples/openwiki-update.yml)와 [GitLab 예시](https://github.com/langchain-ai/openwiki/blob/main/examples/openwiki-update.gitlab-ci.yml): full history와 변경 지시 파일을 확인했다. 문서의 CI 템플릿에 `fetch-depth: 0`, provider 명시, 관리 파일 PR 범위를 보완했다. tracing은 필수처럼 넣지 않는다. 공식 부분 성공 보존/실패 전파와 기존 간소화 템플릿의 차이를 명시했다. action 버전·토큰 권한·CI 실행은 미확인이다.
- pm_notes 전체 루트에서 자동 생성하면 폴더 독립성·공통 지시 파일에 영향을 주므로 단일 주제의 독립 시험 Git 저장소를 사용하도록 설명했다. 브랜치만으로 도구 범위가 제한되지 않음을 명시했다. 실제 생성·설치·커밋·푸시는 하지 않았다.

## Claude 협의

동일 작업 환경은 `HERDR_ENV=1`이며 현재 작업 pane 조회가 `pane_not_found`다. 전용 Claude pane을 확보하지 못해 의견을 받지 않았다. 다른 pane 조작은 없다. 공식 문서로 확인한 사실만 수정하고 생성 문서 구조를 자동 도구로 바꿀지의 판단은 보류한다.

## 세 차례 검증

1. 원문 snapshot의 README 1개와 현재 파일을 대조했다. 원래 제목·절, provider별 고유 예제와 CI 흐름을 유지했다. 버전·모델 ID·실행 범위·CI 조건 수정은 위 기록에 남겼다.
2. 공식 README/package/parser/CI 5개 일차 자료와 대조했다. 로컬 `command -v openwiki`는 실행 파일을 찾지 못했다. 설치나 외부 모델 호출은 하지 않는다. fenced bash는 구문, CI YAML은 파싱만 검사한다. 이 검사는 CLI/API/CI 실행 성공을 뜻하지 않는다.
3. 현재 문서 2개 YAML·상대 링크·앵커 검사와 정확한 Obsidian vault properties·읽기 화면 확인 결과를 아래에 남긴다.

## 남은 미확인

OpenWiki 실제 설치/배포 버전, 모델 사용 권한, provider API, CI 실행과 token 권한, 전용 Claude 협의는 미완료다. GitHub main 자료는 후속 변경될 수 있다.

## 최종 확인

- 원래 README의 제목·소제목 모두 보존. 고유 provider 및 CI 예제는 수정 조건을 기록하고 유지했다.
- bash 10개 fence의 `bash -n`과 CI YAML 1개/metadata 2개 파싱 통과. 실행 테스트는 미완료다.
- 현재 2개 상대 링크·anchor 검사: 기존/신규 깨진 참조 0. 첨부·wiki·reference-style 참조 없음. `git diff --check` 통과.
- Obsidian 1.13.7의 정확한 `pm_notes` vault CLI properties 2개 확인. 읽기 화면에서 README의 한국어·검토 callout·읽기 순서를 확인하고 정리 기록 링크를 클릭해 `ai-dt/openwiki/organization-log`로 이동했다. 모든 아래쪽 코드 fence를 시각 검사한 것은 아니다.
- 실제 CLI/API/CI와 Claude 협의 미완료를 `partial` 상태로 남긴다.
