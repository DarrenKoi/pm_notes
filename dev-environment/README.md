---
tags: [dev-environment, terminal, tools]
level: beginner
last_updated: 2026-07-22
---

# 개발 환경 (Dev Environment)

> 개발 작업에 필요한 터미널, 도구, 환경 설정을 체계적으로 정리하는 섹션

## 왜 필요한가? (Why)

- 효율적인 개발을 위해 터미널과 도구 사용법을 숙지하는 것은 기본기에 해당
- 반복 작업을 자동화하고 생산성을 높이기 위한 기반 지식

## 하위 문서

| 주제 | 설명 |
|------|------|
| [회사에서 Git과 GitLab으로 협업하기](./git/README.md) | fetch·merge·rebase 차이, 작업 브랜치와 MR, 충돌 해결 실습 |
| [터미널 필수 명령어](./terminal/README.md) | Mac 터미널(zsh) 기본~중급 명령어 정리 |
| [Codex CLI 실전 가이드](./codex/README.md) | Codex를 터미널에서 효율적으로 사용하는 기능/워크플로우 정리 |
| [Pi 코딩 에이전트 실전 가이드](./pi/README.md) | Pi 설치 이후 사용법, 세션·안전 운영, PC 개발용 패키지 추천 |
| [UI 특화 VLM 가이드](./vlm/README.md) | 폐쇄망에서 UI VLM 다운로드, 전송, 서빙 가이드 |

## 관련 문서

- [루트 README](../README.md)

## 2026-10-04 읽기 순서와 검토 안내

1. 명령 실행의 기초는 [터미널 목차](./terminal/README.md)의 네 문서를 순서대로 읽는다.
2. 코드 협업은 [Git 안내](./git/README.md)를 따른다. 이 문서는 작업 전 사용자 변경으로 보호했다.
3. 에이전트 CLI는 [Codex 버전별 계약](./codex/current-cli-contract.md)을 먼저 확인한 뒤 기존 실전 예제로 이어간다. [Pi](./pi/README.md)는 별도 런타임이며 Codex 설정을 공유하지 않는다.
4. 외부 기기에서 개발할 때는 [원격 접속 목차](./remote-coding-setup/README.md)를 읽는다.
5. 이미지 모델 실험은 [VLM 목차](./vlm/README.md)에서 UI grounding과 문서 OCR을 구분해 시작한다.

기존 사용자 작성 내용과 링크는 보존했다. 파일별 검토 결과, 확인한 출처, 미확인·보류 항목은 [정리 기록](./organization-log.md)에 있다. CLI 도움말 확인은 모델 호출·원격 서버·GPU 실측의 성공을 뜻하지 않는다.
