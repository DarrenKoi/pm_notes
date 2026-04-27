---
tags: [codex, harness, vibe-coding, governance]
level: intermediate
last_updated: 2026-04-28
status: complete
---

# 바이브 코딩 하네스 가이드

> 바이브 코딩을 빠르게 하되, 모델이 볼 수 있는 것과 실행할 수 있는 것, 검증해야 하는 것을 명시적으로 제한하는 운영 가이드.

## 왜 필요한가? (Why)

- 바이브 코딩은 속도가 빠르지만, 작업 범위가 넓어지면 모델이 불필요한 파일을 읽거나 잘못된 방향으로 구현하기 쉽다.
- 프롬프트만 잘 쓰는 것으로는 안전성이 부족하다. 실제 품질은 컨텍스트, 권한, 도구, 검증 루프가 함께 결정한다.
- 하네스는 Codex를 "자율 개발자"처럼 방치하지 않고, "범위가 정해진 작업자"처럼 운영하기 위한 제어 장치다.

## 하네스란 무엇인가? (What)

LLM/에이전트 환경에서 하네스(harness)는 모델 주변의 제어 레이어다.

- **컨텍스트 제어**: 어떤 폴더, 문서, 지침을 읽을 수 있는가
- **권한 제어**: 파일 수정, 명령 실행, 네트워크 접근을 어디까지 허용할 것인가
- **도구 제어**: MCP, shell, 브라우저, 검색, 외부 API를 어떤 조건으로 쓸 것인가
- **출력 계약**: 어떤 형식, 품질 기준, 완료 조건을 만족해야 하는가
- **검증 루프**: 테스트, lint, typecheck, review, 수동 확인을 언제 실행할 것인가
- **관측성**: 변경 내역, 명령 로그, 실패 원인, 비용/시간을 어떻게 남길 것인가

## 추천 읽기 순서

1. [하네스 설정 맵](./01_harness_settings.md): Codex에서 하네스로 봐야 하는 설정 목록
2. [바이브 코딩 제어 워크플로](./02_vibe_coding_workflow.md): 작업 시작부터 검증까지의 운영 흐름
3. [체크리스트와 프롬프트 템플릿](./03_checklists_and_prompts.md): 바로 복사해 쓸 수 있는 점검표

## 기본 운영 원칙

1. 작업 루트는 항상 좁힌다: `codex -C <module-path>`
2. 기본 권한은 `workspace-write` + `on-request`로 둔다.
3. 분석/리뷰만 할 때는 `read-only`를 쓴다.
4. 네트워크와 외부 도구는 필요한 순간에만 켠다.
5. 변경 요청에는 완료 조건과 검증 명령을 같이 준다.
6. 큰 작업은 "탐색 -> 계획 -> 구현 -> 검증 -> 리뷰"로 쪼갠다.
7. `danger-full-access`는 격리된 disposable 환경이 아니면 쓰지 않는다.

## 추천 모드

| 상황 | 권장 모드 | 의도 |
|------|-----------|------|
| 코드 이해, 설계 검토 | `read-only` + `on-request` | 수정 없이 구조와 위험만 파악 |
| 일반 기능 구현 | `workspace-write` + `on-request` | 작업 폴더 안 수정과 검증 허용 |
| 반복 가능한 자동 점검 | `read-only` + `never` | CI/스크립트에서 질문 없이 읽기만 수행 |
| 빠른 프로토타입 | `workspace-write` + `on-request` + 좁은 `-C` | 속도는 확보하되 외부 영향 제한 |
| 위험한 대량 수정 | 먼저 `read-only`, 이후 명시 승인 | 변경 범위와 되돌릴 방법 확인 후 실행 |

## 관련 문서

- [컨텍스트, 샌드박스, 프로필](../03_context_sandbox_profiles.md)
- [세션 재사용, MCP, 고급 기능](../04_sessions_mcp_advanced.md)
- [에이전트와 스킬](../05_agents_and_skills.md)
- [Harness Engineering for LLM](../../../ai-dt/mcp/harness-engineering-llm.md)
