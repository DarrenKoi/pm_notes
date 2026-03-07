---
tags: [codex, cli, resume, mcp, advanced]
level: advanced
last_updated: 2026-03-07
status: complete
---

# 세션 재사용, MCP, 고급 기능 (Sessions, MCP, Advanced)

> 익숙해지면 Codex는 "한 번 쓰고 끝나는 챗봇"보다 "작업 이력과 도구를 가진 CLI 에이전트"에 가깝다.

## 세션 이어가기와 분기

### 최근 세션 이어서 작업

```bash
codex resume --last
```

어제 하던 작업을 같은 맥락으로 바로 이어갈 때 쓴다.

### 특정 세션 이어가기

```bash
codex resume <SESSION_ID>
```

### 이전 세션을 갈라서 새 방향으로 진행

```bash
codex fork --last
```

기존 맥락은 유지하되, 다른 해결책을 실험하고 싶을 때 유용하다.

## MCP 서버 관리

Codex는 MCP(Model Context Protocol) 서버를 붙여 외부 기능을 도구처럼 사용할 수 있다.

### 현재 등록 목록 확인

```bash
codex mcp list
```

### 개별 설정 확인

```bash
codex mcp get <name>
```

### 서버 추가/삭제

```bash
codex mcp add ...
codex mcp remove <name>
```

### 로그인 연동이 필요한 서버

```bash
codex mcp login <name>
codex mcp logout <name>
```

실무에서는 이 기능으로 이슈 트래커, 문서 저장소, 내부 도구 API를 연결하는 경우가 많다.

## Codex를 MCP 서버로 사용

```bash
codex mcp-server
```

다른 MCP 클라이언트에서 Codex를 stdio 서버처럼 붙이는 용도다. 단독 사용보다 도구 체인 일부로 쓸 때 의미가 있다.

## 패치 적용

```bash
codex apply <TASK_ID>
```

다른 Codex 작업 결과가 패치 형태로 존재할 때 로컬 Git 작업 트리에 반영한다. App/Cloud 흐름과 같이 쓸 때 주로 등장한다.

## 샌드박스 자체 테스트

```bash
codex sandbox macos --help
```

Codex가 사용하는 샌드박스 환경 자체를 별도 명령으로 다룰 수 있다. 보안 정책 테스트나 동작 검증에 유용하다.

## 기타 알아둘 기능

| 기능 | 명령어 | 메모 |
|------|--------|------|
| 로그인 관리 | `codex login`, `codex logout` | 인증 상태 점검 및 전환 |
| 셸 자동완성 | `codex completion` | 자주 쓰는 환경이면 생산성 향상 |
| 기능 플래그 확인 | `codex features` | 실험 기능 노출 여부 확인 |
| 디버깅 도구 | `codex debug` | 문제 재현/진단용 |
| 데스크톱 앱 | `codex app` | CLI 외 환경이 필요할 때 |
| 클라우드 작업 탐색 | `codex cloud` | 실험적 기능 |

## 숙련자용 운영 팁

### 1. 긴 작업은 `resume`, 대안 탐색은 `fork`

문맥을 계속 설명하는 비용을 줄일 수 있다.

### 2. 조직 내 도구 연결은 MCP로 표준화

프롬프트에 API 사용법을 매번 길게 넣는 것보다 MCP 쪽이 안정적이다.

### 3. App/Cloud와 로컬 Git 연결은 `apply` 중심으로 생각

로컬 저장소가 최종 진실 원본이라면 패치 반영 경로를 명확히 유지하는 편이 좋다.

## 관련 문서

- [Codex CLI 실전 가이드](./README.md)
- 이전: [컨텍스트, 샌드박스, 프로필](./03_context_sandbox_profiles.md)
