---
tags: [codex, cli, sandbox, approval, profiles, config]
level: intermediate
last_updated: 2026-03-07
status: complete
---

# 컨텍스트, 샌드박스, 프로필 (Context, Sandbox, Profiles)

> Codex 품질은 모델보다도 `무엇을 볼 수 있는지`, `무엇을 실행할 수 있는지` 설정에 크게 좌우된다.

## 작업 범위 제어

### `-C`로 작업 루트 고정

```bash
codex -C web-development/python/flask/job-scheduler
```

가장 먼저 고려해야 할 옵션이다. 루트가 너무 넓으면 관련 없는 파일을 읽고 토큰을 낭비한다.

### `--add-dir`로 추가 쓰기 범위 허용

```bash
codex -C web-development/python/flask/job-scheduler \
  --add-dir /tmp
```

출력 폴더, 생성 파일 위치가 따로 필요할 때만 최소한으로 연다.

## 샌드박스 정책

Codex는 모델이 셸 명령을 실행할 수 있으므로, 샌드박스 선택이 중요하다.

| 모드 | 의미 | 추천 상황 |
|------|------|-----------|
| `read-only` | 읽기 위주, 파일 수정 불가 | 코드 탐색, 분석, 리뷰 |
| `workspace-write` | 작업 폴더 안에서 수정 가능 | 일반적인 개발 작업 기본값 |
| `danger-full-access` | 광범위한 접근 허용 | 외부에서 이미 격리된 환경에서만 제한적으로 |

예시:

```bash
codex -s read-only "구조만 파악하고 수정은 하지 마."
codex -s workspace-write "테스트 실패를 고치고 검증해줘."
```

## 승인 정책

| 정책 | 의미 | 추천 상황 |
|------|------|-----------|
| `untrusted` | 신뢰된 명령만 자동 허용 | 보수적으로 시작할 때 |
| `on-request` | 필요 시 모델이 승인 요청 | 대화형 작업 기본값 |
| `never` | 승인 없이 진행, 실패를 바로 모델에 반환 | 비대화형/자동화 |

예시:

```bash
codex -a on-request
codex exec -a never "pytest 실패 원인을 찾고 가능한 범위만 수정해줘."
```

## 빠른 조합

### 일반적인 실무 기본값

```bash
codex --full-auto
```

`--full-auto`는 `-a on-request` + `--sandbox workspace-write` 조합이다.

### 위험하지만 강한 모드

```bash
codex --dangerously-bypass-approvals-and-sandbox
```

이 옵션은 이름 그대로다. 로컬 머신 안전성을 스스로 보장할 수 있을 때만 고려한다.

## 프로필과 설정 오버라이드

Codex는 `~/.codex/config.toml` 설정을 읽고, 실행 시 `-p`와 `-c`로 덮어쓸 수 있다.

### 프로필 사용

```bash
codex -p work
codex exec -p review "변경분 리뷰해줘."
```

팀/용도별로 기본 모델, 샌드박스, 승인 정책을 분리할 때 적합하다.

### 일회성 오버라이드

```bash
codex -c model='"o3"' -c shell_environment_policy.inherit=all
```

짧은 실험이나 특정 작업에서만 설정을 바꾸고 싶을 때 사용한다.

## 모델과 입력 확장

### 모델 선택

```bash
codex -m o3
codex --oss
codex --oss --local-provider ollama
```

로컬 모델은 민감 데이터나 네트워크 제약이 있을 때 유용하지만, 코드 수정 품질과 도구 사용 능력은 비교가 필요하다.

### 이미지 첨부

```bash
codex -i ./screenshot.png \
  "이 UI 스크린샷을 기준으로 레이아웃 문제를 추정해줘."
```

스크린샷 기반 버그 리포트, 디자인 수정 요청에서 효과가 크다.

### 웹 검색

```bash
codex --search \
  "Nuxt 최신 문서 기준으로 이 경고 메시지 해결책을 확인해줘."
```

최신 외부 정보가 필요한 작업에서만 켜는 편이 좋다.

## 추천 설정 원칙

1. 기본은 `workspace-write` + `on-request` 또는 `--full-auto`
2. 큰 저장소에서는 항상 `-C`로 범위를 줄인다
3. 자동화는 `exec` + `-a never` 쪽이 일관성이 좋다
4. 위험한 모드는 편의보다 추적 가능성과 안전성을 먼저 따진다

## 관련 문서

- [Codex CLI 실전 가이드](./README.md)
- 이전: [비대화형 실행과 리뷰](./02_exec_and_review.md)
- 다음: [세션 재사용, MCP, 고급 기능](./04_sessions_mcp_advanced.md)
