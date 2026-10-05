---
reviewed_on: 2026-10-04
review_status: partial
document_type: learning
tags: [remote-development, tailscale, ssh]
category_major: "개발 환경"
category_middle: "원격 개발"
category_minor: "서버·클라이언트·연결"
note_kind: "목차"
classified_on: "2026-10-05"
---

# 원격 개발 환경 읽기 순서

Tailscale은 기기 간 네트워크 경로, SSH/Termius는 원격 셸, code-server는 브라우저 편집, tmux는 서버 측 터미널 세션을 담당한다. 서로 다른 장애 지점을 순서대로 확인한다.

1. [목표와 역할](./00-overview.md): 원격 셸과 브라우저 편집을 구분한다.
2. [Mac Mini 설정](./01-mac-mini-setup.md): SSH 키, 서비스와 접속 포트를 준비한다.
3. [Galaxy Tab 설정](./02-galaxy-tab-setup.md): 클라이언트·키보드·브라우저를 준비한다.
4. [연결 점검](./03-connection-guide.md): 네트워크→SSH→프로젝트→편집기 순서로 진단한다.

## 적용 조건과 확인된 경계

기존 직접 접속 예제는 `0.0.0.0:8080`에서 대기한다. **Tailscale 전용 노출을 강제하지 않는다.** HTTP에 TLS도 없다. tailnet 접속 정책과 서버 bind/firewall 범위를 각각 확인해야 한다. [code-server 공식 FAQ](https://coder.com/docs/code-server/FAQ)는 loopback·password·TLS 없음이 기본임을 설명한다.

loopback backend를 tailnet HTTPS로 전달하는 [Tailscale Serve](https://tailscale.com/docs/reference/tailscale-cli/serve) 경로도 있다. 기존 직접 접속 URL을 모두 바꾸는 정책 전환은 Claude 협의가 불가능하여 보류했다. 실제 서버·태블릿 설정은 수정하거나 연결하지 않았다. 확인일: **2026-10-04**.

tmux는 서버 프로세스가 살아 있는 동안 SSH 연결 종료와 작업 수명을 분리한다. 서버 절전·재부팅·프로세스 종료 뒤 복구를 보장하지 않는다. 앱 로그인·북마크·PWA도 세션 유지를 보장하지 않는다.

## 근거와 상위 목차

- 로컬 `man ssh-keygen`, `man tmux`, `man grep`.
- [Tailscale Serve](https://tailscale.com/docs/reference/tailscale-cli/serve), [code-server FAQ](https://coder.com/docs/code-server/FAQ).
- [개발 환경 목차](../README.md), [정리 기록](../organization-log.md).
