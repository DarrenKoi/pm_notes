# 03. MCP + Browser Use (사내 전용)

이 문서는 튜토리얼 주제 3, 9, 12를 사내망 기준으로 학습합니다.

## 1) MCP 서버 설정 (주제 3)

설정 파일:
- 전역: `mcp_settings.json`
- 프로젝트: `.roo/mcp.json` (팀 공유 권장)
- 같은 서버명 충돌 시 프로젝트 설정 우선

Transport 선택:
- `stdio`: 로컬 프로세스, 지연 낮고 보안 유리
- `streamable-http`: 사내 HTTP MCP 서버가 있을 때만 사용
- `sse`: 사내 레거시 MCP 호환 시에만 사용

보안 운영:
- `alwaysAllow`는 최소 권한 원칙
- 민감 토큰은 `${env:VAR}`로 주입
- 필요 없는 tool은 `disabledTools`로 차단

## 2) 사내 MCP 템플릿

예시:
```json
{
  "mcpServers": {
    "company-docs": {
      "command": "node",
      "args": ["/opt/company-mcp/docs-server.js"],
      "alwaysAllow": ["search_docs", "get_doc_page"]
    },
    "company-jira": {
      "url": "https://mcp.internal.company/jira",
      "headers": {
        "Authorization": "Bearer ${env:COMPANY_JIRA_TOKEN}"
      },
      "disabledTools": ["delete_issue"]
    }
  }
}
```

운영 원칙:
- 팀 공통 MCP는 프로젝트 `.roo/mcp.json`에 고정
- 개인 실험용 MCP만 전역 설정 사용
- 툴 단위 allow/deny를 명확히 분리

## 3) 주제 9 대응: 사내 카탈로그 운영

외부 접근 없이 운영하려면:
- 승인된 MCP/Mode를 사내 Git 저장소에서 배포
- 팀 템플릿 레포로 `.roo/*` 파일 표준화
- 프로젝트 생성 시 템플릿을 자동 주입해 설정 편차 최소화

## 4) Browser Use (주제 12)

튜토리얼 주제의 Browser Use는 실무에서 아래 흐름으로 학습하면 된다.

핵심 도구 개념:
- `browser_action`으로 `launch/click/type/scroll/close`
- UI 테스트, 웹 플로우 검증, 스크린샷 기반 분석에 사용

중요 제약:
- 브라우저 세션이 열린 동안 다른 도구 사용이 제한될 수 있음
- `launch -> 상호작용 -> close` 순서 준수
- 좌표 클릭은 viewport 기준

Auto-Approve와 함께 쓸 때:
- Browser 권한은 별도 위험도 관리
- 프로덕션/민감 계정에서는 자동 승인 최소화

## 5) MCP + Browser 조합 패턴

패턴 A: 사내 웹 시스템 점검 자동화
1. browser_action으로 페이지 탐색
2. 필요한 데이터 요약
3. MCP 도구(사내 문서/Jira/DB)로 후처리

패턴 B: QA 회귀 테스트
1. 브랜치별 Worktree 분리
2. 브라우저 상호작용 시나리오 반복
3. 실패 로그를 `@terminal`, `@problems`로 묶어 디버그 모드에 전달
