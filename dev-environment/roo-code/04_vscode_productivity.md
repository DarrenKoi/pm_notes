# 04. VS Code 생산성 극대화

튜토리얼 주제 13, 14, 15, 16 중심 문서입니다.

## 1) Code Actions (주제 13)

핵심:
- 전구(Quick Fix), 우클릭, Command Palette에서 Roo 액션 실행
- 주요 액션: Add to Context, Explain, Improve, Fix, New Task

강력한 사용법:
- 코드 선택 후 `Add to Context`로 파일/라인을 정확히 넘기기
- 오류 줄에서는 `Fix Code`, 정상 코드에서는 `Improve/Explain`로 분기

실전 루틴:
1. 문제 줄 선택
2. `Fix Code`
3. 제안 diff 검토
4. 필요한 수정만 승인

## 2) Terminal Integration (주제 14)

핵심:
- Roo가 명령 실행 흐름과 출력 상태를 더 안정적으로 처리
- timeout, command delay, fallback 동작을 설정으로 조정 가능

문제 해결 포인트:
- 통합 실패 시 shell 설정에 수동 integration hook 추가
- WSL 사용 시 WSL 내부에서 `code .`로 여는 방식이 안정적
- 멀티라인 명령 대신 `&&` 체인 권장

## 3) Context Management (주제 15)

핵심 1: Context Mentions (`@`)
- `@/file`, `@/folder/`, `@problems`, `@terminal`, `@git-changes`, `@commit`, `@url`

핵심 2: Context Condensing
- 긴 대화에서 컨텍스트 유지력을 높이고 오류 복구 자동화

베스트 프랙티스:
- 작업 시작 시 핵심 파일/에러를 mentions로 명시
- 대화 길어지면 condense 수동 실행 + 보존 규칙 커스터마이즈

## 4) Worktrees (주제 16)

핵심:
- 브랜치 전환 없이 병렬 작업 공간 생성
- 같은 저장소로 구현안 A/B, 핫픽스/기능개발 병행 가능

추천 운영:
- 브랜치 명명 규칙 통일 (`worktree/기능명`)
- `.worktreeinclude`로 untracked 파일 복제 정책 정의
- 완료된 worktree는 주기적으로 정리

## 5) 실제로 가장 생산성 높은 조합

조합 A (버그 수정):
- `@problems` + Code Action(Fix) + Debug 모드 + Terminal Integration

조합 B (대규모 기능):
- Architect 모드(todo 생성) + Worktree 분기 + Code 모드 구현 + Checkpoint 복원

조합 C (반복 업무 자동화):
- Slash Command + Skills + MCP 서버
