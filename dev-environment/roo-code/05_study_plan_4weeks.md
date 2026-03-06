# 05. 4주 학습 플랜 (튜토리얼 17주제 완주)

## Week 1 - 기반 세팅

목표:
- 설치/연결/프로필/모드 전환이 자연스러워질 것

할 일:
- Installing Roo Code
- Configuring Profiles
- Your First Task + Using Modes

체크:
- [ ] 모드별 기본 용도 설명 가능
- [ ] 프로필 2개 이상 운영
- [ ] 첫 작업 3개를 재현 가능

## Week 2 - 안전한 자동화

목표:
- 빠르지만 위험 통제된 자동화 루틴 만들기

할 일:
- Auto Approve Feature
- Checkpoints
- Todo Lists
- Slash Commands (`/init` 포함)

체크:
- [ ] auto-approve 권한을 최소권한으로 분리
- [ ] checkpoint 복원 2종류 차이 설명 가능
- [ ] 반복 작업을 slash command로 2개 이상 자동화

## Week 3 - 확장성/검색

목표:
- 코드베이스 탐색과 사내 도구 확장을 체계화

할 일:
- Setting up MCP Servers
- 사내 MCP/Mode 템플릿 배포
- Codebase Indexing
- Skills

체크:
- [ ] 프로젝트 `.roo/mcp.json`로 팀 공유
- [ ] codebase_search로 의미 기반 탐색 수행
- [ ] 스킬 1개 이상 제작/적용
- [ ] Kimi-K2.5/GLM-4.7 프로필별 역할 분리 완료

## Week 4 - 고급 생산성

목표:
- VS Code 내 병렬작업/컨텍스트/브라우저 활용 완성

할 일:
- Browser Use
- Code Actions
- Terminal Integration
- Context Management
- Worktrees

체크:
- [ ] Code Actions 중심 워크플로우 정착
- [ ] long-context 작업에서 condense 전략 운용
- [ ] worktree 2개 이상 병렬 운용

## 완료 기준(실전)

아래를 1시간 안에 재현하면 "실무 투입 가능":

1. 새 기능 요청을 Architect로 계획 + TODO 자동 생성
2. Worktree 생성 후 Code 모드 구현
3. Code Actions로 리팩터링 일부 처리
4. 실패 테스트를 Terminal/Problems mention으로 전달해 Debug 모드 해결
5. Checkpoint 복원으로 비교 검증
6. 결과를 Slash Command/Skill로 템플릿화
