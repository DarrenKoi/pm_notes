# Roo Code (VS Code Extension) 학습 가이드

이 폴더는 `docs.roocode.com` 기준으로 Roo Code VS Code 확장을 빠르게 익히고, 사내 OpenAI-compatible API 환경(Kimi-K2.5, GLM-4.7)에 맞춰 실무 적용하기 위한 한국어 학습 노트입니다.

적용 원칙:
- 외부 API/클라우드 중심 예시는 제외
- 사내망에서 접근 가능한 모델/도구만 사용
- 설정 예시는 내부 엔드포인트 기준으로 작성

## 튜토리얼 비디오 기준 핵심 기능(중요도 순)

출처: https://docs.roocode.com/tutorial-videos (Last updated: Feb 11, 2026)

1. Installing Roo Code
2. Configuring Profiles
3. Setting up MCP Servers
4. Auto Approve Feature
5. Custom Modes
6. Checkpoints
7. Codebase Indexing
8. Context Condensing
9. Roo Marketplace
10. Todo Lists
11. Slash Commands
12. Browser Use
13. Code Actions
14. Terminal Integration
15. Context Management
16. Worktrees
17. Skills

## 이 폴더의 파일 구성

- `00_company_local_llm_setup.md`
  - 사내 OpenAI-compatible API 연결, Kimi/GLM 프로필 설계
- `01_setup_and_first_use.md`
  - 설치, 첫 연결, 프로필, 모드, 첫 작업
- `02_tutorial_topics_core.md`
  - 커스텀 모드, 체크포인트, 인덱싱, 컨텍스트 압축, TODO, 슬래시, 스킬
- `03_mcp_marketplace_and_browser.md`
  - MCP/Browser Use를 사내망 기준으로 구성
- `04_vscode_productivity.md`
  - Code Actions, Terminal Integration, Context Management, Worktrees
- `05_study_plan_4weeks.md`
  - 4주 학습/실습 플랜

## 먼저 이렇게 시작

1. `00_company_local_llm_setup.md`로 사내 API 연결 완성
2. `01_setup_and_first_use.md`로 오늘 바로 1개 작업 완료
3. `02`, `03`, `04`를 주제별로 실습
4. `05_study_plan_4weeks.md` 체크리스트로 반복

## 주의

- `Browser Use`는 최신 문서 구조에서 세부 페이지가 분리/이동되어, 본 가이드는
  - 튜토리얼 비디오 주제 목록
  - `browser_action` 도구 문서(legacy path)
  - 최신 Auto-Approve/Terminal/Context 문서를 함께 참고해 정리했습니다.
