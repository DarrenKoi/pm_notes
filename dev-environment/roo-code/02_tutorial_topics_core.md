# 02. 튜토리얼 핵심 주제 실전 적용

이 문서는 튜토리얼 비디오의 핵심 주제(5,6,7,8,10,11,17)를 실무형으로 요약합니다.

## A. Custom Modes (주제 5)

핵심 필드:
- `slug`, `name`, `description`
- `roleDefinition`, `whenToUse`
- `groups`(tool 접근권한)
- `customInstructions`

중요 포인트:
- `groups`에서 `edit`를 `fileRegex`로 제한 가능
- 예: 문서 모드는 `.md`만 수정
- 모드별 지시어는 `.roo/rules-{mode-slug}/` 폴더 방식 우선

추천 커스텀 모드 3개:
- `reviewer`: read + mcp, 편집 금지
- `doc-writer`: read + edit(.md만)
- `release-manager`: read + command(허용 prefix 제한)

## B. Checkpoints (주제 6)

핵심:
- Roo가 편집 전 상태를 shadow Git 저장소로 스냅샷
- 복원 옵션 2가지
  - Files Only
  - Files & Task(대화 이력도 롤백)

운영 권장:
- 기본 ON 유지
- 대형 저장소는 초기화 timeout 상향
- 실험적 리팩터링 전에 checkpoint 기준점 확보

## C. Codebase Indexing (주제 7)

개념:
- Tree-sitter로 코드 블록 파싱 → 임베딩 생성 → Qdrant 벡터 검색
- `codebase_search`로 의미 기반 검색

실전 구성:
- Vector DB: 사내 Qdrant 또는 local Docker
- Embedder: 사내 OpenAI-compatible 임베딩 모델

효과:
- 함수명 몰라도 "인증 흐름", "결제 재시도" 같은 의미 질의 가능
- 대형 코드베이스 온보딩 속도 상승

주의:
- `.gitignore`, `.rooignore`, 파일 크기 제한 영향 받음
- 임베딩 모델 변경 시 재인덱싱 비용 고려
- 사내망 정책상 외부 임베딩 API 호출이 차단되는지 사전 확인

## D. Context Condensing (주제 8)

핵심:
- 긴 대화에서 토큰 한도 근접 시 이전 문맥 요약 압축
- 자동 트리거 임계값/수동 실행/커스텀 condense prompt 가능
- 압축 전후 토큰/비용/요약 표시

실전 팁:
- 디버깅 시 "에러 로그/스택트레이스는 절대 생략 금지"를 custom condense prompt에 명시
- 장기 태스크는 주기적으로 수동 condense

## E. Todo Lists (주제 10)

핵심:
- 복잡 작업에서 `update_todo_list` 기반 체크리스트 자동 생성
- Architect 모드에서 특히 잘 활용됨

권장 패턴:
- 큰 요청을 한 번에 던질 때 "todo list로 단계 관리"를 함께 지시
- 승인 단계에서 할 일 설명/상태를 편집해 AI 진행 방향 고정

## F. Slash Commands (주제 11)

위치:
- 프로젝트: `.roo/commands/`
- 전역: `~/.roo/commands/`

포인트:
- 파일명 = 명령어(`/review`, `/release-check`)
- 내장 `/init`은 코드베이스 분석 + 에이전트 규칙 파일 생성에 매우 유용
- 프로젝트 명령이 전역 명령보다 우선

## G. Skills (주제 17)

핵심:
- `SKILL.md` + frontmatter(`name`, `description`)로 작업 특화 지식 패키징
- 요청과 description이 매칭될 때만 로드(프롬프트 비대화 방지)

구조:
- 전역: `~/.roo/skills/{skill}/SKILL.md`
- 프로젝트: `.roo/skills/{skill}/SKILL.md`

활용 예:
- 사내 API 문서 생성 스킬
- 특정 프레임워크 마이그레이션 스킬
- 릴리즈 체크 스킬
