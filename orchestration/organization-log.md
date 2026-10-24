---
tags: [orchestration, organization-log]
reviewed_on: 2026-10-04
review_status: partial
document_type: maintenance
category_major: "에이전트 오케스트레이션"
category_middle: "문서 관리"
category_minor: "정리·검증 기록"
note_kind: "관리 기록"
classified_on: "2026-10-05"
---

# orchestration 정리 기록

## 2026-10-04 정리

### 범위와 결정

다음을 검토했다.

- 공개 Markdown 6개 (학습 문서와 템플릿)
- 숨김 에이전트 기록 Markdown 2개
- 실행 파일과 설정 파일 5개

Markdown만 고쳤고, [검토 보충](./review-notes.md)과 이 기록을 새로 만들었다. 삭제, 이동, 폴더 간 링크는 없다. 원샷, 반복 스케줄, 정책, 설정 프롬프트는 용도가 달라서 모두 유지했다. 설정 JSON이 문서에 중복으로 들어 있는 것은 복사해서 붙여넣는 프롬프트라는 목적이 있어서 그대로 두고, 값이 같은지만 검사했다. 문서를 크게 나누거나 합치는 판단은 Claude와 협의하지 못해서 보류했다.

공식 문서, 설치된 패키지 자료, 코드로 확인한 오류와 과도한 일반화만 고쳤다. 2026-09-22 사내 성공 기록과 pi 0.86.1 당시의 확인 기록은 당시 사실의 기록으로 남겼고, 이번 검증 결과로 바꿔 쓰지 않았다. 실행 템플릿은 이번 작업에서 실행할 명령이 아니었다. `night-run.ps1` 구현 변경, 설정 적용, 스케줄 등록, 커밋과 push는 하지 않았다.

### 근거

2026-10-04에 다음을 대조했다.

- 로컬 pi-subagents 0.75.0의 `package.json`과 `docs/agents.md`, `docs/models.md`, `docs/configuration.md`
- Pi 공식 보안 문서
- Microsoft의 `schtasks`, `Start-Process` 설명
- Git 공식 `git-add` 설명

링크와 적용 조건은 [검토 보충](./review-notes.md)에 모았다. 사내 모델 별칭, 성능, 한도, 그리고 패키지 전체 릴리스 비교는 확인하지 못했다. npm 웹 조회가 403으로 막혀서 설치된 패키지 자료로 대신했다.

### Claude 협의 결과

`HERDR_ENV=1` 환경에서 현재 pane 조회를 다시 시도했지만 `pane_not_found`였다. 전용 Claude pane에 연결하거나 제어하지 않았고, 다른 주제의 pane도 건드리지 않았다. 의견은 받지 못했다. 그래서 야간 래퍼 교체, 엔진 설정, 역할 재설계, 큰 문서 통합은 보류했다. 고치지 않은 코드의 위험은 기록으로 남기고, 문서 검증과 로컬 검증은 따로 진행했다.

### 문서별 결과

| 문서 | 검토 결과 |
|---|---|
| [README.md](./README.md) | 읽기 순서 추가. 비용, 모델 품질, 429 재시도 성공에 대한 일반화를 고침. 과거 사내 검증 기록은 유지. 현재 엔드포인트는 미확인. |
| [office-setup.md](./office-setup.md) | `PROMPT-2`와 `PROMPT-3`의 바깥 fence를 백틱 4개로 바꿔 안쪽 JSON과 함께 복사되게 함. 두 JSON은 원본 스니펫과 값이 같음. 실행 지시는 수행하지 않음. |
| [night-setup.md](./night-setup.md) | 래퍼 적용 보류를 등록 단계 바로 앞에 명시. 커밋하지 않은 변경은 누가 바꿨는지 확인하고 자기 변경만 되돌리도록 템플릿을 고침. Windows 미검증. |
| [oneshot.md](./oneshot.md) | 한 번 실행과 반복 실행의 용도가 섞인 부분을 고침. `git add -N`을 배정 경로로 제한. 정책과 예제는 모두 보존. |
| [agents-md.snippet.md](./agents-md.snippet.md) | `git ls-files`와 ignore 규칙의 차이를 구분. `git add -N`의 경로를 제한. 상속은 실제로 해석된 역할 설정을 따른다고 명시. |
| [decisions.example.md](./decisions.example.md) | 결정 정책 템플릿 보존. 프롬프트에 고정해도 OS 경계는 아니라는 점, 확신도는 경험칙이라는 점을 검토 보충에 명시. |

새로 만든 검토 보충은 버전과 정적 한계를 모은 대표 문서이고, 이 기록은 유지보수 문서다. 원래 문서마다 같은 검토일과 partial 상태를 넣고 검토 보충으로 연결했다.

### 실행 파일과 설정 파일 검토: 수정 없음

| 파일 | 결과 |
|---|---|
| [merge-settings.py](./merge-settings.py) | 임시 파일로 병합 검증 통과. cadence 삭제, 원자적 쓰기, 동시 편집 처리는 구현되지 않음. |
| [smoke.sh](./smoke.sh) | bash 문법 검사 통과. 추출한 L0 fixture에서 정상 판정과 실패 검출 통과. 실제 L1~L3의 증거 강도는 검토 보충에서 구분. |
| [night-run.ps1](./night-run.ps1) | 잠금 전 job 시작, 죽은 소유자의 잠금 정리, 명령 문자열로 프로세스를 고르는 방식, 종료 코드 전달의 한계를 확인. 적용 보류. |
| [settings.snippet.json](./settings.snippet.json) | JSON 파싱 통과, office 템플릿과 값 일치. 실제 사내 게이트웨이에서는 실행하지 않음. |
| [subagent-config.snippet.json](./subagent-config.snippet.json) | JSON 파싱 통과, office 템플릿과 값 일치. Windows와 실제 런타임 상한은 실행하지 않음. |

### 검증 1: 문서 목록과 고유 내용

공개 Markdown은 6개에서 8개로 늘었다. 원래 문서, 경로, 고유 프롬프트, JSON 예제를 유지했다. 실행 파일과 설정 파일 5개는 작업 시작 시와 SHA256이 같았다. 크게 다시 쓰는 대신 부분 수정과 검토 보충으로 내용 손실을 피했다.

### 검증 2: 기술 동작

임시 파일로 `merge-settings.py`를 확인했다.

- 미리보기는 파일을 바꾸지 않는다.
- `--apply`는 깊은 병합을 하고, 배열은 교체하고, 원래 키를 보존하고, 백업을 남긴다.
- 같은 내용을 다시 적용하면 바뀌는 것이 없다.
- 대상 JSON이 깨져 있으면 멈춘다.
- 스니펫에서 생략한 cadence는 그대로 남는다.

`smoke.sh`는 Python L0 부분만 꺼내서, 가상의 provider 2개와 모델 5개로 FAIL 0을 확인했다. `defaultModel`에 잘못된 provider와 id를 넣으면 FAIL을 검출했다. 이것은 실제 pi 모델 레지스트리 검증이 아니다.

`bash -n`, Python AST, JSON 2개 파싱, 문서 안 JSON 2개와 스니펫의 값 비교가 통과했다. 네트워크 모델 호출, 사용자 설정 읽기와 쓰기, Windows 실행, pwsh 문법 검사는 하지 않았다.

### 검증 3: 참조, 메타데이터, Obsidian

결과는 아래 [최종 확인](#최종-확인)에 있다.

### 보류

남은 것은 Claude 협의, 실제 사내 모델과 속도 제한, Windows PowerShell과 작업 스케줄러, 야간 래퍼 구현 수정, 운영 격리 시험이다. 구조 검증과 로컬 검증을 마쳐도 partial 상태를 유지하고, 저장소 전체를 완료로 기록하지 않는다.

숨김 파일 `.remember/now.md`는 작업 시작 전부터 있던 0바이트 에이전트 상태 파일이다. `.remember/today-2026-09-26.md`는 당시 세션 압축 상태의 기록이다. 두 파일은 학습 문서로 다시 쓰지 않고 원문을 보존했으며, 메타데이터 검사에서만 제외했다. 숨김 상태, 로그, 캐시도 작업 시작 시의 해시를 그대로 유지했다. 공개 Markdown 8개와 별도로 숨김 기록 2개를 검토했으므로, 검토한 Markdown은 모두 10개다.

### 최종 확인

공개 문서 8개에서 링크와 앵커의 새 오류 0건, 기존 오류 0건이었고 `git diff --check`를 통과했다. Obsidian 1.13.7의 pm_notes vault에서 8개 문서의 properties와 `reviewed_on`을 확인했다. 실제 읽기 화면에서 orchestration/README에서 orchestration/review-notes로 가는 링크를 눌러, 폴더, 한국어 본문, 버전, 날짜, 태그, alias가 맞는지 확인했다. 표는 접근성 트리 구조로 확인했다. 8개 문서의 모든 화면과 Windows 터미널 렌더링을 다 확인하지는 않았다. 숨김 에이전트 기록과 캐시, 실행 파일과 설정 파일 5개는 작업 시작 시와 해시가 같았다.

## 2026-10-24 갱신: 에이전틱 코딩, 속도 제한, pi-subagents 0.81

사용자 요청: 오래 보지 않은 내용을 최신 흐름으로 갱신하고, 에이전틱 코딩을 넣고, 사내 500K TPM·50 RPM
한도와 집의 구독 요금제를 고려한 운영법을 넣는다. Codex(별도 herdr 탭)와 웹 조사를 나눠 하고 계획을 검토받았다.

| 변경 | 내용 |
|---|---|
| 신규 `agentic-coding.md` | 에이전틱 코딩의 정의, 2025~2026 합의된 실천, 멀티 에이전트 근거(Anthropic·Cognition·Google·Cursor), 실패 모드, 이 폴더 설계와의 대응 |
| 신규 `rate-limits.md` | TPM/RPM 계산, 자식 수 공식, pi 설정이 막는 것과 못 막는 것, 두 겹 재시도, 운영팀 확인 목록, 집의 구독 요금제 |
| 신규 `rate-usage.py` | pi 세션 로그에서 분당 요청·토큰을 집계. `--selftest` 포함 |
| `settings.snippet.json`·`office-setup.md` | `reviewer` `systemPrompt` 덮어쓰기(빌트인이 셸 금지), `delegate`와 외부 CLI 역할 6개 비활성 |
| `smoke.sh` | 비활성 역할 9개와 reviewer `systemPrompt` 검사 추가 |
| `README.md` | 읽기 순서 표, 0.81 동작, 속도 제한 절 요약, 모바일 읽기 절 |
| `review-notes.md` | 기준을 pi 1.4.0 + pi-subagents 0.81.0으로, 0.75→0.81 변화 표 |
| `night-setup.md` | 내장 스케줄을 미검증 대안으로 기록 |
| 각 템플릿 | 반복되던 2026-10-04 적용 조건 안내 삭제(검토 보충으로 일원화) |

검증: `bash -n smoke.sh`, 임시 fixture로 `smoke.sh -l 0`의 새 검사 정상·변형 검출 확인, `rate-usage.py --selftest`,
로컬 세션 로그 집계 실행, JSON 파싱과 `office-setup.md` PROMPT-2 JSON의 스니펫 일치. 사내 PC에서 0.81로
`PROMPT-4`~`7`을 다시 돌리는 것과 게이트웨이 운영팀 확인은 남아 있다.
