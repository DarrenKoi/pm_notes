---
tags: [orchestration, verification, pi-subagents]
aliases: [Pi 오케스트레이션 적용 조건]
reviewed_on: 2026-10-24
review_status: partial
document_type: learning
category_major: "에이전트 오케스트레이션"
category_middle: "실행 환경"
category_minor: "버전·적용 조건"
note_kind: "검토 기록"
classified_on: "2026-10-05"
---

# 실행 전에 읽는 검토 보충

> 이 폴더의 설정과 템플릿이 어떤 버전을 기준으로 하는지, 무엇이 확인됐고 무엇이 남았는지 정리한 참조 문서. 실행 전에 읽는다.

권장 읽기 순서는 [목차](./README.md) → [사내 PC 세팅](./office-setup.md) → [결정 정책](./decisions.example.md) → [퇴근 원샷](./oneshot.md)이다. [야간 세팅](./night-setup.md)은 반복 스케줄용이고, 쓰기 전에 아래 [야간 래퍼의 한계](#야간-래퍼는-운영-검증-보류)를 먼저 확인한다. 문서를 갱신하는 동안에는 사내 모델 호출, 사용자 설정 병합, 스케줄러 등록을 하지 않았다.

## 버전과 근거

현재 기준은 **pi 1.4.0 + pi-subagents 0.81.0**(2026-10-24 로컬 설치)이다. 이전 기준은 2026-10-04의 pi-subagents 0.75.0이었다. 대조한 파일은 다음과 같다.

- `package.json`
- `docs/agents.md`, `docs/models.md`, `docs/configuration.md`, `docs/missions.md`, `docs/watchdog.md`
- 빌트인 역할 파일 `agents/*.md`

역할 파일은 사용자나 프로젝트 설정으로 덮어쓸 수 있다. 빌트인 이름만 보고 실제 도구와 상속값을 단정하지 않는다.

0.75에서 0.81 사이에 이 폴더에 영향을 준 변화는 다음과 같다.

| 변화 | 영향 | 반영 |
|---|---|---|
| 빌트인 `reviewer` 프롬프트가 셸 사용을 금지 | `bash`만 돌려주면 검증을 다시 돌리지 않는다 | `systemPrompt` 덮어쓰기 추가, `smoke.sh` 검사 |
| 외부 CLI 역할 6개(`claude-code`, `codex-exec`, `cursor-agent`와 각 `-writer`) | 사내 게이트웨이를 거치지 않는 실행 경로 | `disabled: true` |
| `delegate` 역할 | 모델 제한 없는 쓰기 역할. 지정이 없으면 `subagents.defaultModel`(Medium)을 쓴다 | `disabled: true` |
| `oracle`의 `defaultContext: fork` | 부모 대화를 물려받는다. 새 컨텍스트가 아니다 | README에 명시. `reviewer`는 프롬프트로 fresh를 요구 |
| `toolActivation: "auto"` | 새 세션에 `subagents_enable`만 보일 수 있다 | `PROMPT-5`, `PROMPT-6`에 대응 방법 안내 |
| 내장 스케줄(0.76부터) | `night-run.ps1`의 대안 후보 | `night-setup.md`에 미검증 대안으로만 기록 |
| `usageBudget`, `toolBudget` | 실행 단위의 토큰 상한과 도구 호출 상한 | [속도 제한 아래에서 오케스트레이션하기](./rate-limits.md) |
| 도구 정의 축소(0.77), 프롬프트 캐시 재사용(0.78~0.79) | 부모 요청당 토큰 감소 | 같은 문서 |
| Windows 시작 속도, 임시 폴더 권한 수정(0.80~0.81) | Windows에서 pi 시작이 빨라지고 멈춤이 사라짐 | 해당 없음 |

원본은 [pi-subagents 저장소](https://github.com/nicobailon/pi-subagents)와 [모델 설정 문서](https://github.com/nicobailon/pi-subagents/blob/main/docs/models.md)다. 저장소의 main 브랜치와 설치된 0.81.0이 같은 버전이라는 보장은 없다. 최신 릴리스 전체는 확인하지 않았다.

Pi 공식 보안 문서에 따르면, 도구와 확장은 실행한 사용자의 권한으로 동작하고 project trust는 도구 호출의 샌드박스가 아니다. 작업 폴더, 프롬프트 정책, 리뷰, 워크트리는 운영체제 권한을 제한하지 않는다. [공식 보안 문서](https://pi.dev/docs/latest/security) (2026-10-04 확인).

2026-09-22 사내 성공 기록은 당시 작성자의 기록으로 보존했다. 다음은 이번에 접속해서 다시 확인하지 않았다.

- HCP 별칭 뒤의 실제 모델
- context와 maxTokens 값
- RPM과 TPM의 집계 단위
- 실제 Windows 설치 버전

모델 크기나 계열만으로 품질, 처리량, 오류의 독립성이 보장되지는 않는다.

## 현재 파일에서 확인한 한계

| 파일 | 확인 결과 | 적용 방법 |
|---|---|---|
| `merge-settings.py` | 없는 키를 추가하고 깊은 병합을 한다. 배열은 교체하고, 생략한 키는 보존한다 | 미리보기를 확인한다. cadence 같은 키의 삭제는 따로 지시한다. `.bak`은 기존 백업을 덮어쓸 수 있다 |
| `merge-settings.py` | 파일을 직접 열어 쓴다. 동시 편집 처리와 원자적 교체는 구현되지 않았다 | 다른 프로세스가 같은 파일을 고치는 중이면 적용하지 않는다 |
| `smoke.sh` L0 | 설정 교차 검사와 모델 목록 대조 | warn과 skip도 보고한다. 종료 코드 0만으로 모든 항목이 확인됐다고 보지 않는다 |
| `smoke.sh` L1 | 역할 모델과 대안 모델 목록을 순회 | 호출 수는 고정 3회가 아니라 설정에 따라 달라진다 |
| `smoke.sh` L2 | 출력에 숫자 3이 있는지 검사 | 실제 `ls` 호출과 thinking 전달의 증거는 실행 기록에서 따로 확인한다 |
| `smoke.sh` L3 | 결과 파일이 있는지 검사 | 부모가 직접 썼을 가능성을 배제하지 못한다. 자식 실행 기록과 모델 기록도 확인한다 |

위 결과는 저장소 코드를 읽고 임시 fixture로 검증한 범위다. 스니펫은 운영 파일에 자동으로 적용되지 않는다. 공식 패키지가 지원하는 것과 사내 게이트웨이가 받아 주는 것도 별개다.

## 야간 래퍼는 운영 검증 보류

> [!warning] 무인 실행이 겹치지 않는다는 것과, 깔끔하게 끝난다는 것을 보장하지 않는다
> `night-run.ps1`은 예제로 보존했다. 지금 상태로는 검증된 래퍼로 쓰지 않는다.

코드를 직접 읽어 확인한 문제는 네 가지다.

1. `Start-Job`이 `CreateNew` 잠금보다 먼저 실행된다. 두 회차가 동시에 시작하면, 잠금이 실패하기 전에 두 작업이 모두 돌 수 있다. 잠금 확인 뒤의 `Remove-Item`에도 경쟁 상태가 있다.
2. 남은 자식 때문에 잠금을 남겨 둘 때, 잠금의 소유 PID는 래퍼의 PID다. 래퍼가 끝난 뒤 다음 회차가 이 잠금을 죽은 잠금으로 보고 지울 수 있다. 그래서 잠금을 남겨도 다음 실행이 막힌다는 보장이 없다.
3. Repo 문자열로 프로세스를 찾는 방식은 부모와 자식 프로세스 트리를 정확히 따라가지 못한다. 다른 프로세스를 고르거나 실제 자식을 놓칠 수 있다.
4. job 실패나 pi의 종료 코드가 최종 실패 상태로 충분히 전달되지 않는다. "회차가 끝났다"와 "작업이 성공했다"를 구분해야 한다.

실행 코드는 문서 작업 범위에서 고치지 않았다. 적용 여부는 전용 Claude 협의를 거치고, Windows에서 잠금 경쟁, 강제 종료, 남는 자식, 종료 코드를 시험한 뒤에 판단한다. macOS에 pwsh가 없어서 PowerShell 문법 검사와 실행 검증도 하지 못했다.

`schtasks /IT`는 그 사용자가 로그온해 있을 때만 실행하는 옵션이다. 절전 방지, 환경 변수, 키 접근, 실행 성공을 보장하지 않는다. `/F`는 같은 이름의 작업을 덮어쓸 수 있다. [Microsoft 옵션 설명](https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/schtasks-create) (Windows 10/11, Server 문서, 2026-10-04 확인).

PowerShell은 셸 버전마다 인자 인용 방식을 확인해야 한다. README의 `Start-Process` 예제는 여러 줄 프롬프트 전달과 자식 종료를 검증하지 않았다. [공식 ArgumentList 설명](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.management/start-process?view=powershell-7.5).

## 템플릿을 복사할 때

`git add -N`도 인덱스를 바꾼다. 템플릿에서는 배정된 경로로 한정해서 쓰며, 이번 정리에서는 실행하지 않았다. 새 파일만 확인하려면 상태 목록과 해당 파일 내용을 먼저 읽는다. [Git 공식 설명](https://git-scm.com/docs/git-add).

작업 트리에 커밋하지 않은 변경이 있으면 사용자가 고친 것일 수 있다. 누가 바꿨는지 확인하지 않고 한꺼번에 되돌리지 않는다. 워크트리를 지우기 전에는 필요한 결과와 커밋하지 않은 파일을 확인한다.

정책 본문을 프롬프트에 고정하면 파일 바꿔치기로 인한 혼동은 줄지만, 모델이 정책을 지키도록 강제하지는 못한다. "확신도 60%" 같은 숫자는 보정된 확률이 아니라 템플릿의 경험칙이다. 복구, 전송, 삭제를 실제로 제한하려면 실행 환경의 권한 정책이 필요하다.
