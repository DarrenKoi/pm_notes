---
tags: [orchestration, verification, pi-subagents]
aliases: [Pi 오케스트레이션 적용 조건]
reviewed_on: 2026-10-04
review_status: partial
document_type: learning
category_major: "에이전트 오케스트레이션"
category_middle: "실행 환경"
category_minor: "버전·적용 조건"
note_kind: "검토 기록"
classified_on: "2026-10-05"
---

# 실행 전에 읽는 검토 보충

이 폴더는 사내 모델을 역할별로 연결하는 설정·실행 템플릿이다.
[목차](./README.md) → [낮 세팅](./office-setup.md) →
[정책](./decisions.example.md) → [원샷](./oneshot.md) 순으로 읽는다.
[야간 세팅](./night-setup.md)은 반복 스케줄용이며 아래 래퍼 한계의 확인이 선행한다.
이번 문서 정리에서는 모델 호출·사용자 설정 병합·스케줄러 등록·커밋을 하지 않았다.

## 버전과 근거

2026-10-04 로컬 설치 `pi-subagents` **0.75.0**의 package.json과
`docs/agents.md`, `docs/models.md`, `docs/configuration.md`를 대조했다.
역할 override, 모델 우선순위, 프로젝트 컨텍스트 상속, run deadline 설정은
해당 패키지 설명과 일치한다. 역할 파일은 사용자/프로젝트 override가 가능하므로
빌트인 이름만 보고 실제 도구와 상속값을 단정하지 않는다.
[공식 원본 저장소](https://github.com/nicobailon/pi-subagents),
[모델 설정 문서](https://github.com/nicobailon/pi-subagents/blob/main/docs/models.md).
main과 설치 0.75.0은 동일 버전이라는 보장이 없다. 최신 릴리스 전체 확인은 미완료다.

Pi 공식 보안 문서는 도구·확장이 실행 사용자 권한을 사용하고, project trust는
도구 호출의 샌드박스가 아니라고 설명한다. 작업 폴더·프롬프트 정책·리뷰·워크트리가
운영체제 권한을 제한하지 않는다.
[공식 보안 문서](https://pi.dev/docs/latest/security) (2026-10-04 확인).

2026-09-22 사내 성공 기록은 당시 작성자의 기록으로 보존했다. HCP 별칭 뒤의 모델,
context/maxTokens, RPM/TPM 집계 단위와 실제 Windows 설치 버전은 이번에 접속해
재확인하지 않았다. 모델 크기·계열만으로 품질·처리량·오류 독립성을 보장하지 않는다.

## 현재 파일에서 확인한 한계

| 파일 | 확인 결과 | 적용 방법 |
|---|---|---|
| `merge-settings.py` | 없는 키 추가와 깊은 병합; 배열은 교체, 생략한 키는 보존 | preview를 확인하고 cadence 같은 삭제는 별도 명시. .bak은 기존 백업을 덮어쓸 수 있음 |
| `merge-settings.py` | 파일을 직접 열어 기록; 동시 편집·원자적 교체 미구현 | 동일 파일을 다른 프로세스가 수정 중이면 적용하지 않음 |
| `smoke.sh` L0 | 설정 교차 검사와 모델 목록 대조 | warn/skip도 보고. exit 0만으로 모든 항목이 확인됐다고 판단하지 않음 |
| `smoke.sh` L1 | 역할·대안 모델 목록을 순회 | 표의 고정 3회와 달리 설정에 따라 호출 수가 달라짐 |
| `smoke.sh` L2 | 출력에 숫자 3이 포함되는지 검사 | 실제 ls 호출과 thinking 전달의 증거는 별도 transcript로 확인 |
| `smoke.sh` L3 | 결과 파일 존재로 검사 | 부모가 직접 썼을 가능성 배제 못함; 자식 실행·모델 아티팩트도 확인 |

위 결과는 저장소 코드를 읽고 임시 fixture로 검증한 범위다. 스니펫은 운영 파일에
자동 적용하지 않는다. 공식 패키지 지원과 사내 게이트웨이 수용 여부도 별개다.

## 야간 래퍼는 운영 검증 보류

> [!warning] 무인 실행의 배타성과 종료를 보장하지 않는다
> `night-run.ps1`은 작성된 예제로 보존했지만 현재 상태를 검증 완료 래퍼로 사용하지 않는다.

코드를 직접 읽어 확인한 문제는 다음과 같다.

1. `Start-Job`이 `CreateNew` 잠금보다 먼저다. 동시에 시작하면 잠금 실패 전에도
   두 작업이 실행될 수 있다. 잠금 확인 후 `Remove-Item`도 경쟁 상태가 있다.
2. 잔여 자식 때문에 남긴 잠금의 소유 PID는 래퍼 PID다. 래퍼 종료 뒤 다음 회차가
   이를 죽은 잠금으로 지울 수 있어 “잠금 유지가 다음 실행을 막는다”는 보장이 없다.
3. Repo 문자열로 프로세스를 찾는 방식은 정확한 부모/자식 트리 추적이 아니다.
   다른 프로세스를 선택하거나 실제 자식을 놓칠 수 있다.
4. job 실패나 pi의 종료 코드를 최종 실패 상태로 전달하는 경로가 충분하지 않다.
   “회차 종료”와 실제 업무 성공을 구분해야 한다.

실행 코드 수정은 이번 문서 작업 범위에서 하지 않았다. 전용 Claude 협의 및
Windows에서 잠금 경쟁·강제 종료·잔여 자식·종료 코드 시험 후 적용 판단을 한다.
macOS에 pwsh가 없어 PowerShell 문법/실행 검증도 미완료다.

`schtasks /IT`는 해당 사용자의 로그온 상태에서 실행하는 옵션이며 절전 방지·환경변수·
키 접근·실행 성공을 보장하지 않는다. `/F`는 같은 이름 작업을 덮어쓸 수 있다.
[Microsoft 옵션 설명](https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/schtasks-create)
(Windows 10/11·Server 문서, 2026-10-04 확인).
PowerShell은 셸 버전별 인자 인용을 확인해야 한다. 기존 README의 `Start-Process`
예제에서 여러 줄 prompt 전달과 자식 종료는 미검증이다.
[공식 ArgumentList 설명](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.management/start-process?view=powershell-7.5).

## 템플릿을 복사할 때

`git add -N`도 인덱스를 바꾼다. 배정된 경로로 한정한 실습 예이며 이번 정리에서
실행하지 않았다. 새 파일 확인만 하려면 상태 목록과 해당 파일 내용을 먼저 읽는다.
[Git 공식 설명](https://git-scm.com/docs/git-add).
dirty 상태는 사용자의 수정일 수도 있다. 변경 주체를 확인하지 않고 일괄 복구하지 않는다.
워크트리 삭제 전에는 필요한 결과와 미커밋 파일을 확인한다.

정책 본문을 프롬프트에 고정하면 파일 변조 혼동은 줄일 수 있지만 모델의 정책 준수를
강제하지는 않는다. 확신도 60% 같은 숫자는 보정된 확률이 아닌 템플릿의 휴리스틱이다.
복구·전송·삭제를 실제로 제한하려면 실행 환경의 권한 정책이 필요하다.

Herdr 현재 pane이 `pane_not_found`여서 전용 Claude와 협의하지 못했다.
야간 래퍼 변경, 역할 정의 재설계, 사내 모델별 설정 판단은 보류했다.
[문서별 결과와 검증](./organization-log.md)에 협의 부재를 기록한다.
