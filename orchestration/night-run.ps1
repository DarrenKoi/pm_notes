# night-run.ps1 - Windows 작업 스케줄러에서 매시간 부르는 래퍼.
#
# 하는 일은 셋뿐이다.
#   1. 이전 회차가 아직 돌고 있으면 이번 회차를 건너뛴다 (겹침 방지)
#   2. pi 를 한 번 돌리고 로그를 남긴다
#   3. 시간을 넘기면 죽인다
#
# 사용:
#   powershell -ExecutionPolicy Bypass -File night-run.ps1 -Repo C:\work\wt-night
#
# 작업 스케줄러 등록 (1시간 간격):
#   프로그램:  powershell.exe
#   인수:      -ExecutionPolicy Bypass -NoProfile -File "C:\...\night-run.ps1" -Repo "C:\work\wt-night"
#   시작 위치: C:\work\wt-night
#
# 멈추려면 .orch\STOP 파일을 만든다. 에이전트가 작업을 다 끝내면 스스로 만든다.

param(
  [string]$Repo = (Get-Location).Path,
  [string]$PromptFile = ".orch\night-prompt.txt",
  [int]$TimeoutMinutes = 50
)

$ErrorActionPreference = "Stop"
Set-Location $Repo

$orch = Join-Path $Repo ".orch"
$logs = Join-Path $orch "logs"
New-Item -ItemType Directory -Force -Path $orch | Out-Null
New-Item -ItemType Directory -Force -Path $logs | Out-Null

# 중지 스위치. 사람이 만들거나 에이전트가 작업을 끝내고 만든다.
$stop = Join-Path $orch "STOP"
if (Test-Path $stop) {
  Write-Output "STOP 파일이 있다. 실행하지 않는다."
  exit 0
}

# 단일 실행 보장. 겹치면 같은 워크트리를 두 에이전트가 동시에 고친다.
$lock = Join-Path $orch "run.lock"
if (Test-Path $lock) {
  $oldPid = 0
  $raw = (Get-Content $lock -Raw -ErrorAction SilentlyContinue)
  if ($raw) { [void][int]::TryParse($raw.Trim(), [ref]$oldPid) }
  if ($oldPid -gt 0 -and (Get-Process -Id $oldPid -ErrorAction SilentlyContinue)) {
    Write-Output "이전 회차(PID $oldPid)가 아직 돈다. 이번 회차는 건너뛴다."
    exit 0
  }
  Write-Output "죽은 lock 을 정리한다 (PID $oldPid)."
  Remove-Item $lock -Force -ErrorAction SilentlyContinue
}

if (-not (Test-Path $PromptFile)) {
  Write-Output "프롬프트 파일이 없다: $PromptFile"
  exit 1
}
$prompt = Get-Content $PromptFile -Raw

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$log   = Join-Path $logs "run-$stamp.log"
$errlog= Join-Path $logs "run-$stamp.err"

$proc = Start-Process -FilePath "pi" -ArgumentList @("-p", $prompt) `
        -RedirectStandardOutput $log -RedirectStandardError $errlog `
        -PassThru -NoNewWindow
$null = $proc.Handle          # 핸들을 캐시해 둬야 WaitForExit 가 동작한다
Set-Content -Path $lock -Value $proc.Id

try {
  if (-not $proc.WaitForExit($TimeoutMinutes * 60 * 1000)) {
    Write-Output "시간 초과($TimeoutMinutes 분). 종료시킨다."
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
  }
} finally {
  Remove-Item $lock -Force -ErrorAction SilentlyContinue
}

Write-Output "회차 종료. exit=$($proc.ExitCode) log=$log"
