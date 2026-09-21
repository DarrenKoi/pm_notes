# night-run.ps1 - Windows 작업 스케줄러에서 매시간 부르는 래퍼.
#
# 하는 일은 넷이다.
#   1. 이전 회차가 아직 돌고 있으면 이번 회차를 건너뛴다 (겹침 방지)
#   2. pi 를 한 번 돌리고 로그를 남긴다
#   3. 시간을 넘기면 자식 프로세스까지 통째로 죽인다
#   4. 자식이 다 죽은 것을 확인한 뒤에만 lock 을 푼다
#
# 사용:
#   powershell -ExecutionPolicy Bypass -NoProfile -File night-run.ps1 -Repo C:\work\wt-night
#
# 작업 스케줄러 등록은 night-setup.md 의 NIGHT-3 을 참고한다.
# 스케줄러에서 "이미 실행 중이면 새 인스턴스를 시작하지 않음" 을 함께 설정한다.
# 멈추려면 .orch\STOP 파일을 만든다. 에이전트가 작업을 끝내면 스스로 만든다.

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

if (Test-Path (Join-Path $orch "STOP")) {
  Write-Output "STOP 파일이 있다. 실행하지 않는다."
  exit 0
}

$lock = Join-Path $orch "run.lock"

# 살아 있는 lock 인지 본다. PID 재사용을 구분하려고 시작 시각까지 함께 확인한다.
function Test-LockAlive([string]$path) {
  if (-not (Test-Path $path)) { return $false }
  try { $data = Get-Content $path -Raw | ConvertFrom-Json } catch { return $false }
  $proc = Get-Process -Id $data.Pid -ErrorAction SilentlyContinue
  if (-not $proc) { return $false }
  # 같은 PID 라도 시작 시각이 다르면 재사용된 다른 프로세스다.
  return ($proc.StartTime.ToString("o") -eq $data.Start)
}

if (Test-LockAlive $lock) {
  Write-Output "이전 회차가 아직 돈다. 이번 회차는 건너뛴다."
  exit 0
}
Remove-Item $lock -Force -ErrorAction SilentlyContinue

if (-not (Test-Path $PromptFile)) {
  Write-Output "프롬프트 파일이 없다: $PromptFile"
  exit 1
}
$prompt = Get-Content $PromptFile -Raw

$stamp  = Get-Date -Format "yyyyMMdd-HHmmss"
$log    = Join-Path $logs "run-$stamp.log"
$errlog = Join-Path $logs "run-$stamp.err"

# 프롬프트는 여러 줄이고 따옴표·하이픈이 섞여 있다. Start-Process 의 ArgumentList 는
# 인수 경계를 보존하지 못하므로 쓰지 않는다. 호출 연산자는 $prompt 를 인수 하나로 넘긴다.
$job = Start-Job -ScriptBlock {
  param($repo, $p, $out, $err)
  Set-Location $repo
  & pi -p $p 1> $out 2> $err
  $LASTEXITCODE
} -ArgumentList $Repo, $prompt, $log, $errlog

# lock 을 원자적으로 만든다. 이미 있으면 다른 회차가 먼저 잡은 것이다.
$meta = @{ Pid = $PID; Start = (Get-Process -Id $PID).StartTime.ToString("o"); Job = $job.Id } | ConvertTo-Json -Compress
try {
  $fs = [System.IO.File]::Open($lock, 'CreateNew', 'Write')
  $bytes = [System.Text.Encoding]::UTF8.GetBytes($meta)
  $fs.Write($bytes, 0, $bytes.Length); $fs.Close()
} catch {
  Write-Output "다른 회차가 먼저 lock 을 잡았다. 이번 회차는 건너뛴다."
  Stop-Job $job -ErrorAction SilentlyContinue
  Remove-Job $job -Force -ErrorAction SilentlyContinue
  exit 0
}

try {
  $done = Wait-Job $job -Timeout ($TimeoutMinutes * 60)
  if (-not $done) {
    Write-Output "시간 초과($TimeoutMinutes 분). 프로세스 트리를 종료한다."
    Stop-Job $job -ErrorAction SilentlyContinue
    # pi 가 띄운 자식(서브에이전트·테스트·빌드)은 부모를 죽여도 살아남는다.
    # 남으면 다음 회차와 같은 트리를 동시에 고친다. 트리째 정리한다.
    Get-CimInstance Win32_Process |
      Where-Object { $_.CommandLine -like "*$Repo*" -and $_.Name -match '^(pi|node|pwsh|powershell)' } |
      ForEach-Object {
        Write-Output "  종료: PID $($_.ProcessId) $($_.Name)"
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
      }
    Start-Sleep -Seconds 3
    $leftover = @(Get-CimInstance Win32_Process |
      Where-Object { $_.CommandLine -like "*$Repo*" -and $_.Name -match '^(pi|node)' })
    if ($leftover.Count -gt 0) {
      # 아직 살아 있으면 lock 을 남긴 채 끝낸다. 다음 회차가 건너뛰게 해서
      # 두 에이전트가 같은 트리를 고치는 것을 막는다. 사람이 확인해야 한다.
      Write-Output "경고: 자식 프로세스 $($leftover.Count) 개가 남았다. lock 을 유지한다."
      Write-Output "확인 후 .orch\run.lock 을 직접 지워라."
      Receive-Job $job -ErrorAction SilentlyContinue | Out-Null
      Remove-Job $job -Force -ErrorAction SilentlyContinue
      exit 2
    }
  }
  Receive-Job $job -ErrorAction SilentlyContinue | Out-Null
  Remove-Job $job -Force -ErrorAction SilentlyContinue
  Remove-Item $lock -Force -ErrorAction SilentlyContinue
} catch {
  Write-Output "오류: $_"
  Remove-Item $lock -Force -ErrorAction SilentlyContinue
  throw
}

Write-Output "회차 종료. log=$log"
