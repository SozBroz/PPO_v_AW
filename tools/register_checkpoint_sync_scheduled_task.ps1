# Registers a Windows Scheduled Task to run checkpoint*.zip fleet sync every 2 hours.
# Default: runs on THIS machine only, uses SSH to reach workhorse1 (pull + push).
# latest.zip is never synced (see sync_checkpoint_zips_fleet.ps1).
#
# Run in an elevated PowerShell if you want the task to run when no user is logged on
# (optional; otherwise current-user context is fine for SSH keys loaded at login).

param(
    [string]$RepoRoot = '',
    [string]$TaskName = 'AWBW_CheckpointZipSync_2h',
    [string]$RemoteSsh = 'sshuser@192.168.0.160',
    [string]$RemoteCheckpointDir = 'D:\awbw\checkpoints',
    [string]$LocalMachineId = 'pc-b',
    [string]$RemoteMachineId = 'workhorse1',
    [switch]$Unregister
)

$ErrorActionPreference = 'Stop'

if (-not $RepoRoot) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
}

$syncScript = Join-Path $PSScriptRoot 'sync_checkpoint_zips_fleet.ps1'
if (-not (Test-Path -LiteralPath $syncScript)) {
    throw "Missing $syncScript"
}

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Unregistered task: $TaskName"
    exit 0
}

# Hidden window: no PowerShell flash on screen (still logs to logs\checkpoint_zip_sync.log).
$argString = '-WindowStyle Hidden -NoLogo -NonInteractive -NoProfile -ExecutionPolicy Bypass -File "{0}" -Direction Both -RepoRoot "{1}" -RemoteSsh "{2}" -RemoteCheckpointDir "{3}" -LocalMachineId "{4}" -RemoteMachineId "{5}"' -f @(
    $syncScript,
    $RepoRoot,
    $RemoteSsh,
    $RemoteCheckpointDir,
    $LocalMachineId,
    $RemoteMachineId
)

$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $argString
# Every 2 hours, starting now (one-off trigger with repetition is the usual pattern on Windows).
$start = (Get-Date).AddMinutes(1)
$trigger = New-ScheduledTaskTrigger -Once -At $start -RepetitionInterval (New-TimeSpan -Hours 2) -RepetitionDuration (New-TimeSpan -Days 3650)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null

Write-Host "Registered scheduled task: $TaskName"
Write-Host "Runs every 2 hours as $env:USERNAME (Interactive logon; OK for ssh-agent)."
Write-Host "Log file: $(Join-Path $RepoRoot 'logs\checkpoint_zip_sync.log')"
Write-Host ""
Write-Host "One machine is enough: this task pulls checkpoint*.zip from workhorse1 and pushes your checkpoint*.zip to workhorse1\checkpoints\pool\$LocalMachineId."
Write-Host "Do not install a second copy on workhorse1 unless you also run OpenSSH Server on this PC and pass -RemoteSsh to this script."
