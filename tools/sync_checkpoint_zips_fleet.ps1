# Sync checkpoint*.zip only between fleet hosts. Never syncs latest.zip (excluded by pattern + policy).
# Pull:  remote <RemoteCheckpointDir>\checkpoint*.zip -> local <RepoRoot>\checkpoints\pool\<RemoteMachineId>\
# Push:  local  <RepoRoot>\checkpoints\checkpoint*.zip -> remote <RemoteCheckpointDir>\pool\<LocalMachineId>\
#
# Requires: OpenSSH (ssh, scp), non-interactive SSH auth to the remote host.

param(
    [ValidateSet('Pull', 'Push', 'Both')]
    [string]$Direction = 'Both',

    [string]$RepoRoot = '',

    [string]$RemoteSsh = 'sshuser@192.168.0.160',
    [string]$RemoteCheckpointDir = 'D:\awbw\checkpoints',

    [string]$LocalMachineId = 'pc-b',
    [string]$RemoteMachineId = 'workhorse1',

    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not $RepoRoot) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
}

$localCheckpointDir = Join-Path $RepoRoot 'checkpoints'
$localPoolRemote = Join-Path $RepoRoot (Join-Path 'checkpoints' (Join-Path 'pool' $RemoteMachineId))

$logDir = Join-Path $RepoRoot 'logs'
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
}
$logFile = Join-Path $logDir 'checkpoint_zip_sync.log'

function Write-Log {
    param([string]$Message)
    $line = '{0} {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message
    Add-Content -LiteralPath $logFile -Value $line
    Write-Host $line
}

function Test-AllowedCheckpointLeafName {
    param([string]$Name)
    if ([string]::IsNullOrWhiteSpace($Name)) { return $false }
    if ($Name -ieq 'latest.zip') { return $false }
    if ($Name -notlike 'checkpoint*.zip') { return $false }
    return $true
}

function Get-LocalCheckpointInventory {
    param([string]$Dir)
    if (-not (Test-Path -LiteralPath $Dir)) { return @() }
    Get-ChildItem -LiteralPath $Dir -File | Where-Object { Test-AllowedCheckpointLeafName $_.Name } | ForEach-Object {
        $h = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
        [pscustomobject]@{ Name = $_.Name; Length = $_.Length; Sha256 = $h }
    }
}

function Invoke-RemoteCheckpointInventory {
    param(
        [string]$SshTarget,
        [string]$RemoteDir
    )
    $rd = $RemoteDir.Replace("'", "''")
    $invBody = @"
Set-StrictMode -Version Latest
`$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath '$rd')) { exit 0 }
Get-ChildItem -LiteralPath '$rd' -File | ForEach-Object {
  if (`$_.Name -ieq 'latest.zip') { return }
  if (`$_.Name -notlike 'checkpoint*.zip') { return }
  `$h = (Get-FileHash -LiteralPath `$_.FullName -Algorithm SHA256).Hash
  Write-Output (('{0}{1}{2}{3}{4}' -f `$_.Name, "`t", `$_.Length, "`t", `$h))
}
"@
    $tmpLocal = [System.IO.Path]::GetTempFileName() + '_awbw_inv.ps1'
    try {
        Set-Content -LiteralPath $tmpLocal -Value $invBody -Encoding UTF8
        $remoteInv = 'D:/awbw/.awbw_checkpoint_inventory_tmp.ps1'
        scp $tmpLocal "${SshTarget}:${remoteInv}"
        $raw = ssh $SshTarget "powershell -NoProfile -ExecutionPolicy Bypass -File D:/awbw/.awbw_checkpoint_inventory_tmp.ps1"
        ssh $SshTarget "cmd /c del /f /q D:\awbw\.awbw_checkpoint_inventory_tmp.ps1 2>nul"
    }
    finally {
        Remove-Item -LiteralPath $tmpLocal -Force -ErrorAction SilentlyContinue
    }
    $rows = @()
    foreach ($line in ($raw -split "`r?`n")) {
        if (-not $line.Trim()) { continue }
        $parts = $line.Split("`t")
        if ($parts.Count -lt 3) { continue }
        $rows += [pscustomobject]@{
            Name   = $parts[0]
            Length = [long]$parts[1]
            Sha256 = $parts[2]
        }
    }
    return $rows
}

function Ensure-RemoteDir {
    param([string]$SshTarget, [string]$RemoteDirWin)
    $rd = $RemoteDirWin.Replace("'", "''")
    ssh $SshTarget "powershell -NoProfile -Command `"New-Item -ItemType Directory -Force -LiteralPath '$rd' | Out-Null`""
}

function Scp-PullFile {
    param(
        [string]$SshTarget,
        [string]$RemoteFileWin,
        [string]$LocalTmp,
        [string]$LocalFinal,
        [string]$ExpectedSha256
    )
    if (-not (Test-AllowedCheckpointLeafName (Split-Path -Leaf $RemoteFileWin))) {
        throw "Refusing to pull non-checkpoint pattern: $RemoteFileWin"
    }
    $remoteScp = ($RemoteFileWin -replace '\\', '/')
    scp "${SshTarget}:${remoteScp}" $LocalTmp
    $got = (Get-FileHash -LiteralPath $LocalTmp -Algorithm SHA256).Hash
    if ($got -ne $ExpectedSha256) {
        Remove-Item -LiteralPath $LocalTmp -Force -ErrorAction SilentlyContinue
        throw "Hash mismatch after pull for $(Split-Path -Leaf $LocalFinal). expected=$ExpectedSha256 got=$got"
    }
    if (Test-Path -LiteralPath $LocalFinal) {
        Remove-Item -LiteralPath $LocalFinal -Force
    }
    Move-Item -LiteralPath $LocalTmp -Destination $LocalFinal
}

function Scp-PushFile {
    param(
        [string]$SshTarget,
        [string]$LocalFile,
        [string]$RemoteTmpUnix,
        [string]$RemoteFinalWin,
        [string]$ExpectedSha256
    )
    $leaf = Split-Path -Leaf $LocalFile
    if (-not (Test-AllowedCheckpointLeafName $leaf)) {
        throw "Refusing to push non-checkpoint pattern: $leaf"
    }
    scp $LocalFile "${SshTarget}:${RemoteTmpUnix}"
    $rd = $RemoteFinalWin.Replace("'", "''")
    $rt = ($RemoteTmpUnix -replace '\\', '/').Replace("'", "''")
    $verify = @"
Set-StrictMode -Version Latest
`$ErrorActionPreference = 'Stop'
`$got = (Get-FileHash -LiteralPath '$rt' -Algorithm SHA256).Hash
if (`$got -ne '$ExpectedSha256') { throw "Push tmp hash mismatch: `$got" }
if (Test-Path -LiteralPath '$rd') { Remove-Item -LiteralPath '$rd' -Force }
Move-Item -LiteralPath '$rt' -Destination '$rd'
"@
    $tmpV = [System.IO.Path]::GetTempFileName() + '_awbw_pushv.ps1'
    try {
        Set-Content -LiteralPath $tmpV -Value $verify -Encoding UTF8
        scp $tmpV "${SshTarget}:D:/awbw/.awbw_push_verify_tmp.ps1"
        ssh $SshTarget "powershell -NoProfile -ExecutionPolicy Bypass -File D:/awbw/.awbw_push_verify_tmp.ps1"
        ssh $SshTarget "cmd /c del /f /q D:\awbw\.awbw_push_verify_tmp.ps1 2>nul"
    }
    finally {
        Remove-Item -LiteralPath $tmpV -Force -ErrorAction SilentlyContinue
    }
}

Write-Log "sync_checkpoint_zips_fleet start Direction=$Direction RepoRoot=$RepoRoot DryRun=$DryRun"

if (-not (Test-Path -LiteralPath $localCheckpointDir)) {
    New-Item -ItemType Directory -Force -Path $localCheckpointDir | Out-Null
    Write-Log "Created local checkpoint dir $localCheckpointDir"
}

if ($Direction -eq 'Pull' -or $Direction -eq 'Both') {
    if (-not $DryRun) {
        New-Item -ItemType Directory -Force -Path $localPoolRemote | Out-Null
    }
    Write-Log "Pull: $RemoteSsh`:$RemoteCheckpointDir -> $localPoolRemote"
    $remoteRows = Invoke-RemoteCheckpointInventory -SshTarget $RemoteSsh -RemoteDir $RemoteCheckpointDir
    foreach ($r in $remoteRows) {
        if (-not (Test-AllowedCheckpointLeafName $r.Name)) { continue }
        $dest = Join-Path $localPoolRemote $r.Name
        $need = $true
        if ((Test-Path -LiteralPath $dest) -and -not $DryRun) {
            $dh = (Get-FileHash -LiteralPath $dest -Algorithm SHA256).Hash
            if ($dh -eq $r.Sha256) { $need = $false }
        }
        if (-not $need) {
            Write-Log "Pull skip up-to-date $($r.Name)"
            continue
        }
        $remoteFile = Join-Path $RemoteCheckpointDir $r.Name
        $tmp = $dest + '.tmp'
        Write-Log "Pull $($r.Name) ($($r.Length) bytes)"
        if ($DryRun) { continue }
        Scp-PullFile -SshTarget $RemoteSsh -RemoteFileWin $remoteFile -LocalTmp $tmp -LocalFinal $dest -ExpectedSha256 $r.Sha256
        Write-Log "Pull done $($r.Name)"
    }
}

if ($Direction -eq 'Push' -or $Direction -eq 'Both') {
    $remotePoolWin = Join-Path $RemoteCheckpointDir (Join-Path 'pool' $LocalMachineId)
    if (-not $DryRun) {
        Ensure-RemoteDir -SshTarget $RemoteSsh -RemoteDirWin $remotePoolWin
    }
    Write-Log "Push: $localCheckpointDir -> $RemoteSsh`:$remotePoolWin"
    $remoteExisting = @{}
    foreach ($row in (Invoke-RemoteCheckpointInventory -SshTarget $RemoteSsh -RemoteDir $remotePoolWin)) {
        $remoteExisting[$row.Name] = $row.Sha256
    }
    $localRows = Get-LocalCheckpointInventory -Dir $localCheckpointDir
    foreach ($r in $localRows) {
        $remoteFinalWin = Join-Path $remotePoolWin $r.Name
        $remoteFinalScp = ($remoteFinalWin -replace '\\', '/')
        $remoteTmp = $remoteFinalScp + '.tmp'
        $need = $true
        if ($remoteExisting.ContainsKey($r.Name) -and $remoteExisting[$r.Name] -eq $r.Sha256) {
            $need = $false
        }
        if (-not $need) {
            Write-Log "Push skip up-to-date $($r.Name)"
            continue
        }
        Write-Log "Push $($r.Name) ($($r.Length) bytes)"
        if ($DryRun) { continue }
        Scp-PushFile -SshTarget $RemoteSsh -LocalFile (Join-Path $localCheckpointDir $r.Name) -RemoteTmpUnix $remoteTmp -RemoteFinalWin $remoteFinalWin -ExpectedSha256 $r.Sha256
        Write-Log "Push done $($r.Name)"
    }
}

Write-Log 'sync_checkpoint_zips_fleet complete'
