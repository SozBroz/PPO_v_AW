param(
    [switch]$DryRun,
    [string]$RepoRoot = ''
)
$extra = @{}
if ($DryRun) { $extra.DryRun = $true }
if ($RepoRoot) { $extra.RepoRoot = $RepoRoot }
& "$PSScriptRoot\sync_checkpoint_zips_fleet.ps1" -Direction Pull @extra
