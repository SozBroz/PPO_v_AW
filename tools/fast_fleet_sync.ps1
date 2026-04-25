# Deprecated: use sync_checkpoint_zips_fleet.ps1 (checkpoint*.zip only; never latest.zip).
Write-Warning 'fast_fleet_sync.ps1 is deprecated. Use: .\tools\sync_checkpoint_zips_fleet.ps1'
& "$PSScriptRoot\sync_checkpoint_zips_fleet.ps1" @args
