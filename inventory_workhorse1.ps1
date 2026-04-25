$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backup = "D:\awbw_migration_backup\$stamp`_pre_workhorse1_reonboard"
New-Item -ItemType Directory -Force $backup | Out-Null

git rev-parse HEAD > "$backup\git_head.txt"
git status --short > "$backup\git_status.txt"
nvidia-smi > "$backup\nvidia_smi.txt"
python --version > "$backup\python_env.txt"
where python >> "$backup\python_env.txt"

Get-ChildItem -Force checkpoints -Recurse | Select-Object FullName,Length,LastWriteTime > "$backup\checkpoints_manifest.txt"
Get-ChildItem -Force logs -Recurse -ErrorAction SilentlyContinue | Select-Object FullName,Length,LastWriteTime > "$backup\logs_manifest.txt"
Get-ChildItem -Force fleet -Recurse -ErrorAction SilentlyContinue | Select-Object FullName,Length,LastWriteTime > "$backup\fleet_manifest.txt"

Copy-Item checkpoints "$backup\checkpoints" -Recurse -Force -ErrorAction SilentlyContinue
Copy-Item fleet "$backup\fleet" -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "Backup created at $backup"