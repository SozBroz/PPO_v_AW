# workhorse1 Fleet Sync — Ruthless Revised Plan

**Repo:** `SozBroz/PPO_v_AW`  
**Target host:** `workhorse1` / `192.168.0.160`  
**Doctrine:** local training sovereignty, SSH/SCP/rsync for bulk artifacts, no Samba for checkpoint transport  
**Revision date:** 2026-04-25

---

## Executive verdict

The original plan is directionally correct, but too trusting. Its biggest weakness is that it treats “getting files onto workhorse1” as the core problem. That is not the core problem.

The real failure modes are:

1. **Code/checkpoint contract drift** — same-looking `latest.zip`, different encoder/model/env semantics.
2. **Two-writer checkpoint corruption** — two machines promoting or replacing `latest.zip` without strict ownership.
3. **Half-migrated fleet architecture** — repo docs still support shared-root auxiliary semantics, while this plan wants independent hosts with SSH-backed artifact exchange.
4. **Unproven async throughput assumptions** — `--training-backend async` may be correct, but env count, GPU opponent permits, learner batch, and unroll length must be empirically gated.
5. **No rollback protocol** — the plan says “archive/delete stale artifacts” but does not define an immutable preflight snapshot.
6. **Operational ambiguity around who is authoritative** — pc-b is “ahead,” workhorse1 is “eternal main,” and both are potential training hosts. That needs a strict promotion model.

The plan should be treated as a deployment/migration, not a copy operation.

---

## Corrected operating model

### Roles

| Host | Role | Authority |
|---|---|---|
| Operator PC / `pc-b` | current known-good source, hub, evaluator, promotion controller | owns initial seed and promotion decisions |
| `workhorse1` | high-throughput training worker / future main | owns its own local training outputs only |
| Shared filesystem | legacy convenience only | never authoritative for large checkpoints |

### Non-negotiable rule

No machine writes directly into another machine’s live `checkpoints/latest.zip`.

All checkpoint exchange must go through:

1. staged filename
2. sidecar hash
3. verification
4. atomic local replace
5. manifest update

Example destination lifecycle:

```text
incoming/latest.zip.tmp
incoming/latest.zip.sha256
incoming/latest.zip.verified
checkpoints/pool/pc-b/latest_YYYYMMDD_HHMMSS.zip
```

Only after evaluation/promotion should a local `latest.zip` be replaced.

---

## Phase 0 — Stop-the-line preflight

Before touching checkpoints, run this on **both** machines and save output.

### Commands

```powershell
hostname
whoami
Get-Date -Format o
git rev-parse HEAD
git status --short
python --version
where python
python -c "import sys; print(sys.executable)"
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
nvidia-smi
Get-ChildItem -Force checkpoints | Select-Object Name,Length,LastWriteTime
Get-ChildItem -Force checkpoints\pool -Recurse -ErrorAction SilentlyContinue | Select-Object FullName,Length,LastWriteTime
```

### Gate

Proceed only if:

- git commit is known on both hosts
- Python executable is the intended venv
- CUDA visibility is proven in the same session that will run training
- current checkpoint files are inventoried
- there is enough free disk for at least 3 full checkpoint generations plus logs

### Reason

If you cannot reproduce the environment inventory, you cannot trust any result from the migration.

---

## Phase 1 — Immutable backup before cleanup

The original plan says “archive or delete stale artifacts.” That is too loose.

### Required backup layout

On workhorse1:

```text
D:\awbw_migration_backup\
  2026-04-25_pre_workhorse1_reonboard\
    git_head.txt
    git_status.txt
    nvidia_smi.txt
    python_env.txt
    checkpoints_manifest.txt
    logs_manifest.txt
    fleet_manifest.txt
    checkpoints\
    fleet\
```

### Commands

```powershell
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
```

### Gate

Do not delete anything until this backup exists and its manifest is readable.

---

## Phase 2 — Git parity and build proof

The original plan says “git parity.” Good, but insufficient. You need import/build proof.

### Commands

```powershell
git fetch --all --prune
git rev-parse HEAD
git log -1 --oneline

python scripts/rebuild_cython_extensions.py
python -m pytest -q tests
```

If the full test suite is too slow, minimum smoke test:

```powershell
python -c "import engine; import rl; print('imports ok')"
python -c "from rl.fleet_env import load_machine_role; print('fleet import ok')"
python train.py --help
python scripts/start_solo_training.py --help
```

### Gate

Proceed only if:

- Cython extensions rebuild cleanly
- imports resolve from the intended repo path
- `train.py --help` exposes the expected async/backend flags
- the machine does not silently import old packages from another checkout

### Failure action

If imports point anywhere outside `D:\awbw`, stop and fix venv/path contamination before continuing.

---

## Phase 3 — Clean stale training state deliberately

Do **not** blanket-delete. Partition state into three buckets.

### Bucket A — Preserve

- known-good seed zips
- previous best candidates worth evaluating
- manifests
- logs needed to explain current training curves

### Bucket B — Quarantine

Move suspicious artifacts here:

```text
D:\awbw_quarantine\YYYYMMDD_HHMMSS\
```

Suspicious means:

- unknown encoder/policy contract
- old commit
- pre-current training architecture
- no hash
- ambiguous provenance
- stale `fleet/<old-id>` state

### Bucket C — Delete

Only delete:

- obviously broken partial transfers
- temp files
- massive logs already backed up
- known obsolete generated cache

### Command pattern

```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$quarantine = "D:\awbw_quarantine\$stamp"
New-Item -ItemType Directory -Force $quarantine | Out-Null

Move-Item checkpoints\latest.zip "$quarantine\latest.zip" -ErrorAction SilentlyContinue
Move-Item fleet\old_machine_id "$quarantine\fleet_old_machine_id" -ErrorAction SilentlyContinue
```

---

## Phase 4 — Seed weights with manifest discipline

The original plan correctly calls for `scp`/`sftp` plus `sha256`. Add a manifest and local-only promotion.

### Source-side manifest

On pc-b:

```powershell
$src = "C:\Users\phili\AWBW\checkpoints\latest.zip"
Get-FileHash $src -Algorithm SHA256
git rev-parse HEAD
```

Create:

```json
{
  "artifact": "latest.zip",
  "source_host": "pc-b",
  "source_git_head": "<git sha>",
  "sha256": "<hash>",
  "created_at": "<iso timestamp>",
  "intended_destination": "workhorse1",
  "policy_contract": "same repo commit required before load"
}
```

### Transfer rule

Transfer to a staging directory, never directly into `checkpoints`.

```text
D:\awbw\incoming\pc-b\latest.zip.tmp
D:\awbw\incoming\pc-b\latest.zip.sha256
D:\awbw\incoming\pc-b\manifest.json
```

### Destination verification

On workhorse1:

```powershell
Get-FileHash D:\awbw\incoming\pc-b\latest.zip.tmp -Algorithm SHA256
```

Only after hash match:

```powershell
New-Item -ItemType Directory -Force D:\awbw\checkpoints\pool\pc-b | Out-Null
Move-Item D:\awbw\incoming\pc-b\latest.zip.tmp D:\awbw\checkpoints\pool\pc-b\latest_seed.zip
Copy-Item D:\awbw\checkpoints\pool\pc-b\latest_seed.zip D:\awbw\checkpoints\latest.zip
```

### Gate

Do not train until the checkpoint can be loaded by a minimal policy-load smoke test or a short dry-run if available.

---

## Phase 5 — Probe first, async later

The original plan jumps too quickly from probe to “push hard.” Probe should produce conservative args first.

### Commands

```powershell
python tools/probe_machine_caps.py --machine-id workhorse1
python scripts/start_solo_training.py --machine-id workhorse1 --auto-apply --training-backend async --train-extra-args "--fps-diag"
```

If `--fps-diag` is not available in your current CLI, run the shortest supported training slice and record:

- env FPS
- learner FPS
- GPU utilization
- CPU utilization
- RAM
- VRAM
- actor restarts
- queue/backpressure symptoms
- checkpoint write cadence

### Gate

Only increase `n-envs`, GPU opponent permits, async unroll, or learner batch after a clean short run.

### Tuning order

1. prove stable import/build/load
2. prove one short async run
3. increase `n-envs`
4. adjust GPU opponent permits
5. tune async unroll length
6. tune learner batch
7. only then consider `torch.compile`

### Reason

If you enable every speed path at once, you will not know which knob caused instability.

---

## Phase 6 — Training launch profile

Start conservative.

```powershell
python scripts/start_solo_training.py `
  --machine-id workhorse1 `
  --auto-apply `
  --training-backend async
```

Add extra args only after the probe slice.

Potential later form:

```powershell
python scripts/start_solo_training.py `
  --machine-id workhorse1 `
  --auto-apply `
  --training-backend async `
  --train-extra-args "--n-envs <N> --async-unroll-length <U> --async-learner-batch <B>"
```

Do not treat this command as final. Treat it as an experiment whose output chooses the next command.

---

## Phase 7 — Ongoing sync protocol

The original plan is right to avoid Samba, but it needs stronger ownership semantics.

### Recommended layout

```text
checkpoints/
  latest.zip                    # local active policy only
  pool/
    pc-b/
      latest_seed.zip
      candidate_YYYYMMDD_HHMMSS.zip
      manifest_YYYYMMDD_HHMMSS.json
    workhorse1/
      candidate_YYYYMMDD_HHMMSS.zip
      manifest_YYYYMMDD_HHMMSS.json
```

### Sync direction

| Direction | Purpose | Allowed destination |
|---|---|---|
| pc-b → workhorse1 | seed / opponent pool diversity | `checkpoints/pool/pc-b/` |
| workhorse1 → pc-b | candidate evaluation / pool diversity | `checkpoints/pool/workhorse1/` |
| either → `latest.zip` | only after local verification/promotion | local host only |

### Promotion rule

A candidate copied from another host is not “latest.” It is an opponent or candidate until evaluated.

---

## Phase 8 — Orchestrator strategy

Do not attempt unified multi-host orchestration on day one.

### Day-one mode

- workhorse1 runs standalone solo training
- pc-b remains operator/eval hub
- checkpoint exchange is manual/scripted
- symmetric eval uses explicit copied paths

### Later mode

Build a small `tools/fleet_scp_sync.py` or PowerShell script that does:

1. find new candidates
2. write manifest
3. copy to remote `.tmp`
4. copy hash
5. remote verify
6. atomic move into pool
7. optionally emit reload request
8. log sync event

### Do not do this yet

- do not rely on Samba for multi-GB checkpoint traffic
- do not make both machines share a live `checkpoints/`
- do not run one orchestrator assuming a shared root unless you have actually mirrored the required fleet/pool state locally

---

## Phase 9 — Documentation correction

The repo README still describes auxiliary machines using mounted shared-root semantics such as `Z:\`, while this plan intentionally moves bulk checkpoints over SSH/SCP/rsync and keeps training local.

Update internal docs to say:

- Samba/shared-root is legacy/convenience only
- large checkpoint transport uses SSH/SCP/rsync
- each trainer writes local outputs
- cross-host candidates live under `checkpoints/pool/<machine-id>/`
- `latest.zip` is local-active, not a networked truth object

Suggested files to update:

```text
docs/multi_machine_weight_sync_design.md
docs/SOLO_TRAINING.md
.cursor/skills/awbw-auxiliary-main-machines/SKILL.md
README.md fleet section
```

---

## Phase 10 — Concrete first-session checklist

### On pc-b

```powershell
cd C:\Users\phili\AWBW
git rev-parse HEAD
git status --short
Get-FileHash checkpoints\latest.zip -Algorithm SHA256
```

### On workhorse1

```powershell
ssh 192.168.0.160
hostname
cd D:\awbw
git rev-parse HEAD
git status --short
python --version
where python
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvidia-smi
python scripts/rebuild_cython_extensions.py
python -c "import engine; import rl; print('imports ok')"
python tools/probe_machine_caps.py --machine-id workhorse1
```

### Then transfer seed

Use `scp`, `sftp`, or `rsync`. Do not use Samba.

### Then verify

```powershell
Get-FileHash D:\awbw\incoming\pc-b\latest.zip.tmp -Algorithm SHA256
```

### Then short run

```powershell
python scripts/start_solo_training.py --machine-id workhorse1 --auto-apply --training-backend async
```

Stop after a short interval, inspect logs, then tune.

---

## Revised risk register

| Risk | Severity | Mitigation |
|---|---:|---|
| checkpoint incompatible with current code | critical | same git head + load smoke test |
| partial checkpoint transfer | critical | `.tmp` + sha256 + atomic move |
| two machines overwrite `latest.zip` | critical | local-only latest; remote candidates go to pool |
| async settings unstable | high | conservative probe slice before tuning |
| wrong Python/venv | high | `where python`, import checks, rebuild extensions |
| GPU invisible in SSH session | high | `nvidia-smi` and torch CUDA check in same session |
| stale fleet state poisons orchestration | medium-high | quarantine old `fleet/<id>` before launch |
| docs contradict new doctrine | medium | update README/runbooks after first successful run |
| Samba silently re-enters workflow | medium | explicitly ban for zips/log bulk transfer |
| no rollback path | medium-high | immutable preflight backup |

---

## What I would change from the original plan

1. Replace “delete/archive stale artifacts” with “backup, then quarantine, then selectively delete.”
2. Add a hard preflight gate before any transfer.
3. Add build/import/test proof before loading checkpoints.
4. Treat `latest.zip` as local-active only, never cross-host truth.
5. Put copied policies under `checkpoints/pool/<source-machine>/`.
6. Delay aggressive async tuning until after a measured short run.
7. Explicitly document that this plan diverges from the repo’s older shared-root auxiliary model.
8. Add manifests to every transferred checkpoint.
9. Use pc-b as promotion/eval hub until workhorse1 proves stable.
10. Add rollback paths and stop-the-line conditions.

---

## Final recommended doctrine

Workhorse1 should become the heavy training engine, but not by pretending the fleet system is already a clean distributed system.

Start with:

- local repo
- local venv
- local checkpoints
- SSH/SCP artifact exchange
- separate pools
- manifest/hash verification
- conservative async bring-up
- pc-b as evaluation/promotional authority

After it proves stable, automate sync. Do not automate first and debug later.
