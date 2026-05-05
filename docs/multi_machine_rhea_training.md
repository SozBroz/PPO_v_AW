# Multi-Machine RHEA Value Training

## Overview

Allow any number of machines to contribute RHEA self-play transitions to a single value learner running on `workhorse1`. Uses the existing Samba shared filesystem (`Z:\` or `/mnt/awbw`) for both weight synchronization and transition aggregation.

## Architecture

```
                      ┌─────────────────────────────────────────────┐
                      │          workhorse1 (learner)              │
                      │                                           │
                      │  train_rhea_value_parallel.py            │
                      │  ┌─────────────────────────────────┐     │
                      │  │ Learner process                  │     │
                      │  │  - owns replay buffer            │     │
                      │  │  - trains value net             │     │
                      │  │  - saves value_rhea_latest.pt   │     │
                      │  └───────────┬─────────────────────┘     │
                      │              │ reads                    │
                      │  ┌───────────▼─────────────────────┐     │
                      │  │ Z:/checkpoints/value_rhea_latest │     │
                      │  └─────────────────────────────────┘     │
                      └─────────────────────────────────────────────┘
                                          ▲
                                          │ reads weights
                      ┌───────────────────┴──────────────────────┐
                      │                                          │
          ┌───────────┴──────────┐              ┌───────────────┴────────────┐
          │   workhorse1 actors   │              │   other machine actors       │
          │   (local mp.Queue)   │              │   (write to shared disk)    │
          └──────────────────────┘              └─────────────────────────────┘
```

## Files Modified/Created

### New: `scripts/rhea_remote_actor.py`
Remote actor script that runs on any machine with access to the shared Samba mount.

**Launch command (on any auxiliary machine):**
```bash
python -m scripts.rhea_remote_actor \
  --shared-root Z:\ \
  --machine-id workhorse2 \
  --checkpoint Z:/checkpoints/value_rhea_latest.pt \
  --map-id 171596 \
  --co-p0 14,8,28,7 --co-p1 14,8,28,7 \
  --max-days 30 \
  --rhea-autotune \
  --reward-weight 0.8 --value-weight 0.2 \
  --phi-capture-phase-weighting \
  --dual-gradient-self-play \
  --pairwise-zero-sum-reward \
  --actor-refresh-seconds 120 \
  --transition-batch-size 100 \
  --n-envs 8
```

**Key arguments:**
- `--shared-root`: Path to Samba mount (e.g. `Z:\` on Windows, `/mnt/awbw` on Linux)
- `--machine-id`: Unique machine ID (e.g. `workhorse2`, `gpu-box1`)
- `--transition-batch-size`: Number of transitions per JSONL file (default: 100)
- `--checkpoint`: Path to value net checkpoint (default: `<shared-root>/checkpoints/value_rhea_latest.pt`)

### Modified: `scripts/train_rhea_value_parallel.py`
Added remote transition polling to the main learner loop.

**New arguments:**
- `--remote-transition-dir`: Directory to poll for remote transition files (default: polls `fleet/*/transitions/` under the checkpoint directory)
- `--poll-remote-transitions-interval`: Seconds between polling (default: 60)

The learner:
1. Periodically scans for `*.jsonl` files in `fleet/*/transitions/`
2. Reads and ingests transitions into the replay buffer
3. Renames processed files to `*.jsonl.done`

### Modified: `rl/rhea_replay.py`
Added methods for batch ingestion:
- `add_batch(transitions: list[RheaTransition]) -> int`: Add a batch of transitions to the replay buffer
- `payload_to_transition(p: dict) -> RheaTransition`: Convert JSON payload to RheaTransition

### Modified: `rl/fleet_env.py`
Added helpers for remote transition directory layout:
- `remote_transition_dir(shared_root, machine_id) -> Path`: Returns `fleet/<machine_id>/transitions/`
- `iter_remote_transition_files(shared_root) -> list[Path]`: Globs all `*.jsonl` files under `fleet/*/transitions/`

## Transition Flow

1. **Remote actor** runs RHEA self-play, writes transition batches to `Z:/fleet/<machine_id>/transitions/*.jsonl`
2. **Learner** (on workhorse1) polls for new `*.jsonl` files every `--poll-remote-transitions-interval` seconds
3. **Learner** reads transitions, adds them to the replay buffer via `add_batch()`
4. **Learner** renames processed files to `*.jsonl.done`
5. **Learner** trains the value net from the replay buffer as usual
6. **Learner** saves updated weights to `checkpoints/value_rhea_latest.pt`
7. **Remote actors** periodically refresh their value net from `value_rhea_latest.pt`

## Why Filesystem (not Redis/gRPC)

- **Samba already works** — no new services to deploy
- **Simple failure model** — if a machine dies, its partial files are either committed or not; no hanging connections
- **Natural batching** — transitions accumulate in files, learner ingests in batches
- **Observable** — you can `ls Z:/fleet/*/transitions/` to see what's happening
- **AWBW_SHARED_ROOT already standardizes the mount point** across all machines

## Running the Full System

### On workhorse1 (learner):
```bash
python -m scripts.train_rhea_value_parallel \
  --checkpoint checkpoints/value_rhea_latest.pt \
  --map-id 171596 \
  --co-p0 14,8,28,7 --co-p1 14,8,28,7 \
  --max-days 30 \
  --rhea-autotune \
  --reward-weight 0.8 --value-weight 0.2 \
  --value-lr 1e-4 \
  --replay-size 50000 \
  --min-replay-before-train 1000 \
  --updates-per-turn 1 \
  --gamma-turn 0.99 \
  --target-update-interval 1000 \
  --grad-clip 1.0 \
  --device cuda \
  --n-envs 22 \
  --gpu-actors 6 \
  --phi-capture-phase-weighting \
  --phi-safe-neutral-opening-mult 1.50 \
  --phi-safe-neutral-early-mid-mult 1.30 \
  --phi-safe-neutral-mid-mult 1.15 \
  --phi-safe-neutral-late-mult 1.00 \
  --phi-safe-neutral-endgame-mult 0.50 \
  --phi-contested-neutral-opening-mult 1.25 \
  --phi-contested-neutral-mid-mult 1.00 \
  --phi-contested-neutral-late-mult 0.90 \
  --phi-capture-opening-end-day 5 \
  --phi-capture-early-mid-end-day 8 \
  --phi-capture-mid-end-day 12 \
  --phi-capture-late-end-day 15 \
  --dual-gradient-self-play \
  --dual-gradient-hist-prob 0.2 \
  --pairwise-zero-sum-reward \
  --poll-remote-transitions-interval 60 \
  --save-every-transitions 1000 \
  --total-transitions 1000000
```

### On any other machine (remote actor):
```bash
python -m scripts.rhea_remote_actor \
  --shared-root Z:\ \
  --machine-id <unique-machine-id> \
  --transition-batch-size 100 \
  --n-envs 8 \
  --device cuda \
  --map-id 171596 \
  --co-p0 14,8,28,7 --co-p1 14,8,28,7 \
  --max-days 30 \
  --rhea-autotune \
  --reward-weight 0.8 --value-weight 0.2 \
  --phi-capture-phase-weighting \
  --dual-gradient-self-play \
  --pairwise-zero-sum-reward
```

## Monitoring

- **Learner logs**: `logs/games_log.jsonl`
- **Remote transition files**: `ls Z:/fleet/*/transitions/`
- **Processed files**: `ls Z:/fleet/*/transitions/*.done`
- **Latest weights**: `Z:/checkpoints/value_rhea_latest.pt`
