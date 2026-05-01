# Solo training on pc-b (Tier 1 walk-away)

## One-line bootstrap

From the repo root (PowerShell):

```powershell
python scripts/start_solo_training.py --machine-id pc-b --auto-apply
```

Omit `--auto-apply` if you only want the fleet orchestrator to **log** train restarts when `proposed_args.json` drifts, without killing `train.py` (same as `--auto-apply` false on the orchestrator).

## Optional: `torch.compile` (policy)

To enable the Inductor path in `rl/self_play.py`, pass **`--torch-compile`** to the bootstrap (sets `AWBW_TORCH_COMPILE=1` for the `train.py` child only), or set the variable in the shell before starting:

```powershell
python scripts/start_solo_training.py --machine-id pc-b --auto-apply --torch-compile
```

```powershell
$env:AWBW_TORCH_COMPILE = '1'
python scripts/start_solo_training.py --machine-id pc-b --auto-apply
```

On **Windows**, you still need MSVC Build Tools (C++) and a Triton setup compatible with your PyTorch; see the README *Optional: torch.compile on Windows*.

## What runs

1. `tools/probe_machine_caps.py --machine-id <id>` → `fleet/<id>/probe.json`
2. `tools/propose_train_args.py --machine-id <id>` → `fleet/<id>/proposed_args.json`
3. `train.py` with Phase 10f-proposed `--n-envs` / `--n-steps` / `--batch-size` plus fixed early-game defaults (see docstring in `scripts/start_solo_training.py`)
4. `scripts/fleet_orchestrator.py --shared-root . --pools <id> --apply` (not dry-run), optionally with `--auto-apply`

On **Windows**, optional `torch.compile` for the policy is **not** on unless you set `AWBW_TORCH_COMPILE=1` and have MSVC Build Tools (C++). See the README section *Optional: torch.compile on Windows*.

## Ctrl+C

The bootstrap traps SIGINT/SIGTERM, stops the orchestrator then `train.py`, waits up to 60 seconds each, then kills if needed. It removes `fleet/<id>/train.pid` and leaves `train_launch_cmd.json` and `applied_args.json` for inspection.

`train.py` handles `KeyboardInterrupt` by saving a checkpoint in the training loop (`rl/self_play.py`). On Windows the bootstrap first sends **Ctrl+Break** to the train process group, then falls back to `terminate` / `kill` after timeouts. If a run dies without saving, resume from the latest `checkpoint_*.zip` / `latest.zip`.

## Logs and status

| Artifact | Purpose |
|----------|---------|
| `logs/start_solo_training.log` | Bootstrap probe/propose failures and child exit |
| `logs/fleet_orchestrator.jsonl` | Per-tick decisions (including `restart_train`, `mcts_health`, eval, curate, …) |
| `logs/fps_diag.jsonl` | One JSON line per rollout: `env_collect_s`, `ppo_update_s`, `env_steps_per_s_*`, `worker_step_time_p99_*`, RSS, etc. (from `rl/self_play._EpisodeDiagnosticsCallback`) |
| `logs/game_log.jsonl` | Per-episode training log |
| `fleet/<id>/status.json` | Trainer heartbeat (when the trainer writes it) |
| `logs/train_reconfig.jsonl` | Soft reconfig vs hard-restart timing (orchestrator + trainer) |

### Throughput and worker stragglers (env collect vs PPO, per-subprocess `step` skew)

- **Per-worker `step` timings** in each Subproc env are controlled by `AWBW_TRACK_PER_WORKER_TIMES`. Set `=1` to force on, `=0` to force off. If unset, training turns them **on** when `AWBW_FPS_DIAG=1` (e.g. `python train.py --fps-diag` or `python scripts/start_solo_training.py --machine-id <id> --fps-diag`) or when `AWBW_MACHINE_ID` is non-empty (e.g. `--machine-id` or a stamped fleet env). The solo bootstrap also sets `AWBW_TRACK_PER_WORKER_TIMES=1` in the child train env.
- **TensorBoard** (SB3’s run dir under `logs/MaskablePPO_*` or per-machine under `logs/<machine_id>/...`): `diag/env_collect_s`, `diag/ppo_update_s`, `diag/env_steps_per_s_collect`, `diag/env_steps_per_s_total` (env+learn cycle), `diag/worker_step_time_p99_max_across_envs` / `..._min_...` and `diag/per_worker_step_time_s_p99` (straggler spread when step tracking is on), plus `diag/episodes_per_rollout`, `diag/ep_len_*`, and RSS scalars. Same definitions as `rl/self_play._EpisodeDiagnosticsCallback`.

## Operator arg override and apply gate

1. Copy `fleet/<id>/proposed_args.json` to `fleet/<id>/operator_train_args_override.json`, keep a top-level `"args": { ... }` map, and **only** list the flags you want to force (e.g. `"--n-envs": 12`, or `"--max-days": 50` for the end-inclusive engine calendar cap — same value as deprecated `"--max-turns"`). Flags you omit keep the normal probe + curriculum + MCTS merge for that tick. Delete the file to revert; the next orchestrator refresh rebuilds `proposed_args.json` without those keys.
2. To **apply** drift (hard restart or allow soft reconfig on PPO geometry), start the solo bootstrap with **`--auto-apply`** so `fleet_orchestrator` is launched with the same flag. That is the gate for Tier-1 train restarts when `proposed_args.json` and `applied_args.json` differ (including after `operator_train_args_override.json` merges on refresh). The `auto_apply` field inside `proposed_args.json` is preserved when the orchestrator regenerates the file, but it does **not** block restarts; only running without orchestrator `--auto-apply` does (audit: `orchestrator_auto_apply_off`).
3. Hash alignment: `applied_args.json` must match the `args` hash implied by `proposed_args.json` if you want **no** restart on the next tick (same as before).

PowerShell examples:

```powershell
Get-Content fleet/pc-b/status.json
Get-Content fleet/pc-b/applied_args.json
Get-Content fleet/pc-b/proposed_args.json
```

## MCTS gate (Phase 11d wired)

The fleet orchestrator now drives the MCTS health gate end-to-end (read-only by default on `pc-b`):

1. **Periodic refresh.** Every tick (`--mcts-health-refresh-every-ticks N`, default `1`), the orchestrator runs `tools/mcts_health.compute_health` against `<shared>/logs/<id>/game_log.jsonl` for each pool machine and atomically writes `<shared>/fleet/<id>/mcts_health.json`. Failures surface as `applied=false` `mcts_health_refresh` audit rows; the tick continues.
2. **Two-cycle hysteresis.** A passing verdict only merges `--mcts-mode` / `--mcts-sims` into `proposed_args.json` after `--mcts-gate-required-consecutive` consecutive passes (default `2`). Below the threshold the orchestrator emits `mcts_gate_pending` rows; a failing or stale verdict resets the streak. Counters are persisted in `--state-file`.
3. **Operator-only on the host.** On the host machine (`--host-machine-id`, default `pc-b`) the merge is skipped unless the operator passes `--enable-mcts-here`; the audit shows `mcts_skip_host`. Auxiliary machines are unaffected. `proposed_mcts_mode == "train_advisor"` is refused defensively on every machine and surfaces as `mcts_refuse_train_advisor`.
4. **Sim-budget escalator EV gate.** Once a machine has flipped to `--mcts-mode != off`, the escalator (`tools/mcts_escalator.compute_sims_proposal`) decides per cycle whether to DOUBLE `--mcts-sims` (16→32→64→128), HOLD, DROP_TO_OFF (on engine desync), or STOP_ASK_OPERATOR (at the cap with positive ROI). DOUBLE is gated on `train/explained_variance ≥ 0.6`, scraped from SB3's TensorBoard event files. Files live under `<shared>/logs/<machine_id>/MaskablePPO_*/events.out.tfevents.*` (multi-machine, future-facing) with fallback to `<shared>/logs/MaskablePPO_*/events.out.tfevents.*` (current solo layout — `LOGS_DIR = REPO_ROOT/logs` in `rl/paths.py`). The default scalar tag is `train/explained_variance`; samples are aggregated with `median` over a 1-hour wall-clock window. When no recent samples exist (cold boot, stale logs) the orchestrator emits an informational `mcts_ev_unavailable` audit row and `build_cycle_result` falls back to `0.0` so the escalator HOLDs on the EV threshold rather than DOUBLEing on noise. Operators can sanity-check the live signal with `python -m tools.tb_scrape_ev --machine-id pc-b`.
5. **Per-row desync attribution (schema_version 2).** `tools/desync_audit.AuditRow.to_json` now writes `schema_version: 2`, `machine_id` (defaults to the `AWBW_MACHINE_ID` env var the fleet layer sets, else `null`), and `recorded_at` (ISO-8601 UTC `YYYY-MM-DDTHH:MM:SSZ` stamped at write time when not supplied). `tools/mcts_eval_summary._count_recent_desyncs` uses these to scope `engine_desyncs_in_cycle` to a single host inside the cycle window: rows with `machine_id` must match, rows with `recorded_at` must lie within `--mcts-cycle-window-seconds`. Older registers (no per-row attribution on any row) keep the file-mtime fallback so historical JSONLs still gate sanely. Consumers that walk register rows must read every new field via `row.get(...)` (the in-tree readers — `tools/cluster_desync_register.py`, `tools/desync_register_diff.py`, `tools/desync_audit_amarriner_live.py` — already do). Practical impact: a single fleet-wide desync no longer fires DROP_TO_OFF on every machine in the pool.

4. **Auto-baseline capture on the eval daemon.** To stop the orchestrator emitting `mcts_baseline_missing` audit rows for any machine where the operator never ran `tools/capture_mcts_baseline.py` by hand, start the per-machine eval daemon with `--capture-mcts-baseline-on-start`:

   ```powershell
   python scripts/fleet_eval_daemon.py --capture-mcts-baseline-on-start
   ```

   The daemon reads `<shared>/fleet/<machine-id>/mcts_off_baseline.json` once at startup (after `bootstrap_fleet_layout`, before the first eval iteration). If the file is missing, or `is_baseline_stale` returns `True` against the configured stale window (`--mcts-baseline-stale-hours`, default `168.0` = one week), it shells out to `python -m tools.capture_mcts_baseline` for that machine with `--games <N>` (`--mcts-baseline-games`, default `200`), `--seed <S>` (`--mcts-baseline-seed`, default `0`), plus any verbatim `--mcts-baseline-extra-args` (shlex-split, e.g. `"--map-id 123858 --tier T3"`). Subprocess timeout is 30 minutes; failure or timeout never tears the daemon down.

   Status strings printed to stdout (one line, prefix `[fleet_eval_daemon] mcts baseline status:`):

   | Status | Meaning |
   |--------|---------|
   | `skipped` | `--capture-mcts-baseline-on-start` not set, or no `AWBW_MACHINE_ID` |
   | `present` | Baseline file exists and is within the stale window — no work done |
   | `captured` | Baseline file was missing; capture CLI succeeded |
   | `stale-recaptured` | Baseline file existed but was past the stale window; capture CLI succeeded |
   | `failed` | Capture subprocess returned non-zero, timed out, or could not be launched (stderr tail logged) |

## Limitations (Tier 1)

- **No Phase 10g curriculum:** `proposed_args.json` only changes when you re-run probe/propose (e.g. machine RAM changed). The orchestrator does not tune args from competence metrics.
- **MCTS:** Health verdicts in `fleet/<id>/mcts_health.json` are surfaced in the orchestrator audit log only. `--auto-apply` does **not** enable MCTS or restart on MCTS files; default remains `--mcts-mode off`.
- **Crash recovery:** If `train.py` or the orchestrator exits unexpectedly, the bootstrap process exits with a non-zero code; it does not respawn children (Tier 2).

## Dry-run

```powershell
python scripts/start_solo_training.py --machine-id pc-b --dry-run-bootstrap
```

Prints the train and orchestrator command lines without starting processes.
