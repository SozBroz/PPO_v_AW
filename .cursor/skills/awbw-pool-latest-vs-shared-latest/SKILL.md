---
name: awbw-pool-latest-vs-shared-latest
description: >-
  Interprets Z:/checkpoints/pool/<machine>/latest.zip vs Z:/checkpoints/latest.zip and runs the symmetric head-to-head promotion gate. Use when the user mentions pool latest vs root latest, Z:\checkpoints\pool\pc-b, promotion, or whether auxiliary output should replace shared latest. When this skill applies, the agent must execute the 1v1 symmetric eval (not only describe it).
---

# Pool `latest` vs shared `latest` (fleet checkpoints)

## Vocabulary (this repo)

- **`Z:`** — Default `AWBW_SHARED_ROOT` on auxiliary machines (`rl/fleet_env.py`). Confirm `AWBW_CHECKPOINT_DIR` / `AWBW_SHARED_ROOT` if paths differ.

- **`Z:/checkpoints/latest.zip`** — Shared fleet policy line (root `latest`).

- **`Z:/checkpoints/pool/<machine_id>/latest.zip`** — Per-aux export (e.g. `pc-b`). Distinct role from root `latest` until published. Newest `checkpoint_*.zip` is the pool glob used for opponent mixing; `latest.zip` is often a snapshot copy.

## Snapshot integrity

`scripts/symmetric_checkpoint_eval.py` (and `bo3_checkpoint_playoff.py`) **copy both zips once** at startup to `<repo>/.tmp/eval_snap_<run_id>_*.zip`. All games load those frozen copies so **shared `latest.zip` can change mid-run** without invalidating the series. JSON output includes `candidate_snapshot`, `baseline_snapshot`, and `eval_snapshot_run_id`.

## Mandatory workflow when the user asks for a promotion check

**Do not stop at file stats.** From repo root, run symmetric head-to-head:

```powershell
cd $env:AWBW_REPO   # or your clone path
python scripts/symmetric_checkpoint_eval.py `
  --candidate "Z:\checkpoints\pool\pc-b\latest.zip" `
  --baseline "Z:\checkpoints\latest.zip" `
  --map-id 123858 --tier T3 --co-p0 1 --co-p1 1 `
  --games-first-seat 4 --games-second-seat 3 --seed 0 `
  --max-env-steps 0 `
  --max-days 150 `
  --json-out logs/promotion_symmetric_pc-b_vs_shared.json
```

Adjust `--map-id` / COs / tier if the user’s training matchups differ.

### Why `--max-env-steps 0` and `--max-days`

- **`--max-env-steps 0`** — Unlimited P0 steps per episode (default **100** truncates early and yields no decisive wins).
- **`--max-days`** — Raises the engine end-inclusive calendar tiebreak above `engine.game.MAX_TURNS` (100). Implemented on `GameState.max_turns` / `make_initial_state(max_days=...)`. Alias: `--max-turns` (deprecated).

### After the run

1. Read **`promotion_heuristic_ok`** in stdout or JSON (candidate ahead overall; no all-loss collapse in either seat).
2. If **`total_decided`** is 0, the eval is **invalid** for promotion — fix caps (raise `--max-days`, ensure `--max-env-steps 0`) and rerun.
3. **Promote** only if the user asked to promote *and* heuristic is true: back up root `latest.zip`, then `Copy-Item` candidate → `Z:\checkpoints\latest.zip` (or follow their chosen map/CO). Do not overwrite shared `latest` on stats alone.

## Related code

- `scripts/symmetric_checkpoint_eval.py` — symmetric 1v1; `--max-days` (`--max-turns` deprecated), `--max-env-steps`.
- `engine/game.py` — `GameState.max_turns` (calendar day cap), `make_initial_state(..., max_days=...)`.
- `rl/env.py` — `AWBWEnv(max_turns=...)` (kwarg name unchanged; value is calendar days).
- `rl/fleet_env.py` — pool layout, `iter_pool_checkpoint_zips`.

## Do not

- Promote from mtime/size alone.
- Skip the script when the user asked for a promotion decision.
- Use default `--max-env-steps 100` for promotion gates without understanding it will truncate.
