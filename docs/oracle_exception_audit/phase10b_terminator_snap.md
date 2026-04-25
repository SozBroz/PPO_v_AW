# Phase 10B — Terminator snap generalization (Join / Load / nested-Fire / AttackSeam)

**Campaign:** `desync_purge_engine_harden`  
**Scope:** `tools/oracle_zip_replay.py` only (no `engine/` edits). Phase 6 Manhattan, `_resolve_fire_or_seam_attacker`, Lane O tightenings untouched.

## Pattern (extends Phase 9 Lane L)

Lane L: snapshot whether the JSON path tail `(er, ec)` is in `compute_reachable_costs(state, mover)` *before* the `SELECT`+`move_pos` pair; after the engine move, **before** `_finish_move_join_load_capture_wait` / nested-Fire terminator work, reconcile stance when the tail was unreachable.

**Phase 10B additions:**

1. **`_oracle_path_tail_is_friendly_load_boarding`** (`tools/oracle_zip_replay.py` ~3702): tail holds a friendly transport with room that can load the mover (mirrors `get_loadable_into` / capacity).

2. **`_oracle_path_tail_occupant_allows_forced_snap`** (~3721): empty / self / **JOIN** only — **not** load. For **LOAD**, `_move_unit_forced(mover, transport_hex)` before `ActionType.LOAD` would co-place mover and transport; `GameState._apply_load` expects `unit.pos` at the pre-commit tile and `move_pos` on the transport (~`engine/game.py` `_apply_load`). So for load tails we **only** set `state.selected_move_pos` to the JSON tail when `json_path_was_unreachable`, never forced position.

3. **AttackSeam** (~3964): previously `seam_attack_target is not None` forced `json_path_was_unreachable = False`, disabling reconciliation for seam envelopes. We now always compute `json_path_was_unreachable = (json_path_end not in reach)` before picking `_furthest_reachable_path_stop_for_seam_attack` vs `_nearest_reachable_along_path`, so Join/Load/empty-tail snap applies when the seam move is also truncated in the usual way.

4. **Nested `Fire`**: post-kill duplicate Fire (~5764) and post-`ATTACK` nested move (~5960s in file; search `_oracle_path_tail_is_friendly_load_boarding`) mirror the same split: load boarding → `selected_move_pos` only; join/empty/self → `_move_unit_forced` + `selected_move_pos`.

## Bug locations (file:line, shape)

| Shape | Primary site |
|-------|----------------|
| Move / Join / Load / Capt / AttackSeam (nested move) | `_apply_move_paths_then_terminator` — reach snapshot ~3964; snap block ~4005–4025 |
| Helpers | `_oracle_path_tail_is_friendly_load_boarding` ~3702; `_oracle_path_tail_occupant_allows_forced_snap` ~3721 |
| nested Fire (post-kill) | `_apply_oracle_action_json_body` `Fire` branch, post-kill block ~5764 |
| nested Fire (post-ATTACK) | same `Fire` branch, after `ATTACK` step ~5960–5980 |

## Pytest (targeted)

Command:

`python -m pytest tests/test_oracle_terminator_snap.py tests/test_oracle_move_resolve.py tests/test_engine_negative_legality.py -v --tb=short`

**Result:** 50 passed, 3 xpassed (unchanged neg-test xpasses). Log: `logs/phase10b_targeted_pytest.log`.

New file: `tests/test_oracle_terminator_snap.py` — JOIN, LOAD, CAPT integration tests with patched `compute_reachable_costs`, plus helper smoke test.

## Per-bucket audit (`logs/phase10b_targets.jsonl`, 39 rows)

Harness: `tools.desync_audit._audit_one` with `seed=1`, `MAP_POOL_DEFAULT`, `MAPS_DIR_DEFAULT`, catalog `data/amarriner_gl_std_catalog.json`, zips `replays/amarriner_gl/{games_id}.zip`. Output summarized in `logs/phase10b_audit.log` / `logs/phase10b_audit_results.jsonl`.

| `approx_action_kind` (register) | Rows | Outcome (this change) |
|---------------------------------|-----:|------------------------|
| AttackSeam | 1 | **1 FLIPPED_OK** (`games_id` 1634072) |
| Capt | 2 | 2 STUCK (same truncated-path `oracle_gap`) |
| Fire | 24 | 24 STUCK |
| Join | 5 | 5 STUCK |
| Load | 2 | 2 **escalated** to `engine_bug` (`Illegal move: … not reachable`) — no longer the truncated-path message; see escalations |
| Move | 5 | 5 STUCK |

**Net:** 1 / 39 FLIPPED_OK; 36 / 39 same-family STUCK; 2 / 39 new `engine_bug` surface (Load gids **1605367**, **1630794**).

## Escalations / follow-ups

- **Load → `engine_bug` (1605367, 1630794):** First divergence is now `Illegal move: … is not reachable` (engine legality), not `Move: engine truncated path…`. Treat as **progression** past the oracle-only Move tail assert; root move/reachability likely still drift — **queue Phase 11** (oracle move geometry or engine), not a reopen of Phase 6 Manhattan.

- **Still STUCK (36):** Remaining shapes are dominated by **nested Fire** (24) and **Join** (5) with mover/seat/combat stack issues called out in Lane L for e.g. 1607045 — outside pure tail snap.

- **`phase10c_move_truncate_subshape_classification.md`:** Not present in-repo when Phase 10B landed; classification was taken from Phase 9 validation docs + register `approx_action_kind`.

## Artifacts

- `logs/phase10b_targets.jsonl` — 39 filtered rows  
- `logs/phase10b_audit.log`, `logs/phase10b_audit_results.jsonl`  
- `logs/phase10b_targeted_pytest.log`
