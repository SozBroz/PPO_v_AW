# Phase 2 Worklist — Purge Bogus Exceptions, Re-audit

Generated 2026-04-20 from Phase 1 verdict reports under `docs/oracle_exception_audit/` after all six commander escalations were resolved. Drives the Phase 2 lane of the `desync_purge_engine_harden` campaign.

## Scope

Apply every DELETE / REPLACE-WITH-ENGINE-FIX / PARTIAL verdict in `tools/oracle_zip_replay.py`, run the full desync audit, recluster, append a regression-log entry. Do **not** touch helpers marked KEEP. Do **not** add new logic — purge only.

## Pre-flight checks

1. `cd D:\AWBW`
2. Confirm `git status` is clean (or stash) — the campaign tracks before/after by file diff and audit cluster diff.
3. Confirm `logs/desync_register_pre_purge_2026-04-20.jsonl` and `logs/desync_clusters_pre_purge.json` exist (Phase 0 baseline).
4. Confirm `logs/desync_regression_log.md` already has the BEFORE entry for this campaign (Phase 0).

## Worklist (apply in order, top to bottom)

Line numbers reference `tools/oracle_zip_replay.py` at the pre-purge revision. Re-confirm with Read before each StrReplace; do not blindly trust line numbers if the file has shifted.

### A. DRIFT thread

| # | Lines | Action |
|---|---|---|
| A1 | 851–1055 | DELETE function `_oracle_drift_spawn_unloaded_cargo` |
| A2 | 1058–1203 | DELETE function `_oracle_drift_spawn_mover_from_global` |
| A3 | 4585–4589 | DELETE call site of `_oracle_drift_spawn_mover_from_global` inside `_apply_move_paths_then_terminator`; replace with `raise UnsupportedOracleAction("Move: mover not found in engine; refusing drift spawn from global")` |
| A4 | 1206–1269 | DELETE function `_oracle_drift_spawn_capturer_for_property` |
| A5 | 6162–6174 | DELETE the `Capt (no path)` `capture_points` direct-write early return; replace with `raise UnsupportedOracleAction("Capt no-path: no engine capturer bound; refuse to copy capture_points from PHP snapshot")` |
| A6 | 6175–6183 | DELETE the `Capt (no path)` call to `_oracle_drift_spawn_capturer_for_property` |
| A7 | 7035–7190 | DELETE both Unload drift call sites; replace with `raise UnsupportedOracleAction("Unload: drift recovery disabled; transport/target/loaded cargo do not support UNLOAD")` |
| A8 | 713–732 | DELETE function `_oracle_assign_production_property_owner` |
| A9 | 735–766 | DELETE function `_oracle_snap_neutral_production_owner_for_build` |

### B. BUILD thread

| # | Lines | Action |
|---|---|---|
| B1 | 769–780 | DELETE function `_oracle_build_discovered_matches_awbw_player_map` |
| B2 | 783–796 | DELETE function `_oracle_site_trusted_build_envelope` |
| B3 | 799–815 | DELETE function `_oracle_optional_apply_build_funds_hint` |
| B4 | 1272–1317 | DELETE function `_oracle_drift_teleport_blocker_off_build_tile` |
| B5 | 1356–1367 | DELETE the `u.moved` teleport branch inside `_oracle_nudge_eng_occupier_off_production_build_tile`. KEEP the unmoved legal-step branch (1320–1355 + 1368–1415). |
| B6 | 1416–1418 | DELETE the drift-teleport fallthrough at the bottom of `_oracle_nudge_eng_occupier_off_production_build_tile` |
| B7 | 5535–5639 | KILL `ORACLE_STRICT_BUILD` env flag entirely. The `if strict:` block must always run unconditionally. Then within it: (a) DELETE the funds-bump retry (5585–5592); (b) DELETE the ownership-snap + drift-teleport retries (5595–5632). The remaining branch must always raise `UnsupportedOracleAction` if `funds_after == funds_before and alive_after == alive_before`. |
| B8 | (call sites of B1/B2/B3 in BUILD handler) | Remove invocations and clean up local `trusted` variable references that become dead. |

### C. FIRE thread

| # | Lines | Action |
|---|---|---|
| C1 | 1723–1779 | DELETE silent-RNG fallback inside `_oracle_set_combat_damage_override_from_combat_info`. The helper must `raise UnsupportedOracleAction("Fire: combatInfo missing numeric attacker/defender units_hit_points; cannot pin damage/counter to AWBW")` when it cannot derive damage from logged HP. The "useful work" branch (when both `units_hit_points` are present) is kept. |
| C2 | 4370–4409 | DELETE function `_oracle_fire_no_path_snap_foot_unit_neighbor_to_empty_awbw_anchor` (proven dead code: 532 calls / 0 True across 60 sampled GL games). |
| C3 | 6550–6559 | DELETE call site of C2 inside the `Fire` no-path branch. The downstream `_resolve_fire_or_seam_attacker` already handles positional resolution. |
| C4 | 6696–6701 | DELETE the broad `except UnsupportedOracleAction: declared_mover_type = None` swallow in the `Fire` `declared_mover_type` name parse. Let the exception propagate so unknown unit names surface as oracle gaps. |
| C5 | `tests/test_oracle_fire_lane_a.py` | DELETE `test_foot_snap_single_neighbor_gl1625784` (synthetic fixture for a helper that no longer exists). KEEP the other tests in this file. |

### D. MOVEMENT+REPAIR thread

| # | Lines | Action |
|---|---|---|
| D1 | 4278–4367 | DELETE function `_oracle_snap_mover_to_awbw_path_end` |
| D2 | 4665–4675 | DELETE the post-move snap call inside `_apply_move_paths_then_terminator`; replace with `raise UnsupportedOracleAction("Move: engine truncated path vs AWBW path end; upstream drift")` if a divergence is detected (do NOT just delete the divergence detection — only the forced snap response). |
| D3 | 4907–4987 | DELETE function `_oracle_snap_black_boat_toward_repair_ally` and its call sites in the Repair handler |
| D4 | 5088–5200 | DELETE function `_force_adjacent_repair` |
| D5 | 5259–5260 | DELETE call site of `_force_adjacent_repair`; replace with `raise UnsupportedOracleAction("Repair: no REPAIR in legal actions with synchronized ACTION state")` |
| D6 | 5917–5978 | DELETE the `eng_try` dual-seat loop in the Repair no-path branch. Strict seat attribution: only attempt repair under the seat declared by the envelope. |

### E. Test sweep

After A–D are applied, run the existing test suite to find collateral damage:

```powershell
cd D:\AWBW
python -m pytest tests/ -x --tb=short 2>&1 | tee logs\phase2_pytest_post_purge.log
```

Expected breakages:
- `tests/test_oracle_fire_lane_a.py::test_foot_snap_single_neighbor_gl1625784` — already deleted as part of C5.
- `tests/test_oracle_move_no_unit_drift_spawn.py` — references `_oracle_drift_spawn_mover_from_global` which is deleted in A2. DELETE this test file (it guards behavior the campaign explicitly removes).
- Any other tests that import deleted helpers — DELETE them. Do not preserve scaffolding.
- Tests that pass functions/data through deleted code paths and now fail because the path raises — these are *good* signals; document any genuinely surprising failures in the regression log.

Do NOT modify production code to keep tests passing if the test was guarding deleted behavior. The test goes; the deletion stays.

## Re-audit

The audit CLI is single-process sequential (no `--workers`). Expect ~10–30 minutes wall-clock for the full GL pool.

```powershell
cd D:\AWBW
python -m tools.desync_audit --catalog data\amarriner_gl_std_catalog.json --register logs\desync_register.jsonl --seed 1
python -m tools.cluster_desync_register --register logs\desync_register.jsonl --json logs\desync_clusters.json --markdown logs\desync_clusters.md
```

Capture post-purge artifacts:

```powershell
Copy-Item logs\desync_register.jsonl logs\desync_register_post_purge_20260420.jsonl
Copy-Item logs\desync_clusters.json logs\desync_clusters_post_purge.json
```

## Regression log entry

Append to `logs/desync_regression_log.md`:

```markdown
## 2026-04-20 — Phase 2: oracle exception purge

**Campaign:** desync_purge_engine_harden
**Phase:** 2 (purge bogus exceptions, re-audit)

### Pre-purge baseline (from Phase 0)
- ok: <copy from pre-purge cluster>
- oracle_gap: <...>
- engine_bug: <...>
- replay_no_action_stream: <...>
- TOTAL: <...>

### Post-purge (this entry)
- ok: <fill in>
- oracle_gap: <fill in>
- engine_bug: <fill in>
- replay_no_action_stream: <fill in>
- TOTAL: <fill in>

### Helpers deleted (gross count)
- DRIFT: 9 sites
- BUILD: 8 sites (incl. ORACLE_STRICT_BUILD env flag killed)
- FIRE: 5 sites (incl. one synthetic test removed)
- MOVEMENT+REPAIR: 6 sites

### Tests removed
- tests/test_oracle_fire_lane_a.py::test_foot_snap_single_neighbor_gl1625784
- tests/test_oracle_move_no_unit_drift_spawn.py (entire file)
- (any others discovered during E)

### Notable migrations
- ORACLE_STRICT_BUILD env flag eliminated; BUILD failures always raise.
- combatInfo silent RNG fallback eliminated; missing combatInfo always raises.
- All `_move_unit_forced` oracle invocations eliminated except where commander-approved KEEP logic remains.
```

## Acceptance criteria

1. All A–D actions applied; `tools/oracle_zip_replay.py` no longer references the deleted helpers.
2. `grep -n "_move_unit_forced" tools/oracle_zip_replay.py` returns no hits except inside KEEP-marked helpers (verify against Phase 1 docs).
3. `grep -n "ORACLE_STRICT_BUILD" tools/` returns zero hits.
4. `python -m pytest tests/` runs to completion (failures are fine if they correspond to deleted scaffolding tests; document them).
5. `logs/desync_register_post_purge_2026-04-20.jsonl` and `logs/desync_clusters_post_purge.json` exist.
6. `logs/desync_regression_log.md` has the new entry with both pre- and post-purge cluster counts filled in.
7. New `oracle_gap` rows in the post-purge register are NOT triaged in this phase. Phase 5 will diff and Phase 3+4 may resolve them via engine fixes. Surface them, don't fix them.

## Out of scope (do NOT do in Phase 2)

- Engine legality hardening (Phase 3 — STEP-GATE / SEAM / POWER+TURN+CAPTURE / ATTACK-INV).
- Negative tests (Phase 4).
- Triage of newly-surfaced `oracle_gap` rows (Phase 5 differs against pre-purge).
- Modifying the desync-triage-viewer skill (Phase 5 if closure rules change).
- Touching `KEEP`-marked helpers.
