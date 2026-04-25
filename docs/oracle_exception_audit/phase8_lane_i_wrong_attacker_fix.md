# Phase 8 Lane I — Bucket B wrong-attacker resolution

**Campaign:** `desync_purge_engine_harden`  
**Scope:** `tools/oracle_zip_replay.py` only (`_resolve_fire_or_seam_attacker`); no `engine/` changes.

## Bug location and root cause

**File / symbol:** `tools/oracle_zip_replay.py`, `_resolve_fire_or_seam_attacker` (approximately lines 2305–2488).

**Root cause:** After the AWBW-declared attacker (`units_id`) was found on the active engine seat but could not legally strike the resolved defender tile, the resolver still accepted an alternate friendly that could strike (anchor occupant, `cands` tie-break by distance to target, or `adj_one`), which could be a **different** unit than the envelope’s `units_id`—leading to `_apply_attack` range errors with `unit_pos == from` but Manhattan attacker–target distance ≫ 1 (Phase 7 Bucket B: games **1628198**, **1633184**).

## Fix applied

Introduced a **seat pin** when `_unit_by_awbw_units_id` resolves `awbw_units_id` to a live unit on `engine_player`: that unit must be the striker or resolution fails fast.

1. **Anchor tile:** Return the anchor occupant only if `pin_active` is false *or* `unit_id` matches `awbw_units_id` (so we do not return a different unit just because it can attack from the anchor).
2. **Broad striker list (`cands`):** If pinned and any alternate striker exists, raise `UnsupportedOracleAction` with a clear upstream-drift message instead of substituting.
3. **`adj_one` fallback:** If pinned, require the unique Manhattan-adjacent striker whose `unit_id` matches `awbw_units_id`; otherwise raise the same way (covers cases where `cands` was empty but geometry-based adjacency still listed candidates).

The Phase 6 Manhattan filter on direct-fire adjacency in this function (the `abs(er - tr) + abs(ec - tc) == 1` check) was **not** modified.

## Lane G (Fire nested `Move` handler)

Edits are confined to `_resolve_fire_or_seam_attacker` and its docstring. The `kind == "Fire"` branch that walks nested `Move` paths (Lane G territory) was not touched.

## Per-game outcomes (desync_audit, `--seed 1`)

| games_id | Before (post–Phase 6 register) | After |
|---------|--------------------------------|--------|
| **1628198** | `engine_bug` — `_apply_attack: target (11, 6) not in attack range for NEO_TANK from (12, 9) (unit_pos=(12, 9))` | **`ok`** — 30/30 envelopes, 431 actions applied |
| **1633184** | `engine_bug` — `_apply_attack: target (15, 4) not in attack range for INFANTRY from (14, 7) (unit_pos=(14, 7))` | **`oracle_gap`** — first failure is now earlier: `UnsupportedOracleAction: Move: engine truncated path vs AWBW path end; upstream drift` at envelope 25, day 13, `actions_applied=368` (Fire-range bug no longer first divergence) |

Artifact: `logs/phase8_lane_i_two_games.jsonl` / `logs/phase8_lane_i_two_games.log`.

## Pytest

- Targeted: `tests/test_oracle_fire_resolve.py` + `tests/test_engine_negative_legality.py` — green (56 passed, 2 xfailed expected, 3 xpassed).
- Full: `python -m pytest tests/ test_oracle_zip_replay.py --tb=line -q` — **259 passed**, 2 xfailed, 3 xpassed (log: `logs/phase8_lane_i_full_pytest.log`).

New regression: `TestOracleFireResolve.test_resolve_raises_when_pinned_awbw_units_id_cannot_strike_but_alternate_can_bucket_b`.

## Escalation

None for `engine/`: failures are classified as `oracle_gap` / upstream drift where the oracle refuses to attribute the strike to a different unit than AWBW’s `units_id`. **1633184** still does not complete the full replay; the remaining blocker is the existing Move truncation check, not the Fire attacker resolver.
