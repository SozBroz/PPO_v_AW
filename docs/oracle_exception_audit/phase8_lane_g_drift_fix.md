# Phase 8 Lane G — Bucket A drift root cause and fix

**Campaign:** `desync_purge_engine_harden`  
**Scope:** Oracle-only (`tools/oracle_zip_replay.py`). No `engine/` changes. Phase 6 Manhattan direct-fire rules unchanged.

## Bug location and root cause

**Primary:** `tools/oracle_zip_replay.py` — `\_oracle_resolve_fire_move_pos` (approximately lines 131–204) and the **`Fire` branch with nested `Move.paths`** (approximately lines 5736–5775).

**One-sentence root cause:** For nested-move `Fire`, the resolver could pick a **reachable waypoint along the reversed path** (e.g. `(15,16)`) that **was not** a valid **Manhattan-1 direct-fire stance** for the defender, while the **JSON path end** `(14,16)` **was** the correct firing tile per `get_attack_targets` / AWBW; the final fallback then returned the snapped `(er,ec)` even when it **could not** strike `(dr,dc)`, producing `_apply_attack: target … not in attack range … from (wrong) (unit_pos=old)` (Bucket A: `unit_pos != from` in the message).

**Secondary (replay truth):** When the **ZIP path end** is **not** in `compute_reachable_costs` for the striker in the re-simulated state (terrain/blocker drift vs live AWBW), `_move_unit` would **reject** the attack even if `move_pos` were correct. The replay still records the strike from the path end.

## Fix applied (summary)

1. **`_oracle_resolve_fire_move_pos` (direct units)**  
   - Walk **`paths.global` from tail to head** and return the **first** waypoint that is **reachable** (`pos in costs`) **and** can strike the resolved defender (`attacks_from(pos)`), **before** `_nearest_reachable_along_path` + ranked search.  
   - **Tighten** the “no strike tile” fallback: **do not** `return (er, ec)` unless `(tr,tc) in get_attack_targets(state, unit, (er, ec))`, so we never deliberately emit an illegal direct-fire `move_pos`.

2. **`Fire` handler (after `fire_pos = _oracle_resolve_fire_move_pos(...)`)**  
   - If the resolved stance **cannot** hit `(dr,dc)` but the **JSON path end** `(er, ec)` **can**, set **`fire_pos = (er, ec)`** (ZIP path tail is the AWBW combat stance).  
   - If **`fire_pos` still not in `compute_reachable_costs`** but `get_attack_targets` accepts the strike from `fire_pos`, call **`state._move_unit_forced(u, fire_pos)`** so the striker sits on the recorded firing tile before `ATTACK` (same family as `export_awbw_replay_actions.py` when `step()` fails on replay replay — narrow, replay-only reconciliation).

Phase 6 Manhattan logic at ~2419 (`_resolve_fire_or_seam_attacker`) and `engine/action.py` direct-range **unchanged**.

## Cross-pattern verification (Bucket A sample)

From `logs/phase7_44_classified.json`, these five rows were **cross-checked** against the **post-fix** diff (`logs/phase8_lane_g_diff_vs_phase6.log`): all were **engine_bug** in Phase 6 and **drift** to `oracle_gap` with `Move: engine truncated path vs AWBW path end; upstream drift` after the Fire fix (replay advances past the former Fire failure; **same** shape: **nested Fire path end vs engine reachability** later in the run).

| games_id | Unit (Phase 7) | Post-fix class |
|----------|----------------|----------------|
| 1623738 (INFANTRY) | oracle_gap (truncated Move) |
| 1620450 (B_COPTER) | oracle_gap (truncated Move) |
| 1630669 (ANTI_AIR) | oracle_gap (truncated Move) |
| 1626642 (MED_TANK) | oracle_gap (truncated Move) |
| 1629563 (MECH) | oracle_gap (truncated Move) |

**Shared shape:** **Fire** `move_pos` / path-end alignment **was** wrong; after fixing, **first** divergence is often a **later** `Move` **where** `selected_move_pos` / `u.pos` **≠** JSON path end **—** **oracle_gap**, not `engine_bug` at the old Fire site.

## Regression test

`tests/test_oracle_fire_resolve.py` — `test_fire_move_pos_prefers_zip_path_end_when_snap_cannot_strike_gl_1618770` — builds the **1618770** path tail / defender relationship with **mocked** `compute_reachable_costs` / `get_attack_targets` so the **reversed waypoint walk** must prefer **`(14,16)`** over **`(15,16)`** when both are reachable.

## Validation results

| Metric | Result |
|--------|--------|
| Targeted pytest | `119 passed, 2 xfailed, 3 xpassed` — `logs/phase8_lane_g_targeted_pytest.log` |
| Full pytest | `258 passed` (≥ 257 baseline) — `logs/phase8_lane_g_full_pytest.log` |
| `desync_audit` (741 games) | `engine_bug` **43**, `ok` **440**, `oracle_gap` **258** — `logs/phase8_lane_g_desync_audit.log` |
| vs Phase 6 `engine_bug` | **149 → 43** (Δ −106) |
| `desync_register_diff` vs `logs/desync_register_post_phase6.jsonl` | **0** `ok → non-ok` regressions; **10** fixed `non-ok → ok`; **96** class drifts (mostly `engine_bug` → `oracle_gap` on later Move) — `logs/phase8_lane_g_diff_vs_phase6.log` |

## Unexpected findings

- **1618770** no longer fails at day 13 Fire; it **replays until** `Move: engine truncated path vs AWBW path end` (later day). **Forced** `fire_pos` **alignment** **unlocks** **long** **replays** **that** **previously** **died** **on** **Fire** **range** **.
- **Bucket B** example **1628198** appears in **fixed** `non-ok → ok` list (10 games) — **not** the Lane G focus but **positive** **side** **effect** **of** **cleaner** **Fire** **path** **.
- Remaining **43** `engine_bug` rows are **mostly** `\_apply_attack` **range** **messages** **still** **orthogonal** **to** **Phase** **6** **diagonal** **hypothesis** **—** **per** **orchestrator** **footnote**, **do** **not** **revert** **Manhattan** **correction** **;** **next** **lanes** **may** **need** **target** **/`move_pos`** **pair** **review** **for** **air** **/ **mech** **clusters** **.

## Escalation

**None** to engine: failures are **oracle** **replay** **reconciliation** **and** **later** **Move** **truncation** **oracle_gap** **,** **not** **engine** **invariant** **bugs** **in** **this** **lane** **.
