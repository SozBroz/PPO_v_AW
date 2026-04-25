# Phase 9 — Lane L: Move family path-end reconciliation

**Campaign:** `desync_purge_engine_harden`  
**Scope:** `tools/oracle_zip_replay.py` only (no `engine/` changes). Phase 6 Manhattan / `_resolve_fire_or_seam_attacker` untouched.

## Bug location and root cause

**Primary:** `tools/oracle_zip_replay.py:3702` — `_apply_move_paths_then_terminator` (post-commit invariant at ~3975).

**One-sentence root cause:** When the AWBW `paths.global` tail is missing from `compute_reachable_costs(state, mover)` in the re-simulated state, the engine stops at `_nearest_reachable_along_path` and the post-terminator invariant raises `Move: engine truncated path vs AWBW path end; upstream drift`, even though replay truth still records the mover on the recorded tail.

**Same message, Fire envelopes (secondary, in scope per Lane J family):** nested-move `Fire` (~5834) and the post-kill `Fire` short-circuit (~5698) had the symmetric gap — `u.pos != (er, ec)` after `ATTACK` when the tail was unreachable in-engine.

## Fix applied

In `_apply_move_paths_then_terminator`, snapshot `json_path_was_unreachable = (json_path_end not in reach)` before the move step; after the step (and **before** `_finish_move_join_load_capture_wait`), call `state._move_unit_forced(u, json_path_end)` and align `state.selected_move_pos` when the tail is empty, already `u`, or held by a legal `units_can_join` partner. Mirror snap for nested-move `Fire` after `ATTACK` (direct units only, transport-stack guard) and for the post-kill duplicate `Fire` branch. Same family as Lane G's `_move_unit_forced` for pre-strike `fire_pos`.

## Sample audit (10 representative gids; `logs/phase9_lane_l_sample_audit.log`)

| Result | Count | Notes |
|--------|------:|--------|
| `ok` | **8** | initial expectation was 5; the in-scope Fire reconcile branch lifted 3 more |
| `oracle_gap` (cleaner messages) | **1** | `1607045` — Join with seat / mover-id drift on tail (escalation below) |
| NEW `engine_bug` | **1** | `1614665` — B_COPTER Bucket A drift in a Fire envelope, surfaced because the fix moved this game past the prior failure point; **out of scope**, queued for Phase 10 |

Note on row counts: extracting on `logs/desync_register_post_phase8_g.jsonl` for `class=oracle_gap` + `approx_action_kind=Move` + truncated-path message yields **182** rows (artifact: `logs/phase9_lane_l_move_targets.jsonl`), not Lane J's pre-phase-8 **112** — the post–Lane-G register mix has more first-failure surfaces in Move geometry. Four parallel validation lanes (**L-VAL-1..4**) will sweep the remaining ~172 rows; this report does not pre-empt their work.

## Pytest (targeted)

`logs/phase9_lane_l_targeted_pytest.log` — **49 collected, all green** for the requested pair (`tests/test_oracle_move_resolve.py` 2 / 2, `tests/test_engine_negative_legality.py` 47 / 47 incl. 3 xpassed).

`tests/test_oracle_move_resolve.py` is committed to disk:

- `test_plain_move_forces_zip_path_end_when_reachability_omits_tail_gl_1607045_shape`
- `test_plain_move_truncation_still_raises_when_tail_occupied_by_other_unit`

## Escalation

- **`games_id` 1607045** — Join nested `Move`; engine has no live unit by PHP `joinID` 191439706 and the tail occupant is the wrong seat. Needs **mover-resolution / seat** work, not blind forcing.
- **`games_id` 1614665** — B_COPTER drift in a Fire envelope (Bucket A); same family as Phase 7 / Lane J non-Move, queued for Phase 10.

---

*"In preparing for battle I have always found that plans are useless, but planning is indispensable."* — Dwight D. Eisenhower, NATO Headquarters, Paris (1957)  
*Eisenhower: Supreme Allied Commander Europe in WWII, later 34th U.S. President.*
