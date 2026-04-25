# Thread FIRE — verdicts

Audit of `tools/oracle_zip_replay.py` Fire skip helpers, post-kill duplicates, foot-anchor snap, and silent-RNG fallback (Phase 1 of `desync_purge_engine_harden` campaign). STRICT bar: only AWBW-canon citations earn KEEP.

## Site: `_oracle_set_combat_damage_override_from_combat_info` (`tools/oracle_zip_replay.py`:1723–1779)

**Verdict:** DELETE
**Justification:** Under the STRICT bar, "no AWBW HP → engine `random.randint` via `calculate_damage` / `calculate_counterattack`" is not citable AWBW behavior; it is nondeterministic engine noise. `docs/desync_audit.md` already states that oracle replay without pinned luck makes audits luck-stream-dependent (~103–106), and `tools/desync_audit.py` ~49–53 pins `CANONICAL_SEED` partly because `Fire (no path)` can hit this fallback when combatInfo is missing — so the silent skip is a band-aid that masks an oracle honesty gap, not canonical site rules. The helper still does useful work when it can derive damages from `units_hit_points` (AWBW's post-strike display HP).
**Replacement message:** `raise UnsupportedOracleAction("Fire: combatInfo missing numeric attacker/defender units_hit_points; cannot pin damage/counter to AWBW (oracle would fall back to engine RNG)")`

---

## Site: `_oracle_fire_defender_row_is_postkill_noop` (`tools/oracle_zip_replay.py`:1793–1813)

**Verdict:** KEEP
**Justification:** Docstring ties this to GL 1628985: duplicate `Fire` after the defender died — JSON shows `units_hit_points` ≤ 0 and the recorded tile has no live unit. That matches the repo's accepted position that AWBW's `p:` stream can re-emit `Fire` rows after resolution (see `docs/desync_audit.md` ~255–261 on stale `combatInfo` / attacker tiles and `desync-triage-viewer` §7 on `oracle_fire`). The check keys off AWBW JSON + engine tile emptiness, not zip-only fantasy state.

---

## Site: `_oracle_fire_no_path_postkill_dead_defender_orphan_tile_reoccupied` (`tools/oracle_zip_replay.py`:1860–1928)

**Verdict:** KEEP
**Justification:** Docstring documents real zip-level failure modes with cited games: duplicate re-emit (1628985, 1631194), unsafe apply when anchor/seat drifted (1631858), and the lane-specific fact that `_unit_by_awbw_units_id` is often useless in zip replay (engine `unit_id` ≠ AWBW `units_id`). The gates are heuristic disambiguation for the oracle lane, but anchored to cited replays and the stated zip artifact (post-kill duplicate vs tile reoccupied).

---

## Site: `_oracle_fire_no_path_low_hp_orphan_unmodelled_vs_air` (`tools/oracle_zip_replay.py`:1931–1973)

**Verdict:** KEEP
**Justification:** Docstring cites 1632124 and 1631068 for duplicate `Move: []` / `Fire` while the defender row is orphaned and the tile holds an unrelated live unit; re-firing would hit the wrong unit. Same "stale/orphan row" family as other `oracle_fire` mitigations (`docs/desync_audit.md` § `oracle_fire` resolver; `desync-triage-viewer` §7). Class-based air/copter scoping is a narrowing heuristic after damage-table coverage changed.

---

## Site: `_oracle_fire_no_path_snap_foot_unit_neighbor_to_empty_awbw_anchor` (`tools/oracle_zip_replay.py`:4370–4409)

**Verdict:** DELETE (escalation resolved 2026-04-20 by commander after live drill-down)
**Justification:** Empirical evidence — instrumented audit pipeline ran the helper across 60 sampled GL games. Result: **532 invocations, 0 returned True**. Helper has never fired on the live audit. Probe of the canonical scenario in 1625784 (Day 17 P0 Fire from `(5,10)` → `(5,11)`, AWBW `units_id=192152861`) shows engine and AWBW agree on attacker position, defender position, and unit types. The first guard `_unit_by_awbw_units_id` returns None universally because `desync_audit.py` does not bind AWBW `units_id` to engine `unit_id` (cf. `tools/desync_audit_amarriner_live.py:194-215` which does — but only in the live pipeline). The second guard `state.get_unit_at(anchor)` returns truthy on every observed call because move handlers correctly mirror AWBW paths. Snap is pure scaffolding for a phantom symptom that never materializes on real replays. The synthetic regression test (`tests/test_oracle_fire_lane_a.py::test_foot_snap_single_neighbor_gl1625784`) constructs a board state that does not occur in any audited replay; it must be removed alongside the helper.
**Replacement message:** `raise UnsupportedOracleAction("Fire (no path): AWBW anchor empty; engine has no unit at (anchor_r, anchor_c) — refuse foot snap")` — only if the empty-anchor path is ever genuinely reached.
**Action items:**
1. Delete helper at lines 4370–4409.
2. Delete call site at lines 6550–6559 (call is unconditional after the guard chain).
3. Delete `test_foot_snap_single_neighbor_gl1625784` from `tests/test_oracle_fire_lane_a.py`.

---

## Site: `apply_oracle_action_json` — `Fire` no-path branch (`tools/oracle_zip_replay.py`:6459–6652)

**Verdict:** KEEP
**Justification:** Inline comments cite concrete games: 1629178 / `AttackSeam` (do not early-return on `defender.units_hit_points <= 0` alone), 1628539 (dead attacker: JSON hp≤0 + no `units_id` + empty anchor — three-way duplicate), 1619108 (empty defender tile after post-kill duplicate `Fire`), 1631943 (batched rows / attacker gone from JSON anchor), 1635658 (post-kill defender + no `get_attack_targets` reachability). Aligns with `docs/desync_audit.md` and `desync-triage-viewer` §7: duplicate or obsolete combat rows in the export. The try-other-seat resolver block (~6570–6580) is a documented mitigation for seat/`active_player` skew vs `combatInfo`.

---

## Site: `apply_oracle_action_json` — `Fire` with-path post-kill duplicate snap (`tools/oracle_zip_replay.py`:6669–6690)

**Verdict:** KEEP
**Justification:** Comment cites GL 1635846 (day 12 j=11): when the defender row is a post-kill duplicate, AWBW still records the attacker's post-move path end; without snapping the mover to `(er, ec)`, later envelopes that key off `unit.pos` hit `engine_pos_mismatch_post_move`. AWBW export shape (path geometry in JSON) vs engine position drift — same class of fix as other `_oracle_snap_mover_to_awbw_path_end` uses.
**Engine fix target (if future REPLACE):** If engine always advanced the mover to path end before duplicate rows, oracle might not need the snap — REPLACE-WITH-ENGINE-FIX only after triage proves the engine is wrong relative to AWBW rules, not just export ordering.

---

## Site: `apply_oracle_action_json` — `declared_mover_type` name parse (`tools/oracle_zip_replay.py`:6696–6701)

**Verdict:** DELETE
**Justification:** `except UnsupportedOracleAction: declared_mover_type = None` swallows a signal that the zip named an unmapped or unsupported unit string. Under STRICT bar, that can hide oracle gaps (wrong `units_name` / symbol) and let a looser scan pick the wrong mover. Not AWBW canon — error masking.
**Replacement message:** Remove the broad `except`; or `raise UnsupportedOracleAction(f"Fire Move: invalid units_name/units_symbol for mover: {raw_nm_mv!r}")` from a narrow handler only if the caller distinguishes "unknown name" vs other oracle failures.

---

## Site: `apply_oracle_action_json` — `Fire` with-path post-attack snap (`tools/oracle_zip_replay.py`:6845–6854)

**Verdict:** KEEP
**Justification:** Documents the same state-drift / truncated fire position pattern as `_apply_move_paths_then_terminator` snaps: `_oracle_resolve_fire_move_pos` can leave the attacker short of AWBW's path end; snapping to `(er, ec)` prevents `engine_pos_mismatch_post_move` on following envelopes. Guardrails reference `_ORACLE_MOVE_SNAP_MAX_TELEPORT` in the comment chain. Oracle harness recovery for export/engine mismatch, explicitly scoped (attacker died / occupied / distance cap in surrounding logic).
**Engine fix target (if future REPLACE):** If truncation is purely an engine reachability bug, `engine/action.py` / `engine/game.py` move+attack path application could be fixed so `move_pos` always matches AWBW — only REPLACE after viewer proves engine illegality.

---

## ESCALATIONS (resolved)

1. **`_oracle_fire_no_path_snap_foot_unit_neighbor_to_empty_awbw_anchor` (4370–4409):** **RESOLVED 2026-04-20 → DELETE.** Empirical drill-down: 532 helper invocations across 60 GL games, 0 returned True. Probe of canonical scenario (1625784 Day 17 Fire from `(5,10)` → `(5,11)`) confirmed engine and AWBW agree on positions; helper bails at the `state.get_unit_at(anchor)` guard. The snap is scaffolding for a symptom that never materializes — `_resolve_fire_or_seam_attacker` resolves attackers by position downstream and works correctly. Synthetic regression test must be removed alongside.

---

## Recommendation on `_oracle_set_combat_damage_override_from_combat_info`

**Should the silent fallback to RNG die?** Yes for a STRICT oracle whose job is to replay AWBW's combat: if merged `combatInfo` cannot supply numeric `units_hit_points` for at least one side needed to pin damage/counter, the honest outcome is `UnsupportedOracleAction` (or a dedicated `OracleMissingCombatInfo`), not `random.randint` inside `engine/combat.py::calculate_damage`.

**Reasoning:** AWBW's rolled outcome is only recoverable from logged post-strike HP (per the helper's own docstring ~1731–1738 and `docs/desync_audit.md` ~103–106 on nondeterministic luck). Silent skip makes the audit seed-dependent exactly as `tools/desync_audit.py` ~49–53 warns: `CANONICAL_SEED` stabilizes borderline rows that still use RNG when combatInfo is absent. Removing the silent path forces missing combatInfo to surface as `oracle_gap` / explicit failure instead of hidden luck divergence.

**Impact on `CANONICAL_SEED`:** Pinning remains valuable for non-oracle code paths (`random` used elsewhere), tests, and any residual RNG in combat when overrides are partial or engine features still call `calculate_damage` without a zip-derived roll. It would no longer be justified as a cover for "Fire without combatInfo" if that path raises instead of falling back.

---

## Summary

| Verdict | Count |
|---------|------:|
| KEEP | 6 |
| DELETE | 3 |
| REPLACE-WITH-ENGINE-FIX | 0 |
| ESCALATE | 0 (all resolved) |
