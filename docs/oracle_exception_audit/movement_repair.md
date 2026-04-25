# Thread MOVEMENT+REPAIR — verdicts

Audit of `tools/oracle_zip_replay.py` movement-snap, repair-forcing, and active-player snap helpers (Phase 1 of `desync_purge_engine_harden` campaign). STRICT bar: only AWBW-canon citations earn KEEP.

## Site: `_oracle_ensure_envelope_seat` (`tools/oracle_zip_replay.py`:282–311)

**Verdict:** KEEP
**Justification:** Bar explicitly allows helpers grounded in site zip / `p:` stream behavior, not only in-match rules. Docstring states AWBW archives can interleave both players within the same `day` without an `End` between envelopes while the engine assumes strict alternating half-turns; register taxonomy still clusters `active_player` issues as `oracle_turn_active_player` (`docs/desync_audit.md`, subtype note). After `_oracle_advance_turn_until_player`, snapping `active_player` and clearing selection is a replay harness alignment with that envelope's declared seat — documented as `oracle_turn_active_player`.

---

## Site: `_oracle_snap_active_player_to_engine` (`tools/oracle_zip_replay.py`:314–350)

**Verdict:** KEEP
**Justification:** Same zip / half-turn bookkeeping class as above: docstring ties drift to nested `WAIT` / `END_TURN` sequencing and `oracle_turn_active_player`, and explains why `state.units[eng]` repair resolution needs `active_player` aligned with the Black Boat's engine seat. The branch that sets `active_player` without nuking selection when the selected unit already belongs to `want_eng` preserves a real post-move `ACTION` state — still replay-only, not an in-game rule violation. Citation: local docstring + `docs/desync_audit.md` (lines 232–233) on `active_player` clustering.

---

## Site: `_resolve_active_player_for_repair` (`tools/oracle_zip_replay.py`:353–362)

**Verdict:** KEEP
**Justification:** Thin wrapper around `_oracle_snap_active_player_to_engine` with the same rationale and citations.

---

## Site: `_oracle_snap_mover_to_awbw_path_end` (`tools/oracle_zip_replay.py`:4278–4367)

**Verdict:** DELETE
**Justification:** AWBW resolves movement via legal paths and occupancy; the engine's normal path is `GameState._move_unit`, which validates reachability and fuel (`engine/game.py` 1311–1337). This helper uses `_move_unit_forced` (`game.py` 1339–1352), which explicitly bypasses validation "for trace-replay tools" — i.e. not AWBW gameplay. Teleporting to the JSON path end to mask drift is not supported by cited AWBW rules; it only hides oracle/engine divergence (including `engine_pos_mismatch_post_move` mentioned in the docstring).
**Replacement message:** `raise UnsupportedOracleAction("Move: post-commit engine position disagrees with AWBW path end; refusing _move_unit_forced snap to zip coordinates")`

---

## Site: `_apply_move_paths_then_terminator` — path start vs `path_anchors` (`tools/oracle_zip_replay.py`:4622–4630)

**Verdict:** KEEP
**Justification:** Adjusts `start` when the unit is not listed in decompressed path anchors but is physically on `(sr, sc)`, so `SELECT_UNIT` hits the real tile. Addresses recorded path JSON shape (omitted waypoint) vs engine needs for legal selection, as stated in the comment (`oracle_move_terminator` cluster). Same "site export / envelope data" class as `desync_audit.md` triage on unmapped shapes — not an assertion that units teleport in AWBW.

---

## Site: `_apply_move_paths_then_terminator` — post-move snap call (`tools/oracle_zip_replay.py`:4665–4675)

**Verdict:** DELETE
**Justification:** Invokes `_oracle_snap_mover_to_awbw_path_end` after normal `_engine_step` moves — same analysis as that helper: forced repositioning after commit is not AWBW-canonical movement; duplicates the per-envelope teleport mitigation noted in the helper docstring (`oracle_state_sync` reference) and masks drift rather than failing loud.
**Replacement message:** `raise UnsupportedOracleAction("Move: engine truncated path vs AWBW path end; remove post-move snap or fix upstream drift instead of _move_unit_forced")`
**Engine fix target (if REPLACE):** If triage shows systematic wrong truncation, consider `engine/game.py` / reachability helpers used by `_nearest_reachable_along_path` — but not proven here; default DELETE.

---

## Site: `_finish_move_join_load_capture_wait` — empty `paths.global` fallback (`tools/oracle_zip_replay.py`:4684–4723)

**Verdict:** KEEP
**Justification:** When `paths.global` is empty, the code uses `selected_move_pos` or `selected_unit.pos` from the already committed engine step — does not invent a tile off the legal move graph; reconciles a degenerate envelope to the engine's committed terminator tile. Comment ties this to real register IDs (1635418 / 1635708) and "site zips can disagree by one square" in the `end` computation, matching the documented focus on export / JSON quirks.

---

## Site: `_oracle_snap_black_boat_toward_repair_ally` (`tools/oracle_zip_replay.py`:4907–4987)

**Verdict:** DELETE
**Justification:** Repeated `_move_unit_forced` toward an ally (GL 1634030 cited) moves the Black Boat without `compute_reachable_costs` / `_move_unit` validation — same non-canonical teleport family as `_oracle_snap_mover_to_awbw_path_end`. AWBW naval movement consumes fuel along legal paths; this loop is pure oracle geometry repair.
**Replacement message:** `raise UnsupportedOracleAction("Repair: Black Boat not adjacent to resolved repair target; refusing forced sea slides via _move_unit_forced")`

---

## Site: `_force_adjacent_repair` and call site (`tools/oracle_zip_replay.py`:5088–5200, 5259–5260)

**Verdict:** DELETE
**Justification:** `get_legal_actions` already emits `REPAIR` for every orthogonally adjacent ally because `_black_boat_repair_eligible` is effectively always true for non-`None` allies (`engine/action.py` 586–608, 679–689), matching `_apply_repair`'s wiki-aligned behavior (resupply even when heal is skipped; `engine/game.py` 895–912). So "mask omits REPAIR" is not explained by AWBW being stricter than the engine — it indicates wrong stage/tiles/selection or prior forced geometry. The nested `_move_unit_forced` mid-function (5154–5159) then applies another non-canonical move before `_engine_step(REPAIR, ...)`, i.e. it can step a crafted repair off a masked state. STRICT bar: DELETE and surface `UnsupportedOracleAction` rather than force-step.
**Replacement message:** `raise UnsupportedOracleAction("Repair: no REPAIR in legal actions with synchronized ACTION state; refusing _force_adjacent_repair")`

---

## Site: Repair handler — no-path branch opening (`tools/oracle_zip_replay.py`:5917–5978)

**Verdict:** DELETE the `eng_try` dual-seat loop (escalation resolved 2026-04-20 by commander — strict seat attribution).
**Justification:** Commander ruled `Repair` envelopes must be attributed strictly to the seat declared by the `p:` envelope; trying the other engine seat is non-canon hedging that hides oracle/engine attribution bugs.
**Replacement message:** `raise UnsupportedOracleAction("Repair: no Black Boat resolves under strict seat attribution; refusing dual-seat fallback")`

---

## Site: `_resolve_repair_target_tile` — HP restore and `boat_bid` swallow (`tools/oracle_zip_replay.py`:2479–2496)

**Verdict:** KEEP
**Justification:** Restoring `pairs` from `pairs_unc` when HP hints over-filter after drift (2480–2483) avoids picking no target when the hint is stale — explicitly documented in-comment; prefers honest multi-candidate resolution over a false empty set. Swallowing `UnsupportedOracleAction` from `_repair_boat_awbw_id` (2493–2496) falls back to other disambiguators instead of aborting the whole repair when the `unit` field is partially malformed — within oracle parsing tolerance, not a gameplay teleport.

---

## ESCALATIONS (resolved)

1. **`Repair` handler no-path branch (`oracle_zip_replay.py` ~5917–5978):** **RESOLVED 2026-04-20 → DELETE the `eng_try` loop.** Commander: strict seat attribution; mismatch surfaces as oracle gap.

---

## Recommendation on `_force_adjacent_repair`

**DELETE.** `_black_boat_repair_eligible` is already maximally permissive (`action.py` 679–689), and `_apply_repair` implements wiki-aligned resupply-even-when-heal-skips (`game.py` 895–912). The gap is not "AWBW allows repairs the mask forbids"; it is oracle state drift or invalid ACTION geometry, compounded by `_move_unit_forced`. Proposed engine change if REPLACE were chosen: none for the REPAIR mask — first remove forced teleports and seat hacks, then re-triage; only if concrete games show `get_legal_actions` missing `REPAIR` with a valid ortho ally at committed `move_pos`, investigate `get_legal_actions` / `ActionStage` for that snapshot (`engine/action.py` Black Boat branch under `ACTION`).

---

## Summary

| Verdict | Count |
|---------|------:|
| KEEP | 7 |
| DELETE | 6 |
| REPLACE-WITH-ENGINE-FIX | 0 |
| ESCALATE | 0 (all resolved) |
