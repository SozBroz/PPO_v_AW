# Phase 10M — Pytest triage (11 failures from Phase 10L watchdog)

**Date:** 2026-04-21  
**Lane:** Read-only classification for the desync campaign. **No code or test edits** in this lane.  
**Source log:** `logs/phase10l_pytest.log` (11 failed, 455 passed, …).

## ESCALATION (bucket B — Phase 10 regression)

**None.** None of the 11 failures are attributed to a **new regression** introduced specifically by Phase **10A**, **10B**, or **10E** in a way that warrants reverting those lanes.

- **10B / 10E** touch `tools/oracle_zip_replay.py` only. The **10 stack-frame failures** already analyzed in Phase **10J** show **`IllegalActionError` at `GameState.step` STEP-GATE** with **no** `oracle_zip_replay` frame — consistent with **Phase 3 STEP-GATE + test/harness contracts**, not oracle edits.
- **10A** (`data/damage_table.json`, `engine/game.py` MG ammo gating) is **explicitly documented** as unrelated to `test_cargo_dies_with_lander` (Phase 10A report: same failure with 10A reverted but STEP-GATE kept).
- The **11th** failure (`test_picks_nearest_attacker_to_zip_anchor_when_ambiguous`) is an **`AssertionError` on ammo**, not STEP-GATE. Phase **10E** APPLY edits (per `phase10e_suspect_cleanup.md`) are **diagnostics / `want_t` re-raise / Unload int parse** — **no** changes to `_resolve_fire_or_seam_attacker` tie-break or fire application. **No surgical revert of 10E is indicated** for that test.

**Final verdict:** **GREEN** — **0** bucket **B**; **no Phase 10 work to revert** on the evidence below.

---

## Bucket counts

| Bucket | Count | Meaning |
|--------|------:|---------|
| **A** — STALE-TEST / contract | **11** | Tests or harness assume pre–STEP-GATE or pre-mask behavior, or assert mask vs `step` mismatch; **recommendations only** (see table). |
| **B** — NEW REGRESSION (10A/10B/10E) | **0** | — |
| **C** — REAL ENGINE BUG | **0** (see trace **footnote**) | No failure is classified **C** on current evidence; trace replay merits **Phase 11** drill if `oracle_mode=True` does not resolve. |

---

## Per-failure table

| # | Test | Bucket | Suspected cause | Recommended action |
|---|------|--------|-----------------|-------------------|
| 1 | `test_action_space_prune.py::TestWaitPruningOnProperty::test_step_accepts_hand_crafted_wait_on_neutral_city` | **A** | Docstring expects **WAIT** while mask pruned; **STEP-GATE** rejects `action not in get_legal_actions()` (mask size 1). | Update test: assert gate rejection, or `step(..., oracle_mode=True)` if intentionally exercising `_apply_*` / AWBW superset behavior. |
| 2 | `test_action_space_prune.py::TestWaitPruningOnProperty::test_step_accepts_wait_on_partially_capped_city` | **A** | Same as #1 (partial capture). | Same as #1. |
| 3 | `test_black_boat_repair.py::TestBlackBoatLegalRepair::test_repair_not_offered_for_full_hp_full_supply` | **A** | **`AssertionError`:** REPAIR **is** in `get_legal_actions`. `engine/action.py::_black_boat_repair_eligible` documents AWBW **permissive** Repair listing (adjacent ally; no-op handled in `_apply_repair`). Test expects the **old** strict “no REPAIR if full HP/fuel/ammo” mask. | Align test with documented engine/AWBW rule or change engine doc if product intent differs (not a Phase 10 lane issue). |
| 4 | `test_black_boat_repair.py::TestBlackBoatRepairBehaviour::test_self_repair_is_refused` | **A** | Hand-crafted **REPAIR** with `target_pos == boat.pos` not in legal set → **STEP-GATE** before `_apply_repair`. | Use `oracle_mode=True` or assert via `_apply_repair` / error type at apply layer only. |
| 5 | `test_build_guard.py::TestBuildGuard::test_build_on_neutral_factory_rejected` | **A** | Crafted illegal **BUILD**; test expects `_apply_build` rejection, **STEP-GATE** fires first (`SELECT`, mask size 1). | Expect `IllegalActionError` at gate, or `step(..., oracle_mode=True)` then assert no mutation. |
| 6 | `test_build_guard.py::TestBuildGuard::test_build_on_opponent_factory_rejected` | **A** | Same pattern as #5. | Same as #5. |
| 7 | `test_lander_and_fuel.py::TestTransportDeathKillsCargo::test_cargo_dies_with_lander` | **A** | **Battleship** at `(0,2)` attacks lander at `(1,2)` (Manhattan **1**). Indirect min range **2** → **ATTACK** not in mask (`mask size 1`). Phase **10A** report already marks this **pre-existing vs STEP-GATE**, not 10A. | Fix fixture (place battleship at range ≥ 2) or use `oracle_mode=True` for crafted illegal geometry. |
| 8 | `test_naval_build_guard.py::TestNavalBuildTerrain::test_crafted_black_boat_on_base_rejected` | **A** | Crafted **BUILD** Black Boat on base; **STEP-GATE** before `_apply_build` (`SELECT`, mask size 14). | Same pattern as #5–#6. |
| 9 | `test_oracle_zip_replay.py::TestOracleFireNoPathAttacker::test_picks_nearest_attacker_to_zip_anchor_when_ambiguous` | **A** (*see note*) | **`AssertionError`:** `near.ammo` still at max (`9 not less than 9`) after fire — witness assumes **near** tank consumed primary ammo. **Not** `IllegalActionError`; **no** oracle frame. Phase **10E** did **not** change attacker resolution logic. Test was **rewritten in Phase 10A** (defender **TANK** so MG does not mask ammo witness). | **Note:** On this workspace, `pytest` for **this test alone** and **full `test_oracle_zip_replay.py`** both **PASS** (2026-04-21). Phase **10L** failure may be **order-dependent / flaky** (global `unit_id` or suite pollution). Phase **11:** pin deterministic `unit_id`s or isolate module order; **do not** revert 10E for this signal alone. |
| 10 | `test_trace_182065_seam_validation.py::TestTrace182065SeamValidation::test_export_turn_bucket_count_explains_viewer_turn_119` | **A** | `_rebuild_and_emit` calls `state.step(action)` **without** `oracle_mode` on **END_TURN** (`export_awbw_replay_actions.py` ~660). **STEP-GATE:** `END_TURN` not in mask at turn 54, `SELECT`, mask size 1. **Contrast:** same helper **swallows** exceptions on **BUILD** (~641–643). | Thread **`oracle_mode=True`** through export replay (mirror `oracle_zip_replay._engine_step`) **or** align export with strict legality; then re-validate. |
| 11 | `test_trace_182065_seam_validation.py::TestTrace182065SeamValidation::test_full_trace_replays_without_error` | **A** | Plain `state.step(_trace_to_action(entry))` — **BUILD** Mech not in mask at turn 20 (`ACTION`, mask size 1). | Same as #10: replay harness vs STEP-GATE. If **`oracle_mode=True`** still diverges, **escalate to Phase 11** as possible trace/engine parity (**C**). |

---

## Bucket B detail

**No rows.** No full tracebacks beyond log citation required.

---

## Bucket C — Phase 11 charter additions

**None mandatory** from this triage pass.

**Optional Phase 11 follow-ups:**

1. **182065 trace** — If, after replay with `oracle_mode=True`, state still disagrees with the golden trace, treat as **parity / export** defect (true **C**).
2. **Oracle ambiguous attacker test (#9)** — If full-suite-only failure reproduces, bisect **test order** / **`unit_id` determinism**; confirm `_resolve_fire_or_seam_attacker` tie-break `(distance to defender, row, col, unit_id)` vs test expectations.
3. **Black Boat mask** — Confirm product intent matches `_black_boat_repair_eligible` docstring (permissive vs test expectation).

---

## Method notes (Phase 10M)

- Read: `phase10j_validation_rerun.md`, `phase10e_suspect_cleanup.md`, `phase10b_terminator_snap.md`, `phase10a_b_copter_pathing.md`, `phase3_step_gate.md`, `logs/phase10l_pytest.log`, relevant tests, `engine/game.py::step`, `export_awbw_replay_actions._rebuild_and_emit`.
- Isolated pytest: `test_picks_nearest_attacker_to_zip_anchor_when_ambiguous` — **PASSED** locally; full `test_oracle_zip_replay.py` — **62 PASSED**.

---

## One-line summary for command

**All 11 failures classify as STALE-TEST / harness vs STEP-GATE (A); zero Phase 10A/B/E regressions requiring revert (GREEN).**
