# Phase 10R — Stale-test cleanup (Phase 10M bucket A)

**Date:** 2026-04-21  
**Scope:** Align eight STALE-TEST cases with STEP-GATE (`GameState.step` + `IllegalActionError`); defer Phase 10M items #9–#11 to Phase 11 where required.

## Per-test table

| # | Test | Phase 10M recommendation taken | Edit shape | Result |
|---|------|------------------------------|------------|--------|
| 1 | `test_action_space_prune.py::TestWaitPruningOnProperty::test_step_accepts_hand_crafted_wait_on_neutral_city` | Assert STEP-GATE rejection (`IllegalActionError`); staged move unchanged (`selected_move_pos`, property capture points) | `assertRaises(IllegalActionError)`; import `IllegalActionError` | **pass** |
| 2 | `test_action_space_prune.py::TestWaitPruningOnProperty::test_step_accepts_wait_on_partially_capped_city` | Same as #1 | Same | **pass** |
| 3 | `test_black_boat_repair.py::TestBlackBoatLegalRepair::test_repair_not_offered_for_full_hp_full_supply` | **Option A:** REPAIR remains in mask; `step` REPAIR on full ally is treasury/state no-op | Rewrote body; class docstring softened; method docstring cites 10M + `_black_boat_repair_eligible` | **pass** |
| 4 | `test_black_boat_repair.py::TestBlackBoatRepairBehaviour::test_self_repair_is_refused` | Gate: `IllegalActionError`; then `oracle_mode=True` to assert `_apply_repair` self-target no-op | Two-part test | **pass** |
| 5 | `test_build_guard.py::TestBuildGuard::test_build_on_neutral_factory_rejected` | Expect `IllegalActionError` at STEP-GATE | `assertRaises`; module docstring updated | **pass** |
| 6 | `test_build_guard.py::TestBuildGuard::test_build_on_opponent_factory_rejected` | Same as #5 | Same | **pass** |
| 7 | `test_lander_and_fuel.py::TestTransportDeathKillsCargo::test_cargo_dies_with_lander` | Battleship at Manhattan ≥2 from lander: `(0, 0)` vs lander `(1, 2)` → distance 3 | One-line position fix + comment (Phase 10M) | **pass** |
| 8 | `test_naval_build_guard.py::TestNavalBuildTerrain::test_crafted_black_boat_on_base_rejected` | Expect `IllegalActionError` at STEP-GATE | `assertRaises`; docstring | **pass** |

**Note:** Test #3 **method name** is left as `test_repair_not_offered_*` for audit continuity; behavior now matches permissive REPAIR listing (docstring states this).

## Pytest delta

| Metric | Phase 10L (reference) | After Phase 10R |
|--------|----------------------|-----------------|
| Failed | 11 | **2** |
| Passed | 455 | **464** (full run; includes skips/xfails) |

**Residual failures (deferred to Phase 11):**

- `test_trace_182065_seam_validation.py::TestTrace182065SeamValidation::test_export_turn_bucket_count_explains_viewer_turn_119` — **fail**
- `test_trace_182065_seam_validation.py::TestTrace182065SeamValidation::test_full_trace_replays_without_error` — **fail**

**`test_oracle_zip_replay.py::TestOracleFireNoPathAttacker::test_picks_nearest_attacker_to_zip_anchor_when_ambiguous` (#9):** **pass** on this full-suite run (Phase 10M: order-dependent / flaky; treat as Phase 11 if it regresses in CI).

## Regression gate results

| Gate | Command / check | Floor | Result |
|------|-----------------|-------|--------|
| Negative legality | `pytest tests/test_engine_negative_legality.py -v` | 44 passed, 3 xpassed, 0 failed | **GREEN** (44 passed, 3 xpassed) |
| Manhattan canon | Subset of negative legality | — | **GREEN** (held) |
| Andy SCOP | `pytest tests/test_andy_scop_movement_bonus.py` | 2 passed | **GREEN** |
| Property / step equivalence | `pytest tests/test_engine_legal_actions_equivalence.py::test_legal_actions_step_equivalence` | 1 passed | **GREEN** (~29s) |
| Full pytest | `pytest` | Only deferred trace pair failing | **GREEN** (2 failed, none outside deferred scope) |

## Deferred edge cases

- **#9 (oracle ambiguous attacker):** Not failing in this workspace run; keep **unit_id** pinning / order bisect on Phase 11 if full-suite CI reproduces Phase 10L.
- **#10–#11 (182065 trace):** Require `oracle_mode=True` through `tools/export_awbw_replay_actions.py::_rebuild_and_emit` (Lane 10G high-risk); **no test/engine edits** in Phase 10R per constraints.

## Phase 11 charter additions

1. **Export replay harness:** Thread `oracle_mode=True` into `_rebuild_and_emit` (and align swallow/re-raise policy with `oracle_zip_replay._engine_step`) so trace golden tests replay illegal AWBW envelopes without STEP-GATE abort; re-validate trace 182065.
2. **Oracle fire ambiguous attacker (#9):** If flaky: deterministic `unit_id`s, test order isolation, confirm tie-break vs `_resolve_fire_or_seam_attacker`.
3. **Optional:** Product review of Black Boat REPAIR mask vs AWBW wiki (already documented permissive in `_black_boat_repair_eligible`).

## Verdict

**GREEN:** All eight targeted tests updated and passing individually; regression gates held; full pytest shows **2** failures, both the deferred 182065 trace pair. Failure count improved **11 → 2** (better than the **11 → 3** floor tied to the three originally listed deferrals, because #9 passes here).
