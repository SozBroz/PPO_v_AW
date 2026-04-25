# Phase 10J — Full validation re-run (post Phase 10B + 10E)

**Date:** 2026-04-21  
**Scope:** Read-only confirmation lane. Validates that recent `tools/oracle_zip_replay.py` work (Phase 10B move terminator-snap generalization, Phase 10E slack tightening) did not regress the Phase 4 canonical harness.

**Commands mirrored from:** `logs/desync_regression_log.md` (Phase 6 Lane A, §739–744; Phase 9 Lane O, §966–967). Fuzzer parameters taken from `logs/fuzzer_run_n1000_post_phase6.jsonl` (per-game `seed`: **1**, `days_played` cap **30** → `--max-days 30`).

| Suite | Phase 6 / 9 baseline | Phase 10J result | Delta | Verdict |
|--------|----------------------|------------------|-------|---------|
| Full pytest (`tests/` + root `test_*.py`) | Phase 9 Lane O: **261 passed** (subset note: lane-only run); Phase 6 full sweep: **250 passed**, 0 failed | **444 passed**, **10 failed**, 5 skipped, 2 xfailed, 3 xpassed, 3853 subtests passed; wall ~43.4s | +failures vs 0-fail baseline | **RED** (suite not clean) |
| Negative legality `tests/test_engine_negative_legality.py` | 44 passed, 3 xpassed, 0 failed | 44 passed, 3 xpassed, 0 failed | 0 | **GREEN** |
| Property equivalence `tests/test_engine_legal_actions_equivalence.py::test_legal_actions_step_equivalence` | 1 passed, 0 defects | 1 passed, 0 defects | 0 | **GREEN** |
| Self-play fuzzer N=1000 | 1000 games, `defects_by_type` {}, 0 mask_step_disagree (`logs/phase6_fuzzer_n1000.log`) | 1000 games, `defects_by_type` {}; all 1000 JSONL rows have `"defects": []`; wall ~12.4 min | 0 | **GREEN** |

**Artifacts**

- `logs/phase10j_pytest.log`
- `logs/phase10j_neg.log`
- `logs/phase10j_property_equiv.log`
- `logs/phase10j_fuzzer_n1000.log`
- `logs/fuzzer_run_n1000_post_phase10j.jsonl`

## Full pytest — failures (10)

All failures are `engine.game.IllegalActionError` from `GameState.step` STEP-GATE: **`action not in get_legal_actions(state)`** with `oracle_mode=False` (default). **No stack frame involves `tools/oracle_zip_replay.py` or oracle replay imports.** Attribution to Phase 10B / 10E is **not supported** by the tracebacks.

| Test | Mechanism (short) |
|------|-------------------|
| `test_action_space_prune.py::TestWaitPruningOnProperty::test_step_accepts_hand_crafted_wait_on_neutral_city` | Hand-crafted WAIT on capturable tile; docstring expects `step` to accept while mask prunes — gate rejects |
| `test_action_space_prune.py::TestWaitPruningOnProperty::test_step_accepts_wait_on_partially_capped_city` | Same pattern (partial capture) |
| `test_black_boat_repair.py::TestBlackBoatLegalRepair::test_repair_not_offered_for_full_hp_full_supply` | REPAIR appears in `get_legal_actions` when test expects it absent |
| `test_black_boat_repair.py::TestBlackBoatRepairBehaviour::test_self_repair_is_refused` | Self REPAIR not in legal set → `step` raises |
| `test_build_guard.py::TestBuildGuard::test_build_on_neutral_factory_rejected` | Crafted illegal BUILD; expectation is `_apply_build` rejection, but STEP-GATE blocks before apply |
| `test_build_guard.py::TestBuildGuard::test_build_on_opponent_factory_rejected` | Same |
| `test_lander_and_fuel.py::TestTransportDeathKillsCargo::test_cargo_dies_with_lander` | `IllegalActionError` via gate (detail in log) |
| `test_naval_build_guard.py::TestNavalBuildTerrain::test_crafted_black_boat_on_base_rejected` | Crafted BUILD; gate vs deep reject |
| `test_trace_182065_seam_validation.py::TestTrace182065SeamValidation::test_export_turn_bucket_count_explains_viewer_turn_119` | Trace replay: `END_TURN` not legal at turn 54 |
| `test_trace_182065_seam_validation.py::TestTrace182065SeamValidation::test_full_trace_replays_without_error` | Trace replay: `BUILD` Mech not in mask at turn 20 |

**Classification:** Regression vs Phase 6/9 **in the narrow sense of “oracle_zip_replay edits”** — **no.** These are **engine STEP-GATE / test contract** mismatches (mask vs `step`, or replay export vs strict legality). **Recommendation:** triage under `engine/game.py` STEP-GATE policy and affected tests (`oracle_mode=True` where intentional, or update tests to only assert via legal actions). **Not** a revert of Phase 10B/10E on the evidence above.

## Counts summary

- **New failures vs Phase 6/9 “0 failed” full-suite expectation:** **10** (full pytest).
- **Canonical Phase 4 harness (neg + property + fuzzer):** **0** new failures; fuzzer defects **0**.

## Verdict

- **Phase 10J mission (oracle 10B + 10E did not break the canonical harness):** **GREEN** — negative legality, property equivalence, and N=1000 fuzzer match Phase 6 baselines.
- **Repository full pytest bar (“no test that passed before now fails”):** **RED** — 10 failing tests as listed; causes are **orthogonal** to `oracle_zip_replay.py` per stack traces.

**Combined counsel:** Ship confidence on oracle lanes; schedule **Phase 11** (or parallel) to reconcile STEP-GATE with tests that still assume pre-gate behavior (carve-outs, `oracle_mode`, or export path).
