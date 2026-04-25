# Phase 3 — Thread STEP-GATE (SOURCE OF TRUTH)

**Campaign:** `desync_purge_engine_harden`
**Plan:** `.cursor/plans/desync_purge_engine_harden_d85bd82c.plan.md` § Phase 3 Thread STEP-GATE
**Owner:** Opus (this thread)
**Date:** 2026-04-20

## Mission

Make `GameState.step` enforce its own legality contract. After this thread,
`get_legal_actions(state)` is the single source of truth for what the engine
will accept; `step` raises `IllegalActionError` (a `ValueError` subclass) on
anything outside the mask. Oracle / replay tooling opts out via
`oracle_mode=True` at the chokepoint (`tools/oracle_zip_replay.py::_engine_step`).

This converts every legal-action gap into an automatic `step` rejection. The
remaining Phase 3 threads (SEAM, POWER+TURN+CAPTURE, ATTACK-INV) now write
their primary fix into `get_legal_actions` / `get_attack_targets` and inherit
strict enforcement for free.

## Diff summary

### `engine/game.py` (+34 / −2)

* `IllegalActionError(ValueError)` declared at module scope (right after
  `MAX_TURNS`, ~line 36).
* `GameState.step` signature: added keyword-only `oracle_mode: bool = False`.
* New invariant block at the top of `step` (~lines 198–230): when
  `oracle_mode is False`, the action must appear in
  `get_legal_actions(self)`; otherwise raises `IllegalActionError` with
  turn / active_player / action_stage / mask-size context.
* No `_apply_*` handlers touched. No legality logic widened. Defense-in-depth
  asserts inside handlers remain other threads' job.

### `tools/oracle_zip_replay.py` (+8 / −1)

* `_engine_step` (~line 84) now passes `oracle_mode=True` to `state.step`.
  This is the **only** edit to the oracle layer — the helper is the
  chokepoint every oracle action funnels through, so threading the kwarg
  here covers every call site without a per-site sweep.
* Comment block at the function head documents the contract: oracle envelope
  reconstruction is intentionally outside the mask (capture-timer
  convergence, multi-action terminators, drift snaps).

### Internal `engine/` step call sites

`grep -n "\.step(" engine/` returns one hit:
`engine/belief.py:14` — docstring example, not an executable call. **No
internal engine code path issues `state.step`**, so RL / agent / test
callers inherit the strict default automatically. Nothing else to thread.

### `Action` equality

`Action` is `@dataclass` (default `eq=True`); `__eq__` compares fields
elementwise (tuples, IntEnums, `Optional`). No change required for the
membership test `action not in legal` to work.

## Smoke test

Throwaway script `tools/_phase3_step_gate_smoke.py` reproduced **Phase 2.5
Probe 5** (END_TURN with an unmoved Infantry on the active seat).

```
[PASS] strict path rejected END_TURN: IllegalActionError: Action Action(END_TURN)
       not in get_legal_actions() at turn=1 active_player=0 action_stage=SELECT;
       mask size=1
[PASS] oracle_mode bypass succeeded (active_player 0 -> 1)
```

Both invariants hold:

1. Default (`oracle_mode=False`) raises `IllegalActionError` — Probe 5's
   silent-pass bug is now loud.
2. `oracle_mode=True` still completes the turn — the escape hatch is intact
   for legitimate oracle replay.

Script deleted post-run as instructed.

## Pytest delta

Log: `logs/phase3_step_gate_pytest.log`.
Command: `python -m pytest tests/ --tb=short -q` (dropped `-x` to enumerate
all failures for classification — captured in the same log).

| Run                | Passed | Failed | xpassed | xfail markers I added |
|--------------------|-------:|-------:|--------:|----------------------:|
| Before STEP-GATE * | 181    | 0      | 3       | —                     |
| After gate, pre-fix| 172    | 9      | 3       | —                     |
| After fixes        | 181    | 0      | 3       | **0**                 |

\* "Before" reconstructed from clean baseline — the 3 xpassed are
pre-existing markers on `test_engine_negative_legality.py` SEAM tests
(`pending Phase 3 SEAM canon decision`); the gate did **not** create new
xpasses or new xfails. Total runtime steady at ~36 s.

### Failure classification (the 9 surfaced by the gate)

| # | Test                                                                                       | Class      | Resolution                                                                                                              |
|---|--------------------------------------------------------------------------------------------|------------|-------------------------------------------------------------------------------------------------------------------------|
| 1 | `tests/test_capture_terrain.py::test_full_capture_updates_terrain_on_misery_neutral_city`  | TEST_BUG   | Test parachuted CAPTURE onto SELECT-stage state to exercise the handler. Added `oracle_mode=True` (handler isolation).  |
| 2 | `tests/test_capture_terrain.py::test_full_capture_neutral_comm_tower_swaps_tid`            | TEST_BUG   | Same shape — `oracle_mode=True`.                                                                                        |
| 3 | `tests/test_capture_terrain.py::test_full_capture_neutral_lab_swaps_tid`                   | TEST_BUG   | Same shape — `oracle_mode=True`.                                                                                        |
| 4 | `tests/test_engine_awbw_subset.py::test_relax_wait_on_capturable_property_does_not_raise`  | TEST_BUG   | Test explicitly verifies "step accepts even though mask hides" — the documented capture-vs-WAIT carve-out (plan line 140). Added `oracle_mode=True` with comment citing STEP-GATE. |
| 5 | `tests/test_engine_awbw_subset.py::test_select_unit_id_pins_engine_unit_when_tile_stacked` | TEST_BUG   | `select_unit_id` is the oracle drawable-stack disambiguator (`Action` docstring); never set by RL legal actions, so the mask never carries it. Test now opts into `oracle_mode=True`. |
| 6 | ~~`tests/test_engine_negative_legality.py::test_mech_can_attack_diagonal_chebyshev_1`~~ | ~~TEST_BUG~~ | ~~Positive guard parachuted ATTACK onto SELECT stage. Walked the proper SELECT_UNIT → MOVE pipeline via new `_walk_select_to_action` helper so the mask is exercised end-to-end.~~ |

**AMENDED IN PHASE 6 (2026-04-20):** This test was DELETED entirely in Phase 6.
It codified an engine bug (Chebyshev-1 attack range for direct r1 units) that
was wrong per AWBW canon. The "fix" applied here in Phase 3 made the bad test
pass via the proper pipeline; the test itself was the problem.

| # | Test                                                                                       | Class      | Resolution                                                                                                              |
|---|--------------------------------------------------------------------------------------------|------------|-------------------------------------------------------------------------------------------------------------------------|
| 7 | `tests/test_engine_negative_legality.py::test_piperunner_can_fire_on_pipe_seam_within_range` | TEST_BUG | Same — `_walk_select_to_action`.                                                                                        |
| 8 | `tests/test_engine_negative_legality.py::test_direct_adjacent_attack_on_unit_standing_on_seam_tile` | TEST_BUG | Same — `_walk_select_to_action`.                                                                                  |
| 9 | `tests/test_unit_join.py::test_illegal_wait_on_join_tile`                                  | TEST_BUG   | Asserted `pytest.raises(ValueError, match="JOIN")`; STEP-GATE rejects WAIT before `_apply_wait` reaches its JOIN-specific message. Relaxed regex to `"JOIN|get_legal_actions"`. |

**No `ENGINE_GAP` failures.** No `xfail` markers were needed in this thread —
every failure resolved as a test-side fix. The plan anticipated a "flood" of
xfails handed off to SEAM / POWER+TURN+CAPTURE / ATTACK-INV; in practice the
existing test suite was already remarkably mask-clean. The narrow positive
guards in `test_engine_negative_legality.py` (#6–#8) had been written before
the pipeline-walk helper existed; they are now stronger because they
exercise the mask too.

**No `ORACLE_GAP` failures.** The single chokepoint edit at `_engine_step`
covered every oracle call site; no per-site sweep was required.

## Oracle call sites discovered

Single chokepoint, already documented in the comment block I added at
`tools/oracle_zip_replay.py:84`. `grep -n "_engine_step\|state\.step("
tools/oracle_zip_replay.py` returns ~30 call sites — all funnel through
`_engine_step`. The integration lane should verify no future oracle helper
bypasses `_engine_step` to reach `state.step` directly; if one is added,
it must pass `oracle_mode=True` explicitly or it will trip the gate.

## What this unlocks for downstream Phase 3 threads

* **SEAM**: any addition to the seam exclusion list inside
  `get_attack_targets` immediately causes `step(ATTACK→seam)` to raise via
  STEP-GATE — no per-method assert needed in `_apply_seam_attack`. The thin
  defense-in-depth assert is still encouraged.
* **POWER+TURN+CAPTURE**: filtering `ACTIVATE_COP` out of
  `_get_select_actions` when `co.can_activate_cop()` is False is now the
  whole fix — STEP-GATE rejects the call. Probe 6 (COP at `power_bar=0`)
  closes the moment the mask is corrected. Same for Tank CAPTURE (Probe 7).
* **ATTACK-INV**: friendly-fire / already-attacked / out-of-range attacks
  raise the moment they leave the mask. Probe 4 (friendly-fire) closes
  with a single line in `get_attack_targets`.
* **PROPERTY-EQUIV (Phase 4)**: the equivalence test
  (`tests/test_engine_legal_actions_equivalence.py`) now has both ends of
  the contract enforceable — `mask_overpermits` is still a real failure
  mode (mask says yes, handler raises), but `false_positive_in_step`
  (action accepted by `step` but missing from mask) cannot occur outside
  `oracle_mode`.

## Blockers / open questions

None. Gate is in, all tests green, oracle path verified. Ready for the
remaining Phase 3 threads to land their `get_legal_actions` /
`get_attack_targets` work behind the gate.

## Files touched

```
engine/game.py                           (signature, gate, exception class)
tools/oracle_zip_replay.py               (_engine_step opt-out)
tests/test_capture_terrain.py            (3 tests + module docstring)
tests/test_engine_awbw_subset.py         (2 tests, comments)
tests/test_engine_negative_legality.py   (3 positive guards + helper)
tests/test_unit_join.py                  (1 regex relaxation)
docs/oracle_exception_audit/phase3_step_gate.md   (this file)
logs/phase3_step_gate_pytest.log         (pytest run log)
```
