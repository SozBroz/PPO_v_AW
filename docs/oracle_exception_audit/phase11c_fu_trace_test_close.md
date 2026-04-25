# Phase 11C-FU — Trace 182065 test harness `oracle_mode` follow-up

**Date:** 2026-04-21  
**Scope:** `test_trace_182065_seam_validation.py` only (test harness).

## Section 1 — File changed + line modified

| File | Line |
|------|------|
| `test_trace_182065_seam_validation.py` | **62** |

## Section 2 — Before / after

**Before:**

```python
            state.step(_trace_to_action(entry))
```

**After:**

```python
            state.step(_trace_to_action(entry), oracle_mode=True)
```

## Section 3 — Pytest results

### Targeted

**Command:** `python -m pytest test_trace_182065_seam_validation.py -v --tb=short`

| Outcome | Count |
|---------|-------|
| Passed | 2 |
| Failed | 1 |

- `TestTrace182065SeamValidation::test_attack_seam_log_matches_expected_two_hit_breaks` — **PASSED**
- `TestTrace182065SeamValidation::test_export_turn_bucket_count_explains_viewer_turn_119` — **PASSED**
- `TestTrace182065SeamValidation::test_full_trace_replays_without_error` — **FAILED**

**Failure (post-fix, not `IllegalActionError`):**

- `ValueError` raised from `engine/game.py::_move_unit` via `_apply_wait` during `GameState.step`
- Message (abbrev.): `Illegal move: Infantry (move_type=infantry) from (9, 8) to (11, 7) (terrain id=29, fuel=73) is not reachable.`
- Failing line in test: **62** (`state.step(..., oracle_mode=True)`).

### Full suite

**Command:** `python -m pytest --tb=no -q`

| Metric | Value |
|--------|-------|
| Failed | **1** |
| Passed | **480** |
| Skipped | 5 |
| xfailed / xpassed | 2 / 3 |

## Section 4 — Verdict

**YELLOW**

- The Phase 11C–documented one-line harness change was applied: STEP-GATE opt-out matches `oracle_zip_replay._engine_step` (`oracle_mode=True` on trace-driven `step`).
- The prior failure mode (`IllegalActionError` on BUILD Mech — mask vs AWBW envelope) is **no longer** the blocker; replay advances until a **different** invariant fires (`ValueError` on unreachable move during `WAIT`).
- Full pytest remains **not** green (1 failure). Escalation: engine/oracle parity for full-trace replay vs `_trace_to_action` + post-BUILD state, or harness strategy beyond a single `oracle_mode=True` (out of scope for this FU).
