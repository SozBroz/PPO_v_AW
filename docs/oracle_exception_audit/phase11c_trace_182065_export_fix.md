# Phase 11C — Trace 182065 export `oracle_mode` threading

**Date:** 2026-04-21  
**Scope:** `tools/export_awbw_replay_actions.py` only (per Phase 11C constraints).  
**Engine / tests:** untouched.

## Files changed

| File | Approx. line ranges |
|------|---------------------|
| `tools/export_awbw_replay_actions.py` | Import **75**; `_emit_move_or_fire` **512–526**; `_rebuild_and_emit` **641–683**; `_rebuild_and_emit_with_snapshots` **736–772** |

## `state.step()` callsites (AWBW trace / export pipeline)

All trace-driven `state.step` calls in this file now pass `oracle_mode=True` **except** the intentional first attempt in `_emit_move_or_fire`, which mirrors `oracle_zip_replay._engine_step` (strict apply, then oracle on `IllegalActionError`).

| File | Function | Line(s) | Reasoning |
|------|----------|---------|-----------|
| `tools/export_awbw_replay_actions.py` | `_emit_move_or_fire` | 514 | First `state.step(action)` — strict legality; if `IllegalActionError`, fall through. |
| same | `_emit_move_or_fire` | 517 | `state.step(action, oracle_mode=True)` after `IllegalActionError` — STEP-GATE opt-out for AWBW envelope replay (same contract as `oracle_zip_replay._engine_step`). |
| same | `_rebuild_and_emit` | 644 | BUILD from `full_trace` — AWBW envelope, not mask-derived. |
| same | `_rebuild_and_emit` | 663 | END_TURN from trace — fixes STEP-GATE abort (e.g. turn 54 / `SELECT` mask). |
| same | `_rebuild_and_emit` | 682 | SELECT_UNIT / ACTIVATE_* from trace — AWBW envelope. |
| same | `_rebuild_and_emit_with_snapshots` | 738 | BUILD — same as `_rebuild_and_emit`. |
| same | `_rebuild_and_emit_with_snapshots` | 757 | END_TURN — replaces prior `try/except Exception: continue` that skipped turns on `IllegalActionError`. |
| same | `_rebuild_and_emit_with_snapshots` | 771 | Residual trace actions — AWBW envelope. |

**Note:** `_emit_move_or_fire` does not use `oracle_mode=True` on the first `step` call by design (match `_engine_step` ordering).

## `_emit_move_or_fire` exception handling — Option A

**Taken:** **Option A.**

- `IllegalActionError` is handled **before** `ValueError` so STEP-GATE failures are not misclassified as generic `ValueError` (critical: `IllegalActionError` subclasses `ValueError`).
- On `IllegalActionError`, retry: `state.step(action, oracle_mode=True)`.
- The existing `except ValueError` branch remains for **non–`IllegalActionError`** `ValueError` paths (e.g. re-exec divergence / forced move), preserving prior behavior for that flank.

## Regression gate results (2026-04-21, Windows, `python -m pytest`)

| Gate | Command | Floor | Result |
|------|---------|-------|--------|
| Negative legality | `tests/test_engine_negative_legality.py -v` | 44 passed, 3 xpassed, 0 failed | **GREEN** — 44 passed, 3 xpassed |
| Andy SCOP | `tests/test_andy_scop_movement_bonus.py` | 2 passed | **GREEN** — 2 passed |
| Step equivalence | `tests/test_engine_legal_actions_equivalence.py::test_legal_actions_step_equivalence` | 1 passed | **GREEN** — 1 passed (~29s) |
| Full pytest | `pytest --tb=no -q` | 0 failures attributable to export change | **YELLOW** — 1 failure (see trace tests below) |

## Trace 182065 tests

**Path:** `test_trace_182065_seam_validation.py` at **repository root** (not under `tests/`; Phase 11 charter path is slightly stale).

| Test | Result |
|------|--------|
| `test_export_turn_bucket_count_explains_viewer_turn_119` | **PASS** |
| `test_full_trace_replays_without_error` | **FAIL** — `IllegalActionError` on `BUILD` Mech at turn 20: action not in `get_legal_actions()` (Phase 10M #11: harness uses plain `state.step` without `oracle_mode`). |

**Escalation (not an export defect):** Phase 10M classifies #11 as **replay harness vs STEP-GATE**, same *policy* as export but implemented in the **test body** (`state.step(_trace_to_action(entry))`). Fixing it requires `oracle_mode=True` on that loop (or a dedicated helper), which is **out of scope** for Phase 11C’s “export file only” constraint. If replay with `oracle_mode=True` later diverges from golden state, treat as **Phase 11 parity / engine** per Phase 10M footnote.

## Export CLI sanity

`replays/amarriner_gl/*.zip` not present in this workspace; used:

`PYTHONPATH=<repo root> python tools/export_awbw_replay_actions.py --from-trace %TEMP%\phase11c_smoke.zip replays/182065.trace.json`

**Result:** completed without crash; zip written to temp.

## Verdict

**YELLOW**

- Export pipeline: **GREEN** — `oracle_mode` threaded; `_rebuild_and_emit` / `_rebuild_and_emit_with_snapshots` aligned with `oracle_zip_replay._engine_step`; Option A for `_emit_move_or_fire`.
- Trace campaign closure: **incomplete** — `test_full_trace_replays_without_error` still fails until the **test harness** opts into `oracle_mode=True` for full-trace replay (or strict legality is restored for that trace).

**Recommended follow-up (Phase 11D or test lane):** one-line change in `test_full_trace_replays_without_error`: `state.step(_trace_to_action(entry), oracle_mode=True)`, then re-run for parity vs `game_log` / golden expectations.
