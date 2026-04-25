# Phase 3 — Thread POWER + TURN + CAPTURE

**Campaign:** `desync_purge_engine_harden`  
**Scope:** COP/SCOP/CAPTURE/END_TURN legality alignment with `get_legal_actions`, plus defense-in-depth in `GameState._apply_capture`.

## Files touched

### `engine/action.py`

| Area | Lines (approx.) | Change |
|------|-----------------|--------|
| `_get_select_actions` — COP/SCOP | 416–422 | **No logic change.** Confirmed emission is already guarded by `co.can_activate_cop()` / `can_activate_scop()`. Added short comment tying emission to `COState` helpers. |
| `_get_select_actions` — END_TURN | 433–441 | **No logic change.** Documented that unmoved-unit blocking (with loaded-transport carve-out) is correct here; crafted `END_TURN` while unmoved units remain is a **STEP-GATE** concern (`step(..., oracle_mode=False)` vs `get_legal_actions`). |
| `_get_action_actions` — CAPTURE | 548–556 | **No logic change.** Confirmed `stats.can_capture` and `prop.owner != player` already gate CAPTURE. Added comment. |

### `engine/game.py`

| Method | Lines (approx.) | Change |
|--------|-----------------|--------|
| `_apply_capture` | 753–777 | **Defense-in-depth:** `move_pos` required; unit must exist; `UNIT_STATS[...].can_capture`; `PropertyState` must exist at `move_pos`; property must not already belong to the acting unit’s player. Raises `ValueError` with explicit messages. Replaces prior silent no-ops (`return 0.0`) for missing unit / missing property. |
| `_activate_power` | — | **No change.** The plan asked for `can_activate_*` asserts here; they were **not** added. Reason: (1) `step()` already enforces the legal mask when `oracle_mode=False` (`IllegalActionError`), which covers Probe 6 for normal callers; (2) `oracle_zip_replay` uses `step(..., oracle_mode=True)` and may activate COP with a power bar that is **below** the engine threshold but valid on AWBW — a strict assert would break replay completion; (3) unit tests call `_activate_power()` directly with an empty meter to test power **effects** (e.g. Jess refuel). Adding asserts without threading `oracle_mode` (or a dedicated test-only bypass) through `step()` would contradict those call sites. |

## Smoke test (`tools/_phase3_power_turn_capture_smoke.py`)

- **Status:** **PASS** (run once, then file **deleted** per campaign instructions).
- **Checks:** (1) `ACTIVATE_COP` with `power_bar=0` raises (via `IllegalActionError` from STEP-GATE, not `ValueError` from `_activate_power`); (2) Tank `CAPTURE` on neutral city raises before mutating `capture_points`; (3) Andy COP at full meter succeeds; (4) Infantry `CAPTURE` on neutral city reduces capture progress.

## Pytest

**Command:** `python -m pytest tests/ -x --tb=short` (output logged to `logs/phase3_power_turn_capture_pytest.log`).

**First failure (stop-after-first):** `tests/test_capture_terrain.py::test_full_capture_updates_terrain_on_misery_neutral_city` — `IllegalActionError`: crafted `CAPTURE` from `action_stage=SELECT` is not in `get_legal_actions()`. This is **STEP-GATE** behavior (global `step` gate), not introduced by `_apply_capture` edits.

**Broader run (without `-x`):** Many tests still expect pre-gate `step()` semantics (direct `CAPTURE`/`WAIT`/`ATTACK` without going through the mask). Those fail with `IllegalActionError` until tests pass `oracle_mode=True` or drive the three-stage UI. **PROPERTY-EQUIV** (`test_engine_legal_actions_equivalence.py`) reports thousands of `false_positive_in_step` defects — same class: mask vs `step` parity is the STEP-GATE / Phase 4 harness scope.

**Delta attributable to this thread:** `_apply_capture` asserts do **not** appear in the failure list for the default test order; Jess / GL oracle replays **pass** once `_activate_power` was left without meter asserts.

## END_TURN (Probe 5)

**Owned by STEP-GATE**, not this thread. `get_legal_actions` already omits `END_TURN` when any non-carved-out unmoved unit exists (`engine/action.py` ~424–442). No engine assert was added in `_end_turn` (per instructions).

## References

- Plan: `.cursor/plans/desync_purge_engine_harden_d85bd82c.plan.md` — § Thread POWER + TURN + CAPTURE  
- Recon: `docs/oracle_exception_audit/phase2p5_legality_recon.md` — Probes 5–7  
