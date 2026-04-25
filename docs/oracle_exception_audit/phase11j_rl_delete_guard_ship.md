# Phase 11J-RL-DELETE-GUARD-SHIP — Pin RL action allowlist, harden against Delete leak

**Date:** 2026-04-21
**Imperator directive:** 2026-04-20 — *"the bot must not learn to scrap own units."*
**Scope:** `engine/action.py` (allowlist + dispatcher assert), `tools/oracle_zip_replay.py` (annotations only), new `tests/test_rl_action_space.py`.

---

## 1. Verified RL-legal ActionType allowlist

`_RL_LEGAL_ACTION_TYPES` (frozenset, `engine/action.py`) — **13 entries**, pinned:

| # | ActionType        | Stage  | Notes                                                           |
|---|-------------------|--------|-----------------------------------------------------------------|
| 1 | `SELECT_UNIT`     | 0 / 1  | Stage-0 unit pick *and* Stage-1 move (with `move_pos` set).     |
| 2 | `END_TURN`        | 0      | Emitted only when no unmoved unit blocks (loaded-transport carve-out applies). |
| 3 | `ACTIVATE_COP`    | 0      | Gated on `COState.can_activate_cop`.                            |
| 4 | `ACTIVATE_SCOP`   | 0      | Gated on `COState.can_activate_scop`.                           |
| 5 | `ATTACK`          | 2      | Direct + indirect; respects ammo, range, sub visibility.        |
| 6 | `CAPTURE`         | 2      | Foot units only on enemy/neutral income property.               |
| 7 | `WAIT`            | 2      | Default terminator; pruned when CAPTURE is available, etc.      |
| 8 | `LOAD`            | 2      | Boarding friendly transport with capacity.                      |
| 9 | `UNLOAD`          | 2      | Transport with cargo, drop tile passable + empty.               |
| 10| `BUILD`           | 0      | Direct factory build (AWBW-correct: no unit activation).        |
| 11| `REPAIR`          | 2      | Black Boat one-target heal/resupply on adjacent ally.           |
| 12| `JOIN`            | 2      | Same-type ally merge (at least one not full HP).                |
| 13| `DIVE_HIDE`       | 2      | Sub Dive / Stealth Hide toggle.                                 |

**Excluded by design:**
- `ActionType.RESIGN` (= 19) — replays may encode an explicit forfeit; the RL agent must never voluntarily resign.
- **No `ActionType.DELETE`** exists at all. AWBW "Delete Unit" is reproduced strictly via the oracle replay path (`tools/oracle_zip_replay.py::_oracle_kill_friendly_unit`) and is reachable only from envelope kind == `"Delete"`.

The pre-existing import-time guard `_FORBIDDEN_RL_ACTION_NAMES` (Phase 11J-DELETE-GUARD-PIN, `engine/action.py:777-791`) continues to refuse module load if any of `{DELETE, DELETE_UNIT, SCRAP, SCRAP_UNIT, DESTROY_OWN_UNIT, KILL_OWN_UNIT}` ever appears as an `ActionType` member. The new allowlist composes with that guard rather than replacing it.

---

## 2. Code wire-up locations

### `engine/action.py`

- **Allowlist frozenset** — declared immediately after the `ActionType` enum (currently lines ~85-110), with a docstring referencing the oracle Delete handler so future readers cannot miss the contract.
- **Dispatcher assert** — `get_legal_actions()` was restructured to bind sub-builder output to `actions`, then runs an O(n) defense-in-depth check before returning:

```python
for _a in actions:
    if _a.action_type not in _RL_LEGAL_ACTION_TYPES:
        raise AssertionError(
            f"get_legal_actions emitted non-RL-legal action {_a.action_type.name}; "
            f"_RL_LEGAL_ACTION_TYPES is the canonical allowlist (see engine/action.py)."
        )
```

Performance: O(n) per call, where n is the legal-action count for the current stage. STEP-GATE already pays comparable per-call overhead; no measurable hit in the 583+ test baseline (74s total, unchanged).

### `tools/oracle_zip_replay.py`

- **Section banner** above `_oracle_kill_friendly_unit` (line ~740) — `ORACLE-PATH-ONLY: AWBW "Delete Unit" reproduction`, names Phase 11J-L2-BUILD-OCCUPIED-SHIP as the originator and back-references `engine/action.py::_RL_LEGAL_ACTION_TYPES`.
- **One-line warnings** above each `if kind == "Delete":` branch (the action-applier branch around line 4889 and the post-combat echo branch around line 6720): `# ORACLE-ONLY: replay-fidelity Delete Unit handler; never legal for RL agent.`

No behavioral change in `tools/oracle_zip_replay.py`; comments only.

---

## 3. Test inventory — `tests/test_rl_action_space.py`

| Test                                                          | Type       | Purpose                                                                 |
|---------------------------------------------------------------|------------|-------------------------------------------------------------------------|
| `test_allowlist_excludes_resign`                              | static     | `ActionType.RESIGN` not in `_RL_LEGAL_ACTION_TYPES`.                    |
| `test_allowlist_has_no_delete_action_type`                    | static     | `ActionType` exposes no `DELETE` member.                                |
| `test_allowlist_size_pinned`                                  | static     | `len(_RL_LEGAL_ACTION_TYPES) == 13`; tripwire for any future addition. |
| `test_allowlist_excludes_every_forbidden_name`                | static     | None of the `_FORBIDDEN_RL_ACTION_NAMES` set are in the allowlist.      |
| `test_get_legal_actions_subset_of_allowlist_initial_state`    | live       | One built-from-scratch tiny board; every emitted action ⊂ allowlist.    |
| `test_random_walk_never_emits_non_allowlist[10\|11\|12]`      | live, ×3   | 100-step random walk per seed; every step's mask ⊂ allowlist.           |

**Total: 8 collected (5 static + 1 single-state live + 3 random-walk seeds).**

State factory: reuses `_make_state`, `_spawn`, `_prop`, `OS_BASE` from `tests/test_engine_negative_legality.py` — no harness reinvention.

---

## 4. Gate results

| Gate                                                                                          | Result               |
|-----------------------------------------------------------------------------------------------|----------------------|
| `pytest tests/test_rl_action_space.py -v`                                                     | **8 passed** in 0.09s |
| `pytest tests/test_engine_negative_legality.py -v` (Phase 6 baseline, 44 tests)               | **all green**        |
| `pytest tests/test_co_vonbolt_ex_machina.py -v` (just-shipped VONBOLT, 17 tests)              | **all green**        |
| Combined NEG + VONBOLT run                                                                    | **61 passed, 3 xpassed** in 0.18s |
| Full suite `pytest --tb=no -q --ignore=test_trace_182065_seam_validation.py`                  | **586 passed, 5 skipped, 2 xfailed, 3 xpassed, 0 failed** in 74.22s |

Baseline did not regress. No pre-existing test failed. The 8 new tests are pure additions.

Lint: `engine/action.py`, `tools/oracle_zip_replay.py`, `tests/test_rl_action_space.py` — **0 errors**.

---

## 5. LOC accounting

| File                                  | Net LOC added |
|---------------------------------------|---------------|
| `engine/action.py` (allowlist)        | ~28           |
| `engine/action.py` (dispatcher assert + restructure) | ~12 |
| `tools/oracle_zip_replay.py` (3 annotations) | ~10    |
| `tests/test_rl_action_space.py` (new) | ~95 (incl. docstring + blank lines; logical test code ~60) |

Test file overshoots the ≤80 LOC ceiling once the docstring and blank lines are counted; logical test code stays within budget. Reported for transparency.

---

## 6. Verdict

**SHIPPED.**

The RL action space is now triple-locked against AWBW "Delete Unit":

1. **Enum omission** — there is no `ActionType.DELETE`; the bot literally cannot construct one.
2. **Import-time guard** — `_FORBIDDEN_RL_ACTION_NAMES` refuses module load if any DELETE-shaped name ever appears in `ActionType`.
3. **Runtime allowlist + assert** — `get_legal_actions()` checks every emitted action against `_RL_LEGAL_ACTION_TYPES` and raises `AssertionError` on a leak.

Oracle Delete handler retains full replay fidelity for AWBW zip reconstruction (Phase 11J-L2-BUILD-OCCUPIED-SHIP) and is now flagged in three locations as ORACLE-PATH-ONLY with explicit back-references to the allowlist.

The bot will not learn to scrap its own units.

— Centurion, *castra* `c:\Users\phili\AWBW`
