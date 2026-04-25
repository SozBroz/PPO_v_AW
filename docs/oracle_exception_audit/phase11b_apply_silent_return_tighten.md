# Phase 11B — `_apply_*` silent return tightening (oracle_strict)

**Campaign:** `desync_purge_engine_harden`  
**Scope:** `engine/game.py` only for edits — `_apply_build`, `_apply_join`, `_apply_repair` (partition with Phase 11A on `game.py`).  
**Consumer:** `tools/oracle_zip_replay.py::_engine_step` continues `state.step(act, oracle_mode=True)`; `oracle_strict` defaults to **False**, so zip replay behavior is unchanged except the intentional REPAIR guard fix below.

---

## Per-function summary

| Function | Branches tightened (S10O ids) | `oracle_strict=True` | `oracle_strict=False` behavior change |
|----------|------------------------------|----------------------|--------------------------------------|
| `_apply_build` | 6 (S10O-18 … S10O-23) | `IllegalActionError` with distinct messages per precondition | **No** (silent `return` preserved) |
| `_apply_join` | 1 (S10O-10) | `IllegalActionError("JOIN: no merge partner at target")` | **No** |
| `_apply_repair` | 1 (S10O-04, lines 1021–1023) | `IllegalActionError("REPAIR: not a Black Boat or unit missing")` | **Yes** — see below |

### REPAIR 1021–1023 — `_finish_action` / stage closure

**Before:** Missing unit or non–Black Boat at `unit_pos` → bare `return` with no `_finish_action`, leaving `action_stage` / selection inconsistent (stage drift under oracle).

**After (`oracle_strict=False`):**

- If a **unit** exists at `unit_pos` but is not a Black Boat → `_finish_action(that unit)` (same state transition as other repair early-outs that already finished).
- If **no unit** at `unit_pos` → `action_stage = SELECT`, `selected_unit` / `selected_move_pos` cleared (no unit to mark `moved`).

**After (`oracle_strict=True`):** raises `IllegalActionError` as specified.

---

## Wiring

- `GameState.step(..., oracle_mode=False, oracle_strict=False)` — new keyword only; existing call sites unchanged.
- `oracle_strict` is passed into `_apply_build`, `_apply_join`, `_apply_repair` only.

---

## Regression gates (mandatory)

| # | Gate | Result |
|---|------|--------|
| 1 | `pytest tests/test_engine_negative_legality.py -v` | **44 passed, 3 xpassed, 0 failed** |
| 2 | `pytest tests/test_andy_scop_movement_bonus.py` | **2 passed** |
| 3 | `pytest tests/test_engine_legal_actions_equivalence.py::test_legal_actions_step_equivalence` | **1 passed** |
| 4 | `pytest --tb=no -q` (full suite) | **472 passed, 1 failed** — `test_trace_182065_seam_validation.py::...test_full_trace_replays_without_error` (within deferred trace_182065 budget; ≤2 failures allowed) |
| 5 | `python -m tools.desync_audit --catalog data/amarriner_gl_std_catalog.json --max-games 50 --seed 1` | **45 ok / 4 oracle_gap / 1 engine_bug** (90% ok on this slice; no unexpected class epidemic) |
| 6 | `pytest tests/test_oracle_strict_apply_invariants.py` | **7 passed** |

---

## 50-game `desync_audit` sample (seed 1)

| Class | Count |
|------|------:|
| ok | 45 |
| oracle_gap | 4 |
| engine_bug | 1 |
| **Total** | **50** |

Compared to Phase 10Q full-run rate (~91.8% ok on 741 games), **45/50 = 90%** is in-family for a small deterministic slice. No investigation triggered for class flips on this sample.

---

## Verdict: **GREEN**

- Partition respected (no edits to `_grant_income` / non-owned `_apply_*` ranges).
- Default oracle consumer: only REPAIR guard path intentionally changed on `oracle_strict=False`.
- Strict lane adds fail-fast `IllegalActionError`s for BUILD/JOIN/REPAIR preconditions listed above.
- Full pytest: single failure matches known deferred trace seam area; not treated as Phase 11B regression.

---

## Phase 11 follow-up — remaining SUSPECT `_apply_*` branches

Phase 10O catalog: **24** suspect guard rows. This phase tightened **8** rows (6 BUILD + 1 JOIN + 1 REPAIR at 1021–1023).

**Remaining 16** suspect rows in `game.py` `_apply_*` (defer; prioritize by Phase 10O table / 10F funds linkage):

- `_apply_repair`: S10O-05 … S10O-08 (still silent `_finish_action` skips without full repair semantics)
- `_apply_wait` S10O-01, `_apply_dive_hide` S10O-02/03, `_apply_load` S10O-09
- `_apply_unload` S10O-11 … S10O-17 (7)
- `_apply_attack` S10O-24

*(If counting “top 3 pattern groups” only, the mission scoped to BUILD+JOIN+REPAIR-head; the numeric **16** is branch-level remainder from the JSON inventory.)*

---

## Artifacts

| Path | Role |
|------|------|
| `tests/test_oracle_strict_apply_invariants.py` | Property tests for strict raises, non-strict non-raise, REPAIR `_finish_action` / stage clear |
| This file | Phase 11B report |

---

*“In any moment of decision, the best thing you can do is the right thing, the next best thing is the wrong thing, and the worst thing you can do is nothing.”* — attributed to Theodore Roosevelt (early 20th c., leadership aphorism; wording varies)  
*Roosevelt: 26th President of the United States; reformer and naturalist.*
