# Phase 11Z — Post-merge integration gate

**Date:** 2026-04-21  
**Workspace:** `C:\Users\phili\AWBW`  
**Scope:** READ + TEST only — validates combined **11A + 11B + 11C** state (no edits to engine, tests, or tools in this phase).

---

## Section 1 — Mandatory regression gates

| # | Command | Expected floor | Result |
|---|---------|------------------|--------|
| 1 | `python -m pytest tests/test_engine_negative_legality.py -v --tb=no` | 44 passed, 3 xpassed, 0 failed | **PASS** — 44 passed, 3 xpassed, 0 failed |
| 2 | `python -m pytest tests/test_andy_scop_movement_bonus.py --tb=no` | 2 passed | **PASS** — 2 passed |
| 3 | `python -m pytest tests/test_engine_legal_actions_equivalence.py::test_legal_actions_step_equivalence --tb=no` | 1 passed | **PASS** — 1 passed (~29 s) |
| 4 | `python -m pytest tests/test_co_build_cost_hachi.py tests/test_co_income_kindle.py tests/test_oracle_strict_apply_invariants.py -v --tb=short` | 15 passed (5+3+7) | **PASS** — 15 passed |
| 5 | `python -m pytest --tb=no -q` (full suite) | ≤2 failures (deferred `trace_182065` only) | **PASS floor** — 480 passed, 1 failed, 5 skipped, 2 xfailed, 3 xpassed |
| 6 | `python tools/desync_audit.py --max-games 50 --seed 1 --register logs/desync_register_post_phase11z_smoke.jsonl` | ~45 ok / 4 oracle_gap / 1 engine_bug | **PASS** — 45 ok / 4 oracle_gap / 1 engine_bug |

**Gate 5 failure detail (expected deferred):**

- `test_trace_182065_seam_validation.py::TestTrace182065SeamValidation::test_full_trace_replays_without_error` — matches Phase 11C / 10M harness policy (`state.step` without `oracle_mode=True` on full trace); not treated as an 11A/11B/11C integration regression.

---

## Section 2 — Combined-state verification (integration-only)

### 2.1 `GameState.step` signature

Command: `python -c "import inspect; from engine.game import GameState; print(inspect.signature(GameState.step))"`

Output:

```text
(self, action: 'Action', *, oracle_mode: 'bool' = False, oracle_strict: 'bool' = False) -> 'tuple[GameState, float, bool]'
```

**Result:** Both `oracle_mode` and `oracle_strict` keywords present.

### 2.2 Hachi `_build_cost` + `oracle_strict=True` on a legal BUILD

Minimal fixture: P0 Andy (`co_id` 1), P1 Hachi (`co_id` 17), P1 active, owned neutral base at `(0,1)`, empty, sufficient funds.

- `_build_cost(TANK, state, player=1, pos=(0,1))` → **6300** (0.9 × 7000).
- `state.step(Action(BUILD, move_pos=(0,1), unit_type=TANK), oracle_strict=True)` completes without error; tank appears on P1.

**Result:** PASS.

### 2.3 REPAIR default path (`oracle_strict=False`) — missing unit at `unit_pos`

Minimal state: `ACTION` stage, `selected_unit=None`, `selected_move_pos=(0,0)`, no unit at `(0,0)`, `REPAIR` action with `oracle_mode=True`, `oracle_strict=False` (default).

**Result:** No exception; `action_stage == SELECT`, `selected_unit` and `selected_move_pos` cleared. PASS (matches `test_repair_missing_unit_oracle_strict_false_clears_stage`).

---

## Section 3 — 50-game audit comparison vs Phase 10Q baseline

**Baseline:** `logs/desync_register_post_phase10q.jsonl`  
**Current:** `logs/desync_register_post_phase11z_smoke.jsonl` (seed **1**, first **50** games, same catalog ordering as audit tool output)

| Check | Outcome |
|--------|---------|
| Same 50 `games_id` order | Yes (row-for-row alignment on `games_id`) |
| `ok` → `engine_bug` (regression) | **None** |
| `engine_bug` → `ok` (improvement) | **None** |
| Per-game `class` field mismatches | **0** |

**Counts (11Z smoke register):** 45 `ok`, 4 `oracle_gap`, 1 `engine_bug` — identical to baseline for those games.

**Exception shape:** First-row `engine_bug` (game `1605367`) matches baseline: `exception_type` `ValueError`, same `message` prefix (`Illegal move: Mech ... is not reachable.`). No new exception-type epidemic observed on this slice.

---

## Section 4 — Verdict

**YELLOW**

- Integrated **11A + 11B + 11C** state: **no new regressions** vs Phase 10Q on the deterministic 50-game slice; all lane gates and cross-checks pass.
- Full pytest remains **1 failure** in the **known deferred** trace-182065 seam area (within the ≤2 deferred budget documented in the campaign).

---

## Section 5 — Deferred items (YELLOW)

| Item | Owner / notes |
|------|----------------|
| `test_trace_182065_seam_validation.py::test_full_trace_replays_without_error` | **Test / replay harness lane** — full-trace loop uses plain `state.step` without `oracle_mode=True`; Phase 11C documents fix as one-line harness change or policy alignment with `oracle_zip_replay._engine_step` (out of scope for export-only 11C). |

**Escalation if this were RED:** N/A — no combined-state regression identified; single failure is the pre-documented trace seam, not a new break in Hachi/Kindle/`oracle_strict`/export integration.

---

## Code verification checklist (skim / spot-check)

| Check | Status |
|-------|--------|
| `engine/game.py` `step(..., oracle_mode=..., oracle_strict=...)` | Present (see Section 2.1) |
| `_grant_income` docstring — Kindle rollback rationale | Present (`co_id` 23 not branched; PHP/`1628546` evidence cited) |
| `engine/action.py::_build_cost` — Hachi `co_id` 17 → `int(cost * 0.9)` | Present |
| `tests/test_oracle_strict_apply_invariants.py` | Exists — 7 tests, all passed in gate 4 |
| `tests/test_co_build_cost_hachi.py`, `tests/test_co_income_kindle.py` | Exist — 5 + 3 tests, all passed in gate 4 |
