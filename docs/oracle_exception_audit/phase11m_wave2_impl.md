# Phase 11M-WAVE2-IMPL — `oracle_strict` LOAD / UNLOAD (top 5 from Wave 2 recon)

**Campaign:** `desync_purge_engine_harden`  
**Lane:** ENGINE WRITE — `_apply_load`, `_apply_unload` in `engine/game.py`  
**Pattern:** mirror `docs/oracle_exception_audit/phase11b_apply_silent_return_tighten.md`

---

## Section 1 — Top 5 branches chosen (S10O-XX)

From `docs/oracle_exception_audit/phase11m_suspect_wave_2.md` Section 5 (“Priority ranking”), the recon’s **Top 5 — ship in Wave 2 implementation** are:

| Rank | ID | Function | Notes |
|-----:|-----|----------|--------|
| 1 | **S10O-14** | `_apply_unload` | Drop not orthogonally adjacent to transport **after** `_move_unit` |
| 2 | **S10O-16** | `_apply_unload` | Drop tile occupied |
| 3 | **S10O-17** | `_apply_unload` | Drop terrain impassable for cargo (`effective_move_cost >= INF_PASSABLE`) |
| 4 | **S10O-15** | `_apply_unload` | Drop position **OOB** |
| 5 | **S10O-09** | `_apply_load` | Mover or transport missing at `unit_pos` / `move_pos` |

**Skipped (forbidden / other lanes):**

- `_apply_repair` (S10O-05 … S10O-08) and `_apply_repair` tail — **not touched** (Phase 11Y-RACHEL-IMPL in flight).
- `_apply_attack` (S10O-24) — **excluded** (FIRE-DRIFT zone).
- Middle-six items (WAIT, DIVE_HIDE, UNLOAD S10O-11..13, etc.) — **deferred** to Wave 3 per recon; this implementation lane only ships the **five** above.

---

## Section 2 — Files changed

| Path | Change |
|------|--------|
| `engine/game.py` | `step()` passes `oracle_strict` into `_apply_load` and `_apply_unload`; signatures and five guarded branches (see §3). Approx. lines **336–337**, **342–343**, **1229–1240**, **1333–1405** (line numbers drift with edits). |
| `tests/test_oracle_strict_apply_invariants.py` | Helpers + 10 tests (5 strict + 5 non-strict) for S10O-09 / S10O-14..17. |

---

## Section 3 — Per-branch tightening (before / after)

**Convention:** under `oracle_strict` (default `False`), silent `return` unchanged unless `oracle_strict=True` → `IllegalActionError` with stable prefix.

### S10O-09 — `_apply_load`

- **Before:** `if unit is None or transport is None: return`
- **After:** same, plus `if oracle_strict: raise IllegalActionError("_apply_load: mover or transport missing at unit_pos or move_pos …")`

### S10O-14 — `_apply_unload`

- **Before:** `if dr + dc != 1: return` (after transport move)
- **After:** `IllegalActionError` prefix `_apply_unload: drop tile not orthogonally adjacent to transport after move …`

### S10O-15 — `_apply_unload`

- **Before:** OOB bounds check → `return`
- **After:** same + `_apply_unload: drop position out of bounds …`

### S10O-16 — `_apply_unload`

- **Before:** occupied drop → `return`
- **After:** same + `_apply_unload: drop tile occupied …`

### S10O-17 — `_apply_unload`

- **Before:** `effective_move_cost >= INF_PASSABLE` → `return`
- **After:** same + `_apply_unload: drop terrain impassable for cargo …`

---

## Section 4 — New tests (10 total)

| Branch | Strict (`oracle_strict=True`) | Non-strict (`oracle_strict=False`) |
|--------|------------------------|-------------------------------------|
| S10O-09 | `test_apply_load_missing_mover_or_transport_oracle_strict_raises` | `test_apply_load_missing_mover_or_transport_oracle_strict_false_no_raise` |
| S10O-14 | `test_apply_unload_drop_not_adjacent_after_move_oracle_strict_raises` | `test_apply_unload_drop_not_adjacent_after_move_oracle_strict_false_partial_move` |
| S10O-15 | `test_apply_unload_drop_out_of_bounds_oracle_strict_raises` | `test_apply_unload_drop_out_of_bounds_oracle_strict_false_no_raise` |
| S10O-16 | `test_apply_unload_drop_tile_occupied_oracle_strict_raises` | `test_apply_unload_drop_tile_occupied_oracle_strict_false_no_raise` |
| S10O-17 | `test_apply_unload_drop_terrain_impassable_for_cargo_oracle_strict_raises` | `test_apply_unload_drop_terrain_impassable_oracle_strict_false_no_raise` |

**Module total:** Phase 11B **7** + Wave 2 **10** = **17** tests in `tests/test_oracle_strict_apply_invariants.py` (the mission’s “15 + 10 = 25” figure does not match the pre-existing file count; actual collected count is **17**).

---

## Section 5 — Regression gates (8)

| # | Gate | Result |
|---|------|--------|
| 1 | `pytest tests/test_engine_negative_legality.py -v --tb=no` | **PASS** — 44 passed, 3 xpassed |
| 2 | `pytest tests/test_andy_scop_movement_bonus.py tests/test_co_movement_koal_cop.py --tb=no` | **PASS** — 7 passed |
| 3 | `pytest tests/test_engine_legal_actions_equivalence.py::test_legal_actions_step_equivalence --tb=no` | **PASS** — 1 passed |
| 4 | `pytest tests/test_co_build_cost_hachi.py tests/test_co_income_kindle.py --tb=no` | **PASS** — 8 passed (mission listed 6; suite expanded) |
| 5 | `pytest tests/test_oracle_strict_apply_invariants.py -v --tb=short` | **PASS** — 17 passed |
| 6 | `pytest --tb=no -q` (full suite) | **PASS** — 507 passed, 1 failed (`test_trace_182065_seam_validation` — within ≤2 deferred-trace budget) |
| 7 | `python tools/desync_audit.py --max-games 50 --seed 1 --register logs/desync_register_post_m_wave2_50.jsonl` | **PASS** — see §6 |
| 8 | `python tools/desync_audit.py --max-games 100 --seed 1 --register logs/desync_register_post_m_wave2_100.jsonl` | **PASS** — see §6 |

---

## Section 6 — 50 / 100-game sample vs Phase 11J floor

**Command:** `python tools/desync_audit.py --max-games N --seed 1 --register logs/desync_register_post_m_wave2_{N}.jsonl`

| Sample | ok | oracle_gap | engine_bug |
|--------|---:|-----------:|-----------:|
| **50** | 45 | 5 | **0** |
| **100** | 89 | 11 | **0** |

**Phase 11J reference (50-game, seed 1, fire-drift closeout doc):** `ok=45`, `oracle_gap=5`, `engine_bug=0`.

**Verdict:** `engine_bug ≤ 0` on both samples; `oracle_gap` count on the **50-game** slice **matches** the 11J reference (5), not lower — no evidence that new strict raises hide oracle_gap surfaces on this run.

---

## Section 7 — Verdict

**GREEN**

- Five branches (S10O-09, S10O-14, S10O-15, S10O-16, S10O-17) tightened under `oracle_strict=True`; default `oracle_strict=False` preserves legacy silent paths.
- Partition respected: no edits to `_apply_attack`, `_apply_build` / `_apply_join` / `_apply_repair` bodies beyond existing wiring, `engine/action.py`, `tools/oracle_zip_replay.py`, or `tools/desync_audit.py`.

---

*“The harder the conflict, the more glorious the triumph.”* — Thomas Paine, *The American Crisis* (1776–1783)  
*Paine: English-American political philosopher and revolutionary; author of *Common Sense*.*
