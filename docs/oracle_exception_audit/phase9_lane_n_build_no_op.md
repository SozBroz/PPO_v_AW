# Phase 9 — Lane N: Family B Build no-op investigation

**Campaign:** `desync_purge_engine_harden`  
**Scope:** Eight Global League replays where the oracle raises `UnsupportedOracleAction` with `Build no-op … engine refused BUILD` after `_oracle_diagnose_build_refusal` (`tools/oracle_zip_replay.py` ~701–726, Build branch ~4596–4625).

**Regression constraint:** `logs/desync_regression_log.md` — `## 2026-04-20 — Phase 7` — ORCHESTRATOR FOOTNOTE: do **not** loosen post-Phase-6 Manhattan / direct-fire tightening; fixes must align the engine with AWBW canon, not relax the oracle.

---

## Executive summary

| Aggregate | Count |
|-----------|------:|
| **MASKED ENGINE BUG** (production / `can_build` / cost rule wrong vs AWBW) | **0** |
| **DOWNSTREAM DESYNC (occupancy)** | **6** |
| **DOWNSTREAM DESYNC (funds)** | **2** |
| **REQUIRES_HUMAN_REVIEW** | **0** |

All eight failures are explained by `_oracle_diagnose_build_refusal` as either **tile occupied** or **insufficient funds**. None show **property owner None**, **wrong owner**, **non-production terrain**, **unit not producible**, or **active player mismatch**. AWBW recorded a legal build at an empty factory with sufficient funds; the engine’s preconditions fail because **simulated state already diverged** (unit positions and/or wallet) before the Build envelope — the same failure mode family as Lane J Family A (upstream drift), not a missing AWBW production rule in `engine/game.py::_apply_build`.

**Engine changes:** none.  
**New tests:** none (`tests/test_engine_build.py` does not exist; not required without an engine fix).

---

## Per-`games_id` classification

Audits: `tools.desync_audit._audit_one`, catalog `data/amarriner_gl_std_catalog.json`, `seed=1`, zips `replays/amarriner_gl/{gid}.zip`.  
Build envelopes confirmed via `parse_p_envelopes_from_zip` (gz `a<gid>` member); AWBW lists the new unit at the factory coordinates consistent with the oracle’s `(row,col)` = `(units_y, units_x)`.

| games_id | Refusal reason (`_oracle_diagnose_build_refusal`) | Register class | Lane N classification | Fix / upstream |
|----------|---------------------------------------------------|----------------|----------------------|----------------|
| 1625178 | tile occupied | oracle_gap | DOWNSTREAM DESYNC (occupancy) | Trace first position drift (Move/Fire/Capt); factory tile still blocked in engine vs empty on site. Env index 26, day 14. |
| 1627563 | insufficient funds (need 1000, have 370) | oracle_gap | DOWNSTREAM DESYNC (funds) | Income / spend / cost stream diverged before build; engine wallet too low. Env 23, day 12. |
| 1628287 | tile occupied | oracle_gap | DOWNSTREAM DESYNC (occupancy) | Same as 1625178 pattern. Env 12, day 7. |
| 1630064 | tile occupied | oracle_gap | DOWNSTREAM DESYNC (occupancy) | Env 15, day 8. |
| 1632006 | tile occupied | oracle_gap | DOWNSTREAM DESYNC (occupancy) | Env 41, day 21. |
| 1632778 | tile occupied | oracle_gap | DOWNSTREAM DESYNC (occupancy) | Env 7, day 4. |
| 1634587 | tile occupied | oracle_gap | DOWNSTREAM DESYNC (occupancy) | Env 16, day 9. |
| 1634961 | insufficient funds (need 3000, have 2580) | oracle_gap | DOWNSTREAM DESYNC (funds) | Env 26, day 14. |

**Note on occupancy:** `GameState._apply_build` intentionally no-ops when `get_unit_at(factory) is not None` (`engine/game.py`). The oracle may attempt `_oracle_nudge_eng_occupier_off_production_build_tile` first; if the blocker is already moved, trapped, or the occupying unit is not the expected friendly unmoved case, the build still correctly refuses while AWBW’s stream assumes a clear tile — i.e. **drift**, not a wrong “empty factory” rule.

**Note on funds:** `_build_cost` and CO discounts are exercised by existing tests (e.g. `tests/test_engine_negative_legality.py::test_co_discount_build_applies`). The gaps here (630 and 420) are consistent with **cumulative economy drift**, not a single missing discount edge case in isolation.

---

## AWBW canon (reference only)

Production eligibility (which units on base / airport / port) and empty-factory requirement match Advance Wars / AWBW wiki treatment of factories; this investigation did **not** find a case where the engine allowed/denied the wrong **unit–terrain** pairing. The refusal reasons were strictly **occupancy** and **funds**, which are preconditions AWBW also enforces — the divergence is **state sync**, not wiki rule interpretation.

---

## Pytest

Command run (requested `tests/test_engine_build.py` is **not present** in the repo):

```text
python -m pytest tests/test_engine_negative_legality.py -v --tb=short
```

**Result:** 44 passed, 3 xpassed (47 collected), ~0.06 s.  
Log: `logs/phase9_lane_n_pytest.log`

---

## Artifacts

| File | Purpose |
|------|---------|
| `logs/phase9_lane_n_8_audits.jsonl` | Full audit rows for the eight gids |
| `logs/phase9_lane_n_8_audits.log` | Short text summary |
| `logs/phase9_lane_n_8_post_audit.log` | Re-audit after investigation (unchanged classes) |
| `logs/phase9_lane_n_pytest.log` | Pytest transcript |
| This file | Lane N report |

---

## Escalations

- **REQUIRES_HUMAN_REVIEW:** none for this family after per-gid drill; reasons are unambiguous downstream symptoms.
- **Expected remediation path:** resolve upstream drift (Phase 9 Lane L move/position work, economy/capture consistency) so factory tiles and funds match the zip before Build; these eight rows should then migrate off `oracle_gap` without touching the Build diagnostic in `oracle_zip_replay.py`.
