# Phase 11A — Kindle + Hachi engine income / cost canon fixes

ENGINE WRITE lane. Implements the two HIGH-priority CO canon gaps flagged
by Phase 10T (Kindle income, Hachi build cost) under aggressive regression
gating. **One fix shipped, one rolled back on PHP-oracle evidence.**

## Executive return (briefing)

| Fix | CO | Rule (canon) | Status | Files | Lines |
|-----|----|--------------|--------|-------|-------|
| 1 | Kindle (23) | +50% funds from owned cities (`data/co_data.json` + wiki) | **ROLLED BACK** | `engine/game.py::_grant_income` | docstring update only (logic reverted) |
| 2 | Hachi (17)  | 90% unit cost on **all** builds (AWBW CO Chart "Units cost 10% less") | **SHIPPED** | `engine/action.py::_build_cost` | function body + docstring (~12 lines) |

**Verdict: YELLOW — partial.** Hachi 90% lands cleanly with full regression gates green and zero drift movement on the 10F sample. Kindle was reverted because the live PHP oracle on game `1628546` (Kindle vs Max) **does not** apply the +50%/city bonus that `co_data.json` + the community wiki claim — the AWBW CO Chart (the cited primary source for this campaign) is silent on Kindle income, and the chart wins.

---

## Section 1 — Files changed

### `engine/game.py` (`_grant_income`, lines ~440–480)

- Branch logic for Kindle (`co.co_id == 23`) **NOT** added.
- Docstring expanded to record the rollback rationale, primary-source
  conflict (chart vs `co_data.json`), and the `_phase10n_drilldown.py`
  evidence on game `1628546`. Future agents reading the docstring will
  understand exactly why the obvious bonus is intentionally absent.

```440:482:c:\Users\phili\AWBW\engine\game.py
    def _grant_income(self, player: int) -> None:
        """
        Apply per-turn income to ``player``'s treasury using AWBW rules:
        1000g per owned income-property (HQ/base/city/airport/port), excluding
        comm towers and labs.
        ...
        Kindle (co_id 23) is **deliberately not branched here.** Phase 11A
        attempted a +50% city-income bonus per ``data/co_data.json`` but
        ``tools/_phase10n_drilldown.py`` on game ``1628546`` (Kindle vs Max,
        map 159501) showed PHP rejecting the bonus on the very first Kindle
        city capture (turn 4 grant: PHP +4000 / engine +4500), pulling the
        first funds mismatch from envelope 11 (pre-fix) up to envelope 5
        (post-fix, +500 to engine). ...
        """
```

### `engine/action.py` (`_build_cost`, lines ~697–718)

- Old "50% on `terrain.is_base` only" Hachi heuristic **replaced** with
  `int(cost * 0.9)` applied unconditionally for `co.co_id == 17`.
- `get_terrain` lookup removed from this function (no longer needed; still
  imported for other callers).
- Docstring cites the AWBW CO Chart and 10T section 3.

```697:718:c:\Users\phili\AWBW\engine\action.py
def _build_cost(ut: UnitType, state: GameState, player: int, pos: tuple[int, int]) -> int:
    """Adjusted build cost after CO modifiers.

    Source: AWBW CO Chart https://awbw.amarriner.com/co.php
      * Kanbei  (3)  — units cost +20% more  → ×1.20
      * Colin   (15) — units cost  20% less  → ×0.80
      * Hachi   (17) — units cost  10% less  → ×0.90 (D2D, **all build sites**,
        not just bases). ...
    """
    cost = UNIT_STATS[ut].cost
    co   = state.co_states[player]
    if co.co_id == 3:
        cost = int(cost * 1.2)
    elif co.co_id == 15:
        cost = int(cost * 0.8)
    elif co.co_id == 17:
        cost = int(cost * 0.9)
    return cost
```

### New tests

- `tests/test_co_build_cost_hachi.py` — 5 tests: Hachi tank on base, Hachi
  B-Copter on airport, Hachi Lander on port, Andy tank baseline, Colin
  80% sanity. **All pass.**
- `tests/test_co_income_kindle.py` — 3 tests **pinning the rollback**:
  Kindle on 5 cities + 2 bases grants flat 7000 (not 9500), Andy baseline,
  Kindle/Andy parity until canon resolved. **All pass.**

---

## Section 2 — Per-fix regression gate results

### Fix 2 (Hachi 90% — SHIPPED)

| # | Gate | Floor | Result | Verdict |
|---|------|-------|--------|---------|
| 1 | `tests/test_engine_negative_legality.py -v` | 44 passed / 3 xpassed / 0 failed | **44 passed / 3 xpassed / 0 failed** | GREEN |
| 2 | `tests/test_andy_scop_movement_bonus.py` | 2 passed | **2 passed** | GREEN |
| 3 | `test_engine_legal_actions_equivalence::test_legal_actions_step_equivalence` | 1 passed | **1 passed** | GREEN |
| 4 | Full `pytest --tb=no -q` | ≤ 2 failures (deferred trace_182065) | **1 failed, 472 passed, 5 skipped, 2 xfailed, 3 xpassed** — only `test_trace_182065_seam_validation` (the deferred 10R pair) | GREEN |
| 5 | `tools/desync_audit.py --max-games 50 --seed 1` | no NEW engine_bug | **1 engine_bug (1605367 — pre-existing Mech illegal move, unrelated), 4 oracle_gap, 45 ok** — identical to baseline | GREEN |
| 6 | Hachi smoke test | Tank cost = 6300 | **6300 confirmed (also B-Copter on airport, Lander on port)** | GREEN |

### Fix 1 (Kindle +50% — ROLLED BACK)

| # | Gate | Floor | Pre-rollback result | Verdict |
|---|------|-------|--------|---------|
| 1 | Negative legality | 44/3/0 | 44/3/0 | GREEN |
| 2 | Andy SCOP | 2 | 2 | GREEN |
| 3 | Legal actions equivalence | 1 | 1 | GREEN |
| 4 | Full pytest | ≤ 2 fail | 1 fail (trace_182065 only) | GREEN |
| 5 | Spot-audit 50 games | no new engine_bug | unchanged baseline | GREEN |
| 6 | **Kindle drill on 1628546 — drift at envelope 11 REDUCED or eliminated** | reduce or eliminate | **FAILED — drift moved from envelope 11 (+200 P0) to envelope 5 (+500 P0); +50% bonus added 500g/turn that PHP does not have** | **RED** |

Per the protocol ("If ANY gate fails, ROLL BACK that fix and document why; do not proceed to next"), Fix 1 was reverted before Fix 2 began. Rollback verified by re-running the drill: drift returned to envelope 11 with the original +200 P0 funds delta and HP-bars mismatch at `(0,6,5)` — byte-identical to the Phase 10N baseline.

---

## Section 3 — Funds drift delta on game 1628546 (Kindle vs Max)

| State | First mismatch envelope | P0 funds (engine vs PHP) | Δ engine − PHP | Income event explanation |
|-------|--------------------------|--------------------------|----------------|--------------------------|
| Baseline (Phase 10N) | **11** | 9000 vs 8800 | **+200** | Engine flat 9 × 1000 = +9000; PHP 8800; +200 unexplained-by-Kindle (10N attributes to HP/repair) |
| Fix 1 candidate (+50% city) | **5** | 7500 vs 7000 | **+500** | Engine 4 × 1000 + 1 × 500 = +4500 on turn 4; PHP +4000 — PHP **does not** apply the bonus |
| After rollback | **11** | 9000 vs 8800 | **+200** | Identical to baseline — clean revert confirmed |

**Root cause of the rollback signal:** PHP behavior matches the AWBW CO Chart, which is silent on Kindle income. The `co_data.json` + community-wiki "+50% from owned cities" line is a primary-source discrepancy already flagged in `phase10t_co_income_audit.md` Section 3 (rule class "INCOME% (disputed)"). The chart and the live oracle agree: no D2D income bonus for Kindle.

---

## Section 4 — Phase 10F drift baseline comparison (5-game sample)

`tools/_phase10n_drilldown.py` against 5 games from
`phase10f_silent_drift_recon.md` table (post-Hachi-fix run; Kindle reverted):

| games_id | COs (P0 / P1) | 10F first mismatch | 11A first mismatch | Δ |
|----------|---------------|--------------------|--------------------|---|
| 1628546  | Kindle / Max  | 11 | 11 | 0 |
| 1620188  | Lash / Andy   | 13 | 13 | 0 |
| 1628609  | Andy / Lash   | 13 | 13 | 0 |
| 1632233  | Sasha / Hawke | 12 | 12 | 0 |
| 1634522  | Olaf / Kindle | 16 | 16 | 0 |

**Zero movement.** Expected: per Phase 10T section 4, Hachi (CO 17) appears in **0 / 39** silent-drift games, so a Hachi-only fix touches no funds-drift on this cohort. The rollback of Kindle restores the baseline at game 1628546 and 1634522 (the two Kindle-bearing games in the sample).

The Hachi fix's value will surface in **other cohorts** — any future Hachi replay that AWBW records (none in the 800-game catalog sampled so far per 10T) will now match PHP build costs at every site, not just bases.

---

## Section 5 — Closure verdict

**YELLOW** — Hachi 90% canon shipped clean (every regression gate green, no drift movement on the 10F sample, +5 new tests). Kindle +50% city income was correctly identified as a HIGH-priority gap by Phase 10T but the live PHP oracle rejects the bonus, so the implementation was rolled back per the regression-gate protocol; the rollback is now pinned by `tests/test_co_income_kindle.py` and the docstring in `_grant_income` records the conflict for downstream agents. The next Kindle pass should re-derive the income rule from a wider PHP sample (e.g. drill the 8 Kindle-bearing 10F games end-to-end) before any new attempt — the chart-vs-`co_data.json` discrepancy from 10T Section 3 is now empirically resolved in favor of the chart.

## Citations

| # | Source | URL / path |
|---|--------|-----------|
| 1 | AWBW CO Chart (primary) | https://awbw.amarriner.com/co.php |
| 2 | Phase 10T audit | `docs/oracle_exception_audit/phase10t_co_income_audit.md` |
| 3 | Phase 10N drilldown evidence | `docs/oracle_exception_audit/phase10n_funds_drift_recon.md` |
| 4 | Phase 10F drift register | `docs/oracle_exception_audit/phase10f_silent_drift_recon.md` |
| 5 | Engine income | `engine/game.py::_grant_income` |
| 6 | Engine build cost | `engine/action.py::_build_cost` |
| 7 | Hachi unit tests | `tests/test_co_build_cost_hachi.py` |
| 8 | Kindle rollback pin | `tests/test_co_income_kindle.py` |
| 9 | Kindle drill (post-fix candidate) | `logs/phase11a_kindle_drill_after.json` |
| 10 | Kindle drill (post-rollback) | `logs/phase11a_kindle_drill_rollback.json` |
| 11 | 5-game drift sample after | `logs/phase11a_drift_sample_after.json` |
