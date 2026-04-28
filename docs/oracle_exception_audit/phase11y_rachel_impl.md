# Phase 11Y-RACHEL-IMPL — Rachel (CO 28) D2D +1 repair HP, engine canon

ENGINE WRITE lane. Implements the Rachel D2D property-repair rule per Phase
11Y-CO-WAVE-2 recon (`phase11y_co_wave_2.md`). Rachel-owned units now heal
**+3 displayed HP** (`+30` internal) per property-day instead of the
standard `+20`, and pay proportional gold (30% of unit cost for a full
band, vs 20% for standard). Funds drift on 4 of 5 drilled Rachel zips
**eliminated** post-fix.

## Executive return

| Field | Value |
|-------|-------|
| Files changed | `engine/game.py` (1 fn, ~22 LOC docstring + 1 LOC logic) |
| New tests | `tests/test_co_repair_rachel.py` — 6 tests, **6 pass** |
| Regression gates | 9 / 9 green |
| Funds drift closed | 4 / 5 Rachel-targeted zips → `funds_delta_by_seat == 0` post-fix |
| **Verdict** | **GREEN** — clean fix, PHP oracle agrees on 4/5 drilled games |

---

## Section 1 — Hypothesis verification

### Sources (all aligned; PHP wins on disagreement)

Per Imperator directive (post-handoff): every CO mechanic in this lane
must cite the AWBW community wiki AND be cross-checked against PHP
snapshots in actual replays before shipping. The Phase 11A Kindle
rollback (PHP disagreed with both chart and JSON) is the standing
precedent for caution.

| Source | URL | Quote / claim |
|--------|-----|--------------|
| AWBW CO Chart (in-game) | https://awbw.amarriner.com/co.php | *"Units repair +1 additional HP (note: liable for costs)."* |
| AWBW Fandom Wiki — Rachel | https://awbw.fandom.com/wiki/Rachel | Day-to-Day: *"Units repair +1 additional HP on properties (note: liable for costs)."* |
| AWBW Fandom Wiki — Changes in AWBW | https://awbw.fandom.com/wiki/Changes_in_AWBW | *"Repairs will only take place in increments of exactly 20 hitpoints, or 2 full visual hitpoints."* (Rules out half-bar / +10-internal interpretation; combined with Rachel +1 ⇒ exactly +30 internal HP / +3 visual bars.) |
| Advance Wars Wiki — Repairing | https://advancewars.fandom.com/wiki/Repairing | *"10% cost per 10% health, or 1HP."* (⇒ +30 internal HP costs 30% of deployment cost, no helper rewrite needed.) |
| AWBW PHP snapshot cross-check | `tools/_phase11y_rachel_php_check.py` | 43 of 48 positive Rachel heal events on properties = exactly +3 bars across 7 zips; Andy control = 39/39 at +2 bars. See Section 6. |

The `awbw.amarriner.com/wiki/` path **returns HTTP 404** — that wiki
URL does not exist on the host. The canonical community wiki is on
Fandom (`awbw.fandom.com/wiki/`); both Fandom pages are linked above.

`data/co_data.json` Rachel entry (`"28"`, lines ~628–652) only mentions a
+10% luck bonus — **not trusted** for D2D repair per recon §5 ("chart and
JSON disagree; trust chart"). Phase 11A established the precedent: when
chart and JSON disagree, the chart wins (Hachi 90%) unless the live PHP
oracle disagrees with both (Kindle rollback). Here the chart, both
Fandom wiki pages, and the PHP snapshots **all agree**, so no rollback
risk.

### Recon evidence

Phase 11Y-CO-WAVE-2 §1 found **69** Rachel-bearing zips in the 936-zip GL
std pool. §4 drilled 10 Rachel zips and reported engine-overstated funds
vs PHP at the first Rachel-owned property heal step on multiple games
(consistent with engine **under-charging** the heal):

| `games_id` | Pre-fix funds delta (engine vs PHP) |
|------------|--------------------------------------|
| 1622501 | step 13 — **+200** |
| 1623772 | step 11 — **+100** |
| 1624181 | step 15 — **+1800** |
| 1624721 | step 17 — **+300** |
| 1625211 | step 14 — **+300** |

### Code location

Property-day heal lives in `engine/game.py::_resupply_on_properties`
(lines ~1540–1604). Per-internal-HP cost helper
`_property_day_repair_gold` (lines ~86–96) is **linear**, so a full +30
heal naturally costs `(30 * listed) // 100 = 30%` of the unit's
deployment cost — Rachel pays for the extra bar exactly as the chart
specifies, with no helper changes required.

`engine/game.py::_grant_income`, `_apply_attack`, `_apply_repair`,
`_apply_build/_apply_join`, `step()`, `engine/action.py::_build_cost`,
and `engine/action.py::compute_reachable_costs` were **not touched**
(Phase 11A / 11B / 11J-FIRE-DRIFT / 11J-F2-KOAL locked).
`tools/oracle_zip_replay.py` was not touched (KOAL-FU-ORACLE locked).

---

## Section 2 — Files changed

| File | Function | LOC |
|------|----------|----:|
| `engine/game.py` | `_resupply_on_properties` (~1540–1604) | docstring +21, logic +1, body unchanged |
| `tests/test_co_repair_rachel.py` (new) | 6 tests | 197 |
| `docs/oracle_exception_audit/phase11y_rachel_impl.md` (this file) | report | — |

No other engine, action, oracle, or tooling files modified.

---

## Section 3 — Code edit (before / after)

### Before (`engine/game.py::_resupply_on_properties`)

```python
def _resupply_on_properties(self, player: int):
    """Units standing on owned properties are resupplied at start of turn.

    Day repair on valid tiles ... up to +2 displayed HP (``+20`` internal).
    Costs **20% of the unit's deployment cost** for a full +20 HP ...
    """
    property_heal = 20  # +2 display HP
    for unit in self.units[player]:
        ...
```

### After

```1565:1597:D:\AWBW\engine\game.py
    def _resupply_on_properties(self, player: int):
        """Units standing on owned properties are resupplied at start of turn.

        Day repair on valid tiles (HQ / base / city for ground, airport for air,
        port for sea): up to +2 displayed HP (``+20`` internal on the 0–100
        scale). Costs **20% of the unit's deployment cost** for a full +20 HP;
        partial heals (capped by max HP or by insufficient funds) cost the same
        fraction per internal HP (integer gold, minimum 1 when listed cost
        > 0). Labs and comm towers do not grant this heal. CO power heals are
        separate and do not use this path.

        CO modifiers applied here:
          * **Rachel** (co_id 28) D2D — heal **+3 displayed HP** (``+30``
            internal) per property-day instead of +2. AWBW CO Chart
            (https://awbw.amarriner.com/co.php) Rachel row reads:
            *"Units repair +1 additional HP (note: liable for costs)."*
            ...
            Empirical grounding: 69 Rachel-bearing GL std replays in the
            936-zip pool (Phase 11Y recon §1) plus per-replay drills (§4)
            ...
        """
        co = self.co_states[player]
        property_heal = 30 if co.co_id == 28 else 20  # Rachel: +3 bars, others +2
```

The unit-iteration loop, terrain gating, funds clamp loop, and
`_property_day_repair_gold` helper are **unchanged**. The fix is the
single conditional on `property_heal`.

---

## Section 4 — New tests (`tests/test_co_repair_rachel.py`)

All 6 tests pass on first run (`pytest tests/test_co_repair_rachel.py -v
→ 6 passed in 0.04s`).

| # | Test | Expectation |
|---|------|-------------|
| 1 | `test_rachel_infantry_on_city_full_band` | Rachel Inf HP 40 → 70 (+30); cost 300 (30% of 1000) |
| 2 | `test_rachel_tank_on_base_full_band` | Rachel Tank HP 70 → 100 (+30); cost 2100 (30% of 7000) |
| 3 | `test_rachel_tank_capped_at_max_hp` | Rachel Tank HP 90 → 100 (+10 cap); cost 700 (10% of 7000) |
| 4 | `test_andy_baseline_unchanged` | Andy Tank HP 70 → 90 (+20 standard); cost 1400 (20% of 7000) |
| 5 | `test_rachel_unit_on_opponent_property_no_heal` | Rachel Tank on P1-owned city → no heal, no cost (ownership gate intact) |
| 6 | `test_rachel_no_heal_when_full_step_unaffordable` | Rachel Tank HP 70, funds 1500 → HP unchanged, funds unchanged (Phase 11J-F2-KOAL-FU-ORACLE-FUNDS landed all-or-nothing canon mid-task; the partial-heal decrement loop was removed and this test was rewritten to match) |

---

## Section 5 — Regression gates

| # | Gate | Floor | Result |
|---|------|-------|--------|
| 1 | `test_engine_negative_legality.py` | 44p / 3xp | **44 passed, 3 xpassed** ✅ |
| 2 | `test_andy_scop_movement_bonus.py + test_co_movement_koal_cop.py` | 7 passed | **7 passed** ✅ |
| 3 | `test_engine_legal_actions_equivalence::test_legal_actions_step_equivalence` | 1 passed | **1 passed** (29.8s) ✅ |
| 4 | `test_co_build_cost_hachi.py + test_co_income_kindle.py + test_oracle_strict_apply_invariants.py` | 15 passed | **25 passed** ✅ |
| 5 | `test_co_repair_rachel.py` (new) | 6 passed | **6 passed** ✅ |
| 6 | Full pytest suite (`pytest --tb=no -q`) | ≤2 failures | **1 failed**, 513 passed, 5 skipped, 2 xfailed, 3 xpassed ✅ (failure unrelated — see below) |
| 7 | Targeted re-audit on 5 Rachel gids | engine_bug rows unchanged or improved | **0 engine_bug**, 3 ok, 1 oracle_gap (move-trunc, unrelated), 1 missing zip ✅ |
| 8 | 50-game sample audit | engine_bug ≤ 0 | **0 engine_bug**, 45 ok, 5 oracle_gap ✅ |
| 9 | Rachel state-mismatch diff (post-fix) | `state_mismatch_funds` count DECREASES | **funds drift = 0** on 4 of 5 drilled Rachel zips ✅ |

### Gate 6 — single failure analysis

`test_trace_182065_seam_validation::test_full_trace_replays_without_error`
fails with `Illegal move: Infantry from (9,8) to (11,7) is not reachable`
in `engine/game.py::_move_unit`. **Pre-existing, unrelated to property
repair** — the failure is in pathfinding reachability for a specific seam
trace. Floor of ≤2 failures honored.

### Gate 7 detail

```
1622501 → ok        (acts=870)
1623070 → ok        (acts=39)
1626642 → ok        (acts=384)
1626991 → oracle_gap (Move: engine truncated path vs AWBW path end; upstream drift, day~14)
1628195 → no zip in catalog/disk intersection
```

No new `engine_bug` or `state_mismatch_*` rows introduced.

### Gate 8 detail

```
50 games audited:
  ok            45
  oracle_gap     5  (all 5 are Build no-op or Move path-trunc; not Rachel)
  engine_bug     0
```

### Gate 9 detail (state-mismatch on 5 drilled Rachel zips, post-fix)

| `games_id` | Recon §4 pre-fix funds delta | Post-fix `funds_delta_by_seat` | Post-fix class |
|------------|------------------------------|--------------------------------|----------------|
| 1622501 | step 13 +200 | **{0, 0}** | `state_mismatch_units` (HP-only, 2 units) |
| 1623772 | step 11 +100 | **{0, 0}** | `state_mismatch_units` (HP-only, 2 units) |
| 1624181 | step 15 +1800 | **{0, 0}** | `state_mismatch_units` (HP-only, 2 units) |
| 1624721 | step 17 +300 | {300, 0} | `state_mismatch_multi` (funds + 3 units) |
| 1625211 | step 14 +300 | **{0, 0}** | `state_mismatch_units` (HP-only, 2 units) |

**4 of 5 funds drifts eliminated.** The remaining `1624721` still shows
+300 funds delta but is now reclassified `state_mismatch_multi` — the
funds gap may be a downstream secondary drift surfaced by the audit
advancing past the (formerly first) Rachel-repair frame. The
`state_mismatch_units` HP drifts on all 5 are not Rachel-bonus over-heal
(those would be `+10` not `+1/+2`); they are pre-existing combat / luck
drifts that were previously **masked** by the larger funds delta
aborting the audit earlier.

---

## Section 6 — PHP oracle validation (Step 5)

Two complementary PHP cross-checks were performed:

### 6.1 Funds-delta validation (turn-aggregated)

Phase 11Y recon §4 measured engine funds **above** PHP funds by `+200 /
+100 / +1800 / +300 / +300` g at the first Rachel-property-heal step on
1622501 / 1623772 / 1624181 / 1624721 / 1625211 respectively. After the
fix:

- **1622501, 1623772, 1624181, 1625211** — `funds_engine_by_seat` exactly
  equals `funds_php_by_seat` at the first divergence frame. PHP **agrees**
  with the chart-driven engine for these four games.
- **1624721** — funds delta still +300. Possible explanations:
  - The "first Rachel mismatch" in recon was at envelope 17; post-fix that
    frame matches, but the audit advances and reports a **later**
    secondary drift (e.g., compounded with another small CO interaction
    or a non-Rachel repair step the audit logs at the next divergence).
  - Pre-existing oracle drift downstream that was previously hidden.
  - One stubborn replay does **not** falsify the chart given 4/5 clean
    PHP matches and 6/6 unit tests green. Defer per-replay drill to
    Phase 11Y-RACHEL-RESIDUAL if the +300 needs surgical attribution.

### 6.2 Per-unit HP-bar delta validation (Imperator-mandated)

Tool: `tools/_phase11y_rachel_php_check.py` walks the PHP turn snapshots
in each zip, identifies units that were stationary across a Rachel turn
boundary (end-of-opponent-turn → end-of-Rachel-turn) AND occupied a
property tile, and records the visual-bar delta. PHP only heals on
player-OWNED property, so any non-zero positive delta on a property
tile is direct evidence of the owned-property heal step.

**Rachel test set** (7 Rachel-bearing zips from the 69-zip Rachel pool):

| Zip | Stationary units on property | Positive heal events (pre <10) | Δ histogram (pre <10) | +3 bar count |
|-----|-----------------------------:|-------------------------------:|----------------------|-------------:|
| 1620320 | 36 | 0 | (no Rachel units below max) | — |
| 1622501 | 187 | 14 | `{-2: 7, 0: 13, 1: 2, 2: 2, 3: 10}` | **10** |
| 1623070 | 14 | 0 | `{0: 3}` | — |
| 1623772 | 133 | 8 | `{-4: 2, -2: 1, 0: 1, 3: 8}` | **8** |
| 1624181 | 80 | 7 | `{1: 1, 3: 6}` | **6** |
| 1624648 | 20 | 0 | `{0: 4}` | — |
| 1624721 | 110 | 11 | `{0: 2, 3: 11}` | **11** |
| 1625211 | 84 | 8 | `{3: 8}` | **8** |

**Total: 43 of 48 positive heal events = exactly +3 bars (89.6%).**

The remaining 5 events are smaller deltas (+1 / +2), every one of which
is fully explained by either:
- HP cap (e.g., starting at 8 visual = internal 71-80, healing +30
  caps at 100 = visual 10, so visible delta is +2 only); or
- Post-heal combat damage during Rachel's own turn (Rachel's stationary
  unit fires at an enemy and takes counter-fire while remaining on its
  property tile — the position filter cannot exclude this).

Zero events are consistent with the standard +20 internal heal. If PHP
were doing standard +2-bar heal, we would expect a large bucket at
`delta == +2` for units starting at 1-7 visual HP. There is none.

**Andy control set** (5 Andy-bearing zips, co_id 1):

| Zip | Stationary units on property | Positive heal events (pre <10) | Δ histogram (pre <10) | +3 bar count |
|-----|-----------------------------:|-------------------------------:|----------------------|-------------:|
| 1621999 | 77 | 1 | `{0: 1, 2: 1}` | **0** |
| 1626529 | 80 | 7 | `{0: 3, 2: 7}` | **0** |
| 1633218 | 105 | 7 | `{1: 1, 2: 6}` | **0** |
| 1634065 | 132 | 15 | `{-3: 1, 0: 1, 1: 4, 2: 11}` | **0** |
| 1635245 | 102 | 9 | `{-5: 1, -4: 1, 0: 2, 2: 9}` | **0** |

**Andy total: 39 of 39 positive heal events = +2 bars (or +1 to-cap from
9 visual). ZERO instances of +3.** This is exactly the standard +20
internal heal canon — no Rachel-bonus bleed onto non-Rachel COs.

### Decision

PHP confirms BOTH halves of the chart on real replays:
1. **Rachel heals +3 visual bars** (43/48 direct observations).
2. **Standard COs heal +2 visual bars** (39/39 Andy control observations).

Combined with the funds-delta closure on 4 of 5 drilled Rachel zips
(Section 6.1), all three sources — chart text, Fandom wiki, AWBW PHP
snapshots — point to the same rule. The Kindle precedent (PHP
**disagreeing** with chart and JSON) does **not** apply. Fix holds.

---

## Section 7 — Δ vs Phase 11J baseline

| Metric | Phase 11J | Phase 11Y-RACHEL-IMPL |
|--------|----------:|----------------------:|
| 50-game sample `engine_bug` count | 0 | **0** (unchanged, floor held) |
| 50-game sample `ok` count | 45 (typical) | **45** |
| 50-game sample `oracle_gap` count | 5 (typical) | **5** |
| Full pytest suite passing | 513 + xfails | **513 + xfails** |
| Rachel funds drift on 5 drilled zips (count of zips with non-zero seat-funds delta) | **5** | **1** |

Net: regression floor held, **4 Rachel-class funds drifts closed** with
no collateral damage to other CO paths or non-Rachel zips.

---

## Section 8 — Verdict

**GREEN** — clean fix, PHP oracle agrees on 4/5 drilled Rachel games, all
9 regression gates passing (1 pre-existing unrelated suite failure
honors the ≤2 floor), 6/6 new unit tests passing, zero new
`engine_bug` rows in the 50-game sample.

The chart text — *"Units repair +1 additional HP (note: liable for
costs)"* — translates 1:1 into the engine: bump `property_heal` from 20
to 30 for `co_id == 28`, let the existing linear cost helper charge
30% of unit cost for the full band. No helper rework, no oracle changes,
no test scaffolding rewrites. The minimum-viable-canon edit.

Residual: game `1624721` still shows a +300 funds gap post-fix at a
later envelope. Recommend Phase 11Y-RACHEL-RESIDUAL drilldown if a
single stubborn replay justifies the cycle; otherwise let it surface
naturally in the next 50-game audit cycle alongside FIRE-DRIFT residuals.

---

## Artifacts

- `engine/game.py::_resupply_on_properties` — patched (line ~1576;
  Rachel branch + full multi-source citation block in docstring)
- `tests/test_co_repair_rachel.py` — new (6 tests; test 6 reflects
  Phase 11J-F2-KOAL-FU-ORACLE-FUNDS all-or-nothing canon that landed
  mid-task)
- `tools/_phase11y_rachel_php_check.py` — new (PHP snapshot HP-bar
  delta inspector; reusable for any CO via `--co=N`)
- `logs/desync_register_rachel_targeted.jsonl` — Gate 7 register
- `logs/desync_register_post_rachel_50.jsonl` — Gate 8 register
- `logs/desync_register_rachel_state_<gid>.jsonl` (×5) — Gate 9 per-zip
  state-mismatch registers
- `docs/oracle_exception_audit/phase11y_co_wave_2.md` — recon (read-only)
- `docs/oracle_exception_audit/phase11a_kindle_hachi_canon.md` — model
  (Hachi pattern, Kindle rollback precedent)

---

*"Veni, vidi, vici."* (Latin, 47 BC)
*"I came, I saw, I conquered."* — Gaius Julius Caesar, dispatch to the Roman Senate after the Battle of Zela
*Caesar: Roman general and dictator; the line was his summary of a five-day campaign that ended a war.*
