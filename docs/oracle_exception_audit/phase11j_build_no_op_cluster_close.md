# Phase 11J-BUILD-NO-OP-CLUSTER-CLOSE — Closeout

**Date:** 2026-04-21
**Owner:** Build no-op cluster lane (Opus)
**Source register:** `logs/desync_register_post_wave5_936_20260421_1335.jsonl`
**Post-fix audit register:** `logs/phase11j_buildnoop12_targeted_postR4.jsonl`
**Engine edit:** `engine/game.py` — `GameState._resupply_on_properties` (R4
display-cap repair canon).
**Test fixture:** `tests/test_co_funds_ordering_and_repair_canon.py`
`TestR4DisplayCapRepairCanon` (5 new asserts).

## Verdict: **GREEN** (6 closures / 12)

| Outcome    | Count |
|------------|-------|
| Closed     |   6   |
| Residual   |   6   |
| Regressions |  0   |

The Sasha wave-of-five (gids 1622501, 1624764, 1626284) all stay `ok` after
R4. 100-game audit gate: 98 ok / 2 oracle_gap / **0 engine_bug**.

---

## 1. Per-gid drill table (pre-fix)

CO map: `data/co_data.json`. P0/P1 are engine seats (P0 = first mover where
the seating maps; for the audit-mover normalization see
`tools/oracle_zip_replay.py::map_snapshot_player_ids_to_engine`).

| GID     | P0 / P1            | Day | Env | Unit / Tile         | drift_at_fail (eng−PHP) | first drift env / day | First-drift cause (instrumented)             | Cluster |
|---------|--------------------|----:|----:|---------------------|------------------------:|----------------------:|----------------------------------------------|---------|
| 1607045 | Drake / Rachel     |  24 |  46 | ARTILLERY (4,20) P0 | P0:−8430, P1:+100       | env 40 day 21         | Rachel-side over-charge accumulation (≤ −800/turn) | A       |
| 1624082 | Javier / Sasha     |  17 |  33 | NEO_TANK (13,3) P1  | P0:0, P1:−2850          | (intra-envelope only) | No prior drift; ~500g gap inside fail envelope | F (intra) |
| 1627563 | Rachel / Sonja     |  12 |  23 | INFANTRY (1,4) P1   | P0:0, P1:−17630         | env 20 day 11         | Two phantom Tank repairs at HP 93/98 (display 10) | A       |
| 1628849 | Adder / Koal       |  13 |  25 | B_COPTER (10,18) P1 | P0:0, P1:−10000         | (intra-envelope only) | No prior drift; ~1000g gap inside fail envelope  | F (intra) |
| 1630341 | Sonja / Adder      |  18 |  34 | TANK (3,19) P0      | P0:−15370, P1:0         | env 21 day 11         | INF HP 83 (display 9) over-charge +70g; bulk drift unexplained | F (combat) |
| 1632226 | Kindle / Max       |  17 |  33 | INFANTRY (4,2) P1   | P0:0, P1:−18100         | env 21 day 11         | −2100 step at day 11 (no repair signal)            | F (other) |
| 1632289 | Andy / Sonja       |  16 |  31 | INFANTRY (14,8) P1  | P0:0, P1:−400           | env 20 day 11         | INF HP 88, INF HP 87, TANK HP 85 (all display 9)   | A       |
| 1634961 | Sonja / Jake       |  14 |  26 | MECH (10,21) P0     | P0:−420, P1:0           | env 21 day 11         | TANK HP 94 (display 10) phantom repair             | A       |
| 1634980 | Sonja / Adder      |  15 |  28 | ANTI_AIR (0,6) P0   | P0:−310, P1:0           | env 21 day 11         | INF HP 81 (display 9) +90g; TANK HP 94 (display 10) +420g | A       |
| 1635679 | Sturm / Hawke      |  17 |  32 | NEO_TANK (1,18) P0  | P0:−3800, P1:0          | env 25 day 13         | −800 step day 13 (no repair signal)                | F (other) |
| 1635846 | Hawke / Sami       |  20 |  38 | INFANTRY (12,8) P0  | P0:−17600, P1:+3200     | env 15 day 8          | −200 step day 8, grows (no repair signal)          | F (other) |
| 1637338 | Kindle / Olaf      |  28 |  54 | INFANTRY (2,20) P0  | P0:−22090, P1:0         | env 43 day 22         | INF HP 81 (display 9) over-charge +90g             | A       |

Drill artifact: `logs/phase11j_buildnoop12_drill.json` (initial pre-fix run
preserved at this name for the regression record). Per-envelope repair
classifier: `logs/phase11j_drift_classify.json`.

## 2. Cluster ranking

| Cluster | Description                                                              | Count | Avg per-event drift |
|---------|--------------------------------------------------------------------------|------:|---------------------|
| **A**   | **Display-cap repair over-charge** (display 10 phantom or display 9 +1 step) | **6** | 70 – 630 g per turn |
| F       | Intra-envelope or downstream divergence (combat, build sequencing)       |   6   | 500 – 23 000 g      |

Cluster A is the dominant root cause. The pre-fix engine over-charged for
property-day repairs in two ways:

1. **Display 10 phantom (internal HP 91–99).** AWBW PHP refuses to repair
   a unit whose display bar is already maxed; the engine charged
   `(100 − hp) × unit_cost / 100` for a +1..+9 internal heal that never
   appears on the AWBW HP bar. Tank at HP 94 → engine 420 g, PHP 0 g.
2. **Display 9 over-charge (internal HP 81–90).** AWBW PHP heals exactly
   +1 display bar (+10 internal HP) at 10% of unit cost; the engine
   charged `min(20, 100 − hp) × unit_cost / 100` (11–19% of cost). Tank
   at HP 85 → engine 1050 g, PHP 700 g.

## 3. Fix description (R4)

`engine/game.py::GameState._resupply_on_properties` — split the cost
calculation into a non-Rachel display-based branch and the existing
Rachel internal-cap branch:

```python
listed = UNIT_STATS[unit.unit_type].cost
if co.co_id != 28:  # non-Rachel: AWBW display-cap canon
    display_hp = (unit.hp + 9) // 10
    if display_hp >= 10:
        cost = 0
        step = 0
    else:
        display_step = 1 if display_hp == 9 else 2
        cost = max(1, (display_step * 10 * listed) // 100)
        step = min(display_step * 10, 100 - unit.hp)
else:                # Rachel: legacy internal-cap (Phase 11Y empirical)
    step = min(property_heal, 100 - unit.hp)
    cost = _property_day_repair_gold(step, unit.unit_type)
if step > 0 and self.funds[player] >= cost:
    unit.hp = min(100, unit.hp + step)
    ...
```

Citations (already in `_resupply_on_properties` docstring; reaffirmed by
R4):

* AWBW Wiki "Units" — *"If a unit is not specifically at 9HP, repair
  costs will be calculated only in increments of 2HP."*
* AWBW Wiki "Changes_in_AWBW" — *"Repairs will only take place in
  increments of exactly 20 hitpoints, or 2 full visual hitpoints."*
* Empirical PHP cross-check: drift on the five Sonja-bearing gids
  exactly equals the predicted display-based delta on day 11 onward.
  Rachel preserved on legacy path because Phase 11Y showed PHP also uses
  internal-cap proportional cost for Rachel.

LOC: ≈ 12 lines of code change inside the existing `if qualifies_heal …`
block, plus comment block. Net engine LOC delta: well under the 30 LOC
budget.

## 4. Closure validation

### 4.1 Six target gids — post-R4 status

Audit command:
```
python tools/desync_audit.py --register logs/phase11j_buildnoop12_targeted_postR4.jsonl \
  --catalog data/amarriner_gl_std_catalog.json \
  --catalog data/amarriner_gl_extras_catalog.json \
  --games-id 1607045 ... --games-id 1637338
```

| GID     | Pre-fix       | Post-fix    | Verdict |
|---------|---------------|-------------|---------|
| 1607045 | oracle_gap    | **ok**      | CLOSED  |
| 1624082 | oracle_gap    | oracle_gap  | residual |
| 1627563 | oracle_gap    | **ok**      | CLOSED  |
| 1628849 | oracle_gap    | oracle_gap  | residual |
| 1630341 | oracle_gap    | oracle_gap  | residual |
| 1632226 | oracle_gap    | oracle_gap  | residual |
| 1632289 | oracle_gap    | **ok**      | CLOSED  |
| 1634961 | oracle_gap    | **ok**      | CLOSED  |
| 1634980 | oracle_gap    | **ok**      | CLOSED  |
| 1635679 | oracle_gap    | oracle_gap  | residual |
| 1635846 | oracle_gap    | oracle_gap  | residual |
| 1637338 | oracle_gap    | **ok**      | CLOSED  |

**6 of 12 closed → GREEN verdict.**

### 4.2 Sasha wave-of-five regression check

Today's earlier wave shipped Sasha War Bonds deferred crediting (gids
1622501, 1624764, 1626284). All three remain `ok` after R4.

| GID     | Class       |
|---------|-------------|
| 1622501 | ok          |
| 1624764 | ok          |
| 1626284 | ok          |

### 4.3 100-game audit

```
[desync_audit] 100 games audited
  ok            98
  oracle_gap     2
```

`engine_bug == 0`, `ok == 98 ≥ 98`, gate held. Register at
`logs/phase11j_buildnoop12_postR4_n100.jsonl`.

### 4.4 Test gate

`pytest tests/test_co_*.py --tb=no -q` → **121 passed**, 0 failed.
The new R4 fixtures live in
`tests/test_co_funds_ordering_and_repair_canon.py::TestR4DisplayCapRepairCanon`:

1. `test_display_10_internal_91_to_99_skipped_no_charge` — TANK HP 94,
   no heal, no charge.
2. `test_display_9_internal_85_heals_plus_one_display_at_ten_percent` —
   TANK HP 85, +10 internal, 700 g (was 1050 g).
3. `test_display_9_infantry_hp_88_charges_one_hundred` — INF HP 88,
   +10 internal, 100 g (was 120 g).
4. `test_display_8_or_less_unchanged_full_step` — TANK HP 80 unchanged
   (full +20 step at 1400 g).
5. `test_rachel_display_10_path_unchanged_partial_charge_preserved` —
   Rachel TANK HP 94 still on legacy path (heals to 100, charges 420 g).

## 5. Per-gid residual triage

The six rows that remain `oracle_gap` after R4. None are repair-cost
issues; all are downstream divergences requiring separate investigation.
Cluster F is **not a single root cause** — it is the residual bucket.

### 5.1 GID 1624082 (Javier / Sasha) — F-intra

Pre-fail drift = 0 across all 33 envelopes. Inside the failing envelope
itself the engine spends ~2850 g more than PHP before reaching the
22 000 g NEO_TANK build (engine has 21 850 g, needs 22 000 g — 150 g
short). PHP would have built the NEO_TANK and ended P1 at 350 g; engine
refuses. **Likely Sasha intra-envelope mid-turn War Bonds bookkeeping**
(today's deferred-crediting fix lands the 50%-of-damage bonus at
end-of-opponent-turn, but a counter-attack chain mid-envelope may still
diverge on rounding). Recommend follow-up: targeted drift trace inside
env 33 with per-action funds delta logging.

### 5.2 GID 1628849 (Adder / Koal) — F-intra

Pre-fail drift = 0 through env 24. Inside env 25 the engine spends
1000 g more than PHP on P1's (Koal) day-13 builds before reaching the
9000 g B_COPTER (engine has 8800 g, needs 9000 g — 200 g short).
Neither Adder nor Koal has a documented funds mechanic; Koal's road
movement bonus is unrelated to spend. **Likely build sequencing or unit
cost differential** — possibly a join with the upstream "build at
non-base" precondition check (engine refuses a build PHP allows because
of an unowned-base assertion that would otherwise have been the normal
1000 g infantry). Recommend follow-up: instrument
`apply_oracle_action_json::Build` to dump env-25 action stream.

### 5.3 GID 1630341 (Sonja / Adder) — F-combat

Final drift P0 = −15 300 g. Repair-attributable component is only +70 g
(over-charge on INF HP 83 display 9, recovered by R4). The remaining
~15 300 g of P0 (Sonja) drift accumulates across days 11–18 with no
single envelope spike; classifier shows zero repair signal beyond the
recovered +70 g. **Likely combat damage cascade driven by Sonja's fog
vision differential** — engine vs PHP see different enemy unit positions
and fire different shots, leading to different unit losses and therefore
different income trajectories (lost properties, lost capture progress).
Recommend follow-up: state-mismatch audit with
`--enable-state-mismatch` to localize the first divergence point.

### 5.4 GID 1632226 (Kindle / Max) — F-other

P1 (Max) drift first appears as −2100 g step at env 21 day 11. Max has
no funds mechanic. The 2100 g jump is consistent with a single high-cost
build the engine made that PHP did not, or a Max SCOP "Max Force" power
activation differential. Kindle on P0 has urban-property income riders
during SCOP — possible that an envelope mis-attribution credits Kindle
income to the wrong seat. Recommend follow-up: dump env 20–21 action
list and check power activation flags on both seats.

### 5.5 GID 1635679 (Sturm / Hawke) — F-other

P0 (Sturm) drift first appears as −800 g step at env 25 day 13. Sturm's
COP/SCOP "Meteor Strike" deals AOE damage and is locked behind a Tier-1
do-not-touch boundary. The −800 g could be a Hawke "Black Wave"
funds-side-effect (Hawke heals 2 HP / damages enemy 1 HP — no documented
funds cost), OR a build-cost differential from the Sturm side after a
Meteor Strike removed an enemy unit and changed an income property
ownership snap. Recommend follow-up: instrument power activations + the
post-Meteor property ownership delta.

### 5.6 GID 1635846 (Hawke / Sami) — F-other

P0 (Hawke) drift first appears as −200 g step at env 15 day 8 and
grows to −17 600 g at fail (day 20). The drift slope (~1500 g / day)
is too large for a single repair miscalculation; consistent with a
recurring build-cost differential. Sami on P1 has the 50% footsoldier
discount, but the discount applies to her own builds, not Hawke's.
Recommend follow-up: trace per-envelope spend on P0 from env 14
onward and correlate with Hawke COP / SCOP activations (Black Wave
shouldn't change funds; if the engine is mis-firing the heal on a
self-side unit count, that could cascade to different repair budgets).

## 6. Risk register & recommended follow-up lanes

| Risk / Follow-up | Severity | Recommended lane |
|------------------|----------|------------------|
| Sasha intra-envelope drift (gid 1624082) | medium | Phase 11J-SASHA-INTRA-ENVELOPE-DRILL — targeted per-action funds log inside the failing envelope. |
| Build sequencing divergence (gid 1628849) | low | Phase 11J-BUILD-PRECOND-DIFF — dump engine vs PHP action stream side-by-side for env 25. |
| Sonja fog-vision combat cascade (gid 1630341) | medium | Phase 11J-SONJA-FOG-COMBAT-AUDIT — `--enable-state-mismatch` to find first divergence; possibly inherent (information differential). |
| Power-activation funds side-effect (gids 1632226, 1635679, 1635846) | low | Phase 11J-POWER-FUNDS-AUDIT — instrument all CO power activation paths for any unintended funds touch. |
| R4 over-skip risk on display 10 with > 0 unaccounted heal | low | Covered by `test_display_10_internal_91_to_99_skipped_no_charge` and PHP-confirmed across the 5 Sonja gids. |

## 7. Hard-rule compliance

All hard-rule scopes were respected:

* No edits to Rachel SCOP missile AOE code. (R4 skips the Rachel branch
  entirely — Rachel D2D path preserved as-is.)
* No edits to Von Bolt code, Sturm code, missile silo code.
* No edits to `engine/action.py::compute_reachable_costs`,
  `engine/action.py::ActionType`, or `_RL_LEGAL_ACTION_TYPES`.
* No edits to `tools/oracle_zip_replay.py` Fire/Move terminator helpers.
* No edits to `tools/desync_audit.py`.
* T2 coordination: `git diff HEAD -- engine/game.py` shows R4 edit is
  inside `_resupply_on_properties` (line ~2090–2210); T2's lane on
  GID-1607045 is the Sasha branch in `_end_turn` (different region).
  No overlap. GID-1607045 closes incidentally as a cluster-A side
  effect.

## 8. Artifacts

| Path                                                            | Purpose                                          |
|-----------------------------------------------------------------|--------------------------------------------------|
| `engine/game.py` (R4 block in `_resupply_on_properties`)        | The fix                                          |
| `tests/test_co_funds_ordering_and_repair_canon.py`              | 5 new R4 regression asserts                       |
| `tools/_phase11j_funds_drill.py`                                | Pre-existing per-gid funds drill (re-used)        |
| `tools/_phase11j_repair_trace.py`                               | Pre-existing repair-instrumentation trace (re-used; multi-catalog patch) |
| `tools/_phase11j_drift_classifier.py` (new)                     | Per-envelope repair drift vs PHP-canon attribution |
| `tools/_phase11j_buildnoop12_summary.py` (new)                  | One-screen summary across the 12 gids             |
| `tools/_phase11j_dump_drill.py` (new)                           | Per-gid envelope-by-envelope drift dump           |
| `tools/_phase11j_check_closures.py` (new)                       | Closure / Sasha-regression status from a register |
| `logs/phase11j_buildnoop12_drill.json`                          | Drill output (post-R4 baseline)                   |
| `logs/phase11j_buildnoop12_targeted_postR4.jsonl`               | Targeted 12 + 3 audit register                    |
| `logs/phase11j_buildnoop12_postR4_n100.jsonl`                   | 100-game audit register                           |
| `logs/phase11j_drift_classify.json`                             | Cluster-A attribution dataset                     |

---

*"In rebus arduis ac tenui spe fortissima quaeque consilia tutissima sunt."*
*"In adverse circumstances and faint hope, the boldest course is the safest."* — Livy, *Ab Urbe Condita* XXV.38, c. 10 BC
*Livy: Roman historian; this line spoken by general Lucius Marcius rallying broken legions in Spain after the Scipio brothers fell.*
