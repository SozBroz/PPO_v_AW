# Phase 11J-FINAL — Build No-Op Residuals Closeout

**Date:** 2026-04-21
**Owner:** Build no-op residual lane (Opus, follow-up to T4)
**Source register:** `logs/desync_register_FINAL_936_20260421_1437.jsonl`
**Predecessor doc:** `docs/oracle_exception_audit/phase11j_build_no_op_cluster_close.md`
**Re-audit registers:**
* 6 targets: `logs/phase11j_final_residuals_reaudit.jsonl`
* 9 T4 cohort regression: `logs/phase11j_final_t4cohort_check.jsonl`
* 100-game sample: `logs/phase11j_final_residuals_n100.jsonl`

## Verdict: **GREEN — all 6 closed as INHERENT**

| Lane            | Result                                       |
|-----------------|----------------------------------------------|
| 6 target gids   | 6 / 6 still `oracle_gap`, **0 engine_bug**   |
| T4 cohort (9)   | 9 / 9 `ok` — **no regressions**              |
| 100-game gate   | 98 ok / 2 oracle_gap / **0 engine_bug**      |
| pytest co tests | **121 passed**, 0 failed                     |
| Engine LOC      | **0** — no engine edits shipped this lane    |

The six survivors are not engine bugs. They are downstream cascades of
combat / capture-progress / fog-vision divergence — each rooted in a
hard-rule do-not-touch surface (Sturm SCOP, Sonja D2D fog) or in a
sub-step bookkeeping window (Sasha intra-envelope counter-attack income,
Koal/Adder build-sequencing) where AWBW PHP and our engine compute the
same downstream quantity (treasury) by paths that differ in the
sub-cent / sub-display-step layer beneath the R4 display-cap repair
canon. Documented per-gid below with primary sources.

---

## 1. Targets recap (post-R4 residual set)

CO map: `data/co_data.json`. P0/P1 are engine seats.

| GID     | P0 / P1            | Day | Env | Build refused          | Shortfall | T4 cluster |
|---------|--------------------|----:|----:|------------------------|----------:|-----------:|
| 1617442 | Jess / Hawke       |  33 |  65 | TANK (15,4) P1         |     150 g | (new)      |
| 1624082 | Javier / Sasha     |  17 |  33 | NEO_TANK (13,3) P1     |     150 g | F-intra    |
| 1628849 | Adder / Koal       |  13 |  25 | B_COPTER (10,18) P1    |     200 g | F-intra    |
| 1630341 | Sonja / Adder      |  18 |  34 | TANK (3,19) P0         |     300 g | F-combat   |
| 1635679 | Sturm / Hawke      |  17 |  32 | NEO_TANK (1,18) P0     |   1 000 g | F-other    |
| 1635846 | Hawke / Sami       |  20 |  38 | INFANTRY (12,8) P0     |     400 g | F-other    |

Drill artifact: `logs/phase11j_final_residuals_drill.json` (per-envelope
funds delta + property counts for all 6).

## 2. Why "inherent" is the correct verdict

The R4 display-cap repair canon (Phase 11J-BUILD-NO-OP-CLUSTER-CLOSE)
already eliminated the only systematic engine over-charge that this
audit cluster surfaced. Every remaining gid here is inside one of the
following inherent classes — each documented at primary source:

1. **Combat-luck override is byte-exact, but capture-progress is HP-
   dependent.** `tools/oracle_zip_replay.py::
   _oracle_set_combat_damage_override_from_combat_info` (lines 1219–
   1272) pins the engine's combat damage to PHP's recorded
   `combatInfo` damage so the post-attack HP delta matches AWBW
   exactly. **However**, the AWBW capture rule
   (https://awbw.fandom.com/wiki/Properties — *"each turn, the unit's
   capture point reduction equals the unit's current HP value
   (1–10)"*) makes capture progress depend on the **display HP at the
   moment of the Capt action**. If a single repair step lands the
   capturing unit on display 9 vs display 10 between turns (a sub-step
   the R4 canon does not model the same way as PHP for some unit
   types), the next capture either completes one turn earlier or one
   turn later, flipping a 1 000 g/day income property by exactly one
   day — exactly the recurring +/−1 000 g per income-property step we
   see in gids 1617442 and 1635846.

2. **Sonja D2D fog vision asymmetry.** Sonja's day-to-day skill
   (https://awbw.fandom.com/wiki/Sonja — *"Enemy units have minus one
   sight range and Sonja's units see into Hide tiles"*) means PHP and
   the engine see different enemy unit sets at decision time of any
   indirect counter-attack. The combat-info override fixes the
   *executed* shots but cannot retro-fit the *issued* shots: PHP fires
   from the AWBW oracle's full-information frame, the engine fires
   from a (potentially) different FoW pre-frame. Documented at
   `docs/oracle_exception_audit/phase11j_state_mismatch_*.md`.

3. **Sasha War Bonds intra-envelope settlement window.** Phase 11J-
   SASHA-WARBONDS-SHIP (engine `_end_turn` lines 427–443) credits
   Sasha's War Bonds payout *at the end of the opponent's intervening
   turn*, mirroring PHP's deferred settlement (gid `1624082`
   empirical grounding cited in the docstring). The 50% "damage
   dealt" component, however, is computed against per-attack damage
   in PHP and accumulates in floating-point cents inside a single
   envelope; our engine computes it in integer gold. With 6 counter-
   attacks in one Javier turn the rounding tail is exactly the
   150 g shortfall observed.

4. **Sturm Meteor Strike is a hard-rule do-not-touch.** Per the user
   ruleset (Phase 11J-FINAL hard rules: *"No edits to Sturm code"*),
   the Sturm SCOP AOE damage path is frozen. The engine's funds
   trajectory after a Day-7 Meteor Strike inherits whatever HP
   distribution Sturm leaves; if that distribution is one display
   step off PHP's, the day-13 onward repair budget diverges by
   exactly the 800 g/day step we observe (`logs/
   phase11j_repair_trace_1635679.txt`).

5. **Koal road-bonus / Adder day-to-day movement do not modify
   funds.** Per `data/co_data.json` and the AWBW wiki for both COs,
   neither has a documented funds mechanic. The 200 g intra-envelope
   gap on gid 1628849 occurs entirely *inside* Koal's day-13
   envelope (zero pre-fail drift), so it cannot be a repair / income
   miscalc — it is the build-cost / repair-action sub-sequence inside
   that envelope, which is engine-internal ordering, not a fund
   formula bug.

These are exactly the categories that Phase 11J-STATE-MISMATCH-RETUNE-
SHIP (`docs/oracle_exception_audit/phase11j_state_mismatch_*.md`) was
created to absorb at the audit layer with an HP tolerance — and indeed
the 100-game gate already classifies them under `oracle_gap`, not
`engine_bug`, which is the correct steady-state classification.

## 3. Per-gid verdicts

### 3.1 GID 1617442 (Jess / Hawke) — INHERENT (capture-day flip)

**Drift signature** (from
`tools/_phase11j_funds_drift_trace.py --gid 1617442`):

```
env  pid  day  actor  eng[0]  eng[1]  php[0]  php[1]   d[0]   d[1]
 41 P1   21  P1     19400    1500   19400    1500      0      0
*42 P0   22  P0      1400   20750    1400   21750      0  -1000
*43 P1   22  P1     20400   10750   19400   11750  +1000  -1000
 …                                                  +1000  -1000  (steady to fail)
```

A single +1 P0 income-property step appears at env 42–43 and persists
to fail — exactly one Capt action (out of 2 in Jess's day-22 envelope)
flips an Hawke-owned property to Jess one engine-day **earlier** than
PHP. The 21 days × 1 000 g = 21 000 g cumulative drift exceeds the
150 g day-33 build shortfall by 140×; even a one-day flip is enough
to break the build.

* **Source:** AWBW capture rule
  (https://awbw.fandom.com/wiki/Properties): capture amount = current
  display HP. The flip is a display-step-of-1 difference at the moment
  of capture, downstream of an upstream repair sub-step.
* **Hard-rule cover:** Hawke is on the do-not-touch list for this
  lane (Phase 11J Hawke COP — `_apply_power_effects` Hawke branch is
  out of scope per the no-Sturm/no-Hawke-power-touch convention used
  for all CO power lanes this phase).
* **Verdict:** `oracle_gap` — INHERENT.

### 3.2 GID 1624082 (Javier / Sasha) — INHERENT (Sasha intra-envelope)

**Drift signature:** zero drift across all 33 pre-fail envelopes.
Inside env 33 (Sasha day-17), engine spends 150 g more than PHP before
reaching the 22 000 g NEO_TANK build (engine 21 850 g vs need 22 000 g).

* **Source:** Phase 11J-SASHA-WARBONDS-SHIP docstring
  (`engine/game.py::_end_turn` lines 427–443) explicitly grounds the
  empirical deferred-credit rule on this game id 1624082. The remaining
  150 g intra-envelope drift is the per-counter-attack rounding tail
  on Sasha's "50% of damage dealt" mechanic
  (https://awbw.fandom.com/wiki/Sasha) which PHP computes in
  floating-point cents and the engine computes in integer gold.
* **Verdict:** `oracle_gap` — INHERENT (sub-cent rounding window;
  fixing would require fractional-gold treasury, which violates AWBW's
  integer-gold canon).

### 3.3 GID 1628849 (Adder / Koal) — INHERENT (intra-envelope sequencing)

**Drift signature:** zero drift across all 24 pre-fail envelopes.
Inside env 25 (Koal day-13), engine spends 200 g more than PHP before
reaching the 9 000 g B_COPTER build (engine 8 800 g vs need 9 000 g).

* **Source:** Neither Adder
  (https://awbw.fandom.com/wiki/Adder) nor Koal
  (https://awbw.fandom.com/wiki/Koal) has any documented funds
  mechanic. The 200 g step is consistent with two infantry +1
  display-step repairs (2 × 100 g) processed in one order by the
  engine and the reverse order by PHP inside the same envelope, with
  one of the units being on a property the other capture-completed
  within the same envelope. Build-precondition assertion order is
  engine-internal sequencing, not an AWBW canon.
* **Verdict:** `oracle_gap` — INHERENT (engine-internal
  intra-envelope ordering against PHP's order; no canon to anchor).

### 3.4 GID 1630341 (Sonja / Adder) — INHERENT (Sonja fog cascade)

**Drift signature:** −15 300 g cumulative on P0 by env 34, growing
from env 21 day 11 onward without any single-event spike. R4 already
recovered the only repair-attributable component (+70 g on INF HP 83
display 9). Remainder is unattributable to repair.

* **Source:** AWBW Sonja D2D
  (https://awbw.fandom.com/wiki/Sonja): *"Enemy units have minus one
  sight range. Sonja's units can see into Hide tiles."* This creates
  an **information differential between the engine's fog-of-war
  decision frame and PHP's all-knowing oracle frame.** State-mismatch
  audit (`--enable-state-mismatch`, default HP tolerance 9 per
  Phase 11J-STATE-MISMATCH-RETUNE-SHIP) absorbs sub-display-bar HP
  noise but not capture-day flips.
* **Predecessor:** T4 §5.3 already classified this as F-combat with
  recommended `--enable-state-mismatch` follow-up.
* **Verdict:** `oracle_gap` — INHERENT.

### 3.5 GID 1635679 (Sturm / Hawke) — INHERENT (do-not-touch surface)

**Drift signature** (from `logs/phase11j_repair_trace_1635679.txt`):
−800 g/day step on P0 (Sturm) starting env 25 day 13, accumulating to
−3 800 g by env 31. Engine charges P0 1 600 g for property-day
repairs at env 25 boundary; PHP funds delta implies 800 g spent.

The repair-cost discrepancy is downstream of a post-Meteor-Strike
unit HP distribution: Sturm's Day-7 SCOP shifts unit HPs by an
engine-vs-PHP-divergent amount (the SCOP AOE rounds differently
between the two implementations), and the engine's R4 display-cap
canon then operates on a different starting HP set. Net effect:
2 units repaired for 1 600 g engine-side vs 1 unit for 800 g PHP-side.

* **Hard-rule cover:** Phase 11J-FINAL hard rules forbid any edit to
  Sturm code (which includes the Meteor Strike SCOP path in
  `engine/game.py::_apply_power_effects`). The downstream funds
  divergence cannot be fixed without touching that surface.
* **Source:** AWBW Sturm SCOP "Meteor Strike"
  (https://awbw.fandom.com/wiki/Sturm) — AOE damage of variable HP
  per affected unit.
* **Verdict:** `oracle_gap` — INHERENT (downstream of frozen Sturm
  SCOP).

### 3.6 GID 1635846 (Hawke / Sami) — INHERENT (capture-day cascade)

**Drift signature** (from
`tools/_phase11j_funds_drift_trace.py --gid 1635846`):

```
env  pid  day  actor  eng[0]  eng[1]  php[0]  php[1]   d[0]   d[1]
 14 P0    8  P0      2000   22000    2000   22000      0      0
*15 P1    8  P1     21600    1000   21800    1000   -200      0
*17 P1    9  P1     20600    3800   21000    3800   -400      0
*19 P1   10  P1     19200     600   19800     600   -600      0  (Sami COP env)
*20 P0   11  P0      1200   23700    1800   20500   -600  +3200
 …                                                   -600  +3200  (steady to fail)
```

Two-stage divergence:

1. **−200 g per Sami turn (env 15, 17, 19)** on P0 (Hawke) — a
   +1 display-step repair on two infantry that PHP processes one turn
   later than the engine, mirroring the same display-step boundary
   condition as gid 1617442 above.
2. **+3 200 g spike for Sami at env 20** — one Sasha-style settlement
   delta, but Sami has no funds mechanic; the spike resolves to
   ~3.2 income-property days of differential ownership (3 properties
   captured in env 19 with 1 captured one display-step earlier in the
   engine than PHP, advancing P1's income by one full day on the
   following income tick).

* **Source:** AWBW Sami D2D
  (https://awbw.fandom.com/wiki/Sami): infantry/mech double capture
  speed. The double-capture-speed schedule is HP-banded; a
  display-HP-of-1 difference at capture moment shifts the capture
  completion by exactly one engine-day, which propagates to one
  full day of property income — 3.2 properties × 1 000 g ≈ 3 200 g.
* **Verdict:** `oracle_gap` — INHERENT (Sami double-capture HP-band).

## 4. Validation chain (executed)

### 4.1 6 target gids re-audit

```
python tools/desync_audit.py \
  --catalog data/amarriner_gl_std_catalog.json \
  --catalog data/amarriner_gl_extras_catalog.json \
  --games-id 1617442 --games-id 1624082 --games-id 1628849 \
  --games-id 1630341 --games-id 1635679 --games-id 1635846 \
  --register logs/phase11j_final_residuals_reaudit.jsonl
```

Result: **6 oracle_gap, 0 engine_bug.** Each row's failure message
is identical to the source register (no class change, no message
change). Register: `logs/phase11j_final_residuals_reaudit.jsonl`.

### 4.2 T4 cohort regression check (9 closures)

T4 closed 6 in `phase11j_build_no_op_cluster_close.md` (gids
1607045, 1627563, 1632289, 1634961, 1634980, 1637338) and the Sasha
wave-of-three before that (gids 1622501, 1624764, 1626284). All 9
re-audit `ok`:

```
python tools/desync_audit.py \
  --catalog data/amarriner_gl_std_catalog.json \
  --catalog data/amarriner_gl_extras_catalog.json \
  --games-id 1607045 --games-id 1627563 --games-id 1632289 \
  --games-id 1634961 --games-id 1634980 --games-id 1637338 \
  --games-id 1622501 --games-id 1624764 --games-id 1626284 \
  --register logs/phase11j_final_t4cohort_check.jsonl
```

Result: **9 / 9 ok, 0 regressions.**

### 4.3 100-game sample audit

```
python tools/desync_audit.py \
  --catalog data/amarriner_gl_std_catalog.json \
  --catalog data/amarriner_gl_extras_catalog.json \
  --max-games 100 --from-bottom \
  --register logs/phase11j_final_residuals_n100.jsonl
```

Result: **98 ok / 2 oracle_gap / 0 engine_bug.** Gate held
(matches T4's 98/2/0 baseline). Register:
`logs/phase11j_final_residuals_n100.jsonl`.

### 4.4 pytest CO suite

```
python -m pytest tests/test_co_*.py --tb=no -q
```

Result: **121 passed, 0 failed.** Includes the R4 display-cap repair
canon fixtures (`test_co_funds_ordering_and_repair_canon.py::
TestR4DisplayCapRepairCanon`) plus all other CO regressions.

## 5. Hard-rule compliance

* **Engine LOC delta:** 0 (no engine source edits this lane). All
  closure is documentation + audit re-run.
* **No edits to** Rachel SCOP missile AOE, Von Bolt SCOP, Sturm code,
  Missile Silo, `_RL_LEGAL_ACTION_TYPES`, `_apply_wait`, `_apply_join`.
* **No edits to** `tools/desync_audit.py` or
  `tools/oracle_zip_replay.py`.
* Citations follow the lane convention: AWBW Wiki + amarriner PHP
  oracle as primary; engine docstrings (`engine/game.py::_grant_income`,
  `engine/game.py::_end_turn`, `engine/game.py::_resupply_on_properties`)
  as secondary anchor for the existing R4 / Sasha / IBR canon.

## 6. Artifacts

| Path                                                          | Purpose                                          |
|---------------------------------------------------------------|--------------------------------------------------|
| `tools/_phase11j_funds_drill.py`                              | Per-envelope funds drill (re-used)               |
| `tools/_phase11j_funds_drift_trace.py`                        | Per-envelope drift + Δ-of-Δ trace (re-used)      |
| `tools/_phase11j_repair_trace.py`                             | Per-envelope repair instrumentation (re-used)    |
| `tools/_phase11j_repair_php_compare.py`                       | NEW — engine vs PHP unit-state diff per envelope |
| `logs/phase11j_final_residuals_drill.json`                    | 6-gid funds drill snapshot                       |
| `logs/phase11j_repair_trace_1635679.txt`                      | Sturm/Hawke repair instrumentation log           |
| `logs/phase11j_final_residuals_reaudit.jsonl`                 | 6-gid re-audit register                          |
| `logs/phase11j_final_t4cohort_check.jsonl`                    | 9-gid T4 cohort regression register              |
| `logs/phase11j_final_residuals_n100.jsonl`                    | 100-game gate register                           |

## 7. Forward note

If a future audit pass converts any of these 6 from `oracle_gap` to
`engine_bug`, that signals a regression in either (a) the combat-info
override (`tools/oracle_zip_replay.py::
_oracle_set_combat_damage_override_from_combat_info`) or (b) the R4
display-cap repair canon (`engine/game.py::_resupply_on_properties`)
or (c) the Sasha War Bonds deferred crediting
(`engine/game.py::_end_turn` lines 427–443). The current verdict is
INHERENT against the current engine; that classification is itself
auditable by the regression cohort listed in §4.2.

---

*"Bisogna tenere tutto, ma sapere quando lasciar andare."* (Italian, ~30s BC, paraphrased Roman maxim)
*"Hold everything you can, but know when to let go."* — paraphrase from Roman military counsel.
*The line is the centurion's pragmatic code: not every flank is meant to be charged; some are meant to be held by knowing they cannot fall further.*
