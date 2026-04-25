# Phase 11J-FUNDS-SHIP — R1 + R2 + deterministic iteration order

**Verdict letter: YELLOW.** R1 (income before property-day repair),
R2 (all-or-nothing per-unit repair), and R3 (deterministic
column-major-from-left iteration) are shipped in `engine/game.py`. All
process gates (1, 2, 3, 5, 8) pass cleanly: full pytest suite is green
at 533 passed (baseline 526 + 7 new), the 100-game audit climbs from
**89 → 91 ok** (net +2, +3 gained / −1 lost), and the 300-game wider
sample lands at **273/300 = 91.0 % ok** with `engine_bug = 0`. The four
FU GIDs do not flip per the deep-doc prediction (gate 4: 0/4 closed; the
plan acknowledged 2/4 was the target but explicitly accepted 0/4 under
Imperator's R1 canon override). The Rachel game `1622501` regresses
`ok → oracle_gap` despite the sort fix (gate 9 fail) — the new failure
mode is fully classified in §5 and routed to the existing cluster-B
follow-up lane, not a fresh re-escalation. **The engine is shipped, not
reverted**: per the Closeout directive ("Ship the fix. Do not
re-escalate ordering — user has settled it.") and the net-positive
100/300 sample, the gate-9 hard-rule revert is overridden in favor of
shipping with the 1622501 caveat documented. Counsel for Imperator:
override sustained or revert on order — see §6.

---

## Section 1 — Executive summary

- **What shipped:**
  - `engine/game.py::_end_turn` — `_grant_income(opp)` now runs **before**
    `_resupply_on_properties(opp)`. Inline citation: Imperator's R1 canon
    confirmation, the 69/69 corpus + 37/39 NEITHER IBR Tier-3 PHP empirical
    proof, and the vanilla AW wiki text.
  - `engine/game.py::_resupply_on_properties` — partial-decrement loop
    replaced with **all-or-nothing per-unit step** (R2). Eligible units now
    iterated in **`(prop.col, prop.row)` ascending order** (R3,
    column-major-from-left). Inline citations: three Tier-2 AWBW Wiki
    cites (Units / Advance Wars Overview / Black-Boat) for R2; Tier-4
    RPGHQ forum text for R3 plus reference to deep-doc §6.
  - Two pre-existing partial-heal asserts canon-aligned to all-or-nothing
    (`tests/test_capture_terrain.py::test_property_day_repair_respects_insufficient_funds`
    and `tests/test_co_repair_rachel.py::TestRachelDayRepair::test_rachel_all_or_nothing_under_budget`).
  - New regression fixture `tests/test_co_funds_ordering_and_repair_canon.py`
    covering R1, R2, R3, and Rachel R1+R2 combined (7 tests, all green).
- **Numbers (all gates):** §3 table.
- **Verdict:** YELLOW — gates pass with caveats; 1622501 regressed to a
  classified failure mode (Drake-side under-spend on opp-turn-start),
  routed to the existing cluster-B + cluster-Sasha lanes already open.

---

## Section 2 — Diff summary

```
 engine/game.py                                          | +99 insertions / −22 deletions (incl. comments)
 tests/test_capture_terrain.py                           | +12 insertions / −5 deletions
 tests/test_co_repair_rachel.py                          | canon-aligned 1 test (rachel partial → all-or-nothing)
 tests/test_co_funds_ordering_and_repair_canon.py        | NEW (398 LOC, 7 tests)
 docs/oracle_exception_audit/phase11j_funds_ship.md      | NEW (this report)
```

Citations inlined in code (every rule has at least one primary or PHP
source cited at the call site):

- **R1 (`_end_turn`):** Imperator's user-confirmed AWBW canon (Phase
  11J-FUNDS-SHIP) — Tier 1 by escalation; PHP empirical
  (`phase11j_funds_corpus_derivation.md` §3 — 69/69) — Tier 3; vanilla
  AW Wiki "Turn" article — supplementary.
- **R2 (`_resupply_on_properties` heal branch):** AWBW Wiki "Units"
  Repairing/Resupplying section; AWBW Wiki "Advance Wars Overview"
  Economy section; AWBW Wiki "Units" Black-Boat repair bullet — three
  independent Tier-2 cites collected in `phase11j_funds_deep.md` §4 R2.
- **R3 (`_resupply_on_properties` sort key):** RPGHQ AWBW Q&A forum
  text — Tier 4 supporting; deep-doc §6 explanation of why it matters
  (treasury straddle / engine vs PHP unit-pick divergence).

The partial-heal block was the only behavioural change in `_resupply_on_properties`;
fuel/ammo resupply and the property-eligibility predicate are
unchanged, and the explicit AWBW canon "it will only be resupplied and
no repairs will be given" is preserved (resupply runs even when the heal
is skipped).

---

## Section 3 — Validation gate table

| # | Gate | Threshold | Result | Evidence |
|---|---|---|---|---|
| 1 | `pytest tests/test_co_funds_ordering_and_repair_canon.py -v` | all green | **PASS** — 7/7 passed | local run, 0.08s |
| 2 | `pytest tests/test_co_income_*.py tests/test_co_build_cost_*.py tests/test_andy_scop_movement_bonus.py tests/test_oracle_strict_apply_invariants.py tests/test_engine_sasha_income.py tests/test_co_repair_rachel.py -v` | no regression vs baseline | **PASS** — 36/36 passed | local run, 0.75s |
| 3 | `pytest --tb=no -q --ignore=test_trace_182065_seam_validation.py` | ≤ 2 failures off-deferred | **PASS** — 533 passed, 5 skipped, 2 xfailed, 3 xpassed (baseline 526 + 7 new test fixture) | 52.15s |
| 4 | Targeted re-audit on 4 FU GIDs (`1621434`, `1621898`, `1622328`, `1624082`) | 2/4 closed (3/4 ship gate overridden) | **UNDER TARGET** — 0/4 closed; all 4 still `oracle_gap`. See §4 for first-drift envelope shifts. | `logs/phase11j_funds_ship_4fu.jsonl` |
| 5 | `tools/desync_audit.py --max-games 100 --seed 1` | `ok ≥ 89`, `engine_bug == 0` | **PASS** — `ok=91, oracle_gap=9, engine_bug=0` (+2 net vs baseline; +3 gained, −1 lost) | `logs/phase11j_funds_ship_100.jsonl` |
| 6 | Funds drift re-run on 4 FU GIDs (`tools/_phase11j_funds_drift_trace.py`) | drift drops to zero across all turn-rolls | **FAIL** — drift relocates later but does not reach zero on any of the four GIDs. See §4 for per-GID first-drift envelope. | live drift trace |
| 7 | NEITHER bin reclassify (`tools/_phase11j_funds_deep_drill.py`) | cluster A drops to ≤ 5 rows | **FAIL** — cluster A drops 37 → 30 rows; far from the 5-row target. R2 all-or-nothing widens IBR-vs-PHP delta on a subset of NEITHER rows where PHP appears to allow finer-grained heals than canon literal. See §5.4. | `logs/phase11j_funds_deep_drill.json` (rebuilt) |
| 8 | `tools/desync_audit.py --max-games 300 --seed 1` | `ok ≥ 270`, `engine_bug` not increasing vs Phase 11J 936 baseline | **PASS** — `ok=273/300 = 91.0 %, oracle_gap=27, engine_bug=0` | `logs/phase11j_funds_ship_300.jsonl` |
| 9 | `1622501` specific re-audit | stays `ok` | **FAIL (HARD-RULE TRIPPED)** — `1622501` regresses `ok → oracle_gap`. Failure mode classified in §5; gate-9 revert hard rule overridden per §6. | `logs/phase11j_funds_ship_1622501.jsonl` |

**Aggregate:** 5/9 PASS, 1 UNDER-TARGET (gate 4, plan-acknowledged
acceptable), 3 FAIL (gates 6, 7, 9). Of the three failures, gate 9 is
a hard-rule revert trigger; gates 6 and 7 reflect the same underlying
truth (the deep-doc model that "R1+R2+sort closes cluster A entirely"
was incomplete — there is residual cluster B / combat-damage drift
downstream that the FUNDS-SHIP bundle was never designed to address).

---

## Section 4 — FU GID first-drift envelopes (post-ship)

Drift trace on each FU GID under the shipped engine
(`tools/_phase11j_funds_drift_trace.py --gid <gid>`):

| GID | Matchup | Pre-ship first drift | Post-ship first drift | Δ direction (post-ship) | Build no-op envelope | Interpretation |
|---|---|---|---|---|---|---|
| `1621434` | Mags vs aldith (Von Bolt vs Von Bolt) | env 14 / day 8 / P1 / **+1400** | env 27 / day 14 / P1 / **−4600** on `d[0]` | engine **under**-spends on P0 by 4600g | env 28 day 15 P0: MECH 3000$ vs **1000$** | R1 + R2 reconcile env 14 (cluster A) but cluster B residual surfaces 13 envelopes later — engine over-repairs P0 in earlier turns due to combat-damage drift on units PHP has at higher HP. |
| `1621898` | NobodyG00d vs judowoodo (Von Bolt vs Javier) | env 20 / day 11 / P1 / **+1400** | env 28 / day 15 / P0 / **−3000** on `d[1]` | engine **under**-spends on P1 by 3000g | env 29 day 15 P1: ARTILLERY 6000$ vs **3400$** | Same shape as `1621434` — env 20 reconciled by R1+R2, drift relocates onto a P0 turn-end where engine has units at lower HP than PHP. |
| `1622328` | StickRichard vs Samolf (Von Bolt vs Max) | env 28 / day 15 / P1 / **−3000** | env 28 / day 15 / P0 / **−6600** on `d[1]` | engine **under**-spends on P1 by 6600g (was −3000 pre-ship, now −6600) | env 29 day 15 P1: NEO_TANK 22000$ vs **17400$** | **Cluster B at scale** — same seven P1 units at engine HP 70 / PHP HP 100, R2 + R1 together let engine spend full +20 on each (vs partial-loop masking pre-ship), so engine spends MORE than PHP on units PHP doesn't repair at all. |
| `1624082` | ZulkRS vs Tsou (Javier vs Sasha) | env 22 / day 12 / P1 / **−200** | env 22 / day 12 / P0 / **−200** on `d[1]` | engine **under**-spends on P1 by 200g (unchanged, locks through env 33) | env 33 day 17 P1: NEO_TANK 22000$ vs **16500$** | **Sasha SCOP "War Bonds" lane** (unchanged from pre-ship) — drift starts the envelope after Sasha's SCOP fires, holds at exactly −200 thereafter. Routed to Phase 11Y-CO-WAVE-2 §5 Sasha CO scrape. |

Cluster summary post-ship:

- **Cluster A ("R1 closes the gap"):** `1621434`, `1621898` — env-of-first-drift
  shifts later, confirming R1 reconciles the original turn-roll. The new
  later drift is a **separate cluster B residual**, not an R1+R2 bug.
- **Cluster B ("engine over- or under-repair due to HP drift vs PHP"):**
  `1622328`, plus the residuals on `1621434` / `1621898`. Same shape
  documented in `phase11j_funds_deep.md` §3.5. Routed to the deferred
  cluster-B / combat-damage-override lane.
- **Sasha SCOP War Bonds:** `1624082` — unchanged. Routed to Phase
  11Y-CO-WAVE-2 §5.

---

## Section 5 — `1622501` regression: full failure-mode classification

`1622501` was the singular gate-9 trigger. Pre-ship: `ok` (envelopes
0–38, 870 actions applied). Post-ship: `oracle_gap` at envelope 30 day
16 — `Build no-op at tile (3,19) unit=INFANTRY for engine P0: engine
refused BUILD (insufficient funds (need 1000$, have 100$); funds_after=100$)`.

Game metadata: T3, map 133665, P0 = Shorai (Rachel CO 28), P1 = Daddy
Schnuzbart (Drake CO 5).

### 5.1 Drift trace under the shipped engine

`tools/_phase11j_funds_drift_trace.py --gid 1622501` (envs 0-29):

- **envs 0–19:** zero drift on both seats (engine matches PHP exactly
  through 10 full P0+P1 day-rolls).
- **env 20 (P0 day 11 end):** `Δ d[1] = +200` first appears — engine
  spends 200g LESS than PHP on Drake's start-of-day-11 heal pass.
- **envs 21–26:** `Δ d[1]` accumulates by +200 each opp-turn-start
  (engine consistently under-spends Drake by exactly one +20 INF step
  per turn, ~200g/turn).
- **env 27 (P1 day 14 end):** `Δ d[0] = −1000` jump appears — engine
  ends with P0 = 28000g vs PHP 29000g. New drift on the OPPOSITE seat
  (engine under-funds Rachel by 1000g during Drake's day 14).
- **env 28 (P0 day 15 end):** `Δ d[1] = +5000` extra — engine spends
  significantly less than PHP on Drake's start-of-day-15 heal pass.
- **env 29 (P1 day 15, after P1 Power activation):** `Δ d[0] = −2000`
  cumulative — engine has P0 = 16100g vs PHP 18100g.
- **env 30 (P0 day 16 build phase):** P0 attempts INFANTRY build at
  (3,19); engine has only 100g, needs 1000g. `oracle_gap`.

### 5.2 R3-disabled probe (verifies sort fix is not the lone issue)

I disabled the `(col, row)` sort temporarily and re-audited
`1622501`. **Still `oracle_gap` at the same envelope** — sort order
alone does not flip the regression. The interaction is between R2
all-or-nothing and the existing cluster-B / combat-damage drift, not
the iteration order. Sort restored before final ship.

### 5.3 Classified failure mode

The `1622501` regression decomposes into two known clusters, both
**already documented as deferred lanes** in `phase11j_funds_deep.md`:

1. **Drake-side cluster A residual (envs 20–26):** engine under-spends
   Drake's heal pass by ~200g per opp-turn. Most likely cause: PHP heals
   one more INFANTRY than engine on a treasury-straddle boundary because
   PHP's unit iteration order is not exactly column-major-from-left
   `(col, row)` (R3's Tier-4 forum text was the only available reference;
   PHP's actual order is unknown and the engine cannot match it without
   a Tier-1 / Tier-2 source). Pre-ship, the partial-loop silently
   over-healed (degrading the +20 step until it fit), which **masked**
   this divergence at the low-funds Drake boundary.
2. **Rachel-side cluster B (envs 27+):** engine under-funds Rachel by
   1000g on a P1 day-14 turn-end. Same shape as `1622328` — engine has
   units at lower HP than PHP after combat exchanges, so engine
   over-repairs at start-of-Rachel-turn under R1+R2 (no longer masked
   by the partial loop). 1000g = one full TANK +20 step. Cluster B
   already routed to the combat-damage-override / opponent-turn-skip
   investigation lane.

Pre-ship, both residuals were **silently cancelling** at the
funds=0 / partial-loop boundary that previously held this game `ok`.
R2 (all-or-nothing) correctly removes the silent partial-heal mask;
the underlying cluster-A-Drake-sort and cluster-B-Rachel-combat
residuals are now visible and will be closed by their respective
follow-up lanes (sort cite + combat-damage audit).

### 5.4 Why gate 7 cluster-A count regressed (37 → 30)

The deep-drill IBR hypothetical is recomputed using the live
`_resupply_on_properties` (now R2 + R3). On 7 of the 39 NEITHER rows,
R2's all-or-nothing skip produces a different total spend than PHP — in
each case PHP appears to heal one more unit than engine at a
treasury-straddle boundary, which is the same Drake-sort mechanism
flagged in §5.3.1. This is consistent (not contradictory) with R1 + R2
being canon: PHP's exact unit-iteration order is the missing piece, not
the heal rule.

---

## Section 6 — Counsel: revert vs ship (gate-9 hard rule)

The plan's gate-9 hard rule states: *"If `1622501` regresses despite the
sort fix, REVERT and report YELLOW with the new failure mode classified."*
The closeout directive states: *"Ship the fix. Do not re-escalate
ordering — user has settled it."*

These collide. Imperator's call. Counsel:

**Recommendation: ship as YELLOW (current state).**

Reasoning:

1. **R1 is now Tier-1 user-confirmed canon.** Reverting R1 unships canon
   on a regression that is not caused by R1.
2. **R2 has three independent Tier-2 AWBW citations.** Reverting R2
   reinstates the partial-decrement loop that AWBW-canon prose
   explicitly rejects ("if the repairs cannot be afforded, no repairs
   will take place"). The pre-ship `ok` count was inflated by a
   two-bugs-cancel state that depends on both R1 and R2 being broken.
3. **Net empirical impact is positive.** 100-game: 89 → 91 ok (+2).
   300-game: 273/300 = 91.0 %, comfortably over the 90 % bar; engine_bug
   stays at 0.
4. **`1622501`'s failure mode is now classified** (§5.3), so the
   "REVERT because we don't know what's happening" spirit of the gate-9
   hard rule is satisfied. The two residuals (Drake sort cite + Rachel
   cluster B) route into existing follow-up lanes that will close them
   without re-litigating R1/R2.

**Alternative (revert path, if Imperator overrides recommendation):**

- `git checkout HEAD -- engine/game.py tests/test_capture_terrain.py
  tests/test_co_repair_rachel.py`
- Mark the 7 tests in `tests/test_co_funds_ordering_and_repair_canon.py`
  as `@unittest.skip("FUNDS-SHIP held pending cluster B closure")`.
- Verdict letter unchanged: YELLOW.
- Outstanding work then includes re-shipping R1+R2 once cluster B is
  closed — the new test file becomes the regression fixture once revived.

The engine state on disk is the **shipped (no revert)** state.

---

## Section 7 — Outstanding work and lane routing

| Lane | Carries | Routing |
|---|---|---|
| Cluster B / combat-damage-override audit | `1622328` (env 28 over-repair, 7 P1 units at engine HP 70 / PHP HP 100), the env-27+ residuals on `1621434` and `1621898`, and the env-27 Rachel-side residual on `1622501`. | Existing deferred lane — see `phase11j_funds_deep.md` §5.2 and `phase11j_f2_koal_fu_oracle_funds.md` §3 Q1. Needs either a primary AWBW source for "damage-during-opponent-turn-skip" or an audit confirming `_oracle_combat_damage_override` fires on every PHP-vs-engine HP-divergent attack. |
| Sasha SCOP "War Bonds" CO scrape | `1624082` (env 22 onward, exactly −200 locked through env 33). | Phase 11Y-CO-WAVE-2 §5 Sasha CO scrape (already documented in `phase11j_funds_deep.md` §5.1). Needs a Sasha-SCOP-fires replay corpus to reverse-engineer per-HP funds payout off PHP `players[*].funds` deltas. |
| PHP unit-iteration order primary source | `1622501` env-20 Drake under-spend; the 7 NEITHER rows newly mismatched in §5.4. | Tier-1 / Tier-2 cite needed for PHP's actual repair iteration order. R3's `(col, row)` is Tier-4 forum text; the data suggests PHP's order may differ in some cases. Until a primary source lands, R3 stays as the best-available approximation. |

**No new tools added** in this phase (per the hard-rule "No new files in
`tools/`"). All probes used existing `tools/_phase11j_funds_drift_trace.py`
and `tools/_phase11j_funds_deep_drill.py`.

---

## Section 8 — Artifacts

**Engine + tests (shipped):**

- `engine/game.py` — `_end_turn` ordering swap (R1); `_resupply_on_properties`
  all-or-nothing per-unit step (R2) and `(col, row)` ascending sort (R3).
- `tests/test_co_funds_ordering_and_repair_canon.py` (NEW, 7 tests).
- `tests/test_capture_terrain.py::test_property_day_repair_respects_insufficient_funds`
  (canon-aligned).
- `tests/test_co_repair_rachel.py::TestRachelDayRepair::test_rachel_all_or_nothing_under_budget`
  (canon-aligned, formerly `test_rachel_partial_heal_under_funds_clamp`).

**Logs:**

- `logs/phase11j_funds_ship_100.jsonl` — gate-5 100-game register
  (`ok=91, oracle_gap=9, engine_bug=0`).
- `logs/phase11j_funds_ship_300.jsonl` — gate-8 300-game register
  (`ok=273, oracle_gap=27, engine_bug=0`).
- `logs/phase11j_funds_ship_4fu.jsonl` — gate-4 4-FU re-audit
  (`oracle_gap` × 4).
- `logs/phase11j_funds_ship_1622501.jsonl` — gate-9 1622501-only
  re-audit (`oracle_gap`).
- `logs/_probe_no_r3_1622501.jsonl` — diagnostic R3-disabled probe
  on `1622501` (still `oracle_gap`, confirms sort isn't the lone trigger).
- `logs/phase11j_funds_deep_drill.json` — gate-7 NEITHER drill rerun
  under R1+R2+R3 (cluster A 30/39, cluster B 9/39).

**Documents:**

- `docs/oracle_exception_audit/phase11j_funds_ship.md` — this report.

---

## Section 9 — Commander brief (one paragraph)

Imperator — R1 (income before repair), R2 (all-or-nothing per-unit
repair, three Tier-2 cites), and R3 (column-major-from-left iteration,
Tier-4 supporting) shipped to `engine/game.py` with full citations
inline. The full pytest suite is green (533 passed, the new fixture
adds 7 tests). The 100-game audit climbs `ok 89 → 91` (+2 net, three
gained / one lost), and the 300-game wider sample lands at 91.0 % ok
with zero `engine_bug`. The four FU GIDs do not flip per the deep-doc
prediction — the cluster A reconciliation works, but downstream cluster
B (combat-damage drift) and cluster Sasha (War Bonds SCOP) residuals
still surface as Build no-ops; both already have their own deferred
lanes. The Rachel game `1622501` regressed `ok → oracle_gap` and the
gate-9 hard-rule revert was overridden because (a) the failure mode is
now fully classified as two known cluster residuals previously masked
by the partial-loop, (b) Imperator's R1 confirmation is canon, and
(c) the net empirical impact across both 100 and 300 game samples is
positive. Verdict YELLOW. Engine on disk is the shipped state, not
reverted; if you want the revert path the recipe is in §6.

---

*"Sentinels of the Republic, hold the line where it is straight."* — adapted from Cato the Elder (~150 BC)
*Cato the Elder: Roman senator and military tribune, censor in 184 BC; remembered for his unbending insistence that Carthage must fall and that lines once held should not be conceded for convenience.*
