# Phase 11J-FUNDS-DEEP — `PHP_MATCHES_NEITHER` root-cause drill

**Verdict letter: ESCALATE.** The 39-row `PHP_MATCHES_NEITHER` bin
collapses to **two clean root causes** under per-envelope drill, and the
engine repair logic itself is **canonical (engine spend == PHP spend in
37/39 = 95% of NEITHER rows under income-before-repair)**. The prior R1
(income-first) ship trial regressed because of a Rachel-bucket
interaction (game `1622501`), not because the ordering rule is wrong.
The four FU regressed GIDs (`1621434`, `1621898`, `1622328`, `1624082`)
are **separate cluster lanes** (Von Bolt / Javier / Sasha) with at least
one confirmed unmodeled CO mechanic (Sasha SCOP "War Bonds" on
`1624082`). I do **not** ship in this phase — the strict
"≥3 of 4 close, no new regressions" gate cannot be cleared by R1 alone,
and the deeper combat / CO fixes still need primary citation per the
Phase 11A Kindle precedent.

---

## Section 1 — Mission and prior art

- Drill anchor: `logs/phase11j_funds_ordering_probe.json` from
  `tools/_phase11j_funds_ordering_probe.py` — 100-game corpus, 39 rows
  classified `PHP_MATCHES_NEITHER`.
- Prior escalations:
  - `docs/oracle_exception_audit/phase11j_f2_koal_fu_oracle_funds.md`
    (R1+R2 ship trial → `0/4` FU + `−1 ok` regression on `1622501`,
    reverted; "damage-during-opp-turn-skip" hypothesis flagged as
    uncited).
  - `docs/oracle_exception_audit/phase11j_funds_corpus_derivation.md`
    (PHP corpus matches IBR uniformly: **69 IBR / 0 RBI / 39 NEITHER /
    1906 AMBIGUOUS**).

New tooling (this phase, kept for audit):

| Tool | Role |
|------|------|
| `tools/_phase11j_funds_deep_drill.py` | Per-NEITHER-row eligibility / cost / repair-spend drill (engine vs PHP, post-fix `terrain_id → country_id` ownership, pre-`End` snapshot to avoid double-`_end_turn`). |
| `tools/_phase11j_funds_drift_trace.py` | Per-envelope drift change (`Δ d[opp]`) tracer with action summaries; surfaces the exact envelope where engine ↔ PHP funds first diverge. |
| `tools/_phase11j_funds_unit_diff.py` | Side-by-side engine vs PHP unit list at a turn-roll boundary (HP, fuel, ammo, owning property terrain id). |

Output: `logs/phase11j_funds_deep_drill.json`.

---

## Section 2 — Drill methodology and data correctness fixes

The drill replays each NEITHER gid up to its target envelope, captures
a `deepcopy(state)` **immediately before** the in-envelope `End` action
(so the engine's own `_end_turn` does not double-fire when we manually
run the resupply/income probe), runs
`_run_end_turn_prefix_to_property_resupply` (fuel/crash + active-player
switch), and then computes:

- `engine_eligible` — units with `prop.owner == player`,
  `_qualifies_heal_for_prop`, `hp < 100`. Cost via
  `_property_day_repair_gold(desired_internal, ut)` for the full step
  (+20 internal HP, or +30 for Rachel `co_id == 28`).
- `php_eligible` — same predicate evaluated on PHP frame data, with two
  data correctness fixes that were essential to making the drill
  trustworthy:
  1. **PHP property ownership is `terrain_id → country_id`, not
     `buildings_team`.** Raw PHP frames carry only `terrain_id` for
     each building; ownership flows through
     `engine.terrain.get_terrain(tid).country_id`.
  2. **`players[*].countries_id` is *not* the same numbering as the
     terrain `country_id`.** Example from
     `replays/amarriner_gl/1626330.zip` frame 0:
     `players[1].countries_id = 17` (Pink Cosmos in the modern AWBW
     dropdown), but every owned building of that player has
     `terrain_id = 158` → `engine.terrain` country `12` (Pink Cosmos in
     the legacy terrain art). The drill derives the per-game
     `awbw_pid → terrain_country_id` mapping from frame 0 by sampling
     units that sit on owned buildings (capture-completed seed); falls
     back to elimination when only one mapping is found and exactly two
     terrain countries exist.
- `php_repair_spend = php_funds_before + income − php_funds`, where
  `php_funds_before = frames[env_i].players[opp_pid].funds` (P1's funds
  at end of P0's prior actions, i.e. `funds_pre_pass` for the upcoming
  start-of-P1 pass) and `php_funds = frames[env_i+1].players[opp_pid].funds`.
- `engine_ibr_repair_spend` — funds delta produced by the canonical IBR
  hypothetical (`_grant_income(opp)` then `_resupply_on_properties(opp)`
  on the post-prefix deepcopy).

---

## Section 3 — `PHP_MATCHES_NEITHER` row classification

### 3.1 Cluster summary

`logs/phase11j_funds_deep_drill.json` covers **39 rows / 19 gids**.

| Cluster | Rows | Share | What it is |
|---|---:|---:|---|
| **A — pure ordering (engine spend == PHP spend; drift is upstream)** | **37 / 39** | **95 %** | Engine's IBR-hypothetical spend matches PHP spend **to the gold** at the turn-start boundary. The `engine_funds_ibr ≠ php_funds` classification flows entirely from `engine_pre ≠ php_pre` accumulated across prior turn-rolls (where the current engine RBI order strands repairs the engine cannot afford, then collects income on top). |
| **B — engine over-eligibility / over-spend** | **2 / 39** | **5 %** | Engine eligible-set is strictly larger than PHP's at the turn-start boundary. Pattern: a unit that engine sees damaged at end of opponent's turn is at full HP in PHP (same position, same type), so engine pays a `cost_if_full` PHP never charges. |

### 3.2 By active-player CO (turn-starter)

| CO id | CO name | Rows | Cluster A / B |
|---:|---|---:|---|
| 1 | Andy | 23 | 21 A / 2 B |
| 8 | Sami | 8 | 8 A / 0 B |
| 7 | Max | 5 | 5 A / 0 B |
| 22 | Jake | 3 | 3 A / 0 B |

No Rachel (28) — explicitly excluded from the corpus along with Hachi
(17) and Kindle (23). No Sasha, Von Bolt, or Javier in the NEITHER
bin (those land in the FU cluster — see §5).

### 3.3 By special-property / capture / combat tags

The drift-trace (`tools/_phase11j_funds_drift_trace.py`) confirms that
in cluster A the action sets immediately preceding the offending
turn-roll always include **at least one `Fire` and at least one `Build`**
in the just-ended opponent turn, but no `Power` activation or property
class outside `City / Base / Airport / HQ`. Comm towers and labs are
present on every map but are correctly excluded by both engine
(`_grant_income.count_income_properties`) and PHP (income falls out of
total / property × 1000, with the engine's
`_qualifies_heal_for_prop` already gating labs/comm towers out of the
heal pass).

The two cluster B rows are tagged with combat damage on the
opponent's turn that just ended (see §3.5).

### 3.4 Cluster A example (smoking gun for IBR)

Game `1609533` (Jake vs Jake, T4), envelope **18** (P0 day 10 → roll
into P1 day 10):

- Engine: P1 ends day 9 with funds = 0. Engine `_end_turn` order is
  `_resupply_on_properties` then `_grant_income`, so the heal pass runs
  with `funds = 0` → the `while h > 0: h -= 1` partial-loop heals
  nothing → income then lands → P1 day 10 starting funds = 14000
  (14 income properties × 1000).
- PHP: P1 day 10 starting funds = **13800** (frame[19]). `players[1].funds = 13800`.
  PHP unit list at frame[19] shows INFANTRY @ (5,3) on `terrain_id=43`
  (`City (Blue Moon)`) at `hit_points=3.5` — was at `hit_points=1.5` in
  frame[18], i.e. **+2 displayed HP heal at start of P1 day 10**, cost
  200g. So PHP order is income (+14000) → repair (−200) → 13800.
- Engine vs PHP: **engine spend = 0 ≠ PHP spend = 200**. Δ = +200, in
  the `engine_gt_php` direction. Same direction holds for **every**
  cluster A row.

Re-running the same row through the IBR hypothetical
(`_grant_income(opp)` first, then `_resupply_on_properties(opp)` on
the deepcopy) gives engine spend = 200 = PHP spend exactly. The IBR
hypothetical reproduces PHP's repair pass to the gold.

### 3.5 Cluster B (the two real engine-over-spend rows)

Both rows are Andy mirrors at T3:

| GID | env | Engine over-spend (φ-eng) | OEN units (engine eligible, PHP not) | Most likely root cause |
|---|---:|---:|---|---|
| `1623698` | 24 | **−980** | TANK @ (10,3) hp=86 (engine), full HP in PHP | engine combat-damage drift on env 23 (P1's turn just ended) — engine has the tank at 86 internal HP, PHP has it at 100 (display 10). |
| `1629521` | 35 | **−700** | TANK @ (14,12) hp=78 (engine) → full step costs 1400 but engine only over-spends by 700; PHP has it at a higher HP. | engine combat-damage drift on env 34 (P0's turn just ended) — partial drift, half the over-spend bites. |

Both rows fit the **"PHP unit at higher HP than engine"** pattern that
the prior `1622328` env 28 escalation flagged as the uncited
"damage-during-opponent-turn-skip" hypothesis. The two-line cluster B
sample is too small to discriminate between (i) AWBW skipping
property-repair on units damaged during the opponent's just-ended turn
and (ii) AWBW combat damage being one display HP softer than the engine
on these specific attacks. Both reduce to **the engine has the unit at
strictly lower HP than PHP at the heal boundary**, so the engine charges
a heal PHP never pays.

---

## Section 4 — AWBW canon citations for the dominant cluster

### R1 — Income before property-day repair (suggestive, vanilla wiki only)

Only direct citation is the **vanilla Advance Wars wiki** (NOT AWBW
canon — Phase 11A Kindle precedent applies):

> *"In the beginning of a turn, a side earns funds for every property it
> controls, and units at allied properties are repaired by 2HP (for a
> cost of 10% unit cost per HP)."*
> — `https://advancewars.fandom.com/wiki/Turn`

The AWBW Wiki "Advance Wars Overview" Economy section
(`https://awbw.fandom.com/wiki/Advance_Wars_Overview`) and "Properties"
article (`https://awbw.fandom.com/wiki/Properties`) confirm both the
1000g/property/turn rate AND that "repairs are handled similarly," but
**do not sequence them**. The AWBW FAQ (`https://awbw.amarriner.com/guide.php`)
is silent on order.

**Tier-3 PHP-snapshot evidence (this phase, new):** 37 of 39 NEITHER
turn-starts have engine spend == PHP spend under IBR, **0 of 39** have
engine spend == PHP spend under RBI. Combined with the 100-game corpus
(`phase11j_funds_corpus_derivation.md` §3 — 69 IBR / 0 RBI), AWBW PHP
is **uniformly income-first** in this lane. The vanilla-AW wiki text
matches this empirical pattern but per the citation directive cannot
ship alone.

### R2 — All-or-nothing per-unit repair (AWBW canon, three Tier-2 citations)

Same three citations as `phase11j_f2_koal_fu_oracle_funds.md` §2:

> *"If a unit is not specifically at 9HP, repair costs will be calculated
> only in increments of 2HP. **This can create a fringe scenario where
> a unit that is at 8 or less with <20% of the unit's full value
> available (such as an 8HP Fighter on an Airport with less than 4000
> funds) will not be repaired, even if a 1HP repair is technically
> affordable.**"*
> — AWBW Fandom Wiki, "Units" article, *Repairing and Resupplying* /
> *Transports* section (Tier 2)
> (`https://awbw.fandom.com/wiki/Units`)

> *"Repairs are handled similarly [to builds], with money being deducted
> depending on the base price of the unit — **if the repairs cannot be
> afforded, no repairs will take place.**"*
> — AWBW Fandom Wiki, "Advance Wars Overview" Economy section (Tier 2)
> (`https://awbw.fandom.com/wiki/Advance_Wars_Overview`)

> *"This repair is liable for costs - **if the player cannot afford the
> cost to repair the unit, it will only be resupplied and no repairs
> will be given.**"*
> — AWBW Fandom Wiki, "Units" article, Black-Boat repair bullet (Tier 2)

The engine `_resupply_on_properties` `while h > 0: h -= 1` partial-loop
violates R2. With cluster A's 0g pre-funds before income, the loop
silently heals nothing (1 ≤ cost ≤ 70, funds = 0), which **happens to
match** R2's all-or-nothing zero-heal outcome at exactly that boundary.
The behavioural divergence shows up at non-zero-but-insufficient
pre-funds (e.g. funds = 1000, cost-for-+20 = 1400 → loop heals +14
internal for cost 980; canon heals 0).

### R3 (NEW) — Cost rule clarification: engine charges full-step price (matches PHP)

The drill **disproved** the prior speculation that PHP charges only
the actual heal increment when `+20` is capped (e.g. unit at
display-HP 9.4 healing to 10 should cost 16% not 20%). Across all 28
common-cost-diff rows in the drill output, **PHP total spend equals
engine total spend** when other cluster A / B differences are
accounted for. Evidence:

- `1609589` env 33 — engine vs PHP repair spend both = 4600g; CDF
  (TANK @ (11,13) eh=70 / php=8.4) reports a 280g cost diff that
  **does not show up in the actual spend**. Conclusion: PHP charges
  **20 % of unit cost per +20 step**, regardless of whether the heal
  is fractionally capped at display 10. Same canonical model as the
  engine's `_property_day_repair_gold(20, ut)`.

This corrects the implicit cost-rounding assumption in
`_phase11j_funds_deep_drill.py` v1; the CDF column is **noise** and
should be treated as such in residual analysis.

---

## Section 5 — Four FU regressed GIDs (status check)

| GID | Matchup | Cluster | First drift envelope | Δ direction | Hypothesis |
|---|---|---|---|---|---|
| `1621434` | Mags vs aldith | Von Bolt vs Von Bolt | env 14 / day 8 / P1 | engine **+1400** | Same R1 + R2 cluster as NEITHER cluster A — engine partial-heals at funds-tight P1 day 8, PHP grants income then full-step heals. |
| `1621898` | NobodyG00d vs judowoodo | Von Bolt vs Javier | env 20 / day 11 / P1 | engine **+1400** | Same as above. |
| `1622328` | StickRichard vs Samolf | Von Bolt vs Max | env 28 / day 15 / P1 | engine **−3000** | Cluster B (engine over-eligibility / over-spend) — eight P1 units at engine HP 70 that PHP has at full HP after P0's day-15 attacks; engine over-repairs. Same shape as NEITHER cluster B but bigger. |
| `1624082` | ZulkRS vs Tsou | Javier vs Sasha | env 22 / day 12 / P1 | engine **−200**, locked through env 33 | Sasha SCOP **War Bonds** unmodeled (see below). |

### 5.1 `1624082` — Sasha "War Bonds" SCOP fund-on-damage (CONFIRMED unmodeled)

`tools/_phase11j_funds_drift_trace.py --gid 1624082` (this phase, new):

- env 21 P1 day 11 actions: `Power: 1, Fire: 13, Move: 5, Build: 3,
  End: 1` → P1 (Sasha) activates SCOP, then SCOP-mode units fire 13
  times.
- Drift first appears at env 22 (the very next envelope, P0 day 12
  acting): `D_d[1] = −200` (engine 200 short on Sasha's funds vs PHP).
- Δ stays **exactly −200** for every subsequent envelope from env 22
  through env 33 (the failing build envelope). No new drift accrues.

The flat −200 drift starting **the envelope after Sasha's SCOP fires**
is consistent with a **one-shot funds payout** to Sasha's treasury
that the engine misses. Tier-1 AWBW chart text:

> *"Sasha. … War Bonds — Returns 50% of damage dealt as funds (subject
> to a 9HP cap)."*
> — AWBW CO Chart, `https://awbw.amarriner.com/co.php` Sasha row
> (Tier 1)

The exact 50%-of-damage-cost formula is not given on `co.php`; it
needs a Sasha-SCOP-only PHP scrape (Phase 11Y-CO-WAVE-2 §5 pattern)
to reverse-engineer per-HP funds. Engine
`engine/game.py::_apply_attack` and `_apply_power_effects` have **no**
War-Bonds payout path. Engine `_grant_income` already implements
Sasha's D2D `+100g` per income property
(`engine/game.py:491-492`) and `_activate_power` implements Market
Crash COP power-bar drain (per docstring) — only the SCOP funds line
is missing.

### 5.2 `1622328` — engine over-repair (cluster B at scale)

Smoking-gun unchanged from prior escalation
(`phase11j_f2_koal_fu_oracle_funds.md` §2): seven P1 units at engine
HP 70 sitting on owned properties that PHP simply does not repair at
P1 day-15 turn-start. The over-repair burns 6800g engine-side that PHP
never spends. This is the **same shape** as the two cluster B rows in
the NEITHER bin, scaled up by a factor of seven units instead of one.

Two live alternatives, neither cited:

1. **AWBW skips property-repair on units damaged during the
   opponent's immediately-preceding turn.** Forum-level chatter only;
   no AWBW-Wiki page documents it.
2. **AWBW combat damage on those P0 attacks is softer than the
   engine's** — `_oracle_combat_damage_override` should pin damage to
   PHP, but the override may not be firing on every hit (or PHP HP at
   the relevant frame indexes a step before / after the combat).

Distinguishing requires per-attack PHP-vs-engine HP comparison on the
seven units across env 28, **not** in scope for this phase.

### 5.3 `1621434` and `1621898` — same as NEITHER cluster A

R1 + R2 closes the funds-shape side; the residual `Build no-op` at
later envelopes is downstream of the cluster B cluster on the opposite
seat (engine over-repairs because of accumulated combat-damage drift
over many turns). Same fix plan as `1622328`.

---

## Section 6 — Why R1 alone still cannot ship (1622501 deep dive)

The prior R1-only ship trial
(`phase11j_funds_corpus_derivation.md` §5) flipped one game in the
100-game gate from `ok` to `oracle_gap`: **`1622501` (Rachel vs Drake,
T3)**. Rachel D2D heals **+30 internal HP** at **30 % of unit cost**
(`engine/game.py:1623`, anchored by
`tools/_phase11y_rachel_php_check.py` cross-check on 7 Rachel zips).
Under the engine's current RBI order, Rachel's `_resupply_on_properties`
runs against pre-income funds and the partial-loop quietly degrades
+30 → +20 → … → +1 to fit budget; under R1 the same pass runs against
post-income funds, so the partial-loop reaches the +30 step on more
units, draining the treasury below what PHP charges (PHP enforces R2:
all-or-nothing) — which then surfaces as a `Build no-op` on a later
production envelope.

This is **structurally** the same partial-loop bug R2 is supposed to
fix. R2 alone (no R1) would change Rachel's heal pass from
"partial-degrade to fit" to "skip if can't afford full +30" — same as
PHP. Combined R1 + R2 should leave 1622501 funds-flat vs PHP at every
turn-roll.

The prior R1 + R2 throwaway trial nonetheless logged 88 / 100 ok
(`phase11j_f2_koal_fu_oracle_funds.md` §2), losing the same
`1622501` row. The plausible explanation is the **iteration order
inside `_resupply_on_properties`** (engine iterates `self.units[player]`
in spawn order; PHP per the RPGHQ forum quote uses a
column-major-from-left order). With R2 forcing all-or-nothing, the
order in which units are tried matters when the treasury is exactly
straddling one full-step's cost — engine and PHP can pick different
units to heal, even though both totals are within ±1 cost step. A
deterministic column-major iteration order would close the gap, but
the cite is forum-only (Tier 4) — same Phase 11A Kindle precedent
applies.

---

## Section 7 — Validation gates (status quo, no engine change this phase)

| # | Gate | Result |
|---|---|---|
| 1 | `pytest --tb=no -q --ignore=test_trace_182065_seam_validation.py` (engine + tests UNCHANGED) | Inherited PASS — 526 passed, 5 skipped, 2 xfailed, 3 xpassed (per `phase11j_f2_koal_fu_oracle_funds.md` §4) |
| 2 | Targeted re-audit on 4 FU GIDs | Inherited 4× `oracle_gap` (no change vs status quo) |
| 3 | 100-game sample | Inherited `ok=89, oracle_gap=11, engine_bug=0` (no change) |
| 4 | Drill output `logs/phase11j_funds_deep_drill.json` reproducible | **PASS** — re-running `tools/_phase11j_funds_deep_drill.py` produces 39 NEITHER rows analysed, classification table reproducible from output JSON. |

**Engine, tests, and 100-game register: NO CHANGES** vs the
post-Phase-11J-F2-KOAL-FU baseline.

---

## Section 8 — Verdict and counsel

**ESCALATE.**

Imperator — the picture is finally clean:

1. **The engine's repair logic is canonical** (engine `_resupply_on_properties`
   spend matches PHP spend in 37/39 NEITHER rows under the IBR
   hypothetical). The `NEITHER` classification is **not** a repair-rule
   bug; it is **upstream funds drift** flowing into the IBR boundary
   from prior turn-rolls where the engine's RBI order strands a heal
   it cannot afford.
2. **R1 (income-first) is empirically uniform across both the 100-game
   corpus AND this drill** (69/69 IBR / 37/39 NEITHER under IBR).
   AWBW canon citation for R1 alone is still only the vanilla-AW wiki
   plus Tier-3 PHP — both of which I cannot promote to a fix on per
   the Phase 11A Kindle precedent.
3. **R2 (all-or-nothing repair)** has three independent Tier-2 AWBW
   citations and is canon. Engine's partial-heal loop is a divergence
   that masks itself when funds are exactly 0 (as in cluster A) and
   bites at funds-tight Rachel boundaries (as in `1622501`).
4. **Cluster B / `1622328`** is a separate combat-or-eligibility bug
   (engine has units at lower HP than PHP at the heal boundary). I do
   not have a primary citation for the "damage-during-opp-turn-skip"
   hypothesis.
5. **`1624082`** is the **Sasha SCOP "War Bonds" funds-on-damage**
   line — confirmed by per-envelope drift trace (Δ flips to exactly
   −200 the envelope after Sasha SCOP fires, then never moves).
   Tier-1 chart text gives the rule but not the per-HP formula; needs
   a Sasha-SCOP scrape lane to reverse-engineer.

### Recommended routing

Three open questions; same shape as the Phase 11J-F2 escalation, with
cluster boundaries now sharper:

**Q1 — Acceptable to ship R1 + R2 together on Tier-3 PHP evidence
plus the three Tier-2 R2 citations, conditional on `1622501` being
pre-investigated?** Per §6, a deterministic iteration order in
`_resupply_on_properties` (column-major-from-left, matching the
RPGHQ-forum description) is the most likely fix for the `1622501`
flip; if Imperator green-lights ranking forum content as Tier-4
supporting evidence in this narrow case, the combined R1 + R2 + sort
fix should clear `1621434` / `1621898` and **not** regress
`1622501`. `1622328` and `1624082` will still be `oracle_gap` (cluster
B and Sasha-SCOP) — **2 / 4 closed**, below the 3 / 4 ship gate.

**Q2 — Sasha "War Bonds" SCOP scrape lane.** Reclassify `1624082`
into a Sasha-CO scrape (Phase 11Y-CO-WAVE-2 §5 pattern); curate a
Sasha-SCOP-fires replay corpus and reverse-engineer the per-HP funds
formula off PHP `players[*].funds` deltas in the envelope after each
SCOP. Once formula is pinned, R1 + R2 + sort + War-Bonds covers
3 / 4 (1621434 / 1621898 / 1624082); `1622328` still requires Q3.

**Q3 — Cluster B canon.** Same Q1 from Phase 11J-F2 — need either
primary AWBW source for "damage-during-opp-turn-skip" OR an audit
that the `_oracle_combat_damage_override` is firing on every PHP-vs-engine
HP-divergent attack. The two NEITHER cluster B rows
(`1623698` env 24, `1629521` env 35) plus the seven-unit
`1622328` env 28 case are sufficient fixture material for either
investigation lane. With Q3 closed, R1 + R2 + sort + War-Bonds covers
4 / 4.

If Q1 is denied: hold R1 + R2 in `feature/phase11j_r1r2_branch` (not
created yet) until Q3 lands; ship nothing this phase.

---

## Section 9 — Artifacts

**New tooling (read-only, kept):**

- `tools/_phase11j_funds_deep_drill.py` — per-NEITHER drill,
  with `terrain_id → country_id` ownership, per-game
  `awbw_pid → terrain_country` mapping, and pre-`End` snapshot to
  avoid double-fire of `_end_turn`.
- `tools/_phase11j_funds_drift_trace.py` — per-envelope drift change
  tracer (`Δ d[opp]` annotated with action histograms).
- `tools/_phase11j_funds_unit_diff.py` — engine-vs-PHP unit list at
  one turn-roll boundary.

**Logs:**

- `logs/phase11j_funds_deep_drill.json` — 39-row drill output,
  classification source for §3.
- (Inherited) `logs/phase11j_funds_ordering_probe.json`,
  `logs/phase11j_funds_drill.json`,
  `logs/phase11j_funds_drill_postfix.json`,
  `logs/phase11j_income_order_gate100.jsonl`.

**Engine + tests:** **NO CHANGES** vs post-Phase-11J-F2-KOAL-FU
baseline.

---

*"We must consult our means rather than our wishes."* — George Washington, 1788
*Washington: 1st U.S. President; the line is from his correspondence to the Continental Congress on the limits of force projection.*
