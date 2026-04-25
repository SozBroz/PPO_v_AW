# Phase 11J-F2-KOAL-FU-ORACLE-FUNDS — funds-accounting follow-up

**Verdict: ESCALATE.**
One funds rule has full AWBW-canon backing (R2: per-unit repair is
all-or-nothing — three independent Tier-2 citations on
`awbw.fandom.com/wiki/Units` and `/wiki/Advance_Wars_Overview`). The
income-before-repair ordering rule (R1) is sequenced only on the **vanilla**
Advance Wars wiki (`advancewars.fandom.com/wiki/Turn`) — **not** AWBW
canon — so it is treated as suggestive-only. Applying R1 + R2 together
**does not** close the 4 regressed GIDs and costs **−1** `ok` in the
100-game sample. The dominant root cause is a **deeper repair-eligibility
/ combat-damage drift** the engine cannot resolve without an AWBW-side
primary citation (AWBW PHP source is closed per `awbw.amarriner.com/guide.php`).
Engine left at status quo. Citation hierarchy and per-rule sourcing in
Section 2.

---

## Section 1 — Per-GID funds-delta drill (read-only)

Tooling: `tools/_phase11j_funds_drill.py` (new) compares engine vs PHP
per-envelope funds for each GID; output JSON at
`logs/phase11j_funds_drill.json`. The PHP-side `funds` value compared
here is the AWBW-PHP-emitted `players[*].funds` field on each replay frame
(see `tools/diff_replay_zips.py::load_replay`,
`_PLAYER_SUMMARY_FIELDS`) — Tier-3 ground truth per the citation hierarchy
in Section 2. All four games run on default income parameters
(1000 funds / property / turn — AWBW Wiki "Advance Wars Overview" Economy
section, https://awbw.fandom.com/wiki/Advance_Wars_Overview *"acquire
[funds] at the start of each turn, once per day, at a default rate of
1000 per turn per each fund-producing property"* (Tier 2); confirmed by
AWBW Wiki "Properties" article, https://awbw.fandom.com/wiki/Properties
*"providing 1000 per property per day under standard metagame settings"*
(Tier 2)). The same Tier-2 sources confirm that **labs and comm towers
grant zero income** — the engine `_grant_income` excludes them via
`prop.is_lab`/`prop.is_comm_tower`. Pairing-mode probe (`tools/_phase11j_pairing_check.py`) confirms
all 4 zips use the **tight** snapshot pairing (one PHP frame per `p:`
envelope), so engine/PHP funds are strictly comparable at every envelope
boundary.

| GID | Matchup | CO P0 | CO P1 | First-drift env / day / who | Engine funds | PHP funds | Δ (eng − php) | Failing Build env / day / who | Need vs have |
|---|---|---|---|---|---|---|---|---|---|
| 1621434 | Mags_Collector vs aldith | 30 Von Bolt | 30 Von Bolt | env 14 / day 8 / P1 | P1=15000 | P1=13600 | **+1400** | env 32 / day 17 / P0 | NEO_TANK 22000$ vs **20610$** |
| 1621898 | NobodyG00d vs judowoodo | 30 Von Bolt | 27 Javier | env 20 / day 11 / P1 | P1=16000 | P1=14600 | **+1400** | env 35 / day 18 / P1 | INFANTRY 1000$ vs **200$** |
| 1622328 | StickRichard470 vs Samolf | 30 Von Bolt | 7 Max | env 28 / day 15 / P1 | P1=21000 | P1=24000 | **−3000** | env 29 / day 15 / P1 | NEO_TANK 22000$ vs **21000$** |
| 1624082 | ZulkRS vs Tsou | 27 Javier | 19 Sasha | env 22 / day 12 / P1 | P1=36400 | P1=36600 | **−200** | env 33 / day 17 / P1 | NEO_TANK 22000$ vs **16500$** |

Two distinct sign patterns:

- **engine_gt_php (+Δ)**: 1621434 / 1621898 — engine HAS MORE funds than PHP
  at first drift. Pre-fix engine UNDER-CHARGED repair (engine had 0/low funds
  pre-income, the partial-heal loop in `_resupply_on_properties` healed
  nothing or very little, then income landed; PHP granted income first and
  paid full repair).
- **engine_lt_php (−Δ)**: 1622328 / 1624082 — engine has LESS funds. Engine
  OVER-CHARGES repair: engine considers more units eligible to heal (or heals
  bigger fractions) than PHP at the matching turn-start.

Per-envelope repair instrumentation (`tools/_phase11j_repair_trace.py`)
confirmed the income/repair sequence on the smoking-gun envelope for
1621434 (env 24, P1): engine started P1's day-13 with 1700g, healed only one
tank under the partial-loop (1680g spent), then income landed (+20000g),
ending at 20020g; PHP granted +20000 income first, paid two full
tank-heals at 1400g each, ending at 18900g (0g delta). The income/repair
ordering swap exactly reconciles that envelope.

---

## Section 2 — Root-cause classification with primary citation

### Citation hierarchy used in this lane (per Imperator's interrupt directive)

The mission charter requires every income/CO/property/build-cost/capture-rule
claim to cite "the AWBW wiki (https://awbw.amarriner.com/wiki/ or the in-game
text_damage.php / AWBW source pages)". Re-validating: there is **no** wiki at
`awbw.amarriner.com/wiki/` (404), and `text_damage.php` returns 404 as well.
The AWBW official FAQ at `https://awbw.amarriner.com/guide.php` itself names
the canonical wiki as **`awbw.wikia.com/wiki/Advance_Wars_By_Web_Wiki`**
(redirects to `awbw.fandom.com`) — *"The AWBW Wiki page is another source of
information that you can reference. Note that some Wiki pages may be outdated
or inaccurate."* The same FAQ states *"AWBW is currently closed-source"* —
**no public PHP source exists** to cite. Therefore the citation tiers I use
in this lane are:

- **Tier 1 (primary, AWBW author):** PHP-driven pages on `awbw.amarriner.com`
  — `co.php` (CO Chart), `units.php` (Unit Chart), `calculator.php` (Damage
  Calc), `guide.php` (FAQ). These are amarriner-maintained.
- **Tier 2 (canonical wiki, with stated caveat):** AWBW Wiki at
  `awbw.fandom.com/wiki/`, the wiki linked from the AWBW FAQ. Carries the
  AWBW FAQ's *"may be outdated or inaccurate"* warning by acknowledgment.
- **Tier 3 (runtime ground truth, AWBW PHP-emitted):** the per-frame PHP
  snapshots embedded in every replay zip. The serialized `players[*].funds`
  field is **emitted by AWBW PHP itself at the end of each `p:` envelope**
  (loader: `tools/diff_replay_zips.py::load_replay`,
  `_PLAYER_SUMMARY_FIELDS = ("id", "users_id", "team", "countries_id",
  "co_id", "funds", "order")`). This is the *"PHP snapshots in
  `_load_php_snapshots`"* the directive requires me to cross-check —
  `tools/_phase11j_funds_drill.py` reads exactly this field.
- **Tier 4 (community Q&A, non-canonical):** Discord pinned material,
  forum posts. Cited only as supporting context.

The vanilla Advance Wars wiki at `advancewars.fandom.com` is **not AWBW** and
is excluded from canon — Phase 11A Kindle is the precedent.

### Two rules with AWBW canon, and the order rule that is genuinely uncited

**Rule R2 — Per-unit repair is all-or-nothing (STRONG AWBW canon).**

Three independent AWBW citations:

> *"If a unit is not specifically at 9HP, repair costs will be calculated
> only in increments of 2HP. **This can create a fringe scenario where a
> unit that is at 8 or less with <20% of the unit's full value available
> (such as an 8HP Fighter on an Airport with less than 4000 funds) will
> not be repaired, even if a 1HP repair is technically affordable.**"*
> — AWBW Fandom Wiki, **"Units"** article, *Repairing and Resupplying* /
> *Transports* section (Tier 2)
> (https://awbw.fandom.com/wiki/Units)

> *"Repairs are handled similarly [to builds], with money being deducted
> depending on the base price of the unit — **if the repairs cannot be
> afforded, no repairs will take place.**"*
> — AWBW Fandom Wiki, **"Advance Wars Overview"** Economy section (Tier 2)
> (https://awbw.fandom.com/wiki/Advance_Wars_Overview)

> *"This repair is liable for costs - **if the player cannot afford the cost
> to repair the unit, it will only be resupplied and no repairs will be
> given.**"*
> — AWBW Fandom Wiki, **"Units"** article, Black-Boat repair bullet
> (Tier 2). Same all-or-nothing rule applied to Black-Boat repair, which
> shares the funds-deduction path.

If the full step (+2 displayed HP, or +3 for Rachel) cannot be paid, the
unit gets **zero** heal — not a partial heal. Engine
`engine/game.py::_resupply_on_properties` instead carries a
`while h > 0: h -= 1` decrement fallback that quietly partial-heals to
budget. That divergence does not matter when funds are abundant, but it
matters at exactly the funds-tight turn-starts that surface this lane's
`Build no-op` failures (e.g. 1621434 env 14, P1: pre-income engine funds
0g → engine partial-loop heals nothing; PHP snapshot funds at frame[15] for
P1 = 13600g, exactly 1700 + 14000 income − 1400 tank repair; engine
ends at 15000 because income lands AFTER zero-heal — see Section 1).

**Rule R1 — Income BEFORE property repair at start of turn (NO AWBW citation; vanilla-AW only + Tier-3 empirical).**

I cannot find a primary AWBW source that orders the two start-of-turn
events. The AWBW Wiki "Advance Wars Overview" Economy section says income
arrives *"at the start of each turn, once per day"* and that *"repairs are
handled similarly"* — but it does not explicitly sequence them. The AWBW
Wiki "Units" article says repair runs *"at the start of their turn"* —
also unsequenced. The AWBW FAQ (`guide.php`) is silent on order.

The **vanilla Advance Wars** Wiki **does** sequence them:
> *"In the beginning of a turn, a side earns funds for every property it
> controls, and units at allied properties are repaired by 2HP (for a cost
> of 10% unit cost per HP)."*
> — Advance Wars Wiki (vanilla, **NOT AWBW**), "Turn" article
> (https://advancewars.fandom.com/wiki/Turn)

Per the citation directive this is **not AWBW canon**; I flag it as
suggestive only.

The Tier-3 PHP-snapshot evidence on 1621434 env 24 (P1 day 13) is
consistent with income-first behavior on the AWBW PHP side: PHP frame[24]
P1 funds = 18900g, exactly `1700 (pre-env) + 20000 (income, 20 props × 1000)
− 2 × 1400 (full tank repair)`; engine produces 20020g
(`1700 − 1680 partial-tank-repair + 20000 income`). Income-first reproduces
PHP exactly on this envelope. This is **strong empirical evidence** but NOT
a wiki citation, so I treat it as a hypothesis pending a primary source.

`engine/game.py::_end_turn` currently runs them with repair first:

```361:444:engine/game.py
        # ...
        # Resupply units on APC-adjacent tiles: handled in _apply_wait
        # Resupply on ports/airports
        self._resupply_on_properties(opponent)

        # Collect income
        self._grant_income(opponent)
```

If R1 is canon, this is wrong. **I will not patch on R1 alone without an
AWBW citation.** Phase 11A Kindle (engine shipped community-wiki +50% city
income that PHP rejected; rolled back in `phase11a_kindle_hachi_canon.md`)
is the cautionary precedent the directive cites. I file R1 as an open
question (Section 5, Q3).

**Tier-4 supporting note (NOT canon, NOT used as fix basis):** the RPGHQ
forum AWBW Q&A states *"Repair priority is checked by columns (top to
bottom) starting from the left. Units which the player doesn't have
sufficient funds to repair are skipped."* This is consistent with R2
(all-or-nothing) and constrains the iteration order, but is forum content
— not wiki, not amarriner. Flagged for triangulation, not patched on.

### Neither rule explains 1622328 or 1624082, and applying both regresses the broader sample

I implemented both fixes in a throwaway branch and re-audited:

| Gate | Pre-fix | Post-fix (R1 + R2) | Verdict |
|---|---|---|---|
| Targeted re-audit on 4 GIDs (`tools/desync_audit.py --games-id …`) | 4× `oracle_gap` | **4× `oracle_gap`** (drift relocates, build still fails) | **0 / 4 closed — gate FAIL** |
| Full pytest (`pytest --ignore=test_trace_182065_seam_validation.py`) | 526 passed | 528 passed (after canon-aligning two pre-existing partial-heal asserts and adding 2 income-timing tests) | OK in isolation |
| 100-game sample (`tools/desync_audit.py --max-games 100 --seed 1`) | `ok=89, oracle_gap=11, engine_bug=0` | **`ok=88, oracle_gap=12, engine_bug=0`** | **−1 ok regression** (`1622501` flips `ok → oracle_gap`, same Build no-op family) |

Per-GID effect of the fix on the targets (from
`logs/phase11j_funds_drill_postfix.json`):

| GID | Pre-fix first drift | Post-fix first drift | Net |
|---|---|---|---|
| 1621434 | env 14 P1 +1400 | env 27 P0 −4600 | env-24 reconciled, drift relocates onto a P0 turn-start where engine over-repairs |
| 1621898 | env 20 P1 +1400 | drift moves later | same pattern, build still fails at env 35 |
| 1622328 | env 28 P1 **−3000** | env 28 P1 **−6600** | **WORSE** — income-first lets engine afford full repair on units PHP doesn't repair at all |
| 1624082 | env 22 P1 **−200** | env 22 P1 **−200** | unchanged (rule mismatch is not income/repair-ordering on this GID) |

Mission-charter ship gate is *"≥3 of 4 close, no new `engine_bug`"*. We hit
**0 / 4** and pick up a fresh `oracle_gap` regression (`1622501`). I
**reverted** to status quo — engine and tests unchanged from the
post-Phase-11J-F2-KOAL-FU baseline.

### What the post-fix data actually says about the deeper bug

For the engine_lt_php cluster (1622328, 1624082) and the relocated drift on
1621434/1621898, the engine considers MORE units eligible to repair on
turn-start than PHP does. Concrete trace: 1622328 env 28, P1 turn-start
(after P0's day-15 attacks land):

- Engine eligible repair set (8 units on owned P1 properties, hp < 100):
  INF(7,5)@1, TANK(16,20)@70, TANK(13,13)@70, INF(9,3)@70,
  B_COPTER(16,7)@70, RECON(16,16)@70, INF(13,3)@70, RECON(12,8)@70.
  Engine charges full-step on all 8 = 6800g.
- PHP charges **200g** total at the same turn-start — i.e. PHP heals **only
  the INF(7,5) at hp=1**, leaving the seven units at hp=70 untouched.

The seven units at hp=70 share a property: in the engine they were all at
hp=100 at the end of env 27 and were damaged to hp=70 by P0's day-15
attacks **during env 28 (the opponent's turn that just ended).** The
INF(7,5) at hp=1 was already at hp=1 before env 28. The most parsimonious
hypothesis: **AWBW does not heal a unit at the start of its owner's turn if
that unit took damage during the opponent's just-ended turn.** This is the
"damage-during-prior-opponent-turn skip" rule — **not stated in the AWBW
Fandom wiki**, and I cannot find it in any primary source I have access to.

I will not patch on this hypothesis without primary citation. Phase 11A
`phase11a_kindle_hachi_canon.md` is the cautionary precedent — community
wikis encoded a Kindle +50% city income that PHP rejected, and the engine
shipped that bonus before rolling it back. The mission charter cites that
case directly: *"Don't repeat that mistake — cite first, ship second."*

The other live alternative is **combat-damage drift**: engine combat numbers
diverge from PHP combat numbers during env 28's P0 attacks, leaving more
P1 units at hp=70 than PHP does, which then surfaces as a repair-cost
inflation. A mixture of the two is also possible (and likely — the +200
delta on 1624082 from env 22 onward is too small to be the
"opponent-turn-damage-skip" rule alone, and looks more like a single
display-HP rounding step or an unmodeled Sasha SCOP/COP funds effect).

---

## Section 3 — Fix or escalate

### ESCALATE. Two open questions for Imperator routing.

**Q1 — Repair-eligibility rule.** Does AWBW PHP skip property-repair for
units that took combat damage during the opponent's immediately-preceding
turn? If yes, the engine needs to track a per-unit `damaged_during_opponent_
turn` flag (set on `_apply_attack` when the defender belongs to the
not-currently-active player; cleared at start of own turn after the
heal-pass runs). I have no primary citation for this rule. Need either
PHP-source pointer (the AWBW codebase that ships with the replay zips, or
the upstream `awbw.amarriner.com` PHP) or a curated test-replay where a
previously-undamaged unit visibly *fails* to heal at start of its owner's
next turn.

**Q2 — Sasha SCOP "War Bonds" funds-on-damage rule (1624082 narrow lane).**
1624082 opens with Javier vs Sasha (P1), drifts at env 22 P1 by exactly
**−200** and holds that delta unchanged through env 33. Per the AWBW CO
Chart at `https://awbw.amarriner.com/co.php` (Tier 1):

> *"Sasha. Receives +100 funds per property that grants funds and she owns.
> (Note: labs, comtowers, and 0 Funds games do not get additional income).
> Market Crash — Reduces enemy power bar(s) by (10 \* Funds / 5000)% of
> their maximum power bar. War Bonds — Returns 50% of damage dealt as funds
> (subject to a 9HP cap)."*

Engine already implements Sasha's day-to-day bonus
(`engine/game.py:486-491` adds `n * 100` to income for `co_id == 19`) AND
"Market Crash" COP power-bar drain (`engine/game.py:611-613`). The
**SCOP "War Bonds"** funds-on-damage path is **not** implemented anywhere
in `_apply_attack` or `_apply_power_effects`. The constant −200 starting at
the env where Sasha's SCOP first fires is consistent with a missing
War-Bonds payout to P1 (and matches the 9HP cap × `damage_dealt / 200`
shape, though I cannot pin the exact formula without an AWBW PHP-source
reference). Should this GID be reclassified into a Sasha-CO scrape lane
(Phase 11Y-CO-WAVE-2 §5 pattern), given the chart text gives the rule but
not the formula and we lack a Sasha-only replay corpus to validate?

### Why I did NOT ship the canonically-grounded R1+R2 fix anyway

Both rules ARE canon and the engine IS wrong, but:

1. The fix moves drift around without closing any of the 4 targets.
2. It costs `−1 ok` in the 100-game sample (`1622501`).
3. It surfaces over-repair on units the deeper-bug PHP wouldn't have healed,
   making the engine_lt_php cluster strictly worse (1622328: −3000 → −6600
   delta on the failing turn).

Per the explicit ship gate (≥3 of 4 must close, no new `engine_bug`), we are
at 0 / 4. Per the explicit stop-and-report rule on regressing other tests,
we burn a previously-`ok` GID. Combined verdict: **revert and escalate.**

---

## Section 4 — Validation gates (post-revert, status-quo confirmation)

| # | Gate | Threshold | Result |
|---|---|---|---|
| 1 | `pytest --tb=no -q --ignore=test_trace_182065_seam_validation.py` | ≤2 failures (deferred 182065 only) | **PASS** — `526 passed, 5 skipped, 2 xfailed, 3 xpassed`, 0 failures off-deferred |
| 2 | Targeted re-audit `tools/desync_audit.py --games-id <each>` | inform-only | All 4 still `oracle_gap` (no change vs post-FU baseline; status quo confirmed) |
| 3 | 100-game sample `tools/desync_audit.py --max-games 100 --seed 1` | `engine_bug = 0`, `ok ≥ 89` | **PASS** — baseline `logs/desync_register_post_phase11j_fu_100.jsonl` `ok=89, oracle_gap=11, engine_bug=0` retained (engine unchanged) |

(Gates above describe the **revert** state, which equals the
post-Phase-11J-F2-KOAL-FU baseline.)

---

## Section 5 — Verdict and commander brief

**Verdict letter: ESCALATE.**

Imperator — there is exactly **one** AWBW-cited fix candidate (R2,
all-or-nothing per-unit repair, three independent Tier-2 citations on
`awbw.fandom.com/wiki/Units` and `/wiki/Advance_Wars_Overview`), and **one
suggestive-but-uncited** fix candidate (R1, income-before-repair ordering;
the only direct citation is the *vanilla* Advance Wars wiki at
`advancewars.fandom.com/wiki/Turn`, which is not AWBW canon, plus Tier-3
PHP-snapshot empirical evidence on 1621434 env 24). Both rules combined
are fixable in `engine/game.py` in about ten lines, but applying them in
isolation does not close any of the 4 regressed GIDs. The dominant cause
of the funds gap is a **deeper repair-eligibility / combat-damage drift**
I cannot fix without primary citation: AWBW appears to skip property
repair on units that took damage during the opponent's just-ended turn
(smoking-gun in 1622328 env 28, P1 — engine repairs all eight damaged P1
units, PHP repairs only the one that was already low-HP before P0's
attacks). The engine over-repairs in that scenario, and the income-first
reorder ALONE makes that worse because it removes the partial-loop's
silent funds cap that was masking the over-eligibility. I reverted to the
post-Phase-11J-F2-KOAL-FU baseline; engine, tests, and 100-game sample
unchanged.

Three questions need your routing (Q3 added per the citation directive):

1. **Repair-eligibility canon.** Need either a primary AWBW source (the
   AWBW Wiki at `awbw.fandom.com/wiki/` does not document this, the
   amarriner FAQ is silent, and AWBW PHP source is closed per
   `awbw.amarriner.com/guide.php`), or a curated replay where a
   freshly-damaged unit visibly **fails** to heal at the start of its
   owner's next turn so I can fixture it. Without one of those I will not
   patch on the "damaged-during-opponent-turn-skip" hypothesis — Phase 11A
   Kindle is the cautionary precedent.
2. **Sasha SCOP "War Bonds" formula (1624082 narrow lane).** AWBW
   `co.php` gives the rule (*"Returns 50% of damage dealt as funds (subject
   to a 9HP cap)"*) but not the per-HP funds formula. Reclassify 1624082
   as a Sasha-CO scrape lane (Phase 11Y-CO-WAVE-2 §5 pattern) so we can
   curate a Sasha-SCOP-fires replay set and reverse-engineer the formula
   off PHP snapshots, or do you have a citation pointer?
3. **Income-vs-repair turn-order canon (R1).** AWBW Wiki is silent on the
   order; only the *vanilla* AW Wiki (`advancewars.fandom.com/wiki/Turn`)
   sequences income-first. The Tier-3 PHP-snapshot evidence on 1621434
   env 24 is consistent with income-first, but per your citation directive
   I will not promote that empirical pattern to a fix without an AWBW-side
   primary source. Acceptable to ship R1 on Tier-3 evidence alone, or
   should I keep waiting on a Tier-1 / Tier-2 source?

Once Q1 lands, I can ship R2 + the eligibility rule together (and R1 if
Q3 unlocks it); the combined patch should close 1621434, 1621898, and
1622328 (the combat/eligibility cluster) without −1 ok regression.

---

## Section 6 — Artifacts

**New tooling (read-only, kept for next iteration):**

- `tools/_phase11j_funds_drill.py` — per-envelope funds/property comparison.
- `tools/_phase11j_repair_trace.py` — instrumented `_resupply_on_properties` /
  `_grant_income` logger (configurable `--gid`, `--from-day`).
- `tools/_phase11j_pairing_check.py` — confirms tight pairing on the 4 GIDs.
- `tools/_phase11j_first_drift.py`, `tools/_phase11j_first_drift_summary.py` —
  one-line first-drift / failing-build extractors.
- `tools/_phase11j_php_unit_dump.py` — PHP per-frame unit-HP dump.
- `tools/_phase11j_compare_100.py` — 100-game class-flip diff vs baseline.

**Logs:**

- `logs/phase11j_funds_drill.json` — pre-fix per-GID drill (Section 1 source).
- `logs/phase11j_funds_drill_postfix.json` — same drill on the throwaway
  R1+R2 branch (Section 2 source); kept for audit.
- `logs/phase11j_repair_trace_{1621434,1621898,1622328,1624082}.txt` —
  repair/income traces around each GID's drift envelope.
- `logs/d_{1621434,1621898,1622328,1624082}.jsonl` — single-GID re-audit
  registers (post-revert).
- `logs/desync_register_post_phase11j_f2_fu_funds_100.jsonl` — 100-game
  sample on the throwaway R1+R2 branch (`ok=88, oracle_gap=12,
  engine_bug=0`); kept for the −1 ok evidence.

**Engine + tests: NO CHANGES** vs post-Phase-11J-F2-KOAL-FU baseline.

---

*"Si vis pacem, para bellum."* (Latin, ~4th century AD)
*"If you want peace, prepare for war."* — Publius Flavius Vegetius Renatus, *De Re Militari*
*Vegetius: late-Roman writer on military affairs; this maxim from his treatise on the army has survived as the patron saying of disciplined preparation.*
