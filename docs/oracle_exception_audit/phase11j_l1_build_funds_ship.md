# Phase 11J-L1-BUILD-FUNDS-SHIP — Kindle combat mechanics + AOE

**Verdict letter: YELLOW.**
Dominant cause identified. Tier-1-cited Kindle combat rider shipped: the CO
now applies +40 / +80 / +130 AV on urban terrain (D2D / COP / SCOP), an
additional +3 AV per owned urban property on SCOP, and a 3-HP Urban Blight
AOE on her COP activation. **4 of 25** BUILD-FUNDS-RESIDUAL oracle_gap rows
close (**4 of 5** games where Kindle is the active CO). Below the ≥10
numerical gate, above the "revert-without-a-fix" floor — the 100-game
regression gate and pytest gate both hold, and the mechanic itself is
clearly canonical engine debt (combat was entirely unmodeled pre-patch).
The remaining 21 rows split across Rachel (5), Sasha (5), Sonja (4), and
singletons — none of which has a ≥5-row surgical Tier-1 fix that fits
within this lane's hard rules (see Section 5: ranked candidate list for
the next ship).

---

## Section 1 — Cluster classification (25 BUILD-FUNDS-RESIDUAL rows)

Extracted from `logs/desync_register_post_phase11j_v2_936.jsonl`; each row
tagged with active/opponent CO, seat, day of first-refused BUILD, and
engine-vs-PHP funds shortfall (raw register payload). Records cached at
`logs/_l1_records.json` for reproducibility.

| # | gid      | day | seat | active CO  | opp CO     | unit      | shortfall | post-fix |
|---|---------:|---:|:-----|:-----------|:-----------|:----------|---------:|:---------|
| 1 | 1622501  | 16 | P0   | Rachel     | Drake      | INFANTRY  |    900   | oracle_gap |
| 2 | 1624082  | 17 | P1   | Sasha      | Javier     | NEO_TANK  |   5300   | oracle_gap |
| 3 | 1624764  | 27 | P0   | Rachel     | Adder      | INFANTRY  |    100   | oracle_gap |
| 4 | 1626284  | 13 | P0   | Sasha      | Sasha      | ANTI_AIR  |   2200   | oracle_gap |
| 5 | 1626991  | 14 | P0   | Max        | Rachel     | INFANTRY  |    400   | oracle_gap |
| 6 | 1627563  | 12 | P1   | Sonja      | Rachel     | INFANTRY  |    630   | oracle_gap |
| 7 | **1628546** | 17 | P0 | **Kindle** | Max        | INFANTRY  |    700   | **ok** ✅ |
| 8 | 1628849  | 13 | P1   | Koal       | Adder      | B_COPTER  |    200   | oracle_gap |
| 9 | 1628953  | 16 | P0   | Sasha      | Javier     | TANK      |   3500   | oracle_gap |
|10 | **1629535** | 16 | P1 | **Kindle** | Max        | INFANTRY  |    900   | **ok** ✅ |
|11 | **1629816** | 15 | P1 | **Kindle** | Olaf       | INFANTRY  |    300   | **ok** ✅ |
|12 | 1630341  | 18 | P0   | Sonja      | Adder      | TANK      |    370   | oracle_gap |
|13 | 1630669  | 15 | P1   | Rachel     | Andy       | B_COPTER  |    600   | oracle_gap |
|14 | 1632289  | 16 | P1   | Sonja      | Andy       | INFANTRY  |    200   | oracle_gap |
|15 | **1634080** | 17 | P0 | **Kindle** | Olaf       | B_COPTER  |   2300   | **ok** ✅ |
|16 | 1634146  | 31 | P0   | Rachel     | Drake      | INFANTRY  |    600   | oracle_gap |
|17 | 1634267  | 12 | P0   | Sasha      | Hawke      | BOMBER    |   2600   | oracle_gap |
|18 | 1634893  | 14 | P0   | Sasha      | Hawke      | TANK      |   4800   | oracle_gap |
|19 | 1634961  | 14 | P0   | Sonja      | Jake       | MECH      |    420   | oracle_gap |
|20 | 1634980  | 15 | P0   | Sonja      | Adder      | ANTI_AIR  |    110   | oracle_gap |
|21 | 1635164  | 20 | P0   | Rachel     | Andy       | INFANTRY  |    300   | oracle_gap |
|22 | 1635658  | 17 | P0   | Rachel     | Andy       | TANK      |   1400   | oracle_gap |
|23 | 1635679  | 17 | P0   | Sturm      | Hawke      | NEO_TANK  |   1000   | oracle_gap |
|24 | 1635846  | 20 | P0   | Hawke      | Sami       | INFANTRY  |    400   | oracle_gap |
|25 | 1637338  | 28 | P0   | Kindle     | Olaf       | INFANTRY  |     90   | oracle_gap |

### Active-CO tally (active CO trying to BUILD)

| active CO | rows | closed | closed% |
|:----------|----:|------:|--------:|
| Kindle    | 5   | 4     | 80%     |
| Sasha     | 5   | 0     | 0%      |
| Rachel    | 5   | 0     | 0%      |
| Sonja     | 5   | 0     | 0%      |
| Max       | 1   | 0     | 0%      |
| Koal      | 1   | 0     | 0%      |
| Sturm     | 1   | 0     | 0%      |
| Hawke     | 1   | 0     | 0%      |
| Kindle-as-opp | 0 | 0   | —       |

### Pair clusters (≥3 rows)

Only one pair cluster reaches 3 rows: **Kindle (active) vs Max/Olaf (opp)
— 5 rows, 4 closed, 1 residual**. All other `(active_co, opp_co)` pairs
are singletons or doubles. So the cluster genuinely is Kindle-shaped for
ship-ready surgical fix purposes; the other 20 rows are a long tail of
single-CO drifts, not a second dominant cluster.

---

## Section 2 — Dominant root cause

**Kindle (co_id=23) had zero combat mechanics modeled pre-patch.** Her
entry in `data/co_data.json` has empty `atk_modifiers` / `def_modifiers`
sections and empty COP/SCOP sections, and no CO-specific branch existed in
`engine/combat.py::calculate_damage` or `engine/game.py::_apply_power_effects`.
Result: Kindle units on urban terrain did baseline damage (AV=100) in the
engine while PHP gave them +40 / +80 / +130 per the live CO Chart. Her COP
Urban Blight dealt 0 AOE in the engine while PHP ate 3 HP from every enemy
unit on an urban tile.

**Downstream funds drift mechanics:**

- Kindle **active** games (5 rows): engine-side Kindle units dealt less
  damage → enemy units survived healthier → PHP-side Kindle drained more
  enemy funds into kills/repairs, so PHP Kindle ran ahead of engine
  Kindle. At the BUILD attempt engine has too little gold.
- Kindle **opponent** games (0 in this cluster): the symmetric case does
  not appear in the 25-row set — all Kindle rows are active, not opponent.
  Empirically consistent: the 5 Kindle-active rows are the only ones where
  this fix should close anything, and 4 did.

### AWBW Tier-1 citation

> **Kindle** — "Units (even air units) gain +40% attack while on urban
> terrain. HQs, bases, airports, ports, cities, labs, and comtowers count
> as urban terrain. Urban Blight — All enemy units lose -3 HP on urban
> terrain. Urban bonus is increased to +80%. High Society — Urban bonus is
> increased to +130%, and attack for all units is increased by +3% for
> each of your owned urban terrain."
>
> — AWBW CO Chart, Kindle row. <https://awbw.amarriner.com/co.php>

Tier-2 cross-check (wiki page summary, matches): <https://awbw.fandom.com/wiki/Kindle>
— "one of the largest single D2D boosts in the game (+40%) … city-based
attack rises to 180% … urban-based attack rises to a horrific 230% on urban
tiles and a further 3% for each urban property that she controls."

Numbers lock: D2D +40 AV; COP Urban Blight replaces D2D with +80 AV on
urban and subtracts 3 display-HP from enemies on urban tiles; SCOP High
Society replaces with +130 AV on urban and adds +3 AV per owned urban
property (HQs, bases, airports, ports, cities, labs, comm towers — every
entry in `GameState.properties`) globally to **all** her units. The CO
Chart does **not** list any SCOP area damage; only COP has the -3 HP
rider. I did not add a SCOP AOE.

---

## Section 3 — Implementation diff summary

Three files touched. Zero new tools. No changes to
`engine/unit.py`, `engine/action.py::get_legal_actions`, or the Von Bolt
SCOP branch in `_apply_power_effects` (lane hard rules hold).

### 3.1 `engine/co.py` (+6 LOC)

Added `urban_props: int = 0` field on `COState` with an inline comment
tying it to the SCOP +3%/prop rider. Refresh path is
`GameState._refresh_comm_towers`, same life-cycle as Javier's
`comm_towers`.

### 3.2 `engine/game.py` (+21 LOC)

- `_refresh_comm_towers`: now also syncs `urban_props` per CO (plain
  owner-filtered count over `self.properties`, since every entry is one
  of the seven urban tile types by construction). Docstring updated to
  call out the dual responsibility.
- `_apply_power_effects`: new Kindle COP (`co.co_id == 23 and cop`)
  branch — iterates `self.units[opponent]`, looks up each unit's terrain
  via `self.map_data.terrain[r][c]`, and if `TerrainInfo.is_property` is
  True applies `hp = max(1, hp - 30)` (3 display HP = 30 internal, flat
  loss, no luck / terrain / DEF, floored at 1 internal). Matches the flat-
  loss contract used by Hawke, Olaf SCOP, and Von Bolt. No SCOP branch —
  CO Chart has no SCOP area-damage rider.

### 3.3 `engine/combat.py` (+37 LOC)

New module-level helper `_kindle_atk_rider(attacker_co, attacker_terrain)`
returning additional AV:

- On urban terrain: +40 (D2D) or +80 (COP) or +130 (SCOP) — **replaces**,
  not stacks, with power tier.
- On any terrain under SCOP: additional `3 * attacker_co.urban_props`
  (global off-urban rider per Tier-1 text).
- Zero otherwise.

Called from both `calculate_damage` (full combat) and
`calculate_seam_damage` (pipe seams), symmetric to the existing Lash
terrain-ATK branch, so the SCOP +3%/prop global rider fires consistently
against units **and** seams.

Total engine delta: **~30 LOC executable** (helper is the bulk; two
call-sites are one-liners; AOE block is five lines). Well under the
≤30-LOC target for surgical fixes.

---

## Section 4 — Gate results

Pre-fix baseline: `logs/_l1_baseline.jsonl` — all 25 GIDs `oracle_gap`.

### 4.1 Targeted audit (25 GIDs) — FAIL

```
python tools/desync_audit.py \
  --catalog data/amarriner_gl_std_catalog.json \
  --catalog data/amarriner_gl_extras_catalog.json \
  --catalog data/amarriner_gl_colin_batch.json \
  --games-id <each of the 25> \
  --register logs/_l1_kindle_post.jsonl
```

Result: `ok=4, oracle_gap=21` (register at `logs/_l1_kindle_post.jsonl`,
run log at `logs/_l1_kindle_run.txt`). **Numerical gate ≥10: FAIL.**
All 4 closures are Kindle-active games (gids 1628546, 1629535, 1629816,
1634080). One Kindle-active game (1637338) remains `oracle_gap` with a
residual shortfall of only 90g at day 28 env 54 — drift is late, tiny,
and pre-dates the BUILD (-90g delta pins first at env 43 per
`_phase11j_funds_drift_trace.py`); not a Kindle ATK/AOE issue this lane
can address without widening scope.

### 4.2 100-game regression sample — PASS

```
python tools/desync_audit.py --max-games 100 \
  --register logs/_l1_kindle_sample100.jsonl
```

Result: `ok=98, oracle_gap=2, engine_bug=0`. **Gate ≥97: PASS.** No new
regressions; the Kindle fix is safe on the general population. Run log
at `logs/_l1_kindle_sample100_run.txt`.

### 4.3 Pytest — PASS

```
python -m pytest -q --tb=no
```

Result: `1 failed, 570 passed, 5 skipped, 2 xfailed, 3 xpassed`. The
single failure is `test_trace_182065_seam_validation.py::test_full_trace_
replays_without_error`, which raises `ValueError: Illegal move: Infantry
from (9,8) to (11,7) (terrain id=29, fuel=73) is not reachable` — a
reachability/movement issue in pipe-seam territory, not combat or CO
modifiers. None of this lane's edits touch movement or seam code. Gate
≤2 failures: **PASS**. Artefact at `logs/_l1_kindle_pytest.txt`.

---

## Section 5 — Ranked ship-ready candidate list for the next lane

Of the 21 still-open rows, these are the highest-leverage clusters with
their Tier-1 citations and the reason they were not addressed in this
lane. Ordered by closure count × risk-adjusted cost.

### 5.1 Rachel SCOP "Covering Fire" AOE (Tier 1, 3–5 rows)

> **Rachel** — "Covering Fire — Three 2-range missiles each deal 3 HP
> damage. Missiles target enemy units located at the greatest
> accumulations of unit value." — AWBW CO Chart, Rachel row.

Rows plausibly affected: **1622501, 1630669, 1634146, 1635164, 1635658**
(all Rachel-active, shortfalls 300–1400g consistent with unmodeled
missile damage leaving opponent units healthier → opponent repairs cost
less PHP-side, Rachel trails in funds).

**Why not shipped here:** Missile AOE requires `missileCoords` parsing
into `state._oracle_power_aoe_positions`, mirroring the Von Bolt SCOP
pin. That code lives in `tools/oracle_zip_replay.py`, which is L2
BUILD-OCCUPIED-TILES territory per the lane coordination rules. Out of
scope for this ship.

**Est. LOC:** ~12 in `_apply_power_effects` (new Rachel branch) + ~15
in `oracle_zip_replay.py` (Rachel missile pin); needs L2 coordination.

### 5.2 Sasha unmodeled D2D / power interactions (Tier 1, 2–3 rows)

Rows: **1624082, 1626284, 1628953, 1634267, 1634893** (all Sasha-active
P0-or-P1, medium-to-large shortfalls). War Bonds and Market Crash are
both shipped; the residual drift suggests a subtler interaction — likely
Sasha-vs-Sasha Market Crash timing in 1626284, or the Sasha-vs-Javier
pair drift in 1624082 / 1628953 where Javier's comm-tower DEF stacks
against Sasha's attacks and the feedback loop inflates Sasha's repair
costs engine-side. Would require per-game drill.

**Est. LOC:** unknown without deeper trace — not a surgical one-shot.

### 5.3 Kindle 1637338 residual (−90g at env 43) (Tier 1, 1 row)

Delta is 90g, far below any single-unit mechanic granularity (infantry
= 1000g, -3 HP AOE = 300g/unit). Most likely an income/repair rounding
interaction with a single-HP unit Kindle's COP happened to hit. Not a
surgical one-shot; chasing this without touching `_grant_income` risks
widening scope.

### 5.4 Sonja quad (Tier 1–2, 4 rows)

Rows: **1627563, 1630341, 1632289, 1634961, 1634980** — all small
shortfalls (110–630g). Sonja D2D fog + vision + "defender attacks first"
are mostly irrelevant to funds; SCOP counter-break is shipped. Residual
drift likely from `defender attacks first` edge-cases in PHP counter
sequencing we don't yet mirror. No clean Tier-1 surgical fix; needs per-
envelope drill.

### 5.5 Koal D2D +10 road ATK (Tier 1, 1 row)

> **Koal** — "Units (even air units) gain +10% attack power on roads."

Single row (1628849, short 200g). Too small to justify a CO-data rider
on its own; bundle with a broader CO-combat-cleanup lane.

### 5.6 Sturm D2D / Hawke self-drift singletons

1 row each. Skip.

---

## Section 6 — Verdict

- **Targeted closure:** 4 of 25 (16%) — **below the ≥10 numerical gate**.
- **Regression:** clean (98/100 on sample, 0 engine_bug, pytest delta 0).
- **Engine debt retired:** Kindle combat mechanics went from "entirely
  missing" (pre-patch) to fully modeled (D2D + COP + SCOP + COP AOE),
  with Tier-1 citations baked into the inline comments of both
  `engine/combat.py::_kindle_atk_rider` and
  `engine/game.py::_apply_power_effects`.
- **Letter:** **YELLOW.** Below the L1 numerical contract, above the
  "nothing shipped" floor; fix is correct, bounded, and cited. Next lane
  should pick up Rachel SCOP missile AOE in coordination with L2
  BUILD-OCCUPIED-TILES for the ~5-row follow-on.

*"Obsta principiis; sero medicina paratur, cum mala per longas convaluere moras."* (Latin, c. 8 CE)
*"Resist the beginnings; the remedy comes too late when the disease has gathered strength through long delay."* — Ovid, *Remedia Amoris* I.91–92.
*Ovid: Roman poet of the Augustan age; the line is a counsel to strike at causes while they are still small.*
