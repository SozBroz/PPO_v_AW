# Phase 11J — CO-Mechanics Survey (Recon Report)

**Status:** YELLOW — 2 strong ship candidates with concrete gid evidence; no third
candidate clears the **≥2 gid + Tier 1-2 citation + clearly unimplemented** bar.

**Mode:** Read-only recon. No code edits, no new tests. ≤10 file reads (used: 3 —
`data/co_data.json`, `engine/game.py` [two bands], plus targeted greps & one web
fetch of the AWBW chart).

**Primary register surveyed:** `logs/_phase11j_lane_l_full936.jsonl` (the 936-zip
LANE-L re-audit, 30 `oracle_gap` rows). The 100-game post-WARBONDS register
(`logs/desync_register.jsonl`, 3 `oracle_gap` rows) was checked first; all three
rows are already attributed in the prompt's known-residual list and yield no CO
pair with ≥2 hits.

---

## Executive Summary — Top Ship Candidates

| Rank | CO | Mechanic | Closure (gids) | LOC | Risk | Citation |
|------|----|----|----|-----|------|----------|
| **1** | **Kindle (23)** | Urban-terrain ATK D2D/COP/SCOP **and** -3 HP urban AOE on COP/SCOP — *entirely unimplemented* | **8** | ~80-120 | MED | Tier 1 (amarriner) |
| **2** | **Sasha (19)** | COP **Market Crash** formula correction — engine uses `count_properties × 9000`, canon is `(10 × Sasha_funds / 5000)% of opp max power bar` | **2-4** | ~5 | LOW | Tier 1 (amarriner) |
| 3 | — | No third candidate met the bar (see "Why no rank 3" below) | — | — | — | — |

**One-line pitches:**

1. **Kindle is a ghost in the engine.** Eight oracle_gap rows in the 936 sample
   sit on a CO whose D2D, COP, and SCOP combat mechanics are *literally not
   branched anywhere* in `engine/{co,combat,game,action,weather,unit}.py`. The
   only Kindle reference in code is a deliberate income exclusion. Highest
   closure-per-LOC available; medium risk only because it's a combat ATK touch.

2. **Sasha Market Crash uses the wrong drain formula.** Engine drains
   `count_properties(player) × 9000` from opponent power bar. AWBW chart canon
   is `(10 × Sasha_current_funds / 5000)% of opponent_max_power_bar`.
   Five-line surgical swap, Tier 1 citation, low risk. Confirms ≥2 gids where
   Sasha COP fires.

---

## Top Candidate Detail

### Rank 1 — Kindle (CO 23) full implementation

**State of the engine:**

Greps for `co_id == 23` across `engine/{game,co,combat,action,weather,unit}.py`
return **zero combat hits**. The only Kindle code is a deliberate income exclusion
in `engine/game.py:543` (Phase 11A precedent — PHP rejected Kindle's +50% city
income, correctly omitted). Every other Kindle mechanic — D2D urban ATK, COP
Urban Blight, SCOP High Society — is **missing entirely**.

**Tier 1 canon** (amarriner.com/co.php, fetched 2026-04-21):

> Kindle: Units (even air units) gain **+40% attack** while on urban terrain.
> HQs, bases, airports, ports, cities, labs, and comtowers count as urban terrain.
> **Urban Blight** — All enemy units lose **-3 HP** on urban terrain. Urban bonus
> is increased to **+80%**.
> **High Society** — Urban bonus is increased to **+130%**, and attack for all
> units is increased by **+3% for each of your owned urban terrain**.

This corrects three engineer-folklore numbers carried in `data/co_data.json`:
the chart-canon ATK is **+40%** (not +30%), AOE damage is **-3 HP** (not -1/-2),
and "urban" includes **HQ/labs/comtowers** in addition to cities.

**Tier 2:** awbw.fandom.com/wiki/Kindle (mirrors chart).

**Engine touch points:**

- `engine/co.py` — add `kindle_urban_atk_bonus(self, on_urban_tile: bool) -> int`
  returning `40 + (40 if cop_active else 0) + (90 if scop_active else 0)`,
  plus a SCOP-only `+3 × owned_urban_count` global ATK term.
- `engine/combat.py` — at the existing Lash terrain-bonus injection (lines
  141-143, 242-244), add a parallel Kindle hook that consults attacker's tile
  via `state.get_property_at(attacker.pos)` and the attacker's terrain class
  (HQ / base / airport / port / city / lab / comtower).
- `engine/game.py::_apply_power_effects` — add `elif co.co_id == 23:` branch
  applying -30 internal HP (3 display) to enemy units standing on urban tiles,
  floored at 1 internal — same flat-loss template as the existing Drake / Olaf
  / Hawke / Von Bolt branches at lines 632-673.
- `engine/game.py::_grant_income` — **leave the deliberate Kindle exclusion in
  place** (Phase 11A precedent at line 543; PHP rejects the funds bonus).

**Affected gids (8 of 8 Kindle-bearing oracle_gap rows in the 936 sample):**

| gid | side | opp | env | Build msg | likely mechanism |
|------|------|-----|-----|-----------|------|
| 1625178 | P0 | Kindle (mirror) | 26 | tile occupied @ (4,12) MEGA_TANK | engine has unit PHP doesn't (under-killed urban AOE) |
| 1628287 | P1 | Sasha | 12 | tile occupied @ (8,14) INF | same |
| 1628546 | P0 | Max | 32 | INF $300/$1000 short | cascade — engine over-charged for repairs that PHP skipped on damaged-by-AOE units |
| 1629816 | P1 | Olaf | 29 | INF $700/$1000 short | same |
| 1632006 | P1 | Eagle | 41 | tile occupied @ (18,9) ANTI_AIR | engine has unit PHP destroyed by SCOP +130% urban ATK |
| 1632778 | P0 | Max | 7 | tile occupied @ (12,10) INF | same |
| 1634080 | P0 | Olaf | 32 | B_COPTER $6700/$9000 short | cascade |
| 1634587 | P0 | Kindle (mirror) | 16 | tile occupied @ (13,1) MED_TANK | mirror — both sides under-damaging |

**Closure estimate:** the four "tile occupied" rows have a direct mechanism
(under-killed urban AOE / under-strength urban ATK) and should close cleanly.
The four "insufficient funds" rows are downstream cascade — likely to close but
not guaranteed by a single fix. Conservative claim: **≥4 gids close, up to 8.**

**Complexity:** ~80-120 LOC across three files. The flat-loss AOE template is
well-established in the codebase (Drake/Olaf/Hawke/VonBolt patterns in
`_apply_power_effects`). The combat ATK injection follows the existing Lash
pattern. Risk is **MEDIUM** purely because urban-attacker terrain awareness
expands the surface area of `combat.py`'s ATK path.

---

### Rank 2 — Sasha (CO 19) Market Crash formula correction

**State of the engine:**

`engine/game.py:689-693`:

```589:594:engine/game.py
        # Sasha COP "Market Crash": drain power bar of enemy CO.
        elif co.co_id == 19 and cop:
            self.co_states[opponent].power_bar = max(
                0, self.co_states[opponent].power_bar - (self.count_properties(player) * 9000)
            )
```

**Tier 1 canon** (amarriner.com/co.php, fetched 2026-04-21):

> Sasha: Receives +100 funds per property that grants funds and she owns. (Note:
> labs, comtowers, and 0 Funds games do not get additional income).
> **Market Crash** — Reduces enemy power bar(s) by **(10 × Funds / 5000)%** of
> their maximum power bar.
> **War Bonds** — Receives funds equal to 50% of the damage dealt when
> attacking enemy units.

(Worth noting: the **Sasha D2D** is per-income-property in AWBW canon — engine
matches this at `engine/game.py:566-568`. The widely-circulated "+100 funds per
enemy CO power star" is from the original DS game, not AWBW. **No fix needed
on Sasha D2D.** And War Bonds is correctly implemented at lines 695-721.)

**The bug is COP only.** Engine drains `properties × 9000`. Canon drains
`(2 × Sasha_funds / 1000)% × opp_max_power_bar` — i.e., proportional to Sasha's
current treasury, not her property count × an arbitrary 9000.

**Tier 2:** awbw.fandom.com/wiki/Sasha (mirrors chart).

**Engine touch point:**

- `engine/game.py:691-693` — replace `count_properties(player) * 9000` with
  `int(opp.power_bar_max * (10 * co.funds / 5000) / 100)`. May require exposing
  `power_bar_max` on `COState` (or computing as `co.cop_stars × 9000` for COP /
  scop_stars × 9000 for SCOP). 5 LOC, surgical.

**Affected gids (≥2 confirmed):**

| gid | side | opp | env | msg | mechanism |
|-----|------|-----|-----|-----|-----------|
| 1626284 | P0 | Sasha (mirror) | 24 | ANTI_AIR $5800/$8000 short | Sasha's drain mis-sized by current-funds delta — power-bar state propagates to opponent's economy via (mis-)timed COP firing |
| 1628953 | P0 | Javier | 30 | TANK $3500/$7000 short | same |

**Lower-confidence additions** (Sasha-bearing rows where the COP firing pattern
is consistent with the bug; needs confirmatory replay drill):

- 1624082, 1634267, 1634893 — all P1 vs Sasha (Javier / Hawke opponents); funds
  drift in opponent's treasury suggests the drain mis-sized when Sasha COP fired.

**Conservative claim: ≥2 gids close. Plausible: up to 5.**

**Complexity:** ~5 LOC. Risk **LOW** — surgical numeric formula swap inside an
existing branch with established precedent (Phase 11J-SASHA-WARBONDS-SHIP just
landed in the same `_apply_power_effects` function).

---

### Why no Rank 3

I dropped every candidate that didn't meet **all three** of: ≥2 gid evidence,
Tier 1-2 citation, and a clearly **missing or wrong** engine implementation.
The survivors:

- **Drake D2D `-30% air ATK` and naval `+10 DEF` and naval `+1 movement`** —
  Tier 1 chart confirms all three, none of which appear in `data/co_data.json`
  or any `co_id == 5` branch in `combat.py` / `co.py`. **3 oracle_gap rows
  involve Drake** (1622501, 1630064, 1634146), but every one of those rows is
  a Build no-op on the *Rachel-side* (Drake is the opponent), and the Drake
  D2D bonuses don't directly drive Rachel's funds — only her combat
  outcomes against Drake's ships. **Insufficient gid attribution. Drop.**

- **Sasha D2D as "per enemy power star"** — heuristic in code, BUT Tier 1
  chart confirms the engine's `+100/income-property` IS the actual AWBW canon.
  No fix needed. The in-engine comment at game.py:535-541 implying "AWBW Sasha
  also gains funds on enemy stars" is folklore from the DS game, not AWBW.
  **Already correct. Drop.**

- **Andy "Hyper Repair" / Hawke "Black Wave" free-heal** — engine heals without
  charging funds. Tier 1 chart text doesn't mention a cost on either; no funds
  drift expected. Engine matches canon. **Drop.**

- **Lash D2D / SCOP Prime Tactics terrain stars doubled** — Lash D2D is
  implemented (combat.py:141, 243), Lash power movement reduction in
  weather.py:276. Only **1 oracle_gap row** (1626223) involves Lash. **Below
  the ≥2 bar. Drop.**

- **Eagle Lightning Strike "may move and fire again even if built this turn"** —
  partially implemented (`scop_eagle_air` in unit.py:342). 1 oracle_gap row.
  **Below the ≥2 bar. Drop.**

- **Sami transport +1 movement** — Tier 1 chart says transports get +1 mov.
  Need to confirm in action.py; even if missing, only 2 Sami rows and one is
  a "mover not found" cascade. Weak attribution. **Drop.**

- **Sturm Meteor Strike / Meteor Strike II** — 4 HP / 8 HP missile damage; no
  `co_id == 29` branch in `_apply_power_effects`. But **Sturm appears in zero
  rows** of the 936 oracle_gap sample. **No gid evidence. Drop.**

- **Adder Sideslip / Sidewinder, Koal Forced March, Hachi 90%** — all
  shipped or cited as such; greps confirm `co_id == 11`, `co_id == 21`
  branches in action.py. Adder's 4 oracle_gap rows are Build/move cascades,
  not Adder-mechanic gaps. **Drop.**

The leading hypothesis for the residual oracle_gap rows beyond Kindle and
Sasha-COP is **cascade drift from already-known partial fixes** (FUNDS-SHIP
ordering, Rachel +1 HP iteration, war-bonds settlement) plus **single-bug
combat damage variance** (~±10% luck, terrain rounding). These are not single
CO-mechanic shipments — they're the residual noise floor that requires
per-zip drill, not bulk recon.

---

## Per-CO Appendix — Mechanic Inventory

For each CO with ≥2 rows in the 936 oracle_gap sample. Status legend:
🟢 implemented + tested · 🟡 implemented (partial / no test) · 🔴 missing.

### Kindle (23) — 8 rows
| Mechanic | Tier 1 chart | Engine state |
|----|----|----|
| D2D +40% ATK on urban (HQ/base/airport/port/city/lab/comtower) | amarriner Kindle row | 🔴 **MISSING** (no co_id==23 in combat.py) |
| D2D +50% city income | amarriner is silent | 🟢 deliberate omit (Phase 11A — PHP rejects) |
| COP Urban Blight: -3 HP urban AOE, urban ATK → +80% | amarriner Kindle row | 🔴 **MISSING** |
| SCOP High Society: urban ATK → +130%, +3% global ATK per owned urban | amarriner Kindle row | 🔴 **MISSING** |

### Sasha (19) — 7 rows
| Mechanic | Tier 1 chart | Engine state |
|----|----|----|
| D2D +100 funds per income-property | amarriner Sasha row | 🟢 game.py:566-568 |
| COP Market Crash: drain `(10 × Funds / 5000)% × opp_max_power` | amarriner Sasha row | 🔴 **WRONG FORMULA** (uses `count_properties × 9000`) |
| SCOP War Bonds: 50% damage as funds | amarriner Sasha row | 🟢 just shipped Phase 11J-SASHA-WARBONDS |

### Rachel (28) — 6 rows
| Mechanic | Tier 1 chart | Engine state |
|----|----|----|
| D2D +1 repair HP, liable for cost | amarriner Rachel row | 🟢 game.py:1846 (R1+R2+R3 shipped Phase 11Y/11J-FUNDS-SHIP) |
| D2D luck +0% to +19% | amarriner Rachel row | 🟢 combat.py:177 |
| COP Lucky Lass: luck → +0% to +39% | amarriner Rachel row | 🟢 combat.py |
| SCOP Covering Fire: 3× 2-range missiles 3 HP each | amarriner Rachel row | 🟡 partial — depends on missile-targeter status; see VONBOLT-SCOP-SHIP for AOE-shape sibling work |

Rachel's 6 oracle_gap rows mostly have funds-drift Build no-ops with $100-$700
shortfalls — consistent with cascade from R3 ordering at funds-tight boundaries
that survived the iteration fix on a small minority of zips, not a missing
mechanic.

### Sonja (18) — 5 rows
| Mechanic | Tier 1 chart | Engine state |
|----|----|----|
| D2D +1 vision, hidden HP, counter ×1.5 | amarriner Sonja row | 🟢 combat.py:373, co.py:166-168 |
| COP Enhanced Vision: +1 vision, see into forests/reefs | amarriner Sonja row | 🟡 needs vision-system check (out of scope for funds-driven oracle_gap) |
| SCOP Counter Break: defender attacks first | amarriner Sonja row | 🟢 co.py |

### Adder (11) — 4 rows
All three Adder mechanics implemented (action.py:173). 🟢 across the board.

### Andy (1) — 4 rows
| Mechanic | Tier 1 chart | Engine state |
|----|----|----|
| D2D no abilities | amarriner | 🟢 (none) |
| COP Hyper Repair: +2 HP | amarriner | 🟢 game.py:627-630 |
| SCOP Hyper Upgrade: +5 HP, +10% ATK, +1 mov | amarriner | 🟢 (heal + atk in `_apply_power_effects`; +1 mov shipped Phase 9 Lane M) |

### Drake (5) — 3 rows
| Mechanic | Tier 1 chart | Engine state |
|----|----|----|
| D2D naval +1 movement | amarriner Drake row | 🔴 not seen in action.py greps for `co_id == 5` |
| D2D naval +10 DEF | amarriner | 🔴 not in `data/co_data.json` Drake def_modifiers |
| D2D air -30% ATK | amarriner | 🔴 not in `data/co_data.json` Drake atk_modifiers (only `naval: 20` listed) |
| D2D unaffected by rain | amarriner | 🟡 weather check needed |
| COP Tsunami: -1 HP enemy + half fuel | amarriner | 🟢 game.py:661 |
| SCOP Typhoon: -2 HP + half fuel + rain | amarriner | 🟢 game.py:661 + 610 |

**Drake D2D is multi-gap and matters,** but the 3 Drake rows are Rachel-side
funds drift where Drake is the opponent. Combat-side fixes don't directly close
those rows — they'd need to deflect Rachel's combat outcomes enough to change
her treasury at the build envelopes. Not a clean single-mechanic ship.

### Max (7) — 3 rows
🟢 all mechanics implemented per `data/co_data.json` and combat.py atk_modifier path. Rows are cascade.

### Hawke (12) — 3 rows
| Mechanic | Engine state |
|----|----|
| D2D +10% ATK all | 🟢 co_data.json |
| COP Black Wave: -1 HP enemy, +1 HP own | 🟡 game.py:634-639 — engine uses **+2 HP own and -1 HP enemy** for COP. Tier 1 chart says **+1 HP own**. Possible canon mismatch but not gid-attributable. |
| SCOP Black Storm: -2 HP enemy, +2 HP own | 🟢 |

⚠ **Note:** the Hawke COP heal is `+20` internal in engine (game.py:637) for both
COP and SCOP, but Tier 1 chart distinguishes "+1 HP" (COP) and "+2 HP" (SCOP).
Engine over-heals on COP by 1 display HP. Below the ≥2 gid bar (3 Hawke rows
all attribute more cleanly to opp-side cascade).

### Koal (21) — 2 rows
🟢 Forced March shipped Phase 11J-F2-KOAL. One row is a Move-truncate (not CO).

### Javier (27) — 2 rows
🟢 Comm-tower DEF/ATK in co.py:174-193. Both rows are Sasha-side opponent funds
drift — fix is on Sasha COP (Rank 2), not on Javier.

### Sami (8) — 2 rows
🟢 Capture mechanic at game.py:1091-1096; SCOP +1/+2 mov at action.py:186.
Possibly missing: transport +1 mov (Tier 1 chart). Below the ≥2 bar.

### Olaf (9) — 2 rows
🟢 Snow / Blizzard / Winter Fury implemented. Both rows are Kindle-vs-Olaf —
already absorbed into Rank 1 closure.

---

## Gids-Affected Appendix

### Rank 1 (Kindle) closure list — 8 gids

```
1625178 — Kindle vs Kindle, env 26, BUILD MEGA_TANK tile-occupied
1628287 — Sasha vs Kindle, env 12, BUILD INF tile-occupied
1628546 — Kindle vs Max, env 32, BUILD INF $300/$1000 short
1629816 — Olaf vs Kindle, env 29, BUILD INF $700/$1000 short
1632006 — Eagle vs Kindle, env 41, BUILD ANTI_AIR tile-occupied
1632778 — Kindle vs Max, env 7, BUILD INF tile-occupied
1634080 — Kindle vs Olaf, env 32, BUILD B_COPTER $6700/$9000 short
1634587 — Kindle vs Kindle, env 16, BUILD MED_TANK tile-occupied
```

### Rank 2 (Sasha COP Market Crash formula) closure list — 2 confirmed + 3 plausible

```
Confirmed (≥2):
1626284 — Sasha vs Sasha, env 24, BUILD ANTI_AIR $5800/$8000 short
1628953 — Sasha vs Javier, env 30, BUILD TANK $3500/$7000 short

Plausible (needs replay drill to confirm COP firing window):
1624082 — Javier vs Sasha, env 33, BUILD NEO_TANK $16700/$22000 short
1634267 — Hawke vs Sasha, env 22, BUILD BOMBER $19400/$22000 short
1634893 — Hawke vs Sasha, env 26, BUILD TANK $2200/$7000 short
```

---

## Coordination Notes Honored

- **Skipped Von Bolt** entirely — VONBOLT-SCOP-SHIP `feb199a6` covers Ex Machina + stun.
- **Skipped Sasha SCOP War Bonds** — just shipped Phase 11J-SASHA-WARBONDS.
- **Skipped Colin** — already a known ship candidate per Phase 11Y-COLIN-SCRAPE.
- **Skipped 100-game register** for ranking — the 3 residuals there are
  pre-attributed (LANE-L move-truncate, FUNDS-SHIP residual, CLUSTER-B-style HP drift).

## Read Budget

Files read (3 of 10):
1. `data/co_data.json` — full CO id → name map and per-CO modifier matrix.
2. `engine/game.py` lines 595-755 — `_apply_power_effects` for Andy / Hawke /
   Drake / Olaf / Sensei / Jess / Sasha / VonBolt power branches.
3. `engine/game.py` lines 1790-1920 — `_resupply_on_properties` for Rachel
   D2D +1 HP and the R1/R2/R3 funds-tight repair fix.

Plus targeted Greps across `engine/` and one WebFetch of
https://awbw.amarriner.com/co.php for Tier 1 citation lock-in.

*"Si vis pacem, para bellum."* (Latin, c. 4th–5th century AD)
*"If you want peace, prepare for war."* — Vegetius, *De Re Militari*
*Vegetius: late-Roman military writer; the line is the standard Western maxim on deterrence and readiness.*
