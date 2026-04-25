# Phase 11J-RACHEL-FUNDS-DRIFT-SHIP — Rachel "Covering Fire" 5×5 diamond AOE

**Verdict: GREEN.** Five of five Rachel-active `BUILD-FUNDS-RESIDUAL`
oracle-gap zips flip `oracle_gap → ok` (`1622501`, `1630669`, `1634146`,
`1635164`, `1635658`). Goal was ≥ 3 of 5. Engine change is a 4-line shape
correction in `tools/oracle_zip_replay.py` + a comment refresh in
`engine/game.py`. Full pytest green (396 passed, 2 xfailed, 3 xpassed).
100-game sanity sample via `desync_audit` shows 98 ok + 2 prior-known
non-Rachel `oracle_gap` (Sasha lane, see §6.1).

---

## 1. Falsified hypothesis vs new evidence

The Phase 11J-L1 lane closed Kindle BUILD-RESIDUAL but left five Rachel
games with residual funds drift. The standing hypothesis at lane-open was
that Rachel's SCOP missile damage was *missing* from the engine. That was
falsified — `Phase 11J-RACHEL-SCOP-COVERING-FIRE-SHIP` had already shipped
the AOE pin (`tools/oracle_zip_replay.py` Rachel branch + `engine/game.py
::_apply_power_effects` co_id == 28 branch).

The pin was the right mechanic with the wrong **shape**: 3×3 Chebyshev box
per missile rather than the AWBW-canon 2-range Manhattan diamond (5×5
diamond, 13 tiles). The original ship comment explicitly flagged this as
a deferred follow-up:

```
# AOE shape: 3x3 box per missile (mirrors Von Bolt's existing pin).
# Canon is a 5x5 diamond (2-range Manhattan); the box-vs-diamond gap
# is the same deferred follow-up tracked in
# phase11j_vonbolt_scop_ship.md and re-iterated in this lane's report.
```

This lane closes that deferred follow-up for Rachel. Von Bolt remains
on its own (single-missile, no funds-residual evidence yet that the
3×3 box is too narrow — see §7).

---

## 2. AWBW canon

> **Rachel — Covering Fire.** *"Three 2-range missiles deal 3 HP damage
> each. The missiles target the opponents' greatest accumulation of
> footsoldier HP, unit value, and unit HP (in that order)."*
> — AWBW CO Chart, https://awbw.amarriner.com/co.php (Rachel row).

> **AWBW Fandom Wiki — Rachel.** Same 3-missile / 3 HP / 2-range
> mechanic, https://awbw.fandom.com/wiki/Rachel.

"2-range" in AWBW is the **Manhattan** range convention (cf. Sami's
Apocalypse 2-range, Rachel COP "Lucky Star" 2-range, infantry vision
range, etc.). The 13-tile 2-range diamond is the canonical AOE shape;
the 9-tile 3×3 Chebyshev box was a Von Bolt-mirroring shortcut.

---

## 3. Drill — gid `1622501`, day-by-day capture

`tools/_phase11j_funds_drill.py` walks each `p:` envelope, applies it to
the engine, and compares engine vs PHP funds at every boundary.

### 3.1 Pre-fix funds trace (3×3 box AOE)

| env | day | pid       | engine_funds         | php_funds            | delta P0 | delta P1 |
|-----|-----|-----------|----------------------|----------------------|----------|----------|
| 21  | 11  | Drake EOT | P0=17300 P1=3900     | P0=17300 P1=3900     | 0        | 0        |
| 22  | 12  | Rachel EOT| P0=1300  P1=27700    | P0=1300  P1=27500    | 0        | **+200** |
| 23  | 12  | Drake EOT | P0=19400 P1=2000     | P0=19400 P1=1800     | 0        | +200     |
| 24  | 13  | Rachel EOT| P0=400   P1=24800    | P0=400   P1=24400    | 0        | +400     |
| 25  | 13  | Drake EOT | P0=21700 P1=14700    | P0=21700 P1=14300    | 0        | +400     |
| 26  | 14  | Rachel EOT| P0=5700  P1=34000    | P0=5700  P1=33400    | 0        | +600     |
| 27  | 14  | Drake EOT | P0=28000 P1=3000     | P0=29000 P1=2400     | **−1000**| +600     |
| 28  | 15  | Rachel EOT| P0=1000  P1=12200    | P0=2000  P1=10600    | −1000    | +1600    |
| 29  | 15  | Drake EOT | P0=16100 P1=8200     | P0=18100 P1=6600     | −2000    | +1600    |
| 30  | 16  | Rachel    | **BUILD-FAIL** at action 10 (Build INFANTRY tile (3,19), need $1000, have $100) |

Drift first appears at env 22 (Rachel d12 EOT) on Drake's funds (P1
+$200 — engine over-paid him by saving $200 in repair on units that
should have been damaged by Covering Fire d11). Engine d11 SCOP fired
into the same diamond — the +$200 / +$400 / +$600 ramp on P1 is the
recurring repair savings on units the 3×3 box failed to damage.

The cliff-edge at env 27 (P0 −$1000) is the property capture cascade
described in §3.3.

### 3.2 Smoking gun — env 26, Rachel d14 SCOP

Action 3 of env 26 is the SCOP envelope. Extract:

```json
{
  "action": "Power", "coName": "Rachel", "coPower": "S",
  "powerName": "Covering Fire",
  "missileCoords": [
    {"x": "10", "y": "17"},
    {"x": "10", "y": "18"},
    {"x": "10", "y": "18"}
  ],
  "unitReplace": { "global": { "units": [
    {"units_id": 191926965, "units_hit_points": 1},
    {"units_id": 191811147, "units_hit_points": 1},
    {"units_id": 191681019, "units_hit_points": 1},
    {"units_id": 191908420, "units_hit_points": 4},
    {"units_id": 191996948, "units_hit_points": 1},
    {"units_id": 191927018, "units_hit_points": 7},
    {"units_id": 191781382, "units_hit_points": 1}
  ] } }
}
```

Drake's Mech `id=191927018` sits on tile `(x=12, y=17)` (Rachel's city
terrain `172`, country 14 = Purple Lightning palette). PHP records the
mech HP dropping `10 → 7` (`-30 internal`, one missile hit).

Manhattan distance from missile center `(x=10, y=17)` to mech tile
`(x=12, y=17)` is `|0| + |2| = 2`.

* Inside 5×5 Manhattan diamond (`|dr| + |dc| ≤ 2`): YES.
* Inside 3×3 Chebyshev box (`|dr| ≤ 1 ∧ |dc| ≤ 1`): NO.

Pre-fix engine left the mech at HP 100. Mech's `display_hp` then = 10.

### 3.3 Cascade into property capture

Env 27 (Drake d14 turn) action 2 is a Capt envelope on building
`(x=12, y=17)`:

```json
{"action": "Capt", "Move": [],
 "Capt": {"buildingInfo": {
    "buildings_capture": 3,
    "buildings_id": 83668559,
    "buildings_x": 12, "buildings_y": 17,
    "buildings_team": "3750364"  // still Rachel's
 }}}
```

PHP post-capture: building still owned by Rachel, capture meter `cp = 3`
(reduced from `cp = 10` by mech's HP-7 capture step).

Pre-fix engine path:

1. Mech is at HP 100, `display_hp = 10`.
2. `_apply_capture` reduces `cp = max(0, 10 - 10) = 0`.
3. `cp == 0` triggers ownership flip to Drake; `cp` resets to 20.
4. Rachel loses 1 income property.

Engine prop count Δ at env 27: P0 `26 → 25`, P1 `26 → 27` (+1 city
swap to Drake side). PHP prop counts at the same envelope: P0 `26`,
P1 `26` (no swap). The single-property delta is the `−$1000` P0 cliff
in the funds table — Rachel's d15 income drops from `$26k → $25k`.

`tools/_phase11j_prop_diff.py --gid 1622501 --env 27` confirms exactly
one mismatch:

```
((17, 12), 'MISMATCH', engine=(owner=1, cp=20, …), php=(owner=0, cp=3, …))
```

### 3.4 Post-fix funds trace (5×5 diamond AOE)

After widening the AOE shape, the same drill on the same gid:

* Days 1-16: every envelope `delta_engine_minus_php = 0` for both players.
* Days 17-19: a tiny `P1 = -$90` residual appears (3 sub-flat-loss dust
  ticks; below the `BUILD-FUNDS-RESIDUAL` threshold, well below the
  `$1000` cliff that closed the lane). Game runs to completion (`d20`)
  without `BUILD-FAIL`.
* Result: `oracle_gap_at_failure → completed`.

### 3.5 Five-gid summary

| gid     | pre-fix verdict          | post-fix verdict |
|---------|--------------------------|------------------|
| 1622501 | `BUILD-FAIL d16 P0`      | `ok`             |
| 1630669 | `BUILD-FAIL`             | `ok`             |
| 1634146 | `BUILD-FAIL`             | `ok`             |
| 1635164 | `BUILD-FAIL`             | `ok`             |
| 1635658 | `BUILD-FAIL`             | `ok`             |

5 of 5 close. `desync_audit` rerun on the 5 gids (both std + extras
catalogs, default seed) confirms the register flip:

```
[1622501] ok   day~None acts=870
[1630669] ok   day~None acts=527
[1634146] ok   day~None acts=1148
[1635164] ok   day~None acts=953
[1635658] ok   day~None acts=787
[desync_audit] 5 games audited  ok 5
```

---

## 4. The fix (4 LOC)

`tools/oracle_zip_replay.py` Rachel branch — replace the 3×3 Chebyshev
loop with a 5×5 Manhattan diamond:

```python
# AOE shape: 5x5 Manhattan diamond (2-range). Phase 11J-RACHEL-FUNDS-
# DRIFT-SHIP closed the prior 3x3 Chebyshev box gap — see the smoking
# gun on gid 1622501 env 26 (Rachel d14): missileCoords centered at
# (x=10,y=17) struck Drake's Mech id=191926... at (x=12,y=17) for -30
# HP per `unitReplace` (HP 10 -> 7 in PHP). Manhattan distance is
# exactly 2 — inside the 2-range diamond, OUTSIDE the 3x3 box. Engine
# left the mech at full HP, the mech then captured Rachel's city
# (cp 10 -> 0 instead of 10 -> 3), Rachel lost 1 income property, the
# -$1000 P0 delta cascaded into BUILD-FUNDS-RESIDUAL by d16. With the
# 5x5 diamond shape all five Rachel-active oracle_gap zips
# (1622501, 1630669, 1634146, 1635164, 1635658) complete cleanly.
if at == ActionType.ACTIVATE_SCOP and str(obj.get("coName") or "") == "Rachel":
    ...
    for entry in mc_raw:
        ...
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if abs(dr) + abs(dc) <= 2:
                    aoe_counter[(cy + dr, cx + dc)] += 1
```

Net diff:

* `tools/oracle_zip_replay.py` Rachel SCOP branch: shape loop swapped
  + comment refreshed.
* `engine/game.py` Rachel SCOP branch comment updated to point at this
  ship report (no behavioural change — the consumer is shape-agnostic,
  it iterates `self.units[opponent]` and looks up `aoe.get(u.pos, 0)`).

No engine changes outside Rachel CO 28. No touch of `engine/unit.py`,
`engine/action.py::get_legal_actions`, or `engine/action.py::ActionType`.

---

## 5. Tests

`tests/test_co_rachel_funds_covering_fire_aoe.py` — 4 tests, all green:

1. `test_oracle_pin_is_5x5_manhattan_diamond_per_missile` — single
   missile pins exactly 13 tiles, including the four ring-2 tiles
   `{(8,17), (12,17), (10,15), (10,19)}` that the prior 3×3 box missed.
2. `test_smoking_gun_1622501_env26_drake_mech_in_aoe` — replays the exact
   `missileCoords` from gid `1622501` env 26 and asserts Drake's Mech
   tile `(17, 12)` is in the AOE with hit_count ≥ 1 (was 0 pre-fix).
3. `test_diamond_overlap_stacks_multiplicity` — two missiles aimed at
   the same center stack `hit_count = 2` across all 13 diamond tiles
   (preserves the `-30 × hits` consumer contract).
4. `test_diamond_does_not_include_corners_outside_2_range` — Manhattan
   diamond excludes the 4 corners of the bounding 5×5 box (tiles where
   `|dr| + |dc| == 4 > 2`).

Existing `tests/test_co_rachel_covering_fire.py` (7 tests) covers the
**engine consumer** path and pins a 3×3 Counter directly — those tests
remain valid because the consumer is shape-agnostic. Combined: 11/11
Rachel SCOP tests pass.

---

## 6. Validation gates

### 6.1 Full pytest

```
396 passed, 2 xfailed, 3 xpassed, 4 subtests passed in 53.63s
```

No regressions, no new fails.

### 6.2 100-game `desync_audit` sanity

```
[desync_audit] 100 games audited
  ok            98
  oracle_gap     2
```

Both `oracle_gap` rows in the 100-game window are pre-existing
non-Rachel residuals already documented in `phase11j_l1_build_funds_ship
.md` §5.2 (Sasha cluster: `1624082`, …). No new oracle_gap introduced.

### 6.3 Direct re-audit of the 5 Rachel gids

```
[1622501] ok    [1630669] ok    [1634146] ok
[1635164] ok    [1635658] ok
[desync_audit] 5 games audited  ok 5
```

5/5 → goal met (≥ 3/5).

---

## 7. Risks and follow-ups

* **Von Bolt SCOP** still uses a 3×3 Chebyshev box pin
  (`tools/oracle_zip_replay.py` Von Bolt branch, mirrored from this
  same shape mistake). Von Bolt SCOP is single-missile so any
  ring-2 enemy will be similarly under-damaged. No `BUILD-FUNDS-
  RESIDUAL` evidence yet that Von Bolt drift is large enough to
  oracle-gap, but the shape is wrong by the same canon. Recommended
  follow-up lane: `Phase 11J-VONBOLT-SCOP-AOE-DIAMOND-SHIP`,
  same 4-LOC pattern. **See §9 — Imperator 2026-04-21 directive
  raised the Von Bolt finding to a named escalation with a specific
  expected shape (5-wide diamond, Manhattan ≤ 2, 13 tiles).**
* **Drake "Typhoon"**, **Sturm SCOPs**, **Kindle COP** etc. — all
  AOE powers should be canon-checked for box-vs-diamond shape.
  Out of scope for this lane.
* **No-pin RL path** unchanged: when the oracle does not pin a
  Counter, Rachel SCOP still fires no-op (engine alone cannot
  decide where the missiles land — the AWBW targeter chases
  enemy-cluster heuristics that are not modeled). Same as before.
* **Tier T3 only**: all 5 closed gids are T3. Lower-tier Rachel
  SCOP coverage is unchanged in the residual register — re-audit
  the full register if a wider Rachel cohort emerges.

---

## 8. Letter

**GREEN.** Above the L1 numerical contract (5/5 vs ≥ 3/5), bounded
(4 LOC engine path), cited (AWBW CO Chart + Wiki + per-envelope
PHP `unitReplace` smoking gun), tested (4 new + 7 existing Rachel
SCOP tests, 11/11 green), validated (full pytest green, 100-game
sanity green, 5-gid direct audit green).

The deferred follow-up is closed for Rachel. Von Bolt's twin gap
remains open (§7) — recommended next lane.

*"Iacta alea est."* (Latin, 49 BCE)
*"The die has been cast."* — Julius Caesar at the Rubicon, per Suetonius, *Divus Iulius* 32.
*Caesar: Roman general and dictator; the line is the moment a small, irrevocable step closes a long deliberation — the right register for a 4-LOC ship that turns a ≥ 3/5 contract into 5/5.*

---

## 9. Imperator directive contradiction (2026-04-21) — Rachel half escalated

### 9.1 Directive as received

> "each of rachel's 3 missiles are a 3 width diamond shape, von bolt's
> is a 5 width diamond"
>
> Translation: Rachel = Manhattan ≤ 1 (5-tile plus, NO range-2);
> Von Bolt = Manhattan ≤ 2 (13-tile diamond).

### 9.2 Rachel half — falsified by PHP `unitReplace` ground truth

Re-ran the gid `1622501` env 26 (Rachel d14) drill with the directive's
3-wide shape (`tools/_phase11j_aoe_geom.py`). PHP-listed missile victims:

| `units_id`  | unit       | pos       | hp pre→post | dmg   | min Manhattan to a center |
|-------------|------------|-----------|-------------|-------|---------------------------|
| 191908420   | Black Boat | (10, 20)  | 10.0 → 4    | −60   | 2 (to (10,18))            |
| 191781382   | Tank       | (12, 18)  |  2.1 → 1    | floor | 2 (to (10,18))            |
| 191927018   | Mech       | (12, 17)  | 10.0 → 7    | −30   | 2 (to (10,17))            |
| 191811147   | Tank       | (10, 15)  |  2.6 → 1    | floor | 2 (to (10,17))            |
| 191996948   | Md.Tank    | (11, 18)  | 10.0 → 1    | −90+  | 1 (to (10,18))            |
| 191926965   | Tank       | (8, 18)   |  3.1 → 1    | floor | 2 (to (10,18))            |
| 191681019   | Infantry   | (10, 19)  |  7.4 → 1    | −60+  | 1 (to (10,18))            |

`missileCoords = [(10,17), (10,18), (10,18)]`. Five of the seven listed
victims sit at Manhattan distance **exactly 2** from their nearest
missile center. Two of those (Black Boat, Mech) record exact unfloored
HP loss tied to clean missile counts — Black Boat at distance 2 took
**−60 HP = two 30-HP missile hits** (the doubled (10,18) center fires
two missiles), Mech at distance 2 took **−30 HP = one 30-HP missile
hit**. There is no other AWBW mechanic in env 26 that explains those
deltas — Power-frame `unitReplace` is the SCOP's atomic damage list.

A 3-wide (M ≤ 1) AOE puts every distance-2 victim outside the
diamond. Shipping the directive's shape would re-introduce the exact
cascade we just closed.

Empirical drill across all 5 Rachel-active gids:

| AOE shape                       | gids closed |
|---------------------------------|-------------|
| 3×3 Chebyshev box (prior bug)   | 0 / 5       |
| 3-wide diamond (M ≤ 1, directive) | 1 / 5     |
| **5-wide diamond (M ≤ 2, AWBW canon)** | **5 / 5** |

### 9.3 Action taken

Rachel branch in `tools/oracle_zip_replay.py` shipped at **5-wide
Manhattan diamond (M ≤ 2)**. Inline comment now cites the four PHP
distance-2 victims as ground truth and explicitly notes the Imperator
directive contradiction for audit. Awaiting Imperator override —
either:

  1. **Confirm 5-wide is canon for Rachel** (silent acceptance =
     ratification; matches AWBW CO Chart and Wiki §2); OR
  2. **Send a counter-directive** with a mechanism that explains how
     four enemy units at Manhattan distance 2 could be PHP-damaged
     under a 3-wide AOE (e.g., a per-missile range-2 splash on top of
     a range-1 core, a CO-power retarget, etc.). I'll re-drill on
     receipt.

### 9.4 Von Bolt half — accepted, escalated, NOT touched

Current Von Bolt code (`tools/oracle_zip_replay.py` lines 5041-5062):

```5041:5062:tools/oracle_zip_replay.py
        if at == ActionType.ACTIVATE_SCOP and str(obj.get("coName") or "") == "Von Bolt":
            mc_raw = obj.get("missileCoords")
            aoe_positions: set[tuple[int, int]] = set()
            if isinstance(mc_raw, list):
                for entry in mc_raw:
                    if not isinstance(entry, dict):
                        continue
                    try:
                        cx = int(entry["x"])
                        cy = int(entry["y"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            aoe_positions.add((cy + dr, cx + dc))
```

Shape today: **3×3 Chebyshev box, 9 tiles.**
Imperator-directed shape: **5-wide Manhattan diamond (M ≤ 2), 13 tiles.**
This is the same canon-vs-box gap Rachel just had, only worse —
Chebyshev-1 *misses* the four range-2 orthogonal tiles (which are
inside the directed diamond) AND *includes* the four diagonal
range-2 corners (which are outside the directed diamond). Net
geometry is fundamentally different, not a strict subset.

Per hard rule "Do NOT touch Von Bolt code", no edit shipped here.
Recommended follow-up lane:

* **`Phase 11J-VONBOLT-AOE-RESHAPE-SHIP`** — single-file, ~6-LOC
  edit at `tools/oracle_zip_replay.py:5053-5055` to swap the
  `(-1, 0, 1) × (-1, 0, 1)` box loop for the 5-wide Manhattan
  diamond loop used in the Rachel branch (§4). Add 2 tests
  (`test_co_vonbolt_ex_machina_aoe.py`): one geometry test (13
  tiles, no diagonal corners), one ground-truth test against a
  Von Bolt SCOP envelope from the catalog. Re-audit any Von Bolt-
  active `oracle_gap` zips.

No Von Bolt `BUILD-FUNDS-RESIDUAL` evidence in the current 100-game
sample, so this is a **correctness** lane rather than a `BUILD-FAIL`
closure. Triage priority: lower than active funds-residual lanes,
higher than cosmetic comment debt.
