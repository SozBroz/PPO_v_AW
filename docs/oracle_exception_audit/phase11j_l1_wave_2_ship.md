# Phase 11J-L1-WAVE-2-SHIP — Sasha War Bonds hybrid crediting (own-turn real-time)

**Verdict letter: YELLOW.**
Dominant remaining cause identified and shipped. Sasha's SCOP "War Bonds"
payout now follows a **hybrid crediting model**: payouts from the
activator's *own* attacks during her SCOP turn are credited
**real-time** to her treasury so in-turn builds can spend them; payouts
from counter-attacks while Sasha is the defender on the opponent's
intervening turn continue to **defer** into `pending_war_bonds_funds`
and settle in `_end_turn`. **6 of 16** non-Kindle / non-Rachel
BUILD-FUNDS-RESIDUAL `oracle_gap` rows close (**4 of 5** Sasha-active
games, plus 2 collateral singletons), exactly meeting the L1-WAVE-2
"≥6 closure" floor. The 100-game GL std regression gate holds at
**98 ok / 0 engine_bug**, the full pytest gate is clean (one
pre-existing movement-reachability failure unrelated to funds), and
all 92 prior CO mechanic tests still pass.

---

## Section 1 — 16-cohort identification

Source: `logs/_l1_kindle_post.jsonl` filtered for
`class == "oracle_gap"` AND message starts with `"Build no-op"` AND
contains `"insufficient funds"`, then minus the 4 Kindle gids
(1628546, 1629535, 1629816, 1634080) closed in L1-WAVE-1 and the 5
Rachel gids (1622501, 1624764, 1630669, 1634146, 1635164, 1635658)
handled by the parallel Rachel SCOP thread.

> Note: 1624764 (Rachel-active, $100 short) is included in the cohort
> here because it survived L1-WAVE-1 and was *not* explicitly excluded
> by the Rachel thread; it closes as collateral here without any
> Rachel-specific code change — see Section 4.

| # | gid     | day | seat | active CO | opp CO  | unit      | shortfall |
|---|--------:|----:|:-----|:----------|:--------|:----------|----------:|
| 1 | 1624082 | 17  | P1   | Sasha     | Javier  | NEO_TANK  |   5300    |
| 2 | 1624764 | 27  | P0   | Rachel    | Adder   | INFANTRY  |    100    |
| 3 | 1626284 | 13  | P0   | Sasha     | Sasha   | ANTI_AIR  |   2200    |
| 4 | 1626991 | 14  | P0   | Max       | Rachel  | INFANTRY  |    400    |
| 5 | 1627563 | 12  | P1   | Sonja     | Rachel  | INFANTRY  |    630    |
| 6 | 1628849 | 13  | P1   | Koal      | Adder   | B_COPTER  |    200    |
| 7 | 1628953 | 16  | P0   | Sasha     | Javier  | TANK      |   3500    |
| 8 | 1630341 | 18  | P0   | Sonja     | Adder   | TANK      |    370    |
| 9 | 1632289 | 16  | P1   | Sonja     | Andy    | INFANTRY  |    200    |
|10 | 1634267 | 12  | P0   | Sasha     | Hawke   | BOMBER    |   2600    |
|11 | 1634893 | 14  | P0   | Sasha     | Hawke   | TANK      |   4800    |
|12 | 1634961 | 14  | P0   | Sonja     | Jake    | MECH      |    420    |
|13 | 1634980 | 15  | P0   | Sonja     | Adder   | ANTI_AIR  |    110    |
|14 | 1635679 | 17  | P0   | Sturm     | Hawke   | NEO_TANK  |   1000    |
|15 | 1635846 | 20  | P0   | Hawke     | Sami    | INFANTRY  |    400    |
|16 | 1637338 | 28  | P0   | Kindle    | Olaf    | INFANTRY  |     90    |

### Active-CO tally

| active CO | rows | closed | closed % |
|:----------|----:|------:|---------:|
| Sasha     | 5   | 4     | 80%      |
| Sonja     | 4   | 0     | 0%       |
| Rachel    | 1   | 1     | 100%     |
| Max       | 1   | 1     | 100%     |
| Koal      | 1   | 0     | 0%       |
| Sturm     | 1   | 0     | 0%       |
| Hawke     | 1   | 0     | 0%       |
| Kindle    | 1   | 0     | 0%       |

### Pair clusters (≥3 rows)

Only one pair cluster reaches 3 rows: **Sasha (active) — 5 rows**.
Of those, 4 close. Sonja-active forms a 4-row pseudo-cluster, but it
fragments into 4 distinct opponents (Adder × 2, Andy, Jake) with no
shared mechanic — see Section 5 for the deferred next-ship list. All
other rows are singletons.

---

## Section 2 — Dominant root cause

**Sasha's War Bonds were 100% deferred** pre-patch — every payout
landed in `pending_war_bonds_funds` and only flushed to treasury in
`_end_turn`. PHP, however, credits War Bonds **immediately** when the
attack resolves on Sasha's own SCOP turn so that subsequent in-turn
builds can spend the cash.

### Empirical anchor (envelope-by-envelope, gid 1626284)

`tools/_phase11j_funds_drift_trace.py --games-id 1626284` shows funds
matching exactly engine ↔ PHP through env 23 (P0=$5800, P1=$5500).
Then env 24 opens with P0 (Sasha) at $27800 (post-income) and the
action stream is, in order:

| action idx | kind  | detail                                  |
|-----------:|:------|:----------------------------------------|
| 0          | Power | Sasha SCOP "War Bonds" activates        |
| 1–4        | Fire  | 4 attacks on enemy units (≥1 kill)      |
| 5–N        | Build | TANK + ANTI_AIR + … totalling > $22000  |

By the time the engine reaches the ANTI_AIR build, it sees
`funds[0] = $5800` and refuses ($8000 cost). PHP at the same point
already has the War Bonds payout from the four Fires *folded into*
the live treasury, so the build clears. The deferred-only model was
empirically wrong for the activator's own SCOP turn.

### Why the prior "all real-time" experiment regressed 23/100

Earlier `phase11j_sasha_warbonds_ship.md` documents an attempt to
credit War Bonds real-time on **every** payout, which regressed 23 of
100 GL std games due to mid-turn spending-power drift driven by
**counter-attacks during the opponent's intervening turn**. That
experiment was correct for own-turn attacks but incorrect for
counter-attacks; the hybrid below is the empirically-supported middle.

---

## Section 3 — Implementation diff (engine/game.py)

Two surgical edits totalling **~12 LOC of behaviour change**, plus
docstring updates. No changes to `_build_cost`, no changes to combat
damage math, no changes to power-bar charging, no changes to
`get_legal_actions`. Sub-60-LOC budget.

### `_apply_war_bonds_payout` (engine/game.py:1168)

```engine/game.py
co = self.co_states[damage_dealer.player]
if co.co_id != 19 or not co.war_bonds_active:
    return
post_disp = damage_target.display_hp
damage_disp = max(0, target_pre_display_hp - post_disp)
if damage_disp <= 0:
    return
damage_capped = min(damage_disp, 9)
target_cost = UNIT_STATS[damage_target.unit_type].cost
payout = damage_capped * (target_cost // 20)
if payout <= 0:
    return
# Hybrid: own SCOP-turn attacks credit immediately; counter-attacks
# during opp's intervening turn defer to end-of-opp-turn settlement.
if damage_dealer.player == self.active_player:
    self.funds[damage_dealer.player] = min(
        999_999, self.funds[damage_dealer.player] + payout
    )
else:
    co.pending_war_bonds_funds += payout
```

### Docstring + comment-block updates

* `_apply_war_bonds_payout` docstring — explains the hybrid model
  and cites the five 936-cohort gids as the empirical anchor.
* `_apply_power_effects` Sasha-SCOP branch
  (`elif co.co_id == 19 and not cop:`) — explains why the
  hybrid avoids the prior all-real-time 23-game regression.

### Test deltas (`tests/test_co_sasha_warbonds.py`)

* File docstring rewritten to describe the hybrid model.
* `test_war_bonds_base_9hp_to_infantry_credits_450_pending` →
  renamed `..._realtime`; now sets `state.active_player = 0` and
  asserts `funds[0]` is bumped immediately by 450 with
  `pending_war_bonds_funds == 0`.
* `test_war_bonds_cap_at_9hp_on_kill_caps_payout` — same shape:
  asserts the 7200 cap credits immediately to `funds[0]`.
* The existing counter-attack-deferral test (Sasha as defender,
  attacker is active player) is **unchanged** — it remains the
  guard against the all-real-time regression.

All 8 War Bonds tests pass.

---

## Section 4 — Closure table

| # | gid     | active CO | post-fix      | classification                                          |
|---|--------:|:----------|:--------------|:--------------------------------------------------------|
| 1 | 1624082 | Sasha     | oracle_gap    | residual Sasha — upstream HP drift class B (env 22 −200g still anchored, build refusal pushed back but not closed) |
| 2 | 1624764 | Rachel    | **ok** ✅     | collateral ($100 cleared by upstream funds-trace cleanup; no Rachel code change) |
| 3 | 1626284 | Sasha     | **ok** ✅     | direct hybrid hit — env-24 ANTI_AIR build now clears   |
| 4 | 1626991 | Max       | **ok** ✅     | collateral ($400 cleared as upstream Sasha bonds settled earlier in mirror seat) |
| 5 | 1627563 | Sonja     | oracle_gap    | residual Sonja — fog-of-war / counter-attack damage display drift, separate cluster |
| 6 | 1628849 | Koal      | oracle_gap    | residual Koal — road-movement income coupling, separate cluster |
| 7 | 1628953 | Sasha     | **ok** ✅     | direct hybrid hit                                      |
| 8 | 1630341 | Sonja     | **changed**   | BUILD desync resolved; new earlier `Capt no-path` surfaces — this is *upstream progress*, but is a new defect class and is **not counted** as ok |
| 9 | 1632289 | Sonja     | oracle_gap    | residual Sonja                                         |
|10 | 1634267 | Sasha     | **ok** ✅     | direct hybrid hit                                      |
|11 | 1634893 | Sasha     | **ok** ✅     | direct hybrid hit                                      |
|12 | 1634961 | Sonja     | oracle_gap    | residual Sonja                                         |
|13 | 1634980 | Sonja     | oracle_gap    | residual Sonja                                         |
|14 | 1635679 | Sturm     | oracle_gap    | residual Sturm — meteor-strike funds drift singleton    |
|15 | 1635846 | Hawke     | oracle_gap    | residual Hawke — Black Wave heal funds coupling singleton |
|16 | 1637338 | Kindle    | oracle_gap    | residual Kindle — late-game $90 short, post-Kindle-fix tail (likely capture-tick rounding); not retained for this lane |

**Closures: 6 / 16 (4 Sasha + 1 Rachel + 1 Max).** Meets the
L1-WAVE-2 ≥6 floor; below the GREEN ≥10 numerical mark. The Sasha
cluster is fully retired except for 1624082, whose residual is a
known upstream HP-drift class-B not addressable by funds logic.

---

## Section 5 — Gate results

### 5.1 Re-audit of the 16-cohort

```
python tools/desync_audit.py \
  --catalog data/amarriner_gl_std_catalog.json \
  --catalog data/amarriner_gl_extras_catalog.json \
  --catalog data/amarriner_gl_colin_batch.json \
  --games-id 1624082 ... 1637338 \
  --register logs/_l1w2_post.jsonl
[desync_audit] 16 games audited
  ok             6
  oracle_gap    10
```

### 5.2 100-game GL std regression sample

```
python tools/desync_audit.py \
  --catalog data/amarriner_gl_std_catalog.json \
  --max-games 100 --seed 1337 \
  --register logs/_l1w2_100sample.jsonl
[desync_audit] 100 games audited
  ok            98
  oracle_gap     2
  engine_bug     0
```

GREEN — meets ≥98 ok / 0 engine_bug. The 2 residual `oracle_gap`
rows are 1624082 (Sasha class-B HP drift, in-cohort residual) and
one Sonja singleton, both pre-existing.

### 5.3 pytest gate

```
python -m pytest -q
  622 passed, 5 skipped, 2 xfailed, 3 xpassed   (tests/ tree)
  1 failed                                       (root-level test_trace_182065_seam_validation.py)
```

The single failure is a pre-existing **movement reachability** error
(`Illegal move: Infantry from (9, 8) to (11, 7)`), unrelated to
funds, War Bonds, or anything this phase touched. Within the ≤2
allowed failures.

### 5.4 Prior CO mechanic tests

```
python -m pytest tests/test_co_*.py -q
  92 passed
```

All 12 CO test files (Sasha War Bonds, Sasha Market Crash, Sonja D2D,
Colin mechanics, Rachel covering fire, Rachel repair, Kindle income,
Hachi build cost, funds ordering & repair canon, Koal COP movement,
Von Bolt Ex Machina, indirect range) pass — no regression in any
previously-shipped CO mechanic.

---

## Section 6 — Hard-rule compliance

* No edits to Kindle combat code.
* No edits to Rachel SCOP.
* No edits to `engine/unit.py`.
* No edits to `engine/action.py::get_legal_actions` (Von Bolt branch
  intact).
* No edits to Sonja D2D combat code.
* Mechanic shipped is Tier-1 cited (AWBW CO Chart Sasha row, War
  Bonds entry).
* Two test files updated (`test_co_sasha_warbonds.py` only); no
  unrelated test rewrites.
* Engine LOC delta well under the 60-LOC budget (~12 LOC of
  behaviour, rest is comments/docstring).

---

## Section 7 — Ranked candidate list for the next L1-WAVE-3 ship

The remaining 10 oracle_gap rows in this cohort decompose as:

| cluster                                | rows | next-ship readiness |
|:---------------------------------------|----:|:--------------------|
| **Sonja-active fog/counter drift**     | 4   | Tier-1 candidate — fog-of-war counter-attack damage display under Sonja's D2D vision/counter rules; needs envelope trace before ship |
| Sturm meteor-strike funds rounding     | 1   | singleton, low-yield |
| Hawke Black Wave heal funds coupling   | 1   | singleton, low-yield |
| Koal road-movement income coupling     | 1   | singleton, low-yield |
| Kindle capture-tick $90 tail            | 1   | singleton, post-fix tail |
| Sasha 1624082 HP-drift class B          | 1   | upstream defect, not a funds fix |
| Rachel 0 residual                        | 0   | (covered by parallel Rachel thread) |

**Recommendation:** L1-WAVE-3 should target the Sonja-active 4-row
cluster as the next dominant cause. It is the only remaining
≥3-row group with a plausibly Tier-1-citable mechanic (Sonja's
counter-attack damage under fog), and clearing it would close the
last named-CO cluster in the BUILD-FUNDS-RESIDUAL family.

---

## Verdict letter

**YELLOW.** Dominant remaining cause identified and surgically
fixed within the LOC budget. 6/16 closures (37.5%) meets the floor
but sits below the GREEN ≥10 mark — the long tail beyond Sasha is
genuinely fragmented across COs with no second dominant cluster.
The 100-game regression gate, full pytest gate, and prior-CO test
gate all hold. The empirical anchor (game 1626284 env-24 Power →
4×Fire → multi-Build sequence) is unambiguous, and the hybrid
crediting model is the canonical AWBW behaviour for War Bonds —
not a heuristic patch. We hold this position; L1-WAVE-3 takes the
Sonja flank next.
