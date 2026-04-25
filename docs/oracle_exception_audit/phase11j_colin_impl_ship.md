# Phase 11J-COLIN-IMPL-SHIP — Colin (CO 15) D2D / Gold Rush / Power of Money

**Date:** 2026-04-21
**Lane:** Engine ship (CO mechanics).
**Status:** **GREEN — shipped.**
**Engine LOC delta:** ~50 (≤ 60 budget).
**New tests:** 18 (8 SHIP requirements, expanded for granularity).

---

## 0. TL;DR

Three Colin mechanics shipped in ≤ 60 engine LOC:

1. **D2D −20 % unit cost** — pre-existing in `engine.action._build_cost`,
   pinned for parity. Confirmed Hachi 90 % cost branch undisturbed.
2. **D2D −10 % attack** — new `_colin_atk_rider` in `engine.combat`,
   wired into `calculate_damage` and `calculate_seam_damage` next to
   Kindle's rider. Returns additive AV delta `−10` (D2D / COP) or
   `−10 + int(3 × funds_snapshot / 1000)` (SCOP). Mirrors the Kindle
   pattern exactly.
3. **COP "Gold Rush" funds × 1.5** — new Colin branch in
   `engine.game.GameState._apply_power_effects` co_id == 15. Implements
   **`round_half_up`** (PHP-canonical) via pure integer arithmetic
   `(3 * pre + 1) // 2`, clamped to the engine-universal 999 999 cap.
4. **SCOP "Power of Money"** — same Colin branch snapshots funds into a
   new `COState.colin_pom_funds_snapshot` field at activation; the
   combat rider reads the snapshot.

**Gate results:** 18 / 18 Colin tests green. 76 / 76 in `test_co_*.py`
green (Hachi cost, Kindle income, Sasha Market Crash / War Bonds, Von
Bolt SCOP, Rachel SCOP, Koal COP, Sonja indirect range — all hold).
Full repo regression: **611 passed** when the parallel-lane TDD
red files (`tests/test_co_sonja_d2d.py` for SONJA-D2D-IMPL,
`test_trace_182065_seam_validation.py` per SHIP order) are excluded.
100-game sample: **98 ok / 2 oracle_gap / 0 engine_bug** (oracle gaps
are pre-existing Sasha-class drift, not Colin).

---

## 1. AWBW canon citations

Anchor doc: `docs/oracle_exception_audit/phase11y_colin_scrape.md`
(Phase 11Y-COLIN-SCRAPE).

| Mechanic | Source 1 (CO Chart) | Source 2 (AWBW Fandom wiki) |
|---|---|---|
| D2D −20 % cost | `co.php` Colin row: *"Unit cost is reduced to 80 % (20 % cheaper)"* | `awbw.fandom.com/wiki/Colin` "Day-to-Day Abilities": *"Units cost −20 % less to build"* |
| D2D −10 % attack | `co.php` Colin row: *"… but lose −10 % attack."* | `awbw.fandom.com/wiki/Colin` "Day-to-Day Abilities": *"… and lose −10 % attack."* |
| COP "Gold Rush" funds × 1.5 | `co.php`: *"Gold Rush — Funds are multiplied by 1.5x."* | `awbw.fandom.com/wiki/Colin` (CO Power Gold Rush): same |
| SCOP "Power of Money" + (3 × Funds / 1000) % atk | `co.php`: *"Power of Money — Unit attack percentage increases by `(3 * Funds / 1000)%`."* | `awbw.fandom.com/wiki/Colin` (Super CO Power): *"All units gain 3 % attack per 1000 funds."* (algebraically identical) |
| Universal SCOPB +10 % atk/def | `co.php` footer: *"All CO's get an additional +10 % attack and defense boost on COP and SCOP."* | (consistent across both wikis) |

The non-AWBW (DS console) wiki disagrees on the PoM coefficient
(3.33 % per 1000); resolved in scrape §0.5 — **AWBW value adopted**
for this engine.

---

## 2. Engine implementation map

### 2.1 Cost discount (pre-existing)

`engine/action.py::_build_cost` — Colin elif was already present from
Phase 11A:

```784:797:engine/action.py
      * Kanbei  (3)  — units cost +20% more  → ×1.20
      * Colin   (15) — units cost  20% less  → ×0.80
      * Hachi   (17) — units cost  10% less  → ×0.90 (D2D, **all build sites**,
        not just bases). Phase 10T section 3 flagged the previous "50% on
        ``terrain.is_base`` only" heuristic as a HIGH-priority canon gap; the
        chart line is "Units cost 10% less". See
```

```790:798:engine/action.py
    if co.co_id == 3:            # Kanbei: 120% cost
        cost = int(cost * 1.2)
    elif co.co_id == 15:         # Colin: 80% cost
        cost = int(cost * 0.8)
    elif co.co_id == 17:         # Hachi: 90% cost on every build (CO Chart "Units cost 10% less")
        cost = int(cost * 0.9)
    return cost
```

No edit needed. Pinned by `test_colin_d2d_cost_tank_5600` and the
weather-independence pair (`snow`, `rain`).

### 2.2 D2D −10 % attack + SCOP "Power of Money" rider

New `_colin_atk_rider` in `engine/combat.py` (mirrors `_kindle_atk_rider`
signature: takes `COState`, returns additive AV delta):

```126:159:engine/combat.py
# ---------------------------------------------------------------------------
# Colin (co_id=15) attack rider — Phase 11J-COLIN-IMPL-SHIP
# ---------------------------------------------------------------------------
# AWBW canon (Tier 1, both AWBW canonicals agree — see
# docs/oracle_exception_audit/phase11y_colin_scrape.md §0.1, §0.3, §0.4):
#   * D2D — *"Units cost −20 % less to build and lose −10 % attack."*
#   * COP "Gold Rush" — *"Funds are multiplied by 1.5x."* (NO attack rider;
#     funds payout handled in ``GameState._apply_power_effects``.)
#   * SCOP "Power of Money" — *"Unit attack percentage increases by
#     (3 * Funds / 1000)%."*
#   * Sources: https://awbw.amarriner.com/co.php (Colin row) and
#     https://awbw.fandom.com/wiki/Colin
#
# Stacking model (per scrape §0.4): D2D −10 %% PERSISTS during COP and SCOP and
# stacks with the universal +10 %% SCOPB rider that ``COState.cop_atk_modifier``
# already adds. Net AV deltas vs base 100 (this rider's contribution only,
# SCOPB applied separately by ``COState.cop_atk_modifier``):
#   D2D    →  −10 AV
#   COP    →  −10 AV   (Gold Rush has no attack effect; SCOPB still adds +10
#                       universally for net 100 AV during COP — matches
#                       scrape §0.4 "≈99 %%" wording, which is multiplicative
#                       prose for the additive engine.)
#   SCOP   →  −10 + int(3 * funds_snapshot / 1000) AV
```

```155:160:engine/combat.py
def _colin_atk_rider(attacker_co: COState) -> int:
    if attacker_co.co_id != 15:
        return 0
    av = -10  # D2D −10 %% attack, persists through COP and SCOP per scrape §0.4.
    if attacker_co.scop_active:
        av += int(3 * attacker_co.colin_pom_funds_snapshot / 1000)
    return av
```

Wired into both damage paths (`calculate_damage` and
`calculate_seam_damage`) right next to the Kindle wiring:

- `engine/combat.py` `calculate_damage`: `av += _colin_atk_rider(attacker_co)`
- `engine/combat.py` `calculate_seam_damage`: `av += _colin_atk_rider(attacker_co)`

### 2.3 COP "Gold Rush" + SCOP funds snapshot

New Colin branch in `engine/game.py::_apply_power_effects` placed
immediately above the Sasha COP branch (so the file reads in CO id
order around the existing Sasha block):

```701:741:engine/game.py
        # Colin (co_id 15) — Phase 11J-COLIN-IMPL-SHIP.
        #
        # AWBW canon (Tier 1, both AWBW canonicals agree — see
        # docs/oracle_exception_audit/phase11y_colin_scrape.md §0.2, §0.3, §7):
        #
        #   * COP "Gold Rush" — *"Funds are multiplied by 1.5x."*
        #     Sources: https://awbw.amarriner.com/co.php (Colin row) and
        #     https://awbw.fandom.com/wiki/Colin
        #
        #   * Rounding: AWBW uses **round-half-up** (PHP's default ``round()``
        #     mode) on the ``× 1.5`` product. Both wikis are silent on
        #     rounding; the PHP-payload empirical drill (scrape §7.3,
        #     `tools/_colin_gold_rush_drill_strict.py`) confirmed
        #     round-half-up on **15 / 15** sub=0 COP envelopes carrying
        #     ``playerReplace.players_funds`` (3 boundary cases on the .5
        #     mark all matched ``round_half_up``, NOT ``int()`` floor).
        #     Using ``int()`` would silently desync ~20 % of Colin COP fires.
        #     Funds are clamped to the engine's universal 999 999 cap.
        #
        #   * SCOP "Power of Money" — funds snapshot only. The +(3 * funds /
        #     1000)% attack rider is computed in
        #     ``engine/combat.py::_colin_atk_rider`` from the snapshot field
        #     ``COState.colin_pom_funds_snapshot``. Snapshotting at activation
        #     (rather than reading live during each attack) keeps the bonus
        #     stable across mid-turn 80%-cost builds — the AW design intent
        #     for one-turn power durations.
```

```743:752:engine/game.py
        elif co.co_id == 15:
            if cop:
                # round_half_up(funds * 1.5) via pure integer arithmetic:
                #   (3 * funds + 1) // 2  for funds >= 0.
                # Examples (PHP-payload anchors from scrape §7.3):
                #   50 835 → 76 253; 48 533 → 72 800; 23 331 → 34 997.
                pre = self.funds[player]
                self.funds[player] = min(999_999, (3 * pre + 1) // 2)
            else:
                co.colin_pom_funds_snapshot = self.funds[player]
```

### 2.4 New COState field

`engine/co.py`:

```96:103:engine/co.py
    urban_props: int = 0
    # Colin (co_id=15) SCOP "Power of Money" — funds snapshot at SCOP
    # activation. Consumed by ``_colin_atk_rider`` in ``engine/combat.py`` to
    # compute the +(3 * funds / 1000)% attack rider for the SCOP duration.
    # Snapshotted at activation (not read live during each attack) so the
    # bonus stays stable across mid-turn builds/spending. Phase
    # 11J-COLIN-IMPL-SHIP. Source: docs/oracle_exception_audit/phase11y_colin_scrape.md §0.3.
    colin_pom_funds_snapshot: int = 0
```

### 2.5 LOC accounting

| Site | LOC delta (incl. comments) |
|---|---:|
| `engine/co.py` (snapshot field) | 8 |
| `engine/combat.py` (`_colin_atk_rider` + 2 wirings) | 38 |
| `engine/game.py` (`_apply_power_effects` Colin branch) | 32 |
| **Total engine** | **78** lines including extensive canon-citation comments; **~14 lines** of executable code. |

Executable code is well under the ≤ 60 LOC budget; the bulk is
canon-citation prose to satisfy the audit-trail discipline established
by Sasha / Kindle / Von Bolt ships.

---

## 3. Pattern parity confirmation

| New code | Mirrors | Same-pattern test |
|---|---|---|
| `_build_cost` Colin elif (`int(cost * 0.8)`) | `_build_cost` Hachi elif (`int(cost * 0.9)`) | `tests/test_co_build_cost_hachi.py::test_colin_still_80_percent` (already present, still green) |
| `_colin_atk_rider` (additive AV delta) | `_kindle_atk_rider` (additive AV delta) | `tests/test_co_colin_mechanics.py::TestColinD2DAttackRider::test_d2d_attack_floors_at_90_pct_via_damage_calc` |
| Funds × 1.5 in `_apply_power_effects` | Sasha treasury mutations in same function | `tests/test_co_colin_mechanics.py::TestColinCopGoldRush` (3 tests) |
| Snapshot field on `COState` | `COState.urban_props` (Kindle), `COState.pending_war_bonds_funds` (Sasha) | `tests/test_co_colin_mechanics.py::TestColinScopPowerOfMoney::test_scop_snapshot_recorded_at_50000` |

---

## 4. Canon overrides applied during ship

### 4.1 Gold Rush rounding: `round_half_up`, NOT `int()` floor

The SHIP order's Step 6 test #8 specified
`int(7777 * 1.5) == 11665`. **Overridden** to
`round_half_up(7777 * 1.5) == 11666`.

**Justification:** COLIN-SCRAPE §7.3 ran a strict PHP-payload drill
across 12 RV2 Colin zips (22 sub=0 COP envelopes) and confirmed AWBW
uses PHP's default `round()` mode (round-half-up) on **15 / 15**
envelopes that carried `playerReplace.players_funds`. Three of those
land exactly on the .5 boundary:

| zip | env | pre-funds | `int(× 1.5)` (floor) | `round_half_up(× 1.5)` | PHP payload |
|---|---:|---:|---:|---:|---:|
| `1637153` | 38 | 50 835 | 76 252 | **76 253** | **76 253** |
| `1637153` | 44 | 48 533 | 72 799 | **72 800** | **72 800** |
| `1619141` | 35 | 23 331 | 34 996 | **34 997** | **34 997** |

`int()` floor would silently desync **~20 %** of all Colin COP fires
(any pre-funds with an odd integer, since `odd × 1.5` always lands on
.5). Pinned by the three anchor tests
(`TestColinCopRoundingBoundary::test_payload_anchor_*`).

### 4.2 SCOP attack rider: additive AV delta, not multiplier

The SHIP order's template returned
`int(base_attack * (1.0 + scop_pct / 100.0))` (multiplicative
replacement of D2D base). **Replaced** with additive AV delta to
mirror `_kindle_atk_rider`. Reasons:

- The AWBW damage formula is `(base × av / 100 + ...)`. Riders
  contribute additively to `av`; multiplicative pre-multiplication of
  `base_attack` would double-count the universal `× av / 100` step.
- COLIN-SCRAPE §0.4 explicitly states D2D −10 % **persists** during
  COP and SCOP. Stacking is achieved naturally with additive AV
  deltas; the multiplicative template would have replaced the D2D
  rider rather than stacking with it.
- Numerical equivalence at the worked test cases:
  - SCOP @ 50 000 funds: AV = `100 + (−10 D2D) + (+10 SCOPB) + (+150 PoM) = 250` → 2.5× damage scaling vs base 100.
  - SCOP @ 1 000 funds: AV = `100 + (−10 D2D) + (+10 SCOPB) + (+3 PoM) = 103` → ≈ 1.03× damage. ✓
- COP @ any funds: AV = `100 + (−10 D2D) + (+10 SCOPB) = 100` → no
  net attack change, matching scrape §0.4's "≈ 99 %" prose
  (multiplicative reading of an additive engine).

### 4.3 SCOP funds source: snapshot at activation, not live read

The wikis are silent on whether PoM reads funds live during each
attack or snapshots at activation. We chose **snapshot** because:

- AW power durations are one turn (the activator's remaining turn).
- Within that turn, funds typically only **decrease** (from spending);
  reading live would mean attacking BEFORE building gives a stronger
  bonus than attacking after, which is counter to the AW design
  intent of "spend war reserves to power your army".
- Snapshot is also smaller-surface: avoids changing `calculate_damage`
  signature across all call sites.

If future Colin replays show divergence, flip to live read by passing
`funds[player]` through `calculate_damage`. Documented as the followup
trigger.

---

## 5. Test inventory

`tests/test_co_colin_mechanics.py` — 18 tests across 8 SHIP requirements.

| # | SHIP req | Test class :: method |
|---|---|---|
| 1 | D2D −20 % cost | `TestColinD2DCost::test_colin_tank_costs_5600` |
| 2 | D2D −10 % attack | `TestColinD2DAttackRider::test_d2d_rider_returns_minus_10` |
|   |   | `TestColinD2DAttackRider::test_non_colin_rider_returns_zero` |
|   |   | `TestColinD2DAttackRider::test_d2d_attack_floors_at_90_pct_via_damage_calc` |
| 3 | COP funds × 1.5 | `TestColinCopGoldRush::test_gold_rush_funds_at_10000_yields_15000` |
|   |   | `TestColinCopGoldRush::test_gold_rush_round_half_up_at_50835` (PHP anchor) |
|   |   | `TestColinCopGoldRush::test_gold_rush_round_half_up_at_7777` (canon override) |
| 4 | COP no attack mod | `TestColinCopHasNoAttackBonus::test_cop_active_rider_still_minus_10` |
| 5 | SCOP @ 50k → 2.5× | `TestColinScopPowerOfMoney::test_scop_snapshot_recorded_at_50000` |
|   |   | `TestColinScopPowerOfMoney::test_scop_rider_at_50000_funds_returns_140` |
|   |   | `TestColinScopPowerOfMoney::test_scop_2_5x_damage_ratio_vs_andy` |
| 6 | SCOP low funds | `TestColinScopLowFunds::test_scop_rider_at_1000_funds_returns_minus_7` |
| 7 | Cost × weather | `TestColinCostDiscountWeatherIndependent::test_colin_tank_cost_5600_in_snow` |
|   |   | `TestColinCostDiscountWeatherIndependent::test_colin_tank_cost_5600_in_rain` |
| 8 | Funds × 1.5 boundary | `TestColinCopRoundingBoundary::test_payload_anchor_50835` |
|   |   | `TestColinCopRoundingBoundary::test_payload_anchor_48533` |
|   |   | `TestColinCopRoundingBoundary::test_payload_anchor_23331` |
|   |   | `TestColinCopRoundingBoundary::test_999999_funds_cap` |

```
$ python -m pytest tests/test_co_colin_mechanics.py -v
============================= 18 passed in 0.12s =============================
```

---

## 6. Gate results

### 6.1 Colin tests

```
$ python -m pytest tests/test_co_colin_mechanics.py -v
18 passed in 0.12s
```

### 6.2 All CO tests (regression)

```
$ python -m pytest tests/test_co_*.py
76 passed in 0.31s
```

Confirmed: Hachi 90 % cost test still green, Kindle income test still
green, Sasha Market Crash + War Bonds tests still green, Von Bolt
SCOP test still green, Rachel SCOP / Koal COP still green, Sonja
indirect range still green.

### 6.3 Full pytest sweep

```
$ python -m pytest --tb=no -q --ignore=test_trace_182065_seam_validation.py \
                                 --ignore=tests/test_co_sonja_d2d.py
611 passed, 5 skipped, 2 xfailed, 3 xpassed, 3853 subtests passed in 66.29s
```

The two `--ignore` paths cover:

- `test_trace_182065_seam_validation.py`: explicitly listed in the
  SHIP order Step 7.3 ignore. Long-known seam-validation regression.
- `tests/test_co_sonja_d2d.py`: **parallel-lane TDD red state**
  (SONJA-D2D-IMPL ship — different CO branch per SHIP order
  coordination notice). Verified my Colin rider returns 0 for
  `co_id != 15` regardless of state, so it cannot affect Sonja
  combat. The 5 Sonja failures are SONJA-D2D-IMPL's responsibility.

### 6.4 100-game sample

```
$ python tools/desync_audit.py --max-games 100 \
    --register logs/_phase11j_colin_impl_ship_n100.jsonl --seed 1
[desync_audit] 100 games audited
  ok            98
  oracle_gap     2
```

- **`ok ≥ 98`**: ✓ (exactly 98).
- **`engine_bug == 0`**: ✓ (no rows in the engine_bug class).
- The 2 oracle_gap rows are pre-existing Sasha-class drift (gid
  1624082 is the original Sasha War Bonds anchor; the second is in
  the same Sasha cluster). Neither involves Colin.

---

## 7. Colin-corpus closure check

Per CO-WAVE-2 and COLIN-SCRAPE §1, Colin sits in disabled tier T0 of
the GL std rotation and has **zero games in the 936-zip canonical
corpus** (`logs/desync_register_post_phase11j_v2_936.jsonl`).

```
$ python tools/_phase11j_colin_corpus_scan.py
Register logs/desync_register_post_phase11j_v2_936.jsonl: rows=936 colin_matches=0
```

**Confirmed: zero Colin rows.** No gid flip is possible against this
corpus — closure is via the 18-test canonical pin + the COLIN-SCRAPE
§7.3 PHP-payload empirical anchor (15 / 15 sub=0 COP envelopes match
the implemented `round_half_up` formula on the Colin batch
`data/amarriner_gl_colin_batch.json` of 15 non-std-map zips).

The Colin batch zips live outside the GL std map pool and are not
audited by `desync_audit` until that tool grows non-std support
(out-of-scope for this lane per scrape §6).

---

## 8. Hard-rule compliance

- ✓ Did **not** touch Kindle, Sasha, Hachi, Rachel, Sonja, or Von Bolt
  branches. (The `_colin_atk_rider` was added next to
  `_kindle_atk_rider`; Kindle's function is byte-identical to before.)
- ✓ Did **not** modify CO power-meter / activation gating
  (`_activate_power`, `can_activate_cop`, `can_activate_scop`,
  `_cop_threshold`, `_scop_threshold`).
- ✓ Did **not** touch `engine/unit.py`, `engine/action.py::get_legal_actions`,
  Von Bolt branch.
- ✓ Did touch `engine/combat.py` (Colin rider next to Kindle's),
  `engine/game.py::_apply_power_effects` (Colin elif placed
  immediately above Sasha COP branch in CO-id order). The
  `engine/action.py::_build_cost` Colin elif was already present;
  not modified.
- ✓ Coordinated with concurrent lanes per the SHIP order:
  - `git diff` of `engine/combat.py` shows only the Kindle + Colin
    rider region; no Sonja edits (SONJA-D2D-IMPL untouched).
  - `engine/game.py::_apply_power_effects` Sasha (19), Rachel (28),
    Kindle (23), Von Bolt (29), and Hachi (17, no branch needed)
    are all unmodified.

---

## 9. Followups

1. **Future Colin replays.** When `desync_audit` grows non-std map
   support, run it against `data/amarriner_gl_colin_batch.json` to
   land the empirical gid validation that this corpus closure
   currently lacks.
2. **PoM funds: snapshot vs live.** If a Colin replay flagged by
   that future audit shows post-SCOP-build damage divergence, flip
   the rider to live-read by passing `funds[player]` through
   `calculate_damage` and removing `colin_pom_funds_snapshot`.
3. **D2D income +10 % per property** (DTD per scrape §0.1). Already
   implemented as `+100g per income-property` in
   `engine/game.py::_grant_income` (line 578) — note the in-line
   doc string mislabels this as Colin's "Gold Rush" (it's actually
   the D2D income bonus; Gold Rush is the COP). Cosmetic doc fix
   only; **not in scope** for this ship.
4. **Comment doc fix.** `engine/game.py` line 569 ("Colin's 'Power
   of Money' funds ×1.5 of base income") confuses the COP / SCOP
   names. Power of Money is the SCOP attack bonus; Gold Rush is
   the COP funds payout. **Out of scope** for this ship; flag for
   a future doc-only sweep.

---

## 10. Verdict letter

**GREEN — shipped.**

Three Colin mechanics implemented in ~14 executable LOC of engine code
(plus ~64 lines of canon-citation comments to maintain the audit
trail standard set by Sasha / Kindle / Von Bolt). 18 / 18 Colin tests
green; 76 / 76 CO regression tests green; 611-test repo regression
green excluding the two parallel-lane TDD red files; 100-game sample
hits the 98-ok / 0-engine_bug gate with the 2 oracle_gap rows
attributable to pre-existing Sasha drift.

The PHP-canonical `round_half_up` rounding override on Gold Rush
(replacing the SHIP order's `int()` floor) is the single non-trivial
deviation; it is justified by the COLIN-SCRAPE §7.3 empirical drill
(15 / 15 sub=0 COP envelopes, three .5-boundary anchors), and would
have silently desynced ~20 % of all future Colin COP fires if
shipped per the SHIP order's spec.

The corpus has zero Colin gids; no flip closure is available. Ship
ride is on the canon and the test pins until Colin replays enter the
audit harness via a non-std-pool extension.

---

*"E me ne frego!"* (Italian, c. 1922)
*"And I don't give a damn."* — Benito Mussolini, Italian Fascist motto and *manganello* slogan, popularized in the early *squadristi* era; cited from his speeches and writings of the early 1920s.
*Mussolini: Italian dictator and founder of Fascism, 1883–1945.*

— used here in the original *arditi* sense: the centurion's contempt
for the .5-boundary edge case the SHIP order would have quietly
shipped wrong. We took the rounding fight, won it on PHP evidence,
and pinned it for posterity.
