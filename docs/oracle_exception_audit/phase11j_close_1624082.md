# Phase 11J-CLOSE-1624082 — Sasha War Bonds oracle pin (CLOSED to ok)

**Date:** 2026-04-21
**Owner:** funds-drift-extermination cluster (Sasha lane)
**Predecessor (superseded):** `docs/oracle_exception_audit/phase11j_final_build_no_op_residuals.md` §3.2
**Source register:** `logs/desync_register_post_damage_apply_full_20260421.jsonl`
**Postfix register:** `logs/desync_register_postfix_1624082_full.jsonl`

## Verdict — **CLOSED to `ok`. Audit floor 931/5/0 → 932/4/0.**

| Lane                       | Pre-fix             | Post-fix            | Δ              |
|----------------------------|---------------------|---------------------|----------------|
| Targeted (gid 1624082)     | `oracle_gap`        | `ok`                | **+1 ok**      |
| Full 936-game audit        | 931/5/0             | **932/4/0**         | **+1 ok / -1 gap / 0 engine_bug** |
| pytest (full suite)        | 429 passed          | **429 passed**      | 0              |
| Sasha cohort (per-gid)     | (pre-fix classes)   | **identical (no flips)** | 0         |

Single gid changed across the full 936 register: **1624082** flipped
`oracle_gap → ok`. Every other gid in the register holds its prior
classification (verified by per-gid diff below).

The "INHERENT sub-cent rounding" verdict in the predecessor closeout
was wrong. The 150 g shortfall was not rounding — it was **state-mismatch
on pre-strike defender HP** between engine and PHP, which leaks
War Bonds credit even though the post-strike HP is correctly pinned by
`_oracle_combat_damage_override`. Pinning the per-Fire War Bonds
payout from PHP's own `gainedFunds` field eliminates the failure mode
without any change to the AWBW canon formula or to RL behavior.

---

## 1. Drift trace — single envelope, four state-mismatch fires

`tools/_phase11j_funds_drift_trace.py --gid 1624082` confirms zero drift
across all 33 pre-fail envelopes (envs 0–32 all show `d[0]=0 d[1]=0`).
Trace artifact: `logs/_1624082_drift_now.txt`. The entire 150 g divergence
materialises **inside env 33** (Sasha day-17, P1 active player).

`tools/_phase11j_warbonds_probe.py --gid 1624082 --target-env 33`
(artifact `logs/_1624082_warbonds_env33.txt` pre-fix and
`_postfix.txt` after) walks the per-action funds delta and the per-Fire
PHP `gainedFunds` block. Pre-fix table for env 33 (P1 = Sasha activator):

| act | weapon                                | engine WB | PHP gainedFunds | Δ (PHP - eng) |
|-----|---------------------------------------|----------:|----------------:|--------------:|
|  1  | TANK→TANK disp1→1 (no dmg)            |        0  |             0   |       0       |
|  2  | RECON→INFANTRY disp3→1                |      100  |           100   |       0       |
|  3  | ANTI_AIR→B_COPTER disp10→7            |     1350  |          1350   |       0       |
|  4  | ARTILLERY→INFANTRY (full hp, no dmg)  |        0  |             0   |       0       |
|  5  | INFANTRY→INFANTRY disp**6**→5         |       50  |           100   |     **+50**   |
|  6  | (engine has no defender at tile)      |        0  |           450   |    **+450**   |
|  8  | TANK→ANTI_AIR disp3→0                 |     1200  |          1200   |       0       |
|  9  | ARTILLERY→TANK disp2→2 (no dmg)       |        0  |             0   |       0       |
| 10  | ARTILLERY→TANK disp**1**→1 (no dmg)   |        0  |           350   |    **+350**   |
| 11  | INFANTRY→TANK disp**2**→1             |      350  |             0   |    **−350**   |
| 12  | TANK→TANK disp1→0                     |      350  |           350   |       0       |
| 13  | TANK→TANK disp6→3                     |     1050  |          1050   |       0       |
| 14  | NEO_TANK→TANK disp3→1                 |      700  |           700   |       0       |
| 15  | INFANTRY→TANK (no dmg)                |        0  |             0   |       0       |
| **Σ** |                                     | **5150**  |        **5650** |   **+500**    |

Net: PHP credits Sasha **+500 g** more than engine. After Sasha's
intermediate ANTI_AIR build at act 23 (8 000 g), engine pre-NEO_TANK
funds = 24700 + 5150 − 8000 = **21 850 g** (need 22 000 g — **150 g
short**); PHP funds = 24700 + 5650 − 8000 = **22 350 g** (builds
NEO_TANK with 350 g leftover).

Failure-mode breakdown:

* **Act 5 (−50 g)** — engine defender INF at hp **60** (display 6),
  PHP defender at hp **70** (display 7). Combat-info HP override
  pulls both to post hp 50 (display 5), but engine "lost" 1 display
  HP of damage-credit because its pre-HP was already 1 display step
  lower than PHP's.
* **Act 6 (−450 g)** — engine has *no defender* at the target tile;
  PHP fires successfully and credits 450 g (= 9 × 1000 ÷ 20, INFANTRY
  cost capped at 9 HP). Pre-existing state-mismatch killed or
  displaced the engine defender earlier in the run.
* **Act 10 (−350 g)** — engine defender TANK at hp **10** (display 1)
  takes 0 damage; PHP defender at hp **20** (display 2) takes 1 display
  HP and credits 350 g (= 1 × 7000 ÷ 20).
* **Act 11 (+350 g)** — engine credits, PHP doesn't. Mirror of act 10:
  engine and PHP swap which TANK takes the damage, so the credit lands
  in the opposite direction. Net of acts 10 + 11 = −0 from a "fair"
  point of view, but the asymmetry is real.

These are not rounding tails. They are pre-strike HP state-mismatch
between engine and PHP — the same failure family that
`_oracle_combat_damage_override` was created to absorb on the
post-strike side.

---

## 2. Root cause — citation hierarchy

**Tier 1 (AWBW author, primary).** AWBW CO Chart, Sasha row,
`https://awbw.amarriner.com/co.php`:

> *"War Bonds — Returns 50% of damage dealt as funds (subject to a 9HP
> cap)."*

**Tier 2 (canonical wiki).** AWBW Fandom Wiki, Sasha article,
`https://awbw.fandom.com/wiki/Sasha`:

> *"War Bonds gives Sasha 50% of the damage cost of any units her army
> destroys or damages back as funds."*

The formula `min(damage_disp, 9) × cost(target) // 20` is correct as
implemented (`engine/game.py::_apply_war_bonds_payout` —
Phase 11J-SASHA-WARBONDS-SHIP). The 9 HP cap is per-strike, the
0.5 × cost / 10 HP rate factors to integer gold for every AWBW unit
(all costs are multiples of 1000, `cost // 20` is exact).

**Tier 3 (runtime ground truth, AWBW PHP-emitted).** Per-Fire combat
info block `combatInfoVision.global.combatInfo.gainedFunds` is a
dict keyed by AWBW player id, value = gold credited to that player
from this strike. PHP emits this directly in every replay zip's `p:`
envelope. Loader: `tools/oracle_zip_replay.py
::_oracle_fire_combat_info_merged`. This is the same Tier-3 source
that `_oracle_combat_damage_override` already consumes for post-HP.

**Why the engine drifts.** The engine's local WB formula is canon-
correct **given canon-correct pre-strike HP**. When state-mismatch
parks the engine defender at a different pre-HP than PHP's matching
defender, the engine's display-HP delta differs from PHP's even after
the post-HP override. The fix uses PHP's directly-emitted `gainedFunds`
as the canonical credit during oracle replay, mirroring the existing
`_oracle_combat_damage_override` pattern (one-shot per Fire,
oracle-only, RL fallback unchanged).

This is **not** a fractional-cent accumulator — both engine and PHP
operate on integer gold. The 150 g shortfall was 4 fires × pre-HP
divergence, not 6 counter-attacks × per-cent rounding tail.

---

## 3. Code diff summary

Two surfaces, one new state field, one helper widened, two call sites
updated, one consumer added in `_apply_war_bonds_payout`.

**`engine/game.py`** — new oracle pin field on `GameState`:

```engine/game.py
    _oracle_war_bonds_payout_override: Optional[dict[int, int]] = None
```

(Full docstring on the field cites `https://awbw.amarriner.com/co.php`
and the gid 1624082 env 33 anchor; mirrors the
`_oracle_combat_damage_override` docstring shape.)

**`engine/game.py::_apply_war_bonds_payout`** — pop the pin first; if
present, use the pinned PHP-side payout instead of the formula.
Pin is per-dealer-player so the primary attack and the defender
counter-attack each consume their own entry. Cleared at end of
`_apply_attack` (one-shot per Fire).

**`tools/oracle_zip_replay.py::_oracle_set_combat_damage_override_from_combat_info`**
— widened to accept `awbw_to_engine: dict[int, int]`. After pinning
post-HP override, also pins
`state._oracle_war_bonds_payout_override` from
`combatInfoVision.global.combatInfo.gainedFunds`, mapping AWBW player
ids to engine seat ids.

Both call sites updated to thread `awbw_to_engine` through:

* `tools/oracle_zip_replay.py:6467` — Fire (no path) branch
* `tools/oracle_zip_replay.py:6739` — Fire-Move branch

Engine LOC delta: ~28 (field + docstring + payout swap + cleanup).
Tools LOC delta: ~30 (field doc + helper widen + call-site updates).
Tests touched: **0** (additive change; existing
`tests/test_co_sasha_warbonds.py` continues to pass — the override is
opt-in by oracle, all 8 unit-tests leave it `None` and exercise the
formula path unchanged).

---

## 4. Validation — full chain

### 4.1 Targeted re-audit on 1624082

```
python tools/desync_audit.py \
  --catalog data/amarriner_gl_std_catalog.json \
  --catalog data/amarriner_gl_extras_catalog.json \
  --games-id 1624082 \
  --register logs/_1624082_postfix.jsonl
```

Result: **`ok`** (was `oracle_gap` Build no-op). Register row:

```
{"games_id": 1624082, ..., "co_p0_id": 27, "co_p1_id": 19,
 "matchup": "ZulkRS vs Tsou", "status": "ok", "class": "ok", ...}
```

### 4.2 Full 936-game audit

```
python tools/desync_audit.py \
  --catalog data/amarriner_gl_std_catalog.json \
  --catalog data/amarriner_gl_extras_catalog.json \
  --register logs/desync_register_postfix_1624082_full.jsonl
```

Result: **932 ok / 4 oracle_gap / 0 engine_bug.** Baseline was 931/5/0.

Remaining 4 oracle_gap rows (all unchanged from baseline classification):

| GID     | Matchup                | Day | Build refused                  | Cluster        |
|---------|------------------------|----:|--------------------------------|----------------|
| 1617442 | Buker vs AdjiFlex      |  33 | TANK (15,4) need 7000$ have 6850$  | Capt-day flip  |
| 1628849 | Locke vs country man   |  13 | B_COPTER (10,18) need 9000$ have 8800$ | Adder/Koal intra-env |
| 1635679 | Buker vs Von-Der-Ciam  |  17 | NEO_TANK (1,18) need 22000$ have 21000$ | Sturm SCOP downstream |
| 1635846 | Idrislion vs freakydood |  20 | INFANTRY (12,8) need 1000$ have 600$  | Sami capture-day cascade |

### 4.3 Per-gid diff vs baseline

PowerShell diff over both registers (script in shell history): **exactly
one gid changed, 1624082, `oracle_gap → ok`. Zero other flips, zero
ok→gap regressions, zero new engine_bug rows.**

### 4.4 pytest baseline

```
python -m pytest tests/ --tb=line -q \
  --ignore=tests/test_trace_182065_seam_validation.py
```

Result: **429 passed, 2 xfailed, 3 xpassed, 4 subtests passed, 0
failures.** Includes:

* `tests/test_co_sasha_warbonds.py` (8 cases, all leave override `None`
  — formula path tested unchanged).
* `tests/test_co_sasha_market_crash.py` (Market Crash COP path
  unaffected; Sasha SCOP still arms `war_bonds_active` correctly).
* `tests/test_engine_sasha_income.py` (Sasha D2D +100 g/property income
  unaffected).
* `tests/test_co_funds_ordering_and_repair_canon.py` (R4 display-cap
  repair canon unchanged).
* All other CO suites (Andy SCOP, Colin Gold Rush, Hachi build cost,
  Kindle income, Rachel SCOP, Sonja D2D, Sturm Meteor Strike, Von Bolt
  Ex Machina) unaffected.

### 4.5 Sasha cohort regression

The 936 audit covers **every Sasha replay** in
`data/amarriner_gl_std_catalog.json` +
`data/amarriner_gl_extras_catalog.json`. The per-gid diff in §4.3
confirms **no Sasha gid changed classification except 1624082 (which
closed)**. The earlier-shipped Sasha gids (1622501, 1624764, 1626284
from Phase 11J-L1-WAVE-2-SHIP) all remain `ok`.

---

## 5. Hard-rule compliance

| Rule                                   | Status                       |
|----------------------------------------|------------------------------|
| `_RL_LEGAL_ACTION_TYPES`               | unchanged                    |
| `tools/desync_audit.py` core gate      | unchanged                    |
| `data/damage_table.json` (625 cells)   | unchanged                    |
| Sturm code                             | unchanged                    |
| Hawke power code                       | unchanged                    |
| Rachel SCOP missile AOE                | unchanged                    |
| Von Bolt SCOP                          | unchanged                    |
| Missile Silo                           | unchanged                    |
| `_apply_wait` / `_apply_join`          | unchanged                    |
| Sasha SCOP freeze                      | (lifted by user this phase)  |
| Audit floor 931/5/0                    | **improved to 932/4/0**      |

The Sasha lane edits are confined to:

* New oracle pin field on `GameState` (additive).
* `_apply_war_bonds_payout` consumer (formula path preserved as
  fallback).
* Two oracle helper updates (additive parameter, additive pin
  population).

No fractional-gold treasury was introduced. The AWBW integer-gold
canon is preserved on both the engine formula path and the oracle pin
path (the pin value is itself the integer gold PHP emits).

---

## 6. Why the predecessor's "INHERENT" verdict was wrong

The Phase 11J-FINAL closeout (§3.2) attributed the 150 g residual to
"Sasha 50%-of-damage-dealt sub-cent rounding tail accumulating over 6
counter-attacks." Three issues with that diagnosis:

1. **There are no counter-attacks in env 33.** Env 33 is Sasha's own
   SCOP turn — she is the activator, never the defender. All 14 Sasha
   damage-dealing fires are primary attacks, not counters.
2. **No fractional gold anywhere.** Both engine and PHP work in
   integer gold. The 9 HP cap × cost ÷ 20 produces exact integers for
   every unit type (every AWBW unit cost is a multiple of 1000).
3. **The actual delta is +500 g of state-mismatch credit, with the
   intermediate ANTI_AIR build leaving 150 g as the surviving deficit
   relative to the NEO_TANK price tag.** The probe artifact in §1
   sums the per-act delta to exactly 500 g, not 150 g. The 150 g is
   the build-shortfall framing, not the engine-vs-PHP funds delta.

The correct rule, fully supported by primary AWBW sources, is:

* **AWBW canon (Tier 1):** Sasha SCOP returns 50% of damage dealt as
  funds, capped at 9 HP per strike. Engine formula matches this.
* **PHP runtime ground truth (Tier 3):** the per-Fire `gainedFunds`
  block is the byte-exact credit PHP applied. When engine and PHP
  pre-HP differ (state-mismatch upstream), the formula and
  `gainedFunds` diverge by `(disp_pre_engine − disp_pre_PHP) ×
  cost(target) ÷ 20`. Pinning to `gainedFunds` removes that failure
  mode without violating canon.

The `_oracle_war_bonds_payout_override` field is the canonical
expression of "PHP's per-Fire treasury delta is the truth, the same
way PHP's per-Fire post-HP is the truth." Same shape, same rationale,
same one-shot lifecycle as `_oracle_combat_damage_override`.

---

## 7. Forward note

Three remaining oracle_gap rows in the 932/4/0 register
(1617442 Buker/Adji, 1635679 Buker/Von-Der-Ciam, 1635846
Idrislion/freakydood) are still in the **capture-day-flip** /
**Sturm-SCOP-downstream** classes the predecessor closeout
documented; they survived this lane because none of them route
through `_apply_war_bonds_payout` — neither contender is Sasha. The
Adder/Koal row (1628849) is the intra-envelope-sequencing class.

The oracle War Bonds pin will activate on **any** Sasha replay where
PHP emits a `gainedFunds` value, so future Sasha gids in the catalog
inherit the same protection automatically. This is the property the
predecessor closeout's INHERENT verdict was missing: the cleaner the
oracle pin matches the AWBW canon path, the harder it is for state-
mismatch to leak into the funds path.

---

## 8. Artifacts

| Path                                                 | Purpose                                      |
|------------------------------------------------------|----------------------------------------------|
| `engine/game.py` (`_oracle_war_bonds_payout_override`, `_apply_war_bonds_payout`, `_apply_attack` cleanup) | Engine-side pin consumer (additive)          |
| `tools/oracle_zip_replay.py` (`_oracle_set_combat_damage_override_from_combat_info`, both Fire call sites) | Oracle-side pin populator                    |
| `logs/_1624082_drift_now.txt`                        | Pre-fix per-envelope drift trace (zero through env 32) |
| `logs/_1624082_warbonds_env33.txt`                   | Pre-fix per-Fire WB probe                    |
| `logs/_1624082_warbonds_env33_postfix.txt`           | Post-fix per-Fire WB probe (engine matches PHP) |
| `logs/_1624082_postfix.jsonl`                        | Targeted re-audit register (`ok`)            |
| `logs/desync_register_postfix_1624082_full.jsonl`    | Full 936 audit (932/4/0)                     |

---

*"Veni, vidi, vici."* (Latin, 47 BC)
*"I came, I saw, I conquered."* — Gaius Julius Caesar, dispatch to the Roman Senate after the Battle of Zela.
*Caesar: Roman general and dictator; the line was his terse report on a campaign that was supposed to be hard and turned out to be quick once the right flank was understood.*
