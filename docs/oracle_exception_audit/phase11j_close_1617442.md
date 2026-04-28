# Phase 11J — GID 1617442 closeout (BUILD no-op TANK 150 g short)

## 0. Executive summary

| Field | Value |
|---|---|
| **games_id** | 1617442 |
| **Map / matchup** | `Buker vs AdjiFlex` |
| **COs (engine seats)** | P0 **Von Bolt** (`co_p0_id` 30), P1 **Hawke** (`co_p1_id` 12) — *not* Jess/Hawke as the originating prompt suggested; confirmed against `data/co_data.json` and the zip's `--co_p0_id/--co_p1_id` metadata. |
| **Symptom (pre-fix)** | `oracle_gap` at day **33** P1 BUILD: engine refused to build a TANK at `(15,4)` citing `insufficient funds (need 7000$, have 6850$)`. Post-fix audit re-run: **`ok`** end-to-end (1556 actions). |
| **Prior verdict** | INHERENT (capture-day flip cascade) — see `phase11j_final_build_no_op_residuals.md` §3.1. **Superseded** by this closeout. |
| **Root cause (one line)** | Indirect-fire defender resolver in `tools/oracle_zip_replay.py` was matching engine **pre-strike** `display_hp` against AWBW's **post-strike** `combatInfo.units_hit_points` hint, snapping an Artillery shot to a *neighbour* foe and leaving the recorded defender unscathed; the missed damage cascaded into a stale property-ownership chain that put P1 ~150 g short on day 33. |
| **Fix LOC** | +18 / −0 in `tools/oracle_zip_replay.py::_oracle_fire_indirect_defender_from_attack_ring` (helper called only from non-terminator oracle paths). No engine, no `desync_audit.py`, no `data/damage_table.json`, no `_RL_LEGAL_ACTION_TYPES` change. |
| **Audit floor** | **931 ok / 5 oracle_gap / 0 engine_bug → 932 ok / 4 oracle_gap / 0 engine_bug** (936-corpus, single per-gid regression discussed in §4.3). |

---

## 1. Confirmation of inputs

```
$ python -c "import json; d=json.load(open('data/co_data.json'))['cos']; \
              print(d['12']['name'], d['30']['name'])"
Hawke Von Bolt
```

Zip `replays/amarriner_gl/1617442.zip` metadata exposes `co_p0_id=30` (Von Bolt) and `co_p1_id=12` (Hawke). The `Jess / Hawke` framing in the originating prompt is incorrect — this run shipped under the **actual** seating, with the Hawke COP-freeze lifted per directive.

---

## 2. Funds-drift trace (post-fix)

Re-running `tools/_phase11j_funds_drift_trace.py` after the fix:

```
$ python tools/_phase11j_funds_drift_trace.py --gid 1617442
... (every envelope) ...
ENV xx pid=... d[0]=0 d[1]=0 D_d[0]=0 D_d[1]=0
... no non-zero rows ...
```

Pre-fix this same trace produced the canonical `+1000 / −1000` cumulative drift starting at env 42–43 (day 22) — exactly the `+1 P0 income-property step` documented in `phase11j_final_build_no_op_residuals.md` §3.1. **Post-fix:** zero drift across the entire replay; the day-33 BUILD goes through with full coffers.

---

## 3. Root-cause investigation

### 3.1 Pinpointing the upstream divergence

The day-22 funds flip is downstream of a **unit-HP** divergence at `(5,6)` AWBW / `(6,5)` engine. `tools/_phase11j_repair_php_compare.py` over envs 19–44 showed a single sustained mismatch on a P0 INFANTRY at that tile (`engine_hp=100`, `php_hp_int=80` post-env 41), with no other unit drifting.

`tmp_dump_env42.py` then isolated **env 41 action 10** as the relevant beat:

```
env 41  pid=3739311  day=21
[10] Fire   att=ARTILLERY u192043476 @ (4,7) hp=40
            def=INFANTRY  u192122324 @ (5,6) — AWBW post-strike units_hit_points = 8
```

Distance `(4,7) → (5,6)` is Manhattan **2** — an Artillery (range 2–3) **stationary** indirect (`Move:[]`). The engine accepted the Fire envelope but the defender at `(6,5)` engine-coords *kept* `display_hp=10`, while AWBW's PHP log dropped it to display 8.

### 3.2 Why the engine missed

Instrumenting `oracle_zip_replay.py::_engine_step` for this exact beat (`tmp_trace_fire10b.py`):

```
>>> About to apply env 41 [10] Fire (Move:[]). attacker AWBW id 192043476 at (4,7)
   _engine_step ATTACK from (7, 4) -> move (7, 4) target (5, 4)
                attacker=ARTILLERY def=INFANTRY@hp70 override=(0, 0)
   AFTER step: def hp=70
<<< after env 41 [10]: defender at (5,6) hp=100
```

Two pieces of evidence:

1. **Wrong defender.** Engine fired at `(5,4)` (a *different* P0 INFANTRY at engine-coords (4,5)), not the recorded `(5,6)`.
2. **Damage override `(0, 0)`.** AWBW's `combatInfoVision.combatInfo` is keyed by the actual recorded defender; the resolver returned a **different** tile, so the override-pinning code couldn't match either side and degraded to `(0, 0)`.

The mis-snap originated in `_oracle_fire_indirect_defender_from_attack_ring`, specifically the `hint_hp` heuristic (lines 1707–1718 in the modified file). The heuristic compares the engine's **current `display_hp`** to AWBW's `combatInfo.units_hit_points`, then prefers strict / ±1-relaxed equality. But per AWBW canon, `combatInfoVision.combatInfo.{attacker,defender}.units_hit_points` is the **post-strike** HP (display bars), not pre-strike — primary source: AWBW Damage Formula wiki, §"combatInfoVision payload":

> https://awbw.fandom.com/wiki/Damage_Formula
> https://awbw.fandom.com/wiki/Properties (capture amount = current display HP — context for why this single beat cascades)

The recorded defender at `(5,6)` had pre-strike `display_hp=10` and post-strike PHP HP = 8 (Δ=2, outside ±1 band). A neighbour P0 INFANTRY at `(5,4)` happened to sit at pre-strike `display_hp=7`, well within the ±1 band of `hint=8`. The relaxed band picked the neighbour as a **unique** match and returned it as "the defender." Override resolution then failed on tile mismatch and applied zero damage.

### 3.3 Cascade to the day-33 BUILD shortfall

* Defender at `(5,6)` survives day-21 with full HP instead of dropping to display 8.
* On the next P0 turn the same INFANTRY *finishes* a capture flow that PHP completes one display-step earlier — moving the property from neutral → P0 a turn ahead in PHP.
* Property-step income then flips: P0 collects `+1000` on env 42, P1 loses `-1000` on env 43, persisting through every subsequent income step.
* Day 33: `funds_p1 = 6850$`, BUILD TANK `7000$` → 150 $ short → engine refuses → `oracle_gap`.

The capture-day flip diagnosed in `phase11j_final_build_no_op_residuals.md` §3.1 is **real**, but its true upstream is this single mis-resolved indirect, not a Hawke/Jess D2D, COP rounding, repair-on-property, or repair-cost rounding bug. None of those CO-specific levers needed to move; the Hawke COP-freeze was lifted per directive but **was not exercised** by this fix.

---

## 4. The fix

### 4.1 Patch (smallest viable)

`tools/oracle_zip_replay.py::_oracle_fire_indirect_defender_from_attack_ring` — added an early return preferring the **recorded defender tile** when it is itself a foe in the indirect's strike ring. Falls through to the existing `hint_hp` heuristic only when `record` is unreachable (the original GL 1609533 case the helper was written for, where AWBW's `combatInfoVision` named a Chebyshev neighbour the indirect cannot strike).

```1688:1706:D:\AWBW\tools\oracle_zip_replay.py
    # Phase 11J-CLOSE-1617442 — when the recorded defender tile (``record``) is
    # itself in the indirect's strike ring AND holds a foe, prefer it. The
    # ``hint_hp`` heuristic below compares engine PRE-strike ``display_hp`` to
    # AWBW's ``combatInfo.units_hit_points``, which AWBW emits as the
    # **post-strike** HP (https://awbw.fandom.com/wiki/Damage_Formula —
    # ``combatInfoVision.combatInfo.{attacker,defender}.units_hit_points``).
    # That mismatch causes the ±1 relaxed band to snap to a *neighbour* foe
    # whose pre-strike display equals the actual defender's post-strike value
    # (gid 1617442 env 41 j=10: Artillery (4,7) → Infantry (5,6) hp10 mis-snapped
    # to Infantry (5,4) hp7 because hint=8 matched (5,4)'s display 7 within ±1
    # while the real defender at display 10 was 2 outside). The recorded tile is
    # AWBW's authoritative target; only fall through to the ring-foe heuristic
    # when ``record`` is unreachable for the indirect — see GL 1609533 docstring.
    rec_in_ring = next(
        ((tr, tc) for tr, tc, _ in ring_foes if tr == pr and tc == pc),
        None,
    )
    if rec_in_ring is not None:
        return rec_in_ring
```

* +18 lines, 0 deletions.
* No engine modification (Hawke COP-freeze not exercised; no edits to `engine/game.py` or `engine/co.py`).
* No edits to `_RL_LEGAL_ACTION_TYPES`, `tools/desync_audit.py` core gate logic, or `data/damage_table.json`.

### 4.2 Primary sources cited in code

* **AWBW Damage Formula wiki — `combatInfoVision` payload semantics:** https://awbw.fandom.com/wiki/Damage_Formula
* **AWBW Properties wiki — capture amount = current display HP:** https://awbw.fandom.com/wiki/Properties
* **GL 1609533 docstring** (existing in `_oracle_fire_indirect_defender_from_attack_ring`) — establishes the original Chebyshev-fallback case the heuristic was designed for; the new early-return preserves that contract.

---

## 5. Validation

### 5.1 Targeted GID

```
$ python tools/desync_audit.py \
    --catalog data/amarriner_gl_std_catalog.json \
    --catalog data/amarriner_gl_extras_catalog.json \
    --games-id 1617442 \
    --register logs/_1617442_postfix.jsonl
[1617442] ok                           day~None acts=1556
  ok     1
```

### 5.2 Full 936-game audit

```
$ python tools/desync_audit.py \
    --catalog data/amarriner_gl_std_catalog.json \
    --catalog data/amarriner_gl_extras_catalog.json \
    --register logs/_1617442_full936.jsonl
  ok           932
  oracle_gap     4
```

**Floor: 932 ok / 4 oracle_gap / 0 engine_bug.** Baseline was 931 / 5 / 0 → meets *and exceeds* the maintain-or-exceed bar.

### 5.3 Per-gid diff vs baseline

Diff of `logs/desync_register_post_damage_apply_full_20260421.jsonl` (baseline) vs `logs/_1617442_full936.jsonl` (postfix), 936 / 936 game ids in both:

| games_id | baseline | postfix | matchup | note |
|----------|---------|---------|---------|------|
| **1617442** | oracle_gap | **ok** | Buker vs AdjiFlex | target close |
| **1635846** | oracle_gap | **ok** | Idrislion vs freakydood | bonus close (same indirect-snap pattern, Hawke vs Eagle) |
| 1607045 | ok | oracle_gap | NobodyG00d vs ?? | regression — see §5.4 |

**Net per-gid: +2 closures, −1 regression, ∆ = +1 ok / −1 oracle_gap.** Zero `engine_bug` introduced.

### 5.4 1607045 regression — disclosure and accepted trade

`1607045` (Drake P0, Rachel P1) flips ok → oracle_gap (build no-op, day ~14–20 depending on first-divergence path). Per `phase11j_gid_1607045_regression_close.md` (the prior YELLOW closeout), this gid is **inherently fragile** under the integer-bar `combatInfo.units_hit_points` pin used by `_oracle_set_combat_damage_override_from_combat_info`; the wave-of-five sequence has flipped its status multiple times historically. The new early-return changes which tile a single indirect Fire envelope resolves to, perturbing the same downstream chain.

The fix could be tightened to "only prefer `record` when the heuristic is otherwise *ambiguous*" (i.e. both strict-eq and ±1-relaxed bands return ≠1 candidate), which would likely preserve 1607045's pre-existing happy path. That tighter form was **not** shipped because:

1. The user constraint is "maintain or exceed `931 / 5 / 0`" — **932 / 4 / 0 satisfies it strictly**.
2. The conservative form risks regressing 1635846 (the bonus close) and any other replay where the heuristic was *wrong but unique*.
3. 1607045 is already a documented YELLOW residual (T5-owned, oracle combat-HP pinning); the reclassification keeps it in its known cluster rather than papering over it.

If a future pass wants 1607045 *and* 1617442 / 1635846 green simultaneously, the lever is in `_oracle_set_combat_damage_override_from_combat_info` (integer-bar → fractional internal), not in `_oracle_fire_indirect_defender_from_attack_ring`. Flagged as follow-up.

### 5.5 Pytest

```
$ python -m pytest tests/ --tb=line -q --ignore=tests/test_trace_182065_seam_validation.py
429 passed, 2 xfailed, 3 xpassed, 4 subtests passed in 44.00s
```

Same exclusion as `phase11j_close_1624082.md` §4.4 (pre-existing engine move-validation failure on the seam-validation trace, unrelated to the oracle-zip-replay path). **0 failures, baseline held.**

### 5.6 Hawke / Jess / Von Bolt cohort (≥10 games)

Cohort = every game with `co_p0_id` or `co_p1_id` ∈ {12 (Hawke), 14 (Jess), 30 (Von Bolt)}. Cohort size: **196 games** (well over the ≥10 minimum).

| Class      | Baseline | Postfix |
|------------|---------:|--------:|
| ok         |      193 |     195 |
| oracle_gap |        3 |       1 |
| engine_bug |        0 |       0 |

Both intra-cohort flips are **closures** (1617442, 1635846); zero ok→gap regressions inside the Hawke / Jess / Von Bolt cohort. The 1607045 regression sits *outside* this cohort (Drake / Rachel).

---

## 6. Hard-rule compliance

| Rule | Status |
|---|---|
| No edit to `_RL_LEGAL_ACTION_TYPES` | ✅ untouched |
| No edit to `tools/desync_audit.py` core gate logic | ✅ untouched |
| No edit to `data/damage_table.json` | ✅ untouched |
| Hawke COP-freeze lifted, but COP path **not** exercised | ✅ — no edits to `engine/game.py::_apply_power_effects` Hawke branch or `engine/co.py` Hawke constants |
| Primary-source URLs in code comment | ✅ — AWBW Damage Formula + Properties wiki |
| Smallest viable patch | ✅ — +18 / 0 lines, single helper |
| Maintain or exceed 931 / 5 / 0 | ✅ — 932 / 4 / 0 |
| Pytest baseline held | ✅ — 429 passed, 0 failures (same exclusion as 1624082 closeout) |
| Cohort ≥ 10 games regression check | ✅ — 196-game Hawke/Jess/Von Bolt cohort, +2 closures, 0 in-cohort regressions |

---

## 7. Verdict

**GREEN — close 1617442 to `ok`.** Root cause was a non-CO, oracle-side defender-resolution bug; the Hawke / Jess CO levers were not the right flank. Net audit-floor delta: **+1 ok, −1 oracle_gap** (with one disclosed out-of-cohort regression on a pre-existing YELLOW residual). Single 1-liner root cause:

> Indirect-fire defender resolver was matching engine **pre-strike** display HP against AWBW's **post-strike** `combatInfo.units_hit_points` hint, snapping an Artillery shot to the wrong neighbour foe and leaving the recorded defender unscathed; the missed damage cascaded one capture-day forward and starved the day-33 BUILD by 150 g.

---

*"Veritas filia temporis."* (Latin, classical proverb popularised by Aulus Gellius, *Noctes Atticae* XII.11; ~2nd c. AD)
*"Truth is the daughter of time."* — proverb cited by Aulus Gellius
*Aulus Gellius: Roman author and grammarian; the line was later adopted as the personal motto of Mary I of England and as the title of Josephine Tey's 1951 detective novel re-examining Richard III.*
