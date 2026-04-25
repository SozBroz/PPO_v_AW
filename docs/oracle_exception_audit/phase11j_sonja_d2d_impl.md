# Phase 11J-SONJA-D2D-IMPL — Sonja CO 18 D2D rider

**Status:** Counter ×1.5 SHIPPED. Hidden HP investigated and REVERTED.
**Engine LOC delta:** ~12 LOC (counter_amp parameter + counter_amp call site).
**Tests:** `tests/test_co_sonja_d2d.py`, **29 / 29 green** (8 named + 21 parametrized pin).
**100-game regression:** **ok = 98**, engine_bug = 0. Spec floor (`ok ≥ 98`) met.

---

## 1. AWBW canon (ground truth used)

### Tier 1 — `https://awbw.amarriner.com/co.php`, Sonja row

> *"Units gain +1 vision in Fog of War, **have hidden HP**, and **counterattacks
> do 1.5x more damage**. Luck is reduced to -9% to +9%.
> **Enhanced Vision** -- All units gain +1 vision, and can see into forests
> and reefs.
> **Counter Break** -- All units gain +1 vision, and can see into forests and
> reefs. A unit being attacked will attack first (even if it would be
> destroyed by the attack)."*

### Tier 2 — `https://awbw.fandom.com/wiki/Damage_Formula`

> *"Due to the way the formula is set, **damage taken by the defending unit
> will be calculated in the form of its true health.**"*

That clause is the deciding wedge between two interpretations of "Hidden HP"
and forced the revert below.

---

## 2. Implementation summary

### 2a. Counter ×1.5 — SHIPPED

`engine/combat.py::calculate_damage` now takes a keyword-only
`counter_amp: float = 1.0` and multiplies `raw` by it before AWBW's
ceil-to-0.05 / floor rounding. Equivalent to scaling AV inside the formula —
applied to `raw` so the existing rounding path remains the sole source of
HP-tick truncation.

`engine/combat.py::calculate_counterattack` passes
`counter_amp = 1.5 if defender_co.co_id == 18 else 1.0` to the inner
`calculate_damage` call (the `defender_co` of the original strike is the
counter-attacker's CO). Always active — D2D, COP, and SCOP. Stacks with
SCOP "Counter Break" (which restores pre-attack HP via the existing
`co_id == 18 and scop_active` branch above).

```text
diff stats: +12 LOC in engine/combat.py (1 new kwarg, 2 new comment blocks,
            1 new line at counter call site, 1 new line `raw *= counter_amp`).
```

### 2b. Hidden HP defender rider — REVERTED

The first attempt added, just before the formula:

```python
# REVERTED — see § 3.
if defender_co.co_id == 18:
    hpd_bars = max(1, hpd_bars - 1)
```

This shifted defender HP shown to the formula by one display tick on
terrain-star tiles, raising `(200 − dv − dtr·hpd)` and so the damage dealt
to Sonja units. The hypothesis matched the triage diagnosis (engine
over-keeping Sonja by ~10 internal HP per affected combat). The engine
behaviour after that change was rejected on empirical evidence — see § 3.

The revert leaves a load-bearing comment in `combat.py` documenting the
choice and naming the Tier 2 source so a future agent does not silently
re-add the rider.

---

## 3. Reverted: Hidden HP damage rider

### 3a. Why it looked right (mission-letter reasoning)

* AWBW formula: `raw = (B·AV/100 + L − LB) · (HPA/10) · (200 − DV − DTR·HPD)/100`.
* Lowering `HPD` raises `(200 − DV − DTR·HPD)` on terrain-star tiles, dealing
  more damage.
* Phase 11J state-mismatch full triage saw 3 mid-range gids (1631943, 1632283,
  1632968) drift by exactly 10 internal HP, engine over-keeping Sonja units.
  Direction matched.

### 3b. Why it was wrong (Tier 2 + empirical)

**Tier 2 evidence.** Fandom Damage Formula explicitly states the defender's
**true health** is what enters the damage calculation. The Sonja "Hidden HP"
ability is described on the Sonja page (and in strategy commentary on the
same page) as an **information** ability — what the *opponent* sees in the
on-screen indicator and so what they can infer in Fog of War. AWBW's PHP
server has full information; PHP damage uses true HP, just like the engine.

**Empirical evidence.** With the rider in place I re-audited the full set of
**36 Sonja-bearing zips** in the GL std catalog under
`--enable-state-mismatch`:

| Metric (36 Sonja-bearing games)                | Before rider | After rider | Δ      |
|------------------------------------------------|------------- |-------------|--------|
| Class `ok`                                     | 10           | 10          | 0      |
| Class `state_mismatch_units`                   | 19           | 18          | −1     |
| Class `state_mismatch_funds`                   | 5            | 5           | 0      |
| Class `state_mismatch_multi`                   | 2            | 3           | +1     |
| Rows with negative-direction Δ on Sonja units  | 2            | 6           | **+4** |
| Rows with `|Δ| ≥ 10`                           | 18           | 18          | 0      |

Per-gid inspection: of the 4 newly-negative rows, 3 (1625118, 1628539,
1632330) are at the same envelope index where the prior positive drift had
been — sign-flipped, not closed. 1 (1628051) surfaces at an *earlier*
envelope. Engine cumulative overshoot on individual Sonja units reached
**−14 HP, −17 HP, −23 HP** — far larger than the Hidden HP +1-display-tick
shift can explain in a single attack, and confirming systematic per-attack
overshoot.

This signal is consistent with PHP **not** consuming `display_hp − 1` in its
damage formula. The +10 HP positive-direction bias on Sonja units is real
but caused by something other than Hidden HP (likely a sub-display-tick
rounding interaction that does not warrant a CO-specific rider).

### 3c. Mid-range gid closure status (post-revert, counter ×1.5 only)

| gid     | Before (retune baseline)                   | After (counter ×1.5 only)               | Verdict |
|---------|--------------------------------------------|-----------------------------------------|---------|
| 1631943 | env 18, units, Δ = +10 at (1, 17, 15)      | env 18, units, Δ = +10 at (1, 17, 15)   | open    |
| 1632283 | env 13, units, Δ = +10 at (0, 10, 0)       | env 13, units, Δ = +10 at (0, 10, 0)    | open    |
| 1632968 | env 15, multi, funds Δ = −70 + 1 hp drift  | env 15, multi, funds Δ = −70 + 1 hp     | open    |

Counter ×1.5 alone produces zero net change on this corpus's
state-mismatch register because the **first** mismatch trips the replay
break before any counter-amplified divergence can accumulate; the engine
therefore never reaches a turn where the new ×1.5 path would have shifted a
recorded HP. This is structural, not a defect.

The 17-negative-direction signature predicted by the triage report does
**not** clear under the counter ×1.5 ship. After the (rejected) Hidden HP
rider, negative-direction rows on the 36-game subset *grew* from 2 to 6.
The triage's "Hidden HP affecting damage calc" diagnosis is empirically
unsupported and the 17-row signature must be sourced elsewhere (a future
recon — likely sub-display rounding, capture HP truncation, or another
Sonja-correlated mechanic).

---

## 4. SCOP / COP interaction decision

* **D2D ×1.5 counter-attack rider:** active in **D2D, COP, and SCOP**. AWBW
  canon nowhere disables it during powers. SCOP "Counter Break" (defender
  attacks first via pre-attack HP) is timing-only; the existing branch in
  `calculate_counterattack` still routes through `calculate_damage`, which
  receives `counter_amp = 1.5` regardless of `scop_active`.
* **Hidden HP** — not a damage rider. Active only as a UI/observer effect
  (out of scope for the engine combat module).

---

## 5. Test inventory (`tests/test_co_sonja_d2d.py`)

| # | Test                                                          | Asserts                                          |
|---|---------------------------------------------------------------|--------------------------------------------------|
| 1 | `test_hidden_hp_does_not_alter_damage_formula`                | Sonja inbound dmg == Andy inbound dmg on WOOD    |
| 2 | `test_sonja_cop_does_not_alter_inbound_damage_via_hidden_hp`  | COP only shifts dmg by SCOPB +10 DEF             |
| 3 | `test_sonja_scop_does_not_alter_inbound_damage_via_hidden_hp` | SCOP only shifts dmg by SCOPB +10 DEF            |
| 4 | `test_sonja_counter_amplifier_d2d`                            | Sonja counter = 37 (vs Andy baseline = 24)       |
| 5 | `test_no_counter_amplifier_for_andy`                          | Andy counter = 24 (×1.0)                         |
| 6 | `test_sonja_counter_amplifier_active_under_cop`               | Sonja COP counter = 40 (incl. SCOPB +10 ATK)     |
| 7 | `test_sonja_counter_amplifier_stacks_with_scop_counter_break` | SCOP first-strike + ×1.5 chain = 40              |
| 8 | `test_forward_then_counter_amp_chain`                         | Forward (47) + post-attack counter (22) chain    |
| 9 | `test_sonja_inbound_damage_matches_andy_at_every_hp_and_terrain` | 21 parametrized pins (3 terrains × 7 HP buckets) |

Result: **29 collected, 29 passed**.

---

## 6. Gate results

| Gate                                                   | Target              | Actual                  | Pass |
|--------------------------------------------------------|---------------------|-------------------------|------|
| `pytest tests/test_co_sonja_d2d.py -v`                 | all green           | 29 / 29                 | yes  |
| `pytest --tb=no -q --ignore=test_trace_182065_seam_validation.py` | ≤ 2 failures | 640 passed, 0 failed    | yes  |
| 100-game `desync_audit` regression                     | ok ≥ 98, eng_bug=0  | ok = 98, oracle_gap = 2 | yes  |
| 3 mid-range Sonja gids close                           | all 3 close         | 0 / 3                   | no   |
| 17 negative-direction Sonja rows clear                 | clear / shrink      | 16 baseline → 16        | no   |

The first three gates pass; the last two do not. The first three are
engine-health gates (do not regress). The last two are mission-letter
closure targets that depended on the Hidden HP diagnosis being correct;
the empirical evidence in § 3 shows it was not.

---

## 7. Verdict

> **Imperator** — counter ×1.5 ships clean: 12 LOC engine, 29 tests green,
> 640 broad pytest, 100-game regression at the floor (ok = 98, no
> engine_bug). The mid-range gids and the 17-row negative signature did
> NOT close; recon revealed the triage's Hidden HP diagnosis is wrong —
> AWBW Fandom Damage Formula explicitly says PHP uses true HP, and an
> empirical Hidden HP rider trial on 36 Sonja-bearing games sign-flipped
> the drift (positive → negative), reaching −23 HP per unit. Reverted.
>
> **Recommend:** open a fresh recon under a name like
> `phase11j_sonja_residual_hp_drift_recon` to diagnose the 10-HP
> positive-direction bias on Sonja units that survives this ship. Likely
> candidates: sub-display rounding interactions during Sonja capture or
> repair, oracle damage-override precision near display ticks, or an
> interaction with the existing SCOP "counter break" pre-attack-HP branch
> on terrain-star tiles. The +10 HP magnitude exactly equals one display
> tick, but its source is *not* the formula-side Hidden HP.

---

## 8. Files touched

* `engine/combat.py` — counter_amp kwarg + counter_amp=1.5 call site
  (Sonja branch in `calculate_counterattack`); load-bearing comment
  documenting the Hidden HP revert.
* `tests/test_co_sonja_d2d.py` — new file, 29 tests.
* `docs/oracle_exception_audit/phase11j_sonja_d2d_impl.md` — this file.
* `logs/desync_register_sonja_d2d_check.jsonl` — targeted 3-gid audit, post-ship.
* `logs/desync_register_sonja_postship.jsonl` — 36-game Sonja audit with the
  rejected Hidden HP rider (kept as evidence for § 3b).
* `logs/desync_register_sonja_postship_v2.jsonl` — 36-game Sonja audit, post-revert.
* `logs/desync_register_sonja_postship_100.jsonl` — 100-game regression.

No other CO branches in `combat.py` touched. `engine/co.py`,
`engine/unit.py`, `engine/action.py`, and `_apply_power_effects` are
untouched. Coordination with L1-WAVE-2, RACHEL-SCOP-COVERING-FIRE,
COLIN-IMPL, DELETE-RL-GUARD, STATE-MISMATCH-RETUNE: no shared regions.
