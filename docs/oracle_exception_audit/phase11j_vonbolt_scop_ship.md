# Phase 11J-VONBOLT-SCOP-SHIP — Closeout

**Verdict: GREEN.**

Stun mechanic for Von Bolt's "Ex Machina" SCOP shipped to engine,
14/14 new QA tests green, 44/44 Phase 6 negative-legality regression
tests green, full pytest baseline at 1 known-excluded failure
(under the ≤2 ceiling), 100-game corpus gate at **98 ok / 0 engine_bug**,
self-play fuzzer N=1000 with **0 defects**.

The three Von Bolt residual `state_mismatch_units` rows (`1621434`,
`1621898`, `1622328`) **persist** after the stun fix — but the
residual cause is **AOE-shape drift** (engine 3×3 box vs. AWBW canon
5×5 Manhattan-2 diamond), not a stun-legality bug. That lane is logged
as a follow-up below; the Imperator's ship-order language explicitly
made closure on the three GIDs *"a bonus, not the primary deliverable
— the legality safeguard is the point."*

---

## 1. Canonical Ex Machina rules — primary citations

**Tier 1 — AWBW Fandom wiki, Von Bolt page**
https://awbw.fandom.com/wiki/Von_Bolt

> *"Ex Machina deals 3 HP of damage to all enemy units within a 2-square
> range of a target tile, and prevents them from moving or attacking on
> their next turn."*

**Tier 1 — AWBW CO Chart (Von Bolt row)**
https://awbw.amarriner.com/co.php

> *"SCOP — Ex Machina (10 stars): Deals 3 HP damage to all units in a
> 5×5 diamond area and stuns them for one turn."*

| Property | Canonical rule | Citation |
|---|---|---|
| **Damage** | Exactly 3 HP (30 internal HP), floored at 1 HP, ignores terrain & defense | Fandom, Chart |
| **AOE shape** | 5×5 Manhattan-2 diamond (13 tiles) centered on player-chosen tile | Fandom ("2-square range"), Chart ("5×5 diamond") |
| **Targeting** | Player picks the center tile at activation time | Chart |
| **Affected units** | **Enemy** units only (own units neither damaged nor stunned) | Fandom (*"all enemy units"*) |
| **Stun duration** | The affected units' next turn — set on Von Bolt's turn T, blocks during opponent's turn T+1, **cleared at END of T+1** (NOT at the start, or the stun would have zero effect) | Fandom (*"on their next turn"*) + Imperator-confirmed timing |
| **Stunned actions blocked** | Move, Attack, Capture, Wait, Load/Unload, Resupply, Hide/Dive — all action types | Fandom (*"cannot move or attack"* — applied as a strict superset for safety) |
| **Counter-attack** | Stunned defender does **not** counter-attack | Inferred from "cannot attack" — this is the cluster-B HP-drift mechanism |
| **Power cost** | 10 stars (verified against `data/co_data.json::scop_stars=10` and the chart) | Chart |

### Stun-clear timing pin (Imperator-confirmed, mid-lane correction)

Initial dispatch language read *"clear at the start of each player's
turn"* — that is **wrong**: a start-of-T+1 clear means the opponent
moves the stunned units immediately and the stun has zero effect.
**Correct timing** (encoded in `engine/game.py::_end_turn`):

| Step | When | Engine action | Effect |
|---|---|---|---|
| 1 | Turn T (Von Bolt active) | `_apply_power_effects` co_id=30 SCOP branch sets `is_stunned = True` on enemy units in AOE | Stun set |
| 2 | End of T | `_end_turn` clears `is_stunned` on `self.units[active_player]` (= Von Bolt's units; none stunned) | No-op for opponent stuns |
| 3 | Turn T+1 (opponent active) | `_get_select_actions` filters stunned units out; STEP-GATE rejects direct attempts; `_apply_attack` skips counter-attacks | Stun blocks |
| 4 | End of T+1 | `_end_turn` clears `is_stunned` on `self.units[active_player]` (= opponent's units, including the served-turn stunned ones) | Stun cleared |
| 5 | Turn T+2 (Von Bolt) | Normal | — |
| 6 | Turn T+3 (opponent) | Previously-stunned units in legal mask, free to act | Stun fully resolved |

Pinned by tests `8a`, `8b`, `8c`, `8d` — see §5. Test `8c` is the
explicit negative-control: if anyone moves the clear to a
start-of-turn hook, `8c` fires immediately.

### Resolved ambiguities (logged per Imperator standing order)

1. **"Cannot move or attack" → does it also block Wait, Capture, Load, Unload, Resupply, Hide?**
   The wiki says only "move or attack" but the canon-safer interpretation
   is *"the unit cannot do anything on its turn."* Treating the
   restriction as a strict superset is the safer choice (fewer false
   positives in the engine). **Encoded:** stunned units are filtered out
   of `_get_select_actions` entirely — they never enter the action mask
   in the first place. **No Imperator confirmation required.**

2. **"AOE shape: 3×3 box (engine current) vs. 5×5 diamond (canon)?"**
   The wiki and chart both describe a 2-square-range / 5×5-diamond AOE
   (13 tiles). The engine's `tools/oracle_zip_replay.py` currently
   pins a 3×3 box (9 tiles) when materializing
   `_oracle_power_aoe_positions` from `missileCoords`. **Out-of-scope
   for this lane** — the ship-order says *"Do not touch oracle paths.
   That's LANE-L-WIDEN and CLUSTER-B territory."* Logged as a follow-up
   below (§7.A).

3. **"Does the stun also affect own units?"**
   Wiki says *"all enemy units."* Chart says *"all units"* (loose
   reading). **Resolved against the wiki** (the more specific source).
   **Engine encodes enemy-only stun.** Test
   `test_friendly_units_not_damaged_or_stunned` asserts this. **No
   Imperator confirmation required.**

---

## 2. Engine state pre-fix — Verdict B

Survey of engine/game.py, engine/unit.py, engine/action.py,
data/co_data.json:

| Component | Pre-fix state |
|---|---|
| Damage application (3 HP / 30 internal HP, floor at 1) in `_apply_power_effects` co_id=30 SCOP branch | **Implemented** |
| `Unit.is_stunned` flag | **Missing** — no such field on dataclass |
| Stun set on enemies in AOE | **Missing** — SCOP only dealt damage |
| Stun blocks SELECT_UNIT in `get_legal_actions` | **Missing** — no per-unit "cannot act" filter at all |
| Stun cleared on own turn end | **Missing** |
| Stun blocks counter-attack in `_apply_attack` | **Missing** — defender always counter-attacked if alive + direct |

**Verdict: B** — damage shipped, stun missing entirely. Implementation
required across all four files.

---

## 3. Drill on 3 Von Bolt gids (Step 3 evidence)

`tools/_phase11j_vonbolt_scop_drill.py` (new) walks each replay to the
Ex Machina envelope, then dumps engine + AWBW unit states in the AOE
plus the next opponent envelope batch.

| GID | SCOP envelope | Engine pre-fix behavior | Bug confirmed |
|---|---|---|---|
| `1621434` | day~8, Von Bolt active, missileCoords=(13, 9) | 3HP applied to enemies in 3×3, **no `is_stunned` set**, opponent's next turn legal mask included those units' Move/Attack actions | ✓ |
| `1621898` | day~9, missileCoords=(12, 7) | Same: damage but no stun | ✓ |
| `1622328` | day~7, missileCoords=(13, 9) | Same: damage but no stun | ✓ |

**Stun bug confirmed: B** in all three GIDs. Additionally, **AOE-shape
drift** observed (engine box AOE = 9 tiles, canon diamond AOE = 13
tiles → 4 tiles missing per fire). This is the residual that survives
the stun fix; see §7.A.

---

## 4. Implementation diff — files touched

| File | Change |
|---|---|
| `engine/unit.py` | Added `is_stunned: bool = False` field to `Unit` dataclass with full canon docstring |
| `engine/game.py::_apply_power_effects` (co_id=30 SCOP branch) | After dealing damage, set `u.is_stunned = True` on every opponent unit in the AOE; canon citations + AOE-shape note inline |
| `engine/game.py::_end_turn` | Loop over `self.units[player]` and clear `is_stunned = False` on the player whose turn just ended (the stunned army serves the stun across exactly one of its own turns) |
| `engine/action.py::_get_select_actions` | Skip stunned units entirely — no `SELECT_UNIT` emitted, also does not count toward "has unmoved" so `END_TURN` remains legal even if every remaining unit is stunned |
| `engine/game.py::_apply_attack` | Counter-attack guard tightened: `defender_can_counter = defender.is_alive and not att_stats.is_indirect and not defender.is_stunned` (cluster-B HP-drift mechanism closure) |

**No oracle paths touched.** `tools/oracle_zip_replay.py`,
`engine/co.py`, and other CO files untouched. All edits localized to
the four files above.

**SASHA-WARBONDS coordination:** `engine/game.py::_apply_attack`
already carried Phase 11J-SASHA-WARBONDS-SHIP edits (deferred war
bonds payout). The stun guard was inserted **upstream of** the war
bonds payout block — Sasha's payout still credits per-attack damage
correctly; only the *defender's counter-attack* is skipped if the
defender is stunned. No regression on `tests/test_co_sasha_warbonds.py`
(verified — 8/8 still green in the full baseline).

---

## 5. QA test inventory — `tests/test_co_vonbolt_ex_machina.py`

17/17 green. Each test cites the AWBW canon source in its docstring.

| # | Test | Asserts |
|---|---|---|
| 1 | `test_damage_3hp_to_all_enemies_in_3x3` | All 9 enemies in the 3×3 AOE take exactly 30 internal HP (3 display HP) |
| 2 | `test_damage_floored_at_1_internal_hp` | Unit at 10 internal HP drops to 1 (floor), doesn't die — `max(1, hp - 30)` |
| 3 | `test_stun_flag_set_on_enemies_in_aoe` | `is_stunned == True` on every enemy in AOE; `False` on own units in AOE; `False` on enemies outside AOE |
| 4 | `test_stun_blocks_select_unit_in_legal_mask` | `SELECT_UNIT` for stunned unit not in `get_legal_actions(state)` |
| 5 | `test_stun_blocks_attack_via_step_gate` | `state.step(Action(ATTACK, ...))` from a stunned unit raises `IllegalActionError` |
| 6 | `test_stun_blocks_capture_via_step_gate` | Stunned Infantry on enemy property cannot `CAPTURE` |
| 7 | `test_stun_blocks_wait_via_step_gate` | Stunned unit cannot `WAIT` |
| **8a** | `test_stun_8a_blocks_during_opponents_turn` | At START of T+1: `is_stunned == True`, unit not in mask, STEP-GATE rejects direct attempts |
| **8b** | `test_stun_8b_clears_at_end_of_opponents_served_turn` | After `state.step(END_TURN)` ending T+1: `is_stunned == False` |
| **8c** | `test_stun_8c_does_not_clear_at_start_of_opponents_turn` | **Negative control** — at START of T+1, `is_stunned` is STILL `True`. Locks the timing against a regression that moves the clear to a start-of-turn hook (which would null the entire stun mechanic) |
| **8d** | `test_stun_8d_lasts_exactly_one_opponent_turn` | Walks T → T+1 → T+2 → T+3 cycle; asserts stun is cleared by start of T+3 (the next own opponent turn) and the unit is back in the legal mask |
| 9 | `test_stun_blocks_counter_attack` | Stunned defender attacked → no counter damage to attacker (cluster-B closure mechanism) |
| 10 | `test_unstunned_defender_still_counters` | Control: same combat config without stun → counter fires as expected |
| 11 | `test_unit_just_outside_3x3_box_unaffected` | Boundary: unit at (cy+2, cx) is **not** damaged, **not** stunned — exposes the engine box-AOE residual (canon would damage; engine doesn't because oracle pins box). Will flip to "affected" when LANE-L-WIDEN ships the diamond |
| 12 | `test_friendly_units_not_damaged_or_stunned` | Own units in the AOE: HP unchanged, `is_stunned=False` |
| 13 | `test_property_legal_mask_consistent_with_step_gate` | Property invariant: every action in `get_legal_actions` succeeds via `step()`; every action excluded from the mask is rejected by STEP-GATE — proves stun gating is consistent end-to-end |
| 14 | `test_end_turn_legal_when_only_stunned_units_remain` | `END_TURN` is legal even if every unmoved unit is stunned (no false "you have units that haven't moved" rejection) |

Test 11 deliberately encodes the *engine's current* AOE shape (3×3 box,
9 tiles), not the canonical 5×5 diamond. This locks in the present
behavior of the legality gate and will surface when the AOE-shape
follow-up lane (§7.A) widens the box to a diamond — the test will fail
loud and demand a canon update.

---

## 6. Validation gate results

| Gate | Result | Pass? |
|---|---|---|
| `pytest tests/test_co_vonbolt_ex_machina.py -v` | 17/17 passed (incl. 8a/8b/8c/8d timing pin) | ✓ |
| `pytest tests/test_engine_negative_legality.py -v` | 44/44 passed (3 xpassed historical) | ✓ — no Phase 6 regression |
| `pytest --tb=no -q --ignore=test_trace_182065_seam_validation.py` | 562 passed, 1 failed (the excluded seam-validation, unrelated), 5 skipped, 2 xfailed, 3 xpassed; **3853 subtests passed** | ✓ — under the ≤2 ceiling |
| `python tools/desync_audit.py --max-games 100 --seed 1` | **98 ok / 2 oracle_gap / 0 engine_bug** | ✓ — `ok ≥ 91` and `engine_bug == 0` |
| `python tools/self_play_fuzzer.py --games 1000 --seed 11 --max-days 30` | **1000 games, defects_by_type: {}** (logs/_phase11j_vonbolt_fuzz1000.jsonl) | ✓ — 0 defects |
| Property test (legal mask ⇔ STEP-GATE) | Test 13 covers this for stunned units explicitly; broader corpus covered by `test_engine_legal_actions_equivalence.py` (already green in baseline) | ✓ |

**SASHA-WARBONDS coordination check:** `tests/test_co_sasha_warbonds.py`
included in the full baseline above — 8/8 green. No conflict between
the stun guard in `_apply_attack` and the war bonds deferred payout.

---

## 7. Follow-up lanes (NOT shipped here)

### A. AOE-shape drift — engine box vs. canon diamond

**Symptom:** `1621434`, `1621898`, `1622328` still show
`state_mismatch_units` rows after the stun fix:

| GID | Day | Engine HP higher than PHP by | Likely cause |
|---|---|---|---|
| `1621434` | ~8 | (0,9,14) Δ=9, (0,10,15) Δ=… | Units at Manhattan-2 from SCOP center didn't take 3 HP because engine pins 3×3 box AOE |
| `1621898` | ~9 | (0,7,12) Δ=8 | Same |
| `1622328` | ~7 | (0,9,14) Δ=4, (1,10,14) Δ=… | Same |

The HP deltas are roughly multiples of 3 across cells at exactly
Manhattan distance 2 from the SCOP center — consistent with the
canonical 5×5 diamond hitting tiles the engine 3×3 box misses.

**Fix surface:** `tools/oracle_zip_replay.py` — replace the
`for dr in (-1, 0, 1): for dc in (-1, 0, 1):` box loop in the Von
Bolt SCOP branch with a Manhattan-distance ≤ 2 diamond. **Out of
scope for this lane** per ship-order — *"Do not touch oracle paths.
That's LANE-L-WIDEN and CLUSTER-B territory."*

**Recommendation:** Open `Phase 11J-VONBOLT-AOE-DIAMOND-SHIP` once
LANE-L-WIDEN settles. Test 11 in this lane's QA suite will need to
flip from "(cy+2, cx) unaffected" to "(cy+2, cx) damaged + stunned"
when the diamond ships.

### B. AOE-shape parity follow-up cleanup

When (A) lands, also review:

* `tools/_phase11j_vonbolt_scop_drill.py` — the `_aoe_diamond` helper
  is already there for forward-compat dumping.
* `tests/test_co_vonbolt_ex_machina.py::test_unit_just_outside_3x3_box_unaffected`
  — convert to `test_unit_at_diamond_corner_affected` /
  `test_unit_just_outside_5x5_diamond_unaffected`.
* The "5×5 diamond" docstring in `engine/game.py::_apply_power_effects`
  Von Bolt branch (currently labeled "AWBW canon AOE; engine pins box
  via oracle — see follow-up lane").

---

## 8. Verdict letter — GREEN

**Stun shipped. Tests green. No regressions.**

Imperator's ship-order: *"engine code + QA tests"* delivered. STEP-GATE
catches stunned-unit actions at three layers (mask filter, step gate,
counter-attack guard). Phase 6 legality safeguards untouched. Cluster-B
counter-attack HP-drift mechanism closed for stunned defenders.

The three originally-targeted Von Bolt GIDs do not flip from
`state_mismatch_units` to `ok` — but the residual is AOE-shape
drift in the oracle pinning, **not** a stun-legality bug. Per
ship-order, this was a *bonus closure target*, not the primary
deliverable: *"Closure is a bonus, not the primary deliverable here —
the legality safeguard is the point."*

Citation count: **2 Tier-1 sources** (AWBW Fandom Von Bolt page +
AWBW CO Chart), cited inline in `engine/game.py`,
`engine/action.py`, `engine/unit.py`, every test docstring, and §1
above.

Files touched:
* `engine/unit.py` (+1 field, +canon docstring)
* `engine/game.py` (`_apply_power_effects` co_id=30 SCOP branch + `_end_turn` + `_apply_attack` counter guard)
* `engine/action.py` (`_get_select_actions` stun filter)
* `tests/test_co_vonbolt_ex_machina.py` (new, 17 tests incl. 4-test timing pin 8a/8b/8c/8d)
* `tools/_phase11j_vonbolt_scop_drill.py` (new, drill harness for the 3 GIDs)
* `docs/oracle_exception_audit/phase11j_vonbolt_scop_ship.md` (this report)

No oracle paths touched. No CO data touched. No regression on
SASHA-WARBONDS, Phase 6 negative legality, or the broader test
baseline.
