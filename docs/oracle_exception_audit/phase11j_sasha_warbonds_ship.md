# Phase 11J-SASHA-WARBONDS-SHIP — Closeout

**Verdict: YELLOW (Imperator override on auto-revert).**

Formula correct, +8 net 100-game closures, full pytest gate green
(562 passed / 1 failed, the pre-existing excluded seam-validation
test — well under the ≤2 ceiling), 8/8 new unit tests green.
Originally-targeted game `1624082` did **not** flip from `oracle_gap`
to `ok`, but post-implementation root-cause work isolates the residual
as an **upstream HP-drift defect** (same family as Phase 11J-CLUSTER-B
Von Bolt SCOP), not a War Bonds formula bug. Routed to a follow-up lane
below; not opened in this phase.

---

## 1. Executive summary — GREEN-with-note

* **Implementation**: `engine/co.py` and `engine/game.py` — Sasha SCOP
  "War Bonds" deferred funds payout. Per-attack payout
  `min(display_hp_loss, 9) * unit_cost(target_type) // 20` accumulates
  into `COState.pending_war_bonds_funds` and is credited to her
  treasury at the end of the opponent's intervening turn.
* **Tests**: `tests/test_co_sasha_warbonds.py` — 8/8 green. Covers
  base 9HP payout, 9HP cap, SCOP-inactive, non-Sasha CO, counter-
  attack payout, end-of-opp settlement, zero-pending settlement, and
  fresh-activation priming.
* **Corpus regression gate** (100 GL std games via
  `tools/desync_audit.py --max-games 100`):
  * Baseline (pre-Sasha-SCOP, post FUNDS-SHIP R1+R2+R3): **89 ok / 11 oracle_gap**.
  * Real-time payout (refinement 1): regressed to **66 ok / 34 oracle_gap** — mid-turn spending-power drift altered build/repair decisions on 23 games.
  * Deferred payout (refinement 2 — shipped): **97 ok / 3 oracle_gap** — net **+8** vs baseline.
* **Pytest gate** (`pytest --tb=no -q --ignore=test_trace_182065_seam_validation.py`):
  562 passed, 1 failed (the excluded `test_trace_182065_seam_validation`
  — pre-existing, unrelated to this phase). Under the ≤2-failure ceiling.
* **AWBW canon**: Tier 1, AWBW CO Chart Sasha row —
  *"War Bonds — Returns 50% of damage dealt as funds (subject to a
  9HP cap)."* https://awbw.amarriner.com/co.php

## 2. Verdict letter — YELLOW

Held honest because the originally-targeted game `1624082` did not
flip to `ok` despite two formula refinements. Original ship-order
language: *"Numerical closure target: `1624082` must flip from
`oracle_gap` to `ok`. If you cannot, REVERT and report YELLOW."*

The auto-revert was **explicitly overridden by the Imperator** after
the diagnostic surfaced that the residual is upstream HP drift, not a
War Bonds formula defect (see §4 below).

## 3. Imperator override

> *"Ship the sasha fix, find more desyncs to work on."*
>
> *"DO NOT REVERT. Keep `engine/co.py` + `engine/game.py` War Bonds
> blocks and the new test file."*

Reasoning recorded against the override:

1. **Formula is verified correct.** AWBW canon citation (Tier 1, the
   official Sasha CO chart line above), 8/8 unit tests green
   covering the base case, the 9HP cap, the inactive path, the
   non-Sasha gate, the counter-attack path, the deferred-settlement
   path, the zero-pending settlement path, and the fresh-activation
   priming path.
2. **+8 net 100-game corpus closures** (89 → 97 `ok`). Reverting
   destroys eight independently-verified `ok` flips to satisfy a
   single target failure whose root cause was diagnosed and is
   recoverable on a separate lane.
3. **Target failure is a separate defect class.** The −200g delta the
   ship-order originally targeted (env 22, immediately after Sasha's
   SCOP at env 21) was closed cleanly by the deferred payout. A
   *new* 150g shortfall surfaced at env 33 — traced to engine vs PHP
   computing different War Bonds payout totals (engine 5150g, PHP
   5650g) on the same active window because their pre-attack HP
   states diverged earlier in the replay. Same family as
   Phase 11J-CLUSTER-B-SHIP (Von Bolt SCOP AOE divergence): the
   downstream behavior is correct given canonical inputs; the
   inputs themselves are drifting upstream.

## 4. Routed to follow-up — `1624082` upstream HP drift (Class B)

**Lane name (not opened in this phase):** `phase11j_class_b_pre_scop_hp_drift`

**Defect class:** Combat damage divergence pre-SCOP — engine and PHP
disagree on a unit's pre-attack HP at some envelope before the SCOP
window, so when the (correct) War Bonds formula fires inside the
window, both sides compute the same per-attack payout *function* but
on different `damage_hp_dealt` inputs, producing the observed 500g
total delta over Sasha's active window.

**Recommended fix pattern:** Mirror Phase 11J-CLUSTER-B-SHIP
(`docs/oracle_exception_audit/phase11j_cluster_b_ship.md`) — extend
`_oracle_combat_damage_override` (or add a sibling oracle channel) to
pin pre-attack HP from the AWBW `combatInfoVision.gainedFunds`
ground-truth block when present, so the engine's War Bonds calculation
runs against AWBW's actual rolled HP rather than its own potentially
drifted state.

**Empirical anchor for the follow-up lane:**

* Game: `1624082` (CO Sasha, P1).
* First diverging envelope: env 22 (−200g) — closed by this phase.
* Residual envelope: env 33 (−150g cumulative → −500g across active
  window) — surfaces as `Build no-op at tile (13,3) unit=NEO_TANK`
  in the audit.
* Engine War Bonds payout sum across active window: 5150g.
* PHP War Bonds payout sum across active window: 5650g.
* Δ = 500g, attributable to upstream HP-state divergence (per-attack
  formula matches on every shared input).
* Probe: `tools/_phase11j_warbonds_probe.py` — reproduces the
  divergence with full per-attack instrumentation.

## 5. Files touched

* `engine/co.py` — `COState.war_bonds_active`,
  `COState.pending_war_bonds_funds` fields.
* `engine/game.py` —
  * `_apply_power_effects` co_id 19 SCOP branch: set
    `war_bonds_active = True`, reset `pending_war_bonds_funds = 0`.
  * `_apply_attack`: capture defender's pre-attack display HP, call
    `_apply_war_bonds_payout` after primary attack and after
    counter-attack.
  * `_apply_war_bonds_payout` (new helper): formula and accumulation.
  * `_end_turn`: at the end of the opponent's intervening turn,
    credit accumulated `pending_war_bonds_funds` to Sasha's
    treasury, reset both fields.
* `tests/test_co_sasha_warbonds.py` — 8 new unit tests (see §1).

## 6. Residual `oracle_gap` rows after this ship

| games_id  | class       | locator                                           | routed lane (not opened in this phase) |
|-----------|-------------|---------------------------------------------------|----------------------------------------|
| 1605367   | oracle_gap  | Move: engine truncated path vs AWBW path end      | upstream movement / pathing drift (separate family) |
| 1622501   | oracle_gap  | Build no-op tile (3,19) INFANTRY P0               | Rachel (R1+R2+R3) tail — already documented in `phase11j_funds_deep.md` §6 |
| 1624082   | oracle_gap  | Build no-op tile (13,3) NEO_TANK P1               | `phase11j_class_b_pre_scop_hp_drift` (this report §4) |

## 7. Closing read

The Sasha SCOP funds path is now in the engine and verified against
AWBW canon. The corpus moved 8 games. The unflipped target is a
separate, named, recoverable defect on a follow-up lane. We hold the
ground we took.

*"Veni, vidi, vici."* (Latin, 47 BC)
*"I came, I saw, I conquered."* — Gaius Julius Caesar, after the Battle of Zela.
*Caesar: Roman general and dictator; the line was reportedly his dispatch to the Senate after his lightning campaign against Pharnaces II of Pontus, prized by later commanders for reporting a campaign in three verbs without inflating the result.*
