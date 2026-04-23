"""Black Boat REPAIR command — AWBW parity.

Pins the rules from https://awbw.fandom.com/wiki/Black_Boat and the
``awbw-engine-parity`` plan:

1. REPAIR is an explicit one-target ACTION; a Black Boat does *not*
   auto-heal every adjacent ally on WAIT.
2. Heal amount is +10 internal HP (1 AWBW bar), not +20.
3. Heal cost is 10% of the target's listed deployment cost, charged only
   when HP actually ticks up. If the boat's player cannot afford it, the
   heal is skipped but resupply (fuel + ammo) still fires.
4. Full-HP neighbours that need fuel / ammo still get resupplied at $0.
5. Self-repair is refused — a boat cannot target itself.
"""
from __future__ import annotations

import unittest

from engine.action import (
    Action, ActionType, ActionStage,
    get_legal_actions,
)
from engine.game import IllegalActionError, make_initial_state
from engine.unit import Unit, UnitType, UNIT_STATS

from test_lander_and_fuel import _fresh_state, _make_unit, _select_and_move


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state_with_boat() -> tuple:
    """Fresh 5x5 state, P0 active, Black Boat at (1, 2) on shoal."""
    state = _fresh_state()
    state.active_player = 0
    # Starting funds high enough for every test that assumes "affordable".
    state.funds[0] = 100_000
    bb = _make_unit(state, UnitType.BLACK_BOAT, 0, (1, 2))
    return state, bb


# ---------------------------------------------------------------------------
# Legal action mask
# ---------------------------------------------------------------------------

class TestBlackBoatLegalRepair(unittest.TestCase):
    """REPAIR listing follows AWBW permissive rules (``_black_boat_repair_eligible``)."""

    def test_repair_offered_per_adjacent_ally_needing_help(self) -> None:
        state, bb = _state_with_boat()
        damaged = _make_unit(state, UnitType.INFANTRY, 0, (2, 2), hp=40)
        _select_and_move(state, bb, bb.pos)
        acts = get_legal_actions(state)
        repairs = [a for a in acts if a.action_type == ActionType.REPAIR]
        self.assertEqual(len(repairs), 1)
        self.assertEqual(repairs[0].target_pos, damaged.pos)

    def test_repair_not_offered_for_full_hp_full_supply(self) -> None:
        """Phase 10M / Option A: REPAIR stays in the legal mask; apply-layer is a no-op
        when ally is already full HP/fuel/ammo (AWBW permissive listing)."""
        state, bb = _state_with_boat()
        ally = _make_unit(state, UnitType.INFANTRY, 0, (2, 2))  # 100 HP, max fuel/ammo
        fuel_before = ally.fuel
        ammo_before = ally.ammo
        funds_before = state.funds[0]
        _select_and_move(state, bb, bb.pos)
        types = {a.action_type for a in get_legal_actions(state)}
        self.assertIn(ActionType.REPAIR, types)
        state.step(Action(
            ActionType.REPAIR,
            unit_pos=bb.pos, move_pos=bb.pos, target_pos=ally.pos,
        ))
        self.assertEqual(ally.hp, 100)
        self.assertEqual(ally.fuel, fuel_before)
        self.assertEqual(ally.ammo, ammo_before)
        self.assertEqual(state.funds[0], funds_before)

    def test_repair_offered_for_full_hp_when_ally_needs_fuel(self) -> None:
        state, bb = _state_with_boat()
        _make_unit(state, UnitType.INFANTRY, 0, (2, 2), fuel=5)
        _select_and_move(state, bb, bb.pos)
        types = {a.action_type for a in get_legal_actions(state)}
        self.assertIn(ActionType.REPAIR, types)

    def test_repair_ignores_enemy_neighbours(self) -> None:
        state, bb = _state_with_boat()
        _make_unit(state, UnitType.INFANTRY, 1, (2, 2), hp=10)
        _select_and_move(state, bb, bb.pos)
        types = {a.action_type for a in get_legal_actions(state)}
        self.assertNotIn(ActionType.REPAIR, types)


# ---------------------------------------------------------------------------
# HP / cost / resupply behaviour
# ---------------------------------------------------------------------------

class TestBlackBoatRepairBehaviour(unittest.TestCase):

    def test_heal_applies_ten_hp_and_charges_ten_percent(self) -> None:
        state, bb = _state_with_boat()
        inf = _make_unit(state, UnitType.INFANTRY, 0, (2, 2), hp=50)
        funds_before = state.funds[0]

        _select_and_move(state, bb, bb.pos)
        state.step(Action(
            ActionType.REPAIR,
            unit_pos=bb.pos, move_pos=bb.pos, target_pos=inf.pos,
        ))

        self.assertEqual(inf.hp, 60, "Heal must add exactly 10 HP (1 AWBW bar).")
        expected_cost = max(1, UNIT_STATS[UnitType.INFANTRY].cost // 10)
        self.assertEqual(state.funds[0], funds_before - expected_cost)

    def test_heal_never_exceeds_max_hp(self) -> None:
        """Residue band: internal 91–99 shows 10/10 bars; AWBW charges $0 but tops HP to 100.

        GL **1635742** (env 38): Md.Tank @ 97 internal — PHP ``Repair.funds`` unchanged while
        internal HP reaches 100. Engine matches that **display-cap residue** policy; see
        ``engine/game.py`` ``_apply_repair`` (~lines 1787–1798).
        """
        state, bb = _state_with_boat()
        inf = _make_unit(state, UnitType.INFANTRY, 0, (2, 2), hp=95)
        funds_before = state.funds[0]

        _select_and_move(state, bb, bb.pos)
        state.step(Action(
            ActionType.REPAIR,
            unit_pos=bb.pos, move_pos=bb.pos, target_pos=inf.pos,
        ))
        self.assertEqual(inf.hp, 100)
        # Display-cap residue: no gold charged (bars already read 10/10 at 95 internal).
        self.assertEqual(state.funds[0], funds_before)

    def test_heal_below_display_cap_charges_cost(self) -> None:
        """Guard: only the maxed-bar residue band is $0; normal ticks still charge 10% cost.

        50→60 internal is 5→6 bars — well below the display cap — so full Black Boat heal
        cost must apply (pairs with ``test_heal_never_exceeds_max_hp`` / GL 1635742 residue).
        """
        state, bb = _state_with_boat()
        inf = _make_unit(state, UnitType.INFANTRY, 0, (2, 2), hp=50)
        funds_before = state.funds[0]

        _select_and_move(state, bb, bb.pos)
        state.step(Action(
            ActionType.REPAIR,
            unit_pos=bb.pos, move_pos=bb.pos, target_pos=inf.pos,
        ))

        self.assertEqual(inf.hp, 60)
        expected_cost = max(1, UNIT_STATS[UnitType.INFANTRY].cost // 10)
        self.assertEqual(state.funds[0], funds_before - expected_cost)

    def test_broke_player_skips_heal_but_still_resupplies(self) -> None:
        state, bb = _state_with_boat()
        # Expensive target so cost is definitely more than the player can pay.
        mech = _make_unit(state, UnitType.MECH, 0, (2, 2), hp=40, fuel=5)
        mech.ammo = 0
        state.funds[0] = 50  # floor: Mech heal costs 300 (3000 // 10)
        hp_before = mech.hp

        _select_and_move(state, bb, bb.pos)
        state.step(Action(
            ActionType.REPAIR,
            unit_pos=bb.pos, move_pos=bb.pos, target_pos=mech.pos,
        ))

        self.assertEqual(mech.hp, hp_before, "Heal must be skipped when broke.")
        self.assertEqual(state.funds[0], 50, "Broke treasury must be untouched.")
        # Resupply must still fire (wiki rule).
        stats = UNIT_STATS[UnitType.MECH]
        self.assertEqual(mech.fuel, stats.max_fuel)
        self.assertEqual(mech.ammo, stats.max_ammo)

    def test_full_hp_target_zero_cost_still_resupplies(self) -> None:
        state, bb = _state_with_boat()
        inf = _make_unit(state, UnitType.INFANTRY, 0, (2, 2), fuel=3)
        inf.ammo = 0
        funds_before = state.funds[0]

        _select_and_move(state, bb, bb.pos)
        state.step(Action(
            ActionType.REPAIR,
            unit_pos=bb.pos, move_pos=bb.pos, target_pos=inf.pos,
        ))

        self.assertEqual(inf.hp, 100)
        self.assertEqual(state.funds[0], funds_before,
                         "Full-HP target must not be charged.")
        stats = UNIT_STATS[UnitType.INFANTRY]
        self.assertEqual(inf.fuel, stats.max_fuel)
        self.assertEqual(inf.ammo, stats.max_ammo)

    def test_self_repair_is_refused(self) -> None:
        """Phase 10M: crafted self-target is not in the legal mask — STEP-GATE fires first.
        With ``oracle_mode=True``, ``_apply_repair`` still refuses self-heal (apply-layer)."""
        state, bb = _state_with_boat()
        # Damage the boat itself (hypothetically) to make the test observable.
        bb.hp = 50
        fuel_before = bb.fuel
        _select_and_move(state, bb, bb.pos)
        with self.assertRaises(IllegalActionError):
            state.step(Action(
                ActionType.REPAIR,
                unit_pos=bb.pos, move_pos=bb.pos, target_pos=bb.pos,
            ))
        self.assertEqual(bb.hp, 50)
        self.assertEqual(bb.fuel, fuel_before)
        # Apply path: explicit self-repair guard, no heal and no self-resupply branch.
        state.step(Action(
            ActionType.REPAIR,
            unit_pos=bb.pos, move_pos=bb.pos, target_pos=bb.pos,
        ), oracle_mode=True)
        self.assertEqual(bb.hp, 50, "Boat must not be its own heal target.")
        self.assertEqual(bb.fuel, fuel_before)

    def test_wait_does_not_mass_repair(self) -> None:
        """WAIT must no longer heal every adjacent ally (old auto-behavior)."""
        state, bb = _state_with_boat()
        inf_n = _make_unit(state, UnitType.INFANTRY, 0, (0, 2), hp=30)
        inf_s = _make_unit(state, UnitType.INFANTRY, 0, (2, 2), hp=30)
        _select_and_move(state, bb, bb.pos)
        state.step(Action(
            ActionType.WAIT, unit_pos=bb.pos, move_pos=bb.pos,
        ))
        self.assertEqual(inf_n.hp, 30)
        self.assertEqual(inf_s.hp, 30)


if __name__ == "__main__":
    unittest.main()
