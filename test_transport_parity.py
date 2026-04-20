"""Load/unload parity across APC, T-Copter, and Black Boat transports.

The lander overhaul centralised LOAD/UNLOAD in shared paths
(``_get_action_actions``, ``_apply_load``, ``_apply_unload``,
``compute_reachable_costs``). These tests mirror ``test_lander_and_fuel.py``
patterns against the other three transports so a future regression in the
shared code is caught per transport class, not just for Lander.
"""
from __future__ import annotations

import unittest

from engine.action import (
    Action, ActionType, ActionStage,
    get_legal_actions, get_reachable_tiles,
)
from engine.unit import Unit, UnitType, UNIT_STATS

from test_lander_and_fuel import (
    SEA, SHOAL, PLAIN,
    _fresh_state, _make_unit, _select_and_move,
)


# ---------------------------------------------------------------------------
# APC (capacity 1, ground transport, INF/MECH only)
# ---------------------------------------------------------------------------

class TestAPCLoadingRules(unittest.TestCase):
    """APC: capacity 1, ground transport; AWBW allows only Infantry and Mech as cargo."""

    def setUp(self) -> None:
        self.state = _fresh_state()
        self.state.active_player = 0
        # APC parked on plain (3, 2); infantry below.
        self.apc = _make_unit(self.state, UnitType.APC, 0, (3, 2))
        self.inf = _make_unit(self.state, UnitType.INFANTRY, 0, (4, 2))

    def test_load_is_only_terminator_on_friendly_apc(self) -> None:
        _select_and_move(self.state, self.inf, (3, 2))
        types = {a.action_type for a in get_legal_actions(self.state)}
        self.assertEqual(types, {ActionType.LOAD})

    def test_mech_can_load_into_apc(self) -> None:
        self.state.units[0].remove(self.inf)
        mech = _make_unit(self.state, UnitType.MECH, 0, (4, 2))
        _select_and_move(self.state, mech, (3, 2))
        types = {a.action_type for a in get_legal_actions(self.state)}
        self.assertEqual(types, {ActionType.LOAD})

    def test_wait_onto_apc_is_rejected(self) -> None:
        _select_and_move(self.state, self.inf, (3, 2))
        with self.assertRaises(ValueError):
            self.state.step(Action(
                ActionType.WAIT, unit_pos=self.inf.pos, move_pos=(3, 2),
            ))

    def test_full_apc_blocks_second_infantry(self) -> None:
        # APC capacity is 1; pre-load directly to simulate full state.
        extra = _make_unit(self.state, UnitType.INFANTRY, 0, (3, 3))
        self.state.units[0].remove(extra)
        self.apc.loaded_units.append(extra)
        self.assertEqual(len(self.apc.loaded_units), 1)

        # The APC tile must drop out of the candidate's reachable set.
        self.assertNotIn((3, 2), get_reachable_tiles(self.state, self.inf))

        # Hand-crafted LOAD must raise on capacity overflow.
        self.inf.moved = False
        with self.assertRaises(ValueError):
            self.state._apply_load(Action(
                ActionType.LOAD, unit_pos=self.inf.pos, move_pos=(3, 2),
            ))

    def test_apc_rejects_incompatible_cargo(self) -> None:
        # APC only carries INF/MECH; a Recon must be refused.
        recon = _make_unit(self.state, UnitType.RECON, 0, (4, 1))
        with self.assertRaises(ValueError):
            self.state._apply_load(Action(
                ActionType.LOAD, unit_pos=recon.pos, move_pos=self.apc.pos,
            ))


class TestAPCUnload(unittest.TestCase):
    def setUp(self) -> None:
        self.state = _fresh_state()
        self.state.active_player = 0
        self.apc = _make_unit(self.state, UnitType.APC, 0, (3, 2))
        cargo = _make_unit(self.state, UnitType.INFANTRY, 0, (4, 2))
        self.state.units[0].remove(cargo)
        self.apc.loaded_units.append(cargo)
        self.cargo = cargo

    def test_unload_emits_four_adjacent_plains(self) -> None:
        _select_and_move(self.state, self.apc, self.apc.pos)
        unloads = [a for a in get_legal_actions(self.state)
                   if a.action_type == ActionType.UNLOAD]
        drop_tiles = {a.target_pos for a in unloads}
        self.assertEqual(drop_tiles, {(2, 2), (4, 2), (3, 1), (3, 3)})

    def test_unload_places_cargo_and_finalizes(self) -> None:
        _select_and_move(self.state, self.apc, self.apc.pos)
        self.state.step(Action(
            ActionType.UNLOAD,
            unit_pos=self.apc.pos,
            move_pos=self.apc.pos,
            target_pos=(4, 2),
            unit_type=UnitType.INFANTRY,
        ))
        self.assertEqual(self.apc.loaded_units, [])
        dropped = self.state.get_unit_at(4, 2)
        self.assertIsNotNone(dropped)
        self.assertTrue(dropped.moved)
        self.assertEqual(self.state.action_stage, ActionStage.SELECT)
        self.assertTrue(self.apc.moved)


# ---------------------------------------------------------------------------
# T-Copter (capacity 1, air transport, drop respects cargo passability)
# ---------------------------------------------------------------------------

class TestTCopterLoadingRules(unittest.TestCase):
    def setUp(self) -> None:
        self.state = _fresh_state()
        self.state.active_player = 0
        # T-Copter on shoal (1, 2); infantry on plain below.
        self.tcop = _make_unit(self.state, UnitType.T_COPTER, 0, (1, 2))
        self.inf  = _make_unit(self.state, UnitType.INFANTRY, 0, (2, 2))

    def test_load_is_only_terminator_on_friendly_tcopter(self) -> None:
        _select_and_move(self.state, self.inf, (1, 2))
        types = {a.action_type for a in get_legal_actions(self.state)}
        self.assertEqual(types, {ActionType.LOAD})

    def test_full_tcopter_blocks_second_infantry(self) -> None:
        extra = _make_unit(self.state, UnitType.INFANTRY, 0, (3, 2))
        self.state.units[0].remove(extra)
        self.tcop.loaded_units.append(extra)
        self.assertNotIn((1, 2), get_reachable_tiles(self.state, self.inf))


class TestTCopterUnload(unittest.TestCase):
    def setUp(self) -> None:
        self.state = _fresh_state()
        self.state.active_player = 0
        # T-Copter on shoal so adjacent north (sea) tests cargo passability.
        self.tcop = _make_unit(self.state, UnitType.T_COPTER, 0, (1, 2))
        cargo = _make_unit(self.state, UnitType.INFANTRY, 0, (2, 0))
        self.state.units[0].remove(cargo)
        self.tcop.loaded_units.append(cargo)

    def test_unload_skips_sea_for_infantry_cargo(self) -> None:
        _select_and_move(self.state, self.tcop, self.tcop.pos)
        unloads = [a for a in get_legal_actions(self.state)
                   if a.action_type == ActionType.UNLOAD]
        drop_tiles = {a.target_pos for a in unloads}
        # Shoal east/west and plain south are walkable for infantry; sea north is not.
        self.assertIn((2, 2), drop_tiles)
        self.assertIn((1, 1), drop_tiles)
        self.assertIn((1, 3), drop_tiles)
        self.assertNotIn((0, 2), drop_tiles)


# ---------------------------------------------------------------------------
# Black Boat (capacity 2, naval, INF/MECH only)
# ---------------------------------------------------------------------------

class TestBlackBoatLoadingRules(unittest.TestCase):
    def setUp(self) -> None:
        self.state = _fresh_state()
        self.state.active_player = 0
        # Black Boat parked on shoal (1, 2); infantry on plain below.
        self.bb  = _make_unit(self.state, UnitType.BLACK_BOAT, 0, (1, 2))
        self.inf = _make_unit(self.state, UnitType.INFANTRY, 0, (2, 2))

    def test_load_is_only_terminator_on_friendly_blackboat(self) -> None:
        _select_and_move(self.state, self.inf, (1, 2))
        types = {a.action_type for a in get_legal_actions(self.state)}
        self.assertEqual(types, {ActionType.LOAD})

    def test_wait_onto_blackboat_is_rejected(self) -> None:
        _select_and_move(self.state, self.inf, (1, 2))
        with self.assertRaises(ValueError):
            self.state.step(Action(
                ActionType.WAIT, unit_pos=self.inf.pos, move_pos=(1, 2),
            ))

    def test_third_unit_cannot_board_full_blackboat(self) -> None:
        # Black Boat capacity = 2 — same guard pattern as Lander.
        for r in (3, 4):
            extra = _make_unit(self.state, UnitType.INFANTRY, 0, (r, 2))
            self.state.units[0].remove(extra)
            self.bb.loaded_units.append(extra)
        self.assertEqual(len(self.bb.loaded_units), 2)

        self.assertNotIn((1, 2), get_reachable_tiles(self.state, self.inf))

        self.inf.moved = False
        with self.assertRaises(ValueError):
            self.state._apply_load(Action(
                ActionType.LOAD, unit_pos=self.inf.pos, move_pos=(1, 2),
            ))

    def test_blackboat_rejects_incompatible_cargo(self) -> None:
        # Black Boat carries INF/MECH only; a Tank must be refused.
        tank = _make_unit(self.state, UnitType.TANK, 0, (2, 1))
        with self.assertRaises(ValueError):
            self.state._apply_load(Action(
                ActionType.LOAD, unit_pos=tank.pos, move_pos=self.bb.pos,
            ))


class TestBlackBoatUnload(unittest.TestCase):
    def setUp(self) -> None:
        self.state = _fresh_state()
        self.state.active_player = 0
        self.bb = _make_unit(self.state, UnitType.BLACK_BOAT, 0, (1, 2))
        c0 = _make_unit(self.state, UnitType.INFANTRY, 0, (3, 0))
        c1 = _make_unit(self.state, UnitType.MECH,     0, (3, 1))
        for c in (c0, c1):
            self.state.units[0].remove(c)
            self.bb.loaded_units.append(c)

    def test_unload_skips_sea_for_infantry_cargo(self) -> None:
        _select_and_move(self.state, self.bb, self.bb.pos)
        unloads = [a for a in get_legal_actions(self.state)
                   if a.action_type == ActionType.UNLOAD]
        drop_tiles = {a.target_pos for a in unloads}
        # Sea (0, 2) is impassable for INF/MECH; shoal/plain neighbours are OK.
        self.assertNotIn((0, 2), drop_tiles)
        self.assertIn((2, 2), drop_tiles)
        self.assertIn((1, 1), drop_tiles)
        self.assertIn((1, 3), drop_tiles)

    def test_first_unload_keeps_action_stage_for_second_drop(self) -> None:
        _select_and_move(self.state, self.bb, self.bb.pos)
        self.state.step(Action(
            ActionType.UNLOAD,
            unit_pos=self.bb.pos,
            move_pos=self.bb.pos,
            target_pos=(2, 2),
            unit_type=UnitType.INFANTRY,
        ))
        # One cargo remaining; transport stays selected in ACTION stage so
        # the player can drop the second unit or finalise with WAIT.
        self.assertEqual(len(self.bb.loaded_units), 1)
        self.assertEqual(self.state.action_stage, ActionStage.ACTION)
        self.assertIs(self.state.selected_unit, self.bb)
        self.assertFalse(self.bb.moved)

        # Drop the second cargo (Mech) — now the transport finalises.
        self.state.step(Action(
            ActionType.UNLOAD,
            unit_pos=self.bb.pos,
            move_pos=self.bb.pos,
            target_pos=(1, 1),
            unit_type=UnitType.MECH,
        ))
        self.assertEqual(self.bb.loaded_units, [])
        self.assertEqual(self.state.action_stage, ActionStage.SELECT)
        self.assertTrue(self.bb.moved)


if __name__ == "__main__":
    unittest.main()
