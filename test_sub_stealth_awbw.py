"""Sub / Stealth rules pinned to https://awbw.fandom.com/wiki/Units (Fuel) and
https://awbw.fandom.com/wiki/Stealth (Hide / targeting).
"""
from __future__ import annotations

import unittest

from engine.action import (
    Action,
    ActionType,
    ActionStage,
    get_attack_targets,
    get_legal_actions,
)
from engine.game import GameState, make_initial_state
from engine.map_loader import MapData
from engine.unit import Unit, UnitType, UNIT_STATS, idle_start_of_day_fuel_drain


SEA = 28


def _tiny_sea_map() -> MapData:
    terrain = [[SEA] * 4 for _ in range(4)]
    return MapData(
        map_id=888_888,
        name="sub_stealth_test",
        map_type="std",
        terrain=terrain,
        height=4,
        width=4,
        cap_limit=99,
        unit_limit=50,
        unit_bans=[],
        tiers=[],
        objective_type=None,
        properties=[],
        hq_positions={},
        lab_positions={},
        country_to_player={},
        predeployed_specs=[],
    )


def _unit(
    state: GameState,
    ut: UnitType,
    player: int,
    pos: tuple[int, int],
    *,
    submerged: bool = False,
    fuel: int | None = None,
) -> Unit:
    st = UNIT_STATS[ut]
    u = Unit(
        unit_type=ut,
        player=player,
        hp=100,
        ammo=st.max_ammo if st.max_ammo > 0 else 0,
        fuel=st.max_fuel if fuel is None else fuel,
        pos=pos,
        moved=False,
        loaded_units=[],
        is_submerged=submerged,
        capture_progress=20,
        unit_id=state._allocate_unit_id(),
    )
    state.units[player].append(u)
    return u


class TestIdleFuelFandom(unittest.TestCase):
    def test_sub_surfaced_vs_submerged(self) -> None:
        st = make_initial_state(_tiny_sea_map(), 1, 1, starting_funds=0, tier_name="T2")
        st.units = {0: [], 1: []}
        sub = _unit(st, UnitType.SUBMARINE, 1, (1, 1), submerged=False)
        self.assertEqual(idle_start_of_day_fuel_drain(sub, 11), 1)
        sub.is_submerged = True
        self.assertEqual(idle_start_of_day_fuel_drain(sub, 11), 5)

    def test_stealth_visible_vs_hidden(self) -> None:
        st = make_initial_state(_tiny_sea_map(), 1, 1, starting_funds=0, tier_name="T2")
        st.units = {0: [], 1: []}
        s = _unit(st, UnitType.STEALTH, 1, (0, 0), submerged=False)
        self.assertEqual(idle_start_of_day_fuel_drain(s, 11), 5)
        s.is_submerged = True
        self.assertEqual(idle_start_of_day_fuel_drain(s, 11), 8)

    def test_eagle_air_discount(self) -> None:
        st = make_initial_state(_tiny_sea_map(), 1, 1, starting_funds=0, tier_name="T2")
        st.units = {0: [], 1: []}
        cop = _unit(st, UnitType.B_COPTER, 1, (0, 0))
        self.assertEqual(idle_start_of_day_fuel_drain(cop, 10), 0)
        fight = _unit(st, UnitType.FIGHTER, 1, (1, 0))
        self.assertEqual(idle_start_of_day_fuel_drain(fight, 10), 3)
        hid = _unit(st, UnitType.STEALTH, 1, (2, 0), submerged=True)
        self.assertEqual(idle_start_of_day_fuel_drain(hid, 10), 6)
        sub = _unit(st, UnitType.SUBMARINE, 1, (3, 0), submerged=True)
        self.assertEqual(idle_start_of_day_fuel_drain(sub, 10), 5)

    def test_end_turn_applies_idle_drain_to_next_player(self) -> None:
        md = _tiny_sea_map()
        st = make_initial_state(md, 1, 1, starting_funds=0, tier_name="T2")
        st.units = {0: [], 1: []}
        sub = _unit(st, UnitType.SUBMARINE, 1, (1, 1), submerged=True, fuel=60)
        start = sub.fuel
        st.active_player = 0
        st.action_stage = ActionStage.SELECT
        st.step(Action(ActionType.END_TURN))
        self.assertEqual(st.active_player, 1)
        self.assertEqual(sub.fuel, start - 5)


class TestDiveHideAction(unittest.TestCase):
    def test_dive_toggle(self) -> None:
        md = _tiny_sea_map()
        st = make_initial_state(md, 1, 1, starting_funds=0, tier_name="T2")
        st.units = {0: [], 1: []}
        sub = _unit(st, UnitType.SUBMARINE, 0, (1, 1), submerged=False)
        st.active_player = 0
        st.step(Action(ActionType.SELECT_UNIT, unit_pos=sub.pos))
        st.step(Action(ActionType.SELECT_UNIT, unit_pos=sub.pos, move_pos=sub.pos))
        legal = {a.action_type for a in get_legal_actions(st)}
        self.assertIn(ActionType.DIVE_HIDE, legal)
        st.step(Action(ActionType.DIVE_HIDE, unit_pos=sub.pos, move_pos=sub.pos))
        self.assertTrue(sub.is_submerged)
        self.assertEqual(st.action_stage, ActionStage.SELECT)


class TestRLDiveHideFlatIndex(unittest.TestCase):
    """DIVE_HIDE must not fall through to flat index 0 (END_TURN) in ``rl.env``."""

    def test_dive_hide_distinct_from_wait_and_end_turn(self) -> None:
        from rl.env import _action_to_flat, _DIVE_HIDE_IDX, _WAIT_IDX

        w = Action(ActionType.WAIT, unit_pos=(0, 0), move_pos=(0, 0))
        d = Action(ActionType.DIVE_HIDE, unit_pos=(0, 0), move_pos=(0, 0))
        self.assertEqual(_action_to_flat(w), _WAIT_IDX)
        self.assertEqual(_action_to_flat(d), _DIVE_HIDE_IDX)
        self.assertNotEqual(_action_to_flat(d), 0)


class TestHiddenStealthTargeting(unittest.TestCase):
    def test_fighter_can_target_hidden_stealth_tank_cannot(self) -> None:
        md = _tiny_sea_map()
        st = make_initial_state(md, 1, 1, starting_funds=0, tier_name="T2")
        st.units = {0: [], 1: []}
        stealth = _unit(st, UnitType.STEALTH, 1, (2, 1), submerged=True)
        tank = _unit(st, UnitType.TANK, 0, (2, 0))
        fighter = _unit(st, UnitType.FIGHTER, 0, (1, 1))
        st.active_player = 0

        t_targets = get_attack_targets(st, tank, (2, 0))
        self.assertNotIn((2, 1), t_targets)

        f_targets = get_attack_targets(st, fighter, (1, 1))
        self.assertIn((2, 1), f_targets)


if __name__ == "__main__":
    unittest.main()
