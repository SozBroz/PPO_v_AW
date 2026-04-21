"""CO powers that apply flat HP loss: wiki parity (integer steps, cannot kill).

AWBW: Drake Tsunami/Typhoon, Olaf Winter Fury, Hawke Black Wave/Black Storm,
Von Bolt Ex Machina (global simplification) — damage is whole display-HP units
on the engine's 0–100 internal scale (10 per display HP), not combat formula
rounding. Survivors bottom out at 1 internal HP (~0.1 display bar on the wiki).
"""
from __future__ import annotations

import unittest

from engine.game import GameState, make_initial_state
from engine.unit import Unit, UnitType

from test_lander_and_fuel import _build_map, _make_unit


def _empty_state(p0_co: int, p1_co: int) -> GameState:
    md = _build_map()
    st = make_initial_state(md, p0_co, p1_co, starting_funds=0, tier_name="T2")
    st.units = {0: [], 1: []}
    return st


class TestDirectDamageCannotKill(unittest.TestCase):
    def test_typhoon_leaves_one_internal_hp(self) -> None:
        st = _empty_state(5, 1)  # P0 Drake
        st.active_player = 0
        enemy = _make_unit(st, UnitType.TANK, 1, (2, 2))
        enemy.hp = 5
        st._apply_power_effects(player=0, cop=False)  # Typhoon: −2 display HP = 20
        self.assertEqual(enemy.hp, 1)
        self.assertIn(enemy, st.units[1])

    def test_winter_fury_leaves_one_internal_hp(self) -> None:
        st = _empty_state(9, 1)
        st.active_player = 0
        enemy = _make_unit(st, UnitType.TANK, 1, (2, 2))
        enemy.hp = 3
        st._apply_power_effects(player=0, cop=False)
        self.assertEqual(enemy.hp, 1)
        self.assertIn(enemy, st.units[1])

    def test_ex_machina_leaves_one_internal_hp(self) -> None:
        st = _empty_state(30, 1)
        st.active_player = 0
        enemy = _make_unit(st, UnitType.TANK, 1, (2, 2))
        enemy.hp = 8
        st._apply_power_effects(player=0, cop=False)  # −3 display HP = 30
        self.assertEqual(enemy.hp, 1)
        self.assertIn(enemy, st.units[1])


class TestHawkeWaveVsStorm(unittest.TestCase):
    def test_black_wave_one_hp_black_storm_two(self) -> None:
        st = _empty_state(12, 1)
        st.active_player = 0
        foe = _make_unit(st, UnitType.TANK, 1, (2, 2))
        foe.hp = 100
        st._apply_power_effects(player=0, cop=True)
        self.assertEqual(foe.hp, 90)

        foe.hp = 100
        st._apply_power_effects(player=0, cop=False)
        self.assertEqual(foe.hp, 80)

    def test_both_powers_heal_own_two_display_hp(self) -> None:
        st = _empty_state(12, 1)
        st.active_player = 0
        ally = _make_unit(st, UnitType.TANK, 0, (3, 2))
        ally.hp = 50
        st._apply_power_effects(player=0, cop=True)
        self.assertEqual(ally.hp, 70)
        ally.hp = 50
        st._apply_power_effects(player=0, cop=False)
        self.assertEqual(ally.hp, 70)


class TestDrakeFuelDrain(unittest.TestCase):
    def test_tsunami_drains_one_fuel_per_tile_worth(self) -> None:
        st = _empty_state(5, 1)
        st.active_player = 0
        copter = _make_unit(st, UnitType.T_COPTER, 1, (2, 2))
        start_fuel = copter.fuel
        st._apply_power_effects(player=0, cop=True)
        self.assertEqual(copter.fuel, max(0, start_fuel - 10))


if __name__ == "__main__":
    unittest.main()
