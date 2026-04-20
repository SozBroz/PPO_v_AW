"""Jess (CO 14) COP "Turbo Charge" and SCOP "Overdrive" refuel + reload ALL units.

Per AWBW Fandom (Jess entry): both powers refill the fuel and ammunition of
all of her units (not just vehicles — vehicles get the +mov / +atk bonuses on
top, but every unit gets the resupply).

Regression for game 1632380: P1 Drake's BB (sic — actually P1 Jess's BB)
drained to fuel=3 by env 34, then env 35's Power activation should have
refueled it to max=60 before the Move at ai=15. Without the refuel, the engine
raised "Illegal move: Black Boat ... fuel=0 is not reachable".
"""

from __future__ import annotations

import unittest

from engine.action import ActionStage
from engine.co import make_co_state_safe
from engine.game import GameState
from engine.map_loader import MapData
from engine.unit import UNIT_STATS, Unit, UnitType


def _state_with_low_fuel_units(active_co_id: int = 14) -> GameState:
    """Tiny map; P0 has a low-fuel BB, low-fuel Tank, low-ammo Artillery."""
    terrain = [[1, 1, 1, 1]]
    map_data = MapData(
        map_id=0, name="jess-refuel", map_type="std",
        terrain=terrain, height=1, width=4,
        cap_limit=99, unit_limit=50, unit_bans=[], tiers=[],
        objective_type=None, properties=[],
        hq_positions={0: [], 1: []}, lab_positions={0: [], 1: []},
        country_to_player={},
    )
    bb_stats = UNIT_STATS[UnitType.BLACK_BOAT]
    tank_stats = UNIT_STATS[UnitType.TANK]
    art_stats = UNIT_STATS[UnitType.ARTILLERY]
    inf_stats = UNIT_STATS[UnitType.INFANTRY]
    bb = Unit(
        UnitType.BLACK_BOAT, 0, 100, bb_stats.max_ammo, 3,
        (0, 0), True, [], False, 0, 1,
    )
    tank = Unit(
        UnitType.TANK, 0, 100, 0, 5,
        (0, 1), True, [], False, 0, 2,
    )
    art = Unit(
        UnitType.ARTILLERY, 0, 100, 0, 10,
        (0, 2), False, [], False, 0, 3,
    )
    p1 = Unit(
        UnitType.INFANTRY, 1, 100, inf_stats.max_ammo, inf_stats.max_fuel,
        (0, 3), False, [], False, 0, 4,
    )
    co_states = [make_co_state_safe(active_co_id), make_co_state_safe(1)]
    co_states[0].cop_active = False
    co_states[0].scop_active = False
    return GameState(
        map_data=map_data,
        units={0: [bb, tank, art], 1: [p1]},
        funds=[0, 0],
        co_states=co_states,
        properties=[],
        turn=1,
        active_player=0,
        action_stage=ActionStage.SELECT,
        selected_unit=None,
        selected_move_pos=None,
        done=False,
        winner=None,
        win_reason=None,
        game_log=[],
        tier_name="T2",
        full_trace=[],
    )


class TestJessPowerRefuelsAllUnits(unittest.TestCase):
    def test_jess_cop_refuels_and_reloads_all_owned_units(self) -> None:
        st = _state_with_low_fuel_units(active_co_id=14)
        st._activate_power(cop=True)
        bb, tank, art = st.units[0]
        self.assertEqual(bb.fuel, UNIT_STATS[UnitType.BLACK_BOAT].max_fuel)
        self.assertEqual(bb.ammo, UNIT_STATS[UnitType.BLACK_BOAT].max_ammo)
        self.assertEqual(tank.fuel, UNIT_STATS[UnitType.TANK].max_fuel)
        self.assertEqual(tank.ammo, UNIT_STATS[UnitType.TANK].max_ammo)
        self.assertEqual(art.fuel, UNIT_STATS[UnitType.ARTILLERY].max_fuel)
        self.assertEqual(art.ammo, UNIT_STATS[UnitType.ARTILLERY].max_ammo)

    def test_jess_scop_refuels_and_reloads_all_owned_units(self) -> None:
        st = _state_with_low_fuel_units(active_co_id=14)
        st._activate_power(cop=False)
        bb, tank, art = st.units[0]
        self.assertEqual(bb.fuel, UNIT_STATS[UnitType.BLACK_BOAT].max_fuel)
        self.assertEqual(tank.fuel, UNIT_STATS[UnitType.TANK].max_fuel)
        self.assertEqual(art.fuel, UNIT_STATS[UnitType.ARTILLERY].max_fuel)

    def test_jess_power_does_not_touch_enemy_units(self) -> None:
        st = _state_with_low_fuel_units(active_co_id=14)
        p1_before = (st.units[1][0].fuel, st.units[1][0].ammo)
        st._activate_power(cop=True)
        p1_after = (st.units[1][0].fuel, st.units[1][0].ammo)
        self.assertEqual(p1_before, p1_after)

    def test_non_jess_co_power_does_not_refuel(self) -> None:
        st = _state_with_low_fuel_units(active_co_id=1)  # Andy
        st._activate_power(cop=True)
        bb = st.units[0][0]
        self.assertEqual(bb.fuel, 3, "Andy COP must not refuel BB (only heals HP)")


if __name__ == "__main__":
    unittest.main()
