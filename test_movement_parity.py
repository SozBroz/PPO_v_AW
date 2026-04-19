"""AWBW movement parity: terrain x move-type reachability rules.

Locks the rule that only foot (MOVE_INF), boot (MOVE_MECH), and air units can
enter mountain tiles (terrain id 2). Wheeled/tread/sea types are impassable on
mountains. The artillery-on-mountain bug from replay 131375 is pinned via a
golden check on map 166877 tile (1,18) with a unit staged at (2,17).
"""
from __future__ import annotations

import unittest
from pathlib import Path

from engine.action import Action, ActionType, get_reachable_tiles
from engine.game import make_initial_state
from engine.map_loader import load_map
from engine.unit import Unit, UnitType, UNIT_STATS

ROOT = Path(__file__).parent
POOL = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"

MOUNTAIN_ID = 2
MAP_166877 = 166877


def _make_unit(unit_type: UnitType, player: int, pos: tuple[int, int]) -> Unit:
    stats = UNIT_STATS[unit_type]
    return Unit(
        unit_type=unit_type,
        player=player,
        hp=100,
        ammo=stats.max_ammo if stats.max_ammo > 0 else 0,
        fuel=stats.max_fuel,
        pos=pos,
        moved=False,
        loaded_units=[],
        is_submerged=False,
        capture_progress=20,
    )


class TestMountainReachability(unittest.TestCase):
    """Artillery (tire B) must NOT reach mountains; infantry/mech must."""

    def setUp(self) -> None:
        self.md = load_map(MAP_166877, POOL, MAPS_DIR)
        self.assertEqual(
            self.md.terrain[1][18], MOUNTAIN_ID,
            "Map 166877 tile (1,18) expected to be a mountain (terrain id 2).",
        )
        self.state = make_initial_state(self.md, 1, 7, starting_funds=0, tier_name="T2")
        self.state.units = {0: [], 1: []}

    def _place(self, unit_type: UnitType, player: int, pos: tuple[int, int]) -> Unit:
        u = _make_unit(unit_type, player, pos)
        self.state.units[player].append(u)
        return u

    def test_artillery_cannot_enter_mountain_from_adjacent_plain(self) -> None:
        unit = self._place(UnitType.ARTILLERY, 0, (2, 17))
        reachable = get_reachable_tiles(self.state, unit)
        self.assertNotIn(
            (1, 18), reachable,
            "Artillery (MOVE_TIRE_B) must not be able to enter a mountain tile.",
        )

    def test_tank_cannot_enter_mountain(self) -> None:
        unit = self._place(UnitType.TANK, 0, (2, 17))
        reachable = get_reachable_tiles(self.state, unit)
        self.assertNotIn((1, 18), reachable)

    def test_recon_cannot_enter_mountain(self) -> None:
        unit = self._place(UnitType.RECON, 0, (2, 17))
        reachable = get_reachable_tiles(self.state, unit)
        self.assertNotIn((1, 18), reachable)

    def test_infantry_can_enter_mountain(self) -> None:
        unit = self._place(UnitType.INFANTRY, 0, (2, 17))
        reachable = get_reachable_tiles(self.state, unit)
        self.assertIn(
            (1, 18), reachable,
            "Infantry (MOVE_INF) must be able to enter a mountain tile.",
        )

    def test_mech_can_enter_mountain(self) -> None:
        unit = self._place(UnitType.MECH, 0, (2, 17))
        reachable = get_reachable_tiles(self.state, unit)
        self.assertIn((1, 18), reachable)

    def test_bcopter_can_enter_mountain(self) -> None:
        unit = self._place(UnitType.B_COPTER, 0, (2, 17))
        reachable = get_reachable_tiles(self.state, unit)
        self.assertIn((1, 18), reachable)


class TestMoveUnitGuard(unittest.TestCase):
    """`_move_unit` must reject illegal destinations rather than silently apply."""

    def setUp(self) -> None:
        self.md = load_map(MAP_166877, POOL, MAPS_DIR)
        self.state = make_initial_state(self.md, 1, 7, starting_funds=0, tier_name="T2")
        self.state.units = {0: [], 1: []}

    def test_move_artillery_to_mountain_raises(self) -> None:
        art = _make_unit(UnitType.ARTILLERY, 0, (2, 17))
        self.state.units[0].append(art)
        with self.assertRaises(ValueError):
            self.state._move_unit(art, (1, 18))

    def test_move_infantry_to_mountain_ok(self) -> None:
        inf = _make_unit(UnitType.INFANTRY, 0, (2, 17))
        self.state.units[0].append(inf)
        self.state._move_unit(inf, (1, 18))
        self.assertEqual(inf.pos, (1, 18))


class TestDay1Income(unittest.TestCase):
    """Treasuries start 0g; P0 receives income at start of their first turn."""

    def test_starting_funds_are_zero_plus_income(self) -> None:
        md = load_map(MAP_166877, POOL, MAPS_DIR)
        st = make_initial_state(md, 1, 7, starting_funds=0, tier_name="T2")
        expected_p0 = st.count_income_properties(0) * 1000
        self.assertEqual(st.funds[0], expected_p0)
        # P1 has not had a turn start yet -> still 0g.
        self.assertEqual(st.funds[1], 0)

    def test_income_excludes_comm_towers_and_labs(self) -> None:
        md = load_map(MAP_166877, POOL, MAPS_DIR)
        st = make_initial_state(md, 1, 7, starting_funds=0, tier_name="T2")
        all_p0 = st.count_properties(0)
        income_p0 = st.count_income_properties(0)
        non_income = sum(
            1 for p in st.properties
            if p.owner == 0 and (p.is_comm_tower or p.is_lab)
        )
        self.assertEqual(all_p0 - non_income, income_p0)


if __name__ == "__main__":
    unittest.main()
