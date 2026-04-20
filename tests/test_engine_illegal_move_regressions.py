"""Regression pins for the ``engine_illegal_move`` / reachability cluster (GL Prompt 4).

The historical ``desync_register`` rows for games
1624281, 1625844, 1627935, 1629178, 1630151, 1632283, 1634030
were dominated by :class:`ValueError` from ``GameState._move_unit`` when the
oracle replay asked for a destination tile that ``compute_reachable_costs``
did not include.

Current ``tools/desync_audit.py`` runs for four of those zips without raising
(engine + oracle harness); three now stop earlier on unrelated ``oracle_gap``
actions. These tests lock **engine** rules that were wrong or missing in older
builds:

- **Neutral port (terrain 37)** must use :func:`engine.terrain._port_property_costs`
  so ``MOVE_LANDER`` is passable (Black Boat / Lander paths next to sea).
- **Adder** COP/SCOP move bonuses must include enough range for infantry to
  cross long road chains (GL **1625844** snapshot on map **140000**).

Other cases (Lash flattening, ``Capt`` approach skipping pipe tiles, stale
ACTION finish) are covered in ``test_movement_parity.py``,
``tests/test_oracle_capt_approach_tile.py``, and ``tests/test_oracle_fire_resolve.py``.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from engine.action import compute_reachable_costs
from engine.game import make_initial_state
from engine.map_loader import MapData, load_map
from engine.terrain import get_move_cost
from engine.unit import Unit, UnitType, UNIT_STATS

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"

SEA = 28
NEUTRAL_PORT = 37
NEUTRAL_CITY = 34


def _minimal_sea_port_map() -> MapData:
    """One neutral port orthogonally adjacent to sea (lander may enter the port)."""
    terrain = [
        [SEA, NEUTRAL_PORT],
        [SEA, SEA],
    ]
    return MapData(
        map_id=990_001,
        name="sea_port_probe",
        map_type="std",
        terrain=terrain,
        height=2,
        width=2,
        cap_limit=999,
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


class TestNeutralPortNavalReachability(unittest.TestCase):
    """Ports must not be ``INF`` for lander move-type (regression: neutral port cluster)."""

    def test_move_cost_lander_on_neutral_port_not_impassable(self) -> None:
        self.assertEqual(get_move_cost(NEUTRAL_PORT, "lander"), 1)
        self.assertGreaterEqual(get_move_cost(NEUTRAL_CITY, "lander"), 90)

    def test_lander_and_black_boat_reach_adjacent_port_from_sea(self) -> None:
        md = _minimal_sea_port_map()
        st = make_initial_state(md, 1, 1, starting_funds=0, tier_name="T2")
        st.units = {0: [], 1: []}
        for ut in (UnitType.LANDER, UnitType.BLACK_BOAT):
            with self.subTest(unit=ut):
                st.units[0].clear()
                stats = UNIT_STATS[ut]
                u = Unit(
                    unit_type=ut,
                    player=0,
                    hp=100,
                    ammo=stats.max_ammo if stats.max_ammo > 0 else 0,
                    fuel=stats.max_fuel,
                    pos=(0, 0),
                    moved=False,
                    loaded_units=[],
                    is_submerged=False,
                    capture_progress=20,
                    unit_id=1,
                )
                st.units[0].append(u)
                reach = compute_reachable_costs(st, u)
                self.assertIn((0, 1), reach, f"{ut.name} must reach adjacent neutral port from sea")


class TestAdderRoadReachGL1625844(unittest.TestCase):
    """Infantry on long ESRoad chains during Adder COP (map 140000, GL 1625844)."""

    def test_infantry_reaches_esroad_destination_during_adder_cop(self) -> None:
        md = load_map(140000, POOL, MAPS_DIR)
        self.assertEqual(md.terrain[12][6], 18)
        self.assertEqual(md.terrain[14][9], 18)
        st = make_initial_state(md, 1, 11, starting_funds=0, tier_name="T3")
        st.units = {0: [], 1: []}
        st.active_player = 1
        st.co_states[1].co_id = 11
        st.co_states[1].cop_active = True
        stats = UNIT_STATS[UnitType.INFANTRY]
        u = Unit(
            unit_type=UnitType.INFANTRY,
            player=1,
            hp=100,
            ammo=0,
            fuel=93,
            pos=(12, 6),
            moved=False,
            loaded_units=[],
            is_submerged=False,
            capture_progress=20,
            unit_id=99014000001,
        )
        st.units[1].append(u)
        reach = compute_reachable_costs(st, u)
        self.assertIn((14, 9), reach)
        self.assertEqual(reach[(14, 9)], 5)


if __name__ == "__main__":
    unittest.main()
