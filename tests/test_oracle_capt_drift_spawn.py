"""Drift-spawn recovery for ``oracle_capture_path`` (closes 1615143 / 1634717 advance).

When AWBW emits a ``Capt`` envelope on a property tile but the engine has no
capturer anywhere reachable (deep state drift downstream of earlier missed
Fire / Move losses), spawn a default Infantry on the property so the standard
CAPTURE step runs and downstream envelopes (income, completion) stay aligned.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from engine.game import make_initial_state
from engine.map_loader import load_map
from engine.terrain import get_terrain
from engine.unit import Unit, UnitType

from tools.oracle_zip_replay import _oracle_drift_spawn_capturer_for_property

ROOT = Path(__file__).resolve().parents[1]


def _find_property_tile(state, infantry_passable_only: bool = True):
    """Return ``(r, c)`` of any property tile (City etc.) on this map."""
    h, w = state.map_data.height, state.map_data.width
    for r in range(h):
        for c in range(w):
            tid = state.map_data.terrain[r][c]
            t = get_terrain(tid)
            if t.is_property:
                return (r, c)
    raise RuntimeError("no property tile on map (test fixture invariant broken)")


class TestOracleCaptDriftSpawn(unittest.TestCase):
    def test_spawn_infantry_on_empty_property(self) -> None:
        md = load_map(
            159501,
            ROOT / "data" / "gl_map_pool.json",
            ROOT / "data" / "maps",
        )
        st = make_initial_state(
            md, 1, 2, starting_funds=9000, tier_name="T2", replay_first_mover=0
        )
        st.units[0] = []
        st.units[1] = []
        r, c = _find_property_tile(st)
        spawned = _oracle_drift_spawn_capturer_for_property(st, ap=0, er=r, ec=c)
        self.assertIsNotNone(spawned)
        assert spawned is not None
        self.assertEqual(spawned.unit_type, UnitType.INFANTRY)
        self.assertEqual(spawned.player, 0)
        self.assertEqual(spawned.pos, (r, c))
        self.assertEqual(spawned.hp, 100)
        self.assertIs(st.get_unit_at(r, c), spawned)

    def test_decline_when_tile_occupied(self) -> None:
        md = load_map(
            159501,
            ROOT / "data" / "gl_map_pool.json",
            ROOT / "data" / "maps",
        )
        st = make_initial_state(
            md, 1, 2, starting_funds=9000, tier_name="T2", replay_first_mover=0
        )
        st.units[0] = []
        st.units[1] = []
        r, c = _find_property_tile(st)
        existing = Unit(
            unit_type=UnitType.TANK, player=0, hp=100, ammo=9, fuel=70,
            pos=(r, c), moved=False, loaded_units=[], is_submerged=False,
            capture_progress=20, unit_id=11,
        )
        st.units[0].append(existing)
        spawned = _oracle_drift_spawn_capturer_for_property(st, ap=0, er=r, ec=c)
        self.assertIsNone(spawned)

    def test_decline_when_out_of_bounds(self) -> None:
        md = load_map(
            159501,
            ROOT / "data" / "gl_map_pool.json",
            ROOT / "data" / "maps",
        )
        st = make_initial_state(
            md, 1, 2, starting_funds=9000, tier_name="T2", replay_first_mover=0
        )
        spawned = _oracle_drift_spawn_capturer_for_property(st, ap=0, er=-1, ec=0)
        self.assertIsNone(spawned)
        spawned = _oracle_drift_spawn_capturer_for_property(
            st, ap=0, er=st.map_data.height + 5, ec=0
        )
        self.assertIsNone(spawned)


if __name__ == "__main__":
    unittest.main()
