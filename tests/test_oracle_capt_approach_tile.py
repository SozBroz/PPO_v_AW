"""Oracle ``Capt`` no-path: approach tile on the building orth ring must be passable."""

from __future__ import annotations

import unittest
from pathlib import Path

from engine.action import ActionStage
from engine.game import make_initial_state
from engine.map_loader import load_map
from engine.unit import UNIT_STATS, Unit, UnitType

from tools.oracle_zip_replay import (
    _oracle_capt_no_path_empty_orth_touching_unit,
    apply_oracle_action_json,
)

ROOT = Path(__file__).resolve().parents[1]


class TestOracleCaptApproachTile(unittest.TestCase):
    def test_map_159501_skips_pipe_for_infantry_approach(self) -> None:
        """Map 159501 row 0: ``(0,1)`` is ESPipe (104); port capture at ``(1,1)`` from ``(0,2)``.

        Grounded in ``data/maps/159501.csv`` and replay ``1630151`` (Global League):
        the empty orth neighbour of the port that is one step from a diagonal
        capturer on ``(0,2)`` must be shoal ``(1,2)``, not lexicographic pipe ``(0,1)``.
        """
        md = load_map(
            159501,
            ROOT / "data" / "gl_map_pool.json",
            ROOT / "data" / "maps",
        )
        st = make_initial_state(
            md,
            1,
            2,
            starting_funds=9000,
            tier_name="T2",
            replay_first_mover=0,
        )
        st.units[0] = [u for u in st.units[0] if u.pos != (0, 2)]
        st.units[1] = []
        inf = Unit(
            unit_type=UnitType.INFANTRY,
            player=0,
            hp=100,
            ammo=0,
            fuel=99,
            pos=(0, 2),
            moved=False,
            loaded_units=[],
            is_submerged=False,
            capture_progress=20,
            unit_id=99015950101,
        )
        st.units[0].append(inf)
        t = _oracle_capt_no_path_empty_orth_touching_unit(st, 1, 1, inf)
        self.assertEqual(t, (1, 2))

    def test_buildings_capture_mid_syncs_property_when_no_capturer_reachable(self) -> None:
        """Mid-capture ``Capt`` / ``Move:[]`` with no engine occupant on the tile: oracle syncs ``capture_points``."""
        md = load_map(
            159501,
            ROOT / "data" / "gl_map_pool.json",
            ROOT / "data" / "maps",
        )
        st = make_initial_state(
            md,
            1,
            2,
            starting_funds=9000,
            tier_name="T2",
            replay_first_mover=0,
        )
        st.units[0] = [u for u in st.units[0] if u.pos != (0, 2)]
        st.units[1] = []
        tt = UNIT_STATS[UnitType.TANK]
        st.units[1].append(
            Unit(
                UnitType.TANK,
                1,
                100,
                tt.max_ammo,
                tt.max_fuel,
                (1, 2),
                False,
                [],
                False,
                20,
                99015950199,
            )
        )
        inf = Unit(
            unit_type=UnitType.INFANTRY,
            player=0,
            hp=100,
            ammo=0,
            fuel=99,
            pos=(0, 2),
            moved=False,
            loaded_units=[],
            is_submerged=False,
            capture_progress=20,
            unit_id=99015950102,
        )
        st.units[0].append(inf)
        st.active_player = 0
        st.action_stage = ActionStage.SELECT
        capt = {
            "action": "Capt",
            "Move": [],
            "Capt": {
                "buildingInfo": {
                    "buildings_y": 1,
                    "buildings_x": 1,
                    "buildings_capture": 1,
                },
            },
        }
        apply_oracle_action_json(
            st,
            capt,
            {90001: 0},
            envelope_awbw_player_id=90001,
        )
        prop = st.get_property_at(1, 1)
        self.assertIsNotNone(prop)
        assert prop is not None
        self.assertEqual(prop.capture_points, 1)


if __name__ == "__main__":
    unittest.main()
