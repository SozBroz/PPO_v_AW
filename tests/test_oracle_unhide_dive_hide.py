"""Oracle ``Unhide`` mirrors ``Hide``: nested ``Move`` then ``ActionType.DIVE_HIDE`` (surface / unhide)."""

from __future__ import annotations

import unittest
from pathlib import Path

from engine.action import ActionStage
from engine.game import make_initial_state
from engine.map_loader import load_map
from engine.unit import UNIT_STATS, Unit, UnitType

from tools.oracle_zip_replay import apply_oracle_action_json

ROOT = Path(__file__).resolve().parents[1]


class TestOracleUnhideDiveHide(unittest.TestCase):
    def test_unhide_no_path_surfaces_sub_via_dive_hide(self) -> None:
        md = load_map(
            126428,
            ROOT / "data" / "gl_map_pool.json",
            ROOT / "data" / "maps",
        )
        st = make_initial_state(md, 14, 21, tier_name="T4", starting_funds=0)
        st.units[0] = []
        st.units[1] = []
        st_sub = UNIT_STATS[UnitType.SUBMARINE]
        sub = Unit(
            UnitType.SUBMARINE,
            1,
            70,
            st_sub.max_ammo,
            47,
            (0, 12),
            False,
            [],
            True,
            20,
            424242,
        )
        st.units[1].append(sub)
        st.active_player = 1
        st.action_stage = ActionStage.SELECT
        aw_pid = 90001
        move_empty = {
            "action": "Move",
            "unit": {
                "global": {
                    "units_id": 424242,
                    "units_players_id": aw_pid,
                    "units_y": 0,
                    "units_x": 12,
                    "units_name": "Sub",
                }
            },
            "paths": {str(aw_pid): []},
        }
        apply_oracle_action_json(
            st,
            {"action": "Unhide", "Move": move_empty},
            {aw_pid: 1},
            envelope_awbw_player_id=aw_pid,
        )
        self.assertFalse(sub.is_submerged)


if __name__ == "__main__":
    unittest.main()
