"""Unit tests for GL-style ``Move`` path / ``unit`` envelope resolution (oracle_move_no_unit)."""

from __future__ import annotations

import unittest

from tools.oracle_zip_replay import (
    _oracle_resolve_move_global_unit,
    _oracle_resolve_move_paths,
)


class TestOracleMoveResolve(unittest.TestCase):
    def test_per_seat_paths_and_unit_when_global_missing(self) -> None:
        """Some exports nest ``paths`` / ``unit`` only under the AWBW player id."""
        pid = 3712502
        move = {
            "paths": {
                str(pid): [
                    {"y": 5, "x": 5},
                    {"y": 5, "x": 6},
                ]
            },
            "unit": {
                str(pid): {
                    "units_id": 191234567,
                    "units_y": 5,
                    "units_x": 5,
                    "units_players_id": pid,
                    "units_name": "Infantry",
                }
            },
        }
        paths = _oracle_resolve_move_paths(move, pid)
        self.assertEqual(len(paths), 2)
        gu = _oracle_resolve_move_global_unit(move, pid)
        self.assertEqual(gu["units_id"], 191234567)
        self.assertEqual(int(gu["units_players_id"]), pid)

    def test_global_preferred_when_present(self) -> None:
        move = {
            "paths": {"global": [{"y": 1, "x": 1}, {"y": 2, "x": 1}]},
            "unit": {
                "global": {
                    "units_id": 100,
                    "units_y": 1,
                    "units_x": 1,
                    "units_players_id": 9,
                    "units_name": "Tank",
                }
            },
        }
        paths = _oracle_resolve_move_paths(move, None)
        self.assertEqual(len(paths), 2)
        gu = _oracle_resolve_move_global_unit(move, None)
        self.assertEqual(gu["units_id"], 100)


if __name__ == "__main__":
    unittest.main()
