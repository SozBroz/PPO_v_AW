"""Per-seat overlay when ``Move.unit.global`` uses fog HP — GL 1628722 oracle_move_no_unit."""

from __future__ import annotations

import unittest

from tools.oracle_zip_replay import (
    _oracle_merge_global_move_unit_with_seat_hints,
    _oracle_resolve_move_global_unit,
)


class TestOracleMoveNoUnitSeatHintMerge(unittest.TestCase):
    def test_global_question_mark_hp_pulls_from_envelope_seat(self) -> None:
        """Site ``global`` can hide HP as ``?`` while the acting seat lists real bars."""
        uwrap = {
            "global": {
                "units_id": 192428454,
                "units_players_id": 3763678,
                "units_name": "Mech",
                "units_y": 8,
                "units_x": 16,
                "units_hit_points": "?",
            },
            "3763678": {
                "units_id": 192428454,
                "units_players_id": 3763678,
                "units_name": "Mech",
                "units_y": 8,
                "units_x": 16,
                "units_hit_points": 1,
            },
        }
        gl = uwrap["global"]
        merged = _oracle_merge_global_move_unit_with_seat_hints(
            uwrap, gl, envelope_awbw_player_id=3763678
        )
        self.assertEqual(merged["units_hit_points"], 1)

    def test_resolve_move_global_unit_end_to_end(self) -> None:
        move = {
            "action": "Move",
            "unit": {
                "global": {
                    "units_id": 192428454,
                    "units_players_id": 3763678,
                    "units_name": "Mech",
                    "units_y": 8,
                    "units_x": 16,
                    "units_hit_points": "?",
                },
                "3763678": {
                    "units_id": 192428454,
                    "units_players_id": 3763678,
                    "units_name": "Mech",
                    "units_y": 8,
                    "units_x": 16,
                    "units_hit_points": 1,
                },
            },
            "paths": {"global": [{"y": 7, "x": 17}, {"y": 8, "x": 16}]},
        }
        gu = _oracle_resolve_move_global_unit(move, envelope_awbw_player_id=3763678)
        self.assertEqual(gu.get("units_hit_points"), 1)

    def test_mismatching_units_id_does_not_overlay(self) -> None:
        uwrap = {
            "global": {"units_id": 1, "units_hit_points": "?"},
            "3763678": {"units_id": 2, "units_hit_points": 9},
        }
        merged = _oracle_merge_global_move_unit_with_seat_hints(
            uwrap, uwrap["global"], envelope_awbw_player_id=3763678
        )
        self.assertEqual(merged["units_hit_points"], "?")


if __name__ == "__main__":
    unittest.main()
