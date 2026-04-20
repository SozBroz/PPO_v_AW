"""PHP ``'?'`` / non-numeric JSON must become ``oracle_gap``, not ``engine_bug``."""

from __future__ import annotations

import unittest

from test_build_guard import _minimal_state

from tools.oracle_zip_replay import (
    UnsupportedOracleAction,
    _oracle_awbw_scalar_int_optional,
    apply_oracle_action_json,
)


class TestOracleMalformedJsonInt(unittest.TestCase):
    def test_awbw_scalar_int_optional_placeholders(self) -> None:
        self.assertIsNone(_oracle_awbw_scalar_int_optional("?"))
        self.assertIsNone(_oracle_awbw_scalar_int_optional(""))
        self.assertIsNone(_oracle_awbw_scalar_int_optional(None))
        self.assertIsNone(_oracle_awbw_scalar_int_optional(True))
        self.assertEqual(_oracle_awbw_scalar_int_optional("192463031"), 192463031)
        self.assertEqual(_oracle_awbw_scalar_int_optional(7), 7)

    def test_build_units_y_placeholder_becomes_oracle_gap(self) -> None:
        state = _minimal_state(active_player=0, factory_owner=0)
        obj = {
            "action": "Build",
            "unit": {
                "global": {
                    "units_y": "?",
                    "units_x": 1,
                    "units_name": "Infantry",
                    "units_players_id": 1,
                }
            },
        }
        with self.assertRaises(UnsupportedOracleAction) as ctx:
            apply_oracle_action_json(state, obj, {1: 0}, envelope_awbw_player_id=1)
        msg = str(ctx.exception)
        self.assertIn("Malformed AWBW action JSON", msg)
        self.assertIn("action='Build'", msg)
        self.assertIn("?", msg.lower())


if __name__ == "__main__":
    unittest.main()
