"""Predeploy ``*_units.json`` payload builder for OS/BM normalize."""
from __future__ import annotations

import unittest

from engine.predeployed import PredeployedUnitSpec
from engine.unit import UnitType

from tools.normalize_map_to_os_bm import _build_units_json_payload


class TestNormalizeMapOsBmUnitsPayload(unittest.TestCase):
    def test_preserves_non_units_keys_and_schema(self) -> None:
        specs = [
            PredeployedUnitSpec(row=3, col=6, player=0, unit_type=UnitType.INFANTRY, hp=100)
        ]
        preserved = {"schema_version": 1, "_source": "fixture note"}
        got = _build_units_json_payload(specs, preserved)
        self.assertEqual(got["_source"], "fixture note")
        self.assertEqual(got["schema_version"], 1)
        self.assertEqual(len(got["units"]), 1)
        self.assertEqual(got["units"][0]["player"], 0)
        self.assertEqual(got["units"][0]["unit_type"], "INFANTRY")

    def test_force_engine_player_round_trip(self) -> None:
        specs = [
            PredeployedUnitSpec(
                row=1,
                col=1,
                player=0,
                unit_type=UnitType.TANK,
                hp=100,
                force_engine_player=0,
            )
        ]
        got = _build_units_json_payload(specs, {})
        self.assertEqual(got["units"][0].get("force_engine_player"), 0)


if __name__ == "__main__":
    unittest.main()
