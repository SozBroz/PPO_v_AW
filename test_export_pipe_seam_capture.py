"""Pipe seam buildings in PHP snapshots use capture=99 for full seam HP (AWBW)."""
from __future__ import annotations

import unittest

from tools.export_awbw_replay import _awbw_non_property_building_capture


class TestNonPropertyBuildingCapture(unittest.TestCase):
    def test_pipe_seams_use_99(self) -> None:
        self.assertEqual(_awbw_non_property_building_capture(113), 99)
        self.assertEqual(_awbw_non_property_building_capture(114), 99)

    def test_missile_silos_use_20(self) -> None:
        self.assertEqual(_awbw_non_property_building_capture(111), 20)
        self.assertEqual(_awbw_non_property_building_capture(112), 20)


if __name__ == "__main__":
    unittest.main()
