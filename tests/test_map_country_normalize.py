"""OS/BM terrain normalization for debugging-standard maps."""
from __future__ import annotations

import unittest

from engine.map_country_normalize import remap_property_terrain_id_to_os_bm
from engine.terrain import get_terrain


class TestRemapPropertyToOsBm(unittest.TestCase):
    def test_yc_base_to_os_when_engine_p0_is_yc(self) -> None:
        # Yellow Comet base (country 4) -> Orange Star base when P0 is YC
        tid = 54
        self.assertEqual(get_terrain(tid).country_id, 4)
        got = remap_property_terrain_id_to_os_bm(
            tid, engine_p0_country_id=4, engine_p1_country_id=7
        )
        self.assertEqual(got, 39)
        self.assertEqual(get_terrain(got).country_id, 1)

    def test_gs_base_to_bm_when_engine_p1_is_gs(self) -> None:
        tid = 87  # Grey Sky base, country 7
        got = remap_property_terrain_id_to_os_bm(
            tid, engine_p0_country_id=4, engine_p1_country_id=7
        )
        self.assertEqual(got, 44)
        self.assertEqual(get_terrain(got).country_id, 2)

    def test_neutral_city_unchanged(self) -> None:
        self.assertEqual(
            remap_property_terrain_id_to_os_bm(
                34, engine_p0_country_id=4, engine_p1_country_id=7
            ),
            34,
        )

    def test_plain_unchanged(self) -> None:
        self.assertEqual(
            remap_property_terrain_id_to_os_bm(
                1, engine_p0_country_id=4, engine_p1_country_id=7
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
