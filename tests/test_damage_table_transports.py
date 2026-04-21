"""Ground units vs Lander/Gunboat/Black Boat — oracle_fire / ``get_attack_targets`` coverage.

The matrix had nulls for several direct-fire units vs shallow-water transports, so
``get_base_damage`` returned None and adjacent strikes were hidden. AWBW allows
these attacks (e.g. Tank vs Black Boat); game 1628008 failed in ``Fire (no path)``
until the table was filled.
"""
from __future__ import annotations

import unittest

from engine.action import get_attack_targets
from engine.combat import get_base_damage
from engine.unit import UnitType

from test_lander_and_fuel import _fresh_state, _make_unit


class TestDamageTableNavalTransports(unittest.TestCase):
    def test_tank_black_boat_base_damage(self) -> None:
        # 2026-04 (Phase 11J-DAMAGE-CANON): PHP canon (https://awbw.amarriner.com/damage.php)
        # row Tank, col BlackBoat = 10. Earlier value 55 was a hypothesis; site
        # never disagreed with the audit either way (zero regressions on flip).
        self.assertEqual(get_base_damage(UnitType.TANK, UnitType.BLACK_BOAT), 10)

    def test_tank_can_select_adjacent_black_boat(self) -> None:
        st = _fresh_state()
        st.active_player = 0
        tank = _make_unit(st, UnitType.TANK, 0, (1, 2))
        _make_unit(st, UnitType.BLACK_BOAT, 1, (0, 2))
        self.assertIn((0, 2), get_attack_targets(st, tank, tank.pos))


if __name__ == "__main__":
    unittest.main()
