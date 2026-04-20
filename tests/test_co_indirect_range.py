"""Indirect attack range from CO powers (Grit, Jake) — used by ``get_attack_targets``."""
from __future__ import annotations

import unittest

from engine.action import get_attack_targets
from engine.co import make_co_state
from engine.unit import UnitType

from test_lander_and_fuel import _fresh_state, _make_unit


class TestCoIndirectRange(unittest.TestCase):
    def test_jake_cop_extends_artillery_max_range(self) -> None:
        """Jake Beat Down / Block Rock: +1 max range for land indirects (AWBW wiki)."""
        st = _fresh_map()
        st.co_states[0] = make_co_state(22)
        st.co_states[0].cop_active = False
        ar = _make_unit(st, UnitType.ARTILLERY, 0, (2, 0))
        _make_unit(st, UnitType.INFANTRY, 1, (2, 4))
        self.assertNotIn((2, 4), get_attack_targets(st, ar, ar.pos))
        st.co_states[0].cop_active = True
        self.assertIn((2, 4), get_attack_targets(st, ar, ar.pos))

    def test_grit_cop_plus_one_indirect(self) -> None:
        st = _fresh_map()
        st.co_states[0] = make_co_state(2)
        st.co_states[0].cop_active = True
        ar = _make_unit(st, UnitType.ARTILLERY, 0, (2, 0))
        _make_unit(st, UnitType.TANK, 1, (2, 4))
        self.assertIn((2, 4), get_attack_targets(st, ar, ar.pos))


def _fresh_map():
    st = _fresh_state()
    st.active_player = 0
    return st


if __name__ == "__main__":
    unittest.main()
