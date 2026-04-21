"""Unit tests for ``Hide`` / ``Unhide`` when ``Move`` is omitted or ``[]`` (hide/dive in place)."""

from __future__ import annotations

import unittest

from engine.unit import UNIT_STATS, Unit, UnitType
from tools.oracle_zip_replay import (
    _oracle_resolve_nested_hide_unhide_units_id,
    _oracle_sole_dive_hide_actor_for_player,
)


def _mk_unit(ut: UnitType, player: int, pos: tuple[int, int], uid: int) -> Unit:
    s = UNIT_STATS[ut]
    return Unit(
        unit_type=ut,
        player=player,
        hp=100,
        ammo=s.max_ammo if s.max_ammo > 0 else 0,
        fuel=s.max_fuel,
        pos=pos,
        moved=False,
        loaded_units=[],
        is_submerged=False,
        capture_progress=20,
        unit_id=uid,
    )


class TestOracleHideNoMove(unittest.TestCase):
    def test_compact_vision_unit_map_prefers_envelope_seat(self) -> None:
        """GL replay 1637988: ``unit`` maps AWBW player id -> units_id (scalars)."""
        pid = 3783288
        nested = {
            "action": "Hide",
            "unit": {"3783287": 192834303, "3783288": 192834303},
            "vision": {"3783287": True, "3783288": True},
        }
        self.assertEqual(
            _oracle_resolve_nested_hide_unhide_units_id(nested, pid),
            192834303,
        )

    def test_rich_global_unit_dict(self) -> None:
        pid = 100
        nested = {
            "action": "Hide",
            "unit": {
                "global": {
                    "units_id": 42,
                    "units_y": 3,
                    "units_x": 4,
                    "units_players_id": pid,
                }
            },
        }
        self.assertEqual(
            _oracle_resolve_nested_hide_unhide_units_id(nested, pid),
            42,
        )

    def test_sole_dive_actor_when_unique(self) -> None:
        stealth = _mk_unit(UnitType.STEALTH, 0, (6, 3), 51)
        inf = _mk_unit(UnitType.INFANTRY, 0, (5, 5), 2)
        st = type("S", (), {})()
        st.units = {0: [stealth, inf], 1: []}
        self.assertIs(_oracle_sole_dive_hide_actor_for_player(st, 0), stealth)

    def test_sole_dive_actor_ambiguous_returns_none(self) -> None:
        a = _mk_unit(UnitType.SUBMARINE, 0, (1, 1), 1)
        b = _mk_unit(UnitType.SUBMARINE, 0, (2, 2), 2)
        st = type("S", (), {})()
        st.units = {0: [a, b], 1: []}
        self.assertIsNone(_oracle_sole_dive_hide_actor_for_player(st, 0))


if __name__ == "__main__":
    unittest.main()
