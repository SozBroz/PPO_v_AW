"""``_resolve_unload_transport``: carrier vs cargo ``transportID``, hull vs map (PHP ids)."""

from __future__ import annotations

import unittest

from engine.game import make_initial_state
from engine.map_loader import load_map
from engine.unit import UNIT_STATS, Unit, UnitType

from server.play_human import MAPS_DIR, POOL_PATH
from tools.oracle_zip_replay import (
    UnsupportedOracleAction,
    _merge_move_gu_fields,
    _oracle_unload_unit_global_for_envelope,
    _resolve_unload_transport,
)


def _foot(ut: UnitType, unit_id: int, player: int = 1) -> Unit:
    st = UNIT_STATS[ut]
    return Unit(
        ut,
        player,
        100,
        st.max_ammo,
        st.max_fuel,
        (0, 0),
        True,
        [],
        False,
        20,
        unit_id,
    )


def _apc(pos: tuple[int, int], uid: int, cargo: list[Unit]) -> Unit:
    st = UNIT_STATS[UnitType.APC]
    return Unit(
        UnitType.APC,
        1,
        100,
        st.max_ammo,
        st.max_fuel,
        pos,
        False,
        cargo,
        False,
        20,
        uid,
    )


class TestOracleUnloadMergeGlobals(unittest.TestCase):
    def test_merge_seat_units_players_with_global_units_name_gl_1622104(
        self,
    ) -> None:
        """games_id 1622104: GL splits ``units_players_id`` (seat) vs ``units_name`` (global)."""
        obj = {
            "unit": {
                "555": {
                    "units_players_id": 12345,
                    "units_y": 5,
                    "units_x": 6,
                },
                "global": {"units_name": "INFANTRY"},
            },
        }
        gu = _oracle_unload_unit_global_for_envelope(obj, 555)
        self.assertIsNotNone(gu.get("units_players_id"))
        flat = obj["unit"]["global"]
        merged = _merge_move_gu_fields(gu, flat)
        self.assertEqual(str(merged.get("units_name")), "INFANTRY")


class TestOracleUnloadTransportResolve(unittest.TestCase):
    def setUp(self) -> None:
        m = load_map(126428, POOL_PATH, MAPS_DIR)
        self.state = make_initial_state(m, 14, 21, tier_name="T4", starting_funds=0)
        self.state.units[0] = []
        self.state.units[1] = []

    def test_carrier_id_requires_matching_cargo_not_any_hull(self) -> None:
        """``transportID`` = carrier A; A holds wrong class; pick B by drop adjacency."""
        s = self.state
        mech_c = _foot(UnitType.MECH, 200)
        inf_c = _foot(UnitType.INFANTRY, 201)
        a = _apc((10, 10), 100, [mech_c])
        b = _apc((10, 11), 300, [inf_c])
        s.units[1].extend([a, b])
        t = _resolve_unload_transport(
            s,
            100,
            UnitType.INFANTRY,
            (9, 11),
            1,
            cargo_awbw_units_id=201,
        )
        self.assertIs(t, b)

    def test_transport_id_cargo_in_loaded_units(self) -> None:
        """Site keys ``transportID`` to the drawable cargo id (not on ``state.units``)."""
        s = self.state
        inf_c = _foot(UnitType.INFANTRY, 888)
        b = _apc((10, 11), 300, [inf_c])
        s.units[1].append(b)
        t = _resolve_unload_transport(
            s,
            888,
            UnitType.INFANTRY,
            (9, 11),
            1,
            cargo_awbw_units_id=888,
        )
        self.assertIs(t, b)

    def test_carrier_id_single_matching_cargo(self) -> None:
        s = self.state
        inf_c = _foot(UnitType.INFANTRY, 201)
        b = _apc((10, 11), 300, [inf_c])
        s.units[1].append(b)
        t = _resolve_unload_transport(
            s,
            300,
            UnitType.INFANTRY,
            (9, 11),
            1,
            cargo_awbw_units_id=201,
        )
        self.assertIs(t, b)

    def test_two_same_type_cargo_disambiguate_by_units_id(self) -> None:
        s = self.state
        i1 = _foot(UnitType.INFANTRY, 501)
        i2 = _foot(UnitType.INFANTRY, 502)
        cop = _apc((5, 5), 400, [i1, i2])
        s.units[1].append(cop)
        t = _resolve_unload_transport(
            s,
            400,
            UnitType.INFANTRY,
            (4, 5),
            1,
            cargo_awbw_units_id=502,
        )
        self.assertIs(t, cop)

    def test_two_carriers_equidistant_raises_without_id_signals(self) -> None:
        """No ``transportID`` / ``units_id`` tie-break when geometry ties (do not guess)."""
        s = self.state
        a = _apc((5, 4), 400, [_foot(UnitType.INFANTRY, 501)])
        b = _apc((5, 6), 401, [_foot(UnitType.INFANTRY, 502)])
        s.units[1].extend([a, b])
        with self.assertRaises(UnsupportedOracleAction):
            _resolve_unload_transport(
                s,
                999,
                UnitType.INFANTRY,
                (4, 5),
                1,
                cargo_awbw_units_id=None,
            )


if __name__ == "__main__":
    unittest.main()
