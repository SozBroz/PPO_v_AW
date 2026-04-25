"""Koal Forced March (COP) grants +1 movement to all own units (AWBW parity).

Phase 11J-F2-KOAL: AWBW Power envelope for Koal COP carries
``global.units_movement_points: 1`` and unit snapshots show ``base + 1`` move
points regardless of starting tile (recon: phase11d_f2_recon.md). The wiki text
"+1 on road tiles" is a known mismatch with live-site behavior. The road -1
cost discount is a separate effect handled in
``engine/weather.py::effective_move_cost`` and stacks with the global +1.

Replay anchors:
- gid 1605367 (Mech, Koal vs Jess, T4, day 17, env 32) — needs Mech base 2 + 1.
- gid 1630794 (Inf, Jess vs Koal, T4, day 19, env 37) — needs Inf base 3 + 1.

SCOP "Trail of Woe" is intentionally NOT bumped globally here; only the road
cost discount applies (already handled in weather.py). See Test 3.
"""

from __future__ import annotations

import unittest

from engine.action import compute_reachable_costs
from engine.game import make_initial_state
from engine.map_loader import MapData
from engine.unit import UNIT_STATS, Unit, UnitType


def _strip_map(width: int, terrain_row: list[int]) -> MapData:
    return MapData(
        map_id=990_211,
        name="koal_cop_move_probe",
        map_type="std",
        terrain=[list(terrain_row)],
        height=1,
        width=width,
        cap_limit=999,
        unit_limit=50,
        unit_bans=[],
        tiers=[],
        objective_type=None,
        properties=[],
        hq_positions={0: [], 1: []},
        lab_positions={0: [], 1: []},
        country_to_player={},
        predeployed_specs=[],
    )


def _spawn(state, unit_type: UnitType, player: int, pos: tuple[int, int],
           unit_id: int = 1) -> Unit:
    stats = UNIT_STATS[unit_type]
    u = Unit(
        unit_type,
        player,
        100,
        stats.max_ammo,
        stats.max_fuel,
        pos,
        False,
        [],
        False,
        20,
        unit_id=unit_id,
    )
    state.units[player].append(u)
    return u


class TestKoalCopMovementBonus(unittest.TestCase):
    """Pin Koal CO-Power Forced March +1 global movement parity (AWBW)."""

    def test_mech_three_plain_tiles_requires_cop_bonus(self) -> None:
        """gid 1605367 mirror: Mech base 2, needs 3 MP (3 plains in a row)."""
        md = _strip_map(width=4, terrain_row=[1, 1, 1, 1])
        st = make_initial_state(md, 21, 14, starting_funds=0, tier_name="T4",
                                replay_first_mover=0)
        st.units = {0: [], 1: []}
        mech = _spawn(st, UnitType.MECH, 0, (0, 0))

        st.co_states[0].cop_active = False
        st.co_states[0].scop_active = False
        r0 = compute_reachable_costs(st, mech)
        self.assertNotIn((0, 3), r0,
                         "base Mech move is 2; cost-3 plain path must be unreachable without COP")

        st.co_states[0].cop_active = True
        r1 = compute_reachable_costs(st, mech)
        self.assertIn((0, 3), r1,
                     "Koal COP must lift Mech effective MP cap to 3 (base 2 + 1)")
        self.assertEqual(r1[(0, 3)], 3, "cost equals summed terrain (1+1+1)")

    def test_infantry_four_mixed_tiles_requires_cop_bonus(self) -> None:
        """gid 1630794 mirror: Infantry base 3, needs 4 MP across plain+wood+plain+plain."""
        md = _strip_map(width=5, terrain_row=[1, 1, 3, 1, 1])
        st = make_initial_state(md, 21, 14, starting_funds=0, tier_name="T4",
                                replay_first_mover=0)
        st.units = {0: [], 1: []}
        inf = _spawn(st, UnitType.INFANTRY, 0, (0, 0))

        st.co_states[0].cop_active = False
        st.co_states[0].scop_active = False
        r0 = compute_reachable_costs(st, inf)
        self.assertNotIn((0, 4), r0,
                         "base Infantry move is 3; cost-4 mixed path must be unreachable without COP")

        st.co_states[0].cop_active = True
        r1 = compute_reachable_costs(st, inf)
        self.assertIn((0, 4), r1,
                     "Koal COP must lift Infantry effective MP cap to 4 (base 3 + 1)")
        self.assertEqual(r1[(0, 4)], 4, "cost equals 1 (plain) + 1 (wood inf) + 1 + 1")

    def test_scop_does_not_grant_global_plus_one(self) -> None:
        """Trail of Woe SCOP is road-only (handled in weather.py); no global +1.

        If a future replay proves SCOP also grants global +1, add the bump in
        ``compute_reachable_costs`` and update this test.
        """
        md = _strip_map(width=4, terrain_row=[1, 1, 1, 1])
        st = make_initial_state(md, 21, 14, starting_funds=0, tier_name="T4",
                                replay_first_mover=0)
        st.units = {0: [], 1: []}
        mech = _spawn(st, UnitType.MECH, 0, (0, 0))

        st.co_states[0].cop_active = False
        st.co_states[0].scop_active = True
        r = compute_reachable_costs(st, mech)
        self.assertNotIn((0, 3), r,
                         "Koal SCOP currently grants only the road-cost discount, not global +1")
        self.assertIn((0, 2), r, "Mech base 2 still reaches 2 plain tiles under SCOP")

    def test_non_koal_co_does_not_get_koal_bonus(self) -> None:
        """Andy COP (heal-only) must NOT receive the +1 movement bump.

        The Koal branch is gated on ``co_id == 21``; this test guards against
        an over-broad change where a different CO accidentally inherits +1.
        """
        md = _strip_map(width=4, terrain_row=[1, 1, 1, 1])
        st = make_initial_state(md, 1, 14, starting_funds=0, tier_name="T4",
                                replay_first_mover=0)
        st.units = {0: [], 1: []}
        mech = _spawn(st, UnitType.MECH, 0, (0, 0))

        st.co_states[0].cop_active = True
        st.co_states[0].scop_active = False
        r = compute_reachable_costs(st, mech)
        self.assertNotIn((0, 3), r,
                         "Andy COP (Hyper Repair) is heal-only; no +1 movement")
        self.assertIn((0, 2), r)

    def test_cop_combines_with_road_discount(self) -> None:
        """Koal COP should stack: global +1 (action.py) AND road cost -1 (weather.py).

        Setup: Mech (base 2 MP) on a 5-tile road strip (HRoad tid 15). Without
        Koal effects each road tile costs the road table value (1 for mech) —
        cap = 2 → reaches col 2. With COP: cap = 3, road cost drops to max(0,
        1-1) = 0 per tile, so the Mech can sweep the entire 4-step strip.
        """
        md = _strip_map(width=5, terrain_row=[15, 15, 15, 15, 15])
        st = make_initial_state(md, 21, 14, starting_funds=0, tier_name="T4",
                                replay_first_mover=0)
        st.units = {0: [], 1: []}
        mech = _spawn(st, UnitType.MECH, 0, (0, 0))

        st.co_states[0].cop_active = False
        st.co_states[0].scop_active = False
        r0 = compute_reachable_costs(st, mech)
        self.assertIn((0, 2), r0, "base Mech reaches 2 road tiles (cost 1 each)")
        self.assertNotIn((0, 3), r0, "base Mech cap is 2 — third road tile out of range")

        st.co_states[0].cop_active = True
        r1 = compute_reachable_costs(st, mech)
        self.assertIn((0, 4), r1,
                     "with Koal COP: cap = 3 and road cost -> 0; reach the far end")
        self.assertEqual(r1[(0, 4)], 0,
                         "road discount drops every step to 0 MP; cumulative cost = 0")


if __name__ == "__main__":
    unittest.main()
