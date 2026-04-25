"""Kindle (CO 23) D2D income — canon-of-record pin.

**Phase 11A outcome: rollback.**

Original 11A plan (per ``data/co_data.json`` + community wiki) was to grant
Kindle **+50% funds from owned cities** in ``GameState._grant_income``.
``tools/_phase10n_drilldown.py`` on game ``1628546`` (Kindle vs Max, map
159501) demonstrated that the live PHP oracle does **not** apply this bonus:
the very first Kindle city capture (turn 4 grant) put engine 500g over PHP,
pulling the first funds mismatch from envelope 11 (pre-fix) to envelope 5
(post-fix).

The AWBW CO Chart (https://awbw.amarriner.com/co.php) is **silent** on
Kindle income — the +50% line is a ``co_data.json`` / wiki claim, flagged
as a primary-source discrepancy in
``docs/oracle_exception_audit/phase10t_co_income_audit.md`` Section 3 and
documented in ``phase11a_kindle_hachi_canon.md``. The chart and the live
PHP oracle agree: **no Kindle income D2D**.

This test pins the rollback. If a future audit proves the bonus IS applied
(e.g. via a more representative drill), revisit the rollback record before
flipping the assertions.
"""

from __future__ import annotations

import unittest

from engine.action import ActionStage
from engine.co import make_co_state_safe
from engine.game import GameState
from engine.map_loader import MapData, PropertyState


# ---- terrain ids (engine/terrain.py) -------------------------------------
NEUTRAL_CITY = 34
NEUTRAL_BASE = 35


def _state_with_props(*, p0_co_id: int, n_cities: int, n_bases: int) -> GameState:
    """Build a 1-row fixture map with ``n_cities`` neutral cities then
    ``n_bases`` neutral bases, all owned by P0."""
    width = n_cities + n_bases
    if width == 0:
        width = 1  # MapData refuses width=0
    terrain = [[1] * width]  # default plains
    properties: list[PropertyState] = []
    col = 0
    for _ in range(n_cities):
        terrain[0][col] = NEUTRAL_CITY
        properties.append(PropertyState(
            terrain_id=NEUTRAL_CITY, row=0, col=col, owner=0, capture_points=20,
            is_hq=False, is_lab=False, is_comm_tower=False,
            is_base=False, is_airport=False, is_port=False,
        ))
        col += 1
    for _ in range(n_bases):
        terrain[0][col] = NEUTRAL_BASE
        properties.append(PropertyState(
            terrain_id=NEUTRAL_BASE, row=0, col=col, owner=0, capture_points=20,
            is_hq=False, is_lab=False, is_comm_tower=False,
            is_base=True, is_airport=False, is_port=False,
        ))
        col += 1
    map_data = MapData(
        map_id=0, name="kindle-income", map_type="std",
        terrain=terrain, height=1, width=width,
        cap_limit=99, unit_limit=50, unit_bans=[], tiers=[],
        objective_type=None, properties=properties,
        hq_positions={0: [], 1: []}, lab_positions={0: [], 1: []},
        country_to_player={},
    )
    return GameState(
        map_data=map_data,
        units={0: [], 1: []},
        funds=[0, 0],
        co_states=[make_co_state_safe(p0_co_id), make_co_state_safe(0)],
        properties=map_data.properties,
        turn=1,
        active_player=0,
        action_stage=ActionStage.SELECT,
        selected_unit=None,
        selected_move_pos=None,
        done=False,
        winner=None,
        win_reason=None,
        game_log=[],
        tier_name="T2",
        full_trace=[],
    )


class TestKindleIncomeRollbackPin(unittest.TestCase):
    """Pin: Kindle currently grants flat 1000g per income property — the
    +50%/city bonus from ``co_data.json`` is **not** applied.
    """

    def test_kindle_on_5_cities_2_bases_grants_flat_7000(self) -> None:
        # If/when the +50% city bonus is re-introduced (with PHP-verified
        # support), the expected value flips to 9500 (5 × 1500 + 2 × 1000).
        st = _state_with_props(p0_co_id=23, n_cities=5, n_bases=2)
        st._grant_income(0)
        self.assertEqual(
            st.funds[0], 7000,
            "Kindle currently matches non-Kindle baseline (PHP oracle "
            "rejected the +50%/city bonus on game 1628546 — see "
            "phase11a_kindle_hachi_canon.md).",
        )

    def test_andy_on_5_cities_2_bases_grants_7000(self) -> None:
        st = _state_with_props(p0_co_id=1, n_cities=5, n_bases=2)
        st._grant_income(0)
        self.assertEqual(st.funds[0], 7000)

    def test_kindle_matches_andy_until_canon_resolved(self) -> None:
        # Cross-CO parity check: until the income-discrepancy is settled
        # against PHP, Kindle and Andy must produce the same daily income.
        st_k = _state_with_props(p0_co_id=23, n_cities=5, n_bases=2)
        st_a = _state_with_props(p0_co_id=1,  n_cities=5, n_bases=2)
        st_k._grant_income(0)
        st_a._grant_income(0)
        self.assertEqual(st_k.funds[0], st_a.funds[0])


if __name__ == "__main__":
    unittest.main()
