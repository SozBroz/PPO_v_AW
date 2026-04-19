"""APC refuel correctness and no-op WAIT pruning.

Two rules pinned here:

  1. AWBW APCs resupply every adjacent allied unit on WAIT — regardless of
     unit class (ground, air, naval). ``GameState._apc_resupply`` must bring
     each neighbour to their respective ``max_fuel`` and, if armed, to
     ``max_ammo``.
  2. RL action-space pruning: an empty APC whose current MOVE destination
     has no adjacent allied unit needing supply drops its ``WAIT`` when
     another reachable tile **would** benefit an ally. WAIT stays available
     whenever (a) this tile would trigger a real resupply, (b) there is no
     better tile, (c) the APC has cargo aboard, or (d) an UNLOAD is legal.
"""
from __future__ import annotations

import unittest

from engine.action import (
    Action, ActionType, ActionStage,
    get_legal_actions,
)
from engine.game import make_initial_state
from engine.map_loader import MapData, PropertyState
from engine.unit import Unit, UnitType, UNIT_STATS

from test_lander_and_fuel import _fresh_state, _make_unit, _select_and_move

PLAIN = 1
NEUTRAL_BASE = 35


def _state_apc_on_owned_base_with_needy_inf() -> tuple:
    """5×5 plain; tile (3,3) is an owned base (APC can BUILD there)."""
    terrain = [[PLAIN] * 5 for _ in range(5)]
    terrain[3][3] = NEUTRAL_BASE
    prop = PropertyState(
        terrain_id=NEUTRAL_BASE,
        row=3,
        col=3,
        owner=0,
        capture_points=20,
        is_hq=False,
        is_lab=False,
        is_comm_tower=False,
        is_base=True,
        is_airport=False,
        is_port=False,
    )
    md = MapData(
        map_id=999_997,
        name="apc_prune_build_test",
        map_type="std",
        terrain=terrain,
        height=5,
        width=5,
        cap_limit=999,
        unit_limit=50,
        unit_bans=[],
        tiers=[],
        objective_type=None,
        properties=[prop],
        hq_positions={},
        lab_positions={},
        country_to_player={},
        predeployed_specs=[],
    )
    st = make_initial_state(md, 1, 1, starting_funds=0, tier_name="T2")
    st.units = {0: [], 1: []}
    st.funds[0] = 50_000
    st.active_player = 0
    apc = _make_unit(st, UnitType.APC, 0, (3, 2))
    return st, apc


# ---------------------------------------------------------------------------
# Refuel correctness
# ---------------------------------------------------------------------------

class TestAPCRefuel(unittest.TestCase):
    """``_apc_resupply`` must restore every adjacent ally to max fuel/ammo."""

    def setUp(self) -> None:
        self.state = _fresh_state()
        self.state.active_player = 0
        self.apc = _make_unit(self.state, UnitType.APC, 0, (3, 2))

    def test_all_four_neighbours_refuelled(self) -> None:
        inf_n = _make_unit(self.state, UnitType.INFANTRY, 0, (2, 2), fuel=5)
        inf_s = _make_unit(self.state, UnitType.INFANTRY, 0, (4, 2), fuel=0)
        inf_e = _make_unit(self.state, UnitType.INFANTRY, 0, (3, 3), fuel=1)
        inf_w = _make_unit(self.state, UnitType.INFANTRY, 0, (3, 1), fuel=99)
        # Ammo stress: an armed neighbour with 0 ammo must be rearmed.
        inf_s.ammo = 0
        inf_s_max = UNIT_STATS[UnitType.INFANTRY].max_ammo

        self.state._apc_resupply(self.apc)

        inf_max = UNIT_STATS[UnitType.INFANTRY].max_fuel
        for u in (inf_n, inf_s, inf_e, inf_w):
            self.assertEqual(u.fuel, inf_max, f"fuel not restored for {u.pos}")
        self.assertEqual(inf_s.ammo, inf_s_max)

    def test_enemy_neighbour_is_not_resupplied(self) -> None:
        enemy = _make_unit(self.state, UnitType.INFANTRY, 1, (2, 2), fuel=3)
        self.state._apc_resupply(self.apc)
        self.assertEqual(enemy.fuel, 3)

    def test_wait_refuels_like_resupply(self) -> None:
        # End-to-end through ``step`` rather than the private helper.
        inf = _make_unit(self.state, UnitType.INFANTRY, 0, (4, 2), fuel=0)
        _select_and_move(self.state, self.apc, self.apc.pos)
        self.state.step(Action(
            ActionType.WAIT, unit_pos=self.apc.pos, move_pos=self.apc.pos,
        ))
        self.assertEqual(inf.fuel, UNIT_STATS[UnitType.INFANTRY].max_fuel)

    def test_air_and_naval_allies_are_eligible(self) -> None:
        # AWBW: APCs supply *all* adjacent allied classes, not just ground.
        copter = _make_unit(self.state, UnitType.B_COPTER, 0, (2, 2), fuel=10)
        self.state._apc_resupply(self.apc)
        self.assertEqual(copter.fuel, UNIT_STATS[UnitType.B_COPTER].max_fuel)


# ---------------------------------------------------------------------------
# Action-space pruning
# ---------------------------------------------------------------------------

class TestAPCWaitPruning(unittest.TestCase):
    """Empty APC WAITs that refuel nobody are pruned when a useful tile exists."""

    def setUp(self) -> None:
        self.state = _fresh_state()
        self.state.active_player = 0
        self.apc = _make_unit(self.state, UnitType.APC, 0, (3, 2))

    def test_wait_kept_when_neighbour_needs_fuel(self) -> None:
        _make_unit(self.state, UnitType.INFANTRY, 0, (2, 2), fuel=5)
        _select_and_move(self.state, self.apc, self.apc.pos)
        types = {a.action_type for a in get_legal_actions(self.state)}
        self.assertIn(ActionType.WAIT, types)

    def test_wait_kept_when_better_supply_tile_exists_but_wait_is_only_action(
        self,
    ) -> None:
        # Same geometry as the old "prune WAIT" case: (3, 2) would resupply the
        # needy inf at (2, 2), but we already committed the MOVE to (3, 3).
        # WAIT is the only ACTION terminator here — dropping it yields *no*
        # legal actions, so we must keep WAIT (see engine/action.py APC prune).
        _make_unit(self.state, UnitType.INFANTRY, 0, (2, 2), fuel=5)
        _select_and_move(self.state, self.apc, (3, 3))
        types = {a.action_type for a in get_legal_actions(self.state)}
        self.assertIn(
            ActionType.WAIT,
            types,
            "If WAIT is the only legal terminator, it must not be pruned.",
        )

    def test_wait_pruned_when_other_tile_supplies_ally_and_build_exists(
        self,
    ) -> None:
        # APC cannot attack (no ammo); use BUILD on an owned base as a second
        # legal ACTION so dominated WAIT can be dropped without emptying the list.
        state, apc = _state_apc_on_owned_base_with_needy_inf()
        _make_unit(state, UnitType.INFANTRY, 0, (2, 2), fuel=5)
        _select_and_move(state, apc, (3, 3))
        acts = get_legal_actions(state)
        types = {a.action_type for a in acts}
        self.assertTrue(
            any(a.action_type == ActionType.BUILD for a in acts),
            "Expected at least one BUILD on the owned base.",
        )
        self.assertNotIn(
            ActionType.WAIT,
            types,
            "With BUILD available, useless-resupply WAIT may be pruned.",
        )

    def test_wait_kept_when_no_tile_resupplies_anyone(self) -> None:
        # No allied neighbours anywhere — every WAIT is equally useless.
        # Rather than deadlock the APC, keep WAIT so it can end its turn.
        _select_and_move(self.state, self.apc, (3, 3))
        types = {a.action_type for a in get_legal_actions(self.state)}
        self.assertIn(ActionType.WAIT, types)

    def test_wait_kept_when_apc_carries_cargo(self) -> None:
        cargo = _make_unit(self.state, UnitType.INFANTRY, 0, (4, 4))
        self.state.units[0].remove(cargo)
        self.apc.loaded_units.append(cargo)
        # Make the useless tile have an alternative (same setup as pruned case).
        _make_unit(self.state, UnitType.INFANTRY, 0, (2, 2), fuel=5)
        _select_and_move(self.state, self.apc, (3, 3))
        types = {a.action_type for a in get_legal_actions(self.state)}
        # With cargo aboard UNLOAD would also appear; we care that WAIT stays.
        self.assertIn(ActionType.WAIT, types)

    def test_wait_kept_when_current_tile_benefits_ally(self) -> None:
        # Neighbour on (2, 2) needs fuel; APC already ends on (3, 2) adjacent.
        _make_unit(self.state, UnitType.INFANTRY, 0, (2, 2), fuel=5)
        # Also add an alternative useful tile so the "better tile exists" arm
        # is non-empty — pruning must still not fire because THIS tile is OK.
        _make_unit(self.state, UnitType.INFANTRY, 0, (4, 4), fuel=0)
        _select_and_move(self.state, self.apc, self.apc.pos)
        types = {a.action_type for a in get_legal_actions(self.state)}
        self.assertIn(ActionType.WAIT, types)


if __name__ == "__main__":
    unittest.main()
