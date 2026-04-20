"""Opening-player rule: the side that starts with predeployed units moves second.

Covers:
  * All `type == "std"` maps in `data/gl_map_pool.json` (14 maps). Each one is
    loaded through the real pipeline and its computed opening player is
    checked against the asymmetric-predeploy rule in ``make_initial_state``.
  * An asymmetric fixture (only P0 starts with a unit) to confirm P1 opens —
    the case absent from the current std pool.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from engine.game import make_initial_state
from engine.map_loader import load_map
from engine.predeployed import PredeployedUnitSpec
from engine.unit import UnitType

from tools.oracle_zip_replay import (
    replay_first_mover_engine,
    replay_first_mover_from_snapshot_turn,
    resolve_replay_first_mover,
)

ROOT     = Path(__file__).parent
POOL     = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"


def _expected_opener(n0: int, n1: int) -> int:
    """Opener under the AWBW rule: empty side moves first; tie -> P0."""
    if n0 == 0 and n1 > 0:
        return 0
    if n1 == 0 and n0 > 0:
        return 1
    return 0


def _std_pool_map_ids() -> list[int]:
    pool = json.loads(POOL.read_text(encoding="utf-8"))
    return [m["map_id"] for m in pool if m.get("type") == "std"]


class TestStdPoolOpeningPlayer(unittest.TestCase):
    """Every std map: active_player after make_initial_state matches the rule."""

    def test_std_maps_opener_matches_rule(self) -> None:
        std_ids = _std_pool_map_ids()
        self.assertGreater(len(std_ids), 0, "no std maps in pool")

        for map_id in std_ids:
            with self.subTest(map_id=map_id):
                md = load_map(map_id, POOL, MAPS_DIR)
                n0 = sum(1 for s in md.predeployed_specs if s.player == 0)
                n1 = sum(1 for s in md.predeployed_specs if s.player == 1)
                expected = _expected_opener(n0, n1)

                st = make_initial_state(md, 1, 7, starting_funds=0, tier_name="T2")
                self.assertEqual(
                    st.active_player, expected,
                    f"map {map_id}: P0 units={n0} P1 units={n1} "
                    f"-> expected opener={expected}, got {st.active_player}",
                )


class TestReplayFirstMoverResolution(unittest.TestCase):
    """``resolve_replay_first_mover`` — envelope stream vs PHP snapshot ``turn``."""

    def test_engine_skips_empty_action_envelopes(self) -> None:
        m = {100: 0, 200: 1}
        envs = [
            (100, 1, []),
            (200, 1, [{"action": "End"}]),
        ]
        self.assertEqual(replay_first_mover_engine(envs, m), 1)

    def test_snapshot_turn_fallback(self) -> None:
        m = {55: 1, 66: 0}
        snap = {"turn": 66}
        self.assertEqual(replay_first_mover_from_snapshot_turn(snap, m), 0)
        self.assertIsNone(replay_first_mover_from_snapshot_turn({}, m))
        self.assertIsNone(replay_first_mover_from_snapshot_turn({"turn": 999}, m))

    def test_resolve_prefers_nonempty_envelope_over_snapshot(self) -> None:
        m = {10: 1, 20: 0}
        envs = [(10, 1, [{"action": "End"}])]
        snap = {"turn": 20}
        self.assertEqual(resolve_replay_first_mover(envs, snap, m), 1)

    def test_resolve_falls_back_to_snapshot_when_all_actions_empty(self) -> None:
        m = {10: 1, 20: 0}
        envs = [(10, 1, []), (20, 1, [])]
        snap = {"turn": 20}
        self.assertEqual(resolve_replay_first_mover(envs, snap, m), 0)


class TestReplayFirstMoverOverride(unittest.TestCase):
    """Oracle / site zips can force opener when it disagrees with predeploy heuristic."""

    def test_replay_first_mover_overrides_asymmetric_rule(self) -> None:
        md = load_map(171596, POOL, MAPS_DIR)
        md.predeployed_specs = [
            PredeployedUnitSpec(row=11, col=16, player=0, unit_type=UnitType.INFANTRY),
        ]
        st = make_initial_state(
            md, 1, 7, starting_funds=0, tier_name="T2", replay_first_mover=0,
        )
        self.assertEqual(st.active_player, 0)


class TestAsymmetricOnlyP0Units(unittest.TestCase):
    """When only P0 starts with units, P1 must open and receive day-1 income."""

    def test_only_p0_units_makes_p1_open(self) -> None:
        md = load_map(171596, POOL, MAPS_DIR)
        # 171596 ships with a single P1 infantry; replace with a single P0 unit
        # so the rule has to flip the opener.
        md.predeployed_specs = [
            PredeployedUnitSpec(row=11, col=16, player=0, unit_type=UnitType.INFANTRY),
        ]

        st = make_initial_state(md, 1, 7, starting_funds=0, tier_name="T2")

        self.assertEqual(len(st.units[0]), 1)
        self.assertEqual(len(st.units[1]), 0)
        self.assertEqual(st.active_player, 1, "P1 should open when only P0 has units")

        # Day-1 income went to the opener (P1), not to P0.
        p1_income_props = st.count_income_properties(1)
        self.assertEqual(st.funds[0], 0)
        self.assertEqual(st.funds[1], p1_income_props * 1000)


if __name__ == "__main__":
    unittest.main()
