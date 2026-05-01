"""Regression: replay 179877 and assert transport cargo never exceeds capacity.

Background: game 179877 turn 11 was reviewed for a possible Lander holding
three units. The trace shows only legal loads; this test pins the invariant
``len(loaded_units) <= carry_capacity`` for every transport after each
full_trace step so a future engine regression cannot silently overflow cargo.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from engine.game import make_initial_state
from engine.map_loader import load_map
from engine.unit import UNIT_STATS

from tools.export_awbw_replay_actions import _trace_to_action

REPO_ROOT = Path(__file__).resolve().parent
TRACE_179877 = REPO_ROOT / "replays" / "179877.trace.json"
MAP_POOL = REPO_ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = REPO_ROOT / "data" / "maps"


def _assert_all_transports_within_capacity(state) -> None:
    for player in (0, 1):
        for u in state.units.get(player, []):
            cap = UNIT_STATS[u.unit_type].carry_capacity
            n = len(u.loaded_units)
            if n > cap:
                raise AssertionError(
                    f"Transport {u.unit_type.name} at {u.pos} (P{player}) "
                    f"has {n} loaded units but carry_capacity={cap}."
                )


@unittest.skipUnless(TRACE_179877.exists(), "179877 trace fixture not present")
class TestTrace179877TransportCapacity(unittest.TestCase):
    """Full replay of replays/179877.trace.json with per-step cargo checks."""

    @classmethod
    def setUpClass(cls) -> None:
        with open(TRACE_179877, encoding="utf-8") as f:
            cls.record = json.load(f)
        cls.map_data = load_map(
            cls.record["map_id"], MAP_POOL, MAPS_DIR,
        )
        cls.full_trace = cls.record["full_trace"]

    def test_replay_never_exceeds_transport_carry_capacity(self) -> None:
        state = make_initial_state(
            self.map_data,
            self.record["co0"],
            self.record["co1"],
            starting_funds=0,
            tier_name=self.record.get("tier", "T2"),
        )
        _assert_all_transports_within_capacity(state)

        for i, entry in enumerate(self.full_trace):
            with self.subTest(step=i, type=entry.get("type")):
                action = _trace_to_action(entry)
                # AWBW zip/trace envelopes can include factory BUILDs PHP accepted under a
                # roster-only unit-cap view while cargo sits aboard transports; this test
                # pins transport carry overflow only, not global BUILD legality.
                state.step(action, oracle_mode=True)
                _assert_all_transports_within_capacity(state)


if __name__ == "__main__":
    unittest.main()
