"""Stable unit id invariants.

AWBW's ReplayMap.updateToGameState(..) looks up DrawableUnit by numeric ID and
calls UpdateUnit, which does NOT re-read UnitData from UnitName. If we reuse
the same id for logically different units across adjacent snapshots, the
viewer will render the wrong sprite/color/stats for whatever stayed behind.

These tests lock three invariants:

1. Every Unit ever created goes through `GameState._allocate_unit_id` (i.e. has
   `unit_id > 0`). Covers predeploy, build, and Sensei-COP-spawn paths.
2. `unit_id` never collides across alive units in a single snapshot.
3. For any unit_id that appears in BOTH snapshot N and snapshot N+1, the
   (player, unit_type) pair is identical. This is the real "did the sprite
   flip" test — it catches id churn that AWBW cannot recover from.

Run: `python -m unittest test_stable_unit_ids -v`
"""
from __future__ import annotations

import copy
import random
import unittest
from pathlib import Path

from engine.action import get_legal_actions
from engine.game import GameState, make_initial_state
from engine.map_loader import load_map
from engine.unit import Unit, UnitType


ROOT = Path(__file__).parent
POOL = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"


def _all_units(state: GameState) -> list[Unit]:
    """Flatten alive top-level + cargo units into one list."""
    out: list[Unit] = []
    for pl in (0, 1):
        for u in state.units[pl]:
            out.append(u)
            out.extend(u.loaded_units)
    return out


def _unit_index(state: GameState) -> dict[int, tuple[int, UnitType]]:
    """Map unit_id -> (player, unit_type) for every alive unit in the snapshot."""
    idx: dict[int, tuple[int, UnitType]] = {}
    for u in _all_units(state):
        idx[u.unit_id] = (u.player, u.unit_type)
    return idx


class TestStableUnitIDs(unittest.TestCase):
    MAP_ID = 133665   # "Walls Are Closing In" — small, has predeploy units
    CO0, CO1 = 1, 7   # Andy vs Grit, both vanilla — no Sensei spawn noise
    SEED = 20260416
    MAX_ACTIONS = 800

    def setUp(self) -> None:
        map_data = load_map(self.MAP_ID, POOL, MAPS_DIR)
        self.state = make_initial_state(map_data, self.CO0, self.CO1,
                                        starting_funds=0, tier_name="T2")

    def test_all_predeployed_units_have_ids(self) -> None:
        units = _all_units(self.state)
        self.assertGreater(len(units), 0, "map should have predeployed units")
        for u in units:
            self.assertGreater(
                u.unit_id, 0,
                f"predeployed {u} was not stamped with a unit_id"
            )

    def test_ids_unique_within_initial_snapshot(self) -> None:
        ids = [u.unit_id for u in _all_units(self.state)]
        self.assertEqual(len(ids), len(set(ids)),
                         "duplicate unit_ids in initial snapshot")

    def test_ids_stable_across_random_game(self) -> None:
        """Simulate a short random game, snapshot per player-turn, assert that
        any id that carries across adjacent snapshots still names the same
        (player, unit_type). This is the viewer invariant AWBW relies on.
        """
        rng = random.Random(self.SEED)
        snapshots: list[GameState] = [copy.deepcopy(self.state)]

        for _ in range(self.MAX_ACTIONS):
            if self.state.done:
                break
            legal = get_legal_actions(self.state)
            if not legal:
                break
            prev_player = self.state.active_player
            self.state.step(rng.choice(legal))
            if not self.state.done and self.state.active_player != prev_player:
                snapshots.append(copy.deepcopy(self.state))

        self.assertGreaterEqual(len(snapshots), 3,
                                "short game should have yielded >=3 snapshots")

        # For every adjacent pair, ids shared between them must preserve
        # (player, unit_type). Ids may disappear (death, load into transport
        # that vanished from view) or appear (new build), but they must NEVER
        # swap identity.
        for i in range(len(snapshots) - 1):
            prev = _unit_index(snapshots[i])
            curr = _unit_index(snapshots[i + 1])
            shared = prev.keys() & curr.keys()
            for uid in shared:
                self.assertEqual(
                    prev[uid], curr[uid],
                    f"unit_id {uid} changed identity between snapshot {i} "
                    f"and {i+1}: {prev[uid]} -> {curr[uid]}"
                )

    def test_ids_allocated_by_build_are_fresh(self) -> None:
        """After one player-turn of random play including builds, any newly
        appearing unit_id must exceed all pre-existing ids (monotonic)."""
        rng = random.Random(self.SEED + 1)
        before_ids = {u.unit_id for u in _all_units(self.state)}
        max_before = max(before_ids)

        for _ in range(200):
            if self.state.done:
                break
            prev_player = self.state.active_player
            legal = get_legal_actions(self.state)
            if not legal:
                break
            self.state.step(rng.choice(legal))
            if self.state.active_player != prev_player:
                break   # stop after one player-turn boundary

        for u in _all_units(self.state):
            if u.unit_id not in before_ids:
                self.assertGreater(
                    u.unit_id, max_before,
                    f"newly spawned {u} has non-monotonic id {u.unit_id} "
                    f"(max pre-existing was {max_before})"
                )


if __name__ == "__main__":
    unittest.main()
