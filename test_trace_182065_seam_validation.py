"""Validation for replays/182065.trace.json — pipe seams vs Mech and turn-119 viewer context.

Game 182065: both COs Sami (co_id 8). Full engine replay succeeds; seam damage matches
``game_log`` attack_seam entries.

In this engine, ``full_trace`` entries' ``turn`` field is the **calendar day** (it
advances when Player 1 ends a turn). So e.g. ``"turn": 24`` is **Day 24** on the
in-game clock.

Facts pinned here:
- Pipe seams start at 99 HP (``SEAM_MAX_HP`` in ``engine/game.py``).
- No seam is destroyed in a single hit from full HP in this game: each break takes
  two logged strikes. The (11,14) seam *looks* like a Mech solo-kill on **Day 24**
  if only that attack is noticed — **Day 21** Artillery already reduced the seam
  (63 damage; 36 HP left).
- Exported turn buckets from ``_rebuild_and_emit`` number **121** (indices 0–120), so
  AWBW Replay Player **turn index 119** is near endgame (not the same as calendar
  ``turn`` 1–60 in ``full_trace``). Viewer failures at setup often come from
  ``AttackPipeUnitAction.SetupAndUpdate`` → ``ReplayMissingBuildingException`` if the
  seam tile is missing from that turn's ``Buildings`` snapshot — see
  ``third_party/.../AttackPipeUnitAction.cs``.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from engine.game import make_initial_state
from engine.map_loader import load_map

from tools.export_awbw_replay_actions import _rebuild_and_emit, _trace_to_action

REPO_ROOT = Path(__file__).resolve().parent
TRACE_182065 = REPO_ROOT / "replays" / "182065.trace.json"
MAP_POOL = REPO_ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = REPO_ROOT / "data" / "maps"


@unittest.skipUnless(TRACE_182065.exists(), "182065 trace fixture not present")
class TestTrace182065SeamValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with open(TRACE_182065, encoding="utf-8") as f:
            cls.record = json.load(f)
        cls.map_data = load_map(
            cls.record["map_id"], MAP_POOL, MAPS_DIR,
        )
        cls.full_trace = cls.record["full_trace"]
        cls.game_log = cls.record["game_log"]

    def test_full_trace_replays_without_error(self) -> None:
        state = make_initial_state(
            self.map_data,
            self.record["co0"],
            self.record["co1"],
            starting_funds=0,
            tier_name=self.record.get("tier", "T2"),
        )
        for entry in self.full_trace:
            state.step(_trace_to_action(entry))

    def test_attack_seam_log_matches_expected_two_hit_breaks(self) -> None:
        seams = [x for x in self.game_log if x.get("type") == "attack_seam"]
        self.assertEqual(len(seams), 4)

        # Seam (7, 4): two Mech strikes (Sami D2D + towers → 71 dmg/hit at full HP).
        self.assertEqual(seams[0]["target"], [7, 4])
        self.assertEqual(seams[0]["dmg"], 71)
        self.assertEqual(seams[0]["seam_hp"], 99 - 71)
        self.assertFalse(seams[0]["broken"])

        self.assertEqual(seams[1]["target"], [7, 4])
        self.assertEqual(seams[1]["dmg"], 71)
        self.assertTrue(seams[1]["broken"])

        # Seam (11, 14): Artillery first, then Mech — not a one-shot.
        self.assertEqual(seams[2]["target"], [11, 14])
        self.assertEqual(seams[2]["dmg"], 63)
        self.assertEqual(seams[2]["seam_hp"], 99 - 63)
        self.assertFalse(seams[2]["broken"])

        self.assertEqual(seams[3]["target"], [11, 14])
        self.assertEqual(seams[3]["dmg"], 71)
        self.assertTrue(seams[3]["broken"])

    def test_export_turn_bucket_count_explains_viewer_turn_119(self) -> None:
        buckets = _rebuild_and_emit(
            self.full_trace,
            self.map_data,
            self.record["co0"],
            self.record["co1"],
            tier_name=self.record.get("tier", "T2"),
        )
        # 121 entries → valid indices 0..120; "Failed to setup turn 119" is i=119 in ReplayController.
        self.assertEqual(len(buckets), 121)
        self.assertLess(119, len(buckets))


if __name__ == "__main__":
    unittest.main()
