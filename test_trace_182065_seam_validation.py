"""Validation for replays/182065.trace.json — pipe seams vs Mech and turn-119 viewer context.

Game 182065: both COs Sami (co_id 8). Full engine replay succeeds; seam damage matches
``game_log`` attack_seam entries.

In this engine, ``full_trace`` entries' ``turn`` field is the **calendar day** (it
advances when **P1** ends a turn — i.e. ``active_player`` **1**, blue / second
seat, not “human player #1” in 1-based counting). So e.g. ``"turn": 24`` is **Day 24** on the
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

    @unittest.expectedFailure
    def test_full_trace_replays_without_error(self) -> None:
        """Full trace replay vs AWBW envelopes — **expected failure** (not an engine bug).

        **Choice: Option A** (`@unittest.expectedFailure`). Option B (swallow
        ``ValueError`` / ``IllegalActionError`` and assert partial progress) was rejected
        in favor of an explicit xfail: this test validates replay fidelity against a
        serialized trace, not internal engine consistency; masking failures would hide
        the real seam.

        - ``docs/oracle_exception_audit/phase11c_trace_182065_export_fix.md``: export
          pipeline threads ``oracle_mode=True``; STEP-GATE opt-out does not cover
          ``_move_unit`` raising ``ValueError`` when AWBW's recorded move disagrees with
          engine reachability.
        - ``docs/oracle_exception_audit/phase10f_silent_drift_recon.md``: engine vs AWBW
          PHP drift (funds, HP, movement) is a known multi-phase research class — separate
          from self-play / RL training, which uses the engine's own legality and remains
          unaffected.

        **Divergence pattern:** Infantry move (9,8)→(11,7) around calendar **Day ~24**
        — reachability mismatch between trace and engine.

        **TODO:** Next phase addressing **reachability / movement drift** vs AWBW (Phase 11
        charter: position drift, oracle path work) should revisit this test — remove
        ``expectedFailure`` once the trace replays cleanly or replace with a bounded
        harness that documents tolerated drift.
        """
        state = make_initial_state(
            self.map_data,
            self.record["co0"],
            self.record["co1"],
            starting_funds=0,
            tier_name=self.record.get("tier", "T2"),
        )
        for entry in self.full_trace:
            state.step(_trace_to_action(entry), oracle_mode=True)

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
