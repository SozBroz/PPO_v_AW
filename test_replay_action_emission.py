"""Same-tile WAIT/ATTACK/CAPTURE/LOAD must still emit action JSON.

Background
----------
The AWBW Replay Player desktop viewer animates each player-turn from the
``a<game_id>`` action stream produced by
``tools/export_awbw_replay_actions.py``. Empty per-turn action arrays are
still emitted as ``a:0:{}`` so every player-turn has a ``p:`` envelope; any
code path that returns ``None`` from ``_emit_move_or_fire`` still shrinks
the *action list* for that turn. Same-tile WAIT/ATTACK must return JSON so
indirect-heavy days are not silent no-ops.

These tests pin the contract: same-tile WAIT, CAPTURE, indirect ATTACK, and
moving variants all return a non-``None`` payload, and the full 166901 trace
re-replays without dropping any WAIT/ATTACK/CAPTURE/LOAD step.
"""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from engine.action import Action, ActionType, ActionStage
from engine.game import GameState, make_initial_state
from engine.map_loader import MapData, load_map
from engine.unit import Unit, UnitType, UNIT_STATS

from tools.export_awbw_replay_actions import (
    P0_PLAYER_ID, P1_PLAYER_ID,
    _emit_move_or_fire, _trace_to_action,
    build_action_stream_text,
    _rebuild_and_emit_with_snapshots,
)

from test_lander_and_fuel import _fresh_state, _make_unit


REPO_ROOT = Path(__file__).resolve().parent
TRACE_166901 = REPO_ROOT / "replays" / "166901.trace.json"
TRACE_167911 = REPO_ROOT / "replays" / "167911.trace.json"
TRACE_182065 = REPO_ROOT / "replays" / "182065.trace.json"
MAP_POOL = REPO_ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = REPO_ROOT / "data" / "maps"


# ---------------------------------------------------------------------------
# Same-tile emission contract
# ---------------------------------------------------------------------------

class TestSameTileEmission(unittest.TestCase):
    """`_emit_move_or_fire` must return JSON for `start == end` actions."""

    def setUp(self) -> None:
        self.state = _fresh_state()
        self.state.active_player = 0
        self.pid_of = {0: P0_PLAYER_ID, 1: P1_PLAYER_ID}

    def _select(self, unit: Unit, dest: tuple[int, int]) -> None:
        self.state.action_stage      = ActionStage.SELECT
        self.state.selected_unit     = None
        self.state.selected_move_pos = None
        self.state.step(Action(ActionType.SELECT_UNIT, unit_pos=unit.pos))
        self.state.step(Action(
            ActionType.SELECT_UNIT, unit_pos=unit.pos, move_pos=dest,
        ))

    def test_same_tile_wait_emits_move(self) -> None:
        artillery = _make_unit(self.state, UnitType.ARTILLERY, 0, (3, 2))
        self._select(artillery, (3, 2))
        payload = _emit_move_or_fire(
            self.state,
            Action(ActionType.WAIT, unit_pos=(3, 2), move_pos=(3, 2)),
            self.pid_of, P0_PLAYER_ID, P1_PLAYER_ID,
        )
        self.assertIsNotNone(payload)
        self.assertEqual(payload["action"], "Move")
        self.assertEqual(len(payload["paths"]["global"]), 1)
        self.assertEqual(payload["dist"], 0)

    def test_moving_wait_still_emits_move(self) -> None:
        inf = _make_unit(self.state, UnitType.INFANTRY, 0, (3, 2))
        self._select(inf, (3, 3))
        payload = _emit_move_or_fire(
            self.state,
            Action(ActionType.WAIT, unit_pos=(3, 2), move_pos=(3, 3)),
            self.pid_of, P0_PLAYER_ID, P1_PLAYER_ID,
        )
        self.assertIsNotNone(payload)
        self.assertEqual(payload["action"], "Move")
        self.assertGreaterEqual(len(payload["paths"]["global"]), 2)

    def test_same_tile_indirect_attack_emits_fire(self) -> None:
        # Artillery (range 2-3) attacking from its own tile is the canonical
        # same-tile ATTACK that the old early-return silently dropped.
        artillery = _make_unit(self.state, UnitType.ARTILLERY, 0, (3, 2))
        target = _make_unit(self.state, UnitType.INFANTRY, 1, (3, 4))
        self._select(artillery, (3, 2))
        payload = _emit_move_or_fire(
            self.state,
            Action(
                ActionType.ATTACK,
                unit_pos=(3, 2), move_pos=(3, 2), target_pos=(3, 4),
            ),
            self.pid_of, P0_PLAYER_ID, P1_PLAYER_ID,
        )
        self.assertIsNotNone(payload)
        self.assertEqual(payload["action"], "Fire")
        self.assertIn("Move", payload)
        self.assertEqual(payload["Move"]["action"], "Move")
        # combatInfo is wrapped per-player; the global view always has both
        # attacker and defender post-combat blocks.
        ci = payload["Fire"]["combatInfoVision"]["global"]["combatInfo"]
        self.assertIn("attacker", ci)
        self.assertIn("defender", ci)


# ---------------------------------------------------------------------------
# Full-trace regression (replays/166901)
# ---------------------------------------------------------------------------

@unittest.skipUnless(TRACE_166901.exists(), "166901 trace fixture not present")
class TestTrace166901NoDrops(unittest.TestCase):
    """Re-replay 166901 and verify every WAIT/ATTACK/CAPTURE/LOAD emits JSON.

    Uses the public ``build_action_stream_text`` so the test catches both the
    per-action drop bug and the empty-envelope drop downstream.
    """

    @classmethod
    def setUpClass(cls) -> None:
        with open(TRACE_166901, encoding="utf-8") as f:
            cls.record = json.load(f)
        cls.map_data = load_map(cls.record["map_id"], MAP_POOL, MAPS_DIR)
        cls.full_trace = cls.record["full_trace"]

    def test_no_movefire_action_is_silently_dropped(self) -> None:
        from tools.export_awbw_replay_actions import _rebuild_and_emit

        buckets = _rebuild_and_emit(
            self.full_trace,
            self.map_data,
            self.record["co0"],
            self.record["co1"],
            self.record.get("tier", "T2"),
        )

        # Count expected vs emitted move/fire actions across the trace.
        expected_movefire = sum(
            1 for e in self.full_trace
            if e["type"] in ("WAIT", "ATTACK", "CAPTURE", "LOAD")
        )
        emitted_movefire = sum(
            1 for b in buckets for a in b.actions
            if a["action"] in ("Move", "Fire")
        )
        self.assertEqual(
            emitted_movefire, expected_movefire,
            f"Dropped {expected_movefire - emitted_movefire} WAIT/ATTACK/CAPTURE/LOAD "
            f"emissions for trace 166901 (expected {expected_movefire}, "
            f"got {emitted_movefire}).",
        )

    def test_every_player_turn_envelope_is_non_empty(self) -> None:
        # The viewer matches envelopes by (ActivePlayerID, Day) — an empty
        # envelope is dropped by build_action_stream_text and the next
        # turn appears "skipped" with desynced numbering.
        text = build_action_stream_text(
            self.full_trace,
            self.map_data,
            self.record["co0"],
            self.record["co1"],
            self.record.get("tier", "T2"),
        )

        envelope_count = text.count("\np:") + (1 if text.startswith("p:") else 0)
        end_turns = sum(1 for e in self.full_trace if e["type"] == "END_TURN")
        # Each END_TURN closes one envelope; the trace should yield one
        # envelope per player-turn (some final turns may lack END_TURN if the
        # game ended mid-turn, hence >= rather than ==).
        self.assertGreaterEqual(
            envelope_count, end_turns,
            f"Expected at least {end_turns} player-turn envelopes "
            f"(one per END_TURN), got {envelope_count}.",
        )


# ---------------------------------------------------------------------------
# Full-trace regression (replays/167911)
# ---------------------------------------------------------------------------
# 167911 was reported as "skipped turn blue (P1) turn 7, maybe related to cap."
# P1 day 7 includes an in-place CAPTURE at (3, 4) — the exact same_tile case
# the earlier fix addressed for 166901. These tests pin that behavior for the
# new trace so the bug cannot quietly re-appear for captures specifically.

@unittest.skipUnless(TRACE_167911.exists(), "167911 trace fixture not present")
class TestTrace167911NoDrops(unittest.TestCase):
    """Re-replay 167911 and verify every WAIT/ATTACK/CAPTURE/LOAD emits JSON."""

    @classmethod
    def setUpClass(cls) -> None:
        with open(TRACE_167911, encoding="utf-8") as f:
            cls.record = json.load(f)
        cls.map_data = load_map(cls.record["map_id"], MAP_POOL, MAPS_DIR)
        cls.full_trace = cls.record["full_trace"]

    def test_no_movefire_action_is_silently_dropped(self) -> None:
        from tools.export_awbw_replay_actions import _rebuild_and_emit

        buckets = _rebuild_and_emit(
            self.full_trace,
            self.map_data,
            self.record["co0"],
            self.record["co1"],
            self.record.get("tier", "T2"),
        )

        expected_movefire = sum(
            1 for e in self.full_trace
            if e["type"] in ("WAIT", "ATTACK", "CAPTURE", "LOAD")
        )
        emitted_movefire = sum(
            1 for b in buckets for a in b.actions
            if a["action"] in ("Move", "Fire")
        )
        self.assertEqual(
            emitted_movefire, expected_movefire,
            f"Dropped {expected_movefire - emitted_movefire} WAIT/ATTACK/CAPTURE/LOAD "
            f"emissions for trace 167911 (expected {expected_movefire}, "
            f"got {emitted_movefire}).",
        )

    def test_every_player_turn_envelope_is_non_empty(self) -> None:
        text = build_action_stream_text(
            self.full_trace,
            self.map_data,
            self.record["co0"],
            self.record["co1"],
            self.record.get("tier", "T2"),
        )

        envelope_count = text.count("\np:") + (1 if text.startswith("p:") else 0)
        end_turns = sum(1 for e in self.full_trace if e["type"] == "END_TURN")
        self.assertGreaterEqual(
            envelope_count, end_turns,
            f"Expected at least {end_turns} player-turn envelopes "
            f"(one per END_TURN), got {envelope_count}.",
        )

    def test_p1_day7_in_place_capture_emits_move(self) -> None:
        # Specific pin for the reported bug: the (3, 4) CAPTURE on P1 day 7
        # must land in the day-7 envelope as a Move with a single-tile path.
        from tools.export_awbw_replay_actions import _rebuild_and_emit

        buckets = _rebuild_and_emit(
            self.full_trace,
            self.map_data,
            self.record["co0"],
            self.record["co1"],
            self.record.get("tier", "T2"),
        )
        p1_day7 = next(
            (b for b in buckets if b.day == 7 and b.player_id == P1_PLAYER_ID),
            None,
        )
        self.assertIsNotNone(p1_day7, "No P1/day-7 bucket produced for 167911.")

        def _is_in_place_move(a: dict, tile: tuple[int, int]) -> bool:
            if a.get("action") != "Move":
                return False
            paths = a.get("paths", {}).get("global", [])
            if not paths:
                return False
            first, last = paths[0], paths[-1]
            return (first["y"], first["x"]) == tile and (last["y"], last["x"]) == tile

        self.assertTrue(
            any(_is_in_place_move(a, (3, 4)) for a in p1_day7.actions),
            "P1/day-7 envelope is missing the in-place Move for the (3,4) CAPTURE.",
        )


class TestEndTurnJsonDay(unittest.TestCase):
    """``End`` action ``updatedInfo.day`` must be post-END_TURN ``state.turn`` (viewer NextDay)."""

    @classmethod
    def setUpClass(cls) -> None:
        with open(TRACE_182065, encoding="utf-8") as f:
            cls.record = json.load(f)
        cls.map_data = load_map(
            cls.record["map_id"], MAP_POOL, MAPS_DIR,
        )

    def test_calendar_day_sequence_matches_engine(self) -> None:
        _, buckets = _rebuild_and_emit_with_snapshots(
            self.record["full_trace"],
            self.map_data,
            self.record["co0"],
            self.record["co1"],
            tier_name=self.record.get("tier", "T2"),
        )
        ends: list[int] = []
        for b in buckets:
            for a in b.actions:
                if a.get("action") == "End":
                    ends.append(a["updatedInfo"]["day"])
        self.assertGreaterEqual(len(ends), 4)
        # P0 D1→P1 (1); P1 D1→P0 (2); P0 D2→P1 (2); P1 D2→P0 (3)
        self.assertEqual(ends[:4], [1, 2, 2, 3], f"got ends[:8]={ends[:8]!r}")


if __name__ == "__main__":
    unittest.main()
