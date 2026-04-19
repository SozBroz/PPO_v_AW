"""Round-trip: trace -> AWBW zip -> oracle_zip_replay matches trace replay."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from engine.action import Action, ActionType
from engine.game import make_initial_state
from engine.map_loader import load_map

from tools.export_awbw_replay import write_awbw_replay_from_trace
from tools.export_awbw_replay_actions import _trace_to_action
from tools.oracle_zip_replay import replay_oracle_zip

ROOT = Path(__file__).resolve().parent
MAP_POOL = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"
TRACE = ROOT / "replays" / "272176.trace.json"


def _replay_trace(record: dict):
    md = load_map(record["map_id"], MAP_POOL, MAPS_DIR)
    st = make_initial_state(
        md, record["co0"], record["co1"], starting_funds=0, tier_name=record.get("tier", "T2")
    )
    for e in record["full_trace"]:
        st.step(_trace_to_action(e))
    return st


class TestOracleZipReplayRoundTrip(unittest.TestCase):
    @unittest.skipUnless(TRACE.exists(), "fixture trace missing")
    def test_zip_from_trace_replays_like_trace(self) -> None:
        with open(TRACE, encoding="utf-8") as f:
            record = json.load(f)
        ref = _replay_trace(record)
        with tempfile.TemporaryDirectory() as td:
            zpath = Path(td) / "272176.zip"
            write_awbw_replay_from_trace(record, zpath, MAP_POOL, MAPS_DIR, game_id=272176)
            got = replay_oracle_zip(
                zpath,
                map_pool=MAP_POOL,
                maps_dir=MAPS_DIR,
                map_id=record["map_id"],
                co0=record["co0"],
                co1=record["co1"],
                tier_name=record.get("tier", "T2"),
            ).final_state
        self.assertEqual(ref.done, got.done)
        self.assertEqual(ref.turn, got.turn)
        self.assertEqual(ref.winner, got.winner)
        self.assertEqual(ref.active_player, got.active_player)


if __name__ == "__main__":
    unittest.main()
