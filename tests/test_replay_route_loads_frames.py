"""Replay route reads game_log rows that include ``frames`` (viewer stepping)."""

from __future__ import annotations

import json
from pathlib import Path

from server.routes.replay import _load_game_records


def test_load_game_records_preserves_frames(tmp_path: Path) -> None:
    p = tmp_path / "game_log.jsonl"
    row = {
        "game_id": 1,
        "machine_id": "t",
        "frames": [{"turn": 0, "active_player": 0, "board": {"units": []}}],
        "turns": 1,
        "winner": 0,
    }
    p.write_text(json.dumps(row) + "\n\n", encoding="utf-8")
    recs = _load_game_records(p)
    assert len(recs) == 1
    assert len(recs[0]["frames"]) == 1
