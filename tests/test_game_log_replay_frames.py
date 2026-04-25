"""game_log.jsonl rows include ``frames`` when replay logging is enabled."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
POOL_PATH = ROOT / "data" / "gl_map_pool.json"
MAP_ID = 123858


def _single_map_pool() -> list[dict]:
    with open(POOL_PATH, encoding="utf-8") as f:
        pool = json.load(f)
    return [next(m for m in pool if m.get("map_id") == MAP_ID)]


def test_game_log_row_has_frames_when_awbw_log_replay_frames(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log_path = tmp_path / "game_log.jsonl"
    monkeypatch.setattr("rl.env.GAME_LOG_PATH", log_path)
    monkeypatch.delenv("AWBW_SESSION_GAME_COUNTER_DB", raising=False)
    monkeypatch.setenv("AWBW_LOG_REPLAY_FRAMES", "1")

    from rl.env import AWBWEnv

    env = AWBWEnv(
        map_pool=_single_map_pool(),
        opponent_policy=None,
        co_p0=1,
        co_p1=1,
        tier_name="T3",
    )
    assert env.log_replay_frames is True
    env.reset(seed=0)
    env.state.done = True
    env.state.winner = 0
    env.state.win_reason = "test"
    env._log_finished_game()

    raw = log_path.read_text(encoding="utf-8").strip()
    rows = [json.loads(chunk) for chunk in raw.split("\n\n") if chunk.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert isinstance(row.get("frames"), list)
    assert len(row["frames"]) >= 1
