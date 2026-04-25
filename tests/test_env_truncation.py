"""Forced episode truncation (max_env_steps) and game_log rows (schema 1.9)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
POOL_PATH = ROOT / "data" / "gl_map_pool.json"
MAP_ID = 123858


def _single_map_pool() -> list[dict]:
    with open(POOL_PATH, encoding="utf-8") as f:
        pool = json.load(f)
    return [next(m for m in pool if m.get("map_id") == MAP_ID)]


def _pick_avoid_end_turn(env, rng: np.random.Generator) -> int:
    """Prefer any legal flat index except 0 (END_TURN in this repo's layout)."""
    mask = env.action_masks()
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        return 0
    non_end = legal[legal != 0]
    pool = non_end if non_end.size > 0 else legal
    return int(rng.choice(pool))


def test_max_env_steps_truncates_and_logs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log_path = tmp_path / "game_log.jsonl"
    monkeypatch.setattr("rl.env.GAME_LOG_PATH", log_path)
    monkeypatch.delenv("AWBW_SESSION_GAME_COUNTER_DB", raising=False)

    from rl.env import AWBWEnv

    env = AWBWEnv(
        map_pool=_single_map_pool(),
        opponent_policy=None,
        co_p0=1,
        co_p1=1,
        tier_name="T3",
        max_env_steps=10,
        max_p1_microsteps=4000,
    )
    env.reset(seed=1)
    rng = np.random.default_rng(0)
    saw_trunc = False
    for _ in range(25):
        _, _, term, trunc, _ = env.step(_pick_avoid_end_turn(env, rng))
        if trunc:
            saw_trunc = True
            break
        if term:
            break
    assert saw_trunc, "expected max_env_steps truncation within bounded loop"

    raw = log_path.read_text(encoding="utf-8").strip()
    assert raw, "game log should contain a row after truncated episode"
    rows = [json.loads(chunk) for chunk in raw.split("\n\n") if chunk.strip()]
    assert len(rows) >= 1
    row = rows[-1]
    assert row["truncated"] is True
    assert row["truncation_reason"] == "max_env_steps"
    assert row["terminated"] is False
    assert row["log_schema_version"] == "1.9"
