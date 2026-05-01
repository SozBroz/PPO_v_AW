"""Phase 10c: checkpoint opponent pool refresh without process restart."""
from __future__ import annotations

import json
import os
from pathlib import Path

from rl.env import AWBWEnv
from rl.self_play import _CheckpointOpponent

_ROOT = Path(__file__).resolve().parents[1]


def _one_std_map() -> list[dict]:
    with open(_ROOT / "data" / "gl_map_pool.json", encoding="utf-8") as f:
        pool = json.load(f)
    return [next(m for m in pool if m.get("type") == "std")]


def _touch(p: Path) -> None:
    p.write_bytes(b"")


def test_checkpoint_opponent_reload_pool_returns_count(tmp_path: Path) -> None:
    for i in range(3):
        _touch(tmp_path / f"checkpoint_{i}.zip")
    opp = _CheckpointOpponent(str(tmp_path))
    n = opp.reload_pool()
    assert n == 3
    assert opp._pool_candidates is not None
    assert len(opp._pool_candidates) == 3


def test_reload_pool_with_explicit_paths() -> None:
    opp = _CheckpointOpponent("/tmp/unused", refresh_every=1)
    n = opp.reload_pool(zip_paths=["/fake/a.zip", "/fake/b.zip"])
    assert n == 2
    assert opp._pool_candidates == ["/fake/a.zip", "/fake/b.zip"]


def test_reload_pool_picks_from_refreshed_set(tmp_path: Path) -> None:
    for i in range(3):
        _touch(tmp_path / f"checkpoint_{i}.zip")
    fourth = tmp_path / "checkpoint_99.zip"
    opp = _CheckpointOpponent(str(tmp_path), refresh_every=10**9)
    assert opp.reload_pool() == 3
    _touch(fourth)
    assert opp._pool_candidates is not None
    assert str(fourth) not in opp._pool_candidates
    assert opp.reload_pool() == 4
    assert str(fourth) in (opp._pool_candidates or [])


def test_reload_pool_tails_to_opponent_newest_k(tmp_path: Path) -> None:
    base = 10_000
    for i in range(30):
        p = tmp_path / f"checkpoint_{i:03d}.zip"
        p.write_bytes(b"x")
        os.utime(p, (base + i, base + i))
    opp = _CheckpointOpponent(str(tmp_path), opponent_pool_newest_k=24)
    assert opp.reload_pool() == 24
    cand = opp._pool_candidates or []
    assert len(cand) == 24
    names = sorted(Path(x).name for x in cand)
    assert names[0] == "checkpoint_006.zip"
    assert names[-1] == "checkpoint_029.zip"


def test_reload_pool_newest_k_zero_keeps_all(tmp_path: Path) -> None:
    for i in range(7):
        _touch(tmp_path / f"checkpoint_{i}.zip")
    opp = _CheckpointOpponent(str(tmp_path), opponent_pool_newest_k=0)
    assert opp.reload_pool() == 7


def test_awbw_env_reload_opponent_pool_delegates() -> None:
    class O7:
        def reload_pool(self) -> int:
            return 7

    env = AWBWEnv(map_pool=_one_std_map(), opponent_policy=O7())
    assert env.reload_opponent_pool() == 7

    env2 = AWBWEnv(
        map_pool=_one_std_map(),
        opponent_policy=lambda o, m: 0,  # type: ignore[arg-type,assignment]
    )
    assert env2.reload_opponent_pool() is None
