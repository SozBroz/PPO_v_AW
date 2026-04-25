"""Phase 6b: ``fps_diag.jsonl`` callback + per-worker step time stats."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
POOL_PATH = ROOT / "data" / "gl_map_pool.json"


def _pool_single_map(map_id: int = 123858) -> list[dict]:
    with open(POOL_PATH, encoding="utf-8") as f:
        pool = json.load(f)
    return [next(m for m in pool if m.get("map_id") == map_id)]


@pytest.mark.parametrize("track_on", [False, True])
def test_get_step_time_stats_respects_env_flag(monkeypatch: pytest.MonkeyPatch, track_on: bool) -> None:
    monkeypatch.setenv("AWBW_LEARNER_GREEDY_MIX", "0")
    monkeypatch.delenv("AWBW_MACHINE_ID", raising=False)
    monkeypatch.delenv("AWBW_FPS_DIAG", raising=False)
    if track_on:
        monkeypatch.setenv("AWBW_TRACK_PER_WORKER_TIMES", "1")
    else:
        monkeypatch.delenv("AWBW_TRACK_PER_WORKER_TIMES", raising=False)

    monkeypatch.setattr("rl.env._append_game_log_line", lambda _r: None)

    from rl.env import AWBWEnv

    pool = _pool_single_map()
    env = AWBWEnv(map_pool=pool, opponent_policy=None, render_mode=None)
    env.reset(seed=0)
    for _ in range(5):
        m = env.action_masks()
        env.step(int(np.random.choice(np.flatnonzero(m))))

    stats = env.get_step_time_stats()
    if not track_on:
        assert stats == {}
    else:
        assert set(stats.keys()) == {"p50", "p95", "p99", "max", "count"}
        assert stats["count"] == 5.0
        assert stats["max"] >= stats["p99"] >= stats["p50"]


def test_step_time_tracking_zero_overhead_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWBW_LEARNER_GREEDY_MIX", "0")
    monkeypatch.delenv("AWBW_TRACK_PER_WORKER_TIMES", raising=False)
    monkeypatch.delenv("AWBW_MACHINE_ID", raising=False)
    monkeypatch.delenv("AWBW_FPS_DIAG", raising=False)
    monkeypatch.setattr("rl.env._append_game_log_line", lambda _r: None)

    from rl.env import AWBWEnv

    pool = _pool_single_map()
    env = AWBWEnv(map_pool=pool, opponent_policy=None, render_mode=None)
    assert env._step_times is None
    env.reset(seed=1)
    t0 = time.perf_counter()
    for _ in range(80):
        m = env.action_masks()
        env.step(int(np.random.choice(np.flatnonzero(m))))
    dt = time.perf_counter() - t0
    assert env._step_times is None
    assert dt < 30.0, "smoke: disabled path should complete quickly on a single map"


def test_effective_track_defaults_on_for_machine_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset AWBW_TRACK_PER_WORKER_TIMES + non-empty machine_id => ring buffer on."""
    monkeypatch.delenv("AWBW_TRACK_PER_WORKER_TIMES", raising=False)
    monkeypatch.delenv("AWBW_FPS_DIAG", raising=False)
    monkeypatch.setenv("AWBW_MACHINE_ID", "x")
    from rl.env import AWBWEnv, effective_track_per_worker_times

    assert effective_track_per_worker_times() is True
    monkeypatch.setattr("rl.env._append_game_log_line", lambda _r: None)
    pool = _pool_single_map()
    env = AWBWEnv(map_pool=pool, opponent_policy=None, render_mode=None)
    assert env._step_times is not None


def test_fps_diag_callback_jsonl_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("sb3_contrib")
    from rl import self_play as sp_module
    from rl.env import AWBWEnv
    from rl.self_play import (
        _CheckpointOpponent,
        _build_diagnostics_callback,
        _mask_fn,
    )
    from sb3_contrib import MaskablePPO  # type: ignore[import]
    from sb3_contrib.common.wrappers import ActionMasker  # type: ignore[import]
    from stable_baselines3.common.vec_env import DummyVecEnv  # type: ignore[import]

    monkeypatch.setenv("AWBW_LEARNER_GREEDY_MIX", "0")
    monkeypatch.setenv("AWBW_TRACK_PER_WORKER_TIMES", "1")
    monkeypatch.setenv("AWBW_MACHINE_ID", "test-machine")
    monkeypatch.setattr("rl.env._append_game_log_line", lambda _r: None)

    diag_path = tmp_path / "fps_diag.jsonl"
    monkeypatch.setattr(sp_module, "FPS_DIAG_PATH", diag_path)

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    pool = _pool_single_map()

    def _make() -> ActionMasker:
        opp = _CheckpointOpponent(
            str(ckpt),
            cold_opponent="random",
            pool_from_fleet=False,
        )
        env = AWBWEnv(
            map_pool=pool,
            opponent_policy=opp,
            render_mode=None,
            co_p0=1,
            co_p1=1,
            tier_name="T3",
            curriculum_broad_prob=0.0,
        )
        opp.attach_env(env)
        return ActionMasker(env, _mask_fn)

    vec = DummyVecEnv([_make])
    model = MaskablePPO(
        "MultiInputPolicy",
        vec,
        n_steps=32,
        batch_size=32,
        n_epochs=1,
        verbose=0,
        device="cpu",
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.05,
        clip_range=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )
    cb = _build_diagnostics_callback()
    assert cb is not None
    model.learn(total_timesteps=96, callback=cb, progress_bar=False)

    text = diag_path.read_text(encoding="utf-8").strip()
    assert text, "fps_diag.jsonl should have at least one line"
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert len(lines) >= 2
    row = json.loads(lines[-1])
    for key in (
        "schema_version",
        "iteration",
        "total_timesteps",
        "time_elapsed_s",
        "env_collect_s",
        "main_proc_rss_mb",
        "main_proc_rss_delta_mb",
        "sum_worker_rss_mb",
        "system_ram_used_pct",
        "worker_step_time_p99_max_s",
        "worker_step_time_p99_min_s",
        "n_envs",
        "machine_id",
    ):
        assert key in row
    assert row["schema_version"] == "1.0"
    assert row["machine_id"] == "test-machine"
    assert row["n_envs"] == 1
    assert isinstance(row["env_collect_s"], (int, float))
