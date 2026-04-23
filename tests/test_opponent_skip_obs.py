"""Phase 1a: _CheckpointOpponent.needs_observation and env conditional _get_obs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rl.env import AWBWEnv, _get_action_mask
from rl.self_play import _CheckpointOpponent

ROOT = Path(__file__).resolve().parents[1]
POOL_PATH = ROOT / "data" / "gl_map_pool.json"
MAP_ID = 123858


def _single_map_pool() -> list[dict]:
    with open(POOL_PATH, encoding="utf-8") as f:
        pool = json.load(f)
    return [next(m for m in pool if m.get("map_id") == MAP_ID)]


def test_checkpoint_opponent_needs_observation_returns_false_when_cold(
    tmp_path: Path,
) -> None:
    opp = _CheckpointOpponent(checkpoint_dir=str(tmp_path))
    assert opp.needs_observation() is False


def test_checkpoint_opponent_needs_observation_returns_true_after_model_loaded() -> None:
    opp = _CheckpointOpponent(checkpoint_dir="/nonexistent_empty")
    assert opp.needs_observation() is False
    opp._model = object()
    assert opp.needs_observation() is True
    opp._model = None
    assert opp.needs_observation() is False


def _p0_choose_index(mask: np.ndarray) -> int:
    return 0 if mask[0] else int(np.flatnonzero(mask)[0])


def _p1_made_calls(fake: object) -> bool:
    if hasattr(fake, "obs_list") and len(getattr(fake, "obs_list")) >= 1:
        return True
    return getattr(fake, "last_obs", "unset") != "unset"


def _run_until_p1_opponent_fires(
    env: AWBWEnv,
    fake: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reset and, if needed, take P0 steps until the opponent has acted at least once."""
    monkeypatch.setattr("rl.env._append_game_log_line", lambda _record: None)
    env.reset(seed=0)
    assert env.state is not None
    if _p1_made_calls(fake):
        return
    for _ in range(600):
        if env.state is None or env.state.done:
            break
        assert env.state.active_player == 0, "expected P0 clock before env.step in harness"
        m = _get_action_mask(env.state)
        if not m.any():
            break
        env.step(_p0_choose_index(m))
        if _p1_made_calls(fake):
            return
    pytest.fail("P1 opponent was never invoked — tighten map/co or raise step budget")


class FakeOppDecline:
    def __init__(self) -> None:
        self.last_obs: object | None = "unset"
        self.obs_list: list[object | None] = []

    def needs_observation(self) -> bool:
        return False

    def __call__(self, obs: object, mask: np.ndarray) -> int:
        self.last_obs = obs
        self.obs_list.append(obs)
        if mask.any():
            return int(np.where(mask)[0][0])
        return 0


class FakeOppRequire(FakeOppDecline):
    def needs_observation(self) -> bool:
        return True


class FakeOppLegacy:
    def __init__(self) -> None:
        self.last_obs: object | None = "unset"
        self.obs_list: list[object] = []

    def __call__(self, obs: object, mask: np.ndarray) -> int:
        self.last_obs = obs
        self.obs_list.append(obs)
        if mask.any():
            return int(np.where(mask)[0][0])
        return 0


def test_run_policy_opponent_passes_none_when_opponent_declines_obs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Short episode: P0 steps until P1 has acted; all opponent calls get obs is None."""
    fake = FakeOppDecline()
    env = AWBWEnv(
        map_pool=_single_map_pool(),
        opponent_policy=fake,
        co_p0=1,
        co_p1=1,
        tier_name="T3",
    )
    _run_until_p1_opponent_fires(env, fake, monkeypatch)
    assert len(fake.obs_list) >= 1, "P1 opponent should have been called at least once"
    assert all(x is None for x in fake.obs_list)
    assert fake.last_obs is None


def test_run_policy_opponent_passes_obs_when_opponent_demands_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeOppRequire()
    env = AWBWEnv(
        map_pool=_single_map_pool(),
        opponent_policy=fake,
        co_p0=1,
        co_p1=1,
        tier_name="T3",
    )
    _run_until_p1_opponent_fires(env, fake, monkeypatch)
    assert len(fake.obs_list) >= 1
    assert all(isinstance(x, dict) for x in fake.obs_list)
    assert isinstance(fake.last_obs, dict)


def test_run_policy_opponent_handles_opponent_without_needs_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeOppLegacy()
    env = AWBWEnv(
        map_pool=_single_map_pool(),
        opponent_policy=fake,
        co_p0=1,
        co_p1=1,
        tier_name="T3",
    )
    _run_until_p1_opponent_fires(env, fake, monkeypatch)
    assert len(fake.obs_list) >= 1
    assert isinstance(fake.last_obs, dict), (
        "opponent without needs_observation must still receive a real obs dict (default-True)"
    )
