"""
Phase 1b: golden checksums for AWBWEnv reused numpy buffers (mask + spatial/scalars).

Buffer reuse without a full zero/fill discipline causes silent training corruption.
These tests compare SHA-256 digests of observations and action masks across env swaps
and across reset() on the same instance.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest

from rl.encoder import GRID_SIZE, N_SPATIAL_CHANNELS, N_SCALARS
from rl.env import AWBWEnv

ROOT = Path(__file__).resolve().parents[1]
POOL_PATH = ROOT / "data" / "gl_map_pool.json"
MAP_ID = 123858


def _pool_single_map() -> list[dict]:
    with open(POOL_PATH, encoding="utf-8") as f:
        pool = json.load(f)
    return [next(m for m in pool if m.get("map_id") == MAP_ID)]


def _digest_obs(obs: dict) -> str:
    h = hashlib.sha256()
    h.update(obs["spatial"].tobytes())
    h.update(obs["scalars"].tobytes())
    return h.hexdigest()


def _digest_mask(mask: np.ndarray) -> str:
    return hashlib.sha256(mask.tobytes()).hexdigest()


def _record_obs_mask(env: AWBWEnv) -> tuple[str, str]:
    obs = env._get_obs()
    mask = env.action_masks()
    return _digest_obs(obs), _digest_mask(mask)


def _run_segment(
    env: AWBWEnv,
    *,
    reset_seed: int,
    rng_py: int,
    rng_np: np.random.RandomState,
    n_steps: int,
) -> list[tuple[str, str]]:
    """reset + n_steps; record (obs_digest, mask_digest) after reset and after each step."""
    random.seed(rng_py)
    env.reset(seed=reset_seed)
    out: list[tuple[str, str]] = [_record_obs_mask(env)]
    for _ in range(n_steps):
        if env.state is not None and env.state.done:
            break
        m = env.action_masks()
        legal = np.flatnonzero(m)
        assert legal.size > 0, "no legal actions — cannot continue golden segment"
        a = int(rng_np.choice(legal))
        env.step(a)
        out.append(_record_obs_mask(env))
    return out


def test_golden_a_b_a_no_cross_env_bleed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap env instance B between two A runs; obs/mask digests must match (no buffer bleed)."""
    monkeypatch.setenv("AWBW_LEARNER_GREEDY_MIX", "0")
    monkeypatch.setattr("rl.env._append_game_log_line", lambda _record: None)

    pool = _pool_single_map()

    def make_a() -> AWBWEnv:
        return AWBWEnv(
            map_pool=pool,
            opponent_policy=None,
            co_p0=1,
            co_p1=1,
            tier_name="T3",
            curriculum_broad_prob=0.0,
        )

    env_a1 = make_a()
    pre = _run_segment(
        env_a1, reset_seed=42, rng_py=100, rng_np=np.random.RandomState(0), n_steps=20
    )

    # Misery T3 co_ids are {1,5,11,16,28} — pick two ≠ (1,1) so B walks a different path.
    env_b = AWBWEnv(
        map_pool=pool,
        opponent_policy=None,
        co_p0=11,
        co_p1=5,
        tier_name="T3",
        curriculum_broad_prob=0.0,
    )
    _run_segment(
        env_b, reset_seed=99, rng_py=200, rng_np=np.random.RandomState(7), n_steps=15
    )

    env_a2 = make_a()
    post = _run_segment(
        env_a2, reset_seed=42, rng_py=100, rng_np=np.random.RandomState(0), n_steps=20
    )

    assert len(pre) == len(post), (len(pre), len(post))
    for i, ((o1, m1), (o2, m2)) in enumerate(zip(pre, post)):
        assert o1 == o2, f"obs drift at frame {i}"
        assert m1 == m2, f"mask drift at frame {i}"


def test_golden_reset_same_instance_no_bleed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same env: two identical segments separated by reset must yield identical digests."""
    monkeypatch.setenv("AWBW_LEARNER_GREEDY_MIX", "0")
    monkeypatch.setattr("rl.env._append_game_log_line", lambda _record: None)

    env = AWBWEnv(
        map_pool=_pool_single_map(),
        opponent_policy=None,
        co_p0=1,
        co_p1=1,
        tier_name="T3",
        curriculum_broad_prob=0.0,
    )
    first = _run_segment(
        env, reset_seed=42, rng_py=300, rng_np=np.random.RandomState(0), n_steps=20
    )
    second = _run_segment(
        env, reset_seed=42, rng_py=300, rng_np=np.random.RandomState(0), n_steps=20
    )
    assert first == second


def test_reset_map_id_override_no_crash_and_fixed_obs_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """reset(options={'map_id': ...}) switches episode map; spatial obs stays padded GRID_SIZE.

    Map-specific H×W is clamped into a fixed (GRID_SIZE, GRID_SIZE, C) tensor in the
    encoder; changing maps does not change observation outer shape — only map_id and
    encoded cell content differ.
    """
    monkeypatch.setenv("AWBW_LEARNER_GREEDY_MIX", "0")
    monkeypatch.setattr("rl.env._append_game_log_line", lambda _record: None)

    with open(POOL_PATH, encoding="utf-8") as f:
        full = json.load(f)
    std_maps = [m for m in full if m.get("type") == "std"]
    other_id = next(
        int(m["map_id"]) for m in std_maps if int(m.get("map_id", -1)) != MAP_ID
    )

    env = AWBWEnv(
        map_pool=std_maps,
        opponent_policy=None,
        co_p0=1,
        co_p1=1,
        tier_name="T3",
        curriculum_broad_prob=0.0,
    )

    env.reset(seed=0)
    for _ in range(5):
        if env.state is None or env.state.done:
            break
        m = env.action_masks()
        env.step(int(np.flatnonzero(m)[0]))

    obs, _info = env.reset(seed=1, options={"map_id": other_id})
    assert env._episode_info["map_id"] == other_id
    assert obs["spatial"].shape == (GRID_SIZE, GRID_SIZE, N_SPATIAL_CHANNELS)
    assert obs["scalars"].shape[0] == N_SCALARS
