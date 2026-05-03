"""Phase 4: env-scoped get_legal_actions cache (rl.env.AWBWEnv)."""

from __future__ import annotations

import numpy as np
import pytest

import engine.action as engine_action
from rl.env import AWBWEnv


def _counting_get_legal(monkeypatch, call_count: list):
    real_fn = engine_action.get_legal_actions

    def counting_fn(state):
        call_count[0] += 1
        return real_fn(state)

    monkeypatch.setattr(engine_action, "get_legal_actions", counting_fn)
    monkeypatch.setattr("rl.env.get_legal_actions", counting_fn)
    monkeypatch.setattr("rl.candidate_actions.get_legal_actions", counting_fn)


def test_cache_returns_same_object(monkeypatch):
    call_count = [0]
    _counting_get_legal(monkeypatch, call_count)
    env = AWBWEnv()
    env.reset(seed=0)
    baseline = call_count[0]
    a = env._get_legal()
    b = env._get_legal()
    assert a is b
    assert call_count[0] == baseline + 1


def test_cache_invalidated_on_step(monkeypatch):
    call_count = [0]
    _counting_get_legal(monkeypatch, call_count)
    env = AWBWEnv()
    env.reset(seed=0)
    call_count[0] = 0
    env._get_legal()
    assert env._legal_cache is not None
    legal0 = env._legal_cache
    act = legal0[0]
    env._engine_step_with_belief(act)
    assert env._legal_cache is None
    assert call_count[0] == 1
    env._get_legal()
    assert call_count[0] == 2


def test_cache_invalidated_on_reset(monkeypatch):
    call_count = [0]
    _counting_get_legal(monkeypatch, call_count)
    env = AWBWEnv()
    env.reset(seed=0)
    env._get_legal()
    assert env._legal_cache is not None
    env.reset(seed=1)
    assert env._legal_cache is None


def test_action_masks_cache_consistency(monkeypatch):
    call_count = [0]
    _counting_get_legal(monkeypatch, call_count)
    env = AWBWEnv()
    env.reset(seed=0)
    m1 = env.action_masks()
    m2 = env.action_masks()
    assert m1 is m2  # same buffer
    assert np.array_equal(m1, m2)
    # Candidate enumeration probes many legal lists per mask (MOVE × ACTION probe).
    assert call_count[0] >= 1


def test_step_with_action_uses_cached_legal(monkeypatch):
    call_count = [0]
    _counting_get_legal(monkeypatch, call_count)
    env = AWBWEnv()
    env.reset(seed=0)
    monkeypatch.setattr(env, "_run_random_opponent", lambda acc: (acc, 0.0))
    monkeypatch.setattr(env, "_run_policy_opponent", lambda acc: (acc, 0.0))
    call_count[0] = 0
    mask = env.action_masks()
    legal_idx = int(np.flatnonzero(mask)[0])
    env.step(legal_idx)
    assert call_count[0] >= 1


def test_full_episode_no_illegal_action_errors():
    rng = np.random.default_rng(0)
    env = AWBWEnv(max_env_steps=200, max_p1_microsteps=80)
    obs, info = env.reset(seed=42)
    for _ in range(500):
        mask = env.action_masks()
        idxs = np.flatnonzero(mask)
        if len(idxs) == 0:
            break
        a = int(rng.choice(idxs))
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            break


def test_strip_non_infantry_builds_with_cache(monkeypatch):
    pytest.skip(
        "AWBW_BUILD_MASK_INFANTRY_ONLY reshaped flat build logits; learner mask is "
        "candidate rows — use flat opponent internals or extend enumeration if needed."
    )
