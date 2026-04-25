"""
Phase 1b: byte-identical observation + action mask with AWBW_PREALLOCATED_BUFFERS 0 vs 1.

A→B→A: snapshot at state A, transition to B, rebuild A, assert A matches and that
ON-mode consecutive snapshots at the same rebuilt state are identical (no buffer leak).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from engine.action import ActionStage, ActionType
from rl.env import AWBWEnv, _action_to_flat

ROOT = Path(__file__).resolve().parents[1]
POOL_PATH = ROOT / "data" / "gl_map_pool.json"
MAP_ID = 123858
PRE = "AWBW_PREALLOCATED_BUFFERS"


def _pool_single_map() -> list[dict]:
    with open(POOL_PATH, encoding="utf-8") as f:
        pool = json.load(f)
    return [next(m for m in pool if m.get("map_id") == MAP_ID)]


def _make_env(
    monkeypatch: pytest.MonkeyPatch,
    *,
    prealloc: str,
) -> AWBWEnv:
    monkeypatch.setenv("AWBW_LEARNER_GREEDY_MIX", "0")
    monkeypatch.setenv(PRE, prealloc)
    monkeypatch.setattr("rl.env._append_game_log_line", lambda _r: None)
    return AWBWEnv(
        map_pool=_pool_single_map(),
        opponent_policy=None,
        co_p0=1,
        co_p1=1,
        tier_name="T3",
        curriculum_broad_prob=0.0,
    )


def _step_random_legal(env: AWBWEnv, rs: np.random.RandomState) -> None:
    m = env.action_masks()
    leg = np.flatnonzero(m)
    assert leg.size > 0, "no legal actions"
    env.step(int(rs.choice(leg)))


def _assert_arr_equiv(a: np.ndarray, b: np.ndarray) -> None:
    assert a.dtype == b.dtype, (a.dtype, b.dtype)
    assert a.shape == b.shape, (a.shape, b.shape)
    assert np.array_equal(a, b)
    assert bytes(a.tobytes()) == bytes(b.tobytes())


def _assert_obs_equiv(x: dict[str, np.ndarray], y: dict[str, np.ndarray]) -> None:
    _assert_arr_equiv(x["spatial"], y["spatial"])
    _assert_arr_equiv(x["scalars"], y["scalars"])


def _snap_obs_mask(env: AWBWEnv) -> tuple[dict[str, np.ndarray], np.ndarray]:
    o = env._get_obs()
    m = env.action_masks()
    return {
        "spatial": o["spatial"].copy(),
        "scalars": o["scalars"].copy(),
    }, m.copy()


def _assert_prealloc_parity(
    env: AWBWEnv,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Same ``GameState``: prealloc on vs off must be byte-identical."""
    if not env._use_preallocated_buffers:
        env._use_preallocated_buffers = True
    on_o, on_m = _snap_obs_mask(env)
    env._use_preallocated_buffers = False
    off_o, off_m = _snap_obs_mask(env)
    _assert_obs_equiv(on_o, off_o)
    _assert_arr_equiv(on_m, off_m)
    env._use_preallocated_buffers = True
    on_o2, on_m2 = _snap_obs_mask(env)
    _assert_obs_equiv(on_o, on_o2)
    _assert_arr_equiv(on_m, on_m2)
    return on_o, on_m


def _nontrivial(
    obs: dict[str, np.ndarray],
    mask: np.ndarray,
) -> None:
    assert bool(obs["spatial"].any() or obs["scalars"].any() or mask.any()), (
        "expected non-zero encodings for sanity"
    )


def build_turn_start_state(env: AWBWEnv) -> None:
    env.reset(seed=42)


def build_mid_action_select_unit(env: AWBWEnv) -> None:
    """Reach MOVE stage (unit selected) if possible; else a fixed-depth mid-game."""
    env.reset(seed=42)
    rs = np.random.RandomState(3)
    for _ in range(30):
        if env.state is None or env.state.done:
            return
        if env.state.action_stage == ActionStage.MOVE:
            return
        _step_random_legal(env, rs)


def build_endgame_state(env: AWBWEnv) -> None:
    env.reset(seed=42)
    rs = np.random.RandomState(7)
    for _ in range(55):
        if env.state is None or env.state.done:
            break
        _step_random_legal(env, rs)


def build_post_capture_state(env: AWBWEnv) -> None:
    """Reach a state immediately after a CAPTURE, or a deep mid-game if none found."""
    env.reset(seed=123)
    rs = np.random.RandomState(99)
    for _ in range(280):
        if env.state is None or env.state.done:
            return
        legal = env._get_legal()
        cap = next((a for a in legal if a.action_type == ActionType.CAPTURE), None)
        if cap is not None:
            env.step(_action_to_flat(cap))
            return
        _step_random_legal(env, rs)
    build_endgame_state(env)


def _transition_b_for(
    name: str,
) -> Callable[[AWBWEnv], None]:
    if name == "turn_start":
        return lambda e: _step_random_legal(e, np.random.RandomState(0))

    if name == "mid_action_select_unit":

        def _go(e: AWBWEnv) -> None:
            _step_random_legal(e, np.random.RandomState(1))

        return _go

    if name in ("end_of_game_almost", "after_capture"):

        def _go2(e: AWBWEnv) -> None:
            _step_random_legal(e, np.random.RandomState(2))

        return _go2

    raise AssertionError(name)


# (scenario_name, state_builder)
SCENARIOS: list[tuple[str, Callable[[AWBWEnv], None]]] = [
    ("turn_start", build_turn_start_state),
    ("mid_action_select_unit", build_mid_action_select_unit),
    ("end_of_game_almost", build_endgame_state),
    ("after_capture", build_post_capture_state),
]


@pytest.mark.parametrize("scenario_name,build", SCENARIOS)
def test_golden_prealloc_on_matches_off_aba(
    scenario_name: str,
    build: Callable[[AWBWEnv], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """For one env at a fixed :class:`GameState`: ``AWBW_PREALLOCATED_BUFFERS=0`` vs
    ``1`` must yield identical obs/mask (``_assert_prealloc_parity``). A→B exercises
    a different state, then a second / third snap at A/B confirms no within-state drift.

    We do *not* assert A snapshot equality across two fresh envs after a long stochastic
    ``build`` — :func:`get_legal_actions` ordering can make replay diverge. See
    ``test_aba_after_turn_start_idempotent`` for session-level A = A without random walks.
    """
    to_b = _transition_b_for(scenario_name)
    e = _make_env(monkeypatch, prealloc="1")

    build(e)
    oa, ma = _assert_prealloc_parity(e)
    _nontrivial(oa, ma)
    oa_again, ma_again = _snap_obs_mask(e)
    _assert_obs_equiv(oa, oa_again)
    _assert_arr_equiv(ma, ma_again)

    to_b(e)
    ob, mb = _assert_prealloc_parity(e)
    _nontrivial(ob, mb)
    ob_again, mb_again = _snap_obs_mask(e)
    _assert_obs_equiv(ob, ob_again)
    _assert_arr_equiv(mb, mb_again)


def test_aba_after_turn_start_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stochastic-free ``turn_start``: re-``reset(42)`` on the same env reproduces A."""
    e = _make_env(monkeypatch, prealloc="1")
    build_turn_start_state(e)
    o1, m1 = _snap_obs_mask(e)
    e.reset(seed=42)
    o2, m2 = _snap_obs_mask(e)
    _assert_obs_equiv(o1, o2)
    _assert_arr_equiv(m1, m2)


def test_repeated_get_obs_at_same_state_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    e = _make_env(monkeypatch, prealloc="1")
    build_turn_start_state(e)
    _ = e._get_obs()
    a = e._get_obs()
    b = e._get_obs()
    _assert_obs_equiv(a, b)


def test_stored_obs_becomes_stale_across_step_when_buffer_shared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Document contract: in-place buffer reuse means callers must not retain old obs
    across ``step`` — rollout stacks copy (SB3). A stored spatial snapshot before
    step must not match the previous state's bytes after a state change."""
    e = _make_env(monkeypatch, prealloc="1")
    build_endgame_state(e)
    o0 = e._get_obs()
    snap0 = o0["spatial"].copy()
    snap0s = o0["scalars"].copy()
    for _ in range(8):
        if e.state is None or e.state.done:
            return
        m = e.action_masks()
        leg = np.flatnonzero(m)
        if leg.size == 0:
            return
        e.step(int(leg[0]))
        o1 = e._get_obs()
        if e.state is not None and not e.state.done:
            same_sp = np.array_equal(snap0, o1["spatial"])
            same_sc = np.array_equal(snap0s, o1["scalars"])
            if not (same_sp and same_sc):
                return
        snap0 = o1["spatial"].copy()
        snap0s = o1["scalars"].copy()
    pytest.fail("tried 8 legal steps; none changed the encoded obs — unexpected")


def test_env_respects_prealloc_flag_at_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    e1 = _make_env(monkeypatch, prealloc="1")
    assert e1._use_preallocated_buffers is True
    e0 = _make_env(monkeypatch, prealloc="0")
    assert e0._use_preallocated_buffers is False
