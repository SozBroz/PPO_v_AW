"""Phase 5: fast path in AWBWEnv._engine_step_with_belief for SELECT_UNITS
that do not mutate units (stages SELECT / MOVE)."""

from __future__ import annotations

import numpy as np
import pytest

from engine.action import Action, ActionStage, ActionType, get_legal_actions
from engine.game import make_initial_state
from engine.map_loader import load_map
from engine.belief import BeliefState

from rl.env import AWBWEnv

from server.play_human import MAPS_DIR, POOL_PATH

MAP_WITH_FACTORY = 123858  # also used in test_engine_awbw_subset
# Smaller GL map where seed-0 / P0-open has predeploy so SELECT_UNIT exists
# (many maps have P0=0 units → legal is only END_TURN / BUILD).
MAP_WITH_P0_UNITS = 133665


def _reset_env_with_select_unit(
    env: AWBWEnv, monkeypatch, max_seed: int = 64
) -> list:
    """Pick (seed, map) so SELECT stage includes at least one SELECT_UNIT."""
    monkeypatch.setattr(env, "_run_random_opponent", lambda acc: acc)
    for seed in range(max_seed):
        env.reset(seed=seed, options={"map_id": MAP_WITH_P0_UNITS})
        legal = get_legal_actions(env.state)
        if any(a.action_type == ActionType.SELECT_UNIT for a in legal):
            return legal
    raise AssertionError(
        f"no seed produced SELECT_UNIT on map {MAP_WITH_P0_UNITS}"
    )


def _first_select_unit(legal: list) -> Action:
    for a in legal:
        if a.action_type == ActionType.SELECT_UNIT:
            return a
    raise AssertionError("no SELECT_UNIT in legal actions")


def _to_move_stage(env: AWBWEnv) -> None:
    """From SELECT, pick a legal SELECT_UNIT so action_stage becomes MOVE."""
    a = _first_select_unit(get_legal_actions(env.state))
    env._engine_step_with_belief(a)
    assert env.state.action_stage == ActionStage.MOVE


def _to_action_wait(env: AWBWEnv) -> None:
    """SELECT → SELECT(move) so we can take WAIT in ACTION."""
    _to_move_stage(env)
    legal2 = get_legal_actions(env.state)
    move_acts = [a for a in legal2 if a.action_type == ActionType.SELECT_UNIT]
    assert move_acts, "need at least one move tile"
    env._engine_step_with_belief(move_acts[0])
    assert env.state.action_stage == ActionStage.ACTION


def test_select_unit_in_select_stage_skips_snapshot(monkeypatch):
    env = AWBWEnv()
    legal = _reset_env_with_select_unit(env, monkeypatch)
    a = _first_select_unit(legal)
    n = [0]

    def count_snap(*_a, **_k):
        n[0] += 1
        return {}

    monkeypatch.setattr(env, "_snapshot_units", count_snap)
    env._engine_step_with_belief(a)
    assert n[0] == 0


def test_select_unit_in_move_stage_skips_snapshot(monkeypatch):
    env = AWBWEnv()
    _reset_env_with_select_unit(env, monkeypatch)
    _to_move_stage(env)
    legal = get_legal_actions(env.state)
    move_acts = [a for a in legal if a.action_type == ActionType.SELECT_UNIT]
    assert move_acts
    n = [0]

    def count_snap(*_a, **_k):
        n[0] += 1
        return {}

    monkeypatch.setattr(env, "_snapshot_units", count_snap)
    env._engine_step_with_belief(move_acts[0])
    assert n[0] == 0


def test_action_stage_runs_full_belief_diff(monkeypatch):
    env = AWBWEnv()
    _reset_env_with_select_unit(env, monkeypatch)
    _to_action_wait(env)
    n = [0]
    real = env._snapshot_units

    def count_snap(*_a, **_k):
        n[0] += 1
        return real()

    monkeypatch.setattr(env, "_snapshot_units", count_snap)
    wait = next(
        a
        for a in get_legal_actions(env.state)
        if a.action_type == ActionType.WAIT
    )
    env._engine_step_with_belief(wait)
    assert n[0] > 0


def test_end_turn_runs_full_belief_diff(monkeypatch):
    env = AWBWEnv()
    env.reset(seed=0, options={"map_id": MAP_WITH_FACTORY})
    monkeypatch.setattr(env, "_run_random_opponent", lambda acc: acc)
    p = env.state.active_player
    for u in env.state.units[p]:
        u.moved = True
    env._invalidate_legal_cache()
    n = [0]
    real = env._snapshot_units

    def count_snap(*_a, **_k):
        n[0] += 1
        return real()

    monkeypatch.setattr(env, "_snapshot_units", count_snap)
    et = next(
        a
        for a in get_legal_actions(env.state)
        if a.action_type == ActionType.END_TURN
    )
    env._engine_step_with_belief(et)
    assert n[0] > 0


def test_activate_cop_runs_full_belief_diff(monkeypatch):
    env = AWBWEnv()
    env.reset(seed=0)
    monkeypatch.setattr(env, "_run_random_opponent", lambda acc: acc)
    p = env.state.active_player
    co = env.state.co_states[p]
    if not co.can_activate_cop():
        co.power_bar = max(co._cop_threshold, co.power_bar)
    legal = get_legal_actions(env.state)
    cops = [a for a in legal if a.action_type == ActionType.ACTIVATE_COP]
    if not cops:
        pytest.skip("COP not in legal set (CO or charge layout)")
    n = [0]
    real = env._snapshot_units

    def count_snap(*_a, **_k):
        n[0] += 1
        return real()

    monkeypatch.setattr(env, "_snapshot_units", count_snap)
    env._engine_step_with_belief(cops[0])
    assert n[0] > 0


def test_legal_cache_invalidated_on_select_unit(monkeypatch):
    env = AWBWEnv()
    _reset_env_with_select_unit(env, monkeypatch)
    env._get_legal()
    assert env._legal_cache is not None
    a = _first_select_unit(get_legal_actions(env.state))
    env._engine_step_with_belief(a)
    assert env._legal_cache is None


def test_belief_state_consistency_full_episode(monkeypatch):
    env = AWBWEnv(max_env_steps=120, max_p1_microsteps=50)
    monkeypatch.setattr(env, "_run_random_opponent", lambda acc: acc)
    rng = np.random.default_rng(0)
    for ep_seed in (0, 1, 2):
        env.reset(seed=ep_seed, options={"map_id": MAP_WITH_P0_UNITS})
        for _ in range(400):
            assert len(env._beliefs) == 2
            for b in env._beliefs.values():
                assert b is not None
            assert env._get_obs(0) is not None
            m = env.action_masks()
            idxs = np.flatnonzero(m)
            if len(idxs) == 0:
                break
            a = int(rng.choice(idxs))
            _obs, _r, term, trunc, _ = env.step(a)
            for pl in (0, 1):
                bel = env._beliefs[pl]
                for u in env.state.units[pl]:
                    if u.is_alive:
                        assert u.unit_id in bel
            if term or trunc:
                break


def test_unit_built_during_action_stage_recorded(monkeypatch):
    m = load_map(MAP_WITH_FACTORY, POOL_PATH, MAPS_DIR)
    s = make_initial_state(m, 1, 1, tier_name="T3")
    s.funds[0] = 999_999
    s.active_player = 0
    s.action_stage = ActionStage.SELECT
    s.selected_unit = None
    s.selected_move_pos = None
    built = [0]
    real_built = BeliefState.on_unit_built

    def rec_on_built(bself, unit):
        built[0] += 1
        return real_built(bself, unit)

    monkeypatch.setattr(BeliefState, "on_unit_built", rec_on_built)
    env = AWBWEnv()
    env.reset(seed=0, options={"map_id": MAP_WITH_FACTORY})
    monkeypatch.setattr(env, "_run_random_opponent", lambda acc: acc)
    env.state = s
    env._invalidate_legal_cache()
    for b in env._beliefs.values():
        b.seed_from_state(s)

    legal = get_legal_actions(s)
    builds = [a for a in legal if a.action_type == ActionType.BUILD]
    assert builds, "fixture should expose Stage-0 BUILD"
    env._engine_step_with_belief(builds[0])
    assert built[0] > 0