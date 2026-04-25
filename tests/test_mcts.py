"""Phase 11b: turn-level PUCT MCTS (``rl.mcts``)."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.action import Action, ActionType, get_legal_actions  # noqa: E402
from engine.game import GameState, make_initial_state  # noqa: E402
from engine.map_loader import load_map  # noqa: E402
from rl.env import AWBWEnv  # noqa: E402
from rl.mcts import (  # noqa: E402
    MCTSConfig,
    decision_log_context_from_env,
    make_callables_from_sb3_policy,
    run_mcts,
)

POOL_PATH = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"
MAP_ID = 123858


def _map_pool_single() -> list[dict]:
    with open(POOL_PATH, encoding="utf-8") as f:
        pool = json.load(f)
    return [next(m for m in pool if m.get("map_id") == MAP_ID)]


def _fresh_state() -> GameState:
    md = load_map(MAP_ID, POOL_PATH, MAPS_DIR)
    return make_initial_state(
        md, 1, 1, starting_funds=0, tier_name="T2", replay_first_mover=0
    )


def test_mcts_smoke() -> None:
    s = _fresh_state()
    _rng = np.random.default_rng(0)

    def policy_callable(st: GameState) -> Action:
        leg = get_legal_actions(st)
        assert leg
        return leg[int(_rng.integers(0, len(leg)))]

    cfg = MCTSConfig(
        num_sims=4,
        root_plans=2,
        min_depth=2,
        dirichlet_epsilon=0.0,
        temperature=0.0,
        rng_seed=1,
    )
    plan, stats = run_mcts(
        s,
        policy_callable=policy_callable,
        value_callable=lambda _st: 0.0,
        prior_callable=lambda _st, plans: [1.0 / len(plans)] * len(plans),
        config=cfg,
    )
    assert isinstance(plan, list)
    assert all(isinstance(a, Action) for a in plan)
    assert len(plan) >= 1
    assert stats["total_sims_run"] == 4


def test_mcts_determinism_temperature_zero() -> None:
    s0 = _fresh_state()

    def _make_pol(seed: int):
        rng = np.random.default_rng(seed)

        def policy_callable(st: GameState) -> Action:
            leg = get_legal_actions(st)
            return leg[int(rng.integers(0, len(leg)))]

        return policy_callable

    cfg = MCTSConfig(
        num_sims=8,
        root_plans=3,
        min_depth=1,
        dirichlet_epsilon=0.0,
        temperature=0.0,
        rng_seed=42,
    )
    p1, _ = run_mcts(
        copy.deepcopy(s0),
        policy_callable=_make_pol(999),
        value_callable=lambda _st: 0.0,
        prior_callable=lambda _st, plans: [1.0 / len(plans)] * len(plans),
        config=cfg,
    )
    p2, _ = run_mcts(
        copy.deepcopy(s0),
        policy_callable=_make_pol(999),
        value_callable=lambda _st: 0.0,
        prior_callable=lambda _st, plans: [1.0 / len(plans)] * len(plans),
        config=cfg,
    )
    assert p1 == p2


def test_mcts_terminal_short_circuit(monkeypatch: pytest.MonkeyPatch) -> None:
    from engine.game import GameState as GS

    s = _fresh_state()

    def fake_apply(self, plan_or_policy, **kwargs):
        st = copy.deepcopy(self)
        st.done = True
        st.winner = int(self.active_player)
        return st, [Action(ActionType.END_TURN)], 0.0, True

    monkeypatch.setattr(GS, "apply_full_turn", fake_apply)
    cfg = MCTSConfig(
        num_sims=2,
        root_plans=2,
        min_depth=0,
        dirichlet_epsilon=0.0,
        temperature=0.0,
        rng_seed=0,
    )
    plan, stats = run_mcts(
        s,
        policy_callable=lambda st: Action(ActionType.END_TURN),
        value_callable=lambda _st: 0.0,
        prior_callable=lambda _st, plans: [1.0 / len(plans)] * len(plans),
        config=cfg,
    )
    assert plan and plan[0].action_type == ActionType.END_TURN
    assert stats["total_sims_run"] == 2


def test_mcts_visit_counts_sum_matches_sims() -> None:
    s = _fresh_state()

    def policy_callable(st: GameState) -> Action:
        leg = get_legal_actions(st)
        return leg[int(np.random.randint(0, len(leg)))]

    cfg = MCTSConfig(
        num_sims=16,
        root_plans=4,
        min_depth=0,
        dirichlet_epsilon=0.0,
        temperature=0.0,
        rng_seed=7,
    )
    _plan, stats = run_mcts(
        s,
        policy_callable=policy_callable,
        value_callable=lambda _st: 0.0,
        prior_callable=lambda _st, plans: [1.0 / len(plans)] * len(plans),
        config=cfg,
    )
    vc = stats["visit_counts"]
    assert len(vc) >= 1
    assert sum(vc.values()) == stats["total_sims_run"] == 16


def test_mcts_min_depth_pv() -> None:
    s = _fresh_state()

    def policy_callable(st: GameState) -> Action:
        leg = get_legal_actions(st)
        return leg[int(np.random.randint(0, len(leg)))]

    cfg = MCTSConfig(
        num_sims=256,
        root_plans=6,
        min_depth=4,
        dirichlet_epsilon=0.0,
        temperature=0.0,
        rng_seed=123,
    )
    _plan, stats = run_mcts(
        s,
        policy_callable=policy_callable,
        value_callable=lambda _st: 0.0,
        prior_callable=lambda _st, plans: [1.0 / len(plans)] * len(plans),
        config=cfg,
    )
    assert stats["principal_variation_depth"] >= 4


def test_make_callables_sb3_smoke() -> None:
    import torch
    from sb3_contrib.common.maskable.distributions import (  # type: ignore[import]
        MaskableCategorical,
    )

    class _DummyPol:
        def obs_to_tensor(self, obs):
            if isinstance(obs, dict):
                return {
                    k: torch.as_tensor(v, dtype=torch.float32).unsqueeze(0)
                    for k, v in obs.items()
                }, None
            raise TypeError(obs)

        def predict_values(self, obs_t):
            if isinstance(obs_t, dict):
                b = next(iter(obs_t.values())).shape[0]
            else:
                b = obs_t.shape[0]
            return torch.zeros((b, 1), dtype=torch.float32)

        def get_distribution(self, obs_t, action_masks=None):
            if action_masks is None:
                raise ValueError("need masks")
            n = int(action_masks.shape[-1])
            logits = torch.zeros((1, n), dtype=torch.float32)
            return MaskableCategorical(logits=logits, masks=action_masks)

    class _DummyModel:
        def __init__(self) -> None:
            self.device = torch.device("cpu")
            self.policy = _DummyPol()

        def predict(self, obs, action_masks=None, deterministic=False):
            mask = action_masks
            legal = np.where(mask)[0]
            idx = int(legal[0]) if len(legal) > 0 else 0
            return np.array([idx], dtype=np.int64), None

    env = AWBWEnv(
        map_pool=_map_pool_single(),
        opponent_policy=None,
        co_p0=10,
        co_p1=23,
        tier_name="T2",
    )
    env.reset(seed=0)
    s = env.state
    assert s is not None
    pol_c, val_c, prior_c = make_callables_from_sb3_policy(_DummyModel(), env)
    cfg = MCTSConfig(
        num_sims=2,
        root_plans=2,
        min_depth=0,
        dirichlet_epsilon=0.0,
        temperature=0.0,
        rng_seed=99,
    )
    plan, stats = run_mcts(
        copy.deepcopy(s),
        policy_callable=pol_c,
        value_callable=val_c,
        prior_callable=prior_c,
        config=cfg,
    )
    assert len(plan) >= 1
    assert stats["total_sims_run"] == 2


def test_decision_log_context_from_env_matches_game_log_keys() -> None:
    env = AWBWEnv(
        map_pool=_map_pool_single(),
        opponent_policy=None,
        co_p0=10,
        co_p1=23,
        tier_name="T2",
        curriculum_tag="stage1-misery-andy",
    )
    env.reset(seed=0)
    ctx = decision_log_context_from_env(env)
    assert set(ctx.keys()) >= {
        "curriculum_tag",
        "map_id",
        "tier",
        "p0_co_id",
        "p1_co_id",
        "p0_env_steps",
        "turn",
        "truncated",
        "truncation_reason",
    }
    assert ctx["curriculum_tag"] == "stage1-misery-andy"
    assert ctx["truncated"] is False
    assert ctx["truncation_reason"] is None


def test_mcts_root_decision_log_merges_decision_log_context(tmp_path: Path) -> None:
    s = _fresh_state()

    def policy_callable(st: GameState) -> Action:
        leg = get_legal_actions(st)
        assert leg
        return leg[0]

    log_path = tmp_path / "root_mcts.jsonl"
    ctx = {
        "curriculum_tag": "stage2",
        "map_id": 123858,
        "tier": "T2",
        "p0_co_id": 1,
        "p1_co_id": 2,
        "p0_env_steps": 5,
        "turn": 3,
        "truncated": False,
        "truncation_reason": None,
    }
    cfg = MCTSConfig(
        num_sims=2,
        root_plans=1,
        min_depth=0,
        dirichlet_epsilon=0.0,
        temperature=0.0,
        rng_seed=42,
        root_decision_log_path=str(log_path),
    )
    _plan, stats = run_mcts(
        s,
        policy_callable=policy_callable,
        value_callable=lambda _st: 0.0,
        prior_callable=lambda _st, plans: [1.0 / len(plans)] * len(plans),
        config=cfg,
        decision_log_context=ctx,
    )
    assert stats["decision_log_context"] == ctx
    line = log_path.read_text(encoding="utf-8").strip().splitlines()[-1]
    rec = json.loads(line)
    for k, v in ctx.items():
        assert rec.get(k) == v


def test_mcts_luck_resample_stats_schema() -> None:
    s = _fresh_state()

    def policy_callable(st: GameState) -> Action:
        leg = get_legal_actions(st)
        assert leg
        return leg[0]

    cfg = MCTSConfig(
        num_sims=2,
        root_plans=1,
        min_depth=0,
        dirichlet_epsilon=0.0,
        temperature=0.0,
        rng_seed=1234,
        luck_resamples=2,
        luck_resample_critical_only=False,
        risk_mode="mean_minus_p10",
    )
    plan, stats = run_mcts(
        s,
        policy_callable=policy_callable,
        value_callable=lambda _st: 0.0,
        prior_callable=lambda _st, plans: [1.0 / len(plans)] * len(plans),
        config=cfg,
    )
    assert plan
    assert stats["risk_mode"] == "mean_minus_p10"
    assert stats["luck_resamples"] == 2
    assert isinstance(stats["chosen_risk"], dict)
    assert stats["chosen_risk"]["resample_count"] == 2
    assert isinstance(stats["root_child_stats"], list)
    assert stats["root_child_stats"]


def test_apply_full_turn_return_trace_schema() -> None:
    s = _fresh_state()
    legal = get_legal_actions(s)
    assert legal
    result = s.apply_full_turn(
        [Action(ActionType.END_TURN)],
        copy=True,
        max_actions=8,
        rng_seed=5,
        return_trace=True,
    )
    assert len(result) == 5
    _st, actions, _reward, done, trace = result
    assert actions
    assert done is False
    assert isinstance(trace, list)
    assert trace
    assert "action_type" in trace[0]
    assert "critical_threshold_event" in trace[0]
