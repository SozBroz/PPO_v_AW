"""
Phase 11b: single-process PUCT MCTS at turn boundaries (AlphaZero-style).

Uses ``GameState.apply_full_turn`` as the turn rollout primitive. Default OFF
everywhere until Phase 11c wires this into training/eval.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from engine.action import Action, ActionStage
from engine.game import GameState


@dataclass(slots=True)
class TurnNode:
    state: GameState
    actor: int
    visit_count: int = 0
    total_value: float = 0.0
    prior: float = 0.0
    children: dict[bytes, TurnNode] = field(default_factory=dict)
    plan_actions: list[Action] | None = None
    is_terminal: bool = False
    terminal_value: float = 0.0


@dataclass(slots=True)
class MCTSConfig:
    num_sims: int = 16
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    min_depth: int = 4
    root_plans: int = 8
    max_plan_actions: int = 256
    rng_seed: int | None = None


def plan_key(final_state: GameState, actions: list[Action]) -> bytes:
    """Stable dedup key: action tuple + property ownership snapshot."""
    parts: list[tuple[Any, ...]] = []
    for a in actions:
        ut: int | None = None
        if a.unit_type is not None:
            ut = int(a.unit_type)
        parts.append(
            (
                int(a.action_type),
                a.unit_pos,
                a.move_pos,
                a.target_pos,
                ut,
                a.unload_pos,
                a.select_unit_id,
            )
        )
    prop_bits = tuple(
        sorted(
            ((p.row, p.col), p.owner, p.capture_points)
            for p in final_state.properties
        )
    )
    h = hashlib.sha256()
    h.update(repr(parts).encode("utf-8"))
    h.update(repr(prop_bits).encode("utf-8"))
    return h.digest()


def _normalize_priors(raw: list[float], n: int) -> list[float]:
    if len(raw) != n:
        raw = list(raw) + [1.0 / max(1, n)] * max(0, n - len(raw))
        raw = raw[:n]
    s = float(sum(max(0.0, float(x)) for x in raw))
    if s <= 0:
        return [1.0 / n] * n
    return [max(0.0, float(x)) / s for x in raw]


def _terminal_utility(state: GameState, for_actor: int) -> float:
    if not state.done or state.winner is None:
        return 0.0
    if state.winner == -1:
        return 0.0
    return 1.0 if state.winner == for_actor else -1.0


def _leaf_value(
    node: TurnNode,
    value_callable: Callable[[GameState], float],
) -> float:
    if node.is_terminal:
        return float(node.terminal_value)
    st = node.state
    if st.done and st.winner is not None:
        return float(np.clip(_terminal_utility(st, node.actor), -1.0, 1.0))
    return float(np.clip(value_callable(st), -1.0, 1.0))


def _child_q_for_parent(parent: TurnNode, child: TurnNode) -> float:
    if child.visit_count <= 0:
        return 0.0
    q = child.total_value / child.visit_count
    if child.actor != parent.actor:
        q = -q
    return q


def _puct_select_key(node: TurnNode, c_puct: float) -> bytes:
    best_k: bytes | None = None
    best_score = -1e30
    sqrt_n = math.sqrt(max(1, node.visit_count))
    for k, ch in node.children.items():
        q = _child_q_for_parent(node, ch)
        u = c_puct * ch.prior * sqrt_n / (1.0 + ch.visit_count)
        score = q + u
        if score > best_score:
            best_score = score
            best_k = k
    assert best_k is not None
    return best_k


def _prior_greedy_key(node: TurnNode) -> bytes:
    return max(node.children.keys(), key=lambda kk: node.children[kk].prior)


def _sample_plans(
    node: TurnNode,
    policy_callable: Callable[[GameState], Action],
    *,
    k: int,
    max_plan_actions: int,
    rng: np.random.Generator,
) -> list[tuple[bytes, list[Action], GameState, bool]]:
    out: list[tuple[bytes, list[Action], GameState, bool]] = []
    seen: set[bytes] = set()
    attempts = 0
    max_attempts = max(k * 8, k + 16)
    while len(out) < k and attempts < max_attempts:
        attempts += 1
        seed = int(rng.integers(0, 2**31 - 1))
        final_st, actions, _r, done = node.state.apply_full_turn(
            policy_callable,
            copy=True,
            max_actions=max_plan_actions,
            rng_seed=seed,
        )
        key = plan_key(final_st, actions)
        if key in seen:
            continue
        seen.add(key)
        out.append((key, actions, final_st, done))
    return out


def _expand_node(
    node: TurnNode,
    *,
    policy_callable: Callable[[GameState], Action],
    prior_callable: Callable[[GameState, list[list[Action]]], list[float]],
    config: MCTSConfig,
    rng: np.random.Generator,
    apply_dirichlet: bool,
) -> bool:
    if node.is_terminal or node.children:
        return False
    sampled = _sample_plans(
        node,
        policy_callable,
        k=config.root_plans,
        max_plan_actions=config.max_plan_actions,
        rng=rng,
    )
    if not sampled:
        return False
    plans = [t[1] for t in sampled]
    priors = _normalize_priors(prior_callable(node.state, plans), len(plans))
    if apply_dirichlet and config.dirichlet_epsilon > 0 and len(priors) > 0:
        noise = rng.dirichlet(np.full(len(priors), config.dirichlet_alpha))
        eps = config.dirichlet_epsilon
        priors = (1.0 - eps) * np.asarray(priors, dtype=np.float64) + eps * noise
        priors = (priors / priors.sum()).tolist()
    for (key, plan, st, done), prior in zip(sampled, priors, strict=True):
        terminal = bool(done or st.done)
        child = TurnNode(
            state=st,
            actor=int(st.active_player),
            prior=float(prior),
            plan_actions=plan,
        )
        if terminal:
            child.is_terminal = True
            w = st.winner
            if w is None or w == -1:
                child.terminal_value = 0.0
            else:
                child.terminal_value = 1.0 if w == child.actor else -1.0
        node.children[key] = child
    return True


def _select_path(
    root: TurnNode,
    min_depth: int,
    c_puct: float,
) -> list[TurnNode]:
    path: list[TurnNode] = [root]
    node = root
    while True:
        if node.is_terminal:
            break
        if not node.children:
            break
        depth = len(path) - 1
        if depth < min_depth:
            key = _prior_greedy_key(node)
        else:
            key = _puct_select_key(node, c_puct)
        nxt = node.children[key]
        path.append(nxt)
        node = nxt
        if node.is_terminal:
            break
        if not node.children:
            break
    return path


def _backup(path: list[TurnNode], v_leaf: float) -> None:
    v = float(v_leaf)
    for i in range(len(path) - 1, -1, -1):
        node = path[i]
        node.visit_count += 1
        node.total_value += v
        if i == 0:
            break
        parent = path[i - 1]
        if node.actor != parent.actor:
            v = -v


def principal_variation_depth(root: TurnNode) -> int:
    depth = 0
    node = root
    while node.children:
        best_k = max(
            node.children.keys(),
            key=lambda kk: node.children[kk].visit_count,
        )
        node = node.children[best_k]
        depth += 1
    return depth


def run_mcts(
    root_state: GameState,
    *,
    policy_callable: Callable[[GameState], Action],
    value_callable: Callable[[GameState], float],
    prior_callable: Callable[[GameState, list[list[Action]]], list[float]],
    config: MCTSConfig,
) -> tuple[list[Action], dict[str, Any]]:
    if root_state.action_stage != ActionStage.SELECT:
        raise ValueError("run_mcts requires root action_stage == SELECT")
    t0 = time.perf_counter()
    rng = np.random.default_rng(config.rng_seed)
    root = TurnNode(state=root_state, actor=int(root_state.active_player))
    dirichlet_applied = config.dirichlet_epsilon > 0
    ok = _expand_node(
        root,
        policy_callable=policy_callable,
        prior_callable=prior_callable,
        config=config,
        rng=rng,
        apply_dirichlet=dirichlet_applied,
    )
    if not ok or not root.children:
        raise RuntimeError("MCTS root expansion produced no children")
    total_sims_run = 0
    for _ in range(config.num_sims):
        path = _select_path(root, config.min_depth, config.c_puct)
        leaf = path[-1]
        if leaf.is_terminal:
            v = _leaf_value(leaf, value_callable)
            _backup(path, v)
            total_sims_run += 1
            continue
        if not leaf.children:
            _expand_node(
                leaf,
                policy_callable=policy_callable,
                prior_callable=prior_callable,
                config=config,
                rng=rng,
                apply_dirichlet=False,
            )
        v = _leaf_value(leaf, value_callable)
        _backup(path, v)
        total_sims_run += 1
    visit_counts = {k: root.children[k].visit_count for k in root.children}
    sorted_keys = sorted(root.children.keys())
    if config.temperature == 0.0:
        chosen_key = max(
            sorted_keys, key=lambda kk: root.children[kk].visit_count
        )
    else:
        counts = np.array(
            [root.children[kk].visit_count for kk in sorted_keys],
            dtype=np.float64,
        )
        temp = max(1e-6, float(config.temperature))
        w = np.power(counts, 1.0 / temp)
        s = w.sum()
        if s <= 0:
            p = np.ones(len(sorted_keys), dtype=np.float64) / len(sorted_keys)
        else:
            p = w / s
        idx = int(rng.choice(len(sorted_keys), p=p))
        chosen_key = sorted_keys[idx]
    chosen = root.children[chosen_key]
    plan = list(chosen.plan_actions or [])
    pv_depth = principal_variation_depth(root)
    wall = time.perf_counter() - t0
    stats: dict[str, Any] = {
        "visit_counts": visit_counts,
        "principal_variation_depth": pv_depth,
        "total_sims_run": total_sims_run,
        "wall_time_s": wall,
        "dirichlet_applied": dirichlet_applied,
    }
    return plan, stats


def make_callables_from_sb3_policy(model: Any, env: Any) -> tuple[
    Callable[[GameState], Action],
    Callable[[GameState], float],
    Callable[[GameState, list[list[Action]]], list[float]],
]:
    """
    Build (policy_callable, value_callable, prior_callable) for ``run_mcts``
    from a MaskablePPO-like ``model`` and :class:`rl.env.AWBWEnv`.

    The env's mask/encode path (including infantry-only BUILD stripping when
    ``AWBW_BUILD_MASK_INFANTRY_ONLY`` is set) is honoured via
    ``env.action_masks()`` after binding ``env.state``.
    """
    import copy

    import torch

    from rl.env import AWBWEnv, _action_to_flat, _flat_to_action

    if not isinstance(env, AWBWEnv):
        raise TypeError(f"env must be AWBWEnv, got {type(env)}")

    pol = model.policy
    device = model.device

    def _bind(s: GameState) -> None:
        env.state = s
        env._invalidate_legal_cache()

    def policy_callable(s: GameState) -> Action:
        _bind(s)
        obs = env._get_obs(observer=int(s.active_player))
        mask = env.action_masks()
        action, _ = model.predict(
            obs, action_masks=mask, deterministic=False
        )
        if isinstance(action, int | np.integer):
            idx = int(action)
        else:
            idx = int(np.asarray(action).reshape(-1)[0])
        act = _flat_to_action(idx, s, legal=env._get_legal())
        if act is None:
            legal = env._get_legal()
            if not legal:
                raise RuntimeError("policy_callable: no legal actions")
            return legal[0]
        return act

    def value_callable(s: GameState) -> float:
        _bind(s)
        obs = env._get_obs(observer=int(s.active_player))
        obs_t, _ = pol.obs_to_tensor(obs)
        if isinstance(obs_t, dict):
            obs_t = {k: v.to(device) for k, v in obs_t.items()}
        else:
            obs_t = obs_t.to(device)
        with torch.no_grad():
            vals = pol.predict_values(obs_t)
        v = float(vals.reshape(-1)[0].detach().cpu().numpy())
        return float(np.clip(v, -1.0, 1.0))

    def prior_callable(s: GameState, plans: list[list[Action]]) -> list[float]:
        if not plans:
            return []
        pol_logits: list[float] = []
        for plan in plans:
            st = copy.deepcopy(s)
            logp = 0.0
            for a in plan:
                if st.done:
                    break
                _bind(st)
                obs = env._get_obs(observer=int(st.active_player))
                mask = env.action_masks()
                obs_t, _ = pol.obs_to_tensor(obs)
                if isinstance(obs_t, dict):
                    obs_t = {k: v.to(device) for k, v in obs_t.items()}
                else:
                    obs_t = obs_t.to(device)
                m_arr = np.asarray(mask, dtype=bool)
                m_t = torch.as_tensor(
                    m_arr[np.newaxis, ...], dtype=torch.bool, device=device
                )
                with torch.no_grad():
                    dist = pol.get_distribution(obs_t, action_masks=m_t)
                    idx = int(_action_to_flat(a, st))
                    lp = dist.log_prob(
                        torch.tensor([idx], dtype=torch.long, device=device)
                    )
                logp += float(lp[0].detach().cpu().numpy())
                st, _r, d = st.step(a)
                if d:
                    break
            pol_logits.append(logp)
        logits_t = torch.as_tensor(pol_logits, dtype=torch.float64)
        logits_t = logits_t - logits_t.max()
        w = torch.exp(logits_t)
        ssum = float(w.sum().item())
        if ssum <= 0:
            n = len(plans)
            return [1.0 / n] * n
        return [float(x) / ssum for x in w.tolist()]

    return policy_callable, value_callable, prior_callable
