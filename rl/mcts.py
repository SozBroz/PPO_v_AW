"""
Phase 11b/11f: single-process PUCT MCTS at turn boundaries.

Uses ``GameState.apply_full_turn`` as the turn rollout primitive.  The search is
still intentionally turn-level (not sub-action tree search), but it now carries
root-edge stochastic risk statistics for AWBW combat luck: fixed-plan luck
resampling, edge variance/tail stats, and production-oriented root selection.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from engine.action import Action, ActionStage
from engine.game import GameState


@dataclass(slots=True)
class EdgeStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min_value: float = 1.0
    max_value: float = -1.0
    root_value_samples: list[float] = field(default_factory=list)
    trace_count: int = 0
    critical_threshold_count: int = 0
    defender_kill_count: int = 0
    capture_interrupted_count: int = 0
    attacker_death_count: int = 0

    def update(self, value: float) -> None:
        v = float(np.clip(value, -1.0, 1.0))
        self.count += 1
        delta = v - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (v - self.mean)
        self.min_value = min(self.min_value, v)
        self.max_value = max(self.max_value, v)

    @property
    def variance(self) -> float:
        return 0.0 if self.count < 2 else float(self.m2 / (self.count - 1))

    def add_root_sample(self, value: float, trace: list[dict[str, Any]]) -> None:
        v = float(np.clip(value, -1.0, 1.0))
        self.root_value_samples.append(v)
        self.trace_count += 1
        if any(bool(t.get("critical_threshold_event")) for t in trace):
            self.critical_threshold_count += 1
        if any(bool(t.get("defender_killed")) for t in trace):
            self.defender_kill_count += 1
        if any(bool(t.get("capture_interrupted")) for t in trace):
            self.capture_interrupted_count += 1
        if any(bool(t.get("attacker_killed")) for t in trace):
            self.attacker_death_count += 1

    def risk_summary(self, *, catastrophe_value: float = -0.35) -> dict[str, Any]:
        samples = list(self.root_value_samples)
        if samples:
            arr = np.asarray(samples, dtype=np.float64)
            mean = float(arr.mean())
            variance = float(arr.var(ddof=1)) if len(arr) > 1 else 0.0
            p10 = float(np.percentile(arr, 10))
            worst = float(arr.min())
            catastrophe_probability = float(np.mean(arr <= float(catastrophe_value)))
        else:
            mean = float(self.mean)
            variance = float(self.variance)
            p10 = mean
            worst = mean
            catastrophe_probability = 0.0
        denom = max(1, self.trace_count)
        return {
            "backup_count": int(self.count),
            "backup_mean": float(self.mean),
            "backup_variance": float(self.variance),
            "resample_count": int(len(samples)),
            "resample_mean": mean,
            "resample_variance": variance,
            "p10_value": p10,
            "worst_value": worst,
            "catastrophe_probability": catastrophe_probability,
            "critical_threshold_probability": float(self.critical_threshold_count / denom),
            "kill_probability": float(self.defender_kill_count / denom),
            "capture_interrupted_probability": float(self.capture_interrupted_count / denom),
            "attacker_death_probability": float(self.attacker_death_count / denom),
        }


@dataclass(slots=True)
class TurnNode:
    state: GameState
    actor: int
    visit_count: int = 0
    total_value: float = 0.0
    prior: float = 0.0
    children: dict[bytes, "TurnNode"] = field(default_factory=dict)
    plan_actions: list[Action] | None = None
    is_terminal: bool = False
    terminal_value: float = 0.0
    edge_stats: EdgeStats = field(default_factory=EdgeStats)


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
    luck_resamples: int = 0
    luck_resample_critical_only: bool = True
    risk_mode: str = "visit"  # visit | mean | mean_minus_p10 | constrained
    risk_lambda: float = 0.35
    catastrophe_value: float = -0.35
    max_catastrophe_prob: float = 1.0
    root_decision_log_path: str | None = None
    # MASTERPLAN §14 — optional: set by rl.mcts_rollout_stages presets (telemetry)
    rollout_stage: str | None = None
    # MCTS-4: if set, stop the main sim loop when wall time is reached (still capped by num_sims)
    max_wall_time_s: float | None = None
    # MCTS-2: fraction of P0 SELECT steps that run MCTS (1.0 = always; used by symmetric eval)
    p0_mcts_invocation_fraction: float = 1.0
    # Brute-force optimization: if estimated branching factor is <= this threshold, enumerate all plans
    brute_force_branching_threshold: int = 0  # 0 = disabled


def plan_key(final_state: GameState, actions: list[Action]) -> bytes:
    """Fast plan key — non-cryptographic, per-process deterministic.

    Replaces the original SHA-256 + repr() approach which was 10-50x slower.
    Collisions are acceptable here: they merely merge two nodes' statistics.
    """
    # Build hashable tuples directly — no repr()/encode() overhead.
    parts = tuple(
        (
            int(a.action_type),
            a.unit_pos,
            a.move_pos,
            a.target_pos,
            int(a.unit_type) if a.unit_type is not None else None,
            a.unload_pos,
            a.select_unit_id,
        )
        for a in actions
    )
    prop_bits = tuple(
        sorted(((p.row, p.col), p.owner, p.capture_points) for p in final_state.properties)
    )
    h = hash((parts, prop_bits))
    # Convert to 8 bytes.  to_bytes needs unsigned; we XOR to force positive.
    return (h & 0xFFFFFFFFFFFFFFFF).to_bytes(8, byteorder="little")


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


def _leaf_value(node: TurnNode, value_callable: Callable[[GameState], float]) -> float:
    if node.is_terminal:
        return float(node.terminal_value)
    st = node.state
    if st.done and st.winner is not None:
        return float(np.clip(_terminal_utility(st, node.actor), -1.0, 1.0))
    return float(np.clip(value_callable(st), -1.0, 1.0))


def _state_value_for_actor(st: GameState, *, actor: int, value_callable: Callable[[GameState], float]) -> float:
    if st.done and st.winner is not None:
        return float(np.clip(_terminal_utility(st, actor), -1.0, 1.0))
    v = float(np.clip(value_callable(st), -1.0, 1.0))
    if int(st.active_player) != int(actor):
        v = -v
    return float(np.clip(v, -1.0, 1.0))


def _child_q_for_parent(parent: TurnNode, child: TurnNode) -> float:
    if child.visit_count <= 0:
        return 0.0
    q = child.total_value / child.visit_count
    return -q if child.actor != parent.actor else q


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


def _unpack_full_turn_result(result: Any) -> tuple[GameState, list[Action], float, bool, list[dict[str, Any]]]:
    if not isinstance(result, tuple):
        raise TypeError(f"apply_full_turn returned non-tuple: {type(result)}")
    if len(result) == 4:
        st, actions, reward, done = result
        return st, actions, float(reward), bool(done), []
    if len(result) == 5:
        st, actions, reward, done, trace = result
        return st, actions, float(reward), bool(done), list(trace or [])
    raise ValueError(f"apply_full_turn returned {len(result)} fields; expected 4 or 5")


def _sample_plans(node: TurnNode, policy_callable: Callable[[GameState], Action], *, k: int, max_plan_actions: int, rng: np.random.Generator) -> list[tuple[bytes, list[Action], GameState, bool, list[dict[str, Any]]]]:
    out: list[tuple[bytes, list[Action], GameState, bool, list[dict[str, Any]]]] = []
    seen: set[bytes] = set()
    attempts = 0
    max_attempts = max(k * 8, k + 16)
    while len(out) < k and attempts < max_attempts:
        attempts += 1
        seed = int(rng.integers(0, 2**31 - 1))
        result = node.state.apply_full_turn(policy_callable, copy=True, max_actions=max_plan_actions, rng_seed=seed, return_trace=True)
        final_st, actions, _r, done, trace = _unpack_full_turn_result(result)
        key = plan_key(final_st, actions)
        if key in seen:
            continue
        seen.add(key)
        out.append((key, actions, final_st, done, trace))
    return out


def _expand_node(node: TurnNode, *, policy_callable: Callable[[GameState], Action], prior_callable: Callable[[GameState, list[list[Action]]], list[float]], config: MCTSConfig, rng: np.random.Generator, apply_dirichlet: bool) -> bool:
    if node.is_terminal or node.children:
        return False
    sampled = _sample_plans(node, policy_callable, k=config.root_plans, max_plan_actions=config.max_plan_actions, rng=rng)
    if not sampled:
        return False
    plans = [t[1] for t in sampled]
    priors = _normalize_priors(prior_callable(node.state, plans), len(plans))
    if apply_dirichlet and config.dirichlet_epsilon > 0 and len(priors) > 0:
        noise = rng.dirichlet(np.full(len(priors), config.dirichlet_alpha))
        eps = config.dirichlet_epsilon
        priors = (1.0 - eps) * np.asarray(priors, dtype=np.float64) + eps * noise
        priors = (priors / priors.sum()).tolist()
    for (key, plan, st, done, trace), prior in zip(sampled, priors, strict=True):
        terminal = bool(done or st.done)
        child = TurnNode(state=st, actor=int(st.active_player), prior=float(prior), plan_actions=plan)
        if trace:
            child.edge_stats.add_root_sample(0.0, trace)
            child.edge_stats.root_value_samples.clear()
        if terminal:
            child.is_terminal = True
            w = st.winner
            child.terminal_value = 0.0 if w is None or w == -1 else (1.0 if w == child.actor else -1.0)
        node.children[key] = child
    return True


def _select_path(root: TurnNode, min_depth: int, c_puct: float) -> list[TurnNode]:
    path: list[TurnNode] = [root]
    node = root
    while True:
        if node.is_terminal or not node.children:
            break
        depth = len(path) - 1
        key = _prior_greedy_key(node) if depth < min_depth else _puct_select_key(node, c_puct)
        node = node.children[key]
        path.append(node)
    return path


def _backup(path: list[TurnNode], v_leaf: float) -> None:
    v = float(v_leaf)
    for i in range(len(path) - 1, -1, -1):
        node = path[i]
        node.visit_count += 1
        node.total_value += v
        node.edge_stats.update(v)
        if i == 0:
            break
        parent = path[i - 1]
        if node.actor != parent.actor:
            v = -v


def principal_variation_depth(root: TurnNode) -> int:
    depth = 0
    node = root
    while node.children:
        best_k = max(node.children.keys(), key=lambda kk: node.children[kk].visit_count)
        node = node.children[best_k]
        depth += 1
    return depth


def _trace_has_critical_event(trace: list[dict[str, Any]]) -> bool:
    return any(bool(t.get("critical_threshold_event")) for t in trace)


def _root_child_value(root: TurnNode, child: TurnNode) -> float:
    if child.visit_count > 0:
        return _child_q_for_parent(root, child)
    if child.edge_stats.root_value_samples:
        return float(np.mean(child.edge_stats.root_value_samples))
    return 0.0


def _resample_root_children(root: TurnNode, *, value_callable: Callable[[GameState], float], config: MCTSConfig, rng: np.random.Generator) -> None:
    n = max(0, int(config.luck_resamples))
    if n <= 0:
        return
    for child in root.children.values():
        plan = list(child.plan_actions or [])
        if not plan:
            continue
        if config.luck_resample_critical_only:
            probe_seed = int(rng.integers(0, 2**31 - 1))
            result = root.state.apply_full_turn(plan, copy=True, max_actions=config.max_plan_actions, rng_seed=probe_seed, return_trace=True)
            st, _acts, _reward, _done, trace = _unpack_full_turn_result(result)
            root_v = _state_value_for_actor(st, actor=root.actor, value_callable=value_callable)
            child.edge_stats.add_root_sample(root_v, trace)
            if not _trace_has_critical_event(trace):
                continue
            extra = max(0, n - 1)
        else:
            extra = n
        for _ in range(extra):
            seed = int(rng.integers(0, 2**31 - 1))
            result = root.state.apply_full_turn(plan, copy=True, max_actions=config.max_plan_actions, rng_seed=seed, return_trace=True)
            st, _acts, _reward, _done, trace = _unpack_full_turn_result(result)
            root_v = _state_value_for_actor(st, actor=root.actor, value_callable=value_callable)
            child.edge_stats.add_root_sample(root_v, trace)


def _root_child_selection_score(root: TurnNode, child: TurnNode, config: MCTSConfig) -> float:
    mode = str(config.risk_mode or "visit").strip().lower()
    summary = child.edge_stats.risk_summary(catastrophe_value=config.catastrophe_value)
    mean = float(summary["resample_mean"])
    p10 = float(summary["p10_value"])
    catastrophe_prob = float(summary["catastrophe_probability"])
    if mode == "visit":
        return float(child.visit_count) + 1e-6 * _root_child_value(root, child)
    if mode == "mean":
        return mean if child.edge_stats.root_value_samples else _root_child_value(root, child)
    if mode == "mean_minus_p10":
        downside = max(0.0, mean - p10)
        return mean - float(config.risk_lambda) * downside
    if mode == "constrained":
        if catastrophe_prob > float(config.max_catastrophe_prob):
            return -1e9 + mean
        downside = max(0.0, mean - p10)
        return mean - float(config.risk_lambda) * downside
    raise ValueError(f"unknown MCTS risk_mode={config.risk_mode!r}")


def _root_visit_entropy(root: TurnNode) -> float:
    counts = np.asarray([c.visit_count for c in root.children.values()], dtype=np.float64)
    s = counts.sum()
    if s <= 0:
        return 0.0
    p = counts / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _serialise_root_children(root: TurnNode, config: MCTSConfig) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for k, ch in root.children.items():
        q = _child_q_for_parent(root, ch) if ch.visit_count > 0 else 0.0
        out.append({
            "key": k.hex()[:16],
            "visits": int(ch.visit_count),
            "prior": float(ch.prior),
            "q_root_frame": float(q),
            "risk_score": float(_root_child_selection_score(root, ch, config)),
            "plan_len": len(ch.plan_actions or []),
            "risk": ch.edge_stats.risk_summary(catastrophe_value=config.catastrophe_value),
        })
    out.sort(key=lambda x: (int(x["visits"]), float(x["risk_score"])), reverse=True)
    return out


def _append_decision_log(path: str | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def decision_log_context_from_env(env: Any) -> dict[str, Any]:
    """
    Build per-root-decision context aligned with finished-game ``game_log.jsonl`` (schema ≥1.8)
    so MCTS root JSONL and ``chosen_risk`` / resample telemetry can be sliced by curriculum
    and map/matchup like episode rows.

    While the episode is ongoing, ``truncated`` is false and ``truncation_reason`` is null
    (same keys as the episode record for consistent joins and filters).
    """
    ei = getattr(env, "_episode_info", None) or {}
    st = getattr(env, "state", None)
    tag = getattr(env, "curriculum_tag", None)
    return {
        "curriculum_tag": tag,
        "map_id": ei.get("map_id"),
        "tier": ei.get("tier"),
        "p0_co_id": ei.get("p0_co"),
        "p1_co_id": ei.get("p1_co"),
        "p0_env_steps": int(getattr(env, "_p0_env_steps", 0)),
        "turn": int(st.turn) if st is not None else None,
        "truncated": False,
        "truncation_reason": None,
    }


def _estimate_branching_factor(state: GameState) -> int:
    """Estimate the branching factor (number of legal actions) at the current step.
    
    Note: This is a first-step estimate only. A full turn in AWBW can have
    many steps (move, attack, capture, etc.) with compounding branching.
    The brute-force path is designed for cases where even the first step
    has very few options (threshold <= 5-10).
    """
    from engine.action import get_legal_actions
    legal = get_legal_actions(state)
    return len(legal)


def _brute_force_best_plan(
    root_state: GameState,
    *,
    policy_callable: Callable[[GameState], Action],
    value_callable: Callable[[GameState], float],
    config: MCTSConfig,
    rng: np.random.Generator,
) -> tuple[list[Action], dict[str, Any]]:
    """Brute-force enumeration of all possible full-turn plans when branching is tiny.
    
    This function enumerates all legal action sequences for a full turn by
    repeatedly sampling plans until we've seen all unique ones (up to a limit).
    It then scores each plan and returns the best one.
    """
    # Generate many plans to try to cover all possibilities
    max_attempts = 1000
    seen_plans: dict[bytes, tuple[list[Action], GameState, bool, list[dict[str, Any]]]] = {}
    attempts = 0
    
    while attempts < max_attempts:
        attempts += 1
        seed = int(rng.integers(0, 2**31 - 1))
        result = root_state.apply_full_turn(
            policy_callable, copy=True, max_actions=config.max_plan_actions,
            rng_seed=seed, return_trace=True
        )
        final_st, actions, _r, done, trace = _unpack_full_turn_result(result)
        key = plan_key(final_st, actions)
        
        if key not in seen_plans:
            seen_plans[key] = (actions, final_st, done, trace)
        
        # If we've found many unique plans, branching might not be tiny
        if len(seen_plans) > config.brute_force_branching_threshold * 2:
            break
    
    # Score each plan
    best_plan = None
    best_value = -float('inf')
    best_key = None
    
    for key, (actions, final_st, done, trace) in seen_plans.items():
        # Score the plan using the value function
        value = value_callable(final_st)
        if done:
            # Terminal state: convert winner to value
            w = final_st.winner
            if w is not None and w != -1:
                value = 1.0 if w == root_state.active_player else -1.0
        
        if value > best_value:
            best_value = value
            best_plan = actions
            best_key = key
    
    if best_plan is None:
        raise RuntimeError("Brute force enumeration found no plans")
    
    # Return best plan with basic stats
    stats: dict[str, Any] = {
        "brute_force_used": True,
        "total_plans_evaluated": len(seen_plans),
        "best_value": float(best_value),
        "best_key": best_key.hex()[:16] if best_key else None,
    }
    
    return best_plan, stats


def run_mcts(
    root_state: GameState,
    *,
    policy_callable: Callable[[GameState], Action],
    value_callable: Callable[[GameState], float],
    prior_callable: Callable[[GameState, list[list[Action]]], list[float]],
    config: MCTSConfig,
    decision_log_context: dict[str, Any] | None = None,
) -> tuple[list[Action], dict[str, Any]]:
    if root_state.action_stage != ActionStage.SELECT:
        raise ValueError("run_mcts requires root action_stage == SELECT")
    t0 = time.perf_counter()
    rng = np.random.default_rng(config.rng_seed)
    
    # Check if branching is tiny and we should use brute force
    if config.brute_force_branching_threshold > 0:
        branching_estimate = _estimate_branching_factor(root_state)
        if branching_estimate <= config.brute_force_branching_threshold:
            plan, stats = _brute_force_best_plan(
                root_state,
                policy_callable=policy_callable,
                value_callable=value_callable,
                config=config,
                rng=rng
            )
            stats["wall_time_s"] = time.perf_counter() - t0
            stats["branching_estimate"] = branching_estimate
            stats["brute_force_threshold"] = config.brute_force_branching_threshold
            if decision_log_context:
                stats["decision_log_context"] = dict(decision_log_context)
            return plan, stats
    
    # Otherwise, proceed with normal MCTS
    root = TurnNode(state=root_state, actor=int(root_state.active_player))
    dirichlet_applied = config.dirichlet_epsilon > 0
    ok = _expand_node(root, policy_callable=policy_callable, prior_callable=prior_callable, config=config, rng=rng, apply_dirichlet=dirichlet_applied)
    if not ok or not root.children:
        raise RuntimeError("MCTS root expansion produced no children")
    total_sims_run = 0
    sim_target = max(0, int(config.num_sims))
    wall_s = config.max_wall_time_s
    stop_reason: str = "sims"
    sidx = 0
    while sidx < sim_target:
        if wall_s is not None and float(wall_s) > 0.0 and (time.perf_counter() - t0) >= float(wall_s):
            stop_reason = "time"
            break
        path = _select_path(root, config.min_depth, config.c_puct)
        leaf = path[-1]
        if leaf.is_terminal:
            v = _leaf_value(leaf, value_callable)
            _backup(path, v)
            total_sims_run += 1
            sidx += 1
            continue
        if not leaf.children:
            _expand_node(leaf, policy_callable=policy_callable, prior_callable=prior_callable, config=config, rng=rng, apply_dirichlet=False)
        v = _leaf_value(leaf, value_callable)
        _backup(path, v)
        total_sims_run += 1
        sidx += 1

    _resample_root_children(root, value_callable=value_callable, config=config, rng=rng)

    visit_counts = {k: root.children[k].visit_count for k in root.children}
    sorted_keys = sorted(root.children.keys())
    risk_mode = str(config.risk_mode or "visit").strip().lower()
    if risk_mode != "visit":
        chosen_key = max(sorted_keys, key=lambda kk: _root_child_selection_score(root, root.children[kk], config))
    elif config.temperature == 0.0:
        chosen_key = max(sorted_keys, key=lambda kk: root.children[kk].visit_count)
    else:
        counts = np.array([root.children[kk].visit_count for kk in sorted_keys], dtype=np.float64)
        temp = max(1e-6, float(config.temperature))
        w = np.power(counts, 1.0 / temp)
        s = w.sum()
        p = np.ones(len(sorted_keys), dtype=np.float64) / len(sorted_keys) if s <= 0 else w / s
        idx = int(rng.choice(len(sorted_keys), p=p))
        chosen_key = sorted_keys[idx]
    chosen = root.children[chosen_key]
    plan = list(chosen.plan_actions or [])
    pv_depth = principal_variation_depth(root)
    wall = time.perf_counter() - t0
    root_child_stats = _serialise_root_children(root, config)
    chosen_risk = chosen.edge_stats.risk_summary(catastrophe_value=config.catastrophe_value)
    ctx = dict(decision_log_context) if decision_log_context else {}
    stats: dict[str, Any] = {
        "brute_force_used": False,
        "visit_counts": visit_counts,
        "principal_variation_depth": pv_depth,
        "total_sims_run": total_sims_run,
        "sim_target": int(sim_target),
        "mcts_stop_reason": str(stop_reason),
        "mcts_max_wall_time_s": None if config.max_wall_time_s is None else float(config.max_wall_time_s),
        "rollout_stage": config.rollout_stage,
        "wall_time_s": wall,
        "dirichlet_applied": dirichlet_applied,
        "root_visit_entropy": _root_visit_entropy(root),
        "risk_mode": risk_mode,
        "luck_resamples": int(config.luck_resamples),
        "chosen_key": chosen_key.hex()[:16],
        "chosen_visits": int(chosen.visit_count),
        "chosen_prior": float(chosen.prior),
        "chosen_q_root_frame": float(_child_q_for_parent(root, chosen)) if chosen.visit_count > 0 else 0.0,
        "chosen_risk_score": float(_root_child_selection_score(root, chosen, config)),
        "chosen_risk": chosen_risk,
        "root_child_stats": root_child_stats,
        "decision_log_context": ctx,
    }
    file_payload: dict[str, Any] = {
        "rollout_stage": config.rollout_stage,
        "wall_time_s": wall,
        "total_sims_run": total_sims_run,
        "root_children": len(root.children),
        "pv_depth": pv_depth,
        "root_visit_entropy": stats["root_visit_entropy"],
        "risk_mode": risk_mode,
        "chosen": {"key": stats["chosen_key"], "visits": stats["chosen_visits"], "prior": stats["chosen_prior"], "q_root_frame": stats["chosen_q_root_frame"], "risk_score": stats["chosen_risk_score"], "risk": chosen_risk},
        "root_child_stats": root_child_stats,
    }
    file_payload.update(ctx)
    _append_decision_log(config.root_decision_log_path, file_payload)
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
