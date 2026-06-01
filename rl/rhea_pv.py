"""RHEA principal-variation multi-ply lookahead."""
from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from engine.search_clone import clone_for_search
from rl.rhea_fitness import RheaFitnessBreakdown

if TYPE_CHECKING:
    from engine.game import GameState
    from rl.rhea import RheaConfig, RheaGenome, RheaPlanner


@dataclass(slots=True)
class RheaPVConfig:
    """PV lookahead: leaf-value backup with accurate myopic opponent modeling."""

    enabled: bool = True
    root_width: int = 3
    root_protected_top: int = 3
    root_pool_width: int = 6
    response_width: int = 1
    followup_width: int = 1
    budget_fraction: float = 0.45
    inner_budget_scale: float = 0.45
    random_probes: int = 0
    response_discount: float = 0.85
    followup_discount: float = 0.70
    value_root_slots: int = 0
    value_root_pool_width: int = 8
    value_root_min_advantage: float = 0.04
    pv_min_switch_margin: float = 0.015
    pv_robust_noise_floor: float = 0.01
    pv_max_followup_pairs: int = 4
    iterative_deepening: bool = True


@dataclass(slots=True)
class TurnCandidate:
    actions: list
    state_after: Any
    score: float
    breakdown: RheaFitnessBreakdown
    illegal: int
    move_score: float = 0.0
    diversity_signature: tuple = ()
    source_key: tuple = ()
    is_value_guided: bool = False
    is_protected_base: bool = False
    backup_anchor: float | None = None




def genome_move_identity_key(genome: Any) -> tuple:
    unit_segment = getattr(genome, "unit_segment", None) or []
    build_segment = getattr(genome, "build_segment", None) or []
    # Identity must include the BUILD segment: two turn-plans that move the
    # same units but purchase different units are genuinely different turns,
    # and PV must be able to treat them as distinct alternates. Without this,
    # build-only opening turns (empty move segment) all collapse to one key
    # and the lookahead never has an alternative to switch to.
    build_key = tuple(
        sorted(
            (
                str(getattr(bi, "factory_pos", None)),
                str(getattr(bi, "unit_type", None)),
            )
            for bi in build_segment
        )
    )
    return (
        bool(getattr(genome, "cop_activate", False)),
        tuple(
            (
                getattr(ui, "unit_pos", None),
                getattr(ui, "move_dest", None),
                getattr(ui, "action_type", None),
                getattr(ui, "target_pos", None),
            )
            for ui in unit_segment
        ),
        build_key,
    )


def root_backup_anchor(root: TurnCandidate) -> float:
    if root.backup_anchor is not None:
        return float(root.backup_anchor)
    return float(root.score)


def _turn_hard_wall_breached(*, turn_started_at: float, hard_wall_s: float) -> bool:
    hw = float(hard_wall_s)
    if hw <= 0.0:
        return False
    return (time.perf_counter() - float(turn_started_at)) >= hw


def _pv_time_budget_s(*, turn_started_at: float, hard_wall_s: float, budget_fraction: float) -> float:
    if hard_wall_s <= 0.0:
        return 1e9
    elapsed = time.perf_counter() - float(turn_started_at)
    remaining = max(0.0, float(hard_wall_s) - elapsed)
    return max(0.0, remaining * max(0.0, min(1.0, float(budget_fraction))))


def decision_point_seed(game_seed: int, day: int, seat: int) -> int:
    """Stable per-decision seed for paired myopic base search."""
    x = int(game_seed) & 0xFFFFFFFF
    x ^= (int(day) + 0x9E3779B9) & 0xFFFFFFFF
    x = (x ^ (x >> 16)) * 0x85EBCA6B & 0xFFFFFFFF
    x ^= (int(seat) + 0xC2B2AE35) & 0xFFFFFFFF
    x = (x ^ (x >> 16)) * 0x85EBCA6B & 0xFFFFFFFF
    return int(x & 0x7FFFFFFF)


def leaf_value_at_state(*, fitness, state, root_actor: int) -> float:
    # The raw value net is saturated on this matchup (near-constant win-prob),
    # so a pure-value leaf cannot distinguish materially different PV lines and
    # the lookahead can never justify a switch. Mirror RHEA's own objective
    # (value_weight*value + reward_weight*Phi) at the leaf so deeper lines are
    # ranked by the same discriminating signal RHEA optimizes per-turn.
    seat = int(root_actor)
    v = float(fitness.value(state, seat))
    term = float(fitness.engine_terminal_sparse(state, seat))
    if term != 0.0:
        return v + term
    try:
        phi = float(fitness.phi(state, seat))
    except Exception:
        phi = 0.0
    rw = float(getattr(fitness, "reward_weight", 0.9))
    vw = float(getattr(fitness, "value_weight", 0.1))
    return vw * v + rw * phi


def robust_effective_margin(*, switch_margin: float, noise_floor: float) -> float:
    return max(float(switch_margin), float(noise_floor))


def evaluate_pv_line(
    planner: "RheaPlanner",
    *,
    root: TurnCandidate,
    root_actor: int,
    followup_pairs: int,
    turn_started_at: float,
    hard_wall_s: float,
    deadline_at: float,
) -> tuple[float, bool]:
    """Leaf value after root plus followup_pairs of (opp myopic, our myopic) plies.

    Returns ``(leaf_value, completed)``. ``completed`` is ``False`` when the line
    could not be played out to the requested depth (deadline hit or a myopic
    sub-search failed), so the caller can discard the over-optimistic partial
    leaf instead of switching on it.
    """
    state = clone_for_search(root.state_after)
    fitness = planner.fitness
    if state.winner is not None:
        return leaf_value_at_state(fitness=fitness, state=state, root_actor=root_actor), True
    plies = max(0, int(followup_pairs)) * 2
    completed = True
    for _ in range(plies):
        if state.winner is not None:
            break
        if time.perf_counter() >= float(deadline_at):
            completed = False
            break
        acting = int(state.active_player)
        cand = planner._simulate_myopic_turn_candidate(
            state,
            acting,
            turn_started_at=turn_started_at,
            hard_wall_s=hard_wall_s,
            deadline_at=deadline_at,
        )
        if cand is None:
            completed = False
            break
        state = cand.state_after
    return leaf_value_at_state(fitness=fitness, state=state, root_actor=root_actor), completed


def backed_up_root_score(
    *,
    root: TurnCandidate,
    opponent_responses: list[tuple[TurnCandidate, TurnCandidate | None]],
    root_actor: int,
    fitness,
    response_discount: float = 0.85,
    followup_discount: float = 0.70,
) -> float:
    del opponent_responses, response_discount, followup_discount
    return leaf_value_at_state(fitness=fitness, state=root.state_after, root_actor=root_actor)


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _safe_funds(state: Any, seat: int) -> int:
    funds = getattr(state, "funds", None)
    try:
        if funds is None:
            return 0
        if int(seat) < 0 or int(seat) >= len(funds):
            return 0
        return _safe_int(funds[seat], 0)
    except (TypeError, ValueError, IndexError):
        return 0


def _property_positions(state: Any, seat: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for p in getattr(state, "properties", []):
        if _safe_int(getattr(p, "owner", -1), -1) != int(seat):
            continue
        row = _safe_int(getattr(p, "row", None), -1)
        col = _safe_int(getattr(p, "col", None), -1)
        if row >= 0 and col >= 0:
            out.add((row, col))
    return out


def _unit_summary(state: Any, seat: int) -> tuple[tuple[str, int], ...]:
    counts: dict[str, int] = {}
    for unit in getattr(state, "units", []):
        if _safe_int(getattr(unit, "player", -1), -1) != int(seat):
            continue
        name = str(getattr(getattr(unit, "unit_type", None), "name", getattr(unit, "unit_type", "")))
        counts[name] = counts.get(name, 0) + 1
    return tuple(sorted(counts.items()))


def make_diversity_signature(before: Any, cand: TurnCandidate, seat: int) -> tuple:
    """Coarse strategic signature for avoiding near-duplicate PV root branches."""

    after = cand.state_after
    before_props = _property_positions(before, seat)
    after_props = _property_positions(after, seat)
    props_gained = len(after_props - before_props)
    props_lost = len(before_props - after_props)
    funds_before = _safe_funds(before, seat)
    funds_after = _safe_funds(after, seat)
    spent_bucket = max(0, funds_before - funds_after) // 5000
    action_names = [str(getattr(getattr(a, "action_type", None), "name", "")) for a in cand.actions]
    builds = tuple(
        sorted(
            str(getattr(getattr(a, "unit_type", None), "name", getattr(a, "unit_type", "")))
            for a in cand.actions
            if str(getattr(getattr(a, "action_type", None), "name", "")) == "BUILD"
        )
    )
    return (
        props_gained,
        props_lost,
        action_names.count("CAPTURE"),
        action_names.count("ATTACK"),
        action_names.count("JOIN"),
        action_names.count("LOAD"),
        bool(action_names.count("ACTIVATE_COP") or action_names.count("ACTIVATE_SCOP")),
        spent_bucket,
        builds,
        _unit_summary(after, seat),
    )


def _signature_distance(a: tuple, b: tuple) -> int:
    total = 0
    for av, bv in zip(a, b, strict=False):
        if av == bv:
            continue
        if isinstance(av, int) and isinstance(bv, int):
            total += min(3, abs(int(av) - int(bv)))
        else:
            total += 1
    total += abs(len(a) - len(b))
    return total


def _contains_identity(items: list[TurnCandidate], cand: TurnCandidate) -> bool:
    return any(item is cand for item in items)


def select_top_roots(candidates: list[TurnCandidate], *, width: int) -> list[TurnCandidate]:
    if width <= 0:
        return []
    ordered = sorted(candidates, key=lambda c: (float(c.score), float(c.move_score)), reverse=True)
    return list(ordered[:width])




def select_pv_root_after_backup(
    *,
    backed_by_root: list[tuple[TurnCandidate, float]],
    protected_base: TurnCandidate | None,
    base_anchor_score: float,
    switch_margin: float,
    protected_move_key: tuple | None = None,
    effective_margin: float | None = None,
) -> tuple[TurnCandidate, float, bool, float, float]:
    if effective_margin is None:
        effective_margin = float(switch_margin)
    if not backed_by_root:
        raise ValueError("backed_by_root must not be empty")

    protected_root: TurnCandidate | None = None
    protected_backed = -1e30
    best_alternate: TurnCandidate | None = None
    best_alternate_backed = -1e30

    if protected_base is not None:
        protected_root = protected_base
        for root, backed in backed_by_root:
            if root is protected_base or getattr(root, "is_protected_base", False):
                protected_backed = float(backed)
                protected_root = root
                break
        if protected_backed <= -1e29:
            protected_backed = float(getattr(protected_base, "score", 0.0))

    for root, backed in backed_by_root:
        if protected_root is not None and root is protected_root:
            continue
        if getattr(root, "is_protected_base", False):
            if protected_backed <= -1e29:
                protected_root = root
                protected_backed = float(backed)
            continue
        move_key = getattr(root, "source_key", None) or ()
        if protected_move_key and move_key == protected_move_key:
            continue
        if float(backed) > best_alternate_backed:
            best_alternate_backed = float(backed)
            best_alternate = root

    if protected_root is not None:
        threshold = protected_backed + float(effective_margin)
    else:
        threshold = float(base_anchor_score) + float(effective_margin)
    if best_alternate is not None and best_alternate_backed > threshold:
        return best_alternate, best_alternate_backed, False, best_alternate_backed, protected_backed

    if protected_root is not None:
        return protected_root, protected_backed, True, best_alternate_backed, protected_backed

    if best_alternate is not None:
        return best_alternate, best_alternate_backed, bool(getattr(best_alternate, "is_protected_base", False)), best_alternate_backed, protected_backed

    root, backed = backed_by_root[0]
    return root, float(backed), bool(getattr(root, "is_protected_base", False)), best_alternate_backed, protected_backed


def select_diverse_roots(
    candidates: list[TurnCandidate],
    *,
    width: int,
    protected_top: int,
) -> list[TurnCandidate]:
    if width <= 0:
        return []
    ordered = sorted(candidates, key=lambda c: (float(c.score), float(c.move_score)), reverse=True)
    selected = list(ordered[: min(max(0, protected_top), width)])
    for cand in ordered:
        if _contains_identity(selected, cand):
            continue
        if len(selected) >= width:
            break
        min_dist = min(
            (_signature_distance(cand.diversity_signature, other.diversity_signature) for other in selected),
            default=999,
        )
        if min_dist >= 2:
            selected.append(cand)
    for cand in ordered:
        if len(selected) >= width:
            break
        if not _contains_identity(selected, cand):
            selected.append(cand)
    return selected

def run_rhea_pv_lookahead(
    planner: "RheaPlanner",
    *,
    before: "GameState",
    root_actor: int,
    v_before: float,
    root_move_genomes: list[tuple[float, "RheaGenome"]],
    value_move_genomes: list[tuple[float, "RheaGenome"]] | None = None,
    protected_base: TurnCandidate | None = None,
    turn_started_at: float,
    hard_wall_s: float,
) -> dict | None:
    pv = planner.cfg.pv_config
    if not planner.cfg.use_pv_lookahead or not pv.enabled:
        return None
    if _turn_hard_wall_breached(turn_started_at=turn_started_at, hard_wall_s=hard_wall_s):
        return None

    t0 = time.perf_counter()
    pv_budget_s = _pv_time_budget_s(
        turn_started_at=turn_started_at,
        hard_wall_s=hard_wall_s,
        budget_fraction=pv.budget_fraction,
    )
    if pv_budget_s <= 0.0:
        return None

    width_root = max(1, int(pv.root_width))
    root_pool_width = max(width_root, int(pv.root_pool_width))
    protected_top = max(0, min(width_root, int(pv.root_protected_top)))
    width_resp = max(1, int(pv.response_width))
    width_follow = max(1, int(pv.followup_width))
    use_diverse_roots = root_pool_width > width_root or protected_top < width_root

    root_genomes = list(root_move_genomes[:root_pool_width])
    if not root_genomes:
        return None

    root_candidates: list[TurnCandidate] = []
    nodes = 0
    stop_reason = "complete"

    for move_score, genome in root_genomes:
        if time.perf_counter() - t0 >= pv_budget_s:
            stop_reason = "pv_time"
            break
        if _turn_hard_wall_breached(turn_started_at=turn_started_at, hard_wall_s=hard_wall_s):
            stop_reason = "turn_wall"
            break
        cand = planner._complete_turn_from_move_genome(
            before,
            genome,
            root_actor,
            v_before,
            turn_started_at=turn_started_at,
            hard_wall_s=hard_wall_s,
        )
        if cand is None:
            continue
        cand.move_score = float(move_score)
        cand.source_key = genome_move_identity_key(genome)
        if use_diverse_roots:
            cand.diversity_signature = make_diversity_signature(before, cand, root_actor)
        root_candidates.append(cand)
        nodes += 1

    if protected_base is not None:
        pb = protected_base
        pb.is_protected_base = True
        pb.source_key = pb.source_key or ("protected_base",)
        seen_pb = {c.source_key for c in root_candidates if c.source_key}
        if pb.source_key not in seen_pb:
            root_candidates.append(pb)
            nodes += 1

    if not root_candidates:
        return None
    if use_diverse_roots:
        root_candidates = select_diverse_roots(
            root_candidates,
            width=width_root,
            protected_top=protected_top,
        )
    else:
        root_candidates = select_top_roots(root_candidates, width=width_root)

    if protected_base is not None:
        pb = protected_base
        pb.is_protected_base = True
        if not any(c is pb for c in root_candidates):
            root_candidates.append(pb)

    value_slots = max(0, int(getattr(pv, "value_root_slots", 0) or 0))
    value_pool_width = max(value_slots, int(getattr(pv, "value_root_pool_width", 8) or 8))
    value_roots_added = 0
    if value_slots > 0:
        seen_keys = {c.source_key for c in root_candidates if c.source_key}
        min_reward_value_adv = min(
            (float(c.breakdown.value) for c in root_candidates),
            default=-1e30,
        )
        value_min_adv = float(getattr(pv, "value_root_min_advantage", 0.04) or 0.04)
        for move_score, genome in list(value_move_genomes or [])[:value_pool_width]:
            if value_roots_added >= value_slots:
                break
            if time.perf_counter() - t0 >= pv_budget_s:
                stop_reason = "pv_time"
                break
            if _turn_hard_wall_breached(turn_started_at=turn_started_at, hard_wall_s=hard_wall_s):
                stop_reason = "turn_wall"
                break
            key = genome_move_identity_key(genome)
            if key in seen_keys:
                continue
            if float(move_score) < float(min_reward_value_adv) + value_min_adv:
                continue
            cand = planner._complete_turn_from_move_genome(
                before,
                genome,
                root_actor,
                v_before,
                turn_started_at=turn_started_at,
                hard_wall_s=hard_wall_s,
            )
            if cand is None:
                continue
            cand.move_score = float(move_score)
            cand.source_key = key
            cand.is_value_guided = True
            cand.backup_anchor = float(cand.breakdown.value)
            root_candidates.append(cand)
            seen_keys.add(key)
            value_roots_added += 1
            nodes += 1

    deadline_at = t0 + pv_budget_s
    effective_margin = robust_effective_margin(
        switch_margin=float(getattr(pv, "pv_min_switch_margin", 0.015) or 0.015),
        noise_floor=float(getattr(pv, "pv_robust_noise_floor", 0.01) or 0.01),
    )
    max_pairs = max(0, int(getattr(pv, "pv_max_followup_pairs", 4) or 4))
    iterative = bool(getattr(pv, "iterative_deepening", True))
    pair_depths = list(range(0, max_pairs + 1)) if iterative else [1]

    # Minimum lookahead depth (in opp+our pairs) required before we trust an
    # override. Depth 0 evaluates the leaf right after our own move with no
    # opponent response, which the value net rates over-optimistically; such a
    # leaf must never trigger a switch.
    min_switch_pairs = 1

    best_root = None
    best_backed = -1e30
    pv_best_is_protected_base = True
    alternate_backed = -1e30
    protected_backed = -1e30
    chosen_pairs = 0
    have_verified_selection = False

    for followup_pairs in pair_depths:
        if time.perf_counter() >= deadline_at:
            stop_reason = "pv_time"
            break
        if _turn_hard_wall_breached(turn_started_at=turn_started_at, hard_wall_s=hard_wall_s):
            stop_reason = "turn_wall"
            break

        # Evaluate every root to the SAME full depth. If any root cannot be
        # completed within budget, the whole depth level is discarded and we
        # keep the last fully-verified depth (anytime-safe, apples-to-apples).
        backed_by_root: list[tuple[TurnCandidate, float]] = []
        depth_complete = True
        for root in root_candidates:
            if time.perf_counter() >= deadline_at:
                depth_complete = False
                stop_reason = "pv_time"
                break
            leaf, line_complete = evaluate_pv_line(
                planner,
                root=root,
                root_actor=root_actor,
                followup_pairs=followup_pairs,
                turn_started_at=turn_started_at,
                hard_wall_s=hard_wall_s,
                deadline_at=deadline_at,
            )
            nodes += 1 + max(0, followup_pairs) * 2
            if not line_complete:
                depth_complete = False
                stop_reason = "pv_time"
                break
            backed_by_root.append((root, float(leaf)))

        if not depth_complete or len(backed_by_root) < len(root_candidates):
            # Partial depth: abandon it and keep the last verified selection.
            break

        base_anchor_score = (
            float(protected_base.score)
            if protected_base is not None
            else max((float(score) for _, score in backed_by_root), default=-1e30)
        )
        protected_move_key = genome_move_identity_key(root_genomes[0][1]) if root_genomes else None
        cand_root, cand_backed, cand_protected, cand_alt, cand_prot = select_pv_root_after_backup(
            backed_by_root=backed_by_root,
            protected_base=protected_base,
            base_anchor_score=base_anchor_score,
            switch_margin=float(getattr(pv, "pv_min_switch_margin", 0.015) or 0.015),
            protected_move_key=protected_move_key,
            effective_margin=effective_margin,
        )
        prot_backed = float(cand_prot)
        if protected_base is not None and prot_backed <= -1e29:
            for root, backed in backed_by_root:
                if root is protected_base or getattr(root, "is_protected_base", False):
                    prot_backed = float(backed)
                    break

        # Below the minimum opponent-response depth, never accept an override:
        # fall back to the protected base so a no-response leaf cannot win.
        if followup_pairs < min_switch_pairs and not cand_protected:
            if protected_base is not None:
                cand_root = protected_base
                cand_backed = prot_backed
                cand_protected = True

        chosen_pairs = followup_pairs
        best_root = cand_root
        best_backed = float(cand_backed)
        pv_best_is_protected_base = bool(cand_protected)
        alternate_backed = float(cand_alt)
        protected_backed = prot_backed
        have_verified_selection = True

    if best_root is None or not have_verified_selection:
        return None

    wall_s = time.perf_counter() - t0
    base_anchor_score = float(protected_base.score) if protected_base is not None else float(best_backed)
    base_leaf_out = None if protected_backed <= -1e29 else float(protected_backed)
    alt_leaf_out = None if alternate_backed <= -1e29 else float(alternate_backed)
    return {
        "source": "rhea_pv",
        "score": float(best_backed),
        "actions": list(best_root.actions),
        "breakdown": best_root.breakdown,
        "illegal": int(best_root.illegal),
        "pv_nodes_evaluated": int(nodes),
        "pv_root_width": width_root,
        "pv_value_roots": int(value_roots_added),
        "pv_reward_roots": int(len(root_candidates) - value_roots_added),
        "pv_response_width": width_resp,
        "pv_followup_width": width_follow,
        "pv_depth_pairs": int(chosen_pairs),
        "pv_robust_margin": float(effective_margin),
        "pv_base_leaf_score": base_leaf_out,
        "pv_alternate_leaf_score": alt_leaf_out,
        "pv_wall_s": float(wall_s),
        "pv_stop_reason": stop_reason,
        "pv_backed_score": float(best_backed),
        "pv_alternate_backed_score": float(alternate_backed),
        "pv_protected_backed_score": float(protected_backed),
        "pv_base_anchor_score": float(base_anchor_score),
        "pv_best_value_guided": bool(getattr(best_root, "is_value_guided", False)),
        "pv_best_is_protected_base": bool(pv_best_is_protected_base),
    }