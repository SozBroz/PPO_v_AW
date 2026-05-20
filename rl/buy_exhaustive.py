"""Deterministic capped DFS enumeration for two-phase RHEA buy phase."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from engine.action import Action, ActionStage, ActionType, get_legal_actions
from engine.game import GameState
from engine.search_clone import clone_for_search
from engine.unit import UNIT_STATS, UnitType
from rl.rhea_fitness import RheaFitness


def _ensure_select(sim: GameState) -> None:
    sim.action_stage = ActionStage.SELECT
    sim.selected_unit = None
    sim.selected_move_pos = None


def _genome_key(factory_order: tuple[tuple[int, int], ...], choices: list) -> tuple:
    return (factory_order, tuple(choices))


@dataclass(slots=True)
class BuyEnumerateResult:
    genomes: list
    truncated: bool
    frontier_depth_at_cap: int | None
    candidates_enumerated: int


@dataclass(slots=True)
class ScoredBuyCandidate:
    total: float
    gold_spent: int
    genome: object
    illegal: int
    build_actions: list
    sim_after: GameState


@dataclass(slots=True)
class ExhaustiveBuyPick:
    genome: object
    build_actions: list
    sim_after: GameState
    illegal: int
    candidates_enumerated: int
    candidates_scored: int
    truncated: bool
    frontier_depth_at_cap: int | None


def enumerate_buy_allocations(
    post_moves: GameState,
    factory_order: tuple[tuple[int, int], ...],
    catalog: dict[tuple[int, int], tuple[UnitType, ...]],
    acting: int,
    *,
    max_count: int,
) -> BuyEnumerateResult:
    """DFS over factory slots: SKIP first, then catalog order per factory."""
    from rl.rhea import BuySpendGenome

    cap = max(1, int(max_count))
    results: list[BuySpendGenome] = []
    seen: set[tuple] = set()
    truncated = False
    frontier_depth_at_cap: int | None = None

    def _append(choices: list[Optional[UnitType]]) -> bool:
        key = _genome_key(factory_order, choices)
        if key in seen:
            return False
        seen.add(key)
        results.append(BuySpendGenome(factory_order, list(choices)))
        if len(results) >= cap:
            return True
        return False

    def dfs(slot_i: int, sim: GameState, choices_so_far: list[Optional[UnitType]]) -> None:
        nonlocal truncated, frontier_depth_at_cap

        if len(results) >= cap:
            if not truncated:
                truncated = True
                frontier_depth_at_cap = slot_i
            return

        n = len(factory_order)
        if slot_i >= n:
            _append(choices_so_far)
            return

        if sim.winner is not None or int(sim.active_player) != acting:
            padded = list(choices_so_far) + [None] * (n - len(choices_so_far))
            _append(padded)
            return

        fac = factory_order[slot_i]
        _ensure_select(sim)
        legal = get_legal_actions(sim)
        coins = int(sim.funds[acting])

        # Branch 1: SKIP. Only terminal/full-length vectors are candidates;
        # scoring prefixes would silently treat missing slots as SKIP.
        choices_so_far.append(None)
        dfs(slot_i + 1, clone_for_search(sim), choices_so_far)
        choices_so_far.pop()
        if len(results) >= cap:
            if not truncated:
                truncated = True
                frontier_depth_at_cap = slot_i
            return

        # Branch 2: each affordable unit in catalog[fac]
        for ut in catalog.get(fac, ()):
            cost = int(UNIT_STATS[ut].cost)
            if cost > coins:
                continue
            sim2 = clone_for_search(sim)
            _ensure_select(sim2)
            legal2 = get_legal_actions(sim2)
            act: Action | None = None
            for a in legal2:
                if a.action_type != ActionType.BUILD:
                    continue
                if a.move_pos is None or a.unit_type is None:
                    continue
                if (int(a.move_pos[0]), int(a.move_pos[1])) != fac:
                    continue
                if a.unit_type == ut:
                    act = a
                    break
            if act is None:
                continue
            try:
                sim2.step(act)
            except Exception:
                continue
            choices_so_far.append(ut)
            dfs(slot_i + 1, sim2, choices_so_far)
            choices_so_far.pop()
            if len(results) >= cap:
                if not truncated:
                    truncated = True
                    frontier_depth_at_cap = slot_i
                return

    sim0 = clone_for_search(post_moves)
    dfs(0, sim0, [])

    return BuyEnumerateResult(
        genomes=results,
        truncated=truncated,
        frontier_depth_at_cap=frontier_depth_at_cap,
        candidates_enumerated=len(results),
    )


def score_buy_candidate(
    execute_buy: Callable[..., tuple],
    fitness: RheaFitness,
    post_moves: GameState,
    genome: object,
    *,
    acting_seat: int,
    reward_weight: float,
    value_weight: float,
    buy_value_scale: float,
    buy_shaping_weight: float,
    illegal_gene_penalty: float,
    v_ref_buy: float,
    phi_ref_buy: float,
    gold_before: int,
    v_terminal_override: float | None = None,
) -> ScoredBuyCandidate:
    """Score one buy genome using the same formula as legacy buy RHEA."""
    acts_b, _ef, sf, igb, rk, rex = execute_buy(
        post_moves, genome, mutate_sim=clone_for_search(post_moves),
    )
    if v_terminal_override is not None:
        v_terminal = float(v_terminal_override)
    else:
        v_terminal = float(fitness.value(sf, acting_seat))
    phi_sf = float(fitness.phi(sf, acting_seat))
    rew_shaped = float(rk) + float(rex)
    v_adv_b = (v_terminal - v_ref_buy) * 2.0
    phi_d_b = phi_sf - phi_ref_buy
    ileg_b = -illegal_gene_penalty * float(igb)
    tot_b = (
        reward_weight * phi_d_b
        + value_weight * v_adv_b * buy_value_scale
        + buy_shaping_weight * rew_shaped
        + ileg_b
    )
    spent = max(0, int(gold_before) - int(sf.funds[acting_seat]))
    return ScoredBuyCandidate(
        total=float(tot_b),
        gold_spent=int(spent),
        genome=genome,
        illegal=int(igb),
        build_actions=list(acts_b),
        sim_after=sf,
    )


def _ensure_greedy_in_list(
    genomes: list,
    greedy: object | None,
    factory_order: tuple[tuple[int, int], ...],
) -> list:
    if greedy is None:
        return genomes
    gkey = _genome_key(factory_order, greedy.choices)
    for g in genomes:
        if _genome_key(factory_order, g.choices) == gkey:
            return genomes
    return list(genomes) + [greedy]


def pick_best_exhaustive_buy(
    planner: object,
    fitness: RheaFitness,
    post_moves: GameState,
    acting_seat: int,
    factory_order: tuple[tuple[int, int], ...],
    catalog: dict[tuple[int, int], tuple[UnitType, ...]],
    *,
    max_candidates: int,
    reward_weight: float,
    value_weight: float,
    buy_value_scale: float,
    buy_shaping_weight: float,
    illegal_gene_penalty: float,
    greedy_seed: object | None,
) -> ExhaustiveBuyPick:
    """Enumerate, score, and return argmax (tot, gold_spent) buy genome."""
    enum = enumerate_buy_allocations(
        post_moves,
        factory_order,
        catalog,
        acting_seat,
        max_count=max_candidates,
    )
    genomes = _ensure_greedy_in_list(enum.genomes, greedy_seed, factory_order)
    candidates_enumerated = len(enum.genomes)
    candidates_scored = len(genomes)

    v_ref_buy = float(fitness.value(post_moves, acting_seat))
    phi_ref_buy = float(fitness.phi(post_moves, acting_seat))
    gold_before = int(post_moves.funds[acting_seat])

    execute_buy = planner._execute_buy_spend_allocation

    terminal_states: list[GameState] = []
    exec_rows: list[tuple] = []
    for g in genomes:
        acts_b, _ef, sf, igb, rk, rex = execute_buy(
            post_moves, g, mutate_sim=clone_for_search(post_moves),
        )
        terminal_states.append(sf)
        exec_rows.append((g, acts_b, sf, igb, rk, rex))

    v_batch: list[float] | None = None
    if float(value_weight) > 0.0 and terminal_states:
        v_batch = list(fitness.batch_value(terminal_states, acting_seat))

    scored: list[ScoredBuyCandidate] = []
    for idx, (g, acts_b, sf, igb, rk, rex) in enumerate(exec_rows):
        if v_batch is not None:
            v_terminal = float(v_batch[idx])
        else:
            v_terminal = float(fitness.value(sf, acting_seat))
        phi_sf = float(fitness.phi(sf, acting_seat))
        rew_shaped = float(rk) + float(rex)
        v_adv_b = (v_terminal - v_ref_buy) * 2.0
        phi_d_b = phi_sf - phi_ref_buy
        ileg_b = -illegal_gene_penalty * float(igb)
        tot_b = (
            reward_weight * phi_d_b
            + value_weight * v_adv_b * buy_value_scale
            + buy_shaping_weight * rew_shaped
            + ileg_b
        )
        spent = max(0, int(gold_before) - int(sf.funds[acting_seat]))
        scored.append(
            ScoredBuyCandidate(
                total=float(tot_b),
                gold_spent=int(spent),
                genome=g,
                illegal=int(igb),
                build_actions=list(acts_b),
                sim_after=sf,
            )
        )

    scored.sort(key=lambda r: (r.total, r.gold_spent), reverse=True)
    best = scored[0]
    return ExhaustiveBuyPick(
        genome=best.genome,
        build_actions=best.build_actions,
        sim_after=best.sim_after,
        illegal=best.illegal,
        candidates_enumerated=candidates_enumerated,
        candidates_scored=candidates_scored,
        truncated=enum.truncated,
        frontier_depth_at_cap=enum.frontier_depth_at_cap,
    )
