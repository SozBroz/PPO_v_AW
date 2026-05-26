"""
Segmented-genome RHEA (Rolling Horizon Evolution Algorithm).

Genome structure replaces the flat ``list[int]`` of ranked candidate indices with
a typed segmented genome:

    Move-phase genome (``RheaGenome``, ``build_segment`` empty when two-phase
    Buy RHEA is enabled):

        cop_activate: bool
        unit_segment: list[UnitIntent]

    Buy-phase genome (``BuySpendGenome``): ordered `(factory_tile → unit | SKIP)`
    evaluated left-to-right with a shared gold budget — see
    ``RheaPlanner._evolve_buy_spend``.

Each ``UnitIntent`` directly encodes the full SELECT → MOVE → ACTION cycle for
one unit, so the executor never depends on a fragile positional index into a
dynamically-regenerated candidate list.  Illegal genes are **soft-failed**
(skipped with an increment of the illegal counter) instead of aborting the
entire genome.

Key design properties:
  - ``max_actions_per_turn`` caps unit intents per turn (default 256, 2× legacy
    128). The unit segment is padded to cover every unmoved unit at pool build.
  - Builds are separated from unit actions, so they never pollute the
    SELECT-stage candidate pool.
  - CO power activation is a separate phase so Eagle-style SCOP interleaving
    (move some units → activate SCOP → move them again) is straightforward.
  - Variable-length ``unit_segment`` — crossover and mutation add/remove intents
    within reasonable bounds.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
import random
import time
from typing import Callable, Optional

from engine.action import (
    Action,
    ActionStage,
    ActionType,
    compute_reachable_costs,
    get_legal_actions,
)
from engine.search_clone import clone_for_search
from engine.game import GameState
from engine.unit import UNIT_STATS, UnitType
from rl.candidate_actions import (
    MAX_CANDIDATES,
    CandidateAction,
    CandidateKind,
    candidate_arrays,
    enumerate_candidates,
)
from rl.buy_exhaustive import pick_best_exhaustive_buy
from rl.rhea_fitness import RheaFitness, RheaFitnessBreakdown
from rl.tactical_beam import TacticalBeamConfig, TacticalBeamPlanner

# Cython acceleration — DISABLED for the new segmented genome.
# The flat-index-lists are gone; the Cython hot-path cannot compile against
# the new RheaGenome / UnitIntent / BuildIntent types.  Rewrite the Cython
# module separately when the stable shape of the new genome is confirmed.
USE_CYTHON_RHEA = False


# ---------------------------------------------------------------------------
# Segmented genome types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class UnitIntent:
    """One unit's full SELECT → MOVE → ACTION cycle.

    Fields are *intentions*, not guarantees — the executor validates each
    against the live simulation state before applying.
    """
    unit_pos: tuple[int, int]
    move_dest: tuple[int, int]
    action_type: ActionType
    target_pos: Optional[tuple[int, int]] = None


@dataclass(slots=True)
class BuildIntent:
    """Build one unit at a factory/airport/port."""
    factory_pos: tuple[int, int]
    unit_type: UnitType


@dataclass(slots=True)
class RheaGenome:
    """Full segmented genome for one turn."""
    cop_activate: bool = False
    unit_segment: list[UnitIntent] = field(default_factory=list)
    build_segment: list[BuildIntent] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Ensure mutable lists (field(default_factory) reuses the default,
        # but dataclass __init__ always creates new ones when called with
        # keyword args.  This is a safety net for manual construction.)
        pass
@dataclass(slots=True)
class BuySpendGenome:
    """Post-move purchasing plan: fixed factory order × unit or SKIP."""

    factory_order: tuple[tuple[int, int], ...]
    choices: list[Optional[UnitType]]  # None = SKIP


# ---------------------------------------------------------------------------
# Action pool — snapshot of legal options at genome-design time
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ActionPool:
    """Snapshot of legal actions at turn start for genome design and mutation.

    Built once from the ``before`` state at the start of ``choose_full_turn``
    and reused across all generations of evolution.  Stays valid because every
    genome is evaluated against a *clone* of the same before-state.

    The pool lazily caches:
      - unmoved unit positions (from SELECT-stage candidates)
      - for each unmoved unit: the reachable destinations and terminal actions
        available at each destination (explored by advancing the probe state to
        MOVE stage then ACTION stage)
      - legal BUILD actions
      - power-activation availability
    """
    unmoved_positions: list[tuple[int, int]]
    unit_options: dict[tuple[int, int], list[UnitIntent]]
    build_options: list[BuildIntent]
    cop_legal: bool
    scop_legal: bool
    power_active: bool  # True if either COP or SCOP is active (cannot activate another)

    @classmethod
    def build(cls, state: GameState) -> ActionPool:
        """Construct an action pool from the current state (SELECT stage)."""
        if state.action_stage != ActionStage.SELECT:
            raise ValueError("ActionPool.build requires SELECT stage state")

        feats, mask, cands = candidate_arrays(state, max_candidates=MAX_CANDIDATES)
        legal: list[CandidateAction] = [
            c for i, c in enumerate(cands) if i < len(mask) and bool(mask[i])
        ]

        # Categorise SELECT-stage candidates.
        select_unit_cands: list[CandidateAction] = []
        build_cands: list[CandidateAction] = []
        cop_legal = False
        scop_legal = False
        power_active = False

        for c in legal:
            ta = c.terminal_action
            if ta.action_type == ActionType.BUILD:
                build_cands.append(c)
            elif ta.action_type in (ActionType.ACTIVATE_COP, ActionType.ACTIVATE_SCOP):
                if ta.action_type == ActionType.ACTIVATE_COP:
                    cop_legal = True
                if ta.action_type == ActionType.ACTIVATE_SCOP:
                    scop_legal = True
            elif (
                c.kind == CandidateKind.SELECT_UNIT
                and ta.unit_pos is not None
            ):
                # Only keep SELECT_UNIT for unmoved non-stunned units.
                unit = state.get_unit_at(*ta.unit_pos)
                if unit is not None and not unit.moved and not unit.is_stunned:
                    select_unit_cands.append(c)

        # Extract unmoved unit positions.
        unmoved_positions: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for c in select_unit_cands:
            pos = c.terminal_action.unit_pos
            if pos is not None and pos not in seen:
                seen.add(pos)
                unmoved_positions.append(pos)

        # For each unmoved unit, enumerate MOVE → ACTION options.
        unit_options: dict[tuple[int, int], list[UnitIntent]] = {}

        for upos in unmoved_positions:
            opts: list[UnitIntent] = []

            # Advance a shallow probe to MOVE stage with this unit selected.
            unit = state.get_unit_at(*upos)
            if unit is None or unit.moved or unit.is_stunned:
                continue

            move_cands = _candidates_for_unit_at_move_stage(state, unit)
            for mc in move_cands:
                if mc.second is None:
                    # Setup intents (MOVE_SETUP_UNLOAD, MOVE_SETUP_REPAIR,
                    # MOVE_SETUP_ACTION).  Skip for the pool — these require
                    # an additional ACTION-stage decision that needs the full
                    # probe chain.  RHEA will evolve away from needing them
                    # because movement-only actions (MOVE_WAIT, MOVE_CAPTURE,
                    # etc.) are more common and easier to mutate.
                    continue

                move_dest = mc.first.move_pos
                if move_dest is None:
                    continue

                # Extract the terminal ActionType and target.
                action_type = mc.second.action_type
                target_pos = mc.second.target_pos

                opts.append(UnitIntent(
                    unit_pos=upos,
                    move_dest=move_dest,
                    action_type=action_type,
                    target_pos=target_pos,
                ))

            if opts:
                unit_options[upos] = opts

        # Build options.
        build_options: list[BuildIntent] = []
        for c in build_cands:
            ta = c.terminal_action
            factory_pos = ta.move_pos
            ut = ta.unit_type
            if factory_pos is not None and ut is not None:
                build_options.append(BuildIntent(factory_pos=factory_pos, unit_type=ut))

        power_active = _power_already_active(state)
        return cls(
            unmoved_positions=unmoved_positions,
            unit_options=unit_options,
            build_options=build_options,
            cop_legal=cop_legal,
            scop_legal=scop_legal,
            power_active=power_active,
        )


def _candidates_for_unit_at_move_stage(
    state: GameState,
    unit,
) -> list[CandidateAction]:
    """Enumerate MOVE-stage candidates for a single unit.

    Creates a minimal probe so ``enumerate_candidates`` sees the correct
    ``selected_unit`` and ``action_stage`` without a full clone.
    """
    probe = copy.copy(state)
    probe.selected_unit = unit
    probe.action_stage = ActionStage.MOVE
    # shallow copy is enough — we only read the candidate list, never step().
    return enumerate_candidates(probe)


def _power_already_active(state: GameState) -> bool:
    """True if the acting player already has a CO power active."""
    co = state.co_states[int(state.active_player)]
    return bool(co.cop_active) or bool(co.scop_active)


def _sample_build_intent_biased_expensive(
    opts: list[BuildIntent],
    rng: random.Random,
    *,
    p_pick_max_cost: float,
) -> BuildIntent:
    """Prefer the most expensive legal unit at this factory (still from ``opts``).

    Uniform random mutation re-rolls infantry far too often relative to its
    search value; bias initial genomes and mutations toward top-cost options
    so RHEA explores spending money on army quality, not only captures.
    """
    if not opts:
        raise ValueError("_sample_build_intent_biased_expensive: empty opts")
    if len(opts) == 1:
        return opts[0]
    max_c = max(UNIT_STATS[bi.unit_type].cost for bi in opts)
    best = [bi for bi in opts if UNIT_STATS[bi.unit_type].cost == max_c]
    if rng.random() < p_pick_max_cost:
        return rng.choice(best)
    return rng.choice(opts)


def _factories_fully_owned_positions(state: GameState, acting: int) -> set[tuple[int, int]]:
    """Production tiles fully owned before this turn's move phase (Rule #4 baseline)."""
    out: set[tuple[int, int]] = set()
    for prop in state.properties:
        if prop.owner != acting:
            continue
        if not (
            getattr(prop, "is_base", False)
            or getattr(prop, "is_airport", False)
            or getattr(prop, "is_port", False)
        ):
            continue
        if prop.capture_points < 20:
            continue
        out.add((prop.row, prop.col))
    return out


def _newly_captured_factories_this_turn(
    owned_before_moves: set[tuple[int, int]],
    post_moves: GameState,
    acting: int,
) -> frozenset[tuple[int, int]]:
    """Factories not fully ours at turn-start snapshot but producible positions after moves.

    No ``capture_day`` in ``PropertyState`` — diff against the pre-move fully-owned set.
    """
    nu: set[tuple[int, int]] = set()
    for prop in post_moves.properties:
        if prop.owner != acting:
            continue
        if not (
            getattr(prop, "is_base", False)
            or getattr(prop, "is_airport", False)
            or getattr(prop, "is_port", False)
        ):
            continue
        if prop.capture_points < 20:
            continue
        pos = (prop.row, prop.col)
        if pos not in owned_before_moves:
            nu.add(pos)
    return frozenset(nu)


def _move_phase_factory_occupation_penalty(
    post_moves: GameState,
    acting: int,
    owned_factories_at_turn_start: set[tuple[int, int]],
    per_factory_penalty: float,
) -> float:
    """Negative reward when our units remain on our own factory tiles after moves.

    Same magnitude per blocked production tile as buy-phase base-skip
    (``per_factory_penalty`` is typically ``buy_base_skip_penalty``).

    Does **not** apply while capturing a neutral/enemy factory: any tile that
    flips to us this move phase is excluded via
    ``_newly_captured_factories_this_turn`` (same rule as buy-catalog fresh
    captures). Contested captures (``capture_points < 20``) are also naturally
    excluded because the property is not yet fully ours.
    """
    if per_factory_penalty == 0.0:
        return 0.0
    newly_captured = _newly_captured_factories_this_turn(
        owned_factories_at_turn_start, post_moves, acting,
    )
    penalizable: set[tuple[int, int]] = set()
    for prop in post_moves.properties:
        if prop.owner != acting:
            continue
        if prop.capture_points < 20:
            continue
        if not (
            getattr(prop, "is_base", False)
            or getattr(prop, "is_airport", False)
            or getattr(prop, "is_port", False)
        ):
            continue
        pos = (int(prop.row), int(prop.col))
        if pos in newly_captured:
            continue
        penalizable.add(pos)
    hits = 0
    for pos in penalizable:
        for u in post_moves.units[acting]:
            if not u.is_alive:
                continue
            if (int(u.pos[0]), int(u.pos[1])) == pos:
                hits += 1
                break
    return -float(per_factory_penalty) * float(hits)


def _ensure_select(sim: GameState) -> None:
    sim.action_stage = ActionStage.SELECT
    sim.selected_unit = None
    sim.selected_move_pos = None


def _collect_buy_catalog(
    post_moves: GameState,
    acting: int,
    exclude: frozenset[tuple[int, int]],
) -> tuple[tuple[tuple[int, int], ...], dict[tuple[int, int], tuple[UnitType, ...]]]:
    """Eligible factory tiles after the move phase and affordable unit flavours per tile."""
    _ensure_select(post_moves)
    if post_moves.winner is not None or int(post_moves.active_player) != acting:
        return (), {}
    legal = get_legal_actions(post_moves)
    by_tile: dict[tuple[int, int], set[UnitType]] = {}
    for a in legal:
        if a.action_type != ActionType.BUILD:
            continue
        if a.move_pos is None or a.unit_type is None:
            continue
        pos = (int(a.move_pos[0]), int(a.move_pos[1]))
        if pos in exclude:
            continue
        by_tile.setdefault(pos, set()).add(a.unit_type)
    if not by_tile:
        return (), {}
    ordered = tuple(sorted(by_tile.keys()))
    catalog = {
        p: tuple(sorted(by_tile[p], key=lambda ut: UNIT_STATS[ut].cost))
        for p in ordered
    }
    return ordered, catalog


@dataclass(slots=True)
class RheaAutotuneConfig:
    """Elastic move-search budget formula for ``--rhea-autotune``."""

    owned_unit_weight: float = 1.0
    factory_weight: float = 1.5
    contested_capture_weight: float = 2.0
    contact_weight: float = 0.5
    juicy_attack_weight: float = 0.0
    pop_base: float = 12.0
    pop_complexity_scale: float = 1.5
    pop_min: int = 24
    pop_max: int = 96
    gen_base: float = 3.0
    gen_complexity_scale: float = 1.0 / 6.0
    gen_min: int = 5
    gen_max: int = 12


def dynamic_buy_budget(
    num_factories: int,
    avg_affordable_unit_types: float,
) -> tuple[int, int]:
    """Tiny search for Buy RHEA — scale with factories × branching factor."""
    c = float(max(num_factories, 1)) * max(float(avg_affordable_unit_types), 1.0)
    pop = max(16, min(48, int(14 + c * 0.75)))
    gen = max(4, min(12, int(4 + c / 12.0)))
    return pop, gen


def dynamic_rhea_budget(
    owned_units: int,
    factories: int,
    contested_captures: int,
    enemy_in_range_contacts: int,
    juicy_attacks: int | RheaAutotuneConfig = 0,
    config: RheaAutotuneConfig | None = None,
) -> tuple[int, int]:
    """
    Compute dynamic RHEA search budget based on game state complexity.

    Returns: (population, generations) — no max_actions_per_turn, the genome
    is self-sizing from the number of unmoved units + factories.
    """
    if isinstance(juicy_attacks, RheaAutotuneConfig) and config is None:
        config = juicy_attacks
        juicy_attacks = 0
    cfg = config or RheaAutotuneConfig()
    complexity = (
        float(cfg.owned_unit_weight) * float(owned_units)
        + float(cfg.factory_weight) * float(factories)
        + float(cfg.contested_capture_weight) * float(contested_captures)
        + float(cfg.contact_weight) * float(enemy_in_range_contacts)
        + float(cfg.juicy_attack_weight) * float(juicy_attacks)
    )

    pop = int(float(cfg.pop_base) + float(cfg.pop_complexity_scale) * complexity)
    gen = int(float(cfg.gen_base) + float(cfg.gen_complexity_scale) * complexity)

    pop_min = int(cfg.pop_min)
    pop_max = max(pop_min, int(cfg.pop_max))
    gen_min = int(cfg.gen_min)
    gen_max = max(gen_min, int(cfg.gen_max))
    pop = max(pop_min, min(pop, pop_max))
    gen = max(gen_min, min(gen, gen_max))

    return pop, gen


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RheaConfig:
    population: int = 64
    generations: int = 10
    elite: int = 8
    mutation_rate: float = 0.20
    autotune: RheaAutotuneConfig = field(default_factory=RheaAutotuneConfig)
    # Upper bound on unit intents simulated per turn (2× legacy 128).
    max_actions_per_turn: int = 256
    top_k_per_state: int = 96
    reward_weight: float = 0.90
    value_weight: float = 0.10
    build_value_weight: float = 20.0  # unused when two-phase buy RHEA is on
    # Two-phase planner: evolve moves-only, then a small Buy RHEA on S_post_moves.
    two_phase_buy_rhea: bool = True
    # Buy-phase search budget (fallback when ``buy_autotune`` is disabled).
    buy_population: int = 16
    buy_generations: int = 4
    buy_elite: int = 4
    buy_autotune: bool = True
    buy_base_skip_penalty: float = 0.015
    buy_gold_hoard_penalty: float = 1.0
    buy_safe_reserve: float = 4000.0
    # Flat penalty per 1000 gold above safe reserve
    buy_gold_abs_penalty_per_1k: float = 0.02
    # Scale skip / hoard shaping in the buy phase independently of ``reward_weight``
    # so value-only eval (--value-weight 1 --reward-weight 0) still dislikes
    # skipping affordable builds.
    buy_shaping_weight: float = 2.5
    # Dampens marginal value in buy search only — the value net often assigns
    # spuriously negative deltas to spending gold, overwhelming skip shaping.
    buy_value_scale: float = 0.70
    # Bank credit: remaining gold (capped at this amount) contributes positively
    # to buy fitness so saving toward expensive units can compete with immediate
    # cheap builds. Per-1k credit rate is added to rew_shaped.
    buy_bank_credit_cap: float = 10000.0
    buy_bank_credit_per_1k: float = 0.01
    buy_mode: str = "rhea"  # "rhea" | "exhaustive" (opt-in)
    buy_exhaustive_max_candidates: int = 8192
    adaptive_extend: bool = False
    adaptive_max_extra_generations: int = 0
    adaptive_patience_generations: int = 1
    adaptive_min_improvement: float = 0.0025
    adaptive_max_wall_s: Optional[float] = None
    adaptive_hard_turn_wall_s: float = 900.0
    log_initial_best: bool = True
    seed: Optional[int] = None
    # Tactical beam config (unchanged)
    use_tactical_beam: bool = False
    tactial_beam_max_width: int = 96
    tactial_beam_max_depth: int = 28
    tactial_beam_max_expand: int = 48


@dataclass(slots=True)
class RheaResult:
    actions: list
    score: float
    breakdown: RheaFitnessBreakdown
    illegal_genes: int
    generations: int
    initial_best_score: float | None = None
    evolved_gain: float | None = None
    genome: Optional[RheaGenome] = None  # best genome for intent-group replay
    population_used: int = 32           # actual population used (autotune override)
    generations_used: int = 6           # actual generations used (autotune override)
    n_move_actions: int = 0             # two-phase: len(move_actions) for phase tagging
    n_buy_actions: int = 0              # two-phase: len(exec_buy_act) for phase tagging
    buy_candidates_enumerated: int = 0
    buy_candidates_scored: int = 0
    buy_exhaustive_truncated: bool = False
    buy_mode_used: str = "rhea"
    buy_exhaustive_frontier_depth_at_cap: int | None = None
    move_generations_floor: int = 0
    move_generations_used: int = 0
    adaptive_stop_reason: str | None = None
    adaptive_extra_generations_used: int = 0
    adaptive_disabled_reason: str | None = None
    adaptive_best_improvement: float | None = None


def _update_adaptive_progress(
    *,
    current_best: float,
    best_score_seen: float | None,
    stale_generations: int,
    min_improvement: float,
) -> tuple[float, int]:
    if best_score_seen is None:
        return float(current_best), 0
    if float(current_best) - float(best_score_seen) >= float(min_improvement):
        return float(current_best), 0
    return float(best_score_seen), int(stale_generations) + 1


def _adaptive_stop_after_generation(
    *,
    generation_done: int,
    floor_generations: int,
    max_extra_generations: int,
    stale_generations: int,
    patience_generations: int,
) -> str | None:
    if int(generation_done) < int(floor_generations):
        return None
    if int(max_extra_generations) <= 0:
        return None
    if int(stale_generations) >= max(1, int(patience_generations)):
        return "patience"
    if int(generation_done) >= int(floor_generations) + int(max_extra_generations):
        return "extra_cap"
    return None


def _adaptive_wall_stop_reason(
    *,
    elapsed_s: float,
    generation_index: int,
    floor_generations: int,
    soft_wall_s: float | None,
    hard_wall_s: float,
) -> str | None:
    if float(hard_wall_s) > 0.0 and float(elapsed_s) >= float(hard_wall_s):
        return "hard_wall"
    if (
        int(generation_index) >= int(floor_generations)
        and soft_wall_s is not None
        and float(elapsed_s) >= float(soft_wall_s)
    ):
        return "soft_wall"
    return None


ReplayStepCallback = Callable[
    [GameState, Action, GameState, int, str],
    None,
]


# ---------------------------------------------------------------------------
# Shared replay helpers  (replay RHEA actions on the real env state)
# ---------------------------------------------------------------------------


def _replay_oracle_step(
    env_state: GameState,
    action: Action,
    acting: int,
    *,
    on_step: ReplayStepCallback | None,
    step_idx: int,
    source: str,
) -> bool:
    """Apply one oracle replay step; invoke ``on_step`` only on success."""
    if env_state is None or env_state.winner is not None:
        return False
    if int(env_state.active_player) != acting:
        return False
    before = copy.deepcopy(env_state) if on_step is not None else env_state
    try:
        env_state.step(action, oracle_mode=True)
    except Exception:
        return False
    if on_step is not None:
        on_step(before, action, env_state, step_idx, source)
    return True


def _salvage_ender(
    env_state: GameState,
    acting: int,
    *,
    on_step: ReplayStepCallback | None = None,
    step_base: int = 0,
) -> int:
    """Force-advance ``env_state`` for ``acting``: WAIT unmoved units then END_TURN.

    Returns the number of successful oracle steps applied.  When ``on_step`` is
    set, each successful WAIT/SELECT/MOVE/END (etc.) emits an individual callback
    with ``source="salvage"`` — step mode must use this path, not the legacy
    ``replay_rhea_actions`` salvage bump of +1.
    """
    if env_state is None or env_state.winner is not None:
        return 0
    if int(env_state.active_player) != acting:
        return 0

    applied = 0
    step_idx = int(step_base)

    def _step(action: Action) -> bool:
        nonlocal step_idx, applied
        if _replay_oracle_step(
            env_state,
            action,
            acting,
            on_step=on_step,
            step_idx=step_idx,
            source="salvage",
        ):
            applied += 1
            step_idx += 1
            return True
        return False

    env_state.action_stage = ActionStage.SELECT
    env_state.selected_unit = None
    env_state.selected_move_pos = None

    legal = get_legal_actions(env_state)
    enders = [a for a in legal if a.action_type == ActionType.END_TURN]
    if enders and _step(enders[0]):
        return applied

    # END_TURN not legal (unmoved units) — WAIT each unmoved unit in place.
    for _u in list(env_state.units[acting]):
        if not _u.is_alive or _u.moved or _u.is_stunned:
            continue
        legal = get_legal_actions(env_state)
        sel = [a for a in legal
               if a.action_type == ActionType.SELECT_UNIT and a.unit_pos == _u.pos]
        if not sel:
            continue
        if not _step(sel[0]):
            continue
        legal = get_legal_actions(env_state)
        moves = [a for a in legal
                 if a.action_type == ActionType.SELECT_UNIT and a.move_pos == _u.pos]
        if not moves:
            continue
        if not _step(moves[0]):
            continue
        if env_state.action_stage == ActionStage.ACTION:
            legal = get_legal_actions(env_state)
            waits = [a for a in legal if a.action_type == ActionType.WAIT]
            if waits:
                _step(waits[0])
            elif legal:
                _step(legal[0])

    env_state.action_stage = ActionStage.SELECT
    env_state.selected_unit = None
    env_state.selected_move_pos = None
    legal = get_legal_actions(env_state)
    enders = [a for a in legal if a.action_type == ActionType.END_TURN]
    if enders:
        _step(enders[0])

    return applied


def _clamp_replay_select_stage(env_state: GameState | None) -> None:
    if (env_state is not None and env_state.winner is None
            and env_state.action_stage != ActionStage.SELECT):
        env_state.action_stage = ActionStage.SELECT
        env_state.selected_unit = None
        env_state.selected_move_pos = None


def replay_rhea_actions_with_steps(
    env_state: GameState,
    actions: list,
    acting: int,
    *,
    on_step: ReplayStepCallback | None = None,
) -> tuple[int, int, int]:
    """Replay planner actions with exact per-step accounting.

    Returns ``(applied_planned, skipped, salvage_steps)``.  ``salvage_steps`` is
    the count of successful engine steps from forced end-turn salvage (each WAIT,
    SELECT, MOVE, END, etc.), not the legacy +1 approximation used by
    ``replay_rhea_actions``.
    """
    applied_planned = 0
    skipped = 0
    step_idx = 0

    for action in actions:
        if env_state is None or env_state.winner is not None:
            break
        if int(env_state.active_player) != acting:
            break
        if _replay_oracle_step(
            env_state,
            action,
            acting,
            on_step=on_step,
            step_idx=step_idx,
            source="planned",
        ):
            applied_planned += 1
            step_idx += 1
        else:
            import traceback as _tb
            print(f"[replay_rhea_actions] step failed: {action!r}", flush=True)
            _tb.print_exc()
            skipped += 1

    salvage_steps = 0
    if (env_state is not None and env_state.winner is None
            and int(env_state.active_player) == acting):
        salvage_steps = _salvage_ender(
            env_state,
            acting,
            on_step=on_step,
            step_base=step_idx,
        )

    _clamp_replay_select_stage(env_state)
    return applied_planned, skipped, salvage_steps


def replay_rhea_actions(env_state: GameState, actions: list, acting: int) -> tuple[int, int]:
    """Replay raw engine ``actions`` one at a time, skip failures, salvage.

    ``actions`` are the real ``Action`` objects produced by the RHEA simulation
    (``result.actions`` from ``choose_full_turn``).

    Each action is replayed with ``oracle_mode=True`` to bypass the step gate's
    identity check (simulation-constructed Action objects are not the same
    instances as those from ``get_legal_actions()`` on the real state).

    If an action still fails (true state divergence), it is skipped.  After all
    actions are processed, the function attempts to salvage: force SELECT stage,
    WAIT any unmoved units in place, then END_TURN.  This ensures the turn
    advances even when all genome intents failed.

    Returns ``(applied, skipped)``.  Salvage still uses the historical +1 bump
    when still acting after the planned loop (see ``replay_rhea_actions_with_steps``
    for exact per-step salvage accounting).
    """
    applied_planned, skipped, salvage_steps = replay_rhea_actions_with_steps(
        env_state, actions, acting, on_step=None,
    )
    applied = applied_planned
    # Legacy +1 when end-of-turn salvage was needed (still acting after planned loop),
    # even if salvage applied multiple engine steps or failed to advance the seat.
    salvage_needed = (
        salvage_steps > 0
        or (
            env_state is not None
            and env_state.winner is None
            and int(env_state.active_player) == acting
        )
    )
    if salvage_needed:
        applied += 1
    return applied, skipped


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class RheaPlanner:
    """Rolling Horizon Evolution Algorithm planner with segmented genome."""

    def __init__(
        self,
        fitness: RheaFitness,
        config: RheaConfig,
        dynamic_budget: bool = False,
        complexity_metrics: Optional[tuple[int, int, int, int]] = None,
    ) -> None:
        self.fitness = fitness
        self.cfg = config
        self.rng = random.Random(config.seed)
        self.dynamic_budget = dynamic_budget
        self.complexity_metrics = complexity_metrics
        self.adaptive_disabled_reason: str | None = None
        # Tactical beam planner (unchanged)
        if config.use_tactical_beam:
            self.tactical_beam = TacticalBeamPlanner(
                fitness,
                TacticalBeamConfig(
                    enabled=True,
                    max_width=config.tactial_beam_max_width,
                    max_depth=config.tactial_beam_max_depth,
                    max_candidates_per_expand=config.tactial_beam_max_expand,
                ),
            )
        else:
            self.tactical_beam = None

    def note_turn_wall_time(self, wall_s: float) -> None:
        hard_wall_s = max(0.0, float(self.cfg.adaptive_hard_turn_wall_s))
        if hard_wall_s <= 0.0 or float(wall_s) < hard_wall_s:
            return
        self.adaptive_disabled_reason = "turn_wall_900s"

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def choose_full_turn(self, state: GameState) -> RheaResult:
        if bool(self.cfg.two_phase_buy_rhea):
            return self._choose_full_turn_two_phase(state)
        return self._choose_full_turn_monolithic(state)

    def _choose_full_turn_monolithic(self, state: GameState) -> RheaResult:
        acting_seat = int(state.active_player)
        before = state

        # Run tactical beam if enabled.
        beam_best = None
        if self.cfg.use_tactical_beam and self.tactical_beam is not None:
            beam_result = self.tactical_beam.search(before)
            if beam_result.lines:
                best_line = beam_result.lines[0]
                beam_best = {
                    'score': best_line.breakdown.total if best_line.breakdown else 0.0,
                    'actions': best_line.actions,
                    'breakdown': best_line.breakdown,
                    'illegal': 0,
                }

        # Determine budget.
        population_size = self.cfg.population
        generations = self.cfg.generations

        if self.dynamic_budget and self.complexity_metrics is not None:
            owned_units, factories, contested_captures, enemy_in_range_contacts = self.complexity_metrics[:4]
            juicy_attacks = self.complexity_metrics[4] if len(self.complexity_metrics) > 4 else 0
            pop, gen = dynamic_rhea_budget(
                owned_units,
                factories,
                contested_captures,
                enemy_in_range_contacts,
                juicy_attacks,
                self.cfg.autotune,
            )
            population_size = pop
            generations = gen

        # Build the action pool ONCE from the before-state.
        action_pool = ActionPool.build(before)

        # Initialise population from the pool.
        population = [
            self._random_genome(action_pool) for _ in range(population_size)
        ]

        if action_pool.unit_options:
            self._seed_initial_population_biases(
                population,
                action_pool,
                include_build_bias=True,
                make_genome=self._random_genome,
            )

        print(f"  [RHEA] pop={population_size} gen={generations} "
              f"combos={population_size * generations} "
              f"units={len(action_pool.unmoved_positions)} "
              f"builds={len(action_pool.build_options)}", flush=True)

        rhea_best = None
        initial_best_score: float | None = None

        for _gen in range(generations):
            after_states: list[tuple] = []

            for genome in population:
                after, actions, illegal, mfo_pen = self._simulate_genome(
                    before, genome,
                )
                after_states.append((after, actions, illegal, mfo_pen))

            # Batch value evaluation.
            all_states = [after for after, _, _, _ in after_states]
            all_values = self.fitness.batch_value(all_states, acting_seat)
            before_value = self.fitness.value(before, acting_seat)

            scored: list = []
            for idx, ((after, actions, illegal, mfo_pen), v_after) in enumerate(
                zip(after_states, all_values)
            ):
                phi_after = self.fitness.phi(after, acting_seat)
                phi_before = self.fitness.phi(before, acting_seat)
                phi_delta = phi_after - phi_before

                win_advantage = (v_after - before_value) * 2.0
                illegal_penalty = -self.fitness.illegal_gene_penalty * float(illegal)

                # Build penalty computation.
                build_punishment, unused_funds_penalty = self._compute_build_penalties(
                    before, population[idx], actions, acting_seat
                )

                # Build army value reward (independent of phi_alpha so RHEA can
                # directly see the value of spending money on expensive units).
                build_value_reward = self._compute_build_value_reward(
                    actions, self.cfg.build_value_weight
                )

                total = (
                    self.fitness.reward_weight * phi_delta
                    + self.fitness.value_weight * win_advantage
                    + illegal_penalty
                    + build_punishment
                    + unused_funds_penalty
                    + build_value_reward
                    + float(mfo_pen)
                )

                breakdown = RheaFitnessBreakdown(
                    phi_delta=float(phi_delta),
                    value=float(win_advantage),
                    illegal_penalty=float(illegal_penalty),
                    total=float(total),
                )

                scored.append((float(total), population[idx], actions, breakdown, illegal))

            scored.sort(key=lambda x: x[0], reverse=True)

            if _gen == 0:
                initial_best_score = float(scored[0][0])

            if rhea_best is None or scored[0][0] > rhea_best[0]:
                rhea_best = scored[0]

            # Per-generation logging
            if _gen == generations - 1 or (_gen % 3 == 0 and generations > 3):
                print(f"  RHEA gen={_gen+1}/{generations} "
                      f"best={scored[0][0]:+.4f} "
                      f"pop={population_size} "
                      f"illegal={scored[0][4]} "
                      f"elite={scored[0][0]:+.4f}", flush=True)

            # Selection: elites + crossover offspring.
            elites = scored[: max(1, self.cfg.elite)]
            next_pop: list[RheaGenome] = []
            for _, g, _, _, _ in elites:
                # Deep-copy the elite genome so future mutation doesn't
                # corrupt the pool's reference.
                next_pop.append(copy.deepcopy(g))

            while len(next_pop) < population_size:
                p1 = copy.deepcopy(self.rng.choice(elites)[1])
                p2 = copy.deepcopy(self.rng.choice(elites)[1])
                child = self._crossover(p1, p2, action_pool)
                self._mutate(child, action_pool)
                next_pop.append(child)

            population = next_pop

        # Compare beam and RHEA, return best.
        candidates: list = []
        if beam_best:
            candidates.append((
                beam_best['score'], None,
                beam_best['actions'], beam_best['breakdown'],
                beam_best['illegal'],
            ))
        if rhea_best:
            candidates.append(rhea_best)

        if not candidates:
            return RheaResult(
                actions=[],
                score=0.0,
                breakdown=RheaFitnessBreakdown(
                    phi_delta=0.0, value=0.0, illegal_penalty=0.0, total=0.0
                ),
                illegal_genes=0,
                generations=generations,
                initial_best_score=None,
                evolved_gain=None,
                population_used=population_size,
                generations_used=generations,
            )

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, _, best_actions, best_breakdown, best_illegal = candidates[0]

        if beam_best and best_score == beam_best['score']:
            gain = (
                None
                if initial_best_score is None
                else float(best_score) - float(initial_best_score)
            )
            return RheaResult(
                actions=beam_best['actions'],
                score=float(best_score),
                breakdown=beam_best['breakdown'],
                illegal_genes=0,
                generations=generations,
                initial_best_score=initial_best_score,
                evolved_gain=gain,
                population_used=population_size,
                generations_used=generations,
            )
        else:
            score, _genome, actions, breakdown, illegal = candidates[0]
            gain = (
                None
                if initial_best_score is None
                else float(score) - float(initial_best_score)
            )
            return RheaResult(
                actions=actions,
                score=float(score),
                breakdown=breakdown,
                illegal_genes=int(illegal),
                generations=generations,
                initial_best_score=initial_best_score,
                evolved_gain=gain,
                genome=_genome,
                population_used=population_size,
                generations_used=generations,
            )

    def _choose_full_turn_two_phase(self, state: GameState) -> RheaResult:
        """Separate Move RHEA (Φ/value on post-move snapshot) → Buy RHEA (value+shaping)."""
        acting_seat = int(state.active_player)
        before = state
        factories_before = _factories_fully_owned_positions(before, acting_seat)

        population_size = self.cfg.population
        generations = self.cfg.generations
        if self.dynamic_budget and self.complexity_metrics is not None:
            owned_units, factories, contested_caps, contacts = self.complexity_metrics[:4]
            juicy_attacks = self.complexity_metrics[4] if len(self.complexity_metrics) > 4 else 0
            population_size, generations = dynamic_rhea_budget(
                owned_units,
                factories,
                contested_caps,
                contacts,
                juicy_attacks,
                self.cfg.autotune,
            )

        pool = ActionPool.build(before)
        population = [
            self._random_move_only_genome(pool) for _ in range(population_size)
        ]
        if pool.unit_options:
            self._seed_initial_population_biases(
                population,
                pool,
                include_build_bias=False,
                make_genome=self._random_move_only_genome,
            )

        try:
            print(
                f"  [RHEA/2phi] MOVE pop={population_size} gen={generations} "
                f"combos={population_size * generations} "
                f"units={len(pool.unmoved_positions)}",
                flush=True,
            )
        except UnicodeEncodeError:
            # Fallback to ASCII representation
            print(
                f"  [RHEA/2phi] MOVE pop={population_size} gen={generations} "
                f"combos={population_size * generations} "
                f"units={len(pool.unmoved_positions)}",
                flush=True,
            )
        except UnicodeEncodeError:
            # Fallback to ASCII representation
            print(
                f"  [RHEA/2phi] MOVE pop={population_size} gen={generations} "
                f"combos={population_size * generations} "
                f"units={len(pool.unmoved_positions)}",
                flush=True,
            )
        except UnicodeEncodeError:
            # Fallback to ASCII representation
            print(
                f"  [RHEA/2phi] MOVE pop={population_size} gen={generations} "
                f"combos={population_size * generations} "
                f"units={len(pool.unmoved_positions)}",
                flush=True,
            )

        rw = float(self.fitness.reward_weight)
        vw = float(self.fitness.value_weight)
        igen_pen = float(self.fitness.illegal_gene_penalty)

        adaptive_best_improvement: float | None = None
        move_track_best: tuple[float, RheaGenome] | None = None
        init_move_best: float | None = None

        phi_before_state = float(self.fitness.phi(before, acting_seat))
        v_before = float(self.fitness.value(before, acting_seat))
        fac_block_pp = float(self.cfg.buy_base_skip_penalty)

        move_wall_start = time.perf_counter()
        move_generations_floor = int(generations)
        move_generations_used = 0
        adaptive_extra_used = 0
        adaptive_stop_reason = "floor_complete"
        adaptive_hard_reason: str | None = None
        best_score_seen: float | None = None
        stale_generations = 0
        max_extra_generations = (
            max(0, int(self.cfg.adaptive_max_extra_generations))
            if bool(self.cfg.adaptive_extend) and self.adaptive_disabled_reason is None
            else 0
        )
        patience_generations = max(1, int(self.cfg.adaptive_patience_generations))
        min_improvement = max(0.0, float(self.cfg.adaptive_min_improvement))
        soft_wall_s = self.cfg.adaptive_max_wall_s
        soft_wall_s = (
            float(soft_wall_s)
            if soft_wall_s is not None and float(soft_wall_s) > 0.0
            else None
        )
        hard_wall_s = max(0.0, float(self.cfg.adaptive_hard_turn_wall_s))
        max_move_generations = move_generations_floor + max_extra_generations
        mv_gen = 0
        while mv_gen < max_move_generations:
            elapsed_s = time.perf_counter() - move_wall_start
            wall_stop_reason = _adaptive_wall_stop_reason(
                elapsed_s=elapsed_s,
                generation_index=mv_gen,
                floor_generations=move_generations_floor,
                soft_wall_s=soft_wall_s,
                hard_wall_s=hard_wall_s,
            )
            if wall_stop_reason is not None:
                adaptive_stop_reason = wall_stop_reason
                if wall_stop_reason == "hard_wall":
                    adaptive_hard_reason = "turn_wall_900s"
                    self.adaptive_disabled_reason = adaptive_hard_reason
                break
            states_after: list[GameState] = []
            ill_moves: list[int] = []

            for g in population:
                am, _mav, ild = self._simulate_move_phase_only(before, g)
                states_after.append(am)
                ill_moves.append(int(ild))

            vals_mv = self.fitness.batch_value(states_after, acting_seat)

            scored_mv: list = []
            for idx, sam in enumerate(states_after):
                phi_delta = (
                    float(self.fitness.phi(sam, acting_seat))
                    - phi_before_state
                )
                wadv = (float(vals_mv[idx]) - v_before) * 2.0
                ileg_p = -igen_pen * float(ill_moves[idx])
                fac_occ = _move_phase_factory_occupation_penalty(
                    sam, acting_seat, factories_before, fac_block_pp,
                )
                total_m = rw * phi_delta + vw * wadv + ileg_p + fac_occ
                scored_mv.append((total_m, population[idx]))

            scored_mv.sort(key=lambda x: x[0], reverse=True)

            current_best = float(scored_mv[0][0])
            if mv_gen == 0:
                init_move_best = current_best
            if move_track_best is None or current_best > move_track_best[0]:
                move_track_best = (current_best, copy.deepcopy(scored_mv[0][1]))
            if init_move_best is not None:
                adaptive_best_improvement = float(move_track_best[0]) - float(init_move_best)

            best_score_seen, stale_generations = _update_adaptive_progress(
                current_best=current_best,
                best_score_seen=best_score_seen,
                stale_generations=stale_generations,
                min_improvement=min_improvement,
            )

            move_generations_used = mv_gen + 1

            if mv_gen == max_move_generations - 1 or (mv_gen % 3 == 0 and max_move_generations > 3):
                try:
                    print(
                        f"  [RHEA/2phi] MOVE gen={mv_gen + 1}/{max_move_generations} "
                        f"best={current_best:+.4f}",
                        flush=True,
                    )
                except UnicodeEncodeError:
                    # Fallback to ASCII representation
                    print(
                        f"  [RHEA/2phi] MOVE gen={mv_gen + 1}/{max_move_generations} "
                        f"best={current_best:+.4f}",
                        flush=True,
                    )

            gen_stop_reason = _adaptive_stop_after_generation(
                generation_done=mv_gen + 1,
                floor_generations=move_generations_floor,
                max_extra_generations=max_extra_generations,
                stale_generations=stale_generations,
                patience_generations=patience_generations,
            )
            if gen_stop_reason is not None:
                adaptive_stop_reason = gen_stop_reason
                break

            elites_mv = scored_mv[: max(1, self.cfg.elite)]
            next_mv: list[RheaGenome] = []
            for _sc, gx in elites_mv:
                next_mv.append(copy.deepcopy(gx))

            while len(next_mv) < population_size:
                p1 = copy.deepcopy(self.rng.choice(elites_mv)[1])
                p2 = copy.deepcopy(self.rng.choice(elites_mv)[1])
                child = self._crossover_move_only(p1, p2, pool)
                self._mutate_move_phase_genome(child, pool)
                next_mv.append(child)

            population = next_mv
            mv_gen += 1

        adaptive_extra_used = max(0, move_generations_used - move_generations_floor)

        mv_genome_best = (
            population[0]
            if move_track_best is None
            else move_track_best[1]
        )
        s_after_moves, move_actions, mv_illegal = self._simulate_move_phase_only(
            before, mv_genome_best,
        )

        newly_cap = _newly_captured_factories_this_turn(
            factories_before, s_after_moves, acting_seat,
        )
        fo, catalog = _collect_buy_catalog(s_after_moves, acting_seat, newly_cap)

        buy_pop_sz = int(self.cfg.buy_population)
        buy_gen_sz = int(self.cfg.buy_generations)
        buy_elite_n = max(2, min(self.cfg.buy_elite, buy_pop_sz // 2))

        if self.cfg.buy_autotune and len(fo):
            denom = float(len(fo))
            avg_br = (
                sum(len(catalog[p]) for p in fo) / denom if denom > 0.0 else 1.0
            )
            buy_pop_sz, buy_gen_sz = dynamic_buy_budget(len(fo), avg_br)
            buy_elite_n = max(2, min(self.cfg.buy_elite, buy_pop_sz // 2))

        ig_tot = int(mv_illegal)

        if not fo:
            sim_fin = clone_for_search(s_after_moves)
            tail: list = []
            ig_tot += self._simulate_end_turn_sequence(sim_fin, acting_seat, tail)
            phi_end = (
                float(self.fitness.phi(sim_fin, acting_seat))
                - float(self.fitness.phi(before, acting_seat))
            )
            v_adv_end = (
                float(self.fitness.value(sim_fin, acting_seat)) - v_before
            ) * 2.0
            ill_c = -igen_pen * float(ig_tot)
            score_tot = rw * phi_end + vw * v_adv_end + ill_c

            bd = RheaFitnessBreakdown(
                phi_delta=float(phi_end),
                value=float(v_adv_end),
                illegal_penalty=float(ill_c),
                total=float(score_tot),
            )
            return RheaResult(
                actions=list(move_actions) + list(tail),
                score=float(score_tot),
                breakdown=bd,
                illegal_genes=int(ig_tot),
                generations=move_generations_used + buy_gen_sz,
                initial_best_score=init_move_best,
                evolved_gain=(
                    float(score_tot) - float(init_move_best)
                    if init_move_best is not None else None
                ),
                genome=mv_genome_best,
                population_used=population_size,
                generations_used=move_generations_used + buy_gen_sz,
                move_generations_floor=move_generations_floor,
                move_generations_used=move_generations_used,
                adaptive_stop_reason=adaptive_stop_reason,
                adaptive_extra_generations_used=adaptive_extra_used,
                adaptive_disabled_reason=adaptive_hard_reason or self.adaptive_disabled_reason,
                adaptive_best_improvement=adaptive_best_improvement,
                buy_mode_used=str(self.cfg.buy_mode),
                n_move_actions=len(move_actions),
                n_buy_actions=0,
            )

        buy_mode_used = str(self.cfg.buy_mode)
        buy_candidates_enumerated = 0
        buy_candidates_scored = 0
        buy_exhaustive_truncated = False
        buy_exhaustive_frontier_depth_at_cap: int | None = None
        buy_generations_used = int(buy_gen_sz)

        if buy_mode_used == "exhaustive":
            try:
                print(
                    f"  [RHEA/2phi] BUY exhaustive cap="
                    f"{self.cfg.buy_exhaustive_max_candidates} slots={len(fo)}",
                    flush=True,
                )
            except UnicodeEncodeError:
                print(
                    f"  [RHEA/2phi] BUY exhaustive cap="
                    f"{self.cfg.buy_exhaustive_max_candidates} slots={len(fo)}",
                    flush=True,
                )

            greedy_seed = self._greedy_cheapest_buy_genome(
                s_after_moves, acting_seat, fo, catalog,
            )
            ex_pick = pick_best_exhaustive_buy(
                self,
                self.fitness,
                s_after_moves,
                acting_seat,
                fo,
                catalog,
                max_candidates=int(self.cfg.buy_exhaustive_max_candidates),
                reward_weight=rw,
                value_weight=vw,
                buy_value_scale=float(self.cfg.buy_value_scale),
                buy_shaping_weight=float(self.cfg.buy_shaping_weight),
                illegal_gene_penalty=igen_pen,
                greedy_seed=greedy_seed,
            )
            best_buy_g = ex_pick.genome
            exec_buy_act = list(ex_pick.build_actions)
            sim_after_buy = ex_pick.sim_after
            sim_terminal = clone_for_search(s_after_moves)
            ileg_buy = int(ex_pick.illegal)
            buy_candidates_enumerated = int(ex_pick.candidates_enumerated)
            buy_candidates_scored = int(ex_pick.candidates_scored)
            buy_exhaustive_truncated = bool(ex_pick.truncated)
            buy_exhaustive_frontier_depth_at_cap = ex_pick.frontier_depth_at_cap
            buy_generations_used = 0
            print(
                json.dumps(
                    {
                        "event": "rhea_buy_exhaustive",
                        "slots": len(fo),
                        "cap": int(self.cfg.buy_exhaustive_max_candidates),
                        "candidates_enumerated": buy_candidates_enumerated,
                        "candidates_scored": buy_candidates_scored,
                        "truncated": buy_exhaustive_truncated,
                        "frontier_depth_at_cap": buy_exhaustive_frontier_depth_at_cap,
                    }
                ),
                flush=True,
            )
        else:
            try:
                print(
                    f"  [RHEA/2phi] BUY pop={buy_pop_sz} gen={buy_gen_sz} "
                    f"slots={len(fo)}{' autotune' if self.cfg.buy_autotune else ''}",
                    flush=True,
                )
            except UnicodeEncodeError:
                print(
                    f"  [RHEA/2phi] BUY pop={buy_pop_sz} gen={buy_gen_sz} "
                    f"slots={len(fo)}{' autotune' if self.cfg.buy_autotune else ''}",
                    flush=True,
                )

            buy_track_best: tuple[float, BuySpendGenome, float] | None = None
            buy_population: list[BuySpendGenome] = []

            seed_buy_build_bias_n = max(1, buy_pop_sz // 4)
            for _bp in range(buy_pop_sz):
                ch_rand: list[Optional[UnitType]] = []
                for p_fac in fo:
                    units_only = list(catalog[p_fac])
                    if _bp < seed_buy_build_bias_n and units_only:
                        ch_rand.append(self.rng.choice(units_only))
                    else:
                        pick_pool = units_only + [None]
                        ch_rand.append(self.rng.choice(pick_pool))
                buy_population.append(
                    self._canonical_buy_genome_after_exec(
                        s_after_moves,
                        BuySpendGenome(fo, ch_rand),
                    )
                )

            greedy_seed = self._greedy_cheapest_buy_genome(
                s_after_moves, acting_seat, fo, catalog,
            )
            if greedy_seed is not None:
                buy_population[0] = greedy_seed

            g0_buy = int(s_after_moves.funds[acting_seat])
            buy_sw = float(self.cfg.buy_shaping_weight)
            buy_vs = float(self.cfg.buy_value_scale)
            v_ref_buy = float(self.fitness.value(s_after_moves, acting_seat))
            phi_ref_buy = float(self.fitness.phi(s_after_moves, acting_seat))

            for by_gen in range(buy_gen_sz):
                scored_b: list = []
                for bg in buy_population:
                    sc = clone_for_search(s_after_moves)
                    acts_b, _ef, sf, igb, rk, rex = (
                        self._execute_buy_spend_allocation(
                            s_after_moves, bg, mutate_sim=sc,
                        )
                    )
                    v_terminal = float(self.fitness.value(sf, acting_seat))
                    phi_sf = float(self.fitness.phi(sf, acting_seat))
                    rew_shaped = float(rk) + float(rex)
                    v_adv_b = (v_terminal - v_ref_buy) * 2.0
                    phi_d_b = phi_sf - phi_ref_buy
                    ileg_b = -igen_pen * float(igb)
                    tot_b = (
                        rw * phi_d_b
                        + vw * v_adv_b * buy_vs
                        + buy_sw * rew_shaped
                        + ileg_b
                    )
                    spent = max(0, g0_buy - int(sf.funds[acting_seat]))
                    scored_b.append((tot_b, spent, bg, igb))

                    if by_gen == 0 and len(acts_b) > 0:
                        names = [
                            a.unit_type.name
                            for a in acts_b
                            if a.action_type == ActionType.BUILD
                        ]
                        print(
                            f"  [RHEA/BUY] cand: builds={names} "
                            f"phi_d={phi_d_b:+.4f} rk={rk:+.4f} rex={rex:+.4f} "
                            f"tot={tot_b:+.4f}",
                            flush=True,
                        )

                scored_b.sort(
                    key=lambda row: (row[0], row[1]),
                    reverse=True,
                )

                cand_key = (float(scored_b[0][0]), float(scored_b[0][1]))
                if buy_track_best is None:
                    buy_track_best = (
                        cand_key[0],
                        copy.deepcopy(scored_b[0][2]),
                        cand_key[1],
                    )
                else:
                    prev_key = (buy_track_best[0], buy_track_best[2])
                    if cand_key > prev_key:
                        buy_track_best = (
                            cand_key[0],
                            copy.deepcopy(scored_b[0][2]),
                            cand_key[1],
                        )

                if (
                    by_gen == buy_gen_sz - 1
                    or (by_gen % max(1, buy_gen_sz // 3) == 0)
                ):
                    try:
                        print(
                            f"  [RHEA/2phi] BUY gen={by_gen + 1}/{buy_gen_sz} "
                            f"best={scored_b[0][0]:+.4f}",
                            flush=True,
                        )
                    except UnicodeEncodeError:
                        print(
                            f"  [RHEA/2phi] BUY gen={by_gen + 1}/{buy_gen_sz} "
                            f"best={scored_b[0][0]:+.4f}",
                            flush=True,
                        )

                elites_b = scored_b[: max(2, buy_elite_n)]
                next_buy: list[BuySpendGenome] = []
                for _tot, _spent, bg_e, _ in elites_b:
                    next_buy.append(copy.deepcopy(bg_e))

                while len(next_buy) < buy_pop_sz:
                    bx = self._crossover_buy_spend(
                        fo,
                        self.rng.choice(elites_b)[2],
                        self.rng.choice(elites_b)[2],
                        s_after_moves,
                    )
                    if self.rng.random() < 0.65:
                        bx = self._mutate_buy_spend_genome(
                            fo, catalog, bx, s_after_moves,
                        )
                    next_buy.append(bx)

                buy_population = next_buy

            best_buy_g = (
                buy_population[0]
                if buy_track_best is None
                else buy_track_best[1]
            )

            sim_terminal = clone_for_search(s_after_moves)
            exec_buy_act, _, sim_after_buy, ileg_buy, *_ = (
                self._execute_buy_spend_allocation(
                    s_after_moves,
                    best_buy_g,
                    mutate_sim=sim_terminal,
                )
            )

        if not exec_buy_act and fo:
            fb = self._greedy_cheapest_buy_genome(
                s_after_moves, acting_seat, fo, catalog,
            )
            if fb is not None:
                sim_fb = clone_for_search(s_after_moves)
                g_acts, _, sim_fb, ileg_fb, *_ = (
                    self._execute_buy_spend_allocation(
                        s_after_moves, fb, mutate_sim=sim_fb,
                    )
                )
                if g_acts:
                    best_buy_g = fb
                    exec_buy_act = g_acts
                    sim_terminal = sim_fb
                    sim_after_buy = sim_fb
                    ileg_buy = ileg_fb
                    print(
                        "  [RHEA/2phi] BUY fallback: greedy cheapest — "
                        f"built {len(g_acts)}",
                        flush=True,
                    )

        ig_tot = int(mv_illegal) + int(ileg_buy)
        end_tail: list = []
        ig_tot += self._simulate_end_turn_sequence(
            sim_after_buy, acting_seat, end_tail,
        )

        phi_end = (
            float(self.fitness.phi(sim_after_buy, acting_seat))
            - float(self.fitness.phi(before, acting_seat))
        )
        v_adv_end = (
            float(self.fitness.value(sim_after_buy, acting_seat)) - v_before
        ) * 2.0
        ill_c = -igen_pen * float(ig_tot)
        score_total = rw * phi_end + vw * v_adv_end + ill_c

        bd = RheaFitnessBreakdown(
            phi_delta=float(phi_end),
            value=float(v_adv_end),
            illegal_penalty=float(ill_c),
            total=float(score_total),
        )

        return RheaResult(
            actions=list(move_actions) + list(exec_buy_act) + list(end_tail),
            score=float(score_total),
            breakdown=bd,
            illegal_genes=int(ig_tot),
            generations=move_generations_used + buy_generations_used,
            initial_best_score=init_move_best,
            evolved_gain=(
                float(score_total) - float(init_move_best)
                if init_move_best is not None
                else None
            ),
            genome=mv_genome_best,
            population_used=population_size,
            generations_used=move_generations_used + buy_generations_used,
            buy_candidates_enumerated=buy_candidates_enumerated,
            buy_candidates_scored=buy_candidates_scored,
            buy_exhaustive_truncated=buy_exhaustive_truncated,
            buy_mode_used=buy_mode_used,
            buy_exhaustive_frontier_depth_at_cap=buy_exhaustive_frontier_depth_at_cap,
            move_generations_floor=move_generations_floor,
            move_generations_used=move_generations_used,
            adaptive_stop_reason=adaptive_stop_reason,
            adaptive_extra_generations_used=adaptive_extra_used,
            adaptive_disabled_reason=adaptive_hard_reason or self.adaptive_disabled_reason,
            adaptive_best_improvement=adaptive_best_improvement,
            n_move_actions=len(move_actions),
            n_buy_actions=len(exec_buy_act),
        )

    # ------------------------------------------------------------------
    # Build penalty helper (extracted, unchanged logic)
    # ------------------------------------------------------------------

    def _player_has_base_factories(self, state: GameState, seat: int) -> bool:
        """Check if player has any build-capable properties (bases/airports/ports)."""
        for prop in state.properties:
            if prop.owner == seat and (
                bool(getattr(prop, "is_base", False))
                or bool(getattr(prop, "is_airport", False))
                or bool(getattr(prop, "is_port", False))
            ):
                return True
        return False

    def _count_available_factories(self, state: GameState, seat: int) -> int:
        """Count build-capable properties (bases, airports, ports) owned by seat.

        Only counts tiles that are either empty or occupied by an unmoved
        friendly unit (which Phase 1 may vacate before Phase 2 builds).
        """
        count = 0
        for prop in state.properties:
            if prop.owner == seat and (
                bool(getattr(prop, "is_base", False))
                or bool(getattr(prop, "is_airport", False))
                or bool(getattr(prop, "is_port", False))
            ):
                occupant = state.get_unit_at(prop.row, prop.col)
                if occupant is None or (
                    occupant.player == seat and not occupant.moved
                ):
                    count += 1
        return count

    def _compute_build_penalties(
        self,
        before: GameState,
        genome: RheaGenome | None,
        actions: list | None,
        acting_seat: int,
    ) -> tuple[float, float]:
        """Return (build_punishment, unused_funds_penalty).

        ``genome.build_segment`` is used to detect deliberate base-skipping
        (empty or too-few encode intents) independently of the executed
        actions list which includes salvage fills.  This is critical because
        salvage always fills empty factories with infantry — looking only at
        executed BUILD actions would mask the genome's decision to skip.
        """
        if actions is None or genome is None:
            return 0.0, 0.0

        end_turn_index = None
        build_happened = False
        build_actions: list = []
        units_built = 0
        total_build_cost = 0
        infantry_builds_executed = 0

        for i, action in enumerate(actions):
            atype = getattr(action, 'action_type', None)
            name = getattr(atype, 'name', '')
            if name == "END_TURN":
                end_turn_index = i
            elif name == "BUILD":
                build_happened = True
                build_actions.append(action)
                units_built += 1
                ut = getattr(action, 'unit_type', None)
                if ut is not None:
                    cost = UNIT_STATS[ut].cost
                    total_build_cost += cost
                    if ut == UnitType.INFANTRY:
                        infantry_builds_executed += 1

        build_punishment = 0.0
        unused_funds_penalty = 0.0

        # Base-skip penalty: the genome must *encode* builds explicitly,
        # otherwise salvage fills mask deliberate skipping.
        if end_turn_index is not None and self._player_has_base_factories(before, acting_seat):
            bp = self.fitness.env_template._build_punishment
            if bp > 0.0:
                n_encoded = len(genome.build_segment)
                if n_encoded < 1:
                    # Genome deliberately encoded zero build intents — exploit
                    # salvage fill to dodge the penalty.  This must hurt.
                    build_punishment = -5.0

        if build_happened and end_turn_index is not None:
            base_count = self._count_available_factories(before, acting_seat)
            if base_count > 0:
                    available_funds = before.funds[acting_seat]
                    funds_needed = base_count * 1000
                    if available_funds >= funds_needed:
                        if units_built < base_count:
                            missing = base_count - units_built
                            unused_funds_penalty = -0.5 * missing
                    else:
                        max_affordable = available_funds // 1000
                        if units_built < max_affordable:
                            missing = max_affordable - units_built
                            unused_funds_penalty = -0.5 * missing

                    # Value-awareness: penalize spending only a tiny fraction
                    # of available funds.  Building 5 infantry for 5k when you
                    # have 50k in the bank should hurt.
                    value_spent = float(total_build_cost)
                    value_available = float(available_funds)
                    if value_available > 0 and units_built > 0:
                        spend_ratio = value_spent / value_available
                        if spend_ratio < 0.5:
                            value_gap_penalty = -0.5 * (1.0 - spend_ratio) * float(base_count)
                            if value_gap_penalty < unused_funds_penalty:
                                unused_funds_penalty = value_gap_penalty

        # Infantry when the treasury can afford heavier production: Φ/capture
        # often dominates turn fitness; stamp on cheap-build spam explicitly.
        if end_turn_index is not None and infantry_builds_executed > 0:
            avail_f = float(before.funds[acting_seat])
            if avail_f >= 8000.0:
                scale = min(3.0, 1.0 + avail_f / 25000.0)
                unused_funds_penalty -= 1.15 * float(infantry_builds_executed) * scale

        return build_punishment, unused_funds_penalty

    @staticmethod
    def _compute_build_value_reward(
        actions: list | None,
        weight: float,
    ) -> float:
        """Independent reward for army value created by builds.

        This is separate from phi_delta so RHEA can directly see the value
        of spending money on expensive units, independent of the tiny
        phi_alpha coefficient that drowns the build signal in combat noise.
        """
        if actions is None or weight <= 0.0:
            return 0.0
        total_value = 0.0
        for action in actions:
            atype = getattr(action, 'action_type', None)
            if atype is not None and atype.name == "BUILD":
                ut = getattr(action, 'unit_type', None)
                if ut is not None:
                    c = float(UNIT_STATS[ut].cost)
                    if ut != UnitType.INFANTRY:
                        c *= 1.25
                    total_value += c
        # Normalise to ~0.01 per infantry, ~0.28 per mega tank.
        # weight=1.0 → infantry gives +0.01, mech +0.03, tank +0.07, etc.
        return weight * (total_value / 100000.0)

    # ------------------------------------------------------------------
    # Genome operations
    # ------------------------------------------------------------------

    def _unit_intent_cap(self) -> int:
        return max(1, int(self.cfg.max_actions_per_turn))

    def _ensure_unit_segment_covers_pool(
        self, genome: RheaGenome, pool: ActionPool
    ) -> None:
        """One intent per unmoved unit at turn start (crossover-safe)."""
        cap = self._unit_intent_cap()
        if not pool.unmoved_positions:
            genome.unit_segment = []
            return

        by_pos: dict[tuple[int, int], UnitIntent] = {}
        for intent in genome.unit_segment:
            by_pos[intent.unit_pos] = intent

        ordered: list[UnitIntent] = []
        for upos in pool.unmoved_positions:
            if upos in by_pos:
                ordered.append(by_pos[upos])
                continue
            opts = pool.unit_options.get(upos) or []
            if not opts:
                continue
            waits = [o for o in opts if o.action_type == ActionType.WAIT]
            ordered.append(
                self.rng.choice(waits) if waits else self.rng.choice(opts)
            )

        genome.unit_segment = ordered[:cap]

    def _random_genome(self, pool: ActionPool) -> RheaGenome:
        """Generate a random genome from the action pool.

        Fields that cannot be sampled (no unmoved units, no build options, etc.)
        are simply omitted — the executor will reach END_TURN naturally.
        """
        power_avail = pool.cop_legal or pool.scop_legal
        cop_activate = power_avail and self.rng.random() < 0.33

        unit_segment: list[UnitIntent] = []
        for upos in pool.unmoved_positions:
            opts = pool.unit_options.get(upos)
            if not opts:
                continue
            unit_segment.append(self.rng.choice(opts))

        build_segment: list[BuildIntent] = []
        # One build per factory — pick a random unit type for each.
        # pool.build_options has one entry per (factory, unit_type) pair;
        # grouping by factory avoids flooding the genome with intents that
        # will all-but-one fail as illegal.
        by_factory: dict[tuple[int, int], list[BuildIntent]] = {}
        for bi in pool.build_options:
            by_factory.setdefault(bi.factory_pos, []).append(bi)
        for factory_pos, opts in by_factory.items():
            build_segment.append(
                _sample_build_intent_biased_expensive(
                    opts, self.rng, p_pick_max_cost=0.72
                )
            )

        genome = RheaGenome(
            cop_activate=cop_activate,
            unit_segment=unit_segment,
            build_segment=build_segment,
        )
        self._ensure_unit_segment_covers_pool(genome, pool)
        return genome

    def _random_move_only_genome(self, pool: ActionPool) -> RheaGenome:
        """Move-phase genome with no build genes (Purchase RHEA runs later)."""
        power_avail = pool.cop_legal or pool.scop_legal
        cop_activate = power_avail and self.rng.random() < 0.33

        unit_segment: list[UnitIntent] = []
        for upos in pool.unmoved_positions:
            opts = pool.unit_options.get(upos)
            if not opts:
                continue
            unit_segment.append(self.rng.choice(opts))

        genome = RheaGenome(
            cop_activate=cop_activate, unit_segment=unit_segment, build_segment=[]
        )
        self._ensure_unit_segment_covers_pool(genome, pool)
        return genome

    _CAPTURE_BIAS_PROB = 0.85
    _ATTACK_BIAS_PROB = 0.85

    @staticmethod
    def _count_units_with_capture_options(pool: ActionPool) -> int:
        return sum(
            1
            for opts in pool.unit_options.values()
            if any(o.action_type == ActionType.CAPTURE for o in opts)
        )

    @staticmethod
    def _count_units_with_attack_options(pool: ActionPool) -> int:
        return sum(
            1
            for opts in pool.unit_options.values()
            if any(
                o.action_type == ActionType.ATTACK and o.move_dest != o.unit_pos
                for o in opts
            )
        )

    @staticmethod
    def _compute_initial_seed_counts(
        num_capture_units: int,
        num_attack_units: int,
        seed_budget: int,
    ) -> tuple[int, int]:
        """Return (capture_seeded_genomes, attack_seeded_genomes) within *seed_budget*."""
        if seed_budget <= 0:
            return 0, 0
        capture_seeds = (
            math.ceil(math.sqrt(num_capture_units)) if num_capture_units > 0 else 0
        )
        attack_seeds = (
            math.ceil(math.sqrt(num_attack_units)) if num_attack_units > 0 else 0
        )
        has_both = num_capture_units > 0 and num_attack_units > 0

        if has_both and seed_budget >= 2:
            capture_seeds = max(1, capture_seeds)
            attack_seeds = max(1, attack_seeds)
        elif num_capture_units > 0 and num_attack_units == 0:
            attack_seeds = 0
            capture_seeds = min(max(capture_seeds, 1), seed_budget)
            return capture_seeds, 0
        elif num_attack_units > 0 and num_capture_units == 0:
            capture_seeds = 0
            attack_seeds = min(max(attack_seeds, 1), seed_budget)
            return 0, attack_seeds
        else:
            return 0, 0

        min_capture = 1 if has_both and seed_budget >= 2 else 0
        min_attack = 1 if has_both and seed_budget >= 2 else 0
        total = capture_seeds + attack_seeds
        while total > seed_budget:
            if capture_seeds > attack_seeds and capture_seeds > min_capture:
                capture_seeds -= 1
            elif attack_seeds > min_attack:
                attack_seeds -= 1
            elif capture_seeds > min_capture:
                capture_seeds -= 1
            else:
                break
            total = capture_seeds + attack_seeds
        return capture_seeds, attack_seeds

    def _seed_initial_population_biases(
        self,
        population: list[RheaGenome],
        pool: ActionPool,
        *,
        include_build_bias: bool,
        make_genome: Callable[[ActionPool], RheaGenome],
    ) -> None:
        """Replace the first seeded slots with capture- and attack-biased archetypes."""
        seed_budget = max(1, len(population) // 4)
        num_capture = self._count_units_with_capture_options(pool)
        num_attack = self._count_units_with_attack_options(pool)
        capture_seeds, attack_seeds = self._compute_initial_seed_counts(
            num_capture, num_attack, seed_budget,
        )
        seed_idx = 0
        for _ in range(capture_seeds):
            genome = make_genome(pool)
            self._bias_genome_toward_capture(genome, pool)
            if include_build_bias:
                self._bias_genome_toward_expensive_builds(genome, pool)
            population[seed_idx] = genome
            seed_idx += 1
        for _ in range(attack_seeds):
            genome = make_genome(pool)
            self._bias_genome_toward_attack(genome, pool)
            if include_build_bias:
                self._bias_genome_toward_expensive_builds(genome, pool)
            population[seed_idx] = genome
            seed_idx += 1

    def _bias_genome_toward_capture(self, genome: RheaGenome, pool: ActionPool) -> None:
        """Mutate unit intents in-place toward CAPTURE only."""
        for i, intent in enumerate(genome.unit_segment):
            opts = pool.unit_options.get(intent.unit_pos)
            if not opts:
                continue
            captures = [o for o in opts if o.action_type == ActionType.CAPTURE]
            if captures and self.rng.random() < self._CAPTURE_BIAS_PROB:
                genome.unit_segment[i] = self.rng.choice(captures)

    def _bias_genome_toward_attack(self, genome: RheaGenome, pool: ActionPool) -> None:
        """Mutate unit intents in-place toward ATTACK only (not JOIN)."""
        for i, intent in enumerate(genome.unit_segment):
            opts = pool.unit_options.get(intent.unit_pos)
            if not opts:
                continue
            attacks = [
                o
                for o in opts
                if o.action_type == ActionType.ATTACK and o.move_dest != intent.unit_pos
            ]
            if attacks and self.rng.random() < self._ATTACK_BIAS_PROB:
                genome.unit_segment[i] = self.rng.choice(attacks)

    def _bias_genome_toward_expensive_builds(self, genome: RheaGenome, pool: ActionPool) -> None:
        """Mutate build segment in-place toward higher-value units.

        For each build intent, if the pool offers a unit with higher value
        at the same factory, flip the intent toward that option with high
        probability.  This gives evolution a better starting signal than
        random infantry spam.
        """
        for i, intent in enumerate(genome.build_segment):
            opts = [bi for bi in pool.build_options if bi.factory_pos == intent.factory_pos]
            if len(opts) <= 1:
                continue
            current_cost = UNIT_STATS[intent.unit_type].cost
            max_cost = max(UNIT_STATS[bi.unit_type].cost for bi in opts)
            if current_cost < max_cost and self.rng.random() < 0.88:
                best = [bi for bi in opts if UNIT_STATS[bi.unit_type].cost == max_cost]
                genome.build_segment[i] = self.rng.choice(best)

    def _crossover(self, a: RheaGenome, b: RheaGenome, pool: ActionPool) -> RheaGenome:
        def _sp_cut(seq_a: list, seq_b: list) -> list:
            if not seq_a or not seq_b:
                return list(seq_a or seq_b)
            cut = self.rng.randrange(0, min(len(seq_a), len(seq_b)) + 1)
            return list(seq_a[:cut]) + list(seq_b[cut:])

        child = RheaGenome(
            cop_activate=a.cop_activate if self.rng.random() < 0.5 else b.cop_activate,
            unit_segment=_sp_cut(a.unit_segment, b.unit_segment),
            build_segment=self._crossover_builds(a.build_segment, b.build_segment),
        )
        self._ensure_unit_segment_covers_pool(child, pool)
        return child

    def _crossover_move_only(
        self, a: RheaGenome, b: RheaGenome, pool: ActionPool
    ) -> RheaGenome:
        def _sp_cut(seq_a: list, seq_b: list) -> list:
            if not seq_a or not seq_b:
                return list(seq_a or seq_b)
            cut = self.rng.randrange(0, min(len(seq_a), len(seq_b)) + 1)
            return list(seq_a[:cut]) + list(seq_b[cut:])

        child = RheaGenome(
            cop_activate=a.cop_activate if self.rng.random() < 0.5 else b.cop_activate,
            unit_segment=_sp_cut(a.unit_segment, b.unit_segment),
            build_segment=[],
        )
        self._ensure_unit_segment_covers_pool(child, pool)
        return child

    @staticmethod
    def _crossover_builds(
        a: list[BuildIntent], b: list[BuildIntent]
    ) -> list[BuildIntent]:
        """Crossover build segments preserving one-build-per-factory.

        Merges build intents from both parents by factory, keeping only one
        intent per factory tile.  The first parent's choice is kept by default
        (tie-breaking toward the first chromosome's unit type for each factory).
        This avoids duplicate-factory intents that would otherwise waste
        evaluations on illegal genes.
        """
        merged: dict[tuple[int, int], BuildIntent] = {}
        for bi in a:
            merged[bi.factory_pos] = bi
        for bi in b:
            if bi.factory_pos not in merged:
                merged[bi.factory_pos] = bi
        return list(merged.values())

    def _mutate_move_phase_genome(self, genome: RheaGenome, pool: ActionPool) -> None:
        """CO + unit intents only (two-phase Move RHEA)."""
        mr = self.cfg.mutation_rate

        if self.rng.random() < mr:
            power_avail = pool.cop_legal or pool.scop_legal
            genome.cop_activate = power_avail and self.rng.random() < 0.5

        for intent in genome.unit_segment:
            if self.rng.random() < mr:
                opts = pool.unit_options.get(intent.unit_pos)
                if opts:
                    new_intent = self.rng.choice(opts)
                    intent.unit_pos = new_intent.unit_pos
                    intent.move_dest = new_intent.move_dest
                    intent.action_type = new_intent.action_type
                    intent.target_pos = new_intent.target_pos

        if pool.unit_options and self.rng.random() < mr * 0.3:
            upos = self.rng.choice(list(pool.unit_options.keys()))
            opts = pool.unit_options[upos]
            if opts:
                genome.unit_segment.append(self.rng.choice(opts))

        if len(genome.unit_segment) > 1 and self.rng.random() < mr * 0.2:
            idx = self.rng.randrange(len(genome.unit_segment))
            genome.unit_segment.pop(idx)

        self._ensure_unit_segment_covers_pool(genome, pool)

    def _mutate(self, genome: RheaGenome, pool: ActionPool) -> None:
        """In-place mutation for monolithic RHEA (moves + builds)."""
        self._mutate_move_phase_genome(genome, pool)
        self._mutate_build_segment_inplace(genome, pool, self.cfg.mutation_rate)

    def _mutate_build_segment_inplace(
        self, genome: RheaGenome, pool: ActionPool, mr: float
    ) -> None:
        """Monolithic-move RHEA only — evolves ``build_segment``."""
        # --- Build segment ---
        for intent in genome.build_segment:
            if self.rng.random() < mr and pool.build_options:
                same_factory = [
                    bi for bi in pool.build_options
                    if bi.factory_pos == intent.factory_pos
                ]
                if same_factory:
                    picked = _sample_build_intent_biased_expensive(
                        same_factory, self.rng, p_pick_max_cost=0.58
                    )
                    intent.unit_type = picked.unit_type

        # Add a random build gene for a factory not already in the genome.
        if pool.build_options and self.rng.random() < mr * 0.3:
            existing_factories = {bi.factory_pos for bi in genome.build_segment}
            missing = [
                bi for bi in pool.build_options
                if bi.factory_pos not in existing_factories
            ]
            if missing:
                seed = self.rng.choice(missing)
                same = [
                    bi for bi in pool.build_options
                    if bi.factory_pos == seed.factory_pos
                ]
                genome.build_segment.append(
                    _sample_build_intent_biased_expensive(
                        same, self.rng, p_pick_max_cost=0.65
                    )
                )

        # Remove a random build gene (factory will be skip-built in salvage).
        if len(genome.build_segment) > 1 and self.rng.random() < mr * 0.2:
            idx = self.rng.randrange(len(genome.build_segment))
            genome.build_segment.pop(idx)

    # ------------------------------------------------------------------
    # Genome simulation (the hot path)
    # ------------------------------------------------------------------

    def _simulate_move_phase_only(
        self,
        state: GameState,
        genome: RheaGenome,
    ) -> tuple[GameState, list, int]:
        """CO + unit intents only — no builds, no END_TURN (for two-phase planner)."""
        sim = clone_for_search(state)
        acting = int(sim.active_player)
        actions: list[Action] = []
        illegal = 0

        if genome.cop_activate and sim.winner is None and int(sim.active_player) == acting:
            co = sim.co_states[acting]
            power_action: Action | None = None
            if co.can_activate_scop():
                power_action = Action(ActionType.ACTIVATE_SCOP)
            elif co.can_activate_cop():
                power_action = Action(ActionType.ACTIVATE_COP)
            if power_action is not None:
                try:
                    sim.step(power_action)
                    actions.append(power_action)
                except Exception:
                    illegal += 1

        for intent in genome.unit_segment[: self._unit_intent_cap()]:
            if sim.winner is not None:
                break
            if int(sim.active_player) != acting:
                break

            _, il = self._execute_unit_intent(sim, intent, actions)
            illegal += il

        _ensure_select(sim)

        return sim, actions, illegal

    def _simulate_end_turn_sequence(
        self,
        sim: GameState,
        acting: int,
        actions: list,
    ) -> int:
        """WAIT unmoved scraps then END_TURN. Returns incremental illegal genes."""
        illegal = 0
        if sim.winner is not None or int(sim.active_player) != acting:
            return 0
        _ensure_select(sim)

        _p3_legal = get_legal_actions(sim)
        _p3_enders = [a for a in _p3_legal if a.action_type == ActionType.END_TURN]
        if _p3_enders:
            try:
                sim.step(_p3_enders[0])
                actions.append(_p3_enders[0])
            except Exception:
                illegal += 1
        else:
            for _u in list(sim.units[acting]):
                if not _u.is_alive or _u.moved or _u.is_stunned:
                    continue
                _p3_legal = get_legal_actions(sim)
                _p3_sel = [a for a in _p3_legal
                           if a.action_type == ActionType.SELECT_UNIT and a.unit_pos == _u.pos]
                if not _p3_sel:
                    continue
                try:
                    sim.step(_p3_sel[0])
                    actions.append(_p3_sel[0])
                except Exception:
                    continue
                _p3_legal = get_legal_actions(sim)
                _p3_moves = [a for a in _p3_legal
                             if a.action_type == ActionType.SELECT_UNIT and a.move_pos == _u.pos]
                if not _p3_moves:
                    continue
                try:
                    sim.step(_p3_moves[0])
                    actions.append(_p3_moves[0])
                except Exception:
                    continue
                if sim.action_stage == ActionStage.ACTION:
                    _p3_legal = get_legal_actions(sim)
                    _p3_waits = [a for a in _p3_legal if a.action_type == ActionType.WAIT]
                    if _p3_waits:
                        try:
                            sim.step(_p3_waits[0])
                            actions.append(_p3_waits[0])
                        except Exception:
                            pass
                    elif _p3_legal:
                        try:
                            sim.step(_p3_legal[0])
                            actions.append(_p3_legal[0])
                        except Exception:
                            pass
            _ensure_select(sim)
            _p3_legal = get_legal_actions(sim)
            _p3_enders = [a for a in _p3_legal if a.action_type == ActionType.END_TURN]
            if _p3_enders:
                try:
                    sim.step(_p3_enders[0])
                    actions.append(_p3_enders[0])
                except Exception:
                    illegal += 1
            else:
                illegal += 10
        return illegal

    def _simulate_genome(
        self,
        state: GameState,
        genome: RheaGenome,
    ) -> tuple[GameState, list, int, float]:
        """Execute a segmented genome against a clone of ``state``.

        Returns ``(final_state, actions, illegal_count, move_factory_penalty)``.
        ``move_factory_penalty`` is non-positive, from occupying own factories
        after the unit segment (before builds); see
        ``_move_phase_factory_occupation_penalty``.
        """
        sim = clone_for_search(state)
        acting = int(sim.active_player)
        actions: list[Action] = []
        illegal = 0

        # ------------------------------------------------
        # Phase 0: CO power activation (soft fail)
        # ------------------------------------------------
        if genome.cop_activate and sim.winner is None and int(sim.active_player) == acting:
            co = sim.co_states[acting]
            # Try SCOP first (stronger), then COP.
            power_action: Action | None = None
            if co.can_activate_scop():
                power_action = Action(ActionType.ACTIVATE_SCOP)
            elif co.can_activate_cop():
                power_action = Action(ActionType.ACTIVATE_COP)
            if power_action is not None:
                try:
                    sim.step(power_action)
                    actions.append(power_action)
                except Exception:
                    illegal += 1  # soft fail — power not actually available

        # ------------------------------------------------
        # Phase 1: Unit actions (soft fail per gene)
        # ------------------------------------------------
        for intent in genome.unit_segment[: self._unit_intent_cap()]:
            if sim.winner is not None:
                break
            if int(sim.active_player) != acting:
                break

            ok, il = self._execute_unit_intent(sim, intent, actions)
            illegal += il
            # Note: if ok is False, we continue (soft fail) — the genome
            # loses this unit's action but keeps executing.

        owned_factories_at_turn_start = _factories_fully_owned_positions(
            state, acting,
        )
        move_factory_occupation_penalty = _move_phase_factory_occupation_penalty(
            sim, acting, owned_factories_at_turn_start,
            float(self.cfg.buy_base_skip_penalty),
        )

        # ------------------------------------------------
        # Phase 2: Build actions (soft fail per gene)
        # Execute expensive builds first so they get the
        # remaining budget; cheap fills at the end.
        # ------------------------------------------------
        sorted_builds = sorted(
            genome.build_segment,
            key=lambda bi: UNIT_STATS[bi.unit_type].cost,
            reverse=True,
        )
        for intent in sorted_builds:
            if sim.winner is not None:
                break
            if int(sim.active_player) != acting:
                break

            ok, il = self._execute_build_intent(sim, intent, actions)
            illegal += il

        # ------------------------------------------------
        # Phase 2a: Build salvage sweep — fill factories that
        # were occupied at SELECT stage but vacated by Phase 1
        # movement.  The genome couldn't encode these builds
        # because ActionPool.build_options only captured
        # factories that were already empty.
        # ------------------------------------------------
        if sim.winner is None and int(sim.active_player) == acting:
            sim.action_stage = ActionStage.SELECT
            sim.selected_unit = None
            sim.selected_move_pos = None

            # Factory positions already built at by the genome.
            built_factories: set[tuple[int, int]] = set()
            for a in actions:
                if getattr(a, 'action_type', None) == ActionType.BUILD and a.move_pos is not None:
                    built_factories.add(a.move_pos)

            # Re-query for BUILD actions — Phase 1 may have vacated some factories.
            extra_legal = get_legal_actions(sim)
            # Pick cheapest unit per newly-vacated factory so salvage is a
            # fallback, not a superior alternative to encoding builds.
            salvage_builds_by_factory: dict[tuple[int, int], Action] = {}
            for a in extra_legal:
                if (a.action_type == ActionType.BUILD
                    and a.move_pos is not None
                    and a.move_pos not in built_factories
                ):
                    existing = salvage_builds_by_factory.get(a.move_pos)
                    if (existing is None
                        or UNIT_STATS[a.unit_type].cost < UNIT_STATS[existing.unit_type].cost
                    ):
                        salvage_builds_by_factory[a.move_pos] = a

            # Sort by cost ascending to fill more factories before funds deplete.
            salvage_builds = sorted(
                salvage_builds_by_factory.values(),
                key=lambda a: UNIT_STATS[a.unit_type].cost,
            )
            for a in salvage_builds:
                try:
                    sim.step(a)
                    actions.append(a)
                except Exception:
                    pass  # soft fail — funds consumed by earlier salvage build

        # ------------------------------------------------
        # Phase 3: END_TURN if still acting and legal
        # ------------------------------------------------
        if sim.winner is None and int(sim.active_player) == acting:
            illegal += self._simulate_end_turn_sequence(sim, acting, actions)

        return sim, actions, illegal, move_factory_occupation_penalty

    def _execute_buy_spend_allocation(
        self,
        post_moves_template: GameState,
        bm: BuySpendGenome,
        *,
        mutate_sim: GameState | None,
    ) -> tuple[list, BuySpendGenome, GameState, int, float, float]:
        """Execute ordered factory purchases left→RIGHT with budget clamp.

        Runs on ``mutate_sim`` when provided (in-place); otherwise clones.

        Returns:
            ``(build_actions, effective_genome, final_sim_or_mutate_sim, illegal,
               r_skip, r_excess)``
        """
        sim = mutate_sim if mutate_sim is not None else clone_for_search(post_moves_template)
        acting = int(sim.active_player)
        build_actions: list[Action] = []
        illegal = 0

        gold_available = int(sim.funds[acting])
        r_skip = 0.0
        factories_affordable_at_start = 0
        clawed_ch: list[Optional[UnitType]] = []

        # Count how many factories were affordable before any spending
        for _fact_i, factory_pos in enumerate(bm.factory_order):
            _ensure_select(sim)
            legal_pre = get_legal_actions(sim)
            builds_pre = [
                a for a in legal_pre
                if a.action_type == ActionType.BUILD
                and a.move_pos is not None
                and (int(a.move_pos[0]), int(a.move_pos[1])) == factory_pos
            ]
            mn_pre = (
                min(UNIT_STATS[a.unit_type].cost for a in builds_pre) if builds_pre else None
            )
            if mn_pre is not None and mn_pre <= gold_available:
                factories_affordable_at_start += 1

        # Reset sim to beginning state for actual build execution
        sim = mutate_sim if mutate_sim is not None else clone_for_search(post_moves_template)
        acting = int(sim.active_player)
        build_actions: list[Action] = []
        illegal = 0

        for slot_i, factory_pos in enumerate(bm.factory_order):
            gene = bm.choices[slot_i] if slot_i < len(bm.choices) else None
            if sim.winner is not None or int(sim.active_player) != acting:
                break

            _ensure_select(sim)
            legal_now = get_legal_actions(sim)
            builds_here = [
                a for a in legal_now
                if a.action_type == ActionType.BUILD
                and a.move_pos is not None
                and (int(a.move_pos[0]), int(a.move_pos[1])) == factory_pos
            ]
            coins = int(sim.funds[acting])

            want = gene
            if want is not None:
                cand = [
                    a for a in builds_here
                    if a.unit_type == want
                ]
                uc = UNIT_STATS[want].cost
                if not cand or uc > coins:
                    want = None  # clamp to SKIP semantics

            if want is None:
                clawed_ch.append(None)
                continue

            cand_take = [a for a in builds_here if a.unit_type == want]
            if not cand_take:
                illegal += 1
                clawed_ch.append(None)
                continue
            try:
                sim.step(cand_take[0])
                build_actions.append(cand_take[0])
                clawed_ch.append(want)
            except Exception:
                illegal += 1
                clawed_ch.append(None)

        # Post-hoc skip penalty: factories affordable at start but not used.
        factories_used = len(build_actions)
        factories_skipped = max(0, factories_affordable_at_start - factories_used)
        r_skip = -float(self.cfg.buy_base_skip_penalty) * float(factories_skipped)

        while len(clawed_ch) < len(bm.factory_order):
            clawed_ch.append(None)
        eff = BuySpendGenome(bm.factory_order, clawed_ch)

        gold_after = int(sim.funds[acting])
        sr = float(self.cfg.buy_safe_reserve)
        spendable = max(0.0, float(gold_available) - sr)
        if spendable <= 0.0:
            r_excess = 0.0
        else:
            wasted_frac = max(0.0, float(gold_after) - sr) / spendable
            r_excess = -float(self.cfg.buy_gold_hoard_penalty) * wasted_frac
            # Absolute penalty per 1k gold hoarded above safe reserve
            excess_1k = max(0.0, float(gold_after) - sr) / 1000.0
            r_excess -= float(self.cfg.buy_gold_abs_penalty_per_1k) * excess_1k

        # Bank credit: remaining gold (capped) contributes positive reward.
        # E.g. 1 infantry + 6k bank can compete with 2 infantry + 0k bank.
        bank_capped = min(float(self.cfg.buy_bank_credit_cap), float(gold_after))
        bank_1k = bank_capped / 1000.0
        r_bank = float(self.cfg.buy_bank_credit_per_1k) * bank_1k

        return build_actions, eff, sim, illegal, r_skip, r_excess + r_bank

    def _canonical_buy_genome_after_exec(
        self,
        snapshot: GameState,
        raw: BuySpendGenome,
    ) -> BuySpendGenome:
        """Clamp / budget-revalidate purchases by simulating forward once."""
        _, ef, *_ = self._execute_buy_spend_allocation(snapshot, raw, mutate_sim=None)
        return ef

    def _crossover_buy_spend(
        self,
        fo: tuple[tuple[int, int], ...],
        a: BuySpendGenome,
        b: BuySpendGenome,
        post_moves: GameState,
    ) -> BuySpendGenome:
        n = len(fo)
        if n <= 1:
            return self._canonical_buy_genome_after_exec(post_moves, copy.deepcopy(a))
        cut = self.rng.randint(1, n - 1)
        merged = list(a.choices[:cut]) + list(b.choices[cut:])
        while len(merged) < n:
            merged.append(None)
        return self._canonical_buy_genome_after_exec(
            post_moves, BuySpendGenome(fo, merged[:n])
        )

    def _mutate_buy_spend_genome(
        self,
        fo: tuple[tuple[int, int], ...],
        catalog: dict[tuple[int, int], tuple[UnitType, ...]],
        gm: BuySpendGenome,
        post_moves: GameState,
    ) -> BuySpendGenome:
        ch = list(gm.choices)
        if len(ch) != len(fo):
            ch = list(gm.choices[: len(fo)]) + [None] * max(0, len(fo) - len(ch))
            ch = ch[: len(fo)]
        if len(fo) >= 2 and self.rng.random() < 0.45:
            i, j = self.rng.sample(range(len(fo)), k=2)
            ch[i], ch[j] = ch[j], ch[i]
        else:
            idx = self.rng.randrange(len(fo))
            opts = list(catalog[fo[idx]])
            picks = opts + [None]
            ch[idx] = self.rng.choice(picks)
        return self._canonical_buy_genome_after_exec(
            post_moves, BuySpendGenome(fo, ch)
        )

    def _greedy_cheapest_buy_genome(
        self,
        post_moves: GameState,
        acting: int,
        fo: tuple[tuple[int, int], ...],
        catalog: dict[tuple[int, int], tuple[UnitType, ...]],
    ) -> BuySpendGenome | None:
        """Cheapest affordable build per factory left-to-right (for seeding / fallback).

        Returns ``None`` if no sequence of catalog builds can be afforded from
        ``post_moves.funds`` in factory order.
        """
        if not fo or not catalog:
            return None
        try:
            min_uc = min(
                UNIT_STATS[u].cost
                for p in fo
                for u in catalog.get(p, ())
            )
        except ValueError:
            return None
        if int(post_moves.funds[acting]) < min_uc:
            return None
        sim = clone_for_search(post_moves)
        choices: list[Optional[UnitType]] = []
        for fac_pos in fo:
            if sim.winner is not None or int(sim.active_player) != acting:
                choices.append(None)
                continue
            _ensure_select(sim)
            allowed = catalog.get(fac_pos, ())
            if not allowed:
                choices.append(None)
                continue
            allowed_s = set(allowed)
            legal = get_legal_actions(sim)
            coins = int(sim.funds[acting])
            affordable: list[Action] = []
            for a in legal:
                if a.action_type != ActionType.BUILD:
                    continue
                if a.move_pos is None or a.unit_type is None:
                    continue
                if (int(a.move_pos[0]), int(a.move_pos[1])) != fac_pos:
                    continue
                if a.unit_type not in allowed_s:
                    continue
                if UNIT_STATS[a.unit_type].cost <= coins:
                    affordable.append(a)
            if not affordable:
                choices.append(None)
                continue
            pick_a = min(affordable, key=lambda a: UNIT_STATS[a.unit_type].cost)
            try:
                sim.step(pick_a)
                choices.append(pick_a.unit_type)
            except Exception:
                choices.append(None)
        raw = BuySpendGenome(fo, choices)
        return self._canonical_buy_genome_after_exec(post_moves, raw)

    def _execute_unit_intent(
        self,
        sim: GameState,
        intent: UnitIntent,
        actions: list,
    ) -> tuple[bool, int]:
        """Try to execute one UnitIntent against the simulation state.

        Returns (success, illegal_increment).  Actions are always sourced from
        ``get_legal_actions(sim)`` so the step gate never rejects them.
        """
        # 1. Verify unit exists at the encoded position and is unmoved.
        unit = sim.get_unit_at(*intent.unit_pos)
        if unit is None or unit.moved or unit.is_stunned:
            return False, 1

        # 2. Verify the selected unit is still selectable (SELECT stage).
        if sim.action_stage != ActionStage.SELECT:
            return False, 1

        # 3. Select the unit — look up from legal actions.
        legal = get_legal_actions(sim)
        sel_cands = [a for a in legal
                     if a.action_type == ActionType.SELECT_UNIT and a.unit_pos == intent.unit_pos]
        if not sel_cands:
            return False, 1
        try:
            sim.step(sel_cands[0])
            actions.append(sel_cands[0])
        except Exception:
            return False, 1

        # 4. We are now at MOVE stage. Verify the move destination is reachable.
        if sim.action_stage != ActionStage.MOVE:
            return False, 1

        legal = get_legal_actions(sim)
        move_cands = [a for a in legal
                      if a.action_type == ActionType.SELECT_UNIT and a.move_pos == intent.move_dest]
        if not move_cands:
            return False, 1
        try:
            sim.step(move_cands[0])
            actions.append(move_cands[0])
        except Exception:
            return False, 1

        # 5. Now at ACTION stage. Execute the terminal action.
        if sim.action_stage != ActionStage.ACTION:
            return True, 0

        legal = get_legal_actions(sim)
        term_cands = [a for a in legal
                      if a.action_type == intent.action_type
                      and a.unit_pos == intent.unit_pos
                      and (intent.target_pos is None or a.target_pos == intent.target_pos)]
        # Fallback: match on action_type only if unit_pos/target_pos are None
        if not term_cands:
            term_cands = [a for a in legal if a.action_type == intent.action_type]
        if term_cands:
            try:
                sim.step(term_cands[0])
                actions.append(term_cands[0])
            except Exception:
                pass

        # CRITICAL: The terminal action may have failed silently, leaving the sim
        # in ACTION stage with selected_unit still set.  Phase 2 (builds) requires
        # SELECT stage, so force-reset here.  The unit's movement was already
        # consumed by step 4; the failed terminal does not leave the unit movable.
        if sim.action_stage != ActionStage.SELECT:
            sim.action_stage = ActionStage.SELECT
            sim.selected_unit = None
            sim.selected_move_pos = None

        return True, 0

    def _execute_build_intent(
        self,
        sim: GameState,
        intent: BuildIntent,
        actions: list,
    ) -> tuple[bool, int]:
        """Try to execute one BuildIntent against the simulation state.

        If the exact unit type is not affordable (funds exhausted by earlier
        builds or availability changed), falls back to the most expensive
        still-affordable unit at the same factory.  This prevents RHEA from
        learning that expensive builds cause illegal penalties — the genome
        always gets the best build the budget allows.

        Returns (success, illegal_increment).
        """
        if sim.action_stage != ActionStage.SELECT:
            return False, 1

        legal = get_legal_actions(sim)

        # Try the exact intent first.
        build_cands = [a for a in legal
                       if a.action_type == ActionType.BUILD
                       and a.move_pos == intent.factory_pos
                       and a.unit_type == intent.unit_type]
        if build_cands:
            try:
                sim.step(build_cands[0])
                actions.append(build_cands[0])
                return True, 0
            except Exception:
                pass  # fall through to best-affordable fallback

        # Exact intent failed (budget exhausted, tile occupied, etc.).
        # Fall back to the most-expensive still-affordable unit at this
        # factory.  This is NOT an illegal gene — the intent was sound,
        # just over-budget after earlier builds consumed funds.
        same_factory = [a for a in legal
                        if a.action_type == ActionType.BUILD
                        and a.move_pos == intent.factory_pos]
        if not same_factory:
            return False, 0  # no fallback available, not penalised
        # Pick the most expensive affordable unit as the fallback.
        best = max(same_factory, key=lambda a: UNIT_STATS[a.unit_type].cost)
        try:
            sim.step(best)
            actions.append(best)
            return True, 0
        except Exception:
            return False, 0  # truly stuck, not penalised

    # ------------------------------------------------------------------
    # Surviving helpers (unchanged)
    # ------------------------------------------------------------------

    def _rank_candidates_cheap(
        self,
        state: GameState,
        cands: list[CandidateAction],
    ) -> list[CandidateAction]:
        """Cheap ordering — used by the tactical beam, not by genome sim."""
        scored: list[tuple[float, CandidateAction]] = []
        for c in cands:
            f = c.preview
            score = 0.0
            if f is not None:
                score += 2.0 * float(f[10])
                score += 1.0 * float(f[8])
                score += 0.25 * float(f[11])
                score += 1.25 * float(f[17])
                score += 0.75 * float(f[16])
                score -= 0.80 * float(f[19])
                score += 0.50 * float(f[21])
                score -= 1.00 * float(f[22])
            if (
                c.kind == CandidateKind.END_TURN
                or c.terminal_action.action_type.name == "END_TURN"
            ):
                score -= 0.5
            if c.kind == CandidateKind.MOVE_WAIT:
                score -= 0.1
            scored.append((score, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[: self.cfg.top_k_per_state]]

    def _apply_candidate(self, state: GameState, cand: CandidateAction) -> bool:
        """Apply a CandidateAction — used by tactical beam only."""
        try:
            state.step(cand.first)
            if cand.second is not None and state.winner is None:
                state.step(cand.second)
            return True
        except Exception:
            return False

    @staticmethod
    def compute_complexity_metrics(
        state: GameState, observer_seat: int
    ) -> tuple[int, int, int, int, int]:
        """Compute game state complexity metrics for dynamic budgeting."""
        enemy_seat = 1 - observer_seat
        owned_units = sum(1 for u in state.units[observer_seat] if u.is_alive)
        factories = 0
        for prop in state.properties:
            if prop.owner == observer_seat and (
                getattr(prop, "is_base", False)
                or getattr(prop, "is_airport", False)
                or getattr(prop, "is_port", False)
            ):
                factories += 1
        contested_captures = 0
        for prop in state.properties:
            if prop.capture_points < 20:
                capturing_unit = None
                for player in (observer_seat, enemy_seat):
                    for unit in state.units[player]:
                        if unit.is_alive and unit.pos == (prop.row, prop.col):
                            capturing_unit = unit
                            break
                    if capturing_unit:
                        break
                if capturing_unit:
                    contested_captures += 1
        enemy_in_range_contacts = 0
        try:
            from engine.threat import compute_influence_planes
            t_me, t_en, r_me, r_en, c_me, c_en = compute_influence_planes(
                state, me=observer_seat, grid=30
            )
            enemy_units = state.units[enemy_seat]
            for unit in enemy_units:
                if unit.is_alive and t_me[unit.pos[0], unit.pos[1]] > 0:
                    enemy_in_range_contacts += 1
        except Exception:
            enemy_in_range_contacts = len(state.units[enemy_seat]) // 4
        juicy_attacks = RheaPlanner._count_juicy_attacks(state, observer_seat)
        return (
            owned_units,
            factories,
            contested_captures,
            enemy_in_range_contacts,
            juicy_attacks,
        )

    @staticmethod
    def _count_juicy_attacks(state: GameState, observer_seat: int) -> int:
        """Count high-value attack intents available on the current board."""
        count = 0
        old_stage = state.action_stage
        old_selected = state.selected_unit
        old_move = state.selected_move_pos
        try:
            for unit in state.units[observer_seat]:
                if not unit.is_alive or getattr(unit, "has_acted", False):
                    continue
                state.action_stage = ActionStage.MOVE
                state.selected_unit = unit
                state.selected_move_pos = None
                for cand in enumerate_candidates(state):
                    if cand.kind != CandidateKind.MOVE_ATTACK or cand.preview is None:
                        continue
                    f = cand.preview
                    if (
                        float(f[21]) >= 1.0
                        or float(f[17]) >= 0.12
                        or (float(f[13]) >= 0.55 and float(f[19]) <= 0.20)
                    ):
                        count += 1
                        break
        except Exception:
            return 0
        finally:
            state.action_stage = old_stage
            state.selected_unit = old_selected
            state.selected_move_pos = old_move
        return count