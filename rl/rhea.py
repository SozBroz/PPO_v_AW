"""
Segmented-genome RHEA (Rolling Horizon Evolution Algorithm).

Genome structure replaces the flat ``list[int]`` of ranked candidate indices with
a typed segmented genome:

    RheaGenome = {
        cop_activate: bool,               // optional power activation
        unit_segment: list[UnitIntent],   // one per unit action
        build_segment: list[BuildIntent], // one per factory
    }

Each ``UnitIntent`` directly encodes the full SELECT → MOVE → ACTION cycle for
one unit, so the executor never depends on a fragile positional index into a
dynamically-regenerated candidate list.  Illegal genes are **soft-failed**
(skipped with an increment of the illegal counter) instead of aborting the
entire genome.

Key design properties:
  - No ``max_actions_per_turn`` — the genome is naturally bounded by the number
    of unmoved units + owned factories.
  - Builds are separated from unit actions, so they never pollute the
    SELECT-stage candidate pool.
  - CO power activation is a separate phase so Eagle-style SCOP interleaving
    (move some units → activate SCOP → move them again) is straightforward.
  - Variable-length ``unit_segment`` — crossover and mutation add/remove intents
    within reasonable bounds.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import random
from typing import Optional

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


def dynamic_rhea_budget(
    owned_units: int,
    factories: int,
    contested_captures: int,
    enemy_in_range_contacts: int,
) -> tuple[int, int]:
    """
    Compute dynamic RHEA search budget based on game state complexity.

    Returns: (population, generations) — no max_actions_per_turn, the genome
    is self-sizing from the number of unmoved units + factories.
    """
    complexity = (
        owned_units
        + 1.5 * factories
        + 2.0 * contested_captures
        + 0.5 * enemy_in_range_contacts
    )

    pop = int(8 + 1.2 * complexity)
    gen = int(2 + complexity / 8)

    pop = max(12, min(pop, 64))
    gen = max(2, min(gen, 7))

    return pop, gen


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RheaConfig:
    population: int = 32
    generations: int = 6
    elite: int = 4
    mutation_rate: float = 0.20
    # max_actions_per_turn removed — genome is now self-sizing based on
    # unmoved units + factories in the before-state.
    top_k_per_state: int = 24
    reward_weight: float = 0.90
    value_weight: float = 0.10
    # Logging/eval knobs.
    log_initial_best: bool = True
    seed: Optional[int] = None
    # Tactical beam config (unchanged)
    use_tactical_beam: bool = False
    tactial_beam_max_width: int = 48
    tactial_beam_max_depth: int = 14
    tactial_beam_max_expand: int = 24


@dataclass(slots=True)
class RheaResult:
    actions: list
    score: float
    breakdown: RheaFitnessBreakdown
    illegal_genes: int
    generations: int
    initial_best_score: float | None = None
    evolved_gain: float | None = None


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

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def choose_full_turn(self, state: GameState) -> RheaResult:
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
            owned_units, factories, contested_captures, enemy_in_range_contacts = (
                self.complexity_metrics
            )
            pop, gen = dynamic_rhea_budget(
                owned_units, factories, contested_captures, enemy_in_range_contacts
            )
            population_size = pop
            generations = gen

        # Build the action pool ONCE from the before-state.
        action_pool = ActionPool.build(before)

        # Initialise population from the pool.
        population = [
            self._random_genome(action_pool) for _ in range(population_size)
        ]
        rhea_best = None
        initial_best_score: float | None = None

        for _gen in range(generations):
            after_states: list[tuple] = []

            for genome in population:
                after, actions, illegal = self._simulate_genome(before, genome)
                after_states.append((after, actions, illegal))

            # Batch value evaluation.
            all_states = [after for after, _, _ in after_states]
            all_values = self.fitness.batch_value(all_states, acting_seat)
            before_value = self.fitness.value(before, acting_seat)

            scored: list = []
            for idx, ((after, actions, illegal), v_after) in enumerate(
                zip(after_states, all_values)
            ):
                phi_after = self.fitness.phi(after, acting_seat)
                phi_before = self.fitness.phi(before, acting_seat)
                phi_delta = phi_after - phi_before

                win_advantage = (v_after - before_value) * 2.0
                illegal_penalty = -self.fitness.illegal_gene_penalty * float(illegal)

                # Build penalty computation (unchanged logic).
                build_punishment, unused_funds_penalty = self._compute_build_penalties(
                    before, actions, acting_seat
                )

                total = (
                    self.fitness.reward_weight * phi_delta
                    + self.fitness.value_weight * win_advantage
                    + illegal_penalty
                    + build_punishment
                    + unused_funds_penalty
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
                child = self._crossover(p1, p2)
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
            )

    # ------------------------------------------------------------------
    # Build penalty helper (extracted, unchanged logic)
    # ------------------------------------------------------------------

    def _compute_build_penalties(
        self,
        before: GameState,
        actions: list | None,
        acting_seat: int,
    ) -> tuple[float, float]:
        """Return (build_punishment, unused_funds_penalty)."""
        if actions is None:
            return 0.0, 0.0

        end_turn_index = None
        build_happened = False
        build_actions: list = []
        units_built = 0
        total_build_cost = 0

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

        build_punishment = 0.0
        unused_funds_penalty = 0.0

        if end_turn_index is not None and not build_happened:
            if self.fitness.env_template._player_has_bases(before, acting_seat):
                bp = self.fitness.env_template._build_punishment
                if bp > 0.0:
                    build_punishment = -30000.0 * self.fitness.env_template._phi_alpha

        if build_happened and end_turn_index is not None:
            if self.fitness.env_template._player_has_bases(before, acting_seat):
                base_count = 0
                for prop in before.properties:
                    if prop.owner == acting_seat and (
                        getattr(prop, "is_base", False)
                        or getattr(prop, "is_airport", False)
                        or getattr(prop, "is_port", False)
                    ):
                        base_count += 1
                if base_count > 0:
                    available_funds = before.funds[acting_seat]
                    funds_needed = base_count * 1000
                    if available_funds >= funds_needed:
                        if units_built < base_count:
                            missing = base_count - units_built
                            unused_funds_penalty = -0.05 * missing
                    else:
                        max_affordable = available_funds // 1000
                        if units_built < max_affordable:
                            missing = max_affordable - units_built
                            unused_funds_penalty = -0.05 * missing

        return build_punishment, unused_funds_penalty

    # ------------------------------------------------------------------
    # Genome operations
    # ------------------------------------------------------------------

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
        # Add about half the available builds (leaves room for evolution).
        pool_builds = list(pool.build_options)
        self.rng.shuffle(pool_builds)
        n_builds = max(0, len(pool_builds) // 2) if pool_builds else 0
        n_builds = max(1, n_builds) if pool_builds else 0
        build_segment = pool_builds[:n_builds]

        return RheaGenome(
            cop_activate=cop_activate,
            unit_segment=unit_segment,
            build_segment=build_segment,
        )

    def _crossover(self, a: RheaGenome, b: RheaGenome) -> RheaGenome:
        """Single-point crossover within each segment independently."""
        def _sp_cut(seq_a: list, seq_b: list) -> list:
            if not seq_a or not seq_b:
                return list(seq_a or seq_b)
            cut = self.rng.randrange(0, min(len(seq_a), len(seq_b)) + 1)
            return list(seq_a[:cut]) + list(seq_b[cut:])

        child = RheaGenome(
            cop_activate=a.cop_activate if self.rng.random() < 0.5 else b.cop_activate,
            unit_segment=_sp_cut(a.unit_segment, b.unit_segment),
            build_segment=_sp_cut(a.build_segment, b.build_segment),
        )
        return child

    def _mutate(self, genome: RheaGenome, pool: ActionPool) -> None:
        """In-place mutation of all segments using the action pool.

        Each gene field is independently resampled from the pool with
        probability ``mutation_rate``.  Segment length can also change
        (add or remove a gene) with a smaller probability.
        """
        mr = self.cfg.mutation_rate

        # --- CO power ---
        if self.rng.random() < mr:
            power_avail = pool.cop_legal or pool.scop_legal
            genome.cop_activate = power_avail and self.rng.random() < 0.5

        # --- Unit segment ---
        # Mutate each UnitIntent field.
        for intent in genome.unit_segment:
            if self.rng.random() < mr:
                # Re-roll the entire intent from pool options for this unit.
                opts = pool.unit_options.get(intent.unit_pos)
                if opts:
                    new_intent = self.rng.choice(opts)
                    intent.unit_pos = new_intent.unit_pos
                    intent.move_dest = new_intent.move_dest
                    intent.action_type = new_intent.action_type
                    intent.target_pos = new_intent.target_pos

        # Add a random unit gene (small probability).
        if pool.unit_options and self.rng.random() < mr * 0.3:
            upos = self.rng.choice(list(pool.unit_options.keys()))
            opts = pool.unit_options[upos]
            if opts:
                genome.unit_segment.append(self.rng.choice(opts))

        # Remove a random unit gene (small probability, only if >1).
        if len(genome.unit_segment) > 1 and self.rng.random() < mr * 0.2:
            idx = self.rng.randrange(len(genome.unit_segment))
            genome.unit_segment.pop(idx)

        # --- Build segment ---
        for intent in genome.build_segment:
            if self.rng.random() < mr and pool.build_options:
                new_intent = self.rng.choice(pool.build_options)
                intent.factory_pos = new_intent.factory_pos
                intent.unit_type = new_intent.unit_type

        # Add a random build gene.
        if pool.build_options and self.rng.random() < mr * 0.3:
            genome.build_segment.append(self.rng.choice(pool.build_options))

        # Remove a random build gene.
        if len(genome.build_segment) > 0 and self.rng.random() < mr * 0.2:
            idx = self.rng.randrange(len(genome.build_segment))
            genome.build_segment.pop(idx)

    # ------------------------------------------------------------------
    # Genome simulation (the hot path)
    # ------------------------------------------------------------------

    def _simulate_genome(
        self,
        state: GameState,
        genome: RheaGenome,
    ) -> tuple[GameState, list, int]:
        """Execute a segmented genome against a clone of ``state``.

        Returns (final_state, actions, illegal_count) — same contract as the
        old _simulate_genome.
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
        for intent in genome.unit_segment:
            if sim.winner is not None:
                break
            if int(sim.active_player) != acting:
                break

            ok, il = self._execute_unit_intent(sim, intent, actions)
            illegal += il
            # Note: if ok is False, we continue (soft fail) — the genome
            # loses this unit's action but keeps executing.

        # ------------------------------------------------
        # Phase 2: Build actions (soft fail per gene)
        # ------------------------------------------------
        for intent in genome.build_segment:
            if sim.winner is not None:
                break
            if int(sim.active_player) != acting:
                break

            ok, il = self._execute_build_intent(sim, intent, actions)
            illegal += il

        # ------------------------------------------------
        # Phase 3: END_TURN if still acting and legal
        # ------------------------------------------------
        if sim.winner is None and int(sim.active_player) == acting:
            feats, mask, cands = candidate_arrays(sim, max_candidates=MAX_CANDIDATES)
            legal = [c for i, c in enumerate(cands) if i < len(mask) and bool(mask[i])]
            enders = [
                c for c in legal
                if c.kind == CandidateKind.END_TURN
                or getattr(getattr(c.terminal_action, 'action_type', None), 'name', '') == "END_TURN"
            ]
            if enders:
                try:
                    sim.step(enders[0].first)
                    actions.append(enders[0].terminal_action)
                except Exception:
                    illegal += 1

        return sim, actions, illegal

    def _execute_unit_intent(
        self,
        sim: GameState,
        intent: UnitIntent,
        actions: list,
    ) -> tuple[bool, int]:
        """Try to execute one UnitIntent against the simulation state.

        Returns (success, illegal_increment).
        """
        # 1. Verify unit exists at the encoded position and is unmoved.
        unit = sim.get_unit_at(*intent.unit_pos)
        if unit is None or unit.moved or unit.is_stunned:
            return False, 1

        # 2. Verify the selected unit is still selectable (SELECT stage).
        if sim.action_stage != ActionStage.SELECT:
            return False, 1

        # 3. Select the unit.
        select_action = Action(ActionType.SELECT_UNIT, unit_pos=intent.unit_pos)
        try:
            sim.step(select_action)
            actions.append(select_action)
        except Exception:
            return False, 1

        # 4. We are now at MOVE stage. Verify the move destination is reachable.
        if sim.action_stage != ActionStage.MOVE:
            return False, 1

        # Move must use SELECT_UNIT with move_pos = intent.move_dest.
        move_action = Action(ActionType.SELECT_UNIT, move_pos=intent.move_dest)
        try:
            sim.step(move_action)
            actions.append(move_action)
        except Exception:
            return False, 1

        # 5. Now at ACTION stage. Execute the terminal action.
        if sim.action_stage != ActionStage.ACTION:
            # If the unit cannot perform the intended action (e.g. MOVE_WAIT
            # with no WAIT legal), the MOVE stage action still consumed the
            # unit's turn (it moved and waited).  This is not an illegal gene
            # in the genome — the unit's turn was used productively.
            return True, 0

        term_action = Action(
            intent.action_type,
            unit_pos=intent.unit_pos,
            target_pos=intent.target_pos,
        )
        try:
            sim.step(term_action)
            actions.append(term_action)
        except Exception:
            # The move succeeded; the terminal action failed.  The unit has
            # already moved, so we don't mark this as illegal per se.
            pass

        return True, 0

    def _execute_build_intent(
        self,
        sim: GameState,
        intent: BuildIntent,
        actions: list,
    ) -> tuple[bool, int]:
        """Try to execute one BuildIntent against the simulation state.

        Returns (success, illegal_increment).
        """
        if sim.action_stage != ActionStage.SELECT:
            return False, 1

        build_action = Action(
            ActionType.BUILD,
            move_pos=intent.factory_pos,
            unit_type=intent.unit_type,
        )
        try:
            sim.step(build_action)
            actions.append(build_action)
            return True, 0
        except Exception:
            return False, 1

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
    ) -> tuple[int, int, int, int]:
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
        return owned_units, factories, contested_captures, enemy_in_range_contacts