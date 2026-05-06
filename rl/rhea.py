from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Optional

from engine.search_clone import clone_for_search
from engine.game import GameState
from rl.candidate_actions import (
    MAX_CANDIDATES,
    CandidateAction,
    CandidateKind,
    candidate_arrays,
)
from rl.rhea_fitness import RheaFitness, RheaFitnessBreakdown
from rl.candidate_actions import candidate_arrays, CandidateKind
from rl.tactical_beam import TacticalBeamConfig, TacticalBeamPlanner

# Cython acceleration for hot loops
USE_CYTHON_RHEA = True
try:
    from rl._rhea_cython import (
        simulate_genome_cython,
        crossover_cython,
        mutate_cython,
        random_genome_cython,
    )
except ImportError:
    USE_CYTHON_RHEA = False


def dynamic_rhea_budget(
    owned_units: int,
    factories: int,
    contested_captures: int,
    enemy_in_range_contacts: int,
) -> tuple[int, int, int]:
    """
    Compute dynamic RHEA search budget based on game state complexity.
    
    Returns: (population, generations, max_actions_per_turn)
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

    max_actions = int(32 + 5 * owned_units + 8 * factories)
    max_actions = max(48, min(max_actions, 240))

    return pop, gen, max_actions


@dataclass(slots=True)
class RheaConfig:
    population: int = 32
    generations: int = 6
    elite: int = 4
    mutation_rate: float = 0.20
    max_actions_per_turn: int = 128
    top_k_per_state: int = 24
    reward_weight: float = 0.90
    value_weight: float = 0.10
    # Logging/eval knobs. These do not change search semantics, but make it
    # easier to detect when evolution is merely expensive sorting.
    log_initial_best: bool = True
    seed: Optional[int] = None
    # Tactical beam config
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


class RheaPlanner:
    def __init__(
        self, 
        fitness: RheaFitness, 
        config: RheaConfig,
        dynamic_budget: bool = False,
        complexity_metrics: Optional[tuple[int, int, int, int]] = None,
    ) -> None:
        """
        Args:
            fitness: RheaFitness instance for scoring
            config: RheaConfig with search parameters
            dynamic_budget: If True, compute budget dynamically from complexity_metrics
            complexity_metrics: Tuple of (owned_units, factories, contested_captures, enemy_in_range_contacts)
        """
        self.fitness = fitness
        self.cfg = config
        self.rng = random.Random(config.seed)
        self.dynamic_budget = dynamic_budget
        self.complexity_metrics = complexity_metrics
        # Initialize tactical beam planner if enabled
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

    def choose_full_turn(self, state: GameState) -> RheaResult:
        acting_seat = int(state.active_player)
        before = state
        
        # Run tactical beam if enabled
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
        
        # Use dynamic budgeting if enabled and metrics are provided
        population_size = self.cfg.population
        generations = self.cfg.generations
        max_actions_per_turn = self.cfg.max_actions_per_turn
        
        if self.dynamic_budget and self.complexity_metrics is not None:
            owned_units, factories, contested_captures, enemy_in_range_contacts = self.complexity_metrics
            pop, gen, max_acts = dynamic_rhea_budget(
                owned_units, factories, contested_captures, enemy_in_range_contacts
            )
            population_size = pop
            generations = gen
            max_actions_per_turn = max_acts
        
        population = [self._random_genome(before, max_actions_per_turn) for _ in range(population_size)]
        rhea_best = None
        initial_best_score: float | None = None
        
        for _gen in range(generations):
            scored = []
            for genome in population:
                after, actions, illegal = self._simulate_genome(before, genome)
                breakdown = self.fitness.score(
                    before,
                    after,
                    observer_seat=acting_seat,
                    illegal_genes=illegal,
                    actions=actions,
                )
                scored.append((breakdown.total, genome, actions, breakdown, illegal))
            
            scored.sort(key=lambda x: x[0], reverse=True)
            
            if _gen == 0:
                initial_best_score = float(scored[0][0])
            
            if rhea_best is None or scored[0][0] > rhea_best[0]:
                rhea_best = scored[0]
            
            elites = scored[: max(1, self.cfg.elite)]
            next_pop: list[list[int]] = [list(g) for _, g, _, _, _ in elites]
            
            while len(next_pop) < population_size:
                p1 = self.rng.choice(elites)[1]
                p2 = self.rng.choice(elites)[1]
                child = self._crossover(p1, p2)
                self._mutate(child)
                next_pop.append(child)
            
            population = next_pop
        
        # Compare beam and RHEA, return best
        candidates = []
        if beam_best:
            candidates.append((beam_best['score'], None, beam_best['actions'], beam_best['breakdown'], beam_best['illegal']))
        if rhea_best:
            candidates.append(rhea_best)
        
        if not candidates:
            # Fallback: return empty result
            return RheaResult(
                actions=[],
                score=0.0,
                breakdown=RheaFitnessBreakdown(phi_delta=0.0, value=0.0, illegal_penalty=0.0, total=0.0),
                illegal_genes=0,
                generations=generations,
                initial_best_score=None,
                evolved_gain=None,
            )
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, _, best_actions, best_breakdown, best_illegal = candidates[0]
        
        # Determine if best is from beam or RHEA
        if beam_best and best_score == beam_best['score']:
            # Beam result is best
            gain = None if initial_best_score is None else float(best_score) - float(initial_best_score)
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
            # RHEA result is best
            score, _genome, actions, breakdown, illegal = candidates[0]
            gain = None if initial_best_score is None else float(score) - float(initial_best_score)
            return RheaResult(
                actions=actions,
                score=float(score),
                breakdown=breakdown,
                illegal_genes=int(illegal),
                generations=generations,
                initial_best_score=initial_best_score,
                evolved_gain=gain,
            )

    def _random_genome(self, state: GameState, max_actions_per_turn: Optional[int] = None) -> list[int]:
        # Genome stores ranked candidate choices. At each simulated state,
        # gene[i] means "pick rank gene[i] from the local candidate menu."
        if USE_CYTHON_RHEA and random_genome_cython is not None:
            return random_genome_cython(
                max_actions_per_turn or self.cfg.max_actions_per_turn,
                self.cfg.top_k_per_state,
                self.rng,
            )
        if max_actions_per_turn is None:
            max_actions_per_turn = self.cfg.max_actions_per_turn
        return [
            self.rng.randrange(max(1, self.cfg.top_k_per_state))
            for _ in range(max_actions_per_turn)
        ]

    def _crossover(self, a: list[int], b: list[int]) -> list[int]:
        if USE_CYTHON_RHEA and crossover_cython is not None:
            return crossover_cython(a, b, self.rng)
        if not a:
            return list(b)
        cut = self.rng.randrange(0, len(a))
        return list(a[:cut]) + list(b[cut:])

    def _mutate(self, genome: list[int]) -> None:
        if USE_CYTHON_RHEA and mutate_cython is not None:
            mutate_cython(genome, self.cfg.mutation_rate, self.cfg.top_k_per_state, self.rng)
            return

    def _simulate_genome(
        self,
        state: GameState,
        genome: list[int],
    ) -> tuple[GameState, list, int]:
        # Use Cython-accelerated version if available
        if USE_CYTHON_RHEA:
            return simulate_genome_cython(
                state,
                genome,
                self.cfg.top_k_per_state,
                self.cfg.mutation_rate,
                CandidateKind,
            )
        sim = clone_for_search(state)
        acting = int(sim.active_player)
        actions = []
        illegal = 0

        for gene in genome:
            if sim.winner is not None:
                break
            if int(sim.active_player) != acting:
                break

            _feats, mask, cands = candidate_arrays(sim, max_candidates=MAX_CANDIDATES)
            legal = [c for i, c in enumerate(cands) if i < len(mask) and bool(mask[i])]
            if not legal:
                break

            ranked = self._rank_candidates_cheap(sim, legal)
            if not ranked:
                break

            idx = int(gene)
            if len(ranked) == 0:
                illegal += 1
                break
            if idx >= len(ranked):
                # Wrap around instead of clamping to avoid illegal genes
                idx = idx % len(ranked)

            cand = ranked[idx]
            ok = self._apply_candidate(sim, cand)
            if not ok:
                illegal += 1
                break

            actions.append(cand.first)
            if cand.second is not None:
                actions.append(cand.second)

        # If genome did not naturally pass the turn, force END_TURN if legal.
        if sim.winner is None and int(sim.active_player) == acting:
            _feats, mask, cands = candidate_arrays(sim, max_candidates=MAX_CANDIDATES)
            legal = [c for i, c in enumerate(cands) if i < len(mask) and bool(mask[i])]
            enders = [
                c for c in legal
                if c.kind == CandidateKind.END_TURN
                or c.terminal_action.action_type.name == "END_TURN"
            ]
            if enders:
                self._apply_candidate(sim, enders[0])
                actions.append(enders[0].terminal_action)

        return sim, actions, illegal

    def _rank_candidates_cheap(
        self,
        state: GameState,
        cands: list[CandidateAction],
    ) -> list[CandidateAction]:
        """
        Cheap ordering only. RHEA still chooses among these candidates.
        """
        scored: list[tuple[float, CandidateAction]] = []

        for c in cands:
            f = c.preview
            score = 0.0

            # Candidate preview feature layout from rl/candidate_actions.py.
            # Capture features around 8..11, attack-trade features around 12..23.
            if f is not None:
                score += 2.0 * float(f[10])       # capture completes
                score += 1.0 * float(f[8])        # capture progress
                score += 0.25 * float(f[11])      # property value
                score += 1.25 * float(f[17])      # enemy value removed max
                score += 0.75 * float(f[16])      # enemy value removed min
                score -= 0.80 * float(f[19])      # own value lost max
                score += 0.50 * float(f[21])      # target killed max
                score -= 1.00 * float(f[22])      # attacker killed

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
        try:
            state.step(cand.first)
            if cand.second is not None and state.winner is None:
                state.step(cand.second)
            return True
        except Exception:
            return False

    @staticmethod
    def compute_complexity_metrics(state: GameState, observer_seat: int) -> tuple[int, int, int, int]:
        """
        Compute game state complexity metrics for dynamic budgeting.
        
        Returns: (owned_units, factories, contested_captures, enemy_in_range_contacts)
        """
        enemy_seat = 1 - observer_seat
        
        # Count owned units
        owned_units = sum(1 for u in state.units[observer_seat] if u.is_alive)
        
        # Count factories (bases, airports, ports)
        factories = 0
        for prop in state.properties:
            if prop.owner == observer_seat:
                if getattr(prop, "is_base", False) or getattr(prop, "is_airport", False) or getattr(prop, "is_port", False):
                    factories += 1
        
        # Count contested captures (properties with capture progress from both players)
        contested_captures = 0
        for prop in state.properties:
            if prop.capture_points < 20:  # Property is being captured
                # Check if there's a capturing unit from either player
                capturing_unit = None
                for player in [observer_seat, enemy_seat]:
                    for unit in state.units[player]:
                        if unit.is_alive and unit.pos == (prop.row, prop.col):
                            capturing_unit = unit
                            break
                    if capturing_unit:
                        break
                
                if capturing_unit:
                    contested_captures += 1
        
        # Count enemy units in range of our units
        enemy_in_range_contacts = 0
        # This would require attack range calculations; for now, use a simple approximation
        # based on threat influence planes
        try:
            from engine.threat import compute_influence_planes
            # Get threat influence for enemy
            t_me, t_en, r_me, r_en, c_me, c_en = compute_influence_planes(
                state, me=observer_seat, grid=30
            )
            # Count enemy units that are within threat range of our units
            # For simplicity, we'll count enemy units that are on tiles with non-zero threat
            enemy_units = state.units[enemy_seat]
            for unit in enemy_units:
                if unit.is_alive and t_me[unit.pos[0], unit.pos[1]] > 0:
                    enemy_in_range_contacts += 1
        except Exception:
            # Fallback for any error (import failure, runtime error, etc.)
            enemy_in_range_contacts = len(state.units[enemy_seat]) // 4  # Rough estimate
        
        return owned_units, factories, contested_captures, enemy_in_range_contacts