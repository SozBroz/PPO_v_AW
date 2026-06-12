from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from engine.game import GameState
from engine.unit import UNIT_STATS
from rl.encoder import GRID_SIZE, N_SCALARS, N_SPATIAL_CHANNELS, encode_state
from rl.env import AWBWEnv
from rl.value_net import AWBWValueNet, evaluate_value_np, evaluate_value_batch

RHEA_ENGINE_TERMINAL_WIN_REASONS = frozenset({"hq_capture", "army_wipe"})


# Cython acceleration for hot paths
USE_CYTHON_FITNESS = True
try:
    from rl._rhea_fitness_cython import evaluate_value_fast, evaluate_value_batch_fast, phi_cython
except ImportError:
    USE_CYTHON_FITNESS = False
    evaluate_value_batch_fast = None


@dataclass(slots=True)
class RheaFitnessBreakdown:
    phi_delta: float
    value: float
    illegal_penalty: float
    terminal_sparse: float = 0.0
    total: float = 0.0


class RheaFitness:
    """
    One-turn fitness for RHEA.

    Scores exactly this bracket:

        before acting seat's turn -> after acting seat's turn

    It does not auto-run the opponent. The wider zero-sum learning/evaluation
    contract remains outside the inner RHEA planner.
    """

    def __init__(
        self,
        env_template: AWBWEnv,
        value_model: Optional[AWBWValueNet] = None,
        *,
        device: str = "cuda",
        reward_weight: float = 0.90,
        value_weight: float = 0.10,
        illegal_gene_penalty: float = 0.02,
        capture_completion_bonus: float = 0.0,
        blunder_exposure_weight: float = 0.0,
        hq_defense_weight: float = 0.0,
        capture_interrupt_bonus: float = 0.0,
        neutral_income_gap_weight: float = 0.0,
        capture_progress_bonus: float = 0.0,
    ) -> None:
        self.env_template = env_template
        self.value_model = value_model
        self.device = device
        self.reward_weight = float(reward_weight)
        self.value_weight = float(value_weight)
        self.illegal_gene_penalty = float(illegal_gene_penalty)
        self.capture_completion_bonus = float(capture_completion_bonus)
        self.blunder_exposure_weight = float(blunder_exposure_weight)
        self.hq_defense_weight = float(hq_defense_weight)
        self.capture_interrupt_bonus = float(capture_interrupt_bonus)
        self.neutral_income_gap_weight = float(neutral_income_gap_weight)
        self.capture_progress_bonus = float(capture_progress_bonus)
        # Reusable buffers — allocated once, reused across value() calls
        self._spatial_buf = np.zeros((GRID_SIZE, GRID_SIZE, N_SPATIAL_CHANNELS), dtype=np.float32)
        self._scalars_buf = np.zeros((N_SCALARS,), dtype=np.float32)

    def set_value_model(self, value_model: AWBWValueNet) -> None:
        """Swap the value model used for fitness evaluation.

        Used by dual-gradient self-play with hist-prob: the actor swaps
        between the current learner checkpoint and a historical checkpoint
        depending on the episode's opponent mode.
        """
        self.value_model = value_model

    @staticmethod
    def count_capture_completions(
        before: GameState, after: GameState, acting_seat: int,
    ) -> int:
        seat = int(acting_seat)
        before_owners = {(p.row, p.col): p.owner for p in before.properties}
        n = 0
        for prop in after.properties:
            prev = before_owners.get((prop.row, prop.col))
            if prev is not None and prev != seat and prop.owner == seat:
                n += 1
        return n

    def competency_shaping(
        self,
        before: GameState,
        after: GameState,
        *,
        observer_seat: int,
    ) -> float:
        delta = 0.0
        if self.capture_completion_bonus != 0.0:
            delta += self.capture_completion_bonus * self.count_capture_completions(
                before, after, observer_seat,
            )
        if self.blunder_exposure_weight != 0.0:
            delta -= self._blunder_exposure_penalty(after, observer_seat)
        if self.hq_defense_weight != 0.0:
            delta -= self._hq_defense_penalty(after, observer_seat)
        if self.capture_interrupt_bonus != 0.0:
            delta += self.capture_interrupt_bonus * self.count_capture_interruptions(
                before, after, observer_seat,
            )
        if self.neutral_income_gap_weight != 0.0:
            delta -= self.neutral_income_gap_weight * self._neutral_income_gap_fraction(after)
        if self.capture_progress_bonus != 0.0:
            delta += self.capture_progress_bonus * self.count_capture_progress_chips(
                before, after, observer_seat,
            )
        return delta

    @staticmethod
    def _is_income_property(prop) -> bool:
        return (
            not bool(getattr(prop, "is_hq", False))
            and not bool(getattr(prop, "is_comm_tower", False))
            and not bool(getattr(prop, "is_lab", False))
        )

    @classmethod
    def _neutral_income_gap_fraction(cls, after: GameState) -> float:
        """Fraction of capturable income tiles still neutral (end-of-turn).

        Genomes that flip neutrals this turn lower the fraction in ``after``,
        so the penalty creates a persistent capture-saturation gradient without
        needing a sighted value net. Normalized by all income tiles on the map.
        """
        income_total = sum(1 for p in after.properties if cls._is_income_property(p))
        if income_total <= 0:
            return 0.0
        neutral = sum(
            1 for p in after.properties
            if cls._is_income_property(p) and p.owner is None
        )
        return float(neutral) / float(income_total)

    @staticmethod
    def count_capture_progress_chips(
        before: GameState, after: GameState, acting_seat: int,
    ) -> float:
        """Reward partial capture progress on neutral or enemy income tiles.

        Complements ``capture_completion_bonus`` (flip-only): midgame stall in
        v8-light replays came from army-building phi dominating once contact
        started — small per-chip progress keeps neutral saturation in play.
        """
        seat = int(acting_seat)
        before_by = {(p.row, p.col): p for p in before.properties}
        chips = 0.0
        for prop in after.properties:
            if not RheaFitness._is_income_property(prop):
                continue
            prev = before_by.get((prop.row, prop.col))
            if prev is None:
                continue
            if prev.owner == seat and prev.capture_points >= 20:
                continue
            if prev.owner not in (None, 1 - seat):
                continue
            reduced = float(prev.capture_points - prop.capture_points)
            if reduced <= 0.0:
                continue
            if prev.owner != seat and prop.owner == seat:
                continue
            chips += reduced / 20.0
        return chips

    # Tier multipliers for interrupting an enemy capture of MY property.
    # Losing a base/airport costs production; losing the HQ costs the game.
    _INTERRUPT_TIER_HQ: float = 4.0
    _INTERRUPT_TIER_PRODUCTION: float = 2.0  # bases, airports
    _INTERRUPT_TIER_STANDARD: float = 1.0    # cities, comm towers, seaports, labs

    @classmethod
    def count_capture_interruptions(
        cls, before: GameState, after: GameState, observer_seat: int,
    ) -> float:
        """Tier-weighted count of enemy captures of MY property broken this turn.

        A property counts when it was mine and mid-capture in ``before``
        (capture_points < 20: an enemy stood on it making progress) and in
        ``after`` it is still mine with progress wiped back to 20. Within a
        planner rollout the enemy never moves voluntarily, so a reset means
        my genome killed (or crashed) the capturer — exactly the interrupt
        we want rewarded. Ports are plain capturable property here and fall
        into the standard tier alongside cities, comm towers and labs.
        """
        seat = int(observer_seat)
        before_by_pos = {(p.row, p.col): p for p in before.properties}
        total = 0.0
        for prop in after.properties:
            prev = before_by_pos.get((prop.row, prop.col))
            if prev is None:
                continue
            if prev.owner != seat or prev.capture_points >= 20:
                continue
            if prop.owner != seat or prop.capture_points < 20:
                continue
            if prop.is_hq:
                total += cls._INTERRUPT_TIER_HQ
            elif prop.is_base or prop.is_airport:
                total += cls._INTERRUPT_TIER_PRODUCTION
            else:
                total += cls._INTERRUPT_TIER_STANDARD
        return total

    # 4-turn horizon: react to committed snipers only. The original 8-turn
    # horizon fired for nearly any enemy foot unit on a standard map, taxing
    # every forward genome — v6b turtled and lost the macro game it had
    # previously won (baseline 4-0 with captures/props/army all flipped).
    _HQ_DEFENSE_HORIZON_TURNS: float = 4.0

    def _hq_defense_penalty(self, after: GameState, observer_seat: int) -> float:
        """Penalty when an enemy foot unit can finish capturing my HQ before
        any of my units can contest it.

        Motivated by the v4b gauntlet: v4b led baseline on captures,
        properties AND army value yet lost 0-4 — every game ended day 7-11
        by an HQ snipe that no config defended (full-HP infantry walking
        across the map unopposed). Distance proxy is Manhattan / move_range
        (optimistic for the attacker, so the term triggers conservatively
        early); capture itself takes 2 turns at full HP, so a defender
        arriving within threat_turns + 1 still contests.

        The penalty scales smoothly with the *defense gap* (how many turns
        late the nearest defender is), not just with threat proximity. The
        original binary form (full penalty unless a defender contests
        within threat+1) gave RHEA no selection gradient: a genome moving a
        unit homeward but not all the way earned zero credit, so the term
        was inert in the v6 gauntlet (1-3 vs baseline, snipes unchanged).
        """
        seat = int(observer_seat)
        enemy = 1 - seat
        hqs = after.map_data.hq_positions.get(seat, []) or []
        if not hqs:
            return 0.0
        horizon = self._HQ_DEFENSE_HORIZON_TURNS
        total = 0.0
        for hq in hqs:
            hr, hc = int(hq[0]), int(hq[1])
            threat_turns: float | None = None
            for u in after.units[enemy]:
                if not u.is_alive or not UNIT_STATS[u.unit_type].can_capture:
                    continue
                d = abs(int(u.pos[0]) - hr) + abs(int(u.pos[1]) - hc)
                mv = max(1, int(UNIT_STATS[u.unit_type].move_range))
                t = float(-(-d // mv))
                if threat_turns is None or t < threat_turns:
                    threat_turns = t
            if threat_turns is None or threat_turns > horizon:
                continue
            defense_turns: float | None = None
            for u in after.units[seat]:
                if not u.is_alive:
                    continue
                d = abs(int(u.pos[0]) - hr) + abs(int(u.pos[1]) - hc)
                mv = max(1, int(UNIT_STATS[u.unit_type].move_range))
                t = float(-(-d // mv))
                if defense_turns is None or t < defense_turns:
                    defense_turns = t
            if defense_turns is not None and defense_turns <= threat_turns + 1.0:
                continue
            gap = (
                defense_turns - threat_turns - 1.0
                if defense_turns is not None
                else horizon
            )
            gap_frac = min(1.0, gap / horizon)
            total += max(0.0, 1.0 - threat_turns / horizon) * gap_frac
        return self.hq_defense_weight * total

    def _blunder_exposure_penalty(self, after: GameState, observer_seat: int) -> float:
        """Material genuinely at risk from end-of-turn exposure, capture-exempt.

        Two refinements over the original binary form (which charged full
        unit value for ANY nonzero threat and made every v-config
        under-capture vs baseline — contested captures require standing in
        threat, so the tax fought the capture bonus and won):

        - risk-proportional: charge ``cost * min(threat_dmg, hp)`` — the HP
          actually at stake — so chip damage is cheap and kill-threats are
          expensive;
        - purposeful exposure is free: units standing on a property they can
          capture (mid-capture / completing) are exempt.
        """
        from engine.commander_wars_capture import is_capturable_property_at
        from engine.threat import compute_influence_planes

        seat = int(observer_seat)
        t_me, *_rest = compute_influence_planes(after, me=seat, grid=30)
        exposed = 0.0
        for u in after.units[seat]:
            if not u.is_alive:
                continue
            r, c = u.pos
            threat = float(t_me[r, c])  # normalized max incoming damage [0,1]
            if threat <= 0.0:
                continue
            if UNIT_STATS[u.unit_type].can_capture and is_capturable_property_at(
                after, seat, (r, c)
            ):
                continue
            cost = float(UNIT_STATS[u.unit_type].cost)
            hp_frac = max(0.0, float(u.hp)) / 100.0
            at_risk_frac = min(threat, hp_frac)
            exposed += cost * at_risk_frac
        return self.blunder_exposure_weight * exposed / 10000.0

    @staticmethod
    def engine_terminal_sparse(state: GameState, observer_seat: int) -> float:
        if state is None or not state.done or state.winner is None:
            return 0.0
        win_reason = getattr(state, "win_reason", None)
        if win_reason not in RHEA_ENGINE_TERMINAL_WIN_REASONS:
            return 0.0
        winner = int(state.winner)
        if winner == -1:
            return 0.0
        seat = int(observer_seat)
        return 1.0 if winner == seat else -1.0

    def phi(self, state: GameState, observer_seat: int) -> float:
        if USE_CYTHON_FITNESS and phi_cython is not None:
            return float(phi_cython(self.env_template, state, observer_seat))
        # Fallback: reuse AWBWEnv's tuned Φ without stepping the env.
        old_state = self.env_template.state
        old_seat = self.env_template._learner_seat
        try:
            self.env_template.state = state
            self.env_template._learner_seat = int(observer_seat)
            phi_value = float(self.env_template._compute_phi(state))
            return phi_value
        finally:
            self.env_template.state = old_state
            self.env_template._learner_seat = old_seat

    def value(self, state: GameState, observer_seat: int) -> float:
        """Return win probability [0, 1] for observer_seat from this state."""
        if self.value_model is None:
            return 0.5
        if USE_CYTHON_FITNESS and evaluate_value_fast is not None:
            return float(evaluate_value_fast(self.value_model, state, observer_seat, self.device))
        # Fallback with reusable buffers (allocated once in __init__)
        encode_state(
            state,
            observer=int(observer_seat),
            belief=None,
            out_spatial=self._spatial_buf,
            out_scalars=self._scalars_buf,
        )
        win_prob = evaluate_value_np(self.value_model, self._spatial_buf, self._scalars_buf, device=self.device)
        return float(win_prob)

    def batch_value(self, states: list[GameState], observer_seat: int) -> list[float]:
        """Batch evaluate win probabilities for multiple states at once.

        This is MUCH faster than calling value() repeatedly because it batches
        all forward passes into one GPU call.

        Args:
            states: List of GameState objects to evaluate
            observer_seat: Which seat to evaluate from

        Returns:
            List of win probabilities, one per state
        """
        if self.value_model is None:
            return [0.5] * len(states)

        if not states:
            return []

        # Use Cython batch if available (fastest path)
        if USE_CYTHON_FITNESS and evaluate_value_batch_fast is not None:
            return list(evaluate_value_batch_fast(self.value_model, states, observer_seat, self.device))

        # Fallback: encode all states into batches
        spatial_list = []
        scalars_list = []
        for state in states:
            spatial = np.zeros((GRID_SIZE, GRID_SIZE, N_SPATIAL_CHANNELS), dtype=np.float32)
            scalars = np.zeros((N_SCALARS,), dtype=np.float32)
            encode_state(
                state,
                observer=int(observer_seat),
                belief=None,
                out_spatial=spatial,
                out_scalars=scalars,
            )
            spatial_list.append(spatial)
            scalars_list.append(scalars)

        # Batch evaluate
        win_probs = evaluate_value_batch(
            self.value_model, spatial_list, scalars_list, device=self.device
        )

        return [float(p) for p in win_probs]

    def score(
        self,
        before: GameState,
        after: GameState,
        *,
        observer_seat: int,
        illegal_genes: int = 0,
        actions: list = None,  # Actions taken in the sequence
    ) -> RheaFitnessBreakdown:
        # Calculate phi delta (immediate tactical reward)
        phi_after = self.phi(after, observer_seat)
        phi_before = self.phi(before, observer_seat)
        phi_delta = phi_after - phi_before
        
        # Value head: win probability contribution
        # value() returns win probability in [0, 1]
        # We measure the change in win probability, scaled to [-1, 1] to match phi_delta
        v_before = self.value(before, observer_seat)  # [0, 1]
        v_after = self.value(after, observer_seat)      # [0, 1]
        
        # win_advantage: how much did the win probability change?
        # Scaled by 2.0 to map to [-1, 1] range (matching phi_delta magnitude)
        win_advantage = (v_after - v_before) * 2.0
        
        illegal_penalty = -self.illegal_gene_penalty * float(illegal_genes)
        
        # Check for build punishment
        build_punishment = 0.0
        build_punishment_details = ""
        mech_penalty = 0.0
        unused_funds_penalty = 0.0
        
        if actions is not None:
            # Check if END_TURN is in actions and no BUILD action before it
            end_turn_index = None
            build_happened = False
            build_actions = []
            total_build_cost = 0
            units_built = 0
            
            for i, action in enumerate(actions):
                if hasattr(action, 'action_type') and action.action_type.name == "END_TURN":
                    end_turn_index = i
                elif hasattr(action, 'action_type') and action.action_type.name == "BUILD":
                    build_happened = True
                    build_actions.append(action)
                    units_built += 1
                    # Calculate build cost
                    if hasattr(action, 'unit_type'):
                        from engine.unit import UNIT_STATS
                        unit_cost = UNIT_STATS[action.unit_type].cost
                        total_build_cost += unit_cost
            
            if end_turn_index is not None and not build_happened:
                # Player ended turn without building
                # Check if player has bases
                player_has_bases = self.env_template._player_has_bases(before, observer_seat)
                build_punishment_details = f"player_has_bases={player_has_bases}, "
                if player_has_bases:
                    # Apply build punishment
                    build_punishment_val = self.env_template._build_punishment
                    build_punishment_details += f"_build_punishment={build_punishment_val}, "
                    if build_punishment_val > 0.0:
                        phi_alpha = self.env_template._phi_alpha
                        build_punishment_details += f"_phi_alpha={phi_alpha}, "
                        # Calculate punishment: MUCH larger penalty for skipping builds
                        # Increased from -6000 to -30000 (5x larger)
                        build_punishment = -30000.0 * phi_alpha
                        build_punishment_details += f"calculated_punishment={build_punishment}"
            
# Penalize base underutilization
            if build_happened and end_turn_index is not None:
                if self.env_template._player_has_bases(before, observer_seat):
                    # Count available bases (factories, airports, ports)
                    base_count = 0
                    for prop in before.properties:
                        if prop.owner == observer_seat:
                            if getattr(prop, "is_base", False) or getattr(prop, "is_airport", False) or getattr(prop, "is_port", False):
                                base_count += 1
                    
                    if base_count > 0:
                        # Get funds available BEFORE building
                        available_funds_before = before.funds[observer_seat]
                        
                        # Calculate if funds were limiting factor
                        # Check if we could have built more units with available funds
                        # Simplest check: could we have built at least 'base_count' cheapest units (1000 each)?
                        funds_needed_for_all_bases = base_count * 1000
                        
                        if available_funds_before >= funds_needed_for_all_bases:
                            # Had enough money to use all bases with cheapest units
                            # Should have built base_count units
                            if units_built < base_count:
                                missing_units = base_count - units_built
                                # Penalty per missing unit should outweigh phi benefit of any single unit
                                # Mech gives +0.03 phi, so penalty > 0.03
                                base_utilization_penalty = -0.05 * missing_units
                                build_punishment_details += f" base_utilization_penalty={base_utilization_penalty:.4f} (bases={base_count}, built={units_built}, available_funds={available_funds_before})"
                                unused_funds_penalty = base_utilization_penalty
                        else:
                            # Funds were limiting
                            # Maximum units affordable with cheapest units
                            max_affordable_with_cheapest = available_funds_before // 1000
                            # But we might have built expensive units, using up funds
                            # Simple check: if units_built < max_affordable_with_cheapest, underutilized
                            if units_built < max_affordable_with_cheapest:
                                missing_units = max_affordable_with_cheapest - units_built
                                base_utilization_penalty = -0.05 * missing_units
                                build_punishment_details += f" base_utilization_penalty={base_utilization_penalty:.4f} (bases={base_count}, built={units_built}, affordable_cheapest={max_affordable_with_cheapest}, funds={available_funds_before})"
                                unused_funds_penalty = base_utilization_penalty

        terminal_sparse = self.engine_terminal_sparse(after, observer_seat)
        competency = self.competency_shaping(
            before, after, observer_seat=observer_seat,
        )

        total = (
            self.reward_weight * phi_delta
            + self.value_weight * win_advantage
            + illegal_penalty
            + build_punishment
            + mech_penalty
            + unused_funds_penalty
            + terminal_sparse
            + competency
        )

        return RheaFitnessBreakdown(
            phi_delta=float(phi_delta),
            value=float(win_advantage),
            illegal_penalty=float(illegal_penalty),
            terminal_sparse=float(terminal_sparse),
            total=float(total),
        )