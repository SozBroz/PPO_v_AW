from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from engine.game import GameState
from rl.encoder import GRID_SIZE, N_SCALARS, N_SPATIAL_CHANNELS, encode_state
from rl.env import AWBWEnv
from rl.value_net import AWBWValueNet, evaluate_value_np

# Cython acceleration for hot paths
USE_CYTHON_FITNESS = True
try:
    from rl._rhea_fitness_cython import evaluate_value_fast, phi_cython
except ImportError:
    USE_CYTHON_FITNESS = False


@dataclass(slots=True)
class RheaFitnessBreakdown:
    phi_delta: float
    value: float
    illegal_penalty: float
    total: float


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
    ) -> None:
        self.env_template = env_template
        self.value_model = value_model
        self.device = device
        self.reward_weight = float(reward_weight)
        self.value_weight = float(value_weight)
        self.illegal_gene_penalty = float(illegal_gene_penalty)
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

        total = (
            self.reward_weight * phi_delta
            + self.value_weight * win_advantage
            + illegal_penalty
            + build_punishment
            + mech_penalty
            + unused_funds_penalty
        )

        return RheaFitnessBreakdown(
            phi_delta=float(phi_delta),
            value=float(win_advantage),
            illegal_penalty=float(illegal_penalty),
            total=float(total),
        )