"""Thin helpers for stepwise RHEA training (not wired into actors by default)."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from engine.action import ActionType
from engine.game import GameState
from rl.encoder import encode_state
from rl.rhea import replay_rhea_actions_with_steps
from rl.rhea_fitness import RheaFitness
from rl.rhea_replay import RheaStepTransition

_COP_TYPES = frozenset({ActionType.ACTIVATE_COP, ActionType.ACTIVATE_SCOP})


def phase_for_action_index(
    idx: int,
    action,
    *,
    n_move_actions: int | None = None,
    n_buy_actions: int | None = None,
) -> str:
    """Tag replay step phase from planner boundaries or action type heuristics."""
    if action.action_type in _COP_TYPES:
        return "cop"
    if n_move_actions is not None and n_buy_actions is not None:
        if idx < n_move_actions:
            return "move"
        if idx < n_move_actions + n_buy_actions:
            return "buy"
        return "end"
    if action.action_type.name == "BUILD":
        return "buy"
    if action.action_type.name == "END_TURN":
        return "end"
    return "move"


def collect_step_transitions_for_turn(
    env,
    fitness: RheaFitness,
    acting: int,
    actions: list,
    *,
    n_move_actions: int | None = None,
    n_buy_actions: int | None = None,
    turn_id: int = 0,
    phase_for_index: Callable[[int], str] | None = None,
) -> list[RheaStepTransition]:
    """Collect one ``RheaStepTransition`` per successful replayed engine step.

    Not used by default training; callers must enable step mode explicitly.
    """
    steps: list[RheaStepTransition] = []
    phi_prev = float(fitness.phi(env.state, acting))
    day = int(getattr(env.state, "turn", 0))
    planned_action_idx = 0

    def _phase(idx: int, action) -> str:
        if phase_for_index is not None:
            return str(phase_for_index(idx))
        return phase_for_action_index(
            idx,
            action,
            n_move_actions=n_move_actions,
            n_buy_actions=n_buy_actions,
        )

    def on_step(
        before: GameState,
        action,
        after: GameState,
        step_idx: int,
        source: str,
    ) -> None:
        nonlocal phi_prev, planned_action_idx
        spatial, scalars = encode_state(before, observer=acting)
        spatial_next, scalars_next = encode_state(after, observer=acting)
        phi_now = float(fitness.phi(after, acting))
        delta = phi_now - phi_prev
        phi_prev = phi_now
        if source == "salvage":
            phase = "salvage"
        else:
            phase = _phase(planned_action_idx, action)
            planned_action_idx += 1
        steps.append(
            RheaStepTransition(
                spatial=np.asarray(spatial, dtype=np.float32),
                scalars=np.asarray(scalars, dtype=np.float32),
                spatial_next=np.asarray(spatial_next, dtype=np.float32),
                scalars_next=np.asarray(scalars_next, dtype=np.float32),
                phi_delta=float(delta),
                acting_seat=int(acting),
                day=day,
                step_index=int(step_idx),
                turn_id=int(turn_id),
                done=bool(after.done),
                turn_done=False,
                winner=after.winner,
                phase=phase,
                action_type=str(action.action_type.name),
            )
        )

    applied_planned, _skipped, salvage_steps = replay_rhea_actions_with_steps(
        env.state,
        actions,
        acting,
        on_step=on_step,
    )
    if steps:
        steps[-1].turn_done = True
    assert len(steps) == applied_planned + salvage_steps
    return steps
