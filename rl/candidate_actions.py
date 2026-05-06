"""Candidate-action interface for merged AWBW unit intents.

This module is intentionally engine-adapter only: it does not change
``engine.game.GameState``.  The RL env can expose a fixed padded candidate table
where each row is one legal tactical intent for the current decision.

Rows are normalized float32 features.  The env action index is the row number;
``candidate_mask`` marks populated rows.  Compound candidates execute by issuing
ordinary staged engine actions in sequence, so replay/legal invariants stay in one
place.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np

from engine.action import (
    Action,
    ActionStage,
    ActionType,
    get_legal_actions,
)
from engine.combat import damage_range
from engine.game import GameState
from engine.terrain import get_terrain
from engine.unit import UNIT_STATS, Unit, UnitType

# Try to import Cython-optimized version
try:
    from . import _candidate_actions_cython
    CANDIDATE_CYTHON_AVAILABLE = True
except ImportError:
    CANDIDATE_CYTHON_AVAILABLE = False
    _candidate_actions_cython = None  # noqa: F401 — used when available

USE_CANDIDATE_CYTHON = (
    CANDIDATE_CYTHON_AVAILABLE
    and os.environ.get("AWBW_USE_CANDIDATE_CYTHON", "1") == "1"
)

MAX_CANDIDATES = 4096
CANDIDATE_FEATURE_DIM = 24


class CandidateKind(IntEnum):
    # Single legal flat/stage action passthroughs.
    FLAT_ACTION = 0
    SELECT_UNIT = 1
    BUILD = 2
    REPAIR = 3
    UNLOAD = 4
    POWER = 5
    END_TURN = 6

    # MOVE-stage merged intents.  These all start from state.selected_unit.
    MOVE_WAIT = 10
    MOVE_CAPTURE = 11
    MOVE_ATTACK = 12
    MOVE_LOAD = 13
    MOVE_JOIN = 14
    MOVE_DIVE_HIDE = 15

    # MOVE-stage setup intents that intentionally keep a follow-up ACTION stage.
    MOVE_SETUP_UNLOAD = 20
    MOVE_SETUP_REPAIR = 21
    MOVE_SETUP_ACTION = 22


@dataclass(slots=True)
class CandidateAction:
    kind: CandidateKind
    first: Action
    second: Optional[Action] = None
    preview: Optional[np.ndarray] = None
    label: str = ""

    @property
    def terminal_action(self) -> Action:
        return self.second if self.second is not None else self.first


def _norm_pos(pos: Optional[tuple[int, int]]) -> tuple[float, float]:
    if pos is None:
        return 0.0, 0.0
    return float(pos[0]) / 29.0, float(pos[1]) / 29.0


def _unit_cost(ut: UnitType | None) -> float:
    if ut is None:
        return 0.0
    return float(UNIT_STATS[ut].cost)


def _target_unit(state: GameState, pos: Optional[tuple[int, int]]) -> Optional[Unit]:
    if pos is None:
        return None
    return state.get_unit_at(pos[0], pos[1])


def _capture_preview(state: GameState, unit: Unit, dest: tuple[int, int]) -> dict[str, float]:
    prop = state.get_property_at(*dest)
    if prop is None:
        return {}
    progress = min(int(unit.display_hp), int(prop.capture_points))
    remaining = max(0, int(prop.capture_points) - progress)
    return {
        "capture_progress": float(progress) / 20.0,
        "capture_remaining_after": float(remaining) / 20.0,
        "capture_completes": 1.0 if remaining <= 0 else 0.0,
        "property_value": 1.0,
    }


def _counter_range_for_preview(
    state: GameState,
    attacker: Unit,
    defender: Unit,
    attacker_pos: tuple[int, int],
    forward_min: int,
    forward_max: int,
) -> tuple[int, int, bool]:
    """Approximate min/max counter damage for preview features.

    Indirect defenders never counter in AWBW/AWBW-style rules, including when
    struck by another indirect.  Sonja SCOP must not be pruned by "target dies":
    Counter Break strikes first, so we evaluate counters from pre-attack HP.
    """
    def_stats = UNIT_STATS[defender.unit_type]
    if def_stats.is_indirect:
        return 0, 0, False
    if def_stats.max_ammo > 0 and defender.ammo <= 0:
        return 0, 0, False

    attacker_terrain = get_terrain(state.map_data.terrain[attacker_pos[0]][attacker_pos[1]])
    defender_terrain = get_terrain(state.map_data.terrain[defender.pos[0]][defender.pos[1]])
    attacker_co = state.co_states[attacker.player]
    defender_co = state.co_states[defender.player]
    sonja_counter_break = bool(defender_co.co_id == 18 and defender_co.scop_active)

    vals: list[int] = []
    for fwd in (int(forward_min), int(forward_max)):
        dcopy = copy.copy(defender)
        if sonja_counter_break:
            # Counter Break: defender attacks first / pre-attack HP.
            dcopy.hp = defender.hp
        else:
            dcopy.hp = max(0, defender.hp - fwd)
            if dcopy.hp <= 0:
                continue
        rng = damage_range(
            dcopy,
            attacker,
            defender_terrain,
            attacker_terrain,
            defender_co,
            attacker_co,
        )
        if rng is not None:
            vals.extend([int(rng[0]), int(rng[1])])
    if not vals:
        return 0, 0, sonja_counter_break
    return min(vals), max(vals), sonja_counter_break


def _attack_preview(
    state: GameState,
    unit: Unit,
    dest: tuple[int, int],
    target: tuple[int, int],
) -> dict[str, float]:
    defender = state.get_unit_at(target[0], target[1])
    if defender is None:
        # Pipe seam / non-unit target.  Keep candidate legal but dossier empty.
        return {}
    att_copy = copy.copy(unit)
    att_copy.pos = dest
    attacker_terrain = get_terrain(state.map_data.terrain[dest[0]][dest[1]])
    defender_terrain = get_terrain(state.map_data.terrain[target[0]][target[1]])
    fwd = damage_range(
        att_copy,
        defender,
        attacker_terrain,
        defender_terrain,
        state.co_states[unit.player],
        state.co_states[defender.player],
    )
    if fwd is None:
        return {}
    dmin, dmax = int(fwd[0]), int(fwd[1])
    cmin, cmax, sonja_break = _counter_range_for_preview(
        state, att_copy, defender, dest, dmin, dmax
    )
    target_cost = _unit_cost(defender.unit_type)
    my_cost = _unit_cost(unit.unit_type)
    enemy_loss_min = target_cost * min(dmin, defender.hp) / 100.0
    enemy_loss_max = target_cost * min(dmax, defender.hp) / 100.0
    my_loss_min = my_cost * min(cmin, unit.hp) / 100.0
    my_loss_max = my_cost * min(cmax, unit.hp) / 100.0
    return {
        "damage_min": float(dmin) / 100.0,
        "damage_max": float(dmax) / 100.0,
        "counter_min": float(cmin) / 100.0,
        "counter_max": float(cmax) / 100.0,
        "enemy_value_removed_min": enemy_loss_min / 30000.0,
        "enemy_value_removed_max": enemy_loss_max / 30000.0,
        "my_value_lost_min": my_loss_min / 30000.0,
        "my_value_lost_max": my_loss_max / 30000.0,
        "target_killed_min": 1.0 if dmin >= defender.hp else 0.0,
        "target_killed_max": 1.0 if dmax >= defender.hp else 0.0,
        "attacker_killed_min": 1.0 if cmin >= unit.hp else 0.0,
        "attacker_killed_max": 1.0 if cmax >= unit.hp else 0.0,
        "sonja_counter_break": 1.0 if sonja_break else 0.0,
    }


def _base_features(
    kind: CandidateKind,
    unit_pos: Optional[tuple[int, int]],
    dest: Optional[tuple[int, int]],
    target: Optional[tuple[int, int]],
    unit_type: UnitType | None,
) -> np.ndarray:
    f = np.zeros((CANDIDATE_FEATURE_DIM,), dtype=np.float32)
    f[0] = float(int(kind)) / 32.0
    f[1], f[2] = _norm_pos(unit_pos)
    f[3], f[4] = _norm_pos(dest)
    f[5], f[6] = _norm_pos(target)
    if unit_type is not None:
        f[7] = float(int(unit_type)) / 32.0
    return f


def _fill_preview_features(f: np.ndarray, preview: dict[str, float]) -> None:
    # Capture block.
    f[8] = float(preview.get("capture_progress", 0.0))
    f[9] = float(preview.get("capture_remaining_after", 0.0))
    f[10] = float(preview.get("capture_completes", 0.0))
    f[11] = float(preview.get("property_value", 0.0))
    # Attack block.
    f[12] = float(preview.get("damage_min", 0.0))
    f[13] = float(preview.get("damage_max", 0.0))
    f[14] = float(preview.get("counter_min", 0.0))
    f[15] = float(preview.get("counter_max", 0.0))
    f[16] = float(preview.get("enemy_value_removed_min", 0.0))
    f[17] = float(preview.get("enemy_value_removed_max", 0.0))
    f[18] = float(preview.get("my_value_lost_min", 0.0))
    f[19] = float(preview.get("my_value_lost_max", 0.0))
    f[20] = float(preview.get("target_killed_min", 0.0))
    f[21] = float(preview.get("target_killed_max", 0.0))
    f[22] = max(
        float(preview.get("attacker_killed_min", 0.0)),
        float(preview.get("attacker_killed_max", 0.0)),
    )
    f[23] = float(preview.get("sonja_counter_break", 0.0))


def candidate_to_features(state: GameState, cand: CandidateAction) -> np.ndarray:
    if USE_CANDIDATE_CYTHON:
        return _candidate_actions_cython.candidate_to_features_cython(state, cand)
    action = cand.terminal_action
    ut: UnitType | None = action.unit_type
    if action.unit_pos is not None:
        u = state.get_unit_at(*action.unit_pos)
        if u is not None:
            ut = u.unit_type
    f = _base_features(cand.kind, action.unit_pos, action.move_pos, action.target_pos, ut)
    if cand.preview is not None:
        f += cand.preview.astype(np.float32, copy=False)
    return f


def _make_candidate(
    state: GameState,
    kind: CandidateKind,
    first: Action,
    second: Optional[Action] = None,
    preview: Optional[dict[str, float]] = None,
    label: str = "",
) -> CandidateAction:
    p = np.zeros((CANDIDATE_FEATURE_DIM,), dtype=np.float32)
    if preview:
        _fill_preview_features(p, preview)
    return CandidateAction(kind=kind, first=first, second=second, preview=p, label=label)


def enumerate_candidates(state: GameState) -> list[CandidateAction]:
    """Enumerate legal candidates for the current state.

    SELECT/ACTION stages remain mostly pass-through.  MOVE stage is rewritten as
    merged terminal intents where safe, plus setup intents for multi-object
    actions such as unload/repair.
    """
    legal = get_legal_actions(state)
    if state.action_stage != ActionStage.MOVE:
        out: list[CandidateAction] = []
        for a in legal:
            if a.action_type == ActionType.END_TURN:
                k = CandidateKind.END_TURN
            elif a.action_type in (ActionType.ACTIVATE_COP, ActionType.ACTIVATE_SCOP):
                k = CandidateKind.POWER
            elif a.action_type == ActionType.SELECT_UNIT:
                k = CandidateKind.SELECT_UNIT
            elif a.action_type == ActionType.BUILD:
                k = CandidateKind.BUILD
            elif a.action_type == ActionType.REPAIR:
                k = CandidateKind.REPAIR
            elif a.action_type == ActionType.UNLOAD:
                k = CandidateKind.UNLOAD
            else:
                k = CandidateKind.FLAT_ACTION
            out.append(_make_candidate(state, k, a, label=a.action_type.name))
        return out

    unit = state.selected_unit
    if unit is None:
        return []

    candidates: list[CandidateAction] = []
    # The MOVE legal list contains SELECT_UNIT(move_pos=dest) actions.
    for move in legal:
        if move.action_type != ActionType.SELECT_UNIT or move.move_pos is None:
            continue
        dest = move.move_pos
        # Generate legal ACTION-stage terminators for this destination by using a
        # cheap state copy.  This preserves all existing legal-pruning rules.
        probe = copy.copy(state)
        # Copy the selected stage scalars only; unit objects are shared read-only.
        probe.selected_unit = unit
        probe.selected_move_pos = dest
        probe.action_stage = ActionStage.ACTION
        action_legals = get_legal_actions(probe)

        has_capture = any(a.action_type == ActionType.CAPTURE for a in action_legals)
        terminal_added = False
        for a in action_legals:
            if a.action_type == ActionType.CAPTURE:
                candidates.append(
                    _make_candidate(
                        state,
                        CandidateKind.MOVE_CAPTURE,
                        move,
                        a,
                        _capture_preview(state, unit, dest),
                        label="MOVE_CAPTURE",
                    )
                )
                terminal_added = True
            elif a.action_type == ActionType.ATTACK and a.target_pos is not None:
                candidates.append(
                    _make_candidate(
                        state,
                        CandidateKind.MOVE_ATTACK,
                        move,
                        a,
                        _attack_preview(state, unit, dest, a.target_pos),
                        label="MOVE_ATTACK",
                    )
                )
                terminal_added = True
            elif a.action_type == ActionType.LOAD:
                candidates.append(_make_candidate(state, CandidateKind.MOVE_LOAD, move, a, label="MOVE_LOAD"))
                terminal_added = True
            elif a.action_type == ActionType.JOIN:
                candidates.append(_make_candidate(state, CandidateKind.MOVE_JOIN, move, a, label="MOVE_JOIN"))
                terminal_added = True
            elif a.action_type == ActionType.DIVE_HIDE:
                candidates.append(_make_candidate(state, CandidateKind.MOVE_DIVE_HIDE, move, a, label="MOVE_DIVE_HIDE"))
                terminal_added = True
            elif a.action_type == ActionType.UNLOAD:
                # Keep unload staged: Step 2 moves/setup, Step 3 picks cargo/tile.
                candidates.append(_make_candidate(state, CandidateKind.MOVE_SETUP_UNLOAD, move, None, label="MOVE_SETUP_UNLOAD"))
                terminal_added = True
                break
            elif a.action_type == ActionType.REPAIR:
                # Black Boat repair target is multi-object; keep staged.
                candidates.append(_make_candidate(state, CandidateKind.MOVE_SETUP_REPAIR, move, None, label="MOVE_SETUP_REPAIR"))
                terminal_added = True
                break

        # Capture dominates wait on the same tile.  Attack does NOT dominate wait.
        if not has_capture:
            wait = next((a for a in action_legals if a.action_type == ActionType.WAIT), None)
            if wait is not None:
                candidates.append(_make_candidate(state, CandidateKind.MOVE_WAIT, move, wait, label="MOVE_WAIT"))
                terminal_added = True

        if not terminal_added:
            candidates.append(_make_candidate(state, CandidateKind.MOVE_SETUP_ACTION, move, None, label="MOVE_SETUP_ACTION"))

    return candidates


def candidate_arrays(
    state: GameState,
    *,
    max_candidates: int = MAX_CANDIDATES,
    feats_buf: np.ndarray | None = None,
    mask_buf: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[CandidateAction]]:
    """Enumerate candidates and return padded feature array + mask + candidate list.

    Optional *feats_buf* / *mask_buf* let callers reuse allocations across
    repeated calls (e.g. inside a rollout loop) — pass pre-allocated arrays
    of the correct shape to skip the alloc entirely.
    """
    cands = enumerate_candidates(state)
    n = min(len(cands), max_candidates)

    # Reuse or allocate feature buffer.
    if feats_buf is not None and feats_buf.shape == (max_candidates, CANDIDATE_FEATURE_DIM):
        feats = feats_buf
    else:
        feats = np.zeros((max_candidates, CANDIDATE_FEATURE_DIM), dtype=np.float32)

    # Reuse or allocate mask buffer; always reset the active slice.
    if mask_buf is not None and mask_buf.shape == (max_candidates,):
        mask = mask_buf
        mask[:n] = True
        mask[n:] = False
    else:
        mask = np.zeros((max_candidates,), dtype=bool)
        mask[:n] = True

    for i in range(n):
        feats[i] = candidate_to_features(state, cands[i])
    return feats, mask, cands
