"""
PROPERTY-EQUIV (Phase 4 of `desync_purge_engine_harden`): assert that for every
mid-replay `GameState` in the corpus,

    {a for a in candidate_actions(state) if step_succeeds(state, a)}
        == set(get_legal_actions(state))

Both directions:

* ``mask_overpermits``           — action listed by ``get_legal_actions`` but
                                   ``GameState.step`` raises on it.
* ``false_positive_in_step``     — action NOT in the mask, yet ``GameState.step``
                                   accepts it without raising. This is the gap
                                   STEP-GATE (Phase 3) is meant to close.

Corpus is built by ``tools/build_legal_actions_equivalence_corpus.py`` into
``tests/data/legal_actions_equivalence_corpus/*.pkl``. If the directory is
empty the test is skipped with the build instruction in the message.

Candidate enumeration scope (per ``ActionStage``):

* SELECT: END_TURN, ACTIVATE_COP, ACTIVATE_SCOP, SELECT_UNIT for every unit
  on the map (both seats, to catch "step accepts opponent select"), and BUILD
  for every owned producible factory × unit type.
* MOVE: SELECT_UNIT(unit_pos=selected.pos, move_pos=tile) for every map tile
  within ``selected.unit_type``'s effective Manhattan move radius (capped).
* ACTION: WAIT, DIVE_HIDE, ATTACK with target_pos in range, CAPTURE, LOAD,
  JOIN, UNLOAD per cargo per neighbour, REPAIR per neighbour. BUILD is also
  swept defensively, even though AWBW removed it from ACTION stage.

Skipped axes (documented for transparency):

* SELECT_UNIT with the "wrong" ``select_unit_id`` value (oracle-only
  disambiguation; not part of the agent action space).
* RESIGN (oracle-only terminator; not in any RL legal-action surface).
* Per-state random-luck retries on combat: combat damage uses
  ``random.randint`` for luck rolls. The engine does not raise on these
  rolls, so they don't affect "did step raise". We seed ``random.seed(0)``
  before each candidate trial regardless to keep behavior reproducible.

Runtime budget: ~60s for ~150 snapshots when the candidate cap is 500/state.
"""
from __future__ import annotations

import pickle
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import copy

from engine.action import (
    Action,
    ActionStage,
    ActionType,
    get_legal_actions,
    get_loadable_into,
    get_producible_units,
)
from engine.game import GameState
from engine.terrain import get_terrain
from engine.unit import UNIT_STATS


CORPUS_DIR = ROOT / "tests" / "data" / "legal_actions_equivalence_corpus"
CANDIDATE_CAP = 500
DEFECT_REPORT_LIMIT = 20


# ---------------------------------------------------------------------------
# Action key (Action is a dataclass with default eq=True but is unhashable
# because list-typed fields could appear in other dataclasses; play it safe).
# ---------------------------------------------------------------------------
def _action_key(a: Action) -> tuple:
    """Stable tuple key for set membership.

    Intentionally drops ``select_unit_id`` (oracle-only stack disambiguator;
    not part of the agent action space — the test compares mask vs step on
    the *agent* action surface).
    """
    return (
        int(a.action_type),
        a.unit_pos,
        a.move_pos,
        a.target_pos,
        int(a.unit_type) if a.unit_type is not None else None,
        a.unload_pos,
    )


def _action_repr(a: Action) -> str:
    return repr(a)


# ---------------------------------------------------------------------------
# Candidate enumeration per ActionStage
# ---------------------------------------------------------------------------
def _candidates_select(state: GameState) -> list[Action]:
    cands: list[Action] = []
    cands.append(Action(ActionType.END_TURN))
    cands.append(Action(ActionType.ACTIVATE_COP))
    cands.append(Action(ActionType.ACTIVATE_SCOP))

    # SELECT_UNIT for every alive unit on the map (both seats).
    for player in state.units:
        for u in state.units[player]:
            if u.is_alive:
                cands.append(Action(ActionType.SELECT_UNIT, unit_pos=u.pos))

    # BUILD candidates for every property × producible unit type. Cap defensively
    # by the producible product so we don't explode on giant maps.
    active = state.active_player
    for prop in state.properties:
        if prop.owner != active:
            continue
        terrain = get_terrain(state.map_data.terrain[prop.row][prop.col])
        if not (terrain.is_base or terrain.is_airport or terrain.is_port):
            continue
        for ut in get_producible_units(terrain, state.map_data.unit_bans):
            cands.append(Action(
                ActionType.BUILD,
                unit_pos=None,
                move_pos=(prop.row, prop.col),
                unit_type=ut,
            ))
    return cands


def _candidates_move(state: GameState) -> list[Action]:
    """SELECT_UNIT with move_pos for every map tile within an over-approximation
    of the unit's Manhattan move radius. Includes the unit's own tile (no-op
    move). Capped to avoid blowing up on huge maps with high-fuel units."""
    unit = state.selected_unit
    if unit is None:
        # State is in MOVE with no selected unit — that itself is a defect, but
        # we still need *some* candidate set so the test reports it. Sweep
        # SELECT_UNIT to every tile from origin (None) — empty list keeps it
        # bounded.
        return []
    stats = UNIT_STATS[unit.unit_type]
    # Effective max manhattan reach: move_range bounded by fuel, plus a small
    # slop (CO bonuses can stack +3 etc). Sweep tiles in that bounding box.
    reach = max(1, min(stats.move_range, unit.fuel)) + 4
    h, w = state.map_data.height, state.map_data.width
    r0, c0 = unit.pos
    cands: list[Action] = []
    for dr in range(-reach, reach + 1):
        for dc in range(-reach, reach + 1):
            if abs(dr) + abs(dc) > reach:
                continue
            r, c = r0 + dr, c0 + dc
            if not (0 <= r < h and 0 <= c < w):
                continue
            cands.append(Action(
                ActionType.SELECT_UNIT,
                unit_pos=unit.pos,
                move_pos=(r, c),
            ))
    return cands


def _candidates_action(state: GameState) -> list[Action]:
    unit = state.selected_unit
    move_pos = state.selected_move_pos
    if unit is None or move_pos is None:
        return []
    stats = UNIT_STATS[unit.unit_type]
    cands: list[Action] = []
    cands.append(Action(ActionType.WAIT, unit_pos=unit.pos, move_pos=move_pos))
    cands.append(Action(ActionType.DIVE_HIDE, unit_pos=unit.pos, move_pos=move_pos))
    cands.append(Action(ActionType.CAPTURE, unit_pos=unit.pos, move_pos=move_pos))

    # ATTACK: sweep target_pos within over-approximated max range from move_pos.
    max_r = max(stats.max_range, 1) + 2
    h, w = state.map_data.height, state.map_data.width
    mr, mc = move_pos
    for dr in range(-max_r, max_r + 1):
        for dc in range(-max_r, max_r + 1):
            if dr == 0 and dc == 0:
                continue
            tr, tc = mr + dr, mc + dc
            if not (0 <= tr < h and 0 <= tc < w):
                continue
            cands.append(Action(
                ActionType.ATTACK,
                unit_pos=unit.pos,
                move_pos=move_pos,
                target_pos=(tr, tc),
            ))

    # LOAD / JOIN at move_pos (engine treats friendly-occupied move_pos this way).
    cands.append(Action(ActionType.LOAD, unit_pos=unit.pos, move_pos=move_pos))
    cands.append(Action(ActionType.JOIN, unit_pos=unit.pos, move_pos=move_pos))

    # UNLOAD: every cargo type × every neighbour. We sweep ALL cargo unit
    # types that could conceivably be loaded, not just what is currently
    # aboard, to catch "step lets you unload a non-existent cargo type" bugs.
    cargo_types = list(get_loadable_into(unit.unit_type))
    if cargo_types:
        for cargo_ut in cargo_types:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                tr, tc = mr + dr, mc + dc
                if not (0 <= tr < h and 0 <= tc < w):
                    continue
                cands.append(Action(
                    ActionType.UNLOAD,
                    unit_pos=unit.pos,
                    move_pos=move_pos,
                    target_pos=(tr, tc),
                    unit_type=cargo_ut,
                ))

    # REPAIR: every neighbour (Black Boat command, but sweep regardless to
    # catch "step accepts REPAIR for non-Black-Boat" defects).
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        tr, tc = mr + dr, mc + dc
        if not (0 <= tr < h and 0 <= tc < w):
            continue
        cands.append(Action(
            ActionType.REPAIR,
            unit_pos=unit.pos,
            move_pos=move_pos,
            target_pos=(tr, tc),
        ))

    # BUILD candidates at the action stage shouldn't be legal (BUILD is a
    # SELECT-stage action only — see action.py "BUILD removed from Stage-2"
    # comment). Sweep a few defensively to confirm the mask never lists them.
    # We don't sweep all 30 unit types × all factories here — that would blow
    # the cap. We only sweep at the current move_pos.
    terrain = get_terrain(state.map_data.terrain[mr][mc])
    if terrain.is_base or terrain.is_airport or terrain.is_port:
        for ut in get_producible_units(terrain, state.map_data.unit_bans)[:6]:
            cands.append(Action(
                ActionType.BUILD,
                unit_pos=unit.pos,
                move_pos=move_pos,
                unit_type=ut,
            ))

    return cands


def _build_candidates(state: GameState) -> list[Action]:
    if state.action_stage == ActionStage.SELECT:
        cands = _candidates_select(state)
    elif state.action_stage == ActionStage.MOVE:
        cands = _candidates_move(state)
    elif state.action_stage == ActionStage.ACTION:
        cands = _candidates_action(state)
    else:
        cands = []
    if len(cands) > CANDIDATE_CAP:
        # Deterministic head-cap so the report is reproducible. Better than
        # random sampling for debugging.
        cands = cands[:CANDIDATE_CAP]
    return cands


# ---------------------------------------------------------------------------
# Defect aggregation
# ---------------------------------------------------------------------------
@dataclass
class Defect:
    snapshot: str
    stage: str
    active_player: int
    turn: int
    kind: str       # "mask_overpermits" | "false_positive_in_step"
    action: str
    detail: str     # exception type (for mask_overpermits) or "" otherwise


def _step_succeeds(state: GameState, a: Action) -> tuple[bool, str]:
    """Return (succeeded, error_str). Deepcopies state internally."""
    try:
        scratch = copy.deepcopy(state)
    except Exception as exc:  # noqa: BLE001
        return False, f"deepcopy:{type(exc).__name__}:{exc}"
    random.seed(0)
    try:
        scratch.step(a)
    except Exception as exc:  # noqa: BLE001 — any raise = step rejected
        return False, f"{type(exc).__name__}:{str(exc)[:120]}"
    return True, ""


# ---------------------------------------------------------------------------
# Pytest discovery
# ---------------------------------------------------------------------------
def _list_corpus() -> list[Path]:
    if not CORPUS_DIR.is_dir():
        return []
    return sorted(CORPUS_DIR.glob("*.pkl"))


def test_legal_actions_step_equivalence():
    """The single highest-leverage test in the campaign: mask == step on every
    state in the corpus.

    If the corpus is empty, the test is skipped with the rebuild instruction.
    """
    snapshots = _list_corpus()
    if not snapshots:
        pytest.skip(
            "No corpus found — run "
            "`python tools/build_legal_actions_equivalence_corpus.py` first."
        )

    defects: list[Defect] = []
    state_count = 0
    candidate_count = 0
    deepcopy_failures: list[str] = []

    for pkl_path in snapshots:
        try:
            with open(pkl_path, "rb") as f:
                state: GameState = pickle.load(f)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"failed to load {pkl_path.name}: {exc}")
            continue
        state_count += 1

        # Build mask once (do NOT deepcopy state for mask; mask is read-only).
        try:
            mask = get_legal_actions(state)
        except Exception as exc:  # noqa: BLE001
            defects.append(Defect(
                snapshot=pkl_path.name,
                stage=state.action_stage.name,
                active_player=state.active_player,
                turn=state.turn,
                kind="mask_raised",
                action="<get_legal_actions>",
                detail=f"{type(exc).__name__}:{str(exc)[:160]}",
            ))
            continue
        mask_keys = {_action_key(a): a for a in mask}

        cands = _build_candidates(state)
        candidate_count += len(cands)
        cand_keys = {_action_key(c): c for c in cands}

        # Always test every mask action. Some mask actions might not be in the
        # candidate sweep (e.g. SELECT_UNIT with select_unit_id) — we still
        # need to verify step accepts them.
        for k, a in mask_keys.items():
            ok, err = _step_succeeds(state, a)
            if not ok:
                defects.append(Defect(
                    snapshot=pkl_path.name,
                    stage=state.action_stage.name,
                    active_player=state.active_player,
                    turn=state.turn,
                    kind="mask_overpermits",
                    action=_action_repr(a),
                    detail=err,
                ))

        # Test every candidate not in the mask.
        for k, c in cand_keys.items():
            if k in mask_keys:
                continue
            ok, err = _step_succeeds(state, c)
            if ok:
                defects.append(Defect(
                    snapshot=pkl_path.name,
                    stage=state.action_stage.name,
                    active_player=state.active_player,
                    turn=state.turn,
                    kind="false_positive_in_step",
                    action=_action_repr(c),
                    detail="",
                ))

    # Aggregate report
    print()
    print(f"[property-equiv] snapshots checked:   {state_count}")
    print(f"[property-equiv] candidates evaluated: {candidate_count}")
    print(f"[property-equiv] defects found:        {len(defects)}")
    if deepcopy_failures:
        print(f"[property-equiv] deepcopy failures: {len(deepcopy_failures)}")
    if defects:
        # Top-3 most-frequent defect shapes for quick triage. Shape =
        # (kind, action_type_name) so distinct param combos collapse.
        from collections import Counter
        shapes = Counter(
            (d.kind, d.action.split("(")[0] + ":" + (d.action.split(",")[0].split("(")[-1] if "(" in d.action else d.action))
            for d in defects
        )
        print("[property-equiv] top defect shapes:")
        for (kind, shape), n in shapes.most_common(3):
            print(f"    {n:>5}  {kind:<28}  {shape}")
        print(f"[property-equiv] first {DEFECT_REPORT_LIMIT} defects:")
        for d in defects[:DEFECT_REPORT_LIMIT]:
            tail = f" detail={d.detail[:90]}" if d.detail else ""
            print(
                f"  [{d.snapshot}] stage={d.stage} p={d.active_player} t={d.turn} "
                f"{d.kind}: {d.action}{tail}"
            )
    assert not defects, (
        f"PROPERTY-EQUIV failed on {len(defects)} defects across "
        f"{state_count} snapshots. See stdout for top shapes + first "
        f"{DEFECT_REPORT_LIMIT} entries. This is the gate-step inconsistency "
        f"that Phase 3 STEP-GATE must close."
    )
