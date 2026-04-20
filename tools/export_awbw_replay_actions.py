"""
Emit an AWBW Replay Player `p:` action stream alongside the game-state zip.

Overview
--------
AWBW zips contain two gzipped members:
  1. `<game_id>`        – game-state PHP snapshots (O:8:"awbwGame"; one per turn).
  2. `a<game_id>`       – per-action JSON stream wrapped in PHP envelopes:
        p:<playerID>;d:<day>;a:a:3:{ i:0;i:<playerID>;i:1;i:<turnNum>;i:2;a:<N>:{
            i:0;s:<len>:"<action_json>";
            i:1;s:<len>:"<action_json>";
            ...
        }}
     One envelope per player-turn, newline-separated.

The viewer uses (2) to animate individual moves. Without it, the player still
sees turn snapshots but units teleport at turn boundaries — which is exactly
the behaviour the current export has today.

Scope (MVP)
-----------
Emits these action types, compatible with the viewer's ReplayActionDatabase:
  * End    — every END_TURN boundary (required for the viewer to advance).
  * Build  — every new unit spawned by BUILD.
  * Move   — every WAIT (move+wait).

For CAPTURE / LOAD, we emit only the Move sub-action: the unit visibly
walks to the capture/load tile. The Capt/Load wrappers are **not** emitted,
so the viewer won't animate those side-effects, but the next turn snapshot
will sync state correctly.

For ATTACK, we emit a full `Fire` envelope whose nested `Move` shows the
attacker **pre-combat** (full HP/ammo walking to the firing tile), and
whose `Fire.combatInfoVision.global.combatInfo` carries post-combat HP/ammo
for both attacker and defender. Dead units are dropped by the viewer when
`units_hit_points == 0` (see `AttackUnitAction.SetupAndUpdate`).

HP encoding in action JSON
--------------------------
Oracle AWBW emits `units_hit_points` as **integer bars 0–10** in action
payloads (distinct from the fractional 0.0–10.0 float used in the PHP state
snapshot). We therefore write ``unit.display_hp`` here, matching
``ReplayActionHelper.ParseJObjectIntoReplayUnit`` → ``(int)MathF.Ceiling``
and preventing "0-bar ghost" artefacts where internal HP 1–9 previously
serialised to a 0.1–0.9 float.

Strategy
--------
1. Rebuild the GameState deterministically by replaying `state.full_trace`.
2. On each atomic action, inspect the live state before/after and emit the
   appropriate JSON; group by (active_player_id, day).
3. Write the gzipped stream into the existing replay zip as entry
   `a<game_id>` so `AWBWJsonReplayParser.ParseReplayZip` picks it up and
   sets `ReplayVersion = 2`.

Determinism
-----------
This re-execution uses the same engine code paths as the producing game,
so stable unit_ids line up 1:1 with those in the state snapshots. If the
trace and the snapshot disagree, the snapshot always wins (the state zip is
the authoritative ground truth for the viewer).
"""
from __future__ import annotations

import copy
import gzip
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from engine.action import Action, ActionStage, ActionType
from engine.game import GameState, make_initial_state
from engine.map_loader import MapData, load_map
from engine.unit import Unit, UnitType, UNIT_STATS

from tools.export_awbw_replay import (
    P0_PLAYER_ID, P1_PLAYER_ID,
    _AWBW_UNIT_NAMES, _awbw_move_type,
)


# ---------------------------------------------------------------------------
# Trace entry → Action
# ---------------------------------------------------------------------------

def _trace_to_action(entry: dict) -> Action:
    """Hydrate a full_trace dict back into a live Action."""
    atype = ActionType[entry["type"]]
    unit_pos   = tuple(entry["unit_pos"])   if entry["unit_pos"]   is not None else None
    move_pos   = tuple(entry["move_pos"])   if entry["move_pos"]   is not None else None
    target_pos = tuple(entry["target_pos"]) if entry["target_pos"] is not None else None
    unit_type  = UnitType[entry["unit_type"]] if entry["unit_type"] is not None else None
    return Action(
        action_type=atype,
        unit_pos=unit_pos,
        move_pos=move_pos,
        target_pos=target_pos,
        unit_type=unit_type,
    )


# ---------------------------------------------------------------------------
# PHP envelope primitives
# ---------------------------------------------------------------------------

def _php_s(s: str) -> str:
    encoded = s.encode("utf-8")
    return f's:{len(encoded)}:"{s}";'


def _php_i(n: int) -> str:
    return f"i:{n};"


# ---------------------------------------------------------------------------
# Unit JSON (matches AWBW's ReplayActionHelper.ParseJObjectIntoReplayUnit)
# ---------------------------------------------------------------------------

def _unit_to_json(
    unit: Unit,
    player_id: int,
) -> dict[str, Any]:
    """Build the `units_*` JSON dict the viewer expects in action payloads.

    The keys mirror what `_serialize_unit` writes into the PHP state zip.
    Fields: id / players_id / name / position / hp / fuel / ammo / range /
    cost / movement_type / moved / fired / capture / cargo slots.
    """
    stats = UNIT_STATS[unit.unit_type]
    short_range = stats.min_range if stats.max_ammo != 0 else 0
    long_range  = stats.max_range if stats.max_ammo != 0 else 0
    if stats.max_ammo == 0:
        short_range = 0
        long_range = 0
    return {
        "units_id":              unit.unit_id,
        "units_games_id":        0,
        "units_players_id":      player_id,
        "units_name":            _AWBW_UNIT_NAMES.get(unit.unit_type, stats.name),
        "units_movement_points": stats.move_range,
        "units_vision":          stats.vision,
        "units_fuel":            unit.fuel,
        "units_fuel_per_turn":   stats.fuel_per_turn,
        "units_sub_dive":        "Y" if unit.is_submerged else "N",
        "units_ammo":            unit.ammo,
        "units_short_range":     short_range,
        "units_long_range":      long_range,
        "units_second_weapon":   "N",
        "units_symbol":          _AWBW_UNIT_NAMES.get(unit.unit_type, stats.name),
        "units_cost":            stats.cost,
        "units_movement_type":   _awbw_move_type(unit.unit_type),
        "units_x":               unit.pos[1],
        "units_y":               unit.pos[0],
        "units_moved":           1 if unit.moved else 0,
        "units_capture":         0,
        "units_fired":           0,
        # Oracle AWBW uses integer bars 0–10 in action payloads; the viewer
        # applies ``(int)MathF.Ceiling`` to this value (see
        # ``DrawableUnit.UpdateUnit`` in the vendor tree). Passing
        # ``unit.display_hp`` keeps the serialised value on the same scale
        # the parser compares against ``HealthPoints`` (bar count).
        "units_hit_points":      unit.display_hp,
        "units_cargo1_units_id": 0,
        "units_cargo2_units_id": 0,
        "units_carried":         "N",
    }


def _wrap_per_player(obj: dict[str, Any], p0_id: int, p1_id: int) -> dict[str, Any]:
    """Replicate `obj` under keys "global", p0_id, p1_id — AWBW per-player visibility."""
    return {
        "global": obj,
        str(p0_id): obj,
        str(p1_id): obj,
    }


def _manhattan_path(
    start: tuple[int, int],
    end:   tuple[int, int],
) -> list[dict[str, Any]]:
    """Produce a simple L-shaped (Manhattan) path of `unit_visible=true` steps.

    Full AWBW paths are Dijkstra-traced from the engine, but for the viewer's
    animation a monotonic Manhattan path from start→end suffices. Length is
    |dx| + |dy| + 1 (inclusive of both endpoints)."""
    sr, sc = start
    er, ec = end
    path: list[dict[str, Any]] = [{"x": sc, "y": sr, "unit_visible": True}]
    r, c = sr, sc
    while c != ec:
        c += 1 if ec > c else -1
        path.append({"x": c, "y": r, "unit_visible": True})
    while r != er:
        r += 1 if er > r else -1
        path.append({"x": c, "y": r, "unit_visible": True})
    return path


# ---------------------------------------------------------------------------
# Action JSON builders
# ---------------------------------------------------------------------------

def _build_action_json(
    new_unit: Unit,
    player_id: int,
    p0_id: int,
    p1_id: int,
) -> dict[str, Any]:
    unit_json = _unit_to_json(new_unit, player_id)
    return {
        "action":  "Build",
        "newUnit": {"global": unit_json},
        "discovered": {str(p0_id): None, str(p1_id): None},
    }


def _move_action_json(
    moved_unit: Unit,
    path_start: tuple[int, int],
    path_end:   tuple[int, int],
    player_id: int,
    p0_id: int,
    p1_id: int,
    trapped: bool = False,
) -> dict[str, Any]:
    unit_json = _unit_to_json(moved_unit, player_id)
    path = _manhattan_path(path_start, path_end)
    return {
        "action":  "Move",
        "unit":    _wrap_per_player(unit_json, p0_id, p1_id),
        "paths":   {"global": path},
        "dist":    max(0, len(path) - 1),
        "trapped": trapped,
        "discovered": {str(p0_id): None, str(p1_id): None},
    }


def _fire_action_json(
    move_sub: dict[str, Any],
    attacker_post: Unit,
    defender_post: Unit,
    attacker_end_pos: tuple[int, int],
    defender_pos:     tuple[int, int],
    attacker_player_id: int,
    defender_player_id: int,
    attacker_power_bar: int,
    defender_power_bar: int,
) -> dict[str, Any]:
    """Build the full ``Fire`` envelope the viewer expects for an ATTACK.

    Shape matches ``AttackUnitActionBuilder.ParseJObjectIntoReplayAction`` in
    upstream ``AWBWApp.Game/.../AttackUnitActionBuilder.cs`` (DeamonHunter/AWBW-Replay-Player on GitHub):

    - ``Move`` sub-tree: nested ``Move`` action (attacker walking to firing
      tile, carrying *pre-combat* HP/ammo). Pass in pre-built ``move_sub``.
    - ``Fire.combatInfoVision.global.combatInfo.{attacker,defender}``:
      post-combat ``units_ammo`` / ``units_hit_points`` / ``units_id`` /
      ``units_x`` / ``units_y`` (integer bar scale; 0 signals unit death,
      causing the viewer to call ``Map.DeleteUnit``).
    - ``Fire.copValues.{attacker,defender}``: post-combat CO power bar
      absolute values (``copValue``) per side.

    ``tagValue`` is set to ``null`` — we do not emit tag games.
    ``hasVision`` is always ``true``; we never fog-of-war combat visibility
    in generated replays.
    """
    att_end_row, att_end_col = attacker_end_pos
    def_row,     def_col     = defender_pos

    combat_info = {
        "attacker": {
            "units_ammo":        attacker_post.ammo,
            "units_hit_points":  attacker_post.display_hp,
            "units_id":          attacker_post.unit_id,
            "units_x":           att_end_col,
            "units_y":           att_end_row,
        },
        "defender": {
            "units_ammo":        defender_post.ammo,
            "units_hit_points":  defender_post.display_hp,
            "units_id":          defender_post.unit_id,
            "units_x":           def_col,
            "units_y":           def_row,
        },
    }

    return {
        "action": "Fire",
        "Move":   move_sub,
        "Fire": {
            "action": "Fire",
            "combatInfoVision": {
                "global": {
                    "hasVision":  True,
                    "combatInfo": combat_info,
                },
            },
            "copValues": {
                "attacker": {
                    "playerId": attacker_player_id,
                    "copValue": int(attacker_power_bar),
                    "tagValue": None,
                },
                "defender": {
                    "playerId": defender_player_id,
                    "copValue": int(defender_power_bar),
                    "tagValue": None,
                },
            },
        },
    }


def _repair_action_json(
    move_sub: dict[str, Any],
    boat_unit_id: int,
    repaired_unit: Unit,
    funds_after: int,
    p0_id: int,
    p1_id: int,
) -> dict[str, Any]:
    """Build the ``Repair`` envelope for a Black Boat REPAIR action.

    Shape matches ``RepairUnitActionBuilder.ParseJObjectIntoReplayAction`` in
    upstream ``Actions/RepairUnitAction.cs`` (DeamonHunter/AWBW-Replay-Player):

    - ``Move`` sub-tree: the boat's walk to its firing tile (carries
      pre-repair HP/fuel/ammo for the boat).
    - ``Repair.unit``: boat ``units_id``, per-player wrapped.
    - ``Repair.repaired``: target unit's post-repair ``units_id`` +
      ``units_hit_points`` (display bars), per-player wrapped.
    - ``Repair.funds``: active-player treasury after the 10% heal cost is
      debited (wrapped per-player so the viewer's vision system finds it).
    """
    repaired_json = {
        "units_id":         repaired_unit.unit_id,
        "units_hit_points": repaired_unit.display_hp,
    }
    return {
        "action": "Repair",
        "Move":   move_sub,
        "Repair": {
            "unit":     _wrap_per_player(boat_unit_id, p0_id, p1_id),
            "repaired": _wrap_per_player(repaired_json, p0_id, p1_id),
            "funds":    _wrap_per_player(int(funds_after), p0_id, p1_id),
        },
    }


def _attack_seam_action_json(
    move_sub: dict[str, Any],
    attacker_post: Unit,
    attacker_end_pos: tuple[int, int],
    seam_pos: tuple[int, int],
    new_terrain_id: int,
    new_hp_value: int,
    p0_id: int,
    p1_id: int,
) -> dict[str, Any]:
    """Build the ``AttackSeam`` envelope for a seam attack.

    Shape matches ``AttackSeamActionBuilder.ParseJObjectIntoReplayAction`` in
    upstream ``Actions/AttackPipeUnitAction.cs`` (DeamonHunter/AWBW-Replay-Player):

    - ``Move`` sub-tree: attacker walking to the firing tile.
    - ``AttackSeam.buildings_terrain_id``: post-attack terrain (intact id
      when seam survives, broken rubble id when destroyed).
    - ``AttackSeam.buildings_hit_points``: remaining seam HP (0 == destroyed).
    - ``AttackSeam.seamX`` / ``seamY``: seam tile coordinates.
    - ``AttackSeam.unit``: per-player combatInfo with attacker ``units_id``
      and post-fire ``units_ammo`` — the viewer decrements TimesFired and
      updates the attacker's ammo from this block.
    """
    seam_row, seam_col = seam_pos
    att_end_row, att_end_col = attacker_end_pos

    combat_info = {
        "units_id":   attacker_post.unit_id,
        "units_ammo": attacker_post.ammo,
    }

    return {
        "action": "AttackSeam",
        "Move":   move_sub,
        "AttackSeam": {
            "buildings_terrain_id": int(new_terrain_id),
            "buildings_hit_points": int(new_hp_value),
            "seamX":                seam_col,
            "seamY":                seam_row,
            "unit": _wrap_per_player(
                {"combatInfo": combat_info}, p0_id, p1_id,
            ),
        },
    }


def _end_turn_action_json(
    next_player_id: int,
    next_funds: int,
    day: int,
) -> dict[str, Any]:
    return {
        "action": "End",
        "updatedInfo": {
            "event":       "NextTurn",
            "nextPId":     next_player_id,
            "nextFunds":   {"global": int(next_funds)},
            "nextTimer":   0,
            "nextWeather": "C",
            "supplied":    {"global": []},
            "repaired":    {"global": []},
            "day":         int(day),
            "nextTurnStart": "2026-01-01 00:00:00",
        },
    }


# ---------------------------------------------------------------------------
# PHP envelope per player-turn
# ---------------------------------------------------------------------------

def _envelope_player_turn(
    player_id: int,
    day: int,
    turn_number: int,          # 1-indexed count of this player's own turn
    action_jsons: list[dict[str, Any]],
) -> str:
    """Wrap a list of action JSON objects in the PHP p:<pid>;d:<day>;... envelope."""
    body_parts = []
    for i, obj in enumerate(action_jsons):
        action_str = json.dumps(obj, separators=(",", ":"))
        body_parts.append(_php_i(i) + _php_s(action_str))
    actions_array = f"a:{len(action_jsons)}:{{{''.join(body_parts)}}}"

    # The 3-entry outer array shape hard-coded by readReplayActionTurn:
    #   i:0;i:<playerID>;
    #   i:1;i:<turnNumber>;
    #   i:2;<actions_array>;
    inner = (
        _php_i(0) + _php_i(player_id) +
        _php_i(1) + _php_i(turn_number) +
        _php_i(2) + actions_array
    )
    return f"p:{player_id};d:{day};a:a:3:{{{inner}}}"


# ---------------------------------------------------------------------------
# Main: rebuild state from trace and emit per-turn envelopes
# ---------------------------------------------------------------------------

@dataclass
class _TurnBucket:
    player_id: int
    day: int
    turn_number: int
    actions: list[dict[str, Any]]


def _emit_move_or_fire(
    state: GameState,
    action: Action,
    pid_of: dict[int, int],
    p0_id: int,
    p1_id: int,
) -> Optional[dict[str, Any]]:
    """Apply one WAIT/DIVE_HIDE/ATTACK/CAPTURE/LOAD action and return the JSON to append.

    Snapshots attacker (and defender for ATTACK) **before** ``state.step`` so
    the emitted ``Move`` sub-action carries pre-combat HP / ammo / fuel. For
    ATTACK the Move is wrapped in a ``Fire`` envelope populated from
    post-step state (mutated unit references).

    Same-tile actions (``start == end``) are emitted with a single-tile
    ``paths`` array. The vendor viewer's ``MoveUnitAction.PerformAction``
    guards walk animation behind ``Path.Length > 1`` and otherwise just
    calls ``UpdateUnit`` — so a degenerate path is safe.

    Returns ``None`` only when we cannot resolve a moving unit or destination.
    """
    start = action.unit_pos
    end   = action.move_pos
    atype = action.action_type

    moving_unit: Optional[Unit] = None
    if start is not None:
        moving_unit = state.get_unit_at(*start)

    # Pre-step snapshot of the attacker (and the defender for ATTACK). We use
    # ``copy.copy`` because ``Unit`` holds only scalars plus ``loaded_units``
    # (list of other Units, which we don't need to deep-clone for this view).
    attacker_pre: Optional[Unit] = None
    defender_pre: Optional[Unit] = None
    repaired_pre: Optional[Unit] = None
    seam_target: Optional[tuple[int, int]] = None
    if moving_unit is not None:
        attacker_pre = copy.copy(moving_unit)
    if atype == ActionType.ATTACK and action.target_pos is not None:
        defender_pre = state.get_unit_at(*action.target_pos)
        # Track whether this is a seam strike (empty tile w/ terrain 113/114).
        # The post-step terrain flip decides the AttackSeam envelope shape.
        if defender_pre is None:
            tid_pre = state.map_data.terrain[action.target_pos[0]][action.target_pos[1]]
            if tid_pre in (113, 114):
                seam_target = action.target_pos
    if atype == ActionType.REPAIR and action.target_pos is not None:
        repaired_pre = state.get_unit_at(*action.target_pos)

    step_failed = False
    try:
        state.step(action)
    except ValueError:
        # Re-executed state has diverged from the original game (a blocker
        # occupies a tile it didn't hold during the live match). Force-move
        # the attacker to the destination so subsequent lookups succeed; for
        # ATTACK the Fire envelope is skipped because combat never resolved.
        step_failed = True
        if moving_unit is not None and end is not None:
            state._move_unit_forced(moving_unit, end)
            state._finish_action(moving_unit)

    if moving_unit is None or end is None:
        return None

    # Build the Move sub-action from the pre-step snapshot, overriding pos
    # and ``moved`` so the viewer places the unit at the firing tile with
    # the move-spent flag set. Path may be a single tile when ``start == end``
    # (indirect fire, in-place WAIT, on-tile CAPTURE refresh).
    move_unit_snapshot = copy.copy(attacker_pre if attacker_pre is not None else moving_unit)
    move_unit_snapshot.pos = end
    move_unit_snapshot.moved = True
    move_sub = _move_action_json(
        move_unit_snapshot, start, end,
        pid_of[moving_unit.player], p0_id, p1_id,
    )

    # --- REPAIR: emit Repair envelope wrapping the boat's Move ---
    if atype == ActionType.REPAIR:
        if step_failed or repaired_pre is None:
            return move_sub
        # ``repaired_pre`` is the mutated target unit — HP / fuel / ammo are
        # now post-repair. ``moving_unit`` carries the boat's post-step state.
        funds_after = state.funds[moving_unit.player]
        return _repair_action_json(
            move_sub=move_sub,
            boat_unit_id=moving_unit.unit_id,
            repaired_unit=repaired_pre,
            funds_after=funds_after,
            p0_id=p0_id,
            p1_id=p1_id,
        )

    # --- ATTACK vs empty seam: emit AttackSeam envelope ---
    if atype == ActionType.ATTACK and seam_target is not None and not step_failed:
        seam_row, seam_col = seam_target
        # Post-step terrain id tells us whether the seam broke (flipped to
        # 115/116) or survived (still 113/114). New HP comes from seam_hp
        # when intact; 0 when the entry was cleared after breaking.
        new_tid = state.map_data.terrain[seam_row][seam_col]
        new_hp  = state.seam_hp.get(seam_target, 0)
        return _attack_seam_action_json(
            move_sub=move_sub,
            attacker_post=moving_unit,
            attacker_end_pos=end,
            seam_pos=seam_target,
            new_terrain_id=new_tid,
            new_hp_value=new_hp,
            p0_id=p0_id,
            p1_id=p1_id,
        )

    if atype != ActionType.ATTACK or defender_pre is None or step_failed:
        return move_sub

    # For ATTACK, wrap the Move in a Fire envelope with post-combat HP/ammo.
    # ``moving_unit`` and ``defender_pre`` are the same Python objects
    # ``state.step`` mutated — ``hp=0`` means the unit died and was pruned
    # from ``state.units`` but the reference still holds last-known values.
    attacker_post = moving_unit
    defender_post = defender_pre

    att_power = state.co_states[attacker_post.player].power_bar
    def_power = state.co_states[defender_post.player].power_bar

    return _fire_action_json(
        move_sub=move_sub,
        attacker_post=attacker_post,
        defender_post=defender_post,
        attacker_end_pos=end,
        defender_pos=action.target_pos,
        attacker_player_id=pid_of[attacker_post.player],
        defender_player_id=pid_of[defender_post.player],
        attacker_power_bar=att_power,
        defender_power_bar=def_power,
    )


def _rebuild_and_emit(
    full_trace: list[dict],
    map_data: MapData,
    co0: int,
    co1: int,
    tier_name: str = "T2",
    p0_id: int = P0_PLAYER_ID,
    p1_id: int = P1_PLAYER_ID,
) -> list[_TurnBucket]:
    """Replay `full_trace` against a fresh GameState and collect per-turn envelopes."""
    state = make_initial_state(map_data, co0, co1, starting_funds=0, tier_name=tier_name)
    pid_of = {0: p0_id, 1: p1_id}

    buckets: list[_TurnBucket] = []
    turn_counts = [0, 0]     # per-player count of turns taken (1-indexed at emit time)

    def _open_bucket(player: int, day: int) -> _TurnBucket:
        turn_counts[player] += 1
        b = _TurnBucket(
            player_id=pid_of[player],
            day=day,
            turn_number=turn_counts[player],
            actions=[],
        )
        buckets.append(b)
        return b

    # First bucket = P0 day 1 (game opens with P0 to move)
    current = _open_bucket(state.active_player, state.turn)

    for entry in full_trace:
        action = _trace_to_action(entry)
        player_before = state.active_player
        day_before    = state.turn

        atype = action.action_type

        if atype == ActionType.BUILD:
            # Apply, then look up the new unit by position to read its unit_id.
            try:
                state.step(action)
            except Exception:
                continue
            new_unit = state.get_unit_at(*action.move_pos) if action.move_pos else None
            if new_unit is not None:
                current.actions.append(_build_action_json(
                    new_unit, pid_of[new_unit.player], p0_id, p1_id,
                ))
            continue

        if atype in (ActionType.WAIT, ActionType.DIVE_HIDE, ActionType.ATTACK,
                     ActionType.CAPTURE, ActionType.LOAD,
                     ActionType.REPAIR):
            payload = _emit_move_or_fire(state, action, pid_of, p0_id, p1_id)
            if payload is not None:
                current.actions.append(payload)
            continue

        if atype == ActionType.END_TURN:
            state.step(action)
            next_player = state.active_player
            next_day    = state.turn
            # ``updatedInfo.day`` must be the calendar day *after* END_TURN for the
            # player who moves next (matches ``state.turn``). Using the pre-step
            # day breaks C# ``EndTurnAction`` (e.g. fuel burn gated on NextDay > 1).
            current.actions.append(_end_turn_action_json(
                next_player_id=pid_of[next_player],
                next_funds=state.funds[next_player],
                day=next_day,
            ))
            if not state.done:
                current = _open_bucket(next_player, next_day)
            continue

        # SELECT_UNIT, ACTIVATE_COP, ACTIVATE_SCOP — advance state silently.
        # Powers could be surfaced as a PowerAction later; for MVP we let the
        # turn snapshot communicate the transition.
        try:
            state.step(action)
        except Exception:
            pass

    return buckets


def _rebuild_and_emit_with_snapshots(
    full_trace: list[dict],
    map_data: MapData,
    co0: int,
    co1: int,
    tier_name: str = "T2",
    p0_id: int = P0_PLAYER_ID,
    p1_id: int = P1_PLAYER_ID,
) -> tuple[list[GameState], list[_TurnBucket]]:
    """Single-pass variant that returns both turn-start snapshots and buckets.

    The viewer matches each p: envelope to a snapshot via (ActivePlayerID, Day)
    and then looks up units by ``units_id``. Running two independent trace
    replays (one for snapshots, one for actions) risks unit-ID divergence at
    the first force-stepped or skipped action, which breaks every subsequent
    Move lookup and causes the viewer to fall back to turn-start snapping.

    Building snapshots and buckets from the same ``state`` object guarantees
    snapshot unit ids and envelope ``units_id`` values stay in lock-step.
    Snapshots are taken at the start of every player-turn and once more at
    the final state.
    """
    state = make_initial_state(map_data, co0, co1, starting_funds=0, tier_name=tier_name)
    pid_of = {0: p0_id, 1: p1_id}

    snapshots: list[GameState] = [copy.deepcopy(state)]
    buckets: list[_TurnBucket] = []
    turn_counts = [0, 0]

    def _open_bucket(player: int, day: int) -> _TurnBucket:
        turn_counts[player] += 1
        b = _TurnBucket(
            player_id=pid_of[player],
            day=day,
            turn_number=turn_counts[player],
            actions=[],
        )
        buckets.append(b)
        return b

    current = _open_bucket(state.active_player, state.turn)

    for entry in full_trace:
        action = _trace_to_action(entry)
        atype = action.action_type
        day_before = state.turn

        if atype == ActionType.BUILD:
            try:
                state.step(action)
            except Exception:
                continue
            new_unit = state.get_unit_at(*action.move_pos) if action.move_pos else None
            if new_unit is not None:
                current.actions.append(_build_action_json(
                    new_unit, pid_of[new_unit.player], p0_id, p1_id,
                ))
            continue

        if atype in (ActionType.WAIT, ActionType.DIVE_HIDE, ActionType.ATTACK,
                     ActionType.CAPTURE, ActionType.LOAD,
                     ActionType.REPAIR):
            payload = _emit_move_or_fire(state, action, pid_of, p0_id, p1_id)
            if payload is not None:
                current.actions.append(payload)
            continue

        if atype == ActionType.END_TURN:
            try:
                state.step(action)
            except Exception:
                continue
            next_player = state.active_player
            next_day = state.turn
            current.actions.append(_end_turn_action_json(
                next_player_id=pid_of[next_player],
                next_funds=state.funds[next_player],
                day=next_day,
            ))
            if not state.done:
                snapshots.append(copy.deepcopy(state))
                current = _open_bucket(next_player, next_day)
            continue

        try:
            state.step(action)
        except Exception:
            pass

    snapshots.append(copy.deepcopy(state))
    return snapshots, buckets


def build_action_stream_text_from_buckets(buckets: list[_TurnBucket]) -> str:
    """Serialize pre-built buckets into the final `p:` stream text.

    Mirrors ``build_action_stream_text`` but skips the trace replay — used by
    the single-pass path so we don't double-execute the trace.
    """
    # Emit one envelope per bucket, including a:0:{} when a turn had no actions.
    # Dropping empty buckets removed p: lines for those (player, day) pairs while
    # TurnData rows still exist — readReplayActions then never attached Actions,
    # which desynced the viewer against the PHP snapshot stream.
    lines = [
        _envelope_player_turn(b.player_id, b.day, b.turn_number, b.actions)
        for b in buckets
    ]
    return ("\n".join(lines) + "\n") if lines else ""


def write_action_stream_entry(
    replay_zip: str | Path,
    text: str,
    game_id: int,
) -> Path:
    """Compress ``text`` and write/replace entry ``a<game_id>`` in ``replay_zip``.

    Companion to ``append_action_stream_to_zip`` but skips the trace replay;
    the caller supplies an already-built stream (single-pass path).
    """
    replay_zip = Path(replay_zip)
    if not replay_zip.exists():
        raise FileNotFoundError(f"Replay zip not found: {replay_zip}")
    if not text:
        return replay_zip

    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as gz:
        gz.write(text.encode("utf-8"))
    compressed = buf.getvalue()

    action_entry_name = f"a{game_id}"
    existing: dict[str, bytes] = {}
    with zipfile.ZipFile(replay_zip, "r") as zf:
        for info in zf.infolist():
            if info.filename == action_entry_name:
                continue
            existing[info.filename] = zf.read(info.filename)

    with zipfile.ZipFile(replay_zip, "w", zipfile.ZIP_STORED) as zf:
        for name, blob in existing.items():
            zf.writestr(name, blob)
        zf.writestr(action_entry_name, compressed)

    return replay_zip


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def build_action_stream_text(
    full_trace: list[dict],
    map_data: MapData,
    co0: int,
    co1: int,
    tier_name: str = "T2",
    p0_id: int = P0_PLAYER_ID,
    p1_id: int = P1_PLAYER_ID,
) -> str:
    """Full `p:` stream as a single newline-separated string."""
    buckets = _rebuild_and_emit(
        full_trace, map_data, co0, co1, tier_name, p0_id, p1_id,
    )
    lines = [
        _envelope_player_turn(b.player_id, b.day, b.turn_number, b.actions)
        for b in buckets
    ]
    # The viewer's readReplayActions loop checks `text[textIndex++] != '\n'`
    # after each envelope; without a trailing newline it indexes past the end
    # and throws IndexOutOfRangeException (not caught as ArgumentOutOfRange).
    # Oracle replays (e.g. `a1630459`) always terminate with '\n'.
    return ("\n".join(lines) + "\n") if lines else ""


def append_action_stream_to_zip(
    replay_zip: str | Path,
    full_trace: list[dict],
    map_data: MapData,
    co0: int,
    co1: int,
    game_id: int,
    tier_name: str = "T2",
    p0_id: int = P0_PLAYER_ID,
    p1_id: int = P1_PLAYER_ID,
) -> Path:
    """Open `replay_zip` and add/overwrite the `a<game_id>` action-stream entry.

    The viewer picks up the second entry at parse time; presence is what
    flips `ReplayVersion` from 1 → 2 and unlocks per-action stepping.
    """
    replay_zip = Path(replay_zip)
    if not replay_zip.exists():
        raise FileNotFoundError(f"Replay zip not found: {replay_zip}")

    text = build_action_stream_text(
        full_trace, map_data, co0, co1, tier_name, p0_id, p1_id,
    )
    if not text:
        # Nothing to emit — likely an empty trace. Leave the zip untouched.
        return replay_zip

    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as gz:
        gz.write(text.encode("utf-8"))
    compressed = buf.getvalue()

    action_entry_name = f"a{game_id}"

    # Rebuild the zip preserving existing entries, replacing our action entry.
    existing: dict[str, bytes] = {}
    with zipfile.ZipFile(replay_zip, "r") as zf:
        for info in zf.infolist():
            if info.filename == action_entry_name:
                continue
            existing[info.filename] = zf.read(info.filename)

    with zipfile.ZipFile(replay_zip, "w", zipfile.ZIP_STORED) as zf:
        for name, blob in existing.items():
            zf.writestr(name, blob)
        zf.writestr(action_entry_name, compressed)

    return replay_zip


# ---------------------------------------------------------------------------
# CLI — apply to an existing replay + .trace.json pair
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description=(
            "Build or update the p: action stream for an AWBW replay zip.\n\n"
            "If the zip already exists it is updated in-place (a<game_id> entry is\n"
            "replaced). If the zip does NOT exist, pass --from-trace to generate it\n"
            "from scratch using write_awbw_replay_from_trace."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("zip_path", type=Path, help="replays/<game_id>.zip (existing or target path)")
    ap.add_argument("trace_path", type=Path, nargs="?", default=None,
                    help="Matching .trace.json (default: <zip stem>.trace.json)")
    ap.add_argument("--map-pool", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data" / "gl_map_pool.json")
    ap.add_argument("--maps-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data" / "maps")
    ap.add_argument("--from-trace", action="store_true",
                    help="Create the zip from scratch (snapshots + action stream) when no zip exists")
    args = ap.parse_args()

    trace_path = args.trace_path or args.zip_path.with_suffix(".trace.json")
    with open(trace_path, encoding="utf-8") as f:
        trace_record = json.load(f)

    if args.from_trace or not args.zip_path.exists():
        from tools.export_awbw_replay import write_awbw_replay_from_trace
        out = write_awbw_replay_from_trace(
            trace_record=trace_record,
            output_path=args.zip_path,
            map_pool_path=args.map_pool,
            maps_dir=args.maps_dir,
        )
        print(f"[export_actions] zip created from trace -> {out}")
    else:
        map_data = load_map(trace_record["map_id"], args.map_pool, args.maps_dir)
        game_id = int(args.zip_path.stem)
        out = append_action_stream_to_zip(
            replay_zip=args.zip_path,
            full_trace=trace_record["full_trace"],
            map_data=map_data,
            co0=trace_record["co0"],
            co1=trace_record["co1"],
            game_id=game_id,
            tier_name=trace_record.get("tier", "T2"),
        )
        print(f"[export_actions] action stream appended -> {out}")


if __name__ == "__main__":
    _main()
