"""
Replay AWBW Replay Player ``p:`` action JSON through the engine (best-effort).

Designed first for zips **produced by this repo** (``write_awbw_replay_from_trace``),
where Move/Build/Fire/End shapes match ``tools/export_awbw_replay_actions.py``.
Live-site oracle zips may include extra action kinds; those raise
``UnsupportedOracleAction`` until mapped.

PHP snapshot loading reuses ``tools.diff_replay_zips.load_replay`` (first zip
member = gzipped turn lines).
"""
from __future__ import annotations

import gzip
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from engine.action import Action, ActionStage, ActionType, get_legal_actions
from engine.game import GameState, make_initial_state
from engine.map_loader import load_map
from engine.unit import UnitType

from tools.diff_replay_zips import load_replay
from tools.export_awbw_replay import _AWBW_UNIT_NAMES

EngineStepHook = Optional[Callable[[GameState, Action], None]]


class UnsupportedOracleAction(ValueError):
    pass


class ReplayAborted(Exception):
    """Stop applying oracle actions (e.g. site ``Resign``); replay is truncated."""



def _engine_step(state: GameState, act: Action, hook: EngineStepHook) -> None:
    if hook is not None:
        hook(state, act)
    state.step(act)


def _global_unit(obj: dict[str, Any]) -> dict[str, Any]:
    u = obj.get("unit") or obj.get("newUnit") or {}
    if "global" in u:
        return u["global"]
    return u


def _name_to_unit_type(name: str) -> UnitType:
    n = str(name).strip()
    # Site JSON sometimes omits the space used in ``export_awbw_replay`` strings.
    aliases = {"Md.Tank": "Md. Tank"}
    n = aliases.get(n, n)
    for ut, nm in _AWBW_UNIT_NAMES.items():
        if nm == n:
            return ut
    raise UnsupportedOracleAction(f"unknown AWBW unit name {name!r}")


def map_snapshot_player_ids_to_engine(
    snap0: dict[str, Any],
    co0: int,
    co1: int,
) -> dict[int, int]:
    """Map AWBW ``players[].id`` (PHP int) -> engine player 0/1 from CO assignment."""
    players = snap0.get("players") or {}
    rows: list[tuple[int, int, int]] = []
    for _k, p in players.items():
        if not isinstance(p, dict):
            continue
        pid = int(p["id"])
        order = int(p.get("order", 0))
        cid = int(p["co_id"])
        rows.append((order, pid, cid))
    rows.sort(key=lambda t: t[0])
    if len(rows) < 2:
        raise ValueError(f"expected >=2 players in snapshot, got {rows!r}")
    rows = rows[:2]
    (_, id_a, c_a), (_, id_b, c_b) = rows
    if {c_a, c_b} != {co0, co1}:
        raise ValueError(f"snapshot CO ids {c_a},{c_b} != expected {co0},{co1}")
    if c_a == co0:
        return {id_a: 0, id_b: 1}
    return {id_a: 1, id_b: 0}


def extract_json_action_strings_from_envelope_line(line: str) -> list[str]:
    """Scan PHP envelope line for serialized JSON strings (``s:len:"..."``)."""
    bodies: list[str] = []
    i = 0
    while i < len(line):
        j = line.find("s:", i)
        if j < 0:
            break
        k = line.find(":", j + 2)
        try:
            n = int(line[j + 2 : k])
        except ValueError:
            i = j + 2
            continue
        if k + 1 >= len(line) or line[k + 1] != '"':
            i = j + 2
            continue
        start = k + 2
        body = line[start : start + n]
        if body.startswith('{"action":'):
            bodies.append(body)
        i = start + n + 2
    return bodies


def parse_p_envelopes_from_zip(path: Path) -> list[tuple[int, int, list[dict[str, Any]]]]:
    """
    Return list of (awbw_player_id, day, [action_dict, ...]) in file order.

    Reads the gz member whose decompressed text contains ``p:`` lines (``a<game_id>``).
    """
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        action_name = None
        for n in names:
            if n.startswith("a"):
                action_name = n
                break
        if action_name is None:
            raise FileNotFoundError(f"no a<game_id> entry in {path!s}")
        raw = zf.read(action_name)

    txt = gzip.decompress(raw).decode("utf-8", "replace")
    out: list[tuple[int, int, list[dict[str, Any]]]] = []
    for line in txt.split("\n"):
        line = line.strip()
        if not line.startswith("p:"):
            continue
        m = re.match(r"p:(\d+);d:(\d+);", line)
        if not m:
            continue
        pid = int(m.group(1))
        day = int(m.group(2))
        blobs = extract_json_action_strings_from_envelope_line(line)
        actions = []
        for b in blobs:
            try:
                actions.append(json.loads(b))
            except json.JSONDecodeError as e:
                raise ValueError(f"bad JSON in envelope: {e}") from e
        out.append((pid, day, actions))
    return out


def _path_end_rc(path: list[dict[str, Any]]) -> tuple[int, int]:
    last = path[-1]
    return int(last["y"]), int(last["x"])


def _path_start_rc(path: list[dict[str, Any]]) -> tuple[int, int]:
    first = path[0]
    return int(first["y"]), int(first["x"])


def _apply_move_paths_then_terminator(
    state: GameState,
    move: dict[str, Any],
    awbw_to_engine: dict[int, int],
    before_engine_step: EngineStepHook,
    *,
    after_move: Callable[[], None],
) -> None:
    """SELECT + path move like ``Move`` / ``Fire`` / ``Capt``, then caller-supplied terminator."""
    paths = (move.get("paths") or {}).get("global") or []
    if not paths:
        raise UnsupportedOracleAction("Move without paths.global")
    sr, sc = _path_start_rc(paths)
    er, ec = _path_end_rc(paths)
    gu = _global_unit(move)
    pid = int(gu["units_players_id"])
    eng = awbw_to_engine[pid]
    if int(state.active_player) != eng:
        raise UnsupportedOracleAction(
            f"Move for engine P{eng} but active_player={state.active_player}"
        )
    uid = int(gu["units_id"])
    u = None
    for pl in state.units.values():
        for x in pl:
            if x.unit_id == uid and x.is_alive:
                u = x
                break
    if u is None:
        u = state.get_unit_at(sr, sc)
    if u is None:
        raise UnsupportedOracleAction(f"Move: no unit id {uid} at ({sr},{sc})")
    start = u.pos
    if start != (sr, sc):
        start = (sr, sc)
    end = (er, ec)
    _engine_step(
        state, Action(ActionType.SELECT_UNIT, unit_pos=start), before_engine_step
    )
    _engine_step(
        state,
        Action(ActionType.SELECT_UNIT, unit_pos=start, move_pos=end),
        before_engine_step,
    )
    after_move()
    return


def apply_oracle_action_json(
    state: GameState,
    obj: dict[str, Any],
    awbw_to_engine: dict[int, int],
    before_engine_step: EngineStepHook = None,
) -> None:
    """Mutate ``state`` by applying one oracle viewer JSON action."""
    kind = obj.get("action")
    if kind == "Resign":
        raise ReplayAborted()
    if kind == "End":
        _engine_step(state, Action(ActionType.END_TURN), before_engine_step)
        return

    if kind == "Build":
        gu = _global_unit(obj)
        r, c = int(gu["units_y"]), int(gu["units_x"])
        ut = _name_to_unit_type(str(gu["units_name"]))
        pid = int(gu["units_players_id"])
        eng = awbw_to_engine[pid]
        if int(state.active_player) != eng:
            raise UnsupportedOracleAction(
                f"Build for player {eng} but active_player={state.active_player}"
            )
        _engine_step(
            state,
            Action(ActionType.BUILD, move_pos=(r, c), unit_type=ut),
            before_engine_step,
        )
        return

    if kind == "Move":
        def _after_move_move() -> None:
            legal = get_legal_actions(state)
            chosen: Optional[Action] = None
            paths_g = (obj.get("paths") or {}).get("global") or []
            end = _path_end_rc(paths_g)
            # Prefer JOIN/LOAD/CAPTURE over WAIT — AWBW may list WAIT first but site
            # uses a separate ``Capt`` envelope for capture ticks after move-on-tile.
            for a in legal:
                if a.action_type == ActionType.JOIN and a.move_pos == end:
                    chosen = a
                    break
            if chosen is None:
                for a in legal:
                    if a.action_type == ActionType.LOAD and a.move_pos == end:
                        chosen = a
                        break
            if chosen is None:
                for a in legal:
                    if a.action_type == ActionType.CAPTURE and a.move_pos == end:
                        chosen = a
                        break
            if chosen is None:
                for a in legal:
                    if a.action_type == ActionType.WAIT and a.move_pos == end:
                        chosen = a
                        break
            if chosen is None:
                er, ec = _path_end_rc(paths_g)
                raise UnsupportedOracleAction(
                    f"Move resolved to ACTION but no JOIN/LOAD/CAPTURE/WAIT at {(er, ec)}; "
                    f"legal={[a.action_type.name for a in legal]}"
                )
            _engine_step(state, chosen, before_engine_step)

        _apply_move_paths_then_terminator(
            state, obj, awbw_to_engine, before_engine_step, after_move=_after_move_move
        )
        return

    if kind == "Capt":
        move_raw = obj.get("Move")
        if isinstance(move_raw, list) or not move_raw:
            cap = obj.get("Capt")
            if not isinstance(cap, dict):
                raise UnsupportedOracleAction("Capt with empty Move but no nested Capt")
            bi = cap.get("buildingInfo") or {}
            er, ec = int(bi["buildings_y"]), int(bi["buildings_x"])
            u = state.get_unit_at(er, ec)
            if u is None:
                raise UnsupportedOracleAction(f"Capt (no path): no unit on tile ({er},{ec})")
            eng = int(state.active_player)
            if int(u.player) != eng:
                raise UnsupportedOracleAction(
                    f"Capt no-path unit owner P{u.player} != active_player={eng}"
                )
            # AWBW often emits ``Capt`` with ``Move:[]`` on the *next* line after a
            # move-on-tile envelope; the engine may still be in MOVE with the unit
            # selected but ``selected_move_pos`` not yet committed.
            if (
                state.action_stage == ActionStage.MOVE
                and state.selected_unit is not None
                and state.selected_unit.pos == (er, ec)
                and state.selected_move_pos is None
            ):
                _engine_step(
                    state,
                    Action(ActionType.SELECT_UNIT, unit_pos=(er, ec), move_pos=(er, ec)),
                    before_engine_step,
                )
            elif state.action_stage == ActionStage.SELECT:
                _engine_step(
                    state, Action(ActionType.SELECT_UNIT, unit_pos=u.pos), before_engine_step
                )
                _engine_step(
                    state,
                    Action(ActionType.SELECT_UNIT, unit_pos=u.pos, move_pos=u.pos),
                    before_engine_step,
                )
            else:
                raise UnsupportedOracleAction(
                    f"Capt no-path: unexpected stage={state.action_stage.name} "
                    f"sel={state.selected_unit!s} mpos={state.selected_move_pos}"
                )
            legal = get_legal_actions(state)
            cap_act: Optional[Action] = None
            for a in legal:
                if a.action_type == ActionType.CAPTURE and a.move_pos == (er, ec):
                    cap_act = a
                    break
            if cap_act is None:
                raise UnsupportedOracleAction(
                    f"Capt no-path: no CAPTURE at {(er, ec)}; legal={[x.action_type.name for x in legal]}"
                )
            _engine_step(state, cap_act, before_engine_step)
            return

        move = move_raw
        paths = (move.get("paths") or {}).get("global") or []

        def _after_move_capt() -> None:
            legal = get_legal_actions(state)
            end = _path_end_rc(paths)
            chosen: Optional[Action] = None
            for a in legal:
                if a.action_type == ActionType.CAPTURE and a.move_pos == end:
                    chosen = a
                    break
            if chosen is None:
                for a in legal:
                    if a.action_type == ActionType.WAIT and a.move_pos == end:
                        chosen = a
                        break
            if chosen is None:
                raise UnsupportedOracleAction(
                    f"Capt: no CAPTURE/WAIT at {end}; legal={[a.action_type.name for a in legal]}"
                )
            _engine_step(state, chosen, before_engine_step)

        _apply_move_paths_then_terminator(
            state, move, awbw_to_engine, before_engine_step, after_move=_after_move_capt
        )
        return

    if kind == "Fire":
        move = obj.get("Move") or {}
        paths = (move.get("paths") or {}).get("global") or []
        if not paths:
            raise UnsupportedOracleAction("Fire without Move.paths.global")
        sr, sc = _path_start_rc(paths)
        er, ec = _path_end_rc(paths)
        gu = _global_unit(move)
        pid = int(gu["units_players_id"])
        eng = awbw_to_engine[pid]
        if int(state.active_player) != eng:
            raise UnsupportedOracleAction(
                f"Fire for engine P{eng} but active_player={state.active_player}"
            )
        fi = (((obj.get("Fire") or {}).get("combatInfoVision") or {}).get("global") or {}).get(
            "combatInfo"
        ) or {}
        defender = fi.get("defender") or {}
        dr, dc = int(defender["units_y"]), int(defender["units_x"])
        uid = int(gu["units_id"])
        u = None
        for pl in state.units.values():
            for x in pl:
                if x.unit_id == uid and x.is_alive:
                    u = x
                    break
        if u is None:
            u = state.get_unit_at(sr, sc)
        if u is None:
            raise UnsupportedOracleAction(f"Fire: no attacker unit id {uid}")
        start = u.pos if u.pos == (sr, sc) else (sr, sc)
        _engine_step(
            state, Action(ActionType.SELECT_UNIT, unit_pos=start), before_engine_step
        )
        _engine_step(
            state,
            Action(ActionType.SELECT_UNIT, unit_pos=start, move_pos=(er, ec)),
            before_engine_step,
        )
        _engine_step(
            state,
            Action(
                ActionType.ATTACK,
                unit_pos=start,
                move_pos=(er, ec),
                target_pos=(dr, dc),
            ),
            before_engine_step,
        )
        return

    raise UnsupportedOracleAction(f"unsupported oracle action {kind!r}")


@dataclass
class OracleReplayResult:
    final_state: GameState
    envelopes_applied: int
    actions_applied: int


def replay_oracle_zip(
    zip_path: Path,
    *,
    map_pool: Path,
    maps_dir: Path,
    map_id: int,
    co0: int,
    co1: int,
    tier_name: str,
    on_skip: Optional[Callable[[str], None]] = None,
    before_engine_step: EngineStepHook = None,
) -> OracleReplayResult:
    frames = load_replay(zip_path)
    if not frames:
        raise ValueError("empty replay")
    awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co0, co1)
    map_data = load_map(map_id, map_pool, maps_dir)
    state = make_initial_state(map_data, co0, co1, starting_funds=0, tier_name=tier_name)
    envs = parse_p_envelopes_from_zip(zip_path)
    n_act = 0
    aborted = False
    for _pid, _day, actions in envs:
        for obj in actions:
            try:
                apply_oracle_action_json(
                    state, obj, awbw_to_engine, before_engine_step=before_engine_step
                )
                n_act += 1
            except ReplayAborted:
                aborted = True
                break
            except UnsupportedOracleAction as e:
                if on_skip:
                    on_skip(str(e))
                raise
        if aborted:
            break
    return OracleReplayResult(
        final_state=state,
        envelopes_applied=len(envs),
        actions_applied=n_act,
    )
