#!/usr/bin/env python3
"""
Oracle-desync audit for **in-progress** Amarriner games (no replay zip).

Uses the same logged-in session as ``tools/amarriner_download_replays.py`` and
walks ``POST /api/game/load_replay.php`` turn-by-turn (the site replay viewer),
normalizes viewer JSON to the shapes expected by ``apply_oracle_action_json``,
then replays through the engine like ``tools/desync_audit.py``.

Examples::

  python tools/desync_audit_amarriner_live.py --catalog data/amarriner_gl_current_list_p1.json
  python tools/desync_audit_amarriner_live.py --catalog data/amarriner_gl_current_list_p1.json --max-games 5
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import requests
from bs4 import BeautifulSoup
from requests import HTTPError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.action import (  # noqa: E402
    compute_reachable_costs,
    effective_move_cost,
    get_loadable_into,
)
from engine.game import GameState, make_initial_state  # noqa: E402
from engine.map_loader import load_map  # noqa: E402
from engine.unit import Unit  # noqa: E402
from rl.paths import LOGS_DIR, ensure_logs_dir  # noqa: E402
from tools.amarriner_catalog_cos import catalog_row_has_both_cos, pair_catalog_cos_ids  # noqa: E402
from tools.desync_audit import (  # noqa: E402
    CLS_ENGINE_BUG,
    CLS_LOADER_ERROR,
    CLS_OK,
    CLS_ORACLE_GAP,
    AuditRow,
    _ReplayProgress,
    _classify,
    _meta_int,
)
from tools.gl_std_maps import gl_std_map_ids  # noqa: E402
from tools.oracle_zip_replay import (  # noqa: E402
    UnsupportedOracleAction,
    _name_to_unit_type,
    _unit_by_awbw_units_id,
    apply_oracle_action_json,
    map_snapshot_player_ids_to_engine,
    resolve_replay_first_mover,
)

BASE_URL = "https://awbw.amarriner.com"
LOGIN_URL = f"{BASE_URL}/login.php"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}
SECRETS = ROOT / "secrets.txt"


def _login(session: requests.Session, username: str, password: str) -> bool:
    r = session.get(LOGIN_URL, headers=HEADERS, timeout=20)
    soup = BeautifulSoup(r.text, "lxml")
    payload: dict[str, str] = {}
    form = soup.find("form")
    if form:
        for inp in form.find_all("input"):
            name = inp.get("name")
            if name:
                payload[name] = inp.get("value", "")
    payload["username"] = username
    payload["password"] = password
    r2 = session.post(
        LOGIN_URL, data=payload, headers=HEADERS, timeout=20, allow_redirects=True
    )
    return "logout" in r2.text.lower() or username.lower() in r2.text.lower()


def _parse_game_html(html: str, games_id: int) -> tuple[int, int, int]:
    """Return (game_id, turn_p_id, game_day) for first load_replay POST."""
    m = re.search(r"gameId\s*=\s*parseInt\(\s*[\"'](\d+)[\"']\s*\)", html)
    if not m:
        m = re.search(r"gameId\s*=\s*(\d+)", html)
    gid = int(m.group(1)) if m else games_id
    mct = re.search(r"let\s+currentTurn\s*=\s*(\d+)", html)
    mgd = re.search(r"let\s+gameDay\s*=\s*(\d+)", html)
    turn_pid = int(mct.group(1)) if mct else 0
    game_day = int(mgd.group(1)) if mgd else 1
    return gid, turn_pid, game_day


def _live_players_to_snap0(players_block: dict[str, Any]) -> dict[str, Any]:
    out_players: dict[str, Any] = {}
    for key, p in players_block.items():
        if not isinstance(p, dict):
            continue
        pid = int(p.get("players_id", p.get("id", key)))
        out_players[str(pid)] = {
            "id": pid,
            "order": int(p.get("players_order", p.get("order", 0))),
            "co_id": int(p.get("players_co_id", p.get("co_id", 0))),
        }
    return {"players": out_players}


def _normalize_live_viewer_action(act: dict[str, Any]) -> dict[str, Any]:
    a = dict(act)
    k = a.get("action")
    if k == "NextTurn":
        return {"action": "End"}
    # Live ``load_replay.php`` often puts ``buildingInfo`` at the top level; the oracle
    # expects ``Capt: { buildingInfo: ... }`` plus ``Move`` missing/empty to take the
    # no-path ``Capt`` branch (see ``oracle_zip_replay``).
    if k == "Capt" and "Capt" not in a and isinstance(a.get("buildingInfo"), dict):
        a["Capt"] = {"buildingInfo": a["buildingInfo"]}
        if "Move" not in a:
            a["Move"] = []
    if k == "Move" and "path" in a and "paths" not in a:
        raw_path = a.get("path") or []
        gl: list[dict[str, Any]] = []
        if isinstance(raw_path, list):
            for cell in raw_path:
                if isinstance(cell, dict) and "x" in cell and "y" in cell:
                    gl.append({"x": int(cell["x"]), "y": int(cell["y"])})
        a["paths"] = {"global": gl}
    # Live ``load_replay.php`` flattens combatants; zips nest under ``Fire.combatInfoVision``.
    if k == "Unload":
        if a.get("transportID") is None and a.get("transportId") is not None:
            a["transportID"] = int(a.pop("transportId"))
        if "unit" not in a and isinstance(a.get("unloadedUnit"), dict):
            a["unit"] = a["unloadedUnit"]
    if k == "Fire" and "Fire" not in a and (
        isinstance(a.get("attacker"), dict) or isinstance(a.get("defender"), dict)
    ):
        att = dict(a.get("attacker") or {})
        defe = dict(a.get("defender") or {})
        cop = dict(a.get("copValues") or {})
        a["Fire"] = {
            "combatInfoVision": {
                "global": {
                    "hasVision": True,
                    "combatInfo": {"attacker": att, "defender": defe},
                }
            },
            "copValues": cop,
        }
        for drop in ("attacker", "defender", "copValues", "gainedFunds"):
            a.pop(drop, None)
    return a


def _recover_move_path_from_costs(
    state: GameState,
    unit: Unit,
    costs: dict[tuple[int, int], int],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> Optional[list[tuple[int, int]]]:
    if goal not in costs:
        return None
    path: list[tuple[int, int]] = [goal]
    while path[-1] != start:
        cr, cc = path[-1]
        step_cost = effective_move_cost(state, unit, state.map_data.terrain[cr][cc])
        prev: Optional[tuple[int, int]] = None
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            pr, pc = cr - dr, cc - dc
            if (pr, pc) not in costs:
                continue
            if costs[(pr, pc)] + step_cost == costs[(cr, cc)]:
                prev = (pr, pc)
                break
        if prev is None:
            return None
        path.append(prev)
    path.reverse()
    return path


def _sync_engine_unit_ids_from_live_gamestate(state: GameState, live_gs: dict[str, Any]) -> None:
    """Overwrite ``Unit.unit_id`` with AWBW ``units_id`` so live actions match oracle lookups."""
    units_blob = live_gs.get("units") or {}
    if not isinstance(units_blob, dict):
        return
    max_awbw = 0
    for blob in units_blob.values():
        if not isinstance(blob, dict):
            continue
        try:
            awid = int(blob["units_id"])
            r, c = int(blob["units_y"]), int(blob["units_x"])
            ut = _name_to_unit_type(str(blob["units_name"]))
        except (KeyError, TypeError, ValueError):
            continue
        u = state.get_unit_at(r, c)
        if u is None or u.unit_type != ut:
            continue
        u.unit_id = awid
        max_awbw = max(max_awbw, awid)
    cur_max = max((int(u.unit_id) for pl in state.units.values() for u in pl if u.is_alive), default=0)
    state.next_unit_id = max(int(state.next_unit_id), cur_max + 1, max_awbw + 1)


def _unit_from_live_units_blob(
    state: GameState, units_blob: dict[str, Any], awid: int
) -> Optional[Unit]:
    b = units_blob.get(str(awid))
    if b is None:
        b = units_blob.get(awid)
    if not isinstance(b, dict):
        return None
    try:
        r, c = int(b["units_y"]), int(b["units_x"])
        ut = _name_to_unit_type(str(b["units_name"]))
    except (KeyError, TypeError, ValueError):
        return None
    u = state.get_unit_at(r, c)
    if u is None or u.unit_type != ut:
        return None
    return u


def _expand_live_join_to_move(
    state: GameState,
    obj: dict[str, Any],
    awbw_to_engine: dict[int, int],
    *,
    envelope_index: int,
    per_turn_units: list[dict[str, Any]],
    actions_prefix: list[dict[str, Any]],
) -> dict[str, Any]:
    if obj.get("action") != "Join":
        return obj
    if isinstance(obj.get("Move"), dict):
        return obj
    joined = obj.get("joinedUnit") or {}
    if not isinstance(joined, dict):
        return obj
    jid = obj.get("joinId")
    joiner: Optional[Unit] = None
    if jid is not None:
        joiner = _unit_by_awbw_units_id(state, int(jid))
    if joiner is None:
        units_before: dict[str, Any] = {}
        if envelope_index > 0 and envelope_index - 1 < len(per_turn_units):
            units_before = per_turn_units[envelope_index - 1]
        if jid is not None and isinstance(units_before, dict):
            joiner = _unit_from_live_units_blob(state, units_before, int(jid))
    if joiner is None:
        for prev_move in reversed([p for p in actions_prefix if p.get("action") == "Move"]):
            cells = (prev_move.get("paths") or {}).get("global") or []
            if not isinstance(cells, list) or len(cells) == 0:
                continue
            last = cells[-1]
            try:
                er, ec = int(last["y"]), int(last["x"])
                cand = state.get_unit_at(er, ec)
            except (KeyError, TypeError, ValueError):
                continue
            if cand is not None:
                joiner = cand
                break
    if joiner is None:
        raise UnsupportedOracleAction(f"Join joinId={jid!r}: could not resolve joiner unit")
    try:
        goal = (int(joined["units_y"]), int(joined["units_x"]))
    except (KeyError, TypeError, ValueError) as e:
        raise UnsupportedOracleAction(f"Join joinedUnit missing coords: {joined!r}") from e
    costs = compute_reachable_costs(state, joiner)
    path_cells = _recover_move_path_from_costs(state, joiner, costs, joiner.pos, goal)
    if path_cells is None:
        raise UnsupportedOracleAction(f"Join: no path from {joiner.pos} to {goal}")
    inv_awbw = {int(v): int(k) for k, v in awbw_to_engine.items()}
    awbw_pid = inv_awbw.get(int(joiner.player))
    if awbw_pid is None:
        raise UnsupportedOracleAction(f"Join: no AWBW player id for engine seat {joiner.player}")
    from engine.unit import UNIT_STATS  # noqa: PLC0415

    path_json = [{"x": int(c), "y": int(r)} for (r, c) in path_cells]
    move = {
        "paths": {"global": path_json},
        "unit": {
            "units_id": int(joiner.unit_id),
            "units_x": int(joiner.pos[1]),
            "units_y": int(joiner.pos[0]),
            "units_players_id": awbw_pid,
            "units_name": UNIT_STATS[joiner.unit_type].name,
        },
    }
    out = {"action": "Join", "Move": move}
    return out


def _expand_live_load_to_move(
    state: GameState,
    obj: dict[str, Any],
    awbw_to_engine: dict[int, int],
    *,
    envelope_index: int,
    per_turn_units: list[dict[str, Any]],
    actions_prefix: list[dict[str, Any]],
) -> dict[str, Any]:
    """Turn minimal live ``Load`` (ids only) into a zip-shaped ``Load`` + nested ``Move``."""
    if obj.get("action") != "Load":
        return obj
    if isinstance(obj.get("Move"), dict):
        return obj
    tid = obj.get("transportId")
    lid = obj.get("loadedId")
    if tid is None or lid is None:
        return obj
    units_before: dict[str, Any] = {}
    if envelope_index > 0 and envelope_index - 1 < len(per_turn_units):
        units_before = per_turn_units[envelope_index - 1]
    cargo: Optional[Unit] = _unit_by_awbw_units_id(state, int(lid))
    transport: Optional[Unit] = _unit_by_awbw_units_id(state, int(tid))
    if cargo is None and isinstance(units_before, dict):
        cargo = _unit_from_live_units_blob(state, units_before, int(lid))
    if transport is None and isinstance(units_before, dict):
        transport = _unit_from_live_units_blob(state, units_before, int(tid))
    if cargo is None:
        moves = [p for p in actions_prefix if p.get("action") == "Move"]
        for prev_move in reversed(moves):
            cells = (prev_move.get("paths") or {}).get("global") or []
            if not isinstance(cells, list) or len(cells) == 0:
                continue
            last = cells[-1]
            try:
                er, ec = int(last["y"]), int(last["x"])
                cand = state.get_unit_at(er, ec)
            except (KeyError, TypeError, ValueError):
                continue
            if cand is not None:
                cargo = cand
                break
    if cargo is None and transport is not None:
        tr, tc = transport.pos
        moves = [p for p in actions_prefix if p.get("action") == "Move"]
        for prev_move in reversed(moves):
            cells = (prev_move.get("paths") or {}).get("global") or []
            if not isinstance(cells, list) or len(cells) == 0:
                continue
            last = cells[-1]
            try:
                er, ec = int(last["y"]), int(last["x"])
            except (KeyError, TypeError, ValueError):
                continue
            cand = state.get_unit_at(er, ec)
            if cand is None or int(cand.player) != int(transport.player):
                continue
            if abs(er - tr) + abs(ec - tc) != 1:
                continue
            if cand.unit_type in get_loadable_into(transport.unit_type):
                cargo = cand
                break
    if cargo is None or transport is None:
        raise UnsupportedOracleAction(
            f"live Load: could not resolve cargo/transport "
            f"(loadedId={lid}, transportId={tid})"
        )
    inv_awbw = {int(v): int(k) for k, v in awbw_to_engine.items()}
    awbw_pid = inv_awbw.get(int(cargo.player))
    if awbw_pid is None:
        raise UnsupportedOracleAction(f"no AWBW player id for engine seat {cargo.player}")
    start = cargo.pos
    goal = transport.pos
    costs = compute_reachable_costs(state, cargo)
    cells = _recover_move_path_from_costs(state, cargo, costs, start, goal)
    if cells is None:
        raise UnsupportedOracleAction(
            f"live Load: no move path from {start} to transport at {goal}"
        )
    path_json = [{"x": int(c), "y": int(r)} for (r, c) in cells]
    from engine.unit import UNIT_STATS  # noqa: PLC0415 — keep import local (heavy table)

    unit_payload = {
        "units_id": int(cargo.unit_id),
        "units_x": int(start[1]),
        "units_y": int(start[0]),
        "units_players_id": awbw_pid,
        "units_name": UNIT_STATS[cargo.unit_type].name,
    }
    move = {"paths": {"global": path_json}, "unit": unit_payload}
    out = {"action": "Load", "Move": move}
    if isinstance(obj.get("Load"), dict):
        out["Load"] = obj["Load"]
    return out


def _run_live_replay_instrumented(
    state: GameState,
    envelopes: list[tuple[int, int, list[dict[str, Any]]]],
    awbw_to_engine: dict[int, int],
    progress: _ReplayProgress,
    per_turn_units: list[dict[str, Any]],
) -> Optional[BaseException]:
    """Same as ``desync_audit._run_replay_instrumented`` but expands live-only shapes."""
    progress.envelopes_total = len(envelopes)
    for env_i, (_pid, day, actions) in enumerate(envelopes):
        prefix: list[dict[str, Any]] = []
        for raw in actions:
            if state.done:
                return None
            norm0 = _normalize_live_viewer_action(raw)
            try:
                obj = _expand_live_join_to_move(
                    state,
                    norm0,
                    awbw_to_engine,
                    envelope_index=env_i,
                    per_turn_units=per_turn_units,
                    actions_prefix=prefix,
                )
                obj = _expand_live_load_to_move(
                    state,
                    obj,
                    awbw_to_engine,
                    envelope_index=env_i,
                    per_turn_units=per_turn_units,
                    actions_prefix=prefix,
                )
            except UnsupportedOracleAction as exc:
                return exc
            progress.last_day = day
            progress.last_action_kind = str(obj.get("action") or "?")
            progress.last_envelope_index = env_i
            try:
                apply_oracle_action_json(
                    state, obj, awbw_to_engine, envelope_awbw_player_id=_pid
                )
            except Exception as exc:  # noqa: BLE001
                return exc
            progress.actions_applied += 1
            prefix.append(norm0)
            if state.done:
                progress.envelopes_applied = env_i + 1
                return None
        progress.envelopes_applied = env_i + 1
    return None


def _infer_envelope_awbw_pid(actions: list[dict[str, Any]], request_turn_pid: int) -> int:
    for act in actions:
        if not isinstance(act, dict):
            continue
        k = act.get("action")
        if k in (None, "NextTurn"):
            continue
        if k == "Build":
            gu = act.get("newUnit") or {}
            if gu.get("units_players_id") is not None:
                return int(gu["units_players_id"])
        if k == "Move":
            u = act.get("unit") or {}
            if u.get("units_players_id") is not None:
                return int(u["units_players_id"])
        u = act.get("unit") or act.get("newUnit") or {}
        if isinstance(u, dict) and u.get("units_players_id") is not None:
            return int(u["units_players_id"])
    return int(request_turn_pid)


def _http_get_text(session: requests.Session, url: str, *, attempts: int = 4) -> str:
    last: Optional[BaseException] = None
    for i in range(attempts):
        try:
            r = session.get(url, headers=HEADERS, timeout=60)
            r.raise_for_status()
            return r.text
        except (HTTPError, requests.RequestException) as e:
            last = e
            if i + 1 < attempts:
                time.sleep(0.8 * (i + 1))
    assert last is not None
    raise last


def _http_post_json(
    session: requests.Session, url: str, body: dict[str, Any], *, attempts: int = 4
) -> dict[str, Any]:
    last: Optional[BaseException] = None
    for i in range(attempts):
        try:
            r = session.post(
                url,
                json=body,
                headers={**HEADERS, "Content-Type": "application/json"},
                timeout=90,
            )
            r.raise_for_status()
            return r.json()
        except (HTTPError, requests.RequestException) as e:
            last = e
            if i + 1 < attempts:
                time.sleep(0.8 * (i + 1))
    assert last is not None
    raise last


def _fetch_live_envelopes(
    session: requests.Session,
    *,
    games_id: int,
    sleep_s: float,
) -> tuple[
    list[tuple[int, int, list[dict[str, Any]]]],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
]:
    html = _http_get_text(session, f"{BASE_URL}/game.php?games_id={games_id}")
    gid, req_pid, req_day = _parse_game_html(html, games_id)

    envelopes: list[tuple[int, int, list[dict[str, Any]]]] = []
    first_snap: Optional[dict[str, Any]] = None
    per_turn_units: list[dict[str, Any]] = []

    def post_turn(turn: int, pid: int, day: int, initial: bool) -> dict[str, Any]:
        body = {
            "gameId": gid,
            "turn": turn,
            "turnPId": pid,
            "turnDay": day,
            "initial": initial,
        }
        return _http_post_json(session, f"{BASE_URL}/api/game/load_replay.php", body)

    data0 = post_turn(0, req_pid, req_day, True)
    if data0.get("err"):
        raise ValueError(f"load_replay turn 0: {data0!r}")
    gs0 = data0.get("gameState") or {}
    players0 = gs0.get("players") or {}
    if not isinstance(players0, dict):
        raise ValueError("gameState.players missing")
    first_snap = _live_players_to_snap0(players0)
    first_snap["turn"] = int(req_pid)

    day_sel = data0.get("daySelector") or []
    n_turns = len(day_sel) if isinstance(day_sel, list) and len(day_sel) > 0 else 1

    for turn_ndx in range(n_turns):
        if turn_ndx == 0:
            data = data0
        else:
            time.sleep(sleep_s)
            data = post_turn(turn_ndx, req_pid, req_day, False)
        if data.get("err"):
            raise ValueError(f"load_replay turn {turn_ndx}: {data!r}")
        raw_actions = data.get("actions") or []
        if not isinstance(raw_actions, list):
            raw_actions = []
        env_pid = _infer_envelope_awbw_pid(raw_actions, req_pid)
        day = int(data.get("day") or req_day)
        norm = [_normalize_live_viewer_action(a) for a in raw_actions if isinstance(a, dict)]
        envelopes.append((env_pid, day, norm))
        gs = data.get("gameState") or {}
        raw_u = gs.get("units") or {}
        per_turn_units.append(json.loads(json.dumps(raw_u)) if isinstance(raw_u, dict) else {})
        req_pid = int(gs["currentTurnPId"])
        req_day = int(data.get("day") or req_day)

    assert first_snap is not None
    return envelopes, first_snap, gs0, per_turn_units


def _audit_live_game(
    session: requests.Session,
    meta: dict[str, Any],
    *,
    map_pool: Path,
    maps_dir: Path,
    sleep_s: float,
) -> AuditRow:
    gid = int(meta["games_id"])
    co_p0, co_p1 = pair_catalog_cos_ids(meta)
    map_id = _meta_int(meta, "map_id")
    tier = str(meta.get("tier", ""))
    matchup = str(meta.get("matchup", ""))
    base = AuditRow(
        games_id=gid,
        map_id=map_id,
        tier=tier,
        co_p0_id=co_p0,
        co_p1_id=co_p1,
        matchup=matchup,
        zip_path=f"live:{BASE_URL}/game.php?games_id={gid}",
        status="ok",
        cls=CLS_OK,
        exception_type="",
        message="",
        approx_day=None,
        approx_action_kind=None,
        approx_envelope_index=None,
        envelopes_total=0,
        envelopes_applied=0,
        actions_applied=0,
    )
    try:
        envelopes, snap0, live_gs0, per_turn_units = _fetch_live_envelopes(
            session, games_id=gid, sleep_s=sleep_s
        )
        awbw_to_engine = map_snapshot_player_ids_to_engine(snap0, co_p0, co_p1)
        map_data = load_map(map_id, map_pool, maps_dir)
        first_mover = resolve_replay_first_mover(envelopes, snap0, awbw_to_engine)
        state = make_initial_state(
            map_data,
            co_p0,
            co_p1,
            starting_funds=0,
            tier_name=tier or "T2",
            replay_first_mover=first_mover,
        )
        _sync_engine_unit_ids_from_live_gamestate(state, live_gs0)
    except Exception as exc:  # noqa: BLE001
        base.status = "first_divergence"
        cls, et, msg = _classify(exc)
        base.cls = cls if cls != CLS_ENGINE_BUG else CLS_LOADER_ERROR
        base.exception_type = et
        base.message = msg
        return base

    progress = _ReplayProgress()
    exc = _run_live_replay_instrumented(
        state, envelopes, awbw_to_engine, progress, per_turn_units
    )
    base.envelopes_total = progress.envelopes_total
    base.envelopes_applied = progress.envelopes_applied
    base.actions_applied = progress.actions_applied

    if exc is None:
        base.status = "ok"
        base.cls = CLS_OK
        return base

    base.status = "first_divergence"
    base.approx_day = progress.last_day
    base.approx_action_kind = progress.last_action_kind
    base.approx_envelope_index = progress.last_envelope_index
    cls, et, msg = _classify(exc)
    base.cls = cls
    base.exception_type = et
    base.message = msg
    return base


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", type=Path, default=ROOT / "data/amarriner_gl_current_list_p1.json")
    ap.add_argument("--map-pool", type=Path, default=ROOT / "data/gl_map_pool.json")
    ap.add_argument("--maps-dir", type=Path, default=ROOT / "data/maps")
    ap.add_argument("--register", type=Path, default=LOGS_DIR / "desync_register_amarriner_live.jsonl")
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--games-id", type=int, action="append", default=None, help="Restrict to these games_id values")
    ap.add_argument("--sleep", type=float, default=0.35, help="Seconds between load_replay POSTs (after turn 0)")
    args = ap.parse_args()

    if not args.catalog.is_file():
        print(f"[live_audit] missing catalog {args.catalog}", file=sys.stderr)
        return 1
    if not args.map_pool.is_file():
        print(f"[live_audit] missing map pool {args.map_pool}", file=sys.stderr)
        return 1
    if not SECRETS.is_file():
        print(f"[live_audit] missing {SECRETS}", file=sys.stderr)
        return 1

    creds = SECRETS.read_text(encoding="utf-8").strip().splitlines()
    if len(creds) < 2:
        print("[live_audit] secrets.txt: line1 user, line2 password", file=sys.stderr)
        return 1

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    games = catalog.get("games") or {}
    std_ids = gl_std_map_ids(args.map_pool)
    rows_meta: list[tuple[int, dict[str, Any]]] = []
    gid_filter = set(args.games_id) if args.games_id else None
    for _k, g in games.items():
        if not isinstance(g, dict) or "games_id" not in g:
            continue
        gid = int(g["games_id"])
        if gid_filter is not None and gid not in gid_filter:
            continue
        mid = g.get("map_id")
        if mid is None or int(mid) not in std_ids:
            continue
        if not catalog_row_has_both_cos(g):
            continue
        rows_meta.append((gid, g))
    rows_meta.sort(key=lambda t: t[0])
    if args.max_games is not None:
        rows_meta = rows_meta[: max(0, args.max_games)]

    session = requests.Session()
    if not _login(session, creds[0].strip(), creds[1].strip()):
        print("[live_audit] login failed", file=sys.stderr)
        return 1

    ensure_logs_dir()
    args.register.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with open(args.register, "w", encoding="utf-8") as f:
        for gid, meta in rows_meta:
            row = _audit_live_game(
                session,
                meta,
                map_pool=args.map_pool,
                maps_dir=args.maps_dir,
                sleep_s=args.sleep,
            )
            f.write(json.dumps(row.to_json(), ensure_ascii=False) + "\n")
            f.flush()
            counts[row.cls] = counts.get(row.cls, 0) + 1
            tail = (row.message or "")[:88].replace("\n", " ")
            print(f"[{gid}] {row.cls:<22} envs={row.envelopes_total} acts={row.actions_applied} | {tail}")

    print()
    print(f"[live_audit] register -> {args.register}")
    print(f"[live_audit] {len(rows_meta)} games")
    for k in sorted(counts):
        print(f"  {k:<26} {counts[k]:>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
