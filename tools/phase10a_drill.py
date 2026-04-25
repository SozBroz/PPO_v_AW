"""Drill a single B_COPTER engine_bug gid. Print envelope/action context near
the failure plus the engine's compute_reachable_costs view of the attacker.

Usage:
    python tools/phase10a_drill.py 1631621
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.action import compute_reachable_costs, get_attack_targets
from engine.game import GameState, make_initial_state  # noqa
from engine.map_loader import load_map  # noqa
from engine.unit import UNIT_STATS, UnitType  # noqa

from tools.amarriner_catalog_cos import pair_catalog_cos_ids  # noqa
from tools.diff_replay_zips import load_replay  # noqa
from tools.oracle_zip_replay import (  # noqa
    apply_oracle_action_json,
    map_snapshot_player_ids_to_engine,
    parse_p_envelopes_from_zip,
    resolve_replay_first_mover,
)
from tools.desync_audit import _seed_for_game  # noqa

CATALOG = ROOT / "data" / "amarriner_gl_std_catalog.json"
MAP_POOL = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"
ZIPS_DIR = ROOT / "replays" / "amarriner_gl"


def _safe_get(d, key, default=None):
    if isinstance(d, dict):
        return d.get(key, default)
    return default


def _path_global(d) -> list:
    paths = _safe_get(d, "paths", {})
    if isinstance(paths, dict):
        return paths.get("global") or []
    return []


def _unit_global(d) -> dict:
    unit = _safe_get(d, "unit", {})
    if isinstance(unit, dict):
        gl = unit.get("global")
        if isinstance(gl, dict):
            return gl
    return {}


def _summarize_obj(obj: dict) -> str:
    """Compact one-line summary of an oracle action JSON object."""
    kind = _safe_get(obj, "action") or "?"
    bits = [str(kind)]
    if "Move" in obj:
        m = obj["Move"]
        path = _path_global(m)
        unit = _unit_global(m)
        ut = unit.get("units_name") or "?"
        if path:
            head = path[0] if isinstance(path[0], dict) else {}
            tail = path[-1] if isinstance(path[-1], dict) else {}
            bits.append(
                f"{ut} ({head.get('y')},{head.get('x')})->({tail.get('y')},{tail.get('x')}) {len(path)}tiles"
            )
    if "Fire" in obj:
        fire = obj["Fire"]
        cinfo = _safe_get(fire, "combatInfoVision", {})
        if isinstance(cinfo, dict):
            ci = cinfo.get("combatInfo") or {}
            atk = ci.get("attacker") or {}
            de = ci.get("defender") or {}
            bits.append(
                f"atk@({atk.get('units_y')},{atk.get('units_x')})={atk.get('units_name')} "
                f"def@({de.get('units_y')},{de.get('units_x')})={de.get('units_name')}"
            )
    if "Build" in obj:
        b = obj["Build"]
        u = _safe_get(b, "newUnit", {}) or {}
        bits.append(f"{u.get('units_name')}@({u.get('units_y')},{u.get('units_x')})")
    return " | ".join(bits)


def drill(gid: int) -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))["games"]
    meta = None
    for _k, g in catalog.items():
        if isinstance(g, dict) and int(g.get("games_id", -1)) == gid:
            meta = g
            break
    if meta is None:
        raise SystemExit(f"gid {gid} not in catalog")

    co_p0, co_p1 = pair_catalog_cos_ids(meta)
    map_id = int(meta["map_id"])
    tier = str(meta.get("tier", "T2"))
    print(f"gid={gid} map_id={map_id} tier={tier} co_p0={co_p0} co_p1={co_p1}")

    zip_path = ZIPS_DIR / f"{gid}.zip"
    frames = load_replay(zip_path)
    awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co_p0, co_p1)
    print(f"awbw_to_engine: {awbw_to_engine}")
    map_data = load_map(map_id, MAP_POOL, MAPS_DIR)
    envelopes = parse_p_envelopes_from_zip(zip_path)
    first_mover = resolve_replay_first_mover(envelopes, frames[0], awbw_to_engine)
    state = make_initial_state(
        map_data, co_p0, co_p1,
        starting_funds=0, tier_name=tier,
        replay_first_mover=first_mover,
    )

    random.seed(_seed_for_game(1, gid))

    # Step until first divergence; capture exception, surrounding context.
    fail_env = None
    fail_obj = None
    fail_exc = None
    history: list[tuple[int, int, dict]] = []
    for env_i, (_pid, day, actions) in enumerate(envelopes):
        for j, obj in enumerate(actions):
            if state.done:
                break
            history.append((env_i, j, obj))
            try:
                apply_oracle_action_json(state, obj, awbw_to_engine, envelope_awbw_player_id=_pid)
            except Exception as exc:
                fail_env = env_i
                fail_obj = obj
                fail_exc = exc
                break
        if fail_exc is not None:
            break

    if fail_exc is None:
        print("REPLAYED FULLY OK — no divergence")
        return

    print(f"\nFAIL at env={fail_env} day~{day}")
    print(f"  exc: {type(fail_exc).__name__}: {fail_exc}")
    print(f"  failing obj: {_summarize_obj(fail_obj)}")
    if "Fire" in fail_obj:
        fire = fail_obj["Fire"]
        if isinstance(fire, dict) and "Move" in fire:
            mv = fire["Move"]
            print(f"  nested Move path.global: {json.dumps(_path_global(mv), ensure_ascii=False)[:600]}")

    # Print last 8 history lines for context
    print("\nLast 8 actions before fail:")
    for env_i, j, obj in history[-8:]:
        print(f"  env={env_i:>3} act={j:>2}  {_summarize_obj(obj)}")

    # Find the attacker on the engine side.
    msg = str(fail_exc)
    import re
    m = re.search(r"from \((\d+), (\d+)\) \(unit_pos=\((\d+), (\d+)\)\)", msg)
    if m:
        from_pos = (int(m.group(1)), int(m.group(2)))
        unit_pos = (int(m.group(3)), int(m.group(4)))
        target_match = re.search(r"target \((\d+), (\d+)\)", msg)
        target_pos = (int(target_match.group(1)), int(target_match.group(2))) if target_match else None
        print(f"\nFire stance:")
        print(f"  AWBW from = {from_pos}")
        print(f"  engine unit_pos = {unit_pos}")
        print(f"  target = {target_pos}")
        attacker = state.get_unit_at(*unit_pos)
        if attacker is not None:
            print(f"  engine attacker: {attacker.unit_type.name} player={attacker.player} hp={attacker.hp} fuel={attacker.fuel} ammo={attacker.ammo} has_moved={getattr(attacker,'has_moved',None)}")
            costs = compute_reachable_costs(state, attacker)
            print(f"  reachable tiles ({len(costs)}): {sorted(costs)[:30]}{' ...' if len(costs)>30 else ''}")
            print(f"  AWBW from in engine reachable? {from_pos in costs}")
            atk_targets = get_attack_targets(state, attacker, from_pos)
            print(f"  get_attack_targets from AWBW from: {atk_targets}")
            atk_targets_unit = get_attack_targets(state, attacker, attacker.pos)
            print(f"  get_attack_targets from engine pos: {atk_targets_unit}")
            if target_pos is not None:
                tgt_unit = state.get_unit_at(*target_pos)
                if tgt_unit is None:
                    print(f"  engine target tile {target_pos} is EMPTY")
                else:
                    print(f"  engine target {target_pos}: {tgt_unit.unit_type.name} player={tgt_unit.player} hp={tgt_unit.hp}")
        else:
            print(f"  engine HAS NO UNIT at unit_pos={unit_pos}")
            # Find any B_COPTER in map for the attacker player
            for p in (0, 1):
                for u in state.units[p]:
                    if u.unit_type == UnitType.B_COPTER:
                        print(f"   p{p} B_COPTER @ {u.pos} hp={u.hp}")


if __name__ == "__main__":
    drill(int(sys.argv[1]))
