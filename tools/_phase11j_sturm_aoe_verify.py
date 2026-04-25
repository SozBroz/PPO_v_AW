"""Phase 11J-FINAL Sturm AOE shape & damage verification.

For each Sturm Power envelope, replays the zip with engine up to (but not
including) the Power envelope, then dumps each affected unit's pre-strike HP
+ post-strike HP + Manhattan distance to missileCoords center(s).

Usage: python tools/_phase11j_sturm_aoe_verify.py <gid> [<gid> ...]
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.oracle_zip_replay import (
    parse_p_envelopes_from_zip,
    apply_oracle_action_json,
    UnsupportedOracleAction,
    load_replay,
    map_snapshot_player_ids_to_engine,
    load_map,
    resolve_replay_first_mover,
    make_initial_state,
)
from engine.game import GameState
from engine.unit import Unit
import engine.game as game_module


ROOT = Path(__file__).resolve().parents[1]
MAP_POOL = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"
CATALOG = ROOT / "data" / "amarriner_gl_extras_catalog.json"


def _load_catalog():
    out = {}
    for cat in [ROOT / "data" / "amarriner_gl_std_catalog.json", CATALOG]:
        if not cat.exists():
            continue
        with open(cat, "r", encoding="utf-8") as f:
            d = json.load(f)
        games = d.get("games") if isinstance(d, dict) else None
        if not isinstance(games, dict):
            games = d if isinstance(d, dict) else {}
        for k, v in games.items():
            try:
                out[int(k)] = v
            except (ValueError, TypeError):
                pass
    return out


def find_unit_by_id(state, unit_id: int):
    for p in (0, 1):
        for u in state.units[p]:
            if getattr(u, "unit_id", None) == unit_id:
                return p, u
    return None, None


def replay_up_to_envelope(zpath: Path, stop_env_idx: int, gid: int):
    catalog = _load_catalog()
    cinfo = catalog[gid]
    map_id = int(cinfo["map_id"])
    co0 = int(cinfo["co_p0_id"])
    co1 = int(cinfo["co_p1_id"])

    frames = load_replay(zpath)
    awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co0, co1)
    map_data = load_map(map_id, MAP_POOL, MAPS_DIR)
    envs = parse_p_envelopes_from_zip(zpath)
    first_mover = resolve_replay_first_mover(envs, frames[0], awbw_to_engine)
    state = make_initial_state(
        map_data, co0, co1, starting_funds=0,
        tier_name=cinfo.get("tier", "T1"),
        replay_first_mover=first_mover,
    )
    for ei, (pid, day, acts) in enumerate(envs):
        if ei >= stop_env_idx:
            break
        for obj in acts:
            if state.done:
                break
            try:
                apply_oracle_action_json(
                    state, obj, awbw_to_engine,
                    before_engine_step=None,
                    envelope_awbw_player_id=pid,
                )
            except UnsupportedOracleAction as e:
                print(f"   [WARN env {ei}] {e}")
                raise
        if state.done:
            break
    return state


def drill_one(gid: int):
    zpath = ROOT / "replays" / "amarriner_gl" / f"{gid}.zip"
    envs = parse_p_envelopes_from_zip(zpath)

    sturm_evt = []
    for i, (pid, day, acts) in enumerate(envs):
        for a in acts:
            if isinstance(a, dict) and a.get("action") == "Power" and a.get("coName") == "Sturm":
                sturm_evt.append((i, pid, day, a))

    print(f"\n=== gid {gid}: {len(sturm_evt)} Sturm power envelopes ===")
    for evt_idx, pid, day, action in sturm_evt:
        try:
            state = replay_up_to_envelope(zpath, evt_idx, gid)
        except Exception as e:
            print(f"   [evt {evt_idx}] replay error: {e}")
            continue
        ur = action.get("unitReplace", {}).get("global", {}).get("units", [])
        mc = action.get("missileCoords") or []
        centers = []
        for m in mc:
            try:
                centers.append((int(m["y"]), int(m["x"])))
            except Exception:
                pass

        kind = action.get("coPower")
        name = action.get("powerName")
        print(f"\n  --- evt {evt_idx} pid {pid} day {day} kind={kind} name={name!r} centers={centers} ---")
        # All units (both players), to compute "max diamond value" check too
        # First print affected units
        for entry in ur:
            uid = entry.get("units_id")
            post_hp = entry.get("units_hit_points")
            p_idx, u = find_unit_by_id(state, uid)
            if u is None:
                print(f"    AFF uid={uid} post_disp={post_hp} <NOT FOUND>")
                continue
            pre_disp = (u.hp + 9) // 10
            min_dist = min(abs(u.pos[0] - cy) + abs(u.pos[1] - cx) for (cy, cx) in centers) if centers else None
            print(f"    AFF uid={uid} P{p_idx} pos={u.pos} type={u.unit_type.name} pre_int={u.hp} pre_disp={pre_disp} post_disp={post_hp} loss_disp={pre_disp - int(post_hp)} dist_M={min_dist}")
        # Then print enemies WITHIN diamond <= 2 NOT in unitReplace (should be empty)
        # Sturm = current actor; opponent = whoever is not current
        opp = 1 - state.active_player
        affected_ids = {e["units_id"] for e in ur}
        # Print all enemy units within M<=4 of any center
        print(f"    [enemy units in M<=4 of any center, P{opp}]:")
        for u in state.units[opp]:
            uid = getattr(u, "unit_id", None)
            min_dist = min(abs(u.pos[0] - cy) + abs(u.pos[1] - cx) for (cy, cx) in centers) if centers else 999
            if min_dist <= 4:
                pre_disp = (u.hp + 9) // 10
                print(f"       eng_uid={uid} P{opp} pos={u.pos} type={u.unit_type.name} hp={u.hp} pre_disp={pre_disp} dist_M={min_dist}")


def main():
    gids = [int(x) for x in sys.argv[1:]] or [1635679, 1615143, 1637200]
    for gid in gids:
        try:
            drill_one(gid)
        except Exception as e:
            print(f"gid {gid}: outer error: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
