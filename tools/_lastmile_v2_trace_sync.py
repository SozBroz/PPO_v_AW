"""Trace what my HP sync does at env 21 of 1617442."""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools.diff_replay_zips import load_replay
from tools.oracle_zip_replay import (
    parse_p_envelopes_from_zip,
    apply_oracle_action_json,
    resolve_replay_first_mover,
    map_snapshot_player_ids_to_engine,
    UnsupportedOracleAction,
)
from engine.map_loader import load_map
from engine.game import make_initial_state


def _setup_pin(state, env_i, actions, frames):
    if (env_i + 1) >= len(frames):
        state._oracle_post_envelope_units_by_id = None
        state._oracle_post_envelope_multi_hit_defenders = None
        return
    post_frame = frames[env_i + 1]
    pin = {}
    for u in (post_frame.get("units") or {}).values():
        try:
            uid = int(u["id"]); hp = float(u["hit_points"])
        except Exception:
            continue
        pin[uid] = max(0, min(100, int(round(hp * 10))))
    end_rep = set()
    for obj in actions:
        if isinstance(obj, dict) and obj.get("action") == "End":
            ui = obj.get("updatedInfo") or {}
            rep = ui.get("repaired") if isinstance(ui, dict) else None
            if isinstance(rep, dict):
                rep = rep.get("global")
            if isinstance(rep, list):
                for r in rep:
                    if isinstance(r, dict):
                        try:
                            end_rep.add(int(r.get("units_id")))
                        except Exception:
                            pass
    for uid in end_rep:
        pin.pop(uid, None)
    def_hits = {}
    for obj in actions:
        if not isinstance(obj, dict): continue
        if obj.get("action") not in ("Fire", "AttackSeam"): continue
        ci = obj.get("combatInfo")
        if not isinstance(ci, dict): continue
        d = ci.get("defender")
        if not isinstance(d, dict): continue
        try: d_uid = int(d.get("units_id"))
        except Exception: continue
        def_hits[d_uid] = def_hits.get(d_uid, 0) + 1
    multi = {uid for uid, c in def_hits.items() if c > 1}
    state._oracle_post_envelope_units_by_id = pin
    state._oracle_post_envelope_multi_hit_defenders = multi


def main() -> int:
    gid = 1617442
    co0, co1 = 30, 12
    map_id = None
    # Find map_id from catalog
    import json
    catalogs = ["data/amarriner_gl_std_catalog.json", "data/amarriner_gl_extras_catalog.json"]
    for cp in catalogs:
        cat = json.loads((REPO / cp).read_text(encoding="utf-8"))
        games = cat.get("games", {})
        if isinstance(games, dict):
            entry = games.get(str(gid)) or games.get(gid)
            if entry:
                map_id = entry.get("map_id")
                break
        else:
            for entry in games:
                if entry.get("games_id") == gid:
                    map_id = entry.get("map_id")
                    break
            if map_id is not None:
                break
    print(f"map_id={map_id}")

    zip_path = REPO / f"replays/amarriner_gl/{gid}.zip"
    map_pool = REPO / "data/gl_map_pool.json"
    maps_dir = REPO / "data/maps"

    frames = load_replay(zip_path)
    envs = parse_p_envelopes_from_zip(zip_path)
    awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co0, co1)
    first_mover = resolve_replay_first_mover(envs, frames[0], awbw_to_engine)
    map_data = load_map(map_id, map_pool, maps_dir)
    state = make_initial_state(map_data, co0, co1, starting_funds=0,
                               tier_name="T1", replay_first_mover=first_mover)

    target_env = 21
    target_seat, target_row, target_col = 1, 14, 12
    php_target_id = 191534926

    for i, (pid, day, actions) in enumerate(envs):
        if i > target_env:
            break
        _setup_pin(state, i, actions, frames)
        for ai, obj in enumerate(actions):
            if state.done:
                break
            kind = obj.get("action") if isinstance(obj, dict) else None
            f1_before = state.funds[1]
            tgt_hp_before = None
            for u in state.units[1]:
                if getattr(u, "is_alive", True) and u.pos == (14, 12) and int(getattr(u,"unit_id",-1)) == 11:
                    tgt_hp_before = u.hp
                    break
            try:
                apply_oracle_action_json(state, obj, awbw_to_engine, envelope_awbw_player_id=pid)
            except UnsupportedOracleAction as e:
                if i == target_env and kind == "Repair":
                    print(f"  [REPAIR-EXC] action[{ai}]: {e}")
                    continue
                print(f"env={i} ABORT {type(e).__name__}: {e}")
                return 1
            except Exception as e:
                if i == target_env and kind == "Repair":
                    print(f"  [REPAIR-EXC2] action[{ai}]: {type(e).__name__}: {e}")
                    continue
                raise
            if i == target_env:
                f1_after = state.funds[1]
                tgt_hp_after = None
                for u in state.units[1]:
                    if getattr(u, "is_alive", True) and u.pos == (14, 12) and int(getattr(u,"unit_id",-1)) == 11:
                        tgt_hp_after = u.hp
                        break
                if f1_before != f1_after or tgt_hp_before != tgt_hp_after:
                    print(f"  [DELTA] action[{ai}] {kind}: P1 funds {f1_before}->{f1_after} target_hp {tgt_hp_before}->{tgt_hp_after}")
                if kind == "Repair":
                    print(f"  [REPAIR-FULL-JSON] {json.dumps(obj)}")
                    # Dump engine state around (12,14) and Black Boat
                    print(f"  [REPAIR-DEBUG] active_player={state.active_player} action_stage={state.action_stage}")
                    for seat_loop in (0,1):
                        for u in state.units[seat_loop]:
                            if not getattr(u,"is_alive",True): continue
                            try: row,col=u.pos
                            except: continue
                            if (row,col) in {(14,12),(13,12),(14,13),(15,12),(14,11),(13,13)}:
                                print(f"    seat={seat_loop} unit_id={u.unit_id} pos=({row},{col}) type={u.unit_type} hp={u.hp} fuel={u.fuel} ammo={u.ammo}")
                    # legal actions
                    from engine.action import get_legal_actions
                    from engine.action import ActionType as AT
                    for act in get_legal_actions(state):
                        if act.action_type == AT.REPAIR:
                            print(f"    legal_rep: target={act.target_pos} unit={act.unit_pos} move={act.move_pos}")
        # Mirror the production audit: run HP sync at end of every envelope (BEFORE diff).
        if i < target_env and (i + 1) < len(frames):
            post_frame = frames[i + 1]
            php_units_iter = post_frame.get("units") or {}
            php_units_list = list(php_units_iter.values()) if isinstance(php_units_iter, dict) else php_units_iter
            php_by_awbw_id_loop = {}
            php_by_seat_pos_loop = {}
            for pu in php_units_list:
                try:
                    raw_id = int(pu["id"]); raw_hp = float(pu["hit_points"])
                    raw_x = int(pu["x"]); raw_y = int(pu["y"])
                    raw_pid = int(pu["players_id"])
                except Exception:
                    continue
                seat_loop = awbw_to_engine.get(raw_pid)
                if seat_loop is None: continue
                hp_int = max(0, min(100, int(round(raw_hp * 10))))
                php_by_awbw_id_loop[raw_id] = (seat_loop, raw_x, raw_y, hp_int)
                php_by_seat_pos_loop.setdefault((seat_loop, raw_x, raw_y), hp_int)
            for seat_loop in (0, 1):
                for u in state.units[seat_loop]:
                    if not getattr(u, "is_alive", True): continue
                    new_hp = None
                    try: uid_loop = int(u.unit_id)
                    except Exception: uid_loop = None
                    if uid_loop is not None and uid_loop in php_by_awbw_id_loop:
                        ps, px, py, php_hp_int = php_by_awbw_id_loop[uid_loop]
                        if ps == seat_loop:
                            new_hp = php_hp_int
                    if new_hp is None:
                        try:
                            row, col = u.pos
                            key = (seat_loop, int(col), int(row))
                        except Exception:
                            key = None
                        if key is not None and key in php_by_seat_pos_loop:
                            new_hp = php_by_seat_pos_loop[key]
                    if new_hp is not None and new_hp != int(u.hp):
                        u.hp = new_hp
        # Track unit 191534926 in env 21 actions — find any Fire involving it
        if i == target_env:
            print("\n  [ACTIONS-WITH-191534926]")
            for ai, a in enumerate(actions):
                s = json.dumps(a)
                if "191534926" in s:
                    print(f"    action[{ai}] kind={a.get('action')}: {s[:600]}")
        # Building ownership check at (13,15) and (12,14) for frames 20, 21, 22
        if i == target_env and (i + 1) < len(frames):
            for fi in (20, 21, 22):
                if fi < len(frames):
                    bd = frames[fi].get("buildings") or {}
                    bd_list = list(bd.values()) if isinstance(bd, dict) else bd
                    for b in bd_list:
                        if isinstance(b, dict):
                            bx = int(b.get("x", -1)); by = int(b.get("y", -1))
                            if (bx, by) in {(13, 15), (12, 14)}:
                                print(f"  [BLDG] frame {fi} ({bx},{by}): players_id={b.get('players_id')} terrain={b.get('terrain_id')}")
            # engine prop ownership at (13,15) and (12,14) — at end of env 21 actions
            print(f"  [ENG-PROPS] iter state.properties:")
            try:
                props = state.properties
            except Exception as e:
                props = None
                print(f"  no state.properties: {e}")
            if props is not None:
                if isinstance(props, dict):
                    for k, p in props.items():
                        try:
                            if (getattr(p,'col',-1), getattr(p,'row',-1)) in {(13,15),(12,14)} or k in {(15,13),(14,12)}:
                                print(f"    key={k} owner={getattr(p,'owner',None)} kind={getattr(p,'kind',None)}")
                        except Exception:
                            pass
                else:
                    for p in props:
                        try:
                            if (getattr(p,'col',-1), getattr(p,'row',-1)) in {(13,15),(12,14)}:
                                print(f"    owner={getattr(p,'owner',None)} pos=({p.col},{p.row})")
                        except Exception:
                            pass
        # Show all units at (12,14) in PHP frame i+1 (target+1)
        if i == target_env and (i + 1) < len(frames):
            pf = frames[i + 1]
            pf_units = list((pf.get("units") or {}).values()) if isinstance(pf.get("units"), dict) else (pf.get("units") or [])
            print(f"\n  [DUMP] all PHP units at (12,14) in frame {i+1}:")
            for pu in pf_units:
                if isinstance(pu, dict) and int(pu.get("x", -1)) == 12 and int(pu.get("y", -1)) == 14:
                    print(f"    id={pu.get('id')} players_id={pu.get('players_id')} name={pu.get('name')} hp={pu.get('hit_points')}")
            print(f"  [DUMP] all engine units at (12,14):")
            for seat_loop in (0, 1):
                for u in state.units[seat_loop]:
                    if not getattr(u, "is_alive", True): continue
                    try:
                        row, col = u.pos
                    except Exception:
                        continue
                    if int(col) == 12 and int(row) == 14:
                        print(f"    seat={seat_loop} unit_id={u.unit_id} hp={u.hp} type={getattr(u,'unit_type',None)}")
        # Also sync at target_env to mirror production AND check end-of-env state
        if i == target_env and (i + 1) < len(frames):
            print("\n  [*] running production-style sync at end of target_env...")
            post_frame = frames[i + 1]
            php_units_iter = post_frame.get("units") or {}
            php_units_list = list(php_units_iter.values()) if isinstance(php_units_iter, dict) else php_units_iter
            for seat_loop in (0, 1):
                print(f"  engine seat={seat_loop} unit count={sum(1 for x in state.units[seat_loop] if getattr(x,'is_alive',True))}")
                for u in state.units[seat_loop]:
                    if not getattr(u, "is_alive", True): continue
                    try:
                        row, col = u.pos
                    except Exception:
                        continue
                    if int(col) == 12 and int(row) == 14 and seat_loop == 1:
                        print(f"    engine unit at (12,14) seat=1: unit_id={u.unit_id} hp={u.hp} type={getattr(u,'unit_type',None)}")
        # Inspect engine vs PHP at the target tile RIGHT BEFORE sync runs.
        if i == target_env:
            # Show PHP frames at i and i+1 for this unit
            for fi in (i, i + 1):
                if fi < len(frames):
                    pf = frames[fi]
                    pf_units = list((pf.get("units") or {}).values()) if isinstance(pf.get("units"), dict) else (pf.get("units") or [])
                    for pu in pf_units:
                        if isinstance(pu, dict) and int(pu.get("id", -1)) == 191534926:
                            print(f"  PHP frame {fi}: id=191534926 hp={pu.get('hit_points')} day={pf.get('day')}")
                            break
            # Show engine seat=1 funds
            print(f"  engine funds[1]={state.funds[1]}")
            # Print env 21 actions
            print("  env 21 actions:")
            for a in actions[:20]:
                ak = a.get("action") if isinstance(a, dict) else None
                print(f"    {ak}: {json.dumps(a)[:300]}")
            # Track unit 191534926 across frames 18..22
            for fi in range(max(0, target_env - 4), min(len(frames), target_env + 3)):
                pf = frames[fi]
                pf_units = list((pf.get("units") or {}).values()) if isinstance(pf.get("units"), dict) else (pf.get("units") or [])
                for pu in pf_units:
                    if isinstance(pu, dict) and int(pu.get("id", -1)) == 191534926:
                        print(f"  PHP frame {fi}: unit at x={pu.get('x')} y={pu.get('y')} hp={pu.get('hit_points')} day={pf.get('day')}")
                        break
            # Property ownership at (12,14) in engine and in PHP frames 21, 22
            try:
                tile_eng = state.board[14][12]
                print(f"  engine board[14][12] tile attrs: kind={getattr(tile_eng,'kind',None)} owner={getattr(tile_eng,'owner',None)} hp={getattr(tile_eng,'capture_hp',None)}")
            except Exception as e:
                print(f"  engine tile err: {e}")
            for fi in (20, 21, 22):
                if fi < len(frames):
                    bd = frames[fi].get("buildings") or {}
                    bd_list = list(bd.values()) if isinstance(bd, dict) else bd
                    for b in bd_list:
                        if isinstance(b, dict) and int(b.get("x", -1)) == 12 and int(b.get("y", -1)) == 14:
                            print(f"  PHP frame {fi} bldg(12,14): players_id={b.get('players_id')} terrain={b.get('terrain_id')} hp={b.get('capture')}")
                            break
            print(f"\n=== After env {i} actions, BEFORE sync ===")
            for u in state.units[target_seat]:
                if u.pos == (target_row, target_col):
                    print(f"  engine unit at (row={target_row}, col={target_col}) seat={target_seat}: unit_id={u.unit_id} hp={u.hp}")
            post_frame = frames[i + 1]
            for pu in (post_frame.get("units") or {}).values():
                try:
                    if int(pu.get("id")) == php_target_id:
                        print(f"  PHP unit id={php_target_id}: x={pu.get('x')} y={pu.get('y')} hp={pu.get('hit_points')}")
                except Exception:
                    pass
            # Manually run my sync code:
            php_units_iter = post_frame.get("units") or {}
            php_units_list = list(php_units_iter.values()) if isinstance(php_units_iter, dict) else php_units_iter
            php_by_awbw_id = {}
            php_by_seat_pos = {}
            for pu in php_units_list:
                try:
                    raw_id = int(pu["id"]); raw_hp = float(pu["hit_points"])
                    raw_x = int(pu["x"]); raw_y = int(pu["y"])
                    raw_pid = int(pu["players_id"])
                except Exception:
                    continue
                seat = awbw_to_engine.get(raw_pid)
                if seat is None: continue
                hp_int = max(0, min(100, int(round(raw_hp * 10))))
                php_by_awbw_id[raw_id] = (seat, raw_x, raw_y, hp_int)
                php_by_seat_pos.setdefault((seat, raw_x, raw_y), hp_int)
            # Lookup target
            print(f"\n  php_by_awbw_id has {php_target_id}? {php_target_id in php_by_awbw_id}")
            print(f"  php_by_seat_pos has (seat={target_seat}, x={target_col}, y={target_row})? {(target_seat, target_col, target_row) in php_by_seat_pos}")
            for seat in (0, 1):
                for u in state.units[seat]:
                    if not getattr(u, "is_alive", True): continue
                    if u.pos != (target_row, target_col): continue
                    if seat != target_seat: continue
                    new_hp = None
                    try: uid = int(u.unit_id)
                    except Exception: uid = None
                    if uid is not None and uid in php_by_awbw_id:
                        ps, px, py, php_hp_int = php_by_awbw_id[uid]
                        if ps == seat:
                            new_hp = php_hp_int
                            print(f"  AWBW ID match: uid={uid} new_hp={new_hp}")
                    if new_hp is None:
                        try:
                            row, col = u.pos
                            key = (seat, int(col), int(row))
                        except Exception:
                            key = None
                        if key is not None and key in php_by_seat_pos:
                            new_hp = php_by_seat_pos[key]
                            print(f"  position match: key={key} new_hp={new_hp}")
                    print(f"  engine hp pre={u.hp}, post-sync new_hp={new_hp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
