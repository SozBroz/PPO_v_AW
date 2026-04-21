"""Diagnose GL 1609533 envelope 27 (AttackSeam no-path)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from engine.action import get_attack_targets, compute_reachable_costs  # noqa: E402
from engine.game import make_initial_state  # noqa: E402
from engine.map_loader import load_map  # noqa: E402
from tools.oracle_zip_replay import (  # noqa: E402
    apply_oracle_action_json,
    map_snapshot_player_ids_to_engine,
    parse_p_envelopes_from_zip,
    resolve_replay_first_mover,
    UnsupportedOracleAction,
)
from tools.diff_replay_zips import load_replay  # noqa: E402

GID = 1609533
TARGET_ENV = 27
ZP = ROOT / "replays" / "amarriner_gl" / f"{GID}.zip"

CATALOG = ROOT / "data" / "amarriner_gl_std_catalog.json"
MAP_POOL = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"


def _meta_for(gid: int) -> dict:
    cat = json.loads(CATALOG.read_text(encoding="utf-8"))
    for _k, g in (cat.get("games") or {}).items():
        if isinstance(g, dict) and int(g.get("games_id", -1)) == gid:
            return g
    raise KeyError(gid)


def main() -> int:
    meta = _meta_for(GID)
    map_id = int(meta["map_id"])
    co0 = int(meta["co_p0_id"])
    co1 = int(meta["co_p1_id"])
    tier = str(meta.get("tier") or "global_league")

    map_data = load_map(map_id, MAP_POOL, MAPS_DIR)
    frames = load_replay(ZP)
    awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co0, co1)
    envs = parse_p_envelopes_from_zip(ZP)
    first_mover = resolve_replay_first_mover(envs, frames[0], awbw_to_engine)
    state = make_initial_state(
        map_data, co0, co1, starting_funds=0, tier_name=tier,
        replay_first_mover=first_mover,
    )

    for env_i, (pid, day, actions) in enumerate(envs):
        if env_i >= TARGET_ENV:
            break
        for obj in actions:
            if state.done:
                break
            apply_oracle_action_json(state, obj, awbw_to_engine, envelope_awbw_player_id=pid)

    pid, day, actions = envs[TARGET_ENV]
    print(f"=== envelope {TARGET_ENV} day={day} pid={pid} active_player={state.active_player} ===")
    print(f"action_count={len(actions)}")
    for ai, obj in enumerate(actions):
        kind = obj.get("action")
        if kind == "AttackSeam":
            aseam = obj.get("AttackSeam") or {}
            print(f"\n[{ai}] AttackSeam seamY={aseam.get('seamY')} seamX={aseam.get('seamX')}")
            move = obj.get("Move")
            if isinstance(move, dict):
                paths = (move.get("paths") or {}).get("global") or []
            else:
                paths = []
            print(f"     paths.global: {paths}")
            uwrap = aseam.get("unit") or {}
            gu = uwrap.get("global") if isinstance(uwrap, dict) else {}
            ci = gu.get("combatInfo") if isinstance(gu, dict) else None
            print(f"     unit.global keys: {list(gu.keys()) if isinstance(gu, dict) else None}")
            print(f"     combatInfo: {ci}")
            # Apply earlier actions in same envelope
            print(f"     applying [0..{ai-1}] first...")
            for j in range(ai):
                try:
                    apply_oracle_action_json(state, actions[j], awbw_to_engine, envelope_awbw_player_id=pid)
                except UnsupportedOracleAction as e:
                    print(f"     pre-fail at [{j}] {actions[j].get('action')}: {e}")
                    return 1
            sr, sc = int(aseam["seamY"]), int(aseam["seamX"])
            if isinstance(ci, dict):
                ar, ac = int(ci.get("units_y", -1)), int(ci.get("units_x", -1))
                print(f"\n     combatInfo anchor ({ar},{ac})")
                u = state.get_unit_at(ar, ac)
                if u:
                    print(f"     engine unit: uid={u.unit_id} type={u.unit_type.name} player={u.player} hp={u.hp} ammo={u.ammo} moved={u.moved}")
            # Print any unit near (sr,sc)
            print(f"\n     units within Manhattan 4 of seam ({sr},{sc}):")
            for p in (0, 1):
                for u in state.units[p]:
                    r, c = u.pos
                    if abs(r-sr) + abs(c-sc) <= 4:
                        print(f"       P{p} uid={u.unit_id} type={u.unit_type.name} pos={u.pos} hp={u.hp} ammo={u.ammo} moved={u.moved}")
            print(f"\n     terrain near seam ({sr},{sc}):")
            for r in range(max(0, sr-3), min(state.map_data.height, sr+4)):
                row = []
                for c in range(max(0, sc-3), min(state.map_data.width, sc+4)):
                    row.append(f"{state.map_data.terrain[r][c]:3d}")
                print(f"       r={r}: {' '.join(row)}")
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
