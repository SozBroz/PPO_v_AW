"""Compare engine vs PHP property ownership at a specific envelope."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.game import GameState, make_initial_state
from engine.map_loader import load_map
from engine.terrain import get_terrain
from tools.amarriner_catalog_cos import pair_catalog_cos_ids
from tools.desync_audit import CANONICAL_SEED, _seed_for_game
from tools.diff_replay_zips import load_replay
from tools.oracle_zip_replay import (
    UnsupportedOracleAction,
    apply_oracle_action_json,
    map_snapshot_player_ids_to_engine,
    parse_p_envelopes_from_zip,
    resolve_replay_first_mover,
)

CATALOG = ROOT / "data" / "amarriner_gl_std_catalog.json"
CATALOG_EXTRAS = ROOT / "data" / "amarriner_gl_extras_catalog.json"
ZIPS = ROOT / "replays" / "amarriner_gl"
MAP_POOL = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gid", type=int, required=True)
    ap.add_argument("--env", type=int, required=True)
    ap.add_argument("--country-p0", type=int, default=14, help="PHP country palette ID for engine P0")
    ap.add_argument("--country-p1", type=int, default=17, help="PHP country palette ID for engine P1")
    args = ap.parse_args()

    by_id: dict[int, dict] = {}
    for cat_path in (CATALOG, CATALOG_EXTRAS):
        if cat_path.exists():
            cat = json.loads(cat_path.read_text(encoding="utf-8"))
            for g in (cat.get("games") or {}).values():
                if isinstance(g, dict) and "games_id" in g:
                    by_id[int(g["games_id"])] = g
    meta = by_id[args.gid]

    random.seed(_seed_for_game(CANONICAL_SEED, args.gid))
    co0, co1 = pair_catalog_cos_ids(meta)
    map_data = load_map(int(meta["map_id"]), MAP_POOL, MAPS_DIR)

    zpath = ZIPS / f"{args.gid}.zip"
    envs = parse_p_envelopes_from_zip(zpath)
    frames = load_replay(zpath)
    awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co0, co1)
    first_mover = resolve_replay_first_mover(envs, frames[0], awbw_to_engine)

    state = make_initial_state(
        map_data, co0, co1, starting_funds=0,
        tier_name=str(meta.get("tier") or "T2"),
        replay_first_mover=first_mover,
    )

    eng_owned = None
    for env_i, (pid, day, actions) in enumerate(envs):
        try:
            for obj in actions:
                apply_oracle_action_json(state, obj, awbw_to_engine, envelope_awbw_player_id=pid)
        except UnsupportedOracleAction as e:
            print(f"FAIL at env {env_i}: {e}")
            return 1
        if env_i == args.env:
            eng_owned = {(p.row, p.col): (p.owner, p.capture_points, p.is_comm_tower, p.is_lab) for p in state.properties}
            break

    if eng_owned is None:
        print("env not reached"); return 1

    php_frame = frames[args.env + 1]
    php_owned = {}
    for b in (php_frame.get("buildings") or {}).values():
        info = get_terrain(int(b["terrain_id"]))
        if info is None or not info.is_property:
            continue
        if info.country_id == args.country_p0:
            owner = 0
        elif info.country_id == args.country_p1:
            owner = 1
        elif info.country_id is None:
            owner = None
        else:
            owner = -1
        x, y = int(b["x"]), int(b["y"])
        php_owned[(y, x)] = (owner, info.is_comm_tower, info.is_lab, int(b.get("capture", 20)))

    diffs = []
    for pos, eng_v in eng_owned.items():
        eng_o = eng_v[0]
        ph = php_owned.get(pos)
        if ph is None:
            diffs.append((pos, "ENG_ONLY", eng_v, None)); continue
        php_o = ph[0]
        eng_cap = eng_v[1]
        php_cap = ph[3]
        if eng_o != php_o or eng_cap != php_cap:
            diffs.append((pos, "MISMATCH", eng_v, ph))
    for pos, ph in php_owned.items():
        if pos not in eng_owned:
            diffs.append((pos, "PHP_ONLY", None, ph))

    print(f"=== gid {args.gid} env {args.env} ===")
    print(f"  P0: eng={sum(1 for v in eng_owned.values() if v[0] == 0)} php={sum(1 for v in php_owned.values() if v[0] == 0)}")
    print(f"  P1: eng={sum(1 for v in eng_owned.values() if v[0] == 1)} php={sum(1 for v in php_owned.values() if v[0] == 1)}")
    print(f"  diffs: {len(diffs)}")
    for d in diffs:
        print(f"    {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
