"""For each B_COPTER engine_bug gid, classify the failure into:
- ammo_zero       : engine.attacker.ammo == 0 (ammo bookkeeping)
- no_damage_table : ammo>0 but get_base_damage(B_COPTER, defender) is None
- defender_missing: target tile empty in engine view
- defender_friendly: target is a friendly unit in engine view
- defender_other  : there is an enemy at target with damage entry, ammo>0,
                    yet still rejected (true range / state issue)

Also runs for the smallest-drift gid in MECH/RECON/MEGA_TANK/BLACK_BOAT.
"""
from __future__ import annotations

import json
import random
import sys
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.action import compute_reachable_costs, get_attack_targets  # noqa
from engine.combat import get_base_damage  # noqa
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


def _classify_gid(gid: int) -> dict:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))["games"]
    meta = None
    for _k, g in catalog.items():
        if isinstance(g, dict) and int(g.get("games_id", -1)) == gid:
            meta = g
            break
    if meta is None:
        return {"gid": gid, "error": "not_in_catalog"}
    co_p0, co_p1 = pair_catalog_cos_ids(meta)
    map_id = int(meta["map_id"])
    tier = str(meta.get("tier", "T2"))
    zip_path = ZIPS_DIR / f"{gid}.zip"
    try:
        frames = load_replay(zip_path)
        awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co_p0, co_p1)
        map_data = load_map(map_id, MAP_POOL, MAPS_DIR)
        envelopes = parse_p_envelopes_from_zip(zip_path)
        first_mover = resolve_replay_first_mover(envelopes, frames[0], awbw_to_engine)
        state = make_initial_state(
            map_data, co_p0, co_p1, starting_funds=0,
            tier_name=tier, replay_first_mover=first_mover,
        )
    except Exception as exc:
        return {"gid": gid, "error": f"setup: {exc}"}

    random.seed(_seed_for_game(1, gid))
    fail_obj = None
    fail_exc = None
    for env_i, (_pid, day, actions) in enumerate(envelopes):
        for j, obj in enumerate(actions):
            if state.done:
                break
            try:
                apply_oracle_action_json(state, obj, awbw_to_engine, envelope_awbw_player_id=_pid)
            except Exception as exc:
                fail_obj = obj
                fail_exc = exc
                break
        if fail_exc is not None:
            break

    if fail_exc is None:
        return {"gid": gid, "outcome": "ok"}

    msg = str(fail_exc)
    m = re.search(r"_apply_attack: target \((\d+), (\d+)\) not in attack range for (\w+) from \((\d+), (\d+)\) \(unit_pos=\((\d+), (\d+)\)\)", msg)
    if not m:
        return {"gid": gid, "outcome": "other_exc", "msg": msg[:120]}
    target = (int(m.group(1)), int(m.group(2)))
    unit_type_name = m.group(3)
    move_pos = (int(m.group(4)), int(m.group(5)))
    unit_pos = (int(m.group(6)), int(m.group(7)))
    attacker = state.get_unit_at(*unit_pos)
    if attacker is None:
        return {"gid": gid, "outcome": "no_attacker", "unit_pos": unit_pos}
    defender = state.get_unit_at(*target)
    rec = {
        "gid": gid,
        "attacker_type": unit_type_name,
        "attacker_pos": unit_pos,
        "move_pos": move_pos,
        "target": target,
        "ammo": attacker.ammo,
        "fuel": attacker.fuel,
        "max_ammo": UNIT_STATS[attacker.unit_type].max_ammo,
        "reachable_includes_move_pos": move_pos in compute_reachable_costs(state, attacker),
        "defender_type": defender.unit_type.name if defender else None,
        "defender_player": defender.player if defender else None,
        "attacker_player": attacker.player,
    }
    if defender is not None:
        rec["base_damage"] = get_base_damage(attacker.unit_type, defender.unit_type)
        rec["is_friendly"] = defender.player == attacker.player

    if attacker.ammo == 0 and UNIT_STATS[attacker.unit_type].max_ammo > 0:
        rec["bucket"] = "ammo_zero"
    elif defender is None:
        rec["bucket"] = "defender_missing"
    elif defender.player == attacker.player:
        rec["bucket"] = "defender_friendly"
    elif rec.get("base_damage") is None:
        rec["bucket"] = "no_damage_table"
    else:
        rec["bucket"] = "other"
    return rec


def main():
    targets = []
    for line in open("logs/phase10a_b_copter_targets.jsonl", encoding="utf-8"):
        if line.strip():
            targets.append(json.loads(line))
    other_units_targets = []
    for line in open("logs/phase10a_other_unit_targets.jsonl", encoding="utf-8"):
        if line.strip():
            other_units_targets.append(json.loads(line))

    out_path = Path("logs/phase10a_b_copter_classified.jsonl")
    buckets = {}
    with open(out_path, "w", encoding="utf-8") as f:
        for r in targets:
            res = _classify_gid(int(r["games_id"]))
            f.write(json.dumps(res) + "\n")
            f.flush()
            b = res.get("bucket") or res.get("outcome") or "?"
            buckets[b] = buckets.get(b, 0) + 1
            print(
                f"gid={res['gid']:>8} bucket={b:<18} "
                f"def={res.get('defender_type')} ammo={res.get('ammo')} "
                f"base_dmg={res.get('base_damage')}"
            )
    print()
    print("B_COPTER bucket distribution:")
    for k, v in sorted(buckets.items()):
        print(f"  {k}: {v}")

    print()
    print("Other-unit smallest-drift cases:")
    other_buckets = {}
    out_other = Path("logs/phase10a_other_unit_classified.jsonl")
    seen_classes = set()
    with open(out_other, "w", encoding="utf-8") as f:
        for r in sorted(other_units_targets, key=lambda x: (x["_unit_type"], x["_drift"], int(x["games_id"]))):
            ut = r["_unit_type"]
            # take all rows for visibility but mark first per class
            res = _classify_gid(int(r["games_id"]))
            res["_unit_type"] = ut
            f.write(json.dumps(res) + "\n")
            f.flush()
            b = res.get("bucket") or res.get("outcome") or "?"
            other_buckets[(ut, b)] = other_buckets.get((ut, b), 0) + 1
            tag = ""
            if ut not in seen_classes:
                tag = "  <-- smallest drift"
                seen_classes.add(ut)
            print(
                f"gid={res['gid']:>8} ut={ut:<10} bucket={b:<18} "
                f"def={res.get('defender_type')} ammo={res.get('ammo')} "
                f"base_dmg={res.get('base_damage')}{tag}"
            )
    print()
    print("Other-unit (UnitType, bucket) counts:")
    for k, v in sorted(other_buckets.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
