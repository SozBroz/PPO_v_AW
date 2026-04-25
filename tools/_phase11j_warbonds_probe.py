"""Probe War Bonds payouts inside game 1624082 env 33 to diagnose 150g residual."""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.game import GameState, make_initial_state
from engine.map_loader import load_map
from tools.amarriner_catalog_cos import pair_catalog_cos_ids
from tools.desync_audit import CANONICAL_SEED, _seed_for_game
from tools.diff_replay_zips import load_replay
from tools.oracle_zip_replay import (
    UnsupportedOracleAction, apply_oracle_action_json,
    map_snapshot_player_ids_to_engine, parse_p_envelopes_from_zip,
    resolve_replay_first_mover,
)

CATALOG = ROOT / "data" / "amarriner_gl_std_catalog.json"
ZIPS = ROOT / "replays" / "amarriner_gl"
MAP_POOL = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gid", type=int, default=1624082)
    ap.add_argument("--target-env", type=int, default=33)
    args = ap.parse_args()

    cat = json.loads(CATALOG.read_text(encoding="utf-8"))
    games = cat.get("games") or {}
    by_id = {int(g["games_id"]): g for g in games.values()
             if isinstance(g, dict) and "games_id" in g}
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

    GameState_cls = type(state)
    orig = GameState_cls._apply_war_bonds_payout
    payouts = []

    def logged(self, dealer, target, pre_disp):
        funds_before = self.funds[dealer.player]
        # Capture target HP before/after for diagnostics
        target_pre_hp = None
        # The damage has already been applied at this point; reconstruct
        # via the display drop and post-state isn't enough. We instead
        # snapshot pre/post here:
        target_post_hp = target.hp
        orig(self, dealer, target, pre_disp)
        delta = self.funds[dealer.player] - funds_before
        if delta != 0 or (
            self.co_states[dealer.player].co_id == 19
            and self.co_states[dealer.player].war_bonds_active
        ):
            payouts.append((dealer.unit_type.name, target.unit_type.name,
                            pre_disp, target.display_hp, target_post_hp, delta))

    GameState_cls._apply_war_bonds_payout = logged

    # Also patch _apply_attack to log the dmg/counter values
    orig_apply_attack = GameState_cls._apply_attack
    attack_log = []
    def logged_attack(self, action):
        # Save defender pre-hp for log
        if action.target_pos is not None:
            d_pre = self.get_unit_at(*action.target_pos)
            if d_pre is not None:
                attack_log.append(('pre_def_hp', d_pre.hp, d_pre.unit_type.name))
        return orig_apply_attack(self, action)
    GameState_cls._apply_attack = logged_attack

    php_funds_history = []
    for env_i, (pid, day, actions) in enumerate(envs):
        in_target = (env_i == args.target_env)
        if in_target:
            print(f"--- ENV {env_i} (pid={pid} day={day}) actions={len(actions)} ---")
            print(f"  start funds: engine={list(state.funds)} co_state[1].war_bonds={state.co_states[1].war_bonds_active}")
        try:
            for ai, obj in enumerate(actions):
                kind = obj.get("action") or obj.get("type") or "?"
                if in_target:
                    fb = list(state.funds)
                    payouts.clear()
                apply_oracle_action_json(state, obj, awbw_to_engine,
                                         envelope_awbw_player_id=pid)
                if in_target:
                    fa = list(state.funds)
                    wb = ""
                    for p in payouts:
                        wb += f" WB({p[0]}->{p[1]} disp{p[2]}->{p[3]} post_hp={p[4]} +{p[5]})"
                    detail = ""
                    if kind == "Fire":
                        fire_blk = obj.get("Fire") or {}
                        civ = (fire_blk.get("combatInfoVision") or {}).get("global") or {}
                        ci = civ.get("combatInfo") or {}
                        gf = ci.get("gainedFunds") or {}
                        atk_post_disp = (ci.get("attacker") or {}).get("units_hit_points")
                        dfn_post_disp = (ci.get("defender") or {}).get("units_hit_points")
                        atk_pre_e = next((x for x in attack_log if x[0]=='pre_def_hp'), None)
                        attack_log.clear()
                        detail = f" PHP_post(atk={atk_post_disp},dfn={dfn_post_disp}) gainedFunds={gf} engine_def_pre={atk_pre_e}"
                    print(f"  [act {ai:3}] {kind:6} {fb} -> {fa}{wb}{detail}")
        except UnsupportedOracleAction as e:
            print(f"oracle_gap@{env_i}: {e}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
