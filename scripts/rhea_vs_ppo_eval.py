#!/usr/bin/env python3
"""
Head-to-head eval: RHEA (value_rhea_latest.pt) vs PPO (latest.zip).

Plays 7 games on Designed Desires, Jess mirrors, 30 turns max.
Each side plays both P0 and P1 for fair comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.env import AWBWEnv, POOL_PATH
from rl.rhea import RheaConfig, RheaPlanner
from rl.rhea_fitness import RheaFitness
from rl.value_net import load_value_checkpoint

import numpy as np
import torch


def load_ppo_value_as_heuristic(ckpt_path: str, device: str = "cpu"):
    """
    Extract value function from PPO checkpoint and use it as a heuristic agent.
    Returns a function that takes (obs, mask) and returns an action.
    """
    from rl.ckpt_compat import materialize_sb3_zip_with_spatial_compat
    import io
    import zipfile
    
    # Patch if needed
    patched_path, was_patched = materialize_sb3_zip_with_spatial_compat(ckpt_path)
    if was_patched:
        print(f"[eval] Using patched checkpoint: {patched_path}")
        ckpt_to_load = patched_path
    else:
        ckpt_to_load = ckpt_path
    
    # Load the zip file
    with zipfile.ZipFile(ckpt_to_load, "r") as zf:
        # Read policy.pth
        policy_buf = io.BytesIO(zf.read("policy.pth"))
        policy_sd = torch.load(policy_buf, map_location="cpu", weights_only=False)
    
    # Create a minimal env and model to extract the value function
    from rl.env import AWBWEnv
    with open(POOL_PATH) as f:
        pool = json.load(f)
    env = AWBWEnv(map_pool=pool[:1], co_p0=14, co_p1=14)
    from sb3_contrib import MaskablePPO
    model = MaskablePPO("MultiInputPolicy", env, device=device)
    
    # Load policy state dict with strict=False
    model.policy.load_state_dict(policy_sd, strict=False)
    
    # Return a function that uses the policy for action selection
    def select_action(obs, mask):
        action, _ = model.predict(obs, action_masks=mask, deterministic=False)
        if isinstance(action, int | np.integer):
            return int(action)
        return int(np.asarray(action, dtype=np.int64).reshape(-1)[0])
    
    return select_action, model


def run_one_game(
    *,
    rhea_seat: int,
    value_ckpt: str,
    ppo_ckpt: str,
    map_id: int,
    co_id: int,
    max_turns: int,
    device: str = "cuda",
    game_seed: int | None = None,
) -> dict:
    """Run one game. Returns dict with winner and metadata."""
    rng = random.Random(game_seed)
    np_rng = np.random.default_rng(game_seed)
    
    # Load map pool filtered to map_id
    with open(POOL_PATH) as f:
        full_pool = json.load(f)
    filtered_pool = [m for m in full_pool if m.get("map_id") == map_id]
    if not filtered_pool:
        raise ValueError(f"Map ID {map_id} not found in pool")
    
    env = AWBWEnv(
        map_pool=filtered_pool,
        co_p0=co_id,
        co_p1=co_id,
        max_turns=max_turns,
    )
    env.reset(seed=game_seed)
    
    # Load models
    value_model = load_value_checkpoint(value_ckpt, device=device)
    
    # RHEA setup - disable tactical beam to avoid Cython issues
    rhea_config = RheaConfig(
        population=32,
        generations=6,
        reward_weight=0.90,
        value_weight=0.10,
        seed=int(np_rng.integers(0, 1 << 30)),
        use_tactical_beam=False,  # Disable to avoid Cython issues
    )
    fitness = RheaFitness(
        env_template=env,
        value_model=value_model,
        device=device,
        reward_weight=0.90,
        value_weight=0.10,
    )
    rhea_planner = RheaPlanner(fitness, rhea_config)
    
    # PPO model - load on demand
    ppo_select_action = None
    ppo_model = None
    
    def get_ppo_action(obs, mask):
        nonlocal ppo_select_action, ppo_model
        if ppo_select_action is None:
            ppo_select_action, ppo_model = load_ppo_value_as_heuristic(ppo_ckpt, device="cpu")
        return ppo_select_action(obs, mask)
    
    turn_count = 0
    
    while env.state is not None and env.state.winner is None:
        state = env.state
        active = int(state.active_player)
        
        if active == rhea_seat:
            # RHEA plays full turn
            result = rhea_planner.choose_full_turn(state)
            actions = result.actions
            start_active = active
            for action in actions:
                if env.state is None or env.state.winner is not None:
                    break
                if int(env.state.active_player) != start_active:
                    break
                env.state.step(action)
        else:
            # PPO plays one action at a time
            start_active = active
            while env.state is not None and int(env.state.active_player) == start_active:
                if env.state.winner is not None:
                    break
                obs = env._get_obs()
                mask = env.action_masks()
                flat_act = get_ppo_action(obs, mask)
                # Convert flat to Action object
                from rl.env import _flat_to_action
                act_obj = _flat_to_action(flat_act, env.state, legal=env._get_legal())
                if act_obj is None:
                    break
                env.state.step(act_obj)
        
        turn_count += 1
    
    winner = env.state.winner if env.state is not None else None
    rhea_won = winner == rhea_seat
    
    return {
        "winner": winner,
        "rhea_won": rhea_won,
        "turns_played": turn_count,
        "rhea_seat": rhea_seat,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--value-ckpt",
        type=str,
        default=str(ROOT / "checkpoints" / "value_rhea_latest.pt"),
        help="RHEA value network checkpoint",
    )
    parser.add_argument(
        "--ppo-ckpt",
        type=str,
        default=r"\\Ai_machine\awbw\checkpoints\latest.zip",
        help="PPO policy checkpoint",
    )
    parser.add_argument("--map-id", type=int, default=171596, help="Map ID (Designed Desires)")
    parser.add_argument("--co-id", type=int, default=14, help="CO ID (Jess=14)")
    parser.add_argument("--max-turns", type=int, default=30, help="Max turns")
    parser.add_argument("--games", type=int, default=7, help="Total games")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    
    rng = random.Random(args.seed)
    results = []
    
    # Decide seating: first half rhea is P0, second half rhea is P1
    rhea_as_p0_games = (args.games + 1) // 2
    rhea_as_p1_games = args.games - rhea_as_p0_games
    
    print(f"[eval] RHEA vs PPO on map {args.map_id}, CO {args.co_id}")
    print(f"[eval] RHEA as P0: {rhea_as_p0_games} games, RHEA as P1: {rhea_as_p1_games} games")
    print(f"[eval] Max turns: {args.max_turns}")
    print(f"[eval] Value checkpoint: {args.value_ckpt}")
    print(f"[eval] PPO checkpoint: {args.ppo_ckpt}")
    
    game_num = 0
    
    # RHEA as P0
    for i in range(rhea_as_p0_games):
        game_num += 1
        seed = rng.randint(0, 1 << 30)
        print(f"\n[eval] Game {game_num}: RHEA=P0, PPO=P1, seed={seed}")
        r = run_one_game(
            rhea_seat=0,
            value_ckpt=args.value_ckpt,
            ppo_ckpt=args.ppo_ckpt,
            map_id=args.map_id,
            co_id=args.co_id,
            max_turns=args.max_turns,
            device=args.device,
            game_seed=seed,
        )
        r["game"] = game_num
        results.append(r)
        print(f"[eval] Result: winner={r['winner']}, RHEA won={r['rhea_won']}")
    
    # RHEA as P1
    for i in range(rhea_as_p1_games):
        game_num += 1
        seed = rng.randint(0, 1 << 30)
        print(f"\n[eval] Game {game_num}: PPO=P0, RHEA=P1, seed={seed}")
        r = run_one_game(
            rhea_seat=1,
            value_ckpt=args.value_ckpt,
            ppo_ckpt=args.ppo_ckpt,
            map_id=args.map_id,
            co_id=args.co_id,
            max_turns=args.max_turns,
            device=args.device,
            game_seed=seed,
        )
        r["game"] = game_num
        results.append(r)
        print(f"[eval] Result: winner={r['winner']}, RHEA won={r['rhea_won']}")
    
    # Summary
    rhea_wins = sum(1 for r in results if r["rhea_won"])
    ppo_wins = sum(1 for r in results if not r["rhea_won"] and r["winner"] is not None)
    draws = sum(1 for r in results if r["winner"] is None)
    
    print(f"\n{'=' * 60}")
    print(f"[eval] SUMMARY: RHEA vs PPO ({args.games} games)")
    print(f"[eval] RHEA wins: {rhea_wins}")
    print(f"[eval] PPO wins:  {ppo_wins}")
    print(f"[eval] Draws:      {draws}")
    if args.games > 0:
        print(f"[eval] RHEA win rate: {rhea_wins / args.games:.3f}")
    print(f"{'=' * 60}")
    
    if args.json_out:
        summary = {
            "value_ckpt": args.value_ckpt,
            "ppo_ckpt": args.ppo_ckpt,
            "map_id": args.map_id,
            "co_id": args.co_id,
            "max_turns": args.max_turns,
            "total_games": args.games,
            "rhea_wins": rhea_wins,
            "ppo_wins": ppo_wins,
            "draws": draws,
            "rhea_win_rate": rhea_wins / args.games if args.games > 0 else 0,
            "games": results,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[eval] Wrote summary to {args.json_out}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())