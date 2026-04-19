#!/usr/bin/env python3
"""
Post–behaviour-cloning sanity check: run N episodes as trained P0 vs opponent.

Writes ``curriculum_tag=post_imitation_eval`` on game_log rows when the env
supports it. For full MASTERPLAN metrics, also run a short ``train.py`` eval
slice with an explicit run tag — see ``docs/play_ui.md``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True, help="MaskablePPO zip (e.g. checkpoints/post_bc.zip)")
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--opponent", choices=["random", "checkpoints"], default="random")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--map-id", type=int, default=None, help="Restrict pool to this map_id")
    ap.add_argument("--tier", type=str, default=None)
    ap.add_argument("--co-p0", type=int, default=None)
    ap.add_argument("--co-p1", type=int, default=None)
    args = ap.parse_args()

    from sb3_contrib import MaskablePPO  # type: ignore[import]
    from sb3_contrib.common.wrappers import ActionMasker  # type: ignore[import]

    from rl.env import AWBWEnv
    from rl.self_play import CHECKPOINT_DIR, POOL_PATH, _CheckpointOpponent

    def mask_fn(env):
        return env.action_masks()

    pool_path = Path(str(POOL_PATH))
    with pool_path.open(encoding="utf-8") as f:
        map_pool: list = json.load(f)
    if args.map_id is not None:
        map_pool = [m for m in map_pool if m.get("map_id") == args.map_id]
        if not map_pool:
            raise SystemExit(f"No map with map_id={args.map_id} in {pool_path}")

    opponent = None
    if args.opponent == "checkpoints":
        opponent = _CheckpointOpponent(str(CHECKPOINT_DIR))

    env = ActionMasker(
        AWBWEnv(
            map_pool=map_pool,
            opponent_policy=opponent,
            co_p0=args.co_p0,
            co_p1=args.co_p1,
            tier_name=args.tier,
            curriculum_tag="post_imitation_eval",
        ),
        mask_fn,
    )

    model = MaskablePPO.load(str(args.model), device="cpu")
    rng = np.random.default_rng(args.seed)

    wins = losses = draws = 0
    for g in range(args.games):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        done = False
        while not done:
            mask = env.action_masks()
            act, _ = model.predict(
                obs,
                action_masks=mask,
                deterministic=bool(args.deterministic),
            )
            obs, _r, term, trunc, _info = env.step(int(act))
            done = bool(term or trunc)
        w = env.unwrapped.state.winner
        if w == 0:
            wins += 1
        elif w == 1:
            losses += 1
        else:
            draws += 1

    print(
        f"[eval_imitation] games={args.games} W/D/L={wins}/{draws}/{losses} "
        f"opponent={args.opponent} model={args.model}"
    )


if __name__ == "__main__":
    main()
