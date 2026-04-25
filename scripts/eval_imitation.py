#!/usr/bin/env python3
"""
Post–behaviour-cloning checks:

1) **Rollout (default):** run N games as P0 with ``--model`` (MaskablePPO zip) vs
   ``--opponent`` random or checkpoints. Same flag name as the repo; ``--checkpoint``
   is an alias for ``--model``.

2) **Dataset (``--eval-demos``):** top-1 / top-5 / legal action rate on a
   human_demos / opening .jsonl (no new games).

Writes ``curriculum_tag=post_imitation_eval`` on game_log rows when the env
supports rollout mode. See also ``docs/play_ui.md``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _eval_topk(
    model_path: Path,
    demos: Path,
    *,
    topk: int,
    max_rows: int,
    include_move: bool,
    opening_only: bool,
) -> None:
    import json

    import numpy as np
    import torch
    from sb3_contrib import MaskablePPO  # type: ignore[import]

    rows: list[dict] = []
    with open(demos, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not include_move and row.get("action_stage") == "MOVE":
                continue
            if opening_only and not row.get("opening_segment"):
                continue
            rows.append(row)
            if 0 < max_rows <= len(rows):
                break
    if not rows:
        raise SystemExit("No eval rows after filters")

    model = MaskablePPO.load(str(model_path), device="cpu")
    pol = model.policy
    device = model.device
    n = 0
    top1 = 0
    topk_hits = 0
    legal = 0
    for row in rows:
        obs = {
            "spatial": np.asarray(row["spatial"], dtype=np.float32)[None, ...],
            "scalars": np.asarray(row["scalars"], dtype=np.float32)[None, ...],
        }
        mask = np.asarray(row["action_mask"], dtype=bool)
        y = int(row["action_idx"])
        if bool(mask[y]):
            legal += 1
        obs_t, _ = pol.obs_to_tensor(obs)
        if isinstance(obs_t, dict):
            obs_t = {k: v.to(device) for k, v in obs_t.items()}
        else:
            obs_t = obs_t.to(device)
        m_t = torch.as_tensor(mask[np.newaxis, ...], dtype=torch.bool, device=device)
        with torch.no_grad():
            dist = pol.get_distribution(obs_t, action_masks=m_t)
            logits = dist.distribution.logits[0]
        order = torch.argsort(logits, descending=True)
        ar = order.cpu().numpy()
        if int(ar[0]) == y:
            top1 += 1
        if y in ar[: min(topk, ar.shape[0])].tolist():
            topk_hits += 1
        n += 1
    if n:
        print(
            f"[eval_imitation] dataset={demos} rows={n} "
            f"top1={100.0 * top1 / n:.1f}% top{topk}={100.0 * topk_hits / n:.1f}% "
            f"legal_in_demo={100.0 * legal / n:.1f}% model={model_path}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=None, help="MaskablePPO zip (e.g. checkpoints/post_bc.zip)")
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Alias for --model (BC checkpoint zip).",
    )
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--opponent", choices=["random", "checkpoints"], default="random")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--map-id", type=int, default=None, help="Restrict pool to this map_id")
    ap.add_argument("--tier", type=str, default=None)
    ap.add_argument("--co-p0", type=int, default=None)
    ap.add_argument("--co-p1", type=int, default=None)
    ap.add_argument(
        "--eval-demos",
        type=Path,
        default=None,
        help="If set, run top-k accuracy on this .jsonl instead of rollout games (uses --model).",
    )
    ap.add_argument("--topk", type=int, default=5, help="K for top-K hit rate in --eval-demos mode")
    ap.add_argument(
        "--max-dataset-rows",
        type=int,
        default=0,
        help="Cap rows in --eval-demos (0 = all)",
    )
    ap.add_argument(
        "--include-move",
        action="store_true",
        help="In --eval-demos, keep MOVE-stage rows (usually omitted)",
    )
    ap.add_argument(
        "--opening-only",
        action="store_true",
        help="In --eval-demos, only rows with opening_segment true",
    )
    args = ap.parse_args()

    mp = args.model or args.checkpoint
    if args.eval_demos is not None:
        if mp is None:
            raise SystemExit("Need --model (or --checkpoint) with --eval-demos")
        _eval_topk(
            mp,
            args.eval_demos,
            topk=int(args.topk),
            max_rows=int(args.max_dataset_rows),
            include_move=bool(args.include_move),
            opening_only=bool(args.opening_only),
        )
        return

    if mp is None:
        raise SystemExit("Need --model (or --checkpoint) for rollout eval, or use --eval-demos")

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

    model = MaskablePPO.load(str(mp), device="cpu")
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
        f"opponent={args.opponent} model={mp}"
    )


if __name__ == "__main__":
    main()
