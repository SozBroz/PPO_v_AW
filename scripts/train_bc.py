#!/usr/bin/env python3
"""
Behaviour cloning on human_demos.jsonl (plan §7–§7.1).

Skips MOVE-stage rows by default (flat index does not encode destination).
Uses MaskablePPO.policy.evaluate_actions with stored masks — aggressive LR can
destroy generalisation; checkpoint before running.

**Actual CLI (repo):** ``--demos`` / ``--load`` / ``--save`` (not --input/--output),
``--lr``, ``--epochs``, ``--head-only``, ``--include-move``. Optional
``--opening-only`` keeps rows with ``opening_segment=true`` in each JSON line.
See also ``--eval-demos`` in ``scripts/eval_imitation.py`` for top-k on the
same file.

Example:
  python scripts/train_bc.py --demos logs/human_demos.jsonl \\
    --load checkpoints/latest.zip --save checkpoints/post_bc.zip --epochs 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rl.paths import HUMAN_DEMOS_PATH  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", type=Path, default=HUMAN_DEMOS_PATH)
    ap.add_argument("--load", type=Path, required=True, help="MaskablePPO zip to warm-start")
    ap.add_argument("--save", type=Path, required=True, help="Output zip path")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--head-only", action="store_true", help="Train policy head MLP only")
    ap.add_argument("--include-move", action="store_true", help="Keep MOVE rows (degenerate index)")
    ap.add_argument(
        "--opening-only",
        action="store_true",
        help="Keep only demo rows with opening_segment==true (from opening ingest)",
    )
    args = ap.parse_args()

    from sb3_contrib import MaskablePPO  # type: ignore[import]

    if not args.demos.is_file():
        raise SystemExit(f"No demos file: {args.demos}")

    rows: list[dict] = []
    with open(args.demos, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not args.include_move and row.get("action_stage") == "MOVE":
                continue
            if args.opening_only and not row.get("opening_segment"):
                continue
            rows.append(row)
    if not rows:
        raise SystemExit("No training rows after filters")

    print(
        "[train_bc] Rollback: copy your input zip before running (e.g. "
        "`copy checkpoints\\\\latest.zip checkpoints\\\\latest_pre_human.zip`). "
        "Aggressive LR / full-net BC can collapse win rate vs random."
    )

    model = MaskablePPO.load(str(args.load), device="cpu")
    pol = model.policy
    if args.head_only:
        for p in pol.features_extractor.parameters():
            p.requires_grad = False

    opt = torch.optim.Adam([p for p in pol.parameters() if p.requires_grad], lr=args.lr)

    device = model.device
    for epoch in range(args.epochs):
        total = 0.0
        n = 0
        for row in rows:
            obs = {
                "spatial": np.asarray(row["spatial"], dtype=np.float32)[None, ...],
                "scalars": np.asarray(row["scalars"], dtype=np.float32)[None, ...],
            }
            mask = np.asarray(row["action_mask"], dtype=bool)[None, ...]
            act = int(row["action_idx"])
            obs_t, _ = pol.obs_to_tensor(obs)
            if isinstance(obs_t, dict):
                obs_t = {k: v.to(device) for k, v in obs_t.items()}
            else:
                obs_t = obs_t.to(device)
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device)
            act_t = torch.tensor([act], dtype=torch.long, device=device)
            opt.zero_grad()
            _vals, log_prob, _ent = pol.evaluate_actions(obs_t, act_t, action_masks=mask_t)
            loss = -log_prob.mean()
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu())
            n += 1
        print(f"epoch {epoch + 1}/{args.epochs}  mean_nll={total / max(1, n):.4f}")

    model.save(str(args.save))
    print(f"saved {args.save}")


if __name__ == "__main__":
    main()
