#!/usr/bin/env python3
"""
Transplant ``MaskablePPO`` ``latest.zip`` trunk weights into an :class:`rl.network.AWBWNet`
state dict (shared CNN + ``value_head.0`` from ``features_extractor.fc.0``).

Conv policy heads, ``linear_scalar_policy``, ``candidate_mlp``, and ``value_head.2``
stay at fresh init — fine for warm-starting candidate-action training or factored heads.

Examples:

  python scripts/scalpel_latest_to_awbw_net.py \\
    --input checkpoints/latest.zip \\
    --output checkpoints/latest_awbw_net_scalpel.pth

  python scripts/scalpel_latest_to_awbw_net.py \\
    --input Z:/checkpoints/latest.zip \\
    --output Z:/checkpoints/latest_awbw_net_scalpel.pth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO / "checkpoints" / "latest.zip",
        help="MaskablePPO zip (default: repo checkpoints/latest.zip).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "checkpoints" / "latest_awbw_net_scalpel.pth",
        help="Where to write torch.save dict with state_dict + scalpel_copied list.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=256,
        help="AWBWNet hidden size (must match training-time AWBWNet).",
    )
    args = parser.parse_args()

    import torch

    from rl.ckpt_compat import scalpel_checkpoint_zip_to_awbw_net_state

    inp = args.input.expanduser()
    if not inp.is_file():
        print(f"[scalpel] error: not found: {inp}", file=sys.stderr)
        return 1

    sd, copied = scalpel_checkpoint_zip_to_awbw_net_state(
        inp, hidden_size=int(args.hidden_size)
    )
    out = args.output.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": sd, "scalpel_copied": copied}, out)
    print(f"[scalpel] wrote {out} ({len(copied)} tensor copies)")
    for line in copied[:24]:
        print(f"  {line}")
    if len(copied) > 24:
        print(f"  ... +{len(copied) - 24} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
