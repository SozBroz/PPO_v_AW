#!/usr/bin/env python3
"""
Transplant ``latest.zip`` (flat 35k MaskablePPO) into a new zip with candidate actions
``Discrete(MAX_CANDIDATES)`` and :class:`~rl.network.AWBWCandidateFeaturesExtractor`.

CNN trunk + compatible fusion/MLP/value weights are copied; ``action_net`` is fresh.

Examples:

  python scripts/scalpel_latest_to_candidate_ppo.py \\
    --input checkpoints/latest.zip \\
    --output checkpoints/latest_candidate.zip

  python scripts/scalpel_latest_to_candidate_ppo.py \\
    --input Z:/checkpoints/latest.zip \\
    --output Z:/checkpoints/latest_candidate.zip \\
    --features-dim 512
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
        help="Legacy MaskablePPO zip (default: checkpoints/latest.zip).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "checkpoints" / "latest_candidate.zip",
        help="Written MaskablePPO zip (extension added if omitted by SB3 save).",
    )
    parser.add_argument(
        "--features-dim",
        type=int,
        default=512,
        help="AWBWCandidateFeaturesExtractor features_dim (default 512, matches ppo.make_maskable_ppo).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cpu or cuda for the temporary template model (default: auto).",
    )
    args = parser.parse_args()

    from rl.ckpt_compat import scalpel_checkpoint_zip_to_candidate_maskable_ppo_zip

    inp = args.input.expanduser()
    if not inp.is_file():
        print(f"[scalpel-candidate] error: not found: {inp}", file=sys.stderr)
        return 1

    copied = scalpel_checkpoint_zip_to_candidate_maskable_ppo_zip(
        inp,
        args.output,
        features_dim=int(args.features_dim),
        device=args.device,
    )
    out = args.output
    if out.suffix.lower() != ".zip":
        out = out.with_suffix(".zip")
    print(f"[scalpel-candidate] wrote {out} ({len(copied)} operations)")
    for line in copied[:32]:
        print(f"  {line}")
    if len(copied) > 32:
        print(f"  ... +{len(copied) - 32} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
