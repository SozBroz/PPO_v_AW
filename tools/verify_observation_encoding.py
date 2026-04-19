"""
Human-auditable bridge from GameState → encode_state() tensors.

The policy never sees render_ascii() glyphs; it sees (H,W,59) float planes + 13 scalars
documented in rl/encoder.py. This script:

  - Prints scalars with fixed labels (must stay in sync with encode_state).
  - Decodes spatial planes into compact grids: terrain argmax, unit-channel argmax, HP.
  - Prints the engine render_ascii() for the same state so you can compare.

Usage:
  python tools/verify_observation_encoding.py --map-id 123858 --tier T3 --co-p0 1 --co-p1 1
  python tools/verify_observation_encoding.py --map-id 123858 --random-steps 50
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from engine.action import get_legal_actions
from engine.game import make_initial_state
from engine.map_loader import load_map
from rl.encoder import (
    GRID_SIZE,
    N_UNIT_CHANNELS,
    TERRAIN_CATEGORIES,
    encode_state,
)

# Order must match rl.encoder.encode_state scalars array exactly.
SCALAR_LABELS: list[str] = [
    "funds_p0/50k",
    "funds_p1/50k",
    "power_bar_p0",
    "power_bar_p1",
    "cop_p0",
    "scop_p0",
    "cop_p1",
    "scop_p1",
    "turn/MAX_TURNS",
    "active_player",
    "p0_co_id/30",
    "p1_co_id/30",
    "tier_norm",
]

# One char per terrain category index 0..14 (same order as TERRAIN_CATEGORIES values).
_TERRAIN_ORDER = sorted(TERRAIN_CATEGORIES.items(), key=lambda kv: kv[1])
TERRAIN_INDEX_CHAR: list[str] = [
    "p",
    "M",
    "T",
    "+",
    "~",
    "=",
    "S",
    "s",
    "R",
    "C",
    "B",
    "A",
    "P",
    "H",
    "L",
]

# 14 unit-type buckets (many UnitTypes clamp into bucket 13 — same as encoder).
_UNIT_BUCKET_CHAR = "0123456789ABCD"


def _spatial_terrain_grid(spatial: np.ndarray, h: int, w: int) -> list[str]:
    off = N_UNIT_CHANNELS + 1
    lines = []
    for r in range(h):
        row = []
        for c in range(w):
            sl = spatial[r, c, off : off + len(TERRAIN_INDEX_CHAR)]
            idx = int(np.argmax(sl))
            ch = TERRAIN_INDEX_CHAR[idx] if float(np.max(sl)) > 0.5 else "?"
            row.append(ch)
        lines.append("".join(row))
    return lines


def _spatial_unit_grid(spatial: np.ndarray, h: int, w: int) -> list[str]:
    lines = []
    for r in range(h):
        row = []
        for c in range(w):
            sl = spatial[r, c, :N_UNIT_CHANNELS]
            mx = float(np.max(sl))
            if mx < 0.5:
                row.append(".")
            else:
                ch_idx = int(np.argmax(sl))
                bucket = ch_idx % 14
                player = ch_idx // 14
                u = _UNIT_BUCKET_CHAR[bucket]
                row.append(u.upper() if player == 0 else u.lower())
        lines.append("".join(row))
    return lines


def _print_scalar_table(scalars: np.ndarray) -> None:
    print("\n--- Scalars (decoded labels; compare with rl/encoder.py) ---")
    for i, (name, val) in enumerate(zip(SCALAR_LABELS, scalars, strict=True)):
        print(f"  [{i:2d}] {name:16s} = {float(val):.6f}")


def _run_random_steps(state, n: int) -> None:
    for _ in range(n):
        legal = get_legal_actions(state)
        if not legal:
            break
        act = random.choice(legal)
        state, _, _ = state.step(act)
        if state.done:
            break


def main() -> None:
    p = argparse.ArgumentParser(
        description="Print decoded observation tensors next to render_ascii for auditing."
    )
    p.add_argument("--map-id", type=int, required=True)
    p.add_argument("--tier", type=str, default="T2")
    p.add_argument("--co-p0", type=int, default=1)
    p.add_argument("--co-p1", type=int, default=1)
    p.add_argument(
        "--random-steps",
        type=int,
        default=0,
        help="Apply this many random legal actions (P0 steps only) before encoding.",
    )
    args = p.parse_args()

    pool_path = ROOT / "data" / "gl_map_pool.json"
    maps_dir = ROOT / "data" / "maps"
    map_data = load_map(args.map_id, pool_path, maps_dir)
    state = make_initial_state(
        map_data,
        args.co_p0,
        args.co_p1,
        starting_funds=0,
        tier_name=args.tier,
    )
    if args.random_steps > 0:
        _run_random_steps(state, args.random_steps)

    spatial, scalars = encode_state(state)
    h, w = min(state.map_data.height, GRID_SIZE), min(state.map_data.width, GRID_SIZE)

    print("--- Engine render_ascii() (compact glyphs; NOT model input) ---")
    print(state.render_ascii())
    print("\n--- Legend: render uses its own chars; rows below decode from tensor ---")
    print("Terrain index chars (from encoder one-hot): " + " ".join(TERRAIN_INDEX_CHAR))
    print("Unit buckets 0-13: " + _UNIT_BUCKET_CHAR + " (upper=P0 lower=P1)")

    _print_scalar_table(scalars)

    print("\n--- Decoded from spatial tensor: terrain category (argmax per cell) ---")
    for line in _spatial_terrain_grid(spatial, h, w):
        print(line)

    print("\n--- Decoded from spatial tensor: unit presence (argmax of 28 unit channels) ---")
    for line in _spatial_unit_grid(spatial, h, w):
        print(line)

    print("\n--- HP channel (single channel; stacked units: last writer wins in encoder) ---")
    hp_ch = N_UNIT_CHANNELS
    for r in range(h):
        row = "".join(
            f"{spatial[r, c, hp_ch]:.0f}" if spatial[r, c, hp_ch] > 0.01 else "."
            for c in range(w)
        )
        print(row)

    # Cheap sanity checks (fail loud if encoding breaks)
    terr_off = N_UNIT_CHANNELS + 1
    terr_sum = spatial[:, :, terr_off : terr_off + len(TERRAIN_INDEX_CHAR)].sum(axis=-1)
    bad = np.where((terr_sum[:h, :w] < 0.99) | (terr_sum[:h, :w] > 1.01))
    if bad[0].size > 0:
        print(
            "\n[WARN] Some map cells have terrain one-hot sum != 1.0 — check encoder/terrain mapping.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
