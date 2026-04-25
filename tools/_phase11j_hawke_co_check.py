"""Confirm CO IDs and seat assignments for the 3 Hawke residual zips."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.diff_replay_zips import load_replay  # type: ignore


CO_NAMES = {
    1: "Andy", 2: "Grit", 3: "Jake", 4: "Max", 5: "Drake", 6: "Eagle",
    7: "Sami", 8: "Sensei", 9: "Olaf", 10: "Sonja", 11: "Hachi",
    12: "Hawke", 13: "Sensei?",  # 13 might differ
    14: "Jess", 15: "Colin", 16: "Eagle?",
    17: "Sami?", 18: "Sonja", 19: "Sasha", 20: "Adder",
    21: "Flak", 22: "Hawke?", 23: "Kindle", 24: "Nell",
    25: "Flak?", 26: "Jugger", 27: "Javier", 28: "Rachel",
    29: "Sturm", 30: "Von Bolt",
}


def main() -> int:
    targets = sys.argv[1:] or ["1617442", "1635679", "1635846"]
    for gid in targets:
        path = REPO / "replays" / "amarriner_gl" / f"{gid}.zip"
        print(f"\n=== {gid} ===")
        if not path.exists():
            print("  MISSING")
            continue
        frames = load_replay(path)
        print(f"  frames: {len(frames)}")
        if not frames:
            continue
        f0 = frames[0]
        players = f0.get("players", {})
        for k, pl in players.items():
            cid = pl.get("co_id")
            print(
                f"    seat[{k}]: pid={pl.get('id')} order={pl.get('order')} "
                f"co_id={cid} co_name={CO_NAMES.get(int(cid) if cid else -1, '?')!r} "
                f"team={pl.get('team')} country={pl.get('countries_id')}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
