"""Inspect a single tile's building+unit across PHP frames."""
from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.diff_replay_zips import load_replay


def main(gid: int, x: int, y: int, frames_idx: list[int]) -> None:
    f = load_replay(Path(f"replays/amarriner_gl/{gid}.zip"))
    for fi in frames_idx:
        if fi >= len(f):
            print(f"frame {fi} oob"); continue
        print(f"=== frame {fi} ===")
        for b in (f[fi].get("buildings") or {}).values():
            if int(b["x"]) == x and int(b["y"]) == y:
                print(f"  bldg: terrain_id={b.get('terrain_id')} capture={b.get('capture')} last_capture={b.get('last_capture')}")
                break
        else:
            print("  bldg: not found")
        units_here = []
        for u in (f[fi].get("units") or {}).values():
            ux, uy = u.get("x"), u.get("y")
            if ux is not None and uy is not None and int(ux) == x and int(uy) == y:
                units_here.append(u)
        for u in units_here:
            uid = u.get("id"); nm = u.get("name"); pid = u.get("players_id"); hp = u.get("hit_points")
            print(f"  unit: id={uid} {nm} pid={pid} hp={hp}")


if __name__ == "__main__":
    gid = int(sys.argv[1]); x = int(sys.argv[2]); y = int(sys.argv[3])
    fis = [int(s) for s in sys.argv[4:]]
    main(gid, x, y, fis)
