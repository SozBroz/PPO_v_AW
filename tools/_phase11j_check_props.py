"""Quick PHP frame inspector for prop count + funds at specific envelopes."""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.diff_replay_zips import load_replay
from engine.terrain import get_terrain


def main(gid: int, envs: list[int]) -> None:
    f = load_replay(Path(f"replays/amarriner_gl/{gid}.zip"))
    for env_i in envs:
        fi = env_i + 1
        if fi >= len(f):
            print(f"frame {fi} oob (len={len(f)})"); continue
        counts = Counter()
        for b in (f[fi].get("buildings") or {}).values():
            info = get_terrain(int(b["terrain_id"]))
            if info and info.is_property and not info.is_lab and not info.is_comm_tower and info.country_id is not None:
                counts[info.country_id] += 1
        print(f"=== env_i={env_i} (frame {fi}) ===")
        print(f"  prop counts by country = {dict(counts)}")
        for k, pl in (f[fi].get("players") or {}).items():
            pid = pl.get("id"); co = pl.get("co_id"); cn = pl.get("countries_id"); fu = pl.get("funds")
            print(f"  player slot={k} id={pid} co={co} country={cn} funds={fu}")


if __name__ == "__main__":
    gid = int(sys.argv[1])
    envs = [int(x) for x in sys.argv[2:]]
    main(gid, envs)
