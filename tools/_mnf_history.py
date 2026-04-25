"""Probe whether the failing AWBW unit was ever Built earlier in the stream."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tools.oracle_zip_replay import parse_p_envelopes_from_zip


def main() -> None:
    gid = int(sys.argv[1])
    target_uid = int(sys.argv[2])
    envs = parse_p_envelopes_from_zip(Path(f"replays/amarriner_gl/{gid}.zip"))
    print(f"gid={gid} target_uid={target_uid} total_envelopes={len(envs)}")
    for env_idx, (pid, day, actions) in enumerate(envs):
        for ai, a in enumerate(actions):
            kind = a.get("action")
            blob = json.dumps(a, default=str)
            if str(target_uid) not in blob:
                continue
            print(f"\n--- env={env_idx} ai={ai} pid={pid} day={day} kind={kind} ---")
            if kind == "Build":
                print(json.dumps(a, indent=2, default=str)[:2000])
            else:
                inner = a.get(kind) or a
                unit = inner.get("unit") if isinstance(inner, dict) else None
                if isinstance(unit, dict):
                    g = unit.get("global") or unit
                    if isinstance(g, dict):
                        print(
                            f"  units_id={g.get('units_id')} "
                            f"name={g.get('units_name')} "
                            f"x={g.get('units_x')} y={g.get('units_y')} "
                            f"hp={g.get('units_hit_points')} "
                            f"player={g.get('units_players_id')}"
                        )


if __name__ == "__main__":
    main()
