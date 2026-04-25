# -*- coding: utf-8 -*-
"""Emit one snapshot-only .zip per GL ``type: std`` map for the AWBW Replay Player.

Run from repo root::

    python tools/export_gl_std_starter_replays.py

Output: ``.tmp/gl_std_starter_replays/<map_id>_start.zip`` — day-0 / first-turn
state per map, using each map’s first **enabled** tier and the first CO in that
tier for a mirror (p0 = p1). Open any zip in the desktop viewer to validate
starting position, HQs, and predeploy.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.game import make_initial_state
from engine.map_loader import load_map
from rl.paths import REPLAY_PLAYER_EXE_ENV, REPLAY_PLAYER_THIRD_PARTY_DIR, resolve_awbw_replay_player_exe
from tools.export_awbw_replay import write_awbw_replay

_POOL = ROOT / "data" / "gl_map_pool.json"
_MAPS = ROOT / "data" / "maps"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / ".tmp" / "gl_std_starter_replays",
        help="Directory for <map_id>_start.zip files",
    )
    ap.add_argument(
        "--open-first",
        action="store_true",
        help="Start AWBW Replay Player on the first std map zip (if exe found)",
    )
    ap.add_argument(
        "--open-folder",
        action="store_true",
        help="Open the out-dir in Explorer (Windows)",
    )
    ap.add_argument(
        "--map-id",
        type=int,
        default=None,
        help="If set, only export this std map_id (e.g. 133665)",
    )
    args = ap.parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = json.loads(_POOL.read_text(encoding="utf-8"))
    std_maps = [m for m in pool if m.get("type") == "std"]
    if args.map_id is not None:
        std_maps = [m for m in std_maps if int(m["map_id"]) == int(args.map_id)]
        if not std_maps:
            print(f"No std map with map_id={args.map_id} in gl_map_pool.json", file=sys.stderr)
            sys.exit(1)
    for m in std_maps:
        map_id = int(m["map_id"])
        en = [t for t in m.get("tiers", []) if t.get("enabled") and t.get("co_ids")]
        t = en[0] if en else m["tiers"][0]
        tname = str(t["tier_name"])
        co_ids: list = list(t["co_ids"])
        c0 = c1 = int(co_ids[0])  # mirror: same CO on both sides
        map_data = load_map(map_id, _POOL, _MAPS)
        mks: dict = {"starting_funds": 0, "tier_name": tname}
        rfm = getattr(map_data, "replay_first_mover", None)
        if rfm is not None:
            mks["replay_first_mover"] = int(rfm)
        st = make_initial_state(map_data, c0, c1, **mks)
        name = str(m.get("name", map_id))[:64]
        zp = out_dir / f"{map_id}_start.zip"
        write_awbw_replay(
            [st],
            zp,
            game_id=map_id,
            game_name=f"{name} (std {tname})",
        )
        print(f"OK {map_id:6d}  {tname!r}  co {c0} vs {c1}  ->  {zp}")

    print()
    print(f"Wrote {len(std_maps)} zips to {out_dir.resolve()}")

    if args.open_folder and sys.platform == "win32":
        subprocess.Popen(
            ["explorer", str(out_dir.resolve())],
            close_fds=True,
        )

    if args.open_first:
        firsts = sorted(out_dir.glob("*_start.zip")) if out_dir.is_dir() else []
        first = firsts[0] if firsts else None
        exe = resolve_awbw_replay_player_exe(ROOT)
        if first is not None and exe is not None:
            subprocess.Popen([str(exe), str(first.resolve())], close_fds=True)
            print(f"Launched: {exe.name}  {first}")
        else:
            print(
                f"Skip --open-first: no zips or AWBW Replay Player.exe not found "
                f"(set {REPLAY_PLAYER_EXE_ENV} or build under {REPLAY_PLAYER_THIRD_PARTY_DIR}).",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
