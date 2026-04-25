"""Dump End.updatedInfo.repaired for env 17 of gid 1607045."""
from __future__ import annotations
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools.diff_replay_zips import load_replay
from tools.oracle_zip_replay import parse_p_envelopes_from_zip


def main() -> int:
    zip_path = REPO / "replays/amarriner_gl/1607045.zip"
    envs = parse_p_envelopes_from_zip(zip_path)
    for env_i, (pid, day, actions) in enumerate(envs):
        for obj in actions:
            if obj.get("action") != "End":
                continue
            ui = obj.get("updatedInfo") or {}
            rep = ui.get("repaired") if isinstance(ui, dict) else None
            if isinstance(rep, dict):
                rep = rep.get("global")
            if rep:
                ids = []
                if isinstance(rep, list):
                    for r in rep:
                        if isinstance(r, dict):
                            ids.append((r.get("units_id"), r.get("units_hit_points")))
                print(f"env {env_i} day {day} pid={pid}  End.repaired -> {ids}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
