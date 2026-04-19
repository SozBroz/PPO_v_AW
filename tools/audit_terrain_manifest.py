#!/usr/bin/env python3
"""
List terrain integer IDs appearing in data/maps/*.csv that are missing from
server/static/awbw_textures/manifest.json terrainByAwbwId.

Run from repo root: python tools/audit_terrain_manifest.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAPS_DIR = ROOT / "data" / "maps"
MANIFEST = ROOT / "server" / "static" / "awbw_textures" / "manifest.json"


def collect_csv_ids() -> set[int]:
    """Terrain integers from CSV rows (comma-separated cells only)."""
    ids: set[int] = set()
    if not MAPS_DIR.is_dir():
        print(f"No maps dir: {MAPS_DIR}", file=sys.stderr)
        return ids
    for path in sorted(MAPS_DIR.glob("*.csv")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            for part in line.split(","):
                part = part.strip()
                if not part:
                    continue
                if re.fullmatch(r"-?\d+", part):
                    ids.add(int(part))
    return ids


def main() -> None:
    if not MANIFEST.is_file():
        print(f"Missing manifest: {MANIFEST}", file=sys.stderr)
        sys.exit(1)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    keys = set(manifest.get("terrainByAwbwId", {}).keys())

    used = collect_csv_ids()
    missing = sorted(t for t in used if str(t) not in keys)
    print(f"Maps dir: {MAPS_DIR} ({len(list(MAPS_DIR.glob('*.csv')))} csv)")
    print(f"Unique integer tokens in CSVs: {len(used)}")
    print(f"Manifest terrain keys: {len(keys)}")
    if not missing:
        print("OK: no CSV terrain IDs missing from manifest.")
        return
    print(f"Missing from manifest ({len(missing)}): {missing[:80]}{' …' if len(missing) > 80 else ''}")
    print(
        "Note: Replay Player Tiles.json only lists ~170 AWBWIDs; engine maps use many "
        "property/country IDs (e.g. 34-57) absent from that JSON. Add paths in "
        "tools/sync_awbw_textures.py EXTRA_TERRAIN_BY_AWBW_ID or extend Tiles sourcing."
    )


if __name__ == "__main__":
    main()
